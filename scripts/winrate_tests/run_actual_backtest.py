"""
scripts/winrate_tests/run_actual_backtest.py
=============================================

Runs the PerStrategyTester on REAL cached OHLCV data to produce ACTUAL
win rates per strategy / per confidence level / per tactic.

Outputs:
  - actual_backtest_results.json   (machine-readable, full detail)
  - actual_backtest_results.md     (human-readable summary)
  - actual_backtest_trades.csv     (one row per trade, for further analysis)

Usage:
  # Default — runs all pairs + timeframes in data/backtest_cache/
  python scripts/winrate_tests/run_actual_backtest.py

  # Specific pairs / timeframes
  python scripts/winrate_tests/run_actual_backtest.py \\
      --pairs EURUSD GBPUSD \\
      --timeframes M15 H1

  # Quick mode — only 3000 most recent candles
  python scripts/winrate_tests/run_actual_backtest.py --quick

  # Skip specific strategies
  python scripts/winrate_tests/run_actual_backtest.py --skip multi_pa sd_zones_scored

This script DOES NOT read the static confidence_winrate_data.json — it
produces its OWN results from a real backtest run, so you can compare
them against the JSON to verify the numbers.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make the project importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

# Silence chatty per-strategy warnings (e.g. multi_pa signature mismatch)
import logging as _logging
_logging.getLogger("per_strategy").setLevel(_logging.ERROR)
_logging.getLogger("backtest_loader").setLevel(_logging.ERROR)

from backtest.data_loader import HistoricalDataLoader  # noqa: E402
from backtest.per_strategy_tester import PerStrategyTester, StrategyResult, Trade  # noqa: E402
from utils.logger import get_logger  # noqa: E402

log = get_logger("actual_backtest")
log.setLevel(_logging.INFO)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_CACHE_DIR = PROJECT_ROOT / "data" / "backtest_cache"
OUTPUT_DIR = PROJECT_ROOT / "scripts" / "winrate_tests" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Pairs × timeframes expected in the cache
DEFAULT_PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]
DEFAULT_TIMEFRAMES = ["M15", "H1"]

# Strategies that are known to be slow or broken in the current per_strategy_tester.
# These are skipped by default unless explicitly re-enabled with --include-all.
DEFAULT_SKIP = ["multi_pa", "pin_bar", "sd_zones_scored"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_dict(obj: Any) -> Any:
    """Recursively convert dataclasses / tuples / lists to plain dict/list."""
    if is_dataclass(obj):
        return {k: _to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(x) for x in obj]
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    return obj


def _find_data_file(pair: str, timeframe: str, quick: bool) -> Optional[Path]:
    """Locate the cached parquet file for (pair, timeframe)."""
    # Map M15 -> M15, H1 -> H1 (cache uses these literal labels)
    candidates = []
    if quick:
        candidates.append(DATA_CACHE_DIR / f"{pair}_{timeframe}_3000.parquet")
    candidates.append(DATA_CACHE_DIR / f"{pair}_{timeframe}_10000.parquet")
    for c in candidates:
        if c.exists():
            return c
    return None


# Module-level override — when set, _load_ohlcv tails only this many bars.
# Set by the CLI --bars flag to allow fast testing without modifying cache files.
_MAX_BARS_OVERRIDE: Optional[int] = None

# Raw H4 CSVs (data/{PAIR}_H4.csv) exist for every pair and are used live to
# feed the H4 trend-agreement filter inside stop_hunt (see
# agents/analysis_agent.py's "P1 parity fix" comment + per_strategy_tester's
# _test_stop_hunt(df_h4=...)). run_actual_backtest.py previously never loaded
# this file, so df_h4 was always None here and the filter was silently
# skipped for every backtested stop_hunt trade — degrading fidelity for the
# one strategy this refresh exists to validate. Loaded once per pair, cached.
_H4_CACHE: Dict[str, pd.DataFrame] = {}


def _load_h4(pair: str) -> Optional[pd.DataFrame]:
    """Load full H4 OHLCV for `pair` from data/{pair}_H4.csv, cached per pair."""
    if pair in _H4_CACHE:
        return _H4_CACHE[pair]
    csv_path = PROJECT_ROOT / "data" / f"{pair}_H4.csv"
    if not csv_path.exists():
        log.warning(f"No H4 CSV for {pair} at {csv_path} — stop_hunt H4 filter will be skipped for this pair.")
        _H4_CACHE[pair] = None
        return None
    df_h4 = pd.read_csv(csv_path)
    df_h4.columns = [str(c).lower().strip() for c in df_h4.columns]
    if "datetime_utc" in df_h4.columns:
        df_h4 = df_h4.rename(columns={"datetime_utc": "time"})
    if "tick_volume" in df_h4.columns:
        df_h4 = df_h4.rename(columns={"tick_volume": "volume"})
    df_h4["time"] = pd.to_datetime(df_h4["time"], utc=True, errors="coerce")
    df_h4 = df_h4.dropna(subset=["time"]).drop_duplicates(subset=["time"]).sort_values("time")
    df_h4 = df_h4.set_index("time")
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in df_h4.columns:
            df_h4[c] = 0.0
        df_h4[c] = pd.to_numeric(df_h4[c], errors="coerce")
    df_h4 = df_h4.dropna(subset=["open", "high", "low", "close"])
    _H4_CACHE[pair] = df_h4
    return df_h4


def _load_ohlcv(pair: str, timeframe: str, data_file: Path) -> pd.DataFrame:
    """Load parquet into the df format PerStrategyTester expects."""
    df = pd.read_parquet(data_file)

    # Normalize column names
    df.columns = [str(c).lower().strip() for c in df.columns]
    if "datetime" in df.columns and "time" not in df.columns:
        df = df.rename(columns={"datetime": "time"})
    if "date" in df.columns and "time" not in df.columns:
        df = df.rename(columns={"date": "time"})

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.dropna(subset=["time"]).drop_duplicates(subset=["time"]).sort_values("time")
        df = df.set_index("time")

    needed = ["open", "high", "low", "close", "volume"]
    for c in needed:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])

    # The PerStrategyTester enriches its own indicators, so we pass the raw df.
    df.attrs["pair"] = pair
    df.attrs["timeframe"] = timeframe

    # Apply max-bars override if set (used by --bars CLI flag)
    if _MAX_BARS_OVERRIDE is not None and _MAX_BARS_OVERRIDE > 0 and len(df) > _MAX_BARS_OVERRIDE:
        df = df.tail(_MAX_BARS_OVERRIDE).copy()

    return df


# ---------------------------------------------------------------------------
# Per-strategy aggregation across (pair, timeframe)
# ---------------------------------------------------------------------------

def _empty_agg() -> Dict[str, Any]:
    return {
        "n_trades": 0,
        "n_wins": 0,
        "n_losses": 0,
        "n_breakeven": 0,
        "win_rate": 0.0,
        "avg_r": 0.0,
        "total_r": 0.0,
        "by_confidence": defaultdict(lambda: {"trades": 0, "wins": 0, "win_rate": 0.0, "total_r": 0.0}),
        "by_tactic": defaultdict(lambda: {"trades": 0, "wins": 0, "win_rate": 0.0, "total_r": 0.0}),
        "by_pair_tf": defaultdict(lambda: {"trades": 0, "wins": 0, "win_rate": 0.0}),
    }


def _accumulate(strategy_agg: Dict[str, Any], result: StrategyResult, pair: str, tf: str) -> None:
    """Merge a single (pair, tf) result into the strategy-level aggregate."""
    strategy_agg["n_trades"] += result.n_trades
    strategy_agg["n_wins"] += result.n_wins
    strategy_agg["n_losses"] += result.n_losses
    strategy_agg["n_breakeven"] += result.n_breakeven
    strategy_agg["total_r"] += result.total_r

    # Per-confidence merge
    # NOTE: per_strategy_tester uses keys {n_trades, win_rate, avg_r, total_r}
    for conf, stats in (result.by_confidence or {}).items():
        c = strategy_agg["by_confidence"][conf]
        n_trades = int(stats.get("n_trades", stats.get("trades", 0)))
        win_rate = float(stats.get("win_rate", 0.0))
        n_wins = int(round(win_rate * n_trades))  # derive wins from win_rate * n
        total_r = float(stats.get("total_r", 0.0))
        c["trades"] += n_trades
        c["wins"] += n_wins
        c["total_r"] += total_r

    # Per-tactic merge
    # NOTE: per_strategy_tester uses keys {n_trades, win_rate, avg_r}
    for tac, stats in (result.by_tactic or {}).items():
        t = strategy_agg["by_tactic"][tac]
        n_trades = int(stats.get("n_trades", stats.get("trades", 0)))
        win_rate = float(stats.get("win_rate", 0.0))
        n_wins = int(round(win_rate * n_trades))
        avg_r = float(stats.get("avg_r", 0.0))
        total_r = float(stats.get("total_r", avg_r * n_trades))
        t["trades"] += n_trades
        t["wins"] += n_wins
        t["total_r"] += total_r

    # Per-(pair, tf) merge
    ptf = strategy_agg["by_pair_tf"][f"{pair}_{tf}"]
    ptf["trades"] += result.n_trades
    ptf["wins"] += result.n_wins


def _finalize_aggregate(agg: Dict[str, Any]) -> Dict[str, Any]:
    """Compute win rates and avg_r after accumulation is done."""
    n = agg["n_trades"]
    agg["win_rate"] = (agg["n_wins"] / n) if n > 0 else 0.0
    agg["avg_r"] = (agg["total_r"] / n) if n > 0 else 0.0

    for conf, stats in agg["by_confidence"].items():
        if stats["trades"] > 0:
            stats["win_rate"] = stats["wins"] / stats["trades"]
            stats["avg_r"] = stats["total_r"] / stats["trades"]

    for tac, stats in agg["by_tactic"].items():
        if stats["trades"] > 0:
            stats["win_rate"] = stats["wins"] / stats["trades"]
            stats["avg_r"] = stats["total_r"] / stats["trades"]

    for ptf, stats in agg["by_pair_tf"].items():
        if stats["trades"] > 0:
            stats["win_rate"] = stats["wins"] / stats["trades"]

    # Convert defaultdicts to plain dicts for JSON
    agg["by_confidence"] = dict(agg["by_confidence"])
    agg["by_tactic"] = dict(agg["by_tactic"])
    agg["by_pair_tf"] = dict(agg["by_pair_tf"])
    return agg


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_backtest(
    pairs: List[str],
    timeframes: List[str],
    skip_strategies: List[str],
    quick: bool = False,
) -> Dict[str, Any]:
    """Run the per-strategy tester on every (pair, tf) combo and aggregate."""

    loader = HistoricalDataLoader()
    tester = PerStrategyTester()

    # Pre-patch: replace _test_<strategy> methods for skipped strategies
    # so they return instantly instead of running real logic (which is slow
    # even when failing — e.g. multi_pa / pin_bar spam warnings per bar).
    from backtest.per_strategy_tester import StrategyResult as _SR
    skip_set = set(skip_strategies)
    for strat_name in skip_set:
        method_name = f"_test_{strat_name}"
        if hasattr(tester, method_name):
            def _make_skipper(sn):
                def _skip(*a, **k):
                    # Signature: _test_<strat>(self, df, pair, timeframe, pip)
                    # a[0]=self, a[1]=df, a[2]=pair, a[3]=timeframe
                    pair = a[2] if len(a) > 2 else k.get("pair", "unknown")
                    tf = a[3] if len(a) > 3 else k.get("timeframe", "unknown")
                    return _SR(sn, pair, tf)
                return _skip
            setattr(tester, method_name, _make_skipper(strat_name))
            log.info(f"Pre-skipped strategy: {strat_name} (will return empty result)")

    # strategy_name -> aggregate
    aggregates: Dict[str, Dict[str, Any]] = defaultdict(_empty_agg)
    # Per-(pair, tf) raw results, kept for the detail file
    per_pair_tf_results: List[Dict[str, Any]] = []
    # Flat trade list for CSV
    all_trades: List[Dict[str, Any]] = []

    combos = [(p, t) for p in pairs for t in timeframes]
    log.info(f"Running backtest on {len(combos)} (pair, timeframe) combos...")

    for pair in pairs:
        for tf in timeframes:
            data_file = _find_data_file(pair, tf, quick=quick)
            if data_file is None:
                log.warning(f"No data file for {pair} {tf} — skipping.")
                continue

            log.info(f"=== {pair} {tf}  ({data_file.name}) ===")
            t0 = time.time()
            try:
                df = _load_ohlcv(pair, tf, data_file)
                if len(df) < 200:
                    log.warning(f"  Only {len(df)} bars for {pair} {tf} — too few, skipping.")
                    continue

                df_h4 = _load_h4(pair)
                results = tester.run_all(df, pair=pair, timeframe=tf, df_h4=df_h4)
                elapsed = time.time() - t0
                log.info(f"  Done in {elapsed:.1f}s — {len(df)} bars, "
                         f"{sum(r.n_trades for r in results['strategies'].values())} trades total")

                # Accumulate per strategy
                for strat_name, strat_result in results["strategies"].items():
                    if strat_name in skip_strategies:
                        continue
                    _accumulate(aggregates[strat_name], strat_result, pair, tf)

                    # Save individual trades for CSV
                    for tr in strat_result.trades:
                        all_trades.append({
                            "strategy":  tr.strategy,
                            "pair":      tr.pair,
                            "timeframe": tr.timeframe,
                            "direction": tr.direction,
                            "entry_time":  tr.entry_time.isoformat() if tr.entry_time else "",
                            "entry_price": tr.entry_price,
                            "stop_loss":   tr.stop_loss,
                            "take_profit": tr.take_profit,
                            "exit_time":   tr.exit_time.isoformat() if tr.exit_time else "",
                            "exit_price":  tr.exit_price,
                            "exit_reason": tr.exit_reason,
                            "r_multiple":  tr.r_multiple,
                            "pnl_pips":    tr.pnl_pips,
                            "confidence":  tr.confidence,
                            "tactic":      tr.tactic,
                            "win":         tr.win,
                        })

                # Snapshot per (pair, tf) for the detail file
                per_pair_tf_results.append({
                    "pair": pair,
                    "timeframe": tf,
                    "n_bars": int(len(df)),
                    "data_file": str(data_file.relative_to(PROJECT_ROOT)),
                    "strategies": {
                        name: {
                            "n_trades": r.n_trades,
                            "win_rate": r.win_rate,
                            "avg_r":    r.avg_r,
                            "total_r":  r.total_r,
                            "n_wins":   r.n_wins,
                            "n_losses": r.n_losses,
                            "by_confidence": dict(r.by_confidence) if r.by_confidence else {},
                            "by_tactic":     dict(r.by_tactic)     if r.by_tactic     else {},
                        }
                        for name, r in results["strategies"].items()
                        if name not in skip_strategies
                    },
                })
            except Exception as e:
                log.error(f"  FAILED on {pair} {tf}: {e}")
                log.debug(traceback.format_exc())

    # Finalize aggregates
    final_aggregates = {name: _finalize_aggregate(agg) for name, agg in aggregates.items()}

    # Sort tactics within each strategy by win_rate desc
    for name, agg in final_aggregates.items():
        agg["by_tactic"] = dict(
            sorted(agg["by_tactic"].items(),
                   key=lambda kv: kv[1].get("win_rate", 0.0), reverse=True)
        )

    return {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "pairs": pairs,
            "timeframes": timeframes,
            "skipped_strategies": skip_strategies,
            "quick_mode": quick,
            "n_combos_run": len(per_pair_tf_results),
            "n_total_trades": len(all_trades),
        },
        "strategy_aggregates": final_aggregates,
        "per_pair_tf": per_pair_tf_results,
        "trades": all_trades,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_json(out: Dict[str, Any], path: Path) -> None:
    # Strip the trades list from the JSON to keep it readable; trades go to CSV
    slim = {k: v for k, v in out.items() if k != "trades"}
    slim["n_total_trades_in_csv"] = len(out.get("trades", []))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_dict(slim), f, indent=2, ensure_ascii=False, default=str)
    log.info(f"Wrote JSON: {path}  ({path.stat().st_size / 1024:.1f} KB)")


def write_trades_csv(trades: List[Dict[str, Any]], path: Path) -> None:
    if not trades:
        log.warning("No trades to write to CSV.")
        return
    df = pd.DataFrame(trades)
    df.to_csv(path, index=False)
    log.info(f"Wrote CSV: {path}  ({len(df)} trades, {path.stat().st_size / 1024:.1f} KB)")


def write_markdown(out: Dict[str, Any], path: Path) -> None:
    md = _build_markdown(out)
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    log.info(f"Wrote MD:   {path}  ({path.stat().st_size / 1024:.1f} KB)")


def _build_markdown(out: Dict[str, Any]) -> str:
    meta = out["metadata"]
    aggs = out["strategy_aggregates"]
    lines: List[str] = []

    lines.append("# Actual Backtest Results — Real Win Rates")
    lines.append("")
    lines.append(f"**Generated:** {meta['generated_at']}  ")
    lines.append(f"**Pairs:** {', '.join(meta['pairs'])}  ")
    lines.append(f"**Timeframes:** {', '.join(meta['timeframes'])}  ")
    lines.append(f"**Combos run:** {meta['n_combos_run']}  ")
    lines.append(f"**Total trades:** {meta['n_total_trades']}  ")
    lines.append(f"**Quick mode:** {meta['quick_mode']}  ")
    if meta["skipped_strategies"]:
        lines.append(f"**Skipped strategies:** {', '.join(meta['skipped_strategies'])}")
    lines.append("")
    lines.append("> Numbers below come from a REAL backtest run on cached OHLCV data — ")
    lines.append("> not from a static JSON. Use this to verify `confidence_winrate_data.json`.")
    lines.append("")

    # ---- Strategy summary table ----
    lines.append("## Strategy Summary (aggregated across all pairs/timeframes)")
    lines.append("")
    lines.append("| Strategy | Trades | Wins | Losses | Win Rate | Avg R | Total R |")
    lines.append("|----------|-------:|-----:|------:|---------:|------:|--------:|")
    for name in sorted(aggs.keys(), key=lambda n: aggs[n]["win_rate"], reverse=True):
        a = aggs[name]
        lines.append(f"| `{name}` | {a['n_trades']} | {a['n_wins']} | {a['n_losses']} | "
                     f"{a['win_rate']:.1%} | {a['avg_r']:+.2f} | {a['total_r']:+.2f} |")
    lines.append("")

    # ---- Per-confidence winrates ----
    lines.append("## Per-Confidence Win Rates (actual)")
    lines.append("")
    lines.append("| Strategy | Confidence | Trades | Win Rate | Avg R |")
    lines.append("|----------|-----------|-------:|---------:|------:|")
    for name in sorted(aggs.keys()):
        a = aggs[name]
        if not a["by_confidence"]:
            lines.append(f"| `{name}` | (no confidence data) | — | — | — |")
            continue
        for conf in ["High", "Medium", "Low"]:
            stats = a["by_confidence"].get(conf)
            if stats is None:
                lines.append(f"| `{name}` | {conf} | 0 | — | — |")
            else:
                lines.append(f"| `{name}` | {conf} | {stats['trades']} | "
                             f"{stats['win_rate']:.1%} | {stats.get('avg_r', 0):+.2f} |")
    lines.append("")

    # ---- Per-tactic top 5 per strategy ----
    lines.append("## Top Tactics Per Strategy (actual, by win rate)")
    lines.append("")
    for name in sorted(aggs.keys()):
        a = aggs[name]
        lines.append(f"### `{name}`")
        if not a["by_tactic"]:
            lines.append("_(no tactics recorded)_")
            lines.append("")
            continue
        lines.append("| Tactic | Trades | Win Rate | Avg R |")
        lines.append("|--------|-------:|---------:|------:|")
        top = list(a["by_tactic"].items())[:10]
        for tac, stats in top:
            lines.append(f"| `{tac}` | {stats['trades']} | {stats['win_rate']:.1%} | "
                         f"{stats.get('avg_r', 0):+.2f} |")
        lines.append("")

    # ---- Per-(pair, tf) breakdown ----
    lines.append("## Per-(Pair, Timeframe) Trade Counts")
    lines.append("")
    lines.append("| Strategy | " + " | ".join(
        f"{p}_{t}" for p in meta["pairs"] for t in meta["timeframes"]
    ) + " |")
    lines.append("|----------|" + "|".join(["---"] * len(meta["pairs"]) * len(meta["timeframes"])) + "|")
    for name in sorted(aggs.keys()):
        a = aggs[name]
        cells = []
        for p in meta["pairs"]:
            for t in meta["timeframes"]:
                key = f"{p}_{t}"
                stats = a["by_pair_tf"].get(key)
                if stats and stats["trades"] > 0:
                    cells.append(f"{stats['trades']} ({stats['win_rate']:.0%})")
                else:
                    cells.append("—")
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    lines.append("")

    # ---- Comparison with the static JSON ----
    lines.append("## Comparison with `confidence_winrate_data.json` (static report)")
    lines.append("")
    lines.append("If the actual numbers above differ significantly from the static JSON, ")
    lines.append("the static JSON is stale — run `refresh_confidence_data.py` to regenerate it.")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="Run actual per-strategy backtest on cached OHLCV data."
    )
    p.add_argument("--pairs", nargs="*", default=DEFAULT_PAIRS,
                   help=f"Pairs to test (default: {DEFAULT_PAIRS})")
    p.add_argument("--timeframes", nargs="*", default=DEFAULT_TIMEFRAMES,
                   help=f"Timeframes to test (default: {DEFAULT_TIMEFRAMES})")
    p.add_argument("--skip", nargs="*", default=DEFAULT_SKIP,
                   help=f"Strategies to skip (default: {DEFAULT_SKIP})")
    p.add_argument("--include-all", action="store_true",
                   help="Don't skip any strategy (overrides --skip)")
    p.add_argument("--quick", action="store_true",
                   help="Use 3000-candle files instead of 10000 (faster, less reliable)")
    p.add_argument("--bars", type=int, default=None,
                   help="Cap each pair to last N bars (e.g. --bars 1000 for a fast smoke test)")
    p.add_argument("--out-dir", default=str(OUTPUT_DIR),
                   help="Output directory")
    args = p.parse_args()

    if args.include_all:
        args.skip = []

    # Apply bars override
    global _MAX_BARS_OVERRIDE
    _MAX_BARS_OVERRIDE = args.bars

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("ACTUAL BACKTEST RUN")
    log.info("=" * 70)
    log.info(f"Pairs:      {args.pairs}")
    log.info(f"Timeframes: {args.timeframes}")
    log.info(f"Skip:       {args.skip or '(none)'}")
    log.info(f"Quick mode: {args.quick}")
    log.info(f"Out dir:    {out_dir}")
    log.info("=" * 70)

    result = run_backtest(
        pairs=args.pairs,
        timeframes=args.timeframes,
        skip_strategies=args.skip,
        quick=args.quick,
    )

    # Write outputs
    write_json(result,       out_dir / "actual_backtest_results.json")
    write_trades_csv(result["trades"], out_dir / "actual_backtest_trades.csv")
    write_markdown(result,   out_dir / "actual_backtest_results.md")

    # Print a brief summary to console
    print("\n" + "=" * 70)
    print("ACTUAL BACKTEST SUMMARY")
    print("=" * 70)
    print(f"{'Strategy':<25} {'Trades':>8} {'WinRate':>10} {'AvgR':>8}")
    print("-" * 55)
    for name in sorted(result["strategy_aggregates"].keys(),
                       key=lambda n: result["strategy_aggregates"][n]["win_rate"], reverse=True):
        a = result["strategy_aggregates"][name]
        print(f"{name:<25} {a['n_trades']:>8} {a['win_rate']:>10.1%} {a['avg_r']:>+8.2f}")
    print("-" * 55)
    print(f"Total trades: {result['metadata']['n_total_trades']}")
    print(f"\nOutputs written to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())