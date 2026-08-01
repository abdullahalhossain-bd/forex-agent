"""
scripts/winrate_tests/refresh_confidence_data.py
=================================================

Runs an actual backtest, then regenerates `intelligence/confidence_winrate_data.json`
from the REAL numbers. This is the script you run monthly to keep the live
decision system's lookup table fresh.

Workflow:
  1. Run PerStrategyTester on every (pair, timeframe) combo in data cache.
  2. Aggregate results by strategy / confidence / tactic.
  3. Apply the global thresholds (use / caution / disable).
  4. Write the new confidence_winrate_data.json.
  5. Reload the lookup singleton so the live system sees the new data.

Usage:
  python scripts/winrate_tests/refresh_confidence_data.py
  python scripts/winrate_tests/refresh_confidence_data.py --quick
  python scripts/winrate_tests/refresh_confidence_data.py \\
      --pairs EURUSD GBPUSD --timeframes M15 H1
  python scripts/winrate_tests/refresh_confidence_data.py --dry-run   # print, don't write
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Make project importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Import the runner we built in script 1
from scripts.winrate_tests.run_actual_backtest import (  # noqa: E402
    run_backtest,
    DEFAULT_SKIP,
    DEFAULT_PAIRS as RUNNER_DEFAULT_PAIRS,
    DEFAULT_TIMEFRAMES as RUNNER_DEFAULT_TIMEFRAMES,
)
from utils.logger import get_logger  # noqa: E402

log = get_logger("refresh_confidence_data")

DEFAULT_PAIRS = RUNNER_DEFAULT_PAIRS
DEFAULT_TIMEFRAMES = RUNNER_DEFAULT_TIMEFRAMES
# NOTE: skip defaults to DEFAULT_SKIP (same as run_actual_backtest.py) so that
# known-broken / slow strategies (pin_bar, multi_pa, sd_zones_scored) are not
# executed. Without this, refresh spam-warnings on every bar. Use --include-all
# to override (only do this if you have time to debug the broken strategies).

# Global thresholds — match what's in the existing JSON so behavior stays stable
GLOBAL_THRESHOLDS = {
    "min_trades_for_reliability": 10,
    "min_trades_for_high_reliability": 30,
    "winrate_use_threshold": 0.50,
    "winrate_caution_threshold": 0.40,
    "min_avg_r_for_use": 0.20,
    "notes": "WR >= 50% AND n >= 10  => safe to use; 40% <= WR < 50% => caution; WR < 40% OR n < 10 => disable or fix."
}


def _reliability_tier(n: int, wr: float) -> str:
    """Return 'use' | 'caution' | 'disable' | 'no_data'."""
    if n == 0:
        return "no_data"
    if n < GLOBAL_THRESHOLDS["min_trades_for_reliability"]:
        # Small sample — caution at best, even with high WR
        return "use" if wr >= 0.50 else "disable"  # but mark reliability as low_n
    if wr >= GLOBAL_THRESHOLDS["winrate_use_threshold"]:
        return "use"
    if wr >= GLOBAL_THRESHOLDS["winrate_caution_threshold"]:
        return "caution"
    return "disable"


def _reliability_n(n: int) -> str:
    if n >= 30:
        return "high_n"
    if n >= 10:
        return "med_n"
    if n > 0:
        return "low_n"
    return "no_data"


def _build_strategy_summary(aggregates: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build the strategy_summary list (one entry per strategy)."""
    out = []
    for name, a in aggregates.items():
        # Find best confidence (highest WR with n >= 1)
        best_conf = None
        best_wr = -1.0
        for conf, stats in a["by_confidence"].items():
            if stats["trades"] > 0 and stats["win_rate"] > best_wr:
                best_wr = stats["win_rate"]
                best_conf = conf
        # Find best tactic (highest WR with n >= 3)
        best_tactic = "—"
        best_t_wr = -1.0
        for tac, stats in a["by_tactic"].items():
            if stats["trades"] >= 3 and stats["win_rate"] > best_t_wr:
                best_t_wr = stats["win_rate"]
                best_tactic = tac

        tier = _reliability_tier(a["n_trades"], a["win_rate"])
        rec = "use" if tier == "use" else ("caution" if tier == "caution" else "disable")

        out.append({
            "strategy": name,
            "trades": a["n_trades"],
            "win_rate": round(a["win_rate"], 4),
            "avg_r": round(a["avg_r"], 4),
            "best_confidence": best_conf if best_conf else "—",
            "best_tactic": best_tactic,
            "recommendation": rec,
            "reliability_tier": tier,
        })
    # Sort: use first, then caution, then disable, then by WR desc
    order = {"use": 0, "caution": 1, "disable": 2, "no_data": 3}
    out.sort(key=lambda x: (order.get(x["recommendation"], 99), -x["win_rate"]))
    return out


def _build_per_confidence(aggregates: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build per_confidence_winrates mapping."""
    out: Dict[str, Dict[str, Any]] = {}
    for name, a in aggregates.items():
        per: Dict[str, Any] = {}
        for conf in ["High", "Medium", "Low"]:
            stats = a["by_confidence"].get(conf)
            if stats and stats["trades"] > 0:
                wr = stats["win_rate"]
                n = stats["trades"]
                per[conf] = {
                    "trades": n,
                    "win_rate": round(wr, 4),
                    "avg_r": round(stats.get("avg_r", 0.0), 4),
                    "tier": _reliability_tier(n, wr),
                    "sample_reliability": _reliability_n(n),
                }
            else:
                per[conf] = {
                    "trades": 0,
                    "win_rate": None,
                    "avg_r": None,
                    "tier": "no_data",
                    "sample_reliability": "no_data",
                }
        out[name] = per
    return out


def _build_per_tactic_top(aggregates: Dict[str, Any], limit_per_strategy: int = 30) -> Dict[str, List[Dict[str, Any]]]:
    """Build per_tactic_winrates_top — top N tactics by win_rate per strategy."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for name, a in aggregates.items():
        tactics = []
        for tac, stats in a["by_tactic"].items():
            if stats["trades"] == 0:
                continue
            tactics.append({
                "tactic": tac,
                "trades": stats["trades"],
                "win_rate": round(stats["win_rate"], 4),
            })
        # Sort by win_rate desc, then by trades desc
        tactics.sort(key=lambda t: (-t["win_rate"], -t["trades"]))
        out[name] = tactics[:limit_per_strategy]
    return out


def _build_confidence_to_action_map(
    aggregates: Dict[str, Any],
    strategy_summary: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the rules table that the lookup module consults."""
    rules: List[Dict[str, Any]] = []
    for strat in strategy_summary:
        name = strat["strategy"]
        a = aggregates[name]
        per_conf = a["by_confidence"]

        # If we have per-confidence data, emit one rule per confidence
        if per_conf:
            for conf in ["High", "Medium", "Low"]:
                stats = per_conf.get(conf)
                if not stats or stats.get("trades", 0) == 0:
                    rules.append({
                        "if_strategy": name,
                        "if_confidence": conf,
                        "expected_winrate": None,
                        "n": 0,
                        "action": "skip",
                        "position_scale": 0.0,
                        "notes": "No historical data",
                    })
                    continue
                wr = stats["win_rate"]
                n = stats["trades"]
                action, scale, notes = _decide_action(wr, n)
                rules.append({
                    "if_strategy": name,
                    "if_confidence": conf,
                    "expected_winrate": round(wr, 4),
                    "n": n,
                    "action": action,
                    "position_scale": scale,
                    "notes": notes,
                })
        else:
            # No per-confidence breakdown — apply strategy-level WR to 'any'
            wr = a["win_rate"]
            n = a["n_trades"]
            action, scale, notes = _decide_action(wr, n) if n > 0 else ("skip", 0.0, "No trades")
            rules.append({
                "if_strategy": name,
                "if_confidence": "any",
                "expected_winrate": round(wr, 4) if n > 0 else None,
                "n": n,
                "action": action,
                "position_scale": scale,
                "notes": notes,
            })

    return {
        "_purpose": "When the live decision system tags a signal with a confidence level, "
                    "look up the expected win rate and decide whether to trust it.",
        "rules": rules,
        "default_action_when_unknown": {
            "action": "skip",
            "position_scale": 0.0,
            "notes": "No historical data — fail closed. Do NOT trade on unverified confidence."
        }
    }


def _decide_action(wr: float, n: int) -> tuple[str, float, str]:
    """Apply thresholds -> (action, position_scale, notes)."""
    min_n = GLOBAL_THRESHOLDS["min_trades_for_reliability"]
    wr_use = GLOBAL_THRESHOLDS["winrate_use_threshold"]
    wr_caution = GLOBAL_THRESHOLDS["winrate_caution_threshold"]

    if n == 0:
        return ("skip", 0.0, "No historical data")

    if n < min_n:
        if wr >= wr_use:
            return ("caution", 0.5,
                    f"WR={wr:.1%} but n={n}<{min_n} — small sample, reduce size.")
        return ("skip", 0.0,
                f"WR={wr:.1%} and n={n}<{min_n} — insufficient evidence.")

    if wr >= wr_use:
        scale = 1.0 if wr >= 0.60 else 0.7
        return ("trust", scale, f"WR={wr:.1%} with n={n} — safe to trade.")

    if wr >= wr_caution:
        return ("caution", 0.4,
                f"WR={wr:.1%} with n={n} — borderline, reduce size.")

    return ("skip", 0.0, f"WR={wr:.1%} with n={n} — below {wr_caution:.0%} threshold.")


def _build_live_recommendations(strategy_summary: List[Dict[str, Any]]) -> Dict[str, Any]:
    use = []
    caution = []
    disable = []
    for s in strategy_summary:
        entry = {
            "strategy": s["strategy"],
            "win_rate": s["win_rate"],
            "trades": s["trades"],
        }
        if s["recommendation"] == "use":
            n = s["trades"]
            if n < 30:
                entry["caveat"] = (f"Small sample (n={n}). Treat as promising but not yet "
                                   "statistically robust — keep position sizing conservative "
                                   "until n >= 30.")
            use.append(entry)
        elif s["recommendation"] == "caution":
            caution.append(entry)
        else:
            disable.append({"strategy": s["strategy"],
                            "reason": f"WR {s['win_rate']:.1%} or no trades"})
    return {"use": use, "use_with_caution": caution, "disable_or_fix": disable}


def _build_zero_trade_strategies(aggregates: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for name, a in aggregates.items():
        if a["n_trades"] == 0:
            out.append({"strategy": name,
                        "reason": "needs different data or parameter tuning"})
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def refresh(pairs: List[str], timeframes: List[str],
            skip_strategies: List[str], quick: bool,
            dry_run: bool, out_path: Path) -> int:
    log.info("=" * 70)
    log.info("REFRESH confidence_winrate_data.json from actual backtest")
    log.info("=" * 70)

    # 1. Run the actual backtest
    result = run_backtest(pairs, timeframes, skip_strategies, quick=quick)
    aggregates = result["strategy_aggregates"]

    if not aggregates:
        log.error("No strategy results — aborting refresh.")
        return 2

    # 2. Build the JSON structure
    strategy_summary = _build_strategy_summary(aggregates)
    new_json: Dict[str, Any] = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": f"Actual backtest run via refresh_confidence_data.py — "
                      f"{len(result['metadata']['pairs'])} pairs × "
                      f"{len(result['metadata']['timeframes'])} timeframes, "
                      f"{result['metadata']['n_total_trades']} total trades",
            "pairs_tested": len(pairs),
            "timeframes": timeframes,
            "total_combinations": result["metadata"]["n_combos_run"],
            "successful_runs": result["metadata"]["n_combos_run"],
            "suggested_decision_mode": "confluence",
            "description": "Historical win-rate lookup keyed by strategy + confidence level. "
                           "Live decision system queries this to estimate expected win-rate "
                           "when it tags a signal with High/Medium/Low confidence. "
                           "Regenerated from a real backtest run on cached OHLCV data."
        },
        "global_thresholds": GLOBAL_THRESHOLDS,
        "strategy_summary": strategy_summary,
        "per_confidence_winrates": _build_per_confidence(aggregates),
        "per_tactic_winrates_top": _build_per_tactic_top(aggregates),
        "cci_top_tactics_by_winrate": {
            "_note": "Top 25 cci_state tactics by win_rate (filtered to trades >= 3).",
            "top_25": _build_per_tactic_top(aggregates).get("cci_state", [])[:25]
                     if "cci_state" in aggregates else []
        },
        "cci_top_tactics_by_sample_size": {
            "_note": "Top 25 cci_state tactics by trades count (only WR >= 50%).",
            "top_25": [
                t for t in sorted(
                    _build_per_tactic_top(aggregates).get("cci_state", []),
                    key=lambda t: -t["trades"]
                )[:25] if t["win_rate"] >= 0.50
            ] if "cci_state" in aggregates else []
        },
        "strategies_with_zero_trades": _build_zero_trade_strategies(aggregates),
        "live_trading_recommendations": _build_live_recommendations(strategy_summary),
        "confidence_to_action_map": _build_confidence_to_action_map(aggregates, strategy_summary),
        "next_steps": [
            "Reload the lookup singleton: from intelligence.confidence_winrate_lookup import reload_lookup; reload_lookup()",
            "Wire the lookup into core/orphan_consumers.py apply_signal_scoring().",
            "Re-run this script monthly to keep historical win rates current.",
            "Re-run after any strategy parameter change."
        ]
    }

    # 3. Write or print
    if dry_run:
        log.info("[DRY RUN] Not writing JSON. Preview:")
        print(json.dumps(new_json, indent=2, ensure_ascii=False)[:3000])
        return 0

    # Backup the old file
    if out_path.exists():
        backup = out_path.with_suffix(f".json.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        out_path.rename(backup)
        log.info(f"Backed up old JSON to: {backup}")

    out_path.write_text(json.dumps(new_json, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Wrote new JSON: {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")

    # 4. Print summary
    print("\n" + "=" * 70)
    print("REFRESH COMPLETE — new confidence_winrate_data.json")
    print("=" * 70)
    print(f"{'Strategy':<25} {'Trades':>8} {'WinRate':>10} {'Recommendation':>16}")
    print("-" * 65)
    for s in strategy_summary:
        print(f"{s['strategy']:<25} {s['trades']:>8} {s['win_rate']:>10.1%} "
              f"{s['recommendation']:>16}")
    print("-" * 65)
    print(f"Total trades: {result['metadata']['n_total_trades']}")
    print(f"JSON path:    {out_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Refresh confidence_winrate_data.json from a real backtest.")
    p.add_argument("--pairs", nargs="*", default=DEFAULT_PAIRS)
    p.add_argument("--timeframes", nargs="*", default=DEFAULT_TIMEFRAMES)
    # FIX: default to DEFAULT_SKIP so known-broken strategies are not executed.
    # Previously defaulted to [] which caused refresh to spam 1000s of warnings.
    p.add_argument("--skip", nargs="*", default=DEFAULT_SKIP,
                   help=f"Strategies to skip (default: {DEFAULT_SKIP})")
    p.add_argument("--include-all", action="store_true",
                   help="Don't skip any strategy (overrides --skip). "
                        "WARNING: pin_bar/multi_pa/sd_zones_scored are known to be slow/broken.")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--bars", type=int, default=None,
                   help="Cap each pair to last N bars (faster)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print new JSON instead of writing it")
    p.add_argument("--out",
                   default=str(PROJECT_ROOT / "intelligence" / "confidence_winrate_data.json"),
                   help="Output path for the new JSON")
    args = p.parse_args()

    if args.include_all:
        args.skip = []
        log.warning("--include-all: running ALL strategies including known-broken ones "
                    "(pin_bar, multi_pa, sd_zones_scored). This will be slow and noisy.")

    out_path = Path(args.out)

    # Apply bars override via the runner's module-level setting
    if args.bars is not None:
        import scripts.winrate_tests.run_actual_backtest as _runner
        _runner._MAX_BARS_OVERRIDE = args.bars

    return refresh(
        pairs=args.pairs,
        timeframes=args.timeframes,
        skip_strategies=args.skip,
        quick=args.quick,
        dry_run=args.dry_run,
        out_path=out_path,
    )


if __name__ == "__main__":
    sys.exit(main())
