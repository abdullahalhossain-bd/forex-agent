#!/usr/bin/env python3
"""
scripts/run_honest_validation.py
=================================

CRITICAL: Re-runs ALL strategies through the HONEST backtest engine
(no look-ahead, realistic costs, walk-forward, Monte Carlo).

This is the validation pipeline that MUST pass before any live deployment.

If a strategy fails this validation, it CANNOT go live — no exceptions.

Output:
  download/backtest_analysis/HONEST_VALIDATION_REPORT.md
  download/backtest_analysis/honest_validation.json

Usage:
    python scripts/run_honest_validation.py
    python scripts/run_honest_validation.py --pairs EURUSD,GBPUSD --timeframes H1,H4
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/ → my-project/ (parent of forex_ai)
# Add forex_ai itself so internal imports like `from utils.logger import` work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "forex_ai"))
import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
import pandas as pd

from backtest.honest_backtest_engine import (
    HonestBacktester, IncrementalZoneDetector,
    MonteCarloValidator, WalkForwardValidator, DeploymentGate,
    HonestResult, HonestTrade,
)
from backtest.mt5_bulk_fetcher import MT5BulkFetcher
from utils.logger import get_logger

log = get_logger("honest_validation")

# Output dir is always next to the forex_ai project (portable)
_OUTPUT_ROOT = Path(__file__).resolve().parent.parent  # my-project/
OUTPUT_DIR = _OUTPUT_ROOT / "download" / "backtest_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════
#  STRATEGY DEFINITIONS (no look-ahead versions)
# ════════════════════════════════════════════════════════════════

def make_sr_bounce_strategy(pip_size: float = 0.0001):
    """S/R bounce strategy using INCREMENTAL zone detection (no look-ahead)."""
    detector = IncrementalZoneDetector(
        window_size=100, swing_lookback=5,
        zone_tolerance_pips=10, pip_size=pip_size,
    )

    def strategy(visible_df: pd.DataFrame, current_idx: int) -> Optional[Dict[str, Any]]:
        if len(visible_df) < 50:
            return None
        zones = detector.zones_at_bar(visible_df, current_idx)
        if not zones["support"] and not zones["resistance"]:
            return None

        current_close = float(visible_df.iloc[-1]["close"])
        # ATR using only past 14 bars
        if len(visible_df) < 14:
            return None
        recent = visible_df.iloc[-14:]
        atr = float(np.mean(recent["high"].values - recent["low"].values))

        # Long at support
        for s in zones["support"][:3]:
            if abs(current_close - s["price"]) < atr * 0.5:
                return {
                    "direction": "long",
                    "entry": current_close,
                    "stop_loss": s["price"] - atr * 1.5,
                    "take_profit": current_close + atr * 3.0,
                }
        # Short at resistance
        for r in zones["resistance"][:3]:
            if abs(current_close - r["price"]) < atr * 0.5:
                return {
                    "direction": "short",
                    "entry": current_close,
                    "stop_loss": r["price"] + atr * 1.5,
                    "take_profit": current_close - atr * 3.0,
                }
        return None

    return strategy


def make_sr_resistance_only_strategy(pip_size: float = 0.0001):
    """S/R resistance-only bounce (your 'best' setup — let's test it honestly)."""
    detector = IncrementalZoneDetector(
        window_size=100, swing_lookback=5,
        zone_tolerance_pips=10, pip_size=pip_size,
    )

    def strategy(visible_df: pd.DataFrame, current_idx: int) -> Optional[Dict[str, Any]]:
        if len(visible_df) < 50:
            return None
        zones = detector.zones_at_bar(visible_df, current_idx)
        if not zones["resistance"]:
            return None

        current_close = float(visible_df.iloc[-1]["close"])
        if len(visible_df) < 14:
            return None
        recent = visible_df.iloc[-14:]
        atr = float(np.mean(recent["high"].values - recent["low"].values))

        for r in zones["resistance"][:3]:
            if abs(current_close - r["price"]) < atr * 0.5:
                return {
                    "direction": "short",
                    "entry": current_close,
                    "stop_loss": r["price"] + atr * 1.5,
                    "take_profit": current_close - atr * 3.0,
                }
        return None

    return strategy


def make_donchian_breakout_strategy(pip_size: float = 0.0001):
    """Donchian channel breakout — simple, hard to overfit."""
    def strategy(visible_df: pd.DataFrame, current_idx: int) -> Optional[Dict[str, Any]]:
        if len(visible_df) < 55:
            return None
        # 50-bar Donchian channel (using only past data)
        window = visible_df.iloc[-51:-1]  # bars 0..49 (exclude current)
        upper = float(window["high"].max())
        lower = float(window["low"].min())
        current_close = float(visible_df.iloc[-1]["close"])
        current_high = float(visible_df.iloc[-1]["high"])
        current_low = float(visible_df.iloc[-1]["low"])

        atr_window = visible_df.iloc[-14:]
        atr = float(np.mean(atr_window["high"].values - atr_window["low"].values))

        # Long breakout
        if current_close > upper:
            return {
                "direction": "long",
                "entry": current_close,
                "stop_loss": current_close - 2 * atr,
                "take_profit": current_close + 4 * atr,
            }
        # Short breakdown
        if current_close < lower:
            return {
                "direction": "short",
                "entry": current_close,
                "stop_loss": current_close + 2 * atr,
                "take_profit": current_close - 4 * atr,
            }
        return None

    return strategy


def make_random_strategy(pip_size: float = 0.0001):
    """Random strategy — BASELINE. If your strategies can't beat this, they're noise."""
    rng_state = [0]  # mutable for closure

    def strategy(visible_df: pd.DataFrame, current_idx: int) -> Optional[Dict[str, Any]]:
        if len(visible_df) < 14:
            return None
        # Use deterministic pseudo-random based on bar index (for reproducibility)
        seed = (current_idx * 7919) % 100
        if seed > 5:  # only 5% of bars produce signal
            return None
        recent = visible_df.iloc[-14:]
        atr = float(np.mean(recent["high"].values - recent["low"].values))
        current_close = float(visible_df.iloc[-1]["close"])
        direction = "long" if (seed % 2 == 0) else "short"
        return {
            "direction": direction,
            "entry": current_close,
            "stop_loss": current_close - 2 * atr if direction == "long" else current_close + 2 * atr,
            "take_profit": current_close + 4 * atr if direction == "long" else current_close - 4 * atr,
        }

    return strategy


# All strategies to validate
STRATEGIES = {
    "sr_bounce":           make_sr_bounce_strategy,
    "sr_resistance_only":  make_sr_resistance_only_strategy,  # your "best"
    "donchian_breakout":   make_donchian_breakout_strategy,
    "random_baseline":     make_random_strategy,
}


# ════════════════════════════════════════════════════════════════
#  MAIN VALIDATION
# ════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", default="EURUSD,GBPUSD,XAUUSD",
                   help="Comma-separated pairs")
    p.add_argument("--timeframes", default="M15,H1,H4")
    p.add_argument("--max-candles", type=int, default=3000)
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    pairs = [p.strip() for p in args.pairs.split(",")]
    timeframes = [t.strip() for t in args.timeframes.split(",")]

    print("=" * 75)
    print("  🪓 HONEST VALIDATION — No Look-Ahead, Realistic Costs, Walk-Forward")
    print("=" * 75)
    print(f"  Pairs: {pairs}")
    print(f"  Timeframes: {timeframes}")
    print(f"  Strategies: {list(STRATEGIES.keys())}")
    print(f"  Candles per pair/TF: {args.max_candles}")

    # Total number of comparisons for Bonferroni
    n_comparisons = len(pairs) * len(timeframes) * len(STRATEGIES)
    print(f"  Total comparisons (for Bonferroni): {n_comparisons}")
    bonferroni_alpha = 0.05 / n_comparisons
    print(f"  Bonferroni alpha: {bonferroni_alpha:.6f}")

    fetcher = MT5BulkFetcher()
    bt = HonestBacktester()
    mc_validator = MonteCarloValidator(n_simulations=2000)
    wf_validator = WalkForwardValidator(
        train_bars=min(1500, args.max_candles // 2),
        test_bars=min(750, args.max_candles // 4),
        step_bars=min(750, args.max_candles // 4),
    )

    all_results: List[Dict[str, Any]] = []
    summary_table: List[Dict[str, Any]] = []

    total_combos = len(pairs) * len(timeframes) * len(STRATEGIES)
    done = 0

    for pair in pairs:
        # Determine pip size
        if "JPY" in pair:
            pip = 0.01
        elif "XAU" in pair:
            pip = 0.1
        elif any(idx in pair for idx in ["US30", "NAS100", "SPX500"]):
            pip = 1.0
        else:
            pip = 0.0001

        for tf in timeframes:
            # Fetch data once per (pair, tf)
            fetch_result = fetcher.fetch(pair, tf, n_candles=args.max_candles,
                                          use_cache=not args.no_cache)
            if fetch_result.df is None or len(fetch_result.df) < 200:
                print(f"  SKIP {pair} {tf}: insufficient data")
                continue
            df = fetch_result.df

            for strat_name, strat_factory in STRATEGIES.items():
                done += 1
                print(f"\n  [{done}/{total_combos}] {pair} {tf} {strat_name}...")

                strategy_fn = strat_factory(pip_size=pip)

                # Honest backtest
                t0 = time.time()
                honest = bt.test_strategy(df, strategy_fn, pair=pair,
                                          n_comparisons=n_comparisons)
                bt_time = time.time() - t0

                # Monte Carlo (only if enough trades)
                if honest.n_trades >= 30:
                    mc = mc_validator.validate(honest.trades)
                else:
                    mc = {"error": "insufficient trades for MC", "n_trades": honest.n_trades}

                # Walk-forward (only if enough data)
                try:
                    wf = wf_validator.validate(df, strategy_fn, pair=pair)
                except Exception as e:
                    wf = {"error": str(e)}

                # Deployment gate
                gate = DeploymentGate.evaluate(honest, mc, wf)

                # Build result entry
                result_entry = {
                    "pair": pair, "timeframe": tf, "strategy": strat_name,
                    "n_trades": honest.n_trades,
                    "net_win_rate": round(honest.win_rate, 4),
                    "gross_win_rate": round(honest.gross_win_rate, 4),
                    "avg_net_r": round(honest.avg_net_r, 3),
                    "total_net_r": round(honest.total_net_r, 2),
                    "profit_factor": round(honest.profit_factor, 2) if honest.profit_factor != float("inf") else None,
                    "max_drawdown_r": round(honest.max_drawdown_r, 2),
                    "n_gap_losses": honest.n_gap_losses,
                    "avg_cost_pips": round(honest.avg_cost_per_trade_pips, 2),
                    "p_value": round(honest.p_value, 4),
                    "is_significant": honest.is_significant,
                    "bonferroni_significant": honest.bonferroni_significant,
                    "mc_probability_of_ruin": mc.get("probability_of_ruin", 1.0) if "error" not in mc else None,
                    "mc_95th_pct_drawdown": mc.get("95th_percentile_drawdown", 1.0) if "error" not in mc else None,
                    "mc_wr_ci_low": mc.get("win_rate_ci_low", 0) if "error" not in mc else None,
                    "mc_wr_ci_high": mc.get("win_rate_ci_high", 0) if "error" not in mc else None,
                    "wf_verdict": wf.get("verdict", "fail") if "error" not in wf else "error",
                    "wf_oos_wr": wf.get("out_of_sample_avg_wr", 0) if "error" not in wf else None,
                    "wf_degradation": wf.get("degradation", 1.0) if "error" not in wf else None,
                    "deployment_verdict": gate["verdict"],
                    "blocking_failures": gate["blocking_failures"],
                    "backtest_time_seconds": round(bt_time, 1),
                }
                summary_table.append(result_entry)

                # Print summary
                status = "✅ APPROVED" if gate["can_deploy_live"] else "❌ BLOCKED"
                print(f"    Trades={honest.n_trades} NetWR={honest.win_rate*100:.1f}% "
                      f"R={honest.avg_net_r:+.2f} PF={honest.profit_factor:.2f} "
                      f"p={honest.p_value:.3f} {status}")

    # Save JSON
    json_path = OUTPUT_DIR / "honest_validation.json"

    def _json_default(obj):
        """Handle numpy types in JSON serialization."""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "pairs": pairs, "timeframes": timeframes,
                "n_comparisons": n_comparisons,
                "bonferroni_alpha": bonferroni_alpha,
                "candles_per_pair_tf": args.max_candles,
            },
            "results": summary_table,
        }, f, indent=2, default=_json_default)
    print(f"\n  JSON: {json_path}")

    # Generate Markdown report
    _write_markdown_report(summary_table, n_comparisons, bonferroni_alpha, args)
    print(f"  Report: {OUTPUT_DIR / 'HONEST_VALIDATION_REPORT.md'}")

    # Print final summary
    print("\n" + "=" * 75)
    print("  FINAL VERDICT — HONEST VALIDATION")
    print("=" * 75)
    approved = [r for r in summary_table if "APPROVED" in r["deployment_verdict"]]
    blocked = [r for r in summary_table if "BLOCKED" in r["deployment_verdict"]]
    print(f"\n  Approved for live: {len(approved)}/{len(summary_table)}")
    print(f"  Blocked:           {len(blocked)}/{len(summary_table)}")

    if not approved:
        print("\n  ⚠️  NO STRATEGY PASSED HONEST VALIDATION.")
        print("  This is the truth — your previous 'Tier 1' setups had look-ahead bias.")
        print("  Do NOT deploy anything with real money until a strategy passes.")
    else:
        print(f"\n  ✅ Approved strategies:")
        for r in approved:
            print(f"    - {r['pair']} {r['timeframe']} {r['strategy']} "
                  f"(WR={r['net_win_rate']*100:.1f}%, n={r['n_trades']})")

    # Compare to random baseline
    print(f"\n── Random Baseline Comparison ──")
    random_results = [r for r in summary_table if r["strategy"] == "random_baseline"]
    strat_results = [r for r in summary_table if r["strategy"] != "random_baseline"]
    if random_results and strat_results:
        avg_random_wr = np.mean([r["net_win_rate"] for r in random_results])
        avg_strat_wr = np.mean([r["net_win_rate"] for r in strat_results])
        print(f"  Average random strategy WR:  {avg_random_wr*100:.1f}%")
        print(f"  Average your strategies WR:  {avg_strat_wr*100:.1f}%")
        if avg_strat_wr <= avg_random_wr + 0.05:
            print(f"  ⚠️  YOUR STRATEGIES ARE NO BETTER THAN RANDOM.")
            print(f"  Any 'edge' you saw was look-ahead bias + multiple comparisons noise.")

    print("\n" + "=" * 75)
    fetcher.shutdown()


def _write_markdown_report(results, n_comparisons, bonferroni_alpha, args):
    """Write the honest validation Markdown report."""
    md = []
    md.append("# 🪓 HONEST VALIDATION REPORT — No Look-Ahead, Realistic Costs")
    md.append("")
    md.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ")
    md.append(f"**Comparisons made:** {n_comparisons}  ")
    md.append(f"**Bonferroni alpha:** {bonferroni_alpha:.6f} (must be < this for significance)  ")
    md.append(f"**Candles per pair/TF:** {args.max_candles}")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 🚨 What This Report Does")
    md.append("")
    md.append("This is the **honest** backtest. Unlike previous backtests:")
    md.append("- ❌ NO look-ahead bias (zones computed incrementally)")
    md.append("- ❌ NO unrealistic costs (real spread + commission + slippage + gaps)")
    md.append("- ❌ NO cherry-picking (random baseline included for comparison)")
    md.append("- ✅ Walk-forward OOS validation")
    md.append("- ✅ Monte Carlo simulation")
    md.append("- ✅ Bonferroni correction for multiple comparisons")
    md.append("- ✅ Deployment gate (must pass ALL checks)")
    md.append("")
    md.append("---")
    md.append("")

    md.append("## 📊 Results Summary")
    md.append("")
    md.append("| Pair | TF | Strategy | Trades | Net WR | Gross WR | Avg R | PF | p-value | Bonferroni? | Verdict |")
    md.append("|------|----|----------|--------|--------|----------|-------|-----|---------|-------------|---------|")
    for r in sorted(results, key=lambda x: x["net_win_rate"], reverse=True):
        bonf = "✅" if r["bonferroni_significant"] else "❌"
        verdict = "✅ APPROVED" if "APPROVED" in r["deployment_verdict"] else "❌ BLOCKED"
        md.append(f"| {r['pair']} | {r['timeframe']} | `{r['strategy']}` | "
                  f"{r['n_trades']} | {r['net_win_rate']*100:.1f}% | "
                  f"{r['gross_win_rate']*100:.1f}% | {r['avg_net_r']:+.2f} | "
                  f"{r['profit_factor'] or '—'} | {r['p_value']:.3f} | "
                  f"{bonf} | {verdict} |")
    md.append("")
    md.append("---")
    md.append("")

    md.append("## 🎯 Random Baseline Comparison")
    md.append("")
    md.append("**The most important test in this report.**")
    md.append("")
    md.append("If your strategies can't beat the random baseline after costs, they have NO edge.")
    md.append("")
    random_results = [r for r in results if r["strategy"] == "random_baseline"]
    strat_results = [r for r in results if r["strategy"] != "random_baseline"]
    if random_results and strat_results:
        avg_random = np.mean([r["net_win_rate"] for r in random_results])
        avg_strat = np.mean([r["net_win_rate"] for r in strat_results])
        md.append(f"- **Random baseline avg WR:** {avg_random*100:.1f}%")
        md.append(f"- **Your strategies avg WR:** {avg_strat*100:.1f}%")
        md.append(f"- **Difference:** {(avg_strat - avg_random)*100:+.1f}%")
        md.append("")
        if avg_strat <= avg_random + 0.05:
            md.append("### ⚠️ VERDICT: Your strategies are NO BETTER than random.")
            md.append("")
            md.append("Any 'edge' you saw in previous backtests was:")
            md.append("1. Look-ahead bias (zones computed with future data)")
            md.append("2. Multiple comparisons noise (2400+ tests, expect ~120 false positives)")
            md.append("3. Unrealistic costs (1 pip spread vs real 3-5 pips)")
            md.append("")
        else:
            md.append("### ✅ Your strategies beat random — but may still fail deployment gate.")
    md.append("")
    md.append("---")
    md.append("")

    md.append("## 🚦 Deployment Gate Results")
    md.append("")
    md.append("Each strategy must pass ALL 7 checks to deploy live:")
    md.append("")
    md.append("1. ≥100 trades (sufficient sample)")
    md.append("2. Win rate CI lower bound > 50%")
    md.append("3. Bonferroni-significant (p < 0.05/n_comparisons)")
    md.append("4. Walk-forward verdict = 'pass'")
    md.append("5. Monte Carlo probability of ruin < 5%")
    md.append("6. 95th percentile drawdown < 25%")
    md.append("7. Profit factor > 1.30")
    md.append("")
    approved = [r for r in results if "APPROVED" in r["deployment_verdict"]]
    md.append(f"**Approved: {len(approved)}/{len(results)}**")
    md.append("")
    if approved:
        md.append("### ✅ Approved for Live Trading:")
        md.append("")
        for r in approved:
            md.append(f"- `{r['pair']} {r['timeframe']} {r['strategy']}` — "
                      f"WR={r['net_win_rate']*100:.1f}%, n={r['n_trades']}, "
                      f"PF={r['profit_factor']}")
    else:
        md.append("### ❌ NO STRATEGY APPROVED")
        md.append("")
        md.append("This is the truth. Do not deploy anything.")
    md.append("")
    md.append("---")
    md.append("")

    md.append("## 📋 Most Common Blocking Failures")
    md.append("")
    failure_counts: Dict[str, int] = {}
    for r in results:
        for f in r.get("blocking_failures", []):
            failure_counts[f] = failure_counts.get(f, 0) + 1
    if failure_counts:
        md.append("| Check | Times Failed |")
        md.append("|-------|--------------|")
        for check, count in sorted(failure_counts.items(), key=lambda x: x[1], reverse=True):
            md.append(f"| `{check}` | {count}/{len(results)} |")
    md.append("")
    md.append("---")
    md.append("")

    md.append("## 🎓 Lessons from This Report")
    md.append("")
    md.append("1. **Previous backtests were inflated** by look-ahead bias (zone detection)")
    md.append("2. **Small samples lie** — 7-11 trades at 85-100% WR is meaningless")
    md.append("3. **Multiple comparisons create false positives** — need Bonferroni correction")
    md.append("4. **Realistic costs kill marginal strategies** — 1 pip spread ≠ 1 pip in live")
    md.append("5. **Walk-forward is non-negotiable** — in-sample fit ≠ OOS performance")
    md.append("6. **Random baseline is the bar** — if you can't beat random, you have nothing")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 🚀 Next Steps (if any strategy approved)")
    md.append("")
    md.append("1. **Demo account for 3 months minimum** (not 4 weeks)")
    md.append("2. **Start with 0.01 lot** (micro) for first 50 live trades")
    md.append("3. **Use StrictRiskManager** (0.5% per trade, correlation limits)")
    md.append("4. **Re-validate monthly** with new data")
    md.append("5. **Hard stop**: if live WR drops 10% below backtest WR, halt and re-validate")
    md.append("")
    md.append("## ⚠️ If NO strategy approved")
    md.append("")
    md.append("Don't despair — this is normal. Most strategies don't work.")
    md.append("")
    md.append("Options:")
    md.append("1. **Get more data** — 10,000+ candles per pair/TF for proper walk-forward")
    md.append("2. **Try different strategy types** — trend following, mean reversion, momentum")
    md.append("3. **Accept that retail algo trading is extremely hard** — most fail")
    md.append("4. **Consider copy trading or managed accounts** if you can't find an edge")
    md.append("")

    report_path = OUTPUT_DIR / "HONEST_VALIDATION_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


if __name__ == "__main__":
    main()