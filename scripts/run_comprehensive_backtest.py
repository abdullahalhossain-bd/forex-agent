#!/usr/bin/env python3
"""
scripts/run_comprehensive_backtest.py
======================================

⚠️  WARNING: This script uses per_strategy_tester.py which may have
residual look-ahead patterns. For HONEST results, use:
  python scripts/run_honest_validation.py

That script uses honest_backtest_engine.py (no look-ahead, realistic costs,
walk-forward validation, Monte Carlo, swap costs, Bonferroni correction).

Comprehensive backtest runner:
  • Auto-discovers all pairs from MT5 (or uses defaults)
  • Fetches data for each pair × each timeframe
  • Tests each strategy INDEPENDENTLY (per_strategy_tester)
  • Aggregates results across all pairs/timeframes
  • Calibrates the AdaptiveDecisionEngine with the results
  • Exports calibrated weights for the live trading system
  • Generates a final win-rate report (JSON + Markdown)

Usage:
    python scripts/run_comprehensive_backtest.py
    python scripts/run_comprehensive_backtest.py --pairs EURUSD,GBPUSD --timeframes H1,M15
    python scripts/run_comprehensive_backtest.py --max-candles 2000 --no-cache
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Any, Dict, List

# Make forex_ai importable regardless of where script is run from
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent  # forex_ai/
sys.path.insert(0, str(_PROJECT_ROOT))

import warnings
warnings.filterwarnings("ignore")

import argparse
import pandas as pd

from backtest.mt5_bulk_fetcher import MT5BulkFetcher
from backtest.per_strategy_tester import PerStrategyTester, StrategyResult
from analysis.adaptive_decision_engine import AdaptiveDecisionEngine
from utils.logger import get_logger

log = get_logger("comprehensive_bt")

# Output directory: next to the project, portable across OS
OUTPUT_DIR = _PROJECT_ROOT.parent / "download" / "backtest_results"


def parse_args():
    p = argparse.ArgumentParser(description="Comprehensive backtest")
    p.add_argument("--pairs", type=str, default="",
                   help="Comma-separated pairs (default: auto-discover)")
    p.add_argument("--timeframes", type=str, default="M15,H1,H4",
                   help="Comma-separated timeframes (default: M15,H1,H4)")
    p.add_argument("--max-candles", type=int, default=2000,
                   help="Candles per pair/timeframe (default: 2000)")
    p.add_argument("--no-cache", action="store_true",
                   help="Bypass disk cache")
    p.add_argument("--categories", type=str, default="forex_majors,forex_crosses,metals",
                   help="Categories to include (default: forex_majors,forex_crosses,metals)")
    p.add_argument("--max-pairs", type=int, default=10,
                   help="Max pairs to test (default: 10)")
    return p.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  COMPREHENSIVE BACKTEST — All Pairs × All Timeframes × All Strategies")
    print("=" * 70)
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Timeframes: {args.timeframes}")
    print(f"  Max candles per pair/tf: {args.max_candles}")
    print(f"  Use cache: {not args.no_cache}")
    print(f"  Output dir: {OUTPUT_DIR}")

    # ── Step 1: Discover pairs ────────────────────────────
    print("\n── Step 1: Discovering pairs ──")
    fetcher = MT5BulkFetcher()
    all_pairs = fetcher.discover_pairs()
    print(f"  Total discovered: {len(all_pairs)} pairs")

    # Filter by category
    if args.categories:
        cats = [c.strip() for c in args.categories.split(",")]
        all_pairs = fetcher.filter_pairs(all_pairs, categories=cats)
        print(f"  After category filter ({cats}): {len(all_pairs)} pairs")

    # Filter by explicit pair list
    if args.pairs:
        wanted = [p.strip().upper() for p in args.pairs.split(",")]
        all_pairs = [p for p in all_pairs if p.symbol.upper() in wanted]
        print(f"  After explicit filter: {len(all_pairs)} pairs")

    # Limit number of pairs
    if args.max_pairs and len(all_pairs) > args.max_pairs:
        all_pairs = all_pairs[:args.max_pairs]
        print(f"  Capped to {args.max_pairs} pairs for this run")

    if not all_pairs:
        print("  No pairs to test — exiting")
        return

    print(f"\n  Pairs to test: {[p.symbol for p in all_pairs]}")

    # ── Step 2: Run per-strategy backtest for each (pair, timeframe) ──
    print("\n── Step 2: Running per-strategy backtests ──")
    timeframes = [t.strip() for t in args.timeframes.split(",")]
    tester = PerStrategyTester(max_hold_bars=80)

    all_results: List[Dict[str, Any]] = []
    aggregated_by_strategy: Dict[str, Dict[str, Any]] = {}

    total_combos = len(all_pairs) * len(timeframes)
    done = 0

    for pair in all_pairs:
        for tf in timeframes:
            done += 1
            print(f"\n  [{done}/{total_combos}] {pair.symbol} {tf}...")
            t0 = time.time()

            # Fetch data
            fetch_result = fetcher.fetch(pair.symbol, tf,
                                          n_candles=args.max_candles,
                                          use_cache=not args.no_cache)
            if fetch_result.df is None or len(fetch_result.df) < 100:
                print(f"    SKIP: insufficient data ({fetch_result.n_candles} candles)")
                continue

            df = fetch_result.df
            print(f"    Data: {len(df)} candles ({fetch_result.source})")

            # Run all strategies
            try:
                results = tester.run_all(df, pair=pair.symbol, timeframe=tf)
            except Exception as e:
                print(f"    ERROR: {e}")
                continue

            elapsed = time.time() - t0
            n_trades_total = sum(r.n_trades for r in results["strategies"].values())
            print(f"    Done in {elapsed:.1f}s — {n_trades_total} total trades")

            # Save per-pair-tf result
            pair_tf_result = {
                "pair": pair.symbol,
                "timeframe": tf,
                "category": pair.category,
                "n_bars": len(df),
                "data_source": fetch_result.source,
                "strategies": {},
            }
            for sname, sresult in results["strategies"].items():
                pair_tf_result["strategies"][sname] = {
                    "n_trades": sresult.n_trades,
                    "win_rate": round(sresult.win_rate, 4),
                    "avg_r": round(sresult.avg_r, 3),
                    "total_r": round(sresult.total_r, 2),
                    "profit_factor": round(sresult.profit_factor, 2) if sresult.profit_factor != float("inf") else None,
                    "by_confidence": sresult.by_confidence,
                    "by_tactic": sresult.by_tactic,
                    "by_direction": sresult.by_direction,
                }
                # Aggregate
                if sname not in aggregated_by_strategy:
                    aggregated_by_strategy[sname] = {
                        "n_trades": 0, "n_wins": 0, "r_values": [],
                        "by_confidence": {}, "by_tactic": {},
                    }
                agg = aggregated_by_strategy[sname]
                agg["n_trades"] += sresult.n_trades
                # Approximate wins from win_rate
                agg["n_wins"] += int(sresult.win_rate * sresult.n_trades)
                if sresult.n_trades > 0:
                    agg["r_values"].append((sresult.avg_r, sresult.n_trades))

                # Aggregate by confidence
                for conf, cdata in sresult.by_confidence.items():
                    if conf not in agg["by_confidence"]:
                        agg["by_confidence"][conf] = {"n_trades": 0, "n_wins": 0}
                    agg["by_confidence"][conf]["n_trades"] += cdata.get("n_trades", 0)
                    agg["by_confidence"][conf]["n_wins"] += int(
                        cdata.get("win_rate", 0) * cdata.get("n_trades", 0))

                # Aggregate by tactic
                for tac, tdata in sresult.by_tactic.items():
                    if tac not in agg["by_tactic"]:
                        agg["by_tactic"][tac] = {"n_trades": 0, "n_wins": 0}
                    agg["by_tactic"][tac]["n_trades"] += tdata.get("n_trades", 0)
                    agg["by_tactic"][tac]["n_wins"] += int(
                        tdata.get("win_rate", 0) * tdata.get("n_trades", 0))

            all_results.append(pair_tf_result)

    # ── Step 3: Compute final aggregated stats per strategy ──
    print("\n── Step 3: Computing aggregated strategy stats ──")
    final_strategies = {}
    for sname, agg in aggregated_by_strategy.items():
        n = agg["n_trades"]
        if n == 0:
            final_strategies[sname] = {
                "n_trades": 0, "win_rate": 0.0, "avg_r": 0.0,
                "profit_factor": 0.0,
                "by_confidence": {}, "by_tactic": {},
            }
            continue
        # Weighted average R
        total_weight = sum(w for _, w in agg["r_values"])
        if total_weight > 0:
            avg_r = sum(r * w for r, w in agg["r_values"]) / total_weight
        else:
            avg_r = 0.0

        # Win rate
        wr = agg["n_wins"] / n if n > 0 else 0.0

        # Confidence breakdown
        conf_breakdown = {}
        for conf, cdata in agg["by_confidence"].items():
            cn = cdata["n_trades"]
            if cn > 0:
                conf_breakdown[conf] = {
                    "n_trades": cn,
                    "win_rate": round(cdata["n_wins"] / cn, 4),
                }

        # Tactic breakdown
        tac_breakdown = {}
        for tac, tdata in agg["by_tactic"].items():
            tn = tdata["n_trades"]
            if tn > 0:
                tac_breakdown[tac] = {
                    "n_trades": tn,
                    "win_rate": round(tdata["n_wins"] / tn, 4),
                }

        final_strategies[sname] = {
            "n_trades": n,
            "n_wins": agg["n_wins"],
            "win_rate": round(wr, 4),
            "avg_r": round(avg_r, 3),
            "by_confidence": conf_breakdown,
            "by_tactic": tac_breakdown,
        }

    # ── Step 4: Calibrate adaptive decision engine ────────
    print("\n── Step 4: Calibrating Adaptive Decision Engine ──")
    engine = AdaptiveDecisionEngine(mode="confluence")
    n_calibrated = engine.load_backtest_results({"strategies": final_strategies})

    # Export calibrated weights
    weights_path = OUTPUT_DIR / "calibrated_weights.json"
    engine.export_weights(str(weights_path))
    print(f"  Calibrated {n_calibrated} strategies")
    print(f"  Weights saved to: {weights_path}")

    # ── Step 5: Generate final report ─────────────────────
    print("\n── Step 5: Generating final report ──")
    final_report = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pairs_tested": [p.symbol for p in all_pairs],
            "timeframes_tested": timeframes,
            "n_candles_per_pair_tf": args.max_candles,
            "total_combos": total_combos,
            "successful_combos": len(all_results),
        },
        "strategies": final_strategies,
        "per_pair_timeframe": all_results,
        "recommendations": _generate_recommendations(final_strategies),
    }

    json_path = OUTPUT_DIR / "backtest_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False, default=str)
    print(f"  JSON report: {json_path}")

    md_path = OUTPUT_DIR / "backtest_report.md"
    _write_markdown_report(final_report, md_path)
    print(f"  Markdown report: {md_path}")

    # ── Step 6: Print summary ─────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY — Strategy Win Rates (aggregated across all pairs/TFs)")
    print("=" * 70)
    print(f"  {'Strategy':<25} {'Trades':>7} {'Win Rate':>9} {'Avg R':>7} {'Recommendation':<30}")
    print(f"  {'-'*25} {'-'*7} {'-'*9} {'-'*7} {'-'*30}")

    ranked = sorted(
        [(name, data) for name, data in final_strategies.items() if data["n_trades"] > 0],
        key=lambda x: x[1]["win_rate"],
        reverse=True,
    )
    for name, data in ranked:
        rec = "✅ USE in live system" if data["win_rate"] >= 0.50 and data["n_trades"] >= 10 \
            else "⚠️  Use with caution" if data["win_rate"] >= 0.40 \
            else "❌ Disable or fix"
        print(f"  {name:<25} {data['n_trades']:>7} {data['win_rate']*100:>8.1f}% "
              f"{data['avg_r']:>+7.2f} {rec}")

    no_trades = [name for name, data in final_strategies.items() if data["n_trades"] == 0]
    if no_trades:
        print(f"\n  Strategies with 0 trades (need different data/params):")
        for n in no_trades:
            print(f"    - {n}")

    print("\n" + "=" * 70)
    print(f"  Reports saved to: {OUTPUT_DIR}")
    print("=" * 70)

    fetcher.shutdown()


def _generate_recommendations(strategies: Dict[str, Any]) -> Dict[str, Any]:
    """Generate actionable recommendations for the live trading system."""
    recommendations = {
        "use_strategies": [],
        "use_with_caution": [],
        "disable_or_fix": [],
        "best_confidence_levels": {},
        "best_tactics": {},
        "suggested_mode": "confluence",
    }

    for name, data in strategies.items():
        wr = data.get("win_rate", 0)
        n = data.get("n_trades", 0)
        if n == 0:
            recommendations["disable_or_fix"].append({
                "strategy": name, "reason": "No trades generated — check params"})
        elif wr >= 0.50 and n >= 10:
            recommendations["use_strategies"].append({
                "strategy": name, "win_rate": wr, "n_trades": n})
        elif wr >= 0.40:
            recommendations["use_with_caution"].append({
                "strategy": name, "win_rate": wr, "n_trades": n})
        else:
            recommendations["disable_or_fix"].append({
                "strategy": name, "win_rate": wr, "n_trades": n,
                "reason": f"WR {wr*100:.1f}% below 40%"})

        # Best confidence level
        best_conf = None
        best_conf_wr = 0
        for conf, cdata in data.get("by_confidence", {}).items():
            if cdata.get("win_rate", 0) > best_conf_wr and cdata.get("n_trades", 0) >= 5:
                best_conf_wr = cdata["win_rate"]
                best_conf = conf
        if best_conf:
            recommendations["best_confidence_levels"][name] = {
                "level": best_conf, "win_rate": best_conf_wr,
            }

        # Best tactic
        best_tactic = None
        best_tactic_wr = 0
        for tac, tdata in data.get("by_tactic", {}).items():
            if tdata.get("win_rate", 0) > best_tactic_wr and tdata.get("n_trades", 0) >= 5:
                best_tactic_wr = tdata["win_rate"]
                best_tactic = tac
        if best_tactic:
            recommendations["best_tactics"][name] = {
                "tactic": best_tactic, "win_rate": best_tactic_wr,
            }

    return recommendations


def _write_markdown_report(report: Dict[str, Any], path: Path):
    """Write a human-readable Markdown report."""
    lines = [
        "# Comprehensive Backtest Report",
        "",
        f"**Generated:** {report['metadata']['timestamp']}  ",
        f"**Pairs tested:** {len(report['metadata']['pairs_tested'])}  ",
        f"**Timeframes:** {', '.join(report['metadata']['timeframes_tested'])}  ",
        f"**Total combinations:** {report['metadata']['total_combos']}  ",
        f"**Successful runs:** {report['metadata']['successful_combos']}",
        "",
        "---",
        "",
        "## Strategy Performance Summary (Aggregated)",
        "",
        "| Strategy | Trades | Win Rate | Avg R | Best Confidence | Best Tactic | Recommendation |",
        "|----------|--------|----------|-------|-----------------|-------------|----------------|",
    ]

    strategies = report["strategies"]
    recs = report["recommendations"]
    ranked = sorted(
        [(n, d) for n, d in strategies.items() if d["n_trades"] > 0],
        key=lambda x: x[1]["win_rate"], reverse=True,
    )
    for name, data in ranked:
        wr = data["win_rate"]
        n = data["n_trades"]
        ar = data["avg_r"]
        bc = recs["best_confidence_levels"].get(name, {}).get("level", "-")
        bt = recs["best_tactics"].get(name, {}).get("tactic", "-")
        if any(s["strategy"] == name for s in recs["use_strategies"]):
            r = "✅ Use"
        elif any(s["strategy"] == name for s in recs["use_with_caution"]):
            r = "⚠️ Caution"
        else:
            r = "❌ Disable"
        lines.append(f"| `{name}` | {n} | {wr*100:.1f}% | {ar:+.2f} | {bc} | {bt} | {r} |")

    lines.extend([
        "",
        "## Strategies With 0 Trades",
        "",
    ])
    no_trades = [n for n, d in strategies.items() if d["n_trades"] == 0]
    if no_trades:
        for n in no_trades:
            lines.append(f"- `{n}` — needs different data or parameter tuning")
    else:
        lines.append("_None — all strategies produced trades._")

    lines.extend([
        "",
        "## Per-Confidence-Level Win Rates",
        "",
    ])
    for name, data in ranked:
        if data.get("by_confidence"):
            lines.append(f"### {name}")
            lines.append("")
            lines.append("| Confidence | Trades | Win Rate |")
            lines.append("|------------|--------|----------|")
            for conf, cdata in sorted(data["by_confidence"].items()):
                lines.append(f"| {conf} | {cdata['n_trades']} | {cdata['win_rate']*100:.1f}% |")
            lines.append("")

    lines.extend([
        "## Per-Tactic Win Rates",
        "",
    ])
    for name, data in ranked:
        if data.get("by_tactic"):
            lines.append(f"### {name}")
            lines.append("")
            lines.append("| Tactic | Trades | Win Rate |")
            lines.append("|--------|--------|----------|")
            for tac, tdata in sorted(data["by_tactic"].items()):
                lines.append(f"| `{tac}` | {tdata['n_trades']} | {tdata['win_rate']*100:.1f}% |")
            lines.append("")

    lines.extend([
        "---",
        "",
        "## Recommendations for Live Trading System",
        "",
        f"**Suggested decision mode:** `{recs['suggested_mode']}`",
        "",
        "### ✅ Use in live system (WR ≥ 50%, n ≥ 10)",
        "",
    ])
    for s in recs["use_strategies"]:
        lines.append(f"- `{s['strategy']}` — WR {s['win_rate']*100:.1f}%, {s['n_trades']} trades")

    lines.extend(["", "### ⚠️ Use with caution (40% ≤ WR < 50%)", ""])
    for s in recs["use_with_caution"]:
        lines.append(f"- `{s['strategy']}` — WR {s['win_rate']*100:.1f}%, {s['n_trades']} trades")

    lines.extend(["", "### ❌ Disable or fix", ""])
    for s in recs["disable_or_fix"]:
        reason = s.get("reason", "low win rate")
        lines.append(f"- `{s['strategy']}` — {reason}")

    lines.extend([
        "",
        "---",
        "",
        "## Files Generated",
        "",
        "- `calibrated_weights.json` — load this into AdaptiveDecisionEngine",
        "- `backtest_report.json` — full machine-readable results",
        "- `backtest_report.md` — this document",
        "",
        "## Next Steps",
        "",
        "1. Copy `calibrated_weights.json` to your live trading system",
        "2. Initialize `AdaptiveDecisionEngine` with these weights",
        "3. Use `mode='confluence'` for normal trading (recommended)",
        "4. Re-run this backtest monthly to recalibrate as market conditions change",
        "",
    ])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
