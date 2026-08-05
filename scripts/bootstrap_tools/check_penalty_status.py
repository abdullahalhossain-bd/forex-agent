#!/usr/bin/env python3
"""
Check current status of pattern_stats.json and the Bayesian penalty
that would apply for each (pattern × pair × timeframe × regime) combo.

USAGE:
  python scripts/bootstrap_tools/check_penalty_status.py
  python scripts/bootstrap_tools/check_penalty_status.py --pair EURUSD
  python scripts/bootstrap_tools/check_penalty_status.py --pattern Hammer
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from core.constants import MEMORY_DIR
    from learning.confidence_engine import (
        MIN_SAMPLE_SIZE,
        BAYESIAN_PENALTY_FLOOR,
        BAYESIAN_PENALTY_SCALE,
    )
except Exception as e:
    print(f"ERROR: cannot import: {e}")
    sys.exit(1)

PATTERN_STATS_PATH = MEMORY_DIR / "pattern_stats.json"


def compute_penalty(sample_size: int, raw_score: float = 70.0,
                     bootstrap_mode: bool = False) -> float:
    """Mirror of ConfidenceEngine._bayesian_penalty()."""
    if sample_size >= MIN_SAMPLE_SIZE:
        return 0.0
    if sample_size == 0:
        uncertainty = 1.0
    else:
        uncertainty = 1.0 - math.sqrt(sample_size / MIN_SAMPLE_SIZE)
    deviation = max(0.0, raw_score - 50.0)
    penalty = -(BAYESIAN_PENALTY_FLOOR + BAYESIAN_PENALTY_SCALE * deviation) * uncertainty
    if bootstrap_mode:
        penalty *= 0.5
    return round(penalty, 1)


def main():
    parser = argparse.ArgumentParser(description="Check pattern_stats.json status")
    parser.add_argument("--pair", default="", help="Filter by pair (e.g. EURUSD)")
    parser.add_argument("--pattern", default="", help="Filter by pattern (e.g. Hammer)")
    parser.add_argument("--raw-score", type=float, default=70.0,
                        help="Raw confidence score to compute penalty for (default: 70)")
    parser.add_argument("--show-empty", action="store_true",
                        help="Show entries with 0 trades (hidden by default)")
    args = parser.parse_args()

    print(f"Pattern stats: {PATTERN_STATS_PATH}")
    if not PATTERN_STATS_PATH.exists():
        print("  ✗ File does not exist — system has ZERO pattern data")
        print("  → Every pattern will get the maximum Bayesian penalty")
        print("  → Run: python scripts/bootstrap_tools/generate_synthetic_samples.py")
        return

    with open(PATTERN_STATS_PATH, encoding="utf-8") as f:
        stats = json.load(f)

    print(f"  Total entries: {len(stats)}")
    print(f"  MIN_SAMPLE_SIZE: {MIN_SAMPLE_SIZE}")
    print(f"  BAYESIAN_PENALTY_FLOOR: {BAYESIAN_PENALTY_FLOOR}")
    print(f"  BAYESIAN_PENALTY_SCALE: {BAYESIAN_PENALTY_SCALE}")
    print(f"  Raw score for penalty calc: {args.raw_score}")
    print()

    # Categorize entries
    by_sample_size = {0: [], 1: [], 2: [], 3: [], "3+": []}
    for key, entry in stats.items():
        total = entry.get("total_trades", 0)
        if total == 0:
            by_sample_size[0].append((key, entry))
        elif total == 1:
            by_sample_size[1].append((key, entry))
        elif total == 2:
            by_sample_size[2].append((key, entry))
        elif total == 3:
            by_sample_size[3].append((key, entry))
        else:
            by_sample_size["3+"].append((key, entry))

    print("SAMPLE SIZE DISTRIBUTION:")
    print(f"  0 trades  (max penalty): {len(by_sample_size[0])}")
    print(f"  1 trade   (heavy penalty): {len(by_sample_size[1])}")
    print(f"  2 trades  (mild penalty): {len(by_sample_size[2])}")
    print(f"  3 trades  (no penalty):   {len(by_sample_size[3])}")
    print(f"  3+ trades (mature):       {len(by_sample_size['3+'])}")
    print()

    # Compute what penalty would apply right now
    print("CURRENT BAYESIAN PENALTY (assuming raw_score=70%, bootstrap mode):")
    for ss in [0, 1, 2, 3]:
        p = compute_penalty(ss, args.raw_score, bootstrap_mode=True)
        print(f"  sample_size={ss}: penalty = {p:+.1f}pp")
    print()

    # Show worst-offender patterns (0 trades)
    if by_sample_size[0] and args.show_empty:
        print("ENTRIES WITH 0 TRADES (top 20, will get max penalty):")
        for key, entry in by_sample_size[0][:20]:
            if args.pair and args.pair.upper() not in key.upper():
                continue
            if args.pattern and args.pattern.lower() not in key.lower():
                continue
            print(f"  {key}")
        if len(by_sample_size[0]) > 20:
            print(f"  ... and {len(by_sample_size[0]) - 20} more (use --show-empty to see all)")
        print()

    # Show patterns with real trades (mature)
    if by_sample_size["3+"]:
        print("MATURE PATTERNS (3+ trades, no penalty):")
        sorted_mature = sorted(by_sample_size["3+"],
                              key=lambda x: x[1].get("total_trades", 0),
                              reverse=True)
        for key, entry in sorted_mature[:15]:
            wr = entry.get("win_rate", 0)
            total = entry.get("total_trades", 0)
            wins = entry.get("wins", 0)
            print(f"  {total:3d} trades | WR {wr:5.1f}% ({wins}/{total}) | {key}")
        print()

    # Show patterns with 1-2 trades (need more data)
    developing = by_sample_size[1] + by_sample_size[2]
    if developing:
        print(f"DEVELOPING PATTERNS (1-2 trades, still getting penalty):")
        for key, entry in developing[:10]:
            wr = entry.get("win_rate", 0)
            total = entry.get("total_trades", 0)
            wins = entry.get("wins", 0)
            print(f"  {total} trades | WR {wr:.1f}% ({wins}/{total}) | {key}")
        print()

    # Recommendations
    print("=" * 60)
    print("RECOMMENDATIONS:")
    print("=" * 60)
    if not stats:
        print("  1. Run generate_synthetic_samples.py to create initial entries")
    elif len(by_sample_size[0]) > 100:
        print(f"  1. {len(by_sample_size[0])} patterns still have 0 trades.")
        print("     Either:")
        print("       (a) Run a backtest and import outcomes:")
        print("           python main.py --mode backtest --pairs EURUSD --timeframe 1h --bars 500")
        print("           python scripts/bootstrap_tools/import_backtest_samples.py \\")
        print("               --db backtest/backtest_run_EURUSD_H1.db")
        print("       (b) Let the system run live/paper until trades close naturally")
    elif len(by_sample_size["3+"]) < 10:
        print("  1. Most patterns still need 3+ trades to clear the penalty.")
        print("     Continue running the system — every closed trade helps.")
        print(f"  2. {len(by_sample_size['3+'])} patterns are mature (3+ trades).")
    else:
        print("  ✓ System has good coverage.")
        print(f"  ✓ {len(by_sample_size['3+'])} patterns are mature (3+ trades).")
        print("  ✓ Penalty should be minimal for most queries.")
    print()


if __name__ == "__main__":
    main()
