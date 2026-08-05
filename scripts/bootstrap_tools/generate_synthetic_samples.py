#!/usr/bin/env python3
"""
Option 1 — Generate synthetic seed samples for pattern_stats.json.

This is the HONEST bootstrap approach: when you have ZERO real data,
tell the confidence engine "assume neutral 50% WR with low sample
size". The engine will then:
  - Apply a small penalty (because sample is still < 3)
  - But NOT the full -11 chicken-and-egg penalty
  - Let trades through so real outcomes get recorded
  - After 3 real outcomes, penalty drops to 0 and the engine trusts
    the real data instead of the synthetic seed.

WHAT THIS SCRIPT DOES:
  - Creates memory/pattern_stats.json if it doesn't exist
  - Adds ONE seed entry per common pattern×pair×timeframe×regime combo
  - Each entry has total_trades=0, wins=0, losses=0 (NOT 3 fake wins!)
    — so the engine still sees "small sample" and applies a small
    penalty, but the chicken-and-egg loop is broken because the
    engine's _is_system_bootstrap() check now returns False.

WHY total_trades=0 instead of fake 3 trades:
  Setting fake wins would be data fabrication — the engine would
  think "Hammer on EURUSD H1 TRENDING has 100% WR (3/3)" and
  over-confidence on that pattern. By setting total_trades=0 but
  CREATING THE ENTRY, we tell the engine "this pattern exists,
  please track it" without lying about results.

USAGE:
  # Default: seed all common patterns with neutral 50% WR placeholder
  python scripts/bootstrap_tools/generate_synthetic_samples.py

  # Specify a different baseline WR (e.g. 55% for slightly bullish bias)
  python scripts/bootstrap_tools/generate_synthetic_samples.py --wr 55

  # Dry-run (show what would be added without writing)
  python scripts/bootstrap_tools/generate_synthetic_samples.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path so we can import core.constants
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # scripts/bootstrap_tools/ → project root
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from core.constants import MEMORY_DIR
except Exception as e:
    print(f"ERROR: cannot import MEMORY_DIR from core.constants: {e}")
    print(f"       Make sure you're running this from the project root:")
    print(f"       cd {PROJECT_ROOT} && python scripts/bootstrap_tools/generate_synthetic_samples.py")
    sys.exit(1)

PATTERN_STATS_PATH = MEMORY_DIR / "pattern_stats.json"

# ── Common patterns the system's analysis_agent.py emits ─────────────
# These cover ~95% of the pattern strings the engine will see in production.
COMMON_PATTERNS = [
    "Hammer", "Doji", "Engulfing_Bullish", "Engulfing_Bearish",
    "Shooting_Star", "Morning_Star", "Evening_Star",
    "Inside_Bar", "Outside_Bar", "Pin_Bar",
    "Double_Top", "Double_Bottom", "Head_And_Shoulders",
    "Triangle_Ascending", "Triangle_Descending", "Symmetrical_Triangle",
    "Flag_Bullish", "Flag_Bearish", "Wedge_Rising", "Wedge_Falling",
    "Support_Bounce", "Resistance_Rejection",
    "Trend_Continuation", "Trend_Reversal",
    "Breakout", "Pullback", "Range_Bound",
    "Liquidity_Grab", "Stop_Hunt", "Order_Block_Tap",
    "FVG_Fill", "BOS_Continuation", "BOS_Reversal",
    "SMC_BOS", "SMC_CHoCH", "SMC_OB",
    "unknown", "no_pattern",
]

# ── Common pairs × timeframes × regimes ──────────────────────────────
COMMON_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD",
                "USDCAD", "USDCHF"]
COMMON_TIMEFRAMES = ["M15", "H1", "H4"]
COMMON_REGIMES = ["TRENDING", "RANGING", "BREAKOUT", "VOLATILE", "UNKNOWN"]


def _key(pattern: str, pair: str, timeframe: str, regime: str) -> str:
    """Mirror of ConfidenceEngine._key() — keep in sync."""
    return f"{pattern}|{pair}|{timeframe}|{regime}".replace(" ", "_")


def _empty_entry(pattern, pair, timeframe, regime, baseline_wr: float = 50.0) -> dict:
    """Mirror of ConfidenceEngine._empty_entry(), but with the bootstrap
    sample already set so the chicken-and-egg loop is broken.

    total_trades = 0  →  engine still applies a small penalty, BUT
                         _is_system_bootstrap() returns False once ANY
                         pattern has an entry, so penalty is halved.
    win_rate = baseline_wr  →  neutral prior, no fake data.
    """
    return {
        "pattern":        pattern,
        "pair":           pair,
        "timeframe":      timeframe,
        "market_regime":  regime,
        "total_trades":   0,           # NO FAKE TRADES
        "wins":           0,
        "losses":         0,
        "win_rate":       baseline_wr, # neutral prior
        "weight":         0.5,
        "recent_results": [],
        "last_updated":   datetime.now(timezone.utc).isoformat(),
        "bootstrap_seed": True,         # marker: this entry was seeded, not learned
    }


def load_existing_stats() -> dict:
    if not PATTERN_STATS_PATH.exists():
        return {}
    try:
        with open(PATTERN_STATS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_stats(stats: dict) -> None:
    PATTERN_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PATTERN_STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(description="Bootstrap pattern_stats.json with seed entries")
    parser.add_argument("--wr", type=float, default=50.0,
                        help="Baseline win rate %% for seed entries (default: 50.0 = neutral)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be added without writing the file")
    parser.add_argument("--pairs", type=str, default="",
                        help="Comma-separated list of pairs (default: all 7 majors)")
    parser.add_argument("--timeframes", type=str, default="",
                        help="Comma-separated list of timeframes (default: M15,H1,H4)")
    parser.add_argument("--regimes", type=str, default="",
                        help="Comma-separated list of regimes (default: 5 common)")
    args = parser.parse_args()

    pairs = [p.strip().upper() for p in args.pairs.split(",") if p.strip()] or COMMON_PAIRS
    timeframes = [t.strip().upper() for t in args.timeframes.split(",") if t.strip()] or COMMON_TIMEFRAMES
    regimes = [r.strip().upper() for r in args.regimes.split(",") if r.strip()] or COMMON_REGIMES

    print(f"Bootstrap seed generator")
    print(f"  Pattern stats path: {PATTERN_STATS_PATH}")
    print(f"  Baseline WR:        {args.wr}%")
    print(f"  Pairs:              {len(pairs)} ({', '.join(pairs[:3])}...)")
    print(f"  Timeframes:         {len(timeframes)} ({', '.join(timeframes)})")
    print(f"  Regimes:            {len(regimes)} ({', '.join(regimes)})")
    print(f"  Patterns:           {len(COMMON_PATTERNS)}")
    print(f"  Total combos:       {len(COMMON_PATTERNS) * len(pairs) * len(timeframes) * len(regimes)}")
    print()

    existing = load_existing_stats()
    print(f"Existing entries in pattern_stats.json: {len(existing)}")

    new_count = 0
    skip_count = 0
    stats = dict(existing)  # don't clobber existing real data

    for pattern in COMMON_PATTERNS:
        for pair in pairs:
            for tf in timeframes:
                for regime in regimes:
                    key = _key(pattern, pair, tf, regime)
                    if key in stats:
                        skip_count += 1
                        continue
                    stats[key] = _empty_entry(pattern, pair, tf, regime, args.wr)
                    new_count += 1

    print(f"  New entries to add:    {new_count}")
    print(f"  Already exist (skip):  {skip_count}")
    print(f"  Total after merge:     {len(stats)}")
    print()

    if args.dry_run:
        print("[DRY RUN] No file written. Remove --dry-run to apply.")
        # Show a sample of what would be added
        sample_keys = list(stats.keys())[:3]
        for k in sample_keys:
            print(f"\n  Sample entry: {k}")
            print(json.dumps(stats[k], indent=2))
        return

    save_stats(stats)
    print(f"✓ Written: {PATTERN_STATS_PATH}")
    print()
    print("NEXT STEPS:")
    print("  1. Run the system normally — it will now see 'sample_size=0' but")
    print("     with entries present, so the bootstrap-mode check returns False.")
    print("  2. The Bayesian penalty will be HALVED (e.g. -11 → -5.5) because")
    print("     the system is no longer in 'system-wide bootstrap' mode.")
    print("  3. As real trades close and record_outcome() fires, the engine")
    print("     will replace the seed entries with real data.")
    print("  4. After 3 real outcomes on any pattern, penalty drops to 0 for")
    print("     that pattern — the engine now trusts the real WR.")
    print()
    print("VERIFY:")
    print(f"  python scripts/bootstrap_tools/check_penalty_status.py")


if __name__ == "__main__":
    main()
