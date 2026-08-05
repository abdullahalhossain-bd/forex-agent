#!/usr/bin/env python3
"""
Master Bootstrap Runner — runs all 5 scripts in the right order.

USAGE:
  # Full bootstrap using CSV data (Linux/CI — no MT5 needed)
  python scripts/bootstrap_tools/run_full_bootstrap.py --source csv

  # Full bootstrap using MT5 historical data (Windows host with MT5)
  python scripts/bootstrap_tools/run_full_bootstrap.py --source mt5 --months 12

  # Full bootstrap using MT5 trade history (if you have closed trades)
  python scripts/bootstrap_tools/run_full_bootstrap.py --source mt5-history --days 90

  # Quick bootstrap (just CSV fallback, fast)
  python scripts/bootstrap_tools/run_full_bootstrap.py --source csv --quick
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


def run_script(script_name: str, args: list = None) -> tuple:
    """Run a script and return (exit_code, duration_sec)."""
    args = args or []
    cmd = [sys.executable, str(SCRIPT_DIR / script_name)] + args
    print(f"\n{'─' * 70}")
    print(f"  RUNNING: {script_name} {' '.join(args)}")
    print(f"{'─' * 70}\n")
    start = time.time()
    try:
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
        duration = time.time() - start
        return result.returncode, duration
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return 1, time.time() - start


def main():
    parser = argparse.ArgumentParser(description="Master bootstrap runner")
    parser.add_argument("--source", choices=["csv", "mt5", "mt5-history"],
                        default="csv",
                        help="Data source: csv (data/*.csv), mt5 (MT5 historical), "
                             "mt5-history (MT5 trade history)")
    parser.add_argument("--months", type=int, default=12,
                        help="Months of data to replay (default: 12)")
    parser.add_argument("--days", type=int, default=90,
                        help="Days of MT5 trade history (for mt5-history source)")
    parser.add_argument("--pairs", default="EURUSD,GBPUSD,USDJPY,AUDUSD,NZDUSD,USDCAD,USDCHF",
                        help="Comma-separated pairs (default: 7 majors)")
    parser.add_argument("--timeframe", default="H1",
                        choices=["M15", "H1", "H4", "D1"],
                        help="Timeframe (default: H1)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode — fewer trades per pair")
    parser.add_argument("--skip-status-check", action="store_true",
                        help="Skip the final status check")
    args = parser.parse_args()

    max_trades = 30 if args.quick else 100

    print("=" * 70)
    print("  MASTER BOOTSTRAP RUNNER")
    print("=" * 70)
    print(f"  Source:     {args.source}")
    print(f"  Pairs:      {args.pairs}")
    print(f"  Timeframe:  {args.timeframe}")
    print(f"  Months:     {args.months}")
    print(f"  Max trades: {max_trades}/pair")
    print(f"  Quick mode: {args.quick}")
    print()

    total_start = time.time()
    steps = []

    # ── Step 1: Generate synthetic seed entries ─────────────────────
    code, dur = run_script("generate_synthetic_samples.py")
    steps.append(("1. generate_synthetic_samples.py", code, dur))
    if code != 0:
        print(f"\n⚠ Step 1 failed (exit {code}) — continuing anyway")

    # ── Step 2: Import data based on source ─────────────────────────
    if args.source == "csv":
        # Use CSV replay
        csv_args = [
            "--pairs", args.pairs,
            "--timeframe", args.timeframe,
            "--use-csv",
            "--months", str(args.months),
            "--max-trades", str(max_trades),
        ]
        code, dur = run_script("replay_mt5_signals.py", csv_args)
        steps.append(("2. replay_mt5_signals.py (CSV)", code, dur))

        # Also run M15 if H1 was the main timeframe (more patterns)
        if args.timeframe == "H1" and not args.quick:
            csv_args_m15 = [
                "--pairs", args.pairs.split(",")[0],  # Just first pair for speed
                "--timeframe", "M15",
                "--use-csv",
                "--months", str(args.months),
                "--max-trades", "30",
            ]
            code, dur = run_script("replay_mt5_signals.py", csv_args_m15)
            steps.append(("2b. replay_mt5_signals.py (M15)", code, dur))

    elif args.source == "mt5":
        # Use MT5 historical replay
        mt5_args = [
            "--pairs", args.pairs,
            "--timeframe", args.timeframe,
            "--months", str(args.months),
            "--max-trades", str(max_trades),
        ]
        code, dur = run_script("replay_mt5_signals.py", mt5_args)
        steps.append(("2. replay_mt5_signals.py (MT5)", code, dur))

    elif args.source == "mt5-history":
        # Use MT5 trade history
        hist_args = [
            "--days", str(args.days),
        ]
        code, dur = run_script("import_mt5_history.py", hist_args)
        steps.append(("2. import_mt5_history.py", code, dur))

    # ── Step 3: Check final status ──────────────────────────────────
    if not args.skip_status_check:
        code, dur = run_script("check_penalty_status.py")
        steps.append(("3. check_penalty_status.py", code, dur))

    # ── Summary ─────────────────────────────────────────────────────
    total_dur = time.time() - total_start
    print()
    print("=" * 70)
    print("  BOOTSTRAP SUMMARY")
    print("=" * 70)
    print(f"  Total duration: {total_dur:.1f}s")
    print()
    print(f"  {'Step':<45} {'Exit':<6} {'Duration':<10}")
    print(f"  {'─'*45} {'─'*6} {'─'*10}")
    for name, code, dur in steps:
        status = "✓ OK" if code == 0 else f"✗ FAIL({code})"
        print(f"  {name:<45} {status:<6} {dur:.1f}s")
    print()
    ok_count = sum(1 for _, code, _ in steps if code == 0)
    print(f"  {ok_count}/{len(steps)} steps succeeded")
    print()
    if ok_count == len(steps):
        print("  ✓ Bootstrap complete! The Bayesian penalty should now be 0")
        print("    for any pattern with 3+ recorded outcomes.")
        print()
        print("  NEXT STEPS:")
        print("    1. Run the system normally — confidence engine will use real data")
        print("    2. As live trades close, they will MIX with this bootstrap data")
        print("    3. After ~10 live trades per pattern, live data dominates")
    else:
        print("  ⚠ Some steps failed — see output above for details.")
        print("    The bootstrap may still be partially complete.")


if __name__ == "__main__":
    main()
