#!/usr/bin/env python3
"""
Option 3 — Run the system in PAPER mode to bootstrap real samples.

This is the SLOWEST but most LEGITIMATE bootstrap: actually run the
trading system in paper mode for a period, let it take trades against
real market data (via MT5 or API fallback), close them on SL/TP, and
record the outcomes. ConfidenceEngine.record_outcome() fires
automatically on every closed trade.

WHAT THIS SCRIPT DOES:
  - Sets BYPASS_NEWS_GATE=true (so news API failures don't block trades)
  - Sets TEST_MODE=false (we want the real gates active)
  - Sets SIMULATION_MODE=true (no real MT5 orders, paper only)
  - Runs main.py in a subprocess for N minutes/hours
  - Reports periodically on how many trades were taken/closed
  - Saves a snapshot of pattern_stats.json every hour

USAGE:
  # Run for 4 hours (overnight)
  python scripts/bootstrap_tools/run_paper_bootstrap.py --hours 4

  # Run for 30 minutes (quick test)
  python scripts/bootstrap_tools/run_paper_bootstrap.py --minutes 30

  # Run with specific pairs only
  python scripts/bootstrap_tools/run_paper_bootstrap.py --hours 8 --pairs EURUSD,GBPUSD

  # Check progress without starting a new run
  python scripts/bootstrap_tools/run_paper_bootstrap.py --status-only

PREREQUISITES:
  - MT5 terminal running (Windows) OR API keys configured (.env)
  - .env file with at least some LLM keys (or BYPASS_NEWS_GATE=true)
  - The trade_permission.py bugs must be fixed (already done in this audit)

NOTE:
  This script does NOT fabricate data. It runs the real system and
  lets it accumulate real outcomes. The Bayesian penalty will drop
  naturally as real trades close.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from core.constants import MEMORY_DIR
except Exception as e:
    print(f"ERROR: cannot import MEMORY_DIR: {e}")
    sys.exit(1)

PATTERN_STATS_PATH = MEMORY_DIR / "pattern_stats.json"


def snapshot_pattern_stats(label: str = "") -> dict:
    """Take a snapshot of pattern_stats.json for progress tracking."""
    if not PATTERN_STATS_PATH.exists():
        return {"total": 0, "with_trades": 0, "mature": 0}
    try:
        with open(PATTERN_STATS_PATH, encoding="utf-8") as f:
            stats = json.load(f)
    except Exception:
        return {"total": 0, "with_trades": 0, "mature": 0}

    total = len(stats)
    with_trades = sum(1 for e in stats.values() if e.get("total_trades", 0) > 0)
    mature = sum(1 for e in stats.values() if e.get("total_trades", 0) >= 3)
    total_real_trades = sum(e.get("total_trades", 0) for e in stats.values())
    return {
        "total": total,
        "with_trades": with_trades,
        "mature": mature,
        "total_real_trades": total_real_trades,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": label,
    }


def main():
    parser = argparse.ArgumentParser(description="Run paper-mode bootstrap to accumulate real samples")
    parser.add_argument("--hours", type=float, default=0,
                        help="Run for N hours")
    parser.add_argument("--minutes", type=float, default=0,
                        help="Run for N minutes (added to --hours)")
    parser.add_argument("--pairs", default="",
                        help="Comma-separated pairs (default: from config.py SYMBOLS)")
    parser.add_argument("--timeframe", default="",
                        help="Timeframe (default: from config.py DEFAULT_TIMEFRAME)")
    parser.add_argument("--status-only", action="store_true",
                        help="Just show current pattern_stats status and exit")
    parser.add_argument("--snapshot-interval-min", type=int, default=60,
                        help="Take a snapshot every N minutes (default: 60)")
    args = parser.parse_args()

    if args.status_only:
        snap = snapshot_pattern_stats("current")
        print("Current pattern_stats.json status:")
        print(json.dumps(snap, indent=2))
        return

    duration_sec = (args.hours * 3600) + (args.minutes * 60)
    if duration_sec < 60:
        print("ERROR: specify --hours and/or --minutes (minimum 1 minute)")
        parser.print_help()
        return

    print(f"Paper-mode bootstrap runner")
    print(f"  Duration:          {duration_sec/3600:.2f} hours ({duration_sec:.0f}s)")
    print(f"  Snapshot interval: {args.snapshot_interval_min} min")
    print(f"  Pairs:             {args.pairs or '(from config)'}")
    print(f"  Timeframe:         {args.timeframe or '(from config)'}")
    print(f"  Pattern stats:     {PATTERN_STATS_PATH}")
    print()

    # Take before snapshot
    before = snapshot_pattern_stats("before")
    print("BEFORE:")
    print(json.dumps(before, indent=2))
    print()

    # Build command
    cmd = [sys.executable, str(PROJECT_ROOT / "main.py")]
    if args.pairs:
        cmd.extend(["--pairs", args.pairs])
    if args.timeframe:
        cmd.extend(["--timeframe", args.timeframe])

    # Set environment for paper bootstrap
    env = dict(os.environ)
    env["BYPASS_NEWS_GATE"] = "true"   # news API down — don't let it block
    env["TEST_MODE"] = "false"          # we want real gates active
    env["SIMULATION_MODE"] = "true"     # paper trading only

    print(f"Command: {' '.join(cmd)}")
    print(f"Env: BYPASS_NEWS_GATE=true | TEST_MODE=false | SIMULATION_MODE=true")
    print()

    # Start the process
    start_time = time.time()
    log_path = PROJECT_ROOT / "logs" / "paper_bootstrap.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Logging to: {log_path}")
    print()
    print("The system is now running in the background. Press Ctrl+C to stop early.")
    print("Snapshots will be taken every {} minutes.".format(args.snapshot_interval_min))
    print()

    log_file = open(log_path, "a", encoding="utf-8")
    log_file.write(f"\n{'='*60}\nPaper bootstrap started at {datetime.now(timezone.utc).isoformat()}\n{'='*60}\n")
    log_file.flush()

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=env,
        )
    except Exception as e:
        print(f"ERROR starting main.py: {e}")
        log_file.close()
        return

    # Monitor loop
    snapshots = [before]
    try:
        next_snapshot = time.time() + args.snapshot_interval_min * 60
        while time.time() - start_time < duration_sec:
            ret = proc.poll()
            if ret is not None:
                print(f"main.py exited with code {ret} after {time.time()-start_time:.0f}s")
                break
            time.sleep(30)  # check every 30s
            elapsed = time.time() - start_time
            print(f"  [{elapsed/60:.1f} min] running... (PID {proc.pid})")
            if time.time() >= next_snapshot:
                snap = snapshot_pattern_stats(f"t+{elapsed/60:.0f}min")
                snapshots.append(snap)
                print(f"    Snapshot: {snap['mature']} mature, "
                      f"{snap['with_trades']} with trades, "
                      f"{snap['total_real_trades']} total trades")
                next_snapshot = time.time() + args.snapshot_interval_min * 60
    except KeyboardInterrupt:
        print("\nCtrl+C received — stopping main.py...")

    # Stop the process
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    log_file.write(f"\nPaper bootstrap stopped at {datetime.now(timezone.utc).isoformat()}\n")
    log_file.close()

    # Take after snapshot
    after = snapshot_pattern_stats("after")
    print()
    print("AFTER:")
    print(json.dumps(after, indent=2))
    print()

    # Summary
    print("=" * 60)
    print("BOOTSTRAP SUMMARY")
    print("=" * 60)
    print(f"  Duration:        {(time.time()-start_time)/60:.1f} min")
    print(f"  Entries before:  {before['total']}")
    print(f"  Entries after:   {after['total']}")
    print(f"  With trades before: {before['with_trades']}")
    print(f"  With trades after:  {after['with_trades']}")
    print(f"  Mature (3+) before: {before['mature']}")
    print(f"  Mature (3+) after:  {after['mature']}")
    print(f"  Total real trades:  {after['total_real_trades']}")
    print()
    if after['mature'] > before['mature']:
        print(f"  ✓ {after['mature'] - before['mature']} new patterns reached maturity (3+ trades)")
        print(f"  ✓ Bayesian penalty eliminated for those patterns")
    else:
        print(f"  ⚠ No new patterns reached maturity this run.")
        print(f"    Consider running longer, or use import_backtest_samples.py")
        print(f"    to fast-forward with historical backtest data.")
    print()
    print("Next steps:")
    print("  1. Check status:    python scripts/bootstrap_tools/check_penalty_status.py")
    print("  2. Run again:       python scripts/bootstrap_tools/run_paper_bootstrap.py --hours 4")
    print("  3. View logs:       tail -100 logs/paper_bootstrap.log")


if __name__ == "__main__":
    main()
