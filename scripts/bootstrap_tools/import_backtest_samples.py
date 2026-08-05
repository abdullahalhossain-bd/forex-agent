#!/usr/bin/env python3
"""
Option 2 — Import REAL backtest outcomes into pattern_stats.json.

This is the MOST LEGITIMATE bootstrap approach: run a backtest, then
feed the backtest's actual trade outcomes (real BUY/SELL signals that
hit TP or SL) into the confidence engine's pattern_stats.json. The
engine then has REAL data to compute win_rate from, not synthetic
placeholders.

WHAT THIS SCRIPT DOES:
  - Reads a backtest DB (produced by backtest/unified_engine.py)
  - For each closed trade, extracts: pair, timeframe, regime, outcome
  - Calls ConfidenceEngine.record_outcome() to add the trade to
    pattern_stats.json with proper bookkeeping
  - The engine's calculate() will now return real sample_size > 0
    and the Bayesian penalty will be reduced or eliminated

PREREQUISITES:
  - Run a backtest first:
      python main.py --mode backtest --pairs EURUSD --timeframe 1h --bars 500
  - This creates backtest/backtest_run_EURUSD_H1.db with closed trades

USAGE:
  # Import from a single backtest DB
  python scripts/bootstrap_tools/import_backtest_samples.py \
      --db backtest/backtest_run_EURUSD_H1.db

  # Import from multiple backtest DBs (multi-pair)
  python scripts/bootstrap_tools/import_backtest_samples.py \
      --db backtest/backtest_run_EURUSD_H1.db \
      --db backtest/backtest_run_GBPUSD_H1.db

  # Dry-run (show what would be imported without writing)
  python scripts/bootstrap_tools/import_backtest_samples.py --db <path> --dry-run

  # Override the pattern name (backtest DBs may not store it)
  python scripts/bootstrap_tools/import_backtest_samples.py \
      --db <path> --default-pattern Hammer

IMPORTANT CAVEATS:
  1. The backtest must be HONEST — no look-ahead, no overfitting.
     Use the patched unified_engine.py that wires the registry properly.
  2. Backtest outcomes are not the same as live outcomes (different
     execution: BrokerSimulator vs MT5). But they're far better than
     zero data — the engine will recalibrate as real live trades
     replace them.
  3. If your backtest had <10 trades, this won't help much. Run a
     longer backtest (--bars 2000+ or --days 90+).
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from datetime import datetime, timezone

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from core.constants import MEMORY_DIR
    from learning.confidence_engine import ConfidenceEngine, MIN_SAMPLE_SIZE
except Exception as e:
    print(f"ERROR: cannot import from project: {e}")
    print(f"       Run from project root:  cd {PROJECT_ROOT}")
    sys.exit(1)


def inspect_db_schema(db_path: str) -> tuple:
    """Inspect the backtest DB to find the trades table and its columns."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # List all tables
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    # Find a trades-like table
    trades_table = None
    for t in tables:
        if "trade" in t.lower():
            trades_table = t
            break
    if not trades_table:
        conn.close()
        return tables, None, []
    # Get columns
    cur.execute(f"PRAGMA table_info({trades_table})")
    columns = [r[1] for r in cur.fetchall()]
    conn.close()
    return tables, trades_table, columns


def extract_trades(db_path: str) -> list:
    """Extract closed trades from the backtest DB.
    Returns list of dicts with keys: pair, timeframe, regime, direction,
    exit_reason, pnl_usd, entry_time, exit_time.
    """
    tables, trades_table, columns = inspect_db_schema(db_path)
    if not trades_table:
        print(f"  ✗ No trades table found in {db_path}")
        print(f"    Tables present: {tables}")
        return []

    print(f"  Found trades table: '{trades_table}' with {len(columns)} columns")
    print(f"    Columns: {', '.join(columns[:15])}{'...' if len(columns) > 15 else ''}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Build SELECT dynamically based on available columns
    col_map = {c.lower(): c for c in columns}
    select_cols = []
    for needed in ["symbol", "pair", "timeframe", "direction", "side",
                   "exit_reason", "pnl_usd", "pnl", "profit", "result",
                   "entry_time", "exit_time", "regime", "market_regime",
                   "pattern", "strategy"]:
        for alias, real in col_map.items():
            if needed in alias:
                select_cols.append(f"{real} AS {needed}")
                break

    if not select_cols:
        print(f"  ✗ No recognizable columns in {trades_table}")
        conn.close()
        return []

    query = f"SELECT {', '.join(select_cols)} FROM {trades_table}"
    try:
        cur.execute(query)
        rows = cur.fetchall()
    except Exception as e:
        print(f"  ✗ Query failed: {e}")
        conn.close()
        return []

    trades = []
    for row in rows:
        d = dict(row)
        # Normalize direction to BUY/SELL
        direction = (d.get("direction") or d.get("side") or "").upper()
        if direction in ("LONG", "BUY"):
            d["direction"] = "BUY"
        elif direction in ("SHORT", "SELL"):
            d["direction"] = "SELL"
        else:
            continue  # skip non-trade entries
        # Normalize exit_reason to WIN/LOSS
        exit_reason = (d.get("exit_reason") or "").upper()
        pnl = d.get("pnl_usd") or d.get("pnl") or d.get("profit") or 0
        try:
            pnl = float(pnl)
        except Exception:
            pnl = 0
        if exit_reason in ("TP", "TAKE_PROFIT") or pnl > 0:
            d["outcome"] = "WIN"
        elif exit_reason in ("SL", "STOP_LOSS") or pnl < 0:
            d["outcome"] = "LOSS"
        else:
            d["outcome"] = "BE"  # break-even
        d["pnl"] = pnl
        trades.append(d)

    conn.close()
    return trades


def main():
    parser = argparse.ArgumentParser(description="Import backtest outcomes into pattern_stats.json")
    parser.add_argument("--db", action="append", required=True,
                        help="Path to backtest DB (can be specified multiple times)")
    parser.add_argument("--default-pattern", default="unknown",
                        help="Pattern name to use if DB doesn't store one (default: 'unknown')")
    parser.add_argument("--default-regime", default="UNKNOWN",
                        help="Regime to use if DB doesn't store one (default: 'UNKNOWN')")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be imported without writing")
    args = parser.parse_args()

    print("Backtest outcome importer")
    print(f"  Target pattern_stats: {MEMORY_DIR / 'pattern_stats.json'}")
    print(f"  Default pattern:      {args.default_pattern}")
    print(f"  Default regime:       {args.default_regime}")
    print()

    all_trades = []
    for db_path in args.db:
        if not os.path.exists(db_path):
            print(f"  ✗ DB not found: {db_path}")
            continue
        print(f"  Inspecting: {db_path}")
        trades = extract_trades(db_path)
        print(f"    Extracted {len(trades)} closed trades")
        all_trades.extend(trades)
        print()

    if not all_trades:
        print("No trades to import. Run a backtest first:")
        print("  python main.py --mode backtest --pairs EURUSD --timeframe 1h --bars 500")
        return

    print(f"Total trades to import: {len(all_trades)}")

    # Show sample
    if all_trades:
        print("\nSample trade:")
        print(json.dumps(all_trades[0], indent=2, default=str)[:500])

    # Group by (pattern, pair, timeframe, regime) to show distribution
    from collections import Counter
    by_combo = Counter()
    for t in all_trades:
        pair = (t.get("symbol") or t.get("pair") or "EURUSD").upper()
        tf = (t.get("timeframe") or "H1").upper()
        regime = (t.get("regime") or t.get("market_regime") or args.default_regime).upper()
        pattern = (t.get("pattern") or t.get("strategy") or args.default_pattern)
        by_combo[(pattern, pair, tf, regime)] += 1

    print(f"\nUnique (pattern, pair, tf, regime) combos: {len(by_combo)}")
    for combo, count in by_combo.most_common(10):
        print(f"  {count:3d} trades  {combo}")
    print()

    if args.dry_run:
        print("[DRY RUN] No file written. Remove --dry-run to apply.")
        return

    # Import into ConfidenceEngine
    print("Importing into ConfidenceEngine...")
    engine = ConfidenceEngine()
    imported = 0
    skipped = 0
    for t in all_trades:
        pair = (t.get("symbol") or t.get("pair") or "EURUSD").upper()
        tf = (t.get("timeframe") or "H1").upper()
        regime = (t.get("regime") or t.get("market_regime") or args.default_regime).upper()
        pattern = (t.get("pattern") or t.get("strategy") or args.default_pattern)
        outcome = t["outcome"]
        pnl = t.get("pnl", 0)
        try:
            engine.record_outcome(
                pattern=pattern,
                pair=pair,
                timeframe=tf,
                regime=regime,
                outcome=outcome,
                confidence_used=None,  # backtest didn't track this
                pnl=pnl,
            )
            imported += 1
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"  ⚠ Skipped: {e}")

    print()
    print(f"✓ Imported: {imported}")
    print(f"  Skipped:  {skipped}")
    print(f"  Pattern stats now at: {MEMORY_DIR / 'pattern_stats.json'}")
    print()
    print("NEXT STEPS:")
    print("  1. Verify the import:")
    print("     python scripts/bootstrap_tools/check_penalty_status.py")
    print("  2. The Bayesian penalty should now be reduced for any pattern")
    print("     with ≥3 imported outcomes.")
    print("  3. As live trades close, they will MIX with backtest data.")
    print("     After ~10 live trades per pattern, live data dominates.")


if __name__ == "__main__":
    main()
