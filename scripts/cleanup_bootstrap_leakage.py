#!/usr/bin/env python3
"""
cleanup_bootstrap_leakage.py — one-time repair for ml_features.db

Context: an earlier version of ml/data_bootstrap.py generated synthetic
"seed rows" whose features were a deterministic function of row index,
and whose label was ALSO a deterministic function of row index. That made
the label trivially predictable from the features (see the trend_bias /
label correlation) — a pure label leak. Those rows were written straight
into the persistent SQLite store (ml_features.db) and are still there
even after the bootstrap code itself has been fixed, because fixing the
code that WRITES new rows does not remove rows already written.

This script finds and deletes exactly those old rows, using a fingerprint
that is specific to the old synthetic-row formula:
    price_high - price_close == 0.00018   (exact, to float precision)
    price_close - price_low  == 0.00022   (exact, to float precision)
    price_close - price_open == 0.00012   (exact, to float precision)
Real market OHLC data will not hit all three exact constants across
thousands of bars, so this is a safe, low-false-positive filter.

Usage:
    python scripts/cleanup_bootstrap_leakage.py                # dry run, report only
    python scripts/cleanup_bootstrap_leakage.py --apply         # actually delete
    python scripts/cleanup_bootstrap_leakage.py --pair EURUSD --apply
"""
import argparse
import json
import sqlite3
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

TOL = 1e-9  # float equality tolerance


def _is_old_bootstrap_row(feature_vector_json: str) -> bool:
    try:
        f = json.loads(feature_vector_json)
        po, ph, pl, pc = f.get("price_open"), f.get("price_high"), f.get("price_low"), f.get("price_close")
        if None in (po, ph, pl, pc):
            return False
        return (
            abs((ph - pc) - 0.00018) < TOL and
            abs((pc - pl) - 0.00022) < TOL and
            abs((pc - po) - (-0.00012)) < TOL  # old code: open_ = close - 0.00012 -> close - open = 0.00012
        )
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Remove old leaky bootstrap rows from ml_features.db")
    parser.add_argument("--pair", type=str, default=None, help="Limit to one pair (default: all pairs)")
    parser.add_argument("--apply", action="store_true", help="Actually delete matched rows (default: dry run)")
    args = parser.parse_args()

    from core.constants import MEMORY_DIR
    db_path = MEMORY_DIR / "ml_features.db"
    print(f"DB: {db_path}")
    if not db_path.exists():
        print("FAIL: database file not found")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    query = "SELECT id, pair, timeframe, feature_vector FROM features WHERE 1=1"
    params = []
    if args.pair:
        query += " AND pair = ?"
        params.append(args.pair.upper())
    rows = cur.execute(query, params).fetchall()

    print(f"Scanned {len(rows)} total rows" + (f" for {args.pair.upper()}" if args.pair else " across all pairs"))

    per_pair_counts = {}
    matched_ids = []
    for feature_id, pair, timeframe, fv in rows:
        if _is_old_bootstrap_row(fv):
            matched_ids.append(feature_id)
            key = f"{pair} {timeframe}"
            per_pair_counts[key] = per_pair_counts.get(key, 0) + 1

    print()
    print("Old leaky bootstrap rows found, by pair/timeframe:")
    if not per_pair_counts:
        print("  (none found)")
    for key, count in sorted(per_pair_counts.items()):
        print(f"  {key}: {count}")
    print()
    print(f"Total matched: {len(matched_ids)} / {len(rows)}")

    if not matched_ids:
        print("Nothing to clean up.")
        conn.close()
        return

    if not args.apply:
        print()
        print("DRY RUN — no rows deleted. Re-run with --apply to actually delete these rows.")
        conn.close()
        return

    print()
    print(f"Deleting {len(matched_ids)} rows from 'features' and 'labels' tables...")
    CHUNK = 500
    for i in range(0, len(matched_ids), CHUNK):
        chunk = matched_ids[i:i + CHUNK]
        placeholders = ",".join("?" * len(chunk))
        cur.execute(f"DELETE FROM labels WHERE feature_id IN ({placeholders})", chunk)
        cur.execute(f"DELETE FROM features WHERE id IN ({placeholders})", chunk)
    conn.commit()
    conn.close()
    print("Done. Old leaky bootstrap rows removed.")
    print()
    print("Next: re-run training with --no-bootstrap to confirm remaining data is real-data-only:")
    print("  python scripts/train_models.py --pair EURUSD --no-bootstrap")


if __name__ == "__main__":
    main()