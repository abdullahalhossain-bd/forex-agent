#!/usr/bin/env python3
"""
scripts/rebuild_dna_journal.py — Rebuild Market DNA cluster stats
from REAL closed trades.

After the system has been running and closing trades, run this to
replace the synthetic trade outcomes (from setup_and_train_market_dna.py)
with real trade history. This makes the cluster stats accurate.

Usage:
  python scripts/rebuild_dna_journal.py
  python scripts/rebuild_dna_journal.py --model-id dna_20260805150000
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_DIR, MODEL_DIR
from utils.logger import get_logger

log = get_logger("rebuild_dna_journal")


def get_db_path() -> Path:
    try:
        from config import DB_PATH
        return Path(DB_PATH)
    except Exception:
        from database.db import DB_PATH
        return Path(DB_PATH)


def get_active_model() -> dict:
    """Get the currently ACTIVE market_dna model."""
    db = get_db_path()
    with sqlite3.connect(str(db)) as conn:
        row = conn.execute(
            "SELECT model_id, model_path, trained_at FROM market_dna_models "
            "WHERE status='ACTIVE' ORDER BY trained_at DESC LIMIT 1"
        ).fetchone()
    if not row:
        return {}
    return {"model_id": row[0], "model_path": row[1], "trained_at": row[2]}


def get_closed_trades(pair: str = None) -> pd.DataFrame:
    """Get closed trades from the DB."""
    db = get_db_path()
    with sqlite3.connect(str(db)) as conn:
        if pair:
            df = pd.read_sql_query(
                "SELECT * FROM trades WHERE status='CLOSED' AND pair=? ORDER BY entry_time",
                conn, params=(pair,)
            )
        else:
            df = pd.read_sql_query(
                "SELECT * FROM trades WHERE status='CLOSED' ORDER BY entry_time",
                conn
            )
    return df


def assign_clusters_to_trades(trades_df: pd.DataFrame, detector) -> pd.DataFrame:
    """For each trade, look up the entry bar and predict its cluster."""
    if trades_df.empty:
        return trades_df

    from features.indicators_v5 import add_indicators

    results = []
    for _, trade in trades_df.iterrows():
        pair = trade.get("pair") or trade.get("symbol")
        if not pair:
            continue
        # Find the CSV for this pair
        csv_path = DATA_DIR / f"{pair}_H1.csv"
        if not csv_path.exists():
            # Try M15
            csv_path = DATA_DIR / f"{pair}_M15.csv"
        if not csv_path.exists():
            continue

        # Load and add indicators
        df = pd.read_csv(csv_path)
        time_col = None
        for c in ["datetime_utc", "datetime", "time"]:
            if c in df.columns:
                time_col = c
                break
        if time_col:
            df[time_col] = pd.to_datetime(df[time_col], utc=True)
            df.set_index(time_col, inplace=True)
        if "volume" not in df.columns:
            df["volume"] = df.get("tick_volume", 1000)
        if "time" not in df.columns:
            df["time"] = df.index

        try:
            df = add_indicators(df, drop_nan=True)
        except Exception:
            continue

        # Find the entry bar
        entry_time = pd.to_datetime(trade.get("entry_time"), utc=True)
        if entry_time is None:
            continue
        # Find closest bar to entry time
        idx = df.index.get_indexer([entry_time], method="nearest")[0]
        if idx < 0 or idx >= len(df):
            continue
        entry_bar = df.iloc[[idx]]

        # Predict cluster
        try:
            result = detector.predict_live(entry_bar)
            results.append({
                "trade_id": trade.get("id"),
                "cluster_id": result.get("cluster_id"),
                "state": result.get("state"),
                "direction": trade.get("direction") or trade.get("side"),
                "outcome": "WIN" if (trade.get("pnl_usd") or 0) > 0 else "LOSS",
                "pnl": float(trade.get("pnl_usd") or 0),
            })
        except Exception:
            continue

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Rebuild Market DNA journal from real trades")
    parser.add_argument("--model-id", default=None, help="Specific model ID (default: ACTIVE)")
    parser.add_argument("--pair", default=None, help="Filter trades by pair")
    args = parser.parse_args()

    print("=" * 60)
    print("  REBUILD MARKET DNA JOURNAL")
    print("=" * 60)

    # Get active model
    model_info = get_active_model()
    if not model_info:
        print("  FAIL: No ACTIVE market_dna model found.")
        print("        Run: python scripts/setup_and_train_market_dna.py")
        return

    model_id = args.model_id or model_info["model_id"]
    model_path = Path(model_info["model_path"])
    print(f"  Model: {model_id}")
    print(f"  Path:  {model_path}")

    if not model_path.exists():
        print(f"  FAIL: Model file not found: {model_path}")
        return

    # Load detector
    from analysis.market_dna import MarketDNADetector
    detector = MarketDNADetector.load(model_path)
    print(f"  Loaded detector with {detector.n_clusters_} clusters")

    # Get closed trades
    trades_df = get_closed_trades(args.pair)
    print(f"  Found {len(trades_df)} closed trades in DB")

    if trades_df.empty:
        print("\n  No closed trades yet. The journal will remain based on synthetic data.")
        print("  Run this script again after the system has closed some real trades.")
        return

    # Assign clusters to trades
    print("\n  Assigning clusters to trades...")
    clustered = assign_clusters_to_trades(trades_df, detector)
    print(f"  Clustered {len(clustered)} trades")

    if clustered.empty:
        print("  FAIL: Could not assign clusters to any trades")
        return

    # Build new journal
    from analysis.dna_journal import build_cluster_journal
    stats = build_cluster_journal(clustered, model_id=model_id)

    # Save to DB (replace existing)
    db = get_db_path()
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(str(db)) as conn:
        # Delete old stats for this model
        conn.execute("DELETE FROM market_dna_cluster_stats WHERE model_id=?", (model_id,))
        # Insert new stats
        for s in stats:
            conn.execute(
                """INSERT INTO market_dna_cluster_stats
                   (model_id, cluster_id, trades, wins, win_rate, ci_low, ci_high,
                    profit_factor, expectancy_r, tier, position_multiplier, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (s.model_id, s.cluster_id, s.trades, s.wins, s.win_rate,
                 s.ci_low, s.ci_high, s.profit_factor, s.expectancy_r,
                 s.tier, s.position_multiplier, now),
            )

    print(f"\n  ✓ Journal rebuilt: {len(stats)} clusters")
    print("\n  CLUSTER STATS:")
    print(f"  {'Cluster':>8} {'Trades':>7} {'WR':>6} {'CI':>14} {'PF':>6} {'Tier':>20}")
    print(f"  {'-'*8} {'-'*7} {'-'*6} {'-'*14} {'-'*6} {'-'*20}")
    for s in sorted(stats, key=lambda x: -x.trades):
        print(f"  {s.cluster_id:>8} {s.trades:>7} {s.win_rate*100:>5.1f}% "
              f"[{s.ci_low:.2f},{s.ci_high:.2f}] {s.profit_factor or 0:>6.2f} {s.tier:>20}")

    print("\n  ✓ Done. The MarketDNAService will pick up these new stats on next start.")


if __name__ == "__main__":
    main()
