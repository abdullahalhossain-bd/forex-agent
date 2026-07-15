"""
migrate_legacy_trades.py — One-time migration from memory/trader.db (Day-15 schema)
to database/trader.db (Day-43+ schema).

Reads `trades`, `analysis_log`, `performance`, `mistakes` tables from memory/trader.db
and writes them as `legacy_*` tables in database/trader.db so the LearningEngine
keeps its historical data after the consolidation.

Run once:
    python scripts/migrate_legacy_trades.py

Idempotent: uses INSERT OR IGNORE so re-running won't duplicate rows.
"""
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SRC = ROOT / "memory" / "trader.db"
DST = ROOT / "database" / "trader.db"

if not SRC.exists():
    print(f"[skip] Source DB not found: {SRC}")
    sys.exit(0)

print(f"[migrate] {SRC} -> {DST}")

src = sqlite3.connect(str(SRC))
src.row_factory = sqlite3.Row
dst = sqlite3.connect(str(DST))

# Ensure legacy_* tables exist in destination
dst.executescript("""
CREATE TABLE IF NOT EXISTS legacy_trades (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    pair         TEXT    NOT NULL,
    signal       TEXT    NOT NULL,
    entry        REAL,
    sl           REAL,
    tp           REAL,
    lot          REAL,
    result       TEXT,
    pnl          REAL    DEFAULT 0,
    rr_ratio     REAL,
    confidence   INTEGER,
    chart_snapshot TEXT,
    date         TEXT    DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS legacy_analysis_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    pair         TEXT,
    timeframe    TEXT,
    rsi          REAL,
    macd         REAL,
    trend        TEXT,
    regime       TEXT,
    pattern      TEXT,
    sr_location  TEXT,
    mtf_bias     TEXT,
    decision     TEXT,
    confidence   INTEGER,
    indicators   TEXT,
    date         TEXT    DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS legacy_performance (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    date         TEXT    UNIQUE,
    total_trades INTEGER DEFAULT 0,
    wins         INTEGER DEFAULT 0,
    losses       INTEGER DEFAULT 0,
    win_rate     REAL    DEFAULT 0,
    pnl          REAL    DEFAULT 0,
    best_trade   REAL    DEFAULT 0,
    worst_trade  REAL    DEFAULT 0
);
CREATE TABLE IF NOT EXISTS legacy_mistakes (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id     INTEGER,
    pair         TEXT,
    error_type   TEXT,
    what_happened TEXT,
    lesson       TEXT,
    date         TEXT    DEFAULT (datetime('now'))
);
""")

# Migrate each table
for src_table, dst_table, cols in [
    ("trades",        "legacy_trades",        "pair, signal, entry, sl, tp, lot, result, pnl, rr_ratio, confidence, chart_snapshot, date"),
    ("analysis_log",  "legacy_analysis_log",  "pair, timeframe, rsi, macd, trend, regime, pattern, sr_location, mtf_bias, decision, confidence, indicators, date"),
    ("performance",   "legacy_performance",   "date, total_trades, wins, losses, win_rate, pnl, best_trade, worst_trade"),
    ("mistakes",      "legacy_mistakes",      "trade_id, pair, error_type, what_happened, lesson, date"),
]:
    try:
        # Verify the source table exists
        cur = src.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{src_table}'")
        if not cur.fetchone():
            print(f"  [skip] {src_table}: not in source DB")
            continue

        rows = src.execute(f"SELECT {cols} FROM {src_table}").fetchall()
        if not rows:
            print(f"  [skip] {src_table}: 0 rows")
            continue

        placeholders = ", ".join(["?"] * len(rows[0].keys()))
        col_count = len(rows[0].keys())
        placeholders = ", ".join(["?"] * col_count)
        dst.executemany(
            f"INSERT OR IGNORE INTO {dst_table} ({cols}) VALUES ({placeholders})",
            [tuple(r) for r in rows]
        )
        print(f"  [ok]   {src_table} -> {dst_table}: {len(rows)} rows")
    except Exception as e:
        print(f"  [fail] {src_table} -> {dst_table}: {e}")

dst.commit()
dst.close()
src.close()
print("[done] Migration complete.")
print(f"  You can now safely delete: {SRC}")
