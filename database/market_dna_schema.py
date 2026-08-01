# database/market_dna_schema.py
# ============================================================
# Market DNA — model registry + cluster-stats tables.
#
# Kept as a separate module from database/db.py rather than adding
# to init_db() there, so this feature can be dropped in without
# touching the existing schema/migration path. Call
# init_market_dna_tables() once at startup (idempotent, same
# CREATE TABLE IF NOT EXISTS pattern as db.py).
# ============================================================

import sqlite3

from database.db import DB_PATH
from utils.logger import get_logger

log = get_logger(__name__)


def init_market_dna_tables(db_path: str = DB_PATH) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.executescript("""
            -- One row per FROZEN detector. `status` tracks the
            -- frozen-model lifecycle: ACTIVE (currently serving live
            -- predictions) or RETIRED (superseded by a newer fit).
            -- Exactly one row should be ACTIVE at a time per symbol
            -- scope (enforce in application code, not here, since
            -- SQLite partial-unique-index syntax varies by version).
            CREATE TABLE IF NOT EXISTS market_dna_models (
                model_id            TEXT PRIMARY KEY,
                trained_at          TEXT NOT NULL,
                train_window_start  TEXT,
                train_window_end    TEXT,
                n_train_rows        INTEGER,
                n_clusters          INTEGER,
                min_cluster_size    INTEGER,
                pca_components      INTEGER,
                feature_cols_json   TEXT,
                model_path          TEXT NOT NULL,
                status              TEXT DEFAULT 'ACTIVE',   -- ACTIVE | RETIRED
                retired_at          TEXT,
                retired_reason      TEXT                     -- e.g. 'freeze_window_expired', 'psi_mandatory_refit'
            );

            -- Rebuilt from scratch on every refit (see dna_journal.py
            -- REBUILD, DON'T MIGRATE policy) — rows are deleted for a
            -- model_id and reinserted, never patched in place.
            CREATE TABLE IF NOT EXISTS market_dna_cluster_stats (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id            TEXT NOT NULL REFERENCES market_dna_models(model_id),
                cluster_id          INTEGER NOT NULL,
                trades              INTEGER NOT NULL,
                wins                INTEGER NOT NULL,
                win_rate            REAL,
                ci_low              REAL,
                ci_high             REAL,
                profit_factor       REAL,
                expectancy_r        REAL,
                tier                TEXT,       -- NO_STATISTICAL_EDGE | WEAK_EVIDENCE | RELIABLE | HIGH_CONFIDENCE
                position_multiplier REAL,
                updated_at          TEXT NOT NULL,
                UNIQUE(model_id, cluster_id)
            );

            -- Per-symbol/session PSI drift readings, so the alert
            -- history is queryable (dashboarding, post-mortems on
            -- "when did drift actually start creeping up").
            CREATE TABLE IF NOT EXISTS market_dna_drift_log (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id       TEXT NOT NULL REFERENCES market_dna_models(model_id),
                checked_at     TEXT NOT NULL,
                psi            REAL NOT NULL,
                status         TEXT NOT NULL,   -- NORMAL | MONITOR | ALERT | MANDATORY_REFIT
                per_cluster_json TEXT
            );

            -- Live cluster assignment attached to each trade — kept
            -- separate from the `trades` table (rather than adding
            -- columns there) so this feature can be disabled/dropped
            -- without touching the core trades schema.
            CREATE TABLE IF NOT EXISTS market_dna_trade_assignments (
                trade_id       INTEGER PRIMARY KEY REFERENCES trades(id),
                model_id       TEXT NOT NULL REFERENCES market_dna_models(model_id),
                cluster_id     INTEGER,          -- NULL when state = UNKNOWN
                state          TEXT NOT NULL,    -- KNOWN | UNKNOWN
                confidence     REAL,
                assigned_at    TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_dna_cluster_stats_model
                ON market_dna_cluster_stats(model_id);
            CREATE INDEX IF NOT EXISTS idx_dna_trade_assignments_model
                ON market_dna_trade_assignments(model_id);
        """)
    log.info("[market_dna_schema] Tables ready (models, cluster_stats, drift_log, trade_assignments).")


if __name__ == "__main__":
    init_market_dna_tables()
    print("Market DNA schema initialized.")
