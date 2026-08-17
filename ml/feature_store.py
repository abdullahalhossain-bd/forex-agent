"""
ml/feature_store.py — Persistent feature store (Day 68)
==========================================================

SQLite-backed feature store for ML training data. Stores every
feature vector + label + outcome so the ML model can learn over time.

Tables:
  * **features**    — feature_vector (JSON), pair, timeframe, timestamp, source
  * **labels**      — target label + outcome (filled in after trade closes)
  * **importance**  — feature importance rankings over time

CRITICAL: labels are added AFTER the trade closes — no future leakage.
The features table only contains info available at decision time.

CRITICAL (added after the 2026-08 leakage audit): every row now carries a
`source` tag — 'live' for rows that came from real market/trading data,
'bootstrap' for synthetic placeholder rows written by
ml.data_bootstrap.bootstrap_feature_store_if_needed(). load_training_data()
excludes 'bootstrap' rows by default. This exists because an earlier
version of the bootstrap code wrote synthetic rows whose features and
labels were both deterministic functions of row index (i.e. correlated
with each other for no real reason), and because those rows were
indistinguishable from real data there was no way to keep them out of a
training run without wiping the whole database. Never remove the
`source` filter from load_training_data() without an explicit,
opt-in reason.

Usage:
    store = get_feature_store()
    store.save_features(pair="EURUSD", timeframe="15m", features={...}, label=1)
    df = store.load_training_data(pair="EURUSD", min_samples=100)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from config import MIN_TRAINING_SAMPLES

import pandas as pd

from utils.logger import get_logger
from core.constants import MEMORY_DIR

log = get_logger("feature_store")

DB_PATH = MEMORY_DIR / "ml_features.db"


class FeatureStore:
    """SQLite-backed persistent feature store."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _conn(self):
        return sqlite3.connect(str(self.db_path))

    def _init_db(self) -> None:
        with self._lock, self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    feature_vector TEXT NOT NULL,
                    feature_count INTEGER,
                    timestamp TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS labels (
                    feature_id INTEGER PRIMARY KEY,
                    pair TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    label_binary INTEGER,
                    label_ternary INTEGER,
                    forward_pips REAL,
                    outcome TEXT,
                    pnl_usd REAL,
                    closed_at TEXT,
                    FOREIGN KEY (feature_id) REFERENCES features(id)
                )
            """)
            # NEW (Priority #1 — leakage audit): additive, nullable columns
            # so downstream code (DatasetBuilder) can tell which labeling
            # method produced a given row and what weight it should carry
            # during training. Defaults preserve every existing row's
            # current meaning: labeling_method='fixed_horizon' (what every
            # row up to now actually was) and sample_weight=1.0 (the
            # implicit uniform weight naive training already used).
            # ALTER TABLE ADD COLUMN raises if the column already exists —
            # each is wrapped individually so this migration is idempotent
            # and safe to run on every startup.
            for ddl in (
                "ALTER TABLE labels ADD COLUMN labeling_method TEXT DEFAULT 'fixed_horizon'",
                "ALTER TABLE labels ADD COLUMN sample_weight REAL DEFAULT 1.0",
                # NEW (leakage audit follow-up): tag every features row with
                # its origin. Existing rows (pre-migration) default to
                # 'live' — this is not perfectly accurate for any old
                # bootstrap rows written before this column existed, but
                # those should be removed via scripts/cleanup_bootstrap_leakage.py
                # (fingerprint-based), not silently mislabeled here.
                "ALTER TABLE features ADD COLUMN source TEXT DEFAULT 'live'",
            ):
                try:
                    c.execute(ddl)
                except sqlite3.OperationalError:
                    pass  # column already exists — migration already applied
            c.execute("""
                CREATE TABLE IF NOT EXISTS importance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair TEXT NOT NULL,
                    method TEXT NOT NULL,
                    ranking TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_features_pair_tf ON features(pair, timeframe)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_labels_pair_tf ON labels(pair, timeframe)")
            c.commit()

    def save_features(
        self,
        pair: str,
        timeframe: str,
        features: Dict[str, float],
        label: Optional[int] = None,
        forward_pips: Optional[float] = None,
        labeling_method: str = "fixed_horizon",
        sample_weight: float = 1.0,
        source: str = "live",
    ) -> int:
        """Save a feature vector + optional label. Returns the feature_id.

        labeling_method/sample_weight are NEW (Priority #1) and default to
        the values every existing row already implicitly had — omitting
        them is identical to current behavior.

        source: 'live' (default) for real market/trading data, or
            'bootstrap' for synthetic placeholder rows. Only
            ml.data_bootstrap should ever pass source='bootstrap'. This
            lets load_training_data() keep synthetic rows out of any
            training run that didn't explicitly ask for them.
        """
        with self._lock, self._conn() as c:
            cur = c.execute(
                "INSERT INTO features (pair, timeframe, feature_vector, feature_count, timestamp, source) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (pair.upper(), timeframe, json.dumps(features, default=str),
                 len(features), datetime.now(timezone.utc).isoformat(timespec="seconds"), source),
            )
            feature_id = cur.lastrowid
            if label is not None:
                c.execute(
                    """INSERT OR REPLACE INTO labels
                       (feature_id, pair, timeframe, label_binary, forward_pips, closed_at,
                        labeling_method, sample_weight)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (feature_id, pair.upper(), timeframe, int(label),
                     float(forward_pips) if forward_pips is not None else None,
                     datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     labeling_method, float(sample_weight)),
                )
            c.commit()
            return feature_id

    def update_outcome(self, feature_id: int, outcome: str, pnl_usd: float) -> None:
        """Update the actual trade outcome after a position closes."""
        with self._lock, self._conn() as c:
            c.execute(
                "UPDATE labels SET outcome = ?, pnl_usd = ?, closed_at = ? WHERE feature_id = ?",
                (outcome, float(pnl_usd),
                 datetime.now(timezone.utc).isoformat(timespec="seconds"), feature_id),
            )
            c.commit()

    def load_training_data(
        self,
        pair: Optional[str] = None,
        timeframe: Optional[str] = None,
        min_samples: int = None,
        include_bootstrap: bool = False,
    ) -> pd.DataFrame:
        """Load all feature vectors + labels as a DataFrame for ML training.

        Returns DataFrame with one row per sample: feature columns + 'label'.

        include_bootstrap: if False (default), rows tagged source='bootstrap'
            are excluded — training and any accuracy numbers you intend to
            trust should always use the default. Pass True only for
            first-run/dev flows that explicitly want the synthetic
            placeholder rows included (e.g. so the bootstrap fallback in
            ml.data_bootstrap can still let a from-scratch pipeline run
            end-to-end without crashing).
        """
        with self._lock, self._conn() as c:
            query = """
                SELECT f.id, f.pair, f.timeframe, f.feature_vector, f.timestamp,
                       l.label_binary, l.label_ternary, l.forward_pips, l.outcome, l.pnl_usd,
                       l.labeling_method, l.sample_weight, f.source
                FROM features f
                LEFT JOIN labels l ON f.id = l.feature_id
                WHERE 1=1
            """
            params: List[Any] = []
            if pair:
                query += " AND f.pair = ?"
                params.append(pair.upper())
            if timeframe:
                query += " AND f.timeframe = ?"
                params.append(timeframe)
            if not include_bootstrap:
                # f.source defaults to 'live' for pre-migration rows via the
                # ALTER TABLE default, so this only ever excludes rows
                # explicitly tagged 'bootstrap'.
                query += " AND (f.source IS NULL OR f.source != 'bootstrap')"
            query += " ORDER BY f.timestamp ASC"
            rows = c.execute(query, params).fetchall()

        min_samples_use = min_samples if min_samples is not None else MIN_TRAINING_SAMPLES
        if len(rows) < min_samples_use:
            log.info(f"[FeatureStore] only {len(rows)} samples (need ≥{min_samples_use}) — not enough for training")
            return pd.DataFrame()

        # Build DataFrame
        records = []
        for r in rows:
            try:
                feats = json.loads(r[3])
                feats["_id"] = r[0]
                feats["_pair"] = r[1]
                feats["_timeframe"] = r[2]
                feats["_timestamp"] = r[4]
                feats["label"] = r[5]  # label_binary
                feats["label_ternary"] = r[6]
                feats["forward_pips"] = r[7]
                feats["outcome"] = r[8]
                feats["pnl_usd"] = r[9]
                # NEW (Priority #1): default to the values every pre-migration
                # row implicitly had, so DatasetBuilder's sample_weight
                # handling is a no-op for historical data.
                feats["_labeling_method"] = r[10] if r[10] is not None else "fixed_horizon"
                feats["sample_weight"] = r[11] if r[11] is not None else 1.0
                feats["_source"] = r[12] if r[12] is not None else "live"
                records.append(feats)
            except Exception as e:
                log.debug(f"[FeatureStore] row parse failed: {e}")
        df = pd.DataFrame(records)
        log.info(f"[FeatureStore] loaded {len(df)} samples ({len(rows)} raw, include_bootstrap={include_bootstrap})")
        return df

    def stats(self) -> Dict[str, Any]:
        """Return store statistics."""
        with self._lock, self._conn() as c:
            total_features = c.execute("SELECT COUNT(*) FROM features").fetchone()[0]
            total_labels = c.execute("SELECT COUNT(*) FROM labels WHERE label_binary IS NOT NULL").fetchone()[0]
            total_outcomes = c.execute("SELECT COUNT(*) FROM labels WHERE outcome IS NOT NULL").fetchone()[0]
            by_pair = c.execute(
                "SELECT pair, COUNT(*) FROM features GROUP BY pair ORDER BY COUNT(*) DESC"
            ).fetchall()
            by_source = c.execute(
                "SELECT COALESCE(source, 'live'), COUNT(*) FROM features GROUP BY COALESCE(source, 'live')"
            ).fetchall()
            wins = c.execute("SELECT COUNT(*) FROM labels WHERE outcome = 'WIN'").fetchone()[0]
            losses = c.execute("SELECT COUNT(*) FROM labels WHERE outcome = 'LOSS'").fetchone()[0]
        return {
            "total_feature_rows": total_features,
            "total_labels": total_labels,
            "total_outcomes": total_outcomes,
            "wins": wins,
            "losses": losses,
            "win_rate_pct": round((wins / (wins + losses) * 100) if (wins + losses) else 0, 1),
            "by_pair": dict(by_pair),
            "by_source": dict(by_source),
        }

    def save_importance(self, pair: str, method: str, ranking: List[Dict[str, Any]]) -> None:
        """Save a feature importance ranking snapshot."""
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT INTO importance (pair, method, ranking, timestamp) VALUES (?, ?, ?, ?)",
                (pair.upper(), method, json.dumps(ranking, default=str),
                 datetime.now(timezone.utc).isoformat(timespec="seconds")),
            )
            c.commit()


# ── Singleton ───────────────────────────────────────────────────────

_STORE: Optional[FeatureStore] = None


def get_feature_store() -> FeatureStore:
    global _STORE
    if _STORE is None:
        _STORE = FeatureStore()
    return _STORE