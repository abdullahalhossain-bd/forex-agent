"""
ml/feature_store.py — Persistent feature store (Day 68)
==========================================================

SQLite-backed feature store for ML training data. Stores every
feature vector + label + outcome so the ML model can learn over time.

Tables:
  * **features**    — feature_vector (JSON), pair, timeframe, timestamp
  * **labels**      — target label + outcome (filled in after trade closes)
  * **importance**  — feature importance rankings over time

CRITICAL: labels are added AFTER the trade closes — no future leakage.
The features table only contains info available at decision time.

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
    ) -> int:
        """Save a feature vector + optional label. Returns the feature_id.

        labeling_method/sample_weight are NEW (Priority #1) and default to
        the values every existing row already implicitly had — omitting
        them is identical to current behavior.
        """
        with self._lock, self._conn() as c:
            cur = c.execute(
                "INSERT INTO features (pair, timeframe, feature_vector, feature_count, timestamp) VALUES (?, ?, ?, ?, ?)",
                (pair.upper(), timeframe, json.dumps(features, default=str),
                 len(features), datetime.now(timezone.utc).isoformat(timespec="seconds")),
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
    ) -> pd.DataFrame:
        """Load all feature vectors + labels as a DataFrame for ML training.

        Returns DataFrame with one row per sample: feature columns + 'label'.
        """
        with self._lock, self._conn() as c:
            query = """
                SELECT f.id, f.pair, f.timeframe, f.feature_vector, f.timestamp,
                       l.label_binary, l.label_ternary, l.forward_pips, l.outcome, l.pnl_usd,
                       l.labeling_method, l.sample_weight
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
                records.append(feats)
            except Exception as e:
                log.debug(f"[FeatureStore] row parse failed: {e}")
        df = pd.DataFrame(records)
        log.info(f"[FeatureStore] loaded {len(df)} samples ({len(rows)} raw)")
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
