"""
ml/ensemble_store.py — Ensemble decision persistence (Day 70)
================================================================

SQLite-backed store for ensemble decisions + model performance tracking.

Tables:
  * **ensemble_decisions** — every ensemble decision with per-model breakdown
  * **model_performance**  — rolling win/loss count per model (for weight adj)

The performance table is the "Model Performance Memory" — it tracks how
each model has performed over its last N trades and feeds that back into
ConfidenceFusion for dynamic weight adjustment.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

log = get_logger("ensemble_store")

DB_PATH = Path("memory/ensemble_decisions.db")


class EnsembleStore:
    """Persists ensemble decisions + tracks model performance."""

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
                CREATE TABLE IF NOT EXISTS ensemble_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    xgb_signal TEXT,
                    xgb_conf REAL,
                    rf_signal TEXT,
                    rf_conf REAL,
                    lstm_signal TEXT,
                    lstm_conf REAL,
                    rule_signal TEXT,
                    rule_conf REAL,
                    final_signal TEXT,
                    agreement TEXT,
                    confidence REAL,
                    position_size TEXT,
                    has_conflict INTEGER,
                    abstained INTEGER,
                    actual_result TEXT,
                    pnl_usd REAL,
                    timestamp TEXT NOT NULL,
                    closed_at TEXT
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    model_name TEXT PRIMARY KEY,
                    total_predictions INTEGER DEFAULT 0,
                    correct_predictions INTEGER DEFAULT 0,
                    win_rate REAL DEFAULT 0.0,
                    last_updated TEXT
                )
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_ens_pair ON ensemble_decisions(pair)")
            c.commit()

    def save_decision(self, decision: Dict[str, Any]) -> int:
        """Save an ensemble decision. Returns the row id."""
        with self._lock, self._conn() as c:
            cur = c.execute("""
                INSERT INTO ensemble_decisions
                (pair, timeframe, xgb_signal, xgb_conf, rf_signal, rf_conf,
                 lstm_signal, lstm_conf, rule_signal, rule_conf,
                 final_signal, agreement, confidence, position_size,
                 has_conflict, abstained, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                decision.get("pair", ""),
                decision.get("timeframe", ""),
                decision.get("xgb_signal"),
                decision.get("xgb_conf"),
                decision.get("rf_signal"),
                decision.get("rf_conf"),
                decision.get("lstm_signal"),
                decision.get("lstm_conf"),
                decision.get("rule_signal"),
                decision.get("rule_conf"),
                decision.get("final_signal"),
                decision.get("agreement"),
                decision.get("confidence"),
                decision.get("position_size"),
                1 if decision.get("has_conflict") else 0,
                1 if decision.get("abstained") else 0,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ))
            c.commit()
            return cur.lastrowid

    def update_outcome(self, decision_id: int, result: str, pnl_usd: float) -> None:
        """Update the actual outcome after the trade closes."""
        with self._lock, self._conn() as c:
            c.execute(
                "UPDATE ensemble_decisions SET actual_result = ?, pnl_usd = ?, closed_at = ? WHERE id = ?",
                (result, float(pnl_usd),
                 datetime.now(timezone.utc).isoformat(timespec="seconds"), decision_id),
            )
            c.commit()

    def update_model_performance(self, model_name: str, correct: bool) -> None:
        """Update a model's running win/loss count."""
        with self._lock, self._conn() as c:
            row = c.execute(
                "SELECT total_predictions, correct_predictions FROM model_performance WHERE model_name = ?",
                (model_name,),
            ).fetchone()
            if row:
                total = row[0] + 1
                wins = row[1] + (1 if correct else 0)
                wr = (wins / total) * 100
                c.execute(
                    "UPDATE model_performance SET total_predictions = ?, correct_predictions = ?, win_rate = ?, last_updated = ? WHERE model_name = ?",
                    (total, wins, round(wr, 1),
                     datetime.now(timezone.utc).isoformat(timespec="seconds"), model_name),
                )
            else:
                total = 1
                wins = 1 if correct else 0
                wr = (wins / total) * 100
                c.execute(
                    "INSERT INTO model_performance (model_name, total_predictions, correct_predictions, win_rate, last_updated) VALUES (?, ?, ?, ?, ?)",
                    (model_name, total, wins, round(wr, 1),
                     datetime.now(timezone.utc).isoformat(timespec="seconds")),
                )
            c.commit()

    def get_model_performance(self) -> Dict[str, Dict[str, Any]]:
        """Return performance stats for all models."""
        with self._lock, self._conn() as c:
            rows = c.execute(
                "SELECT model_name, total_predictions, correct_predictions, win_rate FROM model_performance"
            ).fetchall()
        return {
            row[0]: {
                "total": row[1],
                "correct": row[2],
                "win_rate": row[3],
            }
            for row in rows
        }

    def stats(self) -> Dict[str, Any]:
        """Return overall ensemble stats."""
        with self._lock, self._conn() as c:
            total = c.execute("SELECT COUNT(*) FROM ensemble_decisions").fetchone()[0]
            with_result = c.execute("SELECT COUNT(*) FROM ensemble_decisions WHERE actual_result IS NOT NULL").fetchone()[0]
            wins = c.execute("SELECT COUNT(*) FROM ensemble_decisions WHERE actual_result = 'WIN'").fetchone()[0]
            losses = c.execute("SELECT COUNT(*) FROM ensemble_decisions WHERE actual_result = 'LOSS'").fetchone()[0]
            abstained = c.execute("SELECT COUNT(*) FROM ensemble_decisions WHERE abstained = 1").fetchone()[0]
            conflicts = c.execute("SELECT COUNT(*) FROM ensemble_decisions WHERE has_conflict = 1").fetchone()[0]
            perf = self.get_model_performance()
        return {
            "total_decisions": total,
            "closed_with_result": with_result,
            "wins": wins,
            "losses": losses,
            "win_rate_pct": round((wins / (wins + losses) * 100) if (wins + losses) else 0, 1),
            "abstained": abstained,
            "conflicts": conflicts,
            "model_performance": perf,
        }


# ── Singleton ───────────────────────────────────────────────────────

_STORE: Optional[EnsembleStore] = None


def get_ensemble_store() -> EnsembleStore:
    global _STORE
    if _STORE is None:
        _STORE = EnsembleStore()
    return _STORE
