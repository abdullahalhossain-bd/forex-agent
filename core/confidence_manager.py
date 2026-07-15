"""
core/confidence_manager.py — Dynamic Weight Adjustment (Day 73)
=================================================================

Tracks each intelligence layer's historical accuracy and dynamically
adjusts weights. Layers that perform better get more weight; layers
that perform poorly get less.

Layers tracked:
  - rule_engine (Day 67 Confluence)
  - ml_ensemble (Day 69-70 ML + Ensemble)
  - rl_agent (Day 71 RL)
  - llm_analyst (Day 42+ MasterAnalyst)

Default weights:
  rule_engine:  0.30
  ml_ensemble:  0.30
  rl_agent:     0.20
  llm_analyst:  0.20

After 20+ recorded outcomes, weights shift toward layers with higher
win rates. The shift is gradual (max ±0.10 from default) to avoid
overreacting to small samples.
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

log = get_logger("confidence_manager")

DB_PATH = Path("memory/confidence_manager.db")

DEFAULT_WEIGHTS = {
    "rule_engine":  0.30,
    "ml_ensemble":  0.30,
    "rl_agent":     0.20,
    "llm_analyst":  0.20,
}

MAX_WEIGHT_ADJUSTMENT = 0.10  # max deviation from default
MIN_SAMPLES_FOR_ADJUSTMENT = 20


class ConfidenceManager:
    """Manages dynamic weight adjustment based on historical accuracy."""

    def __init__(self):
        self._lock = threading.RLock()
        self._weights = DEFAULT_WEIGHTS.copy()
        self._init_db()

    def _init_db(self) -> None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(DB_PATH)) as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS layer_accuracy (
                    layer TEXT PRIMARY KEY,
                    total_predictions INTEGER DEFAULT 0,
                    correct_predictions INTEGER DEFAULT 0,
                    win_rate REAL DEFAULT 0.0,
                    last_updated TEXT
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS decision_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair TEXT,
                    layer TEXT,
                    predicted_signal TEXT,
                    actual_result TEXT,
                    correct INTEGER,
                    timestamp TEXT NOT NULL
                )
            """)
            c.commit()

    def record_outcome(self, layer: str, predicted_signal: str, actual_result: str) -> None:
        """Record one layer's prediction outcome (WIN/LOSS)."""
        correct = 1 if predicted_signal.upper() == actual_result.upper() else 0
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock, sqlite3.connect(str(DB_PATH)) as c:
            c.execute(
                "INSERT INTO decision_outcomes (pair, layer, predicted_signal, actual_result, correct, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                ("", layer, predicted_signal, actual_result, correct, ts),
            )
            row = c.execute(
                "SELECT total_predictions, correct_predictions FROM layer_accuracy WHERE layer = ?",
                (layer,),
            ).fetchone()
            if row:
                total = row[0] + 1
                wins = row[1] + correct
                wr = (wins / total) * 100
                c.execute(
                    "UPDATE layer_accuracy SET total_predictions = ?, correct_predictions = ?, win_rate = ?, last_updated = ? WHERE layer = ?",
                    (total, wins, round(wr, 1), ts, layer),
                )
            else:
                total = 1
                wins = correct
                wr = (wins / total) * 100
                c.execute(
                    "INSERT INTO layer_accuracy (layer, total_predictions, correct_predictions, win_rate, last_updated) VALUES (?, ?, ?, ?, ?)",
                    (layer, total, wins, round(wr, 1), ts),
                )
            c.commit()
        # Recalculate weights after each outcome
        self._recalculate_weights()

    def _recalculate_weights(self) -> None:
        """Adjust weights based on each layer's recent win rate."""
        with self._lock, sqlite3.connect(str(DB_PATH)) as c:
            rows = c.execute(
                "SELECT layer, total_predictions, correct_predictions, win_rate FROM layer_accuracy"
            ).fetchall()

        if not rows:
            return

        layer_stats = {row[0]: {"total": row[1], "wins": row[2], "wr": row[3]} for row in rows}

        # Only adjust if all layers have enough samples
        if any(s["total"] < MIN_SAMPLES_FOR_ADJUSTMENT for s in layer_stats.values()):
            return

        # Calculate average win rate
        avg_wr = sum(s["wr"] for s in layer_stats.values()) / len(layer_stats)

        # Adjust each weight: layers above average get +, below get -
        new_weights = {}
        for layer, default_w in DEFAULT_WEIGHTS.items():
            stats = layer_stats.get(layer, {"wr": avg_wr})
            deviation = (stats["wr"] - avg_wr) / 100  # -1 to +1
            adjustment = deviation * MAX_WEIGHT_ADJUSTMENT
            new_w = max(0.05, min(0.50, default_w + adjustment))
            new_weights[layer] = new_w

        # Normalize to sum = 1.0
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {k: v / total for k, v in new_weights.items()}

        self._weights = new_weights
        log.info(f"[ConfidenceManager] Weights adjusted: { {k: round(v, 3) for k, v in self._weights.items()} }")

    def get_weights(self) -> Dict[str, float]:
        """Return current weights (default or adjusted)."""
        with self._lock:
            return self._weights.copy()

    def get_layer_stats(self) -> Dict[str, Any]:
        """Return accuracy stats per layer."""
        with self._lock, sqlite3.connect(str(DB_PATH)) as c:
            rows = c.execute(
                "SELECT layer, total_predictions, correct_predictions, win_rate FROM layer_accuracy"
            ).fetchall()
        return {
            row[0]: {"total": row[1], "correct": row[2], "win_rate": row[3]}
            for row in rows
        }

    def status(self) -> Dict[str, Any]:
        return {
            "current_weights": {k: round(v, 3) for k, v in self._weights.items()},
            "default_weights": DEFAULT_WEIGHTS,
            "layer_stats": self.get_layer_stats(),
        }


# ── Singleton ───────────────────────────────────────────────────────

_MANAGER: Optional[ConfidenceManager] = None


def get_confidence_manager() -> ConfidenceManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = ConfidenceManager()
    return _MANAGER