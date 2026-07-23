# analysis/shadow_follow_through_logger.py
# ============================================================
# BUG FIX (boot-breaking ModuleNotFoundError): agents/analysis_agent.py
# has imported `from analysis.shadow_follow_through_logger import
# get_shadow_logger` since the FollowThroughEngine shadow-mode wiring was
# added (see analysis_agent.py's "8.15 FollowThroughEngine — SHADOW MODE
# ONLY" comment block), but this module itself was never committed. Any
# `import agents.analysis_agent` — which core/trader.py does at module
# load time, and core/trading_engine.py in turn imports from core/trader
# — raised ModuleNotFoundError before a single line of AITrader ran,
# taking down the entire "runtime" boot phase (TradingEngine never wired,
# "Trader wired: NO", no trading loop starts at all). This is the
# implementation that was missing.
#
# SHADOW MODE, PER THE CALLER'S OWN COMMENT: this logger only OBSERVES
# BOS events + FollowThroughEngine's score, and later checks what
# actually happened. It has ZERO influence on any live trading decision.
# Nothing in this file is read by SignalFusion, RiskEngine, or
# DecisionValidator — it exists purely to accumulate evidence (BOS
# direction/score vs. actual subsequent price action) for the Phase-1
# rollout evidence bar described in analysis_agent.py (>=500 shadow
# events, positive score-vs-outcome correlation, pair/timeframe
# validated) before FollowThroughEngine's output is ever wired into a
# real decision path.
#
# Storage: a small SQLite file at memory/shadow_follow_through.db (one
# row per observed BOS event). Kept dependency-free (stdlib only) since
# this sits on the hot import path for every AITrader instance.
# ============================================================

from __future__ import annotations

import os
import sqlite3
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from utils.logger import get_logger

log = get_logger("shadow_follow_through_logger")

DB_PATH = os.path.join("memory", "shadow_follow_through.db")

# How many bars after the breakout to wait before checking whether price
# actually followed through, independent of FollowThroughEngine's own
# (shorter) confirm_bars window — this is a longer-horizon "did the move
# actually pay off" check for the shadow evidence log, not a re-run of
# the engine's own confirm/fail logic.
RESOLUTION_HORIZON_BARS = 20

# Minimum move (as a fraction of the breakout bar's level) required to
# call the outcome CORRECT rather than a coin-flip NEUTRAL — avoids
# crediting the prediction for noise-level drift in the predicted
# direction.
MIN_MOVE_FRACTION = 0.0005  # 5 pips on a ~1.0000 pair, scales with price


class ShadowFollowThroughLogger:
    """
    Usage (matches agents/analysis_agent.py's call sites exactly):

        shadow_logger = get_shadow_logger()
        shadow_logger.log_prediction(symbol, timeframe, bos, ft_result, df)
        ...
        get_shadow_logger().resolve_pending_outcomes(symbol, timeframe, df)

    Both methods are individually safe to call every cycle: log_prediction
    de-duplicates on (symbol, timeframe, breakout_index) so a BOS event
    that's still the "current" one across several cycles is only logged
    once; resolve_pending_outcomes is a cheap no-op once nothing is
    pending for that symbol/timeframe.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._ensure_schema()

    # ── Schema ────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_schema(self):
        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS shadow_predictions (
                        id                INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol            TEXT NOT NULL,
                        timeframe         TEXT NOT NULL,
                        breakout_index    INTEGER NOT NULL,
                        logged_at         TEXT NOT NULL,
                        direction         TEXT,
                        breakout_level    REAL,
                        status            TEXT,
                        score             INTEGER,
                        raw_score         INTEGER,
                        session           TEXT,
                        bos_event         TEXT,
                        bos_confidence    REAL,
                        resolved          INTEGER NOT NULL DEFAULT 0,
                        outcome           TEXT,
                        resolved_at       TEXT,
                        price_at_resolve  REAL,
                        UNIQUE(symbol, timeframe, breakout_index)
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_shadow_pending "
                    "ON shadow_predictions(symbol, timeframe, resolved)"
                )
        except Exception as e:
            log.warning(f"[ShadowFollowThrough] schema init failed: {e}")

    # ── Log a new observation ───────────────────────────────────

    def log_prediction(self, symbol: str, timeframe: str, bos: dict, ft_result, df: pd.DataFrame):
        """Record one BOS event + FollowThroughEngine's score. Silently
        ignores duplicate (symbol, timeframe, breakout_index) — the same
        BOS is commonly still "current" across multiple cycles until the
        next structure break."""
        try:
            r = ft_result
            if r is None or not getattr(r, "valid", False):
                return
            with self._lock, self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO shadow_predictions
                        (symbol, timeframe, breakout_index, logged_at,
                         direction, breakout_level, status, score, raw_score,
                         session, bos_event, bos_confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        symbol, timeframe, int(r.breakout_index),
                        datetime.now(timezone.utc).isoformat(),
                        r.direction, float(r.breakout_level), r.status,
                        int(r.score), int(r.raw_score), r.session,
                        (bos or {}).get("event"), (bos or {}).get("confidence"),
                    ),
                )
        except Exception as e:
            log.debug(f"[ShadowFollowThrough] log_prediction skipped ({symbol} {timeframe}): {e}")

    # ── Resolve older pending predictions ───────────────────────

    def resolve_pending_outcomes(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """Check any pending predictions for this symbol/timeframe whose
        RESOLUTION_HORIZON_BARS has elapsed, and mark them CORRECT /
        INCORRECT / NEUTRAL based on where price actually went. Cheap
        no-op once nothing is pending."""
        try:
            if df is None or len(df) == 0 or "close" not in df.columns:
                return
            last_index = len(df) - 1
            with self._lock, self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT id, breakout_index, direction, breakout_level
                    FROM shadow_predictions
                    WHERE symbol = ? AND timeframe = ? AND resolved = 0
                    """,
                    (symbol, timeframe),
                ).fetchall()

                if not rows:
                    return

                now_iso = datetime.now(timezone.utc).isoformat()
                current_price = float(df["close"].iloc[-1])

                for row_id, breakout_index, direction, breakout_level in rows:
                    bars_elapsed = last_index - int(breakout_index)
                    if bars_elapsed < RESOLUTION_HORIZON_BARS:
                        continue  # still pending, not enough bars yet

                    outcome = self._score_outcome(direction, breakout_level, current_price)
                    conn.execute(
                        """
                        UPDATE shadow_predictions
                        SET resolved = 1, outcome = ?, resolved_at = ?, price_at_resolve = ?
                        WHERE id = ?
                        """,
                        (outcome, now_iso, current_price, row_id),
                    )
        except Exception as e:
            log.debug(f"[ShadowFollowThrough] resolve_pending_outcomes skipped ({symbol} {timeframe}): {e}")

    @staticmethod
    def _score_outcome(direction: str, breakout_level: float, current_price: float) -> str:
        if not breakout_level:
            return "NEUTRAL"
        moved_frac = (current_price - breakout_level) / breakout_level
        threshold = MIN_MOVE_FRACTION
        if direction == "BULLISH":
            if moved_frac >= threshold:
                return "CORRECT"
            if moved_frac <= -threshold:
                return "INCORRECT"
        elif direction == "BEARISH":
            if moved_frac <= -threshold:
                return "CORRECT"
            if moved_frac >= threshold:
                return "INCORRECT"
        return "NEUTRAL"

    # ── Convenience for offline analysis (Phase-1 evidence bar) ──

    def summary(self) -> dict:
        """Quick counts for checking Phase-1 rollout readiness (>=500
        resolved shadow events, correlation between score and outcome)."""
        try:
            with self._lock, self._connect() as conn:
                total, resolved, correct, incorrect = conn.execute(
                    """
                    SELECT
                        COUNT(*),
                        SUM(resolved),
                        SUM(CASE WHEN outcome = 'CORRECT' THEN 1 ELSE 0 END),
                        SUM(CASE WHEN outcome = 'INCORRECT' THEN 1 ELSE 0 END)
                    FROM shadow_predictions
                    """
                ).fetchone()
            return {
                "total_logged": total or 0,
                "resolved": resolved or 0,
                "correct": correct or 0,
                "incorrect": incorrect or 0,
            }
        except Exception as e:
            log.warning(f"[ShadowFollowThrough] summary failed: {e}")
            return {"total_logged": 0, "resolved": 0, "correct": 0, "incorrect": 0}


_shadow_logger_instance: Optional[ShadowFollowThroughLogger] = None
_instance_lock = threading.Lock()


def get_shadow_logger() -> ShadowFollowThroughLogger:
    """Shared singleton — matches the get_*() accessor pattern used
    elsewhere in this codebase (e.g. get_follow_through_engine())."""
    global _shadow_logger_instance
    if _shadow_logger_instance is None:
        with _instance_lock:
            if _shadow_logger_instance is None:
                _shadow_logger_instance = ShadowFollowThroughLogger()
    return _shadow_logger_instance
