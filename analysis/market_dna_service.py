"""
analysis/market_dna_service.py — Market DNA Live Service
=========================================================

Wires the MarketDNA system into the live trading pipeline as a
runtime service. This is the missing piece that the README describes
but never implements.

Design:
  - Loads the ACTIVE frozen model once at process start
  - Provides predict_live() for the current bar's features
  - Returns a dna_context dict that downstream code reads:
      regime_ctx["market_dna"] = dna_context
  - Falls back gracefully (state=UNKNOWN) if no model is fitted yet
  - Caches cluster stats in memory for fast lookup

Usage in market_agent.py:
    from analysis.market_dna_service import get_market_dna_service
    svc = get_market_dna_service()
    dna_ctx = svc.predict_for_bar(df, symbol, timeframe)
    regime_ctx["market_dna"] = dna_ctx

Usage in trader.py:
    dna_ctx = market_out.get("regime", {}).get("market_dna", {})
    if dna_ctx.get("recommendation") == "REJECT":
        # Reduce position size or block trade
        risk_out["lot"] *= dna_ctx.get("position_multiplier", 1.0)
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from utils.logger import get_logger
from config import MODEL_DIR

log = get_logger("market_dna_service")

DNA_MODEL_DIR = MODEL_DIR / "market_dna"


class MarketDNAService:
    """Live trading service wrapper around MarketDNADetector.

    Loads the active model once, serves fast predict_live() calls.
    If no model is fitted yet, returns UNKNOWN state (safe fallback).
    """

    def __init__(self):
        self._detector = None
        self._cluster_stats: dict = {}  # cluster_id → ClusterStats dict
        self._model_id: Optional[str] = None
        self._loaded = False
        # BUGFIX (2026-08-19 audit): previously `self._loaded = True` was
        # set unconditionally at the top of _try_load(), so if the DB
        # tables didn't exist yet (setup_market_dna.py never run), the
        # service latched into "give up forever" for the rest of the
        # process lifetime. Every predict_for_bar() call after that first
        # failure silently returned state=UNKNOWN -> position_multiplier
        # =0.25, i.e. EVERY trade for the rest of the run was silently
        # sized at 25% with no further indication why. Now: a schema-
        # missing failure is tracked separately and retried periodically
        # (see _SCHEMA_RETRY_INTERVAL_SEC) so the service self-heals once
        # an operator runs setup_market_dna.py, without requiring a
        # process restart. Real (non-schema) errors still latch — those
        # indicate a genuine problem worth investigating, not a "not set
        # up yet" state.
        self._schema_missing = False
        self._last_schema_retry: float = 0.0

    _SCHEMA_RETRY_INTERVAL_SEC = 900  # re-check every 15 min once tables appear

    def _try_load(self) -> bool:
        """Try to load the active model from DB. Returns True if loaded."""
        if self._loaded:
            return self._detector is not None

        if self._schema_missing:
            # We've already confirmed the market_dna tables don't exist.
            # Don't hammer sqlite with a query every single bar — but do
            # periodically retry in case setup_market_dna.py was run
            # since our last check, so the live process can pick up the
            # model without a restart.
            import time as _time
            if _time.monotonic() - self._last_schema_retry < self._SCHEMA_RETRY_INTERVAL_SEC:
                return False
            self._last_schema_retry = _time.monotonic()

        try:
            # Find the ACTIVE model in DB
            with sqlite3.connect(str(_get_db_path())) as conn:
                # Explicitly check table existence first so we can tell
                # "not set up yet" (expected, retry later) apart from a
                # genuine DB error (permission, corruption — latch and
                # stop retrying, that needs a human to look at it).
                exists = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='market_dna_models'"
                ).fetchone()
                if not exists:
                    if not self._schema_missing:
                        log.warning(
                            "[MarketDNA Service] 'market_dna_models' table does not "
                            "exist yet — Market DNA has never been set up on this DB. "
                            "Run setup_market_dna.py to enable regime-aware position "
                            "sizing. Until then, EVERY trade will be sized at 25% "
                            "(position_multiplier=0.25) via the UNKNOWN-state fallback "
                            f"— will re-check every {self._SCHEMA_RETRY_INTERVAL_SEC}s."
                        )
                    self._schema_missing = True
                    return False

                row = conn.execute(
                    "SELECT model_id, model_path FROM market_dna_models "
                    "WHERE status='ACTIVE' ORDER BY trained_at DESC LIMIT 1"
                ).fetchone()
            self._schema_missing = False  # schema exists — stop the periodic-retry path
            if not row:
                log.info("[MarketDNA Service] No ACTIVE model in DB — returning UNKNOWN until trained")
                self._loaded = True
                return False

            model_id, model_path = row
            model_path = Path(model_path)
            if not model_path.exists():
                log.warning(f"[MarketDNA Service] Model file missing: {model_path}")
                return False

            # Load detector
            from analysis.market_dna import MarketDNADetector
            self._detector = MarketDNADetector.load(model_path)
            self._model_id = model_id

            # Load cluster stats
            with sqlite3.connect(str(_get_db_path())) as conn:
                rows = conn.execute(
                    "SELECT cluster_id, trades, wins, win_rate, ci_low, ci_high, "
                    "profit_factor, expectancy_r, tier, position_multiplier "
                    "FROM market_dna_cluster_stats WHERE model_id=?",
                    (model_id,)
                ).fetchall()
            self._cluster_stats = {}
            for r in rows:
                cid = r[0]
                self._cluster_stats[cid] = {
                    "cluster_id": cid,
                    "trades": r[1], "wins": r[2], "win_rate": r[3],
                    "ci_low": r[4], "ci_high": r[5],
                    "profit_factor": r[6], "expectancy_r": r[7],
                    "tier": r[8], "position_multiplier": r[9],
                }
            log.info(
                f"[MarketDNA Service] Loaded model {model_id} "
                f"({len(self._cluster_stats)} clusters tracked)"
            )
            return True
        except Exception as e:
            log.warning(f"[MarketDNA Service] Load failed: {e}")
            return False

    def predict_for_bar(self, df: pd.DataFrame, symbol: str, timeframe: str) -> dict:
        """Predict the market DNA context for the latest bar.

        Args:
            df: OHLCV DataFrame with indicator columns (must have at least 50 rows)
            symbol: Trading pair (e.g. "EURUSD")
            timeframe: e.g. "H1", "M15"

        Returns:
            dict with keys:
              state: "KNOWN" | "UNKNOWN"
              cluster_id: int | None
              confidence: float
              recommendation: "APPROVE" | "REJECT" | "REDUCE_SIZE"
              position_multiplier: float (0.25 to 1.0)
              win_rate: float | None
              tier: str | None
              reason: str
        """
        # Try to load model (lazy load on first call)
        if not self._try_load():
            return self._unknown_context("no model fitted yet — run setup_market_dna.py")

        # Ensure df has the required features
        try:
            from features.indicators_v5 import add_indicators, FEATURE_COLS

            # If df has only 1 row but already has features, use it directly
            # (the caller already computed indicators on the full df)
            if len(df) < 50:
                # Check if features are already present
                missing = [c for c in FEATURE_COLS if c not in df.columns]
                if missing:
                    return self._unknown_context(
                        f"insufficient bars ({len(df)}) and missing features {missing[:3]}"
                    )
                # Features present, proceed with single-row prediction

            # Add indicators ONLY if not already present
            missing_features = [c for c in FEATURE_COLS if c not in df.columns]
            if missing_features:
                df = add_indicators(df, drop_nan=True)
                if df.empty:
                    return self._unknown_context("indicator computation returned empty df")

            # Take the last bar
            last_bar = df.iloc[[-1]]

            # Predict
            result = self._detector.predict_live(last_bar)

            if result["state"] == "UNKNOWN":
                return self._unknown_context(
                    result.get("reason", "low confidence"),
                    confidence=result.get("confidence", 0.0),
                )

            # Look up cluster stats
            cluster_id = result["cluster_id"]
            stats = self._cluster_stats.get(cluster_id)
            if stats is None:
                return {
                    "state": "KNOWN",
                    "cluster_id": cluster_id,
                    "confidence": result["confidence"],
                    "recommendation": "REDUCE_SIZE",
                    "position_multiplier": 0.25,
                    "win_rate": None,
                    "tier": "NO_STATISTICAL_EDGE",
                    "reason": f"cluster {cluster_id} has no journal stats yet",
                    "model_id": self._model_id,
                }

            # Build recommendation from stats
            from analysis.dna_journal import decision_context, ClusterStats
            cs = ClusterStats(
                model_id=self._model_id,
                cluster_id=cluster_id,
                trades=stats["trades"],
                wins=stats["wins"],
                win_rate=stats["win_rate"],
                ci_low=stats["ci_low"],
                ci_high=stats["ci_high"],
                profit_factor=stats["profit_factor"],
                expectancy_r=stats["expectancy_r"],
                avg_holding_bars=None,
                tier=stats["tier"],
                position_multiplier=stats["position_multiplier"],
            )
            ctx = decision_context(cs)
            ctx["confidence"] = result["confidence"]
            ctx["model_id"] = self._model_id
            return ctx

        except Exception as e:
            log.warning(f"[MarketDNA Service] predict failed: {e}")
            return self._unknown_context(f"error: {e}")

    def _unknown_context(self, reason: str, confidence: float = 0.0) -> dict:
        return {
            "state": "UNKNOWN",
            "cluster_id": None,
            "confidence": confidence,
            "recommendation": "REDUCE_SIZE",
            "position_multiplier": 0.25,
            "win_rate": None,
            "tier": None,
            "reason": reason,
            "model_id": self._model_id,
        }

    def status(self) -> dict:
        """Return service status for dashboard."""
        # Trigger lazy load if not yet attempted
        if not self._loaded:
            self._try_load()
        return {
            "loaded": self._detector is not None,
            "model_id": self._model_id,
            "n_clusters_tracked": len(self._cluster_stats),
            "n_clusters_reliable": sum(
                1 for s in self._cluster_stats.values()
                if s["tier"] in ("RELIABLE", "HIGH_CONFIDENCE")
            ),
        }


def _get_db_path() -> Path:
    """Get DB path, handling both old and new config layouts."""
    try:
        from config import DB_PATH
        return Path(DB_PATH)
    except Exception:
        from database.db import DB_PATH
        return Path(DB_PATH)


# ── Singleton ───────────────────────────────────────────────────────

_SERVICE: Optional[MarketDNAService] = None


def get_market_dna_service() -> MarketDNAService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = MarketDNAService()
    return _SERVICE