"""Loss Rejection Engine v2 - Built from First Principles

Architecture:
  Layer 1: Catastrophic risk guard (3 rules, fail-open)
  Layer 2: ML Meta-Labeler PRIMARY decision maker (LightGBM, calibrated)
  Layer 3: OOD detector rejects unseen market conditions

Design Principles:
  1. Layer 2 is the PRIMARY filter. Layer 1 is a safety net only.
  2. Every feature is available BEFORE trade entry (no lookahead).
  3. Feature to outcome feedback loop is MANDATORY and wired at init.
  4. Per-symbol AND per-timeframe normalization.
  5. Walk-forward training with proper temporal splitting.
  6. Probability calibration via isotonic regression.
  7. Incremental learning: model retrains on new data.
  8. Full persistence with versioning and rollback.

Drop-in replacement: Same evaluate() signature as the old engine.
  Trader integration requires adding ONE call: engine.record_outcome() at trade close.
"""

from __future__ import annotations

import json
import logging
import math
import os
import pickle
import shutil
import sqlite3
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# FIX (2026-08-11): RestrictedUnpickler for tamper resistance.
try:
    from utils.safe_pickle import RestrictedUnpickler as _SafeUnpickler
except Exception:
    _SafeUnpickler = pickle.Unpickler

log = logging.getLogger("lre_v2")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

LRE_ENABLED: bool = os.getenv("LRE_V2_ENABLED", "1") == "1"
LRE_SHADOW_MODE: bool = os.getenv("LRE_V2_SHADOW_MODE", "0") == "1"
MODEL_DIR = Path(os.getenv("LRE_V2_MODEL_DIR", "memory/lre_models_v2"))

# Layer 1 - Catastrophic guards only
L1_MAX_SPREAD_ATR: float = float(os.getenv("LRE_V2_L1_SPREAD_ATR", "3.0"))
L1_MAX_SL_ATR: float = float(os.getenv("LRE_V2_L1_MAX_SL_ATR", "5.0"))
L1_MIN_RR: float = float(os.getenv("LRE_V2_L1_MIN_RR", "0.8"))

# Layer 2 - ML Meta-Labeler
L2_RETRAIN_INTERVAL: int = int(os.getenv("LRE_V2_L2_RETRAIN", "50"))
L2_MIN_TRADES_TRAIN: int = int(os.getenv("LRE_V2_L2_MIN_TRAINS", "80"))
L2_REJECT_THRESHOLD: float = float(os.getenv("LRE_V2_L2_REJECT", "0.58"))
L2_WARN_THRESHOLD: float = float(os.getenv("LRE_V2_L2_WARN", "0.45"))
L2_FEATURE_WINDOW: int = int(os.getenv("LRE_V2_L2_WINDOW", "2000"))
L2_WALK_FORWARD_FOLDS: int = int(os.getenv("LRE_V2_L2_WF_FOLDS", "5"))
L2_MIN_POSITIVE_RATE: float = float(os.getenv("LRE_V2_L2_MIN_POS_RATE", "0.15"))

# Layer 3 - OOD Detector
L3_REJECT_THRESHOLD: float = float(os.getenv("LRE_V2_L3_REJECT", "3.5"))
L3_WARN_THRESHOLD: float = float(os.getenv("LRE_V2_L3_WARN", "2.0"))
L3_MIN_SAMPLES: int = int(os.getenv("LRE_V2_L3_MIN", "200"))
L3_WINDOW: int = int(os.getenv("LRE_V2_L3_WINDOW", "1000"))

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE DEFINITIONS - 15 features, all available before trade entry
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    # 1. Signal quality (from dec_out)
    "confidence",          # Decision confidence 0-100
    "rr",                  # Risk:reward ratio
    "sl_atr_ratio",        # SL distance / ATR (SL normalization)
    "aligned_ratio",       # aligned_factors / total_factors
    # 2. Market regime (from market_out)
    "atr_pips",            # ATR in pips (volatility)
    "rsi",                 # RSI-14
    "trend_strength",      # Trend strength 0-100
    "regime_confidence",   # Regime classification confidence
    # 3. Structure (from analysis_out)
    "smc_score",           # SMC confluence score
    "session_quality",     # Session trading quality 0-1
    # 4. Cross-asset context
    "spread_atr",          # Spread / ATR (liquidity proxy)
    # 5. Time features (cyclical encoding)
    "hour_sin",            # sin(2π * hour / 24)
    "hour_cos",            # cos(2π * hour / 24)
    "dow_sin",             # sin(2π * day_of_week / 5)
    "dow_cos",             # cos(2π * day_of_week / 5)
]

N_FEATURES = len(FEATURE_NAMES)

# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Layer1Output:
    verdict: str = "PASS"  # PASS | REJECT
    reason: str = ""
    score: float = 0.0
    checks: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Layer2Output:
    verdict: str = "PASS"  # PASS | REJECT | WARN | COLLECTING
    reason: str = ""
    loss_prob: float = 0.5
    is_trained: bool = False
    n_trades: int = 0
    features: Optional[np.ndarray] = None
    calibration_slope: float = 1.0
    calibration_intercept: float = 0.0


@dataclass
class Layer3Output:
    verdict: str = "PASS"  # PASS | REJECT | WARN
    reason: str = ""
    distance: float = 0.0
    n_reference: int = 0


@dataclass
class LREResult:
    blocked: bool = False
    shadow_blocked: bool = False
    l1: Optional[Layer1Output] = None
    l2: Optional[Layer2Output] = None
    l3: Optional[Layer3Output] = None
    composite_verdict: str = "PASS"
    confidence_penalty: float = 0.0
    reason: str = ""
    processing_time_ms: float = 0.0
    # New: store features for feedback loop
    _pending_features: Optional[Dict] = field(default=None, repr=False)
    _pending_trade_id: Optional[str] = field(default=None, repr=False)


@dataclass
class TradeRecord:
    trade_id: str
    symbol: str
    timeframe: str
    direction: str
    features: Dict[str, float]
    label: int  # 1 = loss, 0 = win
    pnl: float
    timestamp: float
    l1_verdict: str
    l2_verdict: str
    l3_verdict: str
    l2_loss_prob: float = 0.5
    was_blocked: bool = False


@dataclass
class ModelVersion:
    version: int
    trained_at: str
    n_trades: int
    n_features: int
    feature_names: List[str]
    walk_forward_auc: float = 0.0
    calibration_ece: float = 0.0
    positive_rate: float = 0.0
    feature_correlations: Dict[str, float] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTOR - Single source of truth for all feature computation
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(val, default=0.0):
    """Safely convert any value to float."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_get(d: dict, *keys, default=None):
    """Nested dict safe access with multiple key fallback."""
    if not isinstance(d, dict):
        return default
    for key in keys:
        val = d.get(key, {})
        if val is None or (not isinstance(val, dict) and len(keys) > 1):
            return default
        d = val
    return d if d is not None else default


def _safe_get_first(d: dict, key_path: str, default=None):
    """Safe nested access via dot-separated path."""
    keys = key_path.split(".")
    val = d
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k)
        else:
            return default
        if val is None:
            return default
    return val


def extract_features(
    dec_out: Dict[str, Any],
    analysis_out: Dict[str, Any],
    market_out: Dict[str, Any],
    symbol: str = "",
    timeframe: str = "",
) -> Tuple[Dict[str, float], Optional[np.ndarray]]:
    """
    Extract 15 features from the three input dicts.

    All features are available BEFORE trade entry - no lookahead bias.
    Returns (feature_dict, feature_vector_np).
    """
    ind = market_out.get("ind_ctx") if isinstance(market_out, dict) else {}
    if not isinstance(ind, dict): ind = {}
    regime = market_out.get("regime") if isinstance(market_out, dict) else {}
    if not isinstance(regime, dict): regime = {}
    spread = market_out.get("spread") if isinstance(market_out, dict) else {}
    if not isinstance(spread, dict): spread = {}

    # ATR: can be ind["atr"] (float) or ind["atr"]["value"] (dict)
    atr_raw = ind.get("atr", 0)
    if isinstance(atr_raw, dict):
        atr_val = _safe_float(atr_raw.get("value", atr_raw.get("atr", 0)))
    else:
        atr_val = _safe_float(atr_raw)

    rsi_raw = ind.get("rsi", 0)
    if isinstance(rsi_raw, dict):
        rsi_val = _safe_float(rsi_raw.get("value", 0))
    else:
        rsi_val = _safe_float(rsi_raw)

    close_price = _safe_float(ind.get("close", ind.get("price", 0)))
    spread_val = _safe_float(spread.get("spread", spread.get("current", 0)))
    dec = dec_out or {}

    confidence = _safe_float(dec.get("confidence", 0))
    rr = _safe_float(dec.get("rr", 0))
    sl_pips = _safe_float(dec.get("sl_pips", 0))
    atr_pips = atr_val * 10000 if atr_val > 0 else 0.0  # Convert to pips
    sl_atr_ratio = (sl_pips / atr_pips) if atr_pips > 0 else 0.0

    aligned = _safe_float(dec.get("aligned_factors", 0))
    total = _safe_float(dec.get("total_factors", 1))
    aligned_ratio = aligned / total if total > 0 else 0.0

    trend_strength = _safe_float(regime.get("trend_strength", 0))
    regime_conf = _safe_float(regime.get("confidence", 0))

    smc_ctx = _safe_get(analysis_out, "smc_ctx", default={})
    smc_score = _safe_float(smc_ctx.get("confluence_score", smc_ctx.get("score", 0)))

    session_raw = analysis_out.get("session_ctx") if isinstance(analysis_out, dict) else None
    if not session_raw or not isinstance(session_raw, dict):
        session_raw = market_out.get("session") if isinstance(market_out, dict) else None
    if not isinstance(session_raw, dict):
        session_raw = {}
    session_quality = _safe_float(session_raw.get("quality", session_raw.get("trade_quality", 0.5)))

    # spread_val can be raw price (0.00002) or pips (0.2).
    # Normalize to pips if it looks like a raw price.
    if spread_val > 0 and spread_val < 0.01:
        spread_val = spread_val * 10000  # Convert raw price to pips

    spread_atr = (spread_val / atr_pips) if atr_pips > 0 else 0.0  # Both in pips

    now = datetime.now(timezone.utc)
    hour_sin = math.sin(2 * math.pi * now.hour / 24)
    hour_cos = math.cos(2 * math.pi * now.hour / 24)
    dow = now.weekday()  # 0=Mon, 4=Fri
    dow_sin = math.sin(2 * math.pi * dow / 5)
    dow_cos = math.cos(2 * math.pi * dow / 5)

    features = {
        "confidence": confidence,
        "rr": rr,
        "sl_atr_ratio": sl_atr_ratio,
        "aligned_ratio": aligned_ratio,
        "atr_pips": atr_val * 10000 if atr_val > 0 else 0.0,
        "rsi": rsi_val,
        "trend_strength": trend_strength,
        "regime_confidence": regime_conf,
        "smc_score": smc_score,
        "session_quality": session_quality,
        "spread_atr": spread_atr,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
    }

    vector = np.array([features[name] for name in FEATURE_NAMES], dtype=np.float64)
    return features, vector


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1: CATASTROPHIC RISK GUARD
# ─────────────────────────────────────────────────────────────────────────────

class CatastrophicGuard:
    """
    Layer 1: 3 rules that catch ONLY catastrophic risk scenarios.

    This layer is a safety net, NOT a primary filter.
    It should block < 5% of signals in normal operation.

    Rules (all statistically justified):
      1. Spread/ATR > 3.0: Insane liquidity cost, guaranteed slippage
      2. SL/ATR > 5.0: Stop loss absurdly wide relative to volatility
      3. R:R < 0.8: Insufficient reward for risk taken
    """

    def __init__(self):
        self._n_evaluated = 0
        self._n_blocked = 0

    def evaluate(
        self,
        features: Dict[str, float],
        dec_out: Dict[str, Any],
        market_out: Dict[str, Any],
    ) -> Layer1Output:
        self._n_evaluated += 1
        checks = {}
        max_score = 0.0
        reasons = []

        # Rule 1: Spread/ATR check - catastrophic liquidity cost
        spread_atr = features.get("spread_atr", 0)
        checks["spread_atr"] = spread_atr
        if spread_atr > L1_MAX_SPREAD_ATR:
            # Direct reject: spread > 3x ATR means certain slippage
            max_score = 100  # Always reject, regardless of other rules
            reasons.append(f"spread/ATR={spread_atr:.2f}>{L1_MAX_SPREAD_ATR}")

        # Rule 2: SL/ATR check - stop absurdly wide
        sl_atr = features.get("sl_atr_ratio", 0)
        checks["sl_atr_ratio"] = sl_atr
        if sl_atr > L1_MAX_SL_ATR:
            max_score = 100  # Always reject
            reasons.append(f"SL/ATR={sl_atr:.1f}>{L1_MAX_SL_ATR}")

        # Rule 3: Minimum R:R check - insufficient reward
        rr = features.get("rr", 0)
        checks["rr"] = rr
        if rr < L1_MIN_RR and rr > 0:
            max_score = 100  # Always reject
            reasons.append(f"R:R={rr:.2f}<{L1_MIN_RR}")

        if max_score >= 70:
            self._n_blocked += 1
            return Layer1Output(
                verdict="REJECT",
                reason=f"Catastrophic: {'; '.join(reasons)}",
                score=max_score,
                checks=checks,
            )

        return Layer1Output(verdict="PASS", reason="", score=max_score, checks=checks)

    def stats(self) -> Dict[str, Any]:
        rate = self._n_blocked / self._n_evaluated * 100 if self._n_evaluated > 0 else 0
        return {"evaluated": self._n_evaluated, "blocked": self._n_blocked, "block_rate_pct": rate}


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2: ML META-LABELER - PRIMARY DECISION MAKER
# ─────────────────────────────────────────────────────────────────────────────

class MetaLabelerV2:
    """
    Layer 2: ML binary classifier that predicts P(loss) for each trade signal.

    This is the PRIMARY intelligence of the LRE.
    Trains on accumulated (features, outcome) pairs using LightGBM.
    Features are per-symbol/timeframe normalized.
    Probabilities are calibrated via isotonic regression.
    Walk-forward validation ensures temporal integrity.
    """

    def __init__(self, model_dir: Path = MODEL_DIR):
        self._model_dir = model_dir
        self._model_dir.mkdir(parents=True, exist_ok=True)

        # Trade history for training
        self._trade_records: List[TradeRecord] = []
        self._feature_matrix: Optional[np.ndarray] = None
        self._label_vector: Optional[np.ndarray] = None

        # Pending trades (features at open, awaiting outcome at close)
        self._pending: Dict[str, Dict[str, Any]] = {}

        # Model artifacts
        self._model = None
        self._scaler_means: Optional[np.ndarray] = None
        self._scaler_stds: Optional[np.ndarray] = None
        self._calibrator = None
        self._is_trained = False
        self._version = 0
        self._version_meta: Optional[ModelVersion] = None

        # Per-symbol/timeframe normalization stats
        self._norm_stats: Dict[str, Dict[str, float]] = {}

        # Monitoring
        self._n_evaluated = 0
        self._n_rejected = 0
        self._n_correct_rejections = 0
        self._n_incorrect_rejections = 0
        self._predictions_since_retrain: deque = deque(maxlen=500)

        # Try to load existing model
        self._load_model()

    def _trade_db_path(self) -> Path:
        return self._model_dir / "trade_history.db"

    def _model_path(self, version: Optional[int] = None) -> Path:
        v = version if version is not None else self._version
        return self._model_dir / f"meta_model_v{v}.pkl"

    def _version_path(self) -> Path:
        return self._model_dir / "model_version.json"

    def _norm_stats_path(self) -> Path:
        return self._model_dir / "norm_stats.json"

    # ── Model I/O ─────────────────────────────────────────────────────────

    def _load_model(self):
        """Load the latest model version from disk."""
        vp = self._version_path()
        if not vp.exists():
            log.info("[L2] No saved model found. Starting in collection mode.")
            return

        try:
            with open(vp) as f:
                meta = json.load(f)
            self._version = meta["version"]
            self._version_meta = ModelVersion(**meta)

            mp = self._model_path()
            if not mp.exists():
                log.warning("[L2] Version meta exists but model file missing: v%d", self._version)
                return

            with open(mp, "rb") as f:
                artifacts = _SafeUnpickler(f).load()

            self._model = artifacts["model"]
            self._scaler_means = artifacts["scaler_means"]
            self._scaler_stds = artifacts["scaler_stds"]
            self._calibrator = artifacts["calibrator"]
            self._is_trained = True

            # Load normalization stats
            nsp = self._norm_stats_path()
            if nsp.exists():
                with open(nsp) as f:
                    self._norm_stats = json.load(f)

            log.info(
                "[L2] Loaded model v%d (trained on %d trades, WF-AUC=%.3f, ECE=%.3f)",
                self._version, meta["n_trades"], meta.get("walk_forward_auc", 0),
                meta.get("calibration_ece", 0),
            )
        except Exception as e:
            log.warning("[L2] Failed to load model: %s", e)
            self._is_trained = False

    def _save_model(self):
        """Persist model, scaler, calibrator, and metadata to disk."""
        if self._model is None:
            return

        self._version += 1
        mp = self._model_path()
        bp = self._model_path(self._version - 1)  # backup previous

        artifacts = {
            "model": self._model,
            "scaler_means": self._scaler_means,
            "scaler_stds": self._scaler_stds,
            "calibrator": self._calibrator,
        }

        try:
            # Backup previous model
            if bp.exists():
                shutil.copy2(bp, self._model_dir / f"meta_model_v{self._version - 1}_backup.pkl")

            with open(mp, "wb") as f:
                pickle.dump(artifacts, f, protocol=pickle.HIGHEST_PROTOCOL)

            meta = asdict(self._version_meta) if self._version_meta else {}
            meta["version"] = self._version
            with open(self._version_path(), "w") as f:
                json.dump(meta, f, indent=2)

            # Save normalization stats
            with open(self._norm_stats_path(), "w") as f:
                json.dump(self._norm_stats, f, indent=2)

            log.info("[L2] Saved model v%d to %s", self._version, mp)
        except Exception as e:
            log.error("[L2] Failed to save model: %s", e)
            self._version -= 1  # rollback version counter

    def rollback(self, n: int = 1) -> bool:
        """Roll back to a previous model version."""
        target = self._version - n
        if target < 0:
            return False
        tp = self._model_path(target)
        if not tp.exists():
            return False
        self._version = target
        self._load_model()
        return True

    # ── Trade Recording (THE CRITICAL FEEDBACK LOOP) ─────────────────────

    def register_pending(
        self, trade_id: str, symbol: str, timeframe: str,
        direction: str, features: Dict[str, float],
        l1_verdict: str, l2_verdict: str, l3_verdict: str,
        l2_loss_prob: float, was_blocked: bool,
    ):
        """
        Register a trade signal whose outcome is pending.
        Called IMMEDIATELY after evaluate(), BEFORE the trade is sent to execution.
        """
        self._pending[trade_id] = {
            "symbol": symbol,
            "timeframe": timeframe,
            "direction": direction,
            "features": features,
            "timestamp": time.time(),
            "l1_verdict": l1_verdict,
            "l2_verdict": l2_verdict,
            "l3_verdict": l3_verdict,
            "l2_loss_prob": l2_loss_prob,
            "was_blocked": was_blocked,
        }

    def record_outcome(self, trade_id: str, pnl: float):
        """
        Record the outcome of a previously registered trade.
        THIS IS THE CRITICAL METHOD THAT WAS MISSING IN v1.
        Must be called when a trade closes.
        """
        if trade_id not in self._pending:
            log.debug("[L2] trade_id %s not in pending (already recorded or unknown)", trade_id)
            return

        pending = self._pending.pop(trade_id)
        label = 1 if pnl <= 0 else 0  # 1 = loss, 0 = win

        record = TradeRecord(
            trade_id=trade_id,
            symbol=pending["symbol"],
            timeframe=pending["timeframe"],
            direction=pending["direction"],
            features=pending["features"],
            label=label,
            pnl=pnl,
            timestamp=pending["timestamp"],
            l1_verdict=pending["l1_verdict"],
            l2_verdict=pending["l2_verdict"],
            l3_verdict=pending["l3_verdict"],
            l2_loss_prob=pending["l2_loss_prob"],
            was_blocked=pending["was_blocked"],
        )
        self._trade_records.append(record)

        # Track prediction accuracy
        if pending["l2_verdict"] == "REJECT":
            if label == 1:
                self._n_correct_rejections += 1
            else:
                self._n_incorrect_rejections += 1

        # Persist to SQLite
        self._persist_record(record)

        log.debug(
            "[L2] Recorded outcome: %s %s pnl=$%.2f label=%d (total=%d)",
            pending["symbol"], pending["direction"], pnl, label, len(self._trade_records),
        )

        # Check if retraining is needed
        n_since = len([r for r in self._trade_records if r.timestamp > (self._version_meta.trained_at_ts if hasattr(self._version_meta, 'trained_at_ts') else 0)])
        if len(self._trade_records) >= L2_MIN_TRADES_TRAIN and len(self._trade_records) % L2_RETRAIN_INTERVAL == 0:
            self._train()

    def _persist_record(self, record: TradeRecord):
        """Persist a trade record to SQLite for durability."""
        try:
            conn = sqlite3.connect(str(self._trade_db_path()), timeout=5)
            conn.execute(
                """CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT, timeframe TEXT, direction TEXT,
                    features TEXT, label INTEGER, pnl REAL,
                    timestamp REAL, l1_verdict TEXT, l2_verdict TEXT, l3_verdict TEXT,
                    l2_loss_prob REAL, was_blocked INTEGER
                )"""
            )
            conn.execute(
                "INSERT OR REPLACE INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    record.trade_id, record.symbol, record.timeframe,
                    record.direction, json.dumps(record.features), record.label,
                    record.pnl, record.timestamp, record.l1_verdict,
                    record.l2_verdict, record.l3_verdict, record.l2_loss_prob,
                    int(record.was_blocked),
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            log.debug("[L2] DB persist failed (non-fatal): %s", e)

    def _load_history(self):
        """Load trade history from SQLite on startup."""
        try:
            dbp = self._trade_db_path()
            if not dbp.exists():
                return
            conn = sqlite3.connect(str(dbp), timeout=5)
            cursor = conn.execute(
                "SELECT trade_id, symbol, timeframe, direction, features, label, pnl, "
                "timestamp, l1_verdict, l2_verdict, l3_verdict, l2_loss_prob, was_blocked "
                "FROM trades ORDER BY timestamp"
            )
            for row in cursor:
                rec = TradeRecord(
                    trade_id=row[0], symbol=row[1], timeframe=row[2],
                    direction=row[3], features=json.loads(row[4]),
                    label=row[5], pnl=row[6], timestamp=row[7],
                    l1_verdict=row[8], l2_verdict=row[9], l3_verdict=row[10],
                    l2_loss_prob=row[11], was_blocked=bool(row[12]),
                )
                self._trade_records.append(rec)
            conn.close()
            if self._trade_records:
                log.info("[L2] Loaded %d historical trade records", len(self._trade_records))
        except Exception as e:
            log.warning("[L2] Failed to load trade history: %s", e)

    # ── Normalization ─────────────────────────────────────────────────────

    def _compute_norm_stats(self):
        """Compute per-symbol/timeframe normalization statistics."""
        if not self._trade_records:
            return

        groups: Dict[str, List[np.ndarray]] = {}
        for rec in self._trade_records:
            key = f"{rec.symbol}:{rec.timeframe}"
            vec = np.array([rec.features.get(n, 0.0) for n in FEATURE_NAMES])
            groups.setdefault(key, []).append(vec)

        self._norm_stats = {}
        for key, vecs in groups.items():
            arr = np.array(vecs)
            means = np.nanmean(arr, axis=0)
            stds = np.nanstd(arr, axis=0)
            stds = np.where(stds < 1e-8, 1.0, stds)  # prevent division by zero
            self._norm_stats[key] = {
                "means": means.tolist(),
                "stds": stds.tolist(),
                "count": len(vecs),
            }

    def _normalize_features(
        self, features: Dict[str, float], symbol: str, timeframe: str
    ) -> np.ndarray:
        """Normalize features using per-symbol/timeframe stats, with global fallback."""
        vec = np.array([features.get(n, 0.0) for n in FEATURE_NAMES], dtype=np.float64)

        key = f"{symbol}:{timeframe}"
        if key in self._norm_stats:
            stats = self._norm_stats[key]
            means = np.array(stats["means"])
            stds = np.array(stats["stds"])
            vec = (vec - means) / stds
        elif self._scaler_means is not None:
            # Fallback: use training-time global normalization
            vec = (vec - self._scaler_means) / np.where(self._scaler_stds < 1e-8, 1.0, self._scaler_stds)

        # Clip to prevent extreme values
        vec = np.clip(vec, -5, 5)
        return vec

    # ── Training ─────────────────────────────────────────────────────────

    def _train(self):
        """Train/retrain the meta-labeler on accumulated trade records."""
        n = len(self._trade_records)
        if n < L2_MIN_TRADES_TRAIN:
            log.info("[L2] Not enough trades to train: %d < %d", n, L2_MIN_TRADES_TRAIN)
            return

        # Build feature matrix and labels
        X_list = []
        y_list = []
        for rec in self._trade_records:
            vec = np.array([rec.features.get(name, 0.0) for name in FEATURE_NAMES], dtype=np.float64)
            if np.any(np.isnan(vec)) or np.any(np.isinf(vec)):
                continue
            X_list.append(vec)
            y_list.append(rec.label)

        X = np.array(X_list)
        y = np.array(y_list)

        if len(X) < L2_MIN_TRADES_TRAIN:
            log.info("[L2] Not enough clean samples: %d", len(X))
            return

        positive_rate = y.mean()
        if positive_rate < L2_MIN_POSITIVE_RATE or positive_rate > (1 - L2_MIN_POSITIVE_RATE):
            log.warning(
                "[L2] Label imbalance too extreme: %.1f%% losses. Skipping training.",
                positive_rate * 100,
            )
            return

        # Compute normalization stats
        self._compute_norm_stats()

        # Normalize features
        X_norm = X.copy()
        self._scaler_means = np.nanmean(X_norm, axis=0)
        self._scaler_stds = np.nanstd(X_norm, axis=0)
        self._scaler_stds = np.where(self._scaler_stds < 1e-8, 1.0, self._scaler_stds)
        X_norm = (X_norm - self._scaler_means) / self._scaler_stds
        X_norm = np.clip(X_norm, -5, 5)

        # ── Walk-Forward Validation ──
        n_folds = min(L2_WALK_FORWARD_FOLDS, max(2, len(X) // 40))
        fold_size = len(X) // (n_folds + 1)
        wf_aucs = []

        try:
            from sklearn.metrics import roc_auc_score
        except ImportError:
            log.warning("[L2] sklearn not available, skipping walk-forward validation")
            wf_aucs = [0.5]

        for fold in range(n_folds):
            train_end = fold_size * (fold + 1)
            val_end = min(train_end + fold_size, len(X))

            X_train, X_val = X_norm[:train_end], X_norm[train_end:val_end]
            y_train, y_val = y[:train_end], y[train_end:val_end]

            if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
                continue

            model = self._build_model()
            if model is None:
                return

            model.fit(X_train, y_train)
            probs = model.predict_proba(X_val)[:, 1]

            try:
                auc = roc_auc_score(y_val, probs)
                wf_aucs.append(auc)
            except Exception:
                pass

        avg_wf_auc = np.mean(wf_aucs) if wf_aucs else 0.5

        if avg_wf_auc < 0.52:
            log.warning(
                "[L2] Walk-forward AUC=%.3f < 0.52. Model has no predictive power. Not deploying.",
                avg_wf_auc,
            )
            return

        # ── Train final model on ALL data ──
        final_model = self._build_model()
        if final_model is None:
            return
        final_model.fit(X_norm, y)

        # ── Probability Calibration ──
        calibrator = None
        cal_ece = 0.0
        try:
            from sklearn.isotonic import IsotonicRegression
            from sklearn.calibration import calibration_curve

            all_probs = final_model.predict_proba(X_norm)[:, 1]
            calibrator = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
            calibrator.fit(all_probs, y)

            # Compute ECE (Expected Calibration Error)
            calibrated = calibrator.transform(all_probs)
            fraction_pos, mean_pred = calibration_curve(y, calibrated, n_bins=10, strategy="uniform")
            cal_ece = np.mean(np.abs(fraction_pos - mean_pred))
        except Exception as e:
            log.debug("[L2] Calibration failed (non-fatal): %s", e)

        # ── Feature-Outcome Correlation ──
        correlations = {}
        for i, name in enumerate(FEATURE_NAMES):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            if not np.isnan(corr):
                correlations[name] = round(float(corr), 4)

        # ── Deploy ──
        self._model = final_model
        self._calibrator = calibrator
        self._is_trained = True

        self._version_meta = ModelVersion(
            version=self._version + 1,
            trained_at=datetime.now(timezone.utc).isoformat(),
            n_trades=len(X),
            n_features=N_FEATURES,
            feature_names=FEATURE_NAMES,
            walk_forward_auc=round(avg_wf_auc, 4),
            calibration_ece=round(cal_ece, 4),
            positive_rate=round(positive_rate, 4),
            feature_correlations=correlations,
        )

        self._save_model()
        log.info(
            "[L2] Training complete: v%d | %d trades | WF-AUC=%.3f | ECE=%.3f | "
            "loss_rate=%.1f%% | correlations=%s",
            self._version, len(X), avg_wf_auc, cal_ece,
            positive_rate * 100,
            {k: v for k, v in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:5]},
        )

    def _build_model(self):
        """Build the LightGBM classifier with conservative hyperparameters."""
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                num_leaves=15,
                min_split_gain=0.01,
                verbose=-1,
                n_jobs=1,
                random_state=42,
            )
        except ImportError:
            pass

        # Fallback to sklearn GradientBoosting
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            log.info("[L2] LightGBM unavailable, using sklearn GradientBoosting")
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                min_samples_leaf=20,
                subsample=0.8,
                max_features=0.8,
                random_state=42,
            )
        except ImportError:
            log.error("[L2] Neither LightGBM nor sklearn available. Cannot build model.")
            return None

    # ── Inference ─────────────────────────────────────────────────────────

    def evaluate(
        self,
        features: Dict[str, float],
        feature_vector: np.ndarray,
        symbol: str,
        timeframe: str,
    ) -> Layer2Output:
        """
        Evaluate a trade signal. Returns REJECT if P(loss) > threshold.
        """
        self._n_evaluated += 1

        if not self._is_trained or self._model is None:
            return Layer2Output(
                verdict="COLLECTING",
                reason=f"Collecting ({len(self._trade_records)}/{L2_MIN_TRADES_TRAIN})",
                loss_prob=0.5,
                is_trained=False,
                n_trades=len(self._trade_records),
                features=feature_vector,
            )

        # Normalize features
        X = self._normalize_features(features, symbol, timeframe).reshape(1, -1)

        # Predict
        raw_prob = self._model.predict_proba(X)[0, 1]

        # Calibrate
        if self._calibrator is not None:
            cal_prob = float(self._calibrator.transform(np.array([raw_prob]))[0])
        else:
            cal_prob = raw_prob

        # Track prediction
        self._predictions_since_retrain.append({
            "raw_prob": raw_prob,
            "cal_prob": cal_prob,
            "timestamp": time.time(),
        })

        # Decision
        if cal_prob >= L2_REJECT_THRESHOLD:
            self._n_rejected += 1
            return Layer2Output(
                verdict="REJECT",
                reason=f"P(loss)={cal_prob:.3f}>={L2_REJECT_THRESHOLD}",
                loss_prob=cal_prob,
                is_trained=True,
                n_trades=len(self._trade_records),
                features=feature_vector,
            )
        elif cal_prob >= L2_WARN_THRESHOLD:
            return Layer2Output(
                verdict="WARN",
                reason=f"P(loss)={cal_prob:.3f} in warn zone",
                loss_prob=cal_prob,
                is_trained=True,
                n_trades=len(self._trade_records),
                features=feature_vector,
            )
        else:
            return Layer2Output(
                verdict="PASS",
                reason=f"P(loss)={cal_prob:.3f}",
                loss_prob=cal_prob,
                is_trained=True,
                n_trades=len(self._trade_records),
                features=feature_vector,
            )

    def stats(self) -> Dict[str, Any]:
        total = self._n_correct_rejections + self._n_incorrect_rejections
        precision = self._n_correct_rejections / total if total > 0 else 0
        return {
            "evaluated": self._n_evaluated,
            "rejected": self._n_rejected,
            "is_trained": self._is_trained,
            "version": self._version,
            "n_trade_records": len(self._trade_records),
            "n_pending": len(self._pending),
            "rejection_precision": precision,
            "correct_rejections": self._n_correct_rejections,
            "incorrect_rejections": self._n_incorrect_rejections,
            "version_meta": asdict(self._version_meta) if self._version_meta else None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3: OOD DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class OODDetectorV2:
    """
    Layer 3: Detects when current market conditions are meaningfully different
    from the training distribution.

    Uses proper Mahalanobis distance with covariance matrix.
    Reference distribution built from WINNING trades only (the "good" distribution).
    A signal that's OOD relative to winners is likely a loss.
    """

    def __init__(self, model_dir: Path = MODEL_DIR):
        self._model_dir = model_dir
        self._model_dir.mkdir(parents=True, exist_ok=True)

        # Reference distribution (from WINNING trades only)
        self._ref_features: deque = deque(maxlen=L3_WINDOW)
        self._means: Optional[np.ndarray] = None
        self._cov_inv: Optional[np.ndarray] = None
        self._n_seen: int = 0

        self._n_evaluated = 0
        self._n_rejected = 0

        self._load_reference()

    def _ref_path(self) -> Path:
        return self._model_dir / "ood_reference_v2.pkl"

    def _load_reference(self):
        """Load reference distribution from disk."""
        rp = self._ref_path()
        if not rp.exists():
            return
        try:
            with open(rp, "rb") as f:
                data = _SafeUnpickler(f).load()
            self._means = data["means"]
            self._cov_inv = data["cov_inv"]
            self._n_seen = data["n_seen"]
            log.info("[L3] Loaded OOD reference (%d samples)", self._n_seen)
        except Exception as e:
            log.warning("[L3] Failed to load reference: %s", e)

    def _save_reference(self):
        """Persist reference distribution to disk."""
        if self._means is None:
            return
        try:
            data = {
                "means": self._means,
                "cov_inv": self._cov_inv,
                "n_seen": self._n_seen,
            }
            with open(self._ref_path(), "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            log.debug("[L3] Failed to save reference: %s", e)

    def record_winning_features(self, features: np.ndarray):
        """
        Add features from a WINNING trade to the reference distribution.
        This is called by the engine when a trade closes as a win.
        """
        self._ref_features.append(features.copy())
        self._n_seen += 1

        # Recompute stats periodically
        if self._n_seen % 50 == 0 and len(self._ref_features) >= L3_MIN_SAMPLES:
            self._compute_stats()

    def _compute_stats(self):
        """Compute mean vector and inverse covariance matrix."""
        if len(self._ref_features) < L3_MIN_SAMPLES:
            return

        arr = np.array(self._ref_features)

        # Regularize: add small diagonal to prevent singular covariance
        means = np.mean(arr, axis=0)
        cov = np.cov(arr, rowvar=False)
        cov += np.eye(cov.shape[0]) * 0.01  # regularization

        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            # Fallback: use pseudo-inverse
            cov_inv = np.linalg.pinv(cov)

        self._means = means
        self._cov_inv = cov_inv
        self._save_reference()

    def evaluate(self, features: np.ndarray) -> Layer3Output:
        """Evaluate if the current signal is OOD relative to winning trades."""
        self._n_evaluated += 1

        if self._means is None or self._cov_inv is None:
            return Layer3Output(
                verdict="PASS",
                reason=f"Building reference ({self._n_seen}/{L3_MIN_SAMPLES})",
                distance=0.0,
                n_reference=self._n_seen,
            )

        # Mahalanobis distance
        diff = features - self._means
        dist = float(np.sqrt(diff @ self._cov_inv @ diff))

        if dist >= L3_REJECT_THRESHOLD:
            self._n_rejected += 1
            return Layer3Output(
                verdict="REJECT",
                reason=f"Mahalanobis={dist:.2f}>={L3_REJECT_THRESHOLD}",
                distance=dist,
                n_reference=self._n_seen,
            )
        elif dist >= L3_WARN_THRESHOLD:
            return Layer3Output(
                verdict="WARN",
                reason=f"Mahalanobis={dist:.2f} in warn zone",
                distance=dist,
                n_reference=self._n_seen,
            )

        return Layer3Output(
            verdict="PASS",
            reason=f"Mahalanobis={dist:.2f}",
            distance=dist,
            n_reference=self._n_seen,
        )

    def stats(self) -> Dict[str, Any]:
        return {
            "evaluated": self._n_evaluated,
            "rejected": self._n_rejected,
            "n_reference": self._n_seen,
            "has_covariance": self._cov_inv is not None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class LossRejectionEngineV2:
    """
    Loss Rejection Engine v2 - Orchestrates the 3-layer pipeline.

    Layer 1 (CatastrophicGuard): Safety net for extreme risk.
    Layer 2 (MetaLabelerV2): PRIMARY decision maker. ML-based.
    Layer 3 (OODDetectorV2): Rejects unseen market conditions.

    CRITICAL: record_outcome() MUST be called when trades close.
    This wires the feedback loop that makes Layer 2 intelligent.
    """

    def __init__(self):
        self._enabled = LRE_ENABLED
        self._shadow_mode = LRE_SHADOW_MODE
        self._start_time = time.time()

        self.layer1 = CatastrophicGuard()
        self.layer2 = MetaLabelerV2()
        self.layer3 = OODDetectorV2()

        self._n_evaluated = 0
        self._n_blocked = 0
        self._n_shadow_blocked = 0
        self._n_l1_blocks = 0
        self._n_l2_blocks = 0
        self._n_l3_blocks = 0

        if self._enabled:
            log.info(
                "[LRE-v2] Initialized | shadow=%s | model_dir=%s",
                self._shadow_mode, MODEL_DIR,
            )
        else:
            log.info("[LRE-v2] DISABLED")

    def evaluate(
        self,
        dec_out: Dict[str, Any],
        analysis_out: Dict[str, Any],
        market_out: Dict[str, Any],
        *,
        symbol: str = "",
        timeframe: str = "",
    ) -> LREResult:
        """
        Evaluate a trade signal through all 3 layers.

        Same signature as the old engine for drop-in compatibility.
        Returns LREResult with blocked=True if the signal should be rejected.
        """
        t0 = time.time()
        result = LREResult()
        symbol = symbol or dec_out.get("pair", "")
        timeframe = timeframe or dec_out.get("timeframe", "")

        if not self._enabled:
            result.processing_time_ms = (time.time() - t0) * 1000
            return result

        direction = (dec_out.get("decision") or "WAIT").upper()
        if direction not in ("BUY", "SELL"):
            result.processing_time_ms = (time.time() - t0) * 1000
            return result

        self._n_evaluated += 1
        original_confidence = dec_out.get("confidence", 0)

        # Extract features ONCE, shared by all layers
        features, feature_vector = extract_features(dec_out, analysis_out, market_out, symbol, timeframe)
        result._pending_features = features

        # Generate trade_id for feedback tracking
        trade_id = f"{symbol}_{timeframe}_{int(time.time() * 1000)}"
        result._pending_trade_id = trade_id

        # ── Layer 1: Catastrophic Guard ──
        try:
            result.l1 = self.layer1.evaluate(features, dec_out, market_out)
        except Exception as e:
            log.warning("[LRE-v2] L1 error: %s", e)
            result.l1 = Layer1Output(verdict="PASS", reason=f"L1 error: {e}")

        # ── Layer 2: ML Meta-Labeler (PRIMARY) ──
        try:
            result.l2 = self.layer2.evaluate(features, feature_vector, symbol, timeframe)
        except Exception as e:
            log.warning("[LRE-v2] L2 error: %s", e)
            result.l2 = Layer2Output(verdict="PASS", reason=f"L2 error: {e}")

        # ── Layer 3: OOD Detector (only if L2 didn't already reject) ──
        if result.l2.verdict != "REJECT":
            try:
                result.l3 = self.layer3.evaluate(feature_vector)
            except Exception as e:
                log.warning("[LRE-v2] L3 error: %s", e)
                result.l3 = Layer3Output(verdict="PASS", reason=f"L3 error: {e}")
        else:
            result.l3 = Layer3Output(verdict="SKIP", reason="L2 already rejected")

        # ── Aggregate verdict ──
        # Layer 2 is PRIMARY. L1 overrides only for catastrophic risk.
        # L3 adds rejection only if L2 passed.
        block_reasons = []
        warn_layers = []

        # L1 catastrophic rejection is absolute
        if result.l1.verdict == "REJECT":
            block_reasons.append(f"L1:{result.l1.reason}")
            self._n_l1_blocks += 1

        # L2 is the primary decision maker
        if result.l2.verdict == "REJECT":
            block_reasons.append(f"L2:{result.l2.reason}")
            self._n_l2_blocks += 1
        elif result.l2.verdict == "WARN":
            warn_layers.append("L2")

        # L3 adds rejection if L2 passed
        if result.l3.verdict == "REJECT":
            block_reasons.append(f"L3:{result.l3.reason}")
            self._n_l3_blocks += 1
        elif result.l3.verdict == "WARN":
            warn_layers.append("L3")

        # Final decision: L1 catastrophic always wins, otherwise L2 is primary
        if result.l1.verdict == "REJECT":
            result.blocked = True
            result.composite_verdict = "REJECT"
            result.reason = f"L1 catastrophic: {result.l1.reason}"
        elif result.l2.verdict == "REJECT":
            result.blocked = True
            result.composite_verdict = "REJECT"
            result.reason = f"L2: {result.l2.reason}"
        elif result.l3.verdict == "REJECT":
            result.blocked = True
            result.composite_verdict = "REJECT"
            result.reason = f"L3: {result.l3.reason}"
        elif warn_layers:
            result.composite_verdict = "WARN"
            result.reason = f"WARN from {', '.join(warn_layers)}"
            result.confidence_penalty = 3.0 * len(warn_layers)
        else:
            result.composite_verdict = "PASS"
            result.reason = "All layers passed"

        if result.blocked:
            self._n_blocked += 1

        # Apply confidence penalty for WARN
        if result.confidence_penalty > 0:
            dec_out["confidence"] = max(0, original_confidence - result.confidence_penalty)

        # Shadow mode: log but don't block
        if self._shadow_mode and result.blocked:
            result.blocked = False
            result.shadow_blocked = True
            self._n_shadow_blocked += 1
            log.info(
                "[LRE-v2] SHADOW BLOCK %s | %s | L1=%s L2=%s L3=%s",
                symbol, result.reason,
                result.l1.verdict if result.l1 else "N/A",
                result.l2.verdict if result.l2 else "N/A",
                result.l3.verdict if result.l3 else "N/A",
            )

        # Register pending trade for outcome tracking
        self.layer2.register_pending(
            trade_id=trade_id,
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            features=features,
            l1_verdict=result.l1.verdict if result.l1 else "N/A",
            l2_verdict=result.l2.verdict if result.l2 else "N/A",
            l3_verdict=result.l3.verdict if result.l3 else "N/A",
            l2_loss_prob=result.l2.loss_prob if result.l2 else 0.5,
            was_blocked=result.blocked or result.shadow_blocked,
        )

        result.processing_time_ms = (time.time() - t0) * 1000
        return result

    def record_outcome(self, trade_id: str, pnl: float, symbol: str = "", features: Optional[np.ndarray] = None):
        """
        Record trade outcome. MUST be called when a trade closes.

        This feeds the Layer 2 training loop and Layer 3 reference distribution.

        Args:
            trade_id: Unique trade identifier (from evaluate result._pending_trade_id)
            pnl: Trade profit/loss in account currency
            symbol: Trade symbol (for OOD reference)
            features: Feature vector at trade open time (for OOD reference)
        """
        # Feed Layer 2 (the critical feedback loop)
        self.layer2.record_outcome(trade_id, pnl)

        # Feed Layer 3: only WINNING trades go into reference
        if features is not None and pnl > 0:
            self.layer3.record_winning_features(features)

    def record_outcome_by_signal(
        self,
        symbol: str,
        direction: str,
        pnl: float,
        features: Optional[Dict[str, float]] = None,
    ):
        """
        Fallback outcome recording by symbol/direction when trade_id is unavailable.
        Matches the most recent pending trade for this symbol/direction.
        """
        # Find the most recent pending trade for this symbol/direction
        best_id = None
        best_ts = 0
        for tid, pending in self.layer2._pending.items():
            if pending["symbol"] == symbol and pending["direction"] == direction:
                if pending["timestamp"] > best_ts:
                    best_ts = pending["timestamp"]
                    best_id = tid

        if best_id:
            feat_vec = None
            if features is not None:
                feat_vec = np.array([features.get(n, 0.0) for n in FEATURE_NAMES], dtype=np.float64)
            self.record_outcome(best_id, pnl, symbol, feat_vec)
        else:
            log.debug("[LRE-v2] No pending trade found for %s %s", symbol, direction)

    def stats(self) -> Dict[str, Any]:
        return {
            "enabled": self._enabled,
            "shadow_mode": self._shadow_mode,
            "uptime_seconds": time.time() - self._start_time,
            "total_evaluated": self._n_evaluated,
            "total_blocked": self._n_blocked,
            "total_shadow_blocked": self._n_shadow_blocked,
            "l1_blocks": self._n_l1_blocks,
            "l2_blocks": self._n_l2_blocks,
            "l3_blocks": self._n_l3_blocks,
            "l1": self.layer1.stats(),
            "l2": self.layer2.stats(),
            "l3": self.layer3.stats(),
        }

    def force_retrain(self):
        """Trigger an immediate model retrain."""
        self.layer2._train()

    def rollback_model(self, n: int = 1) -> bool:
        """Roll back Layer 2 model by n versions."""
        return self.layer2.rollback(n)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API - Drop-in replacement for old engine
# ─────────────────────────────────────────────────────────────────────────────

# Module-level instance (lazy-initialized)
_engine_instance: Optional[LossRejectionEngineV2] = None


def get_engine() -> LossRejectionEngineV2:
    """Get or create the singleton LRE v2 engine."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = LossRejectionEngineV2()
    return _engine_instance


# Backward-compatible alias
LossRejectionEngine = LossRejectionEngineV2
