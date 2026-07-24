"""
ml/model_predictor.py — Live ensemble prediction (Day 69)
===========================================================

Loads all trained models for a pair and produces a single ensemble
prediction with:
  - Per-model probability
  - Model agreement score (e.g. "3/3 models agree")
  - Ensemble probability (average of all model probabilities)
  - Final prediction (BUY if ensemble > threshold, SELL if < 1-threshold, WAIT otherwise)
  - Top important features (from the best model)

If no models are trained yet, returns a "not ready" prediction — the
agent falls back to rule-based logic.

Usage:
    predictor = get_model_predictor()
    pred = predictor.predict(features_dict, pair="EURUSD", timeframe="15m")
    # pred = {"prediction": "BUY", "probability": 0.78, "agreement": "2/3", ...}
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

from ml.model_store import get_model_store
from ml.data_preprocessor import get_preprocessor
from config import MODEL_BUY_THRESHOLD, MODEL_SELL_THRESHOLD

log = get_logger("model_predictor")

from config import PROJECT_ROOT as _PROJECT_ROOT
PREDICTIONS_DB = _PROJECT_ROOT / "memory" / "ml_predictions.db"

# Threshold for BUY/SELL decision
BUY_THRESHOLD = MODEL_BUY_THRESHOLD
SELL_THRESHOLD = MODEL_SELL_THRESHOLD


class ModelPredictor:
    """Live ensemble predictor combining XGBoost + RF + LSTM.

    AUDIT NOTE (§3): the `prediction`/`probability` fields returned by
    predict() below are an INTERNAL, informational mean-of-probabilities
    ensemble (simple average, fixed 0.58/0.42 thresholds) — they are NOT
    the system's final trading decision. ml.ensemble.EnsembleEngine is the
    authoritative decision layer: it re-fuses these same per-model outputs
    through ConfidenceFusion with regime-aware weighting, performance-based
    weight adjustment, and conflict/dissent handling. Do not act on
    predict()'s output directly — always go through EnsembleEngine.decide().
    """

    def __init__(self):
        self.store = get_model_store()
        self.preprocessor = get_preprocessor()
        self._lock = threading.RLock()
        self._model_cache: Dict[str, Any] = {}  # pair_tf_modeltype → model
        self._scaler_loaded = False
        # (pair, timeframe) pairs known at boot to have ZERO usable models
        # on disk (see ModelStore.get_cold_pairs()). _load_models() checks
        # this before touching disk at all, so a pair with no trained
        # model doesn't pay a failed load_model() attempt x3 model_types
        # on every single trading cycle — it just returns {} immediately.
        # Populated once via mark_cold_pairs() (called from core/runtime.py
        # right after the boot-time registry/disk audit) and cleared per
        # pair via unmark_cold_pair() as soon as a background retrain
        # actually lands a usable model, so it self-heals without a
        # restart instead of staying permanently blacklisted.
        self._cold_pairs: set = set()
        self._init_predictions_db()

    def mark_cold_pairs(self, pairs) -> None:
        """Pre-flag (pair, timeframe) tuples that have zero usable models
        on disk, so _load_models() can short-circuit instead of retrying
        a load every cycle for something guaranteed to fail."""
        with self._lock:
            added = {(p.upper(), tf) for p, tf in pairs} - self._cold_pairs
            self._cold_pairs |= added
        if added:
            log.info(
                f"[Predictor] {len(added)} pair(s) pre-flagged as cold "
                f"(zero usable models) — skipping load attempts until "
                f"retrained: {sorted(added)}"
            )

    def unmark_cold_pair(self, pair: str, timeframe: str) -> None:
        """Clear a pair's cold flag once a background retrain has landed
        a usable model, so the very next predict() call tries loading it
        again instead of waiting for a restart."""
        with self._lock:
            self._cold_pairs.discard((pair.upper(), timeframe))

    def _init_predictions_db(self) -> None:
        PREDICTIONS_DB.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(PREDICTIONS_DB)) as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS ml_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prediction TEXT,
                    probability REAL,
                    actual_result TEXT,
                    timestamp TEXT NOT NULL
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS ml_ensemble (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    ensemble_prediction TEXT,
                    ensemble_probability REAL,
                    model_agreement TEXT,
                    per_model TEXT,
                    timestamp TEXT NOT NULL
                )
            """)
            c.commit()

    def _load_models(self, pair: str, timeframe: str) -> Dict[str, Any]:
        """Load all available models for a pair (cached).

        Enhancement: when a legacy model (no feature_names in registry)
        is detected as the "latest" version, automatically try to find a
        newer compatible version instead of loading a model that will
        fail at prediction time.
        """
        if (pair.upper(), timeframe) in self._cold_pairs:
            return {}

        cache_key_prefix = f"{pair.upper()}_{timeframe}_"
        models: Dict[str, Any] = {}
        for model_type in ("xgboost", "random_forest", "lstm"):
            cache_key = cache_key_prefix + model_type
            if cache_key in self._model_cache:
                models[model_type] = self._model_cache[cache_key]
                continue

            model = self.store.load_model(pair, timeframe, model_type)

            # Auto-promote: if the loaded model is legacy (no feature_names)
            # and has a different feature count than what FeatureEngineer
            # produces (~161), try to load a newer version instead.
            if model is not None:
                feature_names = self.store.get_feature_names(pair, timeframe, model_type)
                expected_count = getattr(model, "n_features_in_", None)
                if not feature_names and expected_count is not None and expected_count < 50:
                    # This is almost certainly a legacy 13-feature model.
                    # Try to find and load a newer version.
                    newer = self._try_load_newer_version(pair, timeframe, model_type, expected_count)
                    if newer is not None:
                        model = newer
                        log.info(
                            f"[Predictor] Auto-promoted {pair}_{timeframe}_{model_type} "
                            f"from legacy v{expected_count}-feature model to newer version "
                            f"({getattr(newer, 'n_features_in_', '?')} features)"
                        )

            if model is not None:
                self._model_cache[cache_key] = model
                models[model_type] = model
        return models

    def _try_load_newer_version(self, pair: str, timeframe: str,
                                 model_type: str, legacy_count: int) -> Any:
        """Try to load a newer, compatible model version from the registry.

        Scans all versions for this pair/tf/model_type and picks the latest
        one that has feature_names AND a compatible feature count.
        Returns None if no suitable version is found.
        """
        try:
            import json as _json
            from ml.model_store import REGISTRY_PATH
            if not REGISTRY_PATH.exists():
                return None

            registry = _json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
            models_dict = registry.get("models", {})
            key_prefix = f"{pair.upper()}_{timeframe}_{model_type}"

            # Collect all versions with their feature info
            candidates = []
            for key, versions in models_dict.items():
                if not key.startswith(key_prefix):
                    continue
                if not isinstance(versions, dict):
                    continue
                for ver_str, ver_info in versions.items():
                    if not isinstance(ver_info, dict):
                        continue
                    fnames = ver_info.get("feature_names", [])
                    if fnames:  # has saved schema
                        candidates.append((ver_str, ver_info))

            if not candidates:
                return None

            # Pick the latest version with a schema (highest version number)
            candidates.sort(key=lambda x: int(x[0].replace("v", "")) if x[0].replace("v", "").isdigit() else 0, reverse=True)
            best_ver, best_info = candidates[0]

            model_path = best_info.get("model_path")
            if not model_path:
                return None

            from pathlib import Path
            path = Path(model_path)
            if not path.is_absolute():
                from ml.model_store import MODELS_DIR
                path = MODELS_DIR / model_path

            if not path.exists():
                return None

            # Load the newer model
            from ml.model_store import safe_pickle_load
            model = safe_pickle_load(path)
            return model

        except Exception as e:
            log.debug(f"[Predictor] _try_load_newer_version failed: {e}")
            return None

    def _load_scaler(self, pair: str, timeframe: str) -> bool:
        """Try to load the scaler saved during training.

        Bug #7 fix: scaler is now per-pair/timeframe, keyed like the model cache.
        Previously a single shared scaler was loaded once and applied to all pairs.
        """
        cache_key = f"{pair.upper()}_{timeframe}_scaler"
        if cache_key in self._model_cache:
            return True
        scaler_path = _PROJECT_ROOT / "memory" / "ml_processed" / f"{pair.upper()}_{timeframe}_scaler.pkl"
        if not scaler_path.exists():
            # Fallback to legacy single scaler for backward compat
            scaler_path = _PROJECT_ROOT / "memory" / "ml_processed" / "scaler.pkl"
        if scaler_path.exists():
            try:
                self.preprocessor.load_scaler(scaler_path)
                self._model_cache[cache_key] = True
                return True
            except Exception:
                pass
        return False

    def is_ready(self, pair: Optional[str] = None, timeframe: str = "15m") -> bool:
        """Check if at least one model is available for prediction.

        Co-founder fix: this method was missing entirely — AnalysisAgent
        checked hasattr(_predictor, 'is_ready') which returned False, so
        ML models were NEVER loaded even when they existed on disk.
        """
        try:
            if pair is not None:
                for model_type in ("xgboost", "random_forest", "lstm"):
                    try:
                        model = self.store.load_model(pair, timeframe, model_type)
                        if model is not None:
                            return True
                    except Exception:
                        continue
                return False
            else:
                registry = getattr(self.store, '_registry', {})
                models = registry.get('models', {})
                return len(models) > 0
        except Exception as e:
            log.debug(f"[ModelPredictor] is_ready check failed: {e}")
            return False

    def predict(
        self,
        features: Dict[str, float],
        pair: str,
        timeframe: str = "15m",
    ) -> Dict[str, Any]:
        """Run ensemble prediction on a single feature vector.

        Returns:
            {
                "prediction": "BUY" | "SELL" | "WAIT" | "NOT_READY",
                "probability": float,           # ensemble BUY probability
                "model_agreement": str,          # e.g. "3/3"
                "per_model": {                   # per-model breakdown
                    "xgboost": {"prediction": "BUY", "probability": 0.78},
                    "random_forest": {...},
                    "lstm": {...},
                },
                "important_features": [...],     # top features (if available)
                "models_used": int,
                "timestamp": str,
            }
        """
        pair = pair.upper()
        result: Dict[str, Any] = {
            "prediction": "NOT_READY",
            "probability": 0.5,
            "model_agreement": "0/0",
            "per_model": {},
            "important_features": [],
            "models_used": 0,
            # ARCHITECTURAL FIX (institutional refactor): explicit `ml_available`
            # flag so downstream consumers can branch cleanly without parsing
            # the "NOT_READY" string. Ensemble + DecisionAgent + dashboard all
            # check this flag. When False, the ensemble dynamically rebalances
            # weights to the remaining voters (rules + LLM + institutional).
            "ml_available": False,
            "ml_unavailable_reason": "models_not_loaded",
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

        # Load models
        models = self._load_models(pair, timeframe)
        if not models:
            # Co-founder fix: log at WARNING level (was debug) so the
            # operator can see WHY models aren't loading.
            #
            # Round-5 audit fix: actionable diagnostic. The old message
            # was generic ("Check directory exists"). Now we tell the
            # operator EXACTLY what's missing:
            #   - whether the pair_dir exists
            #   - whether _registry.json has an entry for this pair/tf
            #   - whether the model files referenced in the registry
            #     actually exist on disk
            # Plus a one-time "suppressed further warnings for this pair"
            # log so the operator doesn't see this 100 times per cycle.
            try:
                from ml.model_store import MODELS_DIR, REGISTRY_PATH, ModelStore
                import os as _os
                pair_dir = MODELS_DIR / f"{pair.upper()}_{timeframe}"
                pair_dir_exists = pair_dir.exists()
                registry_exists = REGISTRY_PATH.exists()
                registry_has_pair = False
                if registry_exists:
                    try:
                        import json as _json
                        reg = _json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
                        models_dict = reg.get("models", {})
                        # Look for any key starting with "{PAIR}_{TF}_"
                        prefix = f"{pair.upper()}_{timeframe}_"
                        registry_has_pair = any(k.startswith(prefix) for k in models_dict)
                    except Exception:
                        pass
                # One-shot suppression per pair/tf so we don't spam logs
                _warn_key = f"_ml_warn_suppressed::{pair}::{timeframe}"
                if not getattr(self, _warn_key, False):
                    log.warning(
                        f"[Predictor] no models loaded for {pair} {timeframe} — NOT_READY. "
                        f"Diagnostic: pair_dir={pair_dir} (exists={pair_dir_exists}) | "
                        f"registry={REGISTRY_PATH.name} (exists={registry_exists}, "
                        f"has_pair_entry={registry_has_pair}). "
                        f"To fix: run `python scripts/train_models.py --pair {pair} --tf {timeframe}` "
                        f"to train and register models. "
                        f"(This warning is logged once per process — further "
                        f"NOT_READY results for {pair} {timeframe} will be silent.)"
                    )
                    setattr(self, _warn_key, True)
                else:
                    # Silent — just debug-level log
                    log.debug(
                        f"[Predictor] {pair} {timeframe} NOT_READY (suppressed)"
                    )
            except Exception as _diag_e:
                # Diagnostic failed — fall back to old generic message
                log.warning(
                    f"[Predictor] no models loaded for {pair} {timeframe} — NOT_READY. "
                    f"(Diagnostic failed: {_diag_e}). "
                    f"Check: memory/ml_models/{pair}_{timeframe}/ directory exists "
                    f"and _registry.json paths are valid."
                )
            return result

        result["models_used"] = len(models)
        # A loaded artifact is not necessarily usable: a legacy model may
        # have an incompatible feature schema. Mark ML available only after
        # at least one prediction succeeds below.
        result["ml_unavailable_reason"] = "no_compatible_model"

        # Load scaler
        self._load_scaler(pair, timeframe)

        # Build feature vector in the right order
        # We need to match the feature names from training. Since we don't have
        # the exact list here, we pass the dict as a single-row DataFrame and
        # let the model handle it (tree-based models are order-independent by name).
        try:
            X = pd.DataFrame([features])
        except Exception as e:
            log.warning(f"[Predictor] feature vector build failed: {e}")
            return result

        # Transform with scaler (if loaded for this pair/timeframe)
        try:
            scaler_key = f"{pair.upper()}_{timeframe}_scaler"
            if scaler_key in self._model_cache:
                X = self.preprocessor.transform(X)
        except Exception:
            pass  # scaler may not have all columns — use raw

        buy_count = 0
        sell_count = 0
        probabilities: List[float] = []

        # AUDIT FIX (2026-07): train_models_quick.py calibrates and saves a
        # per-model, per-pair optimal threshold (metrics["threshold"] — see
        # _find_optimal_threshold(), F1-maximized on a held-out calibration
        # slice, now with a degenerate-prediction guard). Until this fix,
        # that calibrated value was computed, logged, and saved... and then
        # never read again: every live prediction used the same fixed
        # global BUY_THRESHOLD/SELL_THRESHOLD (0.58/0.42) for every model on
        # every pair regardless of what was actually calibrated for it —
        # e.g. a real training run logged threshold=0.36 for EURUSD XGBoost
        # and threshold=0.30 for EURUSD RandomForest, both silently
        # discarded at inference. Now: look up each model's own saved
        # threshold and use it as the BUY cutoff. The training calibration
        # is one-sided (F1 on the "Up" label only — there is no separately-
        # calibrated SELL cutoff), so SELL is derived as the mirror image
        # around 0.5 (sell_threshold = 1 - buy_threshold) to preserve a
        # symmetric WAIT band rather than inventing an unvalidated
        # asymmetric one. Falls back to the global constants if no
        # calibrated threshold was saved (e.g. legacy models trained before
        # this feature existed).
        model_thresholds: Dict[str, tuple] = {}
        for model_type in models.keys():
            buy_t, sell_t = BUY_THRESHOLD, SELL_THRESHOLD
            try:
                m = self.store.get_latest_metrics(pair, timeframe, model_type)
                calibrated = m.get("threshold") if m else None
                if calibrated is not None and 0.0 < float(calibrated) < 1.0:
                    buy_t = float(calibrated)
                    sell_t = 1.0 - buy_t
            except Exception as e:
                log.debug(f"[Predictor] {model_type} threshold lookup failed, using global default: {e}")
            model_thresholds[model_type] = (buy_t, sell_t)

        for model_type, model in models.items():
            model_result: Dict[str, Any] = {"prediction": "WAIT", "probability": 0.5}
            try:
                expected_features = self.store.get_feature_names(pair, timeframe, model_type)
                expected_count = getattr(model, "n_features_in_", None)
                if model_type != "lstm":
                    if expected_features:
                        missing = [name for name in expected_features if name not in X.columns]
                        if missing:
                            raise ValueError(
                                f"model schema requires {len(missing)} unavailable feature(s): {missing[:3]}"
                            )
                        model_X = X.reindex(columns=expected_features)
                    elif expected_count is not None and X.shape[1] != expected_count:
                        # A legacy artifact with no saved schema cannot be
                        # aligned safely.  Never truncate arbitrary columns:
                        # that produces plausible but invalid live signals.
                        #
                        # FIX: Instead of raising (which killed the entire
                        # ensemble prediction for this pair), skip this
                        # specific model gracefully. Log a one-time warning
                        # and let other compatible models (e.g. a newer
                        # version) contribute to the ensemble.
                        log.warning(
                            f"[Predictor] {model_type} SKIPPED — legacy model "
                            f"schema mismatch (expects {expected_count} features, "
                            f"got {X.shape[1]}). Retrain with: "
                            f"python scripts/train_models.py --pair {pair} --tf {timeframe}"
                        )
                        # Record as WAIT with explanation, but don't break the loop
                        model_result = {
                            "prediction": "WAIT",
                            "probability": 0.5,
                            "error": f"legacy schema mismatch (expects {expected_count}, got {X.shape[1]})",
                            "skipped": True,
                        }
                        result["per_model"][model_type] = model_result
                        continue
                    else:
                        model_X = X
                else:
                    model_X = X
                if model_type == "lstm":
                    # LSTM needs 3D input
                    n_features = model_X.shape[1]
                    X_3d = model_X.values.reshape(1, 1, n_features)
                    proba = float(model.predict(X_3d, verbose=0).ravel()[0])
                else:
                    # P4a FIX: use .values to avoid XGBoost ≥ 2.0
                    # feature_names mismatch when the model was trained on
                    # numpy arrays (scripts/train_models_quick.py used
                    # df[cols].values which strips column names, causing
                    # XGBoost to store internal names as f0, f1, ...).
                    # Tree models are position-dependent, not name-dependent,
                    # and the reindex above (line 395) already guarantees the
                    # correct column ordering. Converting to .values ensures
                    # compatibility with both DataFrame-trained and
                    # numpy-array-trained models.
                    _raw = model_X.values if hasattr(model_X, 'columns') else model_X
                    proba_arr = model.predict_proba(_raw)
                    proba = float(proba_arr[0][1]) if proba_arr.shape[1] > 1 else float(proba_arr[0][0])

                model_result["probability"] = round(proba, 4)
                _buy_t, _sell_t = model_thresholds.get(model_type, (BUY_THRESHOLD, SELL_THRESHOLD))
                if proba >= _buy_t:
                    model_result["prediction"] = "BUY"
                    buy_count += 1
                elif proba <= _sell_t:
                    model_result["prediction"] = "SELL"
                    sell_count += 1
                probabilities.append(proba)

                # Record individual prediction
                self._record_prediction(pair, timeframe, model_type, model_result["prediction"], proba)

            except Exception as e:
                log.debug(f"[Predictor] {model_type} predict failed: {e}")
                model_result = {"prediction": "WAIT", "probability": 0.5, "error": str(e)[:100]}

            result["per_model"][model_type] = model_result

        # Ensemble: average probability
        if probabilities:
            result["ml_available"] = True
            result["ml_unavailable_reason"] = None
            ensemble_proba = float(np.mean(probabilities))
            result["probability"] = round(ensemble_proba, 4)

            # Agreement
            total_models = len(probabilities)
            if buy_count > sell_count and buy_count > 0:
                result["prediction"] = "BUY"
                result["model_agreement"] = f"{buy_count}/{total_models}"
            elif sell_count > buy_count and sell_count > 0:
                result["prediction"] = "SELL"
                result["model_agreement"] = f"{sell_count}/{total_models}"
            else:
                result["prediction"] = "WAIT"
                result["model_agreement"] = f"{max(buy_count, sell_count)}/{total_models}"

        # Important features (from xgboost if available)
        try:
            if "xgboost" in models:
                importances = models["xgboost"].feature_importances_
                # Bug #26: use expected_features (model's training schema) instead of
                # X.columns (raw input order) for correct feature name mapping.
                expected_feat_names = self.store.get_feature_names(pair, timeframe, "xgboost")
                if expected_feat_names and len(expected_feat_names) == len(importances):
                    feat_names = expected_feat_names
                else:
                    feat_names = list(X.columns)
                top_idx = np.argsort(importances)[::-1][:5]
                result["important_features"] = [
                    {"feature": feat_names[i], "importance": round(float(importances[i]), 4)}
                    for i in top_idx if i < len(feat_names)
                ]
        except Exception:
            pass

        # Record ensemble prediction
        self._record_ensemble(pair, timeframe, result)

        return result

    def _record_prediction(self, pair, tf, model, prediction, probability):
        try:
            with sqlite3.connect(str(PREDICTIONS_DB)) as c:
                c.execute(
                    "INSERT INTO ml_predictions (pair, timeframe, model, prediction, probability, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                    (pair, tf, model, prediction, float(probability),
                     datetime.now(timezone.utc).isoformat(timespec="seconds")),
                )
                c.commit()
        except Exception:
            pass

    def _record_ensemble(self, pair, tf, result):
        try:
            with sqlite3.connect(str(PREDICTIONS_DB)) as c:
                c.execute(
                    "INSERT INTO ml_ensemble (pair, timeframe, ensemble_prediction, ensemble_probability, model_agreement, per_model, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (pair, tf, result["prediction"], result["probability"],
                     result["model_agreement"], json.dumps(result["per_model"], default=str),
                     result["timestamp"]),
                )
                c.commit()
        except Exception:
            pass

    def update_actual_result(self, pair: str, timeframe: str, prediction_id: int, actual: str):
        """Update the actual result after the trade closes (for accuracy tracking)."""
        try:
            with sqlite3.connect(str(PREDICTIONS_DB)) as c:
                c.execute(
                    "UPDATE ml_predictions SET actual_result = ? WHERE id = ?",
                    (actual, prediction_id),
                )
                c.commit()
        except Exception:
            pass

    def prediction_stats(self, pair: Optional[str] = None) -> Dict[str, Any]:
        """Return prediction accuracy stats."""
        try:
            with sqlite3.connect(str(PREDICTIONS_DB)) as c:
                if pair:
                    rows = c.execute(
                        "SELECT model, COUNT(*), COALESCE(SUM(CASE WHEN prediction = actual_result THEN 1 ELSE 0 END), 0) FROM ml_predictions WHERE pair = ? AND actual_result IS NOT NULL GROUP BY model",
                        (pair.upper(),),
                    ).fetchall()
                else:
                    rows = c.execute(
                        "SELECT model, COUNT(*), COALESCE(SUM(CASE WHEN prediction = actual_result THEN 1 ELSE 0 END), 0) FROM ml_predictions WHERE actual_result IS NOT NULL GROUP BY model",
                    ).fetchall()
            stats = {}
            for model, total, correct in rows:
                stats[model] = {
                    "total": total,
                    "correct": correct,
                    "accuracy_pct": round((correct / total * 100) if total else 0, 1),
                }
            return stats
        except Exception as e:
            return {"error": str(e)}


# ── Singleton ───────────────────────────────────────────────────────

_PREDICTOR: Optional[ModelPredictor] = None


def get_model_predictor() -> ModelPredictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = ModelPredictor()
    return _PREDICTOR
