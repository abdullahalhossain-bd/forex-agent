"""
ml/model_trainer.py — Multi-model training pipeline (Day 69)
==============================================================

Trains three ML models with graceful fallbacks:
  1. **XGBoost**     — primary, fast, non-linear, feature importance
  2. **RandomForest** — confirmation model (sklearn-based)
  3. **LSTM**         — sequential neural network (TensorFlow/Keras)

If a library is not installed, that model is skipped (no crash). The
ensemble predictor (Day 70) will use whatever models are available.

Walk-forward validation is supported via WalkForwardValidator.

Overfitting is detected and logged — models with >15% train/test gap
are flagged but still saved (operator can rollback).

Usage:
    trainer = get_model_trainer()
    results = trainer.train_all(pair="EURUSD", timeframe="15m")
    # results = {"xgboost": ModelMetrics, "random_forest": ..., "lstm": ...}
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger

from ml.dataset_builder import Dataset, get_dataset_builder
from ml.model_evaluator import ModelMetrics, get_evaluator
from ml.model_store import get_model_store
from ml.data_preprocessor import get_preprocessor

log = get_logger("model_trainer")


@dataclass
class TrainingResult:
    """Result of training all models for one pair."""
    pair: str
    timeframe: str
    dataset_summary: Dict[str, Any]
    models_trained: List[str]
    metrics: Dict[str, Dict[str, Any]]
    best_model: str = ""
    training_time_sec: float = 0.0
    errors: List[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelTrainer:
    """Trains XGBoost, RandomForest, and LSTM models."""

    def __init__(self):
        self.store = get_model_store()
        self.evaluator = get_evaluator()
        self.builder = get_dataset_builder()
        self.preprocessor = get_preprocessor()

    # ── Public API ─────────────────────────────────────────────────

    def train_all(
        self,
        pair: str,
        timeframe: str = "15m",
        min_samples: int = 100,
        labeling_method: str = "fixed_horizon",
        use_purged_split: bool = False,
        label_horizon: int = 0,
    ) -> TrainingResult:
        """Train all available models for a pair. Returns TrainingResult.

        New optional params (Priority #1 — leakage audit), all defaulting
        to exact current behavior:
          labeling_method: "fixed_horizon" (default) | "triple_barrier"
          use_purged_split: purge boundary-overlap leakage at the
            train/val/test cut points. Default False = unchanged.
          label_horizon: forward-looking window size used by both the
            labeler (if triple_barrier) and the purger. Ignored when both
            labeling_method="fixed_horizon" and use_purged_split=False.
        """
        t0 = time.time()
        result = TrainingResult(
            pair=pair.upper(),
            timeframe=timeframe,
            dataset_summary={},
            models_trained=[],
            metrics={},
            errors=[],
        )

        # 1. Build dataset
        log.info(f"[Trainer] Building dataset for {pair} {timeframe} "
                  f"(labeling_method={labeling_method}, use_purged_split={use_purged_split})...")
        dataset = self.builder.build_from_store(
            pair=pair, timeframe=timeframe, min_samples=min_samples,
            labeling_method=labeling_method, use_purged_split=use_purged_split,
            label_horizon=label_horizon,
        )
        if dataset is None:
            result.errors.append(f"Insufficient data for {pair} {timeframe} (need ≥{min_samples} samples)")
            return result
        result.dataset_summary = dataset.summary()

        # 2. Preprocess: fit scaler on train, transform all
        log.info("[Trainer] Preprocessing data (scaler fit on train only)...")
        self.preprocessor.fit_scaler(dataset.X_train)
        X_train = self.preprocessor.transform(dataset.X_train)
        X_val = self.preprocessor.transform(dataset.X_val)
        X_test = self.preprocessor.transform(dataset.X_test)

        # sample_weight from compute_label_uniqueness() (triple_barrier only).
        # None for fixed_horizon — passing sample_weight=None to fit() below
        # is identical to not passing the kwarg at all, so existing runs
        # are numerically unaffected.
        sample_weight = dataset.sample_weight
        if sample_weight is not None:
            sample_weight = sample_weight.reindex(X_train.index).fillna(1.0)

        # Tags persisted into ModelStore metadata (meta.json + registry)
        # so a saved model can always be traced back to which labeling/CV
        # pipeline produced it — required for the shadow-vs-champion
        # comparison in the migration plan (never conflate a
        # purged-CV model's score with a naive-CV model's score).
        cv_tags = {"labeling_method": dataset.labeling_method, "cv_method": dataset.cv_method}

        # 3. Train each model
        log.info("[Trainer] Training XGBoost...")
        xgb_metrics = self._train_xgboost(
            X_train, dataset.y_train, X_val, dataset.y_val, X_test, dataset.y_test,
            pair, timeframe, list(X_train.columns), sample_weight=sample_weight, cv_tags=cv_tags,
        )
        if xgb_metrics:
            result.models_trained.append("xgboost")
            result.metrics["xgboost"] = xgb_metrics.to_dict()

        log.info("[Trainer] Training Random Forest...")
        rf_metrics = self._train_random_forest(
            X_train, dataset.y_train, X_val, dataset.y_val, X_test, dataset.y_test,
            pair, timeframe, list(X_train.columns), sample_weight=sample_weight, cv_tags=cv_tags,
        )
        if rf_metrics:
            result.models_trained.append("random_forest")
            result.metrics["random_forest"] = rf_metrics.to_dict()

        log.info("[Trainer] Training LSTM...")
        lstm_metrics = self._train_lstm(
            X_train, dataset.y_train, X_val, dataset.y_val, X_test, dataset.y_test,
            pair, timeframe, dataset.feature_names, cv_tags=cv_tags,
        )
        if lstm_metrics:
            result.models_trained.append("lstm")
            result.metrics["lstm"] = lstm_metrics.to_dict()

        # 4. Determine best model
        if result.metrics:
            best_name = max(result.metrics.items(),
                          key=lambda x: x[1].get("auc_roc", 0) if isinstance(x[1], dict) else 0)
            result.best_model = best_name[0] if isinstance(best_name, tuple) else str(best_name)

        result.training_time_sec = round(time.time() - t0, 1)
        log.info(
            f"[Trainer] Done in {result.training_time_sec}s | "
            f"models={result.models_trained} | best={result.best_model}"
        )
        return result

    # ── XGBoost ───────────────────────────────────────────────────

    def _train_xgboost(self, X_train, y_train, X_val, y_val, X_test, y_test, pair, tf, feature_names,
                        sample_weight: Optional[pd.Series] = None,
                        cv_tags: Optional[Dict[str, Any]] = None) -> Optional[ModelMetrics]:
        try:
            from xgboost import XGBClassifier
        except ImportError:
            log.warning("[Trainer] xgboost not installed — skipping XGBoost")
            return None

        try:
            model = XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0,
            )
            # sample_weight is None for labeling_method="fixed_horizon" (the
            # default) — XGBClassifier.fit(..., sample_weight=None) is
            # identical to omitting the kwarg, so existing training runs
            # are numerically unaffected.
            model.fit(
                X_train, y_train,
                sample_weight=sample_weight,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            metrics = self.evaluator.evaluate(
                model, X_test, y_test, model_name="xgboost",
                X_train=X_train, y_train=y_train,
            )
            log.info(f"  XGBoost: {metrics.summary_line}")
            saved_metrics = metrics.to_dict()
            if cv_tags:
                saved_metrics.update(cv_tags)
            self.store.save_model(
                model=model, pair=pair, timeframe=tf, model_type="xgboost",
                metrics=saved_metrics, feature_names=feature_names,
            )
            return metrics
        except Exception as e:
            log.error(f"[Trainer] XGBoost training failed: {e}")
            return None

    # ── Random Forest ─────────────────────────────────────────────

    def _train_random_forest(self, X_train, y_train, X_val, y_val, X_test, y_test, pair, tf, feature_names,
                              sample_weight: Optional[pd.Series] = None,
                              cv_tags: Optional[Dict[str, Any]] = None) -> Optional[ModelMetrics]:
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError:
            log.warning("[Trainer] sklearn not installed — skipping RandomForest")
            return None

        try:
            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            )
            # sample_weight=None (fixed_horizon default) == omitting the kwarg.
            model.fit(X_train, y_train, sample_weight=sample_weight)
            metrics = self.evaluator.evaluate(
                model, X_test, y_test, model_name="random_forest",
                X_train=X_train, y_train=y_train,
            )
            log.info(f"  RandomForest: {metrics.summary_line}")
            saved_metrics = metrics.to_dict()
            if cv_tags:
                saved_metrics.update(cv_tags)
            self.store.save_model(
                model=model, pair=pair, timeframe=tf, model_type="random_forest",
                metrics=saved_metrics, feature_names=feature_names,
            )
            return metrics
        except Exception as e:
            log.error(f"[Trainer] RandomForest training failed: {e}")
            return None

    # ── LSTM ──────────────────────────────────────────────────────

    def _train_lstm(
        self, X_train, y_train, X_val, y_val, X_test, y_test, pair, tf, feature_names,
        cv_tags: Optional[Dict[str, Any]] = None,
    ) -> Optional[ModelMetrics]:
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.callbacks import EarlyStopping
        except ImportError:
            log.warning("[Trainer] tensorflow not installed — skipping LSTM")
            return None

        try:
            # Reshape to 3D for LSTM: (samples, timesteps, features)
            # We treat each sample as a single timestep (no sequence here —
            # proper sequence modeling would need a different data loader).
            # For simplicity, we use a window of 1 with all features.
            n_features = X_train.shape[1]
            X_train_3d = X_train.values.reshape(X_train.shape[0], 1, n_features)
            X_val_3d = X_val.values.reshape(X_val.shape[0], 1, n_features)
            X_test_3d = X_test.values.reshape(X_test.shape[0], 1, n_features)

            model = Sequential([
                LSTM(64, input_shape=(1, n_features), return_sequences=True),
                Dropout(0.2),
                LSTM(32),
                Dropout(0.2),
                Dense(16, activation="relu"),
                Dense(1, activation="sigmoid"),
            ])
            model.compile(
                optimizer="adam",
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )

            early_stop = EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True,
            )
            model.fit(
                X_train_3d, y_train,
                validation_data=(X_val_3d, y_val),
                epochs=50,
                batch_size=32,
                callbacks=[early_stop],
                verbose=0,
            )

            # Evaluate
            y_proba = model.predict(X_test_3d, verbose=0).ravel()
            y_pred = (y_proba > 0.5).astype(int)

            from ml.model_evaluator import ModelMetrics
            metrics = ModelMetrics(model_name="lstm")
            metrics.accuracy = float(np.mean(y_pred == np.array(y_test).astype(int)))
            metrics.tp = int(np.sum((y_pred == 1) & (np.array(y_test) == 1)))
            metrics.fp = int(np.sum((y_pred == 1) & (np.array(y_test) == 0)))
            metrics.tn = int(np.sum((y_pred == 0) & (np.array(y_test) == 0)))
            metrics.fn = int(np.sum((y_pred == 0) & (np.array(y_test) == 1)))
            prec_den = metrics.tp + metrics.fp
            metrics.precision = metrics.tp / prec_den if prec_den > 0 else 0
            rec_den = metrics.tp + metrics.fn
            metrics.recall = metrics.tp / rec_den if rec_den > 0 else 0
            try:
                from sklearn.metrics import roc_auc_score
                metrics.auc_roc = float(roc_auc_score(y_test, y_proba))
            except Exception:
                metrics.auc_roc = 0.5
            total = metrics.tp + metrics.fp
            metrics.win_rate = metrics.tp / total if total > 0 else 0
            log.info(f"  LSTM: {metrics.summary_line}")

            saved_metrics = metrics.to_dict()
            if cv_tags:
                saved_metrics.update(cv_tags)
            self.store.save_model(
                model=model, pair=pair, timeframe=tf, model_type="lstm",
                metrics=saved_metrics, is_keras=True,
            )
            return metrics
        except Exception as e:
            log.error(f"[Trainer] LSTM training failed: {e}")
            return None

    # ── Walk-forward validation ───────────────────────────────────

    def walk_forward_validate(
        self,
        pair: str,
        timeframe: str = "15m",
        model_type: str = "xgboost",
    ) -> List[Dict[str, Any]]:
        """Run walk-forward validation on a specific model type."""
        from ml.model_evaluator import get_walk_forward_validator
        wf = get_walk_forward_validator()

        dataset = self.builder.build_from_store(pair=pair, timeframe=timeframe, min_samples=200)
        if dataset is None:
            return []

        # Combine train + val + test for walk-forward
        X = pd.concat([dataset.X_train, dataset.X_val, dataset.X_test])
        y = pd.concat([dataset.y_train, dataset.y_val, dataset.y_test])

        # Normalize
        self.preprocessor.fit_scaler(dataset.X_train)
        X = self.preprocessor.transform(X)

        def train_fn(X_tr, y_tr):
            if model_type == "xgboost":
                try:
                    from xgboost import XGBClassifier
                    m = XGBClassifier(n_estimators=100, max_depth=5, verbosity=0, use_label_encoder=False, eval_metric="logloss")
                    m.fit(X_tr, y_tr)
                    return m
                except ImportError:
                    return None
            elif model_type == "random_forest":
                try:
                    from sklearn.ensemble import RandomForestClassifier
                    m = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
                    m.fit(X_tr, y_tr)
                    return m
                except ImportError:
                    return None
            return None

        def predict_fn(model, X_te):
            if model is None:
                return np.zeros(len(X_te)), np.zeros(len(X_te))
            y_p = model.predict(X_te)
            try:
                y_pr = model.predict_proba(X_te)[:, 1]
            except Exception:
                y_pr = y_p.astype(float)
            return y_p, y_pr

        return wf.run(X, y, train_fn, predict_fn)


# ── Singleton ───────────────────────────────────────────────────────

_TRAINER: Optional[ModelTrainer] = None


def get_model_trainer() -> ModelTrainer:
    global _TRAINER
    if _TRAINER is None:
        _TRAINER = ModelTrainer()
    return _TRAINER


def train_all(
    pair: Optional[str] = None,
    timeframe: str = "15m",
    min_samples: int = None,
    labeling_method: str = "fixed_horizon",
    use_purged_split: bool = False,
    label_horizon: int = 0,
) -> Dict[str, "TrainingResult"]:
    """
    Module-level convenience wrapper.

    FIX (2026-07-15): `from ml.model_trainer import train_all` used to
    raise ImportError because `train_all` only existed as a *method* on
    ModelTrainer, not as a module-level function. This wrapper trains
    every configured pair/timeframe (mirroring scripts/train_models.py)
    so `from ml.model_trainer import train_all; train_all()` works
    directly from a REPL or one-off script.

    labeling_method/use_purged_split/label_horizon (Priority #1) default
    to exact current behavior and are forwarded to ModelTrainer.train_all()
    for every pair — use these to run a shadow batch (see migration plan
    in ml/LEAKAGE_AUDIT.md) without touching the champion pipeline.

    Returns a dict of {pair: TrainingResult}.

    NOTE: this still requires enough historical feature data to exist for
    each pair/timeframe (via ml.dataset_builder / ml.feature_store). If a
    pair has insufficient data, that pair's TrainingResult will have
    `errors` populated and no models saved — this function cannot conjure
    training data that isn't there. For batch training with progress
    printouts, prefer `python scripts/train_models.py`.
    """
    from config import MIN_TRAINING_SAMPLES
    trainer = get_model_trainer()

    min_samples_use = min_samples if min_samples is not None else MIN_TRAINING_SAMPLES

    if pair:
        pairs = [pair.upper()]
    else:
        try:
            from config import SYMBOLS
            pairs = [s.upper() for s in SYMBOLS]
        except Exception:
            pairs = ["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD", "XAUUSD"]

    results: Dict[str, "TrainingResult"] = {}
    for p in pairs:
        try:
            results[p] = trainer.train_all(
                pair=p, timeframe=timeframe, min_samples=min_samples_use,
                labeling_method=labeling_method, use_purged_split=use_purged_split,
                label_horizon=label_horizon,
            )
        except Exception as e:
            log.error(f"[train_all] {p} {timeframe} failed: {e}")
    return results
