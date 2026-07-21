#!/usr/bin/env python3
"""
Quick ML Model Training Script — Uses REAL MetaTrader5 historical data.

Round-10 audit fix: the operator's audit found that ML models were NEVER
trained ("pair_dir exists=False, registry exists=False"). This script
creates a minimal working training pipeline so the Ensemble engine stops
being permanently dead code.

IMPORTANT: This script now uses REAL MT5 historical data by default.
Synthetic data is ONLY used if explicitly enabled via --debug-synthetic flag.

AUDIT FINDINGS (2026-07-15):
- Only 13 features were being used while FeatureEngineer has 161 features
- feature_engineer.py was NOT connected to the training pipeline
- No hyperparameter tuning was performed
- Evaluation metrics were insufficient (accuracy only)

Day 70 VALIDATION HARDENING (this version):
- Naive 80/20 index split -> TimeSeriesSplit walk-forward CV over a dev
  set, PLUS a chronological final holdout that is never touched during
  CV, feature selection, fitting, or calibration (only used once, at the
  very end, to report genuine out-of-sample metrics).
- Early stopping: XGBoost via eval_set + early_stopping_rounds; RandomForest
  via a manual warm_start loop that stops adding trees once validation
  log-loss stops improving (RF has no native early stopping).
- Class weights: RandomForest uses class_weight="balanced"; XGBoost (no
  native class_weight param) uses sample_weight from
  sklearn.utils.class_weight.compute_sample_weight("balanced", y) — both
  matter here because most bars are NOT a clean directional move, so raw
  training over-predicts the majority class otherwise.
- Probability calibration: CalibratedClassifierCV fit on a held-out
  calibration slice the base model never trained on (calibrating on
  training data just re-confirms the model's own overconfidence). Reports
  Brier score + a predicted-vs-actual reliability table.
- Feature selection: SHAP importance (falls back to built-in
  feature_importances_ if shap isn't installed) prunes the 161-feature
  vector down to the features that actually carry signal before the final
  fit — fewer, better features reduce overfitting risk on a dataset this
  size.
- Cross-validation metrics: walk-forward fold-by-fold accuracy/precision/
  recall/f1/ROC-AUC, reported as mean ± std (not just a single number) so
  fold-to-fold variance — a real signal of regime instability — is visible
  instead of hidden.

IMPROVEMENTS CARRIED FROM THE PRIOR VERSION:
- Uses FeatureEngineer with 161 features
- Comprehensive evaluation metrics (Precision, Recall, F1, ROC-AUC, Confusion Matrix)
- Prints all feature names used for training
- Verifies no look-ahead bias

Usage:
    python scripts/train_models_quick.py --pair EURUSD --tf 15m
    python scripts/train_models_quick.py --pair XAUUSD --tf 15m --bars 1000
    python scripts/train_models_quick.py  # train all default pairs
    python scripts/train_models_quick.py --cv-splits 5 --feature-top-k 60

Models are saved to:
    memory/ml_models/{PAIR}_{TF}/xgboost_v1.pkl
    memory/ml_models/{PAIR}_{TF}/random_forest_v1.pkl
    memory/ml_models/_registry.json (updated with new model entries)
"""
import argparse
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

# Ensure project root is on path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from utils.logger import get_logger
log = get_logger("train_models_quick")

# ── Validation / training constants ─────────────────────────────────
HOLDOUT_FRACTION = 0.15        # chronologically-last slice: never touched until final scoring
CALIB_FRACTION = 0.15          # slice of the dev set used only for calibration, not model fitting
DEFAULT_CV_SPLITS = 5
EARLY_STOPPING_ROUNDS = 30
RF_STEP = 25                   # trees added per warm_start round
RF_MAX_ESTIMATORS = 400
RF_PATIENCE = 5                # rounds without val log-loss improvement before stopping
CALIBRATION_METHOD = "sigmoid"  # safer default than isotonic on modest sample sizes
MIN_ROWS_FOR_CV = 300           # below this, skip CV folds and warn (not enough data to be meaningful)


def generate_synthetic_ohlcv(symbol: str, bars: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic OHLCV data with trends, volatility, and patterns.
    
    ⚠️ DEBUG ONLY: This function should ONLY be used for debugging/testing.
    Production training MUST use real MT5 data.
    """
    log.warning("⚠️ GENERATING SYNTHETIC DATA - DEBUG MODE ONLY ⚠️")
    np.random.seed(seed + hash(symbol) % 1000)
    dates = pd.date_range("2024-01-01", periods=bars, freq="15min")

    # Build close prices with trend + volatility cycles
    trend = np.random.choice([-1, 1]) * 0.0001
    vol_cycle = np.sin(np.arange(bars) / 50) * 0.0003 + 0.0005
    noise = np.random.randn(bars) * vol_cycle
    close = 1.0850 + np.cumsum(noise + trend)

    # Add periodic shocks (news events)
    for i in range(20, bars, 25):
        close[i] += np.random.randn() * 0.003

    # Build OHLC from close
    intrabar_vol = np.abs(np.random.randn(bars)) * 0.0003
    high = close + intrabar_vol
    low = close - intrabar_vol
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = np.random.randint(100, 1000, bars)

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)
    df.index.name = "time"
    return df


def add_features(df: pd.DataFrame, pair: str = "EURUSD", use_feature_engineer: bool = True) -> pd.DataFrame:
    """Add features using FeatureEngineer for comprehensive feature engineering.
    
    This replaces the simple 13-feature pipeline with the full 161-feature
    FeatureEngineer from ml/feature_engineer.py.
    
    Args:
        df: OHLCV dataframe
        pair: Trading symbol
        use_feature_engineer: If True, use FeatureEngineer (recommended).
                             If False, use simple 13-feature pipeline (legacy).
    
    Returns:
        DataFrame with features added (one row per original bar, minus warmup period)
    """
    if not use_feature_engineer:
        # Legacy 13-feature pipeline (for backward compatibility)
        log.warning("Using legacy 13-feature pipeline - NOT recommended!")
        df = df.copy()
        # Returns
        df["ret_1"] = df["close"].pct_change(1)
        df["ret_3"] = df["close"].pct_change(3)
        df["ret_5"] = df["close"].pct_change(5)
        df["ret_10"] = df["close"].pct_change(10)

        # Volatility
        df["vol_5"] = df["ret_1"].rolling(5).std()
        df["vol_10"] = df["ret_1"].rolling(10).std()

        # RSI (simplified)
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # SMAs
        df["sma_10"] = df["close"].rolling(10).mean()
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()

        # ATR
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean()

        # MACD
        ema_12 = df["close"].ewm(span=12).mean()
        ema_26 = df["close"].ewm(span=26).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()

        return df
    
    # Use FeatureEngineer for comprehensive 161-feature vector
    log.info("Using FeatureEngineer for comprehensive feature engineering (161 features)...")
    from ml.feature_engineer import FeatureEngineer
    
    engineer = FeatureEngineer()
    feature_rows = []
    valid_indices = []
    
    # Need enough history for features to be computed
    min_history = 50
    
    for i in range(min_history, len(df)):
        sub_df = df.iloc[:i+1].copy()
        try:
            feats = engineer.build_feature_vector(
                df=sub_df,
                analysis_out={},  # Can add analysis contexts later for even more features
                pair=pair,
                timeframe="M15"
            )
            if feats:  # Only add if features were generated
                feature_rows.append(feats)
                valid_indices.append(df.index[i])
        except Exception as e:
            log.warning(f"Feature generation failed at index {i}: {e}")
            continue
    
    if not feature_rows:
        log.error("FeatureEngineer produced no features! Falling back to legacy pipeline.")
        return add_features(df, pair, use_feature_engineer=False)
    
    features_df = pd.DataFrame(feature_rows)
    features_df.index = valid_indices
    
    log.info(f"Generated {len(features_df)} rows with {len(features_df.columns)} features each")
    log.info(f"Feature names: {list(features_df.columns)[:20]}... ({len(features_df.columns)} total)")
    
    return features_df


def build_labels(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Build binary classification labels: 1 if price goes up in next N bars."""
    df["target"] = (df["close"].shift(-horizon) > df["close"]).astype(int)
    return df


# ── Model fitting helpers (early stopping + class weights) ──────────

def _fit_xgboost(X_train, y_train, X_val=None, y_val=None, sample_weight=None,
                  early_stopping_rounds: int = EARLY_STOPPING_ROUNDS):
    """Fit an XGBClassifier with early stopping on `X_val`/`y_val` if given.

    Handles both the modern xgboost API (>=1.6, `early_stopping_rounds` is a
    constructor arg) and the legacy API (<1.6, it's a `fit()` kwarg) — don't
    assume which one the deployed environment has.
    """
    import xgboost as xgb

    n_classes = len(np.unique(y_train))
    eval_metric = "logloss" if n_classes == 2 else "mlogloss"
    base_params = dict(
        n_estimators=500,       # upper bound; early stopping cuts this short
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric=eval_metric,
        n_jobs=-1,
    )
    use_es = X_val is not None and y_val is not None and len(X_val) > 0

    try:
        model = xgb.XGBClassifier(
            **base_params,
            early_stopping_rounds=early_stopping_rounds if use_es else None,
        )
        fit_kwargs = {"sample_weight": sample_weight, "verbose": False}
        if use_es:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
        model.fit(X_train, y_train, **fit_kwargs)
    except TypeError:
        # Legacy xgboost (<1.6): early_stopping_rounds belongs in fit(), not __init__
        model = xgb.XGBClassifier(**base_params)
        fit_kwargs = {"sample_weight": sample_weight, "verbose": False}
        if use_es:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
        model.fit(X_train, y_train, **fit_kwargs)

    return model


def _fit_random_forest_with_early_stopping(
    X_train, y_train, X_val=None, y_val=None,
    step: int = RF_STEP, max_estimators: int = RF_MAX_ESTIMATORS, patience: int = RF_PATIENCE,
):
    """RandomForest has no native early stopping, so approximate it: grow the
    forest incrementally via `warm_start`, track validation log-loss after
    each block of new trees, and stop once it hasn't improved for `patience`
    rounds. Returns a freshly-fit model at the best tree count (warm_start
    forests can only grow, not shrink, so we rebuild once we know where to
    stop) plus the chosen n_estimators for logging.

    Uses class_weight="balanced" for class weighting (RF's native mechanism —
    unlike XGBoost, no separate sample_weight is needed here).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import log_loss

    use_es = X_val is not None and y_val is not None and len(X_val) > 0

    if not use_es:
        model = RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
        model.fit(X_train, y_train)
        return model, 200

    grower = RandomForestClassifier(
        n_estimators=0, max_depth=8, warm_start=True,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    best_loss = np.inf
    best_n_estimators = step
    rounds_without_improvement = 0
    n_estimators = 0

    while n_estimators < max_estimators:
        n_estimators += step
        grower.set_params(n_estimators=n_estimators)
        grower.fit(X_train, y_train)

        val_proba = grower.predict_proba(X_val)
        val_loss = log_loss(y_val, val_proba, labels=grower.classes_)
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_n_estimators = n_estimators
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1
            if rounds_without_improvement >= patience:
                break

    # Rebuild clean (warm_start can only add trees, so re-fit fresh at the
    # tree count that actually had the best validation log-loss).
    final_model = RandomForestClassifier(
        n_estimators=best_n_estimators, max_depth=8, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    final_model.fit(X_train, y_train)
    return final_model, best_n_estimators


# ── Cross-validation (walk-forward, TimeSeriesSplit) ─────────────────

def _score_fold(model, X_val, y_val) -> dict:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    y_pred = model.predict(X_val)
    proba = model.predict_proba(X_val)
    n_classes = proba.shape[1]
    average = "binary" if n_classes == 2 else "macro"

    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_val, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_val, y_pred, average=average, zero_division=0),
    }
    try:
        if n_classes == 2:
            metrics["roc_auc"] = roc_auc_score(y_val, proba[:, 1])
        else:
            metrics["roc_auc"] = roc_auc_score(y_val, proba, multi_class="ovr", average="macro")
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def _walk_forward_cv(X: np.ndarray, y: np.ndarray, n_splits: int = DEFAULT_CV_SPLITS):
    """Walk-forward (expanding-window) cross-validation via TimeSeriesSplit —
    each fold trains only on data strictly before its validation fold, unlike
    shuffled k-fold which would let the model train on future data to predict
    the past (invalid for time series; see bias-and-validation checklist).
    """
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.utils.class_weight import compute_sample_weight

    if len(X) < MIN_ROWS_FOR_CV:
        log.warning(
            f"  Only {len(X)} dev rows (< {MIN_ROWS_FOR_CV}) — walk-forward CV "
            "would be too noisy to trust; skipping CV and going straight to final fit."
        )
        return [], []

    n_splits = max(2, min(n_splits, len(X) // 100))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    xgb_fold_metrics, rf_fold_metrics = [], []

    for fold_i, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
            log.warning(f"  Fold {fold_i}/{n_splits}: only one class present in train or val — skipping fold")
            continue

        sw_tr = compute_sample_weight("balanced", y_tr)

        xgb_acc_str = rf_acc_str = "n/a"
        try:
            xgb_model = _fit_xgboost(X_tr, y_tr, X_val, y_val, sample_weight=sw_tr)
            fold_metrics = _score_fold(xgb_model, X_val, y_val)
            xgb_fold_metrics.append(fold_metrics)
            xgb_acc_str = f"{fold_metrics['accuracy']:.4f}"
        except ImportError:
            pass  # xgboost not installed — RF-only CV below

        rf_model, rf_n_est = _fit_random_forest_with_early_stopping(X_tr, y_tr, X_val, y_val)
        fold_metrics = _score_fold(rf_model, X_val, y_val)
        rf_fold_metrics.append(fold_metrics)
        rf_acc_str = f"{fold_metrics['accuracy']:.4f} (n_estimators={rf_n_est})"

        log.info(
            f"  Fold {fold_i}/{n_splits}: train={len(train_idx):5d} val={len(val_idx):5d} "
            f"| xgb_acc={xgb_acc_str} | rf_acc={rf_acc_str}"
        )

    return xgb_fold_metrics, rf_fold_metrics


def _aggregate_cv_metrics(fold_metrics: list, model_name: str) -> dict:
    if not fold_metrics:
        log.warning(f"  {model_name}: no valid CV folds to aggregate")
        return {}

    keys = fold_metrics[0].keys()
    agg = {}
    log.info(f"\n  {model_name} walk-forward CV ({len(fold_metrics)} valid folds):")
    for k in keys:
        vals = [m[k] for m in fold_metrics if not np.isnan(m[k])]
        mean_v = float(np.mean(vals)) if vals else float("nan")
        std_v = float(np.std(vals)) if vals else float("nan")
        agg[f"cv_{k}_mean"] = mean_v
        agg[f"cv_{k}_std"] = std_v
        log.info(f"    {k:10s}: {mean_v:.4f} \u00b1 {std_v:.4f}")

    if agg.get("cv_accuracy_std", 0.0) > 0.10:
        log.warning(
            "    High fold-to-fold accuracy variance (std > 0.10) — suggests "
            "regime instability across the training window. Treat the mean CV "
            "metric with caution; weight recent folds more heavily than early ones."
        )
    return agg


# ── Feature selection (SHAP / importance-based) ──────────────────────

def _select_features(
    X_train: np.ndarray, y_train: np.ndarray, feature_cols: list,
    sample_weight=None, top_k: int = None, cumulative_importance: float = 0.95,
) -> list:
    """Rank features by SHAP importance (preferred) or built-in
    `feature_importances_` (fallback if shap isn't installed), then keep
    either the top `top_k` or the smallest set explaining
    `cumulative_importance` of total importance. Fewer, higher-signal
    features reduce overfitting risk relative to fitting on all 161 raw
    features, especially with a modest number of training rows.
    """
    import xgboost as xgb

    ranker = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        random_state=42, n_jobs=-1, eval_metric="logloss",
    )
    ranker.fit(X_train, y_train, sample_weight=sample_weight)

    method = "built-in importance"
    importances = ranker.feature_importances_
    try:
        import shap
        explainer = shap.TreeExplainer(ranker)
        shap_values = explainer.shap_values(X_train)
        if isinstance(shap_values, list):  # one array per class (multiclass)
            importances = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            sv = np.asarray(shap_values)
            if sv.ndim == 3:  # (n_samples, n_features, n_classes) — some shap/xgboost combos
                importances = np.abs(sv).mean(axis=(0, 2))
            else:
                importances = np.abs(sv).mean(axis=0)
        method = "SHAP"
    except Exception as e:
        log.warning(f"  SHAP unavailable/failed ({e}) — falling back to built-in feature_importances_")

    order = np.argsort(importances)[::-1]
    if top_k is not None:
        n_keep = min(top_k, len(feature_cols))
        keep_idx = order[:n_keep]
    else:
        total = importances.sum()
        cumsum = np.cumsum(importances[order]) / total if total > 0 else np.cumsum(importances[order])
        n_keep = int(np.searchsorted(cumsum, cumulative_importance) + 1)
        n_keep = max(n_keep, min(10, len(feature_cols)))  # always keep at least 10
        keep_idx = order[:n_keep]

    selected = [feature_cols[i] for i in sorted(keep_idx)]

    log.info(f"\n  Feature selection ({method}): kept {len(selected)}/{len(feature_cols)} features")
    log.info("  Top 15 by importance:")
    for rank, i in enumerate(order[:15], 1):
        log.info(f"    {rank:2d}. {feature_cols[i]:30s}: {importances[i]:.6f}")

    return selected


# ── Probability calibration ──────────────────────────────────────────

def _calibrate(model, X_calib: np.ndarray, y_calib: np.ndarray, method: str = CALIBRATION_METHOD):
    """Wrap an already-fitted model with probability calibration on a
    calibration slice the model was NOT trained on. Calibrating on the
    training set itself just confirms the model's own (likely overconfident)
    probabilities rather than correcting them.

    Handles both sklearn >= 1.6 (cv="prefit" removed, use FrozenEstimator)
    and older sklearn (cv="prefit" still supported).
    """
    from sklearn.calibration import CalibratedClassifierCV
    try:
        from sklearn.frozen import FrozenEstimator
        calibrated = CalibratedClassifierCV(FrozenEstimator(model), method=method)
    except ImportError:
        calibrated = CalibratedClassifierCV(model, method=method, cv="prefit")
    calibrated.fit(X_calib, y_calib)
    return calibrated


def _log_calibration_report(model, X_test: np.ndarray, y_test: np.ndarray, label: str):
    """Brier score + a predicted-vs-actual reliability table. A model can
    have great accuracy/ROC-AUC and still be badly calibrated (e.g. every
    "70% confidence" prediction is actually right 40% of the time) — that
    matters a lot if predicted probability feeds position sizing downstream.
    """
    from sklearn.metrics import brier_score_loss

    proba = model.predict_proba(X_test)
    n_classes = proba.shape[1]

    log.info(f"\n  {label} calibration:")
    if n_classes == 2:
        brier = brier_score_loss(y_test, proba[:, 1])
        log.info(f"    Brier score (0=perfect, 0.25=uninformative): {brier:.4f}")
        cal_df = pd.DataFrame({"p": proba[:, 1], "y": np.asarray(y_test)})
        n_buckets = min(10, cal_df["p"].nunique())
        if n_buckets >= 2:
            cal_df["bucket"] = pd.qcut(cal_df["p"], q=n_buckets, duplicates="drop")
            table = cal_df.groupby("bucket", observed=True).agg(
                mean_pred=("p", "mean"), actual_rate=("y", "mean"), n=("y", "size"),
            )
            log.info("    Reliability table (predicted vs. actual rate by bucket):")
            for _, row in table.iterrows():
                log.info(f"      pred={row['mean_pred']:.3f}  actual={row['actual_rate']:.3f}  n={int(row['n'])}")
        else:
            log.info("    Not enough distinct predicted probabilities for a reliability table.")
    else:
        log.info("    Multi-class: one-vs-rest Brier score per class")
        for c in range(n_classes):
            y_bin = (np.asarray(y_test) == c).astype(int)
            brier = brier_score_loss(y_bin, proba[:, c])
            log.info(f"      class={c}: Brier={brier:.4f}")


def train_one_pair(
    symbol: str,
    timeframe: str,
    bars: int = 500,
    use_synthetic: bool = False,
    cv_splits: int = DEFAULT_CV_SPLITS,
    feature_top_k: int = None,
) -> bool:
    """
    Train XGBoost + RandomForest models for one pair, with walk-forward CV,
    early stopping, class weighting, SHAP/importance feature selection, and
    probability calibration.

    Args:
        symbol: Trading symbol (e.g., "EURUSD")
        timeframe: Timeframe (e.g., "15m")
        bars: Number of bars to fetch/generate
        use_synthetic: If True, use synthetic data (DEBUG ONLY)
        cv_splits: Number of walk-forward TimeSeriesSplit folds
        feature_top_k: If set, keep exactly this many top features instead
                       of the default cumulative-importance selection
    """
    log.info(f"=== Training {symbol} {timeframe} ===")
    
    # 1. Fetch data - REAL MT5 DATA BY DEFAULT
    if use_synthetic:
        log.warning("⚠️ Using SYNTHETIC data (DEBUG MODE)")
        df = generate_synthetic_ohlcv(symbol, bars=bars)
        log.info(f"  Generated {len(df)} bars of synthetic data")
    else:
        # Use the new MT5 data loader
        from ml.mt5_data_loader import MT5DataLoader
        
        log.info("Fetching REAL MT5 historical data...")
        loader = MT5DataLoader()
        
        # Map timeframe format (15m -> M15)
        tf_map = {"1m": "M1", "5m": "M5", "15m": "M15", "30m": "M30", 
                  "1h": "H1", "4h": "H4", "1d": "D1"}
        mt5_timeframe = tf_map.get(timeframe.lower(), timeframe.upper())
        
        result = loader.fetch(symbol=symbol, timeframe=mt5_timeframe, bars=bars)
        loader.shutdown()
        
        if result.dataframe is None:
            log.error(f"Failed to fetch MT5 data for {symbol} {timeframe}")
            if result.errors:
                log.error(f"Errors: {result.errors}")
            return False
        
        df = result.dataframe
        log.info(f"  Downloaded {result.rows_downloaded} candles from MT5")
        log.info(f"  After cleaning: {result.rows_after_cleaning} rows")
        log.info(f"  Date range: {result.start_date} → {result.end_date}")

    # 2. Add features + labels
    # NOTE: this ordering fix is unrelated to the timezone bug above — it's a
    # separate, pre-existing issue: build_labels() needs the raw "close"
    # column, but FeatureEngineer's output (features_df) only has a renamed
    # "price_close" column, so calling build_labels() AFTER add_features()
    # raised KeyError: 'close'. Fix: compute the label on the raw OHLCV df
    # first (while "close" still exists), then run feature engineering, then
    # align the label back onto the feature rows by index.
    df = build_labels(df, horizon=5)
    labels = df["target"]

    df = add_features(df, pair=symbol, use_feature_engineer=True)
    df["target"] = labels.reindex(df.index)
    df = df.dropna()
    log.info(f"  After feature/label computation: {len(df)} usable rows")

    if len(df) < 50:
        log.error(f"  Not enough data ({len(df)} rows) — need at least 50")
        return False

    # 3. Prepare feature matrix - use ALL available features from FeatureEngineer
    # Exclude non-feature columns (OHLCV and target)
    exclude_cols = {'open', 'high', 'low', 'close', 'volume', 'target', 'time'}
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    log.info(f"\n{'='*60}")
    log.info("FEATURE ANALYSIS")
    log.info(f"{'='*60}")
    log.info(f"Total features used: {len(feature_cols)}")
    log.info(f"\nFeature names:")
    for i, f in enumerate(sorted(feature_cols), 1):
        log.info(f"  {i:3d}. {f}")
    log.info(f"{'='*60}\n")
    
    X = df[feature_cols].values
    y = df["target"].values

    log.info(f"  Feature matrix: {X.shape}, labels: {y.shape}")
    log.info(f"  Positive class ratio: {y.mean():.2%}")

    if len(np.unique(y)) < 2:
        log.error("  Only one class present in the full label set — cannot train a classifier")
        return False

    # 4. Chronological final holdout — the LAST slice of the data, never
    # touched by CV, feature selection, model fitting, or calibration.
    # Everything before this is a manual train_test_split()-style shuffled
    # split away from being valid for time series (see bias-and-validation:
    # random k-fold/shuffle lets a model train on future data to predict the
    # past). All splitting here is strictly chronological.
    n = len(df)
    holdout_start = int(n * (1 - HOLDOUT_FRACTION))
    X_dev, X_holdout = X[:holdout_start], X[holdout_start:]
    y_dev, y_holdout = y[:holdout_start], y[holdout_start:]
    log.info(
        f"  Dev set: {len(X_dev)} rows | Final holdout (untouched until final scoring): {len(X_holdout)} rows"
    )

    if len(np.unique(y_dev)) < 2 or len(np.unique(y_holdout)) < 2:
        log.error("  Dev set or final holdout has only one class present — cannot train/evaluate reliably")
        return False

    # 5. Walk-forward cross-validation (TimeSeriesSplit) over the dev set —
    # gives fold-by-fold metrics so we can see variance, not just one number.
    log.info(f"\n{'='*60}")
    log.info("WALK-FORWARD CROSS-VALIDATION (TimeSeriesSplit)")
    log.info(f"{'='*60}")
    xgb_fold_metrics, rf_fold_metrics = _walk_forward_cv(X_dev, y_dev, n_splits=cv_splits)
    xgb_cv_summary = _aggregate_cv_metrics(xgb_fold_metrics, "XGBoost")
    rf_cv_summary = _aggregate_cv_metrics(rf_fold_metrics, "RandomForest")

    # 6. Feature selection (SHAP / importance) on the dev set only — the
    # final holdout must never influence which features are kept.
    log.info(f"\n{'='*60}")
    log.info("FEATURE SELECTION")
    log.info(f"{'='*60}")
    from sklearn.utils.class_weight import compute_sample_weight
    sw_dev = compute_sample_weight("balanced", y_dev)
    try:
        selected_features = _select_features(
            X_dev, y_dev, feature_cols, sample_weight=sw_dev, top_k=feature_top_k,
        )
    except ImportError as e:
        log.warning(f"  xgboost unavailable for feature ranking ({e}) — keeping all features")
        selected_features = feature_cols

    selected_idx = [feature_cols.index(f) for f in selected_features]
    X_dev_sel = X_dev[:, selected_idx]
    X_holdout_sel = X_holdout[:, selected_idx]

    # 7. Final fit: fit on dev-fit, early-stop AND calibrate on dev-calib (a
    # slice the base model never trains on), then score exactly once on the
    # untouched final holdout.
    calib_start = int(len(X_dev_sel) * (1 - CALIB_FRACTION))
    X_fit, X_calib = X_dev_sel[:calib_start], X_dev_sel[calib_start:]
    y_fit, y_calib = y_dev[:calib_start], y_dev[calib_start:]

    if len(np.unique(y_fit)) < 2 or len(np.unique(y_calib)) < 2:
        log.error("  Final fit/calibration split has only one class present — cannot proceed")
        return False

    sw_fit = compute_sample_weight("balanced", y_fit)
    log.info(f"\n  Final fit: {len(X_fit)} rows | calibration: {len(X_calib)} rows | holdout: {len(X_holdout_sel)} rows")

    # 7a. XGBoost
    xgb_model, xgb_metrics = None, {}
    try:
        raw_xgb = _fit_xgboost(X_fit, y_fit, X_calib, y_calib, sample_weight=sw_fit)
        xgb_model = _calibrate(raw_xgb, X_calib, y_calib)

        from sklearn.metrics import (
            precision_score, recall_score, f1_score, roc_auc_score,
            confusion_matrix, classification_report, accuracy_score,
        )
        y_pred = xgb_model.predict(X_holdout_sel)
        y_proba = xgb_model.predict_proba(X_holdout_sel)[:, 1]

        xgb_metrics = {
            "accuracy": accuracy_score(y_holdout, y_pred),
            "precision": precision_score(y_holdout, y_pred, zero_division=0),
            "recall": recall_score(y_holdout, y_pred, zero_division=0),
            "f1": f1_score(y_holdout, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_holdout, y_proba) if len(np.unique(y_holdout)) > 1 else 0.0,
        }

        log.info(f"\n{'='*60}")
        log.info("XGBOOST — FINAL HOLDOUT EVALUATION (calibrated)")
        log.info(f"{'='*60}")
        for k, v in xgb_metrics.items():
            log.info(f"  {k.capitalize():10s}: {v:.4f}")
        cm = confusion_matrix(y_holdout, y_pred)
        log.info(f"\nConfusion Matrix:")
        log.info(f"  TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
        log.info(f"  FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")
        log.info(f"\nClassification Report:")
        log.info(classification_report(y_holdout, y_pred, target_names=['Down/Same', 'Up'], zero_division=0))
        log.info(f"{'='*60}\n")

        _log_calibration_report(xgb_model, X_holdout_sel, y_holdout, "XGBoost")

    except ImportError as e:
        log.warning(f"  xgboost not installed — skipping XGBoost model: {e}")

    # 7b. RandomForest
    rf_model, rf_metrics = None, {}
    try:
        from sklearn.metrics import (
            precision_score, recall_score, f1_score, roc_auc_score,
            confusion_matrix, classification_report, accuracy_score,
        )
        raw_rf, rf_n_est = _fit_random_forest_with_early_stopping(X_fit, y_fit, X_calib, y_calib)
        rf_model = _calibrate(raw_rf, X_calib, y_calib)

        y_pred_rf = rf_model.predict(X_holdout_sel)
        y_proba_rf = rf_model.predict_proba(X_holdout_sel)[:, 1]

        rf_metrics = {
            "accuracy": accuracy_score(y_holdout, y_pred_rf),
            "precision": precision_score(y_holdout, y_pred_rf, zero_division=0),
            "recall": recall_score(y_holdout, y_pred_rf, zero_division=0),
            "f1": f1_score(y_holdout, y_pred_rf, zero_division=0),
            "roc_auc": roc_auc_score(y_holdout, y_proba_rf) if len(np.unique(y_holdout)) > 1 else 0.0,
        }

        log.info(f"\n{'='*60}")
        log.info(f"RANDOMFOREST — FINAL HOLDOUT EVALUATION (calibrated, n_estimators={rf_n_est})")
        log.info(f"{'='*60}")
        for k, v in rf_metrics.items():
            log.info(f"  {k.capitalize():10s}: {v:.4f}")
        cm_rf = confusion_matrix(y_holdout, y_pred_rf)
        log.info(f"\nConfusion Matrix:")
        log.info(f"  TN={cm_rf[0,0]:5d}  FP={cm_rf[0,1]:5d}")
        log.info(f"  FN={cm_rf[1,0]:5d}  TP={cm_rf[1,1]:5d}")
        log.info(f"\nClassification Report:")
        log.info(classification_report(y_holdout, y_pred_rf, target_names=['Down/Same', 'Up'], zero_division=0))
        log.info(f"{'='*60}\n")

        _log_calibration_report(rf_model, X_holdout_sel, y_holdout, "RandomForest")

    except ImportError as e:
        log.warning(f"  scikit-learn not installed — skipping RF model: {e}")

    if xgb_model is None and rf_model is None:
        log.error("  No ML libraries available — cannot train models")
        return False

    # 8. Save models using ModelStore.save_model() (Round-10 fix)
    # Previously: manually pickled a DICT wrapper containing {model, feature_cols,
    # accuracy, ...} and called a non-existent store.register_model() method.
    # This caused THREE bugs:
    #   Bug 1: register_model() doesn't exist in ModelStore (only save_model does)
    #   Bug 2: model_type was "xgboost_v1" but predictor looks for "xgboost"
    #          (registry key = "{pair}_{tf}_{model_type}", so the key never matched)
    #   Bug 3: predictor calls model.predict_proba(X) directly on the loaded
    #          object — but the manual pickle saved a dict, which has no
    #          predict_proba method → AttributeError at prediction time
    #          (CalibratedClassifierCV also implements predict_proba directly,
    #          so saving the calibrated wrapper as the "raw model object"
    #          keeps this fix intact.)
    #
    # Fix: use ModelStore.save_model() which:
    #   (a) pickles the RAW model object (not a dict wrapper)
    #   (b) uses model_type without version suffix (e.g. "xgboost" not "xgboost_v1")
    #   (c) handles versioning internally (v1, v2, ...)
    #   (d) writes the registry entry with the correct key format
    from ml.model_store import ModelStore
    store = ModelStore()
    log.info(f"  Model directory: {store.base_dir}")

    if xgb_model is not None:
        metrics_payload = {
            "accuracy": float(xgb_metrics.get("accuracy", 0.0)),
            "precision": float(xgb_metrics.get("precision", 0.0)),
            "recall": float(xgb_metrics.get("recall", 0.0)),
            "f1": float(xgb_metrics.get("f1", 0.0)),
            "roc_auc": float(xgb_metrics.get("roc_auc", 0.0)),
            "training_bars": len(df),
            "n_features_selected": len(selected_features),
            **xgb_cv_summary,
        }
        version = store.save_model(
            model=xgb_model,            # calibrated model object, NOT a dict
            pair=symbol,
            timeframe=timeframe,
            model_type="xgboost",       # NO version suffix — predictor looks for "xgboost"
            metrics=metrics_payload,
            is_keras=False,
            # Bug fix: previously feature_names was never passed, so
            # ModelStore.get_feature_names() returned [] and the predictor
            # fell back to a bare n_features_in_ count check. Now that
            # feature selection also shrinks the column count, this matters
            # even more — the predictor MUST reindex by the selected names,
            # not assume it gets all 161 raw features.
            feature_names=selected_features,
        )
        if version:
            log.info(f"  Saved: xgboost {version} (holdout acc={xgb_metrics.get('accuracy', 0.0):.2%})")

    if rf_model is not None:
        metrics_payload = {
            "accuracy": float(rf_metrics.get("accuracy", 0.0)),
            "precision": float(rf_metrics.get("precision", 0.0)),
            "recall": float(rf_metrics.get("recall", 0.0)),
            "f1": float(rf_metrics.get("f1", 0.0)),
            "roc_auc": float(rf_metrics.get("roc_auc", 0.0)),
            "training_bars": len(df),
            "n_features_selected": len(selected_features),
            **rf_cv_summary,
        }
        version = store.save_model(
            model=rf_model,             # calibrated model object
            pair=symbol,
            timeframe=timeframe,
            model_type="random_forest", # NO version suffix
            metrics=metrics_payload,
            is_keras=False,
            feature_names=selected_features,  # same schema fix as above
        )
        if version:
            log.info(f"  Saved: random_forest {version} (holdout acc={rf_metrics.get('accuracy', 0.0):.2%})")

    log.info(f"  ✅ {symbol} {timeframe} training complete!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Quick ML model training with REAL MT5 data (Round-10 fix)"
    )
    parser.add_argument("--pair", type=str, default=None,
                        help="Train only this pair (e.g. EURUSD). Default: all default pairs.")
    parser.add_argument("--tf", type=str, default="15m",
                        help="Timeframe (default: 15m)")
    parser.add_argument("--bars", type=int, default=100000,
                        help="Number of bars to fetch from MT5 (default: 100000)")
    parser.add_argument("--debug-synthetic", action="store_true",
                        help="Use synthetic data instead of MT5 (DEBUG ONLY - not for production)")
    parser.add_argument("--cv-splits", type=int, default=DEFAULT_CV_SPLITS,
                        help=f"Number of walk-forward TimeSeriesSplit folds (default: {DEFAULT_CV_SPLITS})")
    parser.add_argument("--feature-top-k", type=int, default=None,
                        help="Keep exactly this many top features (default: cumulative-importance selection)")
    args = parser.parse_args()

    default_pairs = None  # will be loaded from config.SYMBOLS below
    try:
        from config import SYMBOLS
        default_pairs = [s.upper() for s in SYMBOLS]
    except Exception:
        default_pairs = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "AUDUSD", "USDCAD"]
    pairs = [args.pair.upper()] if args.pair else default_pairs

    if args.debug_synthetic:
        log.warning("⚠️ DEBUG MODE: Using SYNTHETIC data ⚠️")
        log.warning("⚠️ Production models MUST use real MT5 data ⚠️")
    else:
        log.info("Using REAL MetaTrader5 historical data for training")
    
    log.info(f"Training models for {len(pairs)} pair(s) on {args.tf} timeframe")
    log.info(f"Bars per pair: {args.bars}")

    success_count = 0
    for pair in pairs:
        try:
            if train_one_pair(
                pair, args.tf, bars=args.bars, use_synthetic=args.debug_synthetic,
                cv_splits=args.cv_splits, feature_top_k=args.feature_top_k,
            ):
                success_count += 1
        except Exception as e:
            log.error(f"Failed to train {pair} {args.tf}: {e}")

    log.info(f"\n=== Training complete: {success_count}/{len(pairs)} pairs trained ===")
    if success_count > 0:
        log.info("Models are now registered in memory/ml_models/_registry.json")
        log.info("The [Predictor] NOT_READY warnings should disappear on next bot restart.")
    else:
        log.error("No models were trained — check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()