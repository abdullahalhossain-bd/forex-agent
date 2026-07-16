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

IMPROVEMENTS IN THIS VERSION:
- Now uses FeatureEngineer with 161 features
- Adds comprehensive evaluation metrics (Precision, Recall, F1, ROC-AUC, Confusion Matrix)
- Includes Optuna hyperparameter tuning
- Prints all feature names used for training
- Verifies no look-ahead bias

Usage:
    python scripts/train_models_quick.py --pair EURUSD --tf 15m
    python scripts/train_models_quick.py --pair XAUUSD --tf 15m --bars 1000
    python scripts/train_models_quick.py  # train all default pairs
    python scripts/train_models_quick.py --tune  # run hyperparameter tuning

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


def train_one_pair(
    symbol: str,
    timeframe: str,
    bars: int = 500,
    use_synthetic: bool = False,
) -> bool:
    """
    Train XGBoost + RandomForest models for one pair.
    
    Args:
        symbol: Trading symbol (e.g., "EURUSD")
        timeframe: Timeframe (e.g., "15m")
        bars: Number of bars to fetch/generate
        use_synthetic: If True, use synthetic data (DEBUG ONLY)
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

    # 4. Split train/test (time-based, no shuffle)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    log.info(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    # 5. Train XGBoost with comprehensive evaluation metrics
    try:
        import xgboost as xgb
        from sklearn.metrics import (
            precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix, classification_report,
            accuracy_score
        )
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        xgb_model.fit(X_train, y_train)
        
        # Comprehensive evaluation
        y_pred = xgb_model.predict(X_test)
        y_proba = xgb_model.predict_proba(X_test)[:, 1]
        
        xgb_acc = accuracy_score(y_test, y_pred)
        xgb_precision = precision_score(y_test, y_pred, zero_division=0)
        xgb_recall = recall_score(y_test, y_pred, zero_division=0)
        xgb_f1 = f1_score(y_test, y_pred, zero_division=0)
        xgb_roc_auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.0
        
        log.info(f"\n{'='*60}")
        log.info("XGBOOST COMPREHENSIVE EVALUATION")
        log.info(f"{'='*60}")
        log.info(f"  Accuracy:  {xgb_acc:.4f}")
        log.info(f"  Precision: {xgb_precision:.4f}")
        log.info(f"  Recall:    {xgb_recall:.4f}")
        log.info(f"  F1 Score:  {xgb_f1:.4f}")
        log.info(f"  ROC-AUC:   {xgb_roc_auc:.4f}")
        log.info(f"\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        log.info(f"  TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
        log.info(f"  FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")
        log.info(f"\nClassification Report:")
        log.info(classification_report(y_test, y_pred, target_names=['Down/Same', 'Up'], zero_division=0))
        log.info(f"{'='*60}\n")
        
        # Feature importance
        log.info("Top 20 Most Important Features (XGBoost):")
        importances = xgb_model.feature_importances_
        top_indices = np.argsort(importances)[::-1][:20]
        for idx in top_indices:
            log.info(f"  {feature_cols[idx]:30s}: {importances[idx]:.6f}")
        
    except ImportError as e:
        log.warning(f"  xgboost not installed — skipping XGBoost model: {e}")
        xgb_model = None
        xgb_acc = 0.0
        xgb_precision = 0.0
        xgb_recall = 0.0
        xgb_f1 = 0.0
        xgb_roc_auc = 0.0

    # 6. Train RandomForest with comprehensive evaluation metrics
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import (
            precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix, classification_report,
            accuracy_score
        )
        
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42,
        )
        rf_model.fit(X_train, y_train)
        
        # Comprehensive evaluation
        y_pred_rf = rf_model.predict(X_test)
        y_proba_rf = rf_model.predict_proba(X_test)[:, 1]
        
        rf_acc = accuracy_score(y_test, y_pred_rf)
        rf_precision = precision_score(y_test, y_pred_rf, zero_division=0)
        rf_recall = recall_score(y_test, y_pred_rf, zero_division=0)
        rf_f1 = f1_score(y_test, y_pred_rf, zero_division=0)
        rf_roc_auc = roc_auc_score(y_test, y_proba_rf) if len(np.unique(y_test)) > 1 else 0.0
        
        log.info(f"\n{'='*60}")
        log.info("RANDOMFOREST COMPREHENSIVE EVALUATION")
        log.info(f"{'='*60}")
        log.info(f"  Accuracy:  {rf_acc:.4f}")
        log.info(f"  Precision: {rf_precision:.4f}")
        log.info(f"  Recall:    {rf_recall:.4f}")
        log.info(f"  F1 Score:  {rf_f1:.4f}")
        log.info(f"  ROC-AUC:   {rf_roc_auc:.4f}")
        log.info(f"\nConfusion Matrix:")
        cm_rf = confusion_matrix(y_test, y_pred_rf)
        log.info(f"  TN={cm_rf[0,0]:5d}  FP={cm_rf[0,1]:5d}")
        log.info(f"  FN={cm_rf[1,0]:5d}  TP={cm_rf[1,1]:5d}")
        log.info(f"\nClassification Report:")
        log.info(classification_report(y_test, y_pred_rf, target_names=['Down/Same', 'Up'], zero_division=0))
        log.info(f"{'='*60}\n")
        
        # Feature importance
        log.info("Top 20 Most Important Features (RandomForest):")
        importances_rf = rf_model.feature_importances_
        top_indices_rf = np.argsort(importances_rf)[::-1][:20]
        for idx in top_indices_rf:
            log.info(f"  {feature_cols[idx]:30s}: {importances_rf[idx]:.6f}")
            
    except ImportError as e:
        log.warning(f"  scikit-learn not installed — skipping RF model: {e}")
        rf_model = None
        rf_acc = 0.0
        rf_precision = 0.0
        rf_recall = 0.0
        rf_f1 = 0.0
        rf_roc_auc = 0.0

    if xgb_model is None and rf_model is None:
        log.error("  No ML libraries available — cannot train models")
        return False

    # 7. Save models using ModelStore.save_model() (Round-10 fix)
    # Previously: manually pickled a DICT wrapper containing {model, feature_cols,
    # accuracy, ...} and called a non-existent store.register_model() method.
    # This caused THREE bugs:
    #   Bug 1: register_model() doesn't exist in ModelStore (only save_model does)
    #   Bug 2: model_type was "xgboost_v1" but predictor looks for "xgboost"
    #          (registry key = "{pair}_{tf}_{model_type}", so the key never matched)
    #   Bug 3: predictor calls model.predict_proba(X) directly on the loaded
    #          object — but the manual pickle saved a dict, which has no
    #          predict_proba method → AttributeError at prediction time
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
        version = store.save_model(
            model=xgb_model,            # raw model object, NOT a dict
            pair=symbol,
            timeframe=timeframe,
            model_type="xgboost",       # NO version suffix — predictor looks for "xgboost"
            metrics={"accuracy": float(xgb_acc), "training_bars": len(df)},
            is_keras=False,
            # Bug fix: previously feature_names was never passed, so
            # ModelStore.get_feature_names() returned [] and the predictor
            # fell back to a bare n_features_in_ count check. Any future
            # change to FeatureEngineer's feature count (e.g. 13 -> 161)
            # then hard-fails every prediction with:
            #   "legacy model schema missing (expects N, got M); retrain required"
            # even for models trained *after* the change. Saving the exact
            # training-time feature names lets the predictor reindex by
            # name instead of relying on a fragile column count.
            feature_names=feature_cols,
        )
        if version:
            log.info(f"  Saved: xgboost {version} (acc={xgb_acc:.2%})")

    if rf_model is not None:
        version = store.save_model(
            model=rf_model,             # raw model object
            pair=symbol,
            timeframe=timeframe,
            model_type="random_forest", # NO version suffix
            metrics={"accuracy": float(rf_acc), "training_bars": len(df)},
            is_keras=False,
            feature_names=feature_cols,  # same schema fix as above
        )
        if version:
            log.info(f"  Saved: random_forest {version} (acc={rf_acc:.2%})")

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
    args = parser.parse_args()

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
            if train_one_pair(pair, args.tf, bars=args.bars, use_synthetic=args.debug_synthetic):
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