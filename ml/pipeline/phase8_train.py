"""
ml/pipeline/phase8_train.py — Model Training (Phase 8)
=======================================================
Trains multiple model types:
  Supervised: XGBoost, LightGBM, CatBoost, Random Forest
  RL: PPO, A2C (using EnhancedTradingEnv)

Each model is saved to its own folder under data/trained_models/{symbol}/{model_type}/
"""
from __future__ import annotations

import json, logging, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ml.pipeline.utils import (
    MODEL_OUTPUT_DIR, PIPELINE_CACHE_DIR, PipelineConfig, PipelineTimer,
    dataset_hash, get_pipeline_logger,
)

log = get_pipeline_logger("phase8_train")


def train_all_models(
    datasets: Dict,  # symbol -> DatasetSplit
    config: Optional[PipelineConfig] = None,
) -> Dict[str, Dict[str, Any]]:
    """Train all configured models. Returns {symbol: {model_type: result}}."""
    config = config or PipelineConfig()
    results = {}
    
    with PipelineTimer("Phase 8: Model Training", log):
        for symbol, split in datasets.items():
            results[symbol] = {}
            
            # ── Supervised models ──────────────────────────────────
            X_train = split.train[split.feature_columns].values
            y_train = split.train["signal"].values
            X_val = split.val[split.feature_columns].values
            y_val = split.val["signal"].values
            
            # Replace NaN/inf
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
            X_val = np.nan_to_num(X_val, nan=0.0, posinf=1.0, neginf=-1.0)
            
            for model_type in config.supervised_models:
                try:
                    result = _train_supervised(model_type, X_train, y_train, X_val, y_val, symbol, split.feature_columns, config)
                    if result:
                        results[symbol][model_type] = result
                except Exception as e:
                    log.warning(f"  {symbol} {model_type}: training failed — {e}")
            
            # ── RL models ──────────────────────────────────────────
            for algo in config.rl_algorithms:
                try:
                    result = _train_rl(algo, split.train, split.feature_columns, symbol, config)
                    if result:
                        results[symbol][algo] = result
                except Exception as e:
                    log.warning(f"  {symbol} {algo}: RL training failed — {e}")
    
    return results


def _train_supervised(model_type: str, X_train, y_train, X_val, y_val, symbol: str, feature_cols: list, config: PipelineConfig) -> Optional[Dict]:
    """Train a single supervised model."""
    log.info(f"  Training {symbol} {model_type}...")
    start = time.time()
    
    if model_type == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                              use_label_encoder=False, eval_metric="mlogloss", verbosity=0)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elif model_type == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, verbose=-1)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        except ImportError:
            log.warning("  lightgbm not installed, skipping")
            return None
    elif model_type == "catboost":
        try:
            from catboost import CatBoostClassifier
            model = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.05,
                                       verbose=0, allow_writing_files=False)
            model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        except ImportError:
            log.warning("  catboost not installed, skipping")
            return None
    elif model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)
    else:
        log.warning(f"  Unknown model type: {model_type}")
        return None
    
    # Evaluate
    from sklearn.metrics import accuracy_score, classification_report
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    elapsed = time.time() - start
    
    log.info(f"    {symbol} {model_type}: accuracy={acc:.4f} ({elapsed:.1f}s)")
    
    # Save model
    model_dir = MODEL_OUTPUT_DIR / symbol / model_type
    model_dir.mkdir(parents=True, exist_ok=True)
    
    import pickle
    model_path = model_dir / "model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(model, f)
    
    # Save metadata
    meta = {
        "model_type": model_type, "symbol": symbol,
        "accuracy": round(float(acc), 4),
        "n_features": len(feature_cols), "feature_columns": feature_cols,
        "n_train": len(X_train), "n_val": len(X_val),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_seconds": round(elapsed, 1),
        "data_hash": dataset_hash(pd.DataFrame(X_train)),
    }
    (model_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    
    # Classification report
    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    (model_dir / "classification_report.json").write_text(json.dumps(report, indent=2, default=str))
    
    return meta


def _train_rl(algo: str, train_df: pd.DataFrame, feature_cols: list, symbol: str, config: PipelineConfig) -> Optional[Dict]:
    """Train an RL model using the EnhancedTradingEnv."""
    try:
        from stable_baselines3 import PPO, A2C
    except ImportError:
        log.warning("  stable-baselines3 not installed, skipping RL")
        return None
    
    from ml.pipeline.phase7_rl_env import EnhancedTradingEnv
    
    log.info(f"  Training {symbol} {algo} RL ({config.rl_timesteps} timesteps)...")
    start = time.time()
    
    env = EnhancedTradingEnv(
        df=train_df, feature_columns=feature_cols,
        initial_balance=config.initial_balance,
        risk_per_trade=config.risk_per_trade,
        spread_pips=config.max_spread_pips,
        slippage_pips=config.slippage_pips,
        pair=symbol,
    )
    
    if algo == "ppo":
        model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048, batch_size=64,
                     n_epochs=10, gamma=0.99, clip_range=0.2, ent_coef=0.01, verbose=0)
    elif algo == "a2c":
        model = A2C("MlpPolicy", env, learning_rate=3e-4, n_steps=2048, gamma=0.99,
                     ent_coef=0.01, verbose=0)
    else:
        return None
    
    model.learn(total_timesteps=config.rl_timesteps)
    elapsed = time.time() - start
    
    # Save
    model_dir = MODEL_OUTPUT_DIR / symbol / algo
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.zip"
    model.save(str(model_path))
    
    meta = {
        "model_type": algo, "symbol": symbol, "algorithm": algo.upper(),
        "timesteps": config.rl_timesteps, "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_seconds": round(elapsed, 1), "n_features": len(feature_cols),
        "observation_space": str(env.observation_space.shape),
    }
    (model_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    
    log.info(f"    {symbol} {algo}: saved to {model_path} ({elapsed:.1f}s)")
    return meta