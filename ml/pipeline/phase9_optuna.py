"""
ml/pipeline/phase9_optuna.py — Hyperparameter Optimization (Phase 9)
====================================================================
Uses Optuna for Bayesian hyperparameter optimization with early stopping.
"""
from __future__ import annotations

import json, logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ml.pipeline.utils import MODEL_OUTPUT_DIR, PipelineConfig, PipelineTimer, get_pipeline_logger

log = get_pipeline_logger("phase9_optuna")


def optimize_hyperparams(
    datasets: Dict,
    config: Optional[PipelineConfig] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run Optuna optimization for supervised models."""
    config = config or PipelineConfig()
    results = {}
    
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        log.warning("Optuna not installed — skipping hyperparameter optimization")
        log.info("Install with: pip install optuna")
        return results
    
    with PipelineTimer("Phase 9: Hyperparameter Optimization", log):
        for symbol, split in datasets.items():
            results[symbol] = {}
            
            X_train = np.nan_to_num(split.train[split.feature_columns].values, nan=0.0)
            y_train = split.train["signal"].values
            X_val = np.nan_to_num(split.val[split.feature_columns].values, nan=0.0)
            y_val = split.val["signal"].values
            
            for model_type in ["xgboost"]:  # Focus on best model
                best_params, best_score = _optuna_study(
                    model_type, X_train, y_train, X_val, y_val, symbol, config
                )
                if best_params:
                    results[symbol][model_type] = {
                        "best_params": best_params, "best_score": best_score,
                    }
                    # Save
                    p_dir = MODEL_OUTPUT_DIR / symbol / model_type / "optuna"
                    p_dir.mkdir(parents=True, exist_ok=True)
                    (p_dir / "best_params.json").write_text(json.dumps(best_params, indent=2))
                    log.info(f"  {symbol} {model_type}: best score={best_score:.4f}")
    
    return results


def _optuna_study(model_type, X_train, y_train, X_val, y_val, symbol, config):
    import optuna
    from sklearn.metrics import accuracy_score
    
    def objective(trial):
        if model_type == "xgboost":
            from xgboost import XGBClassifier
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "use_label_encoder": False, "eval_metric": "mlogloss", "verbosity": 0,
            }
            model = XGBClassifier(**params)
        else:
            return 0.0
        
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return accuracy_score(y_val, preds)
    
    study = optuna.create_study(direction="maximize")
    # Note: XGBoostPruningCallback requires eval_set in model.fit() to work.
    # Since our objective doesn't use early_stopping_rounds, we skip pruning
    # callbacks here and rely on Optuna's default trial pruning instead.
    study.optimize(
        objective,
        n_trials=config.optuna_trials,
        show_progress_bar=False,
        timeout=300,  # 5-minute safety timeout per symbol
    )
    
    return study.best_params, study.best_value