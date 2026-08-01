"""
ml/pipeline/phase10_walkforward.py — Walk-Forward Validation (Phase 10)
======================================================================
Rolling window and expanding window validation.
Never evaluates on a single split — generates metrics for every fold.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ml.pipeline.utils import PipelineConfig, PipelineTimer, get_pipeline_logger

log = get_pipeline_logger("phase10_walkforward")


def walk_forward_validation(
    datasets: Dict,
    config: Optional[PipelineConfig] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Run walk-forward validation for each symbol."""
    config = config or PipelineConfig()
    results = {}
    
    with PipelineTimer("Phase 10: Walk-Forward Validation", log):
        for symbol, split in datasets.items():
            # Combine train+val for walk-forward
            full_df = pd.concat([split.train, split.val], ignore_index=True)
            features = split.feature_columns
            
            fold_results = []
            n = len(full_df)
            fold_size = n // (config.walk_forward_folds + 1)
            
            for fold in range(config.walk_forward_folds):
                train_end = int((fold + 1) * fold_size * config.walk_forward_train_pct)
                test_end = int((fold + 1) * fold_size) + fold_size
                
                if test_end > n:
                    break
                
                train_data = full_df.iloc[:train_end]
                test_data = full_df.iloc[train_end:test_end]
                
                if len(test_data) < config.walk_forward_min_test_size:
                    continue
                
                X_tr = np.nan_to_num(train_data[features].values, nan=0.0)
                y_tr = train_data["signal"].values
                X_te = np.nan_to_num(test_data[features].values, nan=0.0)
                y_te = test_data["signal"].values
                
                # Quick model evaluation — try XGBoost, fall back to RandomForest
                try:
                    model = None
                    try:
                        from xgboost import XGBClassifier
                        model = XGBClassifier(n_estimators=100, max_depth=6, verbosity=0,
                                              use_label_encoder=False, eval_metric="mlogloss")
                    except ImportError:
                        from sklearn.ensemble import RandomForestClassifier
                        model = RandomForestClassifier(n_estimators=100, max_depth=8, n_jobs=-1, random_state=42)
                        log.debug(f"  {symbol} fold {fold+1}: using RandomForest (xgboost not available)")
                    
                    model.fit(X_tr, y_tr)
                    y_pred = model.predict(X_te)
                    
                    metrics = {
                        "fold": fold + 1,
                        "train_size": len(X_tr), "test_size": len(X_te),
                        "accuracy": round(accuracy_score(y_te, y_pred), 4),
                        "precision_macro": round(precision_score(y_te, y_pred, average="macro", zero_division=0), 4),
                        "recall_macro": round(recall_score(y_te, y_pred, average="macro", zero_division=0), 4),
                        "f1_macro": round(f1_score(y_te, y_pred, average="macro", zero_division=0), 4),
                        "train_start": str(train_data["timestamp"].iloc[0]) if "timestamp" in train_data else "",
                        "test_start": str(test_data["timestamp"].iloc[0]) if "timestamp" in test_data else "",
                    }
                    fold_results.append(metrics)
                    log.info(f"  {symbol} fold {fold+1}: acc={metrics['accuracy']:.4f} f1={metrics['f1_macro']:.4f}")
                except Exception as e:
                    log.warning(f"  {symbol} fold {fold+1}: failed — {e}")
            
            results[symbol] = fold_results
            
            if fold_results:
                avg_acc = np.mean([f["accuracy"] for f in fold_results])
                avg_f1 = np.mean([f["f1_macro"] for f in fold_results])
                log.info(f"  {symbol} WF avg: acc={avg_acc:.4f} f1={avg_f1:.4f} ({len(fold_results)} folds)")
    
    return results