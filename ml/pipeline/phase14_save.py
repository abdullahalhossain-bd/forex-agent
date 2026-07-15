"""
ml/pipeline/phase14_save.py — Model Persistence (Phase 14)
==========================================================
Saves complete model artifacts: weights, config, feature list,
normalizer, scaler, metadata, training dataset hash, timestamp.
"""
from __future__ import annotations

import json, logging, pickle, hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ml.pipeline.utils import MODEL_OUTPUT_DIR, PipelineConfig, PipelineTimer, dataset_hash, get_pipeline_logger

log = get_pipeline_logger("phase14_save")


def save_production_artifacts(
    datasets: Dict,
    best_models: Dict[str, Dict[str, Any]],
    config: Optional[PipelineConfig] = None,
) -> Dict[str, str]:
    """Save production-ready model artifacts for each symbol's best model."""
    config = config or PipelineConfig()
    artifacts = {}
    
    with PipelineTimer("Phase 14: Save Production Artifacts", log):
        for symbol, info in best_models.items():
            model_type = info["best_model"]
            model_dir = MODEL_OUTPUT_DIR / symbol / model_type
            
            # Save feature list and normalizer
            if symbol in datasets:
                split = datasets[symbol]
                features = split.feature_columns
                
                # Compute normalizer stats from training data
                X_train = split.train[features].select_dtypes(include=np.number)
                means = X_train.mean().to_dict()
                stds = X_train.std().replace(0, 1).to_dict()
                
                normalizer = {"means": means, "stds": stds}
                with (model_dir / "normalizer.pkl").open("wb") as f:
                    pickle.dump(normalizer, f)
                
                # Feature list
                (model_dir / "feature_list.json").write_text(json.dumps(features, indent=2))
                
                # Dataset hash for change detection
                (model_dir / "dataset_hash.txt").write_text(split.train_hash)
            
            # Production metadata
            prod_meta = {
                "symbol": symbol,
                "model_type": model_type,
                "composite_score": info["score"],
                "backtest_metrics": info.get("best_metrics", {}),
                "pipeline_config": config.to_dict(),
                "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "n_features": len(datasets[symbol].feature_columns) if symbol in datasets else 0,
                "training_hash": datasets[symbol].train_hash if symbol in datasets else "",
            }
            
            prod_dir = model_dir / "production"
            prod_dir.mkdir(parents=True, exist_ok=True)
            (prod_dir / "metadata.json").write_text(json.dumps(prod_meta, indent=2, default=str))
            
            artifacts[symbol] = str(prod_dir / "metadata.json")
            log.info(f"  {symbol}: production artifacts saved to {prod_dir}")
    
    return artifacts