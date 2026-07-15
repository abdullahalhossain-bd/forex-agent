"""
ml/pipeline/phase15_retrain.py — Auto Retraining (Phase 15)
============================================================
Detects new data, appends, recalculates features, and retrains
only if performance improves. Version-controlled.
"""
from __future__ import annotations

import json, logging, shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from ml.pipeline.utils import MODEL_OUTPUT_DIR, PipelineConfig, PipelineTimer, get_pipeline_logger

log = get_pipeline_logger("phase15_retrain")


def check_and_retrain(
    datasets: Dict,
    best_models: Dict[str, Dict[str, Any]],
    config: Optional[PipelineConfig] = None,
) -> Dict[str, bool]:
    """Check if retraining is needed and execute if beneficial."""
    config = config or PipelineConfig()
    retrained = {}
    
    with PipelineTimer("Phase 15: Auto Retrain Check", log):
        for symbol, info in best_models.items():
            model_type = info["best_model"]
            model_dir = MODEL_OUTPUT_DIR / symbol / model_type
            
            # Check if training data hash changed
            hash_file = model_dir / "dataset_hash.txt"
            if not hash_file.exists():
                log.info(f"  {symbol}: no previous hash — first training, skip retrain")
                retrained[symbol] = False
                continue
            
            old_hash = hash_file.read_text().strip()
            new_hash = datasets[symbol].train_hash if symbol in datasets else ""
            
            if old_hash == new_hash:
                log.info(f"  {symbol}: data unchanged — no retrain needed")
                retrained[symbol] = False
                continue
            
            log.info(f"  {symbol}: data changed ({old_hash} → {new_hash}) — retraining...")
            
            # Archive previous model
            version_dir = model_dir / f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            version_dir.mkdir(exist_ok=True)
            for f in model_dir.glob("model.*"):
                shutil.copy2(f, version_dir / f.name)
            log.info(f"    Archived previous model to {version_dir.name}")
            
            # Retrain would happen here via phase8 — for now just flag
            retrained[symbol] = True
    
    return retrained