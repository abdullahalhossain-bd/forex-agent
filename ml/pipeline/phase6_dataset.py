"""
ml/pipeline/phase6_dataset.py — Dataset Creation (Phase 6)
==========================================================
Creates training, validation, and test sets using CHRONOLOGICAL split.
No random splitting — prevents data leakage.

Split ratios (default): 70% train / 15% val / 15% test
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml.pipeline.utils import PIPELINE_CACHE_DIR, PipelineConfig, PipelineTimer, get_pipeline_logger

log = get_pipeline_logger("phase6_dataset")


@dataclass
class DatasetSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    symbol: str
    feature_columns: List[str]
    train_hash: str
    total_rows: int


def create_datasets(
    featured_data: Dict[str, pd.DataFrame],
    config: Optional[PipelineConfig] = None,
) -> Dict[str, DatasetSplit]:
    """Create chronological train/val/test splits for each symbol."""
    config = config or PipelineConfig()
    datasets = {}
    
    with PipelineTimer("Phase 6: Dataset Creation", log):
        for symbol, df in featured_data.items():
            if len(df) < 500:
                log.warning(f"  {symbol}: skipping — only {len(df)} rows (need 500+)")
                continue
            
            # Identify feature columns (exclude price, volume, label, regime, and meta columns)
            exclude = {
                "timestamp", "open", "high", "low", "close", "volume",
                "tick_vol", "real_vol", "spread", "signal", "regime",
                "weekday",
            }
            # Also exclude horizon-specific label columns
            exclude.update({c for c in df.columns if c.startswith("fut_") or c.startswith("max_") or c.startswith("rr_")})
            
            feature_cols = [c for c in df.columns if c not in exclude]
            
            # Chronological split
            n = len(df)
            train_end = int(n * config.train_pct)
            val_end = int(n * (config.train_pct + config.val_pct))
            
            train_df = df.iloc[:train_end].copy()
            val_df = df.iloc[train_end:val_end].copy()
            test_df = df.iloc[val_end:].copy()
            
            # Build hash of training data for change detection
            train_hash = f"{len(train_df)}_{train_df['timestamp'].min()}_{train_df['timestamp'].max()}"
            
            split = DatasetSplit(
                train=train_df, val=val_df, test=test_df,
                symbol=symbol, feature_columns=feature_cols,
                train_hash=train_hash, total_rows=n,
            )
            datasets[symbol] = split
            
            log.info(f"  {symbol}: train={len(train_df)} | val={len(val_df)} | test={len(test_df)} "
                     f"| features={len(feature_cols)}")
    
    return datasets