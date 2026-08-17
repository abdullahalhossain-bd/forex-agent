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
            
            # Chronological split, PURGED + EMBARGOED at both boundaries.
            #
            # Every label here is computed from a forward-looking window of
            # up to max(config.label_horizons) candles (see phase4_labels.py).
            # A naive cut at train_end/val_end leaves rows whose label window
            # spans across the boundary — i.e. training rows whose label was
            # partly computed from candles that are nominally "val" or "test"
            # data. This purges those rows and adds an embargo gap on top
            # (serial correlation in rolling-window features can leak
            # backward across the boundary even without direct overlap).
            # See ml/cv_splitter.py for the reference implementation this
            # mirrors; walk-forward validation (phase10) should still be
            # treated as the primary robustness check, not this single split.
            n = len(df)
            train_end = int(n * config.train_pct)
            val_end = int(n * (config.train_pct + config.val_pct))
            horizon = max(config.label_horizons) if config.label_horizons else 0
            embargo = horizon

            train_df = df.iloc[: max(0, train_end - horizon)].copy()
            val_df = df.iloc[train_end + embargo: max(train_end + embargo, val_end - horizon)].copy()
            test_df = df.iloc[val_end + embargo:].copy()

            if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
                log.warning(
                    f"  {symbol}: purge/embargo (horizon={horizon}) emptied a split — "
                    f"falling back to naive chronological split for this symbol"
                )
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