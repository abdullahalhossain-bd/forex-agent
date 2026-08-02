"""
ml/institutional_feature_adapter.py — Live adapter for phase3_features.py
============================================================================

ml/pipeline/phase3_features.py computes features in BATCH over an entire
historical dataframe (used by ml/train_historical.py to train models saved
under data/trained_models/{pair}/{model_type}/). That's a different feature
set (different names, different indicators) than ml/feature_engineer.py,
which is what the LIVE prediction path (ml/model_predictor.py) has always
used to build its `features: Dict[str, float]` input.

Two independent pipelines producing two different feature schemas from the
same OHLCV data means a model trained by one is NOT directly usable by the
other's prediction code -- see the "duplicate pipeline" audit finding.

This module closes that gap for the institutional side: given a recent
OHLCV window, it runs the SAME `_add_*_features` functions phase3_features.py
uses (same code, same math, same column names) and returns the LAST row as
a feature dict -- so a model trained via ml/train_historical.py can be
scored on a live/current bar using features it will actually recognize.

Usage:
    from ml.institutional_feature_adapter import build_institutional_features
    feats = build_institutional_features(recent_ohlcv_df)  # dict, last-row features
"""
from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from utils.logger import get_logger

log = get_logger("institutional_feature_adapter")

# Longest lookback used anywhere in phase3_features.py (sma_200/ema_200).
# Feed at least this many bars of warm-up so the last row's indicators are
# fully formed (not NaN from insufficient history).
MIN_WARMUP_BARS = 220


def build_institutional_features(df_recent: pd.DataFrame) -> Optional[Dict[str, float]]:
    """Build a phase3-schema feature dict for the LAST row of `df_recent`.

    Args:
        df_recent: OHLCV dataframe, chronological, at least MIN_WARMUP_BARS
            rows (more is fine — only the tail matters for indicator
            warm-up; the returned features describe the LAST row).

    Returns:
        Dict of feature_name -> float for the last row, or None if there
        isn't enough history to compute stable indicators.
    """
    if df_recent is None or len(df_recent) < MIN_WARMUP_BARS:
        log.debug(f"[InstitutionalAdapter] need >= {MIN_WARMUP_BARS} bars, got "
                  f"{0 if df_recent is None else len(df_recent)}")
        return None

    try:
        from ml.pipeline.phase3_features import (
            _add_trend_features, _add_momentum_features, _add_volume_features,
            _add_volatility_features, _add_structure_features,
            _add_session_features, _add_time_features,
        )
    except Exception as e:
        log.warning(f"[InstitutionalAdapter] could not import phase3_features internals: {e}")
        return None

    df = df_recent.copy().reset_index(drop=True)
    import numpy as np
    with np.errstate(invalid="ignore"):
        df = _add_trend_features(df)
        df = _add_momentum_features(df)
        df = _add_volume_features(df)
        df = _add_volatility_features(df)
        df = _add_structure_features(df)
        df = _add_session_features(df)
        df = _add_time_features(df)

    last_row = df.iloc[-1]
    exclude = {"timestamp", "open", "high", "low", "close", "volume"}
    feats = {
        col: float(last_row[col]) for col in df.columns
        if col not in exclude and pd.notna(last_row[col])
    }
    return feats