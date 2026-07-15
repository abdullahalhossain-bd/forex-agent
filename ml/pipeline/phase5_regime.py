"""
ml/pipeline/phase5_regime.py — Market Regime Detection (Phase 5)
================================================================
Classifies every candle into one of 6 regime categories:
  - TRENDING (uptrend or downtrend)
  - RANGING (sideways, low ADX)
  - HIGH_VOLATILITY
  - LOW_VOLATILITY
  - BREAKOUT (price breaking range with volume)
  - REVERSAL (trend change detected)

Uses ADX, ATR, volume, and price action for classification.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ml.pipeline.utils import PipelineConfig, PipelineTimer, get_pipeline_logger

log = get_pipeline_logger("phase5_regime")

REGIME_LABELS = {
    0: "TRENDING_UP",
    1: "TRENDING_DOWN",
    2: "RANGING",
    3: "HIGH_VOLATILITY",
    4: "LOW_VOLATILITY",
    5: "BREAKOUT",
    6: "REVERSAL",
}


def detect_regimes(
    featured_data: Dict[str, pd.DataFrame],
    config: Optional[PipelineConfig] = None,
) -> Dict[str, pd.DataFrame]:
    """Add regime labels to featured DataFrames."""
    config = config or PipelineConfig()
    
    with PipelineTimer("Phase 5: Regime Detection", log):
        for symbol, df in featured_data.items():
            df = _classify_regime(df)
            featured_data[symbol] = df
            
            counts = df["regime"].value_counts().sort_index()
            for idx, cnt in counts.items():
                log.info(f"    {REGIME_LABELS.get(idx, '?')}: {cnt} ({cnt/len(df)*100:.1f}%)")
    
    return featured_data


def _classify_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Classify each candle into a regime category."""
    n = len(df)
    df["regime"] = 2  # Default: RANGING
    
    # Extract indicators (use computed features if available, otherwise compute)
    adx = df.get("adx", _simple_adx(df["high"], df["low"], df["close"]))
    atr = df.get("atr", _simple_atr(df["high"], df["low"], df["close"]))
    close = df["close"]
    ema20 = df.get("ema_20", close.rolling(20).mean())
    vol = df.get("volume", pd.Series(1, index=df.index))
    vol_sma = vol.rolling(20).mean()
    
    # Volatility regime (based on ATR percentile)
    atr_pct = atr / close
    atr_pct_rolling = atr_pct.rolling(100).rank(pct=True)
    
    # 1. HIGH_VOLATILITY: ATR in top 20th percentile
    df.loc[atr_pct_rolling > 0.8, "regime"] = 3
    
    # 2. LOW_VOLATILITY: ATR in bottom 20th percentile
    df.loc[atr_pct_rolling < 0.2, "regime"] = 4
    
    # 3. TRENDING: ADX > 25
    df.loc[adx > 25, "regime"] = 0  # Will refine direction below
    
    # Trend direction
    df.loc[(df["regime"] == 0) & (close > ema20), "regime"] = 0  # TRENDING_UP
    df.loc[(df["regime"] == 0) & (close < ema20), "regime"] = 1  # TRENDING_DOWN
    
    # 4. BREAKOUT: Price breaks 20-period range with high volume
    range_high = df["high"].rolling(20).max().shift(1)
    range_low = df["low"].rolling(20).min().shift(1)
    breakout_up = (close > range_high) & (vol > vol_sma * 1.5)
    breakout_down = (close < range_low) & (vol > vol_sma * 1.5)
    df.loc[breakout_up | breakout_down, "regime"] = 5
    
    # 5. REVERSAL: Trend was present but EMA cross just happened
    ema_cross_up = (ema20.shift(1) < ema20.shift(2)) & (ema20 > ema20.shift(1))
    ema_cross_down = (ema20.shift(1) > ema20.shift(2)) & (ema20 < ema20.shift(1))
    was_trending = df["regime"].isin([0, 1])
    df.loc[(was_trending | was_trending.shift(1)) & ema_cross_up, "regime"] = 6
    df.loc[(was_trending | was_trending.shift(1)) & ema_cross_down, "regime"] = 6
    
    # RANGING is default (already set)
    return df


def _simple_atr(high, low, close, period=14):
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def _simple_adx(high, low, close, period=14):
    atr = _simple_atr(high, low, close, period)
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_di = 100 * plus_dm.rolling(period).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(period).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(period).mean()