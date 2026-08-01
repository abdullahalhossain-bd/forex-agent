"""
ml/pipeline/phase4_labels.py — Label Generation (Phase 4)
=========================================================
Creates supervised learning labels and future return metrics.

Labels:
  - signal: BUY (1), SELL (2), HOLD (0) — based on future price movement

Metrics (per horizon):
  - future_return: % change over horizon
  - future_pips: absolute pip change
  - max_drawdown: max adverse excursion in pips
  - max_profit: max favorable excursion in pips
  - rr_ratio: max_profit / max_drawdown (if drawdown > 0)

Horizons: configurable (default: 5, 10, 20, 50 candles)
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ml.pipeline.utils import PIPELINE_CACHE_DIR, PipelineConfig, PipelineTimer, get_pipeline_logger

log = get_pipeline_logger("phase4_labels")


def generate_labels(
    featured_data: Dict[str, pd.DataFrame],
    config: Optional[PipelineConfig] = None,
) -> Dict[str, pd.DataFrame]:
    """Add labels to featured DataFrames. Returns updated dict."""
    config = config or PipelineConfig()
    
    with PipelineTimer("Phase 4: Label Generation", log):
        for symbol, df in featured_data.items():
            if len(df) < 100:
                log.warning(f"  {symbol}: skipping — only {len(df)} rows")
                continue
            
            log.info(f"  {symbol}: generating labels for {len(config.label_horizons)} horizons...")
            
            for h in config.label_horizons:
                df = _add_horizon_labels(df, h, symbol)
            
            # Primary signal label (based on 20-candle horizon or middle)
            primary_h = config.label_horizons[len(config.label_horizons) // 2]
            df = _add_primary_signal(df, primary_h)
            
            featured_data[symbol] = df
    
    return featured_data


def _add_horizon_labels(df: pd.DataFrame, horizon: int, symbol: str) -> pd.DataFrame:
    """Add future return metrics for a specific horizon."""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    
    # Future return
    future_close = close.shift(-horizon)
    df[f"fut_ret_{horizon}"] = (future_close - close) / close
    
    # Future pips (approximate — assumes 5-digit pricing for majors, 3 for JPY/XAU)
    pip_mult = 10000 if not symbol.endswith("JPY") and symbol != "XAUUSD" else 100
    df[f"fut_pips_{horizon}"] = (future_close - close) * pip_mult
    
    # Max drawdown (max adverse excursion within horizon)
    future_low = low.shift(-1).rolling(horizon).min()
    df[f"max_dd_{horizon}"] = (close - future_low) * pip_mult
    
    # Max profit (max favorable excursion within horizon)
    future_high = high.shift(-1).rolling(horizon).max()
    df[f"max_prof_{horizon}"] = (future_high - close) * pip_mult
    
    # RR ratio
    dd = df[f"max_dd_{horizon}"].clip(lower=0.1)  # Avoid division by zero
    df[f"rr_{horizon}"] = df[f"max_prof_{horizon}"].clip(lower=0) / dd
    
    return df


def _add_primary_signal(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Add primary BUY/SELL/HOLD signal based on future return.
    
    Threshold:
      - BUY:  future_return > +0.1% (sufficiently profitable)
      - SELL: future_return < -0.1%
      - HOLD: everything else (insufficient edge)
    """
    ret = df[f"fut_ret_{horizon}"]
    threshold = 0.001  # 0.1% = 10 pips on EURUSD
    
    df["signal"] = 0  # Default HOLD
    df.loc[ret > threshold, "signal"] = 1  # BUY
    df.loc[ret < -threshold, "signal"] = 2  # SELL
    
    log.info(f"    Signal distribution (h={horizon}): "
             f"HOLD={((df['signal']==0).sum())}, "
             f"BUY={((df['signal']==1).sum())}, "
             f"SELL={((df['signal']==2).sum())}")
    
    return df