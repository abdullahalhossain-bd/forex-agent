# analysis/_pattern_context.py
# =============================================================================
# Shared, causal MarketContext for the candlestick pattern engine.
#
# This module was referenced (import analysis._pattern_context import
# MarketContext, build_context / atr) by analysis/candlestick_engine.py and
# analysis/candlestick_patterns_br.py but did not exist anywhere in the
# codebase — every call site was hitting ModuleNotFoundError at import time,
# which meant the entire unified candlestick engine (analysis/candlestick_
# engine.py) could never be imported, let alone run.
#
# Builds ATR, volatility regime, location-in-range, volume z-score, and a
# multi-model trend read (label + cross-model agreement score) ONCE per
# DataFrame, so candlestick_engine.py doesn't recompute the same indicators
# on every pattern-scoring call.
#
# CAUSALITY: every rolling/ewm calculation here only uses data up to and
# including the current row (pandas' default rolling/ewm window is
# backward-looking) — no `.shift(-1)`, no centered windows. Context at bar i
# never depends on bar i+1 or later, matching the "no future leakage, no
# repainting" design principle documented at the top of candlestick_engine.py.
# =============================================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

# Trend-label vocabulary matches analysis/candlestick_patterns_mw.py's own
# _compute_trend() so a MarketContext-derived trend_mw array can be passed
# straight into mw.compute(precomputed_trend=...) without translation.
_TREND_LABELS = ("uptrend", "downtrend", "sideways")


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's Average True Range. Causal — uses only bars up to i."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    # Wilder smoothing is an EMA with alpha = 1/period.
    return true_range.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def _trend_votes(close: pd.Series) -> List[pd.Series]:
    """Three independent, simple causal trend reads (each in
    {"uptrend", "downtrend", "sideways"}), used both to build the primary
    trend_mw series (majority vote) and to score cross-model agreement.
    Bars with insufficient history default to "sideways" (the same
    conservative default candlestick_patterns_mw.py's own trend detector
    uses before it has enough bars).
    """
    votes: List[pd.Series] = []

    # 1) EMA20 vs EMA50 cross, with a small deadband to avoid noise flips.
    ema_fast = close.ewm(span=20, min_periods=20, adjust=False).mean()
    ema_slow = close.ewm(span=50, min_periods=50, adjust=False).mean()
    v1 = pd.Series("sideways", index=close.index, dtype=object)
    v1[ema_fast > ema_slow * 1.0005] = "uptrend"
    v1[ema_fast < ema_slow * 0.9995] = "downtrend"
    votes.append(v1)

    # 2) Price vs SMA50, confirmed by a rising/falling SMA slope.
    sma50 = close.rolling(50, min_periods=50).mean()
    slope = sma50.diff(10)
    v2 = pd.Series("sideways", index=close.index, dtype=object)
    v2[(close > sma50) & (slope > 0)] = "uptrend"
    v2[(close < sma50) & (slope < 0)] = "downtrend"
    votes.append(v2)

    # 3) Higher-high/higher-low (or lower-high/lower-low) swing structure
    # over a 20-bar window, comparing to the same window 10 bars earlier.
    roll_max = close.rolling(20, min_periods=20).max()
    roll_min = close.rolling(20, min_periods=20).min()
    prior_max = roll_max.shift(10)
    prior_min = roll_min.shift(10)
    v3 = pd.Series("sideways", index=close.index, dtype=object)
    v3[(roll_max > prior_max) & (roll_min > prior_min)] = "uptrend"
    v3[(roll_max < prior_max) & (roll_min < prior_min)] = "downtrend"
    votes.append(v3)

    return votes


def _majority_label(row: pd.Series) -> str:
    counts: dict = {}
    for v in row:
        counts[v] = counts.get(v, 0) + 1
    top = max(counts.values())
    winners = [v for v, c in counts.items() if c == top]
    return winners[0] if len(winners) == 1 else "sideways"


@dataclass
class MarketContext:
    """Precomputed, per-bar market context shared across all candlestick
    pattern detectors (br/mw/ml) and the confidence engine in
    analysis/candlestick_engine.py."""

    atr: pd.Series
    atr_pct: pd.Series
    location: pd.Series
    volatility_regime: pd.Series
    volume_z: Optional[pd.Series]
    trend_mw: pd.Series          # per-bar majority trend label
    _trend_agreement: pd.Series  # per-bar fraction of the 3 models agreeing

    def trend_label(self, i: int) -> str:
        """Trend label at bar `i` — one of uptrend/downtrend/sideways."""
        return self.trend_mw.iloc[i]

    def trend_agreement(self, i: int) -> float:
        """Fraction (0-1) of the independent trend models that agree with
        `trend_label(i)` at bar `i` — a crude confidence-in-trend metric."""
        return float(self._trend_agreement.iloc[i])


def build_context(
    df: pd.DataFrame,
    *,
    atr_period: int = 14,
    volatility_lookback: int = 100,
    location_lookback: int = 20,
    volume_lookback: int = 20,
) -> MarketContext:
    """Build the shared MarketContext for a full DataFrame, once.

    Parameters mirror analysis.candlestick_engine.EngineConfig's
    context-build knobs (atr_period, volatility_lookback,
    location_lookback, volume_lookback).
    """
    close = df["close"].astype(float)

    atr_series = atr(df, period=atr_period)
    atr_pct = (atr_series / close).replace([np.inf, -np.inf], np.nan)

    # Location-in-range: 0 = at the recent range low, 1 = at the range
    # high. NaN for the first `location_lookback` bars (insufficient
    # history) — callers already handle NaN here via pd.isna() checks.
    roll_high = df["high"].astype(float).rolling(location_lookback, min_periods=location_lookback).max()
    roll_low = df["low"].astype(float).rolling(location_lookback, min_periods=location_lookback).min()
    span = (roll_high - roll_low).replace(0, np.nan)
    location = ((close - roll_low) / span).clip(0.0, 1.0)

    # Volatility regime: percentile rank of atr_pct within a rolling
    # lookback window, bucketed into low/normal/high. Bars without enough
    # history for a full lookback window are "unknown" rather than
    # guessing from a partial, less representative sample.
    vol_rank = atr_pct.rolling(volatility_lookback, min_periods=volatility_lookback).rank(pct=True)
    volatility_regime = pd.Series("unknown", index=close.index, dtype=object)
    volatility_regime[vol_rank <= 0.25] = "low"
    volatility_regime[(vol_rank > 0.25) & (vol_rank < 0.75)] = "normal"
    volatility_regime[vol_rank >= 0.75] = "high"

    # Volume z-score — optional, since not every feed has real volume
    # (MT5 forex feeds usually only have tick_volume, a proxy).
    volume_z = None
    vol_col = "volume" if "volume" in df.columns else ("tick_volume" if "tick_volume" in df.columns else None)
    if vol_col is not None:
        vol = df[vol_col].astype(float)
        roll_mean = vol.rolling(volume_lookback, min_periods=volume_lookback).mean()
        roll_std = vol.rolling(volume_lookback, min_periods=volume_lookback).std().replace(0, np.nan)
        volume_z = (vol - roll_mean) / roll_std

    votes = _trend_votes(close)
    votes_df = pd.concat(votes, axis=1)
    trend_mw = votes_df.apply(_majority_label, axis=1)
    trend_mw.name = "trend_mw"
    agreement = votes_df.eq(trend_mw, axis=0).sum(axis=1) / len(votes)

    return MarketContext(
        atr=atr_series,
        atr_pct=atr_pct,
        location=location,
        volatility_regime=volatility_regime,
        volume_z=volume_z,
        trend_mw=trend_mw,
        _trend_agreement=agreement,
    )
