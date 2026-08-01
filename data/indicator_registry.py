"""
data/indicator_registry.py — Canonical indicator source of truth
=================================================================

Co-founder fix: CONSOLIDATE INDICATOR MODULES.

The repo has THREE indicator modules that produce overlapping columns
with different values:

  1. data/indicators.py        (Indicators class, `ta` library)
     → produces: rsi, macd, macd_signal, macd_hist, atr, bb_upper,
       bb_middle, bb_lower, bb_width, bb_pct, sma_20, sma_50, sma_200,
       ema_9, ema_21, trend, rsi_signal, macd_cross

  2. data/indicators_ext.py    (ExtendedIndicators class, `pandas-ta`)
     → produces 139 columns including: rsi, macd, atr, bb_upper, etc.
       (SAME NAMES, DIFFERENT VALUES than module 1)

  3. features/indicators_v5.py (ML-focused, `ta` library)
     → produces: rsi_14, macd, macd_signal, ema_20, ema_50, atr_14,
       bb_upper, bb_lower (slightly different NAMES)

The bug: when market_agent.py runs ExtendedIndicators.add_all() and a
downstream analysis module (e.g. smc_engine.py) imports Indicators and
calls add_rsi(), the second call OVERWRITES the first — silently
replacing pandas-ta's RSI with `ta`'s RSI. The two libraries compute
slightly different values (different smoothing, different NaN handling),
so any signal that depends on RSI gets inconsistent inputs.

This module provides a CANONICAL INDICATOR REGISTRY — a single
authoritative source for each indicator. Consumers should call:

    from data.indicator_registry import get_indicator
    rsi_value = get_indicator('rsi', df)  # always from the canonical source

Migration path:
  - New code: use get_indicator() / add_canonical_indicators()
  - Existing code: no change required (backward compatible)
  - Future: gradually migrate consumers from Indicators → registry

The canonical source for ALL overlapping indicators is ExtendedIndicators
(pandas-ta) because:
  1. It's the most comprehensive (139 columns vs 18)
  2. market_agent.py already prefers it (with fallback to Indicators)
  3. pandas-ta is actively maintained; `ta` library is less so
  4. The MT5 data pipeline already normalizes for pandas-ta
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# ── Canonical indicator name → source module mapping ─────────────────
# Each entry maps a "canonical name" to the column name produced by
# ExtendedIndicators (the authoritative source). If a column is NOT
# produced by ExtendedIndicators, it falls back to Indicators.
#
# This is the SINGLE SOURCE OF TRUTH for which module computes which
# indicator. Any module that wants the canonical value should query
# via get_indicator() instead of computing its own.
CANONICAL_SOURCES = {
    # Overlapping indicators — ExtendedIndicators (pandas-ta) wins
    "rsi":          "extended",  # RSI_14 in pandas-ta
    "macd":         "extended",  # MACD_12_26_9
    "macd_signal":  "extended",
    "macd_hist":    "extended",
    "atr":          "extended",  # ATRr_14
    "bb_upper":     "extended",  # BBB_20_2.0 + basis
    "bb_middle":    "extended",
    "bb_lower":     "extended",
    "bb_width":     "extended",
    "bb_pct":       "extended",
    "sma_20":       "extended",
    "sma_50":       "extended",
    "sma_200":      "extended",
    "ema_9":        "extended",
    "ema_21":       "extended",
    "ema_20":       "extended",
    "ema_50":       "extended",
    "vwap":         "extended",
    "obv":          "extended",
    "stoch_k":      "extended",
    "stoch_d":      "extended",
    "cci":          "extended",
    "adx":          "extended",
    # Indicators ONLY in Indicators (ta lib) — keep there
    "trend":          "legacy",
    "rsi_signal":     "legacy",
    "macd_cross":     "legacy",
    # Indicators ONLY in indicators_v5 — keep there (ML-only)
    "hour_sin":       "v5",
    "hour_cos":       "v5",
    "dow_sin":        "v5",
    "dow_cos":        "v5",
}


def add_canonical_indicators(df: pd.DataFrame, include_patterns: bool = False) -> pd.DataFrame:
    """Add ALL canonical indicators to a DataFrame.

    This is the PREFERRED entry point for new code. It uses
    ExtendedIndicators (pandas-ta) as the single source for all
    overlapping indicators, then adds the legacy `trend`/`rsi_signal`/
    `macd_cross` columns on top (which ExtendedIndicators doesn't
    produce).

    Args:
        df: OHLCV DataFrame with columns open/high/low/close/volume.
            A DatetimeIndex is preferred (required for VWAP) but not
            mandatory — ExtendedIndicators will fall back gracefully.
        include_patterns: if True, also add candlestick patterns.

    Returns:
        DataFrame with all canonical indicator columns added.

    ─── Column count: 139 vs 140 explained (P2 audit) ─────────────────
    ExtendedIndicators.add_all() produces **139** columns. This function
    then ADDS one derived column (`trend`, computed from close/sma_20/
    sma_50/sma_200 via `_trend_direction`) that ExtendedIndicators does
    NOT produce — bringing the canonical output to **140** columns.

    `rsi_signal` and `macd_cross` are also recomputed here, but they
    already exist in ExtendedIndicators' 139 (it writes them too), so
    those overwrites do NOT change the column count.

    If you see "[IndicatorsExt] 139 columns" followed by
    "[MarketAgent] canonical registry 140 columns" in the logs, this
    is EXPECTED and CORRECT — the +1 is the legacy `trend` column that
    the registry layer adds for backward compatibility with downstream
    consumers that still reference `df['trend']`.
    """
    from data.indicators_ext import ExtendedIndicators
    ind = ExtendedIndicators()
    df = ind.add_all(df, include_patterns=include_patterns)

    # Cross-file consistency fix (was: only filled these columns "if
    # not already present"). ExtendedIndicators.add_all() ALWAYS writes
    # its own 'rsi_signal' (and 'macd_cross') directly into df, so the
    # old "not in df.columns" guard meant this function's own canonical
    # computation never actually ran — whatever ExtendedIndicators wrote
    # silently won, even where its label scheme differed (e.g. it used
    # to emit "bullish"/"bearish" here instead of the canonical
    # "bullish_zone"/"bearish_zone"). This module is documented as the
    # single source of truth (see module docstring), so it must always
    # recompute these derived columns from the canonical values rather
    # than deferring to whatever a upstream module happened to set.
    if "rsi" in df.columns:
        df["rsi_signal"] = df["rsi"].apply(_rsi_zone)
    if "macd" in df.columns and "macd_signal" in df.columns:
        df["macd_cross"] = df.apply(
            lambda r: "bullish_cross" if r["macd"] > r["macd_signal"] else "bearish_cross",
            axis=1,
        )
    if all(c in df.columns for c in ("close", "sma_20", "sma_50", "sma_200")):
        df["trend"] = df.apply(_trend_direction, axis=1)

    log.info(
        f"[IndicatorRegistry] add_canonical_indicators: "
        f"{len(df.columns)} columns total "
        f"(ExtendedIndicators base 139 + legacy 'trend' derived = 140 expected)"
    )
    return df


def get_indicator(name: str, df: pd.DataFrame, row: int = -1) -> Optional[float | str]:
    """Get a single canonical indicator value from a DataFrame.

    Args:
        name: canonical indicator name (see CANONICAL_SOURCES keys).
        df: DataFrame that already has indicators added (via
            add_canonical_indicators or ExtendedIndicators.add_all).
        row: row index (-1 = last row, the most recent candle).

    Returns:
        The indicator value, or None if the column doesn't exist.
    """
    source = CANONICAL_SOURCES.get(name)
    if source is None:
        log.warning(f"[IndicatorRegistry] unknown indicator '{name}'")
        return None

    # All canonical sources write to the same column name (no suffix)
    # ExtendedIndicators renames pandas-ta's suffixed columns (RSI_14 → rsi)
    if name not in df.columns:
        log.debug(f"[IndicatorRegistry] column '{name}' not in df — call add_canonical_indicators() first")
        return None

    try:
        val = df.iloc[row][name]
        if pd.isna(val):
            return None
        return val
    except Exception:
        return None


def get_ai_context(df: pd.DataFrame) -> dict:
    """Build the AI context dict from canonical indicator values.

    Equivalent to Indicators.get_ai_context() + ExtendedIndicators.get_ai_context()
    combined, using the canonical source for each field.
    """
    if df is None or len(df) == 0:
        return {}
    last = df.iloc[-1]

    def _safe_float(col, default=0.0):
        try:
            v = last.get(col, default)
            return round(float(v), 5) if not pd.isna(v) else default
        except Exception:
            return default

    def _safe_str(col, default="unknown"):
        try:
            v = last.get(col, default)
            return str(v) if not pd.isna(v) else default
        except Exception:
            return default

    return {
        "price":      _safe_float("close"),
        "trend":      _safe_str("trend"),
        "rsi":        _safe_float("rsi"),
        "rsi_signal": _safe_str("rsi_signal"),
        "macd":       _safe_float("macd"),
        "macd_cross": _safe_str("macd_cross"),
        "atr":        _safe_float("atr"),
        "adx":        _safe_float("adx"),
        "bb_upper":   _safe_float("bb_upper"),
        "bb_lower":   _safe_float("bb_lower"),
        "bb_pct":     _safe_float("bb_pct"),
        "sma_20":     _safe_float("sma_20"),
        "sma_50":     _safe_float("sma_50"),
        "sma_200":    _safe_float("sma_200"),
        "ema_9":      _safe_float("ema_9"),
        "ema_21":     _safe_float("ema_21"),
        "stoch_k":    _safe_float("stoch_k"),
        "stoch_d":    _safe_float("stoch_d"),
        "cci":        _safe_float("cci"),
        "vwap":       _safe_float("vwap"),
    }


# ── Legacy helpers (ported from Indicators class for consistency) ─────

def _rsi_zone(x) -> str:
    if pd.isna(x):
        return "unknown"
    if x >= 70:
        return "overbought"
    if x <= 30:
        return "oversold"
    if x >= 55:
        return "bullish_zone"
    if x <= 45:
        return "bearish_zone"
    return "neutral"


def _trend_direction(row) -> str:
    try:
        p, s20, s50, s200 = row["close"], row["sma_20"], row["sma_50"], row["sma_200"]
        if p > s20 > s50 > s200:
            return "strong_bullish"
        elif p > s20 and s20 > s50:
            return "bullish"
        elif p < s20 < s50 < s200:
            return "strong_bearish"
        elif p < s20 and s20 < s50:
            return "bearish"
        else:
            return "sideways"
    except Exception:
        return "unknown"


# ── Smoke test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np
    # Build a small synthetic df
    n = 50
    dates = pd.date_range("2024-01-01", periods=n, freq="15min")
    close = 1.1 + np.cumsum(np.random.randn(n) * 0.001)
    df = pd.DataFrame({
        "open": close, "high": close + 0.0005, "low": close - 0.0005,
        "close": close, "volume": np.random.randint(100, 1000, n),
    }, index=dates)

    df = add_canonical_indicators(df)
    ctx = get_ai_context(df)
    print(f"Canonical indicators added. Columns: {len(df.columns)}")
    print(f"AI context: price={ctx['price']}, trend={ctx['trend']}, rsi={ctx['rsi']}")
    rsi_val = get_indicator("rsi", df)
    print(f"get_indicator('rsi'): {rsi_val}")
    assert rsi_val is not None or len(df) < 14  # RSI needs 14 bars
    print("Indicator registry smoke test passed.")