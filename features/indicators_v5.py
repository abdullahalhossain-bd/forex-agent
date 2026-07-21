# features/indicators_v5.py — 28-feature indicator set with cyclical time encoding
# =============================================================================
# Ported from: https://github.com/bruh7463/forex_bot/blob/master/features/indicators.py
# Original author: bruh7463 — educational project
#
# Comprehensive 28-feature indicator set organized into 8 categories:
#   - Trend (3): ema_fast_mid, adx, adx_diff
#   - Momentum (5): rsi, macd_hist, stoch_rsi, williams_r, cci
#   - Volatility (4): atr_pct, bb_position, bb_width, candle_range
#   - Price Action (3): body_ratio, upper_shadow, lower_shadow
#   - Lag Returns (4): return_1, return_3, return_5, return_10
#   - Swing Distance (2): dist_to_high, dist_to_low
#   - Time Features (4): hour_sin, hour_cos, dow_sin, dow_cos
#   - Z-Scores (3): rsi_z, cci_z, atr_pct_z
#
# The CYCLICAL TIME ENCODING (sin/cos for hour and day-of-week) is the key
# innovation — it lets ML models learn session effects (London open, NY close,
# etc.) without treating time as a linear integer (which would make "hour 23"
# and "hour 0" seem far apart when they're actually adjacent).
#
# Uses the `ta` library for standard indicators.
# =============================================================================

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("indicators_v5")

try:
    import ta
    _HAS_TA = True
except ImportError:
    _HAS_TA = False
    log.warning("`ta` library not installed. Install with: pip install ta")


FEATURE_COLS = [
    "ema_fast_mid", "adx", "adx_diff",
    "rsi", "macd_hist", "stoch_rsi", "williams_r", "cci",
    "atr_pct", "bb_position", "bb_width", "candle_range",
    "body_ratio", "upper_shadow", "lower_shadow",
    "return_1", "return_3", "return_5", "return_10",
    "dist_to_high", "dist_to_low",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "rsi_z", "cci_z", "atr_pct_z",
]


def add_indicators(
    df: pd.DataFrame,
    *,
    time_col: str = "time",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
    drop_nan: bool = True,
) -> pd.DataFrame:
    """
    Add 28 technical indicators to candle data.

    Parameters
    ----------
    df : DataFrame with OHLCV columns.
    time_col : name of the timestamp column (for cyclical time features).
        If the DataFrame has a DatetimeIndex, that's used instead.
    drop_nan : if True, drops rows with NaN/inf (warmup period).

    Returns
    -------
    DataFrame with original columns plus 28 indicator columns.
    """
    if not _HAS_TA:
        raise ImportError("`ta` library required. Install with: pip install ta")

    required = [open_col, high_col, low_col, close_col, volume_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    c, h, l, o = out[close_col], out[high_col], out[low_col], out[open_col]

    # === Trend (3) ===
    ema10 = ta.trend.EMAIndicator(close=c, window=10).ema_indicator()
    ema50 = ta.trend.EMAIndicator(close=c, window=50).ema_indicator()
    out["ema_fast_mid"] = (ema10 - ema50) / c
    out["ema_20"] = ta.trend.EMAIndicator(close=c, window=20).ema_indicator()
    out["ema_50"] = ema50

    adx_ind = ta.trend.ADXIndicator(high=h, low=l, close=c, window=14)
    out["adx"] = adx_ind.adx()
    out["adx_diff"] = adx_ind.adx_pos() - adx_ind.adx_neg()

    # === Momentum (5) ===
    out["rsi"] = ta.momentum.RSIIndicator(close=c, window=14).rsi()
    out["rsi_14"] = out["rsi"]

    macd_ind = ta.trend.MACD(close=c)
    out["macd_hist"] = macd_ind.macd_diff()
    out["macd"] = macd_ind.macd()
    out["macd_signal"] = macd_ind.macd_signal()

    out["stoch_rsi"] = ta.momentum.StochRSIIndicator(close=c, window=14).stochrsi_k()
    out["williams_r"] = ta.momentum.WilliamsRIndicator(
        high=h, low=l, close=c, lbp=14).williams_r()
    out["cci"] = ta.trend.CCIIndicator(high=h, low=l, close=c, window=20).cci()

    # === Volatility (4) ===
    atr = ta.volatility.AverageTrueRange(
        high=h, low=l, close=c, window=14).average_true_range()
    out["atr_pct"] = atr / c
    out["atr_14"] = atr

    bb = ta.volatility.BollingerBands(close=c, window=20, window_dev=2)
    out["bb_position"] = bb.bollinger_pband()
    out["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / c
    out["bb_upper"] = bb.bollinger_hband()
    out["bb_lower"] = bb.bollinger_lband()
    out["candle_range"] = (h - l) / c

    # === Price Action (3) ===
    hl = h - l + 1e-10
    out["body_ratio"] = abs(c - o) / hl
    out["upper_shadow"] = (h - pd.concat([c, o], axis=1).max(axis=1)) / hl
    out["lower_shadow"] = (pd.concat([c, o], axis=1).min(axis=1) - l) / hl

    # === Lag Returns (4) ===
    for lag in [1, 3, 5, 10]:
        out[f"return_{lag}"] = c.pct_change(lag)

    # === Swing Distance (2) ===
    roll_high = c.rolling(20).max()
    roll_low = c.rolling(20).min()
    out["dist_to_high"] = (roll_high - c) / (atr + 1e-10)
    out["dist_to_low"] = (c - roll_low) / (atr + 1e-10)

    # === Time Features (4) — CYCLICAL ENCODING ===
    if isinstance(out.index, pd.DatetimeIndex):
        ts = out.index
    elif time_col in out.columns:
        ts = pd.to_datetime(out[time_col])
    else:
        ts = None

    if ts is not None:
        hour = pd.Series(ts).dt.hour if not isinstance(ts, pd.DatetimeIndex) else pd.Series(ts.hour, index=out.index)
        dow = pd.Series(ts).dt.dayofweek if not isinstance(ts, pd.DatetimeIndex) else pd.Series(ts.dayofweek, index=out.index)
        out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    else:
        for col_name in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
            out[col_name] = 0.0

    # === Z-Scores (3) ===
    for col_name in ["rsi", "cci", "atr_pct"]:
        roll_mean = out[col_name].rolling(100).mean()
        roll_std = out[col_name].rolling(100).std()
        out[f"{col_name}_z"] = (out[col_name] - roll_mean) / (roll_std + 1e-10)

    if drop_nan:
        out = out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    return out


def get_feature_columns() -> list[str]:
    """Return the canonical list of 28 feature column names."""
    return FEATURE_COLS.copy()


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not _HAS_TA:
        print("`ta` library not installed. Install with: pip install ta")
    else:
        np.random.seed(42)
        n = 500
        dates = pd.date_range("2024-01-01", periods=n, freq="H", tz="UTC")
        base = 1.1000
        df = pd.DataFrame({
            "time": dates,
            "open": base + np.random.randn(n) * 0.001,
            "high": base + np.random.randn(n) * 0.001 + 0.0005,
            "low": base + np.random.randn(n) * 0.001 - 0.0005,
            "close": base + np.random.randn(n) * 0.001,
            "volume": np.random.randint(1000, 10000, n),
        })

        result = add_indicators(df)
        print(f"Input: {df.shape} → Output: {result.shape}")
        print(f"Rows dropped: {len(df) - len(result)}")

        missing = [f for f in FEATURE_COLS if f not in result.columns]
        if missing:
            print(f"MISSING: {missing}")
        else:
            print(f"✓ All {len(FEATURE_COLS)} features present")

        assert not result[FEATURE_COLS].isna().any().any()
        assert not np.isinf(result[FEATURE_COLS].values).any()
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
            assert result[col].between(-1, 1).all()

        print("\n28-feature indicators smoke test passed.")
