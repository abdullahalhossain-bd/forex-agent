# analysis/vw_macd.py — Volume-Weighted MACD (VW-MACD)
# =============================================================================
# Inspired by: https://github.com/smartedgetrading/SmartEdge-EA
# SmartEdge EA docs mention "VW-MACD (Volume-Weighted MACD)" as one of its
# signal filters. The source code is proprietary (server-side), but the
# algorithm concept is standard: replace the EMA smoothing in traditional
# MACD with volume-weighted EMAs (VWMA) so that high-volume bars have more
# influence on the MACD line.
#
# Traditional MACD:
#   macd_line   = EMA(close, fast) - EMA(close, slow)
#   signal_line = EMA(macd_line, signal_period)
#   histogram   = macd_line - signal_line
#
# Volume-Weighted MACD:
#   vwma_f      = VWMA(close, volume, fast)
#   vwma_s      = VWMA(close, volume, slow)
#   macd_line   = vwma_f - vwma_s
#   signal_line = EMA(macd_line, signal_period)   # signal stays non-weighted
#   histogram   = macd_line - signal_line
#
# VWMA formula:
#   vwma(n) = sum(close[i] * volume[i] for i in last n bars) / sum(volume[i])
#
# Interpretation:
#   - MACD line crossing above signal → bullish (stronger if high volume)
#   - MACD line crossing below signal → bearish
#   - Histogram > 0 and rising → bullish momentum increasing
#   - Histogram < 0 and falling → bearish momentum increasing
#
# The volume weighting makes the VW-MACD react more to high-volume moves
# and less to low-volume noise. This is especially useful for forex where
# tick volume is a proxy for real volume.
#
# Output columns:
#   vwmacd         : MACD line (volume-weighted)
#   vwmacd_signal  : signal line (non-weighted EMA of MACD line)
#   vwmacd_hist    : histogram (MACD - signal)
#   vwmacd_cross   : +1 (bullish cross), -1 (bearish cross), 0 (no cross)
# =============================================================================

from __future__ import annotations

import numpy as np
import pandas as pd


def _vwma(close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
    """
    Volume-Weighted Moving Average.
    vwma(n) = sum(close * volume) / sum(volume) over last n bars.
    """
    pv = close * volume
    return pv.rolling(period).sum() / volume.rolling(period).sum()


def compute(
    df: pd.DataFrame,
    *,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    close_col: str = "close",
    volume_col: str = "tick_volume",
) -> pd.DataFrame:
    """
    Compute the Volume-Weighted MACD (VW-MACD).

    Parameters
    ----------
    df : DataFrame with close and volume columns.
    fast_period : fast VWMA period (default 12 — MACD standard).
    slow_period : slow VWMA period (default 26 — MACD standard).
    signal_period : signal line EMA period (default 9 — MACD standard).
    close_col : name of the close price column.
    volume_col : name of the volume column. MT5 returns 'tick_volume' by
        default; some brokers also provide 'real_volume'. The function tries
        'real_volume' first, then 'tick_volume', then 'volume'.

    Returns
    -------
    Same DataFrame with `vwmacd`, `vwmacd_signal`, `vwmacd_hist`,
    `vwmacd_cross` columns added.
    """
    if fast_period >= slow_period:
        raise ValueError(f"fast_period ({fast_period}) must be < slow_period ({slow_period})")

    # Resolve volume column.
    # BUG FIX (institutional review): volume_col defaults to "tick_volume",
    # so the old candidate order [volume_col, "real_volume", "tick_volume",
    # "volume"] always matched on tick_volume first whenever the caller left
    # the default in place — "real_volume" was never reached even when both
    # columns were present, directly contradicting this function's own
    # docstring ("tries real_volume first, then tick_volume"). We now only
    # let an *explicitly overridden* volume_col jump the queue; the default
    # case follows the documented real_volume > tick_volume > volume order.
    candidates = (
        [volume_col, "real_volume", "tick_volume", "volume"]
        if volume_col != "tick_volume"
        else ["real_volume", "tick_volume", "volume"]
    )
    vol = None
    for candidate in candidates:
        if candidate in df.columns:
            vol = df[candidate]
            break
    if vol is None:
        raise ValueError(
            f"No volume column found. Tried: {candidates}. "
            f"Available columns: {list(df.columns)}"
        )

    # Replace zero volumes with a small epsilon to avoid division issues
    vol = vol.replace(0, np.nan)
    vol = vol.ffill().fillna(1.0)

    close = df[close_col]

    # Volume-weighted EMAs (using VWMA as the smoothing)
    vwma_fast = _vwma(close, vol, fast_period)
    vwma_slow = _vwma(close, vol, slow_period)

    # MACD line
    macd_line = vwma_fast - vwma_slow

    # Signal line: non-weighted EMA of MACD line (standard MACD convention)
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

    # Histogram
    histogram = macd_line - signal_line

    # Crossover signals
    # +1 when MACD crosses above signal (bullish)
    # -1 when MACD crosses below signal (bearish)
    prev_diff = (macd_line - signal_line).shift(1)
    curr_diff = macd_line - signal_line
    cross = np.where(
        (prev_diff <= 0) & (curr_diff > 0), 1,
        np.where(
            (prev_diff >= 0) & (curr_diff < 0), -1,
            0
        )
    )

    out = df.copy()
    out["vwmacd"] = macd_line
    out["vwmacd_signal"] = signal_line
    out["vwmacd_hist"] = histogram
    out["vwmacd_cross"] = cross
    return out


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(42)
    close = 1.0850 + np.cumsum(rng.normal(0, 0.0005, n))
    # Volume: random with some spikes
    volume = rng.integers(100, 1000, n).astype(float)
    # Inject volume spikes at bars 50, 100, 150
    volume[50] = 5000
    volume[100] = 5000
    volume[150] = 5000

    df = pd.DataFrame({
        "open": close, "high": close + 0.0003, "low": close - 0.0003,
        "close": close, "tick_volume": volume,
    }, index=idx)

    out = compute(df, fast_period=12, slow_period=26, signal_period=9)
    print(f"Rows: {len(out)}")
    print(f"VW-MACD NaN (warmup): {out['vwmacd'].isna().sum()} (expect {26-1}=25)")
    print(f"Bullish crosses: {int((out['vwmacd_cross'] == 1).sum())}")
    print(f"Bearish crosses: {int((out['vwmacd_cross'] == -1).sum())}")
    print(f"Histogram range: [{out['vwmacd_hist'].min():.6f}, {out['vwmacd_hist'].max():.6f}]")

    # Verify warmup (slow_period=26 → 25 NaN bars)
    assert out["vwmacd"].isna().sum() == 25, \
        f"expected 25 warmup NaNs, got {out['vwmacd'].isna().sum()}"

    # Verify cross values
    valid_crosses = set(out["vwmacd_cross"].unique())
    assert valid_crosses.issubset({-1, 0, 1}), f"invalid cross values: {valid_crosses}"

    # Verify histogram = MACD - signal
    valid = out.dropna(subset=["vwmacd"])
    assert np.allclose(valid["vwmacd_hist"], valid["vwmacd"] - valid["vwmacd_signal"],
                       atol=1e-10), "histogram should equal MACD - signal"

    # Test with real_volume column instead
    df2 = df.rename(columns={"tick_volume": "real_volume"})
    out2 = compute(df2)
    assert not out2["vwmacd"].isna().all(), "should work with real_volume too"

    # Test invalid params
    try:
        compute(df, fast_period=26, slow_period=12)
        assert False, "should have raised"
    except ValueError:
        pass

    print("\nVW-MACD smoke test passed.")