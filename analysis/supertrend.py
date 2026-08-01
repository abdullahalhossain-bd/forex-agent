# analysis/supertrend.py — SuperTrend indicator (ported from MQL5)
# =============================================================================
# Ported from: https://github.com/geraked/metatrader5/blob/master/Indicators/SuperTrend.mq5
# Original author: Geraked (Rabist) — MIT license
#
# SuperTrend is an ATR-based trend-following indicator. It plots a single
# line that flips above price (bearish) or below price (bullish) based on
# ATR-adjusted HL2 bands. Color changes mark trend flips.
#
# Algorithm (faithful to the MQL5 source):
#   middle[i]   = (high[i] + low[i]) / 2
#   up[i]       = middle[i] + multiplier * atr[i]
#   down[i]     = middle[i] - multiplier * atr[i]
#
#   trend[i] = +1  if close[i] > up[i-1]
#   trend[i] = -1  if close[i] < down[i-1]
#   trend[i] = trend[i-1]  otherwise
#
#   If trend flipped up this bar, reset down[i] to middle[i] - mult*atr[i].
#   If trend flipped down this bar, reset up[i] to middle[i] + mult*atr[i].
#   If trend is up   and down[i] < down[i-1]: down[i] = down[i-1]  (ratchet)
#   If trend is down and up[i]   > up[i-1]:   up[i]   = up[i-1]    (ratchet)
#
#   supertrend[i] = down[i] if trend[i] == +1 else up[i]
#
# Output columns:
#   supertrend : the SuperTrend line value
#   st_trend   : +1 (bull) / -1 (bear)
#   st_color   : 'green' / 'red' (for charting convenience)
# =============================================================================

from __future__ import annotations

import numpy as np
import pandas as pd


def compute(
    df: pd.DataFrame,
    *,
    period: int = 10,
    multiplier: float = 3.0,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.DataFrame:
    """
    Compute the SuperTrend indicator.

    Parameters
    ----------
    df : DataFrame with high, low, close columns (case-sensitive names
         configurable via *_col params).
    period : ATR period (default 10, matching MQL5 default `Periode = 10`).
    multiplier : ATR multiplier (default 3.0, matching MQL5 `Multiplier = 3`).

    Returns
    -------
    Same DataFrame with `supertrend`, `st_trend`, `st_color` columns added.
    Warmup rows (first `period` bars) are NaN.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    if multiplier <= 0:
        raise ValueError(f"multiplier must be > 0, got {multiplier}")

    out = df.copy()

    high = out[high_col].to_numpy(dtype=float)
    low = out[low_col].to_numpy(dtype=float)
    close = out[close_col].to_numpy(dtype=float)
    n = len(out)

    # ── True Range & Wilder's ATR ─────────────────────────────────────────────
    # MQL5 uses iATR() which is Wilder's smoothing (RMA).
    tr = np.empty(n, dtype=float)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    atr = np.full(n, np.nan, dtype=float)
    if n >= period:
        # Wilder's RMA = EMA with alpha = 1/period
        atr[period - 1] = tr[:period].mean()
        alpha = 1.0 / period
        for i in range(period, n):
            atr[i] = atr[i - 1] + alpha * (tr[i] - atr[i - 1])

    # ── SuperTrend core loop ──────────────────────────────────────────────────
    middle = (high + low) / 2.0
    up = middle + multiplier * atr
    down = middle - multiplier * atr

    trend = np.zeros(n, dtype=int)
    supertrend = np.full(n, np.nan, dtype=float)

    # Start at first bar where ATR is defined
    start = period
    if start >= n:
        out["supertrend"] = np.nan
        out["st_trend"] = 0
        out["st_color"] = None
        return out

    # Initialize trend at `start` based on close vs middle
    trend[start] = 1 if close[start] > middle[start] else -1
    supertrend[start] = down[start] if trend[start] == 1 else up[start]

    for i in range(start + 1, n):
        # Determine trend based on previous band
        if close[i] > up[i - 1]:
            trend[i] = 1
        elif close[i] < down[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]
            # Ratchet the band we're not using
            if trend[i] == 1 and down[i] < down[i - 1]:
                down[i] = down[i - 1]
            elif trend[i] == -1 and up[i] > up[i - 1]:
                up[i] = up[i - 1]

        # On trend flip, reset the new active band
        if trend[i] == 1 and trend[i - 1] == -1:
            # flipped to bull: reset down
            down[i] = middle[i] - multiplier * atr[i]
        elif trend[i] == -1 and trend[i - 1] == 1:
            # flipped to bear: reset up
            up[i] = middle[i] + multiplier * atr[i]

        # Ratchet the active band too
        if trend[i] == 1 and down[i] < down[i - 1]:
            down[i] = down[i - 1]
        elif trend[i] == -1 and up[i] > up[i - 1]:
            up[i] = up[i - 1]

        supertrend[i] = down[i] if trend[i] == 1 else up[i]

    out["supertrend"] = supertrend
    out["st_trend"] = trend
    out["st_color"] = np.where(trend == 1, "green", "red")
    # Warmup rows: color should be None
    out.loc[out["supertrend"].isna(), "st_color"] = None
    out.loc[out["supertrend"].isna(), "st_trend"] = 0

    return out


# ── Smoke test (run with: python -m analysis.supertrend) ─────────────────────
if __name__ == "__main__":
    # Synthetic: clear uptrend, then clear downtrend
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = np.concatenate([
        np.linspace(1.0850, 1.0950, 100),   # up
        np.linspace(1.0950, 1.0820, 100),   # down
    ])
    high = close + 0.0005
    low = close - 0.0005
    df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=idx)

    out = compute(df, period=10, multiplier=3.0)
    print(f"Rows: {len(out)}")
    print(f"NaN warmup rows: {out['supertrend'].isna().sum()}")
    print(f"First non-NaN at row: {out['supertrend'].first_valid_index()}")
    print(f"Unique trend values: {sorted(out['st_trend'].unique())}")
    print(f"Trend flips: {(out['st_trend'].diff().fillna(0) != 0).sum()}")
    # Expect at least one bull→bear flip in the synthetic series
    assert (out["st_trend"] == 1).any(), "expected at least one bull bar"
    assert (out["st_trend"] == -1).any(), "expected at least one bear bar"
    print("SuperTrend smoke test passed.")
