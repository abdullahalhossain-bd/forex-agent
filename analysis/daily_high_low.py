# analysis/daily_high_low.py — Daily High/Low levels (ported from MQL5)
# =============================================================================
# Ported from: https://github.com/geraked/metatrader5/blob/master/Indicators/DailyHighLow.mq5
# Original author: Geraked (Rabist) — MIT license
#
# Plots the previous (or current) day's high/low as horizontal levels on
# intraday charts. Useful for breakout strategies: a break above the
# previous daily high is a bullish signal; a break below the previous
# daily low is bearish.
#
# Algorithm (faithful to the MQL5 source):
#   For each intraday bar, find the bar's calendar day.
#   Look up the High/Low of the *previous* day (Previous=true, default)
#   or the *current* day (Previous=false).
#   Three price modes:
#     DHL_LOWHIGH    → use iHigh / iLow of that day
#     DHL_OPENCLOSE  → use max(close, open) / min(close, open)
#     DHL_CLOSECLOSE → use max(close) / min(close)
#
# Output columns:
#   dhl_high : daily high level (forward-filled across intraday bars)
#   dhl_low  : daily low level (forward-filled)
# =============================================================================

from __future__ import annotations

import numpy as np
import pandas as pd


def compute(
    df: pd.DataFrame,
    *,
    previous: bool = True,
    price_mode: str = "lowhigh",   # "lowhigh" | "openclose" | "closeclose"
    high_col: str = "high",
    low_col: str = "low",
    open_col: str = "open",
    close_col: str = "close",
) -> pd.DataFrame:
    """
    Compute daily high/low levels on an intraday DataFrame.

    Parameters
    ----------
    df : DataFrame with a tz-aware DatetimeIndex and OHLC columns.
    previous : if True (default, MQL5 default), use the PREVIOUS day's
        levels. If False, use the current day's levels (repaints).
    price_mode : one of "lowhigh", "openclose", "closeclose".

    Returns
    -------
    Same DataFrame with `dhl_high`, `dhl_low` columns added.
    """
    if price_mode not in ("lowhigh", "openclose", "closeclose"):
        raise ValueError(f"price_mode must be lowhigh/openclose/closeclose, got {price_mode!r}")

    out = df.copy()

    # Ensure we have a date column (calendar day)
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex")

    day = out.index.normalize()  # midnight of each bar's day
    out["_day"] = day

    # Build a per-day summary
    if price_mode == "lowhigh":
        daily = out.groupby("_day").agg(high=(high_col, "max"), low=(low_col, "min"))
    elif price_mode == "openclose":
        # max(max(close, open)) over the day, min similarly
        oc_high = out[[open_col, close_col]].max(axis=1)
        oc_low = out[[open_col, close_col]].min(axis=1)
        daily = pd.DataFrame({"high": oc_high, "low": oc_low}, index=out.index)
        daily = daily.groupby(day).agg(high=("high", "max"), low=("low", "min"))
        daily.index.name = "_day"
    else:  # closeclose
        daily = out.groupby("_day").agg(high=(close_col, "max"), low=(close_col, "min"))

    # For each bar, pick the level from `previous` day (or current day)
    out["dhl_high"] = np.nan
    out["dhl_low"] = np.nan
    for i, ts in enumerate(out.index):
        bar_day = day[i]
        target_day = bar_day - pd.Timedelta(days=1) if previous else bar_day
        if target_day in daily.index:
            out.iloc[i, out.columns.get_loc("dhl_high")] = daily.loc[target_day, "high"]
            out.iloc[i, out.columns.get_loc("dhl_low")] = daily.loc[target_day, "low"]

    out = out.drop(columns=["_day"])
    return out


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 3 days of hourly bars
    n = 72
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = 1.0850 + np.sin(np.arange(n) / 5.0) * 0.0010
    high = close + 0.0005
    low = close - 0.0005
    open_ = close
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)

    out = compute(df, previous=True, price_mode="lowhigh")
    print(f"Rows: {len(out)}")
    print(f"dhl_high NaN: {out['dhl_high'].isna().sum()} (first day has no previous)")
    print(f"dhl_low NaN:  {out['dhl_low'].isna().sum()}")
    # Day 1 should have NaN (no previous day); Day 2+ should have values
    day1 = out.loc["2024-01-01"]
    day2 = out.loc["2024-01-02"]
    assert day1["dhl_high"].isna().all(), "day 1 should be NaN (no previous)"
    assert day2["dhl_high"].notna().all(), "day 2 should have values from day 1"
    # All rows of day 2 should have the SAME level (previous day's high)
    assert day2["dhl_high"].nunique() == 1, "expected constant level across day 2"
    print(f"Day 2 dhl_high (constant): {day2['dhl_high'].iloc[0]:.5f}")
    print(f"Day 2 dhl_low  (constant): {day2['dhl_low'].iloc[0]:.5f}")
    print("DailyHighLow smoke test passed.")
