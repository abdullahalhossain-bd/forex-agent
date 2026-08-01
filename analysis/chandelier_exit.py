# analysis/chandelier_exit.py — Chandelier Exit (ported from MQL5)
# =============================================================================
# Ported from: https://github.com/geraked/metatrader5/blob/master/Indicators/ChandelierExit.mq5
# Original author: Geraked (Rabist) — MIT license
#
# The Chandelier Exit is a volatility-based trailing exit. It uses
# Heikin-Ashi-smoothed close + ATR (computed on Heikin-Ashi candles)
# to set a long stop and short stop. Direction flips when HA close
# crosses the trailing stop.
#
# Algorithm (faithful to the MQL5 source):
#   1. Compute Heikin-Ashi candles (HA_O, HA_H, HA_L, HA_C) from raw OHLC.
#   2. Compute ATR on the HA candles (Wilder's RMA over `atr_period`).
#   3. For each bar i:
#        longStop1  = max(HA_C[i-P+1..i]) - atr1
#        longStop2  = max(HA_C[i-1-P+1..i-1]) - atr2
#        longStop   = HA_C[i-1] > longStop2 ? max(longStop1, longStop2) : longStop1
#        (same shape for shortStop, using min and +)
#        Dir[i] = HA_C[i] > shortStop2 ? 1 :
#                 HA_C[i] < longStop2  ? -1 :
#                 Dir[i-1]
#        BuySignal  = (Dir[i] == 1 && Dir[i-1] == -1) ? longStop  : 0
#        SellSignal = (Dir[i] == -1 && Dir[i-1] == 1) ? shortStop : 0
#
# Output columns:
#   ce_long_stop   : the long trailing stop value
#   ce_short_stop  : the short trailing stop value
#   ce_dir         : +1 (long) / -1 (short) / 0 warmup
#   ce_buy_signal  : non-zero on buy-flip bars (= longStop value)
#   ce_sell_signal : non-zero on sell-flip bars (= shortStop value)
# =============================================================================

from __future__ import annotations

import numpy as np
import pandas as pd


def _heikin_ashi(op, hi, lo, cl):
    """Standard Heikin-Ashi smoothing. Returns (ha_o, ha_h, ha_l, ha_c)."""
    n = len(cl)
    ha_o = np.empty(n, dtype=float)
    ha_h = np.empty(n, dtype=float)
    ha_l = np.empty(n, dtype=float)
    ha_c = np.empty(n, dtype=float)
    ha_o[0] = op[0]
    ha_c[0] = cl[0]
    ha_h[0] = hi[0]
    ha_l[0] = lo[0]
    for i in range(1, n):
        ha_o[i] = (ha_o[i - 1] + ha_c[i - 1]) / 2.0
        ha_c[i] = (op[i] + hi[i] + lo[i] + cl[i]) / 4.0
        ha_h[i] = max(hi[i], ha_o[i], ha_c[i])
        ha_l[i] = min(lo[i], ha_o[i], ha_c[i])
    return ha_o, ha_h, ha_l, ha_c


def _atr_rma(values, period):
    """Wilder's RMA (= EMA with alpha = 1/period) on a 1-D array."""
    n = len(values)
    out = np.full(n, np.nan, dtype=float)
    if n < period:
        return out
    out[period - 1] = np.mean(values[:period])
    alpha = 1.0 / period
    for i in range(period, n):
        out[i] = out[i - 1] + alpha * (values[i] - out[i - 1])
    return out


def compute(
    df: pd.DataFrame,
    *,
    atr_period: int = 1,
    atr_mult: float = 0.75,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.DataFrame:
    """
    Compute the Chandelier Exit indicator.

    Parameters
    ----------
    df : DataFrame with open, high, low, close columns.
    atr_period : ATR period on HA candles (default 1 — MQL5 default).
    atr_mult : ATR multiplier (default 0.75 — MQL5 default).

    Returns
    -------
    Same DataFrame with `ce_long_stop`, `ce_short_stop`, `ce_dir`,
    `ce_buy_signal`, `ce_sell_signal` columns added.
    """
    if atr_period < 1:
        raise ValueError(f"atr_period must be >= 1, got {atr_period}")

    out = df.copy()
    op = out[open_col].to_numpy(dtype=float)
    hi = out[high_col].to_numpy(dtype=float)
    lo = out[low_col].to_numpy(dtype=float)
    cl = out[close_col].to_numpy(dtype=float)
    n = len(out)

    # Heikin-Ashi
    ha_o, ha_h, ha_l, ha_c = _heikin_ashi(op, hi, lo, cl)

    # ATR on HA candles: TR[i] = max(HA_H[i], HA_C[i-1]) - min(HA_L[i], HA_C[i-1])
    tr = np.zeros(n, dtype=float)
    for i in range(1, n):
        tr[i] = max(ha_h[i], ha_c[i - 1]) - min(ha_l[i], ha_c[i - 1])
    atr = _atr_rma(tr, atr_period)

    long_stop = np.full(n, np.nan, dtype=float)
    short_stop = np.full(n, np.nan, dtype=float)
    direction = np.zeros(n, dtype=int)
    buy_signal = np.zeros(n, dtype=float)
    sell_signal = np.zeros(n, dtype=float)

    start = max(2 + atr_period - 1, atr_period + 1)
    if start >= n:
        out["ce_long_stop"] = np.nan
        out["ce_short_stop"] = np.nan
        out["ce_dir"] = 0
        out["ce_buy_signal"] = 0.0
        out["ce_sell_signal"] = 0.0
        return out

    # Initialize direction at start
    direction[start - 1] = 1

    for i in range(start, n):
        if np.isnan(atr[i]) or np.isnan(atr[i - 1]):
            direction[i] = direction[i - 1]
            continue
        atr1 = atr[i] * atr_mult
        atr2 = atr[i - 1] * atr_mult

        # Long stop: based on max of HA_C over last atr_period bars
        win1 = ha_c[i - atr_period + 1: i + 1] if atr_period > 1 else ha_c[i:i + 1]
        win2 = ha_c[i - 1 - atr_period + 1: i] if atr_period > 1 else ha_c[i - 1:i]
        long_stop1 = float(np.max(win1)) - atr1
        long_stop2 = float(np.max(win2)) - atr2
        long_stop[i] = max(long_stop1, long_stop2) if ha_c[i - 1] > long_stop2 else long_stop1

        # Short stop: based on min of HA_C over last atr_period bars
        short_stop1 = float(np.min(win1)) + atr1
        short_stop2 = float(np.min(win2)) + atr2
        short_stop[i] = min(short_stop1, short_stop2) if ha_c[i - 1] < short_stop2 else short_stop1

        # Direction
        if ha_c[i] > short_stop2:
            direction[i] = 1
        elif ha_c[i] < long_stop2:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

        # Signals on flip
        if direction[i] == 1 and direction[i - 1] == -1:
            buy_signal[i] = long_stop[i]
        elif direction[i] == -1 and direction[i - 1] == 1:
            sell_signal[i] = short_stop[i]

    # Warmup NaNs
    long_stop[:start] = np.nan
    short_stop[:start] = np.nan
    direction[:start] = 0

    out["ce_long_stop"] = long_stop
    out["ce_short_stop"] = short_stop
    out["ce_dir"] = direction
    out["ce_buy_signal"] = buy_signal
    out["ce_sell_signal"] = sell_signal
    return out


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    t = np.arange(n)
    # Sharp reversals (square wave) to force direction flips
    sig = np.where(np.sin(t / 20.0) > 0, 1.0900, 1.0800)
    # Add small noise so ATR isn't zero
    close = sig + np.random.normal(0, 0.0002, n)
    high = close + 0.0005
    low = close - 0.0005
    open_ = close - 0.0001
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)

    out = compute(df, atr_period=1, atr_mult=0.75)
    print(f"Rows: {len(out)}")
    print(f"NaN warmup: {out['ce_long_stop'].isna().sum()}")
    print(f"Buy signals:  {int((out['ce_buy_signal'] != 0).sum())}")
    print(f"Sell signals: {int((out['ce_sell_signal'] != 0).sum())}")
    print(f"Direction distribution: long={int((out['ce_dir'] == 1).sum())}, "
          f"short={int((out['ce_dir'] == -1).sum())}, warmup={int((out['ce_dir'] == 0).sum())}")
    assert (out["ce_dir"] != 0).any(), "expected some non-warmup bars"
    # Square wave should produce at least one flip
    assert ((out["ce_buy_signal"] != 0) | (out["ce_sell_signal"] != 0)).any(), \
        "expected at least one signal flip on square-wave data"
    print("Chandelier Exit smoke test passed.")
