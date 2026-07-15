# analysis/utbot_alerts.py — UT Bot Alerts indicator (ported from MQL5)
# =============================================================================
# Ported from: https://github.com/geraked/metatrader5/blob/master/Indicators/UTBot.mq5
# Original author: Geraked (Rabist) — MIT license
#
# UT Bot is an ATR-trailing-stop breakout signal. It computes a trailing
# "key value" (C1) that follows the close; when price closes above C1
# after being below, it fires a BULL arrow. When price closes below C1
# after being above, it fires a BEAR arrow.
#
# Algorithm (faithful to the MQL5 source, but oldest-first instead of
# the MQL5 newest-first ArraySetAsSeries):
#
#   loss = ATR[i] * atr_coef
#   if close[i] > C1[i-1]:
#       C1[i] = max(C1[i-1], close[i] - loss)
#   elif close[i] < C1[i-1]:
#       C1[i] = min(C1[i-1], close[i] + loss)
#   else:
#       C1[i] = C1[i-1]
#
#   # Arrow fires on cross:
#   if close[i] > C1[i] and close[i-1] <= C1[i-1]:
#       bull_arrow[i] = low[i] - abs(high[i-1] - low[i-1])   # below the bar
#   if close[i] < C1[i] and close[i-1] >= C1[i-1]:
#       bear_arrow[i] = high[i] + abs(high[i-1] - low[i-1])  # above the bar
#
# Output columns:
#   ut_trail       : the C1 trailing value (NaN during warmup)
#   ut_bull_arrow  : non-zero on bull-signal bars (plotted below low)
#   ut_bear_arrow  : non-zero on bear-signal bars (plotted above high)
#   ut_signal      : +1 / -1 / 0  (handy for downstream rules)
# =============================================================================

from __future__ import annotations

import numpy as np
import pandas as pd


def compute(
    df: pd.DataFrame,
    *,
    atr_coef: float = 2.0,
    atr_len: int = 1,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.DataFrame:
    """
    Compute the UT Bot Alerts indicator.

    Parameters
    ----------
    df : DataFrame with high, low, close columns.
    atr_coef : ATR sensitivity multiplier (default 2.0 — MQL5 default).
    atr_len : ATR period (default 1 — MQL5 default).

    Returns
    -------
    Same DataFrame with `ut_trail`, `ut_bull_arrow`, `ut_bear_arrow`,
    `ut_signal` columns added.
    """
    if atr_len < 1:
        raise ValueError(f"atr_len must be >= 1, got {atr_len}")

    out = df.copy()
    high = out[high_col].to_numpy(dtype=float)
    low = out[low_col].to_numpy(dtype=float)
    close = out[close_col].to_numpy(dtype=float)
    n = len(out)

    # ── True Range + Wilder ATR (same as SuperTrend) ──────────────────────────
    tr = np.empty(n, dtype=float)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    atr = np.full(n, np.nan, dtype=float)
    if n >= atr_len:
        atr[atr_len - 1] = tr[:atr_len].mean()
        alpha = 1.0 / max(atr_len, 1)
        for i in range(atr_len, n):
            atr[i] = atr[i - 1] + alpha * (tr[i] - atr[i - 1])

    # ── UT Bot core loop ──────────────────────────────────────────────────────
    # C1 = the trailing key value. Initialized to close[0] - ATR[0]*coef
    # (matches MQL5 behavior when C1[0] is undefined; MQL5 seeds it to 0 but
    # the first computed value overrides immediately).
    c1 = np.full(n, np.nan, dtype=float)
    bull = np.zeros(n, dtype=float)
    bear = np.zeros(n, dtype=float)
    signal = np.zeros(n, dtype=int)

    # Need at least atr_len+1 bars to start computing
    start = atr_len
    if start >= n:
        out["ut_trail"] = np.nan
        out["ut_bull_arrow"] = 0.0
        out["ut_bear_arrow"] = 0.0
        out["ut_signal"] = 0
        return out

    # Seed C1 at `start`
    c1[start] = close[start] - atr[start] * atr_coef if not np.isnan(atr[start]) else close[start]

    for i in range(start + 1, n):
        if np.isnan(atr[i]):
            c1[i] = c1[i - 1]
            continue
        loss = atr[i] * atr_coef
        # MQL5 logic (translated from the original 3-line conditional):
        #   t1 = close > C1_prev ? close - loss : close + loss
        #   t2 = close < C1_prev AND close_prev < C1_prev ? min(C1_prev, close+loss) : t1
        #   C1  = close > C1_prev AND close_prev > C1_prev ? max(C1_prev, close-loss) : t2
        if close[i] > c1[i - 1] and close[i - 1] > c1[i - 1]:
            c1[i] = max(c1[i - 1], close[i] - loss)
        elif close[i] < c1[i - 1] and close[i - 1] < c1[i - 1]:
            c1[i] = min(c1[i - 1], close[i] + loss)
        elif close[i] > c1[i - 1]:
            c1[i] = close[i] - loss
        else:
            c1[i] = close[i] + loss

        # Arrow detection: cross of close over/under C1
        h = abs(high[i - 1] - low[i - 1])
        if close[i] > c1[i] and close[i - 1] <= c1[i - 1]:
            bull[i] = low[i] - h
            signal[i] = 1
        elif close[i] < c1[i] and close[i - 1] >= c1[i - 1]:
            bear[i] = high[i] + h
            signal[i] = -1

    out["ut_trail"] = c1
    out["ut_bull_arrow"] = bull
    out["ut_bear_arrow"] = bear
    out["ut_signal"] = signal
    return out


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    # A sine oscillation around 1.0850 with some trend — should produce arrows
    t = np.arange(n)
    close = 1.0850 + 0.0050 * np.sin(t / 10.0) + 0.00002 * t
    high = close + 0.0005
    low = close - 0.0005
    df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=idx)

    out = compute(df, atr_coef=2.0, atr_len=1)
    print(f"Rows: {len(out)}")
    print(f"NaN warmup rows in ut_trail: {out['ut_trail'].isna().sum()}")
    print(f"Bull arrows: {(out['ut_bull_arrow'] != 0).sum()}")
    print(f"Bear arrows: {(out['ut_bear_arrow'] != 0).sum()}")
    print(f"Signals: bull={int((out['ut_signal'] == 1).sum())}, "
          f"bear={int((out['ut_signal'] == -1).sum())}")
    assert (out["ut_signal"] == 1).any() or (out["ut_signal"] == -1).any(), \
        "expected at least one signal in oscillating series"
    print("UT Bot Alerts smoke test passed.")
