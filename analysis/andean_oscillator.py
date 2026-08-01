# analysis/andean_oscillator.py — Andean Oscillator (ported from MQL5)
# =============================================================================
# Ported from: https://github.com/geraked/metatrader5/blob/master/Indicators/AndeanOscillator.mq5
# Original author: Geraked (Rabist) — MIT license
#
# The Andean Oscillator uses the relationship between price and its
# variance to produce bull/bear/signal lines. It's volatility-aware:
# the bull line measures downside volatility, the bear line measures
# upside volatility.
#
# Algorithm (faithful to the MQL5 source):
#   alpha = 2 / (Length + 1)
#
#   # Up1/Up2 track the upper envelope of price and price²
#   t = max(close, open)
#   Up1[i] = max(t, Up1[i-1] - (Up1[i-1] - close) * alpha)
#   t = max(close², open²)
#   Up2[i] = max(t, Up2[i-1] - (Up2[i-1] - close²) * alpha)
#
#   # Dn1/Dn2 track the lower envelope
#   t = min(close, open)
#   Dn1[i] = min(t, Dn1[i-1] + (close - Dn1[i-1]) * alpha)
#   t = min(close², open²)
#   Dn2[i] = min(t, Dn2[i-1] + (close² - Dn2[i-1]) * alpha)
#
#   Bull[i] = sqrt(max(0, Dn2[i] - Dn1[i]²))   # downside vol
#   Bear[i] = sqrt(max(0, Up2[i] - Up1[i]²))   # upside vol
#   Signal  = EMA(max(Bull, Bear), SignalLength)
#
# Interpretation:
#   - Bull > Bear: bearish phase (downside vol dominates)
#   - Bear > Bull: bullish phase (upside vol dominates)
#   - Signal crossing Bull/Bear marks regime change
#
# Output columns:
#   ao_bull   : downside volatility (bull line in MQL5 naming)
#   ao_bear   : upside volatility   (bear line in MQL5 naming)
#   ao_signal : EMA-smoothed max(Bull, Bear)
#   ao_phase  : +1 (bear>bull = uptrend), -1 (bull>bear = downtrend), 0 warmup
# =============================================================================

from __future__ import annotations

import numpy as np
import pandas as pd


def compute(
    df: pd.DataFrame,
    *,
    length: int = 50,
    signal_length: int = 9,
    open_col: str = "open",
    close_col: str = "close",
) -> pd.DataFrame:
    """
    Compute the Andean Oscillator.

    Parameters
    ----------
    df : DataFrame with open, close columns.
    length : main EMA length (default 50 — MQL5 default).
    signal_length : signal line EMA length (default 9 — MQL5 default).

    Returns
    -------
    Same DataFrame with `ao_bull`, `ao_bear`, `ao_signal`, `ao_phase` added.
    Warmup: first `length` rows are NaN.
    """
    if length < 2:
        raise ValueError(f"length must be >= 2, got {length}")
    if signal_length < 1:
        raise ValueError(f"signal_length must be >= 1, got {signal_length}")

    out = df.copy()
    op = out[open_col].to_numpy(dtype=float)
    cl = out[close_col].to_numpy(dtype=float)
    n = len(out)

    up1 = np.zeros(n, dtype=float)
    up2 = np.zeros(n, dtype=float)
    dn1 = np.zeros(n, dtype=float)
    dn2 = np.zeros(n, dtype=float)
    bull = np.full(n, np.nan, dtype=float)
    bear = np.full(n, np.nan, dtype=float)
    signal = np.full(n, np.nan, dtype=float)

    alpha = 2.0 / (length + 1.0)

    # Seed: MQL5 initializes Up1/Up2/Dn1/Dn2 to 0, then replaces with close
    # on the first non-zero encounter. We seed to close[0] directly.
    if n == 0:
        out["ao_bull"] = np.nan
        out["ao_bear"] = np.nan
        out["ao_signal"] = np.nan
        out["ao_phase"] = 0
        return out

    up1[0] = cl[0]
    up2[0] = cl[0] * cl[0]
    dn1[0] = cl[0]
    dn2[0] = cl[0] * cl[0]

    for i in range(1, n):
        # Up1: max of (max(close,open)) and (Up1_prev - (Up1_prev - close)*alpha)
        t = max(cl[i], op[i])
        up1[i] = max(t, up1[i - 1] - (up1[i - 1] - cl[i]) * alpha)
        if up1[i] == 0:
            up1[i] = cl[i]

        # Up2: same shape on squared prices
        t = max(cl[i] * cl[i], op[i] * op[i])
        up2[i] = max(t, up2[i - 1] - (up2[i - 1] - cl[i] * cl[i]) * alpha)
        if up2[i] == 0:
            up2[i] = cl[i] * cl[i]

        # Dn1: min of (min(close,open)) and (Dn1_prev + (close - Dn1_prev)*alpha)
        t = min(cl[i], op[i])
        dn1[i] = min(t, dn1[i - 1] + (cl[i] - dn1[i - 1]) * alpha)
        if dn1[i] == 0:
            dn1[i] = cl[i]

        # Dn2: same shape on squared prices
        t = min(cl[i] * cl[i], op[i] * op[i])
        dn2[i] = min(t, dn2[i - 1] + (cl[i] * cl[i] - dn2[i - 1]) * alpha)
        if dn2[i] == 0:
            dn2[i] = cl[i] * cl[i]

        # Bull = sqrt(Dn2 - Dn1²)  (downside vol)
        # Bear = sqrt(Up2 - Up1²)  (upside vol)
        bull_val = dn2[i] - dn1[i] * dn1[i]
        bear_val = up2[i] - up1[i] * up1[i]
        bull[i] = np.sqrt(bull_val) if bull_val > 0 else 0.0
        bear[i] = np.sqrt(bear_val) if bear_val > 0 else 0.0

    # Signal = EMA of max(bull, bear) over signal_length
    # MQL5 starts the EMA at the first bar (rough init), then smooths.
    if n > 1:
        raw = np.where(np.isnan(bull) | np.isnan(bear), 0.0, np.maximum(bull, bear))
        smooth = 2.0 / (1.0 + signal_length)
        signal[0] = raw[0]
        for i in range(1, n):
            signal[i] = raw[i] * smooth + signal[i - 1] * (1.0 - smooth)

    # Mark warmup: MQL5 requires rates_total >= SignalLength+1 to compute
    warmup = min(length, n)
    bull[:warmup] = np.nan
    bear[:warmup] = np.nan
    signal[:warmup] = np.nan

    out["ao_bull"] = bull
    out["ao_bear"] = bear
    out["ao_signal"] = signal

    # Phase: +1 when bear > bull (uptrend), -1 when bull > bear (downtrend)
    phase = np.zeros(n, dtype=int)
    valid = ~np.isnan(bull) & ~np.isnan(bear)
    phase[valid & (bear > bull)] = 1
    phase[valid & (bull > bear)] = -1
    out["ao_phase"] = phase

    return out


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    t = np.arange(n)
    # Two regimes: volatile up, calm sideways
    close = np.concatenate([
        1.0850 + 0.0100 * np.sin(t[:150] / 8.0) + 0.00005 * t[:150],
        1.0900 + 0.0010 * np.sin(t[150:] / 5.0),
    ])
    open_ = close + np.random.uniform(-0.0003, 0.0003, n)
    df = pd.DataFrame({"open": open_, "high": close + 0.0005,
                       "low": close - 0.0005, "close": close}, index=idx)

    out = compute(df, length=50, signal_length=9)
    print(f"Rows: {len(out)}")
    print(f"Warmup NaN rows: {out['ao_bull'].isna().sum()}")
    print(f"Phase distribution: uptrend={int((out['ao_phase'] == 1).sum())}, "
          f"downtrend={int((out['ao_phase'] == -1).sum())}, "
          f"flat={int((out['ao_phase'] == 0).sum())}")
    assert out["ao_bull"].isna().sum() == 50, "expected 50 warmup rows"
    assert (out["ao_phase"] != 0).any(), "expected non-trivial phase classification"
    print("Andean Oscillator smoke test passed.")
