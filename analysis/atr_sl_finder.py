# analysis/atr_sl_finder.py — ATR Stop-Loss Finder (ported from MQL5)
# =============================================================================
# Ported from: https://github.com/geraked/metatrader5/blob/master/Indicators/AtrSlFinder.mq5
# Original author: Geraked (Rabist) — MIT license
#
# Computes suggested stop-loss levels above the high and below the low,
# based on a smoothed True Range × multiplier.
#
# Algorithm (faithful to the MQL5 source):
#   TR[i]  = max(high[i] - low[i],
#                |high[i] - close[i-1]|,
#                |low[i]  - close[i-1]|)
#   MA[i]  = mean(TR[i .. i+Length-1])     # forward-looking SMA in MQL5
#   Upper[i] = MA[i] * Multiplier + high[i]
#   Lower[i] = low[i] - MA[i] * Multiplier
#
# Note: MQL5 source uses ArraySetAsSeries(..., true), so the SMA window
# looks FORWARD (toward older bars in MQL5's newest-first layout, which
# in our oldest-first layout means NEWER bars). This makes the original
# indicator NON-causal — it can repaint.
#
# UPDATED (institutional review): `compute()` now defaults to
# `causal=True` (previous bars only — safe for live trading). The
# original repainting behavior is still available via `causal=False`
# for offline/backtest-chart parity with the MQL5 source, but is no
# longer the default — a repainting indicator should never be what a
# live call site gets without explicitly asking for it. This is a
# breaking change from the prior default; see `compute()`'s docstring.
#
# Output columns:
#   atr_sl_upper : suggested upper SL (above high)
#   atr_sl_lower : suggested lower SL (below low)
#   atr_sl_ma    : the smoothed TR value (for debugging)
# =============================================================================

from __future__ import annotations

import numpy as np
import pandas as pd


def compute(
    df: pd.DataFrame,
    *,
    length: int = 14,
    multiplier: float = 1.5,
    causal: bool = True,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.DataFrame:
    """
    Compute the ATR Stop-Loss Finder indicator.

    Parameters
    ----------
    df : DataFrame with high, low, close columns.
    length : TR smoothing window (default 14 — MQL5 default).
    multiplier : band multiplier (default 1.5 — MQL5 default).
    causal : if True (**default, changed from the original False**), uses
        the previous `length` bars only — causal, safe for live/production
        use. If False, faithfully reproduces the original MQL5 behavior,
        where the SMA window looks FORWARD in time — non-causal, will
        repaint. Pass `causal=False` explicitly if you specifically need
        the original repainting behavior (e.g. to visually match the
        MQL5 indicator in a backtest chart) — never for live trading.

        BREAKING CHANGE (institutional review, see report): the previous
        default of `causal=False` meant any call site that didn't
        explicitly pass `causal=True` was silently getting stop-loss
        levels computed from bars that hadn't happened yet — safe for
        offline analysis, but a live-trading hazard if this ever ran
        against a live feed. Defaulting a repainting indicator on in a
        decision layer is a correctness risk that outweighs preserving the
        old default silently. Any existing call site that depends on the
        old (repainting) numbers must now pass `causal=False` explicitly.

    Returns
    -------
    Same DataFrame with `atr_sl_upper`, `atr_sl_lower`, `atr_sl_ma` columns
    added.
    """
    if length < 1:
        raise ValueError(f"length must be >= 1, got {length}")
    if multiplier <= 0:
        raise ValueError(f"multiplier must be > 0, got {multiplier}")

    out = df.copy()
    high = out[high_col].to_numpy(dtype=float)
    low = out[low_col].to_numpy(dtype=float)
    close = out[close_col].to_numpy(dtype=float)
    n = len(out)

    # True Range (single-bar, with previous close)
    tr = np.zeros(n, dtype=float)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    ma = np.full(n, np.nan, dtype=float)
    upper = np.full(n, np.nan, dtype=float)
    lower = np.full(n, np.nan, dtype=float)

    if causal:
        # Causal: use tr[i-length+1 .. i] (the previous `length` bars including current)
        for i in range(length - 1, n):
            ma[i] = float(np.mean(tr[i - length + 1: i + 1]))
            upper[i] = ma[i] * multiplier + high[i]
            lower[i] = low[i] - ma[i] * multiplier
    else:
        # Faithful to MQL5 (forward-looking in our oldest-first layout):
        # use tr[i .. i+length-1]
        for i in range(n):
            end = min(i + length, n)
            if end - i < 1:
                continue
            ma[i] = float(np.mean(tr[i:end]))
            upper[i] = ma[i] * multiplier + high[i]
            lower[i] = low[i] - ma[i] * multiplier

    out["atr_sl_upper"] = upper
    out["atr_sl_lower"] = lower
    out["atr_sl_ma"] = ma
    return out


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    t = np.arange(n)
    close = 1.0850 + 0.0030 * np.sin(t / 8.0)
    high = close + 0.0005
    low = close - 0.0005
    df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=idx)

    # Faithful (forward-looking, MQL5 default)
    out_faithful = compute(df, length=14, multiplier=1.5, causal=False)
    print(f"Faithful: NaN rows = {out_faithful['atr_sl_upper'].isna().sum()} (expect 0)")

    # Causal
    out_causal = compute(df, length=14, multiplier=1.5, causal=True)
    print(f"Causal:   NaN rows = {out_causal['atr_sl_upper'].isna().sum()} (expect 13 warmup)")

    # ── Regression check: default must now be causal (institutional review) ──
    out_default = compute(df, length=14, multiplier=1.5)
    assert out_default["atr_sl_upper"].equals(out_causal["atr_sl_upper"]), (
        "REGRESSION: compute()'s default should now match causal=True, not the "
        "old repainting causal=False behavior"
    )
    print("Default-causal regression check passed (compute(df) now defaults to causal=True).")

    # Sanity: lower < low < high < upper always
    valid = out_causal.dropna()
    assert (valid["atr_sl_lower"] < valid["low"]).all(), "lower must be below low"
    assert (valid["atr_sl_upper"] > valid["high"]).all(), "upper must be above high"
    print("AtrSlFinder smoke test passed.")