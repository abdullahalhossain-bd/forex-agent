# analysis/nadaraya_watson_envelope.py — Nadaraya-Watson Envelope (ported from MQL5)
# =============================================================================
# Ported from: https://github.com/geraked/metatrader5/blob/master/Indicators/NadarayaWatsonEnvelope.mq5
# Original author: Geraked (Rabist) — MIT license
#
# The Nadaraya-Watson Envelope (NWE) is a non-parametric regression
# envelope. It smooths the close price using a Gaussian kernel, then
# builds upper/lower bands at ±(mean absolute deviation × Multiplier).
#
# Algorithm (faithful to the MQL5 source):
#   For each bar i (most recent first in MQL5; we work oldest-first):
#     ws = Σ_{j=i..i+W-1} close[j] * gauss(j-i, BandWidth)
#     s  = Σ_{j=i..i+W-1} gauss(j-i, BandWidth)
#     reg[i] = ws / s                          # Nadaraya-Watson estimator
#     dev[i] = |close[i] - reg[i]|
#
#   For each bar i:
#     avg_dev = mean(dev[i..i+W-1])
#     band    = avg_dev * Multiplier
#     Upper[i] = reg[i] + band
#     Lower[i] = reg[i] - band
#
#   gauss(x, h) = exp(-(x² / (2h²)))
#
# Note: the MQL5 source uses `ArraySetAsSeries(close, true)`, meaning the
# "window" looks FORWARD in time (toward older bars). We replicate that
# exactly — `reg[i]` uses bars `[i, i+1, ..., i+W-1]` (older bars).
# This is a *centered-ish* smoother, NOT a causal filter. Use with care
# in live trading: the value at bar `i` is revised as new bars arrive.
#
# Output columns:
#   nwe_mid    : the Nadaraya-Watson regression value
#   nwe_upper  : upper band (mid + avg_dev * Multiplier)
#   nwe_lower  : lower band (mid - avg_dev * Multiplier)
#   nwe_pos    : +1 (close above upper), -1 (close below lower), 0 (inside)
#   nwe_stable : True if this bar had a full forward window (not subject to
#                repaint). False for the most recent `window_size` bars —
#                do not trade/alert off nwe_mid/upper/lower where this is
#                False, since those values will still change as new bars
#                arrive (see repainting note above).
# =============================================================================

from __future__ import annotations

import numpy as np
import pandas as pd


def _gauss(x: np.ndarray, h: float) -> np.ndarray:
    return np.exp(-(x * x) / (h * h * 2.0))


def compute(
    df: pd.DataFrame,
    *,
    band_width: float = 8.0,
    multiplier: float = 3.0,
    window_size: int = 500,
    close_col: str = "close",
) -> pd.DataFrame:
    """
    Compute the Nadaraya-Watson Envelope.

    Parameters
    ----------
    df : DataFrame with a close column.
    band_width : Gaussian bandwidth h (default 8.0 — MQL5 default).
    multiplier : band multiplier (default 3.0 — MQL5 default).
    window_size : rolling window W (default 500 — MQL5 default). Note:
        for the band to be defined, you need at least `window_size` bars
        AFTER the current bar. We cap the effective window at the
        remaining bars to avoid NaNs at the end of the series.

    Returns
    -------
    Same DataFrame with `nwe_mid`, `nwe_upper`, `nwe_lower`, `nwe_pos` added.
    """
    if band_width <= 0:
        raise ValueError(f"band_width must be > 0, got {band_width}")
    if multiplier <= 0:
        raise ValueError(f"multiplier must be > 0, got {multiplier}")
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")

    out = df.copy()
    close = out[close_col].to_numpy(dtype=float)
    n = len(out)

    reg = np.full(n, np.nan, dtype=float)
    dev = np.full(n, np.nan, dtype=float)
    upper = np.full(n, np.nan, dtype=float)
    lower = np.full(n, np.nan, dtype=float)

    # Precompute Gaussian weights for offsets 0..W-1
    # (We recompute per-bar to allow shorter window near the end.)
    for i in range(n):
        # Effective window: from i forward until i+W-1, capped at n-1
        end = min(i + window_size, n)
        k = np.arange(end - i)              # offsets 0, 1, ..., end-i-1
        g = _gauss(k.astype(float), band_width)
        ws = float(np.sum(close[i:end] * g))
        s = float(np.sum(g))
        if s > 0:
            reg[i] = ws / s
            dev[i] = abs(close[i] - reg[i])

    # Second pass: compute bands using mean dev over the next W bars
    for i in range(n):
        end = min(i + window_size, n)
        if end - i < 1:
            continue
        # Use only valid dev values (avoid NaN-contaminated mean)
        window_dev = dev[i:end]
        valid = window_dev[~np.isnan(window_dev)]
        if len(valid) == 0:
            continue
        avg_dev = float(np.mean(valid))
        band = avg_dev * multiplier
        upper[i] = reg[i] + band
        lower[i] = reg[i] - band

    # Position relative to bands
    pos = np.zeros(n, dtype=int)
    valid_idx = ~np.isnan(upper) & ~np.isnan(lower)
    pos[valid_idx & (close > upper)] = 1
    pos[valid_idx & (close < lower)] = -1

    # nwe_stable: True only for bars that had a FULL forward window
    # (i.e. end == i + window_size, not truncated near the end of the
    # series). As documented at the top of this file, this indicator is
    # non-causal — reg/upper/lower for the most recent `window_size` bars
    # are computed from a truncated (partial) window and WILL repaint as
    # new bars arrive. Live/decision code must not treat nwe_mid/nwe_upper/
    # nwe_lower on unstable bars as a fixed signal; only nwe_stable==True
    # rows are safe to backtest or alert on without repaint risk.
    stable = np.zeros(n, dtype=bool)
    stable[: max(0, n - window_size)] = True

    out["nwe_mid"] = reg
    out["nwe_upper"] = upper
    out["nwe_lower"] = lower
    out["nwe_pos"] = pos
    out["nwe_stable"] = stable
    return out


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    t = np.arange(n)
    # Sine + noise
    close = 1.0850 + 0.0050 * np.sin(t / 15.0) + np.random.normal(0, 0.0002, n)
    df = pd.DataFrame({"open": close, "high": close + 0.0003,
                       "low": close - 0.0003, "close": close}, index=idx)

    # Use a smaller window so we get full coverage on 300 bars
    out = compute(df, band_width=8.0, multiplier=3.0, window_size=50)
    print(f"Rows: {len(out)}")
    print(f"nwe_mid NaN: {out['nwe_mid'].isna().sum()}")
    print(f"nwe_upper NaN: {out['nwe_upper'].isna().sum()}")
    print(f"Above upper: {int((out['nwe_pos'] == 1).sum())}")
    print(f"Below lower: {int((out['nwe_pos'] == -1).sum())}")
    print(f"Inside band: {int((out['nwe_pos'] == 0).sum())}")
    # Sanity: mid should track close on average
    valid = ~out["nwe_mid"].isna()
    assert valid.any(), "expected at least some valid rows"
    corr = float(np.corrcoef(out.loc[valid, "nwe_mid"], out.loc[valid, "close"])[0, 1])
    print(f"Correlation (mid, close): {corr:.3f}")
    assert corr > 0.5, "expected mid to track close"
    print("Nadaraya-Watson Envelope smoke test passed.")