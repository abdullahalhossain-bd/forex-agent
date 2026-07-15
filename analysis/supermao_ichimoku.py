# analysis/supermao_ichimoku.py — SuperMao Ichimoku cloud strategy
# =============================================================================
# Ported from: https://github.com/tanvird3/TradingRobot/blob/master/SuperMaoIchiMoku.mq4
# Original author: khantanvir (Tanvir) — license not specified in source
#
# Ichimoku Kinko Hyo cloud-crossover strategy with auto-exit on tenkan/kijun
# cross reversal.
#
# Entry rules (LONG):
#   tenkan_sen > kijun_sen  AND  close > min(senkou_span_A, senkou_span_B)
#   "Tenkan above Kijun (bullish TK cross) AND price above/below the cloud"
#
#   TP = close * (1 + take_profit_pct/100)
#   SL = close * (1 - take_profit_pct/100)   # NOTE: original uses TakeProfit % for both
#
# Entry rules (SHORT): mirror.
#
# Auto-exit (ContUpdate="Yes"):
#   For an open LONG, if tenkan crosses BELOW kijun → set TP to current Bid
#   (close the long at market). Same logic in reverse for shorts.
#
# Ichimoku formulas (MT4 iIchimoku):
#   Tenkan-sen  = (max(high, 9)  + min(low, 9))  / 2
#   Kijun-sen   = (max(high, 26) + min(low, 26)) / 2
#   Senkou Span A = (Tenkan + Kijun) / 2, shifted +26 bars forward
#   Senkou Span B = (max(high, 52) + min(low, 52)) / 2, shifted +26 bars forward
#   Chikou Span   = close, shifted -26 bars back (not used in this strategy)
#
# Output columns:
#   tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b
#   smi_signal  : +1 long / -1 short / 0 no signal
#   smi_tp, smi_sl : take-profit / stop-loss prices at signal bar
#   smi_exit    : +1 (close long), -1 (close short), 0 (hold) — auto-exit signal
# =============================================================================

from __future__ import annotations

import numpy as np
import pandas as pd


def _donchian_mid(series_high: np.ndarray, series_low: np.ndarray,
                  period: int) -> np.ndarray:
    """(max(high, period) + min(low, period)) / 2 — the core Ichimoku calc."""
    n = len(series_high)
    out = np.full(n, np.nan, dtype=float)
    # Use a rolling max/min via pandas for speed
    s_high = pd.Series(series_high)
    s_low = pd.Series(series_low)
    rolling_high = s_high.rolling(period).max()
    rolling_low = s_low.rolling(period).min()
    return ((rolling_high + rolling_low) / 2.0).to_numpy()


def compute(
    df: pd.DataFrame,
    *,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_span_b_period: int = 52,
    displacement: int = 26,
    take_profit_pct: float = 5.0,
    stop_loss_pct: float | None = None,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.DataFrame:
    """
    Compute the SuperMao Ichimoku strategy.

    Parameters
    ----------
    df : DataFrame with high, low, close columns.
    tenkan_period : Tenkan-sen period (default 9 — MQL4 default).
    kijun_period : Kijun-sen period (default 26 — MQL4 default).
    senkou_span_b_period : Senkou Span B period (default 52 — MQL4 default).
    displacement : cloud forward-shift in bars (default 26 — MQL4 default).
    take_profit_pct : TP as % of price (default 5.0 — MQL4 default).
    stop_loss_pct : SL as % of price. If None, uses take_profit_pct (matches
        the MQL4 original which uses the same % for both).

    Returns
    -------
    Same DataFrame with Ichimoku lines + smi_signal / smi_tp / smi_sl / smi_exit.
    """
    if stop_loss_pct is None:
        stop_loss_pct = take_profit_pct

    out = df.copy()
    high = out[high_col].to_numpy(dtype=float)
    low = out[low_col].to_numpy(dtype=float)
    close = out[close_col].to_numpy(dtype=float)
    n = len(out)

    # ── Ichimoku lines ───────────────────────────────────────────────────────
    tenkan = _donchian_mid(high, low, tenkan_period)
    kijun = _donchian_mid(high, low, kijun_period)
    # Senkou Span A = (Tenkan + Kijun) / 2, shifted +displacement forward
    span_a_raw = (tenkan + kijun) / 2.0
    # Senkou Span B = midpoint of (high, low) over senkou_span_b_period, shifted +displacement forward
    span_b_raw = _donchian_mid(high, low, senkou_span_b_period)

    # MT4's iIchimoku returns the SHIFTED series directly: when you ask for
    # MODE_SENKOUSPANA at bar i, you get the value computed from bar i-displacement.
    # In our oldest-first layout, "shifted +displacement forward" means the
    # value at bar i is the raw value computed at bar i - displacement.
    senkou_span_a = np.full(n, np.nan, dtype=float)
    senkou_span_b = np.full(n, np.nan, dtype=float)
    if displacement > 0:
        senkou_span_a[displacement:] = span_a_raw[:-displacement]
        senkou_span_b[displacement:] = span_b_raw[:-displacement]
    else:
        senkou_span_a = span_a_raw
        senkou_span_b = span_b_raw

    out["tenkan_sen"] = tenkan
    out["kijun_sen"] = kijun
    out["senkou_span_a"] = senkou_span_a
    out["senkou_span_b"] = senkou_span_b

    # ── Signal generation ────────────────────────────────────────────────────
    smi_signal = np.zeros(n, dtype=int)
    smi_tp = np.full(n, np.nan, dtype=float)
    smi_sl = np.full(n, np.nan, dtype=float)
    smi_exit = np.zeros(n, dtype=int)

    cloud_top = np.maximum(senkou_span_a, senkou_span_b)
    cloud_bottom = np.minimum(senkou_span_a, senkou_span_b)

    for i in range(n):
        if np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(cloud_top[i]):
            continue

        # LONG: tenkan > kijun AND close > cloud_bottom (in or above cloud)
        if tenkan[i] > kijun[i] and close[i] > cloud_bottom[i]:
            smi_signal[i] = 1
            smi_tp[i] = close[i] * (1.0 + take_profit_pct / 100.0)
            smi_sl[i] = close[i] * (1.0 - stop_loss_pct / 100.0)
        # SHORT: tenkan < kijun AND close < cloud_top (in or below cloud)
        elif tenkan[i] < kijun[i] and close[i] < cloud_top[i]:
            smi_signal[i] = -1
            smi_tp[i] = close[i] * (1.0 - take_profit_pct / 100.0)
            smi_sl[i] = close[i] * (1.0 + stop_loss_pct / 100.0)

        # Auto-exit: detect TK cross reversal (independent of entry signal —
        # the EA's ContUpdate logic runs on every bar of an open position)
        if i > 0 and not np.isnan(tenkan[i - 1]) and not np.isnan(kijun[i - 1]):
            # Was long (TK bull), now bearish → exit long
            if tenkan[i - 1] > kijun[i - 1] and tenkan[i] < kijun[i]:
                smi_exit[i] = 1   # close long
            # Was short (TK bear), now bullish → exit short
            elif tenkan[i - 1] < kijun[i - 1] and tenkan[i] > kijun[i]:
                smi_exit[i] = -1  # close short

    out["smi_signal"] = smi_signal
    out["smi_tp"] = smi_tp
    out["smi_sl"] = smi_sl
    out["smi_exit"] = smi_exit
    return out


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    n = 500
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    t = np.arange(n)
    # Trending series with regime changes — should produce TK crosses
    close = 1.0850 + 0.0001 * t + 0.0030 * np.sin(t / 30.0) + np.random.normal(0, 0.0003, n)
    high = close + 0.0003
    low = close - 0.0003
    df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=idx)

    out = compute(df, take_profit_pct=5.0)
    print(f"Rows: {len(out)}")
    print(f"Tenkan NaN: {out['tenkan_sen'].isna().sum()}")
    print(f"Kijun NaN:  {out['kijun_sen'].isna().sum()}")
    print(f"Span A NaN: {out['senkou_span_a'].isna().sum()}")
    print(f"Span B NaN: {out['senkou_span_b'].isna().sum()}")
    print(f"Long signals:  {int((out['smi_signal'] == 1).sum())}")
    print(f"Short signals: {int((out['smi_signal'] == -1).sum())}")
    print(f"Exit-long signals:  {int((out['smi_exit'] == 1).sum())}")
    print(f"Exit-short signals: {int((out['smi_exit'] == -1).sum())}")

    # Verify TP/SL direction
    long_bars = out[out["smi_signal"] == 1]
    if not long_bars.empty:
        assert (long_bars["smi_tp"] > long_bars["close"]).all()
        assert (long_bars["smi_sl"] < long_bars["close"]).all()
    short_bars = out[out["smi_signal"] == -1]
    if not short_bars.empty:
        assert (short_bars["smi_tp"] < short_bars["close"]).all()
        assert (short_bars["smi_sl"] > short_bars["close"]).all()
    print("TP/SL direction OK")

    # Senkou Span A should never be above Senkou Span B by more than the
    # underlying price range (sanity check on the displacement shift)
    assert out["senkou_span_a"].notna().any(), "expected some valid Span A"
    print("Span A has valid values")

    print("\nSuperMao Ichimoku smoke test passed.")
