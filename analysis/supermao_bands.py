# analysis/supermao_bands.py — SuperMao multi-band Bollinger + MACD strategy
# =============================================================================
# Ported from: https://github.com/tanvird3/TradingRobot/blob/master/SuperMao.mq4
# Original author: khantanvir (Tanvir) — license not specified in source
#
# SuperMao is a mean-reversion strategy that uses THREE Bollinger Bands of
# different standard-deviation multipliers simultaneously:
#
#   - BB1 (1σ): take-profit target
#   - BB2 (2σ): outer band — price reaching this signals an extreme
#   - BB6 (6σ): catastrophic stop-loss (price rarely reaches this)
#
# Entry rules (LONG):
#   Ask >= LBBAND2  AND  Ask < UBBAND1  AND  MACD_main > MACD_signal
#   "Price is in the lower band zone (between lower-2σ and upper-1σ)
#    AND MACD is bullish → expect reversion to mean"
#
#   TP = UBBAND1     (reversion to upper 1σ band)
#   SL = LBBAND6     (catastrophic stop at lower 6σ band)
#
# Entry rules (SHORT): mirror.
#
# Optional: continuously update the TP/SL of an open position as the
# bands shift (ContUpdate flag in the original).
#
# Output columns added to the DataFrame:
#   sma, std, bb1_upper, bb1_lower, bb2_upper, bb2_lower, bb6_upper, bb6_lower,
#   macd_main, macd_signal,
#   sm_signal  : +1 (long), -1 (short), 0 (no signal)
#   sm_tp       : take-profit price for this bar
#   sm_sl       : stop-loss price for this bar
# =============================================================================

from __future__ import annotations

import numpy as np
import pandas as pd


def compute(
    df: pd.DataFrame,
    *,
    avg_period: int = 50,
    first_band: float = 1.0,
    second_band: float = 2.0,
    third_band: float = 6.0,
    macd_fast: int = 24,
    macd_slow: int = 52,
    macd_signal: int = 18,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
) -> pd.DataFrame:
    """
    Compute the SuperMao multi-band Bollinger + MACD signal.

    Default parameters match SuperMao.mq4 input defaults:
        MyAvgPeriod=50, FirstBBand=1, SecondBBand=2, ThirdBBand=6,
        MacdFast=24, MacdSlow=52, MacdSignal=18.
    """
    if avg_period < 2:
        raise ValueError(f"avg_period must be >= 2, got {avg_period}")
    if macd_slow <= macd_fast:
        raise ValueError(f"macd_slow ({macd_slow}) must be > macd_fast ({macd_fast})")

    out = df.copy()
    price = out[price_col].to_numpy(dtype=float)
    high = out[high_col].to_numpy(dtype=float)
    low = out[low_col].to_numpy(dtype=float)
    n = len(out)

    # ── Bollinger Bands (3 multipliers, same SMA + std window) ────────────────
    sma = pd.Series(price, index=out.index).rolling(avg_period).mean()
    roll_std = pd.Series(price, index=out.index).rolling(avg_period).std()

    out["sma"] = sma
    out["std"] = roll_std
    out["bb1_upper"] = sma + first_band * roll_std
    out["bb1_lower"] = sma - first_band * roll_std
    out["bb2_upper"] = sma + second_band * roll_std
    out["bb2_lower"] = sma - second_band * roll_std
    out["bb6_upper"] = sma + third_band * roll_std
    out["bb6_lower"] = sma - third_band * roll_std

    # ── MACD (EMA-based, faithful to MT4 iMACD) ───────────────────────────────
    # MT4's iMACD: main = EMA(fast) - EMA(slow); signal = EMA(main, signal_period)
    ema_fast = pd.Series(price, index=out.index).ewm(
        span=macd_fast, adjust=False).mean()
    ema_slow = pd.Series(price, index=out.index).ewm(
        span=macd_slow, adjust=False).mean()
    macd_main = ema_fast - ema_slow
    macd_sig = macd_main.ewm(span=macd_signal, adjust=False).mean()

    out["macd_main"] = macd_main
    out["macd_signal"] = macd_sig

    # ── Signal generation ────────────────────────────────────────────────────
    # The MQL4 uses Ask (for long) and Bid (for short). On historical bars we
    # approximate Ask ≈ high and Bid ≈ low of the current bar (worst case
    # spread assumption for backtesting — actual live signal would use the
    # tick Ask/Bid).
    ask = high  # approximation
    bid = low   # approximation

    sm_signal = np.zeros(n, dtype=int)
    sm_tp = np.full(n, np.nan, dtype=float)
    sm_sl = np.full(n, np.nan, dtype=float)

    bb1u = out["bb1_upper"].to_numpy()
    bb1l = out["bb1_lower"].to_numpy()
    bb2u = out["bb2_upper"].to_numpy()
    bb2l = out["bb2_lower"].to_numpy()
    bb6u = out["bb6_upper"].to_numpy()
    bb6l = out["bb6_lower"].to_numpy()
    mmain = macd_main.to_numpy()
    msig = macd_sig.to_numpy()

    for i in range(n):
        if np.isnan(bb1u[i]) or np.isnan(bb2l[i]) or np.isnan(bb6l[i]):
            continue
        # LONG: Ask in [LBBAND2, UBBAND1) AND MACD_main > MACD_signal
        if bb2l[i] <= ask[i] < bb1u[i] and mmain[i] > msig[i]:
            sm_signal[i] = 1
            sm_tp[i] = bb1u[i]
            sm_sl[i] = bb6l[i]
        # SHORT: Bid in (LBBAND1, UBBAND2] AND MACD_main < MACD_signal
        elif bb1l[i] < bid[i] <= bb2u[i] and mmain[i] < msig[i]:
            sm_signal[i] = -1
            sm_tp[i] = bb1l[i]
            sm_sl[i] = bb6u[i]

    out["sm_signal"] = sm_signal
    out["sm_tp"] = sm_tp
    out["sm_sl"] = sm_sl
    return out


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    n = 500
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    t = np.arange(n)
    # Mean-reverting series (good for BB strategies)
    close = 1.0850 + 0.0050 * np.sin(t / 25.0) + np.random.normal(0, 0.0005, n)
    high = close + 0.0003
    low = close - 0.0003
    df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=idx)

    out = compute(df)
    print(f"Rows: {len(out)}")
    print(f"Warmup NaN rows: {out['sma'].isna().sum()}")
    print(f"Long signals:  {int((out['sm_signal'] == 1).sum())}")
    print(f"Short signals: {int((out['sm_signal'] == -1).sum())}")
    print(f"Total signals: {int((out['sm_signal'] != 0).sum())}")

    # Verify band ordering: bb6 > bb2 > bb1 > sma > bb1_lower > bb2_lower > bb6_lower
    valid = out.dropna(subset=["bb1_upper"])
    assert (valid["bb6_upper"] > valid["bb2_upper"]).all()
    assert (valid["bb2_upper"] > valid["bb1_upper"]).all()
    assert (valid["bb1_upper"] > valid["sma"]).all()
    assert (valid["sma"] > valid["bb1_lower"]).all()
    assert (valid["bb1_lower"] > valid["bb2_lower"]).all()
    assert (valid["bb2_lower"] > valid["bb6_lower"]).all()
    print("Band ordering OK")

    # Verify TP/SL only set on signal bars
    sig_bars = out[out["sm_signal"] != 0]
    nonsig_bars = out[out["sm_signal"] == 0]
    assert not sig_bars["sm_tp"].isna().any(), "all signal bars should have TP"
    assert nonsig_bars["sm_tp"].isna().all() or nonsig_bars.empty, \
        "non-signal bars should have NaN TP"
    print("TP/SL assignment OK")

    # Verify TP/SL direction: long TP > price, SL < price
    long_bars = out[out["sm_signal"] == 1]
    if not long_bars.empty:
        assert (long_bars["sm_tp"] > long_bars["close"]).all(), "long TP must be above close"
        assert (long_bars["sm_sl"] < long_bars["close"]).all(), "long SL must be below close"
    short_bars = out[out["sm_signal"] == -1]
    if not short_bars.empty:
        assert (short_bars["sm_tp"] < short_bars["close"]).all(), "short TP must be below close"
        assert (short_bars["sm_sl"] > short_bars["close"]).all(), "short SL must be above close"
    print("TP/SL direction OK")

    print("\nSuperMao Bands smoke test passed.")
