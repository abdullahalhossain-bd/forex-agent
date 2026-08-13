"""
utils/pair_strategies.py — Per-Pair Strategy Systems

Each pair gets a DIFFERENT trading system based on its behavior:
  - TREND FOLLOWING  — for trending pairs (USDJPY, NZDUSD)
  - MEAN REVERSION   — for choppy pairs (EURUSD, GBPUSD)
  - RANGE TRADING    — for range-bound pairs (USDCAD)
  - BREAKOUT         — for volatile pairs (AUDUSD, USDCHF)

Each strategy has its own entry logic, SL/TP calculation, and filters.
The pair_profiles.py assigns which strategy to use per pair.
"""
from __future__ import annotations
import math
from typing import Any, Optional


def _safe(row, key, default=0.0):
    v = row.get(key, default) if hasattr(row, "get") else default
    if v is None: return default
    try:
        v = float(v)
        return default if math.isnan(v) else v
    except: return default


# ═══════════════════════════════════════════════════════════════════
# STRATEGY 1: TREND FOLLOWING (for USDJPY, NZDUSD)
# ═══════════════════════════════════════════════════════════════════
def trend_following_signal(ind_ctx, pat_ctx, sr_ctx, pair=None):
    """Trend-following: enter on pullback to EMA21 in established trend.

    Entry: price near EMA-21, EMA stack aligned (9>21>50>200), ADX>25
    SL: below recent swing low (tight — 1.0×ATR)
    TP: 3.0×ATR (RR 1:3 — trend trades run far)
    """
    price = _safe(ind_ctx, "price")
    ema_9 = _safe(ind_ctx, "ema_9")
    ema_21 = _safe(ind_ctx, "ema_21")
    ema_50 = _safe(ind_ctx, "ema_50")
    ema_200 = _safe(ind_ctx, "ema_200")
    atr = _safe(ind_ctx, "atr")
    adx = _safe(ind_ctx, "adx")
    rsi = _safe(ind_ctx, "rsi", 50)

    if atr <= 0 or adx < 22:
        return {"signal": "WAIT", "confidence": 0, "strategy": "trend_follow"}

    bull_stack = ema_9 > ema_21 > ema_50 > ema_200
    bear_stack = ema_9 < ema_21 < ema_50 < ema_200

    # Pullback: price within 0.8×ATR of EMA-21
    near_ema21 = abs(price - ema_21) <= atr * 0.8

    factors = 0
    if bull_stack: factors += 1
    if near_ema21: factors += 1
    if adx >= 25: factors += 1
    if 40 <= rsi <= 65: factors += 1  # room to run

    if bull_stack and near_ema21 and factors >= 3:
        return {
            "signal": "BUY", "confidence": min(60 + factors * 8, 90),
            "strategy": "trend_follow",
            "sl_price": price - atr * 1.0,
            "tp_price": price + atr * 3.0,
            "rr_ratio": 3.0,
            "factors": factors,
            "reason": f"Trend-follow BUY: stack+pullback+ADX({adx:.0f})",
        }
    if bear_stack and near_ema21 and factors >= 3:
        return {
            "signal": "SELL", "confidence": min(60 + factors * 8, 90),
            "strategy": "trend_follow",
            "sl_price": price + atr * 1.0,
            "tp_price": price - atr * 3.0,
            "rr_ratio": 3.0,
            "factors": factors,
            "reason": f"Trend-follow SELL: stack+pullback+ADX({adx:.0f})",
        }
    return {"signal": "WAIT", "confidence": 0, "strategy": "trend_follow"}


# ═══════════════════════════════════════════════════════════════════
# STRATEGY 2: MEAN REVERSION (for EURUSD, GBPUSD)
# ═══════════════════════════════════════════════════════════════════
def mean_reversion_signal(ind_ctx, pat_ctx, sr_ctx, pair=None):
    """Mean-reversion: enter on RSI extreme + Bollinger Band touch.

    Entry: RSI < 30 (oversold) + price at/below lower BB + ADX < 25 (no strong trend)
    SL: below entry - 1.5×ATR (wider — give room for continued extension)
    TP: 1.5×ATR (RR 1:1 — quick reversion to mean)
    """
    price = _safe(ind_ctx, "price")
    atr = _safe(ind_ctx, "atr")
    rsi = _safe(ind_ctx, "rsi", 50)
    adx = _safe(ind_ctx, "adx")
    bb_upper = _safe(ind_ctx, "bb_upper")
    bb_lower = _safe(ind_ctx, "bb_lower")
    bb_middle = _safe(ind_ctx, "bb_middle")
    ema_50 = _safe(ind_ctx, "ema_50")

    if atr <= 0:
        return {"signal": "WAIT", "confidence": 0, "strategy": "mean_reversion"}

    # Only trade when NOT in strong trend (ADX < 25)
    if adx > 28:
        return {"signal": "WAIT", "confidence": 0, "strategy": "mean_reversion",
                "reason": f"ADX too high ({adx:.0f}) for mean-reversion"}

    factors = 0
    # Oversold + at lower BB → BUY signal
    if rsi <= 35:
        factors += 1
    if price <= bb_lower:
        factors += 1
    if adx < 22:
        factors += 1
    # Price below EMA-50 (extended away from mean)
    if price < ema_50:
        factors += 1

    if rsi <= 35 and price <= bb_lower and factors >= 3:
        return {
            "signal": "BUY", "confidence": min(55 + factors * 8, 85),
            "strategy": "mean_reversion",
            "sl_price": price - atr * 1.5,
            "tp_price": price + atr * 1.5,  # target = BB middle approx
            "rr_ratio": 1.0,
            "factors": factors,
            "reason": f"Mean-revert BUY: RSI({rsi:.0f})+BB lower+low ADX",
        }

    # Overbought + at upper BB → SELL signal
    factors = 0
    if rsi >= 65: factors += 1
    if price >= bb_upper: factors += 1
    if adx < 22: factors += 1
    if price > ema_50: factors += 1

    if rsi >= 65 and price >= bb_upper and factors >= 3:
        return {
            "signal": "SELL", "confidence": min(55 + factors * 8, 85),
            "strategy": "mean_reversion",
            "sl_price": price + atr * 1.5,
            "tp_price": price - atr * 1.5,
            "rr_ratio": 1.0,
            "factors": factors,
            "reason": f"Mean-revert SELL: RSI({rsi:.0f})+BB upper+low ADX",
        }
    return {"signal": "WAIT", "confidence": 0, "strategy": "mean_reversion"}


# ═══════════════════════════════════════════════════════════════════
# STRATEGY 3: RANGE TRADING (for USDCAD)
# ═══════════════════════════════════════════════════════════════════
def range_trading_signal(ind_ctx, pat_ctx, sr_ctx, pair=None):
    """Range trading: enter at support/resistance in low-ADX market.

    Entry: ADX < 20 (range market) + price near S/R + RSI confirmation
    SL: 2.0×ATR (wide — range can spike)
    TP: 1.5×ATR (RR 1:0.75 — quick exit at opposite range edge)
    """
    price = _safe(ind_ctx, "price")
    atr = _safe(ind_ctx, "atr")
    adx = _safe(ind_ctx, "adx")
    rsi = _safe(ind_ctx, "rsi", 50)
    support = _safe(sr_ctx, "support", price)
    resistance = _safe(sr_ctx, "resistance", price)
    location = sr_ctx.get("location", "mid_range") if hasattr(sr_ctx, "get") else "mid_range"

    if atr <= 0:
        return {"signal": "WAIT", "confidence": 0, "strategy": "range_trading"}

    # Only trade in range market
    if adx > 22:
        return {"signal": "WAIT", "confidence": 0, "strategy": "range_trading",
                "reason": f"ADX too high ({adx:.0f}) for range trading"}

    # Distance to S/R
    dist_to_support = abs(price - support)
    dist_to_resistance = abs(price - resistance)
    near_support = dist_to_support <= atr * 1.0
    near_resistance = dist_to_resistance <= atr * 1.0

    factors = 0
    if adx < 20: factors += 1
    if rsi < 40: factors += 1  # oversold in range

    # BUY at support
    if near_support and rsi < 45 and factors >= 2:
        return {
            "signal": "BUY", "confidence": min(55 + factors * 10, 80),
            "strategy": "range_trading",
            "sl_price": support - atr * 1.0,  # below support
            "tp_price": resistance,  # target = resistance (opposite edge)
            "rr_ratio": abs(resistance - price) / (atr * 1.0) if atr > 0 else 1.0,
            "factors": factors,
            "reason": f"Range BUY at support (RSI {rsi:.0f}, ADX {adx:.0f})",
        }

    factors = 0
    if adx < 20: factors += 1
    if rsi > 60: factors += 1

    # SELL at resistance
    if near_resistance and rsi > 55 and factors >= 2:
        return {
            "signal": "SELL", "confidence": min(55 + factors * 10, 80),
            "strategy": "range_trading",
            "sl_price": resistance + atr * 1.0,
            "tp_price": support,
            "rr_ratio": abs(price - support) / (atr * 1.0) if atr > 0 else 1.0,
            "factors": factors,
            "reason": f"Range SELL at resistance (RSI {rsi:.0f}, ADX {adx:.0f})",
        }
    return {"signal": "WAIT", "confidence": 0, "strategy": "range_trading"}


# ═══════════════════════════════════════════════════════════════════
# STRATEGY 4: BREAKOUT (for AUDUSD, USDCHF)
# ═══════════════════════════════════════════════════════════════════
def breakout_signal(ind_ctx, pat_ctx, sr_ctx, pair=None):
    """Breakout: enter on range expansion with volume confirmation.

    Entry: price breaks 20-bar high/low + ADX rising + volume above avg
    SL: 1.5×ATR (medium — give breakout room to develop)
    TP: 3.0×ATR (RR 1:2 — breakouts can run far)
    """
    price = _safe(ind_ctx, "price")
    atr = _safe(ind_ctx, "atr")
    adx = _safe(ind_ctx, "adx")
    volume_ratio = _safe(ind_ctx, "volume_ratio", 1.0)
    rolling_resistance = _safe(ind_ctx, "rolling_resistance_20", price)
    rolling_support = _safe(ind_ctx, "rolling_support_20", price)

    if atr <= 0:
        return {"signal": "WAIT", "confidence": 0, "strategy": "breakout"}

    # Breakout conditions
    broke_up = price > rolling_resistance
    broke_down = price < rolling_support
    adx_rising = adx >= 20
    vol_confirm = volume_ratio >= 1.0

    factors = 0
    if broke_up or broke_down: factors += 1
    if adx_rising: factors += 1
    if vol_confirm: factors += 1

    if broke_up and adx_rising and factors >= 2:
        return {
            "signal": "BUY", "confidence": min(58 + factors * 8, 85),
            "strategy": "breakout",
            "sl_price": price - atr * 1.5,
            "tp_price": price + atr * 3.0,
            "rr_ratio": 2.0,
            "factors": factors,
            "reason": f"Breakout BUY above {rolling_resistance:.5f} (ADX {adx:.0f})",
        }
    if broke_down and adx_rising and factors >= 2:
        return {
            "signal": "SELL", "confidence": min(58 + factors * 8, 85),
            "strategy": "breakout",
            "sl_price": price + atr * 1.5,
            "tp_price": price - atr * 3.0,
            "rr_ratio": 2.0,
            "factors": factors,
            "reason": f"Breakout SELL below {rolling_support:.5f} (ADX {adx:.0f})",
        }
    return {"signal": "WAIT", "confidence": 0, "strategy": "breakout"}


# ═══════════════════════════════════════════════════════════════════
# STRATEGY DISPATCHER
# ═══════════════════════════════════════════════════════════════════
STRATEGIES = {
    "trend_follow": trend_following_signal,
    "mean_reversion": mean_reversion_signal,
    "range_trading": range_trading_signal,
    "breakout": breakout_signal,
}

def get_strategy(name: str):
    """Get a strategy function by name."""
    return STRATEGIES.get(name, trend_following_signal)

def run_strategy(name: str, ind_ctx, pat_ctx, sr_ctx, pair=None):
    """Run the named strategy and return its signal dict."""
    fn = STRATEGIES.get(name)
    if fn is None:
        return {"signal": "WAIT", "confidence": 0, "strategy": name, "reason": "unknown strategy"}
    return fn(ind_ctx, pat_ctx, sr_ctx, pair=pair)


if __name__ == "__main__":
    # Test each strategy signature
    import pandas as pd
    test_ind = {"price": 1.0, "ema_9": 1.001, "ema_21": 1.0, "ema_50": 0.999,
                "ema_200": 0.995, "atr": 0.005, "adx": 26, "rsi": 52,
                "bb_upper": 1.01, "bb_lower": 0.99, "bb_middle": 1.0,
                "volume_ratio": 1.2, "rolling_resistance_20": 0.998,
                "rolling_support_20": 0.992}
    test_sr = {"support": 0.99, "resistance": 1.01, "location": "mid_range"}
    test_pat = {}
    for name in STRATEGIES:
        res = run_strategy(name, test_ind, test_pat, test_sr, "TEST")
        print(f"  {name:20s} → {res['signal']:5s} conf={res.get('confidence',0)}")
    print("All strategies loaded OK")
