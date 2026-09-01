"""risk/structure_stop.py — Structure-based stop loss"""
from __future__ import annotations
import pandas as pd
from utils.logger import get_logger
log = get_logger("structure_stop")

def find_swing_low(df, lookback=20): return float(df.iloc[-lookback:]["low"].min()) if len(df)>=lookback else float(df["low"].min())
def find_swing_high(df, lookback=20): return float(df.iloc[-lookback:]["high"].max()) if len(df)>=lookback else float(df["high"].max())
def find_fractal_swing_low(df, lookback=20, fs=2):
    if len(df) < lookback: return find_swing_low(df, lookback)
    window = df.iloc[-lookback:]; lows = window["low"].values
    for i in range(len(lows)-1-fs, fs-1, -1):
        if all(lows[i] < lows[i-j] and lows[i] < lows[i+j] for j in range(1, fs+1)): return float(lows[i])
    return float(lows.min())
def find_fractal_swing_high(df, lookback=20, fs=2):
    if len(df) < lookback: return find_swing_high(df, lookback)
    window = df.iloc[-lookback:]; highs = window["high"].values
    for i in range(len(highs)-1-fs, fs-1, -1):
        if all(highs[i] > highs[i-j] and highs[i] > highs[i+j] for j in range(1, fs+1)): return float(highs[i])
    return float(highs.max())

def compute_structure_stop(df, direction, method="swing_atr", lookback=50, atr_buffer_mult=0.20, atr=0.0):
    """Structure-based SL. lookback=50 matches entry_quality DEFAULT_SL_SWING_LOOKBACK
    so Devil's Advocate / sl_swing_anchor sees the same swings.
    atr_buffer_mult=0.20 matches risk_engine structure path.
    """
    if direction.upper() == "BUY":
        swing = find_fractal_swing_low(df, lookback) or find_swing_low(df, lookback)
        if atr <= 0:
            atr = float(df.iloc[-1].get("atr", 0.001) or 0.001) if len(df) > 0 else 0.001
        if method == "swing":
            return swing
        return swing - atr * atr_buffer_mult
    else:
        swing = find_fractal_swing_high(df, lookback) or find_swing_high(df, lookback)
        if atr <= 0:
            atr = float(df.iloc[-1].get("atr", 0.001) or 0.001) if len(df) > 0 else 0.001
        if method == "swing":
            return swing
        return swing + atr * atr_buffer_mult

def compute_structure_stop_pips(df, direction, entry_price, method="swing_atr", lookback=50, atr_buffer_mult=0.20, pip_size=0.0001):
    """Structure SL distance, in pips.

    2026-09 audit fix (finding F3): this used to floor the result at a
    flat `5.0` pips regardless of symbol — the same kind of symbol-blind
    noise-floor default that core/da_safety_net.py's DA_MIN_SL_PIPS was
    removed for. "5 pips" means wildly different things across
    instruments depending on pip_size/typical volatility, and this
    function has no idea which symbol it's being called for beyond
    whatever pip_size the caller passed in — floor removed. The caller
    is expected to pass the symbol's authoritative pip_size (e.g. via
    core.constants.get_pip_size) rather than relying on the 0.0001
    default here for non-4/5-digit-quote symbols (JPY pairs, metals,
    indices), the same "no default data" policy documented in
    core/da_safety_net.py.
    """
    sp = compute_structure_stop(df, direction, method, lookback, atr_buffer_mult)
    sl_pips = (entry_price - sp) / pip_size if direction.upper() == "BUY" else (sp - entry_price) / pip_size
    return round(sl_pips, 1)
