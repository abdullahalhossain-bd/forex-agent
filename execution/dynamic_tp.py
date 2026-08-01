"""execution/dynamic_tp.py — Dynamic take profit"""
from __future__ import annotations
import pandas as pd
from utils.logger import get_logger
log = get_logger("dynamic_tp")

def compute_dynamic_tp(entry, direction, df, bars_in_trade=0, method="atr_structure_blend", initial_atr_mult=3.0, max_atr_mult=5.0, time_decay_per_bar=0.05, lookback=50, momentum_threshold=60.0):
    if df is None or len(df) < 10: return entry
    last = df.iloc[-1]; atr = float(last.get("atr",0.001) or 0.001); rsi = float(last.get("rsi",50) or 50)
    direction = direction.upper()
    if method == "atr":
        am = min(initial_atr_mult + bars_in_trade*0.1, max_atr_mult); d = atr*am
        return entry + d if direction == "BUY" else entry - d
    elif method == "structure":
        window = df.iloc[-lookback:] if len(df)>=lookback else df
        if direction == "BUY":
            st = float(window["high"].max()); return st if st > entry else entry + atr * initial_atr_mult
        else:
            st = float(window["low"].min()); return st if st < entry else entry - atr * initial_atr_mult
    elif method == "momentum":
        mf = 1.3 if (direction=="BUY" and rsi>=momentum_threshold) or (direction=="SELL" and rsi<=100-momentum_threshold) else 0.7 if (direction=="BUY" and rsi<45) or (direction=="SELL" and rsi>55) else 1.0
        d = atr * initial_atr_mult * mf
        return entry + d if direction=="BUY" else entry - d
    else:  # blend
        am = min(initial_atr_mult + bars_in_trade*0.05, max_atr_mult)
        atr_d = atr * am
        window = df.iloc[-lookback:] if len(df)>=lookback else df
        if direction=="BUY": sd = max(float(window["high"].max()) - entry, atr_d)
        else: sd = max(entry - float(window["low"].min()), atr_d)
        mm = 1.2 if (direction=="BUY" and rsi>=momentum_threshold) or (direction=="SELL" and rsi<=100-momentum_threshold) else 0.8 if (direction=="BUY" and rsi<45) or (direction=="SELL" and rsi>55) else 1.0
        td = max(1.0 - bars_in_trade*time_decay_per_bar, 0.5)
        bd = ((atr_d*0.4) + (sd*0.4) + (atr_d*mm*0.2)) * td
        return entry + bd if direction=="BUY" else entry - bd

def should_extend_tp(current_tp, entry, direction, df, extension_threshold=0.3):
    if df is None or len(df) < 5: return False, current_tp, "Insufficient"
    last = df.iloc[-1]; rsi = float(last.get("rsi",50) or 50); atr = float(last.get("atr",0.001) or 0.001)
    if direction.upper()=="BUY" and rsi >= 65: return True, current_tp + atr, f"Strong momentum RSI={rsi:.1f}"
    if direction.upper()=="SELL" and rsi <= 35: return True, current_tp - atr, f"Strong momentum RSI={rsi:.1f}"
    return False, current_tp, "Momentum insufficient"
