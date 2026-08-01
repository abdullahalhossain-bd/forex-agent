"""analysis/advanced_filters.py — Elliott Wave, EMA Ribbon, Fake Breakout, Wyckoff"""
from __future__ import annotations
import numpy as np
import pandas as pd
from utils.logger import get_logger
log = get_logger("advanced_filters")

def detect_elliott_wave(df, lookback=100):
    if df is None or len(df) < lookback: return {"wave_position":"UNKNOWN","signal":"NEUTRAL","score":0,"reason":"Insufficient data"}
    window = df.iloc[-lookback:]; highs = window["high"].values; lows = window["low"].values; closes = window["close"].values
    ph = []; pl = []
    for i in range(2, len(highs)-2):
        if highs[i] == max(highs[i-2:i+3]): ph.append((i, highs[i]))
        if lows[i] == min(lows[i-2:i+3]): pl.append((i, lows[i]))
    if len(ph) < 2 or len(pl) < 2: return {"wave_position":"UNKNOWN","signal":"NEUTRAL","score":10,"reason":"Not enough pivots"}
    rh = [h for _,h in ph[-3:]]; rl = [l for _,l in pl[-3:]]
    if len(rh)>=2 and len(rl)>=2:
        if rh[-1]>rh[-2] and rl[-1]>rl[-2]: return {"wave_position":"WAVE_4_CORRECTION","signal":"BUY","score":55,"reason":"Bullish HH+HL"}
        if rh[-1]<rh[-2] and rl[-1]<rl[-2]: return {"wave_position":"WAVE_4_CORRECTION","signal":"SELL","score":55,"reason":"Bearish LH+LL"}
    return {"wave_position":"UNKNOWN","signal":"NEUTRAL","score":15,"reason":"Wave unclear"}

def analyze_ema_ribbon(df):
    if df is None or len(df) < 50: return {"state":"UNKNOWN","signal":"NEUTRAL","score":0,"separation":0,"slope":0,"reason":"Insufficient"}
    for p in (8,21,50):
        c = f"ema_{p}"
        if c not in df.columns: df = df.copy(); df[c] = df["close"].ewm(span=p, adjust=False).mean()
    last = df.iloc[-1]; e8=float(last["ema_8"]); e21=float(last["ema_21"]); e50=float(last["ema_50"]); close=float(last["close"])
    sep = abs(e8-e50)/close*100 if close>0 else 0
    slope = (e21-float(df.iloc[-5]["ema_21"]))/close*100 if len(df)>=5 else 0
    if sep < 0.05: return {"state":"COMPRESSION","signal":"NEUTRAL","score":40,"separation":round(sep,4),"slope":round(slope,4),"reason":"EMA compression"}
    if e8>e21>e50: return {"state":"EXPANSION","signal":"BUY","score":80 if slope>0.05 else 55,"separation":round(sep,4),"slope":round(slope,4),"reason":"Bullish expansion"}
    if e8<e21<e50: return {"state":"EXPANSION","signal":"SELL","score":80 if slope<-0.05 else 55,"separation":round(sep,4),"slope":round(slope,4),"reason":"Bearish expansion"}
    return {"state":"CROSS","signal":"NEUTRAL","score":30,"separation":round(sep,4),"slope":round(slope,4),"reason":"EMA cross"}

def detect_fake_breakout(df, lookback=20, volume_threshold=1.2):
    if df is None or len(df) < lookback+2: return {"is_fake_breakout":False,"direction":"NONE","signal":"NEUTRAL","score":0,"reason":"Insufficient"}
    window = df.iloc[-lookback:]; res = float(window["high"][:-1].max()); sup = float(window["low"][:-1].min())
    last = df.iloc[-1]; lh=float(last["high"]); ll=float(last["low"]); lc=float(last["close"])
    if lh > res and lc < res: return {"is_fake_breakout":True,"direction":"BULLISH_FAKE","signal":"SELL","score":75,"reason":f"Fake breakout above {res:.5f}"}
    if ll < sup and lc > sup: return {"is_fake_breakout":True,"direction":"BEARISH_FAKE","signal":"BUY","score":75,"reason":f"Fake breakout below {sup:.5f}"}
    return {"is_fake_breakout":False,"direction":"NONE","signal":"NEUTRAL","score":10,"reason":"No fake breakout"}

def detect_wyckoff_pattern(df, lookback=50):
    if df is None or len(df) < lookback: return {"pattern":"NONE","signal":"NEUTRAL","score":0,"reason":"Insufficient"}
    window = df.iloc[-lookback:]; sup = float(window["low"][:-5].min()); res = float(window["high"][:-5].max())
    last = df.iloc[-1]; lc=float(last["close"]); ll=float(last["low"]); lh=float(last["high"]); pc=float(df.iloc[-2]["close"])
    if ll < sup and lc > sup and lc > pc: return {"pattern":"SPRING","signal":"BUY","score":80,"reason":"Spring: swept support, closed above"}
    if lh > res and lc < res and lc < pc: return {"pattern":"UPTHRUST","signal":"SELL","score":80,"reason":"Upthrust: spiked resistance, closed below"}
    return {"pattern":"NONE","signal":"NEUTRAL","score":15,"reason":"No Wyckoff pattern"}
