"""strategies/pullback.py — Pullback entry strategy"""
from __future__ import annotations
from utils.logger import get_logger
log = get_logger("pullback_strategy")

class PullbackStrategy:
    name = "Pullback Entry"; version = "v1"; warmup = 60
    def __init__(self, ema_fast=21, ema_slow=50, min_adx=20.0, stop_atr_mult=1.2, rr_ratio=2.0):
        self.ema_fast=ema_fast; self.ema_slow=ema_slow; self.min_adx=min_adx; self.stop_atr_mult=stop_atr_mult; self.rr_ratio=rr_ratio
    def generate(self, history, trend="NEUTRAL"):
        if len(history) < self.warmup: return {"signal":"HOLD","confidence":0}
        last = history.iloc[-1]; atr = float(last.get("atr",0) or 0)
        if atr <= 0: return {"signal":"HOLD","confidence":0}
        close = float(last["close"]); high = float(last["high"]); low = float(last["low"]); open_p = float(last["open"])
        rsi = float(last.get("rsi",50) or 50); adx = float(last.get("adx",0) or 0)
        ef = float(last.get(f"ema_{self.ema_fast}",close) or close); es = float(last.get(f"ema_{self.ema_slow}",close) or close)
        if adx < self.min_adx: return {"signal":"HOLD","confidence":0,"reason":"ADX too low"}
        bt = ef > es and trend.upper() in ("BULLISH","NEUTRAL",""); bear_t = ef < es and trend.upper() in ("BEARISH","NEUTRAL","")
        at_ema = abs(close-ef) <= atr*0.5; rz = 40 <= rsi <= 65
        bc = close > open_p and close > (high+low)/2; bear_c = close < open_p and close < (high+low)/2
        if bt and at_ema and rz and bc:
            return self._signal("BUY", last, f"Bullish pullback to EMA{self.ema_fast}")
        if bear_t and at_ema and rz and bear_c:
            return self._signal("SELL", last, f"Bearish pullback to EMA{self.ema_fast}")
        return {"signal":"HOLD","confidence":0}
    def _signal(self, d, last, reason):
        adx = float(last.get("adx",20) or 20); return {"signal":d,"confidence":min(58+(adx-self.min_adx)*0.8,82),"reason":reason,"pattern":"pullback","rr_ratio":self.rr_ratio,"strategy_name":self.name}
