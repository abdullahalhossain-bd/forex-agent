"""strategies/momentum.py — Momentum entry strategy"""
from __future__ import annotations
import numpy as np
from utils.logger import get_logger
log = get_logger("momentum_strategy")

class MomentumStrategy:
    name = "Momentum Entry"; version = "v1"; warmup = 40
    def __init__(self, roc_period=10, roc_threshold=0.0015, min_volume_ratio=1.5, min_adx=25.0, stop_atr_mult=1.5, rr_ratio=2.0):
        self.roc_period=roc_period; self.roc_threshold=roc_threshold; self.min_volume_ratio=min_volume_ratio; self.min_adx=min_adx; self.stop_atr_mult=stop_atr_mult; self.rr_ratio=rr_ratio
    def generate(self, history):
        if len(history) < self.warmup: return {"signal":"HOLD","confidence":0}
        last = history.iloc[-1]; atr = float(last.get("atr",0) or 0)
        if atr <= 0: return {"signal":"HOLD","confidence":0}
        close = float(last["close"]); high = float(last["high"]); low = float(last["low"]); open_p = float(last["open"])
        rsi = float(last.get("rsi",50) or 50); adx = float(last.get("adx",0) or 0); vr = float(last.get("volume_ratio",1.0) or 1.0)
        if len(history) < self.roc_period+1: return {"signal":"HOLD","confidence":0}
        roc = (close - float(history.iloc[-self.roc_period-1]["close"])) / float(history.iloc[-self.roc_period-1]["close"])
        bm = roc >= self.roc_threshold; bear_m = roc <= -self.roc_threshold
        vc = vr >= self.min_volume_ratio; ao = adx >= self.min_adx
        rb = 55 <= rsi <= 80; rs = 20 <= rsi <= 45
        cr = high-low; body = abs(close-open_p); br = body/cr if cr > 0 else 0; mc = br >= 0.60
        if bm and vc and ao and rb and mc and close > open_p:
            return self._signal("BUY", roc, f"Bullish momentum ROC={roc*100:.3f}%")
        if bear_m and vc and ao and rs and mc and close < open_p:
            return self._signal("SELL", roc, f"Bearish momentum ROC={roc*100:.3f}%")
        return {"signal":"HOLD","confidence":0}
    def _signal(self, d, roc, reason):
        return {"signal":d,"confidence":min(60+abs(roc)*5000,85),"reason":reason,"pattern":"momentum","rr_ratio":self.rr_ratio,"roc":roc,"strategy_name":self.name}
