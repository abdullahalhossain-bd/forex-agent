"""strategies/momentum.py — Momentum entry strategy"""
from __future__ import annotations
import math
import numpy as np
from strategies._common import safe_float
from utils.logger import get_logger
log = get_logger("momentum_strategy")

class MomentumStrategy:
    name = "Momentum Entry"; version = "v1"; warmup = 40
    def __init__(self, roc_period=10, roc_threshold=0.0015, min_volume_ratio=1.5, min_adx=25.0, stop_atr_mult=1.5, rr_ratio=2.0):
        self.roc_period=roc_period; self.roc_threshold=roc_threshold; self.min_volume_ratio=min_volume_ratio; self.min_adx=min_adx; self.stop_atr_mult=stop_atr_mult; self.rr_ratio=rr_ratio
    def generate(self, history):
        if len(history) < self.warmup: return {"signal":"HOLD","confidence":0}
        last = history.iloc[-1]; atr = safe_float(last, "atr", default=0.0)
        if math.isnan(atr) or atr <= 0: return {"signal":"HOLD","confidence":0,"reason":"atr missing, NaN, or non-positive"}
        close = float(last["close"]); high = float(last["high"]); low = float(last["low"]); open_p = float(last["open"])
        rsi = safe_float(last, "rsi", default=50.0); adx = safe_float(last, "adx", default=0.0); vr = safe_float(last, "volume_ratio", default=1.0)
        # NaN inputs must block the trade explicitly, not silently fail every
        # comparison as False (a NaN adx would otherwise make `adx >= min_adx`
        # False -- looks like "filter blocked it", is actually "filter never
        # ran on real data"). See strategies/_common.py docstring.
        if math.isnan(rsi) or math.isnan(adx) or math.isnan(vr):
            return {"signal":"HOLD","confidence":0,"reason":"rsi/adx/volume_ratio missing or NaN"}
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