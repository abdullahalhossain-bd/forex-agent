"""strategies/retest.py — Retest entry strategy"""
from __future__ import annotations
import pandas as pd
from utils.logger import get_logger
log = get_logger("retest_strategy")

class RetestStrategy:
    name = "Retest Entry"; version = "v1"; warmup = 30
    def __init__(self, breakout_lookback=20, retest_atr_mult=0.3, stop_atr_mult=1.0, rr_ratio=2.0, min_volume_ratio=1.0):
        self.breakout_lookback=breakout_lookback; self.retest_atr_mult=retest_atr_mult; self.stop_atr_mult=stop_atr_mult; self.rr_ratio=rr_ratio; self.min_volume_ratio=min_volume_ratio
    def generate(self, history):
        if len(history) < self.warmup: return {"signal":"HOLD","confidence":0}
        last = history.iloc[-1]; atr = float(last.get("atr",0) or 0)
        if atr <= 0: return {"signal":"HOLD","confidence":0}
        close = float(last["close"]); high = float(last["high"]); low = float(last["low"]); open_p = float(last["open"]); vr = float(last.get("volume_ratio",1.0) or 1.0)
        lookback = history.iloc[-self.breakout_lookback:-1]; bh = float(lookback["high"].max()); bl = float(lookback["low"].min())
        rz = atr * self.retest_atr_mult; recent = history.iloc[-5:]
        broke_above = (recent["close"] > bh).any(); broke_below = (recent["close"] < bl).any()
        bc = close > open_p and close > (high+low)/2
        if broke_above and abs(close-bh)<=rz and bc and vr>=self.min_volume_ratio:
            return self._signal("BUY", last, f"Retest of broken resistance at {bh:.5f}")
        bc2 = close < open_p and close < (high+low)/2
        if broke_below and abs(close-bl)<=rz and bc2 and vr>=self.min_volume_ratio:
            return self._signal("SELL", last, f"Retest of broken support at {bl:.5f}")
        return {"signal":"HOLD","confidence":0}
    def _signal(self, d, last, reason):
        return {"signal":d,"confidence":min(60+float(last.get("adx",20))*0.3,80),"reason":reason,"pattern":"retest","rr_ratio":self.rr_ratio,"stop_pips":max(float(last.get("atr",0.001))*self.stop_atr_mult*10000,8),"strategy_name":self.name}
