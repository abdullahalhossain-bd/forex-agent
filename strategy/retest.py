"""strategies/retest.py — Retest entry strategy"""
from __future__ import annotations
import math
import pandas as pd
from strategies._common import safe_float, pip_size_for
from utils.logger import get_logger
log = get_logger("retest_strategy")

class RetestStrategy:
    name = "Retest Entry"; version = "v1"; warmup = 30
    def __init__(self, breakout_lookback=20, retest_atr_mult=0.3, stop_atr_mult=1.0, rr_ratio=2.0, min_volume_ratio=1.0):
        self.breakout_lookback=breakout_lookback; self.retest_atr_mult=retest_atr_mult; self.stop_atr_mult=stop_atr_mult; self.rr_ratio=rr_ratio; self.min_volume_ratio=min_volume_ratio
    def generate(self, history, pair=None):
        if len(history) < self.warmup: return {"signal":"HOLD","confidence":0}
        last = history.iloc[-1]; atr = safe_float(last, "atr", default=0.0)
        if math.isnan(atr) or atr <= 0: return {"signal":"HOLD","confidence":0,"reason":"atr missing, NaN, or non-positive"}
        close = float(last["close"]); high = float(last["high"]); low = float(last["low"]); open_p = float(last["open"])
        vr = safe_float(last, "volume_ratio", default=1.0)
        if math.isnan(vr):
            return {"signal":"HOLD","confidence":0,"reason":"volume_ratio missing or NaN"}
        lookback = history.iloc[-self.breakout_lookback:-1]; bh = float(lookback["high"].max()); bl = float(lookback["low"].min())
        rz = atr * self.retest_atr_mult; recent = history.iloc[-5:]
        broke_above = (recent["close"] > bh).any(); broke_below = (recent["close"] < bl).any()
        bc = close > open_p and close > (high+low)/2
        if broke_above and abs(close-bh)<=rz and bc and vr>=self.min_volume_ratio:
            return self._signal("BUY", last, f"Retest of broken resistance at {bh:.5f}", pair)
        bc2 = close < open_p and close < (high+low)/2
        if broke_below and abs(close-bl)<=rz and bc2 and vr>=self.min_volume_ratio:
            return self._signal("SELL", last, f"Retest of broken support at {bl:.5f}", pair)
        return {"signal":"HOLD","confidence":0}
    def _signal(self, d, last, reason, pair=None):
        adx = safe_float(last, "adx", default=20.0)
        if math.isnan(adx): adx = 20.0
        atr = safe_float(last, "atr", default=0.001)
        if math.isnan(atr) or atr <= 0: atr = 0.001
        # pip-size lookup (was a hardcoded *10000 that mis-sizes JPY/gold
        # stops -- same class of bug documented/fixed in trend_follow.py).
        # Pass `pair` into generate() so this resolves correctly; falls
        # back to the FX-major default otherwise.
        pip = 1.0 / pip_size_for(pair)
        return {"signal":d,"confidence":min(60+adx*0.3,80),"reason":reason,"pattern":"retest","rr_ratio":self.rr_ratio,"stop_pips":max(round(atr*self.stop_atr_mult*pip,1),8.0),"strategy_name":self.name}