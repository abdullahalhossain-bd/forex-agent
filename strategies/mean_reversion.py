"""strategies/mean_reversion.py — Mean reversion strategy"""
from __future__ import annotations
from utils.logger import get_logger
log = get_logger("mean_reversion_strategy")

class MeanReversionStrategy:
    name = "Mean Reversion"; version = "v1"; warmup = 50
    def __init__(self, rsi_oversold=30.0, rsi_overbought=70.0, max_adx=25.0, stop_atr_mult=1.5, rr_ratio=2.0):
        self.rsi_oversold=rsi_oversold; self.rsi_overbought=rsi_overbought; self.max_adx=max_adx; self.stop_atr_mult=stop_atr_mult; self.rr_ratio=rr_ratio
    def generate(self, history):
        if len(history) < self.warmup: return {"signal":"HOLD","confidence":0}
        last = history.iloc[-1]; atr = float(last.get("atr",0) or 0)
        if atr <= 0: return {"signal":"HOLD","confidence":0}
        close = float(last["close"]); high = float(last["high"]); low = float(last["low"]); open_p = float(last["open"])
        rsi = float(last.get("rsi",50) or 50); adx = float(last.get("adx",0) or 0)
        bb_u = float(last.get("bb_upper",close) or close); bb_l = float(last.get("bb_lower",close) or close); bb_m = float(last.get("bb_middle",close) or close)
        if adx >= self.max_adx: return {"signal":"HOLD","confidence":0,"reason":"ADX too high"}
        at_upper = close >= bb_u; at_lower = close <= bb_l
        br = close > open_p and close > (high+low)/2; bear_r = close < open_p and close < (high+low)/2
        if at_lower and rsi <= self.rsi_oversold and br:
            return self._signal("BUY", bb_m, f"Mean reversion BUY at lower BB, RSI={rsi:.1f}")
        if at_upper and rsi >= self.rsi_overbought and bear_r:
            return self._signal("SELL", bb_m, f"Mean reversion SELL at upper BB, RSI={rsi:.1f}")
        return {"signal":"HOLD","confidence":0}
    def _signal(self, d, target, reason):
        adx = 20; return {"signal":d,"confidence":min(55+(self.max_adx-adx)*1.5,80),"reason":reason,"pattern":"mean_reversion","rr_ratio":self.rr_ratio,"target_price":target,"strategy_name":self.name}
