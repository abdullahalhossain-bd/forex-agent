"""strategies/range_trading.py — Range trading strategy"""
from __future__ import annotations
import math
from strategy._common import safe_float
from utils.logger import get_logger
log = get_logger("range_trading_strategy")

class RangeTradingStrategy:
    name = "Range Trading"; version = "v1"; warmup = 55
    def __init__(self, range_lookback=50, max_adx=20.0, min_range_atr_mult=2.0, rsi_buy_max=35.0, rsi_sell_min=65.0, stop_atr_mult=1.0, rr_ratio=2.0):
        self.range_lookback=range_lookback; self.max_adx=max_adx; self.min_range_atr_mult=min_range_atr_mult; self.rsi_buy_max=rsi_buy_max; self.rsi_sell_min=rsi_sell_min; self.stop_atr_mult=stop_atr_mult; self.rr_ratio=rr_ratio
    def generate(self, history):
        if len(history) < self.warmup: return {"signal":"HOLD","confidence":0}
        last = history.iloc[-1]; atr = safe_float(last, "atr", default=0.0)
        if math.isnan(atr) or atr <= 0: return {"signal":"HOLD","confidence":0,"reason":"atr missing, NaN, or non-positive"}
        close = float(last["close"]); high = float(last["high"]); low = float(last["low"]); open_p = float(last["open"])
        rsi = safe_float(last, "rsi", default=50.0); adx = safe_float(last, "adx", default=0.0)
        if math.isnan(rsi) or math.isnan(adx):
            return {"signal":"HOLD","confidence":0,"reason":"rsi/adx missing or NaN"}
        if adx >= self.max_adx: return {"signal":"HOLD","confidence":0,"reason":"ADX too high"}
        lb = history.iloc[-self.range_lookback:-1]; res = float(lb["high"].max()); sup = float(lb["low"].min()); rw = res - sup
        if rw < atr * self.min_range_atr_mult: return {"signal":"HOLD","confidence":0,"reason":"Range too narrow"}
        at_sup = abs(close-sup) <= atr*0.3; at_res = abs(close-res) <= atr*0.3
        br = close > open_p and close > (high+low)/2; bear_r = close < open_p and close < (high+low)/2
        if at_sup and rsi <= self.rsi_buy_max and br: return self._signal("BUY", sup, res, f"Range BUY at support {sup:.5f}")
        if at_res and rsi >= self.rsi_sell_min and bear_r: return self._signal("SELL", sup, res, f"Range SELL at resistance {res:.5f}")
        return {"signal":"HOLD","confidence":0}
    def _signal(self, d, sup, res, reason):
        return {"signal":d,"confidence":58,"reason":reason,"pattern":"range_trade","rr_ratio":self.rr_ratio,"range_support":sup,"range_resistance":res,"strategy_name":self.name}