# analysis/engulfing_bar_strategy.py
# Engulfing Bar Strategy — Book "Candlestick Trading Bible" Pages 111-125
# Nison's 3 Criteria + MA/Fib/SR confluence + Entry/SL/TP
import logging
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
VALID_TIMEFRAMES = {"H4", "D1", "W1"}
FIB_GOLDEN_LOW = 0.50
FIB_GOLDEN_HIGH = 0.618
MA_PROXIMITY_ATR_MULT = 0.5
SL_BUFFER_ATR_MULT = 0.15

@dataclass
class EngulfingSetup:
    detected: bool = False
    direction: str = "neutral"
    candle_index: int = -1
    candle_time: str = ""
    nison_1_trend: bool = False
    nison_2_engulf: bool = False
    nison_3_opposite: bool = False
    c1_open: float = 0
    c1_close: float = 0
    c2_open: float = 0
    c2_close: float = 0
    c2_high: float = 0
    c2_low: float = 0
    confluence_ma: bool = False
    confluence_200: bool = False
    confluence_fib: bool = False
    confluence_fib_level: str = ""
    confluence_sr: bool = False
    confluence_role_reversal: bool = False
    confluence_count: int = 0
    trend_direction: str = "unknown"
    entry_price: float = 0
    stop_loss: float = 0
    take_profit: float = 0
    risk_reward: float = 0
    quality_score: int = 0
    quality_grade: str = "F"
    def to_dict(self): return {k: v for k, v in self.__dict__.items()}

class EngulfingBarStrategy:
    def __init__(self, timeframe="H4"):
        self.timeframe = timeframe.upper()
    def detect(self, df, symbol="", trend_direction="unknown", sr_zones=None, fib_levels=None, atr_value=None):
        if df is None or len(df) < 5: return EngulfingSetup()
        setup = self._detect_engulfing(df)
        if not setup.detected: return setup
        setup.trend_direction = trend_direction
        # FIX (institutional review, item #3): Nison's Rule 1 requires the
        # engulfing pattern to occur AGAINST the prevailing trend (that's
        # what makes it a reversal signal at all). The old check only
        # verified that *some* trend label was passed in
        # (`trend_direction.upper() in ("BULLISH", "BEARISH")`), so a
        # bullish engulfing bar inside an already-bullish trend — a
        # continuation, not a reversal — still scored the full 30 points.
        # Now it requires the engulfing direction to be counter-trend:
        #   bullish engulfing -> valid only against a BEARISH trend
        #   bearish engulfing -> valid only against a BULLISH trend
        td = trend_direction.upper()
        setup.nison_1_trend = (
            (setup.direction == "bullish" and td == "BEARISH") or
            (setup.direction == "bearish" and td == "BULLISH")
        )
        self._check_ma_confluence(df, setup, atr_value)
        self._check_fib_confluence(setup, fib_levels, atr_value)
        self._check_sr_confluence(setup, sr_zones, atr_value)
        setup.confluence_count = sum([setup.confluence_ma, setup.confluence_fib, setup.confluence_sr])
        self._calculate_entries(setup, sr_zones, atr_value)
        self._score_quality(setup)
        return setup
    def _detect_engulfing(self, df):
        if len(df) < 2: return EngulfingSetup()
        c1, c2 = df.iloc[-2], df.iloc[-1]
        c1_o, c1_c = float(c1["open"]), float(c1["close"])
        c2_o, c2_c = float(c2["open"]), float(c2["close"])
        c2_h, c2_l = float(c2["high"]), float(c2["low"])
        c1_bh, c1_bl = max(c1_o, c1_c), min(c1_o, c1_c)
        c2_bh, c2_bl = max(c2_o, c2_c), min(c2_o, c2_c)
        engulfs = (c2_bh >= c1_bh and c2_bl <= c1_bl and abs(c2_c - c2_o) > abs(c1_c - c1_o))
        if not engulfs: return EngulfingSetup()
        c1_bull, c2_bull = c1_c > c1_o, c2_c > c2_o
        if c1_bull == c2_bull: return EngulfingSetup()
        direction = "bullish" if c2_bull else "bearish"
        try: ct = str(df.index[-1])
        except Exception: ct = ""
        return EngulfingSetup(detected=True, direction=direction, candle_index=len(df)-1, candle_time=ct,
            nison_2_engulf=True, nison_3_opposite=True, c1_open=c1_o, c1_close=c1_c, c2_open=c2_o, c2_close=c2_c, c2_high=c2_h, c2_low=c2_l)
    def _check_ma_confluence(self, df, setup, atr_value):
        if atr_value is None or atr_value <= 0: atr_value = setup.c2_close * 0.001
        prox = atr_value * MA_PROXIMITY_ATR_MULT
        for col in ["sma_8", "sma_21", "ema_21"]:
            if col in df.columns:
                v = float(df[col].iloc[-1])
                if not np.isnan(v) and abs(setup.c2_close - v) <= prox:
                    setup.confluence_ma = True; break
        if "sma_200" in df.columns:
            v = float(df["sma_200"].iloc[-1])
            if not np.isnan(v) and v > 0:
                if setup.direction == "bullish" and setup.c2_close > v: setup.confluence_200 = True
                elif setup.direction == "bearish" and setup.c2_close < v: setup.confluence_200 = True
    def _check_fib_confluence(self, setup, fib_levels, atr_value):
        if not fib_levels or atr_value is None: return
        if atr_value <= 0: atr_value = setup.c2_close * 0.001
        prox = atr_value * 0.5
        for name, price in fib_levels.items():
            if price and abs(setup.c2_close - float(price)) <= prox:
                if "50" in str(name) or "61" in str(name) or "618" in str(name):
                    setup.confluence_fib = True; setup.confluence_fib_level = str(name); break
    def _check_sr_confluence(self, setup, sr_zones, atr_value):
        if not sr_zones or atr_value is None: return
        if atr_value <= 0: atr_value = setup.c2_close * 0.001
        prox = atr_value * 0.5
        for z in sr_zones:
            zc = (float(z.get("zone_top", 0)) + float(z.get("zone_bottom", 0))) / 2
            if abs(setup.c2_close - zc) <= prox:
                setup.confluence_sr = True
                if z.get("role_reversal"): setup.confluence_role_reversal = True
                break
    def _calculate_entries(self, setup, sr_zones, atr_value):
        if not setup.detected: return
        if atr_value is None or atr_value <= 0: atr_value = setup.c2_close * 0.001
        setup.entry_price = setup.c2_close
        buf = atr_value * SL_BUFFER_ATR_MULT
        setup.stop_loss = setup.c2_low - buf if setup.direction == "bullish" else setup.c2_high + buf
        if setup.direction == "bullish":
            tps = [float(z.get("zone_bottom", 0)) for z in (sr_zones or []) if float(z.get("zone_bottom", 0)) > setup.entry_price]
            setup.take_profit = min(tps) if tps else setup.entry_price + atr_value * 3
        else:
            tps = [float(z.get("zone_top", 0)) for z in (sr_zones or []) if 0 < float(z.get("zone_top", 0)) < setup.entry_price]
            setup.take_profit = max(tps) if tps else setup.entry_price - atr_value * 3
        risk = abs(setup.entry_price - setup.stop_loss)
        setup.risk_reward = round(abs(setup.take_profit - setup.entry_price) / risk, 2) if risk > 0 else 0
    def _score_quality(self, setup):
        s = 0
        if setup.nison_1_trend: s += 30
        if setup.nison_2_engulf: s += 30
        if setup.nison_3_opposite: s += 30
        if setup.confluence_ma: s += 10
        if setup.confluence_200: s += 5
        if setup.confluence_fib: s += 10
        if setup.confluence_sr: s += 10
        if setup.confluence_role_reversal: s += 5
        setup.quality_score = min(100, s)
        setup.quality_grade = "A" if s >= 90 else "B" if s >= 70 else "C" if s >= 50 else "D" if s >= 30 else "F"
    def get_summary(self, setup):
        if not setup.detected: return "No engulfing bar detected."
        return f"=== ENGULFING BAR ({self.timeframe}) ===\nDirection: {setup.direction.upper()}\nNison: T={setup.nison_1_trend} E={setup.nison_2_engulf} O={setup.nison_3_opposite}\nConfluence: MA={setup.confluence_ma} Fib={setup.confluence_fib} SR={setup.confluence_sr} ({setup.confluence_count} total)\nEntry={setup.entry_price:.5f} SL={setup.stop_loss:.5f} TP={setup.take_profit:.5f} R:R=1:{setup.risk_reward}\nQuality: {setup.quality_score}/100 (Grade: {setup.quality_grade})"