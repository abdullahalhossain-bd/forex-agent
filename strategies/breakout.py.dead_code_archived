class BreakoutStrategy:
    name = "Breakout Expansion"
    version = "v1"
    warmup = 60

    def __init__(
        self,
        volume_ratio_min: float = 1.2,
        adx_min: float = 18,
        stop_atr_mult: float = 1.3,
        rr_ratio: float = 2.0,
    ):
        self.volume_ratio_min = volume_ratio_min
        self.adx_min = adx_min
        self.stop_atr_mult = stop_atr_mult
        self.rr_ratio = rr_ratio

    def generate(self, history):
        if len(history) < self.warmup:
            return self._hold()

        last = history.iloc[-1]
        atr = float(last.get("atr", 0) or 0)
        if atr <= 0:
            return self._hold()

        close = float(last["close"])
        high_break = float(last.get("rolling_resistance_20", close) or close)
        low_break = float(last.get("rolling_support_20", close) or close)
        volume_ratio = float(last.get("volume_ratio", 1.0) or 1.0)
        adx = float(last.get("adx", 0) or 0)

        bullish_breakout = close > high_break and volume_ratio >= self.volume_ratio_min and adx >= self.adx_min
        bearish_breakout = close < low_break and volume_ratio >= self.volume_ratio_min and adx >= self.adx_min

        if bullish_breakout:
            return self._signal("BUY", last, "Resistance breakout + strong volume")
        if bearish_breakout:
            return self._signal("SELL", last, "Support breakdown + strong volume")
        return self._hold()

    def _signal(self, direction, last, reason: str):
        confidence = int(min(62 + float(last.get("volume_ratio", 1)) * 8 + max(float(last.get("adx", 0)) - self.adx_min, 0), 90))
        return {
            "signal": direction,
            "confidence": confidence,
            "reason": reason,
            "pattern": self._pattern(last),
            "regime": last.get("regime", "BREAKOUT"),
            "session": last.get("session_name", "unknown"),
            "rr_ratio": self.rr_ratio,
            "stop_pips": self._stop_pips(last),
            "strategy_name": self.name,
            "strategy_version": self.version,
        }

    def _stop_pips(self, last) -> float:
        atr = float(last.get("atr", 0.001) or 0.001)
        pip = 100 if float(last.get("close", 0)) > 20 else 10000
        return max(round(atr * self.stop_atr_mult * pip, 1), 10.0)

    def _pattern(self, last) -> str:
        pattern = last.get("pattern", "none")
        return pattern if pattern and pattern != "none" else "breakout"

    def _hold(self):
        return {"signal": "HOLD", "confidence": 0}
