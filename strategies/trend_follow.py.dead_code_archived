class TrendFollowStrategy:
    name = "Trend Pullback"
    version = "v1"
    warmup = 220

    def __init__(
        self,
        adx_min: float = 20,
        pullback_atr_mult: float = 0.8,
        stop_atr_mult: float = 1.4,
        rr_ratio: float = 2.4,
    ):
        self.adx_min = adx_min
        self.pullback_atr_mult = pullback_atr_mult
        self.stop_atr_mult = stop_atr_mult
        self.rr_ratio = rr_ratio

    def generate(self, history):
        if len(history) < self.warmup:
            return self._hold()

        last = history.iloc[-1]
        atr = float(last.get("atr", 0) or 0)
        if atr <= 0:
            return self._hold()

        bullish_alignment = (
            float(last["ema_9"]) > float(last["ema_21"]) > float(last["sma_50"])
            and float(last["close"]) > float(last["sma_200"])
        )
        bearish_alignment = (
            float(last["ema_9"]) < float(last["ema_21"]) < float(last["sma_50"])
            and float(last["close"]) < float(last["sma_200"])
        )
        adx_ok = float(last.get("adx", 0) or 0) >= self.adx_min
        pullback_ok = abs(float(last["close"]) - float(last["ema_21"])) <= atr * self.pullback_atr_mult

        if bullish_alignment and adx_ok and pullback_ok and float(last["macd"]) > float(last["macd_signal"]):
            return self._signal("BUY", last, "Bull trend pullback + MACD confirmation")

        if bearish_alignment and adx_ok and pullback_ok and float(last["macd"]) < float(last["macd_signal"]):
            return self._signal("SELL", last, "Bear trend pullback + MACD confirmation")

        return self._hold()

    def _signal(self, direction, last, reason: str):
        trend_strength = min(max((float(last.get("adx", 20)) - self.adx_min) * 1.5, 0), 20)
        confidence = int(min(60 + trend_strength + 10, 90))
        return {
            "signal": direction,
            "confidence": confidence,
            "reason": reason,
            "pattern": self._pattern(last),
            "regime": last.get("regime", "TRENDING"),
            "session": last.get("session_name", "unknown"),
            "rr_ratio": self.rr_ratio,
            "stop_pips": self._stop_pips(last),
            "strategy_name": self.name,
            "strategy_version": self.version,
        }

    def _stop_pips(self, last) -> float:
        atr = float(last.get("atr", 0.001) or 0.001)
        pip = 100 if float(last.get("close", 0)) > 20 else 10000
        return max(round(atr * self.stop_atr_mult * pip, 1), 8.0)

    def _pattern(self, last) -> str:
        for key in ("engulfing", "star_pattern", "pattern"):
            value = last.get(key, "none")
            if value and value != "none":
                return value
        return "trend_pullback"

    def _hold(self):
        return {"signal": "HOLD", "confidence": 0}
