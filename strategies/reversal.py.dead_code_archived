class ReversalStrategy:
    name = "Support Reversal"
    version = "v1"
    warmup = 80

    def __init__(
        self,
        rsi_oversold: float = 32,
        rsi_overbought: float = 68,
        sr_atr_tolerance: float = 0.6,
        stop_atr_mult: float = 1.2,
        rr_ratio: float = 1.8,
    ):
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.sr_atr_tolerance = sr_atr_tolerance
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
        support = float(last.get("rolling_support_20", close) or close)
        resistance = float(last.get("rolling_resistance_20", close) or close)
        near_support = abs(close - support) <= atr * self.sr_atr_tolerance
        near_resistance = abs(resistance - close) <= atr * self.sr_atr_tolerance
        pattern = self._pattern(last)
        bullish_pattern = pattern in {"hammer", "bullish_pin_bar", "bullish_engulfing", "morning_star"}
        bearish_pattern = pattern in {"shooting_star", "bearish_pin_bar", "bearish_engulfing", "evening_star"}
        adx = float(last.get("adx", 0) or 0)

        if float(last.get("rsi", 50)) <= self.rsi_oversold and near_support and bullish_pattern and adx < 32:
            return self._signal("BUY", last, f"RSI oversold + support + {pattern}")

        if float(last.get("rsi", 50)) >= self.rsi_overbought and near_resistance and bearish_pattern and adx < 32:
            return self._signal("SELL", last, f"RSI overbought + resistance + {pattern}")

        return self._hold()

    def _signal(self, direction, last, reason: str):
        rsi = float(last.get("rsi", 50) or 50)
        edge = abs(50 - rsi)
        confidence = int(min(58 + edge * 0.7, 86))
        return {
            "signal": direction,
            "confidence": confidence,
            "reason": reason,
            "pattern": self._pattern(last),
            "regime": last.get("regime", "RANGING"),
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
        return "none"

    def _hold(self):
        return {"signal": "HOLD", "confidence": 0}
