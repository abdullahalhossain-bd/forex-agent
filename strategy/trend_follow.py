import logging
import math

logger = logging.getLogger("strategies.trend_follow")

# Standard pip sizes (mirrors risk/atr_risk_manager.py's PIP_SIZES so the two
# don't silently drift apart). Kept local (not imported) to avoid a hard
# dependency of strategies/ on risk/ -- these are static market conventions,
# not shared mutable state.
_PIP_SIZES = {"JPY": 0.01, "DEFAULT": 0.0001}


def _pip_size_for(pair: str) -> float:
    """Pair-aware pip size (0.01 for JPY crosses, 0.0001 otherwise)."""
    return _PIP_SIZES["JPY"] if "JPY" in pair.upper() else _PIP_SIZES["DEFAULT"]


def _safe_float(row, key: str, default: float = 0.0) -> float:
    """
    Extract a float from a row, treating missing keys AND NaN as `default`.
    Plain `float(row.get(key, default) or default)` lets a NaN survive
    (NaN is truthy), which then silently propagates into stop/target math.
    """
    value = row.get(key, default) if hasattr(row, "get") else default
    if value is None:
        return default
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(value) else value


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

    def generate(self, history, pair=None):
        if len(history) < self.warmup:
            return self._hold()

        last = history.iloc[-1]
        atr = _safe_float(last, "atr", 0.0)
        if atr <= 0:
            return self._hold()

        ema_9 = _safe_float(last, "ema_9")
        ema_21 = _safe_float(last, "ema_21")
        sma_50 = _safe_float(last, "sma_50")
        sma_200 = _safe_float(last, "sma_200")
        close = _safe_float(last, "close")
        macd = _safe_float(last, "macd")
        macd_signal = _safe_float(last, "macd_signal")

        # Required fields missing/NaN -> can't safely evaluate alignment.
        if any(math.isnan(v) for v in (ema_9, ema_21, sma_50, sma_200, close, macd, macd_signal)):
            logger.debug("TrendFollowStrategy HOLD: required field missing/NaN")
            return self._hold()

        bullish_alignment = ema_9 > ema_21 > sma_50 and close > sma_200
        bearish_alignment = ema_9 < ema_21 < sma_50 and close < sma_200
        adx_ok = _safe_float(last, "adx", 0.0) >= self.adx_min
        pullback_ok = abs(close - ema_21) <= atr * self.pullback_atr_mult

        if bullish_alignment and adx_ok and pullback_ok and macd > macd_signal:
            return self._signal("BUY", last, "Bull trend pullback + MACD confirmation", pair)

        if bearish_alignment and adx_ok and pullback_ok and macd < macd_signal:
            return self._signal("SELL", last, "Bear trend pullback + MACD confirmation", pair)

        return self._hold()

    def _signal(self, direction, last, reason: str, pair=None):
        trend_strength = min(max((_safe_float(last, "adx", 20.0) - self.adx_min) * 1.5, 0), 20)
        confidence = int(min(60 + trend_strength + 10, 90))
        return {
            "signal": direction,
            "confidence": confidence,
            "reason": reason,
            "pattern": self._pattern(last),
            "regime": last.get("regime", "TRENDING"),
            "session": last.get("session_name", "unknown"),
            "rr_ratio": self.rr_ratio,
            "stop_pips": self._stop_pips(last, pair),
            "strategy_name": self.name,
            "strategy_version": self.version,
        }

    def _stop_pips(self, last, pair=None) -> float:
        """
        Stop distance in pips, derived from ATR.

        P1 fix (audit vs. Courtney Smith money-management concepts): this
        used to guess pip size from raw price magnitude
        (`100 if close > 20 else 10000`), which is wrong for gold, JPY
        pairs, and indices -- it silently distorts the risk-per-trade
        calculation in backtest/simulator.py and backtest/engine.py, both
        of which consume this field directly for lot sizing.

        If the caller supplies `pair` (e.g. "EURJPY", "XAUUSD"), pip size
        is now looked up correctly (mirrors risk/atr_risk_manager.py's
        `_get_pip_size`, kept in sync deliberately). If no pair is given
        -- e.g. older call sites not yet updated -- this falls back to the
        historical heuristic, but logs a warning instead of guessing
        silently, so a wrong value is at least visible in the logs rather
        than invisibly biasing every backtest run.
        """
        atr = _safe_float(last, "atr", 0.001)
        if atr <= 0:
            atr = 0.001

        if pair:
            pip_size = _pip_size_for(pair)
            pip = 1.0 / pip_size
        else:
            logger.warning(
                "TrendFollowStrategy._stop_pips: no `pair` supplied, "
                "falling back to price-magnitude heuristic for pip size. "
                "Pass `pair` into generate() to fix this properly."
            )
            close = _safe_float(last, "close", 0.0)
            pip = 100 if close > 20 else 10000

        return max(round(atr * self.stop_atr_mult * pip, 1), 8.0)

    def _pattern(self, last) -> str:
        for key in ("engulfing", "star_pattern", "pattern"):
            value = last.get(key, "none")
            if value and value != "none":
                return value
        return "trend_pullback"

    def _hold(self):
        return {"signal": "HOLD", "confidence": 0}
