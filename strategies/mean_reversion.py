"""
strategies/mean_reversion.py — Bollinger Band + RSI mean-reversion strategy
=============================================================================

Concept:
  - Only trade when ADX indicates a RANGING market (mean reversion is a
    counter-trend strategy; trading it during a strong trend is fading the
    trend, i.e. the single riskiest thing this strategy can do).
  - Entry: price closes at/beyond a Bollinger Band extreme, RSI confirms
    overbought/oversold, and the entry candle itself shows an intra-bar
    rejection back toward the mean.
  - Target: the Bollinger middle band (the "mean" being reverted to).
  - Stop: ATR-multiple beyond entry (added in this rewrite — see §Hidden
    Bugs in the accompanying review; v1 accepted `stop_atr_mult` in its
    constructor and never used it anywhere, so v1 produced signals with a
    reward target and NO defined risk).

CONTRACT (same as the other strategies in this suite):
- `history` MUST contain only fully CLOSED bars; `history.iloc[-1]` is
  treated as the most recent CLOSED bar.
- `rsi`, `adx`, `bb_upper/middle/lower`, `atr` are assumed precomputed
  upstream using standard (closed-bar) formulas. This module cannot verify
  that — flagged so it isn't silently assumed correct.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

from utils.logger import get_logger

log = get_logger("mean_reversion_strategy")


class MarketDataError(Exception):
    """Raised when required OHLC fields are missing, malformed, or NaN."""


def _safe_float(row: Any, key: str, default: Optional[float] = None, required: bool = False) -> float:
    """
    NaN-safe field extraction. v1's pattern of `float(row.get(key, d) or d)`
    is broken: if the value is NaN, `NaN or d` evaluates to NaN (NaN is
    truthy in Python), so the "or default" fallback never actually applies
    to NaN — only to falsy values like 0/None. That bug is present in ALL
    THREE strategy files reviewed so far in this suite (breakout, ema_rsi_combo,
    and this one) — strongly suggests they share a common origin/template.
    Recommend extracting this helper into a shared `strategies/_common.py`
    used by every strategy module, so it's fixed once instead of re-broken
    independently in strategy #4, #5, etc. Kept local here to keep this
    deliverable self-contained; say the word and I'll pull it out.
    """
    if key not in row or row[key] is None:
        value = default
    else:
        try:
            value = float(row[key])
        except (TypeError, ValueError) as exc:
            raise MarketDataError(f"Field '{key}' is not numeric: {row[key]!r}") from exc
        if math.isnan(value):
            value = default

    if value is None:
        if required:
            raise MarketDataError(f"Required field '{key}' is missing or NaN")
        return float("nan")
    return float(value)


@dataclass(frozen=True)
class MeanReversionConfig:
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    max_adx: float = 25.0
    stop_atr_mult: float = 1.5
    # Minimum acceptable reward:risk on the actual measured distance to the
    # middle band vs the ATR-based stop. v1 carried an `rr_ratio` field in
    # its output but never enforced or even measured it — the middle band
    # could be a few pips away (band squeeze) while the stop is much wider,
    # producing a trade with terrible realized RR despite a "2.0" rr_ratio
    # sitting unused in the config. This is now actually enforced.
    min_rr: float = 1.0
    min_bars_between_signals: int = 3

    def __post_init__(self) -> None:
        if not (0 < self.rsi_oversold < 50 < self.rsi_overbought < 100):
            raise ValueError("RSI thresholds must satisfy 0 < oversold < 50 < overbought < 100")
        if not (0 < self.max_adx <= 100):
            raise ValueError("max_adx must be in (0, 100]")
        if self.stop_atr_mult <= 0:
            raise ValueError("stop_atr_mult must be > 0")
        if self.min_rr <= 0:
            raise ValueError("min_rr must be > 0")
        if self.min_bars_between_signals < 0:
            raise ValueError("min_bars_between_signals must be >= 0")


class MeanReversionStrategy:
    """
    Bollinger Band + RSI mean-reversion, gated by an ADX ranging-market filter.

    Known gaps vs. an institutional mean-reversion system (documented, not
    silently assumed solved):
      - No band-width/squeeze filter — a touch of a nearly-flat band in a
        squeeze is much less meaningful than a touch of a wide band, and
        this module can't currently tell the two apart.
      - No volume confirmation on the reversion candle (the breakout
        strategy in this suite has one; this one has none).
      - No check for extreme excursions beyond the band (e.g. a close 3+
        ATR past the band is more likely a breakout/trend continuation than
        a reversion opportunity — blindly fading it is the classic way a
        mean-reversion system takes its worst loss).
      - No RSI-divergence confirmation — this uses only the absolute
        overbought/oversold level, not divergence against price, which is
        the more robust institutional version of this filter.
      - Same-bar touch + same-bar rejection confirmation (see Logic
        Problems in the review) is weaker than a next-bar confirmation.
    """

    name = "Mean Reversion"
    version = "v2"
    warmup = 50

    def __init__(self, config: Optional[MeanReversionConfig] = None, **kwargs: Any) -> None:
        """`config` preferred; kwargs kept for backward compatibility with
        the v1 constructor signature (rsi_oversold, rsi_overbought, max_adx,
        stop_atr_mult, rr_ratio -> mapped to min_rr)."""
        if config is not None:
            self.config = config
        else:
            if "rr_ratio" in kwargs:
                kwargs["min_rr"] = kwargs.pop("rr_ratio")
            self.config = MeanReversionConfig(**kwargs)

    def generate(self, history: Any, bars_since_last_signal: Optional[int] = None) -> dict:
        """
        Generate a signal from the most recent CLOSED bar in `history`.

        Raises:
            MarketDataError: if required OHLC fields are missing/NaN — this
                propagates rather than being silently absorbed into a HOLD,
                since it indicates a broken data pipeline, not "no setup".
        """
        if len(history) < self.warmup:
            return self._hold(f"insufficient history: {len(history)} < warmup {self.warmup}")

        last = history.iloc[-1]

        close = _safe_float(last, "close", required=True)
        high = _safe_float(last, "high", required=True)
        low = _safe_float(last, "low", required=True)
        open_p = _safe_float(last, "open", required=True)
        atr = _safe_float(last, "atr", default=0.0)

        if math.isnan(atr) or atr <= 0:
            return self._hold("atr missing, NaN, or non-positive")

        rsi = _safe_float(last, "rsi", default=50.0)
        adx = _safe_float(last, "adx", default=float("nan"))
        bb_u = _safe_float(last, "bb_upper", default=close)
        bb_l = _safe_float(last, "bb_lower", default=close)
        bb_m = _safe_float(last, "bb_middle", default=close)

        # v1 bug: `float(last.get("adx", 0) or 0)` let a NaN adx SURVIVE the
        # `or 0` fallback (NaN is truthy), so `adx >= max_adx` was False for
        # NaN and the trend-strength filter — the single most important
        # safety check for a counter-trend strategy — silently did nothing
        # when ADX data was missing/corrupted. Now an unusable ADX blocks
        # the trade explicitly instead of defaulting to "safe to trade".
        if math.isnan(adx):
            return self._hold("adx missing or NaN — cannot confirm ranging market, refusing to trade blind")
        if adx >= self.config.max_adx:
            return self._hold(f"adx too high for mean reversion: {adx:.1f} >= {self.config.max_adx:.1f}")

        if math.isnan(rsi) or math.isnan(bb_u) or math.isnan(bb_l) or math.isnan(bb_m):
            return self._hold("one or more filter inputs is NaN after fallback")

        if bars_since_last_signal is not None and bars_since_last_signal < self.config.min_bars_between_signals:
            return self._hold(
                f"cooldown active: {bars_since_last_signal} < {self.config.min_bars_between_signals} bars"
            )

        at_upper = close >= bb_u
        at_lower = close <= bb_l
        mid = (high + low) / 2.0
        bullish_rejection = close > open_p and close > mid
        bearish_rejection = close < open_p and close < mid

        if at_lower and rsi <= self.config.rsi_oversold and bullish_rejection:
            return self._signal("BUY", close, bb_m, atr, adx, rsi, f"Mean reversion BUY at lower BB, RSI={rsi:.1f}")
        if at_upper and rsi >= self.config.rsi_overbought and bearish_rejection:
            return self._signal("SELL", close, bb_m, atr, adx, rsi, f"Mean reversion SELL at upper BB, RSI={rsi:.1f}")

        return self._hold("no mean-reversion setup at close")

    def _signal(
        self, direction: str, close: float, bb_m: float, atr: float, adx: float, rsi: float, reason: str
    ) -> dict:
        stop_distance = atr * self.config.stop_atr_mult
        reward_distance = abs(bb_m - close)

        if direction == "BUY":
            stop_price = close - stop_distance
            target_price = max(bb_m, close)  # never target below current price on a BUY
        else:
            stop_price = close + stop_distance
            target_price = min(bb_m, close)  # never target above current price on a SELL

        realized_rr = reward_distance / stop_distance if stop_distance > 0 else 0.0
        if realized_rr < self.config.min_rr:
            # v1 carried an `rr_ratio` field but never measured or enforced
            # it against the actual band distance — a near-flat/squeezed
            # band could sit a few pips from price while the ATR-based stop
            # is much wider, producing a trade with terrible real RR despite
            # a "2.0" sitting unused in the config. Now enforced.
            return self._hold(
                f"reward:risk too small: measured {realized_rr:.2f} < min {self.config.min_rr:.2f} "
                f"(target distance {reward_distance:.5f} vs stop distance {stop_distance:.5f})"
            )

        # v1's `_signal` computed confidence from a HARDCODED local
        # `adx = 20`, completely ignoring the real adx (and rsi) passed
        # into it — every single signal got the exact same confidence
        # value regardless of how oversold/overbought RSI was or how weak
        # the trend actually was. Confidence now scales with BOTH real
        # inputs. Still a bounded heuristic, not a calibrated probability —
        # do not feed into position sizing as a win-rate estimate.
        adx_slack = max(self.config.max_adx - adx, 0.0)
        if direction == "BUY":
            rsi_extremity = max(self.config.rsi_oversold - rsi, 0.0)
        else:
            rsi_extremity = max(rsi - self.config.rsi_overbought, 0.0)
        confidence = int(max(0.0, min(55.0 + adx_slack * 1.5 + rsi_extremity * 0.5, 80.0)))

        return {
            "signal": direction,
            "confidence": confidence,
            "reason": reason,
            "pattern": "mean_reversion",
            "entry_price": close,
            "stop_price": round(stop_price, 8),
            "target_price": round(target_price, 8),
            "stop_distance_price": round(stop_distance, 8),
            "rr_ratio": round(realized_rr, 2),
            "regime": "RANGE",
            "strategy_name": self.name,
            "strategy_version": self.version,
        }

    @staticmethod
    def _hold(reason: str) -> dict:
        log.debug("MeanReversionStrategy HOLD: %s", reason)
        return {"signal": "HOLD", "confidence": 0, "reason": reason}