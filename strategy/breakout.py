"""
Breakout Expansion strategy — institutional-grade implementation.

CONTRACT (read before wiring this into an execution loop):
- `history` MUST contain only fully CLOSED bars. The last row (`history.iloc[-1]`)
  is treated as the most recent CLOSED bar. If your data pipeline still includes
  the currently-forming bar as the last row, you WILL get look-ahead bias /
  repainting — the fix belongs at the data-ingestion boundary, not here.
- `rolling_resistance_20` / `rolling_support_20` MUST be computed using a window
  that EXCLUDES the current bar (e.g. `high.shift(1).rolling(20).max()`), or the
  breakout comparison is self-referential (a bar can never break a level derived
  from itself). This module cannot verify that upstream — it's flagged here so
  it isn't silently assumed correct.
- This module returns a SIGNAL and price-based risk levels. It does NOT do
  position sizing, pip/lot conversion, or order placement. Position sizing must
  be computed by a separate risk-engine component using account equity, risk %,
  and the broker's actual pip/point value for the traded symbol (see
  `SymbolInfoDouble` / `symbol_info(symbol).point` in MT5) — never approximate
  pip size from price magnitude (see MarketDataError / removed `_stop_pips`
  heuristic below).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger("strategies.breakout_expansion")


class MarketDataError(Exception):
    """Raised when required market data is missing, malformed, or NaN."""


@dataclass(frozen=True)
class BreakoutConfig:
    """Configuration for BreakoutStrategy. Validated at construction time."""

    volume_ratio_min: float = 1.2
    adx_min: float = 18.0
    stop_atr_mult: float = 1.3
    rr_ratio: float = 2.0
    # Confirmation buffer: close must clear the level by this many ATRs,
    # not just by an epsilon. Reduces marginal/noise breakouts and partially
    # mitigates liquidity-sweep false signals (see trading-concepts: Liquidity).
    breakout_buffer_atr: float = 0.10
    # Minimum bars that must pass between two signals in the SAME direction
    # off what is effectively the same level, to avoid re-firing on every
    # bar of an ongoing expansion move. Enforced by the caller via
    # `last_signal_bar_index` passed into `generate`.
    min_bars_between_signals: int = 3

    def __post_init__(self) -> None:
        if self.volume_ratio_min <= 0:
            raise ValueError("volume_ratio_min must be > 0")
        if self.adx_min < 0 or self.adx_min > 100:
            raise ValueError("adx_min must be in [0, 100]")
        if self.stop_atr_mult <= 0:
            raise ValueError("stop_atr_mult must be > 0")
        if self.rr_ratio <= 0:
            raise ValueError("rr_ratio must be > 0")
        if self.breakout_buffer_atr < 0:
            raise ValueError("breakout_buffer_atr must be >= 0")
        if self.min_bars_between_signals < 0:
            raise ValueError("min_bars_between_signals must be >= 0")


def _safe_float(row: Any, key: str, default: Optional[float] = None, required: bool = False) -> float:
    """
    Extract a float from a pandas Series/dict-like row, treating missing keys
    AND NaN values as equivalent (unlike `row.get(key, default) or default`,
    which lets a NaN survive because NaN is truthy in Python — this was a
    critical hidden bug: a NaN ATR would pass `atr <= 0` unchanged, since any
    comparison against NaN is False, and later propagate into stop-distance
    math as NaN).

    Raises MarketDataError if `required` and the value is missing/NaN.
    """
    if key not in row or row[key] is None:
        value = default
    else:
        value = row[key]
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise MarketDataError(f"Field '{key}' is not numeric: {value!r}") from exc
        if math.isnan(value):
            value = default

    if value is None:
        if required:
            raise MarketDataError(f"Required field '{key}' is missing or NaN")
        return float("nan")
    return float(value)


class BreakoutStrategy:
    """
    Volatility-expansion breakout strategy.

    Entry logic: close breaks beyond a 20-bar rolling high/low by at least
    `breakout_buffer_atr` ATRs, confirmed by above-average tick volume and a
    minimum ADX (trend-strength) filter.

    NOTE ON SCOPE: this is a momentum/volatility breakout system, not an
    ICT/SMC or Wyckoff implementation — it does not model order blocks, FVGs,
    liquidity sweeps, or Wyckoff phases. If those confirmations are wanted,
    they should be added as additional filters (see class docstring "Known
    gaps" below) rather than assumed present.

    Known gaps vs. an institutional breakout system (intentionally not solved
    here because they require data/state this module doesn't own — documented
    so they aren't silently missed):
      - No liquidity-sweep / stop-hunt filter (a wick-through-then-reverse
        pattern immediately before the breakout bar would invalidate many of
        these signals; not detectable from a single row).
      - No HTF trend filter / multi-timeframe bias confirmation.
      - No session/kill-zone filter, despite carrying `session_name` through.
      - No spread/news-proximity gate — a live wrapper must check spread and
        upcoming high-impact calendar events before acting on this signal.
      - No portfolio-level exposure/correlation check (belongs in the risk
        engine, not the signal generator).
    """

    name = "Breakout Expansion"
    version = "v2"
    warmup = 60

    def __init__(self, config: Optional[BreakoutConfig] = None, **kwargs: Any) -> None:
        """
        Args:
            config: validated BreakoutConfig. If omitted, built from kwargs
                (kept for backward compatibility with the v1 constructor
                signature: volume_ratio_min, adx_min, stop_atr_mult, rr_ratio).
        """
        self.config = config or BreakoutConfig(**kwargs)

    def generate(
        self,
        history: Any,
        bars_since_last_signal: Optional[int] = None,
    ) -> dict:
        """
        Generate a signal from the most recent CLOSED bar in `history`.

        Args:
            history: OHLCV + indicator DataFrame; last row must be a closed bar.
            bars_since_last_signal: bars elapsed since this strategy last fired
                on this symbol, supplied by the caller (this module is
                intentionally stateless/pure for testability). Used to enforce
                `min_bars_between_signals` and avoid re-firing every bar of an
                ongoing expansion move. Pass None to disable the check.

        Returns:
            Signal dict. On HOLD, includes a `reason` for observability.

        Raises:
            MarketDataError: if required fields (`close`) are missing or NaN.
                This is intentionally NOT swallowed into a HOLD — a missing
                `close` means the data pipeline is broken and the caller
                should halt/alert rather than silently trade on partial data.
        """
        if len(history) < self.warmup:
            return self._hold(f"insufficient history: {len(history)} < warmup {self.warmup}")

        last = history.iloc[-1]

        close = _safe_float(last, "close", required=True)
        atr = _safe_float(last, "atr", default=0.0)
        if math.isnan(atr) or atr <= 0:
            return self._hold("atr missing, NaN, or non-positive")

        high_break = _safe_float(last, "rolling_resistance_20", default=close)
        low_break = _safe_float(last, "rolling_support_20", default=close)
        volume_ratio = _safe_float(last, "volume_ratio", default=1.0)
        adx = _safe_float(last, "adx", default=0.0)

        if math.isnan(high_break) or math.isnan(low_break) or math.isnan(volume_ratio) or math.isnan(adx):
            return self._hold("one or more filter inputs is NaN after fallback")

        if bars_since_last_signal is not None and bars_since_last_signal < self.config.min_bars_between_signals:
            return self._hold(
                f"cooldown active: {bars_since_last_signal} < {self.config.min_bars_between_signals} bars"
            )

        buffer_price = self.config.breakout_buffer_atr * atr
        volume_ok = volume_ratio >= self.config.volume_ratio_min
        trend_ok = adx >= self.config.adx_min

        bullish_breakout = close > (high_break + buffer_price) and volume_ok and trend_ok
        bearish_breakout = close < (low_break - buffer_price) and volume_ok and trend_ok

        if bullish_breakout:
            return self._signal("BUY", last, close, atr, "Resistance breakout + strong volume")
        if bearish_breakout:
            return self._signal("SELL", last, close, atr, "Support breakdown + strong volume")

        return self._hold("no breakout condition met")

    def _signal(self, direction: str, last: Any, close: float, atr: float, reason: str) -> dict:
        volume_ratio = _safe_float(last, "volume_ratio", default=1.0)
        adx = _safe_float(last, "adx", default=0.0)

        # Confidence is a bounded heuristic score, NOT a calibrated
        # probability of the trade winning — must not be fed into position
        # sizing as if it were a win-rate estimate.
        raw_confidence = 62.0 + volume_ratio * 8.0 + max(adx - self.config.adx_min, 0.0)
        confidence = int(max(0.0, min(raw_confidence, 90.0)))

        stop_distance = atr * self.config.stop_atr_mult
        target_distance = stop_distance * self.config.rr_ratio

        if direction == "BUY":
            stop_price = close - stop_distance
            target_price = close + target_distance
        else:
            stop_price = close + stop_distance
            target_price = close - target_distance

        return {
            "signal": direction,
            "confidence": confidence,
            "reason": reason,
            "pattern": self._pattern(last),
            "regime": self._regime(last),
            "session": last.get("session_name", "unknown"),
            "entry_price": close,
            "stop_price": round(stop_price, 8),
            "target_price": round(target_price, 8),
            "stop_distance_price": round(stop_distance, 8),
            "rr_ratio": self.config.rr_ratio,
            "strategy_name": self.name,
            "strategy_version": self.version,
        }

    @staticmethod
    def _pattern(last: Any) -> str:
        pattern = last.get("pattern", None)
        if pattern is None or (isinstance(pattern, float) and math.isnan(pattern)):
            return "breakout"
        pattern = str(pattern)
        return pattern if pattern and pattern != "none" else "breakout"

    @staticmethod
    def _regime(last: Any) -> str:
        regime = last.get("regime", None)
        if regime is None or (isinstance(regime, float) and math.isnan(regime)):
            return "UNKNOWN"
        return str(regime)

    @staticmethod
    def _hold(reason: str) -> dict:
        logger.debug("BreakoutStrategy HOLD: %s", reason)
        return {"signal": "HOLD", "confidence": 0, "reason": reason}