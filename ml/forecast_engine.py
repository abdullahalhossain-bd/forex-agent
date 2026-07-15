"""
ml/forecast_engine.py — Day 97 Conservative Time-Series Forecast
=================================================================
Generates a SHORT-TERM price forecast as an EXTRA VOTE (10% weight),
never as a primary signal. This follows the user's principle:

  ❌ "TimeGPT said price will go up → BUY"     ← WRONG usage
  ✅ "Forecast is slightly bullish → +5% confidence boost" ← CORRECT

The forecast is computed from:
  1. EMA crossover momentum (fast vs slow)
  2. RSI mean-reversion tendency
  3. Recent candle body direction
  4. ATR-based expected range

Output:
    {
      "direction":      "BULLISH" | "BEARISH" | "NEUTRAL",
      "confidence":     61,          # 0-100
      "expected_range": "1.0850-1.0890",
      "next_candles":   "slightly bullish",
      "weight":         0.10,        # how much to weight this in fusion
    }

Usage:
    from ml.forecast_engine import ForecastEngine
    engine = ForecastEngine()
    result = engine.forecast(df)
    # result["direction"] → extra vote for MasterDecisionEngine

IMPORTANT: This is intentionally simple (no neural net / no Nixtla).
A complex forecast model would add noise + false confidence. The
goal is a LIGHTWEIGHT, conservative signal that:
  - Agrees with trend most of the time (low information)
  - Flags divergence occasionally (when momentum + RSI disagree)
  - Never overrides the primary SMC + ML + LLM signals
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("forecast_engine")


class ForecastEngine:
    """Conservative short-term price forecast (extra vote only)."""

    # Weight in MasterDecisionEngine fusion (per user's spec)
    FORECAST_WEIGHT = 0.10

    # Lookback periods
    FAST_EMA = 9
    SLOW_EMA = 21
    RSI_PERIOD = 14
    ATR_PERIOD = 14

    def forecast(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate a short-term forecast from OHLCV data.

        Args:
            df: DataFrame with columns: open, high, low, close, volume
                (ideally with ema_9, ema_21, rsi, atr already computed)

        Returns: dict with direction, confidence, expected_range, etc.
        """
        if df is None or len(df) < 30:
            return self._fallback("Insufficient data for forecast")

        try:
            close = df["close"]
            high = df["high"]
            low = df["low"]

            # ── 1. EMA momentum ──
            ema_fast = self._get_or_compute(df, f"ema_{self.FAST_EMA}",
                                              self._compute_ema, close, self.FAST_EMA)
            ema_slow = self._get_or_compute(df, f"ema_{self.SLOW_EMA}",
                                              self._compute_ema, close, self.SLOW_EMA)

            ema_spread = 0
            if ema_fast is not None and ema_slow is not None and ema_slow.iloc[-1] != 0:
                ema_spread = (ema_fast.iloc[-1] - ema_slow.iloc[-1]) / ema_slow.iloc[-1] * 100

            # ── 2. RSI ──
            rsi = self._get_or_compute(df, "rsi", self._compute_rsi, close)
            rsi_value = rsi.iloc[-1] if rsi is not None else 50.0

            # ── 3. Recent candle direction ──
            recent_bodies = (close.iloc[-5:] - df["open"].iloc[-5:])
            bullish_candles = (recent_bodies > 0).sum()
            bearish_candles = (recent_bodies < 0).sum()
            body_direction = bullish_candles - bearish_candles  # -5 to +5

            # ── 4. ATR for expected range ──
            atr = self._get_or_compute(df, "atr", self._compute_atr, high, low, close)
            atr_value = atr.iloc[-1] if atr is not None else 0.0010

            # ── Compute forecast score ──
            # EMA momentum: +1 if fast > slow, -1 if fast < slow
            ema_score = 1 if ema_spread > 0 else -1 if ema_spread < 0 else 0

            # RSI: >55 = bullish bias, <45 = bearish bias, 45-55 = neutral
            if rsi_value > 55:
                rsi_score = 1
            elif rsi_value < 45:
                rsi_score = -1
            else:
                rsi_score = 0

            # Body direction: +1 if more bullish candles, -1 if more bearish
            body_score = 1 if body_direction > 1 else -1 if body_direction < -1 else 0

            # Combined score: -3 to +3
            total_score = ema_score + rsi_score + body_score

            # Direction
            if total_score >= 2:
                direction = "BULLISH"
                label = "slightly bullish" if total_score == 2 else "bullish"
            elif total_score <= -2:
                direction = "BEARISH"
                label = "slightly bearish" if total_score == -2 else "bearish"
            else:
                direction = "NEUTRAL"
                label = "neutral / sideways"

            # Confidence: base 50, +10 per agreeing signal, capped at 80
            confidence = 50 + abs(total_score) * 10
            confidence = min(80, confidence)

            # Expected range for next 10 candles
            current_price = float(close.iloc[-1])
            range_half = atr_value * 2  # 2x ATR for 10-candle range
            expected_low = current_price - range_half
            expected_high = current_price + range_half

            result = {
                "direction":       direction,
                "confidence":      confidence,
                "expected_range":  f"{expected_low:.5f}-{expected_high:.5f}",
                "next_candles":    label,
                "weight":          self.FORECAST_WEIGHT,
                "ema_spread_pct":  round(ema_spread, 3),
                "rsi":             round(rsi_value, 1),
                "body_direction":  body_direction,
                "total_score":     total_score,
                "source":          "ema_rsi_body_composite",
                "timestamp":       pd.Timestamp.now().isoformat(),
            }

            log.info(
                f"[Forecast] dir={direction} conf={confidence}% | "
                f"ema={ema_score} rsi={rsi_score} body={body_score} → "
                f"score={total_score} | range={expected_low:.5f}-{expected_high:.5f}"
            )
            return result

        except Exception as e:
            log.warning(f"[Forecast] failed: {e}")
            return self._fallback(str(e))

    # ─────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _get_or_compute(df: pd.DataFrame, col_name: str, compute_fn, *args):
        """Get a column if it exists, otherwise compute it."""
        if col_name in df.columns:
            series = df[col_name]
            if not series.isna().all():
                return series
        try:
            return compute_fn(*args)
        except Exception:
            return None

    @staticmethod
    def _compute_ma(close: pd.Series, period: int) -> pd.Series:
        return close.rolling(window=period, min_periods=1).mean()

    @staticmethod
    def _compute_ema(close: pd.Series, period: int) -> pd.Series:
        return close.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(alpha=1/period, adjust=False).mean()

    @staticmethod
    def _fallback(reason: str) -> Dict[str, Any]:
        return {
            "direction":       "NEUTRAL",
            "confidence":      0,
            "expected_range":  "unknown",
            "next_candles":    "no forecast",
            "weight":          0.10,
            "source":          "fallback",
            "reason":          reason,
            "timestamp":       pd.Timestamp.now().isoformat(),
        }

    # ─────────────────────────────────────────────────────────
    # AI CONTEXT
    # ─────────────────────────────────────────────────────────

    def get_ai_context(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "forecast_direction":   result.get("direction", "NEUTRAL"),
            "forecast_confidence":  result.get("confidence", 0),
            "forecast_range":       result.get("expected_range", "unknown"),
            "forecast_next":        result.get("next_candles", "unknown"),
            "forecast_weight":      result.get("weight", 0.10),
        }

    def print_summary(self, result: Dict[str, Any]) -> None:
        bar = "═" * 50
        log.info(bar)
        log.info("  📈  FORECAST ENGINE  (Day 97)")
        log.info(bar)
        log.info(f"  Direction       : {result.get('direction','?')}")
        log.info(f"  Confidence      : {result.get('confidence',0)}%")
        log.info(f"  Next candles    : {result.get('next_candles','?')}")
        log.info(f"  Expected range  : {result.get('expected_range','?')}")
        log.info(f"  Weight in fusion: {result.get('weight',0.10)*100:.0f}%")
        log.info(f"  EMA spread %    : {result.get('ema_spread_pct','?')}")
        log.info(f"  RSI             : {result.get('rsi','?')}")
        log.info(f"  Body direction  : {result.get('body_direction','?')}")
        log.info(f"  Total score     : {result.get('total_score','?')}")
        log.info(bar)


# ── Singleton ────────────────────────────────────────────────────

_INSTANCE: Optional[ForecastEngine] = None

def get_forecast_engine() -> ForecastEngine:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = ForecastEngine()
    return _INSTANCE
