"""
analysis/oscillator_regime_gate.py — Day 97+ Oscillator Regime Gating
=====================================================================
Book reference: "The Only Technical Analysis Book You Will Ever Need" (Brian Hale)
Page 33: "Oscillators are best in range-bound markets; can give false signals
in strong trends/high volatility"

Page 34: "Indicators should be used only for confirmation, not as the primary
basis for a trade — price action / chart patterns should lead"

Problem this solves:
  - RSI oversold in a strong DOWNTREND = not a buy signal (price will keep falling)
  - Stochastic overbought in a strong UPTREND = not a sell signal (price will keep rising)
  - Oscillators give false signals in trending markets

Solution:
  Gate oscillator signals based on market regime:
  - Range-bound (ADX < 20): oscillator signals are RELIABLE (full weight)
  - Trending (ADX > 25): oscillator signals are UNRELIABLE (reduced weight or suppressed)
  - High volatility: oscillator signals are NOISY (reduced weight)

Usage:
    from analysis.oscillator_regime_gate import OscillatorRegimeGate
    gate = OscillatorRegimeGate()
    adjusted = gate.adjust_signal(
        signal="BUY", source="RSI", rsi=28, adx=35, trend="BEARISH"
    )
    # → {"allowed": False, "reason": "RSI oversold in BEARISH trend — false signal", "weight": 0.2}
"""

from typing import Any, Dict, Optional
from utils.logger import get_logger

log = get_logger("oscillator_gate")


class OscillatorRegimeGate:
    """Gates oscillator signals based on market regime.

    Book Page 33: "Oscillators are best in range-bound markets"
    Book Page 34: "Indicators for confirmation only, not primary signal"
    """

    # ADX thresholds
    ADX_RANGE_MAX = 20       # below this = range-bound (oscillators OK)
    ADX_TREND_MIN = 25       # above this = trending (oscillators risky)
    ADX_STRONG_TREND = 35    # above this = strong trend (oscillators unreliable)

    # Weight multipliers by regime
    WEIGHT_RANGE = 1.0       # range-bound: full weight (oscillators work well)
    WEIGHT_WEAK_TREND = 0.5  # weak trend: half weight
    WEIGHT_TREND = 0.2       # trending: 20% weight (mostly false signals)
    WEIGHT_STRONG_TREND = 0.0  # strong trend: suppress oscillator signals entirely

    # RSI regime-adjusted thresholds (Book Page 44)
    # In uptrends, RSI tends to range 40-90 (40-50 = support zone)
    # In downtrends, RSI tends to range 10-60 (50-60 = resistance zone)
    RSI_THRESHOLDS = {
        "UPTREND": {
            "oversold": 40,       # raised from 30 → 40 (uptrend RSI rarely goes below 40)
            "overbought": 80,     # raised from 70 → 80 (uptrend RSI stays elevated)
            "support_zone": (40, 50),   # RSI 40-50 = support in uptrend
        },
        "DOWNTREND": {
            "oversold": 20,       # lowered from 30 → 20 (downtrend RSI stays low)
            "overbought": 60,     # lowered from 70 → 60 (downtrend RSI rarely exceeds 60)
            "resistance_zone": (50, 60),  # RSI 50-60 = resistance in downtrend
        },
        "RANGE": {
            "oversold": 30,       # standard 30
            "overbought": 70,     # standard 70
        },
    }

    def adjust_signal(
        self,
        signal: str,
        source: str,
        rsi: Optional[float] = None,
        adx: Optional[float] = None,
        trend: Optional[str] = None,
        volatility: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Gate an oscillator signal based on current market regime.

        Args:
            signal: "BUY" or "SELL" from oscillator (RSI, Stochastic, etc.)
            source: indicator name ("RSI", "STOCHASTIC", "MACD")
            rsi: current RSI value (for regime-adjusted thresholds)
            adx: current ADX value (for trend strength)
            trend: "BULLISH", "BEARISH", "RANGING", "UNKNOWN"
            volatility: "LOW", "NORMAL", "HIGH", "EXTREME"

        Returns:
            {
                "allowed": bool,      # should this signal be acted on?
                "weight": float,      # 0.0-1.0 multiplier for confidence
                "reason": str,        # why allowed/blocked
                "adjusted_threshold": dict,  # regime-adjusted RSI thresholds if applicable
            }
        """
        trend_upper = (trend or "UNKNOWN").upper()
        adx_val = float(adx or 0)
        vol_upper = (volatility or "NORMAL").upper()

        # Step 1: Determine weight based on ADX (trend strength)
        if adx_val >= self.ADX_STRONG_TREND:
            weight = self.WEIGHT_STRONG_TREND
            regime = "STRONG_TREND"
        elif adx_val >= self.ADX_TREND_MIN:
            weight = self.WEIGHT_TREND
            regime = "TREND"
        elif adx_val >= self.ADX_RANGE_MAX:
            weight = self.WEIGHT_WEAK_TREND
            regime = "WEAK_TREND"
        else:
            weight = self.WEIGHT_RANGE
            regime = "RANGE"

        # Step 2: Check if oscillator signal contradicts the trend
        # Book Page 33: "oscillators give false signals in strong trends"
        contradicts_trend = False
        if signal == "BUY" and trend_upper in ("BEARISH", "STRONG_BEARISH"):
            contradicts_trend = True
        elif signal == "SELL" and trend_upper in ("BULLISH", "STRONG_BULLISH"):
            contradicts_trend = True

        # Step 3: Decision
        allowed = True
        reason = f"{source} {signal} in {regime} (ADX={adx_val:.0f})"

        if weight == 0.0:
            allowed = False
            reason = f"{source} {signal} suppressed — ADX={adx_val:.0f} (strong trend, oscillators unreliable per Book Page 33)"
        elif contradicts_trend and weight < 0.5:
            allowed = False
            reason = f"{source} {signal} suppressed — contradicts {trend_upper} trend in {regime} (false signal risk per Book Page 33)"
        elif contradicts_trend:
            allowed = True
            reason = f"{source} {signal} in {regime} — WARNING: contradicts {trend_upper} trend, weight reduced to {weight}"
        elif vol_upper in ("HIGH", "EXTREME") and weight > 0.5:
            weight *= 0.7  # reduce weight in high volatility
            reason = f"{source} {signal} in {regime} — weight reduced (volatility={vol_upper})"

        # Step 4: Get regime-adjusted RSI thresholds
        adjusted_threshold = {}
        if rsi is not None:
            if trend_upper in ("BULLISH", "STRONG_BULLISH"):
                adjusted_threshold = self.RSI_THRESHOLDS["UPTREND"]
            elif trend_upper in ("BEARISH", "STRONG_BEARISH"):
                adjusted_threshold = self.RSI_THRESHOLDS["DOWNTREND"]
            else:
                adjusted_threshold = self.RSI_THRESHOLDS["RANGE"]

        return {
            "allowed": allowed,
            "weight": round(weight, 2),
            "regime": regime,
            "reason": reason,
            "adjusted_threshold": adjusted_threshold,
        }

    def get_rsi_signal(
        self,
        rsi: float,
        trend: str = "UNKNOWN",
    ) -> Dict[str, Any]:
        """Get regime-adjusted RSI signal (Book Page 44).

        Instead of fixed 70/30, adjusts based on trend:
          Uptrend:  oversold=40, overbought=80 (RSI stays elevated)
          Downtrend: oversold=20, overbought=60 (RSI stays depressed)
          Range:    oversold=30, overbought=70 (standard)
        """
        trend_upper = (trend or "UNKNOWN").upper()

        if trend_upper in ("BULLISH", "STRONG_BULLISH"):
            thresholds = self.RSI_THRESHOLDS["UPTREND"]
            regime = "UPTREND"
        elif trend_upper in ("BEARISH", "STRONG_BEARISH"):
            thresholds = self.RSI_THRESHOLDS["DOWNTREND"]
            regime = "DOWNTREND"
        else:
            thresholds = self.RSI_THRESHOLDS["RANGE"]
            regime = "RANGE"

        oversold = thresholds["oversold"]
        overbought = thresholds["overbought"]

        if rsi <= oversold:
            signal = "BUY"
            zone = "oversold"
        elif rsi >= overbought:
            signal = "SELL"
            zone = "overbought"
        else:
            signal = "NEUTRAL"
            zone = "neutral"

        # Check support/resistance zones in trending markets
        in_support_zone = False
        in_resistance_zone = False
        if regime == "UPTREND" and "support_zone" in thresholds:
            low, high = thresholds["support_zone"]
            in_support_zone = low <= rsi <= high
        if regime == "DOWNTREND" and "resistance_zone" in thresholds:
            low, high = thresholds["resistance_zone"]
            in_resistance_zone = low <= rsi <= high

        return {
            "signal": signal,
            "zone": zone,
            "rsi": rsi,
            "regime": regime,
            "oversold_threshold": oversold,
            "overbought_threshold": overbought,
            "in_support_zone": in_support_zone,
            "in_resistance_zone": in_resistance_zone,
            "reason": f"RSI={rsi:.1f} in {regime} (thresholds: {oversold}/{overbought})",
        }


# ── Singleton ─────────────────────────────────────────────────────

_GATE: Optional[OscillatorRegimeGate] = None


def get_oscillator_gate() -> OscillatorRegimeGate:
    global _GATE
    if _GATE is None:
        _GATE = OscillatorRegimeGate()
    return _GATE
