# analysis/trend_level_signal.py
# ============================================================
# "Trend, Level, Signal" Framework
# ============================================================
# Book: "The Candlestick Trading Bible" Pages 79-80
#
# The book's unifying 3-question framework for chart analysis:
#   1. What is the market doing? (Trend: trending/ranging/choppy)
#   2. What are the most powerful levels? (Level: S/R zones, trendlines)
#   3. What is the signal? (Signal: candlestick patterns at those levels)
#
# This module synthesizes the outputs of:
#   - analysis/market_regime.py (Trend)
#   - analysis/support_resistance.py (Level)
#   - analysis/high_reliability_patterns.py (Signal)
#
# Into a single unified decision framework.
# ============================================================

import logging
from typing import Optional, Dict, Any, List
import pandas as pd

log = logging.getLogger(__name__)


class TrendLevelSignalFramework:
    """
    Book Pages 79-80 — "Trend, Level, Signal" unified framework.

    Usage:
        framework = TrendLevelSignalFramework()
        result = framework.analyze(df, symbol="EURUSD", zones=zones, patterns=patterns)
        # result = {"trend": ..., "level": ..., "signal": ..., "action": "BUY/SELL/WAIT/NO_TRADE"}
    """

    def __init__(self, timeframe: str = "H1"):
        self.timeframe = timeframe

    def analyze(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        zones: Optional[List[dict]] = None,
        patterns: Optional[List[dict]] = None,
        ind_ctx: Optional[dict] = None,
    ) -> dict:
        """
        Run the 3-question framework:
          Q1: What is the market doing? → Trend analysis
          Q2: What are the most powerful levels? → S/R zone analysis
          Q3: What is the signal? → Pattern analysis at levels

        Returns unified decision dict with action + reasoning.
        """
        # ── Q1: TREND ──────────────────────────────────
        trend_analysis = self._analyze_trend(df, ind_ctx)

        # Book Page 69: If CHOPPY → don't trade (stop here)
        if trend_analysis["regime"] == "CHOPPY":
            return {
                "framework": "Trend-Level-Signal",
                "trend": trend_analysis,
                "level": {"status": "skipped — choppy market"},
                "signal": {"status": "skipped — choppy market"},
                "action": "NO_TRADE",
                "reason": "CHOPPY market (ADX<15) — Book P69: 'not worth trading'. "
                          "Can't identify S/R boundaries. Stay out.",
                "confidence": 0,
            }

        # ── Q2: LEVEL ─────────────────────────────────
        level_analysis = self._analyze_levels(df, zones, ind_ctx, trend_analysis)

        # ── Q3: SIGNAL ────────────────────────────────
        signal_analysis = self._analyze_signals(patterns, level_analysis, trend_analysis)

        # ── UNIFIED DECISION ──────────────────────────
        action = self._decide_action(trend_analysis, level_analysis, signal_analysis)

        return {
            "framework": "Trend-Level-Signal",
            "trend": trend_analysis,
            "level": level_analysis,
            "signal": signal_analysis,
            "action": action["action"],
            "reason": action["reason"],
            "confidence": action["confidence"],
        }

    # ═══════════════════════════════════════════════════════════
    # Q1: TREND — "What is the market doing?"
    # ═══════════════════════════════════════════════════════════

    def _analyze_trend(self, df: pd.DataFrame, ind_ctx: Optional[dict]) -> dict:
        """Determine market regime: trending, ranging, or choppy."""
        try:
            from analysis.market_regime import MarketRegimeDetector
            detector = MarketRegimeDetector()
            regime_result = detector.detect(df)

            return {
                "regime": regime_result.get("regime", "UNKNOWN"),
                "direction": regime_result.get("direction", "NEUTRAL"),
                "strength": regime_result.get("strength", "WEAK"),
                "volatility": regime_result.get("volatility", "NORMAL"),
                "adx": regime_result.get("adx", 0),
                "strategy": regime_result.get("strategy", {}),
                "answer": self._trend_answer(regime_result),
            }
        except Exception as e:
            log.warning(f"[TLS] Trend analysis failed: {e}")
            return {"regime": "UNKNOWN", "answer": "Cannot determine trend"}

    def _trend_answer(self, regime_result: dict) -> str:
        """Human-readable answer to 'What is the market doing?'"""
        regime = regime_result.get("regime", "UNKNOWN")
        direction = regime_result.get("direction", "NEUTRAL")
        adx = regime_result.get("adx", 0)

        if regime == "TRENDING":
            return f"Market is TRENDING {direction} (ADX={adx:.0f}). Trade with the trend."
        elif regime == "RANGING":
            return f"Market is RANGING (ADX={adx:.0f}). Buy support, sell resistance."
        elif regime == "CHOPPY":
            return f"Market is CHOPPY (ADX={adx:.0f}). Don't trade — no clear S/R."
        elif regime == "BREAKOUT":
            return f"Market is BREAKING OUT (ADX={adx:.0f}). Watch for confirmation."
        return f"Market regime: {regime} (ADX={adx:.0f})"

    # ═══════════════════════════════════════════════════════════
    # Q2: LEVEL — "What are the most powerful levels?"
    # ═══════════════════════════════════════════════════════════

    def _analyze_levels(
        self, df: pd.DataFrame, zones: Optional[List[dict]],
        ind_ctx: Optional[dict], trend: dict
    ) -> dict:
        """Identify key support/resistance levels near current price."""
        if zones:
            # Use provided zones (from S/R engine)
            current_price = float(df["close"].iloc[-1]) if len(df) > 0 else 0
            nearby_zones = []
            for z in zones:
                z_center = (z.get("zone_top", 0) + z.get("zone_bottom", 0)) / 2
                dist = abs(z_center - current_price)
                if dist < current_price * 0.01:  # within 1% of price
                    nearby_zones.append({
                        "type": z.get("type", "unknown"),
                        "zone_top": z.get("zone_top"),
                        "zone_bottom": z.get("zone_bottom"),
                        "touches": z.get("touches", 0),
                        "strength": z.get("strength", "Weak"),
                        "distance_pct": round(dist / current_price * 100, 2) if current_price else 0,
                    })
            return {
                "nearby_zones": nearby_zones,
                "zone_count": len(nearby_zones),
                "answer": f"Found {len(nearby_zones)} nearby S/R zone(s).",
            }

        # No zones provided — run S/R detection
        try:
            from analysis.support_resistance import SupportResistance
            sr = SupportResistance(timeframe=self.timeframe)
            sr_result = sr.analyze(df, symbol="")
            sr_ctx = sr.get_ai_context(sr_result)

            return {
                "nearest_support": sr_ctx.get("nearest_support"),
                "nearest_resistance": sr_ctx.get("nearest_resistance"),
                "price_location": sr_ctx.get("price_location", "mid_range"),
                "answer": (
                    f"Nearest support: {sr_ctx.get('nearest_support')}, "
                    f"resistance: {sr_ctx.get('nearest_resistance')}, "
                    f"location: {sr_ctx.get('price_location', 'mid_range')}"
                ),
            }
        except Exception as e:
            log.warning(f"[TLS] Level analysis failed: {e}")
            return {"answer": "Cannot determine levels"}

    # ═══════════════════════════════════════════════════════════
    # Q3: SIGNAL — "What is the signal?"
    # ═══════════════════════════════════════════════════════════

    def _analyze_signals(
        self, patterns: Optional[List[dict]], level: dict, trend: dict
    ) -> dict:
        """Check candlestick patterns at key levels."""
        if not patterns:
            return {
                "patterns_found": 0,
                "answer": "No candlestick patterns detected at current levels.",
            }

        # Filter patterns that are near a zone (High reliability)
        high_reliability = [p for p in patterns if p.get("reliability") == "High"]
        reversal_patterns = [p for p in high_reliability if p.get("type") == "Reversal"]

        if reversal_patterns:
            direction = reversal_patterns[0].get("direction", "neutral")
            pattern_name = reversal_patterns[0].get("pattern_name", "unknown")
            return {
                "patterns_found": len(patterns),
                "high_reliability_count": len(high_reliability),
                "reversal_at_level": True,
                "pattern_name": pattern_name,
                "direction": direction,
                "answer": f"High-reliability {pattern_name} ({direction}) detected near key level.",
            }

        return {
            "patterns_found": len(patterns),
            "high_reliability_count": len(high_reliability),
            "reversal_at_level": False,
            "answer": f"{len(patterns)} pattern(s) found, but none with High reliability at key levels.",
        }

    # ═══════════════════════════════════════════════════════════
    # UNIFIED DECISION
    # ═══════════════════════════════════════════════════════════

    def _decide_action(self, trend: dict, level: dict, signal: dict) -> dict:
        """Combine all 3 aspects into a final action."""
        regime = trend.get("regime", "UNKNOWN")
        direction = trend.get("direction", "NEUTRAL")
        has_signal = signal.get("reversal_at_level", False)
        signal_dir = signal.get("direction", "neutral")

        # CHOPPY → NO_TRADE (already handled, but double-check)
        if regime == "CHOPPY":
            return {"action": "NO_TRADE", "reason": "Choppy market — stay out",
                    "confidence": 0}

        # RANGING → only trade if signal at S/R level
        if regime == "RANGING":
            if has_signal:
                action = "BUY" if signal_dir == "bullish" else "SELL" if signal_dir == "bearish" else "WAIT"
                return {
                    "action": action,
                    "reason": f"Ranging market + {signal.get('pattern_name')} at S/R level → {action}",
                    "confidence": 60,
                }
            return {"action": "WAIT", "reason": "Ranging but no signal at levels",
                    "confidence": 30}

        # TRENDING → trade with trend if signal confirms
        if regime == "TRENDING":
            if has_signal:
                # Signal must align with trend
                if (direction == "BULLISH" and signal_dir == "bullish"):
                    return {"action": "BUY", "confidence": 75,
                            "reason": f"Trending UP + {signal.get('pattern_name')} confirms → BUY"}
                elif (direction == "BEARISH" and signal_dir == "bearish"):
                    return {"action": "SELL", "confidence": 75,
                            "reason": f"Trending DOWN + {signal.get('pattern_name')} confirms → SELL"}
                else:
                    return {"action": "WAIT", "confidence": 40,
                            "reason": f"Trend {direction} but signal {signal_dir} — conflict, wait"}
            return {"action": "WAIT", "confidence": 35,
                    "reason": "Trending but no confirmation signal at levels"}

        # BREAKOUT → wait for confirmation
        if regime == "BREAKOUT":
            return {"action": "WAIT", "confidence": 40,
                    "reason": "Breakout in progress — wait for confirmation"}

        return {"action": "NO_TRADE", "confidence": 0,
                "reason": f"Unknown regime: {regime}"}


# ============================================================
# CLI entry
# ============================================================

if __name__ == "__main__":
    import numpy as np
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-06-01", periods=n, freq="1h")
    close = 1.0850 + np.cumsum(np.random.randn(n) * 0.0005)
    df = pd.DataFrame({
        "open": close, "high": close + 0.0005, "low": close - 0.0005,
        "close": close,
    }, index=dates)

    framework = TrendLevelSignalFramework(timeframe="H1")
    result = framework.analyze(df, symbol="EURUSD")
    print(f"\n=== Trend-Level-Signal Framework ===")
    print(f"Q1 Trend:  {result['trend']['answer']}")
    print(f"Q2 Level:  {result['level'].get('answer', 'N/A')}")
    print(f"Q3 Signal: {result['signal'].get('answer', 'N/A')}")
    print(f"Action:    {result['action']} (conf={result['confidence']}%)")
    print(f"Reason:    {result['reason']}")
