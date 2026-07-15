"""
analysis/trendline_engine.py — Day 97+ Trendline Detection & Trading
=====================================================================
Book reference: "The Only Technical Analysis Book You Will Ever Need" (Brian Hale)
Pages 63-66: Trendline construction, touchpoint strength, dynamic S/R, channels

What this does:
  1. Detects trendlines by connecting swing highs/lows (linear regression fit)
  2. Counts touchpoints (more touches = stronger line, Book Page 64)
  3. Detects trendline breakouts (Book Page 62)
  4. Identifies channel trading zones (Book Page 66)
  5. Provides pullback entry signals at trendline (Book Page 65)

Usage:
    from analysis.trendline_engine import TrendlineEngine
    te = TrendlineEngine()
    result = te.analyze(df, pair="EURUSD")
    # → {"uptrend_line": {...}, "downtrend_line": {...}, "channel": {...}, "signals": [...]}
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from utils.logger import get_logger

log = get_logger("trendline_engine")


class TrendlineEngine:
    """Detects and trades trendlines (Book Pages 63-66).

    Book Page 63: Connect higher lows for uptrend line, lower highs for downtrend line
    Book Page 64: More touchpoints = stronger trendline
    Book Page 65: Trendline acts as dynamic S/R — buy on pullback to uptrend line
    Book Page 66: Channel = two parallel trendlines; buy at lower, sell at upper
    """

    MIN_TOUCHES = 2        # minimum touchpoints for a valid trendline
    STRONG_TOUCHES = 3     # 3+ touches = strong trendline
    BREAKOUT_BUFFER = 0.0003  # price must break this far beyond line to confirm
    MIN_BARS_FOR_FIT = 10  # need at least this many bars to fit a line

    def analyze(self, df: pd.DataFrame, pair: str = "EURUSD") -> Dict[str, Any]:
        """Full trendline analysis pipeline.

        Returns:
            {
                "uptrend_line": {"exists": bool, "slope": float, "intercept": float,
                                 "touches": int, "current_value": float, "strength": str},
                "downtrend_line": {...},
                "channel": {"exists": bool, "upper": float, "lower": float, "width": float},
                "signals": [{"type": "pullback_buy"|"pullback_sell"|"breakout_bullish"|"breakout_bearish",
                             "level": float, "confidence": int, "reason": str}],
                "current_price": float,
            }
        """
        if len(df) < self.MIN_BARS_FOR_FIT:
            return self._empty_result("Insufficient data for trendline analysis")

        # Sanitize
        high = pd.to_numeric(df["high"], errors="coerce").fillna(0).values
        low = pd.to_numeric(df["low"], errors="coerce").fillna(0).values
        close = pd.to_numeric(df["close"], errors="coerce").fillna(0).values
        current_price = float(close[-1])

        # Find swing points
        swing_lows = self._find_swings(low, direction="low")
        swing_highs = self._find_swings(high, direction="high")

        # Fit trendlines
        uptrend_line = self._fit_trendline(swing_lows, df, direction="up")
        downtrend_line = self._fit_trendline(swing_highs, df, direction="down")

        # Channel detection
        channel = self._detect_channel(uptrend_line, downtrend_line, current_price)

        # Generate signals
        signals = self._generate_signals(
            uptrend_line, downtrend_line, channel, current_price, df
        )

        result = {
            "uptrend_line": uptrend_line,
            "downtrend_line": downtrend_line,
            "channel": channel,
            "signals": signals,
            "current_price": round(current_price, 5),
        }

        # Log summary
        parts = []
        if uptrend_line["exists"]:
            parts.append(f"Up: {uptrend_line['touches']} touches ({uptrend_line['strength']})")
        if downtrend_line["exists"]:
            parts.append(f"Down: {downtrend_line['touches']} touches ({downtrend_line['strength']})")
        if channel["exists"]:
            parts.append(f"Channel: {channel['width_pips']:.0f} pips")
        if signals:
            parts.append(f"{len(signals)} signals")
        log.info(f"[Trendline] {' | '.join(parts) if parts else 'No trendlines detected'}")

        return result

    def _find_swings(self, values: np.ndarray, direction: str = "low",
                     window: int = 3) -> List[Tuple[int, float]]:
        """Find swing points (local extrema)."""
        swings = []
        n = len(values)
        for i in range(window, n - window):
            if direction == "low":
                if values[i] == min(values[i-window:i+window+1]):
                    swings.append((i, float(values[i])))
            else:
                if values[i] == max(values[i-window:i+window+1]):
                    swings.append((i, float(values[i])))
        return swings

    def _fit_trendline(self, swings: List[Tuple[int, float]], df: pd.DataFrame,
                       direction: str = "up") -> Dict[str, Any]:
        """Fit a linear trendline to swing points using least-squares regression.

        Book Page 63: Uptrend line connects higher lows; downtrend connects lower highs.
        """
        result = {
            "exists": False, "slope": 0, "intercept": 0, "touches": 0,
            "current_value": 0, "strength": "none",
        }

        if len(swings) < self.MIN_TOUCHES:
            return result

        # Use last N swing points for fitting
        recent_swings = swings[-min(len(swings), 8):]
        x = np.array([s[0] for s in recent_swings], dtype=float)
        y = np.array([s[1] for s in recent_swings], dtype=float)

        if len(x) < 2:
            return result

        # Linear regression: y = slope * x + intercept
        try:
            slope, intercept = np.polyfit(x, y, 1)
        except Exception as e:
            return result

        # Validate direction
        if direction == "up" and slope < 0:
            return result  # uptrend line must have positive slope
        if direction == "down" and slope > 0:
            return result  # downtrend line must have negative slope

        # Count touchpoints (price points near the line)
        n_bars = len(df)
        current_value = float(slope * n_bars + intercept)
        touches = 0
        tolerance = float(np.std(y)) * 0.5 if len(y) > 1 else 0.0010

        for i in range(n_bars):
            line_val = float(slope * i + intercept)
            close_i = float(df["close"].iloc[i])
            if abs(close_i - line_val) < tolerance:
                touches += 1

        # Strength rating (Book Page 64: more touches = stronger)
        if touches >= self.STRONG_TOUCHES:
            strength = "strong"
        elif touches >= self.MIN_TOUCHES:
            strength = "moderate"
        else:
            strength = "weak"

        result.update({
            "exists": True,
            "slope": round(float(slope), 8),
            "intercept": round(float(intercept), 5),
            "touches": touches,
            "current_value": round(current_value, 5),
            "strength": strength,
        })
        return result

    def _detect_channel(self, up_line: dict, down_line: dict,
                        current_price: float) -> Dict[str, Any]:
        """Detect trading channel (Book Page 66).

        Channel = two roughly parallel trendlines.
        Buy at lower line, sell at upper line.
        """
        result = {"exists": False, "upper": 0, "lower": 0, "width": 0, "width_pips": 0}

        if not up_line["exists"] or not down_line["exists"]:
            return result

        # Check if roughly parallel (similar slope magnitude)
        slope_diff = abs(abs(up_line["slope"]) - abs(down_line["slope"]))
        if slope_diff > abs(up_line["slope"]) * 0.5:  # too divergent
            return result

        upper = max(up_line["current_value"], down_line["current_value"])
        lower = min(up_line["current_value"], down_line["current_value"])
        width = upper - lower

        if width <= 0:
            return result

        result.update({
            "exists": True,
            "upper": round(upper, 5),
            "lower": round(lower, 5),
            "width": round(width, 5),
            "width_pips": round(width / 0.0001, 0),  # approximate pips
        })
        return result

    def _generate_signals(self, up_line: dict, down_line: dict, channel: dict,
                          current_price: float, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate trading signals from trendline analysis.

        Book Page 65: Buy on pullback to uptrend line
        Book Page 62: Breakout signal when price crosses trendline
        Book Page 66: Channel trading — buy at lower, sell at upper
        """
        signals = []
        tolerance = 0.0005  # how close to line = "near"

        # Pullback to uptrend line = buy signal (Book Page 65)
        if up_line["exists"]:
            line_val = up_line["current_value"]
            if abs(current_price - line_val) < tolerance:
                conf = 60 if up_line["strength"] == "strong" else 45
                signals.append({
                    "type": "pullback_buy",
                    "level": line_val,
                    "confidence": conf,
                    "reason": f"Price at uptrend line ({up_line['touches']} touches, {up_line['strength']})",
                })

        # Pullback to downtrend line = sell signal (Book Page 65)
        if down_line["exists"]:
            line_val = down_line["current_value"]
            if abs(current_price - line_val) < tolerance:
                conf = 60 if down_line["strength"] == "strong" else 45
                signals.append({
                    "type": "pullback_sell",
                    "level": line_val,
                    "confidence": conf,
                    "reason": f"Price at downtrend line ({down_line['touches']} touches, {down_line['strength']})",
                })

        # Breakout signals (Book Page 62)
        if up_line["exists"]:
            line_val = up_line["current_value"]
            if current_price > line_val + self.BREAKOUT_BUFFER:
                signals.append({
                    "type": "breakout_bullish",
                    "level": line_val,
                    "confidence": 55,
                    "reason": f"Bullish breakout above uptrend line ({up_line['touches']} touches)",
                })

        if down_line["exists"]:
            line_val = down_line["current_value"]
            if current_price < line_val - self.BREAKOUT_BUFFER:
                signals.append({
                    "type": "breakout_bearish",
                    "level": line_val,
                    "confidence": 55,
                    "reason": f"Bearish breakout below downtrend line ({down_line['touches']} touches)",
                })

        # Channel trading signals (Book Page 66)
        if channel["exists"]:
            if abs(current_price - channel["lower"]) < tolerance:
                signals.append({
                    "type": "channel_buy",
                    "level": channel["lower"],
                    "confidence": 50,
                    "reason": f"Price at channel lower bound ({channel['width_pips']:.0f} pip channel)",
                })
            elif abs(current_price - channel["upper"]) < tolerance:
                signals.append({
                    "type": "channel_sell",
                    "level": channel["upper"],
                    "confidence": 50,
                    "reason": f"Price at channel upper bound ({channel['width_pips']:.0f} pip channel)",
                })

        return signals

    def _empty_result(self, reason: str) -> Dict[str, Any]:
        return {
            "uptrend_line": {"exists": False, "touches": 0, "strength": "none"},
            "downtrend_line": {"exists": False, "touches": 0, "strength": "none"},
            "channel": {"exists": False},
            "signals": [],
            "current_price": 0,
            "error": reason,
        }


# ── Singleton ─────────────────────────────────────────────────────

_TE: Optional[TrendlineEngine] = None


def get_trendline_engine() -> TrendlineEngine:
    global _TE
    if _TE is None:
        _TE = TrendlineEngine()
    return _TE
