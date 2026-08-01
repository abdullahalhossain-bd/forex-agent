# analysis/high_reliability_patterns.py
# ============================================================
# High-Reliability Candlestick Pattern Library (Spec-Compliant)
# ============================================================
# STRICT 20-pattern library — no rare/minor patterns (reduces false detection).
#
# SINGLE-CANDLE (Reversal):
#   1. Hammer            — long lower wick (≥2× body), small/no upper wick, downtrend/support → bullish reversal
#   2. Shooting Star     — long upper wick (≥2× body), small/no lower wick, uptrend/resistance → bearish reversal
#   3. Inverted Hammer   — Hammer's shape but in downtrend (needs confirmation)
#   4. Hanging Man       — Hammer's shape but in uptrend (bearish warning, needs confirmation)
#   5. Doji              — Open≈Close, indecision/momentum exhaustion
#
# SINGLE-CANDLE (Momentum/Continuation):
#   6. Bullish Marubozu  — no/tiny wick, full bullish body, strong buyer momentum
#   7. Bearish Marubozu  — no/tiny wick, full bearish body, strong seller momentum
#
# TWO-CANDLE (Reversal — strongest):
#   8. Bullish Engulfing    — 2nd candle engulfs 1st body, strong bullish reversal
#   9. Bearish Engulfing    — opposite, strong bearish reversal
#  10. Tweezer Top          — equal high rejected twice, resistance → bearish
#  11. Tweezer Bottom       — equal low rejected twice, support → bullish
#  12. Piercing Line        — bullish candle enters ≥50% into prior bearish body, moderate bullish reversal
#  13. Dark Cloud Cover     — opposite, moderate bearish reversal
#  14. Bullish/Bearish Harami — small opposite candle inside large candle, momentum weakening
#
# THREE-CANDLE (Confirmed Reversal — highest reliability):
#  15. Morning Star         — bearish→indecision→bullish, strong confirmed bullish reversal
#  16. Evening Star         — bullish→indecision→bearish, strong confirmed bearish reversal
#  17. Three White Soldiers — 3 consecutive large bullish candles, strong uptrend continuation
#  18. Three Black Crows    — 3 consecutive large bearish candles, strong downtrend continuation
#  19. Three Inside Up      — Bullish Harami + confirmation, confirmed bullish reversal
#  20. Three Inside Down    — Bearish Harami + confirmation, confirmed bearish reversal
#
# DETECTION & VALIDATION RULES:
#  1. Classify each pattern: Reversal / Continuation / Indecision
#  2. Validate zone confluence:
#       - Near zone (Support/Resistance/Supply-Demand/Trendline) → Reliability: High
#       - Mid-range (no zone nearby) → Reliability: Low (info only, not for trade logic)
#  3. Multi-bar repetition:
#       - Same reversal pattern at same zone 2+ times → zone strength boost (Weak→Medium, Medium→Strong)
#       - 3+ consecutive Marubozu/Soldier/Crow same direction → Momentum sequence (input to S/D zone logic)
#       - Multiple consecutive Doji → Consolidation/Accumulation, lean toward "WAIT"
#  4. NO single pattern is a standalone entry trigger — only used as checklist factor
#     ("candlestick_pattern" + "candle_behavior") combined with other confluence.
# ============================================================

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ─── Constants ────────────────────────────────────────────────
HAMMER_WICK_BODY_RATIO   = 2.0    # lower wick ≥ 2× body
SHOOTING_STAR_WICK_RATIO = 2.0    # upper wick ≥ 2× body
DOJI_BODY_THRESHOLD_PCT  = 0.10   # body ≤ 10% of total range (Book Spread 46)
MARUBOZU_BODY_RATIO      = 0.90   # body ≥ 90% of total range (tiny wicks)
MARUBOZU_MAX_WICK_PCT    = 0.05   # each wick ≤ 5% of range
ENGULFING_THRESHOLD      = 1.0    # 2nd body must fully engulf 1st body
PIERCING_MIN_PCT         = 0.50   # bullish candle closes above 50% of prior bearish body
TWEEZER_TOLERANCE_PCT    = 0.0003 # 0.03% tolerance for "equal" high/low
HARAMI_SMALL_BODY_PCT    = 0.50   # 2nd candle body ≤ 50% of 1st body (small inside large)
STAR_MIDDLE_BODY_PCT     = 0.30   # middle candle body ≤ 30% of range (indecision)
SOLDIERS_MIN_BODY_PCT    = 0.60   # each soldier body ≥ 60% of its range
ZONE_PROXIMITY_ATR_MULT  = 1.0    # within 1×ATR of zone = "near zone"


# ─── Dataclass ────────────────────────────────────────────────

@dataclass
class DetectedPattern:
    """Single detected pattern matching spec output schema."""
    pattern_name: str
    type: str                     # "Reversal" | "Continuation" | "Indecision"
    candle_index_or_time: str     # ISO timestamp or index
    near_zone: bool
    zone_type: str                # "Support" | "Resistance" | "Supply" | "Demand" | "Trendline" | "None"
    reliability: str              # "Low" | "High"
    direction: str = "neutral"    # "bullish" | "bearish" | "neutral" (internal use)
    candle_index: int = -1        # internal use

    def to_spec_dict(self) -> dict:
        return {
            "pattern_name":           self.pattern_name,
            "type":                   self.type,
            "candle_index_or_time":   self.candle_index_or_time,
            "near_zone":              bool(self.near_zone),
            "zone_type":              self.zone_type,
            "reliability":            self.reliability,
        }


# ─── Helpers ──────────────────────────────────────────────────

def _candle_metrics(c: pd.Series) -> dict:
    """Extract body/wick metrics from a single OHLC candle."""
    o = float(c["open"]); h = float(c["high"]); l = float(c["low"]); cl = float(c["close"])
    body = abs(cl - o)
    total_range = h - l
    if total_range <= 0:
        return {"body": 0, "range": 0, "upper_wick": 0, "lower_wick": 0,
                "body_pct": 0, "is_bullish": False, "is_bearish": False}
    upper_wick = h - max(o, cl)
    lower_wick = min(o, cl) - l
    body_pct = body / total_range
    is_bullish = cl > o
    is_bearish = cl < o
    return {
        "open": o, "high": h, "low": l, "close": cl,
        "body": body, "range": total_range,
        "upper_wick": upper_wick, "lower_wick": lower_wick,
        "body_pct": body_pct,
        "is_bullish": is_bullish, "is_bearish": is_bearish,
    }


# Shared ATR helper (returns float, with fallback)
from analysis._engine_utils import atr_value as _atr


# ─── Main Pattern Library ─────────────────────────────────────

class HighReliabilityPatternDetector:
    """
    Strict 20-pattern library — spec-compliant.

    Usage:
        detector = HighReliabilityPatternDetector()
        patterns = detector.detect(df, zones=zones_list, atr_value=atr)
        # Returns list[DetectedPattern]
    """

    def __init__(
        self,
        lookback: int = 10,
        zone_proximity_atr: float = ZONE_PROXIMITY_ATR_MULT,
    ):
        self.lookback = lookback
        self.zone_proximity_atr = zone_proximity_atr

    # ═══════════════════════════════════════════════════════════
    # MAIN ENTRY POINT
    # ═══════════════════════════════════════════════════════════

    def detect(
        self,
        df: pd.DataFrame,
        zones: Optional[List[dict]] = None,
        atr_value: Optional[float] = None,
    ) -> List[DetectedPattern]:
        """
        Scan the last `lookback` candles for any of the 20 high-reliability patterns.

        Args:
            df: OHLC DataFrame
            zones: list of zone dicts, each with keys:
                   {"type": "Support/Resistance/Supply/Demand/Trendline",
                    "zone_top": float, "zone_bottom": float}
            atr_value: ATR value for proximity calculation (auto-computed if None)

        Returns:
            List of DetectedPattern objects (most recent first)
        """
        if df is None or len(df) < 3:
            return []

        if atr_value is None or atr_value <= 0:
            atr_value = _atr(df)

        zones = zones or []
        patterns: List[DetectedPattern] = []

        n = len(df)
        # Determine start index based on lookback (but allow single-candle patterns
        # to be detected even at the very beginning of the lookback window)
        start = max(0, n - self.lookback)

        # Iterate from oldest to newest in lookback window
        for i in range(start, n):
            # ── Single-candle patterns ──
            single_candle_detectors = [
                self._detect_hammer, self._detect_shooting_star,
                self._detect_inverted_hammer, self._detect_hanging_man,
                self._detect_doji, self._detect_bullish_marubozu,
                self._detect_bearish_marubozu,
            ]
            for det_fn in single_candle_detectors:
                p = det_fn(df, i, zones, atr_value)
                if p:
                    patterns.append(p)

            # ── Two-candle patterns ──
            if i >= 1:
                two_candle_detectors = [
                    self._detect_bullish_engulfing, self._detect_bearish_engulfing,
                    self._detect_tweezer_top, self._detect_tweezer_bottom,
                    self._detect_piercing_line, self._detect_dark_cloud_cover,
                    self._detect_harami,
                ]
                for det_fn in two_candle_detectors:
                    p = det_fn(df, i, zones, atr_value)
                    if p:
                        patterns.append(p)

            # ── Three-candle patterns ──
            if i >= 2:
                three_candle_detectors = [
                    self._detect_morning_star, self._detect_evening_star,
                    self._detect_three_white_soldiers, self._detect_three_black_crows,
                    self._detect_three_inside_up, self._detect_three_inside_down,
                ]
                for det_fn in three_candle_detectors:
                    p = det_fn(df, i, zones, atr_value)
                    if p:
                        patterns.append(p)

        # Sort by candle_index descending (most recent first)
        patterns.sort(key=lambda p: p.candle_index, reverse=True)
        return patterns

    # ═══════════════════════════════════════════════════════════
    # ZONE CONFLUENCE VALIDATION
    # ═══════════════════════════════════════════════════════════

    def _check_zone_confluence(
        self,
        candle_high: float,
        candle_low: float,
        zones: List[dict],
        atr_value: float,
    ) -> tuple:
        """
        Check if candle is near any zone. Returns (near_zone, zone_type).
        Zone types in priority: Support, Resistance, Supply, Demand, Trendline, None
        """
        if not zones or atr_value <= 0:
            return False, "None"

        candle_center = (candle_high + candle_low) / 2
        proximity = atr_value * self.zone_proximity_atr

        # Check each zone — find closest one whose range overlaps proximity band
        for zone in zones:
            zt = zone.get("type", "None")
            z_top = float(zone.get("zone_top", zone.get("zone_high", 0)))
            z_bot = float(zone.get("zone_bottom", zone.get("zone_low", 0)))
            z_center = (z_top + z_bot) / 2

            # Distance from candle center to nearest zone boundary
            if z_bot <= candle_center <= z_top:
                # Candle inside zone — definitely near
                return True, zt
            else:
                dist = min(abs(candle_center - z_top), abs(candle_center - z_bot))
                if dist <= proximity:
                    return True, zt

        return False, "None"

    def _make_pattern(
        self,
        name: str,
        ptype: str,
        direction: str,
        df: pd.DataFrame,
        i: int,
        zones: List[dict],
        atr_value: float,
    ) -> DetectedPattern:
        """Build a DetectedPattern with zone confluence validation."""
        row = df.iloc[i]
        near_zone, zone_type = self._check_zone_confluence(
            float(row["high"]), float(row["low"]), zones, atr_value
        )
        reliability = "High" if near_zone else "Low"

        # Use ISO timestamp if available, else index
        try:
            ts = str(df.index[i])
        except Exception as e:
            ts = str(i)

        return DetectedPattern(
            pattern_name=name,
            type=ptype,
            candle_index_or_time=ts,
            near_zone=near_zone,
            zone_type=zone_type,
            reliability=reliability,
            direction=direction,
            candle_index=i,
        )

    # ═══════════════════════════════════════════════════════════
    # SINGLE-CANDLE PATTERNS
    # ═══════════════════════════════════════════════════════════

    def _detect_hammer(self, df, i, zones, atr):
        """1. Hammer: long lower wick (≥2× body), small/no upper wick, bullish reversal."""
        c = _candle_metrics(df.iloc[i])
        if c["body"] < 1e-9 or c["range"] <= 0:
            return None
        # Lower wick ≥ 2× body
        if c["lower_wick"] < c["body"] * HAMMER_WICK_BODY_RATIO:
            return None
        # Upper wick small (≤ 30% of range)
        if c["upper_wick"] > c["range"] * 0.30:
            return None
        # Body in upper half of range
        body_pos = (min(c["open"], c["close"]) - c["low"]) / c["range"]
        if body_pos < 0.40:
            return None
        return self._make_pattern("Hammer", "Reversal", "bullish", df, i, zones, atr)

    def _detect_shooting_star(self, df, i, zones, atr):
        """2. Shooting Star: long upper wick (≥2× body), small/no lower wick, bearish reversal."""
        c = _candle_metrics(df.iloc[i])
        if c["body"] < 1e-9 or c["range"] <= 0:
            return None
        if c["upper_wick"] < c["body"] * SHOOTING_STAR_WICK_RATIO:
            return None
        if c["lower_wick"] > c["range"] * 0.25:
            return None
        body_pos = (min(c["open"], c["close"]) - c["low"]) / c["range"]
        if body_pos >= 0.40:
            return None
        return self._make_pattern("Shooting Star", "Reversal", "bearish", df, i, zones, atr)

    def _detect_inverted_hammer(self, df, i, zones, atr):
        """3. Inverted Hammer: Hammer's shape but in downtrend (needs confirmation)."""
        c = _candle_metrics(df.iloc[i])
        if c["body"] < 1e-9 or c["range"] <= 0:
            return None
        # Same shape as Shooting Star (long upper wick, small body, small lower wick)
        if c["upper_wick"] < c["body"] * SHOOTING_STAR_WICK_RATIO:
            return None
        if c["lower_wick"] > c["range"] * 0.25:
            return None
        # Body in lower half (inverted hammer characteristic)
        body_pos = (min(c["open"], c["close"]) - c["low"]) / c["range"]
        if body_pos >= 0.50:
            return None
        # Must be in downtrend context (last 5 candles trending down)
        if i >= 5:
            recent_close = df["close"].iloc[i-5:i+1].values
            if recent_close[-1] >= recent_close[0]:
                return None  # not a downtrend
        return self._make_pattern("Inverted Hammer", "Reversal", "bullish", df, i, zones, atr)

    def _detect_hanging_man(self, df, i, zones, atr):
        """4. Hanging Man: Hammer's shape but in uptrend (bearish warning)."""
        c = _candle_metrics(df.iloc[i])
        if c["body"] < 1e-9 or c["range"] <= 0:
            return None
        # Hammer shape
        if c["lower_wick"] < c["body"] * HAMMER_WICK_BODY_RATIO:
            return None
        if c["upper_wick"] > c["range"] * 0.30:
            return None
        body_pos = (min(c["open"], c["close"]) - c["low"]) / c["range"]
        if body_pos < 0.40:
            return None
        # Must be in uptrend context
        if i >= 5:
            recent_close = df["close"].iloc[i-5:i+1].values
            if recent_close[-1] <= recent_close[0]:
                return None  # not an uptrend
        return self._make_pattern("Hanging Man", "Reversal", "bearish", df, i, zones, atr)

    def _detect_doji(self, df, i, zones, atr):
        """5. Doji: Open≈Close (body ≤ 5% of range), indecision."""
        c = _candle_metrics(df.iloc[i])
        if c["range"] <= 0:
            return None
        if c["body_pct"] > DOJI_BODY_THRESHOLD_PCT:
            return None
        return self._make_pattern("Doji", "Indecision", "neutral", df, i, zones, atr)

    def _detect_bullish_marubozu(self, df, i, zones, atr):
        """6. Bullish Marubozu: no/tiny wick, full bullish body."""
        c = _candle_metrics(df.iloc[i])
        if c["range"] <= 0 or not c["is_bullish"]:
            return None
        if c["body_pct"] < MARUBOZU_BODY_RATIO:
            return None
        if c["upper_wick"] > c["range"] * MARUBOZU_MAX_WICK_PCT:
            return None
        if c["lower_wick"] > c["range"] * MARUBOZU_MAX_WICK_PCT:
            return None
        return self._make_pattern("Bullish Marubozu", "Continuation", "bullish", df, i, zones, atr)

    def _detect_bearish_marubozu(self, df, i, zones, atr):
        """7. Bearish Marubozu: no/tiny wick, full bearish body."""
        c = _candle_metrics(df.iloc[i])
        if c["range"] <= 0 or not c["is_bearish"]:
            return None
        if c["body_pct"] < MARUBOZU_BODY_RATIO:
            return None
        if c["upper_wick"] > c["range"] * MARUBOZU_MAX_WICK_PCT:
            return None
        if c["lower_wick"] > c["range"] * MARUBOZU_MAX_WICK_PCT:
            return None
        return self._make_pattern("Bearish Marubozu", "Continuation", "bearish", df, i, zones, atr)

    # ═══════════════════════════════════════════════════════════
    # TWO-CANDLE PATTERNS
    # ═══════════════════════════════════════════════════════════

    def _detect_bullish_engulfing(self, df, i, zones, atr):
        """8. Bullish Engulfing: 2nd bullish candle engulfs 1st bearish body."""
        if i < 1:
            return None
        c1 = _candle_metrics(df.iloc[i-1])
        c2 = _candle_metrics(df.iloc[i])
        if c1["body"] < 1e-9 or c2["body"] < 1e-9:
            return None
        # 1st bearish, 2nd bullish
        if not (c1["is_bearish"] and c2["is_bullish"]):
            return None
        # 2nd body engulfs 1st body
        if not (c2["open"] <= c1["close"] and c2["close"] >= c1["open"]):
            return None
        # 2nd body larger than 1st
        if c2["body"] <= c1["body"]:
            return None
        return self._make_pattern("Bullish Engulfing", "Reversal", "bullish", df, i, zones, atr)

    def _detect_bearish_engulfing(self, df, i, zones, atr):
        """9. Bearish Engulfing: 2nd bearish candle engulfs 1st bullish body."""
        if i < 1:
            return None
        c1 = _candle_metrics(df.iloc[i-1])
        c2 = _candle_metrics(df.iloc[i])
        if c1["body"] < 1e-9 or c2["body"] < 1e-9:
            return None
        if not (c1["is_bullish"] and c2["is_bearish"]):
            return None
        if not (c2["open"] >= c1["close"] and c2["close"] <= c1["open"]):
            return None
        if c2["body"] <= c1["body"]:
            return None
        return self._make_pattern("Bearish Engulfing", "Reversal", "bearish", df, i, zones, atr)

    def _detect_tweezer_top(self, df, i, zones, atr):
        """10. Tweezer Top: equal high rejected twice, bearish."""
        if i < 1:
            return None
        h1 = float(df.iloc[i-1]["high"])
        h2 = float(df.iloc[i]["high"])
        if h1 <= 0 or h2 <= 0:
            return None
        # Equal highs (within tolerance)
        if abs(h1 - h2) / max(h1, h2) > TWEEZER_TOLERANCE_PCT:
            return None
        # 1st bullish, 2nd bearish (rejection)
        c1 = _candle_metrics(df.iloc[i-1])
        c2 = _candle_metrics(df.iloc[i])
        if not (c1["is_bullish"] and c2["is_bearish"]):
            return None
        return self._make_pattern("Tweezer Top", "Reversal", "bearish", df, i, zones, atr)

    def _detect_tweezer_bottom(self, df, i, zones, atr):
        """11. Tweezer Bottom: equal low rejected twice, bullish."""
        if i < 1:
            return None
        l1 = float(df.iloc[i-1]["low"])
        l2 = float(df.iloc[i]["low"])
        if l1 <= 0 or l2 <= 0:
            return None
        if abs(l1 - l2) / max(l1, l2) > TWEEZER_TOLERANCE_PCT:
            return None
        c1 = _candle_metrics(df.iloc[i-1])
        c2 = _candle_metrics(df.iloc[i])
        if not (c1["is_bearish"] and c2["is_bullish"]):
            return None
        return self._make_pattern("Tweezer Bottom", "Reversal", "bullish", df, i, zones, atr)

    def _detect_piercing_line(self, df, i, zones, atr):
        """12. Piercing Line: bullish candle closes above 50% of prior bearish body."""
        if i < 1:
            return None
        c1 = _candle_metrics(df.iloc[i-1])
        c2 = _candle_metrics(df.iloc[i])
        if c1["body"] < 1e-9 or c2["body"] < 1e-9:
            return None
        # 1st bearish, 2nd bullish
        if not (c1["is_bearish"] and c2["is_bullish"]):
            return None
        # 2nd opens below 1st low (gap down) — strict definition
        # OR 2nd opens within 1st body (relaxed)
        # 2nd close must be above 50% of 1st body
        c1_midpoint = (c1["open"] + c1["close"]) / 2
        if c2["close"] <= c1_midpoint:
            return None
        # 2nd close must be below 1st open (not full engulfing)
        if c2["close"] >= c1["open"]:
            return None
        return self._make_pattern("Piercing Line", "Reversal", "bullish", df, i, zones, atr)

    def _detect_dark_cloud_cover(self, df, i, zones, atr):
        """13. Dark Cloud Cover: bearish candle closes below 50% of prior bullish body."""
        if i < 1:
            return None
        c1 = _candle_metrics(df.iloc[i-1])
        c2 = _candle_metrics(df.iloc[i])
        if c1["body"] < 1e-9 or c2["body"] < 1e-9:
            return None
        if not (c1["is_bullish"] and c2["is_bearish"]):
            return None
        c1_midpoint = (c1["open"] + c1["close"]) / 2
        if c2["close"] >= c1_midpoint:
            return None
        if c2["close"] <= c1["open"]:
            return None
        return self._make_pattern("Dark Cloud Cover", "Reversal", "bearish", df, i, zones, atr)

    def _detect_harami(self, df, i, zones, atr):
        """14. Bullish/Bearish Harami: small opposite candle inside large candle."""
        if i < 1:
            return None
        c1 = _candle_metrics(df.iloc[i-1])
        c2 = _candle_metrics(df.iloc[i])
        if c1["body"] < 1e-9 or c2["body"] < 1e-9:
            return None
        # 2nd body must be ≤ 50% of 1st body
        if c2["body"] > c1["body"] * HARAMI_SMALL_BODY_PCT:
            return None
        # 2nd body must be inside 1st body range
        if not (c2["open"] >= min(c1["open"], c1["close"])
                and c2["close"] <= max(c1["open"], c1["close"])):
            return None
        # Direction: bullish harami (1st bearish, 2nd bullish) or bearish harami (1st bullish, 2nd bearish)
        if c1["is_bearish"] and c2["is_bullish"]:
            return self._make_pattern("Bullish Harami", "Reversal", "bullish", df, i, zones, atr)
        if c1["is_bullish"] and c2["is_bearish"]:
            return self._make_pattern("Bearish Harami", "Reversal", "bearish", df, i, zones, atr)
        return None

    # ═══════════════════════════════════════════════════════════
    # THREE-CANDLE PATTERNS
    # ═══════════════════════════════════════════════════════════

    def _detect_morning_star(self, df, i, zones, atr):
        """15. Morning Star: bearish → indecision (small body) → bullish, confirmed bullish reversal."""
        if i < 2:
            return None
        c1 = _candle_metrics(df.iloc[i-2])
        c2 = _candle_metrics(df.iloc[i-1])
        c3 = _candle_metrics(df.iloc[i])
        if c1["body"] < 1e-9 or c3["body"] < 1e-9:
            return None
        # 1st bearish, 3rd bullish
        if not (c1["is_bearish"] and c3["is_bullish"]):
            return None
        # 2nd small body (indecision)
        if c2["body_pct"] > STAR_MIDDLE_BODY_PCT:
            return None
        # 3rd closes above 1st midpoint
        c1_midpoint = (c1["open"] + c1["close"]) / 2
        if c3["close"] <= c1_midpoint:
            return None
        return self._make_pattern("Morning Star", "Reversal", "bullish", df, i, zones, atr)

    def _detect_evening_star(self, df, i, zones, atr):
        """16. Evening Star: bullish → indecision → bearish, confirmed bearish reversal."""
        if i < 2:
            return None
        c1 = _candle_metrics(df.iloc[i-2])
        c2 = _candle_metrics(df.iloc[i-1])
        c3 = _candle_metrics(df.iloc[i])
        if c1["body"] < 1e-9 or c3["body"] < 1e-9:
            return None
        if not (c1["is_bullish"] and c3["is_bearish"]):
            return None
        if c2["body_pct"] > STAR_MIDDLE_BODY_PCT:
            return None
        c1_midpoint = (c1["open"] + c1["close"]) / 2
        if c3["close"] >= c1_midpoint:
            return None
        return self._make_pattern("Evening Star", "Reversal", "bearish", df, i, zones, atr)

    def _detect_three_white_soldiers(self, df, i, zones, atr):
        """17. Three White Soldiers: 3 consecutive large bullish candles.
        
        Book Spread 32 rules:
          - 3 consecutive bullish candles
          - Each has little/no long shadows (body ≥ 60% of range)
          - Each next candle opens WITHIN prior candle's body range
            (Open(C2) ∈ [Open(C1), Close(C1)], Open(C3) ∈ [Open(C2), Close(C2)])
          - Occurs at end of downtrend
        """
        if i < 2:
            return None
        c1 = _candle_metrics(df.iloc[i-2])
        c2 = _candle_metrics(df.iloc[i-1])
        c3 = _candle_metrics(df.iloc[i])
        # All 3 bullish
        if not (c1["is_bullish"] and c2["is_bullish"] and c3["is_bullish"]):
            return None
        # All 3 large body (≥60% of range) — book: "no long shadows"
        if (c1["body_pct"] < SOLDIERS_MIN_BODY_PCT
            or c2["body_pct"] < SOLDIERS_MIN_BODY_PCT
            or c3["body_pct"] < SOLDIERS_MIN_BODY_PCT):
            return None
        # Higher highs + higher closes (progressive)
        if not (c2["close"] > c1["close"] and c3["close"] > c2["close"]):
            return None
        if not (c2["high"] >= c1["high"] and c3["high"] >= c2["high"]):
            return None
        # Book rule: Each opens WITHIN prior body range [min(open,close), max(open,close)]
        # (not just near close — full body range)
        c1_body_low = min(c1["open"], c1["close"])
        c1_body_high = max(c1["open"], c1["close"])
        c2_body_low = min(c2["open"], c2["close"])
        c2_body_high = max(c2["open"], c2["close"])
        if not (c1_body_low <= c2["open"] <= c1_body_high):
            return None  # C2 open not within C1 body
        if not (c2_body_low <= c3["open"] <= c2_body_high):
            return None  # C3 open not within C2 body
        return self._make_pattern("Three White Soldiers", "Continuation", "bullish", df, i, zones, atr)

    def _detect_three_black_crows(self, df, i, zones, atr):
        """18. Three Black Crows: 3 consecutive large bearish candles.
        
        Book Spread 43 rules (mirror of Three White Soldiers):
          - 3 consecutive bearish candles
          - Each has little/no long shadows (body ≥ 60% of range)
          - Each next candle opens WITHIN prior candle's body range
          - Occurs at end of uptrend
        """
        if i < 2:
            return None
        c1 = _candle_metrics(df.iloc[i-2])
        c2 = _candle_metrics(df.iloc[i-1])
        c3 = _candle_metrics(df.iloc[i])
        if not (c1["is_bearish"] and c2["is_bearish"] and c3["is_bearish"]):
            return None
        if (c1["body_pct"] < SOLDIERS_MIN_BODY_PCT
            or c2["body_pct"] < SOLDIERS_MIN_BODY_PCT
            or c3["body_pct"] < SOLDIERS_MIN_BODY_PCT):
            return None
        # Lower lows + lower closes (progressive)
        if not (c2["close"] < c1["close"] and c3["close"] < c2["close"]):
            return None
        if not (c2["low"] <= c1["low"] and c3["low"] <= c2["low"]):
            return None
        # Book rule: Each opens WITHIN prior body range
        c1_body_low = min(c1["open"], c1["close"])
        c1_body_high = max(c1["open"], c1["close"])
        c2_body_low = min(c2["open"], c2["close"])
        c2_body_high = max(c2["open"], c2["close"])
        if not (c1_body_low <= c2["open"] <= c1_body_high):
            return None
        if not (c2_body_low <= c3["open"] <= c2_body_high):
            return None
        return self._make_pattern("Three Black Crows", "Continuation", "bearish", df, i, zones, atr)

    def _detect_three_inside_up(self, df, i, zones, atr):
        """19. Three Inside Up: Bullish Harami + bullish confirmation candle."""
        if i < 2:
            return None
        c1 = _candle_metrics(df.iloc[i-2])
        c2 = _candle_metrics(df.iloc[i-1])
        c3 = _candle_metrics(df.iloc[i])
        # c1 bearish large, c2 small bullish inside (bullish harami)
        if not c1["is_bearish"]:
            return None
        if not c2["is_bullish"]:
            return None
        if c2["body"] > c1["body"] * HARAMI_SMALL_BODY_PCT:
            return None
        if not (c2["open"] >= min(c1["open"], c1["close"])
                and c2["close"] <= max(c1["open"], c1["close"])):
            return None
        # c3 bullish confirmation (close above c2 high, ideally above c1 open)
        if not c3["is_bullish"]:
            return None
        if c3["close"] <= c2["high"]:
            return None
        return self._make_pattern("Three Inside Up", "Reversal", "bullish", df, i, zones, atr)

    def _detect_three_inside_down(self, df, i, zones, atr):
        """20. Three Inside Down: Bearish Harami + bearish confirmation candle."""
        if i < 2:
            return None
        c1 = _candle_metrics(df.iloc[i-2])
        c2 = _candle_metrics(df.iloc[i-1])
        c3 = _candle_metrics(df.iloc[i])
        if not c1["is_bullish"]:
            return None
        if not c2["is_bearish"]:
            return None
        if c2["body"] > c1["body"] * HARAMI_SMALL_BODY_PCT:
            return None
        if not (c2["open"] >= min(c1["open"], c1["close"])
                and c2["close"] <= max(c1["open"], c1["close"])):
            return None
        if not c3["is_bearish"]:
            return None
        if c3["close"] >= c2["low"]:
            return None
        return self._make_pattern("Three Inside Down", "Reversal", "bearish", df, i, zones, atr)

    # ═══════════════════════════════════════════════════════════
    # MULTI-BAR REPETITION ANALYSIS
    # ═══════════════════════════════════════════════════════════

    def analyze_repetition(self, patterns: List[DetectedPattern]) -> dict:
        """
        Per spec rule 3 — multi-bar repetition analysis:
          - Same reversal pattern at same zone 2+ times → zone strength boost
          - 3+ consecutive Marubozu/Soldier/Crow same direction → Momentum sequence
          - Multiple consecutive Doji → Consolidation → lean toward WAIT

        Returns dict with:
          - "zone_strength_boosts": list of {zone, pattern, new_strength}
          - "momentum_sequence": {detected: bool, direction: str, count: int} | None
          - "consolidation_detected": bool
        """
        if not patterns:
            return {
                "zone_strength_boosts": [],
                "momentum_sequence": None,
                "consolidation_detected": False,
            }

        # ── Group reversal patterns by (name, zone_type) ──
        reversal_groups: Dict[tuple, List[DetectedPattern]] = {}
        for p in patterns:
            if p.type == "Reversal" and p.near_zone:
                key = (p.pattern_name, p.zone_type)
                reversal_groups.setdefault(key, []).append(p)

        zone_boosts = []
        for (pname, ztype), plist in reversal_groups.items():
            if len(plist) >= 2:
                zone_boosts.append({
                    "pattern": pname,
                    "zone_type": ztype,
                    "occurrences": len(plist),
                    "strength_boost": "Weak→Medium" if len(plist) == 2 else "Medium→Strong",
                })

        # ── Check 3+ consecutive Marubozu/Soldier/Crow same direction ──
        momentum_patterns = [p for p in patterns
                             if p.pattern_name in ("Bullish Marubozu", "Bearish Marubozu",
                                                    "Three White Soldiers", "Three Black Crows")]
        # Group by direction and check consecutive indices
        momentum_seq = None
        for direction in ("bullish", "bearish"):
            dir_patterns = sorted(
                [p for p in momentum_patterns if p.direction == direction],
                key=lambda p: p.candle_index,
            )
            # Find longest consecutive run
            longest = 1
            current = 1
            for j in range(1, len(dir_patterns)):
                if dir_patterns[j].candle_index - dir_patterns[j-1].candle_index <= 2:
                    current += 1
                    longest = max(longest, current)
                else:
                    current = 1
            if longest >= 3:
                momentum_seq = {
                    "detected": True,
                    "direction": direction,
                    "count": longest,
                }
                break

        # ── Multiple consecutive Doji → consolidation ──
        doji_patterns = sorted(
            [p for p in patterns if p.pattern_name == "Doji"],
            key=lambda p: p.candle_index,
        )
        consolidation = False
        if len(doji_patterns) >= 2:
            for j in range(1, len(doji_patterns)):
                if doji_patterns[j].candle_index - doji_patterns[j-1].candle_index <= 1:
                    consolidation = True
                    break

        return {
            "zone_strength_boosts": zone_boosts,
            "momentum_sequence": momentum_seq,
            "consolidation_detected": bool(consolidation),
        }

    # ═══════════════════════════════════════════════════════════
    # SUMMARY OUTPUT
    # ═══════════════════════════════════════════════════════════

    def to_spec_dicts(self, patterns: List[DetectedPattern]) -> List[dict]:
        """Convert list of DetectedPattern to spec-compliant list of dicts."""
        return [p.to_spec_dict() for p in patterns]

    def to_prompt_text(self, patterns: List[DetectedPattern]) -> str:
        """Plain-text rendering for LLM prompts."""
        if not patterns:
            return "=== HIGH-RELIABILITY PATTERNS ===\n(none detected in lookback window)\n" + "=" * 40

        lines = ["=== HIGH-RELIABILITY PATTERNS ==="]
        for p in patterns:
            reliability_emoji = "🟢" if p.reliability == "High" else "⚪"
            zone_str = f"near {p.zone_type}" if p.near_zone else "mid-range"
            lines.append(
                f"  {reliability_emoji} {p.pattern_name} ({p.type}, {p.direction}) "
                f"@ {p.candle_index_or_time} | {zone_str} | Reliability: {p.reliability}"
            )
        lines.append("=" * 40)
        return "\n".join(lines)


# ============================================================
# Convenience: one-shot helper
# ============================================================

def detect_high_reliability_patterns(
    df: pd.DataFrame,
    zones: Optional[List[dict]] = None,
    atr_value: Optional[float] = None,
    lookback: int = 10,
) -> List[dict]:
    """
    One-shot helper — returns spec-compliant list of detected pattern dicts.

    Output schema per item:
      {
        "pattern_name": str,
        "type": "Reversal" | "Continuation" | "Indecision",
        "candle_index_or_time": str,
        "near_zone": bool,
        "zone_type": "Support" | "Resistance" | "Supply" | "Demand" | "Trendline" | "None",
        "reliability": "Low" | "High"
      }
    """
    detector = HighReliabilityPatternDetector(lookback=lookback)
    patterns = detector.detect(df, zones=zones, atr_value=atr_value)
    return detector.to_spec_dicts(patterns)


# ============================================================
# CLI entry
# ============================================================
if __name__ == "__main__":
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-06-01", periods=n, freq="h")
    base = 1.0850
    close = base + np.cumsum(np.random.randn(n) * 0.0005)
    df = pd.DataFrame({
        "open":  close + np.random.randn(n) * 0.0002,
        "high":  close + abs(np.random.randn(n)) * 0.0008,
        "low":   close - abs(np.random.randn(n)) * 0.0008,
        "close": close,
    }, index=dates)

    zones = [
        {"type": "Resistance", "zone_top": 1.0900, "zone_bottom": 1.0895},
        {"type": "Support", "zone_top": 1.0800, "zone_bottom": 1.0795},
    ]

    detector = HighReliabilityPatternDetector(lookback=20)
    patterns = detector.detect(df, zones=zones)
    print(detector.to_prompt_text(patterns))
    print()
    import json
    print("--- Spec JSON output ---")
    print(json.dumps(detector.to_spec_dicts(patterns), indent=2))
