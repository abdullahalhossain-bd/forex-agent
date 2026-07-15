# analysis/pin_bar_strategy.py
# ============================================================
# Pin Bar Candlestick Pattern Strategy
# ============================================================
# Book: "The Candlestick Trading Bible" Pages 81-95
#
# Implements the book's full pin bar trading strategy:
#   3 Filter Criteria (pages 83-87):
#     1. Timeframe ≥ 4H/D1 (page 83) — smaller TFs = false signals
#     2. Trend alignment (page 84) — pin bar direction must match trend
#     3. Key level confluence (page 85) — pin bar must form at S/R or 21-MA
#
#   2 Entry Tactics (pages 92-95):
#     A. Aggressive entry (page 92-93):
#        - Entry: after pin bar close
#        - SL: beyond pin bar's tail (wick extreme + buffer)
#        - TP: next opposing S/R level
#     B. Conservative entry (page 93-95):
#        - Entry: 50% retracement into pin bar's range
#        - Better R:R (up to 5:1) but may miss trade if no retracement
#        - SL: same as aggressive (beyond tail)
#        - TP: same as aggressive (next S/R)
#
#   21-MA as Dynamic Level (pages 90-91):
#     - In uptrend: 21-MA acts as dynamic support
#     - In downtrend: 21-MA acts as dynamic resistance
#     - Pin bar at 21-MA = high-confluence entry
# ============================================================

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ─── Constants ────────────────────────────────────────────────
# Book Page 83: only pin bars on 4H or Daily should be taken seriously
VALID_TIMEFRAMES = {"H4", "D1", "W1"}

# Book Page 93: conservative entry = 50% retracement into pin bar range
CONSERVATIVE_RETRACE_PCT = 0.50

# Book Page 93: SL goes beyond the tail
SL_BUFFER_PIPS = 2  # small buffer beyond wick extreme

# Pin bar detection thresholds (from existing patterns.py)
PIN_BAR_WICK_BODY_RATIO = 2.0  # tail must be ≥ 2× body


# ─── Dataclass ────────────────────────────────────────────────

@dataclass
class PinBarSetup:
    """A detected pin bar setup with all 3 filter criteria evaluated."""
    detected: bool = False
    direction: str = "neutral"        # "bullish" | "bearish"
    candle_index: int = -1
    candle_time: str = ""
    open: float = 0
    high: float = 0
    low: float = 0
    close: float = 0
    body: float = 0
    tail: float = 0
    wick_body_ratio: float = 0

    # Filter results
    filter_1_timeframe: bool = False   # Page 83: TF ≥ 4H
    filter_2_trend: bool = False       # Page 84: direction aligns with trend
    filter_3_level: bool = False       # Page 85: at S/R or 21-MA
    trend_direction: str = "unknown"
    level_type: str = "none"           # "support" | "resistance" | "21ma" | "none"
    level_price: float = 0

    # Entry calculations
    aggressive_entry: float = 0       # Page 92: entry at pin close
    conservative_entry: float = 0     # Page 93: 50% retracement
    stop_loss: float = 0              # Page 93: beyond tail
    take_profit: float = 0            # Page 93: next opposing S/R
    risk_reward_aggressive: float = 0
    risk_reward_conservative: float = 0

    # Quality score
    quality_score: int = 0            # 0-100
    quality_grade: str = "F"          # A/B/C/D/F

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ═════════════════════════════════════════════════════════════

class PinBarStrategy:
    """
    Book Pages 81-95 — Pin Bar Trading Strategy.

    Usage:
        strategy = PinBarStrategy(timeframe="H4")
        setup = strategy.detect(df, symbol="EURUSD",
                                 trend_direction="BULLISH",
                                 sr_zones=[...],
                                 ema_21_value=1.0850)
        if setup.detected and setup.quality_grade in ("A", "B"):
            # Trade the setup
    """

    def __init__(self, timeframe: str = "H4"):
        self.timeframe = timeframe.upper()

    # ═══════════════════════════════════════════════════════════
    # MAIN DETECTION METHOD
    # ═══════════════════════════════════════════════════════════

    def detect(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        trend_direction: str = "unknown",
        sr_zones: Optional[List[dict]] = None,
        ema_21_value: Optional[float] = None,
        atr_value: Optional[float] = None,
    ) -> PinBarSetup:
        """
        Detect pin bar on the LAST candle and evaluate all 3 filter criteria.

        Args:
            df: OHLC DataFrame
            symbol: e.g., "EURUSD"
            trend_direction: "BULLISH" | "BEARISH" | "unknown"
            sr_zones: list of {"type": "support/resistance", "zone_top": float, "zone_bottom": float}
            ema_21_value: current 21-MA value (for dynamic level check)
            atr_value: ATR for SL buffer calculation

        Returns:
            PinBarSetup with all filters + entry calculations
        """
        if df is None or len(df) < 3:
            return PinBarSetup()

        # ── Step 1: Detect pin bar on last candle ──
        setup = self._detect_pin_bar(df)
        if not setup.detected:
            return setup

        # ── Step 2: Apply 3 filter criteria ──
        # Filter 1: Timeframe (Page 83)
        setup.filter_1_timeframe = self._check_timeframe()

        # Filter 2: Trend alignment (Page 84)
        setup.trend_direction = trend_direction
        setup.filter_2_trend = self._check_trend_alignment(setup.direction, trend_direction)

        # Filter 3: Key level confluence (Page 85)
        level_result = self._check_level_confluence(
            setup.close, sr_zones, ema_21_value, atr_value
        )
        setup.filter_3_level = level_result["at_level"]
        setup.level_type = level_result["level_type"]
        setup.level_price = level_result["level_price"]

        # ── Step 3: Calculate entry tactics ──
        self._calculate_entries(df, setup, sr_zones, atr_value)

        # ── Step 4: Quality score ──
        self._score_quality(setup)

        return setup

    # ═══════════════════════════════════════════════════════════
    # PIN BAR DETECTION
    # ═══════════════════════════════════════════════════════════

    def _detect_pin_bar(self, df: pd.DataFrame) -> PinBarSetup:
        """Detect pin bar on the last candle (Book Page 81 anatomy)."""
        last = df.iloc[-1]
        o, h, l, c = float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"])
        body = abs(c - o)
        total_range = h - l

        if body < 1e-9 or total_range <= 0:
            return PinBarSetup()

        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        # Pin bar = one wick much longer than body (≥ 2×)
        # Bullish pin bar (hammer): long LOWER wick
        # Bearish pin bar (shooting star): long UPPER wick
        is_bullish_pin = lower_wick >= body * PIN_BAR_WICK_BODY_RATIO and lower_wick > upper_wick * 1.5
        is_bearish_pin = upper_wick >= body * PIN_BAR_WICK_BODY_RATIO and upper_wick > lower_wick * 1.5

        if not (is_bullish_pin or is_bearish_pin):
            return PinBarSetup()

        direction = "bullish" if is_bullish_pin else "bearish"
        tail = lower_wick if is_bullish_pin else upper_wick
        wick_body_ratio = tail / body if body > 0 else 0

        try:
            candle_time = str(df.index[-1])
        except Exception as e:
            candle_time = ""

        return PinBarSetup(
            detected=True,
            direction=direction,
            candle_index=len(df) - 1,
            candle_time=candle_time,
            open=o, high=h, low=l, close=c,
            body=body, tail=tail,
            wick_body_ratio=round(wick_body_ratio, 2),
        )

    # ═══════════════════════════════════════════════════════════
    # FILTER 1: TIMEFRAME (Page 83)
    # ═══════════════════════════════════════════════════════════

    def _check_timeframe(self) -> bool:
        """Book Page 83: only pin bars on 4H/Daily/W1 should be taken seriously."""
        return self.timeframe in VALID_TIMEFRAMES

    # ═══════════════════════════════════════════════════════════
    # FILTER 2: TREND ALIGNMENT (Page 84)
    # ═══════════════════════════════════════════════════════════

    def _check_trend_alignment(self, pin_direction: str, trend_direction: str) -> bool:
        """Book Page 84: pin bar direction must align with prevailing trend.

        Bullish pin bar in BULLISH trend → aligned
        Bearish pin bar in BEARISH trend → aligned
        Counter-trend pin bars → discard for beginner mode
        """
        if trend_direction.upper() == "UNKNOWN":
            return False  # don't trade without trend context

        if pin_direction == "bullish" and trend_direction.upper() == "BULLISH":
            return True
        if pin_direction == "bearish" and trend_direction.upper() == "BEARISH":
            return True
        return False

    # ═══════════════════════════════════════════════════════════
    # FILTER 3: KEY LEVEL CONFLUENCE (Page 85)
    # ═══════════════════════════════════════════════════════════

    def _check_level_confluence(
        self,
        close_price: float,
        sr_zones: Optional[List[dict]],
        ema_21_value: Optional[float],
        atr_value: Optional[float],
    ) -> dict:
        """Book Page 85: pin bar must form at a key S/R level or 21-MA.

        Checks (in priority):
          1. S/R zone proximity (within 0.5×ATR of a zone)
          2. 21-MA proximity (within 0.5×ATR of EMA 21)

        Book Pages 90-91: 21-MA acts as dynamic support (uptrend) / resistance (downtrend).
        """
        if atr_value is None or atr_value <= 0:
            atr_value = close_price * 0.001  # fallback

        proximity = atr_value * 0.5  # within 0.5×ATR = "at level"

        # Check S/R zones first
        if sr_zones:
            for zone in sr_zones:
                z_top = float(zone.get("zone_top", 0))
                z_bot = float(zone.get("zone_bottom", 0))
                z_center = (z_top + z_bot) / 2

                if abs(close_price - z_center) <= proximity:
                    return {
                        "at_level": True,
                        "level_type": zone.get("type", "support/resistance"),
                        "level_price": z_center,
                    }

        # Check 21-MA (Book Pages 90-91)
        if ema_21_value and abs(close_price - ema_21_value) <= proximity:
            return {
                "at_level": True,
                "level_type": "21ma",
                "level_price": ema_21_value,
            }

        return {"at_level": False, "level_type": "none", "level_price": 0}

    # ═══════════════════════════════════════════════════════════
    # ENTRY CALCULATIONS (Pages 92-95)
    # ═══════════════════════════════════════════════════════════

    def _calculate_entries(
        self, df: pd.DataFrame, setup: PinBarSetup,
        sr_zones: Optional[List[dict]], atr_value: Optional[float]
    ) -> None:
        """Calculate aggressive + conservative entry, SL, TP (Book Pages 92-95)."""
        if not setup.detected:
            return

        pip_value = 0.0001  # default for FX
        if atr_value is None or atr_value <= 0:
            atr_value = setup.close * 0.001

        # ── Aggressive entry (Page 92): enter at pin bar close ──
        setup.aggressive_entry = setup.close

        # ── Conservative entry (Page 93): 50% retracement into pin bar range ──
        pin_range = setup.high - setup.low
        if setup.direction == "bullish":
            # Bullish pin: retrace DOWN 50% from high
            setup.conservative_entry = setup.high - pin_range * CONSERVATIVE_RETRACE_PCT
        else:
            # Bearish pin: retrace UP 50% from low
            setup.conservative_entry = setup.low + pin_range * CONSERVATIVE_RETRACE_PCT

        # ── Stop loss (Page 93): beyond the tail ──
        sl_buffer = atr_value * 0.1  # small buffer
        if setup.direction == "bullish":
            setup.stop_loss = setup.low - sl_buffer  # below the lower wick
        else:
            setup.stop_loss = setup.high + sl_buffer  # above the upper wick

        # ── Take profit (Page 93): next opposing S/R level ──
        if setup.direction == "bullish":
            # Look for nearest resistance ABOVE entry
            tp_candidates = [
                float(z.get("zone_bottom", 0)) for z in (sr_zones or [])
                if float(z.get("zone_bottom", 0)) > setup.aggressive_entry
            ]
            setup.take_profit = min(tp_candidates) if tp_candidates else setup.aggressive_entry + atr_value * 3
        else:
            # Look for nearest support BELOW entry
            tp_candidates = [
                float(z.get("zone_top", 0)) for z in (sr_zones or [])
                if float(z.get("zone_top", 0)) < setup.aggressive_entry and float(z.get("zone_top", 0)) > 0
            ]
            setup.take_profit = max(tp_candidates) if tp_candidates else setup.aggressive_entry - atr_value * 3

        # ── Risk:Reward calculations ──
        risk_agg = abs(setup.aggressive_entry - setup.stop_loss)
        reward_agg = abs(setup.take_profit - setup.aggressive_entry)
        setup.risk_reward_aggressive = round(reward_agg / risk_agg, 2) if risk_agg > 0 else 0

        risk_cons = abs(setup.conservative_entry - setup.stop_loss)
        reward_cons = abs(setup.take_profit - setup.conservative_entry)
        setup.risk_reward_conservative = round(reward_cons / risk_cons, 2) if risk_cons > 0 else 0

    # ═══════════════════════════════════════════════════════════
    # QUALITY SCORING
    # ═══════════════════════════════════════════════════════════

    def _score_quality(self, setup: PinBarSetup) -> None:
        """Score the setup quality based on filter pass rate + wick ratio.

        Book Page 89: 3-criteria quality checklist (timeframe + trend + level).
        All 3 must pass for a high-quality setup.
        """
        score = 0

        # Filter 1: Timeframe (25 points)
        if setup.filter_1_timeframe:
            score += 25

        # Filter 2: Trend alignment (25 points)
        if setup.filter_2_trend:
            score += 25

        # Filter 3: Key level confluence (25 points)
        if setup.filter_3_level:
            score += 25

        # Wick/body ratio bonus (up to 25 points)
        # Book Page 81: "longer tails are more powerful"
        if setup.wick_body_ratio >= 4:
            score += 25
        elif setup.wick_body_ratio >= 3:
            score += 20
        elif setup.wick_body_ratio >= 2:
            score += 15

        setup.quality_score = min(100, score)

        # Grade
        if score >= 90:
            setup.quality_grade = "A"
        elif score >= 70:
            setup.quality_grade = "B"
        elif score >= 50:
            setup.quality_grade = "C"
        elif score >= 30:
            setup.quality_grade = "D"
        else:
            setup.quality_grade = "F"

    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════

    def get_summary(self, setup: PinBarSetup) -> str:
        """Human-readable summary of the pin bar setup."""
        if not setup.detected:
            return "No pin bar detected on last candle."

        lines = [
            f"=== PIN BAR STRATEGY ({self.timeframe}) ===",
            f"Direction: {setup.direction.upper()}",
            f"Wick/Body: {setup.wick_body_ratio}×",
            f"",
            f"--- 3 Filter Criteria (Book Pages 83-85) ---",
            f"  1. Timeframe ≥ 4H: {'✅' if setup.filter_1_timeframe else '❌'} ({self.timeframe})",
            f"  2. Trend alignment: {'✅' if setup.filter_2_trend else '❌'} (trend={setup.trend_direction})",
            f"  3. Key level: {'✅' if setup.filter_3_level else '❌'} ({setup.level_type} @ {setup.level_price:.5f})",
            f"",
            f"--- Entry Tactics (Book Pages 92-95) ---",
            f"  Aggressive entry: {setup.aggressive_entry:.5f} (at pin close)",
            f"  Conservative entry: {setup.conservative_entry:.5f} (50% retracement)",
            f"  Stop loss: {setup.stop_loss:.5f} (beyond tail)",
            f"  Take profit: {setup.take_profit:.5f} (next S/R)",
            f"",
            f"  R:R Aggressive: 1:{setup.risk_reward_aggressive}",
            f"  R:R Conservative: 1:{setup.risk_reward_conservative}",
            f"",
            f"  Quality: {setup.quality_score}/100 (Grade: {setup.quality_grade})",
            f"{'='*50}",
        ]
        return "\n".join(lines)


# ============================================================
# CLI entry
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-06-01", periods=n, freq="4h")
    close = 1.0850 + np.cumsum(np.random.randn(n) * 0.0005)
    # Force a bullish pin bar on last candle
    close[-1] = 1.0840
    df = pd.DataFrame({
        "open": close,
        "high": close + 0.0003,
        "low": close - 0.0003,
        "close": close,
    }, index=dates)
    # Make last candle a pin bar
    df.iloc[-1, df.columns.get_loc("open")] = 1.0842
    df.iloc[-1, df.columns.get_loc("high")] = 1.0845
    df.iloc[-1, df.columns.get_loc("low")] = 1.0825  # long lower wick
    df.iloc[-1, df.columns.get_loc("close")] = 1.0840  # small body

    sr_zones = [
        {"type": "support", "zone_top": 1.0845, "zone_bottom": 1.0835},
        {"type": "resistance", "zone_top": 1.0900, "zone_bottom": 1.0895},
    ]

    strategy = PinBarStrategy(timeframe="H4")
    setup = strategy.detect(
        df, symbol="EURUSD",
        trend_direction="BULLISH",
        sr_zones=sr_zones,
        ema_21_value=1.0838,
        atr_value=0.0010,
    )
    print(strategy.get_summary(setup))
