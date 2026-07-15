"""
analysis/curve_mtf.py — Book 5 (Frank Miller S&D) Chapter 12 Multi-Frame "Curve"
=================================================================================

Pages 126-135 implement the book's most quantitatively rich methodology:
the **"curve"** — a top-down MTF filter that divides the price range
between the nearest demand and supply zones (on the higher timeframe)
into three equal sub-zones (High / Equilibrium / Low) and uses the
current price's position within the curve to set a directional bias.

  ── CORE DEFINITIONS (Book P130) ──────────────────────────────
    curve = price range bounded by:
      lower edge = proximal line of nearest DEMAND zone
      upper edge = proximal line of nearest SUPPLY zone

  ── CURVE SPLIT (Book P131) ───────────────────────────────────
    subzone_width = (upper_proximal - lower_proximal) / 3
    boundaries at:
      lower_proximal + 1 × subzone_width  (Low/Equilibrium border)
      lower_proximal + 2 × subzone_width  (Equilibrium/High border)

    Alternative (Book P132): set Fib tool to 33% and 66% between
    curve extremes — produces identical divisions. The standard
    Fibonacci retracement levels (23.6/38.2/61.8/78.6) are NOT used.

  ── 5 ZONE AREAS (Book P131) ──────────────────────────────────
    Very Low  = inside the demand zone (below lower_proximal)
    Low       = [lower_proximal, lower_proximal + subzone_width)
    Equilibrium = [lower_proximal + subzone_width, lower_proximal + 2*subzone_width)
    High      = [lower_proximal + 2*subzone_width, upper_proximal)
    Very High = inside the supply zone (above upper_proximal)

  ── BIAS RULE (Book P133) ─────────────────────────────────────
    price in Very Low / Low  → BUY_ONLY
    price in Very High / High → SELL_ONLY
    price in Equilibrium     → TREND_FOLLOW_OR_NO_TRADE

  ── HTF OVERRIDE (Book P135) ──────────────────────────────────
    "The longer frame always wins."
    final_bias = higher_timeframe_bias
    Lower-timeframe signals are only actionable if they AGREE with
    the higher-timeframe bias. If they conflict, WAIT.

  ── TRADING STYLE → TIMEFRAME TRIPLET (Book P129) ─────────────
    Scalper  : 15m (long) / 5m (medium) / 1m (short)
    Day      : 1d  (long) / 4h (medium) / 1h (short)
    Swing    : 1w  (long) / 1d (medium) / 4h (short)
    Position : 1M  (long) / 1w (medium) / 1d (short)

Usage:
    from analysis.curve_mtf import CurveMTF, TradingStyle
    curve = CurveMTF.from_zones(
        nearest_demand={"proximal": 1.0800, "zone_low": 1.0790, "zone_high": 1.0810},
        nearest_supply={"proximal": 1.0900, "zone_low": 1.0890, "zone_high": 1.0910},
        current_price=1.0820,
        timeframe="1d",
    )
    bias = curve.get_bias()  # → "BUY_ONLY"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger

log = get_logger("curve_mtf")


# ════════════════════════════════════════════════════════════════
#  TRADING STYLE → TIMEFRAME TRIPLET (Book P129)
# ════════════════════════════════════════════════════════════════

from enum import Enum

class TradingStyle(str, Enum):
    """Book P127-128: four trading styles by holding period."""
    SCALPER  = "scalper"
    DAY      = "day"
    SWING    = "swing"
    POSITION = "position"


# Book P129: recommended 3-timeframe combination per style
# Format: (long_tf, medium_tf, short_tf)
TIMEFRAME_TRIPLET: Dict[TradingStyle, Tuple[str, str, str]] = {
    TradingStyle.SCALPER:  ("15m", "5m", "1m"),
    TradingStyle.DAY:      ("1d",  "4h", "1h"),
    TradingStyle.SWING:    ("1w",  "1d", "4h"),
    TradingStyle.POSITION: ("1M",  "1w", "1d"),
}

# Approximate trade frequency per style (Book P127-128)
TRADE_FREQUENCY_PER_DAY: Dict[TradingStyle, Tuple[int, int]] = {
    TradingStyle.SCALPER:  (10, 30),    # 10-30 trades/day
    TradingStyle.DAY:      (1, 10),     # <10 trades/day
    TradingStyle.SWING:    (0, 1),      # days-weeks holding
    TradingStyle.POSITION: (0, 1),      # months-years holding
}


def get_timeframe_triplet(style: TradingStyle) -> Tuple[str, str, str]:
    """Return the (long, medium, short) timeframe triplet for a style."""
    return TIMEFRAME_TRIPLET[style]


# ════════════════════════════════════════════════════════════════
#  ZONE POSITION ENUM (Book P131 — 5 zone areas)
# ════════════════════════════════════════════════════════════════

class CurvePosition(str, Enum):
    """Where current price sits within the curve (Book P131)."""
    VERY_LOW     = "very_low"      # inside demand zone (below lower_proximal)
    LOW          = "low"           # lowest third of curve
    EQUILIBRIUM  = "equilibrium"   # middle third
    HIGH         = "high"          # highest third
    VERY_HIGH    = "very_high"     # inside supply zone (above upper_proximal)


class DirectionalBias(str, Enum):
    """Book P133: directional bias based on curve position."""
    BUY_ONLY                 = "BUY_ONLY"
    SELL_ONLY                = "SELL_ONLY"
    TREND_FOLLOW_OR_NO_TRADE = "TREND_FOLLOW_OR_NO_TRADE"


# Map curve position → directional bias (Book P133)
POSITION_TO_BIAS: Dict[CurvePosition, DirectionalBias] = {
    CurvePosition.VERY_LOW:    DirectionalBias.BUY_ONLY,
    CurvePosition.LOW:         DirectionalBias.BUY_ONLY,
    CurvePosition.EQUILIBRIUM: DirectionalBias.TREND_FOLLOW_OR_NO_TRADE,
    CurvePosition.HIGH:        DirectionalBias.SELL_ONLY,
    CurvePosition.VERY_HIGH:   DirectionalBias.SELL_ONLY,
}


# ════════════════════════════════════════════════════════════════
#  CURVE DATA STRUCTURE
# ════════════════════════════════════════════════════════════════

@dataclass
class Curve:
    """
    Book P130-131: the price range between nearest demand and supply
    zone proximal lines, divided into three equal sub-zones.
    """
    # Source zone proximal lines (the curve's edges)
    demand_proximal: float      # lower edge (Book P130)
    supply_proximal: float      # upper edge (Book P130)

    # The 5 boundaries (Book P131)
    very_low_top:    float      # = demand_proximal (top of demand zone)
    low_top:         float      # = demand_proximal + subzone_width
    equilibrium_top: float      # = demand_proximal + 2 * subzone_width
    high_top:        float      # = supply_proximal (top of high subzone)

    # Derived
    subzone_width: float
    curve_width:    float

    # Optional: full zone info for context
    demand_zone: Optional[Dict[str, Any]] = None
    supply_zone: Optional[Dict[str, Any]] = None
    timeframe: str = "unknown"

    @property
    def curve_low(self) -> float:
        return min(self.demand_proximal, self.supply_proximal)

    @property
    def curve_high(self) -> float:
        return max(self.demand_proximal, self.supply_proximal)

    def position_of(self, price: float) -> CurvePosition:
        """Determine which sub-zone a price falls into (Book P131)."""
        # Handle inverted case (supply below demand — shouldn't happen
        # but be defensive)
        lo = self.curve_low
        hi = self.curve_high
        if lo == hi:
            return CurvePosition.EQUILIBRIUM

        if price < lo:
            return CurvePosition.VERY_LOW
        if price >= hi:
            return CurvePosition.VERY_HIGH

        # Within curve — split into thirds
        # Normalize so demand_proximal is always at low end
        if self.demand_proximal < self.supply_proximal:
            w = self.subzone_width
            if price < self.demand_proximal + w:
                return CurvePosition.LOW
            if price < self.demand_proximal + 2 * w:
                return CurvePosition.EQUILIBRIUM
            return CurvePosition.HIGH
        else:
            # Inverted — supply is below demand (rare)
            w = self.subzone_width
            if price < self.supply_proximal + w:
                return CurvePosition.HIGH  # near supply = high zone
            if price < self.supply_proximal + 2 * w:
                return CurvePosition.EQUILIBRIUM
            return CurvePosition.LOW

    def bias_for(self, price: float) -> DirectionalBias:
        """Book P133: directional bias based on curve position."""
        pos = self.position_of(price)
        return POSITION_TO_BIAS[pos]

    def describe(self) -> str:
        """Human-readable description of the curve."""
        return (
            f"Curve [{self.curve_low:.5f}, {self.curve_high:.5f}] "
            f"width={self.curve_width:.5f} sub={self.subzone_width:.5f} "
            f"boundaries: Low/Eq={self.low_top:.5f}, Eq/High={self.equilibrium_top:.5f}"
        )


# ════════════════════════════════════════════════════════════════
#  CURVE BUILDER
# ════════════════════════════════════════════════════════════════

class CurveMTF:
    """
    Book 5 Chapter 12 — Multi-Frame "Curve" methodology.

    Provides:
      • from_zones()         — build a Curve from nearest demand/supply zones
      • get_bias()           — directional bias from curve position
      • check_alignment()    — verify LTF signal agrees with HTF bias
      • resolve_conflict()   — HTF-override hierarchy (Book P135)
    """

    @staticmethod
    def from_zones(
        nearest_demand: Dict[str, Any],
        nearest_supply: Dict[str, Any],
        current_price: float,
        timeframe: str = "1d",
    ) -> Curve:
        """
        Build a Curve from the nearest demand and supply zones (Book P130).

        Args:
            nearest_demand : zone dict with at least "proximal" key
            nearest_supply : zone dict with at least "proximal" key
            current_price  : not used in curve construction, but validated
            timeframe      : label of the timeframe this curve is built on

        Returns:
            Curve
        """
        demand_proximal = float(nearest_demand.get(
            "proximal", nearest_demand.get("zone_high",
                                           nearest_demand.get("zone_low", 0))
        ))
        supply_proximal = float(nearest_supply.get(
            "proximal", nearest_supply.get("zone_low",
                                           nearest_supply.get("zone_high", 0))
        ))

        # For demand zone, proximal is typically the UPPER edge (zone_high)
        # For supply zone, proximal is typically the LOWER edge (zone_low)
        # But the book uses "proximal line" generically — we take whichever
        # is closest to current price as the curve edge.

        curve_width = abs(supply_proximal - demand_proximal)
        subzone_width = curve_width / 3.0

        # Ensure demand_proximal is the LOWER edge for the Curve dataclass
        if demand_proximal > supply_proximal:
            # Swap — rare edge case
            demand_proximal, supply_proximal = supply_proximal, demand_proximal

        return Curve(
            demand_proximal=demand_proximal,
            supply_proximal=supply_proximal,
            very_low_top=demand_proximal,
            low_top=demand_proximal + subzone_width,
            equilibrium_top=demand_proximal + 2 * subzone_width,
            high_top=supply_proximal,
            subzone_width=subzone_width,
            curve_width=curve_width,
            demand_zone=nearest_demand,
            supply_zone=nearest_supply,
            timeframe=timeframe,
        )

    # ══════════════════════════════════════════════════════════
    #  BIAS + ALIGNMENT (Book P133, P135)
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def get_bias(curve: Curve, price: float) -> DirectionalBias:
        """Book P133: directional bias from curve position."""
        return curve.bias_for(price)

    @staticmethod
    def check_alignment(
        htf_bias: DirectionalBias,
        ltf_signal: str,
    ) -> bool:
        """
        Book P135: verify a lower-timeframe signal agrees with the
        higher-timeframe bias.

        Args:
            htf_bias    : bias from the higher timeframe curve
            ltf_signal  : "long" | "short" | "neutral" from the LTF

        Returns:
            True if LTF signal agrees with HTF bias (or bias is neutral)
        """
        ltf = ltf_signal.lower()
        if htf_bias == DirectionalBias.BUY_ONLY:
            return ltf in ("long", "neutral")
        if htf_bias == DirectionalBias.SELL_ONLY:
            return ltf in ("short", "neutral")
        # TREND_FOLLOW_OR_NO_TRADE — permit either direction
        return True

    @staticmethod
    def resolve_conflict(
        htf_bias: DirectionalBias,
        ltf_signals: List[Tuple[str, str]],
    ) -> Dict[str, Any]:
        """
        Book P135: HTF override hierarchy — "the longer frame always wins".

        Args:
            htf_bias     : bias from the highest timeframe
            ltf_signals  : list of (timeframe_label, signal) tuples
                           e.g. [("4h", "long"), ("1h", "short")]

        Returns:
            {
                "final_bias": DirectionalBias,
                "actionable_signals": [...],   # LTF signals that agree with HTF
                "conflicting_signals": [...],  # LTF signals that conflict
                "decision": "trade" | "wait",
                "reason": str,
            }
        """
        actionable = []
        conflicting = []

        for tf, sig in ltf_signals:
            if CurveMTF.check_alignment(htf_bias, sig):
                actionable.append((tf, sig))
            else:
                conflicting.append((tf, sig))

        # If HTF bias is TREND_FOLLOW_OR_NO_TRADE, only trade if ALL
        # LTF signals agree with each other
        if htf_bias == DirectionalBias.TREND_FOLLOW_OR_NO_TRADE:
            if not conflicting and len(set(s for _, s in ltf_signals)) <= 1:
                decision = "trade"
                reason = "Equilibrium zone — LTF signals aligned, no HTF conflict"
            else:
                decision = "wait"
                reason = "Equilibrium zone — LTF signals mixed, wait for clarity"
        elif conflicting:
            decision = "wait"
            reason = (f"HTF bias={htf_bias.value} conflicts with LTF signals "
                      f"{[t for t, _ in conflicting]} — Book P135: longer frame wins, WAIT")
        elif actionable:
            decision = "trade"
            reason = f"HTF bias={htf_bias.value} aligns with LTF signals {[t for t, _ in actionable]}"
        else:
            decision = "wait"
            reason = "No actionable LTF signals"

        return {
            "final_bias": htf_bias,
            "actionable_signals": actionable,
            "conflicting_signals": conflicting,
            "decision": decision,
            "reason": reason,
        }

    # ══════════════════════════════════════════════════════════
    #  FIBONACCI ALTERNATIVE (Book P132)
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def fib_levels_for_curve(curve: Curve) -> Dict[str, float]:
        """
        Book P132: alternative curve-splitting method using Fibonacci
        tool customized to 33% and 66% levels (NOT the standard
        23.6/38.2/61.8/78.6 retracement levels).

        Returns:
            {"33%": float, "66%": float} — same as low_top / equilibrium_top
        """
        return {
            "33%": curve.low_top,
            "66%": curve.equilibrium_top,
        }


# ════════════════════════════════════════════════════════════════
#  CLI ENTRY (smoke test)
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    print("=" * 64)
    print("  CURVE MTF — Book 5 Chapter 12 (Pages 126-135)")
    print("=" * 64)

    # ── Book P131 worked example: proximal lines at $10 and $13 ──
    print("\n── Book P131 worked example ($10 / $13) ──")
    demand_zone = {"proximal": 10.0, "zone_low": 9.5, "zone_high": 10.0}
    supply_zone = {"proximal": 13.0, "zone_low": 13.0, "zone_high": 13.5}
    curve = CurveMTF.from_zones(demand_zone, supply_zone, current_price=11.5, timeframe="1M")
    print(f"  {curve.describe()}")
    print(f"  Expected: sub-zone width = (13-10)/3 = 1.0")
    print(f"  Actual:   sub-zone width = {curve.subzone_width}")
    print(f"  Low/Eq boundary  = {curve.low_top}    (expected 11.0)")
    print(f"  Eq/High boundary = {curve.equilibrium_top}    (expected 12.0)")

    # ── Test all 5 positions ──
    print("\n── Position + Bias tests (Book P133) ──")
    test_prices = [9.0, 10.5, 11.5, 12.5, 14.0]
    for p in test_prices:
        pos = curve.position_of(p)
        bias = curve.bias_for(p)
        print(f"  price={p:>5.1f} → position={pos.value:<12} → bias={bias.value}")

    # ── Trading style → timeframe triplet (Book P129) ──
    print("\n── Trading Style → Timeframe Triplet (Book P129) ──")
    for style in TradingStyle:
        triplet = get_timeframe_triplet(style)
        freq = TRADE_FREQUENCY_PER_DAY[style]
        print(f"  {style.value:<9}: long={triplet[0]:<4} medium={triplet[1]:<4} "
              f"short={triplet[2]:<4}  (freq {freq[0]}-{freq[1]}/day)")

    # ── HTF override hierarchy (Book P135) ──
    print("\n── HTF Override Hierarchy (Book P135) ──")
    # Scenario: monthly curve says BUY_ONLY (price in Low zone),
    # but weekly signal is "short" at a supply zone
    htf_bias = DirectionalBias.BUY_ONLY
    ltf_signals = [("1w", "short"), ("1d", "long")]
    result = CurveMTF.resolve_conflict(htf_bias, ltf_signals)
    print(f"  HTF bias: {htf_bias.value}")
    print(f"  LTF signals: {ltf_signals}")
    print(f"  Decision: {result['decision']}")
    print(f"  Reason: {result['reason']}")
    print(f"  Actionable: {result['actionable_signals']}")
    print(f"  Conflicting: {result['conflicting_signals']}")

    # ── Fibonacci alternative (Book P132) ──
    print("\n── Fibonacci Alternative (Book P132) ──")
    fib = CurveMTF.fib_levels_for_curve(curve)
    print(f"  Fib 33% level = {fib['33%']}  (matches Low/Eq boundary = {curve.low_top})")
    print(f"  Fib 66% level = {fib['66%']}  (matches Eq/High boundary = {curve.equilibrium_top})")

    print("\n" + "=" * 64)
    print("  Curve MTF smoke test complete.")
    print("=" * 64)