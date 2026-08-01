# analysis/price_target_module.py — Western measured-move price targets
# =============================================================================
# Core principle (per the Nison rule spec): candlestick patterns never give
# a price target by themselves. Any target must come from a separate Western
# measured-move technique, and is meant for EXITING/TRIMMING an existing
# position — never for initiating a new counter-signal trade.
#
# This module is deliberately independent of every candlestick pattern module
# (patterns.py, candlestick_patterns_mw.py/_br.py, high_reliability_patterns.py,
# long_term_patterns.py). Those modules should call INTO this one for a target
# once they have an entry signal — they should not compute targets themselves.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PriceTarget:
    method: str
    direction: str          # "bullish" or "bearish"
    target: float
    invalidation_level: Optional[float] = None
    note: str = ""


def box_breakout_target(box_top: float, box_bottom: float, direction: str) -> PriceTarget:
    """target = box_top + (box_top - box_bottom) for bullish breakout,
    box_bottom - (box_top - box_bottom) for bearish breakdown.
    Invalidation: close back inside the box voids the target."""
    height = box_top - box_bottom
    if direction == "bullish":
        target = box_top + height
        invalidation = box_bottom
    elif direction == "bearish":
        target = box_bottom - height
        invalidation = box_top
    else:
        raise ValueError("direction must be 'bullish' or 'bearish'")
    return PriceTarget(
        method="box_breakout", direction=direction, target=target,
        invalidation_level=invalidation,
        note="Voided if price closes back inside the box.",
    )


def swing_target(leg_a: float, leg_b: float, correction_low_c: float) -> PriceTarget:
    """target = correction_low_C + (leg_B - leg_A).
    Measures the initial impulse A->B and applies the same height from the
    pullback low C. Direction is inferred from the sign of (leg_b - leg_a)."""
    height = leg_b - leg_a
    direction = "bullish" if height > 0 else "bearish"
    target = correction_low_c + height
    return PriceTarget(
        method="swing_target", direction=direction, target=target,
        note="Impulse A->B height projected from correction low C.",
    )


def flag_pennant_target(flagpole_a: float, flagpole_b: float,
                         flag_bottom: Optional[float] = None,
                         flag_top: Optional[float] = None) -> PriceTarget:
    """Bullish: target = flag_bottom + flagpole_height (conservative — uses
    the flag's BOTTOM as the base, not its top).
    Bearish: target = flag_top - flagpole_height.
    flagpole_height = |B - A| of the initial sharp move."""
    flagpole_height = abs(flagpole_b - flagpole_a)
    if flagpole_b > flagpole_a:
        if flag_bottom is None:
            raise ValueError("flag_bottom required for a bullish flag/pennant target")
        return PriceTarget(
            method="flag_pennant", direction="bullish",
            target=flag_bottom + flagpole_height,
            note="Conservative target: flag BOTTOM + flagpole height.",
        )
    else:
        if flag_top is None:
            raise ValueError("flag_top required for a bearish flag/pennant target")
        return PriceTarget(
            method="flag_pennant", direction="bearish",
            target=flag_top - flagpole_height,
            note="Conservative target: flag TOP - flagpole height.",
        )


def triangle_target(breakout_level: float, widest_point: float,
                     horizontal_line: float, direction: str) -> PriceTarget:
    """target = breakout_level +/- (widest_point_of_triangle - horizontal_line)
    for ascending/descending triangles."""
    height = abs(widest_point - horizontal_line)
    if direction == "bullish":
        target = breakout_level + height
    elif direction == "bearish":
        target = breakout_level - height
    else:
        raise ValueError("direction must be 'bullish' or 'bearish'")
    return PriceTarget(
        method="ascending_descending_triangle", direction=direction, target=target,
        note="Widest point of triangle minus the horizontal line, projected from breakout.",
    )


def spring_upthrust_target(range_top: float, range_bottom: float, direction: str) -> PriceTarget:
    """Per the spec, spring/upthrust targets reuse the trading-range height,
    projected from the opposite boundary of the range (same measured-move
    logic as box_breakout, applied to a Wyckoff spring/upthrust context)."""
    return box_breakout_target(range_top, range_bottom, direction)


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t1 = box_breakout_target(box_top=1.1050, box_bottom=1.0950, direction="bullish")
    print(t1)
    assert round(t1.target, 4) == 1.1150

    t2 = swing_target(leg_a=1.0800, leg_b=1.0950, correction_low_c=1.0880)
    print(t2)
    assert round(t2.target, 4) == 1.1030

    t3 = flag_pennant_target(flagpole_a=1.0800, flagpole_b=1.0950, flag_bottom=1.0900)
    print(t3)
    assert round(t3.target, 4) == 1.1050

    t4 = triangle_target(breakout_level=1.1000, widest_point=1.1100,
                          horizontal_line=1.0950, direction="bullish")
    print(t4)
    assert round(t4.target, 4) == 1.1150

    print("\nPrice target module smoke test passed.")