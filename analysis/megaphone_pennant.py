"""
analysis/megaphone_pennant.py — Megaphone / Pennant neutral-formation classifier.

Both formations are defined purely from the two most recent CONFIRMED swing
highs and the two most recent CONFIRMED swing lows:

  MEGAPHONE (expanding range, no trade possible):
      latest swing high > prior swing high   (higher highs)
      AND latest swing low  < prior swing low   (lower lows)

  PENNANT (contracting range, breakout imminent, bracket both sides):
      latest swing high < prior swing high   (lower highs)
      AND latest swing low  > prior swing low   (higher lows)

Anything else (e.g. both highs and lows moving the same direction, which is
just an ordinary trend) is classified UNKNOWN by this module -- that's not
a bug, it's outside the scope of what a megaphone/pennant classifier claims
to detect. A directional trend should be evaluated by trend-following logic
elsewhere, not by this module.

CONTRACT:
- Swing highs/lows passed in MUST already be confirmed (i.e. not a swing
  point that could still be revised by future bars). This module has no way
  to verify that upstream -- if the caller passes an unconfirmed/provisional
  swing point, the classification can repaint. This mirrors the same
  look-ahead-bias contract documented in strategies/breakout.py.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Sequence

logger = logging.getLogger("analysis.megaphone_pennant")


class Formation:
    MEGAPHONE = "MEGAPHONE"
    PENNANT = "PENNANT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class FormationResult:
    formation: str
    reason: str
    # Only populated when formation == PENNANT -- the bracket levels the
    # book says to place: buy-stop above the latest swing high, sell-stop
    # below the latest swing low.
    buy_stop: Optional[float] = None
    sell_stop: Optional[float] = None


def _valid_pair(a: float, b: float) -> bool:
    for v in (a, b):
        if v is None:
            return False
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return False
        if math.isnan(fv):
            return False
    return True


def classify(
    swing_highs: Sequence[float],
    swing_lows: Sequence[float],
) -> FormationResult:
    """
    Classify the current market as MEGAPHONE, PENNANT, or UNKNOWN, using the
    two most recent confirmed swing highs and two most recent confirmed
    swing lows.

    Args:
        swing_highs: confirmed swing-high prices, oldest-to-newest. Only the
            last two entries are used.
        swing_lows: confirmed swing-low prices, oldest-to-newest. Only the
            last two entries are used.

    Returns:
        FormationResult. UNKNOWN (with a reason) if there aren't at least
        two confirmed swings on each side yet, or if the pattern is neither
        a megaphone nor a pennant (e.g. an ordinary trend, or an exact tie
        on either side).
    """
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return FormationResult(
            Formation.UNKNOWN,
            f"insufficient confirmed swings (highs={len(swing_highs)}, lows={len(swing_lows)}, need >=2 each)",
        )

    high_prev, high_last = swing_highs[-2], swing_highs[-1]
    low_prev, low_last = swing_lows[-2], swing_lows[-1]

    if not _valid_pair(high_prev, high_last) or not _valid_pair(low_prev, low_last):
        logger.debug("classify: NaN/missing swing value(s)")
        return FormationResult(Formation.UNKNOWN, "NaN or missing swing value")

    high_prev, high_last = float(high_prev), float(high_last)
    low_prev, low_last = float(low_prev), float(low_last)

    higher_high = high_last > high_prev
    lower_high = high_last < high_prev
    lower_low = low_last < low_prev
    higher_low = low_last > low_prev

    if higher_high and lower_low:
        return FormationResult(
            Formation.MEGAPHONE,
            "higher high + lower low: expanding, directionless range -- no trade",
        )

    if lower_high and higher_low:
        return FormationResult(
            Formation.PENNANT,
            "lower high + higher low: contracting range -- bracket both sides",
            buy_stop=high_last,
            sell_stop=low_last,
        )

    return FormationResult(
        Formation.UNKNOWN,
        "neither megaphone nor pennant (likely an ordinary trend, or an exact tie)",
    )
