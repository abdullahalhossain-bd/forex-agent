"""
analysis/adx_filters.py — ADX day-over-day rising filter + "Bishop" exit signal.

Both rules are point-in-time comparisons between the current bar's ADX and
the prior bar's ADX. Neither requires any new indicator — they consume the
`adx` column already computed by features/indicators_v5.py.

CONTRACT:
- Both functions are pure/stateless: same inputs -> same output, no hidden
  state. Callers own any "have I already acted on this" bookkeeping.
- `adx_prev` MUST be the immediately preceding CLOSED bar's ADX. Passing a
  non-adjacent bar (e.g. skipping a gap in the series) silently produces a
  meaningless comparison — this module cannot detect that from two floats
  alone, so the caller is responsible for passing truly adjacent bars.

NOTE ON SCOPE: these are distinct from (and complementary to) an absolute
ADX-level filter such as `adx >= 18`. An absolute-level filter answers
"is there currently a trend strong enough to trade." The rising-filter here
answers a different question: "is trend strength currently increasing."
The Bishop is an exit-only rule and does not gate entries at all.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

logger = logging.getLogger("analysis.adx_filters")

# The Bishop only arms once ADX has printed above this level.
BISHOP_ARM_LEVEL = 40.0


def _valid(*values: float) -> bool:
    """True if every value is a real, non-NaN float."""
    for v in values:
        if v is None:
            return False
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return False
        if math.isnan(fv):
            return False
    return True


def adx_rising(adx_curr: Optional[float], adx_prev: Optional[float]) -> bool:
    """
    True only if ADX strictly increased from the prior bar to this bar.

    A flat ADX (adx_curr == adx_prev) is NOT rising -- it must not be
    treated as a pass, since the book's filter is specifically about
    trend strength that is actively building, not merely non-declining.

    Returns False (never raises) on missing/NaN input or on the first
    bar of a series where no prior value exists yet -- an unknown
    direction is treated as "don't take the trade," which is the safer
    default for an entry filter.
    """
    if not _valid(adx_curr, adx_prev):
        logger.debug("adx_rising: invalid input (curr=%r, prev=%r)", adx_curr, adx_prev)
        return False
    return float(adx_curr) > float(adx_prev)


def bishop_exit(
    adx_curr: Optional[float],
    adx_prev: Optional[float],
    arm_level: float = BISHOP_ARM_LEVEL,
) -> bool:
    """
    "The Bishop" -- exit-only signal.

    Fires True exactly on the bar where ADX, having been above `arm_level`
    on the PRIOR bar, ticks down (by any amount, however small) on the
    current bar. It intentionally does NOT fire on every day of a
    subsequent decline -- only on the specific down-tick bar right after
    being above the arm level. It is silent whenever the prior bar's ADX
    was at or below `arm_level`, regardless of direction.

    This function is a single-bar test, not a stateful "have we already
    exited" tracker -- if ADX oscillates (e.g. 45 -> 43 -> 44 -> 41), this
    will correctly fire on the 45->43 bar AND again on the 44->41 bar,
    since each is genuinely a fresh down-tick immediately following a
    >arm_level reading. It is the CALLER's responsibility to only act on
    this while a trend-following position is actually open, and to not
    re-open a position purely because this stops firing.

    Args:
        adx_curr: ADX on the current (just-closed) bar.
        adx_prev: ADX on the immediately preceding closed bar.
        arm_level: the level ADX must have been above on the prior bar
            for this rule to be live. Defaults to 40 per the book.

    Returns:
        True if this bar is a Bishop exit bar, else False.
    """
    if not _valid(adx_curr, adx_prev):
        logger.debug("bishop_exit: invalid input (curr=%r, prev=%r)", adx_curr, adx_prev)
        return False

    adx_curr_f = float(adx_curr)
    adx_prev_f = float(adx_prev)

    if adx_prev_f <= arm_level:
        return False  # rule is dormant below/at the arm level

    return adx_curr_f < adx_prev_f
