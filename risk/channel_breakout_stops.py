"""
risk/channel_breakout_stops.py — Channel-breakout stop management.

Implements three related enhancements to a plain N-bar channel breakout,
kept separate from strategies/breakout.py (which stays a pure, stateless
signal generator) because these require per-trade state (bars-in-trade,
the entry-time breakout level) that a signal generator should not own.

  1. Five-day condition: the rejection rule only applies when the breakout
     level itself has been flat or declining (longs) / flat or rising
     (shorts) for the preceding N bars -- i.e. the breakout is coming out
     of a genuinely sideways market, not an already-running trend.

  2. Rejection rule: when the five-day condition holds, exit the trade if
     price closes back through the ORIGINAL breakout level on the entry
     bar or the very next bar. This produces a much tighter loss than
     waiting for the full channel-width stop, at the cost of a higher
     stop-out frequency on trades that would have worked out.

  3. Last-bar stop: once past the first two bars (or once the rejection
     rule is inactive because the five-day condition didn't hold), the
     stop sits at the low (longs) / high (shorts) of the breakout bar --
     or the bar before it, if the breakout bar barely cleared the level
     or gapped straight through it.

  4. Stop-sequencing priority: for the life of the trade, the ACTIVE stop
     is chosen in this order: rejection rule (bars 1-2, if armed) -> last
     bar stop -> the ordinary N-bar channel stop, once the channel stop
     becomes tighter (closer to price) than the last-bar stop. Once the
     sequence has moved to the channel stop, it does not move back.

CONTRACT:
- `breakout_level` passed into `evaluate_rejection` and `select_active_stop`
  MUST be the level captured at trade entry, not a level recomputed on a
  later bar -- reusing a recomputed level breaks the rejection rule's
  intent (see docstring on RejectionRuleState below).
- All functions here are pure given their inputs; the CALLER owns any
  state that needs to persist across bars/restarts (bars_in_trade, the
  entry-time breakout level, whether the channel-stop handoff has already
  happened). This mirrors the stateless-strategy contract used elsewhere
  in this repo (see strategies/breakout.py).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Sequence

logger = logging.getLogger("risk.channel_breakout_stops")

DIRECTIONS = ("BUY", "SELL")


def _validate_direction(direction: str) -> None:
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")


def _is_valid(*values: float) -> bool:
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


def five_day_condition(
    level_history: Sequence[float],
    direction: str,
    lookback: int = 5,
) -> bool:
    """
    True if the breakout level itself has been flat-or-declining (BUY) /
    flat-or-rising (SELL) for the `lookback` bars immediately preceding the
    breakout bar. This is the gate that arms the rejection rule -- it is
    deliberately false during a market that is already trending hard, since
    the book's own intent is to only apply the tight rejection stop when
    the breakout is coming out of sideways conditions.

    Args:
        level_history: the rolling breakout-level series (e.g. the 55-bar
            high) for the `lookback` bars BEFORE the breakout bar itself
            (do not include the breakout bar's own level).
        direction: "BUY" or "SELL".
        lookback: number of bars the level must have been non-advancing.

    Returns:
        False if there isn't enough history to evaluate, or if any value
        is missing/NaN -- an unverifiable condition is treated as "not
        armed," which is the safer default (falls back to the wider
        channel stop rather than assuming the tighter rule applies).
    """
    _validate_direction(direction)

    if len(level_history) < lookback:
        return False

    window = list(level_history[-lookback:])
    if not all(_is_valid(v) for v in window):
        return False

    window = [float(v) for v in window]

    if direction == "BUY":
        # level must be non-increasing across the window (flat or declining)
        return all(window[i] >= window[i + 1] for i in range(len(window) - 1))
    else:
        # level must be non-decreasing across the window (flat or rising)
        return all(window[i] <= window[i + 1] for i in range(len(window) - 1))


@dataclass(frozen=True)
class RejectionCheck:
    triggered: bool
    reason: str


def evaluate_rejection(
    direction: str,
    breakout_level: float,
    bar_close: float,
    bars_in_trade: int,
    five_day_condition_met: bool,
) -> RejectionCheck:
    """
    Evaluate the rejection rule for a SINGLE bar of an open trade.

    Only applies on bars_in_trade in {1, 2} (the entry bar and the one
    immediately after) AND only when `five_day_condition_met` is True.
    Outside that window, or when the five-day condition never held, this
    always returns not-triggered -- the caller should fall through to the
    last-bar / channel stop logic instead.

    Args:
        direction: "BUY" or "SELL".
        breakout_level: the ORIGINAL level captured at entry (not
            recomputed on a later bar -- see module contract).
        bar_close: close price of the CURRENT bar being evaluated.
        bars_in_trade: 1 on the entry bar, 2 on the bar after, etc.
        five_day_condition_met: result of `five_day_condition()` evaluated
            once at entry and held fixed for the life of this check.

    Returns:
        RejectionCheck(triggered=True, ...) if the trade should be
        liquidated on this bar's close.
    """
    _validate_direction(direction)

    if not five_day_condition_met:
        return RejectionCheck(False, "five-day condition not met -- rejection rule inactive")

    if bars_in_trade not in (1, 2):
        return RejectionCheck(False, f"bars_in_trade={bars_in_trade} outside rejection window (1-2)")

    if not _is_valid(breakout_level, bar_close):
        return RejectionCheck(False, "invalid breakout_level or bar_close")

    breakout_level_f = float(breakout_level)
    bar_close_f = float(bar_close)

    if direction == "BUY":
        rejected = bar_close_f < breakout_level_f
    else:
        rejected = bar_close_f > breakout_level_f

    if rejected:
        return RejectionCheck(
            True,
            f"close {bar_close_f} back through original breakout level {breakout_level_f} "
            f"on bar {bars_in_trade} of the trade",
        )
    return RejectionCheck(False, "no rejection on this bar")


def last_bar_stop(
    direction: str,
    breakout_bar_high: float,
    breakout_bar_low: float,
    breakout_bar_open: float,
    breakout_level: float,
    prior_bar_high: float,
    prior_bar_low: float,
    barely_clear_atr_fraction: float,
    atr: float,
) -> float:
    """
    Compute the "last bar" stop: the low (BUY) / high (SELL) of the
    breakout bar itself, UNLESS the breakout bar barely cleared the level
    or gapped straight through it -- in which case use the prior bar's
    low/high instead, per the book.

    "Barely cleared" is defined here as: the breakout bar's open was
    already beyond the breakout level (a gap-through), OR the breakout
    bar's close is within `barely_clear_atr_fraction * atr` of the level.
    This is my own precise operational reading of the book's qualitative
    "barely below" / "gaps above" description -- treat it as an engineering
    judgment call, not a verified restatement of the author's exact test.

    Args:
        direction: "BUY" or "SELL".
        breakout_bar_high/low/open: OHLC of the bar that broke the level.
        breakout_level: the level that was broken.
        prior_bar_high/low: OHLC of the bar immediately before the breakout.
        barely_clear_atr_fraction: threshold (in ATRs) below which the
            breakout bar is considered to have "barely" cleared the level.
        atr: current ATR, used to scale the "barely" threshold.

    Returns:
        The stop price to use.

    Raises:
        ValueError: if `atr` is not a positive, finite number -- silently
            falling back to a 0 or negative ATR would produce a
            "barely clear" threshold of zero, which defeats the purpose
            of this check rather than failing loudly.
    """
    _validate_direction(direction)
    if not _is_valid(atr) or float(atr) <= 0:
        raise ValueError(f"atr must be a positive, finite number, got {atr!r}")

    atr_f = float(atr)
    threshold = barely_clear_atr_fraction * atr_f

    if direction == "BUY":
        gapped_through = breakout_bar_open > breakout_level
        barely_cleared = abs(breakout_bar_high - breakout_level) <= threshold or \
            (breakout_bar_high - breakout_level) <= threshold
        use_prior_bar = gapped_through or barely_cleared
        return float(prior_bar_low) if use_prior_bar else float(breakout_bar_low)
    else:
        gapped_through = breakout_bar_open < breakout_level
        barely_cleared = abs(breakout_level - breakout_bar_low) <= threshold
        use_prior_bar = gapped_through or barely_cleared
        return float(prior_bar_high) if use_prior_bar else float(breakout_bar_high)


def select_active_stop(
    direction: str,
    bars_in_trade: int,
    five_day_condition_met: bool,
    breakout_level: float,
    current_close: float,
    last_bar_stop_price: float,
    channel_stop_price: float,
) -> dict:
    """
    Choose which stop governs the trade on this bar, following the book's
    documented priority: rejection rule (bars 1-2, if armed) -> last-bar
    stop -> channel stop, switching to the channel stop only once it has
    become tighter (closer to price) than the last-bar stop, and never
    switching back.

    Returns a dict: {"action": "EXIT"|"HOLD", "stop_price": float|None,
    "stage": "rejection"|"last_bar"|"channel", "reason": str}
    """
    _validate_direction(direction)

    rejection = evaluate_rejection(
        direction, breakout_level, current_close, bars_in_trade, five_day_condition_met
    )
    if rejection.triggered:
        return {
            "action": "EXIT",
            "stop_price": current_close,
            "stage": "rejection",
            "reason": rejection.reason,
        }

    if not _is_valid(last_bar_stop_price, channel_stop_price):
        # Can't compare stages safely -- fail toward the wider, known-valid
        # stop rather than silently picking one.
        return {
            "action": "HOLD",
            "stop_price": last_bar_stop_price if _is_valid(last_bar_stop_price) else channel_stop_price,
            "stage": "last_bar",
            "reason": "one of last_bar_stop_price/channel_stop_price invalid; defaulting conservatively",
        }

    last_bar_stop_price = float(last_bar_stop_price)
    channel_stop_price = float(channel_stop_price)

    if direction == "BUY":
        channel_is_tighter = channel_stop_price > last_bar_stop_price
    else:
        channel_is_tighter = channel_stop_price < last_bar_stop_price

    if channel_is_tighter:
        return {
            "action": "HOLD",
            "stop_price": channel_stop_price,
            "stage": "channel",
            "reason": "channel stop has tightened past the last-bar stop",
        }

    return {
        "action": "HOLD",
        "stop_price": last_bar_stop_price,
        "stage": "last_bar",
        "reason": "last-bar stop still tighter than the channel stop",
    }
