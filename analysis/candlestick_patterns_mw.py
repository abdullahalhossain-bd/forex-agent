# analysis/candlestick_patterns_mw.py — MotiveWave-style 33-pattern scanner
# =============================================================================
# Ported from: https://github.com/RauchenwaldC/motivewave-candlestick-pattern-study
# Original author: RauchenwaldC — MIT license
# Original: src/CandlestickPatterns.java (MotiveWave Java study)
#
# A unified candlestick pattern scanner that detects 40 patterns across
# three categories (33 from the original MotiveWave port + 7 added to close
# gaps against the Nison rule-engine spec — counterattack lines, separating
# lines, upside gap two crows):
#
#   1-BAR (11): Hammer, Inverted Hammer, Dragonfly Doji, Bullish Marubozu,
#               Shooting Star, Hanging Man, Gravestone Doji, Bearish Marubozu,
#               Doji, Long-Legged Doji, Spinning Top
#   2-BAR (14): Bullish/Bearish Engulfing, Bullish/Bearish Harami,
#               Piercing Line, Dark Cloud Cover, Tweezer Bottom/Top,
#               Bullish/Bearish Kicker, Bullish/Bearish Counterattack,
#               Bullish/Bearish Separating Lines
#   3-BAR (13): Morning/Evening Star, Morning/Evening Doji Star,
#               Bullish/Bearish Abandoned Baby, Three White Soldiers,
#               Three Black Crows, Three Inside Up/Down, Three Outside Up/Down,
#               Upside Gap Two Crows
#
# Plus optional trend-aware filtering using dual 50/200 MA system:
#   - Bullish reversals only show in downtrends
#   - Bearish reversals only show in uptrends
#   - Neutral patterns show in any trend
#
# This module is INDEPENDENT of analysis/patterns.py, analysis/engulfing_bar_strategy.py,
# analysis/pin_bar_strategy.py, analysis/advanced_patterns.py — those are
# book-based single-strategy modules. This is a broad scanner that can run
# alongside them; the confluence engine can fuse outputs.
#
# Output columns added to the DataFrame:
#   csp_pattern      : pattern name (string) or NaN
#   csp_category     : "bullish" / "bearish" / "neutral" or NaN
#   csp_bars         : 1, 2, or 3 (pattern length)
#   csp_signal       : +1 (bullish), -1 (bearish), 0 (neutral or none)
#   csp_trend        : "uptrend" / "downtrend" / "sideways" (when trend_filter on)
# =============================================================================

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ── Pattern categorization (used for trend filtering) ────────────────────────

BULLISH_REVERSAL_PATTERNS = {
    "Hammer", "Inverted Hammer", "Dragonfly Doji",
    "Bullish Engulfing", "Bullish Harami", "Piercing Line",
    "Tweezer Bottom", "Bullish Kicker", "Bullish Counterattack",
    "Morning Star", "Morning Doji Star", "Bullish Abandoned Baby",
    "Three White Soldiers", "Three Inside Up", "Three Outside Up",
}

BEARISH_REVERSAL_PATTERNS = {
    "Shooting Star", "Hanging Man", "Gravestone Doji",
    "Bearish Engulfing", "Bearish Harami", "Dark Cloud Cover",
    "Tweezer Top", "Bearish Kicker", "Bearish Counterattack",
    "Evening Star", "Evening Doji Star", "Bearish Abandoned Baby",
    "Three Black Crows", "Three Inside Down", "Three Outside Down",
    "Upside Gap Two Crows",
}

NEUTRAL_PATTERNS = {"Doji", "Long-Legged Doji", "Spinning Top"}

# Continuation patterns (Marubozu + Nison separating lines) can appear in any trend
# — they confirm the EXISTING trend rather than reversing it, so they are excluded
# from the bullish/bearish "reversal" sets used by the trend filter.
ANY_TREND_PATTERNS = {
    "Bullish Marubozu", "Bearish Marubozu",
    "Bullish Separating Lines", "Bearish Separating Lines",
}

PATTERN_LENGTH = {
    # 1-bar
    "Hammer": 1, "Inverted Hammer": 1, "Dragonfly Doji": 1, "Bullish Marubozu": 1,
    "Shooting Star": 1, "Hanging Man": 1, "Gravestone Doji": 1, "Bearish Marubozu": 1,
    "Doji": 1, "Long-Legged Doji": 1, "Spinning Top": 1,
    # 2-bar
    "Bullish Engulfing": 2, "Bullish Harami": 2, "Piercing Line": 2,
    "Tweezer Bottom": 2, "Bullish Kicker": 2, "Bullish Counterattack": 2,
    "Bullish Separating Lines": 2,
    "Bearish Engulfing": 2, "Bearish Harami": 2, "Dark Cloud Cover": 2,
    "Tweezer Top": 2, "Bearish Kicker": 2, "Bearish Counterattack": 2,
    "Bearish Separating Lines": 2,
    # 3-bar
    "Morning Star": 3, "Morning Doji Star": 3, "Bullish Abandoned Baby": 3,
    "Three White Soldiers": 3, "Three Inside Up": 3, "Three Outside Up": 3,
    "Evening Star": 3, "Evening Doji Star": 3, "Bearish Abandoned Baby": 3,
    "Three Black Crows": 3, "Three Inside Down": 3, "Three Outside Down": 3,
    "Upside Gap Two Crows": 3,
}


def _categorize(pattern: Optional[str]) -> Optional[str]:
    if pattern is None:
        return None
    if pattern in BULLISH_REVERSAL_PATTERNS or pattern == "Bullish Marubozu":
        return "bullish"
    if pattern in BEARISH_REVERSAL_PATTERNS or pattern == "Bearish Marubozu":
        return "bearish"
    if pattern in NEUTRAL_PATTERNS:
        return "neutral"
    return None


# ── Helper accessors (mirror Java's getBody/getUpperShadow/getLowerShadow) ────

def _is_bullish(o, c): return c > o
def _is_bearish(o, c): return c < o
def _body_size(o, c): return abs(c - o)
def _upper_shadow(o, c, h): return h - max(o, c)
def _lower_shadow(o, c, l): return min(o, c) - l
def _range(h, l): return h - l


# ── Per-pattern detectors ────────────────────────────────────────────────────
# Each returns the pattern name (str) if detected at index `i`, else None.
# Faithful to the Java source's `checkXxx(int index, DataSeries series)` methods.

def _check_doji(o, c, h, l, **_):
    body = _body_size(o, c); rng = _range(h, l)
    if rng == 0: return None
    if body / rng < 0.1: return "Doji"
    return None

def _check_long_legged_doji(o, c, h, l, **_):
    body = _body_size(o, c); rng = _range(h, l)
    if rng == 0: return None
    us = _upper_shadow(o, c, h); ls = _lower_shadow(o, c, l)
    if body / rng < 0.1 and us > body * 2 and ls > body * 2:
        return "Long-Legged Doji"
    return None

def _check_dragonfly_doji(o, c, h, l, **_):
    body = _body_size(o, c); rng = _range(h, l)
    if rng == 0: return None
    us = _upper_shadow(o, c, h); ls = _lower_shadow(o, c, l)
    if body / rng < 0.1 and ls > rng * 0.6 and us < rng * 0.1:
        return "Dragonfly Doji"
    return None

def _check_gravestone_doji(o, c, h, l, **_):
    body = _body_size(o, c); rng = _range(h, l)
    if rng == 0: return None
    us = _upper_shadow(o, c, h); ls = _lower_shadow(o, c, l)
    if body / rng < 0.1 and us > rng * 0.6 and ls < rng * 0.1:
        return "Gravestone Doji"
    return None

def _check_spinning_top(o, c, h, l, **_):
    body = _body_size(o, c); rng = _range(h, l)
    if rng == 0: return None
    us = _upper_shadow(o, c, h); ls = _lower_shadow(o, c, l)
    if 0.1 < body / rng < 0.3 and us > body and ls > body:
        return "Spinning Top"
    return None

def _check_hammer(o, c, h, l, **_):
    if not _is_bullish(o, c): return None
    body = _body_size(o, c)
    ls = _lower_shadow(o, c, l); us = _upper_shadow(o, c, h)
    if ls > body * 2 and us < body * 0.5:
        return "Hammer"
    return None

def _check_inverted_hammer(o, c, h, l, **_):
    if not _is_bullish(o, c): return None
    body = _body_size(o, c)
    us = _upper_shadow(o, c, h); ls = _lower_shadow(o, c, l)
    if us > body * 2 and ls < body * 0.5:
        return "Inverted Hammer"
    return None

def _check_shooting_star(o, c, h, l, **_):
    if not _is_bearish(o, c): return None
    body = _body_size(o, c)
    us = _upper_shadow(o, c, h); ls = _lower_shadow(o, c, l)
    if us > body * 2 and ls < body * 0.5:
        return "Shooting Star"
    return None

def _check_hanging_man(o, c, h, l, **_):
    if not _is_bearish(o, c): return None
    body = _body_size(o, c)
    ls = _lower_shadow(o, c, l); us = _upper_shadow(o, c, h)
    if ls > body * 2 and us < body * 0.5:
        return "Hanging Man"
    return None

def _check_bullish_marubozu(o, c, h, l, **_):
    if not _is_bullish(o, c): return None
    body = _body_size(o, c); rng = _range(h, l)
    if rng == 0: return None
    if body / rng > 0.95: return "Bullish Marubozu"
    return None

def _check_bearish_marubozu(o, c, h, l, **_):
    if not _is_bearish(o, c): return None
    body = _body_size(o, c); rng = _range(h, l)
    if rng == 0: return None
    if body / rng > 0.95: return "Bearish Marubozu"
    return None


# 2-bar patterns need current + previous bar

def _check_bullish_engulfing(o, c, h, l, prev_o, prev_c, prev_h, prev_l, **_):
    if not _is_bearish(prev_o, prev_c) or not _is_bullish(o, c): return None
    if o <= prev_c and c >= prev_o:
        return "Bullish Engulfing"
    return None

def _check_bearish_engulfing(o, c, h, l, prev_o, prev_c, prev_h, prev_l, **_):
    if not _is_bullish(prev_o, prev_c) or not _is_bearish(o, c): return None
    if o >= prev_c and c <= prev_o:
        return "Bearish Engulfing"
    return None

def _check_bullish_harami(o, c, h, l, prev_o, prev_c, prev_h, prev_l, **_):
    if not _is_bearish(prev_o, prev_c) or not _is_bullish(o, c): return None
    prev_body = _body_size(prev_o, prev_c); curr_body = _body_size(o, c)
    if (curr_body < prev_body * 0.5
            and o > prev_c and c < prev_o):
        return "Bullish Harami"
    return None

def _check_bearish_harami(o, c, h, l, prev_o, prev_c, prev_h, prev_l, **_):
    if not _is_bullish(prev_o, prev_c) or not _is_bearish(o, c): return None
    prev_body = _body_size(prev_o, prev_c); curr_body = _body_size(o, c)
    if (curr_body < prev_body * 0.5
            and o < prev_c and c > prev_o):
        return "Bearish Harami"
    return None

def _check_piercing_line(o, c, h, l, prev_o, prev_c, prev_h, prev_l, **_):
    if not _is_bearish(prev_o, prev_c) or not _is_bullish(o, c): return None
    prev_mid = (prev_o + prev_c) / 2.0
    if o < prev_c and c > prev_mid and c < prev_o:
        return "Piercing Line"
    return None

def _check_dark_cloud_cover(o, c, h, l, prev_o, prev_c, prev_h, prev_l, **_):
    if not _is_bullish(prev_o, prev_c) or not _is_bearish(o, c): return None
    prev_mid = (prev_o + prev_c) / 2.0
    if o > prev_c and c < prev_mid and c > prev_o:
        return "Dark Cloud Cover"
    return None

# Phase 3 fix (see AUDIT_REPORT.md "mw.py: hardcoded 'near equal' tolerance"):
# the six detectors below originally hardcoded a 0.05 (5% of average range)
# "near enough" tolerance for tweezer highs/lows, counterattack closes, and
# separating-lines opens. That single constant was baked in six separate
# times with no way to tune it per volatility regime — a high-ATR instrument
# needs a looser tolerance or every tweezer/counterattack signal disappears;
# a low-ATR/quiet instrument needs a tighter one or noise starts qualifying.
# All six now accept an optional `tolerance` kwarg (still defaulting to the
# original 0.05, so behaviour is unchanged unless a caller opts in) that
# `compute()` can thread through uniformly, optionally derived from
# `MarketContext.atr_pct` for volatility-adaptive behaviour.

def _check_tweezer_bottom(o, c, h, l, prev_o, prev_c, prev_h, prev_l, tolerance: float = 0.05, **_):
    low_diff = abs(l - prev_l)
    avg_range = (_range(h, l) + _range(prev_h, prev_l)) / 2.0
    if avg_range == 0: return None
    if low_diff / avg_range < tolerance:
        first_bull = _is_bullish(prev_o, prev_c)
        second_bull = _is_bullish(o, c)
        if first_bull != second_bull:
            return "Tweezer Bottom"
    return None

def _check_tweezer_top(o, c, h, l, prev_o, prev_c, prev_h, prev_l, tolerance: float = 0.05, **_):
    high_diff = abs(h - prev_h)
    avg_range = (_range(h, l) + _range(prev_h, prev_l)) / 2.0
    if avg_range == 0: return None
    if high_diff / avg_range < tolerance:
        first_bull = _is_bullish(prev_o, prev_c)
        second_bull = _is_bullish(o, c)
        if first_bull != second_bull:
            return "Tweezer Top"
    return None

def _check_bullish_kicker(o, c, h, l, prev_o, prev_c, prev_h, prev_l, **_):
    if not _is_bearish(prev_o, prev_c) or not _is_bullish(o, c): return None
    if o > prev_o: return "Bullish Kicker"
    return None

def _check_bearish_kicker(o, c, h, l, prev_o, prev_c, prev_h, prev_l, **_):
    if not _is_bullish(prev_o, prev_c) or not _is_bearish(o, c): return None
    if o < prev_o: return "Bearish Kicker"
    return None

def _check_bullish_counterattack(o, c, h, l, prev_o, prev_c, prev_h, prev_l, tolerance: float = 0.05, **_):
    """Nison 'counterattack lines' / meeting lines (bullish): prior bearish candle
    gaps into a bullish candle that opens lower but rallies back to close
    approximately level with the prior candle's close."""
    if not _is_bearish(prev_o, prev_c) or not _is_bullish(o, c): return None
    if o >= prev_c: return None  # must open lower (gap down) than prior close
    avg_range = (_range(h, l) + _range(prev_h, prev_l)) / 2.0
    if avg_range == 0: return None
    if abs(c - prev_c) / avg_range < tolerance:
        return "Bullish Counterattack"
    return None

def _check_bearish_counterattack(o, c, h, l, prev_o, prev_c, prev_h, prev_l, tolerance: float = 0.05, **_):
    """Nison 'counterattack lines' (bearish): prior bullish candle gaps into a
    bearish candle that opens higher but sells off back to close approximately
    level with the prior candle's close."""
    if not _is_bullish(prev_o, prev_c) or not _is_bearish(o, c): return None
    if o <= prev_c: return None  # must open higher (gap up) than prior close
    avg_range = (_range(h, l) + _range(prev_h, prev_l)) / 2.0
    if avg_range == 0: return None
    if abs(c - prev_c) / avg_range < tolerance:
        return "Bearish Counterattack"
    return None

def _check_bullish_separating_lines(o, c, h, l, prev_o, prev_c, prev_h, prev_l, tolerance: float = 0.05, **_):
    """Nison 'separating lines' (bullish, uptrend continuation): a bearish
    pullback candle is followed by a bullish candle that opens at essentially
    the same open as the pullback candle and continues higher."""
    if not _is_bearish(prev_o, prev_c) or not _is_bullish(o, c): return None
    avg_range = (_range(h, l) + _range(prev_h, prev_l)) / 2.0
    if avg_range == 0: return None
    if abs(o - prev_o) / avg_range < tolerance and c > prev_o:
        return "Bullish Separating Lines"
    return None

def _check_bearish_separating_lines(o, c, h, l, prev_o, prev_c, prev_h, prev_l, tolerance: float = 0.05, **_):
    """Nison 'separating lines' (bearish, downtrend continuation): a bullish
    bounce candle is followed by a bearish candle that opens at essentially the
    same open as the bounce candle and continues lower."""
    if not _is_bullish(prev_o, prev_c) or not _is_bearish(o, c): return None
    avg_range = (_range(h, l) + _range(prev_h, prev_l)) / 2.0
    if avg_range == 0: return None
    if abs(o - prev_o) / avg_range < tolerance and c < prev_o:
        return "Bearish Separating Lines"
    return None


# 3-bar patterns need current + 2 previous bars

def _check_morning_star(o, c, h, l, prev_o, prev_c, prev_h, prev_l,
                        prev2_o, prev2_c, prev2_h, prev2_l, **_):
    if not _is_bearish(prev2_o, prev2_c) or not _is_bullish(o, c): return None
    first_body = _body_size(prev2_o, prev2_c)
    second_body = _body_size(prev_o, prev_c)
    third_body = _body_size(o, c)
    if (second_body < first_body * 0.3
            and max(prev_o, prev_c) < prev2_c
            and third_body > first_body * 0.5):
        return "Morning Star"
    return None

def _check_evening_star(o, c, h, l, prev_o, prev_c, prev_h, prev_l,
                        prev2_o, prev2_c, prev2_h, prev2_l, **_):
    if not _is_bullish(prev2_o, prev2_c) or not _is_bearish(o, c): return None
    first_body = _body_size(prev2_o, prev2_c)
    second_body = _body_size(prev_o, prev_c)
    third_body = _body_size(o, c)
    if (second_body < first_body * 0.3
            and min(prev_o, prev_c) > prev2_c
            and third_body > first_body * 0.5):
        return "Evening Star"
    return None

def _check_morning_doji_star(o, c, h, l, prev_o, prev_c, prev_h, prev_l,
                             prev2_o, prev2_c, prev2_h, prev2_l, **_):
    if not _is_bearish(prev2_o, prev2_c) or not _is_bullish(o, c): return None
    second_body = _body_size(prev_o, prev_c)
    second_range = _range(prev_h, prev_l)
    if second_range == 0: return None
    if (second_body / second_range < 0.1
            and prev_h < prev2_c):
        return "Morning Doji Star"
    return None

def _check_evening_doji_star(o, c, h, l, prev_o, prev_c, prev_h, prev_l,
                             prev2_o, prev2_c, prev2_h, prev2_l, **_):
    if not _is_bullish(prev2_o, prev2_c) or not _is_bearish(o, c): return None
    second_body = _body_size(prev_o, prev_c)
    second_range = _range(prev_h, prev_l)
    if second_range == 0: return None
    if (second_body / second_range < 0.1
            and prev_l > prev2_c):
        return "Evening Doji Star"
    return None

def _check_three_white_soldiers(o, c, h, l, prev_o, prev_c, prev_h, prev_l,
                                prev2_o, prev2_c, prev2_h, prev2_l, **_):
    if not (_is_bullish(prev2_o, prev2_c) and _is_bullish(prev_o, prev_c)
            and _is_bullish(o, c)):
        return None
    if prev_c > prev2_c and c > prev_c:
        return "Three White Soldiers"
    return None

def _check_three_black_crows(o, c, h, l, prev_o, prev_c, prev_h, prev_l,
                             prev2_o, prev2_c, prev2_h, prev2_l, **_):
    if not (_is_bearish(prev2_o, prev2_c) and _is_bearish(prev_o, prev_c)
            and _is_bearish(o, c)):
        return None
    if prev_c < prev2_c and c < prev_c:
        return "Three Black Crows"
    return None

def _check_bullish_abandoned_baby(o, c, h, l, prev_o, prev_c, prev_h, prev_l,
                                  prev2_o, prev2_c, prev2_h, prev2_l, **_):
    if not _is_bearish(prev2_o, prev2_c) or not _is_bullish(o, c): return None
    second_body = _body_size(prev_o, prev_c)
    second_range = _range(prev_h, prev_l)
    if second_range == 0: return None
    if (second_body / second_range < 0.1
            and prev_h < prev2_l
            and prev_h < l):
        return "Bullish Abandoned Baby"
    return None

def _check_bearish_abandoned_baby(o, c, h, l, prev_o, prev_c, prev_h, prev_l,
                                  prev2_o, prev2_c, prev2_h, prev2_l, **_):
    if not _is_bullish(prev2_o, prev2_c) or not _is_bearish(o, c): return None
    second_body = _body_size(prev_o, prev_c)
    second_range = _range(prev_h, prev_l)
    if second_range == 0: return None
    if (second_body / second_range < 0.1
            and prev_l > prev2_h
            and prev_l > h):
        return "Bearish Abandoned Baby"
    return None

def _check_three_inside_up(o, c, h, l, prev_o, prev_c, prev_h, prev_l,
                           prev2_o, prev2_c, prev2_h, prev2_l, **_):
    if not (_is_bearish(prev2_o, prev2_c) and _is_bullish(prev_o, prev_c)
            and _is_bullish(o, c)):
        return None
    if (prev_o > prev2_c and prev_c < prev2_o and c > prev2_o):
        return "Three Inside Up"
    return None

def _check_three_inside_down(o, c, h, l, prev_o, prev_c, prev_h, prev_l,
                             prev2_o, prev2_c, prev2_h, prev2_l, **_):
    if not (_is_bullish(prev2_o, prev2_c) and _is_bearish(prev_o, prev_c)
            and _is_bearish(o, c)):
        return None
    if (prev_o < prev2_c and prev_c > prev2_o and c < prev2_o):
        return "Three Inside Down"
    return None

def _check_three_outside_up(o, c, h, l, prev_o, prev_c, prev_h, prev_l,
                            prev2_o, prev2_c, prev2_h, prev2_l, **_):
    if not (_is_bearish(prev2_o, prev2_c) and _is_bullish(prev_o, prev_c)
            and _is_bullish(o, c)):
        return None
    if (prev_o <= prev2_c and prev_c >= prev2_o and c > prev_c):
        return "Three Outside Up"
    return None

def _check_three_outside_down(o, c, h, l, prev_o, prev_c, prev_h, prev_l,
                              prev2_o, prev2_c, prev2_h, prev2_l, **_):
    if not (_is_bullish(prev2_o, prev2_c) and _is_bearish(prev_o, prev_c)
            and _is_bearish(o, c)):
        return None
    if (prev_o >= prev2_c and prev_c <= prev2_o and c < prev_c):
        return "Three Outside Down"
    return None

def _check_upside_gap_two_crows(o, c, h, l, prev_o, prev_c, prev_h, prev_l,
                                prev2_o, prev2_c, prev2_h, prev2_l, **_):
    """Nison 'upside gap two crows': long bullish candle (prev2, uptrend), a
    small bearish candle that gaps up and leaves the window open (prev), then
    a second bearish candle that opens at/above the prior open, engulfs the
    prior body, but still closes above the original window — the gap remains
    unfilled, which is why this is a warning rather than a confirmed reversal."""
    if not _is_bullish(prev2_o, prev2_c): return None
    if not _is_bearish(prev_o, prev_c): return None
    if not _is_bearish(o, c): return None
    if prev_l <= prev2_c:
        return None  # no window between candle 1 and candle 2
    if o >= prev_o and c <= prev_c and c > prev2_c:
        return "Upside Gap Two Crows"
    return None


# ── Registry: ordered list of detectors ──────────────────────────────────────
# Order matters: more-specific patterns checked before less-specific.
# E.g., Long-Legged Doji (specific) before Doji (generic).

ONE_BAR_DETECTORS = [
    _check_long_legged_doji,     # before Doji
    _check_dragonfly_doji,       # before Doji
    _check_gravestone_doji,      # before Doji
    _check_spinning_top,
    _check_hammer,
    _check_inverted_hammer,
    _check_shooting_star,
    _check_hanging_man,
    _check_bullish_marubozu,
    _check_bearish_marubozu,
    _check_doji,                 # generic Doji last
]

TWO_BAR_DETECTORS = [
    _check_bullish_engulfing, _check_bearish_engulfing,
    _check_bullish_harami, _check_bearish_harami,
    _check_piercing_line, _check_dark_cloud_cover,
    _check_tweezer_bottom, _check_tweezer_top,
    _check_bullish_kicker, _check_bearish_kicker,
    _check_bullish_counterattack, _check_bearish_counterattack,
    _check_bullish_separating_lines, _check_bearish_separating_lines,
]

THREE_BAR_DETECTORS = [
    _check_morning_doji_star,    # before Morning Star (more specific)
    _check_evening_doji_star,    # before Evening Star
    _check_bullish_abandoned_baby,  # before Morning Star
    _check_bearish_abandoned_baby,  # before Evening Star
    _check_upside_gap_two_crows,    # before generic 3-black-crows-style checks
    _check_morning_star,
    _check_evening_star,
    _check_three_white_soldiers,
    _check_three_black_crows,
    _check_three_inside_up,
    _check_three_inside_down,
    _check_three_outside_up,
    _check_three_outside_down,
]


# ── Trend detection (dual 50/200 MA system, faithful to Java getTrend()) ─────

def _compute_trend(
    closes: np.ndarray,
    fast_period: int = 50,
    slow_period: int = 200,
    threshold_pct: float = 0.5,
) -> np.ndarray:
    """
    Compute trend classification per bar.
    Returns array of strings: "uptrend", "downtrend", "sideways".

    Faithful to Java's getTrend():
      - Need at least `slow_period` bars
      - Compute fast MA (50) and slow MA (200)
      - Compute slow MA slope over 10 bars
      - UPTREND: price above slow MA by > threshold%, slow MA rising, fast > slow
      - DOWNTREND: price below slow MA by > threshold%, slow MA falling, fast < slow
      - SIDEWAYS: otherwise
    """
    n = len(closes)
    out = np.empty(n, dtype=object)
    out[:] = "sideways"

    if n < slow_period:
        return out

    # Compute fast and slow SMA via rolling
    s = pd.Series(closes)
    fast_ma = s.rolling(fast_period).mean().to_numpy()
    slow_ma = s.rolling(slow_period).mean().to_numpy()
    slow_ma_prev = s.shift(10).rolling(slow_period).mean().to_numpy()

    for i in range(n):
        if np.isnan(slow_ma[i]):
            continue
        sma_prev = slow_ma_prev[i] if not np.isnan(slow_ma_prev[i]) else slow_ma[i]
        ma_slope = slow_ma[i] - sma_prev
        ma_rising = ma_slope > 0
        ma_falling = ma_slope < 0
        fast_above_slow = (not np.isnan(fast_ma[i])) and fast_ma[i] > slow_ma[i]
        fast_below_slow = (not np.isnan(fast_ma[i])) and fast_ma[i] < slow_ma[i]

        current_price = closes[i]
        percent_diff = ((current_price - slow_ma[i]) / slow_ma[i]) * 100.0

        if percent_diff > threshold_pct and ma_rising and fast_above_slow:
            out[i] = "uptrend"
        elif percent_diff < -threshold_pct and ma_falling and fast_below_slow:
            out[i] = "downtrend"
        # else: stays "sideways"

    return out


# ── Main entry point ─────────────────────────────────────────────────────────

def compute(
    df: pd.DataFrame,
    *,
    detect_1bar: bool = True,
    detect_2bar: bool = True,
    detect_3bar: bool = True,
    detect_bullish: bool = True,
    detect_bearish: bool = True,
    detect_neutral: bool = True,
    trend_filter: bool = False,
    fast_ma_period: int = 50,
    slow_ma_period: int = 200,
    trend_threshold_pct: float = 0.5,
    near_equal_tolerance: float = 0.05,
    precomputed_trend: Optional[np.ndarray] = None,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.DataFrame:
    """
    Scan the DataFrame for candlestick patterns.

    Parameters
    ----------
    df : OHLC DataFrame.
    detect_1bar/2bar/3bar : enable pattern categories.
    detect_bullish/bearish/neutral : enable pattern directions.
    trend_filter : if True, only show patterns in their correct trend context
        (bullish reversals in downtrends, bearish reversals in uptrends).
    fast_ma_period, slow_ma_period : MA periods for trend detection (ignored
        if `precomputed_trend` is supplied).
    trend_threshold_pct : % deviation from slow MA to confirm trend (ignored
        if `precomputed_trend` is supplied).
    near_equal_tolerance : Phase 3 fix — the "near enough" tolerance (as a
        fraction of average bar range) used by Tweezer Top/Bottom,
        Bullish/Bearish Counterattack, and Bullish/Bearish Separating Lines.
        Was hardcoded to 0.05 in six places; now a single configurable knob.
        Pass a larger value in high-ATR regimes, a smaller one in low-ATR
        regimes — `candlestick_engine.py` does this automatically using
        `MarketContext.volatility_regime`.
    precomputed_trend : Phase 3 fix — an optional pre-computed trend-label
        array (e.g. from `analysis._pattern_context.trend_mw_style()` or any
        other source, including the reconciled `MarketContext.trend_label`)
        to use INSTEAD of recomputing the 50/200-SMA trend here. This lets
        `candlestick_engine.py` compute trend exactly once and share it with
        both `candlestick_patterns_br.py` and this module, instead of each
        module maintaining its own, subtly different trend definition (see
        RESEARCH_SYNTHESIS.md, "Contradiction: what counts as a trend").
        Must be the same length as `df` with values in
        {"uptrend", "downtrend", "sideways"}. If omitted, behaviour is
        unchanged (this module computes its own 50/200-SMA trend as before).

    Returns
    -------
    Same DataFrame with csp_pattern, csp_category, csp_bars, csp_signal, csp_trend
    columns added. csp_pattern is NaN where no pattern detected.
    """
    out = df.copy()
    n = len(out)

    o = out[open_col].to_numpy(dtype=float)
    h = out[high_col].to_numpy(dtype=float)
    l = out[low_col].to_numpy(dtype=float)
    c = out[close_col].to_numpy(dtype=float)

    # Compute trend up-front if trend_filter is on (unless one was supplied)
    if precomputed_trend is not None:
        if len(precomputed_trend) != n:
            raise ValueError(
                f"precomputed_trend has length {len(precomputed_trend)}, expected {n}"
            )
        trend_arr = np.asarray(precomputed_trend, dtype=object)
    elif trend_filter:
        trend_arr = _compute_trend(c, fast_ma_period, slow_ma_period, trend_threshold_pct)
    else:
        trend_arr = np.array(["sideways"] * n, dtype=object)

    patterns = [None] * n
    categories = [None] * n
    bars_lengths = [0] * n
    signals = [0] * n

    for i in range(n):
        cur_o, cur_h, cur_l, cur_c = o[i], h[i], l[i], c[i]

        # Get previous bars (None if not available)
        if i >= 1:
            prev_o, prev_c, prev_h, prev_l = o[i-1], c[i-1], h[i-1], l[i-1]
        else:
            prev_o = prev_c = prev_h = prev_l = None
        if i >= 2:
            prev2_o, prev2_c, prev2_h, prev2_l = o[i-2], c[i-2], h[i-2], l[i-2]
        else:
            prev2_o = prev2_c = prev2_h = prev2_l = None

        detected = None

        # Try 1-bar patterns first (most specific)
        if detect_1bar and detected is None:
            for det in ONE_BAR_DETECTORS:
                result = det(cur_o, cur_c, cur_h, cur_l)
                if result:
                    cat = _categorize(result)
                    if (cat == "bullish" and not detect_bullish): continue
                    if (cat == "bearish" and not detect_bearish): continue
                    if (cat == "neutral" and not detect_neutral): continue
                    detected = result
                    break

        # Then 2-bar patterns
        if detect_2bar and detected is None and prev_o is not None:
            for det in TWO_BAR_DETECTORS:
                result = det(cur_o, cur_c, cur_h, cur_l,
                             prev_o, prev_c, prev_h, prev_l,
                             tolerance=near_equal_tolerance)
                if result:
                    cat = _categorize(result)
                    if (cat == "bullish" and not detect_bullish): continue
                    if (cat == "bearish" and not detect_bearish): continue
                    if (cat == "neutral" and not detect_neutral): continue
                    detected = result
                    break

        # Then 3-bar patterns
        if detect_3bar and detected is None and prev2_o is not None:
            for det in THREE_BAR_DETECTORS:
                result = det(cur_o, cur_c, cur_h, cur_l,
                             prev_o, prev_c, prev_h, prev_l,
                             prev2_o, prev2_c, prev2_h, prev2_l)
                if result:
                    cat = _categorize(result)
                    if (cat == "bullish" and not detect_bullish): continue
                    if (cat == "bearish" and not detect_bearish): continue
                    if (cat == "neutral" and not detect_neutral): continue
                    detected = result
                    break

        # Apply trend filter (also applies automatically when the caller
        # supplied a precomputed_trend, since that signals deliberate intent
        # to use it — no reason to compute it and then silently ignore it)
        if detected is not None and (trend_filter or precomputed_trend is not None):
            current_trend = trend_arr[i]
            if not _should_display_pattern(detected, current_trend):
                detected = None

        if detected is not None:
            patterns[i] = detected
            cat = _categorize(detected)
            categories[i] = cat
            bars_lengths[i] = PATTERN_LENGTH.get(detected, 0)
            if cat == "bullish":
                signals[i] = 1
            elif cat == "bearish":
                signals[i] = -1
            # neutral stays 0

    out["csp_pattern"] = patterns
    out["csp_category"] = categories
    out["csp_bars"] = bars_lengths
    out["csp_signal"] = signals
    out["csp_trend"] = trend_arr if trend_filter else None
    return out


# Which 1-bar patterns get a directional "did the next bar confirm it"
# check, and which direction confirmation means for each.
_CONFIRMABLE_BULLISH_1BAR = {"Hammer", "Inverted Hammer", "Dragonfly Doji"}
_CONFIRMABLE_BEARISH_1BAR = {"Shooting Star", "Hanging Man", "Gravestone Doji"}


def add_confirmation(
    out: pd.DataFrame,
    *,
    open_col: str = "open", high_col: str = "high",
    low_col: str = "low", close_col: str = "close",
) -> pd.DataFrame:
    """
    Phase 3 fix (see AUDIT_REPORT.md "mw.py: zero confirmation logic despite
    research consensus that single-bar reversal patterns need it"): adds two
    columns, `csp_confirmed` (bool) and `csp_confirmation_pending` (bool), to
    the output of `compute()`.

    Confirmation rule (matches the "candle over candle" / "candle under
    candle" concept used consistently across the video sources — Ross
    Cameron, Rayner Teo, Warrior Trading small-account transcripts — and the
    IG article's "the next day must be bullish to confirm this reversal
    pattern"): a bullish 1-bar pattern (Hammer, Inverted Hammer, Dragonfly
    Doji) is confirmed if the FOLLOWING bar's close breaks above the pattern
    bar's high; a bearish 1-bar pattern (Shooting Star, Hanging Man,
    Gravestone Doji) is confirmed if the following bar's close breaks below
    the pattern bar's low. Multi-bar patterns (engulfing, harami, morning/
    evening star, etc.) already encode their own confirmation bar as part of
    the pattern definition, so they are left untouched here.

    *** CAUSALITY / NO-REPAINTING NOTE — READ BEFORE USING LIVE ***
    `csp_confirmed[i]` is only knowable once bar i+1 has CLOSED. This
    function is safe to use in historical analysis/backtests (where the
    full DataFrame already exists), because it never changes a signal's
    values after computing them (no repainting) — but `csp_confirmed` on
    the LAST row of any live/streaming DataFrame is always undefined, which
    is exactly why the last row's `csp_confirmation_pending` is always
    forced to `True`. Do not backdate a "confirmed" reading on the last row.
    `candlestick_engine.py` enforces this by never using confirmation
    evidence for the most recent bar's confidence score.
    """
    out = out.copy()
    n = len(out)
    h = out[high_col].to_numpy(dtype=float)
    l = out[low_col].to_numpy(dtype=float)
    c = out[close_col].to_numpy(dtype=float)
    patterns = out["csp_pattern"].to_numpy(dtype=object)

    confirmed = np.zeros(n, dtype=bool)
    pending = np.zeros(n, dtype=bool)

    for i in range(n):
        p = patterns[i]
        if p is None or (isinstance(p, float) and np.isnan(p)):
            continue
        if p not in _CONFIRMABLE_BULLISH_1BAR and p not in _CONFIRMABLE_BEARISH_1BAR:
            continue
        if i + 1 >= n:
            pending[i] = True  # can't know yet — the confirming bar hasn't happened
            continue
        if p in _CONFIRMABLE_BULLISH_1BAR:
            confirmed[i] = c[i + 1] > h[i]
        else:
            confirmed[i] = c[i + 1] < l[i]

    out["csp_confirmed"] = confirmed
    out["csp_confirmation_pending"] = pending
    return out


def _should_display_pattern(pattern_name: str, current_trend: str) -> bool:
    """Faithful to Java's shouldDisplayPattern()."""
    if current_trend == "sideways":
        return True
    if pattern_name in BULLISH_REVERSAL_PATTERNS:
        return current_trend == "downtrend"
    if pattern_name in BEARISH_REVERSAL_PATTERNS:
        return current_trend == "uptrend"
    # Neutral + continuation patterns appear in any trend
    return True


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    n = 500
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(42)
    # Random-walk with some volatility to generate patterns
    close = 1.0850 + np.cumsum(rng.normal(0, 0.0005, n))
    open_ = close + rng.normal(0, 0.0002, n)
    high = np.maximum(open_, close) + rng.uniform(0.0001, 0.0005, n)
    low = np.minimum(open_, close) - rng.uniform(0.0001, 0.0005, n)
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)

    # Without trend filter
    out = compute(df)
    sig_count = (out["csp_pattern"].notna()).sum()
    print(f"Rows: {len(out)}")
    print(f"Patterns detected (no trend filter): {sig_count}")
    print(f"  Bullish: {int((out['csp_category'] == 'bullish').sum())}")
    print(f"  Bearish: {int((out['csp_category'] == 'bearish').sum())}")
    print(f"  Neutral: {int((out['csp_category'] == 'neutral').sum())}")

    top_patterns = out["csp_pattern"].value_counts().head(10)
    print("\nTop 10 patterns:")
    print(top_patterns)

    # With trend filter (needs 200-bar warmup)
    out_tf = compute(df, trend_filter=True, fast_ma_period=20, slow_ma_period=100,
                     trend_threshold_pct=0.5)
    sig_count_tf = (out_tf["csp_pattern"].notna()).sum()
    print(f"\nPatterns detected (trend filter): {sig_count_tf}")
    print(f"  (Fewer than {sig_count} because reversals only show in correct trend)")

    # Verify trend distribution
    print("\nTrend distribution:")
    print(out_tf["csp_trend"].value_counts())

    assert sig_count > 0, "expected at least some patterns in random walk"
    assert sig_count_tf <= sig_count, "trend filter should reduce or keep count"
    print("\nCandlestick patterns (MotiveWave port) smoke test passed.")

    # ── Regression check: Phase 3 additions ─────────────────────────────
    print("\n── Regression check: near_equal_tolerance, precomputed_trend, confirmation ──")

    # Tighter tolerance should never find MORE tweezer/counterattack/separating
    # matches than the default; looser should never find fewer.
    out_tight = compute(df, near_equal_tolerance=0.01)
    out_loose = compute(df, near_equal_tolerance=0.20)
    tol_sensitive = {"Tweezer Bottom", "Tweezer Top", "Bullish Counterattack",
                      "Bearish Counterattack", "Bullish Separating Lines",
                      "Bearish Separating Lines"}
    n_tight = out_tight["csp_pattern"].isin(tol_sensitive).sum()
    n_loose = out_loose["csp_pattern"].isin(tol_sensitive).sum()
    assert n_loose >= n_tight, "looser near_equal_tolerance should find >= matches"
    print(f"  Tolerance-sensitive patterns — tight(0.01): {n_tight}, loose(0.20): {n_loose}")

    # precomputed_trend should be honored instead of recomputed, and should
    # auto-apply trend filtering even though trend_filter itself is left False.
    fake_trend = np.array(["downtrend"] * n, dtype=object)  # force "always downtrend"
    out_forced = compute(df, precomputed_trend=fake_trend)
    # every bearish reversal pattern should now be filtered out (forced downtrend
    # only allows bullish reversals + neutral/continuation through)
    bearish_leftover = out_forced["csp_pattern"].isin(BEARISH_REVERSAL_PATTERNS).sum()
    assert bearish_leftover == 0, (
        "precomputed_trend='downtrend' everywhere should suppress all bearish "
        "reversal patterns, but some leaked through"
    )
    print(f"  precomputed_trend honored: {bearish_leftover} bearish reversals "
          f"leaked through a forced all-downtrend context (expected 0)")

    # add_confirmation: confirmed must be a subset of detected 1-bar reversal
    # patterns, and the last row must always be "pending" (can't be confirmed
    # yet — this is the no-lookahead guarantee for live/streaming use).
    out_conf = add_confirmation(out)
    confirmable = out_conf["csp_pattern"].isin(_CONFIRMABLE_BULLISH_1BAR | _CONFIRMABLE_BEARISH_1BAR)
    assert (out_conf.loc[~confirmable, "csp_confirmed"] == False).all(), \
        "non-confirmable patterns must never be marked confirmed"
    assert out_conf["csp_confirmation_pending"].iloc[-1] == True or not confirmable.iloc[-1], \
        "the last row must be 'pending' if it holds a confirmable pattern (can't know the future)"
    print(f"  add_confirmation: {int(out_conf['csp_confirmed'].sum())} confirmed, "
          f"{int(out_conf['csp_confirmation_pending'].sum())} pending (last-row only, expected <=1)")
    assert int(out_conf["csp_confirmation_pending"].sum()) <= 1
    print("  Phase 3 mw.py regression checks passed.")