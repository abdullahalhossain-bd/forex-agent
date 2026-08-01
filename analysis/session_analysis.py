# analysis/session_analysis.py  —  Day 62 | Session-Based Manipulation Detection
# ============================================================
# ICT "London Open Manipulation" concept:
#
#   Asian session range তৈরি হয় (tight, low volatility)
#        ↓
#   London open এসে একদিকে fake breakout দেখায়
#        ↓
#   দ্রুত reverse করে আসল direction-এ চলে যায়
#
# এই module Asian range + London candles দেখে এই pattern confirm করে।
# ============================================================

import warnings

import pandas as pd
from utils.logger import get_logger

log = get_logger("session_analysis")

# UTC session windows (approx, broker-dependent)
# Round-18 audit fix: London open window is now DST-aware.
# Previously: hardcoded LONDON_OPEN_START_HOUR=7, LONDON_OPEN_END_HOUR=10.
# But London actually opens at:
#   - 08:00 UTC in winter (GMT, Nov-Mar)
#   - 07:00 UTC in summer (BST, Mar-Oct)
# The old hardcoded 7:00 meant that in winter, 07:00-08:00 UTC candles
# (which are actually Tokyo/pre-London) were incorrectly tagged as
# "London manipulation" — causing false positive LONDON_LIQUIDITY_SWEEP
# alerts for ~5-6 months per year.
#
# Now: the start/end hours are computed dynamically per-candle based on
# whether EU DST is active at that candle's timestamp. We keep the old
# constants as fallback defaults for compatibility, but the actual
# filtering uses the DST-aware values.
LONDON_OPEN_START_HOUR_WINTER = 8   # GMT (Nov - mid March)
LONDON_OPEN_START_HOUR_SUMMER = 7   # BST (late March - Oct)
LONDON_OPEN_END_HOUR          = 10  # End is the same in both (10:00 UTC)
# Legacy constants (kept for backward compat — code that imports them
# gets the winter/default value, which is the safer/more conservative
# of the two):
LONDON_OPEN_START_HOUR = LONDON_OPEN_START_HOUR_WINTER
LONDON_OPEN_END_HOUR   = LONDON_OPEN_END_HOUR


def _is_eu_dst_for_timestamp(dt) -> bool:
    """Round-18: DST-aware London open window helper.

    Uses zoneinfo (Python 3.9+) to check if Europe/London is observing
    BST (British Summer Time) at the given datetime. Falls back to
    a fixed-date approximation if zoneinfo is unavailable.

    Args:
        dt: datetime (naive or tz-aware). If naive, assumed UTC.

    Returns:
        True if DST is active (summer), False otherwise (winter).
    """
    try:
        from zoneinfo import ZoneInfo
        from datetime import timezone as _tz
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_tz.utc)
        ld_time = dt.astimezone(ZoneInfo("Europe/London"))
        dst = ld_time.dst()
        return dst is not None and dst.total_seconds() > 0
    except Exception:
        # Fallback: fixed-date approximation (less accurate but better
        # than nothing if zoneinfo/tzdata is missing)
        month = dt.month
        day = dt.day
        if month < 3 or month > 10:
            return False
        if month > 3 and month < 10:
            return True
        if month == 3:
            return day >= 25  # approx last Sunday
        if month == 10:
            return day < 25   # approx last Sunday
        return False


class LondonManipulationDetector:
    """
    Usage:
        analyzer = LondonManipulationDetector()
        result = analyzer.detect_london_manipulation(df, asian_range)

    RENAMED (institutional review, Finding H-2/H-3, discovered during
    implementation): this class used to be named `SessionAnalyzer` — the
    exact same class name independently defined in `session_analyzer.py`
    (the newer, comprehensive session engine that supersedes this narrower
    London-manipulation-only detector). Two same-named classes in
    different modules is a silent-shadowing hazard: `from
    analysis.session_analysis import SessionAnalyzer` and `from
    analysis.session_analyzer import SessionAnalyzer` in the same file
    silently overwrite each other with no error, and whichever import runs
    last wins — with no warning that the "wrong" engine got wired in.

    Renamed to reflect what this class actually does. A deprecated
    `SessionAnalyzer` alias is kept below for backward compatibility with
    any existing `from analysis.session_analysis import SessionAnalyzer`
    callers; it emits a DeprecationWarning pointing here and at the
    canonical `session_analyzer.SessionAnalyzer`.

    Round-18 audit fix: London open window is now DST-aware. Previously
    hardcoded 07:00-10:00 UTC, which was correct for summer (BST) but
    WRONG for winter (GMT) — London opens at 08:00 UTC in winter. This
    caused false positive LONDON_LIQUIDITY_SWEEP alerts in winter months.
    Now uses _is_eu_dst_for_timestamp() to compute the correct window
    per candle.
    """

    SWEEP_BUFFER_ATR_MULT = 0.05   # range boundary-র সামান্য বাইরে গেলেও sweep ধরা হবে

    def detect_london_manipulation(
        self,
        df: pd.DataFrame,
        asian_range: dict,
    ) -> dict:
        """
        Asian range + London session candles দেখে fake-breakout → reversal
        pattern detect করো।

        Args:
            df          : OHLCV (atr column থাকা উচিত), DatetimeIndex
            asian_range : LiquidityZoneMapper.asian_session_range() এর output
        """
        if not asian_range.get('valid'):
            return self._empty_result("No valid Asian range available")

        if not isinstance(df.index, pd.DatetimeIndex):
            return self._empty_result("DataFrame index is not datetime")

        # Round-18: DST-aware London open window.
        # Each candle is checked against its OWN DST status — so a df
        # spanning a DST transition (e.g. late March) will correctly
        # use 07:00 for summer candles and 08:00 for winter candles.
        hours = df.index.hour
        # Vectorized DST check: compute DST flag per timestamp
        dst_flags = df.index.to_series().apply(_is_eu_dst_for_timestamp)
        # Start hour is 7 in summer (BST), 8 in winter (GMT)
        start_hours = dst_flags.map({True: LONDON_OPEN_START_HOUR_SUMMER,
                                      False: LONDON_OPEN_START_HOUR_WINTER})
        # Filter: hour >= start_hour (DST-aware) AND hour < end_hour
        london_mask = (hours >= start_hours) & (hours < LONDON_OPEN_END_HOUR)
        london_df = df[london_mask]

        if london_df.empty:
            return self._empty_result("No London session candles found")

        # সবচেয়ে recent London session নাও (same/next day as Asian range)
        last_day     = london_df.index.normalize().max()
        session      = london_df[london_df.index.normalize() == last_day]
        if session.empty:
            return self._empty_result("No recent London session candles")

        asian_high = asian_range['high']
        asian_low  = asian_range['low']
        atr        = self._safe_atr(df)
        buffer     = atr * self.SWEEP_BUFFER_ATR_MULT

        highs  = session['high'].values
        lows   = session['low'].values
        closes = session['close'].values

        swept_above = bool((highs > asian_high + buffer).any())
        swept_below = bool((lows  < asian_low  - buffer).any())

        current_close = float(closes[-1])

        event     = "NONE"
        direction = "NEUTRAL"
        note      = "No London liquidity sweep detected yet"

        # Bearish manipulation: swept above Asian high then closed back inside/below
        if swept_above and current_close < asian_high:
            event     = "LONDON_LIQUIDITY_SWEEP"
            direction = "BEARISH"
            note      = (
                f"London swept Asian high ({asian_high:.5f}) then rejected back "
                f"below — fake breakout, bearish reversal likely"
            )

        # Bullish manipulation: swept below Asian low then closed back inside/above
        elif swept_below and current_close > asian_low:
            event     = "LONDON_LIQUIDITY_SWEEP"
            direction = "BULLISH"
            note      = (
                f"London swept Asian low ({asian_low:.5f}) then rejected back "
                f"above — fake breakout, bullish reversal likely"
            )

        # Genuine breakout — swept and held beyond range, no rejection back in
        elif swept_above and current_close >= asian_high:
            event     = "LONDON_BREAKOUT"
            direction = "BULLISH"
            note      = f"London broke and held above Asian high ({asian_high:.5f}) — genuine breakout"

        elif swept_below and current_close <= asian_low:
            event     = "LONDON_BREAKOUT"
            direction = "BEARISH"
            note      = f"London broke and held below Asian low ({asian_low:.5f}) — genuine breakdown"

        result = {
            'valid':       True,
            'event':       event,
            'direction':   direction,
            'asian_high':  asian_high,
            'asian_low':   asian_low,
            'swept_above': swept_above,
            'swept_below': swept_below,
            'current_close': round(current_close, 5),
            'is_manipulation': event == "LONDON_LIQUIDITY_SWEEP",
            'note': note,
        }

        log.info(f"[SessionAnalyzer] {event} | Direction: {direction}")
        return result

    def _safe_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        try:
            val = df['atr'].iloc[-1]
            if val and val == val:  # not NaN
                return float(val)
        except Exception:
            pass
        return 0.0005

    def _empty_result(self, reason: str) -> dict:
        return {
            'valid': False, 'event': 'NONE', 'direction': 'NEUTRAL',
            'is_manipulation': False, 'note': reason,
        }

    # ─────────────────────────────────────────────
    # AI CONTEXT
    # ─────────────────────────────────────────────

    def get_ai_context(self, result: dict) -> dict:
        return {
            'session_event':        result.get('event', 'NONE'),
            'session_direction':    result.get('direction', 'NEUTRAL'),
            'session_manipulation': result.get('is_manipulation', False),
            'session_note':         result.get('note', ''),
        }

    # ─────────────────────────────────────────────
    # PRINT SUMMARY
    # ─────────────────────────────────────────────

    def print_summary(self, result: dict) -> None:
        bar  = "═" * 52
        icon = "🚨" if result.get('is_manipulation') else ("🟢" if result.get('event') == 'LONDON_BREAKOUT' else "🟡")
        log.info(bar)
        log.info("  🌍  SESSION ANALYSIS — LONDON OPEN  (Day 62)")
        log.info(bar)
        log.info(f"  {icon} Event     : {result.get('event')}")
        log.info(f"  Direction : {result.get('direction')}")
        log.info(f"  Note      : {result.get('note')}")
        log.info(bar)


class SessionAnalyzer(LondonManipulationDetector):
    """
    DEPRECATED backward-compatible alias.

    This name collided with session_analyzer.SessionAnalyzer (the
    comprehensive, superseding session engine). Use
    `LondonManipulationDetector` from this module, or preferably
    `session_analyzer.SessionAnalyzer` directly, going forward.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "session_analysis.SessionAnalyzer is a deprecated alias for "
            "LondonManipulationDetector, kept only to avoid breaking "
            "existing imports. It previously shared its exact class name "
            "with the unrelated, more comprehensive "
            "session_analyzer.SessionAnalyzer — importing both into the "
            "same namespace could silently shadow one with the other. "
            "Prefer session_analyzer.SessionAnalyzer for new code. "
            "See institutional review Findings H-2/H-3.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)