# broker/data_validator.py  —  Day 32 Part 4, 5 | Data Validation + Gap Fill
# ============================================================
# ভুল data মানে ভুল trade — তাই AI analysis brain-এ পাঠানোর আগে
# candle data-কে তিনটা চেক পার হতে হয়:
#   1. Missing candle (gap)  → detect + auto-fill
#   2. Invalid price          → reject (close=None, high<low, ইত্যাদি)
#   3. Duplicate               → same timestamp দুইবার process না হওয়া
#
# Bonus: Data Quality Score — প্রতিটা batch-এর জন্য 0-100 স্কোর,
# যা AIAnalyst-এর confidence-এর সাথে combine করা যায়।
# ============================================================

from datetime import datetime, timedelta
from utils.logger import get_logger

log = get_logger("data_validator")

# Timeframe → প্রত্যাশিত candle ব্যবধান (মিনিটে)
TIMEFRAME_MINUTES = {
    "M1": 1, "M5": 5, "M15": 15,
    "H1": 60, "H4": 240, "D1": 1440,
}


class DataValidator:
    """
    Candle list validate করে, gap detect/fill করে, duplicate বাদ দেয়,
    এবং একটা quality score বের করে।

    Usage:
        validator = DataValidator(data_feed)
        clean, report = validator.validate_and_fill(candles, symbol="EURUSD", timeframe="M15")
    """

    def __init__(self, data_feed=None):
        self.feed = data_feed   # gap fill করার সময় re-fetch করার জন্য MT5DataFeed instance

    # ─────────────────────────────────────────────
    # MAIN ENTRY
    # ─────────────────────────────────────────────

    def validate_and_fill(
        self, candles: list[dict], symbol: str, timeframe: str
    ) -> tuple[list[dict], dict]:
        """
        Returns (clean_candles, report). report-এ থাকে:
        {valid, invalid_count, duplicate_count, gaps_found, gaps_filled, quality_score}
        """
        report = {
            "symbol": symbol, "timeframe": timeframe,
            "input_count": len(candles),
            "invalid_count": 0, "duplicate_count": 0,
            "gaps_found": 0, "gaps_filled": 0,
            "quality_score": 100,
        }

        if not candles:
            report["quality_score"] = 0
            return [], report

        # Step 1 — invalid price rows বাদ দেওয়া
        valid = self._filter_invalid(candles, report)

        # Step 2 — duplicate timestamp বাদ দেওয়া
        deduped = self._dedupe(valid, report)

        # Step 3 — gap detect ও fill (timeframe জানা থাকলেই সম্ভব)
        filled = self._detect_and_fill_gaps(deduped, symbol, timeframe, report)

        report["output_count"] = len(filled)
        report["quality_score"] = self._compute_quality_score(report)

        return filled, report

    # ─────────────────────────────────────────────
    # 1. INVALID PRICE CHECK
    # ─────────────────────────────────────────────

    def _filter_invalid(self, candles: list[dict], report: dict) -> list[dict]:
        valid = []
        for c in candles:
            if self._is_invalid(c):
                report["invalid_count"] += 1
                log.warning(f"[DataValidator] Invalid candle rejected: {c}")
                continue
            valid.append(c)
        return valid

    def _is_invalid(self, c: dict) -> bool:
        required = ("open", "high", "low", "close")
        if any(c.get(k) is None for k in required):
            return True
        if c["high"] < c["low"]:
            return True
        if not (c["low"] <= c["open"] <= c["high"]):
            return True
        if not (c["low"] <= c["close"] <= c["high"]):
            return True
        if any(c.get(k, 0) <= 0 for k in required):
            return True
        return False

    # ─────────────────────────────────────────────
    # 2. DUPLICATE CHECK
    # ─────────────────────────────────────────────

    def _dedupe(self, candles: list[dict], report: dict) -> list[dict]:
        seen = set()
        deduped = []
        for c in candles:
            ts = c.get("time")
            if ts in seen:
                report["duplicate_count"] += 1
                continue
            seen.add(ts)
            deduped.append(c)
        return deduped

    # ─────────────────────────────────────────────
    # 3. GAP DETECT + AUTO FILL
    # ─────────────────────────────────────────────

    def _detect_and_fill_gaps(
        self, candles: list[dict], symbol: str, timeframe: str, report: dict
    ) -> list[dict]:
        expected_minutes = TIMEFRAME_MINUTES.get(timeframe)
        if not expected_minutes or len(candles) < 2:
            return candles

        sorted_candles = sorted(candles, key=lambda c: c["time"])
        filled = [sorted_candles[0]]

        for prev, curr in zip(sorted_candles, sorted_candles[1:]):
            prev_t = datetime.fromisoformat(prev["time"])
            curr_t = datetime.fromisoformat(curr["time"])
            gap_minutes = (curr_t - prev_t).total_seconds() / 60
            expected_gap = expected_minutes

            if gap_minutes > expected_gap * 1.5:
                # P1 fix (audit §3.3): a gap that is fully explained by the
                # market being closed (weekend, and optionally a configured
                # holiday) is not "missing data" — it's expected. Only
                # gaps that overlap actual trading-session time should be
                # counted/recovered/flat-filled. Without this check, every
                # Friday-close→Sunday-open gap on M15 data synthesized
                # ~191 fake flat candles (see docstring at top of file).
                if self._is_market_closed_gap(prev_t, curr_t, tolerance_minutes=expected_gap * 2):
                    log.debug(
                        f"[DataValidator] Gap {prev['time']} → {curr['time']} "
                        f"is market-closed (weekend/holiday) — not treated as missing data"
                    )
                    filled.append(curr)
                    continue

                missing_count = round(gap_minutes / expected_gap) - 1
                report["gaps_found"] += 1
                log.warning(
                    f"[DataValidator] Gap detected: {symbol} {timeframe} "
                    f"{prev['time']} → {curr['time']} ({missing_count} candle missing)"
                )
                recovered = self._attempt_recover(symbol, timeframe, prev_t, curr_t)
                if recovered:
                    filled.extend(recovered)
                    report["gaps_filled"] += len(recovered)
                else:
                    # Recovery না হলে flat-fill করো (prev candle-এর close রিপিট করে) —
                    # এতে gap visually bridge হয়, AI brain-এ ভুল spike না যায়
                    filled.extend(
                        self._flat_fill(prev, curr_t, missing_count, expected_gap)
                    )
                    report["gaps_filled"] += missing_count

            filled.append(curr)

        return filled

    # Forex market: closed roughly Fri 21:00 UTC → Sun 21:00 UTC. Kept as a
    # class constant (not hardcoded inline) so a broker with different
    # session hours can override it without touching the gap-fill logic.
    WEEKEND_CLOSE_UTC = (4, 21, 0)   # (weekday Mon=0..Sun=6, hour, minute) — Fri 21:00
    WEEKEND_OPEN_UTC = (6, 21, 0)    # Sun 21:00
    # Optional: caller can pass known holiday dates (date objects) to treat
    # as additional closed days — off by default since holiday calendars
    # are broker/region specific and not available in this file.
    HOLIDAYS: set = set()

    def _is_market_closed_gap(self, start: datetime, end: datetime, tolerance_minutes: float = 0) -> bool:
        """
        True if the [start, end] interval is (at least mostly) explained by
        the forex weekend close, so it should NOT be treated as a data gap.

        For each calendar day the interval touches, checks whether the
        portion of that day inside [start, end) falls within closed hours:
          - Saturday: always closed, all day.
          - Sunday: closed until WEEKEND_OPEN_UTC's hour:minute (plus a
            small tolerance — the first real candle after reopen naturally
            lands one timeframe-interval after the exact open time, that's
            not "missing data").
          - Friday: closed from WEEKEND_CLOSE_UTC's hour:minute onward
            (same tolerance, symmetric, for the last candle before close).
          - configured holidays: closed all day.
          - any other weekday: NOT closed — a real gap.

        Args:
            tolerance_minutes: slack absorbed around the open/close
                boundary, so normal candle-grid alignment (e.g. the first
                M15 candle after a 21:00 open landing at 21:00, 21:15, ...)
                isn't misread as "market was open but data is missing".
                Callers pass ~2x the timeframe's expected candle spacing.
        """
        # Fast path: if the gap is short, it can't be a weekend gap.
        if (end - start).total_seconds() < 20 * 3600:  # < 20h can't span a weekend close
            return False

        close_wd, close_h, close_m = self.WEEKEND_CLOSE_UTC
        open_wd, open_h, open_m = self.WEEKEND_OPEN_UTC
        tolerance = timedelta(minutes=max(tolerance_minutes, 0))

        # `end` is the timestamp of curr — an actual candle that exists,
        # not a missing point. Treat the checked interval as half-open
        # [start, end) so the reopen candle itself doesn't get compared
        # against the open-time boundary and falsely flagged.
        effective_end = end - timedelta(microseconds=1)

        cursor = start
        while cursor.date() <= effective_end.date():
            day = cursor.date()
            weekday = day.weekday()  # Mon=0 ... Sun=6
            day_start = max(start, datetime.combine(day, datetime.min.time()))
            day_end = min(effective_end, datetime.combine(day, datetime.max.time()))

            if day in self.HOLIDAYS:
                pass  # fully closed
            elif weekday == 5:  # Saturday — always closed
                pass
            elif weekday == close_wd:
                # Friday — closed only from close time onward (+ tolerance).
                day_close = datetime.combine(day, datetime.min.time()) + timedelta(hours=close_h, minutes=close_m)
                if day_start < day_close - tolerance:
                    return False
            elif weekday == open_wd:
                # Sunday — closed only until open time (+ tolerance).
                day_open = datetime.combine(day, datetime.min.time()) + timedelta(hours=open_h, minutes=open_m)
                if day_end > day_open + tolerance:
                    return False
            else:
                # Any other weekday inside the gap — market should have
                # been open, so this is a real gap, not weekend closure.
                return False

            cursor = datetime.combine(day, datetime.min.time()) + timedelta(days=1)

        return True

    def _attempt_recover(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> list[dict] | None:
        """
        MT5DataFeed দিয়ে missing range আবার fetch করার চেষ্টা করে —
        Day 31-এর broker_symbol resolve হয়ে থাকতে হবে (symbol এখানে
        আগেই broker-exact name হিসেবে আসা উচিত)।
        """
        if not self.feed:
            return None
        try:
            # প্র্যাক্টিক্যালি copy_rates_range ব্যবহার করা উচিত — এখানে
            # সরলীকরণের জন্য সাম্প্রতিক candle re-fetch করে filter করা হলো
            fresh = self.feed.get_candles(symbol, timeframe, count=500)
            recovered = [
                c for c in fresh
                if start < datetime.fromisoformat(c["time"]) < end
            ]
            return recovered or None
        except Exception as e:
            log.error(f"[DataValidator] Recovery fetch failed: {e}")
            return None

    def _flat_fill(
        self, prev: dict, next_time: datetime, count: int, step_minutes: int
    ) -> list[dict]:
        """Recovery সম্ভব না হলে — prev candle-এর close repeat করে gap bridge করা (last resort)।"""
        filler = []
        prev_t = datetime.fromisoformat(prev["time"])
        for i in range(1, count + 1):
            t = prev_t + timedelta(minutes=step_minutes * i)
            if t >= next_time:
                break
            filler.append({
                "time": t.isoformat(),
                "open": prev["close"], "high": prev["close"],
                "low": prev["close"], "close": prev["close"],
                "volume": 0, "spread": 0, "synthetic": True,
            })
        return filler

    # ─────────────────────────────────────────────
    # QUALITY SCORE
    # ─────────────────────────────────────────────

    def _compute_quality_score(self, report: dict) -> int:
        if report["input_count"] == 0:
            return 0
        penalty = (
            report["invalid_count"] * 3
            + report["duplicate_count"] * 1
            + report["gaps_found"] * 5
        )
        score = max(0, 100 - penalty)
        return score

    def print_report(self, report: dict) -> None:
        bar = "═" * 40
        icon = "✅" if report["quality_score"] >= 90 else ("🟡" if report["quality_score"] >= 70 else "🔴")
        log.info(bar)
        log.info(f"  {icon}  DATA QUALITY — {report['symbol']} {report['timeframe']}")
        log.info(bar)
        log.info(f"  Quality score : {report['quality_score']}/100")
        log.info(f"  Invalid       : {report['invalid_count']}")
        log.info(f"  Duplicates    : {report['duplicate_count']}")
        log.info(f"  Gaps found    : {report['gaps_found']}  (filled: {report['gaps_filled']})")
        log.info(bar)