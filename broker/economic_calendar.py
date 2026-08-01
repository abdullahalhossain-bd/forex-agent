# broker/economic_calendar.py  —  Day 32 Bonus 1 | Economic Event Awareness
# ============================================================
# decision_agent.py ইতিমধ্যে `analysis_out["news"]["trade_allowed"]`
# আশা করে (Gate 1 — news block), কিন্তু কোনো news-source uploaded
# ছিল না, তাই signal_pipeline.py এখন পর্যন্ত default True পাঠাচ্ছিল।
# এই module সেই gap পূরণ করে একটা real (যদিও lightweight) news
# window check দিয়ে।
#
# কোনো paid news API key লাগে না — এই version manual/CSV-based
# calendar ব্যবহার করে (high-impact event-এর সময় তুমি নিজে যুক্ত
# করবে .csv বা .json ফাইলে)। চাইলে পরে ForexFactory/Finnhub API দিয়ে
# replace করা যাবে — interface same থাকবে।
# ============================================================

import os
import json
from datetime import datetime, timedelta, timezone
from utils.logger import get_logger

log = get_logger("economic_calendar")

CALENDAR_PATH = "data/economic_calendar.json"
DEFAULT_BUFFER_MINUTES = 30   # high-impact event-এর আগে/পরে কত মিনিট trading বন্ধ


class EconomicCalendar:
    """
    Usage:
        cal = EconomicCalendar()
        cal.add_event("USD CPI", "2026-06-25T12:30:00Z", impact="HIGH")
        status = cal.check_news_window(currency="USD")
        # status = {"trade_allowed": False, "reason": "USD CPI in 8 min"}
    """

    def __init__(self, buffer_minutes: int = DEFAULT_BUFFER_MINUTES):
        self.buffer_minutes = buffer_minutes
        self._events = self._load()

    # ─────────────────────────────────────────────
    # EVENT MANAGEMENT
    # ─────────────────────────────────────────────

    def add_event(self, name: str, time_iso: str, impact: str = "HIGH", currency: str = None) -> None:
        self._events.append({
            "name": name, "time": time_iso,
            "impact": impact.upper(), "currency": currency,
        })
        self._save()
        log.info(f"[EconomicCalendar] Event added: {name} @ {time_iso} ({impact})")

    def clear_past_events(self) -> None:
        now = datetime.now(timezone.utc)
        self._events = [
            e for e in self._events
            if datetime.fromisoformat(e["time"].replace("Z", "+00:00")) > now - timedelta(hours=1)
        ]
        self._save()

    def get_today_events(self, currency: str = None) -> list[dict]:
        """Return all HIGH-impact events occurring today (UTC).
        Optional `currency` filter restricts to events for that currency.
        Used by orchestrator/daily_routine.py to build the morning briefing."""
        today = datetime.now(timezone.utc).date()
        out = []
        for event in self._events:
            if event["impact"] != "HIGH":
                continue
            if currency and event.get("currency") and event["currency"] != currency.upper():
                continue
            try:
                event_time = datetime.fromisoformat(event["time"].replace("Z", "+00:00"))
                if event_time.date() == today:
                    out.append(event)
            except Exception as e:
                log.warning(f"Suppressed exception at line 74: {e}")
                continue
        return out

    def get_upcoming_events(self, hours: int = 24, currency: str = None) -> list[dict]:
        """Return HIGH-impact events in the next `hours` hours."""
        now = datetime.now(timezone.utc)
        horizon = now + timedelta(hours=hours)
        out = []
        for event in self._events:
            if event["impact"] != "HIGH":
                continue
            if currency and event.get("currency") and event["currency"] != currency.upper():
                continue
            try:
                event_time = datetime.fromisoformat(event["time"].replace("Z", "+00:00"))
                if now <= event_time <= horizon:
                    out.append(event)
            except Exception as e:
                log.warning(f"Suppressed exception at line 93: {e}")
                continue
        return out

    # ─────────────────────────────────────────────
    # CHECK
    # ─────────────────────────────────────────────

    def check_news_window(self, currency: str = None, symbol: str = None) -> dict:
        """
        currency দিলে শুধু সেই currency-র event চেক হয় (যেমন "USD")।
        symbol দিলে (যেমন "EURUSD") দুটো currency-ই (EUR, USD) চেক হয়।
        কিছু না দিলে সব high-impact event চেক হয়।
        """
        relevant_currencies = self._currencies_for(currency, symbol)
        now = datetime.now(timezone.utc)

        for event in self._events:
            if event["impact"] != "HIGH":
                continue
            if relevant_currencies and event.get("currency") and event["currency"] not in relevant_currencies:
                continue

            event_time = datetime.fromisoformat(event["time"].replace("Z", "+00:00"))
            window_start = event_time - timedelta(minutes=self.buffer_minutes)
            window_end = event_time + timedelta(minutes=self.buffer_minutes)

            if window_start <= now <= window_end:
                minutes_to_event = round((event_time - now).total_seconds() / 60)
                direction = "in" if minutes_to_event >= 0 else "ago"
                return {
                    "trade_allowed": False,
                    "reason": f"{event['name']} {abs(minutes_to_event)} min {direction} — news window active",
                    "event": event,
                }

        return {"trade_allowed": True, "reason": "No high-impact news nearby"}

    def _currencies_for(self, currency: str | None, symbol: str | None) -> set[str]:
        if currency:
            return {currency.upper()}
        if symbol:
            sym = symbol.upper()[:6]
            return {sym[:3], sym[3:6]}
        return set()

    # ─────────────────────────────────────────────
    # INTERNAL
    # ─────────────────────────────────────────────

    def _load(self) -> list[dict]:
        os.makedirs(os.path.dirname(CALENDAR_PATH), exist_ok=True)
        if os.path.exists(CALENDAR_PATH):
            try:
                with open(CALENDAR_PATH) as f:
                    return json.load(f)
            except Exception as e:
                log.warning(f"Suppressed exception at line 150: {e}")
                pass
        return []

    def _save(self) -> None:
        os.makedirs(os.path.dirname(CALENDAR_PATH), exist_ok=True)
        with open(CALENDAR_PATH, "w") as f:
            json.dump(self._events, f, indent=2)