"""
fundamental/economic_calendar_api.py — Day 94 Institutional Economic Calendar
==============================================================================
Multi-source economic calendar with fallback chain:

    FairEconomy JSON → Forex Factory scraper → hardcoded fallback

Day 95 hotfix:
  - FairEconomy JSON added as Layer 0 (primary source).
    URL: https://nfs.faireconomy.media/ff_calendar_thisweek.json
    ForexFactory এর official JSON feed — no API key, no bot-detection.
  - Fxstreet RSS removed (returns 404 consistently).
  - _empty_result() trade_block defaults to True (conservative).
  - _normalize_ff_events() no longer double-filters by time window.
  - get_calendar() preserves correct source label when filtered list is empty.

Later update: Tradermade and Finnhub layers removed by user request (no
paid/registered API keys in use for this project). Both were optional
paid-tier sources used only for forecast/actual numeric fields consumed by
EconomicSurpriseEngine; get_forecast_actual_events() now always returns []
and callers already treat that as "no surprise data available".

Fetch chain:
    Layer 0: FairEconomy JSON   — primary, fast, reliable, no key needed
    Layer 1: FF scraper         — existing Day 90/91 cloudscraper path
    Layer 2: hardcoded fallback — last resort approximate schedule

Output shape (compatible with existing NewsFilter/AnalysisAgent):
    {
      "source":            "faireconomy_json" | "ff_scraper"
                           | "hardcoded_fallback" | "none",
      "events":            [{"title","currency","time","impact","forecast",
                             "previous","actual"}],
      "high_impact_count": int,
      "next_event":        {...} | None,
      "trade_block":       bool,
      "block_reason":      str,
    }
"""
from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pytz
import requests

from utils.logger import get_logger
from fundamental.faireconomy_cache import DEFAULT_HIGH_IMPACT_KEYWORDS

log = get_logger("economic_calendar_api")


# ── Impact level mapping ─────────────────────────────────────────
IMPACT_MAP = {
    # FairEconomy / FF scraper
    "high":   "HIGH",
    "medium": "MEDIUM",
    "low":    "LOW",
}

# FairEconomy JSON URL
FAIRECONOMY_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

# Day 98+ FIX: now points at the single shared list in faireconomy_cache.py
# instead of maintaining its own copy (see news_filter.py for the same fix).
HIGH_IMPACT_KEYWORDS = DEFAULT_HIGH_IMPACT_KEYWORDS


class EconomicCalendarAPI:
    """Multi-source economic calendar with automatic fallback."""

    BLOCK_WINDOW_MINUTES = 30  # block trades ±30min around high-impact events

    def __init__(self):
        # TraderMade / Finnhub sources removed by user request (no paid API
        # keys in use). The calendar now relies only on the free,
        # no-key-needed sources: FairEconomy JSON (primary) and the Forex
        # Factory scraper (fallback). See get_forecast_actual_events() for
        # the forecast/actual-data implication of this.
        pass

    # ─────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────

    def get_calendar(
        self,
        currencies: List[str] = None,
        hours_ahead: int = 24,
    ) -> Dict[str, Any]:
        """Fetch upcoming economic events.

        Args:
            currencies:  filter by currency codes (e.g. ["USD","EUR"]).
                         None = all major currencies.
            hours_ahead: look this many hours forward from now.

        Returns: dict with source, events, high_impact_count, next_event,
                 trade_block, block_reason.
        """
        if currencies is None:
            currencies = ["USD", "EUR", "GBP", "JPY"]

        events = None
        source = "none"

        # ── Layer 0: FairEconomy JSON (Day 95 — primary) ──
        events = self._fetch_faireconomy(currencies)
        if events:
            source = "faireconomy_json"

        # Tradermade / Finnhub layers removed by user request (no paid API
        # keys in use) — the chain now goes straight from FairEconomy JSON
        # to the Forex Factory scraper below.

        # ── Layer 3: Forex Factory scraper (news_filter module) ──
        if not events:
            try:
                from fundamental.news_filter import NewsFilter
                nf = NewsFilter()
                ff_events, ff_source = nf._fetch_events()
                log.debug(
                    f"[EconCal] FF layer: source={ff_source} "
                    f"raw_events={len(ff_events)}"
                )
                if ff_events:
                    events = self._normalize_ff_events(ff_events, currencies)
                    log.debug(
                        f"[EconCal] FF normalized (currency filter only): "
                        f"{len(events)} events"
                    )
                    if events:
                        source = ff_source
            except Exception as e:
                log.warning(f"[EconCal] FF scraper fallback failed: {e}")

        # ── All layers failed — conservative block ──
        if not events:
            log.warning(
                "[EconCal] All calendar sources returned 0 events — "
                "returning conservative trade_block=True"
            )
            return self._empty_result(
                "All calendar sources failed — trading blocked (unknown calendar risk)",
                block=True,
            )

        # Filter by currency + time window
        now        = datetime.now(timezone.utc)
        window_end = now + timedelta(hours=hours_ahead)
        filtered   = []
        for ev in events:
            ev_time = ev.get("time")
            if ev_time is None:
                continue
            if ev.get("currency") not in currencies:
                continue
            if now <= ev_time <= window_end:
                filtered.append(ev)

        # Sort by time
        filtered.sort(key=lambda e: e["time"])

        high_impact = [e for e in filtered if e.get("impact") == "HIGH"]
        next_event  = filtered[0] if filtered else None
        block, reason = self._check_block(filtered, now)

        log.info(
            f"[EconCal] source={source} | raw={len(events)} | "
            f"filtered(24h)={len(filtered)} | high_impact={len(high_impact)} | "
            f"block={block}"
        )

        return {
            "source":            source,
            "events":            filtered,
            "high_impact_count": len(high_impact),
            "next_event":        self._format_event(next_event) if next_event else None,
            "trade_block":       block,
            "block_reason":      reason,
            "fetched_at":        now.isoformat(timespec="seconds"),
        }

    # ─────────────────────────────────────────────────────────
    # Forecast/actual events (Day 98+ — for EconomicSurpriseEngine)
    # ─────────────────────────────────────────────────────────

    def get_forecast_actual_events(
        self,
        currencies: List[str] = None,
        hours_ahead: int = 168,
    ) -> List[Dict]:
        """Fetch events that carry BOTH forecast and actual values.

        NOTE: Tradermade and Finnhub — the two sources that used to provide
        this — were removed by user request (no paid API keys in use).
        FairEconomy JSON and the Forex Factory scraper (this module's other
        sources) don't populate forecast/actual fields, so this method now
        always returns an empty list. Kept as a stable no-op stub (instead
        of deleting it outright) so EconomicSurpriseEngine and any other
        caller don't need code changes — they already treat an empty list
        as "no forecast/actual data available" and fall back to
        NEUTRAL/confidence=0, exactly as before when both keys were unset.

        Returns: [] always (see note above).
        """
        log.debug(
            "[EconCal] get_forecast_actual_events(): no source configured "
            "(Tradermade/Finnhub removed) — returning empty list"
        )
        return []

    # ─────────────────────────────────────────────────────────
    # SOURCE 0: FairEconomy JSON (Day 95 — primary)
    # ─────────────────────────────────────────────────────────

    def _fetch_faireconomy(self, currencies: List[str]) -> Optional[List[Dict]]:
        """
        FairEconomy JSON feed — ForexFactory এর official data, no key needed.
        URL: https://nfs.faireconomy.media/ff_calendar_thisweek.json

        Day 97+ CRITICAL FIX: previously this method called requests.get()
        DIRECTLY — bypassing the shared faireconomy_cache.py module. This meant:
          1. NO caching (every call hit the API)
          2. NO stampede protection (multiple threads → multiple requests)
          3. news_filter.py ALSO called the same API via the cache module
        → 2+ HTTP requests per cycle → 429 Too Many Requests

        Now delegates to the shared cached fetch_faireconomy() function.
        """
        try:
            # Day 97+ FIX: use the shared cached fetcher instead of direct API call
            from fundamental.faireconomy_cache import fetch_faireconomy as _cached_fetch

            raw_events = _cached_fetch(
                watched_currencies=set(currencies),
                high_impact_keywords=HIGH_IMPACT_KEYWORDS,
            )

            if not raw_events:
                return None

            # Convert from cache format to our format (cache returns simplified dicts).
            # Day 99+ FIX (P1): previously every non-high event was hard-coded
            # to "MEDIUM", which inflated the apparent impact level of low-tier
            # events in logs and AI context (the trade-block logic is unaffected
            # since it only blocks on HIGH). The cache now preserves the feed's
            # original impact string ("HIGH"/"MEDIUM"/"LOW"); we honor it and
            # only fall back to "LOW" when the feed didn't classify at all.
            events = []
            for item in raw_events:
                raw_impact = (item.get("impact", "") or "").strip().upper()
                if raw_impact in ("HIGH", "MEDIUM", "LOW"):
                    impact = raw_impact
                elif item.get("high_impact"):
                    impact = "HIGH"
                else:
                    impact = "LOW"
                events.append({
                    "title":    item.get("title", ""),
                    "currency": item.get("currency", ""),
                    "time":     item.get("time"),
                    "impact":   impact,
                    "forecast": "",
                    "previous": "",
                    "actual":   "",
                })

            log.info(f"[FairEconomy] Fetched {len(events)} events this week (via shared cache)")
            return events or None

        except Exception as e:
            log.warning(f"[FairEconomy] fetch failed: {e}")
            return None

    # ─────────────────────────────────────────────────────────
    # SOURCE 1: Tradermade
    # ─────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────
    # SOURCE 3: normalize existing FF scraper events
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_ff_events(ff_events: list, currencies: list) -> list:
        """Convert existing FF scraper events to our format.

        Note: time-window filter removed here — outer get_calendar() loop
        handles that. Only currency filter applied.
        """
        result = []
        for ev in ff_events:
            try:
                t = ev.get("time")
                if not isinstance(t, datetime):
                    continue
                if ev.get("currency") not in currencies:
                    continue
                result.append({
                    "title":    ev.get("title", ""),
                    "currency": ev.get("currency", ""),
                    "time":     t,
                    "impact":   "HIGH" if ev.get("high_impact") else "LOW",
                    "forecast": "",
                    "previous": "",
                    "actual":   "",
                })
            except Exception:
                continue
        return result

    # ─────────────────────────────────────────────────────────
    # Trade-block logic
    # ─────────────────────────────────────────────────────────

    def _check_block(self, events: List[Dict], now: datetime) -> tuple:
        """Check if any high-impact event falls within the block window."""
        for ev in events:
            if ev.get("impact") != "HIGH":
                continue
            ev_time   = ev["time"]
            delta_min = (ev_time - now).total_seconds() / 60
            if abs(delta_min) <= self.BLOCK_WINDOW_MINUTES:
                direction = "in" if delta_min > 0 else "ago"
                return True, (
                    f"HIGH impact {ev['currency']} {ev['title']} "
                    f"@ {ev_time.strftime('%H:%M UTC')} "
                    f"({abs(int(delta_min))}min {direction})"
                )
        return False, ""

    # ─────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _format_event(ev: Dict) -> Dict:
        return {
            "title":    ev.get("title", ""),
            "currency": ev.get("currency", ""),
            "time":     ev["time"].strftime("%Y-%m-%d %H:%M UTC"),
            "impact":   ev.get("impact", "LOW"),
            "forecast": ev.get("forecast", ""),
            "previous": ev.get("previous", ""),
            "actual":   ev.get("actual", ""),
        }

    @staticmethod
    def _empty_result(reason: str, block: bool = True) -> Dict[str, Any]:
        """Return a safe empty result.

        block defaults to True — calendar outage = unknown risk = no trading.
        """
        return {
            "source":            "none",
            "events":            [],
            "high_impact_count": 0,
            "next_event":        None,
            "trade_block":       block,
            "block_reason":      reason,
            "fetched_at":        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    # ─────────────────────────────────────────────────────────
    # AI context (for MasterAnalyst prompt)
    # ─────────────────────────────────────────────────────────

    def get_ai_context(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Compact context dict for MasterAnalyst."""
        return {
            "econcal_source":       result.get("source", "none"),
            "econcal_event_count":  len(result.get("events", [])),
            "econcal_high_impact":  result.get("high_impact_count", 0),
            "econcal_trade_block":  result.get("trade_block", False),
            "econcal_block_reason": result.get("block_reason", ""),
            "econcal_next_event":   result.get("next_event"),
        }

    def print_summary(self, result: Dict[str, Any]) -> None:
        bar = "═" * 50
        log.info(bar)
        log.info("  📅  ECONOMIC CALENDAR  (Day 94)")
        log.info(bar)
        log.info(f"  Source         : {result.get('source', '?')}")
        log.info(f"  Events (24h)   : {len(result.get('events', []))}")
        log.info(f"  High impact    : {result.get('high_impact_count', 0)}")
        log.info(f"  Trade block    : {'⛔ YES' if result.get('trade_block') else '✅ no'}")
        if result.get("block_reason"):
            log.info(f"  Block reason   : {result['block_reason']}")
        nxt = result.get("next_event")
        if nxt:
            log.info(
                f"  Next event     : {nxt['currency']} {nxt['title']} "
                f"@ {nxt['time']} [{nxt['impact']}]"
            )
        log.info(bar)