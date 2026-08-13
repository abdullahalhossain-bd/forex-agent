"""
fundamental/economic_calendar_api.py — Institutional Economic Calendar
==============================================================================
Multi-source economic calendar with fallback chain:

    FairEconomy JSON (primary, shared cache) → Forex Factory scraper → outage

Fetch chain:
    Layer 0: FairEconomy JSON   — primary, fast, reliable, no key needed
    Layer 1: FF scraper         — existing cloudscraper path
    (No hardcoded schedule — dates move; outage path is the deliberate last resort)

Output shape (stable contract):
    {
      "source":            "faireconomy_json" | "ff_scraper" | "none",
      "events":            [{"title","currency","time","impact","forecast",
                             "previous","actual"}],
      "high_impact_count": int,
      "next_event":        {...} | None,
      "trade_block":       bool,
      "block_reason":      str,
      "calendar_outage":   bool,          # always present
      "fetched_at":        ISO-8601 UTC,
    }
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

from utils.logger import get_logger
from fundamental.faireconomy_cache import DEFAULT_HIGH_IMPACT_KEYWORDS

log = get_logger("economic_calendar_api")

# FairEconomy JSON URL (official FF weekly feed, no key)
FAIRECONOMY_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

# Shared keyword list (single source of truth)
HIGH_IMPACT_KEYWORDS = DEFAULT_HIGH_IMPACT_KEYWORDS


class EconomicCalendarAPI:
    """Multi-source economic calendar with automatic fallback and correct
    ±BLOCK_WINDOW trade blocking (including the post-release window).
    """

    BLOCK_WINDOW_MINUTES = 30  # block trades ±30 min around high-impact events

    def __init__(self) -> None:
        # Paid sources (TraderMade / Finnhub) removed by design — no keys in use.
        pass

    # ─────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────

    def get_calendar(
        self,
        currencies: Optional[List[str]] = None,
        hours_ahead: int = 24,
    ) -> Dict[str, Any]:
        """Fetch upcoming economic events and decide whether trading must be blocked.

        Args:
            currencies:  filter by currency codes (e.g. ["USD","EUR"]).
                         None → major set.
            hours_ahead: look this many hours forward from now (for the public
                         events list). Block logic independently examines the
                         past BLOCK_WINDOW_MINUTES as well.

        Returns:
            Dict with the stable contract documented at module top.
        """
        if currencies is None:
            currencies = ["USD", "EUR", "GBP", "JPY"]

        # Back-test path: no live calendar exists for a historical bar.
        from core.constants import is_backtest_mode
        if is_backtest_mode():
            return self._empty_result(
                "backtest mode — live calendar skipped",
                block=False,
                calendar_outage=False,
            )

        events: Optional[List[Dict]] = None
        source = "none"

        # ── Layer 0: FairEconomy JSON (shared cache) ──
        events = self._fetch_faireconomy(currencies)
        if events:
            source = "faireconomy_json"

        # ── Layer 1: Forex Factory scraper ──
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
                    if events:
                        source = ff_source
            except Exception as e:
                log.warning(f"[EconCal] FF scraper fallback failed: {e}")

        # ── All layers failed ──
        if not events:
            # Default policy: a calendar OUTAGE (all sources unreachable)
            # is NOT the same as "high-impact event in progress" — yet the
            # previous default (ECONCAL_OUTAGE_ALLOWS_TRADES=false) treated
            # them identically, hard-blocking all trading whenever the
            # calendar API was unreachable. That produced endless
            # "[EconCal] All calendar sources returned 0 events" warnings
            # in trader.log with trades silently blocked even on quiet
            # trading days. Now default to ALLOW trades during outage
            # (calendar_unreliable=True is still flagged so downstream
            # risk code can choose to apply extra caution). Operators
            # who want the strict behavior can set
            # ECONCAL_OUTAGE_ALLOWS_TRADES=false explicitly.
            allow_trades_during_outage = os.getenv(
                "ECONCAL_OUTAGE_ALLOWS_TRADES", "true"
            ).lower() in ("1", "true", "yes")
            log.warning(
                "[EconCal] All calendar sources returned 0 events — "
                "marking calendar_outage=True. trade_block=%s "
                "(set ECONCAL_OUTAGE_ALLOWS_TRADES=false to hard-block "
                "trading during outages).",
                "False (allowing trades, calendar unreliable)" if allow_trades_during_outage
                else "True (hard-blocking trades)",
            )
            return self._empty_result(
                "All calendar sources failed — calendar outage. "
                "Trading blocked unless ECONCAL_OUTAGE_ALLOWS_TRADES=true.",
                block=not allow_trades_during_outage,
                calendar_outage=True,
            )

        # Ensure every event time is timezone-aware UTC
        for ev in events:
            t = ev.get("time")
            if isinstance(t, datetime) and t.tzinfo is None:
                ev["time"] = t.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        window_end = now + timedelta(hours=hours_ahead)
        block_start = now - timedelta(minutes=self.BLOCK_WINDOW_MINUTES)

        # Public list: future (and current) events only
        upcoming: List[Dict] = []
        # Internal set used solely for the ± window block decision
        block_candidates: List[Dict] = []

        for ev in events:
            ev_time = ev.get("time")
            if not isinstance(ev_time, datetime):
                continue
            if ev.get("currency") not in currencies:
                continue

            if now <= ev_time <= window_end:
                upcoming.append(ev)

            # Anything that falls inside the block window (past or future)
            if block_start <= ev_time <= now + timedelta(minutes=self.BLOCK_WINDOW_MINUTES):
                block_candidates.append(ev)

        upcoming.sort(key=lambda e: e["time"])

        high_impact = [e for e in upcoming if e.get("impact") == "HIGH"]
        next_event = upcoming[0] if upcoming else None
        block, reason = self._check_block(block_candidates, now)

        log.info(
            f"[EconCal] source={source} | raw={len(events)} | "
            f"upcoming(24h)={len(upcoming)} | high_impact={len(high_impact)} | "
            f"block={block}"
        )

        return {
            "source":            source,
            "events":            upcoming,
            "high_impact_count": len(high_impact),
            "next_event":        self._format_event(next_event) if next_event else None,
            "trade_block":       block,
            "block_reason":      reason,
            "calendar_outage":   False,
            "fetched_at":        now.isoformat(timespec="seconds"),
        }

    def get_forecast_actual_events(
        self,
        currencies: Optional[List[str]] = None,
        hours_ahead: int = 168,
    ) -> List[Dict]:
        """Stub — paid forecast/actual sources removed.

        Callers already treat an empty list as “no surprise data”.
        """
        log.debug(
            "[EconCal] get_forecast_actual_events(): no source configured "
            "(Tradermade/Finnhub removed) — returning empty list"
        )
        return []

    # ─────────────────────────────────────────────────────────
    # SOURCE 0: FairEconomy JSON (via shared cache)
    # ─────────────────────────────────────────────────────────

    def _fetch_faireconomy(self, currencies: List[str]) -> Optional[List[Dict]]:
        """Delegate to the shared cached fetcher (stampede-safe, rate-limit aware)."""
        try:
            from fundamental.faireconomy_cache import fetch_faireconomy as _cached_fetch

            raw_events = _cached_fetch(
                watched_currencies=set(currencies),
                high_impact_keywords=HIGH_IMPACT_KEYWORDS,
            )
            if not raw_events:
                return None

            events: List[Dict] = []
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

            log.info(
                f"[FairEconomy] Fetched {len(events)} events this week (via shared cache)"
            )
            return events or None

        except Exception as e:
            log.warning(f"[FairEconomy] fetch failed: {e}")
            return None

    # ─────────────────────────────────────────────────────────
    # SOURCE 1: normalize FF scraper events
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_ff_events(ff_events: list, currencies: list) -> list:
        """Convert FF-scraper events to the common schema.

        Time-window filtering is performed by the caller.
        Non-high events are labelled LOW (scraper only exposes a high-impact flag).
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

    def _check_block(
        self, candidates: List[Dict], now: datetime
    ) -> Tuple[bool, str]:
        """Return (block, reason) if any HIGH-impact event lies inside ±WINDOW."""
        for ev in candidates:
            if ev.get("impact") != "HIGH":
                continue
            ev_time = ev["time"]
            delta_min = (ev_time - now).total_seconds() / 60.0
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
    def _empty_result(
        reason: str,
        block: bool = True,
        calendar_outage: bool = False,
    ) -> Dict[str, Any]:
        """Safe empty result. calendar_outage is always present."""
        return {
            "source":            "none",
            "events":            [],
            "high_impact_count": 0,
            "next_event":        None,
            "trade_block":       block,
            "block_reason":      reason,
            "calendar_outage":   calendar_outage,
            "fetched_at":        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def get_ai_context(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Compact context for MasterAnalyst."""
        return {
            "econcal_source":       result.get("source", "none"),
            "econcal_event_count":  len(result.get("events", [])),
            "econcal_high_impact":  result.get("high_impact_count", 0),
            "econcal_trade_block":  result.get("trade_block", False),
            "econcal_block_reason": result.get("block_reason", ""),
            "econcal_next_event":   result.get("next_event"),
            "econcal_outage":       result.get("calendar_outage", False),
        }

    def print_summary(self, result: Dict[str, Any]) -> None:
        bar = "═" * 50
        log.info(bar)
        log.info("  📅  ECONOMIC CALENDAR")
        log.info(bar)
        log.info(f"  Source         : {result.get('source', '?')}")
        log.info(f"  Events (24h)   : {len(result.get('events', []))}")
        log.info(f"  High impact    : {result.get('high_impact_count', 0)}")
        log.info(f"  Trade block    : {'⛔ YES' if result.get('trade_block') else '✅ no'}")
        if result.get("block_reason"):
            log.info(f"  Block reason   : {result['block_reason']}")
        if result.get("calendar_outage"):
            log.info("  Calendar outage: True")
        nxt = result.get("next_event")
        if nxt:
            log.info(
                f"  Next event     : {nxt['currency']} {nxt['title']} "
                f"@ {nxt['time']} [{nxt['impact']}]"
            )
        log.info(bar)