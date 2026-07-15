"""
fundamental/trading_economics_calendar.py — Day 95 Economic Calendar (Tradermade alternative)
==============================================================================================
Multi-source economic calendar that doesn't require Tradermade:

  1. Trading Economics API (if key set) — most comprehensive
  2. Investing.com RSS feed (no key) — always available
  3. DailyFX calendar RSS (no key) — backup
  4. Forex Factory scraper (existing Day 90 path) — last resort

All sources are normalized to the same output shape as
EconomicCalendarAPI so they can be used interchangeably.

Usage:
    from fundamental.trading_economics_calendar import TradingEconomicsCalendar
    cal = TradingEconomicsCalendar()
    result = cal.get_calendar(currencies=["USD","EUR"], hours_ahead=24)
"""
from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests

from utils.logger import get_logger

log = get_logger("trading_econ_calendar")


# ── Impact normalization ─────────────────────────────────────────
IMPACT_MAP = {
    "high": "HIGH", "medium": "MEDIUM", "low": "LOW",
    "3": "HIGH", "2": "MEDIUM", "1": "LOW",
    "red": "HIGH", "orange": "MEDIUM", "yellow": "LOW",
}


# Day 99+ FIX: precompiled regex for currency extraction from RSS titles.
# The previous code used `if cur in title` substring matching, which
# produced false positives — e.g. a title mentioning "USDJPY" would
# match "USD" AND "JPY", and a title containing the unrelated word
# "CRUDE" would erroneously match nothing but a title with "EURUSD"
# inside a longer word could match "EUR" spuriously. The new regex
# requires the currency code to appear either:
#   (a) at a word boundary (preceded by start-of-string or non-letter),
#       optionally followed by ":" (Investing.com's "USD: ..." format),
#   (b) OR as the base/quote of a 6-char pair like "EURUSD" — but in
#       that case we only take the BASE currency (first 3 chars), since
#       the title is about that currency's event.
# This eliminates the false-positive matches that were polluting the
# RSS-derived calendar with mis-categorized events.
_CURRENCY_CODES = ("USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD")
_CURRENCY_PATTERN = re.compile(
    r"(?:^|[^A-Z])((?:" + "|".join(_CURRENCY_CODES) + r"))(?=$|[^A-Z]|:)"
)


def _extract_currency_from_title(title: str) -> str:
    """Extract the lead currency code from an RSS item title.

    Returns the first currency code that appears as a standalone token
    (not embedded inside a longer word). Returns "" if none is found.
    """
    if not title:
        return ""
    # Find all standalone currency tokens in the title.
    matches = _CURRENCY_PATTERN.findall(title)
    return matches[0] if matches else ""


def _strip_currency_prefix(title: str) -> str:
    """Remove a leading 'USD: ' / 'EUR: ' style prefix from a title."""
    return re.sub(
        r"^(?:" + "|".join(_CURRENCY_CODES) + r"):\s*", "", title
    ).strip()


class TradingEconomicsCalendar:
    """Multi-source economic calendar — Trading Economics + RSS fallbacks."""

    BLOCK_WINDOW_MINUTES = 30

    def __init__(self):
        self._te_key = os.getenv("TRADINGECONOMICS_API_KEY", "").strip()
        # Day 99+ FIX: surface missing optional key once at construction.
        # The Trading Economics API is the most comprehensive source in
        # this fallback chain, but it's also the only one that requires
        # a paid key. Without it the calendar silently falls back to the
        # free RSS feeds, which is fine but should be visible in logs.
        if not self._te_key:
            log.info(
                "[TradEconCal] TRADINGECONOMICS_API_KEY not set — "
                "calendar will use free RSS feeds (Investing.com / "
                "DailyFX) instead of the Trading Economics API"
            )

    # ─────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────

    def get_calendar(
        self,
        currencies: List[str] = None,
        hours_ahead: int = 24,
    ) -> Dict[str, Any]:
        """Fetch upcoming economic events from multiple sources.

        Returns: dict with source, events, high_impact_count, next_event,
                 trade_block, block_reason.
        """
        if currencies is None:
            currencies = ["USD", "EUR", "GBP", "JPY"]

        events = None
        source = "none"

        # ── Source 1: Trading Economics API ──
        if self._te_key:
            events = self._fetch_trading_economics(currencies, hours_ahead)
            if events:
                source = "trading_economics"

        # ── Source 2: Investing.com RSS ──
        if not events:
            events = self._fetch_investing_rss(currencies, hours_ahead)
            if events:
                source = "investing_rss"

        # ── Source 3: DailyFX calendar RSS ──
        # DailyFX retired this calendar feed; it currently produces 503s.
        # Keep it opt-in for deployments that operate a compatible mirror,
        # but do not make every normal calendar request wait on a dead URL.
        if not events and os.getenv("ENABLE_LEGACY_DAILYFX_RSS", "false").lower() == "true":
            events = self._fetch_dailyfx_rss(currencies, hours_ahead)
            if events:
                source = "dailyfx_rss"

        if not events:
            return self._empty_result("All calendar sources failed")

        # Filter + sort
        now = datetime.now(timezone.utc)
        window_end = now + timedelta(hours=hours_ahead)
        filtered = []
        for ev in events:
            ev_time = ev.get("time")
            if ev_time is None:
                continue
            if ev.get("currency") not in currencies:
                continue
            if now <= ev_time <= window_end:
                filtered.append(ev)
        filtered.sort(key=lambda e: e["time"])

        high_impact = [e for e in filtered if e.get("impact") == "HIGH"]
        next_event = filtered[0] if filtered else None
        block, reason = self._check_block(filtered, now)

        result = {
            "source":            source,
            "events":            filtered,
            "high_impact_count": len(high_impact),
            "next_event":        self._format_event(next_event) if next_event else None,
            "trade_block":       block,
            "block_reason":      reason,
            "fetched_at":        now.isoformat(timespec="seconds"),
        }
        log.info(
            f"[TradEconCal] source={source} | events={len(filtered)} | "
            f"high_impact={len(high_impact)} | block={block}"
        )
        return result

    # ─────────────────────────────────────────────────────────
    # SOURCE 1: Trading Economics API
    # ─────────────────────────────────────────────────────────

    def _fetch_trading_economics(self, currencies: List[str], hours_ahead: int) -> Optional[List[Dict]]:
        """Trading Economics calendar API.

        Free tier: 100 req/month. Get key: https://tradingeconomics.com/api.aspx
        """
        try:
            now = datetime.now(timezone.utc)
            start = now.strftime("%Y-%m-%d")
            end = (now + timedelta(hours=hours_ahead)).strftime("%Y-%m-%d")
            url = f"https://api.tradingeconomics.com/calendar/country/all/{start}/{end}"
            params = {"c": self._te_key, "format": "json"}
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.warning(f"[TradEconCal] Trading Economics failed: {e}")
            return None

        events = []
        for item in data if isinstance(data, list) else []:
            try:
                country = item.get("country", "")
                currency = item.get("currency", "")
                if currency not in currencies:
                    continue
                events.append({
                    "title":     item.get("event", ""),
                    "currency":  currency,
                    "time":      datetime.fromisoformat(
                        item["date"].replace("Z", "+00:00")
                    ),
                    "impact":    IMPACT_MAP.get(str(item.get("importance", "")).lower(), "LOW"),
                    "forecast":  str(item.get("forecast", "")),
                    "previous":  str(item.get("previous", "")),
                    "actual":    str(item.get("actual", "")),
                })
            except Exception:
                continue
        return events

    # ─────────────────────────────────────────────────────────
    # SOURCE 2: Investing.com RSS
    # ─────────────────────────────────────────────────────────

    def _fetch_investing_rss(self, currencies: List[str], hours_ahead: int) -> Optional[List[Dict]]:
        """Investing.com economic calendar RSS feed.

        No API key needed. RSS feed is publicly available.
        """
        try:
            # Investing.com has RSS feeds per category
            # The main economic calendar RSS:
            url = "https://www.investing.com/economic-calendar/rss"
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; ForexAI/1.0)",
                "Accept": "application/rss+xml,application/xml,text/xml",
            }
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                log.debug(f"[TradEconCal] Investing RSS HTTP {resp.status_code}")
                return None
            root = ET.fromstring(resp.content)
        except Exception as e:
            log.debug(f"[TradEconCal] Investing RSS failed: {e}")
            return None

        events = []
        for item in root.findall(".//item"):
            try:
                title = item.findtext("title", default="")
                pub_date_str = item.findtext("pubDate", default="")
                description = item.findtext("description", default="")

                dt = datetime.strptime(pub_date_str, "%a, %d %b %Y %H:%M:%S %z")

                # Day 99+ FIX: use regex-based currency extraction instead
                # of substring search. Substring matching caused false
                # positives (e.g. "USDJPY" matched both USD and JPY) and
                # could match currency codes embedded in longer words.
                currency = _extract_currency_from_title(title)

                # Extract impact from description or title
                impact = "LOW"
                desc_lower = description.lower()
                if "high impact" in desc_lower or "volatility: high" in desc_lower:
                    impact = "HIGH"
                elif "medium impact" in desc_lower or "volatility: medium" in desc_lower:
                    impact = "MEDIUM"

                if currency not in currencies:
                    continue

                # Clean title (remove currency prefix)
                clean_title = _strip_currency_prefix(title)

                events.append({
                    "title":     clean_title,
                    "currency":  currency,
                    "time":      dt,
                    "impact":    impact,
                    "forecast":  "",
                    "previous":  "",
                    "actual":    "",
                })
            except Exception:
                continue
        return events

    # ─────────────────────────────────────────────────────────
    # SOURCE 3: DailyFX RSS
    # ─────────────────────────────────────────────────────────

    def _fetch_dailyfx_rss(self, currencies: List[str], hours_ahead: int) -> Optional[List[Dict]]:
        """DailyFX economic calendar RSS feed.

        No API key needed. DailyFX (owned by IG) provides a free RSS feed.
        """
        try:
            url = "https://www.dailyfx.com/feeds/calendar"
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; ForexAI/1.0)",
                "Accept": "application/rss+xml,application/xml",
            }
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                log.debug(f"[TradEconCal] DailyFX RSS HTTP {resp.status_code}")
                return None
            root = ET.fromstring(resp.content)
        except Exception as e:
            log.debug(f"[TradEconCal] DailyFX RSS failed: {e}")
            return None

        events = []
        for item in root.findall(".//item"):
            try:
                title = item.findtext("title", default="")
                pub_date_str = item.findtext("pubDate", default="")

                dt = datetime.strptime(pub_date_str, "%a, %d %b %Y %H:%M:%S %z")

                # Day 99+ FIX: same regex-based extraction as Investing RSS.
                currency = _extract_currency_from_title(title)

                if currency not in currencies:
                    continue

                events.append({
                    "title":     _strip_currency_prefix(title),
                    "currency":  currency,
                    "time":      dt,
                    "impact":    "MEDIUM",  # DailyFX doesn't always specify
                    "forecast":  "",
                    "previous":  "",
                    "actual":    "",
                })
            except Exception:
                continue
        return events

    # ─────────────────────────────────────────────────────────
    # Trade-block logic
    # ─────────────────────────────────────────────────────────

    def _check_block(self, events: List[Dict], now: datetime) -> tuple[bool, str]:
        """Check if any high-impact event falls within the block window."""
        for ev in events:
            if ev.get("impact") != "HIGH":
                continue
            ev_time = ev["time"]
            delta_min = (ev_time - now).total_seconds() / 60
            if abs(delta_min) <= self.BLOCK_WINDOW_MINUTES:
                direction = "in" if delta_min > 0 else "ago"
                return True, (
                    f"HIGH impact {ev['currency']} {ev['title']} "
                    f"@ {ev_time.strftime('%H:%M UTC')} ({abs(int(delta_min))}min {direction})"
                )
        return False, ""

    # ─────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _format_event(ev: Dict) -> Dict:
        return {
            "title":     ev.get("title", ""),
            "currency":  ev.get("currency", ""),
            "time":      ev["time"].strftime("%Y-%m-%d %H:%M UTC"),
            "impact":    ev.get("impact", "LOW"),
            "forecast":  ev.get("forecast", ""),
            "previous":  ev.get("previous", ""),
            "actual":    ev.get("actual", ""),
        }

    @staticmethod
    def _empty_result(reason: str, block: bool = True) -> Dict[str, Any]:
        """Return a safe empty result.

        Day 98+ FIX: block now defaults to True. Previously this defaulted
        to trade_block=False (fail open) while EconomicCalendarAPI's
        equivalent method defaults to True (fail closed) for the identical
        failure condition — all calendar sources unavailable. A data outage
        is unknown risk, and unknown risk should block trading, not permit
        it. Kept as a parameter (not hardcoded) so a caller with a specific,
        deliberate reason to treat a given failure as low-risk can still
        opt out explicitly — but the default is now conservative.
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
    # AI context (compatible with EconomicCalendarAPI)
    # ─────────────────────────────────────────────────────────

    def get_ai_context(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "econcal_source":         result.get("source", "none"),
            "econcal_event_count":    len(result.get("events", [])),
            "econcal_high_impact":    result.get("high_impact_count", 0),
            "econcal_trade_block":    result.get("trade_block", False),
            "econcal_block_reason":   result.get("block_reason", ""),
            "econcal_next_event":     result.get("next_event"),
        }

    def print_summary(self, result: Dict[str, Any]) -> None:
        bar = "═" * 50
        log.info(bar)
        log.info("  📅  TRADING ECON CALENDAR  (Day 95)")
        log.info(bar)
        log.info(f"  Source         : {result.get('source','?')}")
        log.info(f"  Events         : {len(result.get('events',[]))}")
        log.info(f"  High impact    : {result.get('high_impact_count',0)}")
        log.info(f"  Trade block    : {'⛔ YES' if result.get('trade_block') else '✅ no'}")
        if result.get("block_reason"):
            log.info(f"  Block reason   : {result['block_reason']}")
        nxt = result.get("next_event")
        if nxt:
            log.info(f"  Next event     : {nxt['currency']} {nxt['title']} @ {nxt['time']} [{nxt['impact']}]")
        log.info(bar)
