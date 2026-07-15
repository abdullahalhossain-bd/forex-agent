"""
intelligence/news_sources.py — Multi-source news aggregator
============================================================

Pulls news + economic events from multiple sources and unifies them into
a single schema. Sources (in priority order):

1. **Economic Calendar** — Forex Factory (HTML scrape, no key needed)
   + Investing.com fallback. Pulls scheduled HIGH-impact events
   (FOMC, NFP, CPI, etc.) for the next 24 hours.

2. **Central Bank Watch** — Pre-configured schedule for the 4 major
   central banks (Fed, ECB, BoE, BoJ). Detects scheduled announcements
   + speech timestamps.

3. **Financial RSS Feeds** — Reuters, Bloomberg, Forex Live, DailyFX.
   Pulled every cycle (lightweight — last 20 headlines per source).
   Fed to the sentiment_model.

4. **Local Economic Calendar JSON** — `data/economic_calendar.json`
   (maintained by broker/economic_calendar.py). Acts as a manual
   override / supplement.

Output schema (unified):
    {
        "source": "forex_factory" | "central_bank" | "rss" | "local",
        "event": "FOMC Meeting" | "ECB Speech" | "USD CPI" | ...,
        "currency": "USD" | "EUR" | "GBP" | "JPY" | "ALL",
        "impact": "HIGH" | "MEDIUM" | "LOW",
        "time_iso": "2026-06-22T18:00:00+00:00",
        "actual": None | "2.9%" | "0.25%" | ...,
        "forecast": None | "3.2%" | "0.50%" | ...,
        "previous": None | "3.4%" | "0.50%" | ...,
        "headline": "..." (RSS only),
        "url": "..." (RSS only),
    }
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup

from utils.logger import get_logger

log = get_logger("news_sources")

LOCAL_CALENDAR_PATH = Path("data/economic_calendar.json")

# ── Central bank schedule (recurring + known events) ────────────────
# This is a curated list of the most market-moving events. Times are GMT.
#
# KNOWN LIMITATION (flagged, not fixed here — see IMPLEMENTATION_LOG):
# the "day_of_week" + "week" fields are a fixed heuristic (e.g. "3rd
# Wednesday of the month" for FOMC). Real central-bank calendars shift
# around holidays and are set by each bank's own schedule, which this
# heuristic does not track — it will occasionally compute the wrong
# date. Treat CENTRAL_BANK_EVENTS as a fallback signal only. For dates
# that matter (news-block windows before a live trade), cross-check
# against economic_calendar_api.py / trading_economics_calendar.py,
# which pull real scheduled dates, before trusting this list alone.
# Replacing this with a dynamic source (API/CSV) is a larger change
# left as a pending architectural decision, not a targeted bug fix.
CENTRAL_BANK_EVENTS = [
    # Federal Reserve
    {"event": "FOMC Statement",                   "currency": "USD", "impact": "HIGH",
     "recurrence": "monthly", "day_of_week": "Wednesday", "week": 3},
    {"event": "FOMC Press Conference",            "currency": "USD", "impact": "HIGH",
     "recurrence": "monthly", "day_of_week": "Wednesday", "week": 3, "offset_hours": 0.5},
    {"event": "Fed Chair Powell Speech",          "currency": "USD", "impact": "HIGH",
     "recurrence": "ad_hoc"},
    # ECB
    {"event": "ECB Monetary Policy Statement",    "currency": "EUR", "impact": "HIGH",
     "recurrence": "monthly", "day_of_week": "Thursday", "week": 2},
    {"event": "ECB President Lagarde Speech",     "currency": "EUR", "impact": "HIGH",
     "recurrence": "ad_hoc"},
    # BoE
    {"event": "BoE Interest Rate Decision",       "currency": "GBP", "impact": "HIGH",
     "recurrence": "monthly", "day_of_week": "Thursday", "week": 1},
    {"event": "BoE Governor Speech",              "currency": "GBP", "impact": "HIGH",
     "recurrence": "ad_hoc"},
    # BoJ
    {"event": "BoJ Policy Statement",             "currency": "JPY", "impact": "HIGH",
     "recurrence": "monthly", "day_of_week": "Friday", "week": 2},
    {"event": "BoJ Governor Speech",              "currency": "JPY", "impact": "HIGH",
     "recurrence": "ad_hoc"},
]

# ── RSS feeds (financial news) ──────────────────────────────────────
# Day 81+ hotfix: Reuters feed (feeds.reuters.com) is permanently dead
# (Reuters discontinued public RSS years ago — DNS no longer resolves).
# Removed from the list to silence the noise. DailyFX, ForexLive, and
# Investing remain — they need `lxml` installed for BeautifulSoup to
# parse XML:  pip install lxml
RSS_FEEDS = [
    # DailyFX forex
    {"source": "dailyfx", "currency": "ALL",
     "url": "https://www.dailyfx.com/feeds/all"},
    # Forex Live
    {"source": "forexlive", "currency": "ALL",
     "url": "https://www.forexlive.com/feed/"},
    # Investing.com forex news
    {"source": "investing", "currency": "ALL",
     "url": "https://www.investing.com/rss/news_25.rss"},
    # MarketWatch — replacement for Reuters (top stories)
    {"source": "marketwatch", "currency": "ALL",
     "url": "https://feeds.content.dowjones.io/public/rss/mw_topstories"},
]

# ── Cache + freshness window ────────────────────────────────────────
_CACHE_TTL_SEC = 300  # 5 minutes
_cache_lock = threading.RLock()
_cache: Dict[str, Any] = {
    "calendar": None, "calendar_at": 0,
    "rss": None, "rss_at": 0,
    "central_bank": None, "central_bank_at": 0,
}


@dataclass
class NewsItem:
    """Unified news/event item."""
    source: str
    event: str
    currency: str
    impact: str
    time_iso: Optional[str] = None
    actual: Optional[str] = None
    forecast: Optional[str] = None
    previous: Optional[str] = None
    headline: Optional[str] = None
    url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class NewsSources:
    """Multi-source news aggregator with TTL caching."""

    def __init__(self, cache_ttl: int = _CACHE_TTL_SEC):
        self.cache_ttl = cache_ttl
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; ForexAI/1.0)"
        })

    # ── Economic Calendar (Forex Factory scrape) ────────────────────

    def fetch_economic_calendar(self, hours_ahead: int = 24) -> List[NewsItem]:
        """Fetch HIGH-impact events from Forex Factory for the next `hours_ahead` hours."""
        now = time.time()
        with _cache_lock:
            if _cache["calendar"] is not None and (now - _cache["calendar_at"]) < self.cache_ttl:
                return _cache["calendar"]

        items: List[NewsItem] = []
        try:
            # Forex Factory weekly calendar (JSON API)
            # This is a public endpoint that returns this week's events.
            today = datetime.now(timezone.utc)
            week = today.isocalendar()[1]
            url = f"https://nfs.faireconomy.media/ff_calendar_thisweek.json"
            resp = self.session.get(url, timeout=10)
            if resp.status_code == 200:
                events = resp.json()
                cutoff = today + timedelta(hours=hours_ahead)
                for ev in events:
                    impact = (ev.get("impact") or "").upper()
                    if impact not in ("HIGH", "MEDIUM"):
                        continue
                    try:
                        # FF returns "2026-06-22T18:00:00+00:00" format
                        ev_time = datetime.fromisoformat(ev.get("date", "").replace("Z", "+00:00"))
                        ev_time = ev_time if ev_time.tzinfo else ev_time.replace(tzinfo=timezone.utc)
                    except Exception:
                        continue
                    if today - timedelta(hours=2) <= ev_time <= cutoff:
                        items.append(NewsItem(
                            source="forex_factory",
                            event=ev.get("title", "Unknown"),
                            currency=(ev.get("country") or "ALL").upper(),
                            impact=impact,
                            time_iso=ev_time.isoformat(),
                            forecast=ev.get("forecast"),
                            previous=ev.get("previous"),
                        ))
            else:
                log.warning(f"[NewsSources] Forex Factory status {resp.status_code}")
        except Exception as e:
            log.warning(f"[NewsSources] Forex Factory fetch failed: {e}")

        # Always also pull from local calendar (broker/economic_calendar.py)
        items.extend(self._fetch_local_calendar(hours_ahead))

        # Deduplicate by (event, currency, time_iso)
        seen = set()
        deduped = []
        for it in items:
            key = (it.event, it.currency, it.time_iso)
            if key not in seen:
                seen.add(key)
                deduped.append(it)

        with _cache_lock:
            _cache["calendar"] = deduped
            _cache["calendar_at"] = now
        return deduped

    def _fetch_local_calendar(self, hours_ahead: int = 24) -> List[NewsItem]:
        """Pull events from data/economic_calendar.json (manual override)."""
        if not LOCAL_CALENDAR_PATH.exists():
            return []
        items: List[NewsItem] = []
        try:
            data = json.loads(LOCAL_CALENDAR_PATH.read_text(encoding="utf-8"))
            now = datetime.now(timezone.utc)
            cutoff = now + timedelta(hours=hours_ahead)
            for ev in data:
                if (ev.get("impact") or "").upper() != "HIGH":
                    continue
                try:
                    ev_time = datetime.fromisoformat(ev.get("time", "").replace("Z", "+00:00"))
                    if ev_time.tzinfo is None:
                        ev_time = ev_time.replace(tzinfo=timezone.utc)
                except Exception:
                    continue
                if now - timedelta(hours=2) <= ev_time <= cutoff:
                    items.append(NewsItem(
                        source="local",
                        event=ev.get("name", "Unknown"),
                        currency=(ev.get("currency") or "ALL").upper(),
                        impact="HIGH",
                        time_iso=ev_time.isoformat(),
                    ))
        except Exception as e:
            log.warning(f"[NewsSources] Local calendar read failed: {e}")
        return items

    # ── Central Bank events ─────────────────────────────────────────

    def fetch_central_bank_events(self, hours_ahead: int = 48) -> List[NewsItem]:
        """Return scheduled central bank events in the next `hours_ahead` hours.

        For ad_hoc events (speeches), we cannot predict timing — those are
        detected from RSS headlines instead. This method returns only the
        scheduled recurring events.
        """
        now = time.time()
        with _cache_lock:
            if _cache["central_bank"] is not None and (now - _cache["central_bank_at"]) < self.cache_ttl:
                return _cache["central_bank"]

        items: List[NewsItem] = []
        now_dt = datetime.now(timezone.utc)
        cutoff = now_dt + timedelta(hours=hours_ahead)

        for cb in CENTRAL_BANK_EVENTS:
            if cb.get("recurrence") != "monthly":
                continue  # ad_hoc — skip
            # Find this month's matching weekday + week number
            try:
                target_dow = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"].index(cb["day_of_week"])
                first_of_month = now_dt.replace(day=1, hour=18, minute=0, second=0, microsecond=0)  # default 18:00 UTC
                # Find first matching weekday
                offset = (target_dow - first_of_month.weekday()) % 7
                first_match = first_of_month + timedelta(days=offset)
                # Add (week-1) * 7 days
                event_date = first_match + timedelta(days=(cb["week"] - 1) * 7)
                if "offset_hours" in cb:
                    event_date = event_date + timedelta(hours=cb["offset_hours"])
                if now_dt - timedelta(hours=2) <= event_date <= cutoff:
                    items.append(NewsItem(
                        source="central_bank",
                        event=cb["event"],
                        currency=cb["currency"],
                        impact=cb["impact"],
                        time_iso=event_date.isoformat(),
                    ))
            except Exception as e:
                log.debug(f"[NewsSources] CB schedule parse failed for {cb['event']}: {e}")

        with _cache_lock:
            _cache["central_bank"] = items
            _cache["central_bank_at"] = now
        return items

    # ── RSS feeds ───────────────────────────────────────────────────

    def fetch_rss_feeds(self, max_per_feed: int = 15) -> List[NewsItem]:
        """Pull recent headlines from financial RSS feeds."""
        now = time.time()
        with _cache_lock:
            if _cache["rss"] is not None and (now - _cache["rss_at"]) < self.cache_ttl:
                return _cache["rss"]

        items: List[NewsItem] = []
        for feed in RSS_FEEDS:
            try:
                resp = self.session.get(feed["url"], timeout=8)
                if resp.status_code != 200:
                    continue
                # Day 81+ hotfix: prefer lxml for XML, fall back to html.parser
                # if lxml isn't installed. Without lxml, BeautifulSoup can't
                # parse RSS feeds (the "xml" parser is lxml-backed).
                try:
                    soup = BeautifulSoup(resp.content, "xml")
                except Exception:
                    # lxml not installed — try html.parser (works for most
                    # RSS feeds since they're loosely-XML)
                    soup = BeautifulSoup(resp.content, "html.parser")
                for entry in soup.find_all("item")[:max_per_feed]:
                    title = entry.find("title")
                    link = entry.find("link")
                    pub = entry.find("pubDate")
                    if not title:
                        continue
                    # Bug fix: previously this fell back to datetime.now(timezone.utc)
                    # whenever <pubDate> was missing OR failed to parse (many feeds use
                    # non-RFC822 date formats, so parsing failures are common). That
                    # made a stale headline look freshly-published on every single fetch
                    # cycle — since news_ai.py treats "an event whose time is very close
                    # to now" as breaking news, the SAME old headline (e.g. a CPI/Middle
                    # East commentary piece) kept re-qualifying as a brand-new
                    # "0 min ago" HIGH IMPACT event each cycle, spamming Telegram with
                    # duplicate alerts all day. An unknown publish time must stay
                    # unknown (time_iso=None) — the downstream code already knows how
                    # to safely skip items with no time_iso (see news_ai.py's Day 101
                    # hotfix) instead of pretending they're happening right now.
                    #
                    # Fix: only RFC-822 (with seconds+offset) was ever tried, so any
                    # feed using RFC-822 without seconds, a "GMT"/zone-name suffix
                    # instead of a numeric offset, or ISO-8601 (Atom-style feeds) fell
                    # through to None every time even though the timestamp was present
                    # and parseable. Try a small set of formats covering the common
                    # real-world variants before giving up.
                    pub_dt = None
                    if pub:
                        raw = pub.text.strip()
                        for fmt in (
                            "%a, %d %b %Y %H:%M:%S %z",    # RFC-822 w/ numeric offset
                            "%a, %d %b %Y %H:%M:%S %Z",    # RFC-822 w/ zone name (GMT/UTC)
                            "%a, %d %b %Y %H:%M %z",       # RFC-822, no seconds
                            "%Y-%m-%dT%H:%M:%S%z",         # ISO-8601 (Atom feeds)
                        ):
                            try:
                                pub_dt = datetime.strptime(raw, fmt)
                                break
                            except Exception:
                                continue
                        if pub_dt is None:
                            try:
                                # ISO-8601 with 'Z' suffix — strptime %z doesn't
                                # accept literal 'Z', so normalize it first.
                                pub_dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                            except Exception:
                                pub_dt = None
                    items.append(NewsItem(
                        source=feed["source"],
                        event=title.text.strip()[:200],
                        currency=feed["currency"],
                        impact="MEDIUM",  # default; sentiment_model will refine
                        time_iso=pub_dt.isoformat() if pub_dt else None,
                        headline=title.text.strip()[:200],
                        url=link.text.strip() if link else None,
                    ))
            except Exception as e:
                log.debug(f"[NewsSources] RSS {feed['source']} failed: {e}")

        with _cache_lock:
            _cache["rss"] = items
            _cache["rss_at"] = now
        return items

    # ── Aggregate all sources ───────────────────────────────────────

    def fetch_all(self, hours_ahead: int = 24) -> Dict[str, List[NewsItem]]:
        """Fetch from all sources and return a categorized dict."""
        return {
            "calendar": self.fetch_economic_calendar(hours_ahead=hours_ahead),
            "central_bank": self.fetch_central_bank_events(hours_ahead=hours_ahead),
            "rss": self.fetch_rss_feeds(),
        }

    def fetch_all_flat(self, hours_ahead: int = 24) -> List[NewsItem]:
        """Flat list of all news items, sorted by time."""
        all_items = []
        all_items.extend(self.fetch_economic_calendar(hours_ahead=hours_ahead))
        all_items.extend(self.fetch_central_bank_events(hours_ahead=hours_ahead))
        all_items.extend(self.fetch_rss_feeds())
        # Sort by time (None last)
        all_items.sort(key=lambda x: x.time_iso or "9999")
        return all_items