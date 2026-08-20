"""
fundamental/economic_calendar_api.py — Institutional Economic Calendar
==============================================================================
Multi-source economic calendar with fallback chain:

    FairEconomy JSON (primary, disk-cached) → Forex Factory scraper → outage

Fetch chain:
    Layer 0: FairEconomy JSON   — primary; local disk cache survives 429 rate-limits
    Layer 1: FF scraper         — existing cloudscraper / hardcoded path
    (No hardcoded schedule as primary — dates move; outage path is last resort)

Output shape (stable contract):
    {
      "source":            "faireconomy_json" | "ff_scraper" | "hardcoded_fallback" | "none",
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

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from utils.logger import get_logger

log = get_logger("economic_calendar_api")

# FairEconomy JSON URL (official FF weekly feed, no key)
FAIRECONOMY_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

# Disk cache — survives process restarts and Cloudflare 429 windows
_CACHE_DIR = Path(os.getenv("ECONCAL_CACHE_DIR", os.path.join("data", "cache")))
_CACHE_FILE = _CACHE_DIR / "faireconomy_thisweek.json"
_CACHE_META = _CACHE_DIR / "faireconomy_thisweek.meta.json"
# Serve stale cache up to this age when live fetch is rate-limited / fails
_CACHE_MAX_AGE_SEC = int(os.getenv("ECONCAL_CACHE_MAX_AGE_SEC", str(6 * 3600)))  # 6 h
_CACHE_STALE_OK_SEC = int(os.getenv("ECONCAL_CACHE_STALE_OK_SEC", str(48 * 3600)))  # 48 h hard ceiling

# Default watched set — includes commodity / AUDCAD-relevant currencies
DEFAULT_CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "NZD", "CHF"]

# Fallback HIGH-impact keyword list (used when faireconomy_cache is unavailable)
_FALLBACK_HIGH_IMPACT_KEYWORDS = (
    "interest rate", "rate decision", "nfp", "non-farm", "nonfarm",
    "cpi", "inflation", "gdp", "fomc", "ecb", "boe", "rba", "boc",
    "employment", "unemployment", "retail sales", "pmi", "powell",
    "jackson hole", "cash rate",
)

try:
    from fundamental.faireconomy_cache import DEFAULT_HIGH_IMPACT_KEYWORDS
    HIGH_IMPACT_KEYWORDS = DEFAULT_HIGH_IMPACT_KEYWORDS
except Exception:
    HIGH_IMPACT_KEYWORDS = _FALLBACK_HIGH_IMPACT_KEYWORDS


class EconomicCalendarAPI:
    """Multi-source economic calendar with automatic fallback and correct
    ±BLOCK_WINDOW trade blocking (including the post-release window).
    """

    BLOCK_WINDOW_MINUTES = 30  # block trades ±30 min around high-impact events

    def __init__(self) -> None:
        # Paid sources (TraderMade / Finnhub) removed by design — no keys in use.
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
        })

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
                         None → DEFAULT_CURRENCIES (includes AUD/CAD).
            hours_ahead: look this many hours forward from now (for the public
                         events list). Block logic independently examines the
                         past BLOCK_WINDOW_MINUTES as well.

        Returns:
            Dict with the stable contract documented at module top.
        """
        if currencies is None:
            currencies = list(DEFAULT_CURRENCIES)

        # Back-test path: no live calendar exists for a historical bar.
        try:
            from core.constants import is_backtest_mode
            if is_backtest_mode():
                return self._empty_result(
                    "backtest mode — live calendar skipped",
                    block=False,
                    calendar_outage=False,
                )
        except Exception:
            pass

        events: Optional[List[Dict]] = None
        source = "none"

        # ── Layer 0: FairEconomy JSON (shared cache → disk cache → live) ──
        events = self._fetch_faireconomy(currencies)
        if events:
            source = "faireconomy_json"

        # ── Layer 1: Forex Factory scraper / hardcoded ──
        if not events:
            try:
                from fundamental.news_filter import NewsFilter
                nf = NewsFilter()
                ff_events, ff_source = nf._fetch_events()
                log.debug(
                    f"[EconCal] FF layer: source={ff_source} "
                    f"raw_events={len(ff_events) if ff_events else 0}"
                )
                if ff_events:
                    events = self._normalize_ff_events(ff_events, currencies)
                    if events:
                        source = ff_source or "ff_scraper"
            except Exception as e:
                log.warning(f"[EconCal] FF scraper fallback failed: {e}")

        # ── All layers failed ──
        if not events:
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

        # Helpful diagnostic when source has data but nothing in the look-ahead window
        if not upcoming and events:
            future = [
                e for e in events
                if isinstance(e.get("time"), datetime) and e["time"] > now
            ]
            future.sort(key=lambda e: e["time"])
            if future:
                nxt = future[0]
                hours_to = (nxt["time"] - now).total_seconds() / 3600.0
                log.info(
                    f"[EconCal] source={source} has {len(events)} events but "
                    f"0 in next {hours_ahead}h — next is "
                    f"{nxt.get('currency')} {nxt.get('title')} in {hours_to:.1f}h"
                )
            else:
                log.info(
                    f"[EconCal] source={source} has {len(events)} events, "
                    f"none remaining in the future (all past)"
                )

        log.info(
            f"[EconCal] source={source} | raw={len(events)} | "
            f"upcoming({hours_ahead}h)={len(upcoming)} | high_impact={len(high_impact)} | "
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
    # SOURCE 0: FairEconomy JSON (shared cache → disk → live)
    # ─────────────────────────────────────────────────────────

    def _fetch_faireconomy(self, currencies: List[str]) -> Optional[List[Dict]]:
        """Try shared cache module first; on failure use local disk+live fetch."""
        # 1) Prefer project shared cache (stampede-safe) when available
        try:
            from fundamental.faireconomy_cache import fetch_faireconomy as _cached_fetch

            raw_events = _cached_fetch(
                watched_currencies=set(currencies),
                high_impact_keywords=HIGH_IMPACT_KEYWORDS,
            )
            if raw_events:
                return self._normalize_raw_faireconomy(raw_events)
        except Exception as e:
            log.debug(f"[FairEconomy] shared cache unavailable: {e}")

        # 2) Local disk cache + live HTTP (handles Cloudflare 429)
        raw = self._load_disk_cache()
        live = self._http_fetch_faireconomy()
        if live is not None:
            self._save_disk_cache(live)
            raw = live
        elif raw is None:
            log.warning("[FairEconomy] no live data and no usable disk cache")
            return None
        else:
            log.info(
                f"[FairEconomy] serving disk cache "
                f"({len(raw)} raw events) — live fetch rate-limited or failed"
            )

        # Filter by watched currencies + normalize
        return self._normalize_raw_faireconomy(
            [e for e in raw if (e.get("country") or e.get("currency") or "") in currencies
             or not currencies]
            if raw else []
        ) or self._normalize_raw_faireconomy(raw or [])

    def _http_fetch_faireconomy(self) -> Optional[List[Dict]]:
        """Live GET with Retry-After respect. Returns raw JSON list or None."""
        # Honour a previous Retry-After if still active
        retry_until = self._read_retry_until()
        if retry_until and time.time() < retry_until:
            remaining = int(retry_until - time.time())
            log.debug(f"[FairEconomy] skipping live fetch — Retry-After {remaining}s left")
            return None

        try:
            resp = self._session.get(FAIRECONOMY_URL, timeout=15)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", "300"))
                self._write_retry_until(time.time() + retry_after)
                log.warning(
                    f"[FairEconomy] HTTP 429 rate-limited — "
                    f"Retry-After={retry_after}s; will use disk cache if available"
                )
                return None
            if resp.status_code != 200:
                log.warning(f"[FairEconomy] HTTP {resp.status_code}")
                return None
            data = resp.json()
            if not isinstance(data, list) or not data:
                log.warning("[FairEconomy] empty or non-list JSON body")
                return None
            log.info(f"[FairEconomy] live fetch OK — {len(data)} events this week")
            return data
        except Exception as e:
            log.warning(f"[FairEconomy] live fetch failed: {e}")
            return None

    def _normalize_raw_faireconomy(self, raw_events: List[Dict]) -> Optional[List[Dict]]:
        """Convert FairEconomy / cache items to the common schema."""
        if not raw_events:
            return None
        events: List[Dict] = []
        for item in raw_events:
            try:
                # Time parsing — FairEconomy uses "date": "08-19-2026 3:30pm"
                t = item.get("time")
                if not isinstance(t, datetime):
                    t = self._parse_faireconomy_date(
                        item.get("date") or item.get("datetime") or item.get("time")
                    )
                if t is None:
                    continue

                currency = (
                    item.get("currency")
                    or item.get("country")
                    or ""
                ).strip().upper()

                raw_impact = (item.get("impact", "") or "").strip().upper()
                if raw_impact in ("HIGH", "MEDIUM", "LOW"):
                    impact = raw_impact
                elif raw_impact in ("RED", "3"):
                    impact = "HIGH"
                elif raw_impact in ("ORANGE", "2", "MED"):
                    impact = "MEDIUM"
                elif item.get("high_impact"):
                    impact = "HIGH"
                else:
                    # Keyword boost for title
                    title_l = (item.get("title") or item.get("name") or "").lower()
                    impact = "HIGH" if any(k in title_l for k in HIGH_IMPACT_KEYWORDS) else "LOW"

                events.append({
                    "title":    item.get("title") or item.get("name") or "",
                    "currency": currency,
                    "time":     t,
                    "impact":   impact,
                    "forecast": str(item.get("forecast") or item.get("consensus") or ""),
                    "previous": str(item.get("previous") or ""),
                    "actual":   str(item.get("actual") or ""),
                })
            except Exception:
                continue

        if events:
            log.info(f"[FairEconomy] normalized {len(events)} events")
        return events or None

    @staticmethod
    def _parse_faireconomy_date(raw: Any) -> Optional[datetime]:
        """Parse FairEconomy date strings into timezone-aware UTC datetime."""
        if raw is None:
            return None
        if isinstance(raw, datetime):
            return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
        s = str(raw).strip()
        if not s:
            return None
        # Common FF formats: "08-19-2026 3:30pm", "2026-08-19T15:30:00"
        for fmt in (
            "%m-%d-%Y %I:%M%p",
            "%m-%d-%Y %I:%M %p",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%d-%m-%Y %H:%M",
        ):
            try:
                dt = datetime.strptime(s.replace("  ", " "), fmt)
                if dt.tzinfo is None:
                    # FairEconomy times are Eastern (US) — convert to UTC
                    # EST=UTC-5, EDT=UTC-4. Use fixed -4 as safe summer default;
                    # override with ECONCAL_FF_TZ_OFFSET_HOURS if needed.
                    offset_h = float(os.getenv("ECONCAL_FF_TZ_OFFSET_HOURS", "-4"))
                    dt = dt.replace(tzinfo=timezone(timedelta(hours=offset_h)))
                return dt.astimezone(timezone.utc)
            except ValueError:
                continue
        return None

    # ── disk cache helpers ───────────────────────────────────

    def _load_disk_cache(self) -> Optional[List[Dict]]:
        try:
            if not _CACHE_FILE.exists() or not _CACHE_META.exists():
                return None
            meta = json.loads(_CACHE_META.read_text(encoding="utf-8"))
            saved_at = float(meta.get("saved_at", 0))
            age = time.time() - saved_at
            if age > _CACHE_STALE_OK_SEC:
                log.debug(f"[FairEconomy] disk cache too old ({age/3600:.1f}h) — discarding")
                return None
            data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
            if not isinstance(data, list) or not data:
                return None
            if age > _CACHE_MAX_AGE_SEC:
                log.info(
                    f"[FairEconomy] disk cache soft-stale ({age/3600:.1f}h) — "
                    f"usable while rate-limited"
                )
            return data
        except Exception as e:
            log.debug(f"[FairEconomy] disk cache read failed: {e}")
            return None

    def _save_disk_cache(self, data: List[Dict]) -> None:
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            _CACHE_FILE.write_text(json.dumps(data, default=str), encoding="utf-8")
            _CACHE_META.write_text(
                json.dumps({"saved_at": time.time(), "count": len(data)}),
                encoding="utf-8",
            )
        except Exception as e:
            log.debug(f"[FairEconomy] disk cache write failed: {e}")

    def _read_retry_until(self) -> float:
        try:
            p = _CACHE_DIR / "faireconomy_retry_until.txt"
            if p.exists():
                return float(p.read_text().strip())
        except Exception:
            pass
        return 0.0

    def _write_retry_until(self, ts: float) -> None:
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            (_CACHE_DIR / "faireconomy_retry_until.txt").write_text(str(ts))
        except Exception:
            pass

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
                    "time":     t if t.tzinfo else t.replace(tzinfo=timezone.utc),
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
