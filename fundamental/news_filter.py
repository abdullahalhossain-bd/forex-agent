# fundamental/news_filter.py  —  Day 11 (base) + Day 43 + Day 90 + Day 91 + Day 92
#
# Day 92 fix: নিজস্ব _FAIRECONOMY_CACHE সরিয়ে shared
# fundamental.faireconomy_cache.fetch_faireconomy() ব্যবহার করা হচ্ছে।
# এতে economic_calendar_api এবং news_filter দুটোই একই cache থেকে পাবে।

import os
import time
from datetime import datetime, timedelta
from typing import Optional

import pytz
import requests
from bs4 import BeautifulSoup

from utils.logger import get_logger
from fundamental.faireconomy_cache import fetch_faireconomy, DEFAULT_HIGH_IMPACT_KEYWORDS

log = get_logger("news_filter")

try:
    import cloudscraper
    _CLOUDSCRAPER_AVAILABLE = True
except ImportError:
    _CLOUDSCRAPER_AVAILABLE = False

# PERFORMANCE FIX (execution-parity audit follow-up, 2026-07-19): Layer 0
# (fetch_faireconomy) already has proper cooldown/backoff on failure — see
# faireconomy_cache.py's _blocked_until. But Layer 0 returns [] both when
# it genuinely found zero high-impact events AND when it's in cooldown with
# no cached data — _fetch_events() can't tell those apart, so an empty
# result always falls through to Layer 1 (cloudscraper) and Layer 2 (plain
# requests) against forexfactory.com/calendar, with NO cooldown of their
# own. When that site is blocking the scraper (403, as in this backtest's
# sandbox), every single call — i.e. every bar in a backtest — pays the
# full cloudscraper handshake + retry cost again, forever. This module-
# level cooldown (persists across NewsFilter() instances, since a fresh
# instance is constructed per call — see agents/analysis_agent.py) skips
# Layers 1/2 for FF_SCRAPE_COOLDOWN_SECONDS after a failure, going straight
# to the Layer 3 hardcoded fallback instead. No change to what data is
# used on failure (same hardcoded fallback as before) — only how often the
# doomed network call is retried.
_ff_scrape_blocked_until = 0.0
FF_SCRAPE_COOLDOWN_SECONDS = 300.0  # 5 minutes

VOLATILITY_MAP = {
    "non-farm":        {"level": "EXTREME", "pips": (80, 150)},
    "nfp":             {"level": "EXTREME", "pips": (80, 150)},
    "interest rate":   {"level": "EXTREME", "pips": (70, 130)},
    "fomc":            {"level": "EXTREME", "pips": (70, 130)},
    "fed chair":       {"level": "HIGH",    "pips": (40, 90)},
    "cpi":             {"level": "HIGH",    "pips": (50, 100)},
    "inflation":       {"level": "HIGH",    "pips": (40, 90)},
    "unemployment":    {"level": "HIGH",    "pips": (40, 80)},
    "ecb":             {"level": "HIGH",    "pips": (40, 90)},
    "boe":             {"level": "HIGH",    "pips": (35, 80)},
    "boj":             {"level": "HIGH",    "pips": (35, 80)},
    "gdp":             {"level": "MEDIUM",  "pips": (30, 60)},
    "retail sales":    {"level": "MEDIUM",  "pips": (25, 50)},
    "pmi":             {"level": "MEDIUM",  "pips": (20, 45)},
}
DEFAULT_VOLATILITY = {"level": "LOW", "pips": (5, 20)}

CURRENCY_PAIR_MAP = {
    "USD": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "XAUUSD"],
    "EUR": ["EURUSD", "EURGBP", "EURJPY", "EURAUD", "EURCAD", "EURCHF", "EURNZD"],
    "GBP": ["GBPUSD", "EURGBP", "GBPJPY", "GBPAUD", "GBPCAD", "GBPCHF", "GBPNZD"],
    "JPY": ["USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY", "NZDJPY"],
}

AFTERMATH_WAIT_MINUTES = 15

HARDCODED_WEEKLY_EVENTS = [
    (2, 12, 30, "USD", "CPI (typical release day/time — verify live calendar)"),
    (4, 12, 30, "USD", "Non-Farm Payrolls (typical 1st-Friday release — verify live calendar)"),
    (2, 18, 0,  "USD", "FOMC Statement (typical — verify live calendar)"),
    (3, 11, 45, "EUR", "ECB Rate Decision (typical — verify live calendar)"),
    (3, 6, 0,   "GBP", "BOE Rate Decision (typical — verify live calendar)"),
]


class NewsFilter:
    WINDOW_BEFORE = 30
    WINDOW_AFTER  = 30   # was 60 — long post-event blocks wasted valid setups
    WINDOW_AFTER_BY_VOLATILITY = {
        "EXTREME": 45,
        "HIGH":    30,
        "MEDIUM":  20,
        "LOW":     15,
    }
    WATCHED_CURRENCIES = {"USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"}
    # Day 98+ FIX: now points at the single shared list in faireconomy_cache.py
    # instead of maintaining its own copy (was identical to
    # economic_calendar_api.py's copy by coincidence, not by design).
    HIGH_IMPACT_KEYWORDS = DEFAULT_HIGH_IMPACT_KEYWORDS
    FF_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.forexfactory.com/",
    }
    FETCH_MAX_RETRIES   = 1
    FETCH_RETRY_DELAY_S = 2.0

    def __init__(self) -> None:
        self._scraper = None

    def check(self, symbol: str = "EURUSD") -> dict:
        if self._news_block_disabled():
            log.info("News block disabled by configuration — allowing trade to proceed")
            return {
                "trade_allowed":      True,
                "reason":             "News block disabled by configuration",
                "flagged_events":     [],
                "upcoming_events":    [],
                "currencies_checked": [self._extract_currencies(symbol)],
                "risk_level":         "LOW",
                "aftermath":          {"in_confirmation_window": False, "advice": ""},
                "source":             "disabled",
            }

        currencies = self._extract_currencies(symbol)
        log.info(f"Checking news for: {currencies}")

        events, source = self._fetch_events()

        if not events:
            log.warning("Could not fetch any news data — blocking trades (unknown news risk)")
            return self._safe_result("News fetch failed — trading blocked until data source recovers")

        now_utc = datetime.now(pytz.utc)
        flagged = []

        for event in events:
            if event["currency"] not in currencies:
                continue
            if not event["high_impact"]:
                continue

            event_time   = event["time"]
            vol          = self.estimate_volatility(event["title"])
            after_mins   = self.WINDOW_AFTER_BY_VOLATILITY.get(
                vol["level"], self.WINDOW_AFTER
            )
            window_start = event_time - timedelta(minutes=self.WINDOW_BEFORE)
            window_end   = event_time + timedelta(minutes=after_mins)

            if window_start <= now_utc <= window_end:
                mins_to = int((event_time - now_utc).total_seconds() / 60)
                flagged.append({
                    "event":      event["title"],
                    "currency":   event["currency"],
                    "time":       event_time.strftime("%H:%M UTC"),
                    "mins_to":    mins_to,
                    "volatility": vol,
                })

        if flagged:
            ev        = flagged[0]
            aftermath = self.post_news_status(self._event_time_from_label(ev, now_utc))
            reason = (
                f"{ev['currency']} {ev['event']} @ {ev['time']} "
                f"({abs(ev['mins_to'])} min {'until' if ev['mins_to'] > 0 else 'ago'}) "
                f"— expected volatility: {ev['volatility']['level']}"
            )
            return {
                "trade_allowed":      False,
                "reason":             reason,
                "flagged_events":     flagged,
                "currencies_checked": list(currencies),
                "risk_level":         self._max_risk_level(flagged),
                "aftermath":          aftermath,
                "source":             source,
            }

        upcoming_raw = [
            e for e in events
            if e["currency"] in currencies
            and e["high_impact"]
            and e["time"] > now_utc
            and (e["time"] - now_utc).total_seconds() < 3 * 3600
        ]
        upcoming = [
            {
                "event":      e["title"],
                "currency":   e["currency"],
                "time":       e["time"].strftime("%H:%M UTC"),
                "volatility": self.estimate_volatility(e["title"]),
            }
            for e in upcoming_raw[:3]
        ]

        return {
            "trade_allowed":      True,
            "reason":             "No high impact news in window",
            "flagged_events":     [],
            "upcoming_events":    upcoming,
            "currencies_checked": list(currencies),
            "risk_level":         self._max_risk_level(upcoming) if upcoming else "LOW",
            "aftermath":          {"in_confirmation_window": False, "advice": ""},
            "source":             source,
        }

    def estimate_volatility(self, title: str) -> dict:
        title_lower = (title or "").lower()
        for keyword, info in VOLATILITY_MAP.items():
            if keyword in title_lower:
                return dict(info)
        return dict(DEFAULT_VOLATILITY)

    def affected_pairs(self, currency: str) -> list:
        return CURRENCY_PAIR_MAP.get(currency.upper(), [])

    def post_news_status(self, event_time: Optional[datetime]) -> dict:
        if event_time is None:
            return {"in_confirmation_window": False, "advice": ""}
        now_utc     = datetime.now(pytz.utc)
        elapsed_min = (now_utc - event_time).total_seconds() / 60
        if 0 <= elapsed_min < AFTERMATH_WAIT_MINUTES:
            remaining = round(AFTERMATH_WAIT_MINUTES - elapsed_min)
            return {
                "in_confirmation_window": True,
                "minutes_remaining":      remaining,
                "advice": (
                    f"News released {round(elapsed_min)} min ago — first move "
                    f"often fakes out (liquidity grab). Wait {remaining} more "
                    f"min and confirm direction before entering."
                ),
            }
        return {"in_confirmation_window": False, "advice": ""}

    def _event_time_from_label(self, flagged_event: dict, now_utc: datetime) -> datetime:
        # Day 100+ FIX: mins_to is computed as
        #   int((event_time - now_utc).total_seconds() / 60)
        # so it is already signed: negative when the event is in the past,
        # positive when it's still upcoming. The old code special-cased
        # mins_to <= 0 and did `now_utc - timedelta(minutes=mins_to)`,
        # which for a negative mins_to (e.g. -20) computed
        # `now_utc - (-20min)` = `now_utc + 20min` — flipping a past event
        # into the future. That fed post_news_status() a future
        # event_time, made elapsed_min negative, and silently disabled the
        # post-news "confirmation window" advice exactly when it mattered
        # most (right after a high-impact release).
        #
        # Adding the signed mins_to directly handles both past and future
        # cases correctly in one line.
        return now_utc + timedelta(minutes=flagged_event.get("mins_to", 0))

    def _max_risk_level(self, events: list) -> str:
        order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "EXTREME": 3}
        best  = "LOW"
        for e in events:
            lvl = e.get("volatility", {}).get("level", "LOW")
            if order.get(lvl, 0) > order.get(best, 0):
                best = lvl
        return best

    def get_weekly_calendar(self, events: list | None = None) -> dict:
        events = events if events is not None else self._fetch_events()[0]
        by_day: dict[str, list] = {}
        for e in events:
            if not e.get("high_impact"):
                continue
            if e["currency"] not in self.WATCHED_CURRENCIES:
                continue
            day_key = e["time"].strftime("%Y-%m-%d")
            by_day.setdefault(day_key, []).append({
                "time":       e["time"].strftime("%H:%M UTC"),
                "currency":   e["currency"],
                "event":      e["title"],
                "volatility": self.estimate_volatility(e["title"]),
            })
        for day_key in by_day:
            by_day[day_key].sort(key=lambda x: x["time"])
        return dict(sorted(by_day.items()))

    def print_weekly_calendar(self, calendar: dict | None = None) -> None:
        calendar = calendar if calendar is not None else self.get_weekly_calendar()
        bar = "═" * 48
        log.info(bar)
        log.info("  📅  WEEKLY ECONOMIC CALENDAR  (Day 43)")
        log.info(bar)
        if not calendar:
            log.info("  No high-impact events found this week.")
        for day, events in calendar.items():
            log.info(f"  {day}")
            for e in events:
                tag = "⚠️ " if e["volatility"]["level"] in ("HIGH", "EXTREME") else "  "
                log.info(f"    {tag}{e['time']}  {e['currency']}  {e['event']}  [{e['volatility']['level']}]")
        log.info(bar)

    # ── Fetch chain ────────────────────────────────────────────
    def _fetch_events(self) -> tuple[list, str]:
        global _ff_scrape_blocked_until

        # ── Layer 0: shared FairEconomy cache (Day 92) ──
        events = fetch_faireconomy(self.WATCHED_CURRENCIES, self.HIGH_IMPACT_KEYWORDS)
        if events:
            return events, "faireconomy_json"

        # Cooldown check (see module-level comment above): if the FF scrape
        # layers failed recently, skip straight to the hardcoded fallback
        # instead of re-attempting a doomed cloudscraper/requests call.
        now_mono = time.monotonic()
        if now_mono < _ff_scrape_blocked_until:
            log.debug(
                f"[NewsFilter] FF scrape in cooldown "
                f"({_ff_scrape_blocked_until - now_mono:.0f}s remaining) — "
                f"skipping to hardcoded fallback"
            )
            events = self._build_hardcoded_events()
            return events, "hardcoded_fallback"

        # ── Layer 1: cloudscraper ──
        if _CLOUDSCRAPER_AVAILABLE:
            html = self._fetch_html_cloudscraper()
            if html:
                events = self._parse_ff(html)
                if events:
                    return events, "forexfactory_cloudscraper"

        # ── Layer 2: plain requests (1 retry) ──
        html = self._fetch_html_requests_with_retry()
        if html:
            events = self._parse_ff(html)
            if events:
                return events, "forexfactory_requests"

        # Both scrape layers failed — start the cooldown so subsequent
        # calls in this run (e.g. the next bar in a backtest) skip them.
        _ff_scrape_blocked_until = now_mono + FF_SCRAPE_COOLDOWN_SECONDS

        # ── Layer 3: hardcoded fallback ──
        log.warning("Live Forex Factory fetch fully failed → hardcoded fallback")
        fallback_events = self._build_hardcoded_events()
        if fallback_events:
            return fallback_events, "hardcoded_fallback"

        return [], "none"

    def _fetch_html_cloudscraper(self) -> str | None:
        try:
            if self._scraper is None:
                self._scraper = cloudscraper.create_scraper(
                    browser={"browser": "chrome", "platform": "windows", "mobile": False}
                )
            resp = self._scraper.get(
                "https://www.forexfactory.com/calendar",
                headers=self.FF_HEADERS, timeout=15,
            )
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            log.warning(f"cloudscraper fetch failed: {e}")
            return None

    def _fetch_html_requests_with_retry(self) -> str | None:
        url      = "https://www.forexfactory.com/calendar"
        last_err = None
        for attempt in range(1, self.FETCH_MAX_RETRIES + 1):
            try:
                resp = requests.get(url, headers=self.FF_HEADERS, timeout=10)
                resp.raise_for_status()
                return resp.text
            except Exception as e:
                last_err = e
                log.warning(f"Forex Factory fetch failed (attempt {attempt}/{self.FETCH_MAX_RETRIES}): {e}")
                if attempt < self.FETCH_MAX_RETRIES:
                    time.sleep(self.FETCH_RETRY_DELAY_S * attempt)
        log.warning(f"Forex Factory fetch exhausted retries — last error: {last_err}")
        return None

    def _build_hardcoded_events(self) -> list:
        # Day 99+ FIX: previously this computed `days_ahead` using
        # `(weekday - now_utc.weekday()) % 7`, which is correct for the
        # date portion but combined the computed date with a UTC
        # hour/minute and a UTC "now" — meaning a process running at
        # 23:55 UTC on a Tuesday could see `now.weekday() == 1` (Tuesday)
        # while the next event at 00:30 UTC was effectively on Wednesday
        # (weekday 2). The arithmetic still worked, but the comparison
        # against `now_utc` later in the filter could flag or skip
        # events incorrectly near the UTC midnight boundary.
        #
        # The fix below computes the event datetime in a single
        # timezone-aware step: it picks the soonest upcoming occurrence
        # of (weekday, hour, minute) strictly after `now_utc`, then also
        # adds the previous-week occurrence so events that just happened
        # are still visible in the post-event block window. This is
        # robust to running at any time of day, including across UTC
        # midnight.
        now_utc = datetime.now(pytz.utc)
        events = []
        for weekday, hour, minute, currency, title in HARDCODED_WEEKLY_EVENTS:
            # Find the next occurrence of this weekday at the given UTC time
            # that is strictly in the future. If today is the target weekday
            # but the time has already passed, we pick next week.
            days_ahead = (weekday - now_utc.weekday()) % 7
            candidate = now_utc + timedelta(days=days_ahead)
            candidate = candidate.replace(
                hour=hour, minute=minute, second=0, microsecond=0,
            )
            if candidate <= now_utc:
                # Target time already passed today; roll forward 7 days.
                candidate += timedelta(days=7)

            events.append({
                "title":       title,
                "currency":    currency,
                "high_impact": True,
                "time":        candidate,
            })
            # Also include the previous-week occurrence so post-event
            # block windows still trigger for events that just happened.
            events.append({
                "title":       title,
                "currency":    currency,
                "high_impact": True,
                "time":        candidate - timedelta(days=7),
            })
        return events

    def _parse_ff(self, html: str) -> list:
        soup   = BeautifulSoup(html, "html.parser")
        events = []
        now    = datetime.now(pytz.utc)
        rows   = soup.select("tr.calendar__row")
        current_date = now.date()
        current_time = None

        for row in rows:
            try:
                date_cell = row.select_one(".calendar__date span")
                if date_cell and date_cell.text.strip():
                    try:
                        parsed = datetime.strptime(
                            f"{date_cell.text.strip()} {now.year}", "%a %b %d %Y"
                        )
                        current_date = parsed.date()
                    except ValueError:
                        pass

                time_cell = row.select_one(".calendar__time")
                if time_cell and time_cell.text.strip():
                    t_text = time_cell.text.strip()
                    if ":" in t_text:
                        try:
                            current_time = datetime.strptime(t_text, "%I:%M%p").time()
                        except ValueError:
                            pass

                if current_time is None:
                    continue

                cur_cell = row.select_one(".calendar__currency")
                if not cur_cell:
                    continue
                currency = cur_cell.text.strip().upper()
                if currency not in self.WATCHED_CURRENCIES:
                    continue

                impact_cell = row.select_one(".calendar__impact span")
                impact_cls  = impact_cell.get("class", []) if impact_cell else []
                is_high     = any("red" in c or "high" in c for c in impact_cls)

                title_cell = row.select_one(".calendar__event-title")
                title      = title_cell.text.strip() if title_cell else ""

                if not is_high:
                    is_high = any(kw.lower() in title.lower() for kw in self.HIGH_IMPACT_KEYWORDS)

                naive_dt = datetime.combine(current_date, current_time)
                eastern  = pytz.timezone("US/Eastern")
                utc_dt   = eastern.localize(naive_dt).astimezone(pytz.utc)

                events.append({"title": title, "currency": currency, "high_impact": is_high, "time": utc_dt})
            except Exception:
                continue

        log.info(f"Parsed {len(events)} events from Forex Factory")
        return events

    def _extract_currencies(self, symbol: str) -> set:
        symbol = symbol.upper().replace("/", "").replace("=X", "")
        if len(symbol) >= 6:
            return {symbol[:3], symbol[3:6]}
        return {"USD"}

    def _news_block_disabled(self) -> bool:
        try:
            return str(os.getenv("NEWS_BLOCK_ENABLED", "false")).strip().lower() in {"1", "true", "yes", "on"}
        except Exception:
            return False

    def _safe_result(self, reason: str) -> dict:
        # Day 98+ FIX: fail CLOSED, not open. Previously this returned
        # trade_allowed=True on a total data-fetch failure ("proceed with
        # caution") while EconomicCalendarAPI's equivalent failure path
        # (_empty_result) already defaulted to trade_block=True. Two modules
        # serving the same purpose disagreeing on fail-safe behavior meant
        # capital protection depended on which one the Decision Layer
        # happened to call. A calendar/news outage is unknown risk — unknown
        # risk should never default to "trade anyway".
        return {
            "trade_allowed":      False,
            "reason":             reason,
            "flagged_events":     [],
            "upcoming_events":    [],
            "currencies_checked": [],
            "risk_level":         "HIGH",
            "aftermath":          {"in_confirmation_window": False, "advice": ""},
            "source":             "none",
        }

    def currency_strength(self, ind_ctx: dict) -> dict:
        price = ind_ctx.get("close", 0)
        sma20 = ind_ctx.get("sma20", price)
        sma50 = ind_ctx.get("sma50", price)
        rsi   = ind_ctx.get("rsi", 50)
        score = 0
        if price > sma20: score += 1
        if price > sma50: score += 1
        if rsi > 55:      score += 1
        if rsi < 45:      score -= 1
        if price < sma20: score -= 1
        if price < sma50: score -= 1
        label = "STRONG" if score >= 2 else "WEAK" if score <= -2 else "NEUTRAL"
        return {"score": score, "label": label}

    def save_event_memory(self, event: dict, reaction_pips: float = 0) -> None:
        import json, os
        path = "memory/news_history.json"
        os.makedirs("memory", exist_ok=True)
        history = []
        if os.path.exists(path):
            try:
                with open(path) as f:
                    history = json.load(f)
            except Exception:
                pass
        history.append({
            "timestamp":     datetime.now(pytz.utc).isoformat(),
            "event":         event.get("event", ""),
            "currency":      event.get("currency", ""),
            "reaction_pips": reaction_pips,
            "lesson":        f"Avoid entry 30 min before {event.get('event', '')}",
        })
        with open(path, "w") as f:
            json.dump(history[-100:], f, indent=2)
        log.info(f"News memory saved: {event.get('event')}")

    def print_summary(self, result: dict) -> None:
        bar     = "═" * 44
        allowed = result["trade_allowed"]
        log.info(bar)
        log.info(f"  {'✅' if allowed else '⛔'}  NEWS FILTER  (Day 43)")
        log.info(bar)
        log.info(f"  Trade allowed : {allowed}")
        log.info(f"  Reason        : {result['reason']}")
        log.info(f"  Risk level    : {result.get('risk_level', 'LOW')}")
        log.info(f"  Data source   : {result.get('source', 'unknown')}")
        if result.get("aftermath", {}).get("in_confirmation_window"):
            log.info(f"  ⏳ Aftermath  : {result['aftermath']['advice']}")
        if result.get("flagged_events"):
            log.info("  ── Flagged ──")
            for ev in result["flagged_events"]:
                vol = ev.get("volatility", {})
                log.info(
                    f"    {ev['currency']} {ev['event']} @ {ev['time']} "
                    f"[{vol.get('level','?')} | {vol.get('pips', ('?','?'))} pips]"
                )
        if result.get("upcoming_events"):
            log.info("  ── Upcoming (3h) ──")
            for ev in result["upcoming_events"]:
                vol = ev.get("volatility", {})
                log.info(f"    {ev['currency']} {ev['event']} @ {ev['time']} [{vol.get('level','?')}]")
        log.info(bar)

    def get_ai_context(self, result: dict) -> dict:
        return {
            "news_trade_allowed": result["trade_allowed"],
            "news_reason":        result["reason"],
            "news_flagged_count": len(result.get("flagged_events", [])),
            "upcoming_events":    result.get("upcoming_events", []),
            "risk_level":         result.get("risk_level", "LOW"),
            "aftermath":          result.get("aftermath", {}),
            "source":             result.get("source", "unknown"),
            "trade_allowed":      result["trade_allowed"],
        }