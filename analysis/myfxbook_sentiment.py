"""
analysis/myfxbook_sentiment.py — Day 95 Myfxbook Community Outlook (OANDA alternative)
=====================================================================================
Pulls retail trader sentiment from Myfxbook's Community Outlook page.

Why this exists:
  OANDA's v20 API requires a practice account + token. Many users don't
  want to open an OANDA account just for sentiment data. Myfxbook's
  Community Outlook is FREE, public, and requires NO API key — it's
  scraped from their public webpage.

Myfxbook Community Outlook shows:
  - % of retail traders long vs short per pair
  - Average entry price for longs and shorts
  - Total long/short volume
  - Pip P/L distribution

This is a CONTRARIAN indicator: when 80%+ retail is long, smart money
is usually short, and price tends to reverse.

Fallback chain (in retail_sentiment.py):
  1. OANDA v20 (if OANDA_API_KEY set) — most accurate, has order book
  2. Myfxbook Community Outlook (this module, no key needed) — good accuracy
  3. Synthetic sentiment (computed from RSI + price action) — last resort

Usage:
    from analysis.myfxbook_sentiment import MyfxbookSentiment
    api = MyfxbookSentiment()
    result = api.get_sentiment("EURUSD")
    # result = {"long_pct": 72.3, "short_pct": 27.7, "contrarian": "BEARISH", ...}

Notes:
  - Myfxbook's public outlook page is HTML, so we parse it with BeautifulSoup.
  - The page is updated every ~5 minutes.
  - No rate limit on public page views, but be polite (1 req per pair per cycle).
  - If Myfxbook adds bot-detection, we fall back to synthetic sentiment.

Round-13 audit fix (concurrency / reliability hardening):
  A prior audit flagged six production-readiness gaps, all fixed here:
    1. Cache was a bare class dict with no lock -> race conditions under
       multi-threaded callers. Now guarded by `_cache_lock` (RLock).
    2. Concurrent calls for different pairs each triggered their own
       network fetch of the *same* outlook page. Now a single-flight
       `_fetch_lock` collapses concurrent fetches into one request, and
       that one request warms the cache for every pair on the page (not
       just the one that was asked for).
    3. `_cache` grew forever (never evicted). Now an `OrderedDict` used
       as a simple LRU, capped at `MAX_CACHE_ENTRIES`, with a hard
       staleness ceiling (`STALE_MAX_AGE_SEC`) after which even stale
       entries are dropped.
    4. On fetch failure we used to jump straight to a flat 50/50
       neutral fallback. Now we first try to serve a *stale* cache
       entry (clearly labeled `myfxbook_stale_cache` + age) — old real
       positioning data is a better signal than a fabricated neutral.
    5. `cloudscraper.create_scraper()` was called on every single
       request (expensive: re-solves Cloudflare's JS challenge each
       time). Now a singleton scraper is created once and reused.
    6. Transient errors (HTTP 429/503) are now retried with capped
       exponential backoff + jitter before falling through to the next
       layer, instead of failing immediately.
  Also added: fetch/parse latency logging and a `get_metrics()`
  classmethod (cache hit rate, fetch success rate, avg latency) for
  production observability, and a multi-factor synthetic-sentiment
  model (RSI + trend slope + extension z-score instead of RSI alone).

  Compatibility note: `retail_sentiment.py` and `sentiment_data.py` do
  exact-string matching on `result["source"]` (e.g.
  `if source == "synthetic_rsi"`, `if source in (..., "myfxbook_cached")`).
  The synthetic source string is therefore kept as `"synthetic_rsi"`
  even though the model is now multi-factor, and the new
  `"myfxbook_stale_cache"` source has been added to both call sites'
  accepted-source lists (see companion patch) — otherwise the new
  stale-cache fallback would be silently discarded by those callers.
"""
from __future__ import annotations

import os
import random
import re
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from utils.logger import get_logger

log = get_logger("myfxbook_sentiment")


class MyfxbookSentiment:
    """Myfxbook Community Outlook scraper — free, no API key needed.

    Round-5 audit fix: Myfxbook's public outlook page is now behind a
    Cloudflare WAF / bot-detection layer that returns HTTP 403 to plain
    `requests.get()`. Two mitigations added:

      1. **cloudscraper fallback**: if `requests` returns 403, retry
         with the `cloudscraper` library (which solves Cloudflare's
         JS challenge automatically) if it's installed. If cloudscraper
         is not installed, log an actionable warning.
      2. **circuit breaker**: after N consecutive failures (default 5),
         the source is marked DISABLED for a cooldown period (default
         30 min). During cooldown, `get_sentiment()` returns the
         fallback result immediately without attempting a fetch — this
         prevents log spam every cycle when the WAF is permanently
         blocking us.

    To re-enable after cooldown, just call get_sentiment() — it
    auto-resets after the cooldown period elapses.

    Round-13 audit fix: see module docstring for the full list of
    concurrency/reliability fixes (thread-safe cache, single-flight
    fetch locking, LRU eviction, stale-cache fallback, singleton
    scraper, retry+backoff, latency metrics, multi-factor synthetic
    sentiment).
    """

    BASE_URL = "https://www.myfxbook.com/community/outlook"

    # Cache results for 10 minutes to avoid hitting the page too often.
    # Round-13: now an OrderedDict used as a simple LRU (move-to-end on
    # write/fresh-read, evict from the front when over capacity), guarded
    # by `_cache_lock`. Value shape unchanged: pair -> (timestamp, data).
    _cache: "OrderedDict[str, tuple]" = OrderedDict()
    _cache_lock: threading.RLock = threading.RLock()
    CACHE_TTL_SEC = 600  # 10 minutes — entry is "fresh" until this age
    STALE_MAX_AGE_SEC = 6 * 3600  # entries older than this are dropped
    MAX_CACHE_ENTRIES = 300  # hard cap so cache can't grow unbounded

    # Round-13: single-flight lock. The outlook page returns ALL pairs in
    # one response, so instead of a per-symbol lock (which wouldn't stop
    # two threads asking for *different* pairs from both hitting the
    # network), one global fetch lock is used: whichever thread gets it
    # fetches the page once and warms the cache for every pair found,
    # and every other thread that was waiting picks its pair straight out
    # of that freshly-populated cache instead of fetching again.
    _fetch_lock: threading.RLock = threading.RLock()

    # Round-13: singleton cloudscraper instance + creation lock (double-
    # checked locking so it's still only built once under concurrency).
    _scraper: Any = None
    _scraper_lock: threading.Lock = threading.Lock()

    # Round-13: retry/backoff tuning for transient HTTP errors (429/503).
    MAX_RETRIES = 2
    RETRY_BACKOFF_BASE_SEC = 1.0
    # ConnectionResetError (TCP RST from Cloudflare) gets a longer base
    # backoff — the default 1s × 2^attempt caps at 4s, which is far too
    # aggressive for a server that just RST'd us. Cloudflare typically
    # needs 15-30s before accepting a new connection from the same IP
    # after a RST, so retrying within 4s just produces 3 identical
    # failures and exhausts the retry budget for nothing.
    CONNRESET_BACKOFF_BASE_SEC = 15.0

    # Round-13: lightweight in-process metrics (see get_metrics()).
    _metrics_lock: threading.Lock = threading.Lock()
    _metrics: Dict[str, Any] = {
        "cache_hits": 0,
        "cache_misses": 0,
        "fetch_success": 0,
        "fetch_failure": 0,
        "stale_serves": 0,
        "fetch_latencies_ms": [],
    }
    _METRICS_LATENCY_WINDOW = 50  # keep a rolling window, not unbounded

    # Round-5: circuit breaker state (class-level, shared across instances)
    _consecutive_failures: int = 0
    _disabled_until: float = 0.0  # epoch seconds; 0 = not disabled
    MAX_CONSECUTIVE_FAILURES = 5  # disable after this many in a row
    COOLDOWN_SEC = 1800  # 30 min cooldown after disabling

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.myfxbook.com/",
    }

    def __init__(self):
        self._available = True  # public page, no key needed

    @classmethod
    def _is_in_cooldown(cls) -> bool:
        """Check if the circuit breaker has disabled this source."""
        if cls._disabled_until and time.time() < cls._disabled_until:
            return True
        # Cooldown expired — reset
        if cls._disabled_until and time.time() >= cls._disabled_until:
            cls._disabled_until = 0.0
            cls._consecutive_failures = 0
            # A fresh cooldown cycle likely means Myfxbook's WAF state (or
            # our IP's reputation with it) has changed — force a new
            # scraper on the next attempt rather than reusing a scraper
            # whose solved-challenge cookies may now be stale/blocked.
            with cls._scraper_lock:
                cls._scraper = None
            log.info(
                "[Myfxbook] Cooldown expired — re-enabling source. "
                "Next fetch attempt will run."
            )
        return False

    @classmethod
    def _record_failure(cls, reason: str) -> None:
        """Record a fetch failure; trip circuit breaker if threshold hit."""
        cls._consecutive_failures += 1
        if cls._consecutive_failures >= cls.MAX_CONSECUTIVE_FAILURES:
            cls._disabled_until = time.time() + cls.COOLDOWN_SEC
            log.warning(
                f"[Myfxbook] Circuit breaker TRIPPED after "
                f"{cls._consecutive_failures} consecutive failures "
                f"({reason}). Source disabled for {cls.COOLDOWN_SEC}s. "
                f"Retail sentiment will use stale-cache/synthetic RSI "
                f"fallback until cooldown expires."
            )
        else:
            log.warning(
                f"[Myfxbook] Failure {cls._consecutive_failures}/"
                f"{cls.MAX_CONSECUTIVE_FAILURES}: {reason}"
            )

    @classmethod
    def _record_success(cls) -> None:
        """Reset failure counter on a successful fetch."""
        if cls._consecutive_failures > 0:
            log.info(
                f"[Myfxbook] Recovered after {cls._consecutive_failures} "
                f"consecutive failure(s). Counter reset."
            )
        cls._consecutive_failures = 0
        cls._disabled_until = 0.0

    @property
    def available(self) -> bool:
        return self._available

    # ─────────────────────────────────────────────────────────
    # Round-13: metrics helpers
    # ─────────────────────────────────────────────────────────

    @classmethod
    def _record_cache_hit(cls) -> None:
        with cls._metrics_lock:
            cls._metrics["cache_hits"] += 1

    @classmethod
    def _record_cache_miss(cls) -> None:
        with cls._metrics_lock:
            cls._metrics["cache_misses"] += 1

    @classmethod
    def _record_fetch_success(cls) -> None:
        with cls._metrics_lock:
            cls._metrics["fetch_success"] += 1

    @classmethod
    def _record_fetch_failure(cls) -> None:
        with cls._metrics_lock:
            cls._metrics["fetch_failure"] += 1

    @classmethod
    def _record_stale_serve(cls) -> None:
        with cls._metrics_lock:
            cls._metrics["stale_serves"] += 1

    @classmethod
    def _record_latency(cls, ms: float) -> None:
        with cls._metrics_lock:
            lat = cls._metrics["fetch_latencies_ms"]
            lat.append(ms)
            if len(lat) > cls._METRICS_LATENCY_WINDOW:
                del lat[0]

    @classmethod
    def get_metrics(cls) -> Dict[str, Any]:
        """Production observability snapshot — cache efficiency, fetch
        reliability, and latency. Safe to call from a health-check /
        dashboard endpoint; cheap (no I/O)."""
        with cls._metrics_lock:
            latencies = list(cls._metrics["fetch_latencies_ms"])
            hits = cls._metrics["cache_hits"]
            misses = cls._metrics["cache_misses"]
            fetch_success = cls._metrics["fetch_success"]
            fetch_failure = cls._metrics["fetch_failure"]
            stale_serves = cls._metrics["stale_serves"]
        total_lookups = hits + misses
        total_fetches = fetch_success + fetch_failure
        with cls._cache_lock:
            cache_size = len(cls._cache)
        return {
            "cache_hits": hits,
            "cache_misses": misses,
            "cache_hit_rate_pct": round(100 * hits / total_lookups, 1) if total_lookups else None,
            "cache_size": cache_size,
            "cache_capacity": cls.MAX_CACHE_ENTRIES,
            "fetch_success": fetch_success,
            "fetch_failure": fetch_failure,
            "fetch_success_rate_pct": round(100 * fetch_success / total_fetches, 1) if total_fetches else None,
            "stale_serves": stale_serves,
            "avg_fetch_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
            "circuit_breaker_tripped": cls._disabled_until > time.time(),
            "consecutive_failures": cls._consecutive_failures,
        }

    # ─────────────────────────────────────────────────────────
    # Round-13: thread-safe cache helpers (LRU + stale-if-error)
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_pair_key(pair: str) -> str:
        """Canonical cache key, e.g. 'eur/usd' / 'EURUSD' / 'EUR/USD' all
        map to 'EURUSD'. Round-13 fix: the old code cached under the raw,
        un-normalized caller-supplied string, so 'EURUSD' and 'eurusd'
        silently created two separate cache entries."""
        return pair.upper().replace("/", "").replace("=X", "")

    @classmethod
    def _cache_get(cls, key: str, allow_stale: bool = False):
        """Returns (data, age_sec) for a fresh entry, or for a stale-but-
        still-usable entry when `allow_stale=True`. Returns (None, age)
        (age may be None) when nothing usable is cached."""
        with cls._cache_lock:
            entry = cls._cache.get(key)
            if entry is None:
                return None, None
            ts, data = entry
            age = datetime.now(timezone.utc).timestamp() - ts
            if age < cls.CACHE_TTL_SEC:
                cls._cache.move_to_end(key)
                return data, age
            if allow_stale and age < cls.STALE_MAX_AGE_SEC:
                return data, age
            return None, age

    @classmethod
    def _cache_put(cls, key: str, data: Dict[str, Any]) -> None:
        with cls._cache_lock:
            cls._cache[key] = (datetime.now(timezone.utc).timestamp(), data)
            cls._cache.move_to_end(key)
            # LRU eviction: drop least-recently-written/read entries first.
            while len(cls._cache) > cls.MAX_CACHE_ENTRIES:
                cls._cache.popitem(last=False)

    # ─────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────

    def get_sentiment(self, pair: str) -> Dict[str, Any]:
        """Get retail sentiment for a pair from Myfxbook Community Outlook.

        Args:
            pair: e.g. "EURUSD" (will be converted to "EUR/USD")

        Returns: dict with long_pct, short_pct, contrarian_signal, etc.
                 Falls back to stale cache, then neutral, if scrape fails.
        """
        # P1 perf fix (2026-08-03, parity investigation): skip the Myfxbook
        # scrape entirely in backtest mode — a historical bar has no "today's
        # retail positioning" to fetch, and the scrape (HTTPS GET +
        # BeautifulSoup parse of myfxbook.com/outlook) was costing ~1-2s per
        # bar per pair. Mirrors the backtest short-circuit pattern already
        # used in retail_sentiment.py (line 100-101), sentiment_data.py,
        # economic_calendar_api.py, news_filter.py, news_api_provider.py,
        # and macro_data.py.
        from core.constants import is_backtest_mode
        if is_backtest_mode():
            return {
                "long_pct":           50.0,
                "short_pct":          50.0,
                "contrarian_signal":  "NEUTRAL",
                "strength":           "weak",
                "source":             "backtest_skipped",
                "pair":               pair,
            }

        key = self._normalize_pair_key(pair)

        # ── Fast path: fresh cache hit, no locking needed ──
        fresh, _ = self._cache_get(key)
        if fresh is not None:
            self._record_cache_hit()
            result = dict(fresh)
            result["source"] = "myfxbook_cached"
            return result
        self._record_cache_miss()

        # ── Single-flight fetch: collapse concurrent misses into one
        #    network request instead of one-per-caller (Round-13 fix #2).
        with self._fetch_lock:
            # Re-check: another thread may have populated the cache for
            # this pair while we were waiting for the lock.
            fresh, _ = self._cache_get(key)
            if fresh is not None:
                self._record_cache_hit()
                result = dict(fresh)
                result["source"] = "myfxbook_cached"
                return result

            t0 = time.monotonic()
            outlook_data = self._fetch_outlook_page()
            fetch_ms = (time.monotonic() - t0) * 1000
            self._record_latency(fetch_ms)

            # Round-12 audit fix: distinguish "scrape failed" (HTTP error /
            # Cloudflare block) from "parse failed" (page loaded but no
            # pairs extracted — HTML structure changed).
            #
            # Previously: `if not outlook_data` treated both cases as
            # "scrape failed", which was misleading. The operator's audit
            # saw "[Myfxbook] cloudscraper fetch SUCCESS" immediately
            # followed by "[RetailSent] Myfxbook failed" — the fetch
            # worked but parsing returned [].
            if outlook_data is None:
                # None = HTTP/Cloudflare failure (fetch never succeeded)
                self._record_fetch_failure()
                return self._stale_or_fallback(
                    pair, key, fetch_ms, "Myfxbook scrape failed (HTTP/Cloudflare)"
                )
            if len(outlook_data) == 0:
                # Empty list = page loaded but parser found no pairs
                # (HTML structure may have changed)
                self._record_fetch_failure()
                log.warning(
                    f"[Myfxbook] Page fetched successfully but parser found "
                    f"0 pairs — HTML structure may have changed. Falling "
                    f"back to stale cache / synthetic sentiment for {pair}."
                )
                return self._stale_or_fallback(
                    pair, key, fetch_ms,
                    "Myfxbook parse failed (0 pairs extracted — HTML structure changed?)"
                )

            self._record_fetch_success()

            # Round-13 fix #2 (continued): one page fetch returns ALL
            # pairs — warm the cache for every one of them, not just the
            # pair the caller asked for. This is what makes the
            # single-flight lock actually kill duplicate requests: a
            # fetch triggered by an EURUSD lookup also satisfies the next
            # GBPUSD/USDJPY/etc. lookup straight from cache.
            for item in outlook_data:
                item_key = self._normalize_pair_key(item["pair"])
                built = self._build_result_from_pair_data(item_key, item)
                self._cache_put(item_key, built)

            pair_data = self._find_pair(outlook_data, pair)
            if not pair_data:
                return self._fallback_result(
                    pair, f"{pair} not found in Myfxbook outlook ({len(outlook_data)} pairs available)"
                )

            cached_now, _ = self._cache_get(key)
            result = dict(cached_now) if cached_now is not None else self._build_result_from_pair_data(pair, pair_data)
            result["source"] = "myfxbook_live"

            log.info(
                f"[Myfxbook] {pair} | retail {result['sentiment_label']} "
                f"({result['long_pct']:.0f}%L/{result['short_pct']:.0f}%S) | "
                f"contrarian={result['contrarian_signal']}({result['contrarian_strength']}) | "
                f"bias={result['trade_bias']} conf={result['confidence']}% | "
                f"fetch_ms={fetch_ms:.0f} cache_size={len(self._cache)}"
            )
            return result

    def _stale_or_fallback(self, pair: str, key: str, fetch_ms: float, reason: str) -> Dict[str, Any]:
        """Round-13 fix #4: on fetch/parse failure, prefer a stale cache
        entry over a fabricated 50/50 neutral — old real positioning data
        is materially more useful to a contrarian strategy than a
        placeholder that claims retail is perfectly balanced."""
        stale, stale_age = self._cache_get(key, allow_stale=True)
        if stale is not None:
            self._record_stale_serve()
            result = dict(stale)
            result["source"] = "myfxbook_stale_cache"
            result["stale_age_sec"] = round(stale_age, 1)
            log.warning(
                f"[Myfxbook] {reason} — serving stale cache for {pair} "
                f"(age={stale_age:.0f}s, fetch_ms={fetch_ms:.0f})"
            )
            return result
        log.info(f"[Myfxbook] fetch_ms={fetch_ms:.0f} result=fail pair={pair} reason={reason}")
        return self._fallback_result(pair, reason)

    # ─────────────────────────────────────────────────────────
    # Scraping
    # ─────────────────────────────────────────────────────────

    @classmethod
    def _get_scraper(cls):
        """Round-13 fix #5: singleton `cloudscraper` instance. Creating a
        new scraper per request re-solves Cloudflare's JS challenge every
        time — slow (extra round-trips) and more bot-like (repeated
        challenge solves from the same IP in a short window is itself a
        signal Cloudflare can flag on). Reuse one scraper — and its
        solved-challenge cookies — across requests. Thread-safe via
        double-checked locking so it's still only built once.
        """
        if cls._scraper is not None:
            return cls._scraper
        with cls._scraper_lock:
            if cls._scraper is None:
                import cloudscraper  # type: ignore
                cls._scraper = cloudscraper.create_scraper(
                    browser={"browser": "chrome", "platform": "windows", "mobile": False}
                )
            return cls._scraper

    def _fetch_outlook_page(self) -> Optional[List[Dict]]:
        """Fetch and parse Myfxbook's community outlook page.

        Returns: list of dicts, each with pair, long_pct, short_pct, etc.
                 None on failure.

        Round-5 audit fix: added cooldown check + cloudscraper fallback.
        Round-13 audit fix: added retry+backoff on transient errors
        (429/503) and switched to the singleton scraper.
        """
        # ── Circuit breaker: skip fetch entirely during cooldown ──
        if self._is_in_cooldown():
            log.debug(
                "[Myfxbook] Skipping fetch — in circuit-breaker cooldown. "
                "Caller will get stale-cache / fallback result."
            )
            return None

        # ── Attempt 1: plain requests with browser-mimicking headers,
        #    retried with backoff on transient (429/503) errors ──
        need_cloudscraper = False
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                resp = requests.get(self.BASE_URL, headers=self.HEADERS, timeout=15)
            except Exception as e:
                log.warning(f"[Myfxbook] fetch failed (requests, attempt {attempt + 1}): {e}")
                if attempt < self.MAX_RETRIES:
                    self._sleep_backoff(attempt)
                    continue
                need_cloudscraper = True
                break

            if resp.status_code == 200:
                self._record_success()
                return self._timed_parse(resp.text)
            if resp.status_code in (429, 503) and attempt < self.MAX_RETRIES:
                log.info(
                    f"[Myfxbook] HTTP {resp.status_code} from requests — "
                    f"retrying (attempt {attempt + 1}/{self.MAX_RETRIES})"
                )
                self._sleep_backoff(attempt)
                continue
            if resp.status_code in (403, 429, 503):
                # Cloudflare WAF / persistent rate-limit / bot-detection —
                # try cloudscraper if available.
                log.info(
                    f"[Myfxbook] HTTP {resp.status_code} from requests — "
                    f"trying cloudscraper fallback (if installed)."
                )
                need_cloudscraper = True
                break
            log.warning(f"[Myfxbook] HTTP {resp.status_code}")
            self._record_failure(f"HTTP {resp.status_code}")
            return None

        if not need_cloudscraper:
            self._record_failure("requests retries exhausted")
            return None

        # ── Attempt 2: singleton cloudscraper (solves Cloudflare JS
        #    challenge), also retried with backoff on transient errors ──
        try:
            scraper = self._get_scraper()
        except ImportError:
            log.warning(
                "[Myfxbook] HTTP 403/429 received and `cloudscraper` not "
                "installed. Install with `pip install cloudscraper` to "
                "bypass Cloudflare bot detection. Falling back to "
                "stale cache / synthetic RSI sentiment."
            )
            self._record_failure("403 + cloudscraper not installed")
            return None
        except Exception as e:
            log.warning(f"[Myfxbook] cloudscraper init failed: {e}")
            self._record_failure(f"cloudscraper init exception: {e}")
            return None

        for attempt in range(self.MAX_RETRIES + 1):
            try:
                resp2 = scraper.get(self.BASE_URL, timeout=20)
            except ConnectionResetError as e:
                # Cloudflare RST — needs much longer backoff than a 429.
                # The default 1s×2^attempt would fire 3 retries within 7s
                # and all fail identically. Use the CONNRESET base so the
                # retries are 15s → 30s → 60s, giving Cloudflare time to
                # un-blacklist our IP.
                log.warning(
                    f"[Myfxbook] cloudscraper ConnectionResetError (attempt {attempt + 1}): "
                    f"{e} — TCP RST from upstream, using longer backoff"
                )
                if attempt < self.MAX_RETRIES:
                    self._sleep_backoff_connreset(attempt)
                    continue
                self._record_failure(f"cloudscraper ConnectionResetError: {e}")
                return None
            except Exception as e:
                log.warning(f"[Myfxbook] cloudscraper fetch failed (attempt {attempt + 1}): {e}")
                if attempt < self.MAX_RETRIES:
                    self._sleep_backoff(attempt)
                    continue
                self._record_failure(f"cloudscraper exception: {e}")
                return None

            if resp2.status_code == 200:
                log.info(
                    "[Myfxbook] cloudscraper fetch SUCCESS — "
                    "Cloudflare challenge solved."
                )
                self._record_success()
                return self._timed_parse(resp2.text)
            if resp2.status_code in (429, 503) and attempt < self.MAX_RETRIES:
                log.info(
                    f"[Myfxbook] cloudscraper HTTP {resp2.status_code} — "
                    f"retrying (attempt {attempt + 1}/{self.MAX_RETRIES})"
                )
                self._sleep_backoff(attempt)
                continue
            log.warning(
                f"[Myfxbook] cloudscraper also failed: HTTP {resp2.status_code}"
            )
            self._record_failure(f"cloudscraper HTTP {resp2.status_code}")
            return None

        self._record_failure("cloudscraper retries exhausted")
        return None

    @classmethod
    def _sleep_backoff(cls, attempt: int) -> None:
        """Capped exponential backoff with jitter (Round-13 fix #6)."""
        backoff = cls.RETRY_BACKOFF_BASE_SEC * (2 ** attempt) + random.uniform(0, 0.5)
        time.sleep(backoff)

    @classmethod
    def _sleep_backoff_connreset(cls, attempt: int) -> None:
        """Longer exponential backoff for ConnectionResetError (Cloudflare RST).

        The default _sleep_backoff caps at ~4s which is too short for
        Cloudflare to un-block our IP. Use a 15s base so retries land at
        ~15s → ~30s → ~60s, giving the upstream time to recover.
        """
        backoff = cls.CONNRESET_BACKOFF_BASE_SEC * (2 ** attempt) + random.uniform(0, 1.0)
        time.sleep(backoff)

    @staticmethod
    def _timed_parse(html: str) -> List[Dict]:
        """Wraps _parse_outlook_html with latency logging (Round-13 fix)."""
        t0 = time.monotonic()
        parsed = MyfxbookSentiment._parse_outlook_html(html)
        parse_ms = (time.monotonic() - t0) * 1000
        log.debug(f"[Myfxbook] parse_ms={parse_ms:.0f} pairs={len(parsed)}")
        return parsed

    @staticmethod
    def _parse_outlook_html(html: str) -> List[Dict]:
        """Parse Myfxbook outlook HTML to extract per-pair sentiment.

        Myfxbook's outlook page has a table with rows like:
          EUR/USD | 72% long | 28% short | avg long 1.0850 | avg short 1.0820

        We use regex + BeautifulSoup to extract this.
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            log.warning("[Myfxbook] BeautifulSoup not installed")
            return []

        soup = BeautifulSoup(html, "html.parser")
        results = []

        def add_pair(pair_name, long_value, short_value, avg_long=None, avg_short=None):
            """Normalize and validate a record from any page representation."""
            normalized = re.sub(r"[^A-Z]", "", str(pair_name).upper())
            if len(normalized) != 6:
                return
            try:
                long_pct = float(long_value)
                short_pct = float(short_value)
            except (TypeError, ValueError):
                return
            if not (0 <= long_pct <= 100 and 0 <= short_pct <= 100):
                return
            # Labels may include a tiny rounding difference.
            if abs(long_pct + short_pct - 100) > 5:
                return
            results.append({
                "pair": f"{normalized[:3]}/{normalized[3:]}",
                "long_pct": long_pct,
                "short_pct": short_pct,
                "avg_long_price": avg_long,
                "avg_short_price": avg_short,
                "total_long_volume": None,
                "total_short_volume": None,
            })

        # Myfxbook uses a table with class 'outlookTable' or similar
        # Each row has the pair name + long/short percentages
        # Try multiple selectors since the page structure changes

        # Approach 1: look for table rows with pair names
        #
        # Bug note: this previously required a literal "/" between the two
        # currency codes (e.g. "EUR/USD") and matched only upper-case. If
        # Myfxbook's current markup renders the pair as "EURUSD" (no slash)
        # or wraps each character in its own span (so get_text() still joins
        # them into "EURUSD", but case can vary depending on CSS
        # text-transform), the old regex silently matched zero rows and the
        # whole source fell through to synthetic sentiment. Made the slash
        # optional and the match case-insensitive; add_pair() already
        # upper-cases and validates the result, so this doesn't loosen the
        # final data quality check.
        for row in soup.select("tr"):
            try:
                text = row.get_text(separator=" ", strip=True)
                # Look for patterns like "EUR/USD" or "EURUSD" followed by percentages
                match = re.search(
                    r"([A-Za-z]{3}/?[A-Za-z]{3}).*?(\d+(?:\.\d+)?)%.*?(\d+(?:\.\d+)?)%",
                    text
                )
                if match:
                    # Try to extract average prices
                    prices = re.findall(r"(\d+\.\d{4,5})", text)
                    avg_long = float(prices[0]) if len(prices) >= 1 else None
                    avg_short = float(prices[1]) if len(prices) >= 2 else None
                    add_pair(match.group(1), match.group(2), match.group(3), avg_long, avg_short)
            except Exception as e:
                log.debug(f"[myfxbook_sentiment] suppressed: {e}")
                continue

        # Current Myfxbook pages can render card elements instead of table
        # rows.  Read their data attributes as well as the visible text.
        for node in soup.select("[data-symbol], [data-pair], [data-instrument]"):
            pair_name = node.get("data-symbol") or node.get("data-pair") or node.get("data-instrument")
            long_pct = node.get("data-long") or node.get("data-long-percent") or node.get("data-long-percentage")
            short_pct = node.get("data-short") or node.get("data-short-percent") or node.get("data-short-percentage")
            if pair_name and long_pct is not None and short_pct is not None:
                add_pair(pair_name, long_pct, short_pct)

        # The client-side page also embeds records in script JSON.  This is
        # deliberately schema-tolerant so a markup-only redesign does not
        # silently turn a successful fetch into a synthetic fallback.
        for script in soup.find_all("script"):
            payload = script.string or script.get_text() or ""
            for match in re.finditer(
                r'(?is)["\'](?:symbol|pair|instrument)["\']\s*:\s*["\']([A-Z]{3}/?[A-Z]{3})["\']'
                r'.{0,500}?["\'](?:long|longPercent|longPercentage|long_pct)["\']\s*:\s*["\']?([\d.]+)'
                r'.{0,300}?["\'](?:short|shortPercent|shortPercentage|short_pct)["\']\s*:\s*["\']?([\d.]+)',
                payload,
            ):
                add_pair(match.group(1), match.group(2), match.group(3))

        # Deduplicate by pair name (keep first occurrence)
        seen = set()
        unique = []
        for r in results:
            if r["pair"] not in seen:
                seen.add(r["pair"])
                unique.append(r)

        if unique:
            log.debug(f"[Myfxbook] parsed {len(unique)} pairs from outlook page")
        else:
            # Previously this was a silent debug-level no-op, so when
            # Myfxbook changed their markup the source quietly degraded to
            # synthetic sentiment with no actionable signal in the logs.
            # Log a snippet of the raw HTML so a future markup change can be
            # diagnosed without re-adding print statements.
            snippet = re.sub(r"\s+", " ", html)[:500]
            log.warning(
                "[Myfxbook] parsed 0 pairs from outlook page — page structure "
                f"may have changed. HTML snippet: {snippet!r}"
            )
        return unique

    @staticmethod
    def _find_pair(outlook_data: List[Dict], pair: str) -> Optional[Dict]:
        """Find a specific pair in the outlook data.

        Args:
            outlook_data: list of dicts from _parse_outlook_html
            pair: e.g. "EURUSD" (will match "EUR/USD" in data)

        Returns: matching dict or None
        """
        # Normalize: EURUSD → EUR/USD
        target = pair.upper().replace("/", "").replace("=X", "")
        if len(target) >= 6:
            target = f"{target[:3]}/{target[3:6]}"

        for item in outlook_data:
            if item["pair"].upper() == target:
                return item
        return None

    # ─────────────────────────────────────────────────────────
    # Confidence calculation
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _compute_confidence(long_pct: float, short_pct: float, strength: str) -> int:
        """Contrarian confidence — higher when retail is more one-sided."""
        extremes = abs(long_pct - 50)
        base = int(extremes * 2)
        bonus = {"STRONG": 10, "MODERATE": 5, "WEAK": 0}.get(strength, 0)
        return max(0, min(100, base + bonus))

    # ─────────────────────────────────────────────────────────
    # Result construction (shared by live-fetch + pre-warm paths)
    # ─────────────────────────────────────────────────────────

    @classmethod
    def _build_result_from_pair_data(cls, pair_label: str, pair_data: Dict) -> Dict[str, Any]:
        """Round-13: factored out of get_sentiment() so the same logic
        builds the result for the caller's requested pair AND for every
        other pair on the same fetched page (which get pre-cached — see
        fix #2 in the module docstring)."""
        long_pct = pair_data["long_pct"]
        short_pct = pair_data["short_pct"]
        ratio = long_pct / short_pct if short_pct > 0 else float("inf")
        net_pct = long_pct - short_pct

        sentiment_label = "BULLISH" if long_pct > short_pct else "BEARISH"
        contrarian_signal = "BEARISH" if long_pct > 60 else "BULLISH" if long_pct < 40 else "NEUTRAL"
        contrarian_strength = (
            "STRONG" if long_pct > 75 or long_pct < 25
            else "MODERATE" if long_pct > 60 or long_pct < 40
            else "WEAK"
        )
        trade_bias = contrarian_signal
        confidence = cls._compute_confidence(long_pct, short_pct, contrarian_strength)

        return {
            "source":              "myfxbook_live",
            "pair":                pair_label,
            "long_pct":            round(long_pct, 1),
            "short_pct":           round(short_pct, 1),
            "sentiment_label":     sentiment_label,
            "contrarian_signal":   contrarian_signal,
            "contrarian_strength": contrarian_strength,
            "long_short_ratio":    round(ratio, 2),
            "net_position_pct":    round(net_pct, 1),
            "avg_long_price":      pair_data.get("avg_long_price"),
            "avg_short_price":     pair_data.get("avg_short_price"),
            "total_long_volume":   pair_data.get("total_long_volume"),
            "total_short_volume":  pair_data.get("total_short_volume"),
            "order_book":          {"price_levels": [], "stop_cluster": None},
            "trade_bias":          trade_bias,
            "confidence":          confidence,
            "fetched_at":          datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    # ─────────────────────────────────────────────────────────
    # Fallback (synthetic sentiment from RSI — last resort)
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _fallback_result(pair: str, reason: str) -> Dict[str, Any]:
        """When Myfxbook unavailable — return neutral sentiment."""
        return {
            "source":              "fallback",
            "pair":                pair,
            "long_pct":            50.0,
            "short_pct":           50.0,
            "sentiment_label":     "NEUTRAL",
            "contrarian_signal":   "NEUTRAL",
            "contrarian_strength": "WEAK",
            "long_short_ratio":    1.0,
            "net_position_pct":    0.0,
            "avg_long_price":      None,
            "avg_short_price":     None,
            "total_long_volume":   None,
            "total_short_volume":  None,
            "order_book":          {"price_levels": [], "stop_cluster": None},
            "trade_bias":          "NEUTRAL",
            "confidence":          0,
            "reason":              reason,
            "fetched_at":          datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    # ─────────────────────────────────────────────────────────
    # Synthetic sentiment (RSI + trend + extension — no external API needed)
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def compute_synthetic_sentiment(pair: str, df) -> Dict[str, Any]:
        """Compute synthetic retail sentiment from price action.

        Round-13 audit fix: previously this was a single-factor proxy
        (`long_pct = RSI`, clipped to 10-90). Real retail behavior isn't
        just "chase the oscillator" — it's a mix of momentum-chasing and
        blow-off-extension panic, so this now blends three signals:

          1. RSI (overbought/oversold) — the original signal, still the
             anchor, weight 0.5. Retail visibly chases momentum
             indicators near round-number extremes.
          2. Short-term trend slope (least-squares fit over the last 20
             closes, normalized by price level) — trending markets pull
             more retail into the direction of the trend (breakout
             chasing), weight 0.3.
          3. Extension z-score (distance of the latest close from the
             20-bar mean, in standard deviations) — captures
             euphoria/panic extension beyond what RSI alone reflects
             without needing a separate ATR column, weight 0.2.

        Each factor is mapped to a [-1, +1] tilt, blended, then mapped to
        long_pct in [10, 90] (same clipping as before). This is still a
        heuristic proxy for real positioning, not positioning data
        itself — confidence is still penalized the same way as before.

        Compatibility note: `source` is intentionally kept as
        `"synthetic_rsi"` (not renamed to reflect the new model) because
        `retail_sentiment.py` does an exact-string check
        `if result.get("source") == "synthetic_rsi"` — renaming this
        without also patching that call site would silently discard
        every synthetic-sentiment result and degrade straight to neutral
        fallback.

        Args:
            pair: e.g. "EURUSD"
            df: DataFrame with 'close' column + ideally 'rsi' column

        Returns: same shape as get_sentiment() output
        """
        if df is None or len(df) == 0:
            return MyfxbookSentiment._fallback_result(pair, "no data for synthetic sentiment")

        # Get RSI (compute if not present)
        if "rsi" in df.columns:
            rsi = float(df["rsi"].iloc[-1])
        else:
            try:
                import pandas_ta as ta
                rsi = float(ta.rsi(df["close"], length=14).iloc[-1])
            except Exception:
                return MyfxbookSentiment._fallback_result(pair, "RSI computation failed")

        if rsi != rsi:  # NaN check
            return MyfxbookSentiment._fallback_result(pair, "RSI is NaN")

        # RSI 50 → 0 tilt, RSI 100 → +1 tilt, RSI 0 → -1 tilt
        rsi_tilt = (rsi - 50) / 50.0

        trend_tilt = 0.0
        zscore_tilt = 0.0
        closes = df["close"].tail(20)
        if len(closes) >= 5:
            try:
                import numpy as np
                y = closes.to_numpy(dtype=float)
                x = np.arange(len(y))
                slope = np.polyfit(x, y, 1)[0]
                mean_price = float(y.mean())
                std_price = float(y.std())
                if mean_price:
                    # Total drift over the window, as a fraction of price
                    # level; *15 keeps typical FX moves in a usable range
                    # before clipping to [-1, 1].
                    norm_slope = (slope * len(y)) / mean_price
                    trend_tilt = max(-1.0, min(1.0, norm_slope * 15))
                if std_price > 0:
                    z = (float(y[-1]) - mean_price) / std_price
                    zscore_tilt = max(-1.0, min(1.0, z / 2.5))
            except Exception as e:
                log.debug(f"[SyntheticSent] trend/zscore calc skipped: {e}")

        composite = 0.5 * rsi_tilt + 0.3 * trend_tilt + 0.2 * zscore_tilt
        composite = max(-1.0, min(1.0, composite))
        long_pct = max(10.0, min(90.0, 50 + composite * 40))
        short_pct = 100 - long_pct

        sentiment_label = "BULLISH" if long_pct > short_pct else "BEARISH"
        contrarian_signal = "BEARISH" if long_pct > 60 else "BULLISH" if long_pct < 40 else "NEUTRAL"
        contrarian_strength = (
            "STRONG" if long_pct > 75 or long_pct < 25
            else "MODERATE" if long_pct > 60 or long_pct < 40
            else "WEAK"
        )
        confidence = MyfxbookSentiment._compute_confidence(long_pct, short_pct, contrarian_strength)

        result = {
            "source":              "synthetic_rsi",  # kept for compatibility — see docstring
            "pair":                pair,
            "long_pct":            round(long_pct, 1),
            "short_pct":           round(short_pct, 1),
            "sentiment_label":     sentiment_label,
            "contrarian_signal":   contrarian_signal,
            "contrarian_strength": contrarian_strength,
            "long_short_ratio":    round(long_pct / short_pct, 2) if short_pct > 0 else 99,
            "net_position_pct":    round(long_pct - short_pct, 1),
            "rsi_basis":           round(rsi, 1),
            "trend_tilt":          round(trend_tilt, 2),
            "zscore_tilt":         round(zscore_tilt, 2),
            "order_book":          {"price_levels": [], "stop_cluster": None},
            "trade_bias":          contrarian_signal,
            "confidence":          max(0, confidence - 20),  # lower confidence for synthetic
            "fetched_at":          datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        log.info(
            f"[SyntheticSent] {pair} | RSI={rsi:.1f} trend={trend_tilt:+.2f} "
            f"z={zscore_tilt:+.2f} → retail {sentiment_label} "
            f"({long_pct:.0f}%L/{short_pct:.0f}%S) | contrarian={contrarian_signal}"
        )
        return result

    # ─────────────────────────────────────────────────────────
    # AI context (compatible with RetailSentimentAPI)
    # ─────────────────────────────────────────────────────────

    def get_ai_context(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Compact context for MasterAnalyst — same shape as RetailSentimentAPI."""
        return {
            "sentiment_source":         result.get("source", "fallback"),
            "sentiment_retail_long":    result.get("long_pct", 50),
            "sentiment_retail_short":   result.get("short_pct", 50),
            "sentiment_label":          result.get("sentiment_label", "NEUTRAL"),
            "sentiment_contrarian":     result.get("contrarian_signal", "NEUTRAL"),
            "sentiment_strength":       result.get("contrarian_strength", "WEAK"),
            "sentiment_bias":           result.get("trade_bias", "NEUTRAL"),
            "sentiment_confidence":     result.get("confidence", 0),
            "sentiment_stop_cluster":   result.get("order_book", {}).get("stop_cluster"),
        }

    def print_summary(self, result: Dict[str, Any]) -> None:
        bar = "═" * 50
        log.info(bar)
        log.info("  👥  MYFXBOOK SENTIMENT  (Day 95)")
        log.info(bar)
        log.info(f"  Pair           : {result.get('pair','?')}")
        log.info(f"  Source         : {result.get('source','?')}")
        log.info(f"  Retail Long %  : {result.get('long_pct',0):.1f}")
        log.info(f"  Retail Short % : {result.get('short_pct',0):.1f}")
        log.info(f"  Sentiment      : {result.get('sentiment_label','?')} (retail mood)")
        log.info(f"  Contrarian     : {result.get('contrarian_signal','?')} ({result.get('contrarian_strength','?')})")
        log.info(f"  Trade bias     : {result.get('trade_bias','?')} | conf {result.get('confidence',0)}%")
        if result.get("rsi_basis"):
            log.info(f"  RSI basis      : {result['rsi_basis']}")
        if result.get("stale_age_sec"):
            log.info(f"  Stale age      : {result['stale_age_sec']:.0f}s")
        log.info(bar)


# ── Singleton ────────────────────────────────────────────────────

_INSTANCE: Optional[MyfxbookSentiment] = None


def get_myfxbook_sentiment() -> MyfxbookSentiment:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = MyfxbookSentiment()
    return _INSTANCE