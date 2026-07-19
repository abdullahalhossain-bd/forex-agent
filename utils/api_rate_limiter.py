# utils/api_rate_limiter.py  —  Provider-aware rate limiting + retry
# ============================================================
# Day 99+ FIX (Issue #2): TwelveData (8 req/min), Polygon (5 req/min),
# Alpha Vantage (5 req/min, 25 req/day), and Finnhub (60 req/min) all
# enforce strict rate limits on the free tier. The data fetcher
# previously called these endpoints in a tight loop (one call per
# symbol per timeframe per cycle) with NO sleep between calls and NO
# retry on 429 — meaning a multi-symbol, multi-timeframe scan would
# blow through the limit within seconds, then every subsequent call
# in that minute would 429, and the bot would proceed with empty
# data and silently trade on stale / missing values.
#
# This module exposes:
#   - rate_limited_get(url, params, provider): HTTP GET that respects
#     per-provider rate limits + retries 429/5xx with exponential
#     backoff.
#   - register_provider(name, min_interval, max_retries): configure
#     the rate limit for a provider (called once at import time below
#     for the known free-tier providers).
#
# Used by data/fetcher.py for Alpha Vantage, Polygon, Finnhub,
# Twelve Data, and any future HTTP-based data source.
# ============================================================

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

import requests

from utils.logger import get_logger

log = get_logger("api_rate_limiter")


# ── Backtest replay fast-fail (execution-parity audit, §12-13) ──
# backtest/unified_engine.py sets this True before a replay run.
# When True, rate_limited_get() skips rate-limit waiting AND retry/
# backoff entirely and returns None on the first non-2xx response —
# every caller already degrades gracefully on None (see module
# docstrings across news_filter.py, fundamental/*, sentiment/*), so
# this only changes HOW FAST that fallback triggers, not what a
# backtest run's analysis output looks like. Does not affect Demo/Real
# in any way (default False, only ever set by the backtest entry
# point).
BACKTEST_REPLAY_MODE = False


class _ProviderState:
    """Per-provider rate-limit state."""

    __slots__ = ("min_interval", "max_retries", "_last_call", "_lock")

    def __init__(self, min_interval: float, max_retries: int):
        self.min_interval = min_interval
        self.max_retries = max_retries
        self._last_call = 0.0
        self._lock = threading.Lock()


# Registry of known providers + their free-tier limits.
# `min_interval` is the minimum number of seconds between consecutive
# calls to the same provider (enforced process-wide via a lock).
_PROVIDERS: Dict[str, _ProviderState] = {}
_PROVIDERS_LOCK = threading.Lock()


def register_provider(
    name: str,
    min_interval: float,
    max_retries: int = 3,
) -> None:
    """Register (or re-register) a provider's rate-limit parameters.

    Args:
        name: provider key (e.g. "twelve_data", "polygon")
        min_interval: minimum seconds between consecutive calls
            to this provider, process-wide. Sized to stay under the
            free-tier per-minute limit with a 20% safety margin.
            Examples:
              - twelve_data: 8 req/min → 60/8 = 7.5s, +20% → 9s
              - polygon: 5 req/min → 60/5 = 12s, +20% → 14.4s
              - alpha_vantage: 5 req/min → 14.4s
              - finnhub: 60 req/min → 1.2s
        max_retries: how many times to retry on 429/5xx before
            giving up.
    """
    with _PROVIDERS_LOCK:
        _PROVIDERS[name] = _ProviderState(min_interval, max_retries)
    log.debug(
        f"[api_rate_limiter] registered provider '{name}' "
        f"min_interval={min_interval}s max_retries={max_retries}"
    )


# ── Register known providers (free-tier limits, 20% safety margin) ──
register_provider("alpha_vantage",  14.4, max_retries=3)   # 5 req/min
register_provider("polygon",        14.4, max_retries=3)   # 5 req/min
register_provider("finnhub",         1.2, max_retries=3)   # 60 req/min
register_provider("twelve_data",     9.0, max_retries=3)   # 8 req/min


def _wait_for_slot(provider: str) -> None:
    """Block until enough time has elapsed since the last call to
    this provider, then claim the slot (update last-call timestamp).
    """
    with _PROVIDERS_LOCK:
        state = _PROVIDERS.get(provider)
    if state is None:
        return  # unregistered → no rate limiting

    with state._lock:
        now = time.monotonic()
        elapsed = now - state._last_call
        if elapsed < state.min_interval:
            wait = state.min_interval - elapsed
            log.debug(
                f"[api_rate_limiter] {provider}: sleeping {wait:.2f}s "
                f"to respect rate limit (min_interval={state.min_interval}s)"
            )
            time.sleep(wait)
            now = time.monotonic()
        state._last_call = now


def rate_limited_get(
    url: str,
    *,
    provider: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 15.0,
    default_headers: Optional[Dict[str, str]] = None,
) -> Optional[requests.Response]:
    """HTTP GET with per-provider rate limiting + retry on 429/5xx.

    Returns:
        requests.Response on success (HTTP 2xx).
        None on failure (after exhausting retries, or on non-retryable
        errors like 4xx other than 429).

    Side effects:
        - Sleeps as needed to respect the provider's min_interval.
        - Retries 429 and 5xx with exponential backoff (2^attempt
          seconds, capped at 30s). Honors Retry-After header on 429
          if present.

    Day 99+ FIX (Issue #5): adds browser-like User-Agent + Accept
    headers to bypass Cloudflare bot detection on provider endpoints
    (Polygon, Finnhub, Twelve Data, Alpha Vantage all sit behind
    Cloudflare). Also tries curl_cffi first if installed — its
    Chrome-impersonating TLS fingerprint defeats Cloudflare's
    "browser fingerprint" check that 403s Python's requests even
    with a browser User-Agent.
    """
    with _PROVIDERS_LOCK:
        state = _PROVIDERS.get(provider)
    max_retries = state.max_retries if state else 0

    if BACKTEST_REPLAY_MODE:
        # See module-level comment on BACKTEST_REPLAY_MODE. Skip the
        # provider rate-limit wait (replaying historical bars isn't a
        # live request burst against today's rate-limit window) and
        # collapse retries to a single attempt with no backoff sleep.
        state = None
        max_retries = 0

    # Day 99+ FIX (Issue #5): browser-like default headers.
    browser_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }
    merged_headers = browser_headers.copy()
    if default_headers:
        merged_headers.update(default_headers)
    if headers:
        merged_headers.update(headers)

    last_err: Optional[str] = None
    for attempt in range(max_retries + 1):
        # ── Rate-limit gate ───────────────────────────────────
        if state is not None:
            _wait_for_slot(provider)

        # ── Day 99+ FIX (Issue #5): try curl_cffi first ───────
        resp = None
        try:
            from curl_cffi import requests as _curl_requests  # type: ignore
            resp = _curl_requests.get(
                url, params=params, headers=merged_headers,
                timeout=timeout, impersonate="chrome120",
            )
        except ImportError:
            pass  # not installed — fall through to plain requests
        except Exception as e:
            log.debug(
                f"[api_rate_limiter] {provider} curl_cffi failed "
                f"({type(e).__name__}: {e}) — falling back to requests"
            )
            resp = None

        if resp is None:
            try:
                resp = requests.get(
                    url, params=params, headers=merged_headers,
                    timeout=timeout,
                )
            except requests.exceptions.RequestException as e:
                last_err = f"network error: {e}"
                log.warning(
                    f"[api_rate_limiter] {provider} attempt {attempt+1}/"
                    f"{max_retries+1}: {last_err}"
                )
                if attempt < max_retries:
                    backoff = min(2 ** attempt, 30)
                    time.sleep(backoff)
                    continue
                return None

        # ── Success ───────────────────────────────────────────
        if 200 <= resp.status_code < 300:
            return resp

        # ── Rate limited (429) ────────────────────────────────
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    sleep_s = float(retry_after)
                except ValueError:
                    sleep_s = min(2 ** attempt, 30)
            else:
                sleep_s = min(2 ** attempt, 30)
            log.warning(
                f"[api_rate_limiter] {provider} returned 429 "
                f"(attempt {attempt+1}/{max_retries+1}) — sleeping "
                f"{sleep_s:.1f}s before retry"
            )
            last_err = f"HTTP 429 (rate limited)"
            if attempt < max_retries:
                time.sleep(sleep_s)
                continue
            return resp  # caller can inspect .status_code

        # ── Server error (5xx) — retry ────────────────────────
        if 500 <= resp.status_code < 600:
            last_err = f"HTTP {resp.status_code}"
            log.warning(
                f"[api_rate_limiter] {provider} returned {resp.status_code} "
                f"(attempt {attempt+1}/{max_retries+1}) — retrying"
            )
            if attempt < max_retries:
                backoff = min(2 ** attempt, 30)
                time.sleep(backoff)
                continue
            return resp

        # ── Cloudflare 403 — special handling ─────────────────
        if resp.status_code == 403:
            last_err = "HTTP 403 (Cloudflare bot detection)"
            log.warning(
                f"[api_rate_limiter] {provider} returned 403 "
                f"(attempt {attempt+1}/{max_retries+1}) — likely Cloudflare "
                f"bot detection. Install curl_cffi (pip install curl_cffi) "
                f"for Chrome-grade TLS fingerprint emulation."
            )
            # 403 from Cloudflare is sometimes transient (rate-based
            # challenge); retry once with backoff before giving up.
            if attempt < max_retries:
                backoff = min(2 ** attempt, 30)
                time.sleep(backoff)
                continue
            return resp

        # ── Other 4xx — non-retryable ────────────────────────
        last_err = f"HTTP {resp.status_code}"
        log.warning(
            f"[api_rate_limiter] {provider} returned {resp.status_code} "
            f"(non-retryable): {resp.text[:200]}"
        )
        return resp

    log.error(
        f"[api_rate_limiter] {provider} exhausted {max_retries} retries "
        f"(last error: {last_err})"
    )
    return None