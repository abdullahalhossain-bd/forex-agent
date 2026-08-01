# fundamental/faireconomy_cache.py  —  Day 92 shared cache
# ============================================================
# FairEconomy JSON endpoint-এর জন্য process-wide singleton cache।
#
# সমস্যা (Day 92 log থেকে):
#   17:51:51  economic_calendar_api  | [FairEconomy] fetch failed: 429
#   17:51:52  news_filter            | [FairEconomy] fetch failed: 429
#
# দুটো module নিজস্ব _fetch_faireconomy() দিয়ে একই endpoint-ে
# একই সেকেন্ডে call করছিল → rate limit দ্বিগুণ হচ্ছিল।
#
# Fix: এই module-ে একটাই cache dict এবং একটাই fetch_faireconomy()
# function আছে। news_filter.py এবং economic_calendar_api.py দুটোই
# এই function import করে ব্যবহার করবে — তাই HTTP request একবারই হবে।
#
# Day 97+ FIXES:
#   #1: _fetching flag prevents cache stampede (5 threads → 5 HTTP requests → 429)
#   #2: Retry-After log now shows ACTUAL sleep time (was misleading)
#   #3: CACHE_TTL 60s → 300s (calendar doesn't change every minute)
# ============================================================

import time
import threading
from datetime import datetime

import pytz
import requests

from utils.logger import get_logger

log = get_logger("faireconomy_cache")

FAIRECONOMY_URL     = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
# Day 97+ FIX #3: 60s → 300s. Economic calendar doesn't change every minute;
# 5 minutes is safe and reduces API calls by 5x.
CACHE_TTL_SECONDS   = 300.0  # 5 minutes

# Day 98+ FIX: canonical keyword list, single source of truth.
# Previously news_filter.py and economic_calendar_api.py each kept their
# own copy of this exact list — correct today only because they happen to
# match; a future edit to one and not the other would have silently
# diverged. Both modules now import this instead of defining their own.
DEFAULT_HIGH_IMPACT_KEYWORDS = [
    "Non-Farm", "NFP", "CPI", "Interest Rate", "FOMC",
    "GDP", "Unemployment", "Retail Sales", "Fed Chair",
    "ECB", "BOE", "BOJ", "Inflation", "PMI Flash",
]

# ── Singleton cache ───────────────────────────────────────────
_cache_lock = threading.Lock()
_cache: dict = {
    "data":       None,   # list[dict] | None
    "fetched_at": 0.0,    # time.monotonic() timestamp
}

# Day 97+ FIX #1: _fetching flag prevents cache stampede.
# Without this, multiple threads hitting cache-miss simultaneously would all
# fire HTTP requests → 429 Too Many Requests.
_fetching = False

# Day 132 fix: _fetching only stops threads that overlap in the SAME
# instant. It does nothing for the far more common case in production —
# one failed fetch releases the flag, and the very next call (a different
# symbol's cycle, a fraction of a second to ~30s later) sees a cache miss
# again and fires a brand-new HTTP request, immediately drawing another
# 429. Production logs showed exactly this: 429 at :24, :25, then :52 —
# less than 30s apart, even though the server said "wait 300s" / "wait
# 273s" each time. `_blocked_until` is a monotonic deadline, set from the
# server's own Retry-After header (falling back to a fixed cooldown if
# absent/unparseable), that every caller must respect before attempting a
# new fetch — regardless of how much wall-clock time has passed since the
# _fetching flag was released.
_blocked_until = 0.0
_DEFAULT_BLOCK_SECONDS = 60.0  # fallback cooldown if Retry-After is missing/unparseable


def _apply_filters(raw_events: list, watched_currencies: set, high_impact_keywords: list) -> list:
    """Apply currency filtering + high_impact classification to already-
    parsed events. Kept OUT of the cached data itself (see Day 98+ FIX
    below) so the shared cache stays correct for any caller regardless of
    which currencies/keywords they ask for.

    Day 99+ FIX (P1): the original impact level from the feed is now
    preserved in the "impact" field of each returned event (previously
    dropped here, forcing callers to guess "MEDIUM" for anything that
    wasn't high). Callers that only care about the boolean high_impact
    flag are unaffected; callers that want to distinguish MEDIUM from
    LOW (e.g. for AI context / log fidelity) can now read it directly.
    """
    result = []
    for item in raw_events:
        currency = item.get("currency", "")
        if currency not in watched_currencies:
            continue
        title = item.get("title", "")
        raw_impact = (item.get("impact", "") or "").strip().lower()
        # Preserve original logic: trust the feed's own "impact" field
        # first, fall back to keyword matching only if that's missing/low.
        is_high = raw_impact == "high"
        if not is_high:
            is_high = any(kw.lower() in title.lower() for kw in high_impact_keywords)
        # Preserve the original impact string so downstream callers can
        # distinguish MEDIUM from LOW instead of being forced to "MEDIUM".
        if raw_impact in ("high", "medium", "low"):
            preserved_impact = raw_impact.upper()
        elif is_high:
            preserved_impact = "HIGH"
        else:
            preserved_impact = "LOW"
        result.append({
            "title":       title,
            "currency":    currency,
            "high_impact": is_high,
            "impact":      preserved_impact,
            "time":        item.get("time"),
        })
    return result


def fetch_faireconomy(
    watched_currencies: set,
    high_impact_keywords: list = None,
) -> list:
    """
    FairEconomy JSON feed থেকে economic events fetch করে।
    একই process-ে যেকোনো module থেকে call হোক — TTL (300s) শেষ না হওয়া
    পর্যন্ত শুধু প্রথম call-ই HTTP request করবে, বাকিরা cache পাবে।

    Day 97+ FIX #1: _fetching flag prevents cache stampede — if one thread
    is already fetching, other threads return stale cache (or []) instead
    of firing duplicate HTTP requests.

    Day 98+ FIX: the shared cache now stores ALL currencies, unfiltered
    and unclassified. Previously the cache stored the FIRST caller's
    currency-filtered, keyword-classified result — so a second caller
    passing a different watched_currencies set (e.g. a single-pair lookup
    for a currency the first caller didn't request) would silently get
    back events filtered for someone else's request. Filtering and
    high_impact classification now happen per-call, after the cache
    lookup, using each caller's own arguments. high_impact_keywords is
    now optional and defaults to the shared DEFAULT_HIGH_IMPACT_KEYWORDS.

    Returns: list of event dicts, filtered to watched_currencies.
    """
    global _cache, _fetching, _blocked_until

    if high_impact_keywords is None:
        high_impact_keywords = DEFAULT_HIGH_IMPACT_KEYWORDS

    now_mono = time.monotonic()

    # ── Fast path: cache hit ──────────────────────────────────
    with _cache_lock:
        age = now_mono - _cache["fetched_at"]
        if _cache["data"] is not None and age < CACHE_TTL_SECONDS:
            log.debug(f"[FairEconomy] cache hit (age={age:.1f}s) — skipping HTTP request")
            return _apply_filters(_cache["data"], watched_currencies, high_impact_keywords)

        # Day 132 fix: honor the server-provided cooldown from a previous
        # 429/failure. This is checked BEFORE the stampede flag because it
        # applies across time, not just to overlapping in-flight calls.
        if now_mono < _blocked_until:
            remaining = _blocked_until - now_mono
            if _cache["data"] is not None:
                log.debug(
                    f"[FairEconomy] in cooldown ({remaining:.0f}s remaining) — "
                    f"returning stale cache instead of re-fetching"
                )
                return _apply_filters(_cache["data"], watched_currencies, high_impact_keywords)
            else:
                log.debug(
                    f"[FairEconomy] in cooldown ({remaining:.0f}s remaining), "
                    f"no cache available — returning []"
                )
                return []

        # Day 97+ FIX #1: cache stampede prevention
        # If another thread is already fetching, DON'T fire another request.
        # Return stale cache if available, else empty list.
        if _fetching:
            if _cache["data"] is not None:
                log.debug("[FairEconomy] cache miss but fetch in progress — returning stale cache")
                return _apply_filters(_cache["data"], watched_currencies, high_impact_keywords)
            else:
                log.debug("[FairEconomy] cache miss, fetch in progress, no stale cache — returning []")
                return []

        # Claim the fetch — other threads will now see _fetching=True
        _fetching = True

    # ── Cache miss: HTTP fetch (only ONE thread reaches here) ────
    try:
        # Day 97+ FIX #2: retry with backoff, but log the ACTUAL sleep time
        # (was logging server's Retry-After but capping sleep at 30s — misleading)
        raw = None
        last_err = None
        for attempt in range(3):
            try:
                resp = requests.get(FAIRECONOMY_URL, timeout=10)
                if resp.status_code == 429:
                    # Round-19 fix: DO NOT block-sleep on 429. The cache's
                    # entire purpose is to avoid re-fetching within TTL
                    # (300s) — sleeping 30-90s here in the main pipeline
                    # thread defeats that and stalls every symbol's cycle
                    # behind this one HTTP call. Fall through to the
                    # stale-cache fallback immediately instead; the next
                    # natural fetch (when TTL expires) will simply retry.
                    retry_after_raw = resp.headers.get("Retry-After", "unknown")
                    try:
                        block_seconds = float(retry_after_raw)
                    except (TypeError, ValueError):
                        block_seconds = _DEFAULT_BLOCK_SECONDS
                    with _cache_lock:
                        _blocked_until = time.monotonic() + block_seconds
                    log.warning(
                        f"[FairEconomy] 429 Too Many Requests — "
                        f"server says wait {retry_after_raw}s — "
                        f"NOT sleeping this thread, but blocking ALL callers "
                        f"from re-fetching for {block_seconds:.0f}s; falling "
                        f"back to stale cache immediately for this call"
                    )
                    last_err = "HTTP 429 rate-limited"
                    raw = None
                    break
                resp.raise_for_status()
                raw = resp.json()
                break
            except requests.exceptions.HTTPError as he:
                if resp.status_code >= 500 and attempt < 2:
                    backoff = 2 ** attempt
                    log.warning(
                        f"[FairEconomy] HTTP {resp.status_code} — "
                        f"sleep={backoff}s (attempt {attempt+1}/3)"
                    )
                    time.sleep(backoff)
                    last_err = f"HTTP {resp.status_code}"
                    continue
                raise
            except requests.exceptions.RequestException as re:
                if attempt < 2:
                    backoff = 2 ** attempt
                    log.warning(
                        f"[FairEconomy] network error: {re} — "
                        f"sleep={backoff}s (attempt {attempt+1}/3)"
                    )
                    time.sleep(backoff)
                    last_err = str(re)
                    continue
                raise

        if raw is None:
            log.warning(f"[FairEconomy] fetch failed after retries: {last_err}")
            with _cache_lock:
                _fetching = False  # release flag
                # If this failure wasn't a 429 (which already set its own
                # cooldown above), still impose a default cooldown so a
                # non-429 failure (network error, 5xx) doesn't get retried
                # by the next caller a fraction of a second later either.
                if time.monotonic() >= _blocked_until:
                    _blocked_until = time.monotonic() + _DEFAULT_BLOCK_SECONDS
                if _cache["data"] is not None:
                    log.warning("[FairEconomy] using stale cache as fallback")
                    return _apply_filters(_cache["data"], watched_currencies, high_impact_keywords)
            return []

        # Day 98+ FIX: parse ALL currencies here, with NO currency filter
        # and NO high_impact classification — those are caller-specific
        # and now applied per-call via _apply_filters(), not baked into
        # the shared cache. This is what's stored in _cache["data"].
        #
        # Day 99+ CRITICAL FIX (P0): the FairEconomy JSON feed (mirrored
        # from Forex Factory) provides a "currency" field (e.g. "USD",
        # "EUR"), NOT a "country" field. Previously this read
        # item.get("country", "") which always returned "" for every
        # event, so _apply_filters() (which filters on currency) dropped
        # 100% of events. The entire FairEconomy primary source was
        # silently dead — every cycle fell through to Tradermade/Finnhub/
        # scraper, doubling API cost and latency. Reading the correct
        # field name restores the primary source.
        parsed = []
        for item in raw:
            currency = item.get("currency", "").upper()
            title    = item.get("title", "")
            impact   = item.get("impact", "")

            date_str = item.get("date", "")
            try:
                dt = datetime.fromisoformat(date_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=pytz.utc)
                utc_dt = dt.astimezone(pytz.utc)
            except Exception:
                continue

            parsed.append({
                "title":    title,
                "currency": currency,
                "impact":   impact,
                "time":     utc_dt,
            })

        # ── Update cache + release fetch flag ─────────────────
        with _cache_lock:
            _cache["data"]       = parsed
            _cache["fetched_at"] = time.monotonic()
            _fetching = False  # Day 97+ FIX #1: release flag
            _blocked_until = 0.0  # Day 132 fix: clear any prior cooldown on success

        log.info(f"[FairEconomy] fetched {len(parsed)} events across all currencies (cache updated)")
        return _apply_filters(parsed, watched_currencies, high_impact_keywords)

    except Exception as e:
        log.warning(f"[FairEconomy] fetch failed: {e}")
        with _cache_lock:
            _fetching = False  # Day 97+ FIX #1: release flag on error too
            # Day 132 fix: same reasoning as the raw-is-None path above —
            # an unhandled exception shouldn't leave the door open for the
            # very next caller to retry immediately.
            if time.monotonic() >= _blocked_until:
                _blocked_until = time.monotonic() + _DEFAULT_BLOCK_SECONDS
            if _cache["data"] is not None:
                log.warning("[FairEconomy] using stale cache as fallback")
                return _apply_filters(_cache["data"], watched_currencies, high_impact_keywords)
        return []