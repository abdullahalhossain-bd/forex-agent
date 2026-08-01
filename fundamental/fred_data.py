"""
fundamental/fred_data.py — Day 94 FRED API (Federal Reserve Economic Data)
=========================================================================
Pulls macro-economic data directly from the St. Louis Fed's FRED database.
This is the OFFICIAL source for US economic indicators.

Free tier: unlimited requests with free API key (no daily cap).
Get key: https://fredaccount.stlouisfed.org/apikeys

Series we track (configurable via env):
  CPIAUCSL     — Consumer Price Index (US inflation)
  UNRATE       — Unemployment Rate
  DGS10        — 10-Year Treasury Yield
  DGS2         — 2-Year Treasury Yield
  T10Y2Y       — 10Y-2Y Yield Spread (recession indicator)
  FEDFUNDS     — Federal Funds Rate (current interest rate)
  DEXUSEU      — USD/EUR exchange rate (sanity check)
  VIXCLS       — VIX (volatility index)

Usage:
    from fundamental.fred_data import FREDApi
    fred = FREDApi()
    data = fred.get_macro_snapshot()
    # data = {"CPI": {"value":314.4, "date":"2024-12-01"}, "UNRATE": {...}, ...}

    # Or single series:
    cpi = fred.get_series("CPIAUCSL")
"""
from __future__ import annotations

import os
import time
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests

from utils.logger import get_logger

log = get_logger("fred_api")

# Day 98+ FIX: FRED series (CPI, unemployment, Fed funds rate, etc.) update
# at most daily, most monthly or less. Every other module in this layer
# (see faireconomy_cache.py) already learned not to re-fetch unchanging
# data every cycle; fred_data.py previously had no caching at all, meaning
# a Decision Layer polling get_macro_snapshot() every cycle would fire up
# to 8 HTTP requests per cycle for data that hadn't changed since the last
# call. 1 hour is a safe TTL — comfortably fresher than the underlying
# data's own update cadence, while cutting redundant calls dramatically.
#
# Day 99+ FIX: FRED's own update cadence varies by series. Daily series
# (yields, VIX) genuinely change every business day, so a 1h TTL is right.
# But monthly series (CPI, Unemployment, Fed Funds Rate) update at most
# once a month — caching them for only 1h still means ~720 redundant
# calls per month for data that won't change. We now keep a per-series
# TTL table; monthly/quarterly series get a 12h TTL (still 60x fresher
# than the data itself), daily series stay at 1h. The stale-while-
# revalidate fallback below ensures a long outage never serves truly
# ancient data without the caller knowing about it.
_SERIES_CACHE_TTL_SECONDS = 3600.0  # 1 hour default (daily-frequency series)

# Per-series TTL overrides — keyed by series_id. FRED publishes the
# update frequency for each series; we use that to size the cache TTL.
# Anything not listed here falls back to _SERIES_CACHE_TTL_SECONDS.
_SERIES_TTL_OVERRIDE: Dict[str, float] = {
    "CPIAUCSL": 12 * 3600.0,   # CPI — monthly
    "UNRATE":   12 * 3600.0,   # Unemployment — monthly
    "FEDFUNDS": 12 * 3600.0,   # Fed Funds Rate — monthly (changes only at FOMC)
    "T10Y2Y":   6 * 3600.0,    # 10Y-2Y spread — daily, but slow-moving signal
    "DEXUSEU":  1 * 3600.0,    # USD/EUR — daily
    "DGS10":    1 * 3600.0,    # 10Y yield — daily
    "DGS2":     1 * 3600.0,    # 2Y yield — daily
    "VIXCLS":   1 * 3600.0,    # VIX — daily (but very volatile, keep short)
}

# Hard ceiling on how old a stale cached value can be before we refuse to
# serve it (and return None instead). Prevents an outage that lasts days
# from serving month-old data silently. Caller can detect via None and
# fall back to its own degrade path.
_STALE_MAX_AGE_SECONDS = 7 * 24 * 3600.0  # 7 days

# ── Per-series cache: {series_id: {"data": dict|None, "fetched_at": float}} ──
_series_cache_lock = threading.Lock()
_series_cache: Dict[str, Dict[str, Any]] = {}


# ── Tracked FRED series ──────────────────────────────────────────
# Each entry: series_id -> (label, category, description)
TRACKED_SERIES = {
    "CPIAUCSL":  ("CPI",              "inflation",   "Consumer Price Index (US inflation)"),
    "UNRATE":    ("Unemployment",     "labor",       "US Unemployment Rate (%)"),
    "DGS10":     ("10Y Yield",        "rates",       "10-Year Treasury Yield (%)"),
    "DGS2":      ("2Y Yield",         "rates",       "2-Year Treasury Yield (%)"),
    "T10Y2Y":    ("10Y-2Y Spread",    "rates",       "Yield curve spread (recession indicator)"),
    "FEDFUNDS":  ("Fed Funds Rate",   "rates",       "Federal Funds Rate (%)"),
    "DEXUSEU":   ("USD/EUR",          "fx",          "USD to EUR exchange rate"),
    "VIXCLS":    ("VIX",              "volatility",  "CBOE Volatility Index"),
}


class FREDApi:
    """FRED API client for macro-economic data."""

    BASE_URL = "https://api.stlouisfed.org/fred"

    def __init__(self) -> None:
        self._api_key = os.getenv("FRED_API_KEY", "").strip()
        # Day 99+ FIX: validate presence of the key once at construction
        # so a misconfigured deployment is loud, not silent. Each consumer
        # of FRED data (snapshot, surprise, sentiment) checks `.available`
        # anyway, but a missing key used to log nothing until the first
        # call — by then several cycles had silently run with no macro data.
        if not self._api_key:
            log.warning(
                "[FRED] FRED_API_KEY is not set — macro-economic data will be "
                "unavailable. Get a free key at "
                "https://fredaccount.stlouisfed.org/apikeys"
            )

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    # ─────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────

    def get_macro_snapshot(self) -> Dict[str, Any]:
        """Fetch latest value of all tracked series in one call.

        Returns:
            {
              "series": {
                "CPI":          {"value": 314.4, "date": "2024-12-01", "change_pct": 0.3},
                "Unemployment": {"value": 4.1,   "date": "2024-12-01", "change_pct": 0.0},
                ...
              },
              "yield_curve":    "inverted" | "normal" | "flat" | "unknown",
              "inflation_trend":"rising" | "falling" | "stable",
              "rate_environment":"hawkish" | "dovish" | "neutral",
              "fetched_at":     ISO timestamp,
              "source":         "fred_live" | "fred_partial" | "none",
            }
        """
        if not self.available:
            return self._empty_result("FRED_API_KEY not set")

        series_data = {}
        success_count = 0
        for series_id, (label, category, desc) in TRACKED_SERIES.items():
            data = self.get_series(series_id)
            if data:
                series_data[label] = data
                success_count += 1

        if success_count == 0:
            return self._empty_result("All FRED series failed")

        # Compute derived indicators
        yield_curve = self._analyze_yield_curve(series_data)
        inflation_trend = self._analyze_inflation(series_data)
        rate_env = self._analyze_rate_environment(series_data, inflation_trend)

        result = {
            "series":           series_data,
            "yield_curve":      yield_curve,
            "inflation_trend":  inflation_trend,
            "rate_environment": rate_env,
            "fetched_at":       datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source":           "fred_live" if success_count == len(TRACKED_SERIES) else "fred_partial",
        }
        log.info(
            f"[FRED] {success_count}/{len(TRACKED_SERIES)} series | "
            f"yield_curve={yield_curve} | inflation={inflation_trend} | "
            f"rates={rate_env}"
        )
        return result

    def get_series(self, series_id: str) -> Optional[Dict[str, Any]]:
        """Fetch latest value of a single FRED series.

        Day 98+ FIX: now cached (1h TTL, matches the discipline already
        used elsewhere in this layer) and retries transient failures with
        backoff instead of giving up after a single attempt.

        Day 99+ FIX: per-series TTL (see _SERIES_TTL_OVERRIDE). Monthly
        series like CPI / Unemployment / FedFunds are cached for 12h
        instead of 1h — they only update monthly, so caching them for
        1h was 720x more aggressive than necessary. Stale fallback is
        also now bounded by _STALE_MAX_AGE_SECONDS so a multi-day outage
        can't silently serve week-old data.

        Returns: {"value": float, "date": "YYYY-MM-DD", "change_pct": float}
                 or None on failure.
        """
        if not self.available:
            return None

        ttl = _SERIES_TTL_OVERRIDE.get(series_id, _SERIES_CACHE_TTL_SECONDS)

        # ── Cache check ────────────────────────────────────────
        with _series_cache_lock:
            entry = _series_cache.get(series_id)
            if entry is not None:
                age = time.monotonic() - entry["fetched_at"]
                if age < ttl:
                    log.debug(
                        f"[FRED] {series_id} cache hit (age={age:.0f}s, ttl={ttl:.0f}s)"
                    )
                    return entry["data"]

        # ── Cache miss: fetch with retry/backoff ─────────────────
        url = f"{self.BASE_URL}/series/observations"
        params = {
            "series_id":  series_id,
            "api_key":    self._api_key,
            "file_type":  "json",
            "sort_order": "desc",
            "limit":      2,  # latest + previous for change calc
        }

        data = None
        last_err = None
        for attempt in range(3):
            try:
                resp = requests.get(url, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                break
            except requests.exceptions.RequestException as e:
                last_err = e
                if attempt < 2:
                    backoff = 2 ** attempt
                    log.debug(
                        f"[FRED] {series_id} attempt {attempt+1}/3 failed: {e} — "
                        f"retrying in {backoff}s"
                    )
                    time.sleep(backoff)
                    continue

        if data is None:
            log.debug(f"[FRED] {series_id} failed after retries: {last_err}")
            # Stale-while-revalidate fallback. Day 99+ FIX: enforce a hard
            # age ceiling — if the last good cached value is older than
            # _STALE_MAX_AGE_SECONDS, we refuse to serve it (return None)
            # instead of pretending the data is current. The caller's
            # own degrade path is better than silently using month-old
            # numbers during a long outage.
            with _series_cache_lock:
                stale = _series_cache.get(series_id)
                if stale is not None and stale["data"] is not None:
                    stale_age = time.monotonic() - stale["fetched_at"]
                    if stale_age <= _STALE_MAX_AGE_SECONDS:
                        log.warning(
                            f"[FRED] {series_id} using stale cached value as fallback "
                            f"(age={stale_age:.0f}s, max={_STALE_MAX_AGE_SECONDS:.0f}s)"
                        )
                        return stale["data"]
                    else:
                        log.warning(
                            f"[FRED] {series_id} stale cache is too old "
                            f"(age={stale_age:.0f}s > max={_STALE_MAX_AGE_SECONDS:.0f}s) "
                            f"— refusing to serve, returning None"
                        )
            return None

        observations = data.get("observations", [])
        if not observations:
            return None

        latest = observations[0]
        value_str = latest.get("value", ".")
        if value_str == ".":
            return None  # FRED uses "." for missing data

        try:
            value = float(value_str)
        except ValueError:
            return None

        date = latest.get("date", "")
        change_pct = 0.0
        if len(observations) > 1:
            try:
                prev_value = float(observations[1].get("value", "."))
                if prev_value != 0:
                    change_pct = round((value - prev_value) / prev_value * 100, 3)
            except (ValueError, ZeroDivisionError):
                pass

        result = {
            "value":      value,
            "date":       date,
            "change_pct": change_pct,
        }

        with _series_cache_lock:
            _series_cache[series_id] = {"data": result, "fetched_at": time.monotonic()}

        return result

    # ─────────────────────────────────────────────────────────
    # Derived analysis
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _analyze_yield_curve(series: Dict) -> str:
        """Inverted yield curve (2Y > 10Y) is a recession signal."""
        y10 = series.get("10Y Yield", {}).get("value")
        y2  = series.get("2Y Yield", {}).get("value")
        if y10 is None or y2 is None:
            return "unknown"
        spread = y10 - y2
        if spread < -0.2:
            return "inverted"   # recession warning
        if spread > 0.5:
            return "normal"
        return "flat"

    @staticmethod
    def _analyze_inflation(series: Dict) -> str:
        """CPI trend — rising/falling/stable."""
        cpi = series.get("CPI", {})
        change = cpi.get("change_pct", 0)
        if change > 0.5:
            return "rising"
        if change < -0.3:
            return "falling"
        return "stable"

    @staticmethod
    def _analyze_rate_environment(series: Dict, inflation_trend: str) -> str:
        """Hawkish (raising rates) vs dovish (cutting rates)."""
        fed_rate = series.get("Fed Funds Rate", {}).get("value")
        if fed_rate is None:
            return "neutral"
        # Simplified: high rate + rising inflation = hawkish
        # Low rate + falling inflation = dovish
        if fed_rate > 4.0 and inflation_trend == "rising":
            return "hawkish"
        if fed_rate < 2.0 or inflation_trend == "falling":
            return "dovish"
        return "neutral"

    # ─────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _empty_result(reason: str) -> Dict[str, Any]:
        return {
            "series":           {},
            "yield_curve":      "unknown",
            "inflation_trend":  "stable",
            "rate_environment": "neutral",
            "fetched_at":       datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source":           "none",
            "reason":           reason,
        }

    def get_ai_context(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Compact context for MasterAnalyst."""
        s = result.get("series", {})
        return {
            "fred_source":          result.get("source", "none"),
            "fred_yield_curve":     result.get("yield_curve", "unknown"),
            "fred_inflation_trend": result.get("inflation_trend", "stable"),
            "fred_rate_env":        result.get("rate_environment", "neutral"),
            "fred_cpi":             s.get("CPI", {}).get("value"),
            "fred_unemployment":    s.get("Unemployment", {}).get("value"),
            "fred_fed_rate":        s.get("Fed Funds Rate", {}).get("value"),
            "fred_10y_yield":       s.get("10Y Yield", {}).get("value"),
            "fred_vix":             s.get("VIX", {}).get("value"),
        }

    def print_summary(self, result: Dict[str, Any]) -> None:
        bar = "═" * 50
        log.info(bar)
        log.info("  🏛️  FRED MACRO DATA  (Day 94)")
        log.info(bar)
        log.info(f"  Source         : {result.get('source','?')}")
        log.info(f"  Yield curve    : {result.get('yield_curve','?')}")
        log.info(f"  Inflation      : {result.get('inflation_trend','?')}")
        log.info(f"  Rate env       : {result.get('rate_environment','?')}")
        for label, data in result.get("series", {}).items():
            log.info(f"  {label:<16}: {data['value']}  ({data['date']}, {data['change_pct']:+.2f}%)")
        log.info(bar)


# ── Singleton ────────────────────────────────────────────────────

_INSTANCE: Optional[FREDApi] = None


def get_fred_api() -> FREDApi:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = FREDApi()
    return _INSTANCE