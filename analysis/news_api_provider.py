"""
analysis/news_api_provider.py — Day 92 NewsAPI.org integration
================================================================
Pulls real-time financial news headlines from 80,000+ sources
(Bloomberg, Reuters, Yahoo Finance, WSJ, etc.) via NewsAPI.org.

Free tier:
  - 100 requests/day
  - 1-day delay (articles published yesterday onwards)
  - Developer license (no commercial use)

Why this matters:
  Forex Factory gives us SCHEDULED high-impact events (CPI, NFP, FOMC).
  But markets also move on BREAKING news — a Fed official's surprise
  speech, a geopolitical escalation, an unexpected central-bank
  statement. NewsAPI surfaces this kind of unscheduled news so the
  AI can factor in real-time sentiment shifts.

Output shape (compatible with existing news_ctx used by MasterAnalyst):
    {
      "trade_allowed":      bool,
      "reason":             str,
      "news_bias":          "BULLISH" | "BEARISH" | "NEUTRAL",
      "news_score":         -100 to +100,
      "headline_count":     int,
      "top_headlines":      [{"title","source","published_at","sentiment"}],
      "currency_filtered":  bool,
      "source":             "newsapi_live" | "newsapi_cached" | "fallback",
    }

Sentiment scoring:
  Each headline is keyword-classified into BULLISH / BEARISH for the
  pair's two currencies (e.g. EURUSD → check headlines that mention
  EUR + USD, score based on bullish/bearish keywords per currency).
  This is intentionally simple — for production-grade sentiment we'd
  run each headline through the LLM, but that doubles token usage.
  The keyword approach gives a "quick-and-dirty" sentiment that's
  good enough to flag risk windows where the AI should slow down.

Usage:
    from analysis.news_api_provider import NewsAPIProvider
    provider = NewsAPIProvider()
    result = provider.fetch_headlines_for_pair("EURUSD")
    if result["news_bias"] == "BEARISH" and result["news_score"] < -40:
        # AI should be more cautious on EURUSD longs
        ...
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests

from utils.logger import get_logger

log = get_logger("news_api_provider")


# ── Keyword-based sentiment lexicon ──────────────────────────────
# Simple but effective — classifies headlines by scanning for these
# terms. Each bullish hit adds +1, each bearish hit adds -1, then
# we normalize to a -100..+100 score.

BULLISH_KEYWORDS = [
    "rally", "surge", "jump", "soar", "gain", "rise", "boost",
    "bullish", "uptrend", "breakout", "optimism", "strong",
    "beat estimates", "exceeds expectations", "upgrade",
    "recovery", "growth", "expansion", "hawkish", "rate hike",
    "support", "demand", "safe haven inflow",
]

BEARISH_KEYWORDS = [
    "plunge", "crash", "drop", "fall", "slide", "tumble",
    "bearish", "downtrend", "breakdown", "pessimism", "weak",
    "miss estimates", "disappoints", "downgrade", "recession",
    "contraction", "dovish", "rate cut", "risk-off", "risk off",
    "selloff", "sell-off", "fear", "panic", "uncertainty",
    "geopolitical tension", "sanctions", "tariff", "trade war",
]


class NewsAPIProvider:
    """NewsAPI.org client with currency-aware sentiment scoring."""

    BASE_URL = "https://newsapi.org/v2"
    # Cache results for 15 minutes to avoid burning the 100-req/day quota
    CACHE_TTL_SEC = 15 * 60
    # Look back this many hours for headlines (24h is a good balance)
    LOOKBACK_HOURS = 24
    # Free-tier daily request cap (NewsAPI.org). Tracked locally so quota
    # exhaustion is visible in logs/results instead of silently degrading
    # to the neutral fallback with no signal that anything is wrong.
    DAILY_REQUEST_LIMIT = 100

    def __init__(self):
        self._api_key = os.getenv("NEWSAPI_API_KEY", "").strip()
        self._cache: Dict[str, tuple] = {}  # cache_key -> (timestamp, data)
        self._last_call_ts = 0.0
        self._daily_request_count = 0
        self._daily_count_date = datetime.now(timezone.utc).date()

    def _check_quota(self) -> bool:
        """Track/reset the local daily request counter.

        Returns True if a live API call is allowed, False if the local
        estimate of the free-tier daily cap has been reached. This is a
        local estimate (resets at UTC midnight) — it does not query
        NewsAPI's actual remaining quota, but it turns a previously
        silent "every call fails after quota runs out" failure mode
        into a loud, logged, and reportable one.
        """
        today = datetime.now(timezone.utc).date()
        if today != self._daily_count_date:
            self._daily_count_date = today
            self._daily_request_count = 0

        if self._daily_request_count >= self.DAILY_REQUEST_LIMIT:
            log.warning(
                f"[NewsAPI] Daily request quota reached "
                f"({self._daily_request_count}/{self.DAILY_REQUEST_LIMIT}) — "
                f"falling back to NEUTRAL for remaining requests today. "
                f"News-based filtering is now effectively disabled until "
                f"UTC midnight; consider a higher-tier plan or a longer "
                f"CACHE_TTL_SEC if this happens routinely."
            )
            return False
        return True

    @property
    def available(self) -> bool:
        """True if NEWSAPI_API_KEY is configured."""
        return bool(self._api_key)

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────

    def fetch_headlines_for_pair(self, pair: str) -> Dict[str, Any]:
        """Fetch + score news headlines relevant to a forex pair.

        Args:
            pair: e.g. "EURUSD", "GBPUSD", "XAUUSD"

        Returns: dict with news_bias, news_score, top_headlines, etc.
                 If NewsAPI is unavailable (no key, quota exhausted, or the
                 request errors), falls back to scoring the free RSS feeds
                 in intelligence/news_sources.py before giving up to a flat
                 NEUTRAL — RSS has no key/quota, so a 429 from NewsAPI no
                 longer means "zero sentiment signal for the rest of today".
        """
        if not self.available:
            return self._rss_fallback(pair, "NEWSAPI_API_KEY not set")

        # Currency extraction: EURUSD → ["EUR", "USD"]
        currencies = self._extract_currencies(pair)
        cache_key = f"{pair}:{'+'.join(currencies)}"

        # Cache check
        cached = self._cache.get(cache_key)
        if cached and (time.time() - cached[0]) < self.CACHE_TTL_SEC:
            log.debug(f"[NewsAPI] cache hit for {pair}")
            result = dict(cached[1])
            result["source"] = "newsapi_cached"
            return result

        # Quota check — only relevant for a live call (cache hits above are free)
        if not self._check_quota():
            result = self._rss_fallback(pair, "Daily NewsAPI request quota reached")
            result["quota_exhausted"] = True
            return result

        # Build query — search for either currency
        # NewsAPI 'everything' endpoint supports OR / AND in q param
        query = " OR ".join(currencies)
        from_date = (datetime.now(timezone.utc) - timedelta(hours=self.LOOKBACK_HOURS)).strftime("%Y-%m-%dT%H:%M:%S")
        # Sort by relevancy so we get the most on-topic stories first
        params = {
            "q": query,
            "from": from_date,
            "sortBy": "relevancy",
            "language": "en",
            "pageSize": 30,
            "apiKey": self._api_key,
        }

        try:
            # Rate-limit: min 1s between calls
            elapsed = time.time() - self._last_call_ts
            if elapsed < 1.0:
                time.sleep(1.0 - elapsed)

            log.info(
                f"[NewsAPI] fetching headlines for {pair} (q={query!r}) "
                f"[{self._daily_request_count + 1}/{self.DAILY_REQUEST_LIMIT} today]"
            )
            resp = requests.get(
                f"{self.BASE_URL}/everything",
                params=params,
                timeout=15,
            )
            self._last_call_ts = time.time()
            self._daily_request_count += 1
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.warning(f"[NewsAPI] fetch failed: {e}")
            return self._rss_fallback(pair, f"NewsAPI fetch error: {e}")

        if data.get("status") != "ok":
            log.warning(f"[NewsAPI] bad status: {data.get('status')} | {data.get('message','')}")
            return self._rss_fallback(pair, f"NewsAPI status: {data.get('status')}")

        articles = data.get("articles", [])
        if not articles:
            log.info(f"[NewsAPI] no articles found for {pair}")
            return self._neutral_result("No recent headlines for this pair")

        # Score + filter to relevant articles
        scored = []
        for art in articles:
            title = art.get("title", "") or ""
            # Filter: must mention at least one of the pair's currencies
            if not any(c.lower() in title.lower() for c in currencies):
                continue
            sentiment, score = self._score_headline(title)
            scored.append({
                "title":        title,
                "source":       art.get("source", {}).get("name", "unknown"),
                "published_at": art.get("publishedAt", ""),
                "url":          art.get("url", ""),
                "sentiment":    sentiment,
                "score":        score,
            })

        if not scored:
            log.info(f"[NewsAPI] {len(articles)} articles found but 0 mention {currencies}")
            return self._neutral_result("Articles found but none mention this pair's currencies")

        # Aggregate: sum scores, normalize to -100..+100
        total_score = sum(s["score"] for s in scored)
        # Normalize: if all headlines agreed, score would be ±len(scored)
        max_possible = max(1, len(scored))
        normalized = int((total_score / max_possible) * 100)

        # Bias label
        if normalized >= 25:
            bias = "BULLISH"
        elif normalized <= -25:
            bias = "BEARISH"
        else:
            bias = "NEUTRAL"

        # Trade-block threshold: very strong negative sentiment
        trade_allowed = normalized > -50

        result = {
            "trade_allowed":      trade_allowed,
            "reason":             f"{len(scored)} relevant headlines | bias={bias} score={normalized}",
            "news_bias":          bias,
            "news_score":         normalized,
            "headline_count":     len(scored),
            "top_headlines":      scored[:5],   # top 5 by relevancy order
            "currency_filtered":  True,
            "source":             "newsapi_live",
            "currencies_checked": currencies,
            "fetched_at":         datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        # Cache it
        self._cache[cache_key] = (time.time(), result)
        log.info(
            f"[NewsAPI] {pair} | {len(scored)} headlines | "
            f"bias={bias} score={normalized} | trade_allowed={trade_allowed}"
        )
        return result

    # ─────────────────────────────────────────────────────────
    # Sentiment scoring
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _score_headline(title: str) -> tuple[str, int]:
        """Keyword-based sentiment classification.

        Returns (label, score) where:
          - label ∈ {"BULLISH", "BEARISH", "NEUTRAL"}
          - score ∈ {-1, 0, +1}  (per-headline)
        """
        t = title.lower()
        bull_hits = sum(1 for kw in BULLISH_KEYWORDS if kw in t)
        bear_hits = sum(1 for kw in BEARISH_KEYWORDS if kw in t)
        net = bull_hits - bear_hits
        if net > 0:
            return "BULLISH", +1
        if net < 0:
            return "BEARISH", -1
        return "NEUTRAL", 0

    # ─────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _extract_currencies(pair: str) -> List[str]:
        """EURUSD → ['EUR', 'USD']  |  XAUUSD → ['XAU', 'USD']"""
        s = pair.upper().replace("/", "").replace("=X", "")
        if len(s) >= 6:
            return [s[:3], s[3:6]]
        return [s] if s else []

    @staticmethod
    def _neutral_result(reason: str) -> Dict[str, Any]:
        return {
            "trade_allowed":      True,
            "reason":             reason,
            "news_bias":          "NEUTRAL",
            "news_score":         0,
            "headline_count":     0,
            "top_headlines":      [],
            "currency_filtered":  False,
            "source":             "neutral",
            "currencies_checked": [],
            "fetched_at":         datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    @staticmethod
    def _fallback_result(reason: str) -> Dict[str, Any]:
        """When the API is unavailable — don't block trades, just flag it."""
        return {
            "trade_allowed":      True,   # don't block on missing news data
            "reason":             reason,
            "news_bias":          "NEUTRAL",
            "news_score":         0,
            "headline_count":     0,
            "top_headlines":      [],
            "currency_filtered":  False,
            "source":             "fallback",
            "currencies_checked": [],
            "fetched_at":         datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def _rss_fallback(self, pair: str, reason: str) -> Dict[str, Any]:
        """Free, no-key, no-quota fallback for when NewsAPI is unavailable.

        Reuses intelligence/news_sources.py's already-wired RSS feeds
        (DailyFX, ForexLive, Investing.com, MarketWatch) instead of just
        returning a flat NEUTRAL with zero signal. Same keyword scoring as
        the live NewsAPI path, so downstream consumers (MasterAnalyst)
        see a consistent shape either way. Falls through to
        _fallback_result() only if RSS itself has nothing relevant right
        now (feeds unreachable, or no headline mentions this pair).
        """
        currencies = self._extract_currencies(pair)
        try:
            from intelligence.news_sources import NewsSources
            rss_items = NewsSources().fetch_rss_feeds()
        except Exception as e:
            log.debug(f"[NewsAPI] RSS fallback unavailable: {e}")
            return self._fallback_result(reason)

        scored = []
        for item in rss_items:
            title = item.headline or item.event or ""
            if not any(c.lower() in title.lower() for c in currencies):
                continue
            sentiment, score = self._score_headline(title)
            scored.append({
                "title":        title,
                "source":       item.source,
                "published_at": item.time_iso or "",
                "url":          item.url or "",
                "sentiment":    sentiment,
                "score":        score,
            })

        if not scored:
            log.info(f"[NewsAPI] RSS fallback: no {currencies}-relevant headlines "
                      f"in {len(rss_items)} RSS items — using flat neutral")
            result = self._fallback_result(reason)
            result["rss_checked"] = len(rss_items)
            return result

        total_score = sum(s["score"] for s in scored)
        normalized = int((total_score / max(1, len(scored))) * 100)
        if normalized >= 25:
            bias = "BULLISH"
        elif normalized <= -25:
            bias = "BEARISH"
        else:
            bias = "NEUTRAL"

        log.info(
            f"[NewsAPI] RSS fallback ({reason}) | {pair} | "
            f"{len(scored)} headlines | bias={bias} score={normalized}"
        )
        return {
            "trade_allowed":      normalized > -50,
            "reason":             f"RSS fallback ({reason}) | {len(scored)} headlines | bias={bias} score={normalized}",
            "news_bias":          bias,
            "news_score":         normalized,
            "headline_count":     len(scored),
            "top_headlines":      scored[:5],
            "currency_filtered":  True,
            "source":             "rss_fallback",
            "currencies_checked": currencies,
            "fetched_at":         datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    # ─────────────────────────────────────────────────────────
    # AI context (for MasterAnalyst)
    # ─────────────────────────────────────────────────────────

    def get_ai_context(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Compact context dict that MasterAnalyst can inject into its prompt."""
        return {
            "newsapi_bias":       result.get("news_bias", "NEUTRAL"),
            "newsapi_score":      result.get("news_score", 0),
            "newsapi_headlines":  len(result.get("top_headlines", [])),
            "newsapi_top":        [
                {"t": h["title"][:80], "s": h["sentiment"]}
                for h in result.get("top_headlines", [])[:3]
            ],
            "newsapi_source":     result.get("source", "unknown"),
        }

    def print_summary(self, result: Dict[str, Any]) -> None:
        bar = "═" * 50
        log.info(bar)
        log.info("  📰  NEWSAPI SENTIMENT  (Day 92)")
        log.info(bar)
        log.info(f"  Bias          : {result.get('news_bias','?')}")
        log.info(f"  Score         : {result.get('news_score',0):+d}")
        log.info(f"  Headlines     : {result.get('headline_count',0)}")
        log.info(f"  Source        : {result.get('source','?')}")
        log.info(f"  Trade allowed : {result.get('trade_allowed',True)}")
        if result.get("top_headlines"):
            log.info("  ── Top 3 ──")
            for h in result["top_headlines"][:3]:
                icon = {"BULLISH":"🟢","BEARISH":"🔴","NEUTRAL":"⚪"}.get(h["sentiment"],"?")
                log.info(f"    {icon} [{h['source']}] {h['title'][:80]}")
        log.info(bar)


# ── Singleton ─────────────────────────────────────────────────────

_PROVIDER: Optional[NewsAPIProvider] = None


def get_news_api_provider() -> NewsAPIProvider:
    global _PROVIDER
    if _PROVIDER is None:
        _PROVIDER = NewsAPIProvider()
    return _PROVIDER