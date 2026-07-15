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
"""
from __future__ import annotations

import os
import re
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
    """

    BASE_URL = "https://www.myfxbook.com/community/outlook"
    # Cache results for 10 minutes to avoid hitting the page too often
    _cache: Dict[str, tuple] = {}  # pair -> (timestamp, data)
    CACHE_TTL_SEC = 600  # 10 minutes

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
        import time as _t
        if cls._disabled_until and _t.time() < cls._disabled_until:
            return True
        # Cooldown expired — reset
        if cls._disabled_until and _t.time() >= cls._disabled_until:
            cls._disabled_until = 0.0
            cls._consecutive_failures = 0
            log.info(
                "[Myfxbook] Cooldown expired — re-enabling source. "
                "Next fetch attempt will run."
            )
        return False

    @classmethod
    def _record_failure(cls, reason: str) -> None:
        """Record a fetch failure; trip circuit breaker if threshold hit."""
        import time as _t
        cls._consecutive_failures += 1
        if cls._consecutive_failures >= cls.MAX_CONSECUTIVE_FAILURES:
            cls._disabled_until = _t.time() + cls.COOLDOWN_SEC
            log.warning(
                f"[Myfxbook] Circuit breaker TRIPPED after "
                f"{cls._consecutive_failures} consecutive failures "
                f"({reason}). Source disabled for {cls.COOLDOWN_SEC}s. "
                f"Retail sentiment will use synthetic RSI fallback "
                f"until cooldown expires."
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
    # PUBLIC API
    # ─────────────────────────────────────────────────────────

    def get_sentiment(self, pair: str) -> Dict[str, Any]:
        """Get retail sentiment for a pair from Myfxbook Community Outlook.

        Args:
            pair: e.g. "EURUSD" (will be converted to "EUR/USD")

        Returns: dict with long_pct, short_pct, contrarian_signal, etc.
                 Falls back to neutral if scrape fails.
        """
        # Cache check
        cached = self._cache.get(pair)
        if cached and (datetime.now(timezone.utc).timestamp() - cached[0]) < self.CACHE_TTL_SEC:
            result = dict(cached[1])
            result["source"] = "myfxbook_cached"
            return result

        # Try to scrape the outlook page
        outlook_data = self._fetch_outlook_page()

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
            return self._fallback_result(pair, "Myfxbook scrape failed (HTTP/Cloudflare)")
        if len(outlook_data) == 0:
            # Empty list = page loaded but parser found no pairs
            # (HTML structure may have changed)
            log.warning(
                f"[Myfxbook] Page fetched successfully but parser found "
                f"0 pairs — HTML structure may have changed. Falling "
                f"back to synthetic sentiment for {pair}."
            )
            return self._fallback_result(pair, "Myfxbook parse failed (0 pairs extracted — HTML structure changed?)")

        # Find this pair in the outlook data
        pair_data = self._find_pair(outlook_data, pair)
        if not pair_data:
            return self._fallback_result(pair, f"{pair} not found in Myfxbook outlook ({len(outlook_data)} pairs available)")

        # Compute derived metrics
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
        confidence = self._compute_confidence(long_pct, short_pct, contrarian_strength)

        result = {
            "source":              "myfxbook_live",
            "pair":                pair,
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

        # Cache it
        self._cache[pair] = (datetime.now(timezone.utc).timestamp(), result)

        log.info(
            f"[Myfxbook] {pair} | retail {sentiment_label} "
            f"({long_pct:.0f}%L/{short_pct:.0f}%S) | "
            f"contrarian={contrarian_signal}({contrarian_strength}) | "
            f"bias={trade_bias} conf={confidence}%"
        )
        return result

    # ─────────────────────────────────────────────────────────
    # Scraping
    # ─────────────────────────────────────────────────────────

    def _fetch_outlook_page(self) -> Optional[List[Dict]]:
        """Fetch and parse Myfxbook's community outlook page.

        Returns: list of dicts, each with pair, long_pct, short_pct, etc.
                 None on failure.

        Round-5 audit fix: added cooldown check + cloudscraper fallback.
        """
        # ── Circuit breaker: skip fetch entirely during cooldown ──
        if self._is_in_cooldown():
            log.debug(
                "[Myfxbook] Skipping fetch — in circuit-breaker cooldown. "
                "Caller will get fallback / cached result."
            )
            return None

        # ── Attempt 1: plain requests with browser-mimicking headers ──
        try:
            resp = requests.get(self.BASE_URL, headers=self.HEADERS, timeout=15)
            if resp.status_code == 200:
                self._record_success()
                html = resp.text
                return self._parse_outlook_html(html)
            elif resp.status_code in (403, 429, 503):
                # Cloudflare WAF / rate-limit / bot-detection — try
                # cloudscraper if available.
                log.info(
                    f"[Myfxbook] HTTP {resp.status_code} from requests — "
                    f"trying cloudscraper fallback (if installed)."
                )
            else:
                log.warning(f"[Myfxbook] HTTP {resp.status_code}")
                self._record_failure(f"HTTP {resp.status_code}")
                return None
        except Exception as e:
            log.warning(f"[Myfxbook] fetch failed (requests): {e}")
            # Don't return yet — try cloudscraper below
            pass

        # ── Attempt 2: cloudscraper (solves Cloudflare JS challenge) ──
        try:
            import cloudscraper  # type: ignore
        except ImportError:
            log.warning(
                "[Myfxbook] HTTP 403/429 received and `cloudscraper` not "
                "installed. Install with `pip install cloudscraper` to "
                "bypass Cloudflare bot detection. Falling back to "
                "synthetic RSI sentiment."
            )
            self._record_failure("403 + cloudscraper not installed")
            return None

        try:
            scraper = cloudscraper.create_scraper(
                browser={"browser": "chrome", "platform": "windows", "mobile": False}
            )
            resp2 = scraper.get(self.BASE_URL, timeout=20)
            if resp2.status_code == 200:
                log.info(
                    "[Myfxbook] cloudscraper fetch SUCCESS — "
                    "Cloudflare challenge solved."
                )
                self._record_success()
                return self._parse_outlook_html(resp2.text)
            else:
                log.warning(
                    f"[Myfxbook] cloudscraper also failed: HTTP {resp2.status_code}"
                )
                self._record_failure(f"cloudscraper HTTP {resp2.status_code}")
                return None
        except Exception as e:
            log.warning(f"[Myfxbook] cloudscraper fetch failed: {e}")
            self._record_failure(f"cloudscraper exception: {e}")
            return None

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
        for row in soup.select("tr"):
            try:
                text = row.get_text(separator=" ", strip=True)
                # Look for patterns like "EUR/USD" followed by percentages
                match = re.search(
                    r"([A-Z]{3}/[A-Z]{3}).*?(\d+(?:\.\d+)?)%.*?(\d+(?:\.\d+)?)%",
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

        log.debug(f"[Myfxbook] parsed {len(unique)} pairs from outlook page")
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
    # Synthetic sentiment (RSI-based — no external API needed)
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def compute_synthetic_sentiment(pair: str, df) -> Dict[str, Any]:
        """Compute synthetic retail sentiment from price action.

        When both OANDA and Myfxbook are unavailable, we can estimate
        retail sentiment from RSI + recent price action:
          - RSI > 70 (overbought) → retail likely 70%+ long (chasing)
          - RSI < 30 (oversold) → retail likely 70%+ short (panic)
          - RSI 45-55 → balanced

        This is less accurate than real sentiment data but better
        than nothing — it at least captures the "retail chases trends"
        pattern.

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
            except Exception as e:
                return MyfxbookSentiment._fallback_result(pair, "RSI computation failed")

        if rsi != rsi:  # NaN check
            return MyfxbookSentiment._fallback_result(pair, "RSI is NaN")

        # Map RSI to retail long%:
        # RSI 50 → 50% long (balanced)
        # RSI 70 → 70% long (retail chasing up)
        # RSI 30 → 30% long (retail panicking out)
        # RSI 80 → 80% long (euphoria)
        # RSI 20 → 20% long (capitulation)
        long_pct = max(10, min(90, rsi))
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
            "source":              "synthetic_rsi",
            "pair":                pair,
            "long_pct":            round(long_pct, 1),
            "short_pct":           round(short_pct, 1),
            "sentiment_label":     sentiment_label,
            "contrarian_signal":   contrarian_signal,
            "contrarian_strength": contrarian_strength,
            "long_short_ratio":    round(long_pct / short_pct, 2) if short_pct > 0 else 99,
            "net_position_pct":    round(long_pct - short_pct, 1),
            "rsi_basis":           round(rsi, 1),
            "order_book":          {"price_levels": [], "stop_cluster": None},
            "trade_bias":          contrarian_signal,
            "confidence":          max(0, confidence - 20),  # lower confidence for synthetic
            "fetched_at":          datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        log.info(
            f"[SyntheticSent] {pair} | RSI={rsi:.1f} → retail {sentiment_label} "
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
        log.info(bar)


# ── Singleton ────────────────────────────────────────────────────

_INSTANCE: Optional[MyfxbookSentiment] = None


def get_myfxbook_sentiment() -> MyfxbookSentiment:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = MyfxbookSentiment()
    return _INSTANCE
