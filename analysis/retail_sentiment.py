"""
analysis/retail_sentiment.py — Day 94/95 Retail Sentiment + Order Book
=========================================================================
Pulls retail trader positioning data with multi-source fallback:

  1. OANDA v20 (if OANDA_API_KEY set) — most accurate, has order book
  2. Myfxbook Community Outlook (no key needed) — good accuracy, scraped
  3. Price-based sentiment proxy (RSI-based) — last resort, computed locally

NOTE: The "synthetic" sentiment is NOT real retail sentiment.
It is a PRICE-BASED PROXY computed from RSI. It does not reflect
actual retail positioning. Use with caution — it's a fallback when
no real data is available.

Day 95 update: Myfxbook added as OANDA alternative (no account needed).
The fallback chain runs automatically — if OANDA key is missing, it
tries Myfxbook; if Myfxbook is blocked, it computes a price-based proxy.

Output shape (compatible with existing sentiment_ctx):
    {
      "source":              "oanda_live" | "myfxbook_live" | "synthetic_rsi" | "fallback",
      "pair":                "EURUSD",
      "long_pct":            72.3,
      "short_pct":           27.7,
      "sentiment_label":     "BULLISH",
      "contrarian_signal":   "BEARISH",
      "contrarian_strength": "STRONG",
      "trade_bias":          "SELL",
      "confidence":          75,
    }

Usage:
    from analysis.retail_sentiment import get_retail_sentiment_api
    api = get_retail_sentiment_api()
    result = api.get_sentiment("EURUSD")
    # Or with DataFrame for synthetic fallback:
    result = api.get_sentiment("EURUSD", df=candle_df)
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from utils.logger import get_logger

log = get_logger("retail_sentiment")


class RetailSentimentAPI:
    """OANDA v20 retail sentiment + order book API."""

    BASE_URL = "https://api-fxtrade.oanda.com/v3"
    # Use practice URL for demo accounts:
    PRACTICE_URL = "https://api-fxpractice.oanda.com/v3"

    def __init__(self):
        self._api_key    = os.getenv("OANDA_API_KEY", "").strip()
        self._account_id = os.getenv("OANDA_ACCOUNT_ID", "").strip()
        # Practice account = use practice URL; live = production URL
        self._use_practice = os.getenv("OANDA_USE_PRACTICE", "true").lower() == "true"

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    @property
    def _base_url(self) -> str:
        return self.PRACTICE_URL if self._use_practice else self.BASE_URL

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept-Datetime-Format": "RFC3339",
        }

    # ─────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────

    def get_sentiment(self, pair: str, df=None) -> Dict[str, Any]:
        """Get retail sentiment for a pair with multi-source fallback.

        Day 95 fallback chain:
          1. OANDA v20 (if OANDA_API_KEY set) — most accurate, has order book
          2. Myfxbook Community Outlook (no key needed) — scraped from public page
          3. Synthetic sentiment (RSI-based) — computed from df if provided
          4. Neutral fallback — last resort

        Args:
            pair: e.g. "EURUSD"
            df:   optional DataFrame with 'close' + 'rsi' columns (for synthetic fallback)

        Returns: dict with sentiment, contrarian signal, etc.
        """
        # ── Source 1: OANDA v20 (if key available) ──
        if self.available:
            result = self._get_sentiment_oanda(pair)
            if result.get("source") == "oanda_live":
                return result
            log.info("[RetailSent] OANDA failed/unavailable, trying Myfxbook")

        # ── Source 2: Myfxbook Community Outlook (no key needed) ──
        try:
            from analysis.myfxbook_sentiment import get_myfxbook_sentiment
            myfxbook = get_myfxbook_sentiment()
            result = myfxbook.get_sentiment(pair)
            if result.get("source") in ("myfxbook_live", "myfxbook_cached"):
                return result
            log.info("[RetailSent] Myfxbook failed, trying synthetic sentiment")
        except Exception as e:
            log.warning(f"[RetailSent] Myfxbook error: {e}")

        # ── Source 3: Synthetic sentiment (RSI-based, needs df) ──
        if df is not None:
            try:
                from analysis.myfxbook_sentiment import MyfxbookSentiment
                result = MyfxbookSentiment.compute_synthetic_sentiment(pair, df)
                if result.get("source") == "synthetic_rsi":
                    return result
            except Exception as e:
                log.warning(f"[RetailSent] Synthetic sentiment error: {e}")

        # ── Source 4: Neutral fallback ──
        return self._fallback_result(pair, "All sentiment sources failed")

    def _get_sentiment_oanda(self, pair: str) -> Dict[str, Any]:
        """Original OANDA v20 sentiment fetch (Source 1 in chain)."""
        if not self.available:
            return self._fallback_result(pair, "OANDA_API_KEY not set")

        oanda_instrument = self._to_oanda_format(pair)

        # ── Fetch Position Book (retail long/short %) ──
        position_data = self._fetch_position_book(oanda_instrument)

        # ── Fetch Order Book (pending orders by price) ──
        order_data = self._fetch_order_book(oanda_instrument)

        if not position_data and not order_data:
            return self._fallback_result(pair, "OANDA API calls failed")

        # Parse position book
        long_pct, short_pct = 50.0, 50.0
        if position_data:
            long_pct, short_pct = self._parse_position_pct(position_data, oanda_instrument)

        # Parse order book
        stop_cluster = None
        order_levels: List[Dict] = []
        if order_data:
            order_levels, stop_cluster = self._parse_order_book(order_data, oanda_instrument)

        # Compute derived metrics
        ratio = long_pct / short_pct if short_pct > 0 else float("inf")
        net_pct = long_pct - short_pct
        sentiment_label = "BULLISH" if long_pct > short_pct else "BEARISH"
        contrarian_signal = "BEARISH" if long_pct > 60 else "BULLISH" if long_pct < 40 else "NEUTRAL"
        contrarian_strength = (
            "STRONG" if long_pct > 75 or long_pct < 25
            else "MODERATE" if long_pct > 60 or long_pct < 40
            else "WEAK"
        )
        trade_bias = contrarian_signal  # contrarian: fade retail
        confidence = self._compute_confidence(long_pct, short_pct, contrarian_strength)

        result = {
            "source":              "oanda_live",
            "pair":                pair,
            "oanda_instrument":    oanda_instrument,
            "long_pct":            round(long_pct, 1),
            "short_pct":           round(short_pct, 1),
            "sentiment_label":     sentiment_label,        # retail mood
            "contrarian_signal":   contrarian_signal,      # smart-money bias
            "contrarian_strength": contrarian_strength,
            "long_short_ratio":    round(ratio, 2),
            "net_position_pct":    round(net_pct, 1),
            "order_book":          {
                "price_levels": order_levels[:10],
                "stop_cluster": stop_cluster,
            },
            "trade_bias":          trade_bias,
            "confidence":          confidence,
            "fetched_at":          datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        log.info(
            f"[RetailSent] {pair} | retail {sentiment_label} ({long_pct:.0f}%L/{short_pct:.0f}%S) | "
            f"contrarian={contrarian_signal}({contrarian_strength}) | "
            f"bias={trade_bias} conf={confidence}%"
        )
        return result

    # ─────────────────────────────────────────────────────────
    # OANDA API calls
    # ─────────────────────────────────────────────────────────

    def _fetch_position_book(self, instrument: str) -> Optional[Dict]:
        """Fetch OANDA position book for an instrument."""
        try:
            url = f"{self._base_url}/instruments/{instrument}/positionBook"
            resp = requests.get(url, headers=self._headers(), timeout=15)
            if resp.status_code != 200:
                log.debug(f"[RetailSent] positionBook {instrument} HTTP {resp.status_code}")
                return None
            return resp.json()
        except Exception as e:
            log.debug(f"[RetailSent] positionBook failed: {e}")
            return None

    def _fetch_order_book(self, instrument: str) -> Optional[Dict]:
        """Fetch OANDA order book for an instrument."""
        try:
            url = f"{self._base_url}/instruments/{instrument}/orderBook"
            resp = requests.get(url, headers=self._headers(), timeout=15)
            if resp.status_code != 200:
                log.debug(f"[RetailSent] orderBook {instrument} HTTP {resp.status_code}")
                return None
            return resp.json()
        except Exception as e:
            log.debug(f"[RetailSent] orderBook failed: {e}")
            return None

    # ─────────────────────────────────────────────────────────
    # Parsing helpers
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _parse_position_pct(data: Dict, instrument: str) -> tuple[float, float]:
        """Extract long% / short% from OANDA position book response.

        OANDA returns a bucketed position book. We aggregate all longs
        and all shorts to compute the percentages.
        """
        try:
            book = data.get("orderBook", data.get("positionBook", {}))
            buckets = book.get("buckets", [])
            long_total = 0.0
            short_total = 0.0
            for b in buckets:
                # 'longCountPercent' and 'shortCountPercent' are 0-100
                long_total += float(b.get("longCountPercent", 0))
                short_total += float(b.get("shortCountPercent", 0))
            total = long_total + short_total
            if total == 0:
                return 50.0, 50.0
            return (long_total / total * 100, short_total / total * 100)
        except Exception as e:
            return 50.0, 50.0

    @staticmethod
    def _parse_order_book(data: Dict, instrument: str) -> tuple[List[Dict], Optional[float]]:
        """Extract pending order levels + detect stop cluster.

        Returns (price_levels, stop_cluster_price). The stop cluster
        is the price level with the highest concentration of stop-loss
        orders (where price tends to spike to grab liquidity).
        """
        try:
            book = data.get("orderBook", {})
            buckets = book.get("buckets", [])
            levels = []
            max_stop_pct = 0.0
            stop_cluster_price = None
            for b in buckets:
                price = float(b.get("price", 0))
                long_count = float(b.get("longCountPercent", 0))
                short_count = float(b.get("shortCountPercent", 0))
                total = long_count + short_count
                if total > 0:
                    levels.append({
                        "price": price,
                        "long_pct": long_count,
                        "short_pct": short_count,
                        "total_pct": total,
                    })
                    if total > max_stop_pct:
                        max_stop_pct = total
                        stop_cluster_price = price
            return levels, stop_cluster_price
        except Exception as e:
            return [], None

    # ─────────────────────────────────────────────────────────
    # Confidence calculation
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _compute_confidence(long_pct: float, short_pct: float, strength: str) -> int:
        """Contrarian confidence — higher when retail is more one-sided."""
        extremes = abs(long_pct - 50)  # 0 = balanced, 50 = extreme one-side
        base = int(extremes * 2)  # 0-100
        bonus = {"STRONG": 10, "MODERATE": 5, "WEAK": 0}.get(strength, 0)
        return max(0, min(100, base + bonus))

    # ─────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _to_oanda_format(pair: str) -> str:
        """EURUSD → EUR_USD (OANDA instrument format)."""
        s = pair.upper().replace("/", "").replace("=X", "").replace("_", "")
        if len(s) >= 6:
            return f"{s[:3]}_{s[3:6]}"
        return s

    @staticmethod
    def _fallback_result(pair: str, reason: str) -> Dict[str, Any]:
        """When OANDA unavailable — return neutral sentiment."""
        return {
            "source":              "fallback",
            "pair":                pair,
            "oanda_instrument":    RetailSentimentAPI._to_oanda_format(pair),
            "long_pct":            50.0,
            "short_pct":           50.0,
            "sentiment_label":     "NEUTRAL",
            "contrarian_signal":   "NEUTRAL",
            "contrarian_strength": "WEAK",
            "long_short_ratio":    1.0,
            "net_position_pct":    0.0,
            "order_book":          {"price_levels": [], "stop_cluster": None},
            "trade_bias":          "NEUTRAL",
            "confidence":          0,
            "reason":              reason,
            "fetched_at":          datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    # ─────────────────────────────────────────────────────────
    # AI context (for MasterAnalyst)
    # ─────────────────────────────────────────────────────────

    def get_ai_context(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Compact context for MasterAnalyst prompt."""
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
        log.info("  👥  RETAIL SENTIMENT  (Day 94)")
        log.info(bar)
        log.info(f"  Pair           : {result.get('pair','?')}")
        log.info(f"  Source         : {result.get('source','?')}")
        log.info(f"  Retail Long %  : {result.get('long_pct',0):.1f}")
        log.info(f"  Retail Short % : {result.get('short_pct',0):.1f}")
        log.info(f"  Sentiment      : {result.get('sentiment_label','?')} (retail mood)")
        log.info(f"  Contrarian     : {result.get('contrarian_signal','?')} ({result.get('contrarian_strength','?')})")
        log.info(f"  Trade bias     : {result.get('trade_bias','?')} | conf {result.get('confidence',0)}%")
        sc = result.get("order_book", {}).get("stop_cluster")
        if sc:
            log.info(f"  Stop cluster   : {sc} (liquidity-grab target)")
        log.info(bar)


# ── Singleton ────────────────────────────────────────────────────

_INSTANCE: Optional[RetailSentimentAPI] = None


def get_retail_sentiment_api() -> RetailSentimentAPI:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = RetailSentimentAPI()
    return _INSTANCE
