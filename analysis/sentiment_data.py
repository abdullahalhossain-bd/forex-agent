# data/sentiment_data.py  —  Day 41 | Sentiment Data Provider
# ============================================================
# SentimentEngine-এর জন্য data collect করে।
#
# Sources:
#   - Retail positioning  : Myfxbook / broker COT data (simulated)
#   - Fear & Greed Index  : Alternative.me API (free)
#   - Currency Strength   : analysis.currency_strength.CurrencyStrengthEngine
#                            (shared singleton — single source of truth,
#                            no longer calculated in this file, see
#                            get_currency_strengths() below)
#   - DXY                 : yfinance "DX-Y.NYB" symbol
# ============================================================

from utils.logger import get_logger
from analysis.currency_strength import get_currency_strength_engine, MAJOR_CURRENCIES

log = get_logger("sentiment_data")

# 2026-07-22 fix: this module used to keep its own MAJOR_CURRENCIES list and
# a CURRENCY_PAIRS map (7 pairs/currency) alongside a private yfinance-based
# recalculation in get_currency_strengths() below. That gave the AI two
# different "currency strength" numbers for the same currency at the same
# moment — one from this file's 1-day-% -change-on-12-pairs approximation,
# one from analysis.currency_strength's real 28-pair RSI/momentum engine —
# and they never agreed. MAJOR_CURRENCIES is now imported from the real
# engine so the two never drift apart again; CURRENCY_PAIRS is removed
# entirely since strength is no longer computed here.


class SentimentDataProvider:
    """
    Sentiment Engine-এর জন্য সব data এক জায়গা থেকে দেয়।

    Usage:
        provider = SentimentDataProvider()
        data = provider.get_all("EURUSD")

        sentiment_engine.final_sentiment_score(
            pair               = data["pair"],
            retail_long_pct    = data["retail_long_pct"],
            fg_index           = data["fg_index"],
            currency_strengths = data["currency_strengths"],
            dxy_trend          = data["dxy_trend"],
            dxy_change_pct     = data["dxy_change_pct"],
        )
    """

    def __init__(self):
        # No local strength cache anymore — get_currency_strengths() now
        # delegates to the shared CurrencyStrengthEngine singleton, which
        # owns its own cache (see analysis/currency_strength.py).
        pass

    # ═══════════════════════════════════════════════════════════
    # MAIN METHOD
    # ═══════════════════════════════════════════════════════════

    def get_all(self, pair: str) -> dict:
        """
        একটি pair-এর জন্য সব sentiment data এক call-এ।

        Returns:
            {
                "pair": "EURUSD",
                "retail_long_pct": 68.5,
                "fg_index": 45.0,
                "currency_strengths": {"USD": 72, "EUR": 48, ...},
                "dxy_trend": "BULLISH",
                "dxy_change_pct": 0.25,
                "source": "live" | "cached" | "fallback"
            }
        """
        log.info(f"[SentimentData] Fetching all sentiment data for {pair}")

        retail     = self.get_retail_positioning(pair)
        fg         = self.get_fear_greed_index()
        strengths  = self.get_currency_strengths()
        dxy        = self.get_dxy_data()

        result = {
            "pair":               pair,
            "retail_long_pct":    retail["long_pct"],
            "retail_source":      retail["source"],
            "fg_index":           fg["value"],
            "fg_source":          fg["source"],
            "currency_strengths": strengths["strengths"],
            "strength_source":    strengths["source"],
            "dxy_trend":          dxy["trend"],
            "dxy_change_pct":     dxy["change_pct"],
            "dxy_source":         dxy["source"],
        }

        log.info(
            f"[SentimentData] {pair} | "
            f"Retail Long: {retail['long_pct']}% | "
            f"F&G: {fg['value']} | "
            f"DXY: {dxy['trend']}"
        )
        return result

    # ═══════════════════════════════════════════════════════════
    # 1. RETAIL POSITIONING
    # ═══════════════════════════════════════════════════════════

    def get_retail_positioning(self, pair: str) -> dict:
        """
        Retail trader positioning data।

        Real implementation:
            - Myfxbook Community Outlook API
            - OANDA fxTrade sentiment
            - Broker-specific COT-style data

        এখন: yfinance RSI + volume দিয়ে approximate করা হচ্ছে।
        Future: real broker API connect করো।
        """
        try:
            import yfinance as yf
            pair_yf = self._normalize_pair(pair)
            # Skip commodity pairs
            if pair_yf is None:
                return self._fallback_retail(pair)
            
            ticker  = yf.Ticker(pair_yf)
            df      = ticker.history(period="5d", interval="1h")

            if df.empty:
                return self._fallback_retail(pair)

            # Approximate: momentum-based retail positioning
            # যখন price উপরে যায় → retail usually buys more
            close   = df["Close"].values
            recent  = close[-6:]   # last 6 hours
            older   = close[-24:-6] if len(close) >= 24 else close[:-6]

            if len(older) == 0:
                return self._fallback_retail(pair)

            price_change = (recent[-1] - older[0]) / older[0] * 100

            # Approximate retail long%: mean-reversion bias
            # Retail tends to be: ~60% long in uptrend, ~40% long in downtrend
            base_long = 50.0
            if price_change > 1.0:
                long_pct = min(85, base_long + price_change * 8)
            elif price_change < -1.0:
                long_pct = max(15, base_long + price_change * 8)
            else:
                long_pct = base_long + price_change * 5

            long_pct = round(long_pct, 1)

            log.info(f"[RetailData] {pair} | Approx Long: {long_pct}%")
            return {"long_pct": long_pct, "source": "approximated_from_price"}

        except Exception as e:
            log.warning(f"[RetailData] Error: {e} — using fallback")
            return self._fallback_retail(pair)

    def _fallback_retail(self, pair: str) -> dict:
        """Fallback: neutral 50% positioning"""
        return {"long_pct": 50.0, "source": "fallback_neutral"}

    # ═══════════════════════════════════════════════════════════
    # 2. FEAR & GREED INDEX
    # ═══════════════════════════════════════════════════════════

    def get_fear_greed_index(self) -> dict:
        """
        Crypto Fear & Greed Index (Alternative.me API — free, no key needed)।
        Forex/Gold sentiment proxy হিসেবে কাজ করে।

        Endpoint: https://api.alternative.me/fng/
        """
        try:
            import urllib.request
            import json

            url = "https://api.alternative.me/fng/?limit=1"
            req = urllib.request.Request(url, headers={"User-Agent": "ForexAI/1.0"})

            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())

            value = float(data["data"][0]["value"])
            label = data["data"][0]["value_classification"]
            log.info(f"[F&G API] Value: {value} | Label: {label}")
            return {"value": value, "label": label, "source": "alternative.me"}

        except Exception as e:
            log.warning(f"[F&G API] Failed: {e} — using fallback 50")
            return {"value": 50.0, "label": "Neutral", "source": "fallback"}

    # ═══════════════════════════════════════════════════════════
    # 3. CURRENCY STRENGTH
    # ═══════════════════════════════════════════════════════════

    def get_currency_strengths(self) -> dict:
        """
        Major 8 currencies-এর relative strength — এখন সরাসরি
        analysis.currency_strength.CurrencyStrengthEngine (shared singleton)
        থেকে আসে।

        2026-07-22 fix: এই মেথড আগে yfinance দিয়ে নিজের একটা আলাদা
        calculation করত (12 sample pair, শুধু 1-day % change, broker/MT5
        candle বা indicator ছাড়া) — যেটা CurrencyStrengthEngine-এর আসল
        28-pair RSI/momentum-ভিত্তিক calculation-এর সাথে কখনোই মিলত না।
        একই মুহূর্তে একই currency-র জন্য দুইটা ভিন্ন strength number থাকা
        মানে এক dataset ভুল বা stale, অথচ কোনটা সেটা ধরার উপায় ছিল না।
        এখন এই মেথড শুধু shared engine-কে delegate করে, তাই পুরো
        pipeline-এ ঠিক একটাই currency-strength number থাকে এবং cache-ও
        শেয়ার হয় (28-pair fetch একবারই হয়, প্রতি consumer-এর জন্য আলাদা
        করে না)।

        Returns:
            {"strengths": {"USD": 72, "EUR": 48, ...}, "source": "engine" | "fallback"}
        """
        try:
            engine = get_currency_strength_engine()
            result = engine.calculate_strength()
            strengths = result["strengths"]
            log.info(f"[CurrStr] From shared CurrencyStrengthEngine: {strengths}")
            return {"strengths": strengths, "source": "engine"}

        except Exception as e:
            log.warning(f"[CurrStr] Shared engine unavailable: {e} — using fallback")
            fallback = {c: 50.0 for c in MAJOR_CURRENCIES}
            return {"strengths": fallback, "source": "fallback"}

    # ═══════════════════════════════════════════════════════════
    # 4. DXY DATA
    # ═══════════════════════════════════════════════════════════

    def get_dxy_data(self) -> dict:
        """
        DXY (US Dollar Index) — yfinance থেকে।
        Symbol: "DX-Y.NYB"

        Returns:
            {
                "trend": "BULLISH",
                "change_pct": 0.25,
                "current": 104.5,
                "source": "yfinance"
            }
        """
        try:
            import yfinance as yf

            ticker = yf.Ticker("DX-Y.NYB")
            df     = ticker.history(period="5d", interval="1d")

            if df.empty or len(df) < 2:
                return self._fallback_dxy()

            prev    = float(df["Close"].iloc[-2])
            current = float(df["Close"].iloc[-1])
            change  = round((current - prev) / prev * 100, 3)

            # 3-day trend
            if len(df) >= 3:
                three_days_ago = float(df["Close"].iloc[-3])
                three_day_chg  = (current - three_days_ago) / three_days_ago * 100
            else:
                three_day_chg = change

            if three_day_chg > 0.15:
                trend = "BULLISH"
            elif three_day_chg < -0.15:
                trend = "BEARISH"
            else:
                trend = "NEUTRAL"

            log.info(
                f"[DXY] Current: {current:.3f} | "
                f"Change: {change:+.3f}% | Trend: {trend}"
            )
            return {
                "trend":      trend,
                "change_pct": change,
                "current":    round(current, 3),
                "source":     "yfinance",
            }

        except Exception as e:
            log.warning(f"[DXY] Error: {e} — using fallback")
            return self._fallback_dxy()

    def _fallback_dxy(self) -> dict:
        return {"trend": "NEUTRAL", "change_pct": 0.0, "current": 100.0, "source": "fallback"}

    # ═══════════════════════════════════════════════════════════
    # UTILS
    # ═══════════════════════════════════════════════════════════

    def _normalize_pair(self, pair: str) -> str:
        """pair → yfinance symbol"""
        pair = pair.upper().replace("/", "").replace("=X", "")
        # Skip commodity pairs (XAUUSD, XAGUSD) — not available on Yahoo Finance in this format
        if pair in ("XAUUSD", "XAGUSD"):
            return None
        return pair + "=X"

    def print_summary(self, data: dict) -> None:
        """Fetched data-এর summary print করো।"""
        bar = "─" * 48
        print(f"\n{bar}")
        print(f"  📡  SENTIMENT DATA  —  {data.get('pair', '')}")
        print(bar)
        print(f"  Retail Long     : {data.get('retail_long_pct', 0):.1f}%  [{data.get('retail_source', '')}]")
        print(f"  Fear & Greed    : {data.get('fg_index', 0):.0f}  [{data.get('fg_source', '')}]")
        print(f"  DXY Trend       : {data.get('dxy_trend', '')}  ({data.get('dxy_change_pct', 0):+.3f}%)  [{data.get('dxy_source', '')}]")
        print(f"\n  Currency Strengths:")
        for cur, val in sorted(
            data.get("currency_strengths", {}).items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            bar_len = int(val / 5)
            strength_bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"  {cur}  {strength_bar}  {val:.0f}/100")
        print(f"  [{data.get('strength_source', '')}]")
        print(bar + "\n")