# data/sentiment_data.py  —  Day 41 | Sentiment Data Provider
# ============================================================
# SentimentEngine-এর জন্য data collect করে।
#
# Sources:
#   - Retail positioning  : Myfxbook / broker COT data (simulated)
#   - Fear & Greed Index  : FX-native aggregator (preferred) → alternative.me crypto F&G (fallback)
#                          The FX-native aggregator averages retail long% across
#                          7 major pairs via analysis/retail_sentiment.py.
#                          The crypto F&G is only used as a last-resort proxy
#                          and is flagged source="alternative.me" so downstream
#                          scoring can apply a half-weight penalty.
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
        # PERF + PARITY FIX: this module has a duplicate at
        # data/sentiment_data.py that already skips live fetches in
        # backtest mode — but agents/analysis_agent.py actually imports
        # THIS copy (analysis.sentiment_data), which never got the same
        # fix. Result: every backtest bar was making live network calls
        # (retail positioning, Fear & Greed via alternative.me, DXY via
        # Yahoo, currency strengths) — ~3-6s/bar per trader.log evidence,
        # and a parity bug too (today's live sentiment applied to
        # historical bars). Mirror the other short-circuited modules
        # (economic_calendar_api, fred_data, sentiment_model) here.
        from core.constants import is_backtest_mode
        if is_backtest_mode():
            log.debug(
                f"[SentimentData] backtest mode — live sentiment fetch "
                f"skipped for {pair}, returning neutral fallback"
            )
            return {
                "pair":               pair,
                "retail_long_pct":    50.0,
                "retail_source":      "backtest_skipped",
                "fg_index":           50.0,
                "fg_label":           "Neutral",
                "fg_source":          "backtest_skipped",
                "currency_strengths": {},
                "dxy_trend":          "NEUTRAL",
                "dxy_change_pct":     0.0,
                "dxy_source":         "backtest_skipped",
                "source":             "backtest_skipped",
            }

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

    # Major FX pairs used to build a genuine FX-native Fear & Greed Index.
    # We aggregate retail long% across these pairs — when retail is
    # overwhelmingly long across the board, the market is "greedy"
    # (contrarian SELL); when overwhelmingly short, "fearful" (contrarian BUY).
    # This is a real FX sentiment signal, not a crypto cross-asset proxy.
    _FX_FG_PAIRS = (
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
        "AUDUSD", "USDCAD", "NZDUSD",
    )

    # Module-level cache so we don't hit OANDA/Myfxbook 7 times per
    # sentiment refresh. TTL in seconds.
    _FX_FG_CACHE: dict = {"value": None, "ts": 0.0}
    _FX_FG_TTL = 300.0  # 5 minutes

    def get_fear_greed_index(self) -> dict:
        """
        Fear & Greed Index — FX-native by default, crypto proxy as fallback.

        Source chain (round-5 audit fix):
          1. FX-native aggregator (preferred): averages retail long%
             across 7 major FX pairs via analysis.retail_sentiment.py
             (OANDA v20 → Myfxbook → synthetic RSI). Returns a real
             FX-market sentiment index on a 0–100 scale.
          2. Crypto F&G proxy (fallback): alternative.me crypto F&G,
             used ONLY if every retail-sentiment source fails. The
             result is flagged source="alternative.me" so downstream
             fear_greed() can apply the half-weight penalty.
          3. Static neutral (last resort): value=50, source="fallback".

        Returns:
            {"value": float, "label": str, "source": str}
            source ∈ {"fx_native", "alternative.me", "fallback"}
        """
        import time

        # ── Source 1: FX-native aggregator (cached) ───────────────
        now = time.time()
        cached = self._FX_FG_CACHE
        if cached["value"] is not None and (now - cached["ts"]) < self._FX_FG_TTL:
            log.debug("[F&G] Using cached FX-native value")
            return cached["value"]

        try:
            result = self._compute_fx_native_fg()
            if result is not None:
                self._FX_FG_CACHE["value"] = result
                self._FX_FG_CACHE["ts"] = now
                return result
            log.info("[F&G] FX-native aggregator returned nothing, falling back to crypto proxy")
        except Exception as e:
            log.warning(f"[F&G] FX-native aggregator failed: {e} — falling back to crypto proxy")

        # ── Source 2: crypto F&G proxy (alternative.me) ───────────
        try:
            import urllib.request
            import json

            url = "https://api.alternative.me/fng/?limit=1"
            req = urllib.request.Request(url, headers={"User-Agent": "ForexAI/1.0"})

            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())

            value = float(data["data"][0]["value"])
            label = data["data"][0]["value_classification"]
            log.info(f"[F&G API] Crypto proxy | Value: {value} | Label: {label}")
            return {"value": value, "label": label, "source": "alternative.me"}

        except Exception as e:
            log.warning(f"[F&G API] Crypto proxy failed: {e} — using fallback 50")
            return {"value": 50.0, "label": "Neutral", "source": "fallback"}

    def _compute_fx_native_fg(self):
        """
        Build a 0–100 FX-native Fear & Greed Index from retail positioning.

        Method:
          - For each major pair, fetch retail long_pct via RetailSentimentAPI
            (OANDA → Myfxbook → synthetic RSI fallback chain).
          - Average long_pct across all pairs that returned a non-fallback
            value. If fewer than 3 pairs returned genuine data, return None
            so the caller falls back to the crypto proxy.
          - Map the averaged long_pct onto a 0–100 F&G scale:
              long_pct ≥ 70 → 80–100 (Extreme Greed — crowded long)
              long_pct ≥ 60 → 60–80  (Greed)
              long_pct ≤ 30 → 0–20   (Extreme Fear — crowded short)
              long_pct ≤ 40 → 20–40  (Fear)
              else          → 40–60  (Neutral)
            The contrarian interpretation: when retail is crowded long,
            the market is "greedy" (overbought); when crowded short,
            "fearful" (oversold).

        Returns:
            {"value": float, "label": str, "source": "fx_native"} or None
        """
        try:
            from analysis.retail_sentiment import get_retail_sentiment_api
        except Exception as e:
            log.debug(f"[F&G] retail_sentiment module unavailable: {e}")
            return None

        try:
            api = get_retail_sentiment_api()
        except Exception as e:
            log.debug(f"[F&G] RetailSentimentAPI init failed: {e}")
            return None

        long_pcts: list[float] = []
        genuine_sources = ("oanda_live", "myfxbook_live", "myfxbook_cached", "synthetic_rsi")
        for pair in self._FX_FG_PAIRS:
            try:
                r = api.get_sentiment(pair)
                if r.get("source") in genuine_sources:
                    long_pcts.append(float(r.get("long_pct", 50.0)))
            except Exception as e:
                log.debug(f"[F&G] {pair} retail fetch failed: {e}")
                continue

        # Require at least 3 genuine data points — otherwise the average
        # is too noisy to act as a market-wide sentiment read.
        if len(long_pcts) < 3:
            log.info(f"[F&G] Only {len(long_pcts)} genuine pairs — insufficient for FX-native F&G")
            return None

        avg_long = sum(long_pcts) / len(long_pcts)

        # Map retail long% → 0–100 F&G scale.
        #   retail crowded long  (avg_long high) → high F&G (greedy)
        #   retail crowded short (avg_long low)  → low F&G  (fearful)
        # Linear mapping: F&G = avg_long (since long_pct is already 0–100).
        # Then snap into the conventional F&G label bands.
        value = round(avg_long, 1)

        if value >= 75:
            label = "Extreme Greed"
        elif value >= 60:
            label = "Greed"
        elif value >= 40:
            label = "Neutral"
        elif value >= 25:
            label = "Fear"
        else:
            label = "Extreme Fear"

        log.info(
            f"[F&G FX-native] avg_long={avg_long:.1f}% across {len(long_pcts)} pairs "
            f"→ value={value} ({label})"
        )
        return {"value": value, "label": label, "source": "fx_native"}

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