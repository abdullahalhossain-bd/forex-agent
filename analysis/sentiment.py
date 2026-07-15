# analysis/sentiment.py  —  Day 41 | Market Sentiment Engine
# ============================================================
# AI Crowd Psychology Layer
#
# 4 টি ইঞ্জিন:
#   1. Retail Trader Positioning  — contrarian logic
#   2. Fear & Greed Index         — market emotion
#   3. Currency Strength Meter    — relative strength
#   4. DXY Correlation Analysis   — USD impact
#
# Final Sentiment Score = সব একসাথে combine
# ============================================================

from utils.logger import get_logger

log = get_logger("sentiment_engine")


class SentimentEngine:
    """
    Market psychology reading engine।

    Technical analysis-এর পাশাপাশি AI বুঝবে:
    - মানুষ কী ভাবছে
    - কোথায় ভুল করছে
    - Currency relative strength কেমন
    - USD (DXY) কোন দিকে যাচ্ছে
    """

    # ── Thresholds ────────────────────────────────────────────
    RETAIL_EXTREME_LONG  = 75   # % এর বেশি long = contrarian SELL
    RETAIL_EXTREME_SHORT = 25   # % এর কম long = contrarian BUY
    FG_EXTREME_GREED     = 75
    FG_EXTREME_FEAR      = 25

    def __init__(self):
        # Currency strength cache (0–100 scale)
        self._currency_strength: dict[str, float] = {}

    # ═══════════════════════════════════════════════════════════
    # 1. RETAIL TRADER POSITIONING  ⭐⭐⭐⭐⭐
    # ═══════════════════════════════════════════════════════════

    # Sources sentiment_data.get_retail_positioning() can currently return.
    # There is no real broker/COT feed wired in today — only a price-derived
    # approximation and a neutral fallback — so both count as "not genuine"
    # for weighting purposes until a real feed (Myfxbook/OANDA/COT) exists.
    _GENUINE_RETAIL_SOURCES = {"myfxbook", "oanda", "broker_api", "cot"}

    def retail_positioning(
        self, pair: str, long_pct: float, source: str = "approximated_from_price"
    ) -> dict:
        """
        Retail traders কত % long/short আছে সেটা দেখে contrarian signal দাও।

        Args:
            pair     : "EURUSD", "GBPUSD" ইত্যাদি
            long_pct : Retail long percentage (0–100)
            source   : where long_pct came from (see sentiment_data.py's
                "source" field: "approximated_from_price", "fallback_neutral",
                or a genuine feed name). Defaults to the approximation this
                codebase actually uses today.

        Logic:
            Retail 90% LONG  → smart money likely SHORT  → BEARISH signal
            Retail 10% LONG  → smart money likely LONG   → BULLISH signal

        FIX (institutional review, Finding H-3): sentiment_data.py's
        "retail positioning" is not real crowd data — it's derived from
        recent price momentum itself (source="approximated_from_price"),
        which risks double-counting momentum/trend signals computed
        elsewhere (structure.py, strength_calculator.py) under the guise of
        an "independent, contrarian" signal. Previously this was scored at
        full confidence regardless. Non-genuine sources now contribute at
        half weight, and the result carries an explicit data_quality flag
        so downstream consumers/aggregators can see this wasn't real
        positioning data, not just infer it from a "source" string.

        Returns:
            {
                "pair": "EURUSD",
                "retail_long_pct": 90,
                "retail_short_pct": 10,
                "contrarian_bias": "BEARISH",
                "score": -15,
                "data_quality": "synthetic_proxy" | "no_data_fallback" | "verified",
                "reason": "Retail 90% LONG — crowd extreme, SELL bias"
            }
        """
        short_pct = round(100 - long_pct, 1)
        is_genuine = source in self._GENUINE_RETAIL_SOURCES

        if long_pct >= self.RETAIL_EXTREME_LONG:
            bias  = "BEARISH"
            score = -15
            reason = (
                f"Retail {long_pct:.0f}% LONG — crowd extreme overcrowded, "
                f"contrarian SELL bias"
            )
        elif long_pct <= self.RETAIL_EXTREME_SHORT:
            bias  = "BULLISH"
            score = +15
            reason = (
                f"Retail {long_pct:.0f}% LONG ({short_pct:.0f}% SHORT) — "
                f"crowd extreme short, contrarian BUY bias"
            )
        else:
            bias  = "NEUTRAL"
            score = 0
            reason = (
                f"Retail {long_pct:.0f}% LONG — no extreme positioning, "
                f"no contrarian signal"
            )

        if not is_genuine and score != 0:
            score = round(score * 0.5)
            reason += " (halved: positioning is price-derived, not a genuine broker feed)"

        if source == "approximated_from_price":
            data_quality = "synthetic_proxy"
        elif source == "fallback_neutral":
            data_quality = "no_data_fallback"
        elif is_genuine:
            data_quality = "verified"
        else:
            data_quality = "unknown_source"

        result = {
            "pair":              pair,
            "retail_long_pct":   long_pct,
            "retail_short_pct":  short_pct,
            "contrarian_bias":   bias,
            "score":             score,
            "data_quality":      data_quality,
            "reason":            reason,
        }

        log.info(
            f"[Retail] {pair} | Long: {long_pct}% | "
            f"Bias: {bias} | Score: {score:+d} | Quality: {data_quality}"
        )
        return result

    # ═══════════════════════════════════════════════════════════
    # 2. FEAR & GREED INDEX  ⭐⭐⭐⭐
    # ═══════════════════════════════════════════════════════════

    def fear_greed(self, index_value: float, source: str = "alternative.me") -> dict:
        """
        Fear & Greed Index দেখে market emotion বোঝো।

        Args:
            index_value : 0 (Extreme Fear) → 100 (Extreme Greed)
            source      : where index_value came from. Defaults to
                "alternative.me", which is this codebase's actual current
                source — the CRYPTO Fear & Greed Index, applied to FX/Gold
                as an unvalidated cross-asset proxy (see sentiment_data.py).

        FIX (institutional review, Finding H-3): this was previously scored
        at full confidence regardless of source, presenting a crypto-market
        sentiment index as if it were validated forex sentiment. No
        correlation check between crypto sentiment and the specific FX pair
        being scored exists anywhere in this codebase. Until a genuine
        FX-specific sentiment source is wired in, the crypto-proxy source
        contributes at half weight and the result carries an explicit
        data_quality flag.

        Returns:
            {
                "value": 82,
                "label": "EXTREME_GREED",
                "impact": "negative_for_buy",
                "score": -20,
                "data_quality": "crypto_proxy" | "no_data_fallback" | "verified",
                "description": "Market overextended — reduce BUY confidence"
            }
        """
        if index_value >= self.FG_EXTREME_GREED:
            label   = "EXTREME_GREED"
            impact  = "negative_for_buy"
            score   = -20
            desc    = (
                "Market extremely greedy — overextended conditions. "
                "Reduce BUY confidence, reversal risk high."
            )
        elif index_value >= 55:
            label   = "GREED"
            impact  = "slightly_negative_for_buy"
            score   = -10
            desc    = "Market greedy — caution on new BUY entries."
        elif index_value <= self.FG_EXTREME_FEAR:
            label   = "EXTREME_FEAR"
            impact  = "negative_for_sell"
            score   = +20
            desc    = (
                "Market extremely fearful — oversold conditions. "
                "Reduce SELL confidence, bounce risk high."
            )
        elif index_value <= 45:
            label   = "FEAR"
            impact  = "slightly_negative_for_sell"
            score   = +10
            desc    = "Market fearful — caution on new SELL entries."
        else:
            label   = "NEUTRAL"
            impact  = "none"
            score   = 0
            desc    = "Market neutral — no extreme emotion detected."

        # ── Round-5 audit fix: explicit handling of crypto F&G as FX proxy ──
        # The operator's audit flagged this as "Fear & Greed index misuse"
        # — sentiment_engine uses crypto F&G as an unvalidated FX proxy.
        #
        # STATUS: This is INTENTIONAL and already mitigated — the score
        # is halved when source is "alternative.me" (crypto F&G) or
        # "fallback". The halving was already in place BEFORE this audit
        # (it's a documented institutional-review fix from Finding H-3,
        # see module docstring at fear_greed()).
        #
        # What this audit adds:
        #   1. An explicit TODO comment so the next maintainer knows
        #      this is a known tech debt, not an oversight.
        #   2. An env-var escape hatch: if the operator wires in a real
        #      FX-native sentiment source (CFTC CoT, IG Client
        #      Sentiment, OANDA Order Book, etc.) and sets
        #      SENTIMENT_FG_SOURCE=fx_native, the halving is skipped.
        #
        # TODO(round-5 audit): Replace the alternative.me call in
        # analysis/sentiment_data.py:get_fear_greed_index() with a
        # genuine FX-specific sentiment source. Candidates:
        #   - CFTC Commitment of Traders (weekly, free, USD/forex futures)
        #   - IG Client Sentiment API (free with IG account)
        #   - OANDA Order Book snapshot (already wired in
        #     analysis/retail_sentiment.py — could be reused here)
        # Until then, the crypto-proxy contributes at half weight and
        # the result carries data_quality="crypto_proxy".
        import os as _os
        _fx_native_sources = ("cftc_cot", "ig_client", "oanda_orderbook", "fx_native")
        # Operator can declare their source is FX-native via env var
        _operator_declared_native = _os.getenv(
            "SENTIMENT_FG_SOURCE", ""
        ).strip().lower() in _fx_native_sources
        is_fx_native = (source not in ("alternative.me", "fallback")) or _operator_declared_native
        if not is_fx_native and score != 0:
            score = round(score * 0.5)
            desc += " (halved: this is the crypto F&G index used as an unvalidated FX proxy — see TODO in sentiment.py)"

        if source == "alternative.me":
            data_quality = "crypto_proxy"
        elif source == "fallback":
            data_quality = "no_data_fallback"
        else:
            data_quality = "verified"

        result = {
            "value":       index_value,
            "label":       label,
            "impact":      impact,
            "score":       score,
            "data_quality": data_quality,
            "description": desc,
        }

        log.info(
            f"[F&G] Value: {index_value} | Label: {label} | Score: {score:+d} | "
            f"Quality: {data_quality}"
        )
        return result

    # ═══════════════════════════════════════════════════════════
    # 3. CURRENCY STRENGTH METER  ⭐⭐⭐⭐⭐
    # ═══════════════════════════════════════════════════════════

    def currency_strength(
        self,
        strengths: dict[str, float],
        pair: str,
    ) -> dict:
        """
        Major currency-গুলোর relative strength দেখে pair bias বের করো।

        Args:
            strengths : {"USD": 82, "EUR": 45, "GBP": 60, ...}  (0–100 scale)
            pair      : "EURUSD", "GBPJPY" ইত্যাদি

        Returns:
            {
                "base": "EUR",
                "quote": "USD",
                "base_strength": 45,
                "quote_strength": 82,
                "diff": -37,
                "pair_bias": "BEARISH",
                "score": -20,
                "reason": "USD (82) >> EUR (45) — SELL bias"
            }
        """
        # Pair কে base/quote এ ভাগ করো
        pair_clean = pair.upper().replace("/", "").replace("=X", "")

        # Standard 6-char forex pairs
        base  = pair_clean[:3]
        quote = pair_clean[3:6] if len(pair_clean) >= 6 else pair_clean[3:]

        base_str  = strengths.get(base,  50.0)
        quote_str = strengths.get(quote, 50.0)
        diff      = base_str - quote_str

        # Score calculation (-25 to +25)
        score = round(diff * 0.5)
        score = max(-25, min(25, score))

        if diff >= 20:
            bias   = "STRONG_BULLISH"
            reason = f"{base} ({base_str:.0f}) >> {quote} ({quote_str:.0f}) — strong BUY bias"
        elif diff >= 8:
            bias   = "BULLISH"
            reason = f"{base} ({base_str:.0f}) > {quote} ({quote_str:.0f}) — BUY bias"
        elif diff <= -20:
            bias   = "STRONG_BEARISH"
            reason = f"{quote} ({quote_str:.0f}) >> {base} ({base_str:.0f}) — strong SELL bias"
        elif diff <= -8:
            bias   = "BEARISH"
            reason = f"{quote} ({quote_str:.0f}) > {base} ({base_str:.0f}) — SELL bias"
        else:
            bias   = "NEUTRAL"
            reason = f"{base} ({base_str:.0f}) ≈ {quote} ({quote_str:.0f}) — no clear bias"

        result = {
            "pair":           pair,
            "base":           base,
            "quote":          quote,
            "base_strength":  base_str,
            "quote_strength": quote_str,
            "diff":           round(diff, 1),
            "pair_bias":      bias,
            "score":          score,
            "reason":         reason,
        }

        log.info(
            f"[CurrStr] {pair} | {base}: {base_str} vs {quote}: {quote_str} "
            f"| Bias: {bias} | Score: {score:+d}"
        )
        return result

    # ═══════════════════════════════════════════════════════════
    # 4. DXY CORRELATION ANALYSIS  ⭐⭐⭐⭐⭐
    # ═══════════════════════════════════════════════════════════

    def dxy_analysis(
        self,
        dxy_trend: str,
        dxy_change_pct: float = 0.0,
    ) -> dict:
        """
        DXY (USD Index) trend দেখে USD-linked pairs-এ impact calculate করো।

        Args:
            dxy_trend      : "BULLISH" | "BEARISH" | "NEUTRAL"
            dxy_change_pct : DXY-এর % change (positive = DXY rising)

        DXY Correlation:
            DXY ↑  →  EURUSD ↓, GBPUSD ↓, AUDUSD ↓, XAUUSD ↓
            DXY ↓  →  EURUSD ↑, GBPUSD ↑, AUDUSD ↑, XAUUSD ↑
            DXY ↑  →  USDJPY ↑, USDCAD ↑, USDCHF ↑

        Returns:
            {
                "dxy_trend": "BULLISH",
                "dxy_change_pct": 0.3,
                "usd_impact": "STRONG",
                "pair_impacts": {
                    "EURUSD": -15, "GBPUSD": -15,
                    "USDJPY": +15, "XAUUSD": -20
                },
                "score_for_usd_pairs": +15
            }
        """
        # Impact level
        abs_change = abs(dxy_change_pct)
        if abs_change >= 0.5:
            usd_impact = "STRONG"
            base_score = 20
        elif abs_change >= 0.2:
            usd_impact = "MODERATE"
            base_score = 12
        else:
            usd_impact = "WEAK"
            base_score = 6

        # Direction
        if dxy_trend == "BULLISH" or dxy_change_pct > 0:
            direction = "BULLISH"
            # USD-quoted pairs: negative impact (EURUSD, GBPUSD, etc.)
            pair_impacts = {
                "EURUSD": -base_score,
                "GBPUSD": -base_score,
                "AUDUSD": -base_score,
                "NZDUSD": -base_score,
                "XAUUSD": -round(base_score * 1.3),  # gold more sensitive
                "USDJPY": +base_score,
                "USDCAD": +base_score,
                "USDCHF": +base_score,
            }
            score_for_usd_pairs = +base_score
        elif dxy_trend == "BEARISH" or dxy_change_pct < 0:
            direction = "BEARISH"
            pair_impacts = {
                "EURUSD": +base_score,
                "GBPUSD": +base_score,
                "AUDUSD": +base_score,
                "NZDUSD": +base_score,
                "XAUUSD": +round(base_score * 1.3),
                "USDJPY": -base_score,
                "USDCAD": -base_score,
                "USDCHF": -base_score,
            }
            score_for_usd_pairs = -base_score
        else:
            direction        = "NEUTRAL"
            pair_impacts     = {p: 0 for p in ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF", "XAUUSD"]}
            score_for_usd_pairs = 0

        result = {
            "dxy_trend":            direction,
            "dxy_change_pct":       dxy_change_pct,
            "usd_impact":           usd_impact,
            "pair_impacts":         pair_impacts,
            "score_for_usd_pairs":  score_for_usd_pairs,
        }

        log.info(
            f"[DXY] Trend: {direction} | Change: {dxy_change_pct:+.2f}% | "
            f"Impact: {usd_impact} | Score: {score_for_usd_pairs:+d}"
        )
        return result

    # ═══════════════════════════════════════════════════════════
    # 5. FINAL SENTIMENT SCORE  ⭐⭐⭐⭐⭐
    # ═══════════════════════════════════════════════════════════

    def final_sentiment_score(
        self,
        pair:              str,
        retail_long_pct:   float,
        fg_index:          float,
        currency_strengths: dict[str, float],
        dxy_trend:         str,
        dxy_change_pct:    float = 0.0,
        retail_source:     str = "approximated_from_price",
        fg_source:         str = "alternative.me",
    ) -> dict:
        """
        সব sentiment source combine করে final score বের করো।

        Formula:
            Final = Retail score + F&G score + Currency Strength score + DXY score

        Score range: roughly -75 to +75
            Positive = BULLISH bias
            Negative = BEARISH bias

        Args:
            retail_source, fg_source: pass through the "source" field from
                sentiment_data.py's get_retail_positioning()/
                get_fear_greed_index() so retail_positioning()/fear_greed()
                can correctly dampen non-genuine data (institutional review,
                Finding H-3). Previously this metadata was dropped before
                reaching this method — every caller implicitly got
                full-confidence scoring regardless of data quality.

        Returns full sentiment dict with all sub-results, plus a top-level
        "data_quality" summary so a consumer that only reads the numeric
        score can still tell whether inputs were degraded/proxied.
        """
        # ── Sub-analyses ──────────────────────────────────────
        retail_result = self.retail_positioning(pair, retail_long_pct, source=retail_source)
        fg_result     = self.fear_greed(fg_index, source=fg_source)
        cs_result     = self.currency_strength(currency_strengths, pair)
        dxy_result    = self.dxy_analysis(dxy_trend, dxy_change_pct)

        # DXY score for this specific pair
        pair_clean  = pair.upper().replace("/", "").replace("=X", "")[:6]
        dxy_pair_score = dxy_result["pair_impacts"].get(pair_clean, dxy_result["score_for_usd_pairs"])

        # ── Aggregate ──────────────────────────────────────────
        total_score = (
            retail_result["score"]
            + fg_result["score"]
            + cs_result["score"]
            + dxy_pair_score
        )

        # ── Data quality summary (institutional review, item §7.7) ──────
        degraded = [
            f"retail:{retail_result['data_quality']}"
            for _ in [0] if retail_result["data_quality"] != "verified"
        ] + [
            f"fear_greed:{fg_result['data_quality']}"
            for _ in [0] if fg_result["data_quality"] != "verified"
        ]
        data_quality = "degraded" if degraded else "verified"

        # ── Bias label ─────────────────────────────────────────
        if total_score >= 30:
            bias       = "STRONG_BULLISH"
            confidence = min(95, 60 + abs(total_score))
        elif total_score >= 10:
            bias       = "BULLISH"
            confidence = min(80, 50 + abs(total_score))
        elif total_score <= -30:
            bias       = "STRONG_BEARISH"
            confidence = min(95, 60 + abs(total_score))
        elif total_score <= -10:
            bias       = "BEARISH"
            confidence = min(80, 50 + abs(total_score))
        else:
            bias       = "NEUTRAL"
            confidence = 40

        # ── Reasons list ───────────────────────────────────────
        reasons = []
        if retail_result["score"] != 0:
            reasons.append(retail_result["reason"])
        if fg_result["score"] != 0:
            reasons.append(fg_result["description"])
        if cs_result["score"] != 0:
            reasons.append(cs_result["reason"])
        if dxy_pair_score != 0:
            reasons.append(
                f"DXY {dxy_result['dxy_trend']} — "
                f"{pair_clean} impact: {dxy_pair_score:+d}"
            )

        result = {
            "pair":              pair,
            "sentiment_score":   total_score,
            "bias":              bias,
            "confidence":        confidence,
            "data_quality":      data_quality,
            "reasons":           reasons,
            # Sub-results
            "retail":            retail_result,
            "fear_greed":        fg_result,
            "currency_strength": cs_result,
            "dxy":               dxy_result,
            "dxy_pair_score":    dxy_pair_score,
            # Score breakdown
            "score_breakdown": {
                "retail":            retail_result["score"],
                "fear_greed":        fg_result["score"],
                "currency_strength": cs_result["score"],
                "dxy":               dxy_pair_score,
                "total":             total_score,
            },
        }

        log.info(
            f"[SentimentEngine] {pair} | "
            f"Score: {total_score:+d} | Bias: {bias} | Conf: {confidence}%"
        )
        return result

    # ═══════════════════════════════════════════════════════════
    # CONFLICT DETECTION  ⭐
    # ═══════════════════════════════════════════════════════════

    def detect_conflict(
        self,
        technical_signal: str,
        sentiment_result: dict,
    ) -> dict:
        """
        Technical signal এবং Sentiment bias-এর মধ্যে conflict আছে কিনা।

        Args:
            technical_signal : "BUY" | "SELL" | "NO TRADE"
            sentiment_result : final_sentiment_score() এর output

        Returns:
            {
                "has_conflict": True,
                "technical": "BUY",
                "sentiment": "BEARISH",
                "confidence_adjustment": -20,
                "recommendation": "WAIT — sentiment opposes technical setup"
            }
        """
        sent_bias = sentiment_result.get("bias", "NEUTRAL")

        # Bearish sentiment vs BUY signal
        tech_bullish  = technical_signal == "BUY"
        sent_bearish  = "BEARISH" in sent_bias
        tech_bearish  = technical_signal == "SELL"
        sent_bullish  = "BULLISH" in sent_bias

        conflict = (tech_bullish and sent_bearish) or (tech_bearish and sent_bullish)

        if conflict:
            if "STRONG" in sent_bias:
                adj   = -25
                rec   = "WAIT — Strong sentiment opposes technical. High conflict risk."
            else:
                adj   = -15
                rec   = "CAUTION — Sentiment conflicts with technical. Reduce position size."
        else:
            adj = 0
            rec = "ALIGNED — Technical and sentiment agree. Normal confidence."

        return {
            "has_conflict":           conflict,
            "technical":              technical_signal,
            "sentiment_bias":         sent_bias,
            "sentiment_score":        sentiment_result.get("sentiment_score", 0),
            "confidence_adjustment":  adj,
            "recommendation":         rec,
        }

    # ═══════════════════════════════════════════════════════════
    # AI CONTEXT  (Decision Agent-এ inject করার জন্য)
    # ═══════════════════════════════════════════════════════════

    def get_ai_context(self, sentiment_result: dict) -> dict:
        """Decision Agent-এ পাঠানোর জন্য clean context dict।"""
        return {
            "sentiment_score":  sentiment_result.get("sentiment_score", 0),
            "sentiment_bias":   sentiment_result.get("bias", "NEUTRAL"),
            "sentiment_conf":   sentiment_result.get("confidence", 0),
            "retail_long_pct":  sentiment_result.get("retail", {}).get("retail_long_pct"),
            "fg_label":         sentiment_result.get("fear_greed", {}).get("label"),
            "currency_bias":    sentiment_result.get("currency_strength", {}).get("pair_bias"),
            "dxy_trend":        sentiment_result.get("dxy", {}).get("dxy_trend"),
            "sentiment_reasons": sentiment_result.get("reasons", []),
        }

    # ═══════════════════════════════════════════════════════════
    # PRINT SUMMARY
    # ═══════════════════════════════════════════════════════════

    def print_summary(self, sentiment_result: dict) -> None:
        bar = "═" * 52
        pair  = sentiment_result.get("pair", "")
        score = sentiment_result.get("sentiment_score", 0)
        bias  = sentiment_result.get("bias", "NEUTRAL")
        conf  = sentiment_result.get("confidence", 0)

        icons = {
            "STRONG_BULLISH": "🟢🟢",
            "BULLISH":        "🟢",
            "NEUTRAL":        "🟡",
            "BEARISH":        "🔴",
            "STRONG_BEARISH": "🔴🔴",
        }
        icon = icons.get(bias, "⚪")

        print(f"\n{bar}")
        print(f"  🧠  SENTIMENT ENGINE  —  {pair}")
        print(bar)
        print(f"  Final Score     : {score:+d}")
        print(f"  Sentiment Bias  : {icon}  {bias}")
        print(f"  Confidence      : {conf}%")
        print()

        # Score breakdown
        bd = sentiment_result.get("score_breakdown", {})
        print("  ── Score Breakdown ──")
        print(f"  Retail Position : {bd.get('retail', 0):+d}")
        print(f"  Fear & Greed    : {bd.get('fear_greed', 0):+d}")
        print(f"  Currency Str.   : {bd.get('currency_strength', 0):+d}")
        print(f"  DXY Impact      : {bd.get('dxy', 0):+d}")
        print(f"  {'─'*30}")
        print(f"  TOTAL           : {bd.get('total', 0):+d}")
        print()

        # Sub-results
        retail = sentiment_result.get("retail", {})
        fg     = sentiment_result.get("fear_greed", {})
        cs     = sentiment_result.get("currency_strength", {})
        dxy    = sentiment_result.get("dxy", {})

        print("  ── Details ──")
        print(f"  Retail Long     : {retail.get('retail_long_pct', 0):.0f}%  →  {retail.get('contrarian_bias', '')}")
        print(f"  Fear & Greed    : {fg.get('value', 0):.0f}  →  {fg.get('label', '')}")
        print(
            f"  Currency        : {cs.get('base', '')} {cs.get('base_strength', 0):.0f} "
            f"vs {cs.get('quote', '')} {cs.get('quote_strength', 0):.0f}"
        )
        print(f"  DXY             : {dxy.get('dxy_trend', '')}  ({dxy.get('usd_impact', '')} impact)")
        print()

        # Reasons
        reasons = sentiment_result.get("reasons", [])
        if reasons:
            print("  ── Reasons ──")
            for r in reasons:
                print(f"  • {r}")

        print(bar + "\n")