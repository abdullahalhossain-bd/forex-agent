# fundamental/fundamental_sentiment.py  —  Day 43 | Fundamental Sentiment Score
# ============================================================
# Economic event history (database/db.py এর `economic_history` table)
# থেকে একটা currency-র "fundamental score" বের করে — যেমন doc-এ লেখা:
#
#     USD Fundamental Score: +35 Bullish
#
# এই score টা MasterAnalyst-এর context-এ যাবে (sentiment.py-এর
# market-psychology score-এর পাশাপাশি, কিন্তু এটা purely news-driven)।
# ============================================================

from typing import Optional

from database.db import TraderDB
from fundamental.news_filter import NewsFilter
from utils.logger import get_logger

log = get_logger("fundamental_sentiment")

MAJOR_CURRENCIES = ["USD", "EUR", "GBP", "JPY"]

# raw_score (bullish_count - bearish_count) → scaled -100..+100
SCALE_PER_EVENT = 12   # প্রতিটা net event reaction কত পয়েন্ট নাড়াবে

# Day 98+ FIX: sample-size guardrails. Previously a currency with only 2-3
# recent events could hit a STRONG_BULLISH/STRONG_BEARISH label — the same
# label a currency with 20+ events and a clear majority would get. That's
# not a meaningful distinction; 2-3 events is noise, not evidence. These
# thresholds require enough sample size before the score is allowed to
# claim strong conviction, and force NEUTRAL below a bare minimum.
#
# Day 99+ FIX: thresholds raised from 3/6 to 5/10. With only 3 events the
# "directional" label could be triggered by what is effectively a 2-vs-1
# majority — well within normal sampling noise. 5 events is the smallest
# sample where a 4-vs-1 split starts to be meaningfully unlikely by
# chance. Similarly, 6 events was too low for "STRONG" — at 6 events a
# 5-vs-1 split would be labelled STRONG, but a 5-vs-1 split on 6 events
# has a ~p=0.10 chance of occurring even if the true distribution is
# 50/50. 10 events raises the bar so STRONG labels actually mean
# something. Bayesian shrinkage (see score_currency) further pulls the
# effective score toward zero when sample size is small.
MIN_SAMPLE_FOR_DIRECTIONAL = 5   # below this: label forced to NEUTRAL
MIN_SAMPLE_FOR_STRONG      = 10  # below this: capped at BULLISH/BEARISH, no STRONG_*

# Bayesian prior strength — number of "prior neutral" events to mix in
# when computing the effective score. With PRIOR_STRENGTH=3 and a real
# sample of 5 events split 4-1, the effective split becomes
# (4+1.5) vs (1+1.5) = 5.5 vs 2.5 — same direction, but dampened toward
# zero, reflecting our prior belief that an unknown currency is
# approximately neutral. As sample_size grows, the prior's influence
# shrinks (e.g. 20 events 15-5 becomes 16.5 vs 6.5 — barely moved).
# This is a standard Bayesian shrinkage estimator and avoids the cliff
# where a label flips just because the sample crosses a threshold.
PRIOR_STRENGTH = 3.0


class FundamentalSentimentScore:
    """
    Usage:
        fs = FundamentalSentimentScore()
        usd_score = fs.score_currency("USD")
        pair_score = fs.score_pair("EURUSD")
        ctx = fs.get_ai_context(pair_score)
    """

    def __init__(self, db: Optional[TraderDB] = None, news_filter: Optional[NewsFilter] = None) -> None:
        self.db = db or TraderDB()
        self.news_filter = news_filter or NewsFilter()

    # ─────────────────────────────────────────────
    # SINGLE CURRENCY SCORE
    # ─────────────────────────────────────────────

    def score_currency(self, currency: str, lookback: int = 10) -> dict:
        """
        একটা currency-র fundamental score বের করে — সাম্প্রতিক
        economic_history reaction-গুলো + upcoming high-impact risk মিলিয়ে।

        Returns:
            {
                "currency": "USD",
                "score": 35,
                "label": "BULLISH",
                "sample_size": 6,
                "upcoming_risk": "HIGH",
                "reason": "3 bullish vs 1 bearish reaction in recent history; ..."
            }
        """
        currency = currency.upper()
        bias = self.db.get_currency_fundamental_bias(currency, lookback=lookback)

        raw_score   = bias["raw_score"]
        sample_size = bias["sample_size"]

        # Day 99+ FIX: Bayesian shrinkage. Instead of using raw_score
        # (bullish - bearish) directly, mix in PRIOR_STRENGTH neutral
        # "pseudo-events" so small samples are pulled toward zero. The
        # effective raw_score becomes:
        #     raw_score * sample_size / (sample_size + PRIOR_STRENGTH)
        # This is mathematically equivalent to adding PRIOR_STRENGTH/2
        # to each side of the bullish/bearish split. With sample_size=5
        # and PRIOR_STRENGTH=3, the shrinkage factor is 5/8 = 0.625 — a
        # raw_score of +5 becomes +3.1. With sample_size=20, the factor
        # is 20/23 ≈ 0.87 — barely moved. This eliminates the "2-vs-1
        # on 3 events = BULLISH" cliff without discarding the signal
        # when the sample is genuinely large.
        shrinkage = sample_size / (sample_size + PRIOR_STRENGTH) if sample_size > 0 else 0.0
        shrunk_score = raw_score * shrinkage
        score = max(-100, min(100, shrunk_score * SCALE_PER_EVENT))

        # Day 98+ FIX: gate label strength on sample size, not just score
        # magnitude. A large score built on 2 events is not the same
        # quality of signal as the same score built on 15 events.
        # Day 99+ FIX: thresholds raised (see module-level constants).
        if sample_size < MIN_SAMPLE_FOR_DIRECTIONAL:
            label = "NEUTRAL"
        elif score >= 25:
            label = "STRONG_BULLISH" if (score >= 50 and sample_size >= MIN_SAMPLE_FOR_STRONG) else "BULLISH"
        elif score <= -25:
            label = "STRONG_BEARISH" if (score <= -50 and sample_size >= MIN_SAMPLE_FOR_STRONG) else "BEARISH"
        else:
            label = "NEUTRAL"

        # Day 98+ FIX: expose how much evidence backs this score so the
        # Decision Layer can weight it accordingly, independent of the
        # label downgrade above (e.g. a BULLISH label with low sample_size
        # confidence is still meaningfully weaker than one with high
        # confidence, even though the label text is the same).
        sample_confidence = round(min(100, (sample_size / MIN_SAMPLE_FOR_STRONG) * 100))

        # Upcoming news risk — high-impact event আসন্ন থাকলে score-এর
        # উপর confidence কমানো উচিত (volatile reversal সম্ভব)
        upcoming_risk = self._upcoming_risk_for_currency(currency)

        reason = (
            f"{bias['bullish_count']} bullish vs {bias['bearish_count']} bearish "
            f"reaction in last {bias['sample_size']} {currency} events"
            if bias["sample_size"] else
            f"No recent {currency} economic history — neutral by default"
        )
        if sample_size < MIN_SAMPLE_FOR_DIRECTIONAL:
            reason += f" (below minimum sample size of {MIN_SAMPLE_FOR_DIRECTIONAL} — forced NEUTRAL)"

        result = {
            "currency":          currency,
            "score":             round(score),
            "label":             label,
            "sample_size":       bias["sample_size"],
            "sample_confidence": sample_confidence,
            "upcoming_risk":     upcoming_risk,
            "reason":            reason,
        }

        log.info(
            f"[FundamentalSentiment] {currency} | Score: {result['score']:+d} | "
            f"Label: {label} | Sample: {sample_size} (conf={sample_confidence}%) | "
            f"Upcoming risk: {upcoming_risk}"
        )
        return result

    def _upcoming_risk_for_currency(self, currency: str) -> str:
        """এই currency-র জন্য কোনো high-impact event আসন্ন কিনা (3h window)।"""
        try:
            check = self.news_filter.check(f"{currency}USD" if currency != "USD" else "EURUSD")
            for ev in check.get("upcoming_events", []):
                if ev.get("currency") == currency:
                    return ev.get("volatility", {}).get("level", "LOW")
            if not check.get("trade_allowed", True):
                for ev in check.get("flagged_events", []):
                    if ev.get("currency") == currency:
                        return ev.get("volatility", {}).get("level", "HIGH")
        except Exception as e:
            log.warning(f"[FundamentalSentiment] upcoming_risk check failed: {e}")
        return "LOW"

    # ─────────────────────────────────────────────
    # PAIR SCORE (base vs quote)
    # ─────────────────────────────────────────────

    def score_pair(self, pair: str, lookback: int = 10) -> dict:
        """
        EURUSD হলে EUR vs USD fundamental score-এর difference থেকে
        pair-level bias বের করে।

        Returns:
            {
                "pair": "EURUSD",
                "base": "EUR", "quote": "USD",
                "base_score": -10, "quote_score": 35,
                "diff": -45,
                "pair_bias": "BEARISH",
                "reason": "USD fundamentals stronger than EUR"
            }
        """
        pair_clean = pair.upper().replace("/", "").replace("=X", "")
        base  = pair_clean[:3]
        quote = pair_clean[3:6] if len(pair_clean) >= 6 else pair_clean[3:]

        base_result  = self.score_currency(base, lookback=lookback)
        quote_result = self.score_currency(quote, lookback=lookback)

        diff = base_result["score"] - quote_result["score"]

        # Day 98+ FIX: same sample-size gate as score_currency — a pair
        # bias shouldn't claim STRONG_* conviction if either leg's score
        # is itself built on too little evidence.
        both_well_sampled = (
            base_result["sample_size"] >= MIN_SAMPLE_FOR_STRONG
            and quote_result["sample_size"] >= MIN_SAMPLE_FOR_STRONG
        )

        if diff >= 30:
            bias = "STRONG_BULLISH" if both_well_sampled else "BULLISH"
        elif diff >= 10:
            bias = "BULLISH"
        elif diff <= -30:
            bias = "STRONG_BEARISH" if both_well_sampled else "BEARISH"
        elif diff <= -10:
            bias = "BEARISH"
        else:
            bias = "NEUTRAL"

        reason = (
            f"{base} fundamentals stronger than {quote}" if diff > 10 else
            f"{quote} fundamentals stronger than {base}" if diff < -10 else
            f"{base} and {quote} fundamentals roughly balanced"
        )

        result = {
            "pair":        pair,
            "base":        base,
            "quote":       quote,
            "base_score":  base_result["score"],
            "quote_score": quote_result["score"],
            "diff":        diff,
            "pair_bias":   bias,
            "reason":      reason,
            "base_detail":  base_result,
            "quote_detail": quote_result,
        }

        log.info(
            f"[FundamentalSentiment] {pair} | {base}:{base_result['score']:+d} "
            f"vs {quote}:{quote_result['score']:+d} | Bias: {bias}"
        )
        return result

    # ─────────────────────────────────────────────
    # AI CONTEXT  (MasterAnalyst handoff)
    # ─────────────────────────────────────────────

    def get_ai_context(self, pair_result: dict) -> dict:
        base_detail  = pair_result.get("base_detail", {}) or {}
        quote_detail = pair_result.get("quote_detail", {}) or {}
        # Day 98+ FIX: additive field — overall confidence is the weaker
        # of the two legs' sample confidence, since a pair bias is only as
        # trustworthy as its least-supported currency.
        sample_confidence = min(
            base_detail.get("sample_confidence", 0),
            quote_detail.get("sample_confidence", 0),
        )
        return {
            "fundamental_pair_bias":         pair_result.get("pair_bias", "NEUTRAL"),
            "fundamental_base_score":        pair_result.get("base_score", 0),
            "fundamental_quote_score":       pair_result.get("quote_score", 0),
            "fundamental_diff":              pair_result.get("diff", 0),
            "fundamental_reason":            pair_result.get("reason", ""),
            "fundamental_sample_confidence": sample_confidence,
        }

    # ─────────────────────────────────────────────
    # PRINT SUMMARY
    # ─────────────────────────────────────────────

    def print_summary(self, pair_result: dict) -> None:
        bar = "═" * 50
        log.info(bar)
        log.info(f"  💵  FUNDAMENTAL SENTIMENT SCORE  (Day 43)")
        log.info(bar)
        log.info(f"  Pair        : {pair_result['pair']}")
        log.info(
            f"  {pair_result['base']} Score : {pair_result['base_score']:+d}  "
            f"({pair_result['base_detail']['label']})"
        )
        log.info(
            f"  {pair_result['quote']} Score : {pair_result['quote_score']:+d}  "
            f"({pair_result['quote_detail']['label']})"
        )
        log.info(f"  Pair Bias   : {pair_result['pair_bias']}")
        log.info(f"  Reason      : {pair_result['reason']}")
        log.info(bar)