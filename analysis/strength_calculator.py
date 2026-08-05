# analysis/strength_calculator.py  —  Day 64 | Currency Strength — Score Calculator
# ============================================================
# একটা single currency pair (যেমন EURUSD)-এর candle data থেকে
# "এই move-এ base currency কতটা শক্তিশালী আচরণ করছে" সেটা বের করার
# নিচু-স্তরের (low-level) math এখানে থাকে।
#
# CurrencyStrengthEngine প্রতিটা cross pair-এর জন্য এই calculator
# কল করে — base currency-তে +score আর quote currency-তে -score যোগ
# করে (কারণ pair bullish হলে base শক্তিশালী, quote দুর্বল বোঝায়)।
#
# Score components (doc অনুযায়ী):
#   strength_score = price_change + trend + momentum + volatility_adjustment
#   Normalize  ->  0 - 100
# ============================================================

import pandas as pd
from utils.logger import get_logger

log = get_logger("strength_calculator")


class StrengthCalculator:
    """
    Usage:
        calc = StrengthCalculator()
        pair_score = calc.compute_pair_score(df, ind_ctx)
        # pair_score['total'] -> base currency contribution
        # (quote currency-র জন্য caller এটাকে negate করে নেয়)

        normalized = calc.normalize_scores({"USD": 12.4, "EUR": -3.1, ...})
    """

    # ── Component weights (যোগফল = 1.0) ─────────────────────────
    PRICE_CHANGE_WEIGHT = 0.35
    TREND_WEIGHT        = 0.25
    MOMENTUM_WEIGHT      = 0.25
    VOLATILITY_WEIGHT     = 0.15

    # ── Lookback windows (candle count) ─────────────────────────
    PRICE_CHANGE_LOOKBACK = 20
    MOMENTUM_SHORT          = 5
    MOMENTUM_LONG            = 10
    VOLATILITY_PERIOD         = 14

    TREND_SCORE_MAP = {
        "strong_bullish": 100,
        "bullish":         50,
        "strong_bearish": -100,
        "bearish":         -50,
    }

    # ═══════════════════════════════════════════════════════
    # MAIN ENTRY — একটা pair-এর সব component একসাথে
    # ═══════════════════════════════════════════════════════

    def compute_pair_score(self, df: pd.DataFrame, ind_ctx: dict) -> dict:
        """
        Returns:
            {
                'price_change':   float,
                'trend':          float,
                'momentum':       float,
                'volatility_adj': float,
                'total':          float,   # weighted sum — base currency contribution
            }
        """
        price_change = self._price_change_score(df)
        trend        = self._trend_score(ind_ctx)
        momentum     = self._momentum_score(df)
        vol_adj      = self._volatility_adjustment(df, ind_ctx)

        total = (
            price_change * self.PRICE_CHANGE_WEIGHT +
            trend         * self.TREND_WEIGHT +
            momentum      * self.MOMENTUM_WEIGHT +
            vol_adj       * self.VOLATILITY_WEIGHT
        )

        return {
            "price_change":   round(price_change, 2),
            "trend":          round(trend, 2),
            "momentum":       round(momentum, 2),
            "volatility_adj": round(vol_adj, 2),
            "total":          round(total, 2),
        }

    # ═══════════════════════════════════════════════════════
    # 1. PRICE CHANGE SCORE
    # ═══════════════════════════════════════════════════════

    def _price_change_score(self, df: pd.DataFrame) -> float:
        """
        Lookback window জুড়ে % price change — scaled to roughly -100..100।
        Bullish move (base উঠছে) → positive score।
        """
        closes   = df["close"].values
        lookback = min(self.PRICE_CHANGE_LOOKBACK, len(closes) - 1)
        if lookback < 1:
            return 0.0

        start = closes[-lookback - 1]
        if start == 0:
            return 0.0

        change_pct = (closes[-1] - start) / start * 100
        # Typical intraday forex move ~0-1% — trend/momentum component-এর
        # সাথে comparable range-এ আনতে scale up করা হয়েছে
        return float(max(-100.0, min(100.0, change_pct * 50)))

    # ═══════════════════════════════════════════════════════
    # 2. TREND SCORE
    # ═══════════════════════════════════════════════════════

    def _trend_score(self, ind_ctx: dict) -> float:
        """Indicators.get_ai_context()-এর 'trend' string থেকে স্কোর।"""
        trend = ind_ctx.get("trend", "") or ""
        for key, val in self.TREND_SCORE_MAP.items():
            if key in trend:
                return float(val)
        return 0.0

    # ═══════════════════════════════════════════════════════
    # 3. MOMENTUM SCORE  (Rate-of-Change Acceleration)
    # ═══════════════════════════════════════════════════════

    def _momentum_score(self, df: pd.DataFrame) -> float:
        """
        শুধু "দাম বাড়ছে" না — "বাড়ার গতি বাড়ছে নাকি কমছে" সেটা মাপে।

        recent_roc > prior_roc  → momentum accelerating  (positive)
        recent_roc < prior_roc  → momentum decelerating  (negative)
        """
        closes = df["close"].values
        n      = len(closes)
        if n < self.MOMENTUM_LONG + 1:
            return 0.0

        short, long_ = self.MOMENTUM_SHORT, self.MOMENTUM_LONG

        p_now   = closes[-1]
        p_short = closes[-1 - short]
        p_long  = closes[-1 - long_]

        if p_short == 0 or p_long == 0:
            return 0.0

        recent_roc = (p_now - p_short) / p_short * 100
        prior_roc  = (p_short - p_long) / p_long * 100

        momentum = recent_roc - prior_roc
        return float(max(-100.0, min(100.0, momentum * 80)))

    # ═══════════════════════════════════════════════════════
    # 4. VOLATILITY ADJUSTMENT
    # ═══════════════════════════════════════════════════════

    def _volatility_adjustment(self, df: pd.DataFrame, ind_ctx: dict) -> float:
        """
        ATR স্বাভাবিকের চেয়ে expand করছে আর trend direction-এ move হচ্ছে
        → সেই currency-র move-টা "real" — bonus দাও। শুধু noise হলে
        কিছুই যোগ হয় না।
        """
        atr   = ind_ctx.get("atr", 0) or 0
        price = ind_ctx.get("price", ind_ctx.get("close", 0)) or 0
        if price == 0:
            return 0.0

        atr_pct     = atr / price * 100
        avg_atr_pct = self._avg_atr_pct(df)
        if avg_atr_pct == 0:
            return 0.0

        expansion = atr_pct / avg_atr_pct   # >1 মানে এখন স্বাভাবিকের চেয়ে বেশি move হচ্ছে

        trend     = ind_ctx.get("trend", "") or ""
        direction = 1 if "bullish" in trend else (-1 if "bearish" in trend else 0)

        score = direction * min(40.0, (expansion - 1) * 40)
        return float(max(-50.0, min(50.0, score)))

    def _avg_atr_pct(self, df: pd.DataFrame) -> float:
        if "atr" not in df.columns or "close" not in df.columns:
            return 0.0
        recent         = df.tail(self.VOLATILITY_PERIOD * 3)
        atr_pct_series = (recent["atr"] / recent["close"] * 100).dropna()
        if atr_pct_series.empty:
            return 0.0
        return float(atr_pct_series.mean())

    # ═══════════════════════════════════════════════════════
    # NORMALIZATION — raw avg score (per currency) → 0-100
    # ═══════════════════════════════════════════════════════

    # compute_pair_score()'s weighted components are each bounded to
    # roughly ±100 (price_change/trend/momentum) or ±50 (volatility_adj),
    # and the weights sum to 1.0, so a single-pair "total" — and therefore
    # the per-currency average of several such totals — stays within
    # roughly this range in practice. Used as a fixed reference scale for
    # normalization below.
    RAW_SCORE_SCALE = 100.0

    def normalize_scores(self, raw_scores: dict) -> dict:
        """
        Currency raw scores (avg of compute_pair_score()['total'] across
        that currency's pairs) কে 0-100 স্কেলে আনে।

        FIX (audit H1): আগে এখানে min-max normalization হতো — প্রতিটা
        cycle-এর সবচেয়ে দুর্বল currency 0 আর সবচেয়ে শক্তিশালীটা 100 হয়ে
        যেত, independent of actual strength। এতে কিছু সমস্যা হতো:
          - Historical comparable না: আজকের 70 আর গতকালের 70 ভিন্ন জিনিস
            বোঝাতে পারত, কারণ scale প্রতিবার basket-এর min/max-এর উপর
            নির্ভর করত (momentum/cycle detection যেটার উপর নির্ভর করে)।
          - Outlier-sensitive: একটা extreme currency পুরো basket-এর scale
            চেপে দিত (v_min/v_max একাই পুরো mapping ঠিক করে দেয়)।
          - সবগুলো currency দুর্বল হলেও, min-max সবসময় একজনকে "100"
            বানিয়ে দিত — misleading।

        এখন raw score-কে একটা fixed reference scale (RAW_SCORE_SCALE)-এর
        বিপরীতে map করা হয়, যাতে 70 মানে সবসময় একই জিনিস বোঝায় —
        currency সত্যিই শক্তিশালী, শুধু আজকের basket-এ relatively কম
        দুর্বল না। Monotonic, outlier-resistant (কোনো একটা currency-র
        extreme value বাকিদের scale নষ্ট করে না), এবং cycle-to-cycle
        comparable।
        """
        if not raw_scores:
            return {}

        normalized = {}
        for cur, val in raw_scores.items():
            # FIX (audit M1/M2): guard against NaN raw scores (e.g. every
            # pair fetch for a currency failed, leaving a NaN average)
            # reaching the ranker as a bogus number instead of failing loudly.
            if val is None or pd.isna(val):
                log.warning(f"[StrengthCalculator] NaN/None raw score for {cur} — defaulting to neutral 50.0")
                normalized[cur] = 50.0
                continue
            clipped = max(-self.RAW_SCORE_SCALE, min(self.RAW_SCORE_SCALE, val))
            normalized[cur] = round((clipped + self.RAW_SCORE_SCALE) / (2 * self.RAW_SCORE_SCALE) * 100, 1)
        return normalized