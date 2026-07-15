# analysis/currency_ranker.py  —  Day 64 | Currency Ranking & Pair Opportunity Finder
# ============================================================
# CurrencyStrengthEngine থেকে আসা normalized strength scores (0-100)
# নিয়ে:
#
#   ✅ Strongest / Weakest ranking
#   ✅ Strong vs Weak pair opportunity finder (strength_difference filter)
#   ✅ Currency Heatmap matrix
#   ✅ Correlation protection (same-currency-based pairs গ্রুপ করা)
#   ✅ Currency cycle detection (history থেকে strengthening/weakening trend)
# ============================================================

import numpy as np
from utils.logger import get_logger

log = get_logger("currency_ranker")


class CurrencyRanker:
    """
    Usage:
        ranker = CurrencyRanker()
        ranking       = ranker.rank(strengths)
        opportunities = ranker.find_best_pairs(strengths, min_diff=40)
        opportunities = ranker.detect_correlation_risk(opportunities)
        heatmap       = ranker.build_heatmap(strengths)
        cycle         = ranker.detect_cycle(history)
    """

    # ── Strength-difference trade-quality tiers ─────────────────
    VERY_HIGH_DIFF = 60
    HIGH_DIFF      = 40   # doc rule: difference > 40 হলে trade allowed
    MEDIUM_DIFF    = 25

    TOP_N = 3   # strongest/weakest list-এর সাইজ

    # ═══════════════════════════════════════════════════════
    # 1. RANKING
    # ═══════════════════════════════════════════════════════

    def rank(self, strengths: dict) -> dict:
        """
        strengths: {"USD": 72, "EUR": 45, "GBP": 81, "JPY": 28, ...}
        """
        if not strengths:
            return {"ranked": [], "strongest": [], "weakest": []}

        ordered = sorted(strengths.items(), key=lambda kv: kv[1], reverse=True)

        strongest = [c for c, _ in ordered[: self.TOP_N]]
        weakest   = [c for c, _ in ordered[-self.TOP_N:]][::-1]   # সবচেয়ে দুর্বলটা প্রথমে

        return {
            "ranked":    ordered,    # সব currency, strongest → weakest
            "strongest": strongest,
            "weakest":   weakest,
        }

    # ═══════════════════════════════════════════════════════
    # 2. STRONG vs WEAK PAIR FINDER  ⭐⭐⭐⭐⭐
    # ═══════════════════════════════════════════════════════

    def find_best_pairs(self, strengths: dict, min_diff: int = 40, max_pairs: int = 10) -> list[dict]:
        """
        প্রতিটা currency-pair combination-এর strength difference হিসাব করে,
        min_diff-এর নিচে যেগুলো সেগুলো বাদ দেয় (Avoid Bad Trades — doc #11)।

        Returns: strength_difference অনুযায়ী sorted (descending) opportunity list।
        """
        currencies    = list(strengths.keys())
        opportunities = []

        for i in range(len(currencies)):
            for j in range(i + 1, len(currencies)):
                c1, c2 = currencies[i], currencies[j]
                s1, s2 = strengths[c1], strengths[c2]

                if s1 == s2:
                    continue

                strong, weak               = (c1, c2) if s1 > s2 else (c2, c1)
                strong_score, weak_score   = max(s1, s2), min(s1, s2)
                diff                       = round(strong_score - weak_score, 1)

                if diff < min_diff:
                    continue

                opportunities.append({
                    "pair":                f"{strong}{weak}",
                    "direction":           "BUY",   # strong/weak → strong-side কেনা হবে
                    "strong_currency":     strong,
                    "weak_currency":       weak,
                    "strong_strength":     strong_score,
                    "weak_strength":       weak_score,
                    "strength_difference": diff,
                    "trade_quality":       self._grade_difference(diff),
                })

        opportunities.sort(key=lambda o: o["strength_difference"], reverse=True)
        return opportunities[:max_pairs]

    def _grade_difference(self, diff: float) -> str:
        if diff >= self.VERY_HIGH_DIFF:
            return "VERY_HIGH"
        if diff >= self.HIGH_DIFF:
            return "HIGH"
        if diff >= self.MEDIUM_DIFF:
            return "MEDIUM"
        return "LOW"

    # ═══════════════════════════════════════════════════════
    # 3. CORRELATION PROTECTION  ⭐
    # ═══════════════════════════════════════════════════════

    def detect_correlation_risk(self, opportunities: list[dict]) -> list[dict]:
        """
        একাধিক opportunity একই currency-র উপর ভিত্তি করে তৈরি হলে
        (যেমন GBPJPY BUY আর AUDJPY BUY — দুটোই JPY weakness trade)
        সেগুলোকে correlation group-এ চিহ্নিত করো এবং risk note যোগ করো।
        """
        if not opportunities:
            return opportunities

        # currency -> কোন কোন opportunity index এই currency ব্যবহার করছে
        usage: dict[str, list[int]] = {}
        for idx, opp in enumerate(opportunities):
            usage.setdefault(opp["strong_currency"], []).append(idx)
            usage.setdefault(opp["weak_currency"], []).append(idx)

        enriched = [dict(o) for o in opportunities]

        for currency, idxs in usage.items():
            if len(idxs) < 2:
                continue
            for idx in idxs:
                enriched[idx].setdefault("correlation_groups", []).append({
                    "shared_currency": currency,
                    "group_size":      len(idxs),
                })

        for opp in enriched:
            groups = opp.get("correlation_groups", [])
            if groups:
                shared = ", ".join(g["shared_currency"] for g in groups)
                opp["correlated"]            = True
                opp["correlation_note"]      = (
                    f"Shares exposure with other active opportunities via {shared} — "
                    f"reduce combined position size to manage correlated risk"
                )
                opp["suggested_size_divisor"] = max(g["group_size"] for g in groups)
            else:
                opp["correlated"]             = False
                opp["correlation_note"]       = "No correlation risk detected"
                opp["suggested_size_divisor"] = 1

        return enriched

    # ═══════════════════════════════════════════════════════
    # 4. CURRENCY HEATMAP
    # ═══════════════════════════════════════════════════════

    def build_heatmap(self, strengths: dict) -> dict:
        """
        Pairwise strength-difference matrix।
        heatmap['matrix']['USD']['JPY'] = strengths['USD'] - strengths['JPY']
        """
        currencies = list(strengths.keys())
        matrix     = {}
        for row in currencies:
            matrix[row] = {}
            for col in currencies:
                if row == col:
                    matrix[row][col] = None
                else:
                    matrix[row][col] = round(strengths[row] - strengths[col], 1)
        return {"currencies": currencies, "matrix": matrix}

    # ═══════════════════════════════════════════════════════
    # 5. CURRENCY CYCLE DETECTION  ⭐
    # ═══════════════════════════════════════════════════════

    def detect_cycle(self, history: list[dict], lookback: int = 10) -> dict:
        """
        history: CurrencyStrengthEngine._load_history()-এর output —
                 [{'currency': 'USD', 'strength_score': 72, 'timestamp': ...}, ...]

        প্রতিটা currency-র সাম্প্রতিক N স্কোর দিয়ে simple linear slope বের
        করে "STRENGTHENING_CYCLE" / "WEAKENING_CYCLE" / "NEUTRAL_CYCLE" বলে।
        """
        if not history:
            return {}

        by_currency: dict[str, list[float]] = {}
        for entry in history:
            by_currency.setdefault(entry["currency"], []).append(entry["strength_score"])

        cycles = {}
        for cur, scores in by_currency.items():
            recent = scores[-lookback:]
            if len(recent) < 3:
                cycles[cur] = {"cycle": "INSUFFICIENT_DATA", "slope": 0.0}
                continue

            x = np.arange(len(recent))
            slope, _ = np.polyfit(x, recent, 1)

            if slope > 0.8:
                label = "STRENGTHENING_CYCLE"
            elif slope < -0.8:
                label = "WEAKENING_CYCLE"
            else:
                label = "NEUTRAL_CYCLE"

            cycles[cur] = {"cycle": label, "slope": round(float(slope), 2)}

        return cycles

    # ═══════════════════════════════════════════════════════
    # PRINT HELPERS
    # ═══════════════════════════════════════════════════════

    def print_heatmap(self, heatmap: dict) -> None:
        currencies = heatmap.get("currencies", [])
        matrix     = heatmap.get("matrix", {})
        if not currencies:
            log.info("  Heatmap unavailable — no currency data")
            return

        header = "        " + " ".join(f"{c:>6}" for c in currencies)
        log.info(header)
        for row in currencies:
            cells = []
            for col in currencies:
                val = matrix[row][col]
                cells.append("     -" if val is None else f"{val:+6.1f}")
            log.info(f"  {row:<4}  " + " ".join(cells))