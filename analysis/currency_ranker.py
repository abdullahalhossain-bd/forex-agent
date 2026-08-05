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

    def find_best_pairs(
        self, strengths: dict, min_diff: int = 40, max_pairs: int = 10,
        pair_details: dict | None = None,
    ) -> list[dict]:
        """
        প্রতিটা currency-pair combination-এর strength difference হিসাব করে,
        min_diff-এর নিচে যেগুলো সেগুলো বাদ দেয় (Avoid Bad Trades — doc #11)।

        FIX (audit H4): আগে ranking শুধু strength_difference দিয়ে হতো —
        একটা 90 vs 20 diff দেখতে দুর্দান্ত লাগে, কিন্তু সেই move যদি ATR
        spike/noise-চালিত হয়, তাহলে আসলে risk বেশি, reliability কম। এখন
        `pair_details` (CurrencyStrengthEngine.calculate_strength()-এর
        output, প্রতিটা cross-pair-এর volatility_adj component সহ) দেওয়া
        হলে সেটা quality grading-এ ছাড় দেয়/downgrade করে।

        pair_details: {"USDJPY": {"volatility_adj": 32.1, ...}, ...} —
        না দিলে আগের মতোই diff-only grading হয় (backward compatible)।

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

                vol_score, vol_flag = self._pair_volatility(pair_details, strong, weak)

                opportunities.append({
                    "pair":                f"{strong}{weak}",
                    "direction":           "BUY",   # strong/weak → strong-side কেনা হবে
                    "strong_currency":     strong,
                    "weak_currency":       weak,
                    "strong_strength":     strong_score,
                    "weak_strength":       weak_score,
                    "strength_difference": diff,
                    "volatility_score":    vol_score,   # None যদি pair_details না থাকে
                    "volatility_flag":     vol_flag,    # "ELEVATED" | "NORMAL" | None
                    "trade_quality":       self._grade_difference(diff, vol_flag),
                })

        opportunities.sort(key=lambda o: o["strength_difference"], reverse=True)
        return opportunities[:max_pairs]

    def _pair_volatility(self, pair_details: dict | None, strong: str, weak: str) -> tuple:
        """
        strong/weak currency-র underlying cross pair(s)-এর
        volatility_adj magnitude থেকে একটা risk indicator বের করে।
        pair_details না থাকলে (None, None) — grading-এ কোনো প্রভাব পড়ে না।
        """
        if not pair_details:
            return None, None

        candidates = (f"{strong}{weak}", f"{weak}{strong}")
        for sym in candidates:
            detail = pair_details.get(sym)
            if detail is not None:
                vol = abs(detail.get("volatility_adj", 0.0))
                flag = "ELEVATED" if vol >= 25.0 else "NORMAL"
                return round(vol, 1), flag

        return None, None

    def _grade_difference(self, diff: float, vol_flag: str | None = None) -> str:
        if diff >= self.VERY_HIGH_DIFF:
            grade = "VERY_HIGH"
        elif diff >= self.HIGH_DIFF:
            grade = "HIGH"
        elif diff >= self.MEDIUM_DIFF:
            grade = "MEDIUM"
        else:
            grade = "LOW"

        # FIX (audit H4): an ELEVATED-volatility pair gets knocked down one
        # tier — a big strength gap during an ATR spike is more likely to
        # be noise than a clean directional move.
        if vol_flag == "ELEVATED":
            downgrade = {"VERY_HIGH": "HIGH", "HIGH": "MEDIUM", "MEDIUM": "LOW", "LOW": "LOW"}
            grade = downgrade[grade]

        return grade

    # ═══════════════════════════════════════════════════════
    # 3. CORRELATION PROTECTION  ⭐
    # ═══════════════════════════════════════════════════════

    def detect_correlation_risk(self, opportunities: list[dict]) -> list[dict]:
        """
        একাধিক opportunity একই currency-র উপর ভিত্তি করে তৈরি হলে
        (যেমন GBPJPY BUY আর AUDJPY BUY — দুটোই JPY weakness trade)
        সেগুলোকে correlation group-এ চিহ্নিত করো এবং risk note যোগ করো।

        FIX (audit H5): শুধু "shared currency আছে" বলাটা যথেষ্ট না — সেই
        currency দুই opportunity-তে একই role-এ (দুটোতেই strong, বা দুটোতেই
        weak) থাকলে দুটো trade একই দিকে move করে (REINFORCING — exposure
        সত্যিই যোগ হয়)। বিপরীত role-এ থাকলে (একটায় strong, আরেকটায় weak)
        সেই currency-র exposure আংশিক offset করে (OFFSETTING — কম risky)।
        এটা এখনও rolling correlation/covariance না (তার জন্য price
        history লাগবে, এই দুই ফাইলের scope-এ নেই) — কিন্তু বিনামূল্যে
        পাওয়া যাচ্ছে এমন directional তথ্য দিয়ে অন্তত ভুল দিকে
        conservative (REINFORCING-কে worst-case ধরে) হওয়া যাচ্ছে।
        """
        if not opportunities:
            return opportunities

        # currency -> কোন কোন opportunity index এই currency ব্যবহার করছে,
        # আর সেই currency-টা সেই opportunity-তে কোন role-এ (strong/weak)
        usage: dict[str, list[tuple[int, str]]] = {}
        for idx, opp in enumerate(opportunities):
            usage.setdefault(opp["strong_currency"], []).append((idx, "strong"))
            usage.setdefault(opp["weak_currency"], []).append((idx, "weak"))

        enriched = [dict(o) for o in opportunities]

        for currency, entries in usage.items():
            if len(entries) < 2:
                continue
            for idx, role in entries:
                other_roles = [r for i, r in entries if i != idx]
                reinforcing = any(r == role for r in other_roles)
                enriched[idx].setdefault("correlation_groups", []).append({
                    "shared_currency": currency,
                    "group_size":      len(entries),
                    "risk_direction":  "REINFORCING" if reinforcing else "OFFSETTING",
                })

        for opp in enriched:
            groups = opp.get("correlation_groups", [])
            if groups:
                shared = ", ".join(
                    f"{g['shared_currency']} ({g['risk_direction'].lower()})" for g in groups
                )
                any_reinforcing = any(g["risk_direction"] == "REINFORCING" for g in groups)
                opp["correlated"]            = True
                opp["correlation_note"]      = (
                    f"Shares exposure with other active opportunities via {shared} — "
                    + ("reduce combined position size to manage correlated risk"
                       if any_reinforcing else
                       "exposure partially offsets across opportunities, but monitor combined size")
                )
                # Only reinforcing overlaps genuinely stack risk; size divisor
                # should reflect that rather than penalizing offsetting overlaps
                # as if they were additive.
                reinforcing_sizes = [g["group_size"] for g in groups if g["risk_direction"] == "REINFORCING"]
                opp["suggested_size_divisor"] = max(reinforcing_sizes) if reinforcing_sizes else 1
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