# scanner/opportunity_ranker.py  —  Day 36 Part 3 | Opportunity Ranking Engine
# ============================================================
# সব scanner result-এ একটা composite score দেয় এবং rank করে।
#
# Final Score = Σ (component_score × weight)
#   technical_strength : TA signal কতটা clear
#   mtf_alignment      : multi-timeframe কতটা একমত
#   rr_ratio           : Risk/Reward কতটা ভালো
#   news_safety        : news window থেকে কতটা দূরে
#   liquidity          : spread কতটা tight
# ============================================================

from utils.logger import get_logger
from scanner.config import RANK_WEIGHTS, MIN_OPPORTUNITY_SCORE, TOP_N

log = get_logger("opportunity_ranker")


class OpportunityRanker:
    """
    Usage:
        ranker = OpportunityRanker()
        ranked = ranker.rank(scan_results)
        top    = ranker.top_n(ranked)
    """

    def rank(self, scan_results: list[dict]) -> list[dict]:
        """
        scan_results = list of dicts from MarketScanner.scan()
        প্রতিটাতে score যোগ করে sort করে returns।
        """
        scored = []
        for r in scan_results:
            if r.get("signal") == "NO TRADE":
                continue
            if r.get("correlation_blocked"):
                continue

            score = self._compute_score(r)
            r = dict(r)
            r["opportunity_score"] = score
            r["rank_breakdown"] = self._breakdown(r)

            if score >= MIN_OPPORTUNITY_SCORE:
                scored.append(r)

        scored.sort(key=lambda x: x["opportunity_score"], reverse=True)

        for i, opp in enumerate(scored, 1):
            opp["rank"] = i

        return scored

    def top_n(self, ranked: list[dict], n: int = None) -> list[dict]:
        return ranked[: n or TOP_N]

    # ─────────────────────────────────────────────
    # SCORE COMPONENTS
    # ─────────────────────────────────────────────

    def _compute_score(self, r: dict) -> int:
        w = RANK_WEIGHTS
        score = (
            self._technical_strength(r)  * w["technical_strength"]
            + self._mtf_alignment(r)     * w["mtf_alignment"]
            + self._rr_score(r)          * w["rr_ratio"]
            + self._news_safety(r)       * w["news_safety"]
            + self._liquidity_score(r)   * w["liquidity"]
        )
        return round(score)

    def _technical_strength(self, r: dict) -> float:
        """AI confidence + trend clarity → 0-100."""
        confidence = r.get("confidence", 50)
        trend = r.get("trend", "RANGE")
        trend_bonus = 15 if trend in ("BULLISH", "BEARISH") else 0
        return min(100, confidence + trend_bonus)

    def _mtf_alignment(self, r: dict) -> float:
        """Multi-timeframe alignment score — 0, 50, or 100."""
        mtf = r.get("mtf_alignment", "UNKNOWN")
        return {"STRONG": 100, "MODERATE": 60, "WEAK": 20, "CONFLICT": 0}.get(mtf, 50)

    def _rr_score(self, r: dict) -> float:
        """RR ratio → score. 1:2 = 70, 1:3 = 90, 1:1 = 40."""
        rr = r.get("rr_ratio", 1.0)
        if rr >= 3.0:   return 100
        if rr >= 2.5:   return 90
        if rr >= 2.0:   return 75
        if rr >= 1.5:   return 55
        return 30

    def _news_safety(self, r: dict) -> float:
        """News window থেকে দূরে = 100, news window active = 0."""
        if r.get("news_blocked"):
            return 0
        mins = r.get("mins_to_news", 999)
        if mins < 15:    return 20
        if mins < 30:    return 60
        return 100

    def _liquidity_score(self, r: dict) -> float:
        """Spread কম হলে বেশি score।"""
        spread = r.get("spread_pips", 2.0)
        if spread <= 0.8:  return 100
        if spread <= 1.5:  return 85
        if spread <= 2.5:  return 65
        if spread <= 4.0:  return 40
        return 15

    def _breakdown(self, r: dict) -> dict:
        return {
            "technical": round(self._technical_strength(r)),
            "mtf":       round(self._mtf_alignment(r)),
            "rr":        round(self._rr_score(r)),
            "news":      round(self._news_safety(r)),
            "liquidity": round(self._liquidity_score(r)),
        }

    def print_top(self, ranked: list[dict], n: int = None) -> None:
        top = self.top_n(ranked, n)
        bar = "═" * 50
        log.info(bar)
        log.info("  🏆  TOP OPPORTUNITIES")
        log.info(bar)
        for opp in top:
            icon = "🟢" if opp["signal"] == "BUY" else "🔴"
            log.info(
                f"  {opp['rank']}️⃣  {icon} {opp['symbol']:<8} "
                f"{opp['signal']:<5} "
                f"Score: {opp['opportunity_score']}/100  "
                f"Conf: {opp.get('confidence', '?')}%  "
                f"RR: 1:{opp.get('rr_ratio', '?')}"
            )
            bd = opp.get("rank_breakdown", {})
            log.info(
                f"       Tech:{bd.get('technical')} MTF:{bd.get('mtf')} "
                f"RR:{bd.get('rr')} News:{bd.get('news')} Liq:{bd.get('liquidity')}"
            )
        log.info(bar)