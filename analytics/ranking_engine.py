# analytics/ranking_engine.py  —  Day 54 | Strategy Ranking Engine
# ============================================================
# সব setup কে multi-factor score দিয়ে rank করে।
#
# Score Formula:
#   win_rate      × 0.35
#   profit_factor × 0.25
#   avg_rr        × 0.20
#   sample_size   × 0.10   (বেশি trade = বেশি reliable)
#   consistency   × 0.10   (walk-forward স্থিতিশীলতা)
#
# Output:
#   "EURUSD H1 London BOS+OB" → Score 91/100 → TRADE
#   "GBPUSD M5 Asian Weak"    → Score 35/100 → REJECT
# ============================================================

from __future__ import annotations
import math
from typing import Optional
from utils.logger import get_logger

log = get_logger("ranking_engine")

# Minimum trade count — এর কম হলে score reliable নয়
MIN_RELIABLE_TRADES = 20
# Score threshold — এর নিচে হলে trade নেওয়া উচিত নয়
SCORE_THRESHOLD_TRADE   = 60
SCORE_THRESHOLD_CAUTION = 45


class SetupScore:
    """একটা setup-এর সম্পূর্ণ score breakdown।"""

    def __init__(
        self,
        pair:         str,
        timeframe:    str,
        session:      str,
        pattern:      str,
        regime:       str,
        win_rate:     float,      # 0-100
        avg_rr:       float,
        total_pips:   float,
        trades:       int,
        profit_factor: float = 1.0,
        wf_last7_wr:  float = None,   # walk-forward 7-day win rate
        wf_last30_wr: float = None,   # walk-forward 30-day win rate
    ):
        self.pair          = pair
        self.timeframe     = timeframe
        self.session       = session
        self.pattern       = pattern
        self.regime        = regime
        self.win_rate      = win_rate
        self.avg_rr        = avg_rr
        self.total_pips    = total_pips
        self.trades        = trades
        self.profit_factor = profit_factor
        self.wf_last7_wr   = wf_last7_wr
        self.wf_last30_wr  = wf_last30_wr

        # Score calculate করো
        self.score, self.breakdown = self._calculate()
        self.recommendation = self._recommend()

    def _calculate(self) -> tuple[float, dict]:
        """
        Multi-factor score 0-100।
        প্রতিটি factor 0-100 normalize হয়, তারপর weighted sum।
        """

        # ── 1. Win Rate Score (0-100) ──────────────────────
        # 35% = 0 points, 65% = 70 points, 80%+ = 100 points
        wr_score = max(0, min(100, (self.win_rate - 35) / 45 * 100))

        # ── 2. Profit Factor Score (0-100) ─────────────────
        # PF 1.0 = 0, PF 1.5 = 50, PF 2.0+ = 100
        pf = min(self.profit_factor, 3.0)   # cap করো
        pf_score = max(0, min(100, (pf - 1.0) / 2.0 * 100))

        # ── 3. Avg R:R Score (0-100) ───────────────────────
        # RR 1.0 = 20, RR 1.5 = 50, RR 2.5+ = 100
        rr_score = max(0, min(100, (self.avg_rr - 1.0) / 2.0 * 100))

        # ── 4. Sample Size Score (0-100) ───────────────────
        # 5 trades = 25, 20 = 60, 50+ = 100
        sample_score = min(100, math.log10(max(1, self.trades)) / math.log10(50) * 100)

        # ── 5. Walk-Forward Consistency (0-100) ────────────
        # Lifetime vs recent 30d, 7d কতটা consistent
        consistency = 100.0
        if self.wf_last30_wr is not None:
            drop_30 = self.win_rate - self.wf_last30_wr
            consistency -= max(0, drop_30 * 1.5)   # 10% drop → -15 points
        if self.wf_last7_wr is not None:
            drop_7 = self.win_rate - self.wf_last7_wr
            consistency -= max(0, drop_7 * 0.8)
        consistency = max(0, min(100, consistency))

        # ── Weighted Sum ───────────────────────────────────
        weights = {
            "win_rate":    0.35,
            "pf":          0.25,
            "rr":          0.20,
            "sample":      0.10,
            "consistency": 0.10,
        }

        final_score = (
            wr_score     * weights["win_rate"] +
            pf_score     * weights["pf"]       +
            rr_score     * weights["rr"]       +
            sample_score * weights["sample"]   +
            consistency  * weights["consistency"]
        )

        breakdown = {
            "win_rate_score":    round(wr_score, 1),
            "profit_factor_score": round(pf_score, 1),
            "rr_score":          round(rr_score, 1),
            "sample_size_score": round(sample_score, 1),
            "consistency_score": round(consistency, 1),
            "weights":           weights,
        }

        return round(final_score, 1), breakdown

    def _recommend(self) -> str:
        if self.trades < 5:
            return "INSUFFICIENT_DATA"
        if self.score >= SCORE_THRESHOLD_TRADE:
            return "TRADE"
        elif self.score >= SCORE_THRESHOLD_CAUTION:
            return "CAUTION"
        else:
            return "AVOID"

    def to_dict(self) -> dict:
        return {
            "pair":           self.pair,
            "timeframe":      self.timeframe,
            "session":        self.session,
            "pattern":        self.pattern,
            "regime":         self.regime,
            "score":          self.score,
            "recommendation": self.recommendation,
            "win_rate":       self.win_rate,
            "avg_rr":         self.avg_rr,
            "profit_factor":  self.profit_factor,
            "trades":         self.trades,
            "total_pips":     self.total_pips,
            "breakdown":      self.breakdown,
        }


# ════════════════════════════════════════════════════════════
# RANKING ENGINE
# ════════════════════════════════════════════════════════════

class RankingEngine:
    """
    Day 54 — সব setup কে score দিয়ে rank করে।

    DecisionAgent trade নেওয়ার আগে current setup-এর score check করবে।
    Score < 45  →  "AVOID"  →  NO TRADE।
    Score 45-59 →  "CAUTION" →  confidence কমিয়ে trade।
    Score 60+   →  "TRADE"  →  স্বাভাবিকভাবে proceed।
    """

    def __init__(self, tracker=None):
        """
        tracker: StrategyTracker instance।
        None হলে score শুধু provided data দিয়ে calculate হবে।
        """
        self.tracker = tracker

    # ─────────────────────────────────────────────
    # SCORE CURRENT SETUP
    # ─────────────────────────────────────────────

    def score_setup(
        self,
        pair:      str,
        timeframe: str,
        session:   str,
        pattern:   str,
        regime:    str,
        historical_win_rate: float = 50.0,
        historical_rr:       float = 1.5,
        historical_trades:   int   = 0,
        historical_pf:       float = 1.0,
        historical_pips:     float = 0.0,
    ) -> SetupScore:
        """
        Current trade setup-এর score calculate করো।
        Tracker থাকলে DB থেকে data নাও, না হলে provided data ব্যবহার করো।
        """
        wf7  = None
        wf30 = None

        if self.tracker:
            # DB থেকে exact setup data নাও
            db_data = self._fetch_from_tracker(pair, timeframe, session, pattern, regime)
            if db_data:
                historical_win_rate = db_data["win_rate"]
                historical_rr       = db_data["avg_rr"]
                historical_trades   = db_data["trades"]
                historical_pf       = db_data.get("profit_factor", 1.0)
                historical_pips     = db_data.get("total_pips", 0.0)
                wf7                 = db_data.get("wf7_win_rate")
                wf30                = db_data.get("wf30_win_rate")

        return SetupScore(
            pair           = pair,
            timeframe      = timeframe,
            session        = session,
            pattern        = pattern,
            regime         = regime,
            win_rate       = historical_win_rate,
            avg_rr         = historical_rr,
            total_pips     = historical_pips,
            trades         = historical_trades,
            profit_factor  = historical_pf,
            wf_last7_wr    = wf7,
            wf_last30_wr   = wf30,
        )

    def _fetch_from_tracker(
        self,
        pair: str, timeframe: str, session: str,
        pattern: str, regime: str,
    ) -> Optional[dict]:
        """Tracker DB থেকে setup-specific stats নাও।"""
        try:
            import sqlite3
            with sqlite3.connect(self.tracker.db_path) as conn:
                row = conn.execute("""
                    SELECT COUNT(*) as trades,
                           SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) as wins,
                           AVG(rr_actual) as avg_rr,
                           SUM(profit_pips) as total_pips
                    FROM trades
                    WHERE pair=? AND timeframe=? AND session=? AND pattern=? AND regime=?
                      AND result IS NOT NULL
                """, (pair, timeframe, session, pattern, regime)).fetchone()

                if not row or row[0] == 0:
                    return None

                trades = row[0]
                wins   = row[1] or 0
                win_rate = round(wins / trades * 100, 1) if trades else 0

                # Walk-forward আলাদা করে
                from datetime import datetime, timezone, timedelta
                now = datetime.now(timezone.utc)

                def _period_wr(days: int) -> Optional[float]:
                    since = (now - timedelta(days=days)).isoformat()
                    r2 = conn.execute("""
                        SELECT COUNT(*) as t,
                               SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) as w
                        FROM trades
                        WHERE pair=? AND timeframe=? AND session=? AND pattern=? AND regime=?
                          AND result IS NOT NULL AND timestamp >= ?
                    """, (pair, timeframe, session, pattern, regime, since)).fetchone()
                    if r2 and r2[0] >= 3:
                        return round((r2[1] or 0) / r2[0] * 100, 1)
                    return None

                # Profit factor
                pf_row = conn.execute("""
                    SELECT SUM(CASE WHEN profit_pips > 0 THEN profit_pips ELSE 0 END) as gross_win,
                           ABS(SUM(CASE WHEN profit_pips < 0 THEN profit_pips ELSE 0 END)) as gross_loss
                    FROM trades
                    WHERE pair=? AND timeframe=? AND session=? AND pattern=? AND regime=?
                      AND result IS NOT NULL
                """, (pair, timeframe, session, pattern, regime)).fetchone()

                pf = 1.0
                if pf_row and pf_row[1] and pf_row[1] > 0:
                    pf = round((pf_row[0] or 0) / pf_row[1], 2)

                return {
                    "trades":        trades,
                    "wins":          wins,
                    "win_rate":      win_rate,
                    "avg_rr":        round(row[2] or 0, 2),
                    "total_pips":    round(row[3] or 0, 1),
                    "profit_factor": pf,
                    "wf7_win_rate":  _period_wr(7),
                    "wf30_win_rate": _period_wr(30),
                }
        except Exception as e:
            log.warning(f"[RankingEngine] DB fetch error: {e}")
            return None

    # ─────────────────────────────────────────────
    # RANK ALL SETUPS
    # ─────────────────────────────────────────────

    def rank_all_setups(self, min_trades: int = 5) -> list[dict]:
        """
        DB-তে থাকা সব setup rank করো।
        RankingEngine হলো StrategyTracker-এর viewer।
        """
        if not self.tracker:
            log.warning("[RankingEngine] No tracker provided — cannot rank all setups")
            return []

        try:
            import sqlite3
            with sqlite3.connect(self.tracker.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute("""
                    SELECT pair, timeframe, session, pattern, regime,
                           COUNT(*) as trades,
                           SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) as wins,
                           AVG(rr_actual) as avg_rr,
                           SUM(profit_pips) as total_pips,
                           SUM(CASE WHEN profit_pips > 0 THEN profit_pips ELSE 0 END) as gross_win,
                           ABS(SUM(CASE WHEN profit_pips < 0 THEN profit_pips ELSE 0 END)) as gross_loss
                    FROM trades
                    WHERE result IS NOT NULL
                    GROUP BY pair, timeframe, session, pattern, regime
                    HAVING COUNT(*) >= ?
                """, (min_trades,)).fetchall()
        except Exception as e:
            log.error(f"[RankingEngine] rank_all error: {e}")
            return []

        scored = []
        for r in rows:
            trades = r["trades"]
            wins   = r["wins"] or 0
            pf     = 1.0
            if r["gross_loss"] and r["gross_loss"] > 0:
                pf = round((r["gross_win"] or 0) / r["gross_loss"], 2)

            setup_score = SetupScore(
                pair          = r["pair"],
                timeframe     = r["timeframe"],
                session       = r["session"],
                pattern       = r["pattern"],
                regime        = r["regime"],
                win_rate      = round(wins / trades * 100, 1) if trades else 0,
                avg_rr        = round(r["avg_rr"] or 0, 2),
                total_pips    = round(r["total_pips"] or 0, 1),
                trades        = trades,
                profit_factor = pf,
            )
            scored.append(setup_score.to_dict())

        # Score অনুযায়ী sort করো (descending)
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    # ─────────────────────────────────────────────
    # CONFIDENCE ADJUSTMENT
    # ─────────────────────────────────────────────

    def get_confidence_adjustment(self, score: float) -> int:
        """
        Setup score থেকে confidence adjustment বের করো।
        DecisionAgent এটা use করবে।

        Score 80+  → +10% confidence boost
        Score 60-79 → 0% (neutral)
        Score 45-59 → -10% (caution)
        Score <45   → trade block (AVOID)
        """
        if score >= 80:
            return +10
        elif score >= 60:
            return 0
        elif score >= 45:
            return -10
        else:
            return -99   # sentinel: block trade

    # ─────────────────────────────────────────────
    # PRINT RANKINGS
    # ─────────────────────────────────────────────

    def print_rankings(self, setups: list[dict], top_n: int = 10) -> None:
        bar = "═" * 72
        print(f"\n{bar}")
        print("  🏆  STRATEGY RANKINGS  (Day 54)")
        print(bar)
        print(f"  {'#':3s} {'Pair':8s} {'TF':5s} {'Session':22s} {'Pattern':20s} "
              f"{'Score':6s} {'WR':6s} {'RR':5s} {'Rec':7s}")
        print(f"  {'─'*3} {'─'*8} {'─'*5} {'─'*22} {'─'*20} {'─'*6} {'─'*6} {'─'*5} {'─'*7}")

        icons = {"TRADE": "✅", "CAUTION": "⚠️ ", "AVOID": "❌", "INSUFFICIENT_DATA": "❓"}

        for i, s in enumerate(setups[:top_n], 1):
            icon = icons.get(s["recommendation"], "❓")
            print(
                f"  {i:3d} {s['pair']:8s} {s['timeframe']:5s} "
                f"{s['session']:22s} {s['pattern'][:20]:20s} "
                f"{s['score']:6.1f} {s['win_rate']:5.1f}% "
                f"{s['avg_rr']:5.2f} {icon} {s['recommendation']}"
            )

        if len(setups) > top_n:
            print(f"\n  ... and {len(setups) - top_n} more setups")
        print(f"{bar}\n")