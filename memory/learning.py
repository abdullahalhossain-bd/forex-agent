# memory/learning.py  —  Day 15 | AI Self-Learning Engine

"""
AI তার নিজের trade history দেখে শিখবে।

কী দেখবে:
- কোন pattern সবচেয়ে বেশি win করে?
- কোন regime-এ সবচেয়ে বেশি জেতে?
- কোন ভুল বারবার করছে?
- Confidence level কত হলে win বেশি?
"""

from memory.database import Database
import json
from collections import defaultdict


class LearningEngine:
    """
    AI-এর self-improvement engine।

    trader.py এটা দিয়ে past data থেকে
    insight নেবে এবং decision improve করবে।
    """

    def __init__(self):
        self.db = Database()

    # ── Pattern Performance ────────────────────────────────────

    def pattern_win_rate(self) -> dict:
        """
        কোন candlestick pattern সবচেয়ে বেশি win করে?

        Output:
        {
            "hammer":           {"wins": 8, "total": 10, "win_rate": 80.0},
            "bullish_engulfing": {"wins": 6, "total": 8,  "win_rate": 75.0},
        }
        """
        cursor = self.db.conn.cursor()
        cursor.execute("""
        SELECT a.pattern, t.result
        FROM analysis_log a
        JOIN trades t ON date(a.date) = date(t.date) AND a.pair = t.pair
        WHERE t.result IN ('WIN', 'LOSS') AND a.pattern != 'none'
        """)
        rows = cursor.fetchall()

        stats = defaultdict(lambda: {"wins": 0, "total": 0})
        for row in rows:
            pattern, result = row
            stats[pattern]["total"] += 1
            if result == "WIN":
                stats[pattern]["wins"] += 1

        result_dict = {}
        for pattern, data in stats.items():
            total = data["total"]
            wins  = data["wins"]
            result_dict[pattern] = {
                "wins":     wins,
                "total":    total,
                "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
            }

        return dict(sorted(result_dict.items(), key=lambda x: x[1]["win_rate"], reverse=True))

    # ── Regime Performance ─────────────────────────────────────

    def regime_win_rate(self) -> dict:
        """
        কোন market regime-এ AI সবচেয়ে ভালো perform করে?
        """
        cursor = self.db.conn.cursor()
        cursor.execute("""
        SELECT a.regime, t.result
        FROM analysis_log a
        JOIN trades t ON date(a.date) = date(t.date) AND a.pair = t.pair
        WHERE t.result IN ('WIN', 'LOSS')
        """)
        rows = cursor.fetchall()

        stats = defaultdict(lambda: {"wins": 0, "total": 0})
        for row in rows:
            regime, result = row
            if regime:
                stats[regime]["total"] += 1
                if result == "WIN":
                    stats[regime]["wins"] += 1

        result_dict = {}
        for regime, data in stats.items():
            total = data["total"]
            wins  = data["wins"]
            result_dict[regime] = {
                "wins":     wins,
                "total":    total,
                "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
            }

        return result_dict

    # ── Confidence Analysis ────────────────────────────────────

    def confidence_vs_result(self) -> dict:
        """
        Confidence level কত হলে win বেশি?

        Buckets: 50-60%, 60-70%, 70-80%, 80%+
        """
        cursor = self.db.conn.cursor()
        cursor.execute("""
        SELECT confidence, result FROM trades
        WHERE result IN ('WIN', 'LOSS') AND confidence IS NOT NULL
        """)
        rows = cursor.fetchall()

        buckets = {
            "50-60": {"wins": 0, "total": 0},
            "60-70": {"wins": 0, "total": 0},
            "70-80": {"wins": 0, "total": 0},
            "80+":   {"wins": 0, "total": 0},
        }

        for conf, result in rows:
            if conf < 60:
                key = "50-60"
            elif conf < 70:
                key = "60-70"
            elif conf < 80:
                key = "70-80"
            else:
                key = "80+"

            buckets[key]["total"] += 1
            if result == "WIN":
                buckets[key]["wins"] += 1

        for key, data in buckets.items():
            total = data["total"]
            wins  = data["wins"]
            data["win_rate"] = round(wins / total * 100, 1) if total > 0 else 0

        return buckets

    # ── Common Mistakes ────────────────────────────────────────

    def common_mistakes(self) -> list:
        """
        AI কোন ভুল সবচেয়ে বেশি করে?
        """
        cursor = self.db.conn.cursor()
        cursor.execute("""
        SELECT error_type, COUNT(*) as count, GROUP_CONCAT(lesson, ' | ') as lessons
        FROM mistakes
        GROUP BY error_type
        ORDER BY count DESC
        """)
        rows = cursor.fetchall()
        return [
            {
                "error_type": row[0],
                "count":      row[1],
                "lesson":     row[2].split(" | ")[0] if row[2] else "",
            }
            for row in rows
        ]

    # ── AI Improvement Suggestions ─────────────────────────────

    def get_improvement_plan(self) -> dict:
        """
        সব analysis একসাথে করে AI-কে improvement plan দাও।

        এই dict LLM prompt-এ যোগ করা যাবে।
        """
        patterns = self.pattern_win_rate()
        regimes  = self.regime_win_rate()
        conf     = self.confidence_vs_result()
        mistakes = self.common_mistakes()

        # Best pattern
        best_pattern = max(patterns.items(), key=lambda x: x[1]["win_rate"])[0] \
            if patterns else "Unknown"

        # Best regime
        best_regime = max(regimes.items(), key=lambda x: x[1]["win_rate"])[0] \
            if regimes else "Unknown"

        # Optimal confidence
        best_conf_bucket = max(conf.items(), key=lambda x: x[1]["win_rate"])[0] \
            if conf else "70-80"

        # Top mistake
        top_mistake = mistakes[0]["error_type"] if mistakes else "None yet"

        plan = {
            "best_pattern":       best_pattern,
            "best_regime":        best_regime,
            "optimal_confidence": best_conf_bucket,
            "top_mistake":        top_mistake,
            "pattern_stats":      patterns,
            "regime_stats":       regimes,
            "confidence_stats":   conf,
            "mistakes":           mistakes,
            "recommendations": [
                f"Prefer {best_pattern} patterns — highest win rate",
                f"Best performance in {best_regime} market",
                f"Enter only when confidence is {best_conf_bucket}%",
                f"Avoid: {top_mistake} — most common error",
            ],
        }

        return plan

    def print_report(self):
        """Console-এ learning report দেখাও।"""
        plan = self.get_improvement_plan()
        bar = "═" * 52

        print(f"\n{bar}")
        print(f"  🧠  AI SELF-LEARNING REPORT")
        print(bar)
        print(f"  Best Pattern     : {plan['best_pattern']}")
        print(f"  Best Regime      : {plan['best_regime']}")
        print(f"  Optimal Confidence: {plan['optimal_confidence']}%")
        print(f"  Top Mistake      : {plan['top_mistake']}")
        print(f"\n  ── Recommendations ──")
        for rec in plan["recommendations"]:
            print(f"  → {rec}")

        if plan["pattern_stats"]:
            print(f"\n  ── Pattern Win Rates ──")
            for p, s in plan["pattern_stats"].items():
                print(f"  {p:25s}: {s['win_rate']}% ({s['wins']}/{s['total']})")

        if plan["mistakes"]:
            print(f"\n  ── Common Mistakes ──")
            for m in plan["mistakes"][:3]:
                print(f"  [{m['count']}x] {m['error_type']}: {m['lesson']}")
        print(bar)

    def close(self):
        self.db.close()
