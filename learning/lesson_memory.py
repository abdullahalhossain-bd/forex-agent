# learning/lesson_memory.py  —  Day 52 | Lesson Memory System ⭐⭐⭐⭐⭐
# ============================================================
# AI-এর দীর্ঘমেয়াদী শিক্ষার ভাণ্ডার।
#
# Store করে:
#   - lesson_id, pattern, market_condition, mistake, new_rule
#   - success_rate_after_change (rule কাজ করছে কিনা track)
#
# Future use:
#   নতুন setup দেখলে AI জিজ্ঞেস করবে:
#   "এই পরিস্থিতিতে আগে কী হয়েছিল?"
#   Memory থেকে: "গত 10 বার ranging + engulfing = 7 loss → Avoid"
# ============================================================

import json
import os
from datetime import datetime, timezone

from utils.logger import get_logger
from core.constants import MEMORY_DIR

log = get_logger("learning.lesson_memory")

LESSON_DB_PATH   = str(MEMORY_DIR / "lesson_memory.json")
PATTERN_STATS_PATH = str(MEMORY_DIR / "pattern_stats.json")


class LessonMemory:
    """
    AI-এর experience store।

    Usage:
        memory = LessonMemory()

        # Lesson save:
        memory.add_lesson(
            pattern="Bullish Engulfing",
            market_condition="RANGING",
            mistake="Pattern failed in ranging market — price chopped",
            new_rule="Avoid engulfing reversal when regime=RANGING",
        )

        # Query before trading:
        history = memory.recall(pattern="Bullish Engulfing", condition="RANGING")
        print(history["summary"])  # "7/10 losses in RANGING — AVOID"

        # Track if new rule is working:
        memory.update_success_rate(lesson_id=5, outcome="WIN")
    """

    def __init__(self):
        os.makedirs("memory", exist_ok=True)

    # ──────────────────────────────────────────────────────────
    # ADD LESSON
    # ──────────────────────────────────────────────────────────

    def add_lesson(
        self,
        pattern: str,
        market_condition: str,
        mistake: str,
        new_rule: str,
        pair: str = None,
        timeframe: str = None,
        pnl: float = None,
        confidence_at_entry: int = None,
        source: str = "deep_analyzer",
    ) -> dict:
        """Lesson memory-তে নতুন entry যোগ করো।"""
        lessons = self._load()

        lesson = {
            "lesson_id":           len(lessons) + 1,
            "timestamp":           datetime.now(timezone.utc).isoformat(),
            "pattern":             pattern,
            "market_condition":    market_condition,
            "mistake":             mistake,
            "new_rule":            new_rule,
            "pair":                pair,
            "timeframe":           timeframe,
            "pnl":                 pnl,
            "confidence_at_entry": confidence_at_entry,
            "source":              source,
            # Track করো rule কাজ করছে কিনা
            "trades_after_rule":   0,
            "wins_after_rule":     0,
            "success_rate_after_change": None,
        }

        lessons.append(lesson)
        self._save(lessons)

        log.info(f"[LessonMemory] Lesson #{lesson['lesson_id']} saved: {pattern} in {market_condition}")
        return lesson

    # ──────────────────────────────────────────────────────────
    # RECALL — query করো
    # ──────────────────────────────────────────────────────────

    def recall(
        self,
        pattern: str = None,
        condition: str = None,
        pair: str = None,
        limit: int = 10,
    ) -> dict:
        """
        নতুন trade setup-এর আগে memory থেকে জিজ্ঞেস করো।

        Returns:
            {
                "has_memory": bool,
                "total": int,
                "lessons": list,
                "summary": "গত 10 বার ranging + engulfing = 7 loss → Avoid",
                "recommendation": "AVOID | CAUTION | PROCEED",
                "best_rule": "strictest relevant rule"
            }
        """
        lessons = self._load()

        filtered = [
            l for l in lessons
            if (pattern is None or l.get("pattern") == pattern)
            and (condition is None or l.get("market_condition") == condition)
            and (pair is None or l.get("pair") == pair)
        ]

        if not filtered:
            return {
                "has_memory":     False,
                "total":          0,
                "lessons":        [],
                "summary":        f"No past experience with {pattern} in {condition}",
                "recommendation": "PROCEED_WITH_CAUTION",
                "best_rule":      None,
            }

        # Win/loss rate
        # FIX: handle pnl=None gracefully. l.get("pnl", -1) returns None
        # if the key exists but value is None — None < 0 crashes in Python 3.
        losses = [l for l in filtered if (l.get("pnl") or 0) < 0]
        loss_rate = round(len(losses) / len(filtered) * 100, 1)

        # Recommendation
        if loss_rate >= 70:
            recommendation = "AVOID"
        elif loss_rate >= 50:
            recommendation = "CAUTION"
        else:
            recommendation = "PROCEED"

        # Best (most recent) rule
        rules = [l.get("new_rule", "") for l in sorted(filtered, key=lambda x: x.get("timestamp", ""), reverse=True)]
        best_rule = rules[0] if rules else None

        cond_str = condition or "all conditions"
        pat_str  = pattern or "this pattern"

        summary = (
            f"গত {len(filtered)} বার {cond_str} + {pat_str} = "
            f"{len(losses)} loss ({loss_rate}%) — "
            f"{'⛔ AVOID' if recommendation == 'AVOID' else '⚠️ CAUTION' if recommendation == 'CAUTION' else '✅ PROCEED'}"
        )

        return {
            "has_memory":     True,
            "total":          len(filtered),
            "losses":         len(losses),
            "loss_rate_pct":  loss_rate,
            "lessons":        filtered[-limit:],
            "summary":        summary,
            "recommendation": recommendation,
            "best_rule":      best_rule,
        }

    # ──────────────────────────────────────────────────────────
    # TRACK RULE EFFECTIVENESS
    # ──────────────────────────────────────────────────────────

    def update_success_rate(self, lesson_id: int, outcome: str) -> None:
        """
        Rule apply করার পরের trades track করো।
        outcome: "WIN" | "LOSS"
        """
        lessons = self._load()
        for l in lessons:
            if l.get("lesson_id") == lesson_id:
                l["trades_after_rule"] = l.get("trades_after_rule", 0) + 1
                if outcome == "WIN":
                    l["wins_after_rule"] = l.get("wins_after_rule", 0) + 1

                total = l["trades_after_rule"]
                wins  = l.get("wins_after_rule", 0)
                l["success_rate_after_change"] = round(wins / total * 100, 1) if total > 0 else None

                log.info(
                    f"[LessonMemory] Lesson #{lesson_id} updated | "
                    f"after-rule win rate: {l['success_rate_after_change']}%"
                )
                break

        self._save(lessons)

    # ──────────────────────────────────────────────────────────
    # PATTERN PERFORMANCE STATS
    # ──────────────────────────────────────────────────────────

    def get_pattern_stats(self) -> dict:
        """
        Pattern-wise performance breakdown।

        Example:
            Hammer:       trades=100  win_rate=62%
            Engulfing:    trades=80   win_rate=48%
        """
        lessons = self._load()
        stats = {}

        for l in lessons:
            p = l.get("pattern", "unknown")
            if p not in stats:
                stats[p] = {"trades": 0, "losses": 0, "conditions": set()}

            stats[p]["trades"] += 1
            if (l.get("pnl") or 0) < 0:
                stats[p]["losses"] += 1

            cond = l.get("market_condition", "")
            if cond:
                stats[p]["conditions"].add(cond)

        # Calculate rates
        result = {}
        for p, s in stats.items():
            win_rate = round((1 - s["losses"] / s["trades"]) * 100, 1) if s["trades"] > 0 else 0
            result[p] = {
                "trades":          s["trades"],
                "losses":          s["losses"],
                "win_rate":        win_rate,
                "worst_conditions": list(s["conditions"]),
            }

        return result

    def get_regime_stats(self) -> dict:
        """
        Market condition-wise performance।

        Example:
            TRENDING: profit_factor=2.1
            RANGING:  profit_factor=0.8
        """
        lessons = self._load()
        regimes = {}

        for l in lessons:
            r = l.get("market_condition", "UNKNOWN")
            if r not in regimes:
                regimes[r] = {"trades": 0, "losses": 0}
            regimes[r]["trades"] += 1
            if (l.get("pnl") or 0) < 0:
                regimes[r]["losses"] += 1

        result = {}
        for r, s in regimes.items():
            win_rate = round((1 - s["losses"] / max(s["trades"], 1)) * 100, 1)
            result[r] = {
                "trades":   s["trades"],
                "losses":   s["losses"],
                "win_rate": win_rate,
                "verdict":  "✅ Good" if win_rate >= 60 else "⚠️ Caution" if win_rate >= 45 else "⛔ Avoid",
            }

        return result

    # ──────────────────────────────────────────────────────────
    # PRINT SUMMARY
    # ──────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        lessons = self._load()
        bar = "═" * 60
        print(f"\n{bar}")
        print("  🧠  LESSON MEMORY SUMMARY  (Day 52)")
        print(bar)
        print(f"  Total lessons stored: {len(lessons)}")
        print()

        pattern_stats = self.get_pattern_stats()
        if pattern_stats:
            print("  ── Pattern Performance ──")
            for p, s in sorted(pattern_stats.items(), key=lambda x: x[1]["win_rate"]):
                icon = "✅" if s["win_rate"] >= 60 else "⚠️" if s["win_rate"] >= 45 else "⛔"
                print(f"  {icon}  {p:<28} trades={s['trades']:<4} win_rate={s['win_rate']}%")
            print()

        regime_stats = self.get_regime_stats()
        if regime_stats:
            print("  ── Market Regime Performance ──")
            for r, s in regime_stats.items():
                print(f"  {s['verdict']}  {r:<16} win_rate={s['win_rate']}%  trades={s['trades']}")

        print(bar + "\n")

    # ──────────────────────────────────────────────────────────
    # STORAGE
    # ──────────────────────────────────────────────────────────

    def _load(self) -> list:
        if not os.path.exists(LESSON_DB_PATH):
            return []
        try:
            with open(LESSON_DB_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _save(self, lessons: list) -> None:
        with open(LESSON_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(lessons[-1000:], f, indent=2, default=str)