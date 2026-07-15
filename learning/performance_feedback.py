# learning/performance_feedback.py  —  Day 52 | Strategy Performance Feedback ⭐⭐⭐⭐⭐
# ============================================================
# AI আলাদা করে track করে:
#   - Pattern performance (Hammer: 62%, Engulfing: 48%)
#   - Market condition performance (Trending: PF 2.1, Ranging: PF 0.8)
#   - Timeframe performance (H1: Best, M5: Bad)
#   - Overall system health
#
# MasterAnalyst এই feedback দেখে নিজের confidence adjust করে।
# ============================================================

import json
import os
from datetime import datetime, timezone
from typing import Optional

from utils.logger import get_logger

log = get_logger("learning.performance_feedback")

FEEDBACK_DB_PATH = "memory/performance_feedback.json"
TIMEFRAME_DB_PATH = "memory/timeframe_performance.json"


class PerformanceFeedback:
    """
    Strategy-level performance analytics।

    Usage:
        feedback = PerformanceFeedback()

        # Trade record করো:
        feedback.record_trade(
            pattern="Hammer",
            regime="TRENDING",
            timeframe="H1",
            outcome="WIN",
            pnl=35.5,
            rr=1.8,
        )

        # Report দেখো:
        feedback.print_full_report()

        # MasterAnalyst-এর জন্য context:
        ctx = feedback.get_master_context()
    """

    def __init__(self):
        os.makedirs("memory", exist_ok=True)

    # ──────────────────────────────────────────────────────────
    # RECORD A TRADE
    # ──────────────────────────────────────────────────────────

    def record_trade(
        self,
        outcome: str,           # "WIN" | "LOSS" | "BE"
        pnl: float,
        pattern: str = None,
        regime: str = None,
        timeframe: str = None,
        pair: str = None,
        rr: float = None,
        confidence: int = None,
    ) -> None:
        """Trade result record করো।"""
        db = self._load()

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "outcome":   outcome,
            "pnl":       pnl,
            "pattern":   pattern,
            "regime":    regime,
            "timeframe": timeframe,
            "pair":      pair,
            "rr":        rr,
            "confidence": confidence,
            "win":       outcome == "WIN",
        }

        db.append(entry)
        self._save(db)

        log.info(f"[PerformanceFeedback] Recorded: {outcome} | PnL={pnl} | {pattern} | {regime} | {timeframe}")

    # ──────────────────────────────────────────────────────────
    # PATTERN PERFORMANCE
    # ──────────────────────────────────────────────────────────

    def get_pattern_performance(self) -> dict:
        """
        Pattern-wise breakdown।

        Example:
            Hammer:        trades=100  win_rate=62%  profit_factor=2.1
            Engulfing BUY: trades=80   win_rate=48%  profit_factor=0.9
        """
        db = self._load()
        stats = {}

        for t in db:
            p = t.get("pattern") or "Unknown"
            if p not in stats:
                stats[p] = {"trades": 0, "wins": 0, "gross_profit": 0, "gross_loss": 0}

            stats[p]["trades"] += 1
            if t.get("win"):
                stats[p]["wins"] += 1
                stats[p]["gross_profit"] += abs(t.get("pnl", 0))
            else:
                stats[p]["gross_loss"] += abs(t.get("pnl", 0))

        result = {}
        for p, s in stats.items():
            win_rate = round(s["wins"] / s["trades"] * 100, 1) if s["trades"] > 0 else 0
            pf = round(s["gross_profit"] / max(s["gross_loss"], 0.01), 2)
            result[p] = {
                "trades":         s["trades"],
                "wins":           s["wins"],
                "win_rate":       win_rate,
                "profit_factor":  pf,
                "gross_profit":   round(s["gross_profit"], 2),
                "gross_loss":     round(s["gross_loss"], 2),
                "verdict":        "✅ Good" if pf >= 1.5 else "⚠️ Marginal" if pf >= 1.0 else "⛔ Unprofitable",
            }

        return result

    # ──────────────────────────────────────────────────────────
    # MARKET CONDITION PERFORMANCE
    # ──────────────────────────────────────────────────────────

    def get_regime_performance(self) -> dict:
        """
        Market condition-wise breakdown।

        Example:
            TRENDING: profit_factor=2.1, win_rate=65%
            RANGING:  profit_factor=0.8, win_rate=42%
        """
        db = self._load()
        stats = {}

        for t in db:
            r = t.get("regime") or "UNKNOWN"
            if r not in stats:
                stats[r] = {"trades": 0, "wins": 0, "gross_profit": 0, "gross_loss": 0}

            stats[r]["trades"] += 1
            if t.get("win"):
                stats[r]["wins"] += 1
                stats[r]["gross_profit"] += abs(t.get("pnl", 0))
            else:
                stats[r]["gross_loss"] += abs(t.get("pnl", 0))

        result = {}
        for r, s in stats.items():
            win_rate = round(s["wins"] / s["trades"] * 100, 1) if s["trades"] > 0 else 0
            pf = round(s["gross_profit"] / max(s["gross_loss"], 0.01), 2)
            result[r] = {
                "trades":        s["trades"],
                "wins":          s["wins"],
                "win_rate":      win_rate,
                "profit_factor": pf,
                "verdict":       "✅ Favorable" if pf >= 1.5 else "⚠️ Neutral" if pf >= 0.9 else "⛔ Avoid",
            }

        return result

    # ──────────────────────────────────────────────────────────
    # TIMEFRAME PERFORMANCE
    # ──────────────────────────────────────────────────────────

    def get_timeframe_performance(self) -> dict:
        """
        Timeframe-wise breakdown।

        Example:
            H1:  win_rate=65%  — Best
            M15: win_rate=52%  — Acceptable
            M5:  win_rate=38%  — Bad
        """
        db = self._load()
        stats = {}

        for t in db:
            tf = t.get("timeframe") or "UNKNOWN"
            if tf not in stats:
                stats[tf] = {"trades": 0, "wins": 0, "gross_profit": 0, "gross_loss": 0}

            stats[tf]["trades"] += 1
            if t.get("win"):
                stats[tf]["wins"] += 1
                stats[tf]["gross_profit"] += abs(t.get("pnl", 0))
            else:
                stats[tf]["gross_loss"] += abs(t.get("pnl", 0))

        result = {}
        for tf, s in stats.items():
            win_rate = round(s["wins"] / s["trades"] * 100, 1) if s["trades"] > 0 else 0
            pf = round(s["gross_profit"] / max(s["gross_loss"], 0.01), 2)
            result[tf] = {
                "trades":        s["trades"],
                "win_rate":      win_rate,
                "profit_factor": pf,
                "verdict":       "⭐ Best" if win_rate >= 60 else "✅ Good" if win_rate >= 50 else "⚠️ Marginal" if win_rate >= 40 else "⛔ Bad",
            }

        return result

    # ──────────────────────────────────────────────────────────
    # MASTER CONTEXT (MasterAnalyst injection)
    # ──────────────────────────────────────────────────────────

    def get_master_context(self) -> dict:
        """MasterAnalyst-এ inject করার জন্য summarized context।"""
        db = self._load()

        if not db:
            return {"has_feedback": False}

        total = len(db)
        wins = [t for t in db if t.get("win")]
        win_rate = round(len(wins) / total * 100, 1)

        # Best/worst pattern
        pat_perf = self.get_pattern_performance()
        best_pattern  = max(pat_perf.items(), key=lambda x: x[1]["win_rate"])[0] if pat_perf else None
        worst_pattern = min(pat_perf.items(), key=lambda x: x[1]["win_rate"])[0] if pat_perf else None

        # Best/worst regime
        reg_perf = self.get_regime_performance()
        best_regime  = max(reg_perf.items(), key=lambda x: x[1]["profit_factor"])[0] if reg_perf else None
        worst_regime = min(reg_perf.items(), key=lambda x: x[1]["profit_factor"])[0] if reg_perf else None

        # Best timeframe
        tf_perf = self.get_timeframe_performance()
        best_tf = max(tf_perf.items(), key=lambda x: x[1]["win_rate"])[0] if tf_perf else None

        return {
            "has_feedback":   True,
            "total_trades":   total,
            "overall_win_rate": win_rate,
            "best_pattern":   best_pattern,
            "worst_pattern":  worst_pattern,
            "best_regime":    best_regime,
            "worst_regime":   worst_regime,
            "best_timeframe": best_tf,
            "avoid_regime":   worst_regime if reg_perf and reg_perf.get(worst_regime, {}).get("profit_factor", 1) < 0.9 else None,
            "feedback_summary": (
                f"Best: {best_pattern} in {best_regime} on {best_tf}. "
                f"Avoid: {worst_pattern} in {worst_regime}. "
                f"Overall win rate: {win_rate}%"
            ),
        }

    # ──────────────────────────────────────────────────────────
    # FULL REPORT
    # ──────────────────────────────────────────────────────────

    def print_full_report(self) -> None:
        bar = "═" * 62
        print(f"\n{bar}")
        print("  📊  STRATEGY PERFORMANCE FEEDBACK  (Day 52)")
        print(bar)

        # Pattern
        pat_perf = self.get_pattern_performance()
        if pat_perf:
            print("\n  ── Pattern Performance ──")
            print(f"  {'Pattern':<28} {'Trades':<8} {'Win%':<8} {'PF'}")
            for p, s in sorted(pat_perf.items(), key=lambda x: x[1]["win_rate"], reverse=True):
                print(f"  {s['verdict']}  {p:<28} {s['trades']:<8} {s['win_rate']:<8} {s['profit_factor']}")

        # Regime
        reg_perf = self.get_regime_performance()
        if reg_perf:
            print("\n  ── Market Condition Performance ──")
            for r, s in sorted(reg_perf.items(), key=lambda x: x[1]["profit_factor"], reverse=True):
                print(f"  {s['verdict']}  {r:<16} win_rate={s['win_rate']}%  PF={s['profit_factor']}")

        # Timeframe
        tf_perf = self.get_timeframe_performance()
        if tf_perf:
            print("\n  ── Timeframe Performance ──")
            for tf, s in sorted(tf_perf.items(), key=lambda x: x[1]["win_rate"], reverse=True):
                print(f"  {s['verdict']}  {tf:<8} win_rate={s['win_rate']}%  trades={s['trades']}")

        # Master context
        ctx = self.get_master_context()
        if ctx.get("has_feedback"):
            print(f"\n  ── Summary ──")
            print(f"  {ctx['feedback_summary']}")

        print(bar + "\n")

    # ──────────────────────────────────────────────────────────
    # STORAGE
    # ──────────────────────────────────────────────────────────

    def _load(self) -> list:
        if not os.path.exists(FEEDBACK_DB_PATH):
            return []
        try:
            with open(FEEDBACK_DB_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _save(self, db: list) -> None:
        with open(FEEDBACK_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(db[-2000:], f, indent=2, default=str)