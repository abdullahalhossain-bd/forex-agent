# learning/weekly_review.py  —  Day 55 | Weekly Review Scheduler + Report Generator ⭐⭐⭐⭐⭐
# ============================================================
# AI প্রতি রবিবার (বা manual trigger-এ) weekly_optimizer() চালায় এবং
# তার ফলাফল থেকে একটা human-readable Explainable Report তৈরি করে।
#
# "আমি কেন পরিবর্তন করলাম?" — প্রতিটা suggestion-এর reason
# স্পষ্টভাবে দেখানো হয়, যাতে human approval mode-এ সহজে review করা যায়।
# ============================================================

import json
import os
from datetime import datetime, timezone

from utils.logger import get_logger
from learning.auto_optimizer import AutoOptimizer
from learning.performance_feedback import PerformanceFeedback
from learning.strategy_config import StrategyConfig

log = get_logger("learning.weekly_review")

WEEKLY_REPORT_LOG_PATH = "memory/weekly_reports.json"
REVIEW_DAY = 6   # Python: Monday=0 ... Sunday=6


def is_review_day(today: datetime = None) -> bool:
    """আজ কি saptahik review চালানোর দিন (রবিবার)?"""
    today = today or datetime.now(timezone.utc)
    return today.weekday() == REVIEW_DAY


def run_weekly_review(human_approval: bool = True, force: bool = False, days: int = 7) -> dict:
    """
    Day 55 main entry point।

    Usage:
        from learning.weekly_review import run_weekly_review
        report = run_weekly_review()       # human approval mode (default)
        report = run_weekly_review(human_approval=False)  # fully autonomous

    `force=True` দিলে সপ্তাহের যেকোনো দিনেই চালানো যাবে (manual run / testing)।
    """
    if not force and not is_review_day():
        log.info("[WeeklyReview] Not review day (Sunday) — skipping. Use force=True to override.")
        return {"skipped": True, "reason": "Not Sunday"}

    optimizer = AutoOptimizer(human_approval=human_approval)
    run_record = optimizer.weekly_optimizer(days=days)

    report = WeeklyReportGenerator(optimizer, run_record, days=days).build()
    _save_report(report)
    _print_report(report)

    return report


class WeeklyReportGenerator:
    """
    AutoOptimizer-এর run output থেকে spec-format-এ একটা readable report বানায়।

    Output structure মেলে Day 55 spec-এর sample report-এর সাথে:
        📊 Weekly AI Optimization Report
        Period, Best Strategy, Disabled, Risk Adjustment ...
    """

    def __init__(self, optimizer: AutoOptimizer, run_record: dict, days: int = 7):
        self.optimizer = optimizer
        self.run_record = run_record
        self.days = days
        self.feedback = PerformanceFeedback()
        self.config = StrategyConfig()

    def build(self) -> dict:
        period_end = datetime.now(timezone.utc)
        period_start = period_end.fromordinal(period_end.toordinal() - self.days)

        best_strategy = self._best_strategy()
        disabled = self._disabled_summary()
        risk_change = self._risk_change_summary()
        explainability = self._explain_all_actions()

        report = {
            "generated_at":   period_end.isoformat(),
            "period": {
                "start": period_start.strftime("%Y-%m-%d"),
                "end":   period_end.strftime("%Y-%m-%d"),
            },
            "trades_analyzed": self.run_record.get("trades_analyzed", 0),
            "best_strategy":   best_strategy,
            "disabled":        disabled,
            "risk_adjustment": risk_change,
            "applied_changes": self.run_record.get("applied", []),
            "queued_for_approval": self.run_record.get("queued_for_approval", []),
            "version":         self.run_record.get("version"),
            "rollback":        self.run_record.get("rollback"),
            "explainability":  explainability,
        }
        return report

    # ── component builders ──────────────────────────────────

    def _best_strategy(self) -> dict:
        ctx = self.feedback.get_master_context()
        if not ctx.get("has_feedback"):
            return {"summary": "Not enough data yet"}
        return {
            "pair_pattern":  f"{ctx.get('best_pattern')} on {ctx.get('best_timeframe')}",
            "win_rate":      ctx.get("overall_win_rate"),
            "best_regime":   ctx.get("best_regime"),
        }

    def _disabled_summary(self) -> list:
        out = []
        for action in self.run_record.get("applied", []) + self.run_record.get("queued_for_approval", []):
            if action.get("type") == "PAIR_REMOVE":
                out.append({"pair": action["target"], "reason": action["reason"]})
        if not out:
            disabled_pairs = self.config.get_disabled_pairs()
            for p, d in disabled_pairs.items():
                out.append({"pair": p, "reason": d.get("reason")})
        return out

    def _risk_change_summary(self) -> dict:
        for action in self.run_record.get("applied", []) + self.run_record.get("queued_for_approval", []):
            if action.get("type") == "RISK_ADJUSTMENT":
                return {
                    "changed":  f"{action['old_risk']}% → {action['new_risk']}%",
                    "reason":   action["reason"],
                }
        return {"changed": "No change", "reason": "Risk within normal range"}

    def _explain_all_actions(self) -> list:
        """⭐ Explainable Optimization — প্রতিটা action-এর পেছনের কারণ।"""
        all_actions = self.run_record.get("applied", []) + self.run_record.get("queued_for_approval", [])
        return [
            {"type": a.get("type"), "target": a.get("target"), "why": a.get("reason")}
            for a in all_actions
        ]

    # ── storage ──────────────────────────────────────────────


def _save_report(report: dict) -> None:
    reports = []
    if os.path.exists(WEEKLY_REPORT_LOG_PATH):
        try:
            with open(WEEKLY_REPORT_LOG_PATH, encoding="utf-8") as f:
                reports = json.load(f)
        except Exception:
            reports = []
    reports.append(report)
    with open(WEEKLY_REPORT_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(reports[-100:], f, indent=2, default=str)


def _print_report(report: dict) -> None:
    bar = "═" * 64
    print(f"\n{bar}")
    print("  📊  WEEKLY AI OPTIMIZATION REPORT  (Day 55)")
    print(bar)
    print(f"  Period          : {report['period']['start']} → {report['period']['end']}")
    print(f"  Trades analyzed : {report['trades_analyzed']}")
    print()

    bs = report["best_strategy"]
    print("  ── Best Strategy ──")
    if "summary" in bs:
        print(f"  {bs['summary']}")
    else:
        print(f"  {bs['pair_pattern']}  |  Win Rate: {bs['win_rate']}%  |  Best regime: {bs['best_regime']}")
    print()

    print("  ── Disabled ──")
    if report["disabled"]:
        for d in report["disabled"]:
            print(f"  ⛔ {d['pair']} — {d['reason']}")
    else:
        print("  None")
    print()

    print("  ── Risk Adjustment ──")
    print(f"  {report['risk_adjustment']['changed']}")
    print(f"  Reason: {report['risk_adjustment']['reason']}")
    print()

    if report.get("queued_for_approval"):
        print(f"  ⏸️  {len(report['queued_for_approval'])} suggestion(s) awaiting human approval")
        print(f"     Run: optimizer.list_pending() / optimizer.approve(id) / optimizer.reject(id)")
        print()

    if report.get("rollback") and report["rollback"].get("rolled_back"):
        print(f"  ⏮️  ROLLBACK TRIGGERED: {report['rollback']}")
        print()

    if report.get("version"):
        print(f"  💾 Strategy version saved: v{report['version']}")
        print()

    print("  ── Explainability ──")
    for e in report["explainability"]:
        print(f"  • [{e['type']}] {e['target']}: {e['why']}")

    print(bar + "\n")