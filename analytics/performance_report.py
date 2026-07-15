# analytics/performance_report.py  —  Day 54 | Weekly Performance Report
# ============================================================
# AI Trading System-এর weekly performance report তৈরি করে।
#
# Features:
#   ✅ Best setup breakdown (pair + TF + session + pattern)
#   ✅ Worst setup list (disabled setups)
#   ✅ Walk-forward stats (lifetime / 30d / 7d)
#   ✅ Session & Day-of-Week breakdown
#   ✅ Regime performance
#   ✅ Monte Carlo simulation (account survival check)
#   ✅ Strategy Version Control (কোন version-এ কেমন ছিল)
#   ✅ AI Optimization Suggestions
# ============================================================

from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np

from analytics.strategy_tracker import StrategyTracker
from analytics.ranking_engine import RankingEngine
from utils.logger import get_logger

log = get_logger("performance_report")


# ════════════════════════════════════════════════════════════
# STRATEGY VERSION CONTROL
# ════════════════════════════════════════════════════════════

class StrategyVersionControl:
    """
    প্রতিটি strategy version-এর performance আলাদাভাবে track করো।

    Usage:
        svc = StrategyVersionControl(tracker)
        svc.compare_versions()
    """

    def __init__(self, tracker: StrategyTracker):
        self.tracker = tracker

    def compare_versions(self) -> dict:
        """কোন version-এ কেমন performance ছিল।"""
        try:
            with sqlite3.connect(self.tracker.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute("""
                    SELECT strategy_ver,
                           COUNT(*) as trades,
                           SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) as wins,
                           AVG(rr_actual) as avg_rr,
                           SUM(profit_pips) as total_pips,
                           MIN(timestamp) as first_trade,
                           MAX(timestamp) as last_trade
                    FROM trades
                    WHERE result IS NOT NULL
                    GROUP BY strategy_ver
                    ORDER BY first_trade ASC
                """).fetchall()
        except Exception as e:
            log.warning(f"[VersionControl] DB error: {e}")
            return {}

        versions = {}
        for r in rows:
            trades = r["trades"]
            wins   = r["wins"] or 0
            pf_row = None
            try:
                with sqlite3.connect(self.tracker.db_path) as conn:
                    pf_row = conn.execute("""
                        SELECT SUM(CASE WHEN profit_pips > 0 THEN profit_pips ELSE 0 END),
                               ABS(SUM(CASE WHEN profit_pips < 0 THEN profit_pips ELSE 0 END))
                        FROM trades
                        WHERE strategy_ver=? AND result IS NOT NULL
                    """, (r["strategy_ver"],)).fetchone()
            except Exception:
                pass

            pf = 1.0
            if pf_row and pf_row[1] and pf_row[1] > 0:
                pf = round(pf_row[0] / pf_row[1], 2)

            versions[r["strategy_ver"]] = {
                "trades":      trades,
                "wins":        wins,
                "win_rate":    round(wins / trades * 100, 1) if trades else 0,
                "avg_rr":      round(r["avg_rr"] or 0, 2),
                "total_pips":  round(r["total_pips"] or 0, 1),
                "profit_factor": pf,
                "period":      f"{r['first_trade'][:10]} → {r['last_trade'][:10]}",
            }

        return versions

    def print_version_comparison(self, versions: dict) -> None:
        if not versions:
            print("  No version data available.")
            return

        bar = "─" * 60
        print(f"\n  ── Strategy Version Comparison ──")
        print(f"  {bar}")
        for ver, stats in versions.items():
            change = ""
            vers_list = list(versions.keys())
            idx = vers_list.index(ver)
            if idx > 0:
                prev = versions[vers_list[idx - 1]]
                diff = stats["win_rate"] - prev["win_rate"]
                change = f"  ({diff:+.1f}% vs {vers_list[idx-1]})"
            print(
                f"  {ver:12s} | WR: {stats['win_rate']:5.1f}%{change} | "
                f"PF: {stats['profit_factor']:.2f} | "
                f"RR: {stats['avg_rr']:.2f} | "
                f"Pips: {stats['total_pips']:+.1f} | "
                f"{stats['period']}"
            )


# ════════════════════════════════════════════════════════════
# MONTE CARLO — ACCOUNT SURVIVAL CHECK
# ════════════════════════════════════════════════════════════

class MonteCarloSimulator:
    """
    পরপর N loss হলে account survive করবে?
    Day 54-এর risk assessment tool।
    """

    def simulate(
        self,
        tracker: StrategyTracker,
        initial_balance: float = 10_000.0,
        risk_per_trade_pct: float = 1.0,
        runs: int = 1_000,
        consecutive_loss_check: int = 10,
    ) -> dict:
        """
        1000 বার trade sequence shuffle করে worst case বের করো।
        """
        # DB থেকে trade pnl নাও
        try:
            with sqlite3.connect(tracker.db_path) as conn:
                rows = conn.execute(
                    "SELECT profit_pips FROM trades WHERE result IS NOT NULL"
                ).fetchall()
        except Exception as e:
            log.warning(f"[MonteCarlo] DB error: {e}")
            return {"status": "no_data"}

        if len(rows) < 10:
            return {"status": "insufficient_data", "trades_available": len(rows)}

        # pip → dollar conversion (approximate: 1 pip ≈ risk_per_trade_pct% / avg_sl_pips)
        # Simplified: 1 pip = $1 per $10k account at standard lot
        pip_value = initial_balance * 0.001   # rough approximation

        pnl_values = [r[0] * pip_value for r in rows if r[0] is not None]
        rng        = np.random.default_rng(42)

        final_balances = []
        drawdowns      = []
        bankruptcies   = 0

        for _ in range(runs):
            sampled = rng.permutation(pnl_values)
            equity  = initial_balance
            peak    = initial_balance
            max_dd  = 0.0

            for pnl in sampled:
                equity += pnl
                if equity <= 0:
                    bankruptcies += 1
                    equity = 0
                    break
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak * 100 if peak > 0 else 0
                max_dd = max(max_dd, dd)

            final_balances.append(equity)
            drawdowns.append(max_dd)

        # Consecutive loss survival
        sorted_pnl = sorted(pnl_values)[:consecutive_loss_check]  # worst N trades
        consecutive_loss_balance = initial_balance + sum(sorted_pnl)

        return {
            "status":                  "ok",
            "runs":                    runs,
            "initial_balance":         initial_balance,
            "best_final":              round(max(final_balances), 2),
            "worst_final":             round(min(final_balances), 2),
            "median_final":            round(float(np.median(final_balances)), 2),
            "mean_final":              round(float(np.mean(final_balances)), 2),
            "worst_drawdown_pct":      round(float(max(drawdowns)), 2),
            "median_drawdown_pct":     round(float(np.median(drawdowns)), 2),
            "bankruptcy_count":        bankruptcies,
            "bankruptcy_rate_pct":     round(bankruptcies / runs * 100, 2),
            "survives_10_losses":      consecutive_loss_balance > 0,
            "balance_after_10_losses": round(consecutive_loss_balance, 2),
        }


# ════════════════════════════════════════════════════════════
# OPTIMIZATION SUGGESTIONS
# ════════════════════════════════════════════════════════════

class OptimizationSuggester:
    """
    AI নিজে নিজে rule পরিবর্তন সাজেস্ট করে।
    সরাসরি বদলায় না — Observation → Suggestion → Backtest → Approval flow।
    """

    def generate(
        self,
        tracker: StrategyTracker,
        ranker:  RankingEngine,
    ) -> list[dict]:
        """Performance data বিশ্লেষণ করে structured suggestions তৈরি করো।"""
        suggestions = []

        # ── Session suggestions ──────────────────────────
        sess_perf = tracker.session_performance()
        best_sess  = None
        worst_sess = None

        for sess, stats in sess_perf.items():
            if stats["trades"] >= 10:
                if best_sess is None or stats["win_rate"] > sess_perf[best_sess]["win_rate"]:
                    best_sess = sess
                if worst_sess is None or stats["win_rate"] < sess_perf[worst_sess]["win_rate"]:
                    worst_sess = sess

        if best_sess and worst_sess and best_sess != worst_sess:
            best_wr  = sess_perf[best_sess]["win_rate"]
            worst_wr = sess_perf[worst_sess]["win_rate"]
            diff     = best_wr - worst_wr

            if diff >= 20:
                suggestions.append({
                    "type":        "SESSION_PREFERENCE",
                    "observation": f"{best_sess} performs {diff:.0f}% better than {worst_sess}.",
                    "suggestion":  f"Increase confidence +10% during {best_sess}. Reduce or avoid {worst_sess}.",
                    "action":      f"BOOST_{best_sess.upper()}_CONFIDENCE",
                    "impact":      "HIGH" if diff >= 30 else "MEDIUM",
                    "requires_backtest": True,
                })

        # ── Pair suggestions ──────────────────────────────
        pair_perf = tracker.pair_performance()
        if len(pair_perf) >= 2:
            pairs_sorted = sorted(pair_perf.items(), key=lambda x: x[1]["win_rate"], reverse=True)
            best_pair   = pairs_sorted[0]
            worst_pair  = pairs_sorted[-1]

            if (best_pair[1]["trades"] >= 10 and
                    worst_pair[1]["trades"] >= 10 and
                    best_pair[1]["win_rate"] - worst_pair[1]["win_rate"] >= 15):
                suggestions.append({
                    "type":        "PAIR_PREFERENCE",
                    "observation": (
                        f"{best_pair[0]} win rate {best_pair[1]['win_rate']:.1f}% vs "
                        f"{worst_pair[0]} {worst_pair[1]['win_rate']:.1f}%."
                    ),
                    "suggestion":  f"Prioritize {best_pair[0]}. Reduce {worst_pair[0]} position size.",
                    "action":      f"REDUCE_{worst_pair[0].upper()}_ALLOCATION",
                    "impact":      "MEDIUM",
                    "requires_backtest": True,
                })

        # ── Regime suggestions ────────────────────────────
        regime_perf = tracker.regime_performance()
        avoid_regimes = []
        for regime, stats in regime_perf.items():
            if stats["trades"] >= 10 and stats["win_rate"] < 40:
                avoid_regimes.append(regime)
                suggestions.append({
                    "type":        "REGIME_FILTER",
                    "observation": f"{regime} market win rate only {stats['win_rate']:.1f}%.",
                    "suggestion":  f"Disable trading in {regime} regime until conditions improve.",
                    "action":      f"DISABLE_IN_{regime.upper()}",
                    "impact":      "HIGH",
                    "requires_backtest": False,   # ঝুঁকি কমানো — immediate apply করা যায়
                })

        # ── Day of week suggestions ────────────────────────
        dow_perf = tracker.day_of_week_performance()
        for day, stats in dow_perf.items():
            if stats["trades"] >= 8 and stats["win_rate"] < 40:
                suggestions.append({
                    "type":        "DAY_FILTER",
                    "observation": f"{day} win rate only {stats['win_rate']:.1f}%.",
                    "suggestion":  f"Reduce or skip trading on {day}.",
                    "action":      f"REDUCE_{day.upper()}_TRADING",
                    "impact":      "LOW",
                    "requires_backtest": True,
                })

        # ── Walk-forward performance drop ─────────────────
        wf = tracker.walk_forward_stats()
        lifetime_wr = wf["lifetime"]["win_rate"]
        recent7_wr  = wf["last_7_days"]["win_rate"]

        if (wf["last_7_days"]["trades"] >= 5 and
                lifetime_wr - recent7_wr >= 20):
            suggestions.append({
                "type":        "WALK_FORWARD_ALERT",
                "observation": (
                    f"Recent 7-day win rate ({recent7_wr:.1f}%) dropped "
                    f"{lifetime_wr - recent7_wr:.0f}% below lifetime average ({lifetime_wr:.1f}%)."
                ),
                "suggestion":  "Market conditions may have changed. Reduce position size and increase confirmation threshold.",
                "action":      "INCREASE_CONFIRMATION_THRESHOLD",
                "impact":      "HIGH",
                "requires_backtest": False,
            })

        return suggestions


# ════════════════════════════════════════════════════════════
# WEEKLY PERFORMANCE REPORT
# ════════════════════════════════════════════════════════════

class PerformanceReport:
    """
    Day 54 — সব analytics একত্র করে weekly report তৈরি করে।

    Usage:
        tracker = StrategyTracker()
        report  = PerformanceReport(tracker)
        report.generate()      # console
        report.to_json()       # JSON dict
        report.save("reports/week_01.json")
    """

    def __init__(
        self,
        tracker: StrategyTracker,
        initial_balance: float = 10_000.0,
    ):
        self.tracker     = tracker
        self.ranker      = RankingEngine(tracker)
        self.svc         = StrategyVersionControl(tracker)
        self.mc          = MonteCarloSimulator()
        self.suggester   = OptimizationSuggester()
        self.balance     = initial_balance

    # ─────────────────────────────────────────────
    # GENERATE  (main entry point)
    # ─────────────────────────────────────────────

    def generate(self, print_report: bool = True) -> dict:
        """সব analytics run করো এবং full report dict return করো।"""

        report = {
            "generated_at":       datetime.now(timezone.utc).isoformat(),
            "walk_forward":       self.tracker.walk_forward_stats(),
            "pair_performance":   self.tracker.pair_performance(),
            "session_performance": self.tracker.session_performance(),
            "day_of_week":        self.tracker.day_of_week_performance(),
            "regime_performance": self.tracker.regime_performance(),
            "pattern_matrix":     self.tracker.pattern_performance_matrix(),
            "best_worst_setups":  self.tracker.best_worst_setups(),
            "preferred_timeframe": self.tracker.preferred_timeframe(),
            "rankings":           self.ranker.rank_all_setups(),
            "disabled_setups":    self.tracker.get_disabled_setups(),
            "version_comparison": self.svc.compare_versions(),
            "monte_carlo":        self.mc.simulate(self.tracker, self.balance),
            "suggestions":        self.suggester.generate(self.tracker, self.ranker),
        }

        if print_report:
            self._print(report)

        return report

    # ─────────────────────────────────────────────
    # SAVE TO FILE
    # ─────────────────────────────────────────────

    def save(self, filepath: str, report: dict = None) -> None:
        import os
        if report is None:
            report = self.generate(print_report=False)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=str)
        log.info(f"[PerformanceReport] Saved → {filepath}")

    def to_json(self, report: dict = None) -> str:
        if report is None:
            report = self.generate(print_report=False)
        return json.dumps(report, indent=2, default=str)

    # ─────────────────────────────────────────────
    # PRINT REPORT
    # ─────────────────────────────────────────────

    def _print(self, r: dict) -> None:
        bar  = "═" * 62
        bar2 = "─" * 62

        print(f"\n{bar}")
        print("  📊  WEEKLY AI TRADING PERFORMANCE REPORT  (Day 54)")
        print(f"  Generated: {r['generated_at'][:19]} UTC")
        print(bar)

        # ── Walk-Forward ─────────────────────────────────
        print("\n  ── Walk-Forward Stats ──")
        wf = r["walk_forward"]
        for period, stats in wf.items():
            icon = "🟢" if stats["win_rate"] >= 55 else ("🟡" if stats["win_rate"] >= 45 else "🔴")
            print(
                f"  {icon} {period:15s}: {stats['trades']:4d} trades | "
                f"WR: {stats['win_rate']:5.1f}% | "
                f"Total: {stats['total_pips']:+.1f} pips"
            )

        # ── Best Setup ──────────────────────────────────
        bw = r["best_worst_setups"]
        if bw["best"]:
            print(f"\n  ── 🏆 Best Setup ──")
            s = bw["best"][0]
            print(f"  Pair:      {s['pair']} {s['timeframe']}")
            print(f"  Session:   {s['session']}")
            print(f"  Pattern:   {s['pattern']}")
            print(f"  Regime:    {s['regime']}")
            print(f"  Win Rate:  {s['win_rate']:.1f}%")
            print(f"  Avg R:R:   {s['avg_rr']}")
            print(f"  Trades:    {s['trades']}")

        # ── Worst Setup ─────────────────────────────────
        if bw["worst"]:
            print(f"\n  ── ⚠️  Worst Setup ──")
            s = bw["worst"][0]
            print(f"  Pair:      {s['pair']} {s['timeframe']}")
            print(f"  Session:   {s['session']}")
            print(f"  Pattern:   {s['pattern']}")
            print(f"  Win Rate:  {s['win_rate']:.1f}%  ← REVIEW NEEDED")

        # ── Session Performance ──────────────────────────
        sess = r["session_performance"]
        if sess:
            print(f"\n  ── Session Breakdown ──")
            for s, stats in sess.items():
                icon = "✅" if stats["win_rate"] >= 55 else ("⚠️ " if stats["win_rate"] >= 45 else "❌")
                print(
                    f"  {icon} {s:22s}: WR {stats['win_rate']:5.1f}% | "
                    f"{stats['trades']} trades"
                )

        # ── Day of Week ──────────────────────────────────
        dow = r["day_of_week"]
        if dow:
            print(f"\n  ── Day of Week ──")
            for day, stats in dow.items():
                icon = "✅" if stats["win_rate"] >= 60 else ("⚠️ " if stats["win_rate"] >= 45 else "❌")
                flag = "  ← REDUCE TRADING" if stats["win_rate"] < 45 else ""
                print(
                    f"  {icon} {day:12s}: WR {stats['win_rate']:5.1f}%"
                    f" | {stats['trades']} trades{flag}"
                )

        # ── Regime Performance ───────────────────────────
        regime = r["regime_performance"]
        if regime:
            print(f"\n  ── Regime Performance ──")
            for reg, stats in regime.items():
                icon = "✅" if stats["win_rate"] >= 55 else ("⚠️ " if stats["win_rate"] >= 45 else "❌")
                print(
                    f"  {icon} {reg:20s}: WR {stats['win_rate']:5.1f}% | "
                    f"RR: {stats['avg_rr']} | {stats['trades']} trades"
                )

        # ── Rankings (Top 5) ─────────────────────────────
        rankings = r.get("rankings", [])
        if rankings:
            print(f"\n  ── Top 5 Setups (by Score) ──")
            for i, s in enumerate(rankings[:5], 1):
                icon = "✅" if s["recommendation"] == "TRADE" else "⚠️ "
                print(
                    f"  {i}. {icon} Score {s['score']:5.1f} | "
                    f"{s['pair']} {s['timeframe']} {s['session'][:15]} | "
                    f"{s['pattern'][:18]} | WR: {s['win_rate']:.1f}%"
                )

        # ── Disabled Setups ─────────────────────────────
        dis = r.get("disabled_setups", [])
        if dis:
            print(f"\n  ── ⛔ Auto-Disabled Setups ──")
            for d in dis:
                print(f"  ❌ {d['setup_key']}")
                print(f"     {d['reason'][:55]}")

        # ── Strategy Versions ────────────────────────────
        versions = r.get("version_comparison", {})
        if versions:
            self.svc.print_version_comparison(versions)

        # ── Monte Carlo ──────────────────────────────────
        mc = r.get("monte_carlo", {})
        if mc.get("status") == "ok":
            print(f"\n  ── 🎲 Monte Carlo Risk Simulation ──")
            print(f"  Runs:              {mc['runs']:,}")
            print(f"  Initial Balance:   ${mc['initial_balance']:,.2f}")
            print(f"  Median Outcome:    ${mc['median_final']:,.2f}")
            print(f"  Worst Case:        ${mc['worst_final']:,.2f}")
            print(f"  Max Drawdown:      {mc['worst_drawdown_pct']:.1f}%")
            print(f"  Bankruptcy Rate:   {mc['bankruptcy_rate_pct']:.2f}%")
            surv_icon = "✅" if mc["survives_10_losses"] else "❌"
            print(
                f"  {surv_icon} After 10 Consecutive Losses: "
                f"${mc['balance_after_10_losses']:,.2f}"
            )

        # ── AI Suggestions ─────────────────────────────
        suggestions = r.get("suggestions", [])
        if suggestions:
            print(f"\n  ── 🤖 AI Optimization Suggestions ──")
            for i, sug in enumerate(suggestions, 1):
                impact_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(sug["impact"], "⚪")
                bt = " [NEEDS BACKTEST]" if sug["requires_backtest"] else " [IMMEDIATE]"
                print(f"\n  {i}. {impact_icon} [{sug['impact']}]{bt}")
                print(f"     Observation: {sug['observation'][:65]}")
                print(f"     Suggestion:  {sug['suggestion'][:65]}")
                print(f"     Action:      {sug['action']}")

        print(f"\n{bar}\n")