# learning/auto_optimizer.py  —  Day 55 | Auto Strategy Adjustment (Self-Optimizing Trading Brain) ⭐⭐⭐⭐⭐
# ============================================================
# Day 54 পর্যন্ত AI শুধু analysis করতো (কোন pair/timeframe/session/pattern
# ভালো)। Day 55-এ AI নিজের trading system-এর configuration নিজে
# optimize করার ক্ষমতা পায় — কিন্তু সবসময় Safety Layer + Version Control
# এর মধ্য দিয়ে, যাতে impulsive বা overfit পরিবর্তন না হয়।
#
# Flow:
#   Trade History → Performance Tracker → Strategy Optimizer
#       → Pair Optimization / Pattern Optimization
#       → Session Analysis / Risk Adjustment
#       → Strategy Configuration → Decision Agent Update
#
# 10/10 Features implemented here:
#   ⭐ Weekly Review System          (weekly_optimizer)
#   ⭐ Underperforming Pair Mgmt     (_optimize_pairs)
#   ⭐ Pattern Optimization          (_optimize_patterns)
#   ⭐ Best Session Optimization     (_optimize_sessions)
#   ⭐ Dynamic / Volatility Risk     (_optimize_risk)
#   ⭐ Strategy Version Control      (via StrategyConfig)
#   ⭐ Weekly Report Generator       (delegated to weekly_review.py)
#   ⭐ Safety Layer                  (via optimizer_rules.validate_change)
#   ⭐ Human Approval Mode           (queue / approve / reject)
#   ⭐ A/B Testing, Regime Adaptation, Rollback, Explainability
# ============================================================

import json
import os
from datetime import datetime, timezone, timedelta
from typing import Optional

from utils.logger import get_logger
from learning import optimizer_rules as rules
from learning.strategy_config import StrategyConfig
from learning.performance_feedback import PerformanceFeedback, FEEDBACK_DB_PATH
from learning.confidence_engine import ConfidenceEngine
from learning.rule_updater import RuleUpdater
from learning.deep_analyzer import DeepMistakeAnalyzer

log = get_logger("learning.auto_optimizer")

PENDING_OPTIMIZER_PATH = "memory/pending_optimizer_approvals.json"
OPTIMIZER_LOG_PATH     = "memory/optimizer_run_log.json"


class AutoOptimizer:
    """
    Day 55 Main Class — AI-এর Self-Improvement Controller।

    Usage:
        optimizer = AutoOptimizer(human_approval=True)

        # প্রতি রবিবার (বা manually) চালাও:
        result = optimizer.weekly_optimizer()

        # Human approval mode-এ suggestion approve/reject করো:
        optimizer.list_pending()
        optimizer.approve("opt_20240623_1")
        optimizer.reject("opt_20240623_2")

        # Fully autonomous mode চাইলে:
        optimizer = AutoOptimizer(human_approval=False)
    """

    def __init__(self, human_approval: bool = rules.HUMAN_APPROVAL_MODE_DEFAULT):
        os.makedirs("memory", exist_ok=True)
        self.human_approval = human_approval
        self.config       = StrategyConfig()
        self.feedback     = PerformanceFeedback()
        self.confidence   = ConfidenceEngine()
        self.rule_updater = RuleUpdater()
        self.deep_analyzer = DeepMistakeAnalyzer()

    # ══════════════════════════════════════════════════════════
    # 1. WEEKLY REVIEW — MAIN ENTRY POINT
    # ══════════════════════════════════════════════════════════

    def weekly_optimizer(self, days: int = 7) -> dict:
        """
        Flow: Collect last N days data → Analyze performance →
              Find weakness → Generate improvement → Update configuration
        """
        log.info(f"[AutoOptimizer] 🔁 Weekly optimization run started (last {days} days)")

        window_trades = self._load_recent_trades(days)

        suggestions = []
        suggestions += self._optimize_pairs(window_trades)
        suggestions += self._optimize_patterns()
        suggestions += self._optimize_sessions(window_trades)
        suggestions += self._optimize_risk(window_trades)

        applied, queued = self._dispatch_suggestions(suggestions)

        # Version snapshot AFTER applying autonomous changes
        version_record = None
        if applied:
            snapshot = self._performance_snapshot(window_trades)
            new_label = self._next_version_label()
            version_record = self.config.save_version(
                label=new_label,
                notes="; ".join(a["reason"] for a in applied),
                performance_snapshot=snapshot,
            )

        # Rollback check against the previous version
        rollback_result = self._check_rollback()

        run_record = {
            "timestamp":        datetime.now(timezone.utc).isoformat(),
            "trades_analyzed":  len(window_trades),
            "suggestions_total": len(suggestions),
            "applied":          applied,
            "queued_for_approval": queued,
            "version":          version_record["label"] if version_record else None,
            "rollback":         rollback_result,
        }
        self._log_run(run_record)

        log.info(
            f"[AutoOptimizer] ✅ Weekly run complete | "
            f"{len(applied)} applied, {len(queued)} queued, "
            f"{len(suggestions)} total suggestions"
        )
        return run_record

    # ══════════════════════════════════════════════════════════
    # 2. PAIR OPTIMIZATION
    # ══════════════════════════════════════════════════════════

    def _optimize_pairs(self, trades: list) -> list:
        """
        প্রতিটা active pair-এর win rate / profit factor দেখে underperforming
        pair চিহ্নিত করো এবং REMOVE suggestion তৈরি করো।
        """
        suggestions = []
        stats = self._aggregate_by_key(trades, key="pair")

        for pair, s in stats.items():
            if pair not in self.config.get_active_pairs():
                continue  # ইতিমধ্যে disabled

            v = rules.validate_change(
                sample_size=s["total"], wins=s["wins"],
                min_sample=rules.MIN_TRADES_FOR_PAIR_DECISION,
            )
            if not v.ok:
                continue

            win_rate = s["win_rate"]
            pf = s["profit_factor"]

            if win_rate < rules.PAIR_REMOVE_WIN_RATE and pf < rules.PAIR_REMOVE_PROFIT_FACTOR:
                suggestions.append({
                    "type":   "PAIR_REMOVE",
                    "target": pair,
                    "action": "REMOVE",
                    "reason": (
                        f"Negative expectancy — win_rate={win_rate}%, PF={pf} "
                        f"over {s['total']} trades (z={v.z_score})"
                    ),
                    "stats":  s,
                    "validation": v.__dict__,
                    "autonomous_safe": True,
                })

        return suggestions

    # ══════════════════════════════════════════════════════════
    # 3. PATTERN OPTIMIZATION
    # ══════════════════════════════════════════════════════════

    def _optimize_patterns(self) -> list:
        """
        ConfidenceEngine-এর pattern database থেকে underperforming pattern
        খুঁজে confidence কমানোর suggestion তৈরি করো।
        """
        suggestions = []
        pattern_rows = self.confidence.get_pattern_summary()

        for row in pattern_rows:
            total = row["total"]
            win_rate = row["win_rate"]
            wins = round(win_rate / 100 * total)

            v = rules.validate_change(
                sample_size=total, wins=wins,
                min_sample=rules.MIN_TRADES_FOR_PATTERN_UPDATE,
            )
            if not v.ok:
                continue

            if win_rate < rules.PATTERN_LOW_WIN_RATE:
                key_parts = row["key"].split("|")
                pattern = key_parts[0] if key_parts else row["key"]

                current_conf = self.rule_updater.get_confidence(pattern, "GENERAL")
                new_conf = max(20, current_conf - 20)

                suggestions.append({
                    "type":   "PATTERN_CONFIDENCE",
                    "target": pattern,
                    "action": "LOWER_CONFIDENCE",
                    "reason": (
                        f"{pattern} win rate {win_rate}% over {total} trades "
                        f"(z={v.z_score}) — lowering confidence {current_conf}% → {new_conf}%"
                    ),
                    "old_confidence": current_conf,
                    "new_confidence": new_conf,
                    "condition": "Require additional confirmation (e.g. BOS) before use",
                    "validation": v.__dict__,
                    "autonomous_safe": True,
                })

        return suggestions

    # ══════════════════════════════════════════════════════════
    # 4. SESSION OPTIMIZATION
    # ══════════════════════════════════════════════════════════

    def _optimize_sessions(self, trades: list) -> list:
        """
        Trade data-তে 'session' field থাকলে, pair অনুযায়ী best/worst session
        বের করো এবং preference suggestion দাও।
        """
        suggestions = []
        by_pair_session = {}

        for t in trades:
            session = t.get("session")
            pair = t.get("pair")
            if not session or not pair:
                continue
            key = (pair, session)
            by_pair_session.setdefault(key, {"total": 0, "wins": 0})
            by_pair_session[key]["total"] += 1
            if t.get("win"):
                by_pair_session[key]["wins"] += 1

        if not by_pair_session:
            log.info("[AutoOptimizer] No session-tagged trades found — skipping session optimization")
            return suggestions

        # pair-wise group sessions
        per_pair = {}
        for (pair, session), s in by_pair_session.items():
            per_pair.setdefault(pair, {})[session] = s

        for pair, sessions in per_pair.items():
            scored = {}
            for session, s in sessions.items():
                if s["total"] < rules.MIN_TRADES_FOR_SESSION_DECISION:
                    continue
                scored[session] = round(s["wins"] / s["total"] * 100, 1)

            if not scored:
                continue

            best_session = max(scored, key=scored.get)
            worst_session = min(scored, key=scored.get)

            if scored[best_session] >= rules.SESSION_GOOD_WIN_RATE or scored[worst_session] <= rules.SESSION_BAD_WIN_RATE:
                suggestions.append({
                    "type":   "SESSION_PREFERENCE",
                    "target": pair,
                    "action": "SET_SESSION_PREFERENCE",
                    "reason": (
                        f"{pair}: {best_session} win_rate={scored[best_session]}% (preferred), "
                        f"{worst_session} win_rate={scored[worst_session]}% (avoid)"
                    ),
                    "preferred": best_session,
                    "avoid":     worst_session if scored[worst_session] <= rules.SESSION_BAD_WIN_RATE else None,
                    "autonomous_safe": True,
                })

        return suggestions

    # ══════════════════════════════════════════════════════════
    # 5. DYNAMIC RISK ADJUSTMENT  ⭐⭐⭐⭐⭐
    # ══════════════════════════════════════════════════════════

    def _optimize_risk(self, trades: list) -> list:
        """
        AI risk blindly বাড়ায় না। Win rate trend + drawdown + volatility
        দেখে formula অনুযায়ী risk adjust করার suggestion দেয়।

        risk = base_risk / volatility_factor
        """
        suggestions = []
        if len(trades) < rules.MIN_TRADES_FOR_RISK_CHANGE:
            return suggestions

        win_rate = self._win_rate(trades)
        drawdown = self._estimate_drawdown(trades)
        volatility_factor = self._estimate_volatility_factor(trades, drawdown)

        base_risk = self.config._load().get("base_risk_percent", rules.DEFAULT_BASE_RISK_PCT)
        current_risk = self.config.get_risk()

        proposed_risk = rules.volatility_to_risk(base_risk, volatility_factor)
        proposed_risk = rules.clamp_risk_step(current_risk, proposed_risk)

        if abs(proposed_risk - current_risk) < 0.05:
            return suggestions  # negligible change, skip noise

        if volatility_factor >= rules.HIGH_VOLATILITY_FACTOR * 0.8 or drawdown > 8:
            reason = f"High volatility/drawdown detected (drawdown={drawdown}%, vol_factor={volatility_factor})"
        elif win_rate >= 60 and drawdown < 5:
            reason = f"Stable market, win rate improving ({win_rate}%), low drawdown ({drawdown}%)"
        else:
            reason = f"Volatility-adjusted risk recalculation (vol_factor={volatility_factor})"

        suggestions.append({
            "type":   "RISK_ADJUSTMENT",
            "target": "global_risk",
            "action": "SET_RISK",
            "reason": f"Risk {current_risk}% → {proposed_risk}% | {reason}",
            "old_risk": current_risk,
            "new_risk": proposed_risk,
            "autonomous_safe": True,
        })

        return suggestions

    # ══════════════════════════════════════════════════════════
    # 6. DISPATCH: APPLY (autonomous) OR QUEUE (human approval)
    # ══════════════════════════════════════════════════════════

    def _dispatch_suggestions(self, suggestions: list) -> tuple:
        applied, queued = [], []

        for s in suggestions:
            if self.human_approval:
                self._queue_for_approval(s)
                queued.append(s)
            else:
                self._apply_suggestion(s)
                applied.append(s)

        return applied, queued

    def _apply_suggestion(self, s: dict) -> None:
        t = s["type"]
        if t == "PAIR_REMOVE":
            self.config.remove_pair(s["target"], s["reason"])
        elif t == "PATTERN_CONFIDENCE":
            self.rule_updater.apply_rule(
                pattern=s["target"], condition="GENERAL",
                new_confidence=s["new_confidence"], lesson=s["condition"],
                approved_by="auto_optimizer",
            )
        elif t == "SESSION_PREFERENCE":
            self.config.set_session_preference(s["target"], preferred=s["preferred"], avoid=s.get("avoid"))
        elif t == "RISK_ADJUSTMENT":
            self.config.set_risk(s["new_risk"], s["reason"])
        log.info(f"[AutoOptimizer] ⚡ Applied autonomously: {t} | {s['reason']}")

    # ══════════════════════════════════════════════════════════
    # 7. HUMAN APPROVAL MODE  ⭐⭐⭐⭐⭐
    # ══════════════════════════════════════════════════════════

    def _queue_for_approval(self, suggestion: dict) -> dict:
        pending = self._load_json(PENDING_OPTIMIZER_PATH, [])
        suggestion = dict(suggestion)
        suggestion["id"] = f"opt_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{len(pending)+1}"
        suggestion["status"] = "PENDING_APPROVAL"
        suggestion["queued_at"] = datetime.now(timezone.utc).isoformat()
        pending.append(suggestion)
        self._save_json(PENDING_OPTIMIZER_PATH, pending)
        log.info(f"[AutoOptimizer] ⏸️ Queued for approval: {suggestion['id']} | {suggestion['reason']}")
        return suggestion

    def list_pending(self) -> list:
        return self._load_json(PENDING_OPTIMIZER_PATH, [])

    def approve(self, suggestion_id: str) -> dict:
        """Human approve করলে suggestion সরাসরি apply হয়ে যায়।"""
        pending = self._load_json(PENDING_OPTIMIZER_PATH, [])
        match, remaining = None, []
        for s in pending:
            if s["id"] == suggestion_id:
                match = s
            else:
                remaining.append(s)

        if not match:
            return {"success": False, "reason": "Suggestion not found"}

        self._apply_suggestion(match)
        self._save_json(PENDING_OPTIMIZER_PATH, remaining)
        log.info(f"[AutoOptimizer] ✅ APPROVED & applied: {suggestion_id}")
        return {"success": True, "applied": match}

    def reject(self, suggestion_id: str) -> dict:
        pending = self._load_json(PENDING_OPTIMIZER_PATH, [])
        remaining = [s for s in pending if s["id"] != suggestion_id]
        self._save_json(PENDING_OPTIMIZER_PATH, remaining)
        log.info(f"[AutoOptimizer] ❌ REJECTED: {suggestion_id}")
        return {"success": True}

    def print_pending(self) -> None:
        pending = self.list_pending()
        bar = "═" * 64
        print(f"\n{bar}")
        print("  📋  PENDING OPTIMIZER SUGGESTIONS  (Day 55)")
        print(bar)
        if not pending:
            print("  ✅ No pending suggestions")
        for s in pending:
            print(f"\n  [{s['id']}] {s['type']} → {s['target']}")
            print(f"      Reason : {s['reason']}")
            print(f"      Approve: optimizer.approve('{s['id']}')")
            print(f"      Reject : optimizer.reject('{s['id']}')")
        print(bar + "\n")

    # ══════════════════════════════════════════════════════════
    # 8. A/B STRATEGY TESTING  (delegates to DeepMistakeAnalyzer)
    # ══════════════════════════════════════════════════════════

    def ab_test_pattern(self, pattern: str, regime: str) -> dict:
        """Strategy A (no filter) vs Strategy B (filtered) — reuses Day 52 engine."""
        return self.deep_analyzer.run_ab_test(pattern, regime)

    # ══════════════════════════════════════════════════════════
    # 9. MARKET REGIME AUTO ADAPTATION
    # ══════════════════════════════════════════════════════════

    def adapt_to_regime(self, current_regime: str, previous_regime: str) -> dict:
        """
        Market regime পরিবর্তন হলে (Trending → Ranging ইত্যাদি),
        কোন pattern enable/disable করা উচিত তার suggestion দেয়।
        """
        if current_regime == previous_regime:
            return {"changed": False}

        regime_perf = self.feedback.get_regime_performance()
        verdict = regime_perf.get(current_regime, {}).get("verdict", "⚠️ Neutral")

        action = "ENABLE_REGIME_STRATEGY" if "✅" in verdict else "CAUTION_REGIME_STRATEGY"

        result = {
            "changed":         True,
            "from_regime":     previous_regime,
            "to_regime":       current_regime,
            "performance_in_new_regime": regime_perf.get(current_regime, {}),
            "action":          action,
            "explanation": (
                f"Market regime changed {previous_regime} → {current_regime}. "
                f"Historical performance in {current_regime}: {verdict}."
            ),
        }
        log.info(f"[AutoOptimizer] 🔄 Regime adaptation: {result['explanation']}")
        return result

    # ══════════════════════════════════════════════════════════
    # 10. ROLLBACK CHECK  ⭐⭐⭐⭐⭐
    # ══════════════════════════════════════════════════════════

    def _check_rollback(self) -> Optional[dict]:
        """
        সর্বশেষ দুই version compare করে — নতুন version যদি
        ROLLBACK_DEGRADATION_PCT-এর বেশি খারাপ হয়, rollback করো।
        """
        versions = self.config.list_versions()
        if len(versions) < 2:
            return None

        latest, previous = versions[-1], versions[-2]
        latest_perf = latest.get("performance_snapshot", {})
        prev_perf   = previous.get("performance_snapshot", {})

        if latest_perf.get("total_trades", 0) < rules.ROLLBACK_MIN_TRADES:
            return {"checked": True, "rolled_back": False, "reason": "Not enough trades on new version yet"}

        latest_wr = latest_perf.get("win_rate", 0)
        prev_wr   = prev_perf.get("win_rate", 0)
        drop = prev_wr - latest_wr

        if drop >= rules.ROLLBACK_DEGRADATION_PCT:
            result = self.config.rollback_to(previous["label"])
            log.warning(
                f"[AutoOptimizer] ⏮️ Performance dropped {drop:.1f}% "
                f"({prev_wr}% → {latest_wr}%) — rolled back to v{previous['label']}"
            )
            return {
                "checked": True, "rolled_back": True,
                "from_version": latest["label"], "to_version": previous["label"],
                "drop_pct": round(drop, 1),
            }

        return {"checked": True, "rolled_back": False, "drop_pct": round(drop, 1)}

    # ══════════════════════════════════════════════════════════
    # INTERNAL HELPERS
    # ══════════════════════════════════════════════════════════

    def _load_recent_trades(self, days: int) -> list:
        all_trades = self._load_json(FEEDBACK_DB_PATH, [])
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        recent = []
        for t in all_trades:
            ts = t.get("timestamp")
            if not ts:
                continue
            try:
                if datetime.fromisoformat(ts) >= cutoff:
                    recent.append(t)
            except Exception:
                continue
        return recent

    def _aggregate_by_key(self, trades: list, key: str) -> dict:
        stats = {}
        for t in trades:
            k = t.get(key)
            if not k:
                continue
            stats.setdefault(k, {"total": 0, "wins": 0, "gross_profit": 0.0, "gross_loss": 0.0})
            stats[k]["total"] += 1
            pnl = t.get("pnl", 0) or 0
            if t.get("win"):
                stats[k]["wins"] += 1
                stats[k]["gross_profit"] += abs(pnl)
            else:
                stats[k]["gross_loss"] += abs(pnl)

        for k, s in stats.items():
            s["win_rate"] = round(s["wins"] / s["total"] * 100, 1) if s["total"] else 0
            s["profit_factor"] = round(s["gross_profit"] / max(s["gross_loss"], 0.01), 2)

        return stats

    def _win_rate(self, trades: list) -> float:
        if not trades:
            return 50.0
        wins = sum(1 for t in trades if t.get("win"))
        return round(wins / len(trades) * 100, 1)

    def _estimate_drawdown(self, trades: list) -> float:
        """Running PnL থেকে সাধারণ peak-to-trough drawdown % approximate করো।"""
        equity, peak, max_dd = 0.0, 0.0, 0.0
        for t in sorted(trades, key=lambda x: x.get("timestamp", "")):
            equity += t.get("pnl", 0) or 0
            peak = max(peak, equity)
            if peak > 0:
                dd = (peak - equity) / peak * 100
                max_dd = max(max_dd, dd)
        return round(max_dd, 1)

    def _estimate_volatility_factor(self, trades: list, drawdown: float) -> float:
        """
        Trade-level ATR data না থাকলে, drawdown + loss-streak থেকে একটা
        proxy volatility factor বের করো (1.0 = normal)।
        """
        atrs = [t.get("atr") for t in trades if t.get("atr")]
        if atrs:
            avg_atr = sum(atrs) / len(atrs)
            # normalize against a baseline ATR of 0.0035 (EURUSD-ish default)
            factor = round(avg_atr / 0.0035, 2)
        else:
            # fallback proxy: scale with drawdown
            factor = round(1.0 + drawdown / 20, 2)
        return max(0.3, min(3.0, factor))

    def _performance_snapshot(self, trades: list) -> dict:
        return {
            "total_trades": len(trades),
            "win_rate":     self._win_rate(trades),
            "drawdown":     self._estimate_drawdown(trades),
            "captured_at":  datetime.now(timezone.utc).isoformat(),
        }

    def _next_version_label(self) -> str:
        versions = self.config.list_versions()
        if not versions:
            return "1.1"
        try:
            last = float(versions[-1]["label"])
            return f"{last + 0.1:.1f}"
        except Exception:
            return datetime.now(timezone.utc).strftime("%Y%m%d.%H%M")

    def _log_run(self, record: dict) -> None:
        runs = self._load_json(OPTIMIZER_LOG_PATH, [])
        runs.append(record)
        self._save_json(OPTIMIZER_LOG_PATH, runs[-200:])

    def _load_json(self, path: str, default):
        if not os.path.exists(path):
            return default
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default

    def _save_json(self, path: str, data) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)