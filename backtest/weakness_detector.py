"""
backtest/weakness_detector.py — Automatic Weakness Detection Engine

Analyzes backtest results and automatically identifies:
  - Weakest strategies (recommend disable/reduce weight)
  - Best strategies (recommend increase weight)
  - Session performance gaps
  - Pattern strengths/weaknesses
  - Rejection pattern analysis
  - Confidence calibration issues
  - Overall system grade

Outputs actionable recommendations for system improvement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from collections import defaultdict

log = logging.getLogger("weakness_detector")


@dataclass
class Finding:
    """A single weakness or strength finding."""
    severity: str = "INFO"  # CRITICAL, WARNING, INFO, GOOD
    category: str = ""
    title: str = ""
    detail: str = ""
    win_rate: Optional[float] = None
    recommendation: str = ""
    impact_estimate: str = ""

    def to_dict(self):
        return {
            "severity": self.severity,
            "category": self.category,
            "title": self.title,
            "detail": self.detail,
            "win_rate": self.win_rate,
            "recommendation": self.recommendation,
            "impact_estimate": self.impact_estimate,
        }


class WeaknessDetector:
    """Analyzes backtest results and generates actionable recommendations."""

    def __init__(self, min_trades_for_analysis: int = 5):
        self.min_trades = min_trades_for_analysis
        self.findings: List[Finding] = []
        self.suggested_improvements: List[str] = []

    def analyze(self, report_data) -> Dict[str, Any]:
        """
        Run full weakness analysis on ReportData.

        Args:
            report_data: ReportData object from report_generator

        Returns:
            Dict with findings, suggested_improvements, overall_grade
        """
        self.findings = []
        self.suggested_improvements = []

        self._analyze_strategy_quality(report_data)
        self._analyze_session_quality(report_data)
        self._analyze_pattern_quality(report_data)
        self._analyze_confidence_calibration(report_data)
        self._analyze_rejection_patterns(report_data)
        self._analyze_risk_quality(report_data)
        self._analyze_overall_stats(report_data)

        grade = self._compute_grade(report_data)

        return {
            "findings": [f.to_dict() for f in self.findings],
            "suggested_improvements": self.suggested_improvements,
            "overall_grade": grade,
        }

    # ----------------------------------------------------------
    # Strategy Analysis
    # ----------------------------------------------------------

    def _analyze_strategy_quality(self, data):
        strategies = data.strategy_breakdown
        if not strategies:
            return

        sorted_strats = sorted(strategies.items(), key=lambda x: x[1].get("win_rate", 0))

        # Weakest strategy
        for name, d in sorted_strats:
            if d.get("trades", 0) < self.min_trades:
                continue
            wr = d.get("win_rate", 0)
            if wr < 40:
                self.findings.append(Finding(
                    severity="CRITICAL" if wr < 35 else "WARNING",
                    category="Strategy",
                    title=f"Weakest Strategy: {name}",
                    detail=f"{d.get('trades', 0)} trades, WR={wr:.1f}%, P&L=${d.get('pnl_usd', 0):.2f}",
                    win_rate=wr,
                    recommendation="Disable" if wr < 35 else "Reduce weight to 0.5x",
                    impact_estimate=f"System WR would improve by ~{(50-wr)*d.get('trades',0)/max(data.summary.total_trades,1)*0.3:.1f}% points",
                ))
                if wr < 35:
                    self.suggested_improvements.append(f"Disable {name} strategy (WR={wr:.1f}%)")
                else:
                    self.suggested_improvements.append(f"Reduce {name} strategy weight (WR={wr:.1f}%)")

        # Best strategy
        for name, d in sorted(strategies.items(), key=lambda x: x[1].get("win_rate", 0), reverse=True):
            if d.get("trades", 0) < self.min_trades:
                continue
            wr = d.get("win_rate", 0)
            if wr > 60:
                self.findings.append(Finding(
                    severity="GOOD",
                    category="Strategy",
                    title=f"Best Strategy: {name}",
                    detail=f"{d.get('trades', 0)} trades, WR={wr:.1f}%, P&L=${d.get('pnl_usd', 0):.2f}",
                    win_rate=wr,
                    recommendation="Increase weight to 1.5x",
                ))
                self.suggested_improvements.append(f"Increase {name} strategy weight (WR={wr:.1f}%)")

    # ----------------------------------------------------------
    # Session Analysis
    # ----------------------------------------------------------

    def _analyze_session_quality(self, data):
        sessions = data.session_breakdown
        if not sessions:
            return

        best_sess = max(sessions.items(), key=lambda x: x[1].get("win_rate", 0))
        worst_sess = min(sessions.items(), key=lambda x: x[1].get("win_rate", 0))

        if worst_sess[1].get("trades", 0) >= self.min_trades:
            wr = worst_sess[1].get("win_rate", 0)
            if wr < 45:
                self.findings.append(Finding(
                    severity="WARNING",
                    category="Session",
                    title=f"Worst Session: {worst_sess[0]}",
                    detail=f"WR={wr:.1f}%, {worst_sess[1].get('trades', 0)} trades",
                    win_rate=wr,
                    recommendation=f"Reduce or skip trading during {worst_sess[0]} session",
                ))
                self.suggested_improvements.append(f"Remove/reduce trades during {worst_sess[0]} session")

        if best_sess[1].get("trades", 0) >= self.min_trades:
            wr = best_sess[1].get("win_rate", 0)
            if wr > 55:
                self.findings.append(Finding(
                    severity="GOOD",
                    category="Session",
                    title=f"Best Session: {best_sess[0]}",
                    detail=f"WR={wr:.1f}%, {best_sess[1].get('trades', 0)} trades",
                    win_rate=wr,
                ))

    # ----------------------------------------------------------
    # Pattern Analysis
    # ----------------------------------------------------------

    def _analyze_pattern_quality(self, data):
        patterns = data.pattern_breakdown
        if not patterns:
            return

        best_pat = max(patterns.items(), key=lambda x: x[1].get("pnl_usd", 0))
        worst_pat = min(patterns.items(), key=lambda x: x[1].get("pnl_usd", 0))

        if best_pat[1].get("trades", 0) >= self.min_trades and best_pat[1].get("pnl_usd", 0) > 0:
            self.findings.append(Finding(
                severity="GOOD",
                category="Pattern",
                title=f"Most Profitable Pattern: {best_pat[0]}",
                detail=f"P&L=${best_pat[1].get('pnl_usd', 0):.2f}, WR={best_pat[1].get('win_rate', 0):.1f}%",
                recommendation="Prioritize this pattern",
            ))

        if worst_pat[1].get("trades", 0) >= self.min_trades and worst_pat[1].get("pnl_usd", 0) < 0:
            self.findings.append(Finding(
                severity="WARNING",
                category="Pattern",
                title=f"Least Profitable Pattern: {worst_pat[0]}",
                detail=f"P&L=${worst_pat[1].get('pnl_usd', 0):.2f}, WR={worst_pat[1].get('win_rate', 0):.1f}%",
                recommendation="Review entry conditions or disable",
            ))

    # ----------------------------------------------------------
    # Confidence Calibration
    # ----------------------------------------------------------

    def _analyze_confidence_calibration(self, data):
        cal = data.confidence_calibration
        if not cal:
            return

        # Check if high confidence actually means high win rate
        high_conf = cal.get("90-100", {})
        mid_conf = cal.get("70-80", {})
        low_conf = cal.get("Below 50", {})

        if high_conf.get("trades", 0) >= self.min_trades:
            actual_wr = high_conf.get("actual_win_pct", 0)
            if actual_wr < 75:
                self.findings.append(Finding(
                    severity="WARNING",
                    category="Calibration",
                    title="Confidence Miscalibrated (High)",
                    detail=f"90-100% confidence trades only win {actual_wr:.1f}% of the time",
                    win_rate=actual_wr,
                    recommendation="Recalibrate confidence model — high confidence should predict ~85%+ wins",
                ))
                self.suggested_improvements.append("Recalibrate confidence model (90%+ confidence -> {actual_wr:.0f}% actual)")

        if low_conf.get("trades", 0) >= self.min_trades:
            actual_wr = low_conf.get("actual_win_pct", 0)
            if actual_wr > 55:
                self.findings.append(Finding(
                    severity="INFO",
                    category="Calibration",
                    title="Low Confidence Outperforms Expected",
                    detail=f"Below 50% confidence trades win {actual_wr:.1f}% — may be too conservative",
                    win_rate=actual_wr,
                    recommendation="Consider lowering confidence threshold",
                ))

    # ----------------------------------------------------------
    # Rejection Pattern Analysis
    # ----------------------------------------------------------

    def _analyze_rejection_patterns(self, data):
        rej = data.rejection_stats
        if not rej:
            return

        total = sum(rej.values())
        if total == 0:
            return

        # Find dominant rejection reason
        top_reason = max(rej.items(), key=lambda x: x[1])
        top_pct = top_reason[1] / total * 100

        if top_pct > 50:
            self.findings.append(Finding(
                severity="INFO" if top_reason[0] == "WAIT" else "WARNING",
                category="Rejection",
                title=f"Most Rejection Reason: {top_reason[0]}",
                detail=f"{top_reason[1]} rejections ({top_pct:.1f}% of all rejections)",
                recommendation=f"Investigate why {top_reason[0]} dominates — may indicate overly strict gating",
            ))

        # Check if too many are rejected
        total_trades = data.summary.total_trades
        if total_trades + total > 0:
            acceptance = total_trades / (total_trades + total) * 100
            if acceptance < 5:
                self.findings.append(Finding(
                    severity="CRITICAL",
                    category="Rejection",
                    title="Extremely Low Acceptance Rate",
                    detail=f"Only {acceptance:.1f}% of decisions result in trades ({total_trades}/{total_trades+total})",
                    recommendation="Lower confidence threshold or relax permission gates",
                ))
                self.suggested_improvements.append(f"Raise acceptance rate (currently {acceptance:.1f}%)")
            elif acceptance < 15:
                self.findings.append(Finding(
                    severity="WARNING",
                    category="Rejection",
                    title="Low Acceptance Rate",
                    detail=f"Only {acceptance:.1f}% of decisions result in trades",
                ))

    # ----------------------------------------------------------
    # Risk Quality
    # ----------------------------------------------------------

    def _analyze_risk_quality(self, data):
        r = data.risk_report
        if not r:
            return

        if r.get("avg_rr", 0) < 1.5:
            self.findings.append(Finding(
                severity="WARNING",
                category="Risk",
                title="Low Average Risk:Reward",
                detail=f"Average R:R is 1:{r.get('avg_rr', 0):.2f} (target: 1:2.0+)",
                recommendation="Increase minimum R:R requirement in RiskEngine",
            ))

        if r.get("risk_utilization", 0) > 80:
            self.findings.append(Finding(
                severity="WARNING",
                category="Risk",
                title="High Risk Utilization",
                detail=f"Average position sizing uses {r.get('risk_utilization', 0):.1f}% of max lot",
                recommendation="Reduce position sizing to maintain safety buffer",
            ))

    # ----------------------------------------------------------
    # Overall Stats
    # ----------------------------------------------------------

    def _analyze_overall_stats(self, data):
        s = data.summary

        # Max drawdown check
        if s.max_drawdown_pct > 30:
            self.findings.append(Finding(
                severity="CRITICAL",
                category="Drawdown",
                title="Extreme Maximum Drawdown",
                detail=f"Max DD: {s.max_drawdown_pct:.1f}% (${s.max_drawdown_usd:,.2f})",
                recommendation="Reduce risk per trade or add circuit breaker",
            ))
            self.suggested_improvements.append(f"Reduce risk to limit drawdown (current max: {s.max_drawdown_pct:.1f}%)")
        elif s.max_drawdown_pct > 20:
            self.findings.append(Finding(
                severity="WARNING",
                category="Drawdown",
                title="High Maximum Drawdown",
                detail=f"Max DD: {s.max_drawdown_pct:.1f}%",
            ))

        # Sharpe check
        if s.sharpe < 0:
            self.findings.append(Finding(
                severity="CRITICAL",
                category="Performance",
                title="Negative Sharpe Ratio",
                detail=f"Sharpe: {s.sharpe:.2f} — system is losing money on a risk-adjusted basis",
            ))
        elif s.sharpe < 1.0 and s.total_trades > 30:
            self.findings.append(Finding(
                severity="WARNING",
                category="Performance",
                title="Low Sharpe Ratio",
                detail=f"Sharpe: {s.sharpe:.2f} — below institutional threshold of 1.0",
            ))

        # Profit factor check
        if s.profit_factor < 1.0 and s.total_trades > 20:
            self.findings.append(Finding(
                severity="CRITICAL",
                category="Performance",
                title="Profit Factor Below 1.0",
                detail=f"PF: {s.profit_factor:.2f} — system is net losing",
            ))

        # Confidence threshold recommendation
        if hasattr(data, 'confidence_calibration') and data.confidence_calibration:
            high = data.confidence_calibration.get("70-80", {})
            if high.get("actual_win_pct", 0) > 60:
                self.suggested_improvements.append("Raise confidence threshold to 75%")

    # ----------------------------------------------------------
    # Grade Computation
    # ----------------------------------------------------------

    def _compute_grade(self, data) -> str:
        """Compute an overall system grade (A+ to F)."""
        s = data.summary
        score = 0.0

        # Win rate scoring (0-25 points)
        if s.win_rate > 60: score += 25
        elif s.win_rate > 55: score += 20
        elif s.win_rate > 50: score += 15
        elif s.win_rate > 45: score += 10
        elif s.win_rate > 40: score += 5

        # Profit factor scoring (0-25 points)
        if s.profit_factor > 2.0: score += 25
        elif s.profit_factor > 1.5: score += 20
        elif s.profit_factor > 1.2: score += 15
        elif s.profit_factor > 1.0: score += 10
        elif s.profit_factor > 0.8: score += 5

        # Sharpe scoring (0-20 points)
        if s.sharpe > 2.0: score += 20
        elif s.sharpe > 1.5: score += 15
        elif s.sharpe > 1.0: score += 12
        elif s.sharpe > 0.5: score += 8
        elif s.sharpe > 0: score += 4

        # Drawdown scoring (0-15 points)
        if s.max_drawdown_pct < 5: score += 15
        elif s.max_drawdown_pct < 10: score += 12
        elif s.max_drawdown_pct < 15: score += 8
        elif s.max_drawdown_pct < 20: score += 5
        elif s.max_drawdown_pct < 30: score += 2

        # Recovery factor scoring (0-15 points)
        if s.recovery_factor > 5.0: score += 15
        elif s.recovery_factor > 3.0: score += 12
        elif s.recovery_factor > 2.0: score += 8
        elif s.recovery_factor > 1.0: score += 5
        elif s.recovery_factor > 0: score += 2

        # Grade mapping
        if score >= 90: return "A+"
        elif score >= 80: return "A"
        elif score >= 70: return "B+"
        elif score >= 60: return "B"
        elif score >= 50: return "C+"
        elif score >= 40: return "C"
        elif score >= 30: return "D"
        else: return "F"
