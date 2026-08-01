# research/research_report.py — Day 57 | Research Report Generator
# ============================================================
# Weekly AI Research Report generation.
# Summarizes all experiments, discoveries, and recommendations.
# ============================================================

import json
from datetime import datetime, timezone
from pathlib import Path

from utils.logger import get_logger
from core.constants import MEMORY_DIR, REPORTS_DIR

log = get_logger("research.report")

RESEARCH_REPORTS_DIR = REPORTS_DIR / "research"
RESEARCH_REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class ResearchReportGenerator:
    """
    Generates research reports for the AI Trading System.

    Usage:
        gen = ResearchReportGenerator()
        report = gen.generate_weekly(
            experiment_results=[...],
            hypothesis_results=[...],
            market_findings=[...],
        )
        gen.save_report(report)
        gen.print_report(report)
    """

    def generate_weekly(
        self,
        experiment_results: list[dict] = None,
        hypothesis_results: list[dict] = None,
        market_findings: list[dict] = None,
        strategies_approved: list[str] = None,
        strategies_rejected: list[str] = None,
    ) -> dict:
        """
        Generate a comprehensive weekly research report.

        Args:
            experiment_results: List of experiment result dicts
            hypothesis_results: List of hypothesis evaluation results
            market_findings: List of market behavior research findings
            strategies_approved: List of strategy names approved
            strategies_rejected: List of strategy names rejected

        Returns:
            Complete research report dict.
        """
        now = datetime.now(timezone.utc)
        experiment_results = experiment_results or []
        hypothesis_results = hypothesis_results or []
        market_findings = market_findings or []
        strategies_approved = strategies_approved or []
        strategies_rejected = strategies_rejected or []

        # Aggregate stats
        total_experiments = len(experiment_results)
        approved = sum(1 for e in experiment_results if e.get("status") == "APPROVED")
        rejected = sum(1 for e in experiment_results if e.get("status") == "REJECTED")

        total_hypotheses = len(hypothesis_results)
        confirmed = sum(1 for h in hypothesis_results if h.get("verdict") == "CONFIRMED")
        h_rejected = sum(1 for h in hypothesis_results if h.get("verdict") == "REJECTED")
        inconclusive = sum(1 for h in hypothesis_results if h.get("verdict") == "INCONCLUSIVE")

        # Best discovery
        best_discovery = self._find_best_discovery(experiment_results)

        # Best hypothesis insight
        best_hypothesis = self._find_best_hypothesis(hypothesis_results)

        # Key market findings
        key_findings = self._summarize_findings(market_findings)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            experiment_results, hypothesis_results, market_findings
        )

        report = {
            "report_type": "weekly_research",
            "generated_at": now.isoformat(timespec="seconds"),
            "period": {
                "start": now.replace(hour=0, minute=0, second=0).isoformat(),
                "end": now.isoformat(),
            },
            "summary": {
                "total_experiments": total_experiments,
                "experiments_approved": approved,
                "experiments_rejected": rejected,
                "approval_rate": round(approved / total_experiments * 100, 1) if total_experiments > 0 else 0,
                "total_hypotheses": total_hypotheses,
                "hypotheses_confirmed": confirmed,
                "hypotheses_rejected": h_rejected,
                "hypotheses_inconclusive": inconclusive,
            },
            "best_discovery": best_discovery,
            "best_hypothesis_insight": best_hypothesis,
            "market_findings": key_findings,
            "strategies_approved": strategies_approved,
            "strategies_rejected": strategies_rejected,
            "recommendations": recommendations,
            "experiment_details": experiment_results,
            "hypothesis_details": hypothesis_results,
        }

        log.info(
            f"[ResearchReport] Weekly report generated: "
            f"{total_experiments} experiments, {approved} approved, "
            f"{total_hypotheses} hypotheses, {confirmed} confirmed"
        )
        return report

    def generate_single_experiment_report(self, experiment_result: dict) -> dict:
        """Generate a detailed report for a single experiment."""
        now = datetime.now(timezone.utc)

        return {
            "report_type": "single_experiment",
            "generated_at": now.isoformat(timespec="seconds"),
            "experiment": experiment_result,
            "analysis": {
                "strengths": self._analyze_strengths(experiment_result),
                "weaknesses": self._analyze_weaknesses(experiment_result),
                "suggestions": self._generate_experiment_suggestions(experiment_result),
            },
        }

    # ═══════════════════════════════════════════════════════
    # REPORT OUTPUT
    # ═══════════════════════════════════════════════════════

    def save_report(self, report: dict, filename: str = None) -> str:
        """Save report to JSON file."""
        if filename is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            report_type = report.get("report_type", "research")
            filename = f"research_report_{report_type}_{ts}.json"

        filepath = RESEARCH_REPORTS_DIR / filename
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)

        log.info(f"[ResearchReport] Saved to {filepath}")
        return str(filepath)

    def print_report(self, report: dict) -> None:
        """Print a formatted research report to console."""
        bar = "=" * 56
        sep = "-" * 56

        print()
        print(bar)
        print("  RESEARCH REPORT — AI TRADING SYSTEM (Day 57)")
        print(bar)
        print(f"  Generated : {report['generated_at']}")
        print(f"  Type      : {report['report_type'].upper()}")
        print()

        # Summary
        summary = report.get("summary", {})
        print(f"  EXPERIMENTS")
        print(sep)
        print(f"  Total         : {summary.get('total_experiments', 0)}")
        print(f"  Approved      : {summary.get('experiments_approved', 0)}")
        print(f"  Rejected      : {summary.get('experiments_rejected', 0)}")
        print(f"  Approval Rate : {summary.get('approval_rate', 0)}%")
        print()

        print(f"  HYPOTHESES")
        print(sep)
        print(f"  Total     : {summary.get('total_hypotheses', 0)}")
        print(f"  Confirmed : {summary.get('hypotheses_confirmed', 0)}")
        print(f"  Rejected  : {summary.get('hypotheses_rejected', 0)}")
        print(f"  Inconcl.  : {summary.get('hypotheses_inconclusive', 0)}")
        print()

        # Best discovery
        best = report.get("best_discovery", {})
        if best:
            print("  BEST DISCOVERY")
            print(sep)
            print(f"  Strategy  : {best.get('strategy_name', 'N/A')}")
            print(f"  Pair      : {best.get('pair', 'N/A')}")
            print(f"  Win Rate  : {best.get('win_rate', 'N/A')}%")
            print(f"  Profit F. : {best.get('profit_factor', 'N/A')}")
            print(f"  Max DD    : {best.get('max_drawdown', 'N/A')}%")
            print()

        # Market findings
        findings = report.get("market_findings", [])
        if findings:
            print("  MARKET FINDINGS")
            print(sep)
            for i, finding in enumerate(findings[:5], 1):
                print(f"  {i}. {finding.get('finding', 'N/A')[:60]}")
            print()

        # Recommendations
        recs = report.get("recommendations", [])
        if recs:
            print("  RECOMMENDATIONS")
            print(sep)
            for i, rec in enumerate(recs[:5], 1):
                print(f"  {i}. {rec[:70]}")
            print()

        # Approved strategies
        approved = report.get("strategies_approved", [])
        if approved:
            print("  APPROVED STRATEGIES")
            print(sep)
            for s in approved:
                print(f"    + {s}")
            print()

        print(bar)
        print()

    # ═══════════════════════════════════════════════════════
    # INTERNAL ANALYSIS HELPERS
    # ═══════════════════════════════════════════════════════

    def _find_best_discovery(self, experiments: list[dict]) -> dict:
        """Find the best performing experiment."""
        best = None
        best_score = 0

        for exp in experiments:
            bt = exp.get("backtest_result", {})
            summary = bt.get("summary", {}) if bt else {}
            score = (
                summary.get("profit_factor", 0) * 0.4 +
                summary.get("win_rate", 0) * 0.01 * 0.3 +
                (100 - summary.get("max_drawdown", 100)) * 0.01 * 0.3
            )

            if score > best_score:
                best_score = score
                best = {
                    "strategy_name": exp.get("strategy_name", "N/A"),
                    "experiment_id": exp.get("id", "N/A"),
                    "pair": exp.get("pair", "N/A"),
                    "timeframe": exp.get("timeframe", "N/A"),
                    "win_rate": summary.get("win_rate", 0),
                    "profit_factor": summary.get("profit_factor", 0),
                    "max_drawdown": summary.get("max_drawdown", 0),
                    "average_rr": summary.get("average_rr", 0),
                    "trades": summary.get("trades", 0),
                    "score": round(best_score, 2),
                }

        return best or {}

    def _find_best_hypothesis(self, hypotheses: list[dict]) -> dict:
        """Find the most impactful confirmed hypothesis."""
        best = None
        best_improvement = 0

        for h in hypotheses:
            improvement = abs(h.get("improvement_pct", 0))
            if h.get("verdict") == "CONFIRMED" and improvement > best_improvement:
                best_improvement = improvement
                best = {
                    "question": h.get("question", h.get("hypothesis_question", "N/A")),
                    "improvement_pct": h.get("improvement_pct", 0),
                    "verdict": h.get("verdict", "N/A"),
                    "confidence": h.get("confidence", 0),
                }

        return best or {}

    def _summarize_findings(self, findings: list[dict]) -> list[dict]:
        """Summarize market behavior findings."""
        summarized = []
        for f in findings:
            summarized.append({
                "pair": f.get("pair", "N/A"),
                "session": f.get("session", "N/A"),
                "finding": f.get("finding", "No significant finding")[:80],
                "recommendation": f.get("recommendation", "")[:80],
            })
        return summarized

    def _generate_recommendations(
        self,
        experiments: list[dict],
        hypotheses: list[dict],
        findings: list[dict],
    ) -> list[str]:
        """Generate actionable recommendations based on research results."""
        recommendations = []

        # From experiments
        approved = [e for e in experiments if e.get("status") == "APPROVED"]
        if approved:
            best = max(approved, key=lambda x: x.get("backtest_result", {}).get("summary", {}).get("profit_factor", 0))
            name = best.get("strategy_name", "the approved strategy")
            recommendations.append(
                f"Consider deploying '{name}' to paper trading for live validation."
            )

        rejected = [e for e in experiments if e.get("status") == "REJECTED"]
        if len(rejected) > len(approved):
            recommendations.append(
                "High rejection rate detected — consider refining strategy generation parameters."
            )

        # From hypotheses
        confirmed_h = [h for h in hypotheses if h.get("verdict") == "CONFIRMED"]
        if confirmed_h:
            best_h = max(confirmed_h, key=lambda x: abs(x.get("improvement_pct", 0)))
            recommendations.append(
                f"Confirmed hypothesis: {best_h.get('improvement_pct', 0):+.1f}% improvement "
                f"— apply to active strategies."
            )

        # From market findings
        for f in findings:
            rec = f.get("recommendation", "")
            if rec and "Consider" in rec:
                recommendations.append(rec)

        if not recommendations:
            recommendations.append("Continue research cycles to gather more data.")

        return recommendations

    def _analyze_strengths(self, experiment: dict) -> list[str]:
        """Analyze strengths of an experiment result."""
        strengths = []
        bt = experiment.get("backtest_result", {})
        summary = bt.get("summary", {}) if bt else {}

        if summary.get("profit_factor", 0) > 2.0:
            strengths.append(f"Strong profit factor ({summary['profit_factor']:.2f})")
        if summary.get("win_rate", 0) > 60:
            strengths.append(f"Good win rate ({summary['win_rate']:.1f}%)")
        if summary.get("max_drawdown", 100) < 10:
            strengths.append(f"Low drawdown ({summary['max_drawdown']:.1f}%)")
        if summary.get("average_rr", 0) > 2.0:
            strengths.append(f"Good risk-reward ratio (1:{summary['average_rr']:.1f})")

        return strengths or ["No significant strengths identified"]

    def _analyze_weaknesses(self, experiment: dict) -> list[str]:
        """Analyze weaknesses of an experiment result."""
        weaknesses = []
        bt = experiment.get("backtest_result", {})
        summary = bt.get("summary", {}) if bt else {}

        if summary.get("profit_factor", 0) < 1.3:
            weaknesses.append(f"Weak profit factor ({summary['profit_factor']:.2f})")
        if summary.get("win_rate", 0) < 45:
            weaknesses.append(f"Low win rate ({summary['win_rate']:.1f}%)")
        if summary.get("max_drawdown", 0) > 15:
            weaknesses.append(f"High drawdown ({summary['max_drawdown']:.1f}%)")
        if summary.get("trades", 0) < 200:
            weaknesses.append(f"Insufficient sample size ({summary['trades']})")

        return weaknesses or ["No significant weaknesses identified"]

    def _generate_experiment_suggestions(self, experiment: dict) -> list[str]:
        """Generate suggestions for improving a strategy based on experiment results."""
        suggestions = []
        bt = experiment.get("backtest_result", {})
        summary = bt.get("summary", {}) if bt else {}

        if summary.get("win_rate", 0) < 50:
            suggestions.append("Add more entry confirmation filters to improve win rate.")
        if summary.get("max_drawdown", 0) > 12:
            suggestions.append("Tighten stop loss or reduce position size to limit drawdown.")
        if summary.get("average_rr", 0) < 1.5:
            suggestions.append("Look for better entry points or extend take-profit targets.")
        if summary.get("trades", 0) < 200:
            suggestions.append("Test on a longer time period or add more pairs.")

        return suggestions or ["Strategy shows balanced performance — monitor for optimization opportunities."]
