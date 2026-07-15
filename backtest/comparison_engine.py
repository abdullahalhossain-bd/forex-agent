"""
backtest/comparison_engine.py — System Comparison Engine (Day 74)
===================================================================

Compares the 3 system versions (rule_only, rule_intel, full_ai) and
determines:
  1. Which system performed best overall
  2. Whether ML actually improved performance (ML Improvement Detector)
  3. If ML did NOT improve — diagnosis of why
  4. Champion model selection

Output:
    {
        "winner": "full_ai",
        "ml_improved": True,
        "improvement_pct": 11.0,
        "diagnosis": [],
        "champion": "full_ai",
    }
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from utils.logger import get_logger
try:
    from backtest.ml_backtest import BacktestMetrics
except ImportError:
    # Dummy class so code doesn't crash if ml_backtest module is unavailable
    class BacktestMetrics:
        def __init__(self, **kwargs):
            self.system = kwargs.get("system", "")
            self.profit_factor = 0.0
            self.win_rate = 0.0
        def to_dict(self): return {}

log = get_logger("comparison_engine")


@dataclass
class ComparisonResult:
    """Result of comparing 3 system versions."""
    winner: str = ""
    ml_improved: bool = False
    improvement_pct: float = 0.0
    improvement_pf: float = 0.0
    diagnosis: List[str] = field(default_factory=list)
    champion: str = ""
    recommendation: str = ""
    metrics_summary: Dict[str, Dict] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ComparisonEngine:
    """Compares system versions and detects ML improvement."""

    # Minimum improvement required to justify ML complexity
    MIN_IMPROVEMENT_PF = 0.1   # ML PF must be at least 0.1 higher
    MIN_IMPROVEMENT_WR = 3.0   # Win rate must improve by at least 3%

    def compare(self, results: Dict[str, BacktestMetrics]) -> ComparisonResult:
        """Compare backtest results from 3 system versions.

        Args:
            results: {"rule_only": metrics, "rule_intel": metrics, "full_ai": metrics}

        Returns:
            ComparisonResult with winner + ML improvement analysis.
        """
        comp = ComparisonResult()

        if not results:
            comp.recommendation = "No results to compare"
            return comp

        # Build summary
        for name, metrics in results.items():
            comp.metrics_summary[name] = metrics.to_dict()

        # Determine winner by profit factor (primary metric)
        best_pf = -1
        winner = ""
        for name, metrics in results.items():
            if metrics.profit_factor > best_pf:
                best_pf = metrics.profit_factor
                winner = name
        comp.winner = winner
        comp.champion = winner

        # ML Improvement Detection
        rule_pf = results.get("rule_only", BacktestMetrics(system="rule_only")).profit_factor
        rule_wr = results.get("rule_only", BacktestMetrics(system="rule_only")).win_rate
        ai_pf = results.get("full_ai", BacktestMetrics(system="full_ai")).profit_factor
        ai_wr = results.get("full_ai", BacktestMetrics(system="full_ai")).win_rate

        comp.improvement_pct = ai_wr - rule_wr
        comp.improvement_pf = ai_pf - rule_pf

        if comp.improvement_pf >= self.MIN_IMPROVEMENT_PF and comp.improvement_pct >= self.MIN_IMPROVEMENT_WR:
            comp.ml_improved = True
            comp.recommendation = f"ML APPROVED — improvement: +{comp.improvement_pct:.1f}% WR, +{comp.improvement_pf:.2f} PF"
        else:
            comp.ml_improved = False
            comp.recommendation = f"ML NOT APPROVED — improvement too small: +{comp.improvement_pct:.1f}% WR, +{comp.improvement_pf:.2f} PF"
            comp.diagnosis = self._diagnose(results)

        log.info(
            f"[Comparison] Winner: {winner} | ML improved: {comp.ml_improved} | "
            f"+{comp.improvement_pct:.1f}% WR | +{comp.improvement_pf:.2f} PF | "
            f"{comp.recommendation}"
        )
        return comp

    def _diagnose(self, results: Dict[str, BacktestMetrics]) -> List[str]:
        """Diagnose why ML is not improving."""
        issues: List[str] = []
        rule = results.get("rule_only")
        ai = results.get("full_ai")

        if not rule or not ai:
            return ["Missing system results for diagnosis"]

        if ai.total_trades < 20:
            issues.append("Insufficient trades — ML may need more data or different thresholds")

        if ai.profit_factor < rule.profit_factor:
            issues.append("ML profit factor LOWER than rule-based — model may be overfit or features are noisy")

        if ai.max_drawdown_pct > rule.max_drawdown_pct:
            issues.append("ML drawdown HIGHER than rule-based — risk management needs adjustment")

        if ai.win_rate < rule.win_rate:
            issues.append("ML win rate LOWER than rule-based — feature quality or model parameters need tuning")

        if not issues:
            issues.append("ML shows marginal improvement — consider hyperparameter tuning or feature engineering")

        return issues


# ── Singleton ───────────────────────────────────────────────────────

_ENGINE: Optional[ComparisonEngine] = None


def get_comparison_engine() -> ComparisonEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = ComparisonEngine()
    return _ENGINE
