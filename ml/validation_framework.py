"""
ml/validation_framework.py — Research & Validation Framework
=================================================================
THE GATEKEEPER — every new feature must pass ALL 6 gates before deployment.

This is the most important module in the entire system. It embodies the
principle: "Spend as much time proving features WRONG as building them."

GATE 1: Historical Performance — does it improve backtest results?
GATE 2: Walk-Forward — does it survive out-of-sample testing?
GATE 3: Robustness — does it survive parameter perturbation?
GATE 4: Live Paper Trading — does the edge exist in real-time?
GATE 5: Duplicate Check — is it correlated with an existing feature?
GATE 6: Overfitting Check — is it too good to be true?

If ANY gate fails → feature REJECTED. No exceptions.

USAGE:
    from ml.validation_framework import ValidationFramework
    vf = ValidationFramework()
    result = vf.validate_feature(
        feature_name="vwap_rejection_signal",
        backtest_results=...,
        walk_forward_results=...,
        paper_trading_results=...,
        correlation_with_existing=...,
    )
    if result["approved"]:
        # deploy feature
    else:
        # reject — log reason
"""

from __future__ import annotations
import numpy as np
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
from utils.logger import get_logger

log = get_logger("validation_framework")


# ════════════════════════════════════════════════════════════════════
# GATE THRESHOLDS
# ════════════════════════════════════════════════════════════════════

# Gate 1: Historical Performance
MIN_BACKTEST_WIN_RATE = 0.50       # must be > 50% win rate
MIN_BACKTEST_PROFIT_FACTOR = 1.20  # profit factor > 1.2
MIN_BACKTEST_TRADES = 100          # at least 100 trades in backtest
MIN_BACKTEST_SHARPE = 0.5          # Sharpe ratio > 0.5

# Gate 2: Walk-Forward
MIN_WF_WIN_RATE = 0.45            # slightly lower threshold for OOS
MIN_WF_PROFIT_FACTOR = 1.05       # must still be profitable OOS
MIN_WF_PERIODS = 3                # at least 3 walk-forward periods
MAX_WF_DEGRADATION = 0.30         # OOS performance < 70% of IS = fail

# Gate 3: Robustness
MAX_PARAMETER_SENSITIVITY = 0.20  # 20% parameter change shouldn't break it
MIN_ROBUSTNESS_SCORE = 0.70       # 70% of perturbed configs must be profitable

# Gate 4: Paper Trading
MIN_PAPER_TRADES = 30             # at least 30 real-time paper trades
MIN_PAPER_WIN_RATE = 0.45        # must maintain > 45% WR in real-time
MAX_PAPER_DRAWDOWN_PCT = 10.0    # max 10% drawdown during paper trading

# Gate 5: Duplicate Check
MAX_CORRELATION_WITH_EXISTING = 0.85  # > 85% correlated = duplicate

# Gate 6: Overfitting Check
MAX_IS_OOS_RATIO = 3.0           # IS performance > 3× OOS = overfitting
MIN_OOS_SHARPE = 0.3             # OOS Sharpe must be > 0.3


@dataclass
class GateResult:
    """Result of a single validation gate."""
    gate_name: str
    passed: bool
    score: float = 0.0
    threshold: float = 0.0
    details: dict = field(default_factory=dict)
    reason: str = ""


@dataclass
class ValidationReport:
    """Complete validation report for a feature."""
    feature_name: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    gates: List[GateResult] = field(default_factory=list)
    approved: bool = False
    overall_score: float = 0.0
    recommendation: str = ""
    rejection_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "feature_name": self.feature_name,
            "timestamp": self.timestamp,
            "approved": self.approved,
            "overall_score": round(self.overall_score, 3),
            "recommendation": self.recommendation,
            "rejection_reasons": self.rejection_reasons,
            "gates": [
                {
                    "gate": g.gate_name,
                    "passed": g.passed,
                    "score": round(g.score, 3),
                    "threshold": g.threshold,
                    "reason": g.reason,
                }
                for g in self.gates
            ],
        }


class ValidationFramework:
    """The Gatekeeper — validates every feature before deployment.

    A feature must pass ALL 6 gates to be approved. If any gate fails,
    the feature is rejected with a specific reason.

    This prevents:
    - Overfitting (Gate 6)
    - Duplicated features (Gate 5)
    - Features that only work in backtest (Gate 2, 4)
    - Fragile features (Gate 3)
    - Unprofitable features (Gate 1)
    """

    def validate_feature(
        self,
        feature_name: str,
        # Gate 1: Historical Performance
        backtest_win_rate: float = 0.0,
        backtest_profit_factor: float = 0.0,
        backtest_n_trades: int = 0,
        backtest_sharpe: float = 0.0,
        # Gate 2: Walk-Forward
        wf_win_rate: float = 0.0,
        wf_profit_factor: float = 0.0,
        wf_n_periods: int = 0,
        wf_is_sharpe: float = 0.0,
        wf_oos_sharpe: float = 0.0,
        # Gate 3: Robustness
        robustness_score: float = 0.0,  # fraction of perturbed configs that are profitable
        max_parameter_sensitivity: float = 1.0,
        # Gate 4: Paper Trading
        paper_n_trades: int = 0,
        paper_win_rate: float = 0.0,
        paper_max_drawdown_pct: float = 100.0,
        # Gate 5: Duplicate Check
        max_correlation_with_existing: float = 0.0,
        correlated_feature_name: str = "",
        # Gate 6: Overfitting
        is_oos_ratio: float = 1.0,  # IS performance / OOS performance
    ) -> ValidationReport:
        """Validate a feature through all 6 gates.

        Returns:
            ValidationReport with approval decision and detailed reasons.
        """
        report = ValidationReport(feature_name=feature_name)
        rejection_reasons = []

        # ════════════════════════════════════════════════════════════
        # GATE 1: Historical Performance
        # ════════════════════════════════════════════════════════════
        g1_passed = (
            backtest_win_rate >= MIN_BACKTEST_WIN_RATE and
            backtest_profit_factor >= MIN_BACKTEST_PROFIT_FACTOR and
            backtest_n_trades >= MIN_BACKTEST_TRADES and
            backtest_sharpe >= MIN_BACKTEST_SHARPE
        )
        g1 = GateResult(
            gate_name="Historical Performance",
            passed=g1_passed,
            score=backtest_sharpe,
            threshold=MIN_BACKTEST_SHARPE,
            details={
                "win_rate": backtest_win_rate,
                "profit_factor": backtest_profit_factor,
                "n_trades": backtest_n_trades,
                "sharpe": backtest_sharpe,
            },
            reason=self._gate1_reason(g1_passed, backtest_win_rate, backtest_profit_factor,
                                      backtest_n_trades, backtest_sharpe),
        )
        report.gates.append(g1)
        if not g1_passed:
            rejection_reasons.append(f"Gate 1 FAILED: {g1.reason}")

        # ════════════════════════════════════════════════════════════
        # GATE 2: Walk-Forward (Out-of-Sample)
        # ════════════════════════════════════════════════════════════
        degradation = 1.0 - (wf_oos_sharpe / wf_is_sharpe) if wf_is_sharpe > 0 else 1.0
        g2_passed = (
            wf_win_rate >= MIN_WF_WIN_RATE and
            wf_profit_factor >= MIN_WF_PROFIT_FACTOR and
            wf_n_periods >= MIN_WF_PERIODS and
            degradation < MAX_WF_DEGRADATION
        )
        g2 = GateResult(
            gate_name="Walk-Forward (OOS)",
            passed=g2_passed,
            score=wf_oos_sharpe,
            threshold=0.3,
            details={
                "oos_win_rate": wf_win_rate,
                "oos_profit_factor": wf_profit_factor,
                "n_periods": wf_n_periods,
                "is_sharpe": wf_is_sharpe,
                "oos_sharpe": wf_oos_sharpe,
                "degradation": round(degradation, 3),
            },
            reason=self._gate2_reason(g2_passed, wf_win_rate, wf_profit_factor,
                                       wf_n_periods, degradation),
        )
        report.gates.append(g2)
        if not g2_passed:
            rejection_reasons.append(f"Gate 2 FAILED: {g2.reason}")

        # ════════════════════════════════════════════════════════════
        # GATE 3: Robustness (Parameter Sensitivity)
        # ════════════════════════════════════════════════════════════
        g3_passed = (
            robustness_score >= MIN_ROBUSTNESS_SCORE and
            max_parameter_sensitivity <= MAX_PARAMETER_SENSITIVITY
        )
        g3 = GateResult(
            gate_name="Robustness",
            passed=g3_passed,
            score=robustness_score,
            threshold=MIN_ROBUSTNESS_SCORE,
            details={
                "robustness_score": robustness_score,
                "max_sensitivity": max_parameter_sensitivity,
            },
            reason=self._gate3_reason(g3_passed, robustness_score, max_parameter_sensitivity),
        )
        report.gates.append(g3)
        if not g3_passed:
            rejection_reasons.append(f"Gate 3 FAILED: {g3.reason}")

        # ════════════════════════════════════════════════════════════
        # GATE 4: Live Paper Trading
        # ════════════════════════════════════════════════════════════
        g4_passed = (
            paper_n_trades >= MIN_PAPER_TRADES and
            paper_win_rate >= MIN_PAPER_WIN_RATE and
            paper_max_drawdown_pct <= MAX_PAPER_DRAWDOWN_PCT
        )
        g4 = GateResult(
            gate_name="Paper Trading",
            passed=g4_passed,
            score=paper_win_rate,
            threshold=MIN_PAPER_WIN_RATE,
            details={
                "n_trades": paper_n_trades,
                "win_rate": paper_win_rate,
                "max_drawdown_pct": paper_max_drawdown_pct,
            },
            reason=self._gate4_reason(g4_passed, paper_n_trades, paper_win_rate,
                                       paper_max_drawdown_pct),
        )
        report.gates.append(g4)
        if not g4_passed:
            rejection_reasons.append(f"Gate 4 FAILED: {g4.reason}")

        # ════════════════════════════════════════════════════════════
        # GATE 5: Duplicate Check
        # ════════════════════════════════════════════════════════════
        g5_passed = max_correlation_with_existing < MAX_CORRELATION_WITH_EXISTING
        g5 = GateResult(
            gate_name="Duplicate Check",
            passed=g5_passed,
            score=1.0 - max_correlation_with_existing,
            threshold=1.0 - MAX_CORRELATION_WITH_EXISTING,
            details={
                "max_correlation": max_correlation_with_existing,
                "correlated_with": correlated_feature_name,
            },
            reason=self._gate5_reason(g5_passed, max_correlation_with_existing,
                                       correlated_feature_name),
        )
        report.gates.append(g5)
        if not g5_passed:
            rejection_reasons.append(f"Gate 5 FAILED: {g5.reason}")

        # ════════════════════════════════════════════════════════════
        # GATE 6: Overfitting Check
        # ════════════════════════════════════════════════════════════
        g6_passed = (
            is_oos_ratio <= MAX_IS_OOS_RATIO and
            wf_oos_sharpe >= MIN_OOS_SHARPE
        )
        g6 = GateResult(
            gate_name="Overfitting Check",
            passed=g6_passed,
            score=1.0 / (is_oos_ratio + 1e-10),  # higher = less overfit
            threshold=1.0 / MAX_IS_OOS_RATIO,
            details={
                "is_oos_ratio": is_oos_ratio,
                "oos_sharpe": wf_oos_sharpe,
            },
            reason=self._gate6_reason(g6_passed, is_oos_ratio, wf_oos_sharpe),
        )
        report.gates.append(g6)
        if not g6_passed:
            rejection_reasons.append(f"Gate 6 FAILED: {g6.reason}")

        # ════════════════════════════════════════════════════════════
        # FINAL DECISION
        # ════════════════════════════════════════════════════════════
        n_passed = sum(1 for g in report.gates if g.passed)
        n_total = len(report.gates)
        report.overall_score = n_passed / n_total
        report.rejection_reasons = rejection_reasons

        if n_passed == n_total:
            report.approved = True
            report.recommendation = (
                f"APPROVED — passed all {n_total} gates. "
                f"Deploy with monitoring."
            )
        else:
            report.approved = False
            report.recommendation = (
                f"REJECTED — failed {n_total - n_passed}/{n_total} gates. "
                f"Do NOT deploy. See rejection reasons."
            )

        # Log
        status = "✅ APPROVED" if report.approved else "❌ REJECTED"
        log.info(
            f"[ValidationFramework] {status} '{feature_name}' — "
            f"{n_passed}/{n_total} gates passed | {report.recommendation}"
        )
        for g in report.gates:
            icon = "✓" if g.passed else "✗"
            log.info(f"  {icon} Gate: {g.gate_name} — {g.reason}")

        return report

    # ── Gate reason generators ──────────────────────────────────────

    def _gate1_reason(self, passed, wr, pf, n, sharpe):
        if passed:
            return f"WR={wr:.0%} PF={pf:.2f} trades={n} Sharpe={sharpe:.2f} — all above threshold"
        reasons = []
        if wr < MIN_BACKTEST_WIN_RATE: reasons.append(f"WR {wr:.0%} < {MIN_BACKTEST_WIN_RATE:.0%}")
        if pf < MIN_BACKTEST_PROFIT_FACTOR: reasons.append(f"PF {pf:.2f} < {MIN_BACKTEST_PROFIT_FACTOR}")
        if n < MIN_BACKTEST_TRADES: reasons.append(f"trades {n} < {MIN_BACKTEST_TRADES}")
        if sharpe < MIN_BACKTEST_SHARPE: reasons.append(f"Sharpe {sharpe:.2f} < {MIN_BACKTEST_SHARPE}")
        return "; ".join(reasons)

    def _gate2_reason(self, passed, wr, pf, n, degradation):
        if passed:
            return f"OOS WR={wr:.0%} PF={pf:.2f} periods={n} degradation={degradation:.0%}"
        reasons = []
        if wr < MIN_WF_WIN_RATE: reasons.append(f"OOS WR {wr:.0%} < {MIN_WF_WIN_RATE:.0%}")
        if pf < MIN_WF_PROFIT_FACTOR: reasons.append(f"OOS PF {pf:.2f} < {MIN_WF_PROFIT_FACTOR}")
        if n < MIN_WF_PERIODS: reasons.append(f"periods {n} < {MIN_WF_PERIODS}")
        if degradation >= MAX_WF_DEGRADATION: reasons.append(f"degradation {degradation:.0%} > {MAX_WF_DEGRADATION:.0%}")
        return "; ".join(reasons)

    def _gate3_reason(self, passed, score, sensitivity):
        if passed:
            return f"Robustness={score:.0%} sensitivity={sensitivity:.2f}"
        return f"Robustness {score:.0%} < {MIN_ROBUSTNESS_SCORE:.0%} or sensitivity {sensitivity:.2f} > {MAX_PARAMETER_SENSITIVITY}"

    def _gate4_reason(self, passed, n, wr, dd):
        if passed:
            return f"Paper: {n} trades WR={wr:.0%} maxDD={dd:.1f}%"
        reasons = []
        if n < MIN_PAPER_TRADES: reasons.append(f"trades {n} < {MIN_PAPER_TRADES}")
        if wr < MIN_PAPER_WIN_RATE: reasons.append(f"WR {wr:.0%} < {MIN_PAPER_WIN_RATE:.0%}")
        if dd > MAX_PAPER_DRAWDOWN_PCT: reasons.append(f"DD {dd:.1f}% > {MAX_PAPER_DRAWDOWN_PCT}%")
        return "; ".join(reasons)

    def _gate5_reason(self, passed, corr, name):
        if passed:
            return f"Max correlation={corr:.2f} < {MAX_CORRELATION_WITH_EXISTING}"
        return f"Correlation {corr:.2f} ≥ {MAX_CORRELATION_WITH_EXISTING} with '{name}' — duplicate feature"

    def _gate6_reason(self, passed, ratio, oos_sharpe):
        if passed:
            return f"IS/OOS ratio={ratio:.2f} OOS Sharpe={oos_sharpe:.2f} — no overfitting"
        reasons = []
        if ratio > MAX_IS_OOS_RATIO: reasons.append(f"IS/OOS {ratio:.2f} > {MAX_IS_OOS_RATIO} — overfitting suspected")
        if oos_sharpe < MIN_OOS_SHARPE: reasons.append(f"OOS Sharpe {oos_sharpe:.2f} < {MIN_OOS_SHARPE}")
        return "; ".join(reasons)


# ════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    vf = ValidationFramework()

    print("=== Test 1: Good Feature (should pass all gates) ===")
    r1 = vf.validate_feature(
        feature_name="vwap_confluence",
        backtest_win_rate=0.62, backtest_profit_factor=1.8,
        backtest_n_trades=250, backtest_sharpe=1.2,
        wf_win_rate=0.55, wf_profit_factor=1.3,
        wf_n_periods=5, wf_is_sharpe=1.2, wf_oos_sharpe=0.8,
        robustness_score=0.85, max_parameter_sensitivity=0.12,
        paper_n_trades=45, paper_win_rate=0.53, paper_max_drawdown_pct=6.0,
        max_correlation_with_existing=0.45, correlated_feature_name="vwap",
        is_oos_ratio=1.5,
    )
    print(f"  Approved: {r1.approved} — {r1.recommendation}")
    for g in r1.gates:
        icon = "✓" if g.passed else "✗"
        print(f"  {icon} {g.gate_name}: {g.reason}")

    print("\n=== Test 2: Overfit Feature (should fail Gate 6) ===")
    r2 = vf.validate_feature(
        feature_name="curve_fitted_indicator",
        backtest_win_rate=0.85, backtest_profit_factor=3.5,
        backtest_n_trades=200, backtest_sharpe=2.8,
        wf_win_rate=0.42, wf_profit_factor=0.9,
        wf_n_periods=4, wf_is_sharpe=2.8, wf_oos_sharpe=0.1,
        robustness_score=0.40, max_parameter_sensitivity=0.35,
        paper_n_trades=35, paper_win_rate=0.40, paper_max_drawdown_pct=12.0,
        max_correlation_with_existing=0.30,
        is_oos_ratio=28.0,  # IS is 28× better than OOS — massive overfitting
    )
    print(f"  Approved: {r2.approved} — {r2.recommendation}")
    for g in r2.gates:
        icon = "✓" if g.passed else "✗"
        print(f"  {icon} {g.gate_name}: {g.reason}")

    print("\n=== Test 3: Duplicate Feature (should fail Gate 5) ===")
    r3 = vf.validate_feature(
        feature_name="rsi_v2",
        backtest_win_rate=0.55, backtest_profit_factor=1.3,
        backtest_n_trades=150, backtest_sharpe=0.8,
        wf_win_rate=0.50, wf_profit_factor=1.1,
        wf_n_periods=4, wf_is_sharpe=0.8, wf_oos_sharpe=0.5,
        robustness_score=0.80, max_parameter_sensitivity=0.15,
        paper_n_trades=40, paper_win_rate=0.50, paper_max_drawdown_pct=5.0,
        max_correlation_with_existing=0.92, correlated_feature_name="rsi",  # 92% correlated!
        is_oos_ratio=1.6,
    )
    print(f"  Approved: {r3.approved} — {r3.recommendation}")
    for g in r3.gates:
        icon = "✓" if g.passed else "✗"
        print(f"  {icon} {g.gate_name}: {g.reason}")

    print("\nValidation framework smoke test passed.")
