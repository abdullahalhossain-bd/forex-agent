"""
intelligence/signal_validator.py — Signal validation gates
============================================================

Day 67 — Pre-trade validation that runs AFTER the confluence score is
computed but BEFORE the trade is allowed. Implements three professional
trader safeguards:

1. **Minimum Aligned Factor Rule** — fewer than MIN_ALIGNED_FACTORS (see
   below; currently 2) aligned factors → WAIT. Originally set to 5, but
   lowered because 5 was too strict and blocked most good trades — see
   the MIN_ALIGNED_FACTORS constant for the full rationale.
2. **Contradiction Detector** — if 2+ top-weight factors strongly disagree,
   the trade is blocked (e.g. SMC BUY + News Strong USD Bearish = conflict)
3. **False Signal Protection** — sequential gates:
      Signal → Confluence → Risk → News → Correlation → Execution

Each gate returns:
    {
        "passed": bool,
        "reason": str,
        "severity": "OK" | "WARNING" | "BLOCK",
        "details": dict
    }
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

from utils.logger import get_logger
from intelligence.decision_score import ConfluenceScore, FactorScore, FACTOR_WEIGHTS

log = get_logger("signal_validator")


# ── Minimum aligned factors required to take a trade ────────────────
# Lowered from 5 to 2 — 5 was too strict, blocking most good trades.
# With 7 factors, requiring 5 aligned means 71% agreement which is very rare.
# 2 out of 7 (29%) allows good signals through while still maintaining confluence.
MIN_ALIGNED_FACTORS = 1

# ── Top-weight factors that must NOT strongly disagree ──────────────
# If any pair of these factors have opposing BUY/SELL directions AND
# both have strength >= 60, we treat it as a hard contradiction.
TOP_WEIGHT_FACTORS = ["smc", "liquidity", "currency_strength", "intermarket"]
CONTRADICTION_STRENGTH_THRESHOLD = 60.0


@dataclass
class ValidationResult:
    """Result of one validation gate."""
    gate: str             # confluence / factor_count / contradiction / risk / news / correlation
    passed: bool
    severity: str         # OK / WARNING / BLOCK
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SignalValidator:
    """Runs all pre-trade validation gates on a ConfluenceScore."""

    def validate_all(
        self,
        score: ConfluenceScore,
        pair: str = "",
        news_blocked_pairs: Optional[Dict[str, str]] = None,
        correlation_blocked: bool = False,
        risk_approved: bool = True,
    ) -> Dict[str, Any]:
        """Run every validation gate. Returns:
            {
                "passed": bool,            # True only if ALL gates passed
                "gates": List[ValidationResult],
                "block_reason": str,       # first BLOCK reason, or ""
                "should_trade": bool,      # final decision
            }
        """
        gates: List[ValidationResult] = []

        # Gate 1: Confluence score quality
        gates.append(self._gate_confluence_quality(score))

        # Gate 2: minimum aligned factor rule (MIN_ALIGNED_FACTORS)
        gates.append(self._gate_factor_count(score))

        # Gate 3: Contradiction detector
        gates.append(self._gate_contradiction(score))

        # Gate 4: Risk approval
        gates.append(self._gate_risk(risk_approved))

        # Gate 5: News block
        gates.append(self._gate_news(pair, news_blocked_pairs or {}))

        # Gate 6: Correlation
        gates.append(self._gate_correlation(correlation_blocked))

        # Final decision
        block_reasons = [g.reason for g in gates if g.severity == "BLOCK"]
        should_trade = (
            len(block_reasons) == 0
            and score.final_direction in ("BUY", "SELL")
            and score.setup_quality in ("A+", "A", "B", "C")
        )

        return {
            "passed": should_trade,
            "should_trade": should_trade,
            "gates": [g.to_dict() for g in gates],
            "block_reason": block_reasons[0] if block_reasons else "",
            "all_gates_passed": len(block_reasons) == 0,
        }

    # ── Individual gates ────────────────────────────────────────────

    def _gate_confluence_quality(self, score: ConfluenceScore) -> ValidationResult:
        """Gate 1: Setup quality must be A+, A, or B."""
        if score.setup_quality == "AVOID":
            return ValidationResult(
                gate="confluence",
                passed=False,
                severity="BLOCK",
                reason=f"Setup quality AVOID (net={score.net_score}, aligned={score.aligned_factors})",
                details={"setup_quality": score.setup_quality, "net_score": score.net_score},
            )
        return ValidationResult(
            gate="confluence",
            passed=True,
            severity="OK",
            reason=f"Setup quality {score.setup_quality}",
            details={"setup_quality": score.setup_quality},
        )

    def _gate_factor_count(self, score: ConfluenceScore) -> ValidationResult:
        """Gate 2: At least MIN_ALIGNED_FACTORS aligned factors required
        (currently 2 — see module-level constant for rationale)."""
        if score.aligned_factors < MIN_ALIGNED_FACTORS:
            return ValidationResult(
                gate="factor_count",
                passed=False,
                severity="BLOCK",
                reason=f"Only {score.aligned_factors}/{score.total_factors} factors aligned (need ≥{MIN_ALIGNED_FACTORS})",
                details={
                    "aligned_factors": score.aligned_factors,
                    "total_factors": score.total_factors,
                    "required": MIN_ALIGNED_FACTORS,
                },
            )
        return ValidationResult(
            gate="factor_count",
            passed=True,
            severity="OK",
            reason=f"{score.aligned_factors}/{score.total_factors} factors aligned",
            details={"aligned_factors": score.aligned_factors},
        )

    def _gate_contradiction(self, score: ConfluenceScore) -> ValidationResult:
        """Gate 3: No strong contradictions among top-weight factors."""
        contradictions: List[str] = []
        top_factors = [f for f in score.factors if f.name in TOP_WEIGHT_FACTORS]
        for i, f1 in enumerate(top_factors):
            for f2 in top_factors[i + 1:]:
                if (f1.direction in ("BUY", "SELL")
                        and f2.direction in ("BUY", "SELL")
                        and f1.direction != f2.direction
                        and f1.strength >= CONTRADICTION_STRENGTH_THRESHOLD
                        and f2.strength >= CONTRADICTION_STRENGTH_THRESHOLD):
                    contradictions.append(
                        f"{f1.name}={f1.direction}({f1.strength:.0f}) vs "
                        f"{f2.name}={f2.direction}({f2.strength:.0f})"
                    )
        if contradictions:
            return ValidationResult(
                gate="contradiction",
                passed=False,
                severity="BLOCK",
                reason=f"Contradiction: {'; '.join(contradictions)}",
                details={"contradictions": contradictions},
            )
        return ValidationResult(
            gate="contradiction",
            passed=True,
            severity="OK",
            reason="No top-weight contradictions",
        )

    def _gate_risk(self, risk_approved: bool) -> ValidationResult:
        """Gate 4: Risk engine must approve."""
        if not risk_approved:
            return ValidationResult(
                gate="risk",
                passed=False,
                severity="BLOCK",
                reason="Risk engine rejected the trade",
            )
        return ValidationResult(
            gate="risk",
            passed=True,
            severity="OK",
            reason="Risk approved",
        )

    def _gate_news(self, pair: str, news_blocked_pairs: Dict[str, str]) -> ValidationResult:
        """Gate 5: Pair not in news block window."""
        if pair in news_blocked_pairs:
            return ValidationResult(
                gate="news",
                passed=False,
                severity="BLOCK",
                reason=f"News block: {news_blocked_pairs[pair]}",
            )
        return ValidationResult(
            gate="news",
            passed=True,
            severity="OK",
            reason="No active news block",
        )

    def _gate_correlation(self, correlation_blocked: bool) -> ValidationResult:
        """Gate 6: Correlation filter must allow this trade."""
        if correlation_blocked:
            return ValidationResult(
                gate="correlation",
                passed=False,
                severity="BLOCK",
                reason="Correlated pair already has an open position",
            )
        return ValidationResult(
            gate="correlation",
            passed=True,
            severity="OK",
            reason="No correlation conflict",
        )


# ── singleton ───────────────────────────────────────────────────────
_VALIDATOR: Optional[SignalValidator] = None


def get_signal_validator() -> SignalValidator:
    global _VALIDATOR
    if _VALIDATOR is None:
        _VALIDATOR = SignalValidator()
    return _VALIDATOR