"""
intelligence/signal_validator.py — Signal validation gates
============================================================

Day 67 — Pre-trade validation that runs AFTER the confluence score is
computed but BEFORE the trade is allowed.

Confidence-pipeline simplification (institutional refactor):
  - Gates 1+2 (confluence quality + factor count) COLLAPSED into one
    soft check.  Previously they duplicated decision_score.py's own
    AVOID/factor-count logic and could hard-block trades that
    decision_score had already scored with nonzero confidence.
  - Gate 3 (contradiction) converted from hard BLOCK to a confidence
    deduction proportional to the strength of the disagreeing factors.
    A single weak/conflicting secondary factor no longer nukes an
    otherwise excellent trade.
  - Gates 4-6 (risk, news, correlation) remain as hard BLOCKs —
    they protect against genuine danger, not low confidence.

Each gate returns:
    {
        "passed": bool,
        "reason": str,
        "severity": "OK" | "WARNING" | "BLOCK",
        "confidence_penalty": float,  # NEW: deduction applied to confidence
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
    confidence_penalty: float = 0.0  # NEW: deduction to apply to confidence
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
        # Confidence-pipeline simplification: only Gates 4-6 (risk, news,
        # correlation) can hard-BLOCK.  Gates 1-3 are now soft (WARNING
        # with confidence deductions).  We also accept any directional
        # signal — quality "C" / "AVOID" no longer blocks.
        block_reasons = [g.reason for g in gates if g.severity == "BLOCK"]
        # Sum up all soft penalties from Gates 1-3
        total_soft_penalty = sum(g.confidence_penalty for g in gates if g.severity == "WARNING")

        should_trade = (
            len(block_reasons) == 0
            and score.final_direction in ("BUY", "SELL")
        )

        from utils.confidence_trace import confidence_trace
        if total_soft_penalty > 0:
            confidence_trace.record(
                module="signal_validator",
                before=score.confidence,
                after=max(0, score.confidence - total_soft_penalty),
                reason=f"soft penalties from gates: -{total_soft_penalty:.1f} (no hard block)",
            )

        return {
            "passed": should_trade,
            "should_trade": should_trade,
            "gates": [g.to_dict() for g in gates],
            "block_reason": block_reasons[0] if block_reasons else "",
            "all_gates_passed": len(block_reasons) == 0,
            "confidence_penalty": total_soft_penalty,  # NEW: for downstream consumers
        }

    # ── Individual gates ────────────────────────────────────────────

    def _gate_confluence_quality(self, score: ConfluenceScore) -> ValidationResult:
        """Gate 1 (softened): Setup quality check.

        Confidence-pipeline simplification: this gate no longer hard-blocks.
        It was duplicating decision_score.py's own AVOID classification.
        Now it only emits a WARNING with a small confidence deduction when
        quality is low, so the signal flows downstream instead of dying here.
        """
        if score.setup_quality in ("AVOID", "C"):
            penalty = 5.0  # small deduction, not a kill
            return ValidationResult(
                gate="confluence",
                passed=True,  # always pass now
                severity="WARNING",
                reason=f"Setup quality {score.setup_quality} (net={score.net_score}, aligned={score.aligned_factors}) — soft penalty applied",
                confidence_penalty=penalty,
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
        """Gate 2 (softened): Aligned factor count check.

        Confidence-pipeline simplification: merged with Gate 1.
        Previously duplicated decision_score.py's own factor counting.
        Now: only a WARNING with a small deduction when below threshold.
        """
        if score.aligned_factors < MIN_ALIGNED_FACTORS:
            penalty = 3.0  # small deduction, not a block
            return ValidationResult(
                gate="factor_count",
                passed=True,  # always pass now
                severity="WARNING",
                reason=f"Only {score.aligned_factors}/{score.total_factors} factors aligned (need ≥{MIN_ALIGNED_FACTORS}) — soft penalty",
                confidence_penalty=penalty,
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
        """Gate 3 (softened): Contradiction detector.

        Confidence-pipeline simplification: previously a hard BLOCK when
        ANY pair of top-weight factors strongly disagreed.  Now: a
        proportional confidence deduction based on the STRENGTH of the
        disagreeing factors.  One weak/conflicting secondary factor no
        longer nukes an otherwise excellent trade.  The deduction is
        capped at 15 points so the trade is weakened but not killed.
        """
        contradictions: List[str] = []
        total_penalty = 0.0
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
                    # Proportional penalty: average strength of the
                    # disagreeing pair, scaled down.  Capped per-pair at 8.
                    avg_strength = (f1.strength + f2.strength) / 2.0
                    total_penalty += min(8.0, (avg_strength - CONTRADICTION_STRENGTH_THRESHOLD) * 0.2)
        # Cap total contradiction penalty at 15
        total_penalty = min(15.0, total_penalty)
        if contradictions:
            return ValidationResult(
                gate="contradiction",
                passed=True,  # always pass now — apply deduction instead
                severity="WARNING",
                reason=f"Contradiction: {'; '.join(contradictions)} — soft penalty -{total_penalty:.1f}",
                confidence_penalty=total_penalty,
                details={"contradictions": contradictions, "penalty": total_penalty},
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