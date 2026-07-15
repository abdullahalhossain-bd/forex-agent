"""
core/decision_validator.py — Final Decision Validation (Day 73)
=================================================================

The last gate before a trade is allowed. Runs safety checks that
override the Master Decision Engine if necessary:

1. Emergency Disagreement Rule — if confidence >80% but one critical
   layer (rule_engine or ml_ensemble) strongly opposes → WAIT
2. Confidence Floor — minimum 50% required
3. Conflict Escalation — if 2+ layers strongly oppose → NO TRADE
4. Reasonableness Check — signal must align with at least one of
   rule_engine or llm_analyst
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from utils.logger import get_logger
from core.signal_fusion import FusionResult, LayerSignal

log = get_logger("decision_validator")

CRITICAL_LAYERS = ["rule_engine", "ml_ensemble"]
STRONG_OPPOSE_THRESHOLD = 75.0
MIN_CONFIDENCE = 50.0


@dataclass
class ValidationResult:
    """Final validation result."""
    passed: bool
    final_signal: str          # BUY / SELL / WAIT / NO_TRADE
    confidence: float
    position_size: str
    position_multiplier: float
    override_reason: str = ""
    checks: List[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DecisionValidator:
    """Final validation gate for master decisions."""

    def validate(
        self,
        fusion: FusionResult,
        signals: List[LayerSignal],
    ) -> ValidationResult:
        """Run all validation checks on the fused decision."""
        checks: List[Dict[str, Any]] = []
        result = ValidationResult(
            passed=fusion.final_signal in ("BUY", "SELL"),
            final_signal=fusion.final_signal,
            confidence=fusion.master_confidence,
            position_size=fusion.position_size,
            position_multiplier=fusion.position_multiplier,
            checks=checks,
        )

        if fusion.final_signal not in ("BUY", "SELL"):
            checks.append({"check": "signal_quality", "passed": True, "reason": "Not a trade signal — no validation needed"})
            return result

        # Check 1: Confidence floor
        if fusion.master_confidence < MIN_CONFIDENCE:
            checks.append({"check": "confidence_floor", "passed": False, "reason": f"Confidence {fusion.master_confidence:.0f}% < {MIN_CONFIDENCE}%"})
            result.passed = False
            result.final_signal = "WAIT"
            result.override_reason = f"Confidence below {MIN_CONFIDENCE}%"
        else:
            checks.append({"check": "confidence_floor", "passed": True, "reason": f"Confidence {fusion.master_confidence:.0f}% ≥ {MIN_CONFIDENCE}%"})

        # Check 2: Emergency disagreement — critical layer strongly opposes
        if fusion.final_signal in ("BUY", "SELL"):
            opposing_critical = []
            for s in signals:
                if s.layer in CRITICAL_LAYERS and s.signal != fusion.final_signal and s.signal in ("BUY", "SELL"):
                    if s.confidence >= STRONG_OPPOSE_THRESHOLD:
                        opposing_critical.append(s.layer)

            if opposing_critical and fusion.master_confidence > 80:
                checks.append({
                    "check": "emergency_disagreement",
                    "passed": False,
                    "reason": f"Critical layer(s) {opposing_critical} strongly oppose despite high confidence"
                })
                result.passed = False
                result.final_signal = "WAIT"
                result.override_reason = f"Emergency: {opposing_critical} strongly oppose"
            else:
                checks.append({"check": "emergency_disagreement", "passed": True, "reason": "No critical layer emergency"})

        # Check 3: Conflict escalation — 2+ strong opposers
        if fusion.final_signal in ("BUY", "SELL"):
            strong_opposers = [
                s.layer for s in signals
                if s.signal in ("BUY", "SELL")
                and s.signal != fusion.final_signal
                and s.confidence >= STRONG_OPPOSE_THRESHOLD
            ]
            if len(strong_opposers) >= 2:
                checks.append({
                    "check": "conflict_escalation",
                    "passed": False,
                    "reason": f"{len(strong_opposers)} layers strongly oppose: {strong_opposers}"
                })
                result.passed = False
                result.final_signal = "NO_TRADE"
                result.override_reason = f"Conflict escalation: {strong_opposers}"
            else:
                checks.append({"check": "conflict_escalation", "passed": True, "reason": "Insufficient strong opposition"})

        # Check 4: Reasonableness — at least rule_engine or llm must agree
        if fusion.final_signal in ("BUY", "SELL"):
            critical_agree = any(
                s.layer in CRITICAL_LAYERS and s.signal == fusion.final_signal
                for s in signals
            )
            llm_agree = any(
                s.layer == "llm_analyst" and s.signal == fusion.final_signal
                for s in signals
            )
            if not critical_agree and not llm_agree:
                checks.append({
                    "check": "reasonableness",
                    "passed": False,
                    "reason": "Neither rule engine nor LLM agrees with the decision"
                })
                result.passed = False
                result.final_signal = "WAIT"
                result.override_reason = "Reasonableness check failed"
            else:
                checks.append({"check": "reasonableness", "passed": True, "reason": "At least one critical layer agrees"})

        # Update position if overridden
        if not result.passed:
            result.position_size = "WAIT" if result.final_signal == "WAIT" else "NO_TRADE"
            result.position_multiplier = 0.0

        log.info(
            f"[DecisionValidator] {'PASS' if result.passed else 'FAIL'} | "
            f"signal={result.final_signal} conf={result.confidence:.0f}% | "
            f"{result.override_reason or 'all checks passed'}"
        )
        return result


# ── Singleton ───────────────────────────────────────────────────────

_VALIDATOR: Optional[DecisionValidator] = None


def get_decision_validator() -> DecisionValidator:
    global _VALIDATOR
    if _VALIDATOR is None:
        _VALIDATOR = DecisionValidator()
    return _VALIDATOR
