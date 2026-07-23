"""
core/decision_validator.py — Final Decision Validation (Day 73+)
=================================================================

The last gate before a trade is allowed. Runs safety checks that
override the Master Decision Engine if necessary:

1. Emergency Disagreement Rule — if confidence >80% but one critical
   layer (rule_engine or ml_ensemble) strongly opposes → WAIT
2. Confidence Floor — minimum 50% required
3. Conflict Escalation — if 2+ layers strongly oppose → NO TRADE
4. Reasonableness Check — signal must align with at least one of
   rule_engine or llm_analyst
5. Ollama Institutional Check (Qwen3:4B) — local LLM veto gate
   that can reject trades based on full market context analysis
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict, field
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
    checks: List[Dict[str, Any]] = field(default_factory=list)
    # Ollama validation context (populated if Check 5 ran)
    ollama_check: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


class DecisionValidator:
    """Final validation gate for master decisions.

    Pipeline:
      Checks 1-4 (deterministic) → Check 5 (Ollama Qwen3:4B veto)
    """

    def __init__(self):
        # Lazy-init Ollama validator (fail-open if unavailable)
        self._ollama = None
        try:
            from core.ollama_validator import get_ollama_validator
            self._ollama = get_ollama_validator()
        except Exception as e:
            log.debug(f"[DecisionValidator] Ollama validator unavailable: {e}")

    def validate(
        self,
        fusion: FusionResult,
        signals: List[LayerSignal],
        market_data: Optional[Dict[str, Any]] = None,
        entry_price: float = 0.0,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        risk_reward: float = 0.0,
    ) -> ValidationResult:
        """Run all validation checks on the fused decision.

        Args:
            fusion: Fused signal result from SignalFusion.
            signals: Individual layer signals.
            market_data: Market context for Ollama validation (optional).
            entry_price: Proposed entry price for Ollama.
            stop_loss: Proposed SL for Ollama.
            take_profit: Proposed TP for Ollama.
            risk_reward: Proposed R:R for Ollama.
        """
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

        # ── Check 5: Ollama Qwen3:4B Institutional Veto ───────────
        # Only runs if previous checks passed and Ollama is available.
        # This is a local LLM that independently evaluates the trade.
        # It can VETO but cannot promote.
        if result.passed and self._ollama is not None and market_data:
            try:
                ollama_result = self._ollama.validate(
                    market_data=market_data,
                    proposed_signal=fusion.final_signal,
                    proposed_confidence=fusion.master_confidence,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    risk_reward=risk_reward,
                )
                ollama_dict = ollama_result.to_dict()
                result.ollama_check = ollama_dict

                if ollama_result.checked and not ollama_result.approved:
                    checks.append({
                        "check": "ollama_institutional_veto",
                        "passed": False,
                        "reason": ollama_result.reason[:200],
                        "detail": {
                            "model_decision": ollama_result.decision,
                            "model_confidence": ollama_result.confidence,
                            "risk_level": ollama_result.risk_level,
                            "response_ms": ollama_result.response_time_ms,
                        },
                    })
                    result.passed = False
                    result.final_signal = "NO_TRADE"
                    result.override_reason = f"Ollama veto: {ollama_result.reason[:150]}"
                elif ollama_result.checked and ollama_result.approved:
                    checks.append({
                        "check": "ollama_institutional_veto",
                        "passed": True,
                        "reason": f"Approved by Qwen3:4B (conf={ollama_result.confidence:.0f}%, risk={ollama_result.risk_level})",
                        "detail": {
                            "model_decision": ollama_result.decision,
                            "model_confidence": ollama_result.confidence,
                            "risk_level": ollama_result.risk_level,
                            "response_ms": ollama_result.response_time_ms,
                        },
                    })
                else:
                    # Not checked (disabled, skipped, or error with fail-open)
                    checks.append({
                        "check": "ollama_institutional_veto",
                        "passed": True,
                        "reason": ollama_result.reason or "Not checked (fail-open)",
                    })
            except Exception as e:
                log.debug(f"[DecisionValidator] Ollama check skipped: {e}")
                checks.append({
                    "check": "ollama_institutional_veto",
                    "passed": True,
                    "reason": f"Ollama unavailable: {e}",
                })

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
