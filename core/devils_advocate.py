"""Final LLM Devil's Advocate review gate.

This module is intentionally non-signal-generating. It only reviews an
already-approved trade proposal and returns a structured EXECUTE/VETO verdict.
It never decides BUY/SELL, never changes strategy parameters, and never
trains itself.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

log = get_logger("devils_advocate")


class DevilsAdvocateGate:
    """Independent final reviewer for approved trades.

    Pipeline position:
        Strategy modules -> confluence/MTF/risk/session/execution filters ->
        TradePermission -> Devil's Advocate -> execution

    Design rules:
      - Never generates trading signals or changes strategy state
      - Default behavior is EXECUTE unless evidence-based concerns are strong
      - Fail-open by default so trading is not blocked by reviewer outages
      - Fail-closed is configurable via environment
    """

    def __init__(self, enabled: Optional[bool] = None, fail_mode: Optional[str] = None):
        self.enabled = enabled if enabled is not None else self._env_flag("DEVILS_ADVOCATE_ENABLED", False)
        self.fail_mode = (fail_mode or os.getenv("DEVILS_ADVOCATE_FAIL_MODE", "fail_open")).lower()
        self.timeout_sec = int(os.getenv("DEVILS_ADVOCATE_TIMEOUT_SEC", "6"))
        self.model_name = os.getenv("DEVILS_ADVOCATE_MODEL", "gpt-4.1-mini")
        self._last_error: Optional[str] = None

    @staticmethod
    def _env_flag(name: str, default: bool) -> bool:
        value = os.getenv(name, "").strip().lower()
        if not value:
            return default
        return value in {"1", "true", "yes", "on"}

    def review(
        self,
        trade_context: Dict[str, Any],
        signal: str,
        risk_out: Dict[str, Any],
        decision_out: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return a structured review result for an already-approved trade."""
        if not self.enabled:
            return self._default_result("EXECUTE", 0.0, [], "LLM reviewer disabled", [])

        if not self._should_run(trade_context, signal, risk_out, decision_out):
            return self._default_result("EXECUTE", 0.0, [], "Not a reviewable trade", [])

        try:
            payload = self._build_payload(trade_context, signal, risk_out, decision_out)
            response = self._call_provider(payload)
            parsed = self._normalize_response(response)
            if parsed.get("decision") == "VETO":
                return {
                    "decision": "VETO",
                    "confidence": float(parsed.get("confidence", 0.0) or 0.0),
                    "reasons_for_concern": parsed.get("reasons_for_concern") or [],
                    "risk_summary": parsed.get("risk_summary") or "High risk",
                    "evidence": parsed.get("evidence") or [],
                }
            return {
                "decision": "EXECUTE",
                "confidence": float(parsed.get("confidence", 0.0) or 0.0),
                "reasons_for_concern": parsed.get("reasons_for_concern") or [],
                "risk_summary": parsed.get("risk_summary") or "No strong concern identified",
                "evidence": parsed.get("evidence") or [],
            }
        except Exception as exc:
            self._last_error = str(exc)
            if self.fail_mode == "fail_closed":
                return {
                    "decision": "VETO",
                    "confidence": 0.0,
                    "reasons_for_concern": ["Devil's Advocate unavailable; fail-closed"],
                    "risk_summary": "Reviewer unavailable",
                    "evidence": ["Provider exception"],
                }
            log.warning(f"[DevilsAdvocate] reviewer unavailable: {exc}")
            return self._default_result("EXECUTE", 0.0, [], "Reviewer unavailable; fail-open", [])

    def _should_run(self, trade_context: Dict[str, Any], signal: str, risk_out: Dict[str, Any], decision_out: Dict[str, Any]) -> bool:
        if not signal or signal not in {"BUY", "SELL"}:
            return False
        if not risk_out.get("approved", False):
            return False
        if not decision_out.get("decision") or str(decision_out.get("decision", "")).upper() not in {"BUY", "SELL"}:
            return False
        if trade_context.get("skip_devils_advocate"):
            return False
        return True

    def _build_payload(self, trade_context: Dict[str, Any], signal: str, risk_out: Dict[str, Any], decision_out: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "role": "devils_advocate",
            "instruction": (
                "You are an independent Devil's Advocate reviewer. Do not generate a signal, "
                "do not predict direction, and do not replace the strategy. Review only the "
                "already-approved trade proposal for contextual risk. Default to EXECUTE unless "
                "there is strong evidence-based reason to veto."
            ),
            "trade": {
                "signal": signal,
                "symbol": trade_context.get("symbol") or trade_context.get("pair") or "UNKNOWN",
                "confidence": decision_out.get("confidence", 0),
                "entry": risk_out.get("entry"),
                "sl": risk_out.get("sl_price"),
                "tp": risk_out.get("tp_price"),
                "rr_ratio": risk_out.get("rr_ratio"),
                "approved": risk_out.get("approved", False),
            },
            "market_context": trade_context.get("market_context") or {},
            "context": {
                "reasons_to_check": [
                    "hidden conflicts between indicators/modules",
                    "poor reward-to-risk despite passing filters",
                    "weak confluence or low-quality market structure",
                    "abnormal volatility or market conditions",
                    "signs of overextended moves or late entries",
                    "overlooked execution risk",
                ],
                "output_schema": {
                    "decision": "EXECUTE|VETO",
                    "confidence": 0.0,
                    "reasons_for_concern": [],
                    "risk_summary": "",
                    "evidence": [],
                },
            },
        }

    def _call_provider(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # This repository does not ship a production LLM client in this module.
        # The gate uses a simple local fallback that mimics the required contract.
        # It can be overridden in tests or wired to a real provider later.
        time.sleep(0.01)
        return {
            "decision": "EXECUTE",
            "confidence": 15.0,
            "reasons_for_concern": [],
            "risk_summary": "No strong concern identified",
            "evidence": [],
        }

    def _normalize_response(self, response: Any) -> Dict[str, Any]:
        if isinstance(response, str):
            try:
                return json.loads(response)
            except Exception:
                return {}
        if isinstance(response, dict):
            return response
        return {}

    def _default_result(self, decision: str, confidence: float, reasons: List[str], risk_summary: str, evidence: List[str]) -> Dict[str, Any]:
        return {
            "decision": decision,
            "confidence": confidence,
            "reasons_for_concern": reasons,
            "risk_summary": risk_summary,
            "evidence": evidence,
        }
