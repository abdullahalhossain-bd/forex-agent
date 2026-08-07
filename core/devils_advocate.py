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
        """Call a real LLM (Groq primary, Gemini fallback) to review the trade.

        Uses the same LLMKeyManager multi-key rotation the rest of the stack
        relies on (ai/ai_analyst.py), so it benefits from key failover
        without needing its own credential handling. Returns a dict matching
        the ``output_schema`` in the payload; raises on total failure so the
        caller's fail_open/fail_closed logic in ``review()`` takes over.
        """
        prompt = self._build_prompt(payload)
        deadline = time.monotonic() + self.timeout_sec

        from core.llm_key_manager import get_llm_key_manager, log_llm_call_failure
        manager = get_llm_key_manager()

        last_error: Optional[Exception] = None

        # Try Groq first (fast + cheap), then fall back to Gemini.
        for provider in ("groq", "gemini"):
            if time.monotonic() >= deadline:
                break
            try:
                raw = self._call_one_provider(manager, provider, prompt, deadline)
                if raw is None:
                    continue
                from utils.llm_json import parse_llm_json
                parsed = parse_llm_json(raw)
                if isinstance(parsed, dict) and parsed.get("decision") in {"EXECUTE", "VETO"}:
                    return parsed
                last_error = ValueError(f"[DevilsAdvocate] {provider} returned unparseable/invalid payload")
            except Exception as exc:
                last_error = exc
                log_llm_call_failure(log, provider, self.model_name, 0, 1, exc)
                continue

        raise last_error or RuntimeError("Devil's Advocate: no LLM provider available")

    def _call_one_provider(
        self, manager: Any, provider: str, prompt: str, deadline: float
    ) -> Optional[str]:
        """Make a single request to the named provider. Returns raw text or None."""
        if provider == "groq":
            client = manager.get_groq_client()
            if client is None:
                return None
            create_kwargs = dict(
                model=self.model_name if "gpt" not in self.model_name else "llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=400,
                response_format={"type": "json_object"},
            )
            remaining = max(0.5, deadline - time.monotonic())
            create_kwargs["timeout"] = remaining
            try:
                resp = client.chat.completions.create(**create_kwargs)
            except TypeError:
                create_kwargs.pop("timeout", None)
                create_kwargs.pop("response_format", None)
                resp = client.chat.completions.create(**create_kwargs)
            usage = getattr(resp, "usage", None)
            tokens = (getattr(usage, "prompt_tokens", 0) or 0) + (getattr(usage, "completion_tokens", 0) or 0) if usage else 0
            try:
                manager.mark_groq_success(tokens_used=tokens, client=client)
            except Exception:
                pass
            return resp.choices[0].message.content

        if provider == "gemini":
            if (deadline - time.monotonic()) < 10.0:
                # Gemini API rejects deadlines under 10s.
                return None
            client = manager.get_gemini_client()
            if client is None:
                return None
            generate_kwargs = dict(model="gemini-flash-lite-latest", contents=prompt)
            resp = client.models.generate_content(**generate_kwargs)
            usage = getattr(resp, "usage_metadata", None)
            tokens = 0
            if usage is not None:
                tokens = (getattr(usage, "prompt_token_count", 0) or 0) + (getattr(usage, "candidates_token_count", 0) or 0)
            try:
                manager.mark_gemini_success(tokens_used=tokens, client=client)
            except Exception:
                pass
            return resp.text

        return None

    @staticmethod
    def _build_prompt(payload: Dict[str, Any]) -> str:
        return (
            f"{payload['instruction']}\n\n"
            f"Trade proposal:\n{json.dumps(payload['trade'], default=str)}\n\n"
            f"Market context:\n{json.dumps(payload['market_context'], default=str)}\n\n"
            f"Check specifically for: {', '.join(payload['context']['reasons_to_check'])}.\n\n"
            "Respond with ONLY a JSON object (no markdown fences, no prose) matching exactly "
            f"this schema: {json.dumps(payload['context']['output_schema'])}"
        )

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