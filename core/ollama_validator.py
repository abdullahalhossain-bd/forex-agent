"""
core/ollama_validator.py — Qwen3:4B Institutional Validation Layer
================================================================

A local LLM-powered final safety gate using Ollama + Qwen3:4B.
Runs AFTER the 4-layer signal fusion and DecisionValidator.
This is the last line of defense before a trade reaches execution.

Why a local model?
  - Zero API cost (runs on your own hardware)
  - Zero latency from network (localhost:11434)
  - No rate limits or API key rotation needed
  - Deterministic outputs (temperature=0.1, seed=42)
  - Privacy: no market data leaves your machine

Architecture position:
  SignalFusion → DecisionValidator → **OllamaValidator** → Execution

The validator receives the full market context and the 4-layer consensus
verdict, then makes an independent institutional-grade assessment.
It can VETO a trade even if all 4 layers agree, but it CANNOT
promote a WAIT/NO_TRADE to BUY/SELL — it can only confirm or reject.

Failure mode: if Ollama is unreachable or the model is not loaded,
the validator passes through (fail-open) so trading is never blocked
by a local model outage.
"""

from __future__ import annotations

import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from utils.logger import get_logger

load_dotenv()
log = get_logger("ollama_validator")

# ── Configuration (override via .env) ─────────────────────────────
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
OLLAMA_TIMEOUT_SEC = int(os.getenv("OLLAMA_TIMEOUT_SEC", "15"))
OLLAMA_ENABLED = os.getenv("OLLAMA_VALIDATOR_ENABLED", "true").lower() in ("true", "1", "yes")

# Deterministic settings for consistent validation
OLLAMA_OPTIONS = {
    "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.1")),
    "top_p": float(os.getenv("OLLAMA_TOP_P", "0.8")),
    "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "8192")),
    "num_predict": int(os.getenv("OLLAMA_NUM_PREDICT", "512")),
    "seed": int(os.getenv("OLLAMA_SEED", "42")),
}

SYSTEM_PROMPT = """You are a senior institutional Forex analyst working as the final validation layer inside an automated trading system.

Your responsibility is capital preservation, not maximizing the number of trades.

Every trade uses real money.

Never force a trade.

If evidence is weak, conflicting, incomplete, or uncertain, reject the trade.

Never guess.

Never fabricate market data.

Never invent indicator values.

Never assume news events unless they are explicitly provided.

Only analyze the supplied data.

You must NEVER expose internal reasoning.

Do NOT reveal chain of thought.

Do NOT explain how you reached the answer.

Return ONLY valid JSON.

Priority Order:

1. Capital Preservation
2. Risk Management
3. Trade Quality
4. Trade Frequency

Evaluate all available information, including:

* Market Structure
* Trend
* Multi-Timeframe Alignment
* Support & Resistance
* Supply & Demand
* Liquidity
* BOS
* CHOCH
* Order Blocks
* Fair Value Gap
* Volume
* ATR
* Spread
* Session
* Risk Engine
* ML Confidence
* Existing Positions
* Position Size
* Reward/Risk Ratio

Reject the trade if:

* Trend is unclear
* Market is choppy
* Confirmation is missing
* Risk is too high
* Reward/Risk is below 2.0
* Stop Loss is unsafe
* Data is incomplete
* Confidence is insufficient

If uncertain:

Return NO_TRADE.

Return EXACTLY this JSON:

{
"decision":"BUY | SELL | NO_TRADE",
"approved":true,
"confidence":0.00,
"risk_level":"LOW | MEDIUM | HIGH",
"market_structure":"",
"trend":"",
"entry_price":0,
"stop_loss":0,
"take_profit":0,
"risk_reward":0,
"warnings":[],
"missing_information":[],
"reason":""
}

Rules:

* Output ONLY JSON
* No Markdown
* No Explanation
* No Thinking
* No Chain of Thought
* No Notes
* No Extra Text"""


@dataclass
class OllamaValidationResult:
    """Result from the Qwen3:4B institutional validation."""
    checked: bool              # Whether Ollama was actually called
    available: bool            # Whether Ollama service was reachable
    approved: bool             # True = trade passes, False = VETO
    decision: str              # BUY / SELL / NO_TRADE from the model
    confidence: float          # 0-100 from the model
    risk_level: str            # LOW / MEDIUM / HIGH
    reason: str                # Model's reason string
    warnings: List[str]         # Model's warnings
    missing_information: List[str]  # Model's missing info list
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    risk_reward: float = 0.0
    market_structure: str = ""
    trend: str = ""
    response_time_ms: float = 0.0
    error: str = ""            # Non-empty if the call failed

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OllamaValidator:
    """Local LLM validation gate using Ollama + Qwen3:4B.

    Usage:
        validator = OllamaValidator()
        result = validator.validate(market_data, proposed_signal, confidence)
        if not result.approved:
            # VETO — do not execute the trade
    """

    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        enabled: Optional[bool] = None,
    ):
        self.host = host or OLLAMA_HOST
        self.model = model or OLLAMA_MODEL
        self.enabled = enabled if enabled is not None else OLLAMA_ENABLED
        self._client = None
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="ollama_val"
        )
        self._call_count = 0
        self._veto_count = 0
        self._error_count = 0

    def _get_client(self):
        """Lazy-init the Ollama client."""
        if self._client is not None:
            return self._client
        try:
            from ollama import Client
            self._client = Client(host=self.host)
            return self._client
        except ImportError:
            log.warning(
                "[OllamaValidator] 'ollama' package not installed. "
                "Install with: pip install ollama"
            )
            return None

    def _strip_thinking(self, text: str) -> str:
        """Remove Qwen3 <think>...</think> blocks and markdown fences."""
        # Remove <think>...</think> blocks
        import re
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Remove ```json and ``` fences
        cleaned = re.sub(r'^```json\s*', '', cleaned.strip())
        cleaned = re.sub(r'\s*```$', '', cleaned.strip())
        return cleaned.strip()

    def _call_ollama(self, market_data: dict) -> Optional[dict]:
        """Call Ollama and parse the JSON response."""
        client = self._get_client()
        if client is None:
            return None

        response = client.chat(
            model=self.model,
            options=OLLAMA_OPTIONS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(market_data, default=str)},
            ],
        )

        raw_content = response.get("message", {}).get("content", "")
        if not raw_content:
            return None

        cleaned = self._strip_thinking(raw_content)

        # Try to find JSON in the response (handle potential wrapping)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to extract JSON object from the text
            import re
            json_match = re.search(r'\{[^}]+\}', cleaned, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            return None

    def validate(
        self,
        market_data: Dict[str, Any],
        proposed_signal: str,
        proposed_confidence: float,
        entry_price: float = 0.0,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        risk_reward: float = 0.0,
    ) -> OllamaValidationResult:
        """Run the Qwen3:4B institutional validation.

        Args:
            market_data: Full market context dict (symbol, timeframe, trend,
                         ml_confidence, spread, session, market_structure,
                         support_resistance, order_blocks, etc.)
            proposed_signal: The 4-layer consensus signal (BUY/SELL)
            proposed_confidence: The 4-layer consensus confidence (0-100)
            entry_price: Proposed entry price
            stop_loss: Proposed stop loss
            take_profit: Proposed take profit
            risk_reward: Proposed risk/reward ratio

        Returns:
            OllamaValidationResult — approved=True means the trade passes.
        """
        # Build the full context for the model
        context = {
            **market_data,
            "proposed_decision": proposed_signal,
            "proposed_confidence": proposed_confidence,
            "system_confidence": proposed_confidence,
        }
        if entry_price:
            context["proposed_entry"] = entry_price
        if stop_loss:
            context["proposed_sl"] = stop_loss
        if take_profit:
            context["proposed_tp"] = take_profit
        if risk_reward:
            context["proposed_rr"] = risk_reward

        # Fail-open: if disabled or signal is not a trade, skip
        if not self.enabled:
            return OllamaValidationResult(
                checked=False, available=True, approved=True,
                decision="SKIPPED", confidence=0, risk_level="",
                reason="Validator disabled via OLLAMA_VALIDATOR_ENABLED",
            )

        if proposed_signal not in ("BUY", "SELL"):
            return OllamaValidationResult(
                checked=False, available=True, approved=True,
                decision=proposed_signal, confidence=0, risk_level="",
                reason=f"Signal is {proposed_signal} — no validation needed",
            )

        self._call_count += 1
        t0 = time.monotonic()

        try:
            # Run with timeout to avoid blocking the trading loop
            future = self._executor.submit(self._call_ollama, context)
            result = future.result(timeout=OLLAMA_TIMEOUT_SEC)
            elapsed_ms = (time.monotonic() - t0) * 1000

            if result is None:
                self._error_count += 1
                log.warning(
                    f"[OllamaValidator] No valid JSON response "
                    f"(elapsed={elapsed_ms:.0f}ms) — fail-open"
                )
                return OllamaValidationResult(
                    checked=True, available=True, approved=True,
                    decision="PARSE_ERROR", confidence=0, risk_level="",
                    reason="Failed to parse Ollama response — fail-open",
                    response_time_ms=elapsed_ms,
                    error="parse_error",
                )

            decision = str(result.get("decision", "NO_TRADE")).upper()
            approved = str(result.get("approved", "false")).lower() in ("true", "1", "yes")
            confidence = float(result.get("confidence", 0))
            risk_level = str(result.get("risk_level", "MEDIUM")).upper()
            reason = str(result.get("reason", ""))
            warnings = list(result.get("warnings", []) or [])
            missing_info = list(result.get("missing_information", []) or [])
            entry = float(result.get("entry_price", 0) or 0)
            sl = float(result.get("stop_loss", 0) or 0)
            tp = float(result.get("take_profit", 0) or 0)
            rr = float(result.get("risk_reward", 0) or 0)
            market_struct = str(result.get("market_structure", ""))
            trend = str(result.get("trend", ""))

            # VETO logic: model can only reject, not promote
            # If model says NO_TRADE or disagrees with the proposed direction → veto
            veto = False
            veto_reason = ""
            if decision == "NO_TRADE":
                veto = True
                veto_reason = reason or "Model returned NO_TRADE"
            elif decision in ("BUY", "SELL") and decision != proposed_signal:
                # Model disagrees with direction → veto
                veto = True
                veto_reason = (
                    f"Direction mismatch: system={proposed_signal}, "
                    f"model={decision}"
                )
            elif not approved:
                veto = True
                veto_reason = reason or "Model did not approve"
            elif confidence < 40:
                # Even if approved, very low confidence is suspicious
                veto = True
                veto_reason = f"Model confidence too low: {confidence:.0f}%"
            elif risk_level == "HIGH":
                veto = True
                veto_reason = f"Model flagged HIGH risk: {reason}"
            elif rr > 0 and rr < 2.0:
                veto = True
                veto_reason = f"Model R:R too low: {rr:.1f} (min 2.0)"

            if veto:
                self._veto_count += 1
                log.warning(
                    f"[OllamaValidator] VETO — {proposed_signal} rejected | "
                    f"model={decision} conf={confidence:.0f}% risk={risk_level} | "
                    f"{veto_reason[:120]}"
                )
            else:
                log.info(
                    f"[OllamaValidator] APPROVED — {proposed_signal} confirmed | "
                    f"model={decision} conf={confidence:.0f}% risk={risk_level} | "
                    f"{reason[:80]}"
                )

            return OllamaValidationResult(
                checked=True, available=True, approved=not veto,
                decision=decision, confidence=confidence,
                risk_level=risk_level, reason=veto_reason if veto else reason,
                warnings=warnings, missing_information=missing_info,
                entry_price=entry, stop_loss=sl, take_profit=tp,
                risk_reward=rr, market_structure=market_struct,
                trend=trend, response_time_ms=elapsed_ms,
            )

        except FutureTimeoutError:
            self._error_count += 1
            elapsed_ms = (time.monotonic() - t0) * 1000
            log.warning(
                f"[OllamaValidator] Timeout after {elapsed_ms:.0f}ms — fail-open"
            )
            return OllamaValidationResult(
                checked=True, available=True, approved=True,
                decision="TIMEOUT", confidence=0, risk_level="",
                reason=f"Ollama timeout ({OLLAMA_TIMEOUT_SEC}s) — fail-open",
                response_time_ms=elapsed_ms,
                error="timeout",
            )
        except Exception as e:
            self._error_count += 1
            elapsed_ms = (time.monotonic() - t0) * 1000
            log.warning(
                f"[OllamaValidator] Error: {e} — fail-open"
            )
            return OllamaValidationResult(
                checked=True, available=False, approved=True,
                decision="ERROR", confidence=0, risk_level="",
                reason=f"Ollama error: {e}",
                response_time_ms=elapsed_ms,
                error=str(e),
            )

    def check_health(self) -> Dict[str, Any]:
        """Check if Ollama is reachable and the model is available."""
        result = {
            "enabled": self.enabled,
            "host": self.host,
            "model": self.model,
            "reachable": False,
            "model_loaded": False,
            "calls": self._call_count,
            "vetoes": self._veto_count,
            "errors": self._error_count,
        }
        if not self.enabled:
            return result
        try:
            client = self._get_client()
            if client is None:
                return result
            # Check if Ollama is running
            client.list()  # will raise if unreachable
            result["reachable"] = True
            # Check if model is available
            models = client.list()
            model_names = [m.get("name", "") for m in models.get("models", [])]
            result["model_loaded"] = any(
                self.model in name for name in model_names
            )
        except Exception as e:
            result["error"] = str(e)
        return result


# ── Singleton ───────────────────────────────────────────────────────

_VALIDATOR: Optional[OllamaValidator] = None


def get_ollama_validator() -> OllamaValidator:
    global _VALIDATOR
    if _VALIDATOR is None:
        _VALIDATOR = OllamaValidator()
    return _VALIDATOR
