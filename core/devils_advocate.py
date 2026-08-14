"""Final LLM Devil's Advocate review gate.

This module is intentionally non-signal-generating. It only reviews an
already-approved trade proposal and returns a structured
TAKE / REJECT / UNCERTAIN verdict. It never decides BUY/SELL, never changes
strategy parameters, and never trains itself.

--------------------------------------------------------------------------
2026-08-11 redesign (see review notes)
--------------------------------------------------------------------------
The previous version was a "post-gate EXECUTE/VETO filter" that defaulted
to EXECUTE whenever the model was uncertain or unreachable. That is
dangerous for a module whose entire purpose is to test whether the LLM
adds real edge: silently letting trades through on failure/uncertainty
contaminates any later analysis of "did the Devil's Advocate help?".

This version changes five things, in order of importance:

  1. Decision vocabulary is TAKE / REJECT / UNCERTAIN instead of
     EXECUTE / VETO. UNCERTAIN is never returned to the caller as-is —
     it is resolved by a deterministic, conservative policy
     (`_resolve_uncertain`) so the rest of the pipeline always sees a
     definite TAKE/REJECT, but the *raw* model output and the fact that
     it was uncertain are preserved in the result for audit purposes.
  2. The model is given structured, falsifiable evidence (HTF/LTF trend,
     market structure, location, momentum, execution cost, session, and
     whatever confluence/module signal is available) instead of a thin
     generic payload.
  3. The prompt explicitly separates "why this trade could work" from
     "why this trade could fail" and instructs the model to hunt for the
     strongest falsification of the thesis rather than justify it.
  4. The output schema is research-friendly: thesis_quality,
     counter_evidence_strength, expected_edge, risk_level, and a
     critical_failure slot, in addition to the decision itself.
  5. Every review is appended to an outcome-aware audit log
     (`memory/devils_advocate_audit.jsonl`) so that later analysis can
     join decisions to realized R-multiples and measure whether the
     model is removing losers or removing winners.

Design rules (unchanged):
  - Never generates trading signals or changes strategy state
  - Never bypasses existing risk/permission gates; it only runs after they
    have already approved the trade
  - Fail behavior is mode-aware: in "research"/"backtest" mode an
    unavailable/uncertain reviewer NEVER resolves to TAKE (that would be
    statistical contamination of the backtest). In "live" mode the
    resolution is configurable but defaults to the same conservative
    REJECT-on-uncertain behavior.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

log = get_logger("devils_advocate")

DECISION_TAKE = "TAKE"
DECISION_REJECT = "REJECT"
DECISION_UNCERTAIN = "UNCERTAIN"

_VALID_RAW_DECISIONS = {DECISION_TAKE, DECISION_REJECT, DECISION_UNCERTAIN}


class DevilsAdvocateGate:
    """Independent final reviewer for approved trades.

    Pipeline position:
        Strategy modules -> confluence/MTF/risk/session/execution filters ->
        TradePermission -> Devil's Advocate -> execution

    Design rules:
      - Never generates trading signals or changes strategy state
      - Decision vocabulary is TAKE / REJECT / UNCERTAIN. UNCERTAIN is
        always resolved to a definite TAKE/REJECT before being returned,
        using a conservative default policy ("do not take on uncertainty").
      - In research/backtest mode, a reviewer failure NEVER resolves to
        TAKE, to avoid contaminating expectancy statistics.
      - In live mode the failure resolution is configurable, defaulting to
        the same conservative behavior.
    """

    def __init__(
        self,
        enabled: Optional[bool] = None,
        fail_mode: Optional[str] = None,
        mode: Optional[str] = None,
        uncertain_policy: Optional[str] = None,
    ):
        # Audit fix (2026-08-09): this gate was fully implemented and
        # correctly wired into core/trader.py at the right pipeline
        # position (after ALL mandatory gates pass, immediately before
        # execution) but defaulted to OFF via DEVILS_ADVOCATE_ENABLED.
        # Per the audit requirement ("if ALL mandatory trade gates pass,
        # call the existing LLM Devil's Advocate before execution"),
        # default this ON. _should_run() only triggers a review after
        # every other gate (risk, permission, confidence, entry quality,
        # S/R, confluence, etc.) already passed — this can only add an
        # extra reject opportunity, never bypass an existing block.
        self.enabled = enabled if enabled is not None else self._env_flag("DEVILS_ADVOCATE_ENABLED", True)

        # "live" (default) vs "research"/"backtest". Research/backtest mode
        # is stricter: an unreachable/uncertain reviewer can never resolve
        # to TAKE, because that would silently let "no opinion" masquerade
        # as "approved" in expectancy/backtest statistics.
        self.mode = (mode or os.getenv("DEVILS_ADVOCATE_MODE", "live")).strip().lower()
        if self.mode not in {"live", "research", "backtest"}:
            self.mode = "live"

        # B4g fix: default was the string "fail_open", but the actual
        # resolution logic below never implemented an "open" (auto-TAKE)
        # path for it -- any value other than "fail_closed" just fell
        # through to uncertain_policy, which itself defaults to "reject".
        # So the *real* default behavior was always fail-closed; the env
        # var name was actively misleading (an operator reading
        # DEVILS_ADVOCATE_FAIL_MODE=fail_open would reasonably assume a
        # provider outage lets trades through, which it does not).
        # Fix: rename the default to "fail_closed" to match actual
        # behavior, and give "fail_open" a real effect in
        # _resolve_failure() below so the setting is no longer a no-op.
        self.fail_mode = (fail_mode or os.getenv("DEVILS_ADVOCATE_FAIL_MODE", "fail_closed")).lower()
        if self.fail_mode not in {"fail_open", "fail_closed"}:
            self.fail_mode = "fail_closed"

        # 2026-08-11 review fix: UNCERTAIN previously did not exist as a
        # state — anything short of an explicit VETO defaulted to EXECUTE.
        # The default policy now is conservative: uncertainty means "do
        # not take the trade" unless an operator explicitly opts into the
        # old permissive behavior via env/config.
        self.uncertain_policy = (uncertain_policy or os.getenv("DEVILS_ADVOCATE_UNCERTAIN_POLICY", "reject")).lower()
        if self.uncertain_policy not in {"reject", "take"}:
            self.uncertain_policy = "reject"

        self.timeout_sec = int(os.getenv("DEVILS_ADVOCATE_TIMEOUT_SEC", "6"))
        self.model_name = os.getenv("DEVILS_ADVOCATE_MODEL", "gpt-4.1-mini")
        self._last_error: Optional[str] = None
        self._audit = _DevilsAdvocateAuditLog()

    @staticmethod
    def _env_flag(name: str, default: bool) -> bool:
        value = os.getenv(name, "").strip().lower()
        if not value:
            return default
        return value in {"1", "true", "yes", "on"}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def review(
        self,
        trade_context: Dict[str, Any],
        signal: str,
        risk_out: Dict[str, Any],
        decision_out: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return a structured review result for an already-approved trade."""
        trade_id = trade_context.get("trade_id") or uuid.uuid4().hex[:12]

        if not self.enabled:
            result = self._default_result(DECISION_TAKE, 0.0, risk_summary="LLM reviewer disabled")
            self._audit.record(trade_id, trade_context, signal, risk_out, decision_out, result)
            return result

        if not self._should_run(trade_context, signal, risk_out, decision_out):
            result = self._default_result(DECISION_TAKE, 0.0, risk_summary="Not a reviewable trade")
            self._audit.record(trade_id, trade_context, signal, risk_out, decision_out, result)
            return result

        thesis = self._build_thesis(trade_context, signal, risk_out, decision_out)
        evidence = self._build_evidence(trade_context, signal, risk_out, decision_out)

        try:
            payload = self._build_payload(thesis, evidence, signal)
            response = self._call_provider(payload)
            parsed = self._normalize_response(response)
            result = self._finalize(parsed, data_quality="good")
        except Exception as exc:
            self._last_error = str(exc)
            log.warning(f"[DevilsAdvocate] reviewer unavailable: {exc}")
            result = self._resolve_failure(str(exc))

        self._audit.record(trade_id, trade_context, signal, risk_out, decision_out, result, evidence=evidence, thesis=thesis)
        return result

    # ------------------------------------------------------------------
    # Gate / eligibility
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Thesis + evidence construction
    # ------------------------------------------------------------------
    @staticmethod
    def _g(d: Optional[Dict[str, Any]], *path: str, default: Any = "unknown") -> Any:
        """Safely walk a nested dict path, tolerating missing/None keys."""
        cur: Any = d or {}
        for key in path:
            if not isinstance(cur, dict):
                return default
            cur = cur.get(key)
            if cur is None:
                return default
        return cur if cur is not None else default

    @staticmethod
    def _first_present(d: Optional[Dict[str, Any]], *keys: str, default: Any = None) -> Any:
        """Return the first key that is present in d (even if falsy), else default.

        Unlike `d.get(a) or d.get(b)`, this does NOT treat a real empty
        string / 0 / False as "missing" and fall through to the next key.
        """
        if not isinstance(d, dict):
            return default
        for key in keys:
            if key in d and d[key] is not None:
                return d[key]
        return default

    def _build_thesis(self, trade_context: Dict[str, Any], signal: str, risk_out: Dict[str, Any], decision_out: Dict[str, Any]) -> Dict[str, Any]:
        """Build the explicit "why this trade" thesis the model has to attack.

        A caller can supply a fully custom thesis via
        trade_context["devils_advocate_thesis"] (list[str] of claims) if it
        already has richer reasoning available; otherwise this builds a
        best-effort thesis from whatever context fields are present.
        """
        custom = trade_context.get("devils_advocate_thesis")
        if isinstance(custom, list) and custom:
            return {"symbol": trade_context.get("symbol", "UNKNOWN"), "signal": signal, "claims": custom}

        symbol = trade_context.get("symbol") or trade_context.get("pair") or "UNKNOWN"

        # B4f fix: previously used decision_out.get("mtf_bias") -- a
        # separate summary field that can desync from the actual per-TF
        # mtf_trends dict used to build `evidence` (see B4d fix below).
        # In production this produced thesis claims like "bias is NEUTRAL"
        # sitting right next to evidence showing h1_trend=strong_bearish /
        # h4_trend=sideways for the same review, which both confuses the
        # reviewing LLM and creates duplicate/contradictory
        # "Higher-timeframe bias..." entries in its supporting/
        # contradicting evidence output. Its fallback also looked up the
        # wrong key ("h4" instead of "4h"), so if mtf_bias were ever
        # absent it silently degraded to "unknown".
        # Fix: derive the thesis claim from the same mtf_trends dict (and
        # the same "4h"/"1h" keys) that evidence uses, so thesis and
        # evidence can never disagree about what the HTF trend actually is.
        mtf_trends_for_thesis = decision_out.get("mtf_trends") or {}
        h4_bias = mtf_trends_for_thesis.get("4h", "unknown")
        h1_bias = mtf_trends_for_thesis.get("1h", "unknown")
        if h4_bias != "unknown":
            mtf_bias = h4_bias
        elif h1_bias != "unknown":
            mtf_bias = h1_bias
        else:
            mtf_bias = decision_out.get("mtf_bias", "unknown")

        structure_ctx = decision_out.get("structure_ctx") or {}
        smc_ctx = decision_out.get("smc_ctx") or {}
        sr_ctx = decision_out.get("sr_ctx") or {}

        claims: List[str] = []
        sig = str(signal).upper()
        # Only claim HTF alignment when bias actually agrees with the signal.
        # "sideways/neutral aligned with SELL" was confusing the reviewer into
        # inventing contradictions or treating ranging HTF as support.
        _bullish_words = ("bull", "buy", "long")
        _bearish_words = ("bear", "sell", "short")
        _mtf_l = str(mtf_bias).lower()
        _htf_agrees = (
            (sig == "BUY" and any(w in _mtf_l for w in _bullish_words))
            or (sig == "SELL" and any(w in _mtf_l for w in _bearish_words))
        )
        if mtf_bias and mtf_bias != "unknown" and _htf_agrees:
            claims.append(f"Higher-timeframe bias is {mtf_bias}, aligned with the {signal} signal")
        elif mtf_bias and str(mtf_bias).lower() in ("sideways", "neutral", "ranging"):
            claims.append(f"Higher-timeframe is {mtf_bias} (no strong opposing HTF trend against {signal})")

        if structure_ctx:
            bos = self._first_present(
                structure_ctx, "bos", "structure_bos", "break_of_structure"
            )
            choch = self._first_present(
                structure_ctx, "choch", "structure_choch", "change_of_character"
            )
            bos_u = str(bos or "").upper()
            choch_u = str(choch or "").upper()
            # Only claim BOS/CHoCH when they SUPPORT the signal direction.
            # Claiming BULLISH_BOS on a SELL thesis made the reviewer reject
            # every counter-structure trade (correct) AND also polluted
            # aligned trades' thesis with noise.
            if bos_u not in ("NONE", "UNKNOWN", ""):
                if (sig == "BUY" and "BULLISH" in bos_u) or (sig == "SELL" and "BEARISH" in bos_u):
                    claims.append(f"Break of structure supports {signal}: {bos}")
            if choch_u not in ("NONE", "UNKNOWN", ""):
                if (sig == "BUY" and "BULLISH" in choch_u) or (sig == "SELL" and "BEARISH" in choch_u):
                    claims.append(f"Change of character supports {signal}: {choch}")
        if smc_ctx.get("liquidity_sweep"):
            claims.append(f"Liquidity sweep observed: {smc_ctx.get('liquidity_sweep')}")
        if sr_ctx:
            claims.append("Entry is positioned relative to a mapped support/resistance or supply/demand zone")
        rr = risk_out.get("rr_ratio")
        if rr:
            claims.append(f"Reward-to-risk ratio of {rr} clears the minimum bar")
        if not claims:
            claims.append(f"Strategy modules produced a {signal} signal that passed all mandatory gates")

        return {"symbol": symbol, "signal": signal, "claims": claims}

    def _build_evidence(self, trade_context: Dict[str, Any], signal: str, risk_out: Dict[str, Any], decision_out: Dict[str, Any]) -> Dict[str, Any]:
        """Assemble structured, falsifiable evidence for the model.

        A caller can pass a fully custom evidence dict via
        trade_context["devils_advocate_evidence"] to override/extend any of
        this (e.g. once richer module-vote data is wired through). Missing
        fields are reported as "unknown" rather than omitted, so the model
        is explicitly told what it does NOT know instead of silently
        inferring it.
        """
        market_context = trade_context.get("market_context") or {}
        mtf_trends = decision_out.get("mtf_trends") or {}
        ind_ctx = decision_out.get("ind_ctx") or {}
        structure_ctx = decision_out.get("structure_ctx") or {}
        smc_ctx = decision_out.get("smc_ctx") or {}
        liquidity_ctx = decision_out.get("liquidity_ctx") or {}
        sr_ctx = decision_out.get("sr_ctx") or {}
        regime = decision_out.get("regime") or {}
        analysis_out = trade_context.get("analysis_out")
        confluence_ctx = analysis_out.get("confluence") if isinstance(analysis_out, dict) else None

        entry = risk_out.get("entry")
        sl = risk_out.get("sl_price")
        tp = risk_out.get("tp_price")
        atr = ind_ctx.get("atr")
        spread_pips = ind_ctx.get("spread_pips")

        sl_distance = None
        try:
            if entry is not None and sl is not None:
                sl_distance = abs(float(entry) - float(sl))
        except (TypeError, ValueError):
            pass

        # B4e fix: spread_atr_ratio was computing spread_pips (a pip count,
        # e.g. 1.2) divided directly by atr (a raw price delta, e.g.
        # 0.00081). That's a unit mismatch -- pips vs price -- which
        # produces a number with no real meaning even when both inputs
        # are valid.
        # On top of that, spread_pips == 0 (missing/upstream-default, not
        # a genuine zero spread) silently produced ratio = 0.0, which the
        # LLM then read as a red flag ("near-zero spread -> high risk")
        # instead of being told the data was absent.
        #
        # Fix: convert atr into pips using the correct pip size for the
        # symbol before dividing, and treat spread_pips == 0 as missing
        # data ("unknown") rather than a real zero-cost spread, since a
        # true zero spread does not occur in live FX feeds.
        symbol_for_pip = trade_context.get("symbol") or trade_context.get("pair") or ""
        pip_size = 0.01 if "JPY" in str(symbol_for_pip).upper() else 0.0001

        spread_atr_ratio = "unknown"
        try:
            if spread_pips not in (None, 0) and atr not in (None, 0):
                atr_pips = float(atr) / pip_size
                if atr_pips:
                    spread_atr_ratio = round(float(spread_pips) / atr_pips, 4)
        except (TypeError, ValueError, ZeroDivisionError):
            pass

        news_risk = "unknown"
        if isinstance(analysis_out, dict):
            news_ctx = analysis_out.get("news_ctx")
            if isinstance(news_ctx, dict):
                news_risk = news_ctx.get("risk_level", "unknown")

        # ── Normalize upstream key aliases ─────────────────────────────
        # structure_engine.get_ai_context uses structure_bos / structure_choch;
        # DA historically only looked for bos / choch. Also accept values
        # placed on trade_context["market_context"] by trader.py (2026-08-13)
        # when decision_out["mtf_trends"] is empty or uses H4/H1 labels.
        def _tf(*keys, fallback="unknown"):
            for k in keys:
                v = mtf_trends.get(k) if isinstance(mtf_trends, dict) else None
                if v not in (None, "", "unknown"):
                    return v
                v = market_context.get(k) if isinstance(market_context, dict) else None
                if v not in (None, "", "unknown"):
                    return v
            # common alternate names on market_context
            for k in keys:
                alt = {
                    "4h": ("h4_trend", "H4", "h4"),
                    "1h": ("h1_trend", "H1", "h1"),
                    "15m": ("m15_trend", "M15", "m15", "m15_structure"),
                }.get(k, ())
                for a in alt:
                    v = mtf_trends.get(a) if isinstance(mtf_trends, dict) else None
                    if v not in (None, "", "unknown"):
                        return v
                    v = market_context.get(a) if isinstance(market_context, dict) else None
                    if v not in (None, "", "unknown"):
                        return v
            return fallback

        _bos = self._first_present(
            structure_ctx, "bos", "structure_bos", "break_of_structure", default="unknown"
        )
        _choch = self._first_present(
            structure_ctx, "choch", "structure_choch", "change_of_character", default="unknown"
        )
        _nearest_zone = self._first_present(
            sr_ctx,
            "nearest_zone", "location", "nearest_support", "nearest_resistance",
            default="unknown",
        )
        # Prefer market_context SR hints if sr_ctx is empty
        if _nearest_zone in (None, "", "unknown") and isinstance(market_context, dict):
            _nearest_zone = (
                market_context.get("sr_zone")
                or market_context.get("nearest_support")
                or market_context.get("nearest_resistance")
                or "unknown"
            )

        evidence: Dict[str, Any] = {
            "htf": {
                # B4d + 2026-08-13: accept "4h"/"1h"/"15m" and H4/h4_trend aliases
                # so empty mtf_trends no longer forces every review to REJECT.
                "h4_trend": _tf("4h"),
                "h1_trend": _tf("1h"),
                "m15_structure": _tf("15m", fallback=structure_ctx.get("m15", "unknown")),
            },
            "structure": {
                "bos": _bos if _bos not in (None, "") else "unknown",
                "choch": _choch if _choch not in (None, "") else "unknown",
                "liquidity_sweep": (
                    smc_ctx.get("liquidity_sweep")
                    or liquidity_ctx.get("sweep")
                    or structure_ctx.get("liquidity_sweep")
                    or "unknown"
                ),
                "displacement": (
                    smc_ctx.get("displacement")
                    or structure_ctx.get("displacement")
                    or structure_ctx.get("displacement_dir")
                    or "unknown"
                ),
            },
            "location": {
                "pdh": sr_ctx.get("pdh", "unknown"),
                "pdl": sr_ctx.get("pdl", "unknown"),
                "prev_week_high": sr_ctx.get("prev_week_high", "unknown"),
                "prev_week_low": sr_ctx.get("prev_week_low", "unknown"),
                "session_high": sr_ctx.get("session_high", "unknown"),
                "session_low": sr_ctx.get("session_low", "unknown"),
                "support_resistance_zone": _nearest_zone,
                "supply_demand_zone": smc_ctx.get("supply_demand_zone", "unknown"),
            },
            "momentum": {
                "atr": atr if atr is not None else "unknown",
                "candle_size": ind_ctx.get("last_candle_range", "unknown"),
                "volatility_regime": regime.get("regime", market_context.get("volatility", "unknown")),
            },
            "execution": {
                # Treat 0 / 0.0 as missing (never a real FX spread) so the
                # LLM does not invent a "spread_to_atr_ratio is 0.0" critical
                # failure from absent upstream data.
                "spread_pips": (
                    spread_pips if spread_pips not in (None, 0, 0.0) else "unknown"
                ),
                "spread_to_atr_ratio": spread_atr_ratio,
                "estimated_slippage": trade_context.get("estimated_slippage", "unknown"),
                "sl_distance": sl_distance if sl_distance is not None else "unknown",
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "rr_ratio": risk_out.get("rr_ratio", "unknown"),
            },
            "context": {
                "session": market_context.get("session", "unknown"),
                "market_regime": regime.get("regime", "unknown"),
                "news_risk": news_risk,
                "timeframe": market_context.get("timeframe", "unknown"),
            },
            "signal_evidence": {
                "modules_supporting": (confluence_ctx or {}).get("passed_modules", "unknown") if isinstance(confluence_ctx, dict) else "unknown",
                "modules_opposing": (confluence_ctx or {}).get("failed_modules", "unknown") if isinstance(confluence_ctx, dict) else "unknown",
                "confluence_score": (confluence_ctx or {}).get("score", "unknown") if isinstance(confluence_ctx, dict) else "unknown",
                "strategy_confidence": decision_out.get("confidence", "unknown"),
            },
        }

        custom = trade_context.get("devils_advocate_evidence")
        if isinstance(custom, dict):
            for k, v in custom.items():
                if isinstance(v, dict) and isinstance(evidence.get(k), dict):
                    evidence[k].update(v)
                else:
                    evidence[k] = v

        return evidence

    # ------------------------------------------------------------------
    # Provider call
    # ------------------------------------------------------------------
    def _build_payload(self, thesis: Dict[str, Any], evidence: Dict[str, Any], signal: str) -> Dict[str, Any]:
        return {
            "role": "devils_advocate",
            "instruction": (
                "You are an independent Devil's Advocate reviewer for an already-generated "
                "trade thesis. Do not generate a signal, do not predict direction, and do not "
                "replace the strategy. Assume the proposed direction may be wrong: you are not "
                "rewarded for agreeing with the thesis. Your only task is to find the strongest "
                "falsification of the thesis using the evidence provided, and to state plainly "
                "when the evidence is too thin to have an opinion (UNCERTAIN). "
                "IMPORTANT policy on RANGING markets: a RANGING or sideways regime alone is "
                "NOT sufficient grounds to REJECT when evidence.structure.bos / "
                "evidence.structure.choch ACTUALLY agree with the proposed signal direction "
                "(check the real evidence field, not just the absence of an opposing claim in "
                "the thesis) AND a mapped S/R or supply/demand zone is present AND "
                "reward-to-risk meets the minimum. In that case prefer TAKE (or UNCERTAIN if "
                "other hard contradictions exist). Do REJECT when structure breaks the "
                "opposite way of the signal, or HTF trend is strongly against the signal, or "
                "the signal is entering at/near a swing level that would invalidate it on a "
                "normal continuation. A thesis whose only claim is 'no strong opposing HTF "
                "trend' (an absence-of-opposition claim, not real support) is NOT sufficient "
                "grounds for TAKE by itself. "
                "CONSISTENCY REQUIREMENT (hard rule, checked programmatically downstream): "
                "your `expected_edge` field must agree with your `decision`. Never return "
                "decision=TAKE together with expected_edge=negative, and never return "
                "decision=REJECT together with expected_edge=positive. If your honest edge "
                "assessment is negative or unclear, your decision must be REJECT or UNCERTAIN, "
                "not TAKE."
            ),
            "trade_thesis": thesis,
            "evidence": evidence,
            "instructions_detail": {
                "task_1": "List concrete SUPPORTING evidence for why this trade could work.",
                "task_2": "List concrete CONTRADICTING evidence for why this trade could fail. "
                           "This is the primary task -- actively search for it, do not pad it "
                           "with the mere absence of supporting evidence. Do not treat "
                           "'regime is RANGING' as a critical failure by itself when BOS/CHoCH "
                           "agree with the signal.",
                "task_3": "Decide TAKE if supporting structure (direction-aligned BOS/CHoCH and "
                          "location) outweighs contradictions — including in RANGING regimes. "
                          "Decide REJECT if structure or HTF clearly opposes the signal, or "
                          "contradicting evidence materially outweighs support. Decide UNCERTAIN "
                          "if evidence is missing/degraded/mixed enough that no confident call "
                          "can be made.",
                "output_schema": {
                    "decision": "TAKE|REJECT|UNCERTAIN",
                    "confidence": 0.0,
                    "thesis_quality": 0.0,
                    "counter_evidence_strength": 0.0,
                    "expected_edge": "positive|neutral|negative|unknown",
                    "risk_level": "low|medium|high",
                    "supporting_evidence": [],
                    "contradicting_evidence": [],
                    "reasons_for_rejection": [],
                    "critical_failure": None,
                },
            },
        }

    def _call_provider(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Call a real LLM (Groq primary, Gemini fallback) to review the trade.

        Uses the same LLMKeyManager multi-key rotation the rest of the stack
        relies on (ai/ai_analyst.py), so it benefits from key failover
        without needing its own credential handling. Returns a dict matching
        the ``output_schema`` in the payload; raises on total failure so the
        caller's failure-resolution logic in ``review()`` takes over.

        Note: this deliberately still calls a single provider chain
        (Groq -> Gemini as availability fallback, not a disagreement vote)
        rather than querying both models for every trade. Cross-model
        disagreement detection is a promising extension but should only be
        added after the single-model version has demonstrated statistical
        value -- see review notes point 8.
        """
        prompt = self._build_prompt(payload)
        deadline = time.monotonic() + self.timeout_sec

        from core.llm_key_manager import get_llm_key_manager, log_llm_call_failure
        manager = get_llm_key_manager()

        last_error: Optional[Exception] = None

        for provider in ("groq", "gemini"):
            if time.monotonic() >= deadline:
                break
            try:
                raw = self._call_one_provider(manager, provider, prompt, deadline)
                if raw is None:
                    continue
                from utils.llm_json import parse_llm_json
                parsed = parse_llm_json(raw)
                if isinstance(parsed, dict) and parsed.get("decision") in _VALID_RAW_DECISIONS:
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
            # B4h fix: silently swapping any "gpt*"-named model for
            # llama-3.1-8b-instant (Groq doesn't host GPT models) used to
            # happen with no trace anywhere. An operator setting
            # DEVILS_ADVOCATE_MODEL=gpt-4o expecting that model would get
            # llama-3.1-8b-instant instead with nothing in the logs to
            # explain why. Fix: keep the same override (Groq still can't
            # serve GPT models) but log it once so it's visible.
            groq_model = self.model_name
            if "gpt" in groq_model:
                groq_model = "llama-3.1-8b-instant"
                log.warning(
                    f"[DevilsAdvocate] configured model '{self.model_name}' is not "
                    f"available on Groq; using '{groq_model}' instead"
                )
            create_kwargs = dict(
                model=groq_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=700,
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
            f"Trade thesis (the claims to attack):\n{json.dumps(payload['trade_thesis'], default=str)}\n\n"
            f"Evidence:\n{json.dumps(payload['evidence'], default=str)}\n\n"
            f"Tasks:\n"
            f"1. {payload['instructions_detail']['task_1']}\n"
            f"2. {payload['instructions_detail']['task_2']}\n"
            f"3. {payload['instructions_detail']['task_3']}\n\n"
            "Respond with ONLY a JSON object (no markdown fences, no prose) matching exactly "
            f"this schema: {json.dumps(payload['instructions_detail']['output_schema'])}"
        )

    # ------------------------------------------------------------------
    # Response handling
    # ------------------------------------------------------------------
    def _normalize_response(self, response: Any) -> Dict[str, Any]:
        if isinstance(response, str):
            try:
                return json.loads(response)
            except Exception:
                return {}
        if isinstance(response, dict):
            return response
        return {}

    def _finalize(self, parsed: Dict[str, Any], data_quality: str) -> Dict[str, Any]:
        raw_decision = str(parsed.get("decision", DECISION_UNCERTAIN)).upper()
        if raw_decision not in _VALID_RAW_DECISIONS:
            raw_decision = DECISION_UNCERTAIN

        resolved = raw_decision
        critical_failure = parsed.get("critical_failure")
        if raw_decision == DECISION_UNCERTAIN:
            resolved = self._resolve_uncertain()

        expected_edge_raw = parsed.get("expected_edge") or "unknown"
        expected_edge_l = str(expected_edge_raw).lower()
        thesis_quality = float(parsed.get("thesis_quality", 0.0) or 0.0)
        counter_evidence_strength = float(parsed.get("counter_evidence_strength", 0.0) or 0.0)

        # P0 fix (2026-08-14 forensic audit, EURUSD eval_1786641545_EURUSD_fe0d708e):
        # that trade's raw model output was decision=TAKE, expected_edge=negative,
        # reasons_for_rejection=[] -- a self-contradictory verdict that executed
        # anyway because expected_edge/counter_evidence_strength were recorded for
        # audit purposes but never fed back into the resolved decision. Close that
        # gap here: if the model's own structured fields contradict a TAKE, don't
        # trust the raw label -- run it through the same conservative resolution
        # path as an UNCERTAIN verdict (REJECT by default; see _resolve_uncertain).
        # This is a general consistency check, not a rule tailored to this one
        # trade -- it fires on any TAKE whose own expected_edge is negative, or
        # whose own counter_evidence_strength meets/exceeds its own thesis_quality.
        contradiction_reason: Optional[str] = None
        if resolved == DECISION_TAKE:
            if expected_edge_l == "negative":
                contradiction_reason = "expected_edge=negative on a TAKE decision"
            elif counter_evidence_strength > 0.0 and counter_evidence_strength >= thesis_quality:
                contradiction_reason = (
                    f"counter_evidence_strength={counter_evidence_strength} >= "
                    f"thesis_quality={thesis_quality} on a TAKE decision"
                )
        if contradiction_reason:
            log.warning(f"[DevilsAdvocate] TAKE overridden -- internal contradiction: {contradiction_reason}")
            resolved = self._resolve_contradiction()
            if not critical_failure:
                critical_failure = f"internal_contradiction: {contradiction_reason}"

        supporting = parsed.get("supporting_evidence") or []
        contradicting = parsed.get("contradicting_evidence") or []
        reasons_for_rejection = parsed.get("reasons_for_rejection") or []

        risk_summary = self._summarize(resolved, raw_decision, reasons_for_rejection, critical_failure)

        return {
            "decision": resolved,
            "raw_decision": raw_decision,
            "confidence": float(parsed.get("confidence", 0.0) or 0.0),
            "thesis_quality": thesis_quality,
            "counter_evidence_strength": counter_evidence_strength,
            "expected_edge": expected_edge_raw,
            "risk_level": parsed.get("risk_level") or "unknown",
            "supporting_evidence": supporting,
            "contradicting_evidence": contradicting,
            "reasons_for_rejection": reasons_for_rejection,
            "critical_failure": critical_failure,
            "data_quality": data_quality,
            # legacy-compatible fields other callers/tests may still read
            "reasons_for_concern": reasons_for_rejection,
            "risk_summary": risk_summary,
            "evidence": list(supporting) + list(contradicting),
        }

    def _resolve_uncertain(self) -> str:
        """Deterministically resolve UNCERTAIN into TAKE/REJECT.

        Research/backtest mode always rejects on uncertainty regardless of
        the configured policy, to avoid contaminating expectancy stats with
        "no opinion" masquerading as approval.
        """
        if self.mode in {"research", "backtest"}:
            return DECISION_REJECT
        return DECISION_TAKE if self.uncertain_policy == "take" else DECISION_REJECT

    @staticmethod
    def _resolve_contradiction() -> str:
        """Resolve a TAKE whose own structured fields contradict it
        (expected_edge=negative, or counter_evidence_strength >=
        thesis_quality -- see _finalize).

        Deliberately NOT routed through _resolve_uncertain(): that path
        honors self.uncertain_policy, which is a legitimate operator
        preference for genuine model uncertainty ("no opinion" -> take or
        reject, operator's call). A self-contradictory TAKE is not "no
        opinion" -- it's the model asserting two incompatible things in
        the same response, which is a data-integrity failure. If this
        reused _resolve_uncertain(), setting
        DEVILS_ADVOCATE_UNCERTAIN_POLICY=take (a supported, documented
        value) would silently resolve the contradiction back to TAKE and
        defeat the entire P0 fix -- exactly reproducing the 2026-08-13
        EURUSD incident this check exists to prevent. Always REJECT here,
        unconditionally, in every mode.
        """
        return DECISION_REJECT

    def _resolve_failure(self, error_text: str) -> Dict[str, Any]:
        """Resolve a provider-call failure (timeout, no key, bad JSON, etc.).

        Distinct from _finalize's UNCERTAIN path in that the model was never
        actually reached, so data_quality is always "poor" and a
        critical_failure note is always attached.
        """
        # Conservative default: provider failures resolve to REJECT in
        # all modes unless the operator explicitly opts into the unsafe
        # combination: `fail_mode='fail_open'` AND
        # `uncertain_policy='take'`.
        # Research/backtest MUST NEVER auto-TAKE on reviewer outage —
        # that would contaminate simulated expectancy. Only allow TAKE
        # in live mode when both opt-ins are present.
        if self.mode == "research":
            resolved = DECISION_REJECT
        elif self.fail_mode == "fail_open" and self.uncertain_policy == "take":
            resolved = DECISION_TAKE
        else:
            resolved = DECISION_REJECT

        return {
            "decision": resolved,
            "raw_decision": DECISION_UNCERTAIN,
            "confidence": 0.0,
            "thesis_quality": 0.0,
            "counter_evidence_strength": 0.0,
            "expected_edge": "unknown",
            "risk_level": "unknown",
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "reasons_for_rejection": [],
            "critical_failure": f"reviewer_unavailable: {error_text}",
            "data_quality": "poor",
            "reasons_for_concern": [],
            "risk_summary": f"Reviewer unavailable ({self.mode} mode, {self.fail_mode}) -> {resolved}",
            "evidence": [],
        }

    @staticmethod
    def _summarize(resolved: str, raw: str, reasons: List[str], critical_failure: Optional[str]) -> str:
        if critical_failure:
            return f"{resolved}: {critical_failure}"
        if raw == DECISION_UNCERTAIN and resolved != raw:
            return f"{resolved} (model was UNCERTAIN; resolved conservatively)"
        if resolved == DECISION_REJECT:
            return "; ".join(reasons[:3]) if reasons else "Contradicting evidence outweighed thesis"
        return "Thesis survived adversarial review" if raw == DECISION_TAKE else f"{resolved}"

    def _default_result(self, decision: str, confidence: float, risk_summary: str) -> Dict[str, Any]:
        return {
            "decision": decision,
            "raw_decision": decision,
            "confidence": confidence,
            "thesis_quality": 0.0,
            "counter_evidence_strength": 0.0,
            "expected_edge": "unknown",
            "risk_level": "unknown",
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "reasons_for_rejection": [],
            "critical_failure": None,
            "data_quality": "n/a",
            "reasons_for_concern": [],
            "risk_summary": risk_summary,
            "evidence": [],
        }


class _DevilsAdvocateAuditLog:
    """Outcome-aware audit journal for every Devil's Advocate review.

    Appends one JSON line per review to memory/devils_advocate_audit.jsonl
    with the full decision snapshot (evidence, thesis, decision, confidence)
    but WITHOUT the eventual trade outcome, since that isn't known yet. A
    separate offline job is expected to join these rows to realized
    R-multiples (by trade_id) once trades close, producing the
    TAKE->WIN / TAKE->LOSS / REJECT->WIN / REJECT->LOSS breakdown needed to
    tell whether the reviewer is removing losers or removing winners.

    Never raises: a logging failure must not affect trading.
    """

    def __init__(self) -> None:
        try:
            from core.constants import MEMORY_DIR
            self.path = MEMORY_DIR / "devils_advocate_audit.jsonl"
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            self.path = None

    def record(
        self,
        trade_id: str,
        trade_context: Dict[str, Any],
        signal: str,
        risk_out: Dict[str, Any],
        decision_out: Dict[str, Any],
        result: Dict[str, Any],
        evidence: Optional[Dict[str, Any]] = None,
        thesis: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.path is None:
            return
        try:
            market_context = trade_context.get("market_context") or {}
            row = {
                "trade_id": trade_id,
                "timestamp": time.time(),
                "symbol": trade_context.get("symbol") or trade_context.get("pair"),
                "timeframe": market_context.get("timeframe"),
                "signal": signal,
                "llm_decision": result.get("decision"),
                "llm_raw_decision": result.get("raw_decision"),
                "llm_confidence": result.get("confidence"),
                "thesis_quality": result.get("thesis_quality"),
                "counter_evidence_strength": result.get("counter_evidence_strength"),
                "expected_edge": result.get("expected_edge"),
                "risk_level": result.get("risk_level"),
                "data_quality": result.get("data_quality"),
                "entry": risk_out.get("entry"),
                "sl": risk_out.get("sl_price"),
                "tp": risk_out.get("tp_price"),
                "rr_ratio": risk_out.get("rr_ratio"),
                "session": market_context.get("session"),
                "regime": (evidence or {}).get("momentum", {}).get("volatility_regime") if evidence else None,
                "atr": (evidence or {}).get("momentum", {}).get("atr") if evidence else None,
                "spread_pips": (evidence or {}).get("execution", {}).get("spread_pips") if evidence else None,
                "supporting_evidence": result.get("supporting_evidence"),
                "contradicting_evidence": result.get("contradicting_evidence"),
                "reasons_for_rejection": result.get("reasons_for_rejection"),
                "critical_failure": result.get("critical_failure"),
                "thesis_claims": (thesis or {}).get("claims"),
                # Filled in later by an offline join against realized trades.
                "future_outcome": None,
                "r_multiple": None,
            }
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
        except Exception as exc:
            log.warning(f"[DevilsAdvocate] audit log write failed (non-fatal): {exc}")