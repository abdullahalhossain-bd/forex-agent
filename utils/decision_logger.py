"""
utils/decision_logger.py — Structured per-decision audit log
============================================================

Produces a fixed-format, parseable log block for every trading cycle
decision.  Designed for grep/awk parsing, dashboard ingestion, and
human audit of why the system chose what it chose.

Format (one block per decision):
    ╔══ DECISION AUDIT ═════════════════════════════════════╗
    ║ PAIR: EURUSD | TF: M15 | Session: LONDON
    ║ Trend: BULLISH(72%) | Pattern: bearish_engulfing
    ║ SMC: 78 (A) | Session: 85/100 (A)
    ║ Technical: 68% | ML: 74% | RL: 62% | LLM: 80%
    ║ Master: 82% | Fusion: 71% | Confluence: 76% (B+)
    ║ Risk: approved
    ║ Raw Confidence: 73.2% | Bonuses: sentiment_agree(+3)
    ║ Penalties: session_gate(-6), confluence_avoid(-12)
    ║ Final Confidence: 71% | Decision: BUY
    ║ Reason: MasterAnalyst: BUY | Rule: BUY (68%) | LLM: BUY (80%)
    ╚═══════════════════════════════════════════════════════╝

Usage (in trader.py, after _print_final):
    from utils.decision_logger import log_decision_block
    log_decision_block(result_dict, analysis_out)

The function only reads from dicts — it never mutates state or
raises exceptions (all extraction is .get-safe).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger("decision_logger")


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Coerce any value to float, returning default on failure."""
    try:
        if val is None:
            return default
        v = float(val)
        return round(v, 1)
    except (TypeError, ValueError):
        return default


def _safe_str(val: Any, default: str = "N/A") -> str:
    if val is None:
        return default
    return str(val)[:60]


def _extract_confidence_components(
    result: Dict[str, Any],
    analysis_out: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Extract all individual confidence values from the result dict and
    analysis_out.  Every field uses .get() with defaults so missing
    keys never crash.

    Returns a flat dict with all 19+ fields needed for the log block.
    """
    ao = analysis_out if isinstance(analysis_out, dict) else {}

    # --- Individual source confidences ---
    # Technical (rule engine / DecisionScorer)
    tech_conf = _safe_float(result.get("rule_conf"), 0)
    tech_signal = _safe_str(result.get("rule_signal"), "WAIT")

    # Classic LLM analyst (NOT MasterAnalyst)
    llm_conf = _safe_float(result.get("llm_conf"), 0)
    llm_signal = _safe_str(result.get("llm_signal"), "WAIT")

    # MasterAnalyst (LLM-synthesized — different from classic LLM)
    master_ctx = ao.get("master_ctx", {}) or {}
    master_conf = _safe_float(master_ctx.get("master_confidence"), 0)
    master_sig = _safe_str(master_ctx.get("master_signal"), "WAIT")

    # ML Ensemble
    ensemble_ctx = ao.get("ensemble", {}) or {}
    ml_conf = _safe_float(ensemble_ctx.get("confidence"), 0)
    ml_decision = _safe_str(ensemble_ctx.get("decision"), "WAIT")
    ml_available = bool(ensemble_ctx.get("ml_available", True))

    # RL Agent (0-1 scale stored, convert to 0-100)
    rl_ctx = ao.get("rl_agent", {}) or {}
    rl_conf_raw = _safe_float(rl_ctx.get("confidence"), 0)
    rl_conf = min(99.0, rl_conf_raw * 100) if rl_conf_raw <= 1.0 else rl_conf_raw
    rl_action = _safe_str(rl_ctx.get("action_name"), "HOLD")

    # SMC Engine
    smc_ctx = ao.get("smc_ctx", {}) or {}
    smc_score = _safe_float(smc_ctx.get("smc_score"), 0)
    smc_grade = _safe_str(smc_ctx.get("smc_grade"), "N/A")

    # Session Analyzer
    session_ctx = ao.get("session_ctx", {}) or {}
    session_score = _safe_float(session_ctx.get("session_score"), 0)
    session_conf = _safe_float(session_ctx.get("session_confidence"), 0)
    session_grade = _safe_str(session_ctx.get("session_grade"), "N/A")
    current_session = _safe_str(session_ctx.get("current_session"), "UNKNOWN")

    # Confluence Engine
    confluence_ctx = ao.get("confluence", {}) or {}
    confluence_conf = _safe_float(confluence_ctx.get("confidence"), 0)
    confluence_quality = _safe_str(confluence_ctx.get("setup_quality"), "UNKNOWN")
    aligned_factors = int(confluence_ctx.get("aligned_factors", 0) or 0)

    # Signal Fusion
    fusion_conf = 0.0
    # Fusion confidence is harder to extract — it's only in decision_agent's
    # internal variable. Check if dec_out included it.
    dec_out = result.get("_dec_out", {}) or {}
    fusion_conf = _safe_float(dec_out.get("_fusion_conf"), 0)

    # Risk status
    risk_approved = result.get("trade_allowed", False)
    risk_reason = _safe_str(result.get("reject_reason") or result.get("blocked_reason"), "")
    risk_status = "approved" if risk_approved else f"blocked:{risk_reason}"

    # Confidence penalties from signal validation
    signal_ctx = ao.get("signal", {}) or {}
    penalties = signal_ctx.get("confidence_penalties", []) or []
    penalty_strs = []
    for p in penalties if isinstance(penalties, list) else []:
        if isinstance(p, dict):
            penalty_strs.append(
                f"{p.get('source', '?')}:{p.get('reason', '?')}(-{p.get('amount', 0):.0f})"
            )

    # Bonuses (extracted from reasons list)
    bonus_strs = []
    reasons = result.get("reasons", []) or []
    for r in reasons:
        if isinstance(r, str) and ("+" in r or "boost" in r.lower() or "bonus" in r.lower()):
            # Extract the bonus part — look for (+N) or +N%
            import re
            matches = re.findall(r'\(\+[\d.]+\)', r) or re.findall(r'\+[\d.]+%', r)
            if matches:
                bonus_strs.append(f"{r[:50]}")

    # Confidence trace
    trace_entries = []
    try:
        from utils.confidence_trace import confidence_trace
        trace_entries = confidence_trace.to_list()
    except Exception:
        pass

    # Additional trace-based penalties not in the penalties list
    for entry in trace_entries:
        delta = entry.get("after", 0) - entry.get("before", 0)
        if delta < -0.5:  # meaningful reduction
            module = entry.get("module", "?")
            reason = entry.get("reason", "")
            # Avoid duplicates with existing penalty_strs
            if not any(module in ps for ps in penalty_strs):
                penalty_strs.append(f"{module}:{reason[:30]}({delta:+.0f})")

    return {
        "pair": _safe_str(result.get("symbol"), "UNKNOWN"),
        "timeframe": _safe_str(result.get("timeframe"), "M15"),
        "session": current_session,
        "session_score": session_score,
        "session_conf": session_conf,
        "session_grade": session_grade,
        "trend": _safe_str(result.get("trend"), "N/A"),
        "trend_conf": _safe_float(result.get("trend_confidence", 0), 0),
        "pattern": _safe_str(result.get("pattern"), "none"),
        "smc_score": smc_score,
        "smc_grade": smc_grade,
        "tech_conf": tech_conf,
        "tech_signal": tech_signal,
        "ml_conf": ml_conf,
        "ml_available": ml_available,
        "rl_conf": rl_conf,
        "llm_conf": llm_conf,
        "llm_signal": llm_signal,
        "master_conf": master_conf,
        "master_sig": master_sig,
        "fusion_conf": fusion_conf,
        "confluence_conf": confluence_conf,
        "confluence_quality": confluence_quality,
        "aligned_factors": aligned_factors,
        "risk_status": risk_status,
        "penalties": penalty_strs,
        "bonuses": bonus_strs,
        "raw_confidence": _safe_float(result.get("raw_confidence"), 0),
        "final_confidence": _safe_float(result.get("confidence"), 0),
        "decision": _safe_str(result.get("decision"), "WAIT"),
        "execution_action": _safe_str(
            result.get("execution_action") or result.get("final_action"), "WAIT"
        ),
        "reasons": reasons[:3] if isinstance(reasons, list) else [],
        "trace_entries": trace_entries,
    }


def log_decision_block(
    result: Dict[str, Any],
    analysis_out: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Emit the structured decision audit block to the logger.

    This is a pure logging function — it reads from the dicts, formats
    the block, and emits via logging.  It never mutates any state and
    never raises (all exceptions are caught and logged as warnings).

    Parameters
    ----------
    result : dict
        The result dict from ``trader.py:_build_result()``.  Must contain
        at least ``symbol``, ``timeframe``, ``confidence``, ``decision``.
    analysis_out : dict, optional
        The full analysis_out dict from ``analysis_agent.run()``.  When
        provided, individual confidence components (ML, RL, SMC, Session,
        Master, Confluence, Fusion) are extracted and logged.
    """
    try:
        comp = _extract_confidence_components(result, analysis_out)
    except Exception as e:
        log.warning(f"[DecisionLogger] Failed to extract components: {e}")
        return

    try:
        # Build the formatted block
        top = "\u2554\u2550\u2550 DECISION AUDIT \u2550" * 3 + "\u2557"
        bot = "\u255a\u2550" * 29 + "\u255d"
        row = "\u2551"

        lines = [top]
        lines.append(
            f"{row} PAIR: {comp['pair']} | TF: {comp['timeframe']} "
            f"| Session: {comp['session']} {row}"
        )
        lines.append(
            f"{row} Trend: {comp['trend']} | Pattern: {comp['pattern']} {row}"
        )
        lines.append(
            f"{row} SMC: {comp['smc_score']:.0f} ({comp['smc_grade']}) "
            f"| Session: {comp['session_score']:.0f}/100 ({comp['session_grade']}) {row}"
        )
        lines.append(
            f"{row} Technical: {comp['tech_conf']:.0f}% "
            f"| ML: {comp['ml_conf']:.0f}% "
            f"| RL: {comp['rl_conf']:.0f}% "
            f"| LLM: {comp['llm_conf']:.0f}% {row}"
        )
        lines.append(
            f"{row} Master: {comp['master_conf']:.0f}% "
            f"| Fusion: {comp['fusion_conf']:.0f}% "
            f"| Confluence: {comp['confluence_conf']:.0f}% ({comp['confluence_quality']}) {row}"
        )
        lines.append(f"{row} Risk: {comp['risk_status']} {row}")

        # Bonuses line
        if comp["bonuses"]:
            bonus_str = " | ".join(comp["bonuses"][:4])
            lines.append(f"{row} Bonuses: {bonus_str} {row}")

        # Penalties line
        if comp["penalties"]:
            penalty_str = ", ".join(comp["penalties"][:6])
            lines.append(f"{row} Penalties: {penalty_str} {row}")
        else:
            lines.append(f"{row} Penalties: none {row}")

        # Raw → Final confidence
        raw = comp["raw_confidence"]
        final = comp["final_confidence"]
        delta = final - raw
        if abs(delta) > 0.5:
            lines.append(
                f"{row} Raw Confidence: {raw:.1f}% -> {final:.1f}% "
                f"(delta {delta:+.1f}) {row}"
            )
        else:
            lines.append(f"{row} Final Confidence: {final:.1f}% {row}")

        # Decision + execution
        decision = comp["decision"]
        exec_action = comp["execution_action"]
        if exec_action != decision:
            lines.append(
                f"{row} Decision: {decision} -> Execution: {exec_action} {row}"
            )
        else:
            lines.append(f"{row} Decision: {decision} {row}")

        # Primary reason (first non-empty reason)
        for r in comp["reasons"]:
            if isinstance(r, str) and r.strip():
                # Truncate to fit in block
                reason_text = r[:70] + "..." if len(r) > 70 else r
                lines.append(f"{row} Reason: {reason_text} {row}")
                break

        lines.append(bot)

        block = "\n".join(lines)
        log.info(block)

    except Exception as e:
        log.warning(f"[DecisionLogger] Failed to format block: {e}")