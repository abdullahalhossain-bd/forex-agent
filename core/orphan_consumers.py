"""
core/orphan_consumers.py — Real consumers for the Phase-25 orphan services.

BACKGROUND
==========
Phase 25 (`core/_orphan_integration.py`) registered 83 previously-orphaned
production modules into the `ServiceRegistry`. But registration alone is
NOT consumption: those services sat in the registry with zero callers,
so they had **zero impact on trade decisions**.

This module closes that gap. It exposes four hooks that the live decision
pipeline (`core/trader.py::evaluate_decision_core` and `run_cycle`) calls
at the correct stages:

    ┌──────────────────────────────────────────────────────────────────┐
    │ STAGE                   HOOK                            SERVICES  │
    ├──────────────────────────────────────────────────────────────────┤
    │ 1. Pre-analysis enrich  enrich_market_context()         4 svcs    │
    │ 2. Post-decision score  apply_signal_scoring()          6 svcs    │
    │ 3. Pre-execution risk   apply_advanced_risk_gates()     10 svcs   │
    │ 4. Final veto gate      final_decision_gate()           4 svcs    │
    └──────────────────────────────────────────────────────────────────┘

Every hook is **fail-safe**: each service is wrapped in its own
try/except, and any failure logs a WARNING but does NOT block the trade.
The only exceptions are services whose *purpose* is to block (e.g.
`check_chasing_filter`, `book_guardrails`, `monte_carlo_engine`) — those
flip `risk_out["approved"] = False` on a hard-fail, exactly as a real
risk gate must.

USAGE FROM TRADER
=================
    from core.orphan_consumers import (
        enrich_market_context,
        apply_signal_scoring,
        apply_advanced_risk_gates,
        final_decision_gate,
    )

    # 1) Right after MarketAgent, before AnalysisAgent:
    market_out = enrich_market_context(market_out, symbol=self.symbol,
                                       registry=self._registry)

    # 2) Right after DecisionAgent, before RiskEngine:
    dec_out, analysis_out = apply_signal_scoring(
        dec_out, analysis_out, market_out,
        symbol=self.symbol, registry=self._registry,
    )

    # 3) Right after RiskEngine, before TradePermission:
    risk_out = apply_advanced_risk_gates(
        risk_out, dec_out, market_out, analysis_out,
        symbol=self.symbol, balance=self.balance,
        registry=self._registry,
    )

    # 4) Right after TradePermission, before execution:
    perm_out = final_decision_gate(
        perm_out, dec_out, risk_out, market_out, analysis_out,
        symbol=self.symbol, registry=self._registry,
    )

All four hooks return the (possibly modified) primary dict so the caller
can chain them inline.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

# Toggles for individual consumers. Set ORPHAN_CONSUMER_<NAME>=0 to disable.
# Default: all enabled. Useful for A/B testing whether a service helps or
# hurts a given symbol/market.
_ENABLED = {
    "mtf_analyzer":            os.getenv("OC_MTF", "1") == "1",
    "strength_calculator":     os.getenv("OC_STRENGTH", "1") == "1",
    "liquidity_engine":        os.getenv("OC_LIQUIDITY", "1") == "1",
    "indicator_cache":         os.getenv("OC_INDCACHE", "1") == "1",
    "signal_scorer":           os.getenv("OC_SCORER", "1") == "1",
    "confidence_manager":      os.getenv("OC_CONF", "1") == "1",
    "entry_quality_guardrails":os.getenv("OC_EQG", "1") == "1",
    "institutional_entry":     os.getenv("OC_INST", "1") == "1",
    "book_guardrails":         os.getenv("OC_BOOK", "1") == "1",
    "trading_controls":        os.getenv("OC_TCTRL", "1") == "1",
    "structure_stop":          os.getenv("OC_STRUCT_STOP", "1") == "1",
    "rr_policy":               os.getenv("OC_RRPOL", "1") == "1",
    "streak_tracker":          os.getenv("OC_STREAK", "1") == "1",
    "advanced_risk_orchestrator": os.getenv("OC_ARO", "1") == "1",
    "monte_carlo_engine":      os.getenv("OC_MC", "1") == "1",
    "cost_tracker":            os.getenv("OC_COST", "1") == "1",
    "ml_concept_drift_detector": os.getenv("OC_DRIFT", "1") == "1",
    "ml_uncertainty_estimator":os.getenv("OC_UNC", "1") == "1",
    "decision_validator":      os.getenv("OC_DV", "1") == "1",
    "system_watchdog":         os.getenv("OC_WDOG", "1") == "1",
    "network_monitor":         os.getenv("OC_NETMON", "1") == "1",
}


def _resolve(registry: Any, key: str) -> Optional[Any]:
    """Safe registry lookup — returns None if registry is None or service
    is missing/unresolved."""
    if registry is None:
        return None
    try:
        return registry.try_resolve(key)
    except Exception:
        return None


def _is_enabled(name: str) -> bool:
    return _ENABLED.get(name, True)


# ─────────────────────────────────────────────────────────────────────────
# HOOK 1 — Pre-analysis enrichment
# ─────────────────────────────────────────────────────────────────────────
def enrich_market_context(
    market_out: Dict[str, Any],
    *,
    symbol: str,
    registry: Any,
    timeframe: str = "",
) -> Dict[str, Any]:
    """Enrich market_out with multi-timeframe, currency-strength, and
    liquidity context BEFORE AnalysisAgent runs.

    Adds these keys to market_out (all optional, all fail-safe):
      - "mtf_ctx"        — MTFAnalyzer.analyze() output (4-TF top-down bias)
      - "strength_ctx"   — StrengthCalculator.compute_pair_score() output
      - "liquidity_ctx"  — LiquidityEngine.analyze() output (overrides the
                            thinner liquidity_ctx the analysis_agent builds
                            internally, if present)
      - "indicator_cache_hit_ratio" — diagnostic hit ratio (advisory only)

    Returns the (possibly modified) market_out dict.
    """
    if not isinstance(market_out, dict):
        return market_out

    df = market_out.get("df")
    ind = market_out.get("ind_ctx", {}) or {}
    smc_ctx = market_out.get("smc_ctx") or {}

    # ── MTF Analyzer (4-TF top-down: H4 → H1 → M15 → M5) ────────────────
    # Confirms or contradicts the lower-TF signal direction. The result is
    # *advisory* — AnalysisAgent may use it to bump or dampen confidence.
    if _is_enabled("mtf_analyzer"):
        mtf = _resolve(registry, "mtf_analyzer")
        if mtf is not None:
            try:
                # MTFAnalyzer.analyze() takes regime_ctx and pulls its own
                # data from MT5. We pass the current regime so it doesn't
                # have to recompute.
                mtf_result = mtf.analyze(regime_ctx=market_out.get("regime"))
                if isinstance(mtf_result, dict) and mtf_result:
                    market_out["mtf_ctx"] = mtf_result
                    # If MTF gives a clear bias, also expose the string
                    # directly so analysis_agent can compare it with its
                    # own signal without re-parsing.
                    _mtf_bias = mtf_result.get("bias") or mtf_result.get("signal")
                    if _mtf_bias:
                        # Don't overwrite an existing bias — only fill if absent.
                        existing = market_out.get("mtf_bias")
                        if isinstance(existing, dict):
                            existing.setdefault("mtf_analyzer_bias", _mtf_bias)
                        elif not existing:
                            market_out["mtf_bias"] = {"bias": _mtf_bias,
                                                     "source": "mtf_analyzer"}
                    log.debug(f"[OC] mtf_analyzer → bias={_mtf_bias}")
            except Exception as e:
                log.warning(f"[OC] mtf_analyzer failed (non-fatal): {e}")

    # ── Currency Strength Calculator ────────────────────────────────────
    # Computes a per-pair strength score (-100..+100). BUY signals get a
    # small confidence bump when the base currency is strong; SELL signals
    # get the same when it's weak. Stored as advisory context.
    if _is_enabled("strength_calculator"):
        sc = _resolve(registry, "strength_calculator")
        if sc is not None and df is not None:
            try:
                strength = sc.compute_pair_score(df, ind)
                if isinstance(strength, dict) and strength:
                    market_out["strength_ctx"] = strength
                    log.debug(f"[OC] strength_calculator → score={strength.get('score', '?')}")
            except Exception as e:
                log.warning(f"[OC] strength_calculator failed (non-fatal): {e}")

    # ── Liquidity Engine (equal H/L + PDH/PDL/PWH/PWL + stop-hunt) ──────
    # The analysis_agent already builds a thinner liquidity_ctx from
    # sr_ctx/smc_ctx. If the dedicated LiquidityEngine produces a richer
    # result, prefer it.
    if _is_enabled("liquidity_engine"):
        le = _resolve(registry, "liquidity_engine")
        if le is not None and df is not None:
            try:
                liq = le.analyze(df, smc_ctx=smc_ctx)
                if isinstance(liq, dict) and liq and not liq.get("error"):
                    # Only overwrite if the dedicated engine produced a
                    # real result (not just an _empty_result).
                    #
                    # Audit fix: the old check `liq.get("bias") or
                    # liq.get("levels") or liq.get("grade")` never actually
                    # filtered anything out. "levels" isn't a real key on
                    # LiquidityEngine.analyze()'s output (the real key is
                    # "liquidity_levels"), and _empty_result() itself sets
                    # bias='NEUTRAL' and grade='INVALID' -- both non-empty
                    # strings, both truthy -- so the condition was True on
                    # every call, empty result or not. The one field
                    # _empty_result() actually leaves falsy is
                    # 'current_price' (None), so that's the real signal.
                    if liq.get("current_price") is not None:
                        market_out["liquidity_ctx_orphan"] = liq
                        # Merge grade/bias/analysis into the existing
                        # liquidity_ctx so downstream consumers that already
                        # read market_out["liquidity_ctx"] see the richer
                        # data.
                        #
                        # Audit fix: "explanation" and "stop_hunt_target"
                        # were never real keys on this dict either (real
                        # names are "analysis" and target["target_liquidity"]
                        # inside the "target" sub-dict) -- those two merges
                        # were silently dead on every cycle.
                        existing_liq = market_out.get("liquidity_ctx") or {}
                        existing_liq = dict(existing_liq)
                        _target = liq.get("target") or {}
                        merge_vals = {
                            "grade":             liq.get("grade"),
                            "bias":              liq.get("bias"),
                            "explanation":       liq.get("analysis"),
                            "stop_hunt_target":  _target.get("target_liquidity"),
                        }
                        for k, v in merge_vals.items():
                            if v is not None and k not in existing_liq:
                                existing_liq[k] = v
                        market_out["liquidity_ctx"] = existing_liq
                        log.debug(f"[OC] liquidity_engine → grade={liq.get('grade', '?')}")
            except Exception as e:
                log.warning(f"[OC] liquidity_engine failed (non-fatal): {e}")

    # ── Indicator Cache (advisory diagnostic only) ──────────────────────
    # We don't actually feed the cache here (MarketAgent / Indicators
    # would have to opt in), but we surface the hit ratio so the operator
    # can see whether the cache is being used.
    if _is_enabled("indicator_cache"):
        ic = _resolve(registry, "indicator_cache")
        if ic is not None:
            try:
                hit_ratio = getattr(ic, "hit_ratio", None)
                if callable(hit_ratio):
                    market_out["indicator_cache_hit_ratio"] = float(hit_ratio())
                elif hit_ratio is not None:
                    market_out["indicator_cache_hit_ratio"] = float(hit_ratio)
            except Exception as e:
                log.debug(f"[OC] indicator_cache diagnostic skipped: {e}")

    return market_out


# ─────────────────────────────────────────────────────────────────────────
# HOOK 2 — Post-decision signal scoring
# ─────────────────────────────────────────────────────────────────────────
def apply_signal_scoring(
    dec_out: Dict[str, Any],
    analysis_out: Dict[str, Any],
    market_out: Dict[str, Any],
    *,
    symbol: str,
    registry: Any,
) -> tuple:
    """Run multi-factor signal scoring and confidence management AFTER
    DecisionAgent but BEFORE RiskEngine.

    Two effects:
      1. `signal_scorer` — accumulates points from named layers (rule, ml,
         llm, mtf, strength, liquidity, news). If total score < adaptive
         threshold, the decision is downgraded to WAIT.
      2. `confidence_manager` — adjusts dec_out["confidence"] based on
         historical per-layer accuracy. A layer with poor historical
         accuracy contributes less; a high-accuracy layer contributes more.

    Also runs three advisory quality checks (entry_quality_guardrails,
    institutional_entry_framework, ml_uncertainty_estimator). Their
    failures are NOT blocking on their own, but they each subtract points
    from the signal_scorer total, which CAN downgrade the decision to WAIT
    via the threshold mechanism.

    Returns (dec_out, analysis_out) — both possibly modified.
    """
    if not isinstance(dec_out, dict) or not isinstance(analysis_out, dict):
        return dec_out, analysis_out

    direction = (dec_out.get("decision") or "WAIT").upper()
    if direction not in ("BUY", "SELL"):
        # No trade signal — nothing to score.
        return dec_out, analysis_out

    ind = market_out.get("ind_ctx", {}) or {}
    df = market_out.get("df")
    score_components = []

    # ── Signal Scorer — multi-factor gate ───────────────────────────────
    scorer = None
    if _is_enabled("signal_scorer"):
        scorer = _resolve(registry, "signal_scorer")
    if scorer is not None:
        try:
            scorer.reset()
        except Exception:
            pass

        # Layer 1: Rule-based analysis (PA patterns, candlestick)
        try:
            pa_score = 0
            signal_data = analysis_out.get("signal", {}) or {}
            patterns = signal_data.get("patterns_detected") or analysis_out.get("patterns_detected") or []
            if patterns:
                pa_score = min(30, len(patterns) * 10)
            final_sig = analysis_out.get("final_signal", "")
            if "STRONG" in str(final_sig):
                pa_score += 10
            if scorer is not None:
                scorer.add("rule_pa", pa_score, f"{len(patterns)} patterns")
                score_components.append(("rule_pa", pa_score))
        except Exception as e:
            log.debug(f"[OC] signal_scorer.rule_pa failed: {e}")

        # Layer 2: ML signal
        # AUDIT FIX (execution-proof audit): the previous code read
        # `analysis_out.get("ml_ctx")` — but AnalysisAgent's return dict
        # (analysis_agent.py:2261) has the field named `ml_prediction`,
        # NOT `ml_ctx`. So ml_ctx was always {} and the ML layer was
        # ALWAYS contributing 0 points. Also, the actual ML predictor
        # output (ml/model_predictor.py:302-306) uses keys `prediction`
        # and `probability` (not `signal` and `confidence`), so even if
        # ml_ctx had been resolved correctly, the confidence read would
        # have returned 0. Both bugs fixed below.
        try:
            ml_score = 0
            ml_ctx = analysis_out.get("ml_prediction") or analysis_out.get("ml_ctx") or {}
            ml_signal = (
                ml_ctx.get("prediction")
                or ml_ctx.get("signal")
                or ml_ctx.get("decision")
            )
            # ModelPredictor returns `probability` (0-1, BUY-side); legacy
            # path returned `confidence` (0-100). Detect scale and normalize.
            _ml_prob = ml_ctx.get("probability")
            _ml_conf = ml_ctx.get("confidence")
            if _ml_prob is not None:
                ml_conf = float(_ml_prob) * 100.0
            elif _ml_conf is not None:
                ml_conf = float(_ml_conf)
            else:
                ml_conf = 0.0
            if ml_signal and ml_signal.upper() == direction:
                ml_score = int(min(25, ml_conf * 0.25))
            if scorer is not None:
                scorer.add("ml", ml_score, f"conf={ml_conf:.0f}%")
                score_components.append(("ml", ml_score))
        except Exception as e:
            log.debug(f"[OC] signal_scorer.ml failed: {e}")

        # Layer 3: LLM signal
        # AUDIT FIX (execution-proof audit): the previous code read
        # `llm_ctx.get("signal")` and `llm_ctx.get("decision")` — NEITHER
        # key exists in the llm_ctx produced by AIAnalyst.get_ai_context()
        # (see ai/ai_analyst.py:991-1000). The actual keys are
        # `llm_signal`, `llm_confidence`, etc. Result: the LLM layer was
        # ALWAYS contributing 0 points to the signal_scorer's verdict,
        # silently making the threshold stricter than designed — every
        # BUY/SELL needed to clear the threshold without the 30-point LLM
        # layer that was supposed to be part of the score.
        try:
            llm_score = 0
            llm_ctx = analysis_out.get("llm_ctx") or {}
            llm_signal = (
                llm_ctx.get("llm_signal")
                or llm_ctx.get("signal")
                or llm_ctx.get("decision")
            )
            llm_conf = float(
                llm_ctx.get("llm_confidence")
                or llm_ctx.get("confidence")
                or 0
            )
            if llm_signal and llm_signal.upper() == direction:
                llm_score = int(min(30, llm_conf * 0.30))
            if scorer is not None:
                scorer.add("llm", llm_score, f"conf={llm_conf:.0f}%")
                score_components.append(("llm", llm_score))
        except Exception as e:
            log.debug(f"[OC] signal_scorer.llm failed: {e}")

        # Layer 4: MTF bias (from enrich_market_context)
        try:
            mtf_score = 0
            mtf_ctx = market_out.get("mtf_ctx") or {}
            mtf_bias = (mtf_ctx.get("bias") or mtf_ctx.get("signal") or "").upper()
            if mtf_bias and mtf_bias == direction:
                mtf_score = 20
            elif mtf_bias and mtf_bias in ("BUY", "SELL"):
                mtf_score = -10  # contradicts
            if scorer is not None:
                scorer.add("mtf", mtf_score, f"bias={mtf_bias or 'none'}")
                score_components.append(("mtf", mtf_score))
        except Exception as e:
            log.debug(f"[OC] signal_scorer.mtf failed: {e}")

        # Layer 5: Currency strength
        try:
            strength_score = 0
            strength_ctx = market_out.get("strength_ctx") or {}
            sc_score = float(strength_ctx.get("score", 0) or 0)
            # BUY wants positive strength (base>quote); SELL wants negative.
            if (direction == "BUY" and sc_score > 20) or (direction == "SELL" and sc_score < -20):
                strength_score = 10
            elif (direction == "BUY" and sc_score < -20) or (direction == "SELL" and sc_score > 20):
                strength_score = -10
            if scorer is not None:
                scorer.add("strength", strength_score, f"score={sc_score:.0f}")
                score_components.append(("strength", strength_score))
        except Exception as e:
            log.debug(f"[OC] signal_scorer.strength failed: {e}")

        # Layer 6: Liquidity grade
        try:
            liq_score = 0
            liq_ctx = market_out.get("liquidity_ctx") or {}
            grade = (liq_ctx.get("grade") or "").upper()
            # Audit fix: LiquidityEngine._rank_grade() returns 'A+', 'A',
            # 'B', or 'INVALID' -- this map only had 'A'/'B'/'C'/'D'/'F',
            # so the engine's own TOP grade ('A+') silently fell through
            # to the 0-score default, scoring identically to having no
            # liquidity signal at all. 'C'/'D'/'F' are kept even though
            # the engine never emits them, in case a future grading
            # scheme adds finer bands; 'INVALID' is now explicit rather
            # than relying on the same silent default.
            grade_map = {"A+": 20, "A": 15, "B": 10, "C": 5, "D": -5, "F": -15, "INVALID": 0}
            liq_score = grade_map.get(grade, 0)
            if scorer is not None:
                scorer.add("liquidity", liq_score, f"grade={grade or 'none'}")
                score_components.append(("liquidity", liq_score))
        except Exception as e:
            log.debug(f"[OC] signal_scorer.liquidity failed: {e}")

        # 2026-08-13: credit Unified + Adaptive + DecisionAgent when they
        # already agreed on direction. Without this, scorer only saw thin
        # rule/llm layers (ML schema-broken, patterns often empty) and
        # score stayed ~39 while DecisionAgent had a real SELL.
        try:
            _bonus = 0
            _unified = analysis_out.get("unified_signal") or {}
            _cons = (_unified.get("consensus") or {}) if isinstance(_unified, dict) else {}
            _u_action = str(_cons.get("action", "")).upper()
            if _u_action == direction:
                _bonus += 15
            _adaptive = (_unified.get("adaptive_decision") or {}) if isinstance(_unified, dict) else {}
            if str(_adaptive.get("action", "")).upper() == direction:
                _bonus += 15
            _final = str(analysis_out.get("final_signal", "")).upper()
            if _final == direction:
                _bonus += 10
            _dec_conf = float(dec_out.get("confidence") or 0)
            if _dec_conf >= 50:
                _bonus += min(15, int(_dec_conf * 0.15))
            if scorer is not None and _bonus > 0:
                # Use "smc" slot as generic confluence bonus (registered in LAYER_MAX)
                scorer.add("smc", min(20, _bonus), f"unified/adaptive/final bonus={_bonus}")
                score_components.append(("smc", min(20, _bonus)))
        except Exception as e:
            log.debug(f"[OC] signal_scorer.unified_bonus failed: {e}")

        # Apply the scorer's verdict — soft when close to threshold.
        try:
            verdict = scorer.decide(direction, pair=symbol)
            dec_out["signal_score"] = verdict
            if verdict.get("signal") == "WAIT":
                _score = float(verdict.get("score") or 0)
                _thr = float(verdict.get("threshold") or 60)
                # Soft path: within 15 points of threshold → keep direction,
                # only annotate. Hard path: far below → WAIT.
                if _score >= max(20.0, _thr - 15):
                    log.info(
                        f"[OC] signal_scorer SOFT-PASS {symbol} {direction} "
                        f"(score={_score}/{verdict.get('max')}, thr={_thr} — "
                        f"within 15 of threshold, direction preserved)"
                    )
                    dec_out["signal_scorer_soft_pass"] = True
                    dec_out.setdefault("reject_reason", "")
                else:
                    log.info(
                        f"[OC] signal_scorer DOWNGRADED {symbol} {direction} → WAIT "
                        f"(score={verdict.get('score')}/{verdict.get('max')}, "
                        f"threshold={verdict.get('threshold')}, "
                        f"reason={verdict.get('reason')})"
                    )
                    dec_out.setdefault("pre_gate_decision", direction)
                    dec_out.setdefault("pre_gate_confidence", dec_out.get("confidence"))
                    dec_out["decision"] = "WAIT"
                    dec_out["reject_reason"] = (
                        f"Signal scorer below threshold "
                        f"({verdict.get('score')}/{verdict.get('threshold')}): "
                        f"{verdict.get('reason')}"
                    )
                    dec_out["signal_scorer_downgraded"] = True
                    return dec_out, analysis_out
            else:
                log.info(
                    f"[OC] signal_scorer APPROVED {symbol} {direction} "
                    f"(score={verdict.get('score')}/{verdict.get('max')}, "
                    f"threshold={verdict.get('threshold')})"
                )
        except Exception as e:
            log.warning(f"[OC] signal_scorer.decide() failed (non-fatal): {e}")

    # ── Confidence Manager — dynamic per-layer weight adjustment ────────
    if _is_enabled("confidence_manager"):
        cm = _resolve(registry, "confidence_manager")
        if cm is not None:
            try:
                # ConfidenceManager exposes the recalibrated weights; we
                # use them to scale dec_out["confidence"].
                weights = getattr(cm, "get_weights", None)
                if callable(weights):
                    w = weights()
                    # Multiply current confidence by the average weight
                    # of the layers that contributed (clamped to [0.5, 1.5]).
                    if isinstance(w, dict) and w:
                        avg_w = sum(w.values()) / max(1, len(w))
                        avg_w = max(0.5, min(1.5, float(avg_w)))
                        old_conf = float(dec_out.get("confidence", 50) or 50)
                        new_conf = max(0, min(100, old_conf * avg_w))
                        dec_out["confidence"] = round(new_conf, 1)
                        dec_out["confidence_manager_adjustment"] = {
                            "old": old_conf,
                            "new": new_conf,
                            "avg_weight": round(avg_w, 3),
                            "weights": w,
                        }
                        log.debug(
                            f"[OC] confidence_manager adjusted {old_conf:.0f}% → "
                            f"{new_conf:.0f}% (avg_w={avg_w:.3f})"
                        )
            except Exception as e:
                log.warning(f"[OC] confidence_manager failed (non-fatal): {e}")

    # ── ML Uncertainty Estimator — dampen confidence if model uncertain ─
    if _is_enabled("ml_uncertainty_estimator"):
        ue = _resolve(registry, "ml_uncertainty_estimator")
        if ue is not None:
            try:
                # If the ML model is uncertain (entropy high or std-dev high),
                # dampen confidence. We try the cheapest estimator first.
                ml_ctx = analysis_out.get("ml_ctx") or {}
                proba = ml_ctx.get("probabilities")
                if proba and hasattr(ue, "estimate_from_confidence_interval"):
                    # proba is a list like [p_buy, p_hold, p_sell]
                    p_max = max(proba) if proba else 0.5
                    uncertainty = 1.0 - p_max  # higher = more uncertain
                    if uncertainty > 0.3:  # significant uncertainty
                        old_conf = float(dec_out.get("confidence", 50) or 50)
                        # Dampen proportional to uncertainty beyond 0.3.
                        dampen = 1.0 - (uncertainty - 0.3) * 0.5
                        new_conf = old_conf * dampen
                        dec_out["confidence"] = round(new_conf, 1)
                        dec_out["uncertainty_dampened"] = {
                            "uncertainty": round(uncertainty, 3),
                            "dampen_factor": round(dampen, 3),
                            "old_conf": old_conf,
                            "new_conf": new_conf,
                        }
                        log.debug(
                            f"[OC] ml_uncertainty_estimator dampened "
                            f"{old_conf:.0f}% → {new_conf:.0f}% "
                            f"(uncertainty={uncertainty:.3f})"
                        )
            except Exception as e:
                log.debug(f"[OC] ml_uncertainty_estimator failed (non-fatal): {e}")

    return dec_out, analysis_out


# ─────────────────────────────────────────────────────────────────────────
# HOOK 3 — Pre-execution advanced risk gates
# ─────────────────────────────────────────────────────────────────────────
def apply_advanced_risk_gates(
    risk_out: Dict[str, Any],
    dec_out: Dict[str, Any],
    market_out: Dict[str, Any],
    analysis_out: Dict[str, Any],
    *,
    symbol: str,
    balance: float,
    registry: Any,
) -> Dict[str, Any]:
    """Apply the heavy risk subsystems AFTER RiskEngine but BEFORE
    TradePermission.

    Services consulted (in order):
      1. entry_quality_guardrails  — anti-chasing, indecision, exhaustion
                                      (BLOCK on hard-fail)
      2. institutional_entry_framework — 200-pt entry score (advisory)
      3. structure_stop            — replace ATR SL with structure-based SL
                                      when structure is tighter
      4. rr_policy                 — min R:R enforcement (BLOCK if below)
      5. book_guardrails           — correlation + anti-revenge + cost-EV
                                      (BLOCK on hard-fail)
      6. trading_controls          — portfolio limits (BLOCK on violation)
      7. streak_tracker            — pull consolidated consecutive-loss count
      8. advanced_risk_orchestrator — Kelly position-size override + can_trade
      9. monte_carlo_engine        — risk-of-ruin advisory (BLOCK if RoR > 5%)
     10. cost_tracker              — adjust RR for realistic costs

    Returns risk_out (possibly modified).
    """
    if not isinstance(risk_out, dict):
        return risk_out
    # If RiskEngine already rejected, no point running more gates.
    if not risk_out.get("approved"):
        return risk_out

    direction = (dec_out.get("decision") or "WAIT").upper()
    if direction not in ("BUY", "SELL"):
        return risk_out

    df = market_out.get("df")
    ind = market_out.get("ind_ctx", {}) or {}
    entry = float(risk_out.get("entry") or dec_out.get("entry") or 0)
    sl_price = float(risk_out.get("sl_price") or 0)
    tp_price = float(risk_out.get("tp_price") or 0)
    sl_pips = float(risk_out.get("sl_pips") or 0)
    tp_pips = float(risk_out.get("tp_pips") or 0)

    # ── 1. Entry quality guardrails (chasing / indecision / exhaustion) ─
    if _is_enabled("entry_quality_guardrails") and df is not None:
        try:
            from risk.entry_quality_guardrails import check_chasing_filter
            chase_result = check_chasing_filter(
                df=df, symbol=symbol, direction=direction,
            )
            risk_out.setdefault("quality_checks", []).append(
                {"check": "chasing_filter",
                 "passed": chase_result.passed,
                 "severity": chase_result.severity,
                 "reason": chase_result.reason}
            )
            if not chase_result.passed and chase_result.severity.upper() in ("BLOCK", "HARD"):
                risk_out["approved"] = False
                risk_out["lot"] = 0.0
                risk_out["reject_reason"] = f"Entry quality (chasing): {chase_result.reason}"
                log.info(f"[OC] entry_quality.chasing BLOCKED {symbol} {direction}: {chase_result.reason}")
                return risk_out
        except Exception as e:
            log.warning(f"[OC] entry_quality_guardrails failed (non-fatal): {e}")

    # ── 2. Institutional entry framework (200-pt score, advisory) ───────
    if _is_enabled("institutional_entry"):
        try:
            from risk.institutional_entry_framework import evaluate_institutional_entry
            inst_result = evaluate_institutional_entry(
                direction=direction, entry=entry, sl=sl_price, tp=tp_price,
                df=df, ind_ctx=ind,
                regime=market_out.get("regime"),
                mtf_bias=(market_out.get("mtf_bias") or {}).get("bias") if isinstance(market_out.get("mtf_bias"), dict) else market_out.get("mtf_bias"),
                structure_ctx=analysis_out.get("structure_ctx"),
                smc_ctx=analysis_out.get("smc_ctx"),
                session_ctx=market_out.get("session_ctx"),
                news_ctx=analysis_out.get("news_ctx"),
                liquidity_ctx=analysis_out.get("liquidity_ctx") or market_out.get("liquidity_ctx"),
                spread_pips=float(ind.get("spread_pips", 1.5) or 1.5),
            )
            if isinstance(inst_result, dict):
                risk_out["institutional_entry"] = inst_result
                # If score is below 100/200, dampen lot by 50% (advisory).
                score = float(inst_result.get("score", 200) or 200)
                if score < 100:
                    old_lot = float(risk_out.get("lot", 0) or 0)
                    risk_out["lot"] = round(old_lot * 0.5, 2)
                    risk_out.setdefault("advisory_adjustments", []).append(
                        f"institutional_entry: lot halved (score {score}/200)"
                    )
                    log.info(f"[OC] institutional_entry dampened lot {old_lot} → {risk_out['lot']} (score={score}/200)")
        except Exception as e:
            log.warning(f"[OC] institutional_entry_framework failed (non-fatal): {e}")

    # ── 3. Structure stop — replace SL if structure-based is tighter ───
    if _is_enabled("structure_stop") and df is not None and entry > 0:
        try:
            from risk.structure_stop import compute_structure_stop
            atr = float(ind.get("atr", 0) or 0)
            struct_sl = compute_structure_stop(
                df, direction, method="swing_atr",
                lookback=20, atr_buffer_mult=1.0, atr=atr,
            )
            if struct_sl and struct_sl > 0:
                # For BUY, struct_sl < entry; for SELL, struct_sl > entry.
                # Compute pip distance and use it if TIGHTER than current
                # (tighter = less risk per trade = safer).
                # pip_size resolution
                from utils.pip_utils import pip_size as _pip_size
                _pip = _pip_size(symbol)
                if direction == "BUY":
                    struct_pips = (entry - struct_sl) / _pip if struct_sl < entry else 0
                else:
                    struct_pips = (struct_sl - entry) / _pip if struct_sl > entry else 0
                if 0 < struct_pips < sl_pips:
                    # Structure is tighter — use it.
                    risk_out["sl_price_struct"] = struct_sl
                    risk_out["sl_pips_struct"] = round(struct_pips, 1)
                    # Note: we DON'T overwrite sl_pips / sl_price here because
                    # the existing RiskEngine values are used by the execution
                    # router. We expose the structure stop as an advisory
                    # alternative + record it for the journal.
                    risk_out.setdefault("advisory_adjustments", []).append(
                        f"structure_stop: tighter SL available "
                        f"({struct_pips:.1f}p vs current {sl_pips:.1f}p)"
                    )
                    log.debug(f"[OC] structure_stop: tighter SL available ({struct_pips:.1f}p < {sl_pips:.1f}p)")
        except Exception as e:
            log.debug(f"[OC] structure_stop failed (non-fatal): {e}")

    # ── 4. R:R policy — enforce minimum ─────────────────────────────────
    if _is_enabled("rr_policy"):
        try:
            from risk.rr_policy import get_min_rr
            min_rr = get_min_rr()
            rr = float(risk_out.get("rr_ratio", 0) or 0)
            if rr > 0 and rr < min_rr:
                risk_out["approved"] = False
                risk_out["lot"] = 0.0
                risk_out["reject_reason"] = (
                    f"R:R {rr:.2f} below policy minimum {min_rr:.2f}"
                )
                log.info(f"[OC] rr_policy BLOCKED {symbol} {direction}: RR {rr:.2f} < {min_rr:.2f}")
                return risk_out
        except Exception as e:
            log.warning(f"[OC] rr_policy failed (non-fatal): {e}")

    # ── 5. Book guardrails — correlation + anti-revenge + cost-EV ──────
    if _is_enabled("book_guardrails"):
        try:
            from risk.book_guardrails import run_all_guardrails
            # Pull live open positions for correlation check
            open_positions = []
            try:
                # Local import to avoid circular deps when registry is in flux.
                from core.service_registry import get_registry
                _reg = registry or get_registry()
                _pt = _reg.try_resolve("paper_trader") if _reg else None
                if _pt is not None and hasattr(_pt, "get_open_positions"):
                    open_positions = _pt.get_open_positions()
            except Exception:
                pass

            # Consecutive losses for anti-revenge check
            consec_losses = 0
            try:
                from risk.streak_tracker import get_consecutive_losses as _gcl
                consec_losses = int(_gcl())
            except Exception:
                pass

            # Win probability from analysis confidence.
            # 2026-08-13: floor 0.45 (was 0.30) so Bayesian-crushed conf
            # (~24%) does not force negative cost-EV and zero the lot.
            win_prob = max(0.45, min(0.75, float(dec_out.get("confidence", 50) or 50) / 100.0))
            # Normal lot = risk_per_trade * balance / (sl_pips * pip_value)
            try:
                from core.constants import get_pip_value_usd
                pip_val = get_pip_value_usd(symbol)
            except Exception:
                pip_val = 10.0
            normal_lot = max(0.01, (balance * 0.01) / max(1.0, sl_pips * pip_val))

            gr = run_all_guardrails(
                proposed_pair=symbol,
                proposed_direction=direction,
                proposed_lot_size=float(risk_out.get("lot", 0) or 0),
                normal_lot_size=normal_lot,
                consecutive_losses=consec_losses,
                open_positions=open_positions,
                win_probability=win_prob,
                sl_pips=sl_pips, tp_pips=tp_pips,
            )
            risk_out["book_guardrails"] = gr
            if not gr.get("all_passed"):
                block_reason = str(gr.get("block_reason") or "")
                # 2026-08-13 BALANCED: cost_aware_ev / Net EV failures are
                # often Bayesian chicken-egg (low conf → negative EV →
                # lot=0 → no samples). Soften: keep approved, cap lot at
                # 0.02 so first trades can accumulate data without full size.
                # HARD blocks (correlation stack, revenge trading) still kill.
                _soft_ev = (
                    "cost_aware" in block_reason.lower()
                    or "net ev" in block_reason.lower()
                    or "don't ignore fees" in block_reason.lower()
                    or "bootstrap relief" in block_reason.lower()
                )
                if _soft_ev:
                    old_lot = float(risk_out.get("lot", 0) or 0)
                    capped = min(old_lot, 0.02) if old_lot > 0 else 0.02
                    risk_out["lot"] = round(capped, 2)
                    risk_out.setdefault("advisory_adjustments", []).append(
                        f"book_guardrails EV soft: lot capped {old_lot}→{risk_out['lot']} ({block_reason[:120]})"
                    )
                    log.info(
                        f"[OC] book_guardrails EV-SOFT {symbol} {direction}: "
                        f"lot→{risk_out['lot']} (kept approved) | {block_reason[:100]}"
                    )
                else:
                    risk_out["approved"] = False
                    risk_out["lot"] = 0.0
                    risk_out["reject_reason"] = (
                        f"Book guardrails: {block_reason or 'failed'}"
                    )
                    log.info(f"[OC] book_guardrails BLOCKED {symbol} {direction}: {block_reason}")
                    return risk_out
        except Exception as e:
            log.warning(f"[OC] book_guardrails failed (non-fatal): {e}")

    # ── 6. Trading controls — portfolio limits ──────────────────────────
    if _is_enabled("trading_controls"):
        tc = _resolve(registry, "trading_controls")
        if tc is not None:
            try:
                # Build a minimal PortfolioState from live data.
                from risk.trading_controls import PortfolioState
                open_positions = []
                try:
                    from core.service_registry import get_registry
                    _reg = registry or get_registry()
                    _pt = _reg.try_resolve("paper_trader") if _reg else None
                    if _pt is not None and hasattr(_pt, "get_open_positions"):
                        open_positions = _pt.get_open_positions()
                except Exception:
                    pass
                portfolio = PortfolioState(
                    cash=float(balance),
                    positions={p.get("symbol", "?"): p.get("lot", 0) for p in open_positions},
                )
                # Validate the proposed trade
                tc.validate(
                    asset=symbol,
                    amount=float(risk_out.get("lot", 0) or 0),
                    portfolio=portfolio,
                    current_price=entry,
                )
            except Exception as e:
                # TradingControlViolation is a subclass of Exception.
                from risk.trading_controls import TradingControlViolation
                if isinstance(e, TradingControlViolation):
                    risk_out["approved"] = False
                    risk_out["lot"] = 0.0
                    risk_out["reject_reason"] = f"Trading control: {e}"
                    log.info(f"[OC] trading_controls BLOCKED {symbol} {direction}: {e}")
                    return risk_out
                log.warning(f"[OC] trading_controls failed (non-fatal): {e}")

    # ── 7. Streak tracker — pull consolidated count (advisory) ──────────
    if _is_enabled("streak_tracker"):
        try:
            from risk.streak_tracker import (
                get_consecutive_losses as _gcl,
                get_recent_results as _grr,
                get_win_rate as _gwr,
            )
            risk_out["streak_context"] = {
                "consecutive_losses": int(_gcl()),
                "recent_results": _grr(limit=10),
                "win_rate_10": float(_gwr(lookback=10)),
            }
        except Exception as e:
            log.debug(f"[OC] streak_tracker failed (non-fatal): {e}")

    # ── 8. Advanced Risk Orchestrator — can_trade + Kelly sizing ────────
    if _is_enabled("advanced_risk_orchestrator"):
        aro = _resolve(registry, "advanced_risk_orchestrator")
        if aro is not None:
            try:
                # Sync the orchestrator's balance from the live balance.
                try:
                    aro.account_balance = float(balance)
                except Exception:
                    pass

                # Master gate: daily loss / weekly loss / max drawdown.
                if hasattr(aro, "can_trade") and not aro.can_trade():
                    risk_out["approved"] = False
                    risk_out["lot"] = 0.0
                    risk_out["reject_reason"] = "AdvancedRiskOrchestrator: can_trade=False (daily/weekly/DD limit)"
                    log.info(f"[OC] advanced_risk_orchestrator BLOCKED {symbol} {direction}: can_trade=False")
                    return risk_out

                # Kelly-based size override (advisory: only use if it's
                # smaller than the current lot — i.e. act as a cap, never
                # as an uplifter).
                try:
                    from risk.streak_tracker import get_win_rate as _gwr
                    wr = float(_gwr(lookback=20)) / 100.0
                except Exception:
                    wr = 0.0
                if wr > 0 and hasattr(aro, "position_size"):
                    # Use 1.5R avg win, 1.0R avg loss defaults.
                    kelly_lot = aro.position_size(
                        win_rate=wr,
                        avg_win=1.5, avg_loss=1.0,
                        stop_distance_pips=sl_pips,
                        pip_value_per_lot=10.0,  # advisory — actual varies
                    )
                    if kelly_lot > 0:
                        current_lot = float(risk_out.get("lot", 0) or 0)
                        # Cap at Kelly (never exceed).
                        if kelly_lot < current_lot:
                            risk_out["lot"] = round(kelly_lot, 2)
                            risk_out.setdefault("advisory_adjustments", []).append(
                                f"advanced_risk_orchestrator: lot capped to Kelly {kelly_lot:.2f} (was {current_lot:.2f})"
                            )
                            log.debug(f"[OC] aro Kelly cap: {current_lot:.2f} → {kelly_lot:.2f}")
            except Exception as e:
                log.warning(f"[OC] advanced_risk_orchestrator failed (non-fatal): {e}")

    # ── 9. Monte Carlo — risk-of-ruin advisory (BLOCK if RoR > 5%) ─────
    if _is_enabled("monte_carlo_engine"):
        mc = _resolve(registry, "monte_carlo_engine")
        if mc is not None:
            try:
                # Pull win_rate and avg_win/loss from streak/memory context.
                try:
                    from risk.streak_tracker import get_win_rate as _gwr
                    wr = float(_gwr(lookback=20)) / 100.0
                except Exception:
                    wr = 0.5
                if 0 < wr < 1:
                    # Risk per trade from current lot + SL + pip value.
                    try:
                        from core.constants import get_pip_value_usd
                        pip_val = get_pip_value_usd(symbol)
                    except Exception:
                        pip_val = 10.0
                    risk_usd = float(risk_out.get("lot", 0) or 0) * sl_pips * pip_val
                    rpt = risk_usd / max(1.0, float(balance)) if balance > 0 else 0.01
                    rpt = max(0.001, min(0.10, rpt))  # clamp to [0.1%, 10%]
                    mc_result = mc.run(
                        win_rate=wr,
                        avg_win_pct=1.5, avg_loss_pct=1.0,
                        n_simulations=2000,  # fast — advisory
                        n_trades=100,
                        initial_balance=float(balance),
                        risk_per_trade=rpt,
                        ruin_threshold=0.5,
                    )
                    ror = float(mc_result.get("risk_of_ruin", 0) or 0)
                    risk_out["monte_carlo"] = {
                        "risk_of_ruin": ror,
                        "median_final": mc_result.get("median_final"),
                        "p95_drawdown": mc_result.get("p95_drawdown") or mc_result.get("worst_drawdown"),
                        "win_rate_used": wr,
                        "risk_per_trade_used": rpt,
                    }
                    # BLOCK if risk of ruin > 5%.
                    if ror > 0.05:
                        risk_out["approved"] = False
                        risk_out["lot"] = 0.0
                        risk_out["reject_reason"] = (
                            f"Monte Carlo risk-of-ruin {ror*100:.1f}% > 5% threshold"
                        )
                        log.info(f"[OC] monte_carlo BLOCKED {symbol} {direction}: RoR={ror*100:.1f}%")
                        return risk_out
                    log.debug(f"[OC] monte_carlo advisory: RoR={ror*100:.2f}%")
            except Exception as e:
                log.warning(f"[OC] monte_carlo_engine failed (non-fatal): {e}")

    # ── 10. Cost tracker — adjust RR for realistic costs (advisory) ────
    if _is_enabled("cost_tracker"):
        ct = _resolve(registry, "cost_tracker")
        if ct is not None:
            try:
                summary = ct.get_cost_summary(symbol=symbol)
                if isinstance(summary, dict) and summary.get("total_trades", 0) > 0:
                    avg_cost = float(summary.get("avg_cost_per_trade", 0) or 0)
                    # If average cost per trade > 20% of the risked amount,
                    # this trade is marginal — dampen the lot by 25%.
                    try:
                        from core.constants import get_pip_value_usd
                        pip_val = get_pip_value_usd(symbol)
                    except Exception:
                        pip_val = 10.0
                    risk_usd = float(risk_out.get("lot", 0) or 0) * sl_pips * pip_val
                    if risk_usd > 0 and avg_cost > 0.20 * risk_usd:
                        old_lot = float(risk_out.get("lot", 0) or 0)
                        risk_out["lot"] = round(old_lot * 0.75, 2)
                        risk_out.setdefault("advisory_adjustments", []).append(
                            f"cost_tracker: lot dampened 25% (avg_cost ${avg_cost:.2f} > 20% of risk ${risk_usd:.2f})"
                        )
                        log.debug(f"[OC] cost_tracker dampened lot {old_lot} → {risk_out['lot']}")
                    risk_out["cost_summary"] = summary
            except Exception as e:
                log.debug(f"[OC] cost_tracker failed (non-fatal): {e}")

    return risk_out


# ─────────────────────────────────────────────────────────────────────────
# HOOK 4 — Final decision gate
# ─────────────────────────────────────────────────────────────────────────
def final_decision_gate(
    perm_out: Dict[str, Any],
    dec_out: Dict[str, Any],
    risk_out: Dict[str, Any],
    market_out: Dict[str, Any],
    analysis_out: Dict[str, Any],
    *,
    symbol: str,
    registry: Any,
) -> Dict[str, Any]:
    """Final veto gate AFTER TradePermission but BEFORE execution.

    Three independent veto authorities:
      1. system_watchdog        — block if system health degraded
      2. network_monitor        — block if latency too high
      3. ml_concept_drift_detector — block if feature drift detected

    (decision_validator is intentionally NOT called here — it requires a
    FusionResult object that the current pipeline doesn't construct.
    Wiring it would require either refactoring analysis_agent to produce
    a FusionResult or writing a translator. Out of scope for this pass;
    tracked as a follow-up.)

    Returns perm_out (possibly modified).
    """
    if not isinstance(perm_out, dict):
        return perm_out
    # If permission already denied, no point running more gates.
    if not perm_out.get("allowed"):
        return perm_out

    direction = (dec_out.get("decision") or "WAIT").upper()
    if direction not in ("BUY", "SELL"):
        return perm_out

    # ── 1. System Watchdog ──────────────────────────────────────────────
    if _is_enabled("system_watchdog"):
        wd = _resolve(registry, "system_watchdog")
        if wd is not None:
            try:
                # Try several common method names.
                health_fn = (
                    getattr(wd, "is_healthy", None)
                    or getattr(wd, "check_health", None)
                    or getattr(wd, "status", None)
                )
                if callable(health_fn):
                    health = health_fn()
                    # is_healthy() returns bool; status() returns dict.
                    if isinstance(health, bool):
                        healthy = health
                        reason = "system degraded" if not healthy else "ok"
                    elif isinstance(health, dict):
                        healthy = bool(health.get("healthy", health.get("ok", True)))
                        reason = health.get("reason", health.get("message", "system degraded"))
                    else:
                        healthy = True
                        reason = "unknown"
                    if not healthy:
                        perm_out["allowed"] = False
                        perm_out["execution_allowed"] = False
                        perm_out["final_action"] = "NO TRADE"
                        perm_out["execution_action"] = "NO TRADE"
                        perm_out["blocked_reason"] = f"System watchdog: {reason}"
                        perm_out.setdefault("checks", []).append({
                            "check": "system_watchdog",
                            "passed": False,
                            "detail": reason,
                        })
                        perm_out["total"] = perm_out.get("total", 0) + 1
                        log.warning(f"[OC] system_watchdog BLOCKED {symbol} {direction}: {reason}")
                        return perm_out
            except Exception as e:
                log.warning(f"[OC] system_watchdog failed (non-fatal): {e}")

    # ── 2. Network Monitor ──────────────────────────────────────────────
    if _is_enabled("network_monitor"):
        nm = _resolve(registry, "network_monitor")
        if nm is not None:
            try:
                lat_fn = (
                    getattr(nm, "current_latency_ms", None)
                    or getattr(nm, "get_latency", None)
                    or getattr(nm, "latency_ms", None)
                )
                if callable(lat_fn):
                    lat = lat_fn()
                    if isinstance(lat, dict):
                        lat_ms = float(lat.get("ms", lat.get("value", 0)) or 0)
                    else:
                        lat_ms = float(lat or 0)
                    # Block if latency > 1000ms (1s) — order fill quality
                    # would be unreliable.
                    if lat_ms > 1000:
                        perm_out["allowed"] = False
                        perm_out["execution_allowed"] = False
                        perm_out["final_action"] = "NO TRADE"
                        perm_out["execution_action"] = "NO TRADE"
                        perm_out["blocked_reason"] = f"Network latency {lat_ms:.0f}ms > 1000ms"
                        perm_out.setdefault("checks", []).append({
                            "check": "network_monitor",
                            "passed": False,
                            "detail": f"latency {lat_ms:.0f}ms",
                        })
                        perm_out["total"] = perm_out.get("total", 0) + 1
                        log.warning(f"[OC] network_monitor BLOCKED {symbol} {direction}: {lat_ms:.0f}ms")
                        return perm_out
            except Exception as e:
                log.debug(f"[OC] network_monitor failed (non-fatal): {e}")

    # ── 3. ML Concept Drift Detector ────────────────────────────────────
    if _is_enabled("ml_concept_drift_detector"):
        drift = _resolve(registry, "ml_concept_drift_detector")
        if drift is not None:
            try:
                # Build a recent-feature dict from ind_ctx for drift check.
                # We only check the most informative features.
                ind = market_out.get("ind_ctx", {}) or {}
                recent = {}
                for feat in ("rsi", "macd", "atr", "adx", "cci", "stoch_k"):
                    val = ind.get(feat)
                    if val is not None:
                        try:
                            import numpy as np
                            recent[feat] = np.array([float(val)])
                        except Exception:
                            pass
                if recent and hasattr(drift, "check_all_features"):
                    drift_report = drift.check_all_features(recent)
                    if isinstance(drift_report, dict):
                        # If any feature has drift severity > 0.5, block.
                        max_drift = 0.0
                        drifted_features = []
                        for feat, info in drift_report.items():
                            if isinstance(info, dict):
                                sev = float(info.get("drift_score", info.get("psi", 0)) or 0)
                                if sev > max_drift:
                                    max_drift = sev
                                if sev > 0.5:
                                    drifted_features.append(feat)
                        if drifted_features:
                            perm_out["allowed"] = False
                            perm_out["execution_allowed"] = False
                            perm_out["final_action"] = "NO TRADE"
                            perm_out["execution_action"] = "NO TRADE"
                            perm_out["blocked_reason"] = (
                                f"Concept drift on {drifted_features} (max={max_drift:.2f})"
                            )
                            perm_out.setdefault("checks", []).append({
                                "check": "ml_concept_drift",
                                "passed": False,
                                "detail": f"max drift={max_drift:.2f} on {drifted_features}",
                            })
                            perm_out["total"] = perm_out.get("total", 0) + 1
                            log.warning(f"[OC] ml_concept_drift BLOCKED {symbol} {direction}: drift on {drifted_features}")
                            return perm_out
                        perm_out["ml_drift_report"] = drift_report
            except Exception as e:
                log.debug(f"[OC] ml_concept_drift_detector failed (non-fatal): {e}")

    return perm_out


__all__ = [
    "enrich_market_context",
    "apply_signal_scoring",
    "apply_advanced_risk_gates",
    "final_decision_gate",
]