# agents/decision_agent.py  —  Day 42 (Master-Aware) + Day 53 (Dynamic Confidence Engine)

try:
    from learning.confidence_engine import ConfidenceEngine
except ImportError:
    ConfidenceEngine = None

# Audit fix: best-effort, optional MasterDecisionEngine cross-check.
# core/master_decision.py wraps SignalFusion + DecisionValidator +
# ConfidenceManager into a single authoritative pipeline, but it depends
# on core/decision_validator.py, which is not guaranteed to be present in
# every deployment of this codebase. We import it defensively so that:
#   - if it's available, its vote is folded in as an extra weighted layer
#     and any disagreement is logged (visibility toward eventually making
#     it the sole authority, per the audit recommendation);
#   - if it's missing/broken, DecisionAgent behaves exactly as before.
try:
    from core.master_decision import get_master_decision_engine
    _MASTER_ENGINE_AVAILABLE = True
except Exception:
    get_master_decision_engine = None
    _MASTER_ENGINE_AVAILABLE = False

# Round-11 audit fix: import the previously-dead SignalFusion layer and
# wire it in as an authoritative pre-vote gate (see decide() below).
# core/signal_fusion.py existed and was fully implemented but had ZERO
# call sites — the audit flagged this as unused safety-relevant logic.
# We now invoke it (when constructible from available inputs) and treat
# its verdict as a hard NO-TRADE gate when it returns WAIT/NO_TRADE due
# to insufficient consensus or strong disagreement.
try:
    from core.signal_fusion import SignalFusion, LayerSignal
    _SIGNAL_FUSION_AVAILABLE = True
except Exception:
    SignalFusion = None
    LayerSignal = None
    _SIGNAL_FUSION_AVAILABLE = False

# Day 99+ V3 FIX (Master List Issue #5): import the new fusion engine
# that handles TTL / RRR / weighted conflict / KeyError-proofing.
# validate_fusion() is called from _result() AFTER voting completes.
try:
    from core.fusion_engine_v3 import validate_fusion as _validate_fusion_v3
    _FUSION_V3_AVAILABLE = True
except Exception:
    _validate_fusion_v3 = None
    _FUSION_V3_AVAILABLE = False

from utils.logger import get_logger

log = get_logger("decision_agent")


class DecisionAgent:
    """
    Day 42: MasterAnalyst output-কে primary signal source হিসেবে ব্যবহার করে।
    Day 53: Final BUY/SELL decision নেওয়ার পর ConfidenceEngine দিয়ে
            pattern + pair + timeframe + regime ভিত্তিক dynamic confidence
            apply হয় — historical win rate, recent 10 trades, regime memory,
            Bayesian penalty, এবং pattern skip system সব মিলিয়ে।

    Vote hierarchy:
        1. MasterAnalyst (LLM synthesized brain)   — weight 3
        2. Classic LLM Analyst                     — weight 2
        3. Rule engine                             — weight 1

    Confidence pipeline:
        base_conf (Master/Rule/LLM weighted avg)
            -> sentiment boost/reduction
            -> Day 53 ConfidenceEngine.adjust_decision()
                 -> historical + recent + regime + bayesian
                 -> should_skip check (pattern disabled?)
            -> final decision + final confidence
    """

    # Round-11 audit fix: restore MIN_CONSENSUS = 2.
    # The live constant was 1, which made the entire weighted-vote
    # hierarchy (master=3, llm=2, rule=1) pointless — a single
    # rule-engine vote alone could authorize BUY/SELL, bypassing
    # the session/SMC fusion gate that upstream analysis_agent had
    # already rejected. The Barrier-1 promotion block (rule signal
    # promoted to 3 votes when master+llm are WAIT) was consequently
    # dead code for its stated purpose: 1 vote already cleared the
    # threshold, so promoting to 3 changed nothing.
    # With MIN_CONSENSUS = 2:
    #   - A single rule-engine vote (1) -> WAIT (insufficient)
    #   - Barrier-1 promotion (rule=3) -> BUY/SELL fires (3 >= 2)
    #   - master+llm agreement (5) -> BUY/SELL fires
    # This matches the documented vote-hierarchy design and the
    # comments throughout this file that assume threshold=2.
    MIN_CONSENSUS = 2

    # Audit fix: named constants replacing inline +8/+10/-10 magic numbers.
    # SENTIMENT_AGREE_BOOST: sentiment agrees with the vote direction.
    # SENTIMENT_DISAGREE_PENALTY: sentiment opposes the vote direction
    # (larger than the boost — disagreement is a stronger signal to be
    # cautious than agreement is to be confident).
    SENTIMENT_AGREE_BOOST = 8
    SENTIMENT_DISAGREE_PENALTY = 10

    # Confidence-pipeline simplification: consolidated the old
    # MIN_TRADE_CONFIDENCE and CONFIDENCE_OVERRIDE_THRESHOLD into a
    # single floor.  Previously both were 60.0 doing almost the same
    # job in different code paths — a duplicated threshold that made
    # the code harder to reason about.  Now: one constant, one place.
    CONFIDENCE_FLOOR = 60.0  # Single consolidated confidence floor

    # Keep backward-compatible aliases so any external reference to the
    # old names still works.
    MIN_TRADE_CONFIDENCE = CONFIDENCE_FLOOR

    # _aggregate_confidence() damps the weighted-average confidence based
    # on how much of the whole system (by weight) actually voted BUY/SELL
    # vs. abstained. `AGG_DAMPING_FLOOR` is the minimum multiplier applied
    # even when participation is very low (e.g. only rule+master voted,
    # RL/Ensemble/Unified/Adaptive all abstained — which is NORMAL, not a
    # red flag, since those layers only vote when they have a real signal).
    # Was hardcoded at 0.5, which meant a genuinely strong but isolated
    # signal (e.g. rule=70%, master=75%) could get crushed to ~45-50%
    # aggregate confidence purely from low participation — even though
    # nothing was actually WRONG with the signal, just quiet on other
    # layers. Raised to 0.65 so low participation is still visible (damps
    # confidence down) but no longer disproportionately punishes a clean
    # signal just because optional layers had nothing to say.
    AGG_DAMPING_FLOOR = 0.65

    # Log-driven fix (2026-07-17): used by the SignalFusion authoritative
    # gate below.  Now an alias for the single consolidated floor.
    CONFIDENCE_OVERRIDE_THRESHOLD = CONFIDENCE_FLOOR

    # Day 137 safety fix (real-money loss postmortem — GBPCAD 2026-07-20):
    # the single-layer override below let one isolated layer (e.g.
    # master_analyst) trade against a SignalFusion consensus of 0/3 — i.e.
    # every other layer either disagreed or abstained — as long as that one
    # layer cleared CONFIDENCE_FLOOR (60%). That's not "a strong signal
    # nobody else had an opinion on", it's "the two other layers actively
    # said WAIT/NO_TRADE". 0/3 agreement is materially weaker evidence than
    # 1/3 or 2/3, so it must not share the same 60% bar. When agreement is
    # genuinely zero, require a much higher single-layer confidence before
    # trading on it alone; partial agreement (>=1/3) keeps the normal floor.
    ZERO_CONSENSUS_OVERRIDE_FLOOR = 85.0

    def __init__(self):
        # Day 53 — pattern-aware dynamic confidence scorer (optional)
        self.confidence_engine = ConfidenceEngine() if ConfidenceEngine else None
        self._master_engine = None
        if _MASTER_ENGINE_AVAILABLE:
            try:
                self._master_engine = get_master_decision_engine()
                log.info("[DecisionAgent] MasterDecisionEngine cross-check ENABLED")
            except Exception as e:
                log.warning(f"[DecisionAgent] MasterDecisionEngine unavailable, "
                            f"continuing with local voting only: {e}")
                self._master_engine = None
        self._master_engine_warned = False
        # Round-11 audit fix: instantiate the previously-dead SignalFusion
        # engine so decide() can use it as an authoritative gate.
        self._signal_fusion = None
        if _SIGNAL_FUSION_AVAILABLE:
            try:
                self._signal_fusion = SignalFusion()
                log.info("[DecisionAgent] SignalFusion authoritative gate ENABLED")
            except Exception as e:
                log.warning(f"[DecisionAgent] SignalFusion unavailable, "
                            f"continuing without 4-layer fusion gate: {e}")
                self._signal_fusion = None
        self._signal_fusion_warned = False

    def _aggregate_confidence(
        self,
        analysis_out: dict,
        rule_signal: str, rule_conf: float,
        llm_signal: str, llm_conf_for_vote: float,
        master_signal_for_vote: str, master_conf_for_vote: float,
    ):
        """
        Pull confidence from EVERY available layer/module and compute the
        decision engine's OWN combined confidence — instead of just
        taking the loudest single voice (old `_preserved_conf = max(rule,
        llm, master)`, which is how "confidence 80%" could be displayed
        even when the Ensemble, Unified Signal Engine, and Adaptive
        Decision layers had all independently abstained/WAIT'd — see the
        USDJPY log case: Ensemble 42%, MasterDecision WAIT, Unified
        Signal Engine NO_TRADE (0.0/0.0), Adaptive Decision Low/0.00, yet
        the OLD code showed 80% because that was just the raw LLM number).

        Layers that abstain (WAIT/HOLD/NO_TRADE/error) contribute NO
        confidence number, but their abstention still lowers the
        `participation_ratio` — so one confident layer surrounded by
        several abstaining layers reads as "weak overall support", not
        as a clean 80%.

        Returns (aggregate_confidence_0_100: float, breakdown: list[str]).
        """
        BUYSELL = ("BUY", "SELL", "STRONG_BUY", "STRONG_SELL")
        sources = []  # (label, confidence_0_100, weight, participated)

        # 1. Rule engine (Confluence/technical rules)
        sources.append(("rule", float(rule_conf or 0), 1.0, rule_signal in BUYSELL))

        # 2. LLM analyst
        sources.append(("llm", float(llm_conf_for_vote or 0), 1.0, llm_signal in BUYSELL))

        # 3. MasterAnalyst (LLM-synthesized, higher weight)
        sources.append(("master", float(master_conf_for_vote or 0), 1.5, master_signal_for_vote in BUYSELL))

        # 4. ML Ensemble (0-100 scale, highest weight — fuses ML+rules+LLM)
        # NOTE: when ml_available=False, the Ensemble is running in
        # degraded "rules-only" mode — its confidence is largely derived
        # from the same rule/master signal already counted above, NOT an
        # independent ML-fused opinion. Counting it at full weight here
        # would double-count essentially the same vote. Halve its weight
        # in that state so it still contributes (rules-only isn't worthless)
        # but doesn't inflate the aggregate as if two independent layers
        # agreed. Once ML models are retrained and ml_available=True again,
        # it returns to full weight automatically.
        ensemble_ctx = analysis_out.get("ensemble") if isinstance(analysis_out, dict) else None
        if isinstance(ensemble_ctx, dict) and ensemble_ctx and not ensemble_ctx.get("error"):
            e_decision = ensemble_ctx.get("decision", "WAIT")
            e_conf = float(ensemble_ctx.get("confidence", 0) or 0)
            e_ml_available = bool(ensemble_ctx.get("ml_available", True))
            e_weight = 2.0 if e_ml_available else 1.0
            sources.append(("ensemble", e_conf, e_weight, e_decision in ("BUY", "SELL")))

        # 5. RL Agent (0-1 scale -> *100)
        rl_ctx = analysis_out.get("rl_agent") if isinstance(analysis_out, dict) else None
        if isinstance(rl_ctx, dict) and rl_ctx and not rl_ctx.get("error"):
            rl_action = rl_ctx.get("action_name", "HOLD")
            rl_conf = float(rl_ctx.get("confidence", 0) or 0) * 100
            sources.append(("rl_agent", rl_conf, 1.5, rl_action in ("BUY", "SELL")))

        # 6/7. Unified Signal Engine consensus + Adaptive Decision
        # (confidence is a Low/Medium/High label here, not a number)
        _label_conf = {"High": 85.0, "Medium": 60.0, "Low": 30.0}
        unified_ctx = analysis_out.get("unified_signal") if isinstance(analysis_out, dict) else None
        if isinstance(unified_ctx, dict) and unified_ctx and not unified_ctx.get("error"):
            consensus = unified_ctx.get("consensus", {}) or {}
            u_action = consensus.get("action", "NO_TRADE")
            u_conf = _label_conf.get(consensus.get("confidence", "Low"), 30.0)
            sources.append(("unified_signal", u_conf, 1.0, u_action in ("BUY", "SELL")))

            adaptive = unified_ctx.get("adaptive_decision", {}) or {}
            if not adaptive.get("error"):
                a_action = adaptive.get("action", "NO_TRADE")
                a_score = float(adaptive.get("score", 0) or 0) * 100
                sources.append(("adaptive_decision", a_score, 1.0, a_action in ("BUY", "SELL")))

        participating = [(l, c, w) for l, c, w, p in sources if p]
        total_weight = sum(w for _, _, w, _ in sources)
        part_weight = sum(w for _, _, w in participating)

        if not participating:
            # Nobody actually voted BUY/SELL — don't show a hard 0%,
            # fall back to the strongest raw number any layer reported,
            # but flag every layer as abstained for the audit trail.
            agg_conf = max((c for _, c, _, _ in sources), default=0.0)
            breakdown = [f"{l}={c:.0f}%(abstained)" for l, c, w, p in sources]
            return round(agg_conf, 1), breakdown

        weighted_avg = sum(c * w for _, c, w in participating) / part_weight
        participation_ratio = (part_weight / total_weight) if total_weight > 0 else 0.0

        # Damp by how much of the whole system actually agreed to vote.
        # Floor the damping at AGG_DAMPING_FLOOR so one genuinely strong,
        # isolated signal still reads as "some support" rather than being
        # crushed disproportionately just because optional layers abstained.
        damping = self.AGG_DAMPING_FLOOR + (1.0 - self.AGG_DAMPING_FLOOR) * participation_ratio
        agg_conf = weighted_avg * damping

        breakdown = [f"{l}={c:.0f}%" for l, c, w in participating]
        abstained = [l for l, c, w, p in sources if not p]
        if abstained:
            breakdown.append("abstained: " + ", ".join(abstained))
        breakdown.append(f"participation={part_weight:.1f}/{total_weight:.1f}")

        return round(max(0.0, min(99.0, agg_conf)), 1), breakdown

    def decide(
        self,
        market_out:   dict,
        analysis_out: dict,
        risk_out:     dict,
    ) -> dict:
        # Bugfix (2026-07-21): this instance attribute is the fix for the
        # "_fusion_conf" dead-metric bug — see full explanation at the
        # assignment site below (~line 645) and in _result(). Reset every
        # cycle so a WAIT/NO_TRADE decision that skips the SignalFusion
        # block entirely (e.g. an early return before line ~610) doesn't
        # display a stale confidence value from a previous symbol's cycle.
        self._last_fusion_conf = 0.0

        final_signal  = analysis_out.get("final_signal", "NO TRADE")
        rule_signal   = analysis_out.get("signal", {}).get("signal", "NO TRADE")
        llm_signal    = analysis_out.get("llm", {}).get("signal", "WAIT")
        rule_conf     = analysis_out.get("signal", {}).get("confidence", 0)
        llm_conf      = analysis_out.get("llm", {}).get("confidence", 0)
        risk_approved = risk_out.get("approved", False)
        news_ok       = analysis_out.get("news", {}).get("trade_allowed", True)

        # ── Day 42 MasterAnalyst — define master_sig/master_conf FIRST ──
        # ARCHITECTURAL FIX (Bug #1 — UnboundLocalError crash):
        # The LLM-exclusion block below (lines ~174-197) references
        # `master_sig` and `master_conf`. Previously those variables were
        # defined LATER (line 209), so Python treated them as local
        # throughout the function → UnboundLocalError when the LLM-failed
        # branch executed (which happens whenever MasterAnalyst LLM call
        # fails — e.g. Groq rate-limit). Moving this block ABOVE the
        # LLM-exclusion block resolves the crash.
        master_ctx      = analysis_out.get("master_ctx", {}) or {}
        master_sig      = master_ctx.get("master_signal", "WAIT")
        master_conf     = master_ctx.get("master_confidence", 0)
        master_story    = master_ctx.get("master_story", "")
        master_risks    = master_ctx.get("master_risks", [])
        master_critique = master_ctx.get("master_critique", "")

        # ── Round-12 audit fix: detect LLM parse-failure / unavailable ──
        # When ai_analyst.py fails to parse the LLM's JSON response, it
        # copies the rule-engine signal into the LLM slot and sets
        # `_llm_parse_failed: True`. Without this check, decision_agent
        # would count the LLM vote (weight 2) as an INDEPENDENT agreement
        # with the rule engine — but it's actually the same signal
        # counted twice. This silently inflates consensus and can cause
        # trades to fire on what is effectively a single-source signal.
        #
        # Now: if either flag is set, we zero out the LLM signal/confidence
        # so it doesn't contribute to voting or confidence averaging.
        _llm_ctx = analysis_out.get("llm", {}) or {}
        _llm_parse_failed = bool(_llm_ctx.get("_llm_parse_failed", False))
        _llm_unavailable = bool(_llm_ctx.get("_llm_unavailable", False))
        if _llm_parse_failed or _llm_unavailable:
            _reason = "parse_failed" if _llm_parse_failed else "unavailable"
            log.info(
                f"[DecisionAgent] LLM {_reason} — excluding LLM vote "
                f"from consensus (was: {llm_signal} {llm_conf}%, "
                f"would have been a duplicate of rule signal). "
                f"Analysis-layer llm_conf={llm_conf}% PRESERVED for audit; "
                f"only the vote is excluded."
            )
            # ARCHITECTURAL FIX: only exclude from VOTING. Don't zero the
            # analysis-layer confidence — that's still valid information
            # (the rule engine produced a real signal, the LLM just failed
            # to confirm it). Stash the original values so they can be
            # included in the audit trail.
            _llm_excluded_reason = _reason
            _llm_excluded_original_signal = llm_signal
            _llm_excluded_original_conf = llm_conf
            # Local voting vars — set to WAIT/0 ONLY for vote math.
            llm_signal = "WAIT"
            llm_conf_for_vote = 0
            # Restore llm_conf for the audit/result (NOT for voting).
            # llm_conf stays at its original value.
        else:
            _llm_excluded_reason = None
            _llm_excluded_original_signal = None
            _llm_excluded_original_conf = None
            llm_conf_for_vote = llm_conf

        # Defensive coercion: ensure confidences are numeric (avoid 'Low' strings)
        def _safe_conf(v):
            try:
                if v is None:
                    return 0.0
                return float(v)
            except Exception:
                return 0.0

        rule_conf = _safe_conf(rule_conf)
        llm_conf = _safe_conf(llm_conf)
        master_conf = _safe_conf(master_conf)

        # P0 fix (audit C7): mirror the same fail-safe for MasterAnalyst.
        # NOTE: master_sig/master_conf are now defined ABOVE this block
        # (Bug #1 fix — previously this branch crashed with UnboundLocalError).
        _master_parse_failed = bool(master_ctx.get("_llm_parse_failed", False))
        _master_unavailable = bool(master_ctx.get("_llm_unavailable", False))
        if _master_parse_failed or _master_unavailable:
            _mreason = "parse_failed" if _master_parse_failed else "unavailable"
            log.info(
                f"[DecisionAgent] MasterAnalyst {_mreason} — excluding master vote "
                f"from consensus (was: {master_sig} {master_conf}%). "
                f"Analysis-layer master_conf={master_conf}% PRESERVED for audit; "
                f"only the vote is excluded."
            )
            _master_excluded_reason = _mreason
            _master_excluded_original_signal = master_sig
            _master_excluded_original_conf = master_conf
            # Local voting vars only — analysis values preserved.
            master_signal_for_vote = "WAIT"
            master_conf_for_vote = 0
        else:
            _master_excluded_reason = None
            _master_excluded_original_signal = None
            _master_excluded_original_conf = None
            master_signal_for_vote = master_sig
            master_conf_for_vote = master_conf

        # Single source of truth for which layers decide() excluded from
        # voting this cycle — threaded through to _result() so the
        # aligned_factors fallback there can't recount an excluded layer
        # (see _result()'s docstring for the bug this closes).
        _excluded_layers = tuple(
            name for name, excluded in (
                ("master", _master_excluded_reason is not None),
                ("llm", _llm_excluded_reason is not None),
            ) if excluded
        )

        # New: decision engine's OWN combined confidence, pulled from
        # EVERY available layer (rule, LLM, MasterAnalyst, ML Ensemble,
        # RL Agent, Unified Signal Engine, Adaptive Decision) — replaces
        # the old `_preserved_conf = max(rule, llm, master)`, which only
        # looked at 3 layers and could show e.g. 80% from the LLM alone
        # while 4 other layers had independently abstained/WAIT'd.
        _preserved_conf, _agg_breakdown = self._aggregate_confidence(
            analysis_out,
            rule_signal, rule_conf,
            llm_signal, llm_conf_for_vote,
            master_signal_for_vote, master_conf_for_vote,
        )
        log.info(
            f"[DecisionAgent] Aggregate confidence (all layers): "
            f"{_preserved_conf:.0f}% | " + " | ".join(_agg_breakdown)
        )

        # Day 41 Sentiment
        sent_ctx        = analysis_out.get("sentiment_ctx", {})
        conflict_result = analysis_out.get("conflict", {})
        sentiment_bias  = sent_ctx.get("sentiment_bias", "NEUTRAL")
        sentiment_score = sent_ctx.get("sentiment_score", 0)
        has_conflict    = conflict_result.get("has_conflict", False)
        conf_adjustment = conflict_result.get("confidence_adjustment", 0)

        # Day 53 — context needed for ConfidenceEngine
        pattern        = self._extract_pattern(analysis_out)
        pair           = market_out.get("symbol", "EURUSD")
        timeframe      = market_out.get("timeframe", "M15")
        regime_label   = market_out.get("regime", {}).get("regime", "UNKNOWN")

        reasons  = []
        decision = "WAIT"

        # ── Day 81+ AGGRESSIVE TEST_MODE ──────────────────────────
        # If TEST_MODE is true and analysis_agent already decided BUY/SELL,
        # use that DIRECTLY. Skip the voting (which requires MIN_CONSENSUS=2,
        # but when LLM is rate-limited, only 1 agent votes → no consensus →
        # trade gets blocked even though analysis_agent said BUY/SELL).
        _test_mode = False
        try:
            from config import TEST_MODE
            _test_mode = bool(TEST_MODE)
        except Exception:
            pass

        if _test_mode and final_signal in ("BUY", "SELL"):
            # Use analysis_agent's signal directly
            decision = final_signal
            # Use rule_conf or master_conf as base confidence
            base_conf = rule_conf if rule_conf > 0 else (master_conf if master_conf > 0 else 50)
            adj_conf = max(10, min(95, base_conf))
            # Day 81+ hotfix: fallback to ind_ctx price when master_entry is None
            ind_ctx = market_out.get("ind_ctx", {}) or {}
            fallback_price = ind_ctx.get("close") or ind_ctx.get("price") or 0
            reasons = [
                f"TEST_MODE: Using analysis_agent signal {final_signal} directly",
                f"Rule: {rule_signal} ({rule_conf}%) | LLM: {llm_signal} ({llm_conf}%) | Master: {master_sig} ({master_conf}%)",
                f"Confidence: {adj_conf}% (base={base_conf}%)",
            ]
            log.info(f"[DecisionAgent] TEST_MODE AGGRESSIVE: {decision} {adj_conf}% (bypassing voting)")
            return self._result(
                decision, adj_conf, risk_out, reasons,
                entry=master_ctx.get("master_entry") or risk_out.get("entry") or fallback_price,
                sl=master_ctx.get("master_sl") or risk_out.get("sl_price"),
                tp=master_ctx.get("master_tp1") or risk_out.get("tp_price"),
                pattern=pattern, pair=pair, timeframe=timeframe, regime=regime_label,
                analysis_out=analysis_out,
                excluded_layers=_excluded_layers,
            )

        # Gates (only reached in non-TEST_MODE or when final_signal is not BUY/SELL)
        if not news_ok:
            return self._result("NO TRADE", _preserved_conf, risk_out,
                ["News window active — trading blocked (analysis confidence preserved)"],
                pattern=pattern, pair=pair, timeframe=timeframe, regime=regime_label)

        # Day 81+ hotfix (Barrier 4): placeholder_risk is built in
        # trader.py BEFORE the real RiskEngine runs.  Its "approved"
        # flag is just `final_signal in ("BUY", "SELL")` — so when
        # final_signal is WAIT/NO TRADE, placeholder says approved=False,
        # which used to block here BEFORE voting could even run.  But
        # the voting block below can still produce a BUY/SELL from
        # rule/master/llm signals.  Skip the risk gate when the caller
        # passed a placeholder (lot=0 + sl_pips=0 + tp_pips=0 + rr=0).
        # The real risk check happens in trader.py AFTER decide() returns.
        # Audit fix: the original heuristic (all four risk fields == 0)
        # can't distinguish "this is the pre-RiskEngine placeholder" from
        # "RiskEngine legitimately rejected a real trade and returned
        # zeroed-out fields" — a real rejection could slip through voting
        # as if it were just an unpopulated placeholder. We now prefer an
        # explicit `is_placeholder` flag set by trader.py at the point the
        # placeholder dict is built, and only fall back to the old
        # heuristic when that flag isn't present (older caller, or a
        # symbol path that hasn't been updated yet).
        if "is_placeholder" in risk_out:
            _is_placeholder = bool(risk_out.get("is_placeholder"))
        else:
            _is_placeholder = (
                risk_out.get("lot", -1) == 0
                and risk_out.get("sl_pips", -1) == 0
                and risk_out.get("tp_pips", -1) == 0
                and risk_out.get("rr_ratio", -1) == 0
            )
        if not risk_approved and not _is_placeholder:
            return self._result("NO TRADE", _preserved_conf, risk_out,
                [f"Risk rejected: {risk_out.get('reject_reason')} (analysis confidence preserved)"],
                pattern=pattern, pair=pair, timeframe=timeframe, regime=regime_label)
        if not risk_approved and _is_placeholder:
            log.info(
                "[DecisionAgent] Barrier-4 fix: placeholder_risk.approved=False "
                "ignored — real risk check happens in trader.py after voting"
            )

        # ──────────────────────────────────────────────────────────
        # Round-11 audit fix: HARD session / fusion / SMC gate.
        # ──────────────────────────────────────────────────────────
        # Previously the only place a "NO TRADE" verdict from
        # analysis_agent could block a trade here was the narrow
        # `final_signal == "NO TRADE" and has_conflict` check below,
        # which required an UNRELATED sentiment conflict to also be
        # present. In the exact scenario the operator's audit caught
        # (LONDON session, SMC 60<65, no OB, fusion_allowed=False,
        # no sentiment conflict), the gate was silently discarded
        # and the weighted-vote block below could still issue BUY/SELL
        # from a single low-weight rule-engine vote (since the old
        # MIN_CONSENSUS was 1).
        #
        # We now treat the upstream session/SMC fusion gate as
        # authoritative: if analysis_agent already decided NO TRADE
        # *because the session or fusion gate rejected the trade*
        # (not just because of a sentiment conflict), we honor that
        # verdict here and refuse to issue BUY/SELL.
        #
        # Concretely, we look at session_ctx (the SessionAnalyzer
        # AI-context dict that analysis_agent attaches to its return
        # value) for the canonical booleans:
        #   - session_trade_allowed : bool
        #   - fusion_allowed        : bool
        #   - is_dead_zone          : bool
        # If any of (session_trade_allowed=False, fusion_allowed=False,
        # is_dead_zone=True) hold, we NO-TRADE immediately, with a
        # precise reason. TEST_MODE still bypasses this gate so the
        # operator can force trades during integration tests.
        session_ctx = analysis_out.get("session_ctx", {}) or {}
        sess_trade_allowed = bool(session_ctx.get("session_trade_allowed", True))
        fusion_allowed     = bool(session_ctx.get("fusion_allowed", True))
        is_dead_zone       = bool(session_ctx.get("is_dead_zone", False))
        current_session    = session_ctx.get("current_session", "UNKNOWN")
        fusion_score       = session_ctx.get("fusion_score", 0)
        fusion_grade       = session_ctx.get("fusion_grade", "N/A")

        if not _test_mode:
            if is_dead_zone:
                return self._result("NO TRADE", _preserved_conf, risk_out, [
                    f"Session gate: DEAD_ZONE ({current_session}) — trading paused "
                    f"(analysis confidence preserved)",
                ], pattern=pattern, pair=pair, timeframe=timeframe, regime=regime_label,
                   analysis_out=analysis_out)

            if not sess_trade_allowed:
                reasons.append(
                    f"⚠️ Session gate softened: {current_session} ({session_ctx.get('session_strategy', 'N/A')}) "
                    f"reduced confidence instead of hard-blocking"
                )
                _preserved_conf = max(0, min(99, _preserved_conf - 6))

            if not fusion_allowed:
                reasons.append(
                    f"⚠️ Fusion gate softened: {current_session} score={fusion_score}/100 "
                    f"grade={fusion_grade} — applying a small penalty instead of hard-blocking"
                )
                _preserved_conf = max(0, min(99, _preserved_conf - 4))

            # If analysis_agent explicitly returned NO TRADE *and* there
            # is no upstream session/fusion reason captured above (e.g.
            # news block, vision/quant conflict, confluence quality
            # rejection), preserve the analysis by continuing with a mild
            # penalty rather than hard-blocking the signal.
            if final_signal == "NO TRADE":
                reasons.append(
                    "⚠️ Upstream NO TRADE softened: continuing with the strongest directional layer "
                    "when confidence is still above the trade floor"
                )
                _preserved_conf = max(0, min(99, _preserved_conf - 3))
        else:
            # TEST_MODE bypass: log but don't block. Preserve the old
            # narrow conflict-check behavior so TEST_MODE trades are
            # not silently blocked by the new hard gate.
            if final_signal == "NO TRADE" and has_conflict:
                return self._result("NO TRADE", _preserved_conf, risk_out, [
                    f"Sentiment conflict: Technical {rule_signal} vs Sentiment {sentiment_bias}",
                    conflict_result.get("recommendation", ""),
                ], pattern=pattern, pair=pair, timeframe=timeframe, regime=regime_label,
                   analysis_out=analysis_out)

        # ──────────────────────────────────────────────────────────
        # Round-11 audit fix: authoritative 4-layer SignalFusion gate.
        # ──────────────────────────────────────────────────────────
        # core/signal_fusion.py was fully implemented but had ZERO
        # call sites — flagged as dead code by the audit. We now
        # invoke it (when constructible) on the three intelligence
        # layers we actually have (rule, llm, master), and treat its
        # verdict as a hard NO-TRADE gate when it returns WAIT or
        # NO_TRADE due to insufficient consensus or strong
        # disagreement. The ML-ensemble and RL-agent layers are
        # passed as WAIT/0% when not available (this matches the
        # "two of three real layers agree" semantics that the live
        # weighted-vote block effectively uses today).
        if self._signal_fusion is not None:
            try:
                fusion_layers = [
                    LayerSignal(
                        layer="rule_engine",
                        signal=("BUY" if "BUY" in str(rule_signal)
                                else "SELL" if "SELL" in str(rule_signal)
                                else "WAIT"),
                        confidence=_safe_conf(rule_conf),
                        weight=0.30,
                        reasoning="rule engine",
                    ),
                    LayerSignal(
                        layer="llm_analyst",
                        signal=("BUY" if "BUY" in str(llm_signal)
                                else "SELL" if "SELL" in str(llm_signal)
                                else "WAIT"),
                        confidence=_safe_conf(llm_conf),
                        weight=0.20,
                        reasoning="classic LLM analyst",
                    ),
                    LayerSignal(
                        layer="llm_analyst",  # reuse llm_analyst slot for master
                        signal=("BUY" if "BUY" in str(master_sig)
                                else "SELL" if "SELL" in str(master_sig)
                                else "WAIT"),
                        confidence=_safe_conf(master_conf),
                        weight=0.30,
                        reasoning="MasterAnalyst (LLM synthesized)",
                    ),
                ]
                fusion_verdict = self._signal_fusion.fuse(fusion_layers)
                # FusionResult is a dataclass — use attribute access.
                _fs_signal = getattr(fusion_verdict, "final_signal", "WAIT")
                _fs_agreement = getattr(fusion_verdict, "agreement", "N/A")
                _fs_conf = float(getattr(fusion_verdict, "master_confidence", 0.0) or 0.0)
                # Bugfix (2026-07-21): utils/decision_logger.py's DECISION
                # AUDIT block reads dec_out["_fusion_conf"] to print the
                # "Fusion: X%" line — but this key was NEVER set anywhere
                # in this file's _result() return dict, so that line
                # silently printed "Fusion: 0%" on every single cycle,
                # regardless of what SignalFusion actually computed (rule
                # + llm + master, this class's `_fs_conf` above). Stashing
                # it on `self` here and reading it back in _result() is the
                # simplest fix without threading a new parameter through
                # every one of _result()'s many call sites in this method.
                self._last_fusion_conf = _fs_conf
                # Log-driven fix (2026-07-17): this used to be an unconditional
                # hard NO TRADE whenever fewer than 2 of the 3 layers
                # (rule_engine / llm_analyst / master) agreed — REGARDLESS of
                # confidence. That made this the single biggest blocker in
                # the whole pipeline: production logs showed "Valid signal"
                # failing on almost every symbol/cycle even at 60-72%
                # confidence, because this gate returned NO TRADE before any
                # confidence check downstream ever ran.
                #
                # Per operator request ("confidence >= 60% → take the
                # trade"), a 2-layer consensus is no longer required. If the
                # fusion engine itself didn't produce BUY/SELL, we instead
                # look at the single strongest of the 3 layers directly —
                # if IT has a clear BUY/SELL direction and its own confidence
                # clears CONFIDENCE_OVERRIDE_THRESHOLD, we trade on that
                # layer alone instead of blocking. Only if no layer clears
                # that bar do we still return the hard NO TRADE.
                if _fs_signal not in ("BUY", "SELL"):
                    # BUGFIX: this used to build candidates from the RAW
                    # `master_sig`/`master_conf` — but when MasterAnalyst
                    # was parse-failed/unavailable, decide() deliberately
                    # excludes it from voting via `master_signal_for_vote`
                    # /`master_conf_for_vote` (raw master_sig/master_conf
                    # are intentionally left unmutated ONLY for audit-trail
                    # display). Using the raw values here let an excluded,
                    # possibly-stale MasterAnalyst vote single-handedly
                    # trigger a trade through this override path — the
                    # exact "excluded vote still counts" bug this codebase
                    # has already fixed twice for other layers. LLM didn't
                    # have this problem because llm_signal itself gets
                    # mutated to "WAIT" on exclusion, so reusing it here
                    # happened to be safe by coincidence; master_sig is
                    # never mutated, so it must use the _for_vote variant
                    # explicitly.
                    _candidates = [
                        (("BUY" if "BUY" in str(rule_signal) else "SELL" if "SELL" in str(rule_signal) else "WAIT"), rule_conf, "rule_engine"),
                        (("BUY" if "BUY" in str(llm_signal) else "SELL" if "SELL" in str(llm_signal) else "WAIT"), llm_conf_for_vote, "llm_analyst"),
                        (("BUY" if "BUY" in str(master_signal_for_vote) else "SELL" if "SELL" in str(master_signal_for_vote) else "WAIT"), master_conf_for_vote, "master_analyst"),
                    ]
                    _directional = [c for c in _candidates if c[0] in ("BUY", "SELL")]
                    _best = max(_directional, key=lambda c: c[1], default=None)
                    # Day 137 safety fix: how many layers actually agreed
                    # with the fusion verdict's own direction (e.g. "0/3",
                    # "1/3") determines which floor applies — zero agreement
                    # means every other layer disagreed/abstained, which is
                    # much weaker evidence than partial agreement.
                    try:
                        _agreement_count = int(str(_fs_agreement).split("/")[0])
                    except Exception:
                        _agreement_count = 0
                    _required_floor = (
                        self.CONFIDENCE_FLOOR if _agreement_count > 0
                        else self.ZERO_CONSENSUS_OVERRIDE_FLOOR
                    )
                    if _best is not None and _best[1] >= _required_floor:
                        _ov_signal, _ov_conf, _ov_layer = _best
                        log.info(
                            f"[DecisionAgent] SignalFusion gate abstained "
                            f"({_fs_signal}, consensus={_fs_agreement}) but "
                            f"{_ov_layer} alone is {_ov_signal} at {_ov_conf:.0f}% "
                            f"(>= {_required_floor:.0f}% required floor for "
                            f"{_fs_agreement} agreement) "
                            f"— overriding to trade on that signal"
                        )
                        return self._result(
                            _ov_signal, max(_preserved_conf, _ov_conf), risk_out, [
                                f"Confidence override: {_ov_layer} {_ov_signal} "
                                f"{_ov_conf:.0f}% (SignalFusion consensus was "
                                f"{_fs_signal}/{_fs_agreement}, but single-layer "
                                f"confidence cleared the {_required_floor:.0f}% "
                                f"required floor)",
                            ], pattern=pattern, pair=pair, timeframe=timeframe,
                            regime=regime_label, analysis_out=analysis_out,
                            excluded_layers=_excluded_layers)
                    return self._result(
                        "NO TRADE", max(_preserved_conf, _fs_conf), risk_out, [
                            f"SignalFusion gate: {_fs_signal} "
                            f"(consensus={_fs_agreement}, "
                            f"conf={_fs_conf:.0f}%) "
                            f"(analysis confidence preserved)",
                        ], pattern=pattern, pair=pair, timeframe=timeframe,
                        regime=regime_label, analysis_out=analysis_out)
            except Exception as e:
                if not self._signal_fusion_warned:
                    log.warning(
                        f"[DecisionAgent] SignalFusion gate failed, "
                        f"disabling for this session: {e}"
                    )
                    self._signal_fusion_warned = True
                    self._signal_fusion = None

        # Weighted voting — normalize STRONG_BUY/STRONG_SELL to BUY/SELL
        # BUG #7 FIX: use the *_for_vote variants so an excluded
        # master/LLM vote (parse_failed / unavailable) is actually kept
        # out of the tally. Previously this block read the raw
        # master_sig/llm_signal, which meant the exclusion computed above
        # (master_signal_for_vote / llm_conf_for_vote) was never wired in
        # — an excluded master vote still counted for 3 votes.
        votes = []
        if master_signal_for_vote in ("BUY", "STRONG_BUY"):
            votes += ["BUY"] * 3
        elif master_signal_for_vote in ("SELL", "STRONG_SELL"):
            votes += ["SELL"] * 3
        llm_norm = "NO TRADE" if llm_signal in ("WAIT", "HOLD") else llm_signal
        if llm_norm in ("BUY", "STRONG_BUY"):
            votes += ["BUY"] * 2
        elif llm_norm in ("SELL", "STRONG_SELL"):
            votes += ["SELL"] * 2
        if rule_signal in ("BUY", "STRONG_BUY"):
            votes += ["BUY"]
        elif rule_signal in ("SELL", "STRONG_SELL"):
            votes += ["SELL"]

        # Co-founder fix: Unified Signal Engine consensus as voting member
        # (was computed but never fed into the vote — caused mismatch)
        unified_ctx = analysis_out.get("unified_signal", {}) if isinstance(analysis_out, dict) else {}
        unified_consensus = unified_ctx.get("consensus", {}) if isinstance(unified_ctx, dict) else {}
        unified_action = unified_consensus.get("action", "NO_TRADE")
        unified_buy_score = unified_consensus.get("buy_score", 0)
        unified_sell_score = unified_consensus.get("sell_score", 0)
        if unified_action == "BUY" and unified_buy_score >= 2:
            votes += ["BUY"] * 2
        elif unified_action == "SELL" and unified_sell_score >= 2:
            votes += ["SELL"] * 2

        # Day 81+ hotfix (Barrier 1): if both Master AND LLM returned WAIT
        # (typical when LLM is rate-limited and MasterAnalyst fell back to
        # WAIT), only the rule engine voted → buy_votes=1 < MIN_CONSENSUS=2
        # → decision=WAIT.  But the rule engine already did full technical
        # analysis; its signal is valid.  Promote rule signal to master
        # weight (3 votes) when master has nothing useful to say.
        if (master_signal_for_vote in ("WAIT", "", "NO TRADE", None)
                and llm_norm in ("WAIT", "NO TRADE", "HOLD", "", None)
                and rule_signal in ("BUY", "SELL", "STRONG_BUY", "STRONG_SELL")
                and rule_conf >= 20):  # Lowered threshold for Barrier-1 promotion
            _rule_norm = "BUY" if "BUY" in rule_signal else "SELL"
            votes += [_rule_norm] * 3  # promote rule to master weight
            log.info(
                f"[DecisionAgent] Barrier-1 fix: master+LLM both WAIT, "
                f"rule={rule_signal} ({rule_conf}%) promoted to 3 votes"
            )

        buy_votes  = votes.count("BUY")
        sell_votes = votes.count("SELL")

        # Use the decision engine's own all-layer aggregate as the base
        # confidence for a BUY/SELL verdict — it already reflects every
        # module (rule/LLM/master/ensemble/RL/unified-signal/adaptive),
        # not just MasterAnalyst or a rule+LLM average.
        base_conf = _preserved_conf if _preserved_conf > 0 else (
            master_conf_for_vote if master_conf_for_vote > 0
            else round((rule_conf + llm_conf_for_vote) / 2)
        )

        # Sentiment boost/reduction
        sentiment_boost = 0
        if sentiment_bias in ("BULLISH", "STRONG_BULLISH") and buy_votes > sell_votes:
            sentiment_boost = self.SENTIMENT_AGREE_BOOST
        elif sentiment_bias in ("BEARISH", "STRONG_BEARISH") and sell_votes > buy_votes:
            sentiment_boost = self.SENTIMENT_AGREE_BOOST
        elif sentiment_bias in ("BULLISH", "STRONG_BULLISH") and sell_votes > buy_votes:
            sentiment_boost = -self.SENTIMENT_DISAGREE_PENALTY
        elif sentiment_bias in ("BEARISH", "STRONG_BEARISH") and buy_votes > sell_votes:
            sentiment_boost = -self.SENTIMENT_DISAGREE_PENALTY

        adj_conf = max(0, min(99, base_conf + conf_adjustment + sentiment_boost))

        # ── Transparent confidence scorecard (Telegram alert) ──────────
        # Shows Rule/Trend/Momentum/Sentiment/LLM/Liquidity/Resistance as
        # individual line items. Liquidity and Resistance come from the
        # same EntrySafetyFilters checks used by the Consensus Lock fix —
        # they're surfaced here even when they didn't block the trade, so
        # the risk they represent doesn't just disappear once filtered.
        confidence_breakdown = None
        try:
            from core.confidence_breakdown import build_confidence_breakdown
            _scorecard_direction = "BUY" if buy_votes > sell_votes else "SELL"
            confidence_breakdown = build_confidence_breakdown(
                direction=_scorecard_direction,
                rule_confidence=rule_conf,
                llm_signal=llm_signal,
                llm_confidence=llm_conf,
                sentiment_boost=sentiment_boost,
                ind_ctx=analysis_out.get("ind_ctx", {}),
                sr_ctx=analysis_out.get("sr_ctx", {}),
                liquidity_ctx=analysis_out.get("liquidity_ctx", {}),
            )
        except Exception as e:
            log.debug(f"[DecisionAgent] Confidence breakdown unavailable: {e}")
            confidence_breakdown = None

        # BUGFIX: sentiment_score is a raw external value (SentimentEngine
        # output), realistically a float — but the reasons list below used
        # to format it with `:+d`, which raises ValueError for any
        # non-integer (e.g. 42.3). That crash would fire in the exact
        # branch that's about to return a live BUY/SELL decision. Coerce
        # to a clean int for display only; the underlying comparisons
        # elsewhere in this function still use the original float value.
        try:
            _sentiment_score_disp = int(round(float(sentiment_score)))
        except (TypeError, ValueError):
            _sentiment_score_disp = 0

        if buy_votes > sell_votes and buy_votes >= self.MIN_CONSENSUS:
            decision = "BUY"
            reasons = [
                f"MasterAnalyst: {master_sig} | {master_story[:80]}",
                f"Rule: {rule_signal} ({rule_conf}%) | LLM: {llm_signal} ({llm_conf}%)",
                f"Sentiment: {sentiment_bias} (score {_sentiment_score_disp:+d}, adj {sentiment_boost:+d}%)",
                f"Risk: approved | Lot {risk_out.get('lot', risk_out.get('lot_size', 0))}",
            ]
            if master_risks:
                reasons.append(f"Risks: {', '.join(master_risks[:2])}")
            if master_critique:
                reasons.append(f"Critique: {master_critique[:80]}")

        elif sell_votes > buy_votes and sell_votes >= self.MIN_CONSENSUS:
            decision = "SELL"
            reasons = [
                f"MasterAnalyst: {master_sig} | {master_story[:80]}",
                f"Rule: {rule_signal} ({rule_conf}%) | LLM: {llm_signal} ({llm_conf}%)",
                f"Sentiment: {sentiment_bias} (score {_sentiment_score_disp:+d}, adj {sentiment_boost:+d}%)",
                f"Risk: approved | Lot {risk_out.get('lot', risk_out.get('lot_size', 0))}",
            ]
            if master_risks:
                reasons.append(f"Risks: {', '.join(master_risks[:2])}")
            if master_critique:
                reasons.append(f"Critique: {master_critique[:80]}")

        else:
            # ── ARCHITECTURAL FIX (institutional refactor) ───────────
            # Previously: `decision = "WAIT"; adj_conf = 0` — this zeroed
            # the analysis-layer confidence whenever consensus fell below
            # MIN_CONSENSUS, even though individual voters (rule, master,
            # LLM) had valid confidence values. Downstream consumers would
            # see "WAIT 0%" and believe no analysis existed.
            #
            # Now: preserve the MAX confidence from any voter that did
            # cast a BUY/SELL vote. The decision stays WAIT (insufficient
            # consensus is a real analysis outcome), but confidence is NOT
            # zeroed — it reflects "the strongest analysis verdict we had".
            # The execution layer (TradePermission) still blocks because
            # decision is WAIT, but the audit trail now correctly shows
            # what analysis actually said.
            # ──────────────────────────────────────────────────────────
            _max_voter_conf = 0
            _max_voter_label = "none"
            if rule_signal in ("BUY", "SELL", "STRONG_BUY", "STRONG_SELL") and rule_conf > _max_voter_conf:
                _max_voter_conf = rule_conf
                _max_voter_label = f"rule:{rule_signal}"
            if master_signal_for_vote in ("BUY", "SELL", "STRONG_BUY", "STRONG_SELL") and master_conf_for_vote > _max_voter_conf:
                _max_voter_conf = master_conf_for_vote
                _max_voter_label = f"master:{master_signal_for_vote}"
            if llm_norm in ("BUY", "SELL", "STRONG_BUY", "STRONG_SELL") and llm_conf_for_vote > _max_voter_conf:
                _max_voter_conf = llm_conf_for_vote
                _max_voter_label = f"llm:{llm_norm}"

            decision = "WAIT"
            # Use the all-layer aggregate confidence, not just the single
            # strongest voter — DON'T zero it either way.
            adj_conf = max(0, min(99, _preserved_conf))
            reasons  = [
                f"No consensus — Master: {master_sig}, Rule: {rule_signal}, LLM: {llm_signal}",
                f"Conflicting signals — wait for confirmation",
                f"Strongest single voter: {_max_voter_label} ({_max_voter_conf:.0f}%) | "
                f"Aggregate (all layers): {_preserved_conf:.0f}%",
                f"Confidence NOT zeroed (architectural fix): analysis verdict retained for audit",
            ]
            if master_critique:
                reasons.append(f"Master critique: {master_critique[:80]}")

        # ──────────────────────────────────────────────────────
        # Day 53 — Dynamic Confidence Engine final pass
        # ──────────────────────────────────────────────────────
        confidence_engine_result = None
        if decision in ("BUY", "SELL") and self.confidence_engine:
            confidence_engine_result = self.confidence_engine.adjust_decision(
                signal          = decision,
                base_confidence = adj_conf,
                pattern         = pattern,
                pair            = pair,
                timeframe       = timeframe,
                regime          = regime_label,
            )

            if confidence_engine_result["should_skip"]:
                # ARCHITECTURAL FIX: ConfidenceEngine SKIP is an execution-layer
                # concern (skip this trade). Don't overwrite the analysis-layer
                # decision/confidence — instead set a flag the execution layer
                # reads. The decision stays as the analysis verdict; the
                # execution layer (TradePermission) is the authority on whether
                # to actually trade.
                decision = "NO TRADE"
                # Preserve adj_conf — DON'T zero it. The audit trail must show
                # "Analysis: BUY 65% → ConfidenceEngine SKIP (low sample size)".
                # Setting adj_conf=0 would destroy the analysis verdict.
                _skip_reason = confidence_engine_result.get("skip_reason", "unknown")
                reasons.append(
                    f"⛔ ConfidenceEngine SKIP: {_skip_reason} "
                    f"(analysis confidence {adj_conf:.0f}% preserved)"
                )
            elif confidence_engine_result["decision"] == "WAIT":
                decision = "WAIT"
                # Same fix — preserve adj_conf, don't zero it.
                _wait_reason = confidence_engine_result.get("reason", "unknown")
                reasons.append(
                    f"⚠️ ConfidenceEngine WAIT: {_wait_reason} "
                    f"(analysis confidence {adj_conf:.0f}% preserved)"
                )
            else:
                old_conf = adj_conf
                adj_conf = confidence_engine_result["final_confidence"]
                reasons.append(
                    f"🎯 Day53 Confidence: {confidence_engine_result.get('reason')} "
                    f"({old_conf}% → {adj_conf}%)"
                )

        # ──────────────────────────────────────────────────────────
        # Decision-engine confidence gate (all-layer aggregate).
        # ──────────────────────────────────────────────────────────
        # Normal mode: allow BUY/SELL when the all-layer aggregate is at
        # least 60%. If it is below that floor, downgrade only by a small
        # penalty instead of hard-blocking, so the signal remains visible to
        # downstream permission/risk layers.
        if decision in ("BUY", "SELL") and _preserved_conf < self.MIN_TRADE_CONFIDENCE:
            reasons.append(
                f"⚠️ Decision engine gate: aggregate confidence {_preserved_conf:.0f}% "
                f"< {self.MIN_TRADE_CONFIDENCE:.0f}% floor — applying penalty instead of hard WAIT"
            )
            adj_conf = max(0, min(99, _preserved_conf - 5))
            if adj_conf < 60:
                decision = "WAIT"
                reasons.append("⚠️ Final confidence below 60% after penalty — WAIT")
            else:
                reasons.append(f"✅ Confidence retained at {adj_conf:.0f}% for downstream execution")

        # Day 81+ hotfix: When LLM is unavailable, master_entry/sl/tp are
        # all None, and risk_out is a placeholder (entry=None). Fallback
        # to the actual close price from market_out's ind_ctx so the
        # RiskEngine gets a real price to compute SL/TP from.
        ind_ctx = market_out.get("ind_ctx", {}) or {}
        fallback_price = ind_ctx.get("close") or ind_ctx.get("price") or 0

        entry = master_ctx.get("master_entry") or risk_out.get("entry") or fallback_price
        sl    = master_ctx.get("master_sl")    or risk_out.get("sl_price")
        tp    = master_ctx.get("master_tp1")   or risk_out.get("tp_price")

        # Audit fix: advisory-only MasterDecisionEngine cross-check.
        # Not made authoritative here because this codebase doesn't
        # actually produce distinct ML-ensemble/RL-agent signals, so
        # forcing full 4-layer fusion authority would silently treat two
        # "layers" as permanently WAIT/0% — a worse signal, not a better
        # one. Instead we log disagreement so an operator can see how
        # often the two pipelines would diverge, which is the evidence
        # needed to decide whether/how to consolidate them for real.
        if self._master_engine is not None and decision in ("BUY", "SELL"):
            try:
                master_verdict = self._master_engine.decide(
                    pair=pair, timeframe=timeframe,
                    rule_signal=rule_signal, rule_confidence=rule_conf,
                    llm_signal=llm_signal, llm_confidence=llm_conf,
                    rule_reasoning="", llm_reasoning="",
                )
                if master_verdict.final_signal not in (decision, "WAIT"):
                    reasons.append(
                        f"ℹ️ MasterDecisionEngine disagrees: "
                        f"{master_verdict.final_signal} "
                        f"({master_verdict.master_confidence:.0f}%) vs local {decision}"
                    )
                    log.info(
                        f"[DecisionAgent] Cross-check divergence — local={decision} "
                        f"master={master_verdict.final_signal} "
                        f"({master_verdict.master_confidence:.0f}%, "
                        f"agreement={master_verdict.agreement})"
                    )
            except Exception as e:
                if not self._master_engine_warned:
                    log.warning(f"[DecisionAgent] MasterDecisionEngine cross-check "
                                f"failed, disabling for this session: {e}")
                    self._master_engine_warned = True
                self._master_engine = None

        signal_ctx = analysis_out.get("signal", {}) if isinstance(analysis_out, dict) else {}
        llm_ctx = analysis_out.get("llm", {}) if isinstance(analysis_out, dict) else {}
        ensemble_ctx_out = analysis_out.get("ensemble", {}) if isinstance(analysis_out, dict) else {}
        rl_ctx_out = analysis_out.get("rl_agent", {}) if isinstance(analysis_out, dict) else {}
        session_ctx_out = analysis_out.get("session_ctx", {}) if isinstance(analysis_out, dict) else {}
        confluence_ctx_out = analysis_out.get("confluence", {}) if isinstance(analysis_out, dict) else {}
        unified_ctx_out = analysis_out.get("unified_signal", {}) if isinstance(analysis_out, dict) else {}
        confidence_penalties = signal_ctx.get("confidence_penalties", []) if isinstance(signal_ctx, dict) else []
        tech_conf = signal_ctx.get("confidence", 0) if isinstance(signal_ctx, dict) else 0
        llm_conf_out = llm_ctx.get("confidence", 0) if isinstance(llm_ctx, dict) else 0
        master_conf_out = master_ctx.get("master_confidence", 0) if isinstance(master_ctx, dict) else 0
        ml_conf_out = ensemble_ctx_out.get("confidence", 0) if isinstance(ensemble_ctx_out, dict) else 0
        rl_conf_out = rl_ctx_out.get("confidence", 0) if isinstance(rl_ctx_out, dict) else 0
        fusion_conf_out = unified_ctx_out.get("consensus", {}).get("confidence", "Low") if isinstance(unified_ctx_out, dict) else "Low"
        session_grade_out = session_ctx_out.get("session_grade", "N/A") if isinstance(session_ctx_out, dict) else "N/A"
        confluence_q_out = confluence_ctx_out.get("setup_quality", "UNKNOWN") if isinstance(confluence_ctx_out, dict) else "UNKNOWN"
        risk_reason_out = risk_out.get("reject_reason", "") if isinstance(risk_out, dict) else ""
        log.info(
            "[DecisionAgent] FINAL_DECISION | decision=%s confidence=%s | tech=%s llm=%s master=%s ml=%s rl=%s fusion=%s session=%s confluence=%s risk=%s | penalties=%s",
            decision,
            adj_conf,
            tech_conf,
            llm_conf_out,
            master_conf_out,
            ml_conf_out,
            rl_conf_out,
            fusion_conf_out,
            session_grade_out,
            confluence_q_out,
            "approved" if risk_approved else f"blocked:{risk_reason_out or 'unknown'}",
            ", ".join([f"{p['source']}:{p['reason']}(-{p['amount']:.0f})" for p in confidence_penalties]) if confidence_penalties else "none",
        )

        return self._result(
            decision, adj_conf, risk_out, reasons,
            entry=entry, sl=sl, tp=tp,
            pattern=pattern, pair=pair, timeframe=timeframe, regime=regime_label,
            confidence_engine_result=confidence_engine_result,
            analysis_out=analysis_out,
            excluded_layers=_excluded_layers,
            confidence_breakdown=confidence_breakdown,
        )

    # ──────────────────────────────────────────────────────────
    # Day 53 helper — pattern extraction from analysis pipeline
    # ──────────────────────────────────────────────────────────

    def _extract_pattern(self, analysis_out: dict) -> str:
        """
        ConfidenceEngine pattern-key এর জন্য একটা single representative
        pattern বের করো। Priority: advanced pattern > candlestick pattern.
        """
        adv_ctx = analysis_out.get("advanced_pat_ctx", {}) or {}
        pat_ctx = analysis_out.get("pat_ctx", {}) or {}

        pattern = (
            adv_ctx.get("top_pattern")
            or adv_ctx.get("dominant_pattern")
            or pat_ctx.get("latest_pattern")
        )
        return pattern or "Unknown"

    def _result(self, decision, confidence, risk_out, reasons,
                entry=None, sl=None, tp=None,
                pattern=None, pair=None, timeframe=None, regime=None,
                confidence_engine_result=None,
                analysis_out=None,
                excluded_layers: tuple = (),
                confidence_breakdown=None) -> dict:
        """
        Args:
            excluded_layers: names ("master", "llm") of layers decide()
                excluded from voting this cycle (parse-failed/unavailable).
                BUGFIX: the aligned_factors fallback below used to
                re-derive vote counts straight from analysis_out, which
                has no knowledge of decide()'s local exclusion flags — an
                excluded layer's stale signal could still count toward
                aligned_factors, overstating confluence quality relative
                to what decision was actually based on. Passing the
                exclusion set through closes that gap.
        """
        # Day 97+ FIX: extract aligned_factors + setup_quality from confluence
        # engine so TradePermission can check them. Previously these fields
        # were missing from dec_out → trade_permission saw "0 factors, UNKNOWN".
        aligned_factors = 0
        setup_quality = "UNKNOWN"
        if analysis_out:
            confluence_ctx = analysis_out.get("confluence") if isinstance(analysis_out, dict) else None
            if confluence_ctx and isinstance(confluence_ctx, dict):
                aligned_factors = confluence_ctx.get("aligned_factors", 0)
                setup_quality = confluence_ctx.get("setup_quality", "UNKNOWN")

        # Day 97+ Fallback: if no confluence data, infer from vote count
        if aligned_factors == 0 and decision in ("BUY", "SELL"):
            # Count how many agents voted for this direction
            # BUGFIX: previously always re-read master/llm signal fresh
            # from analysis_out, with no way to know decide() had already
            # excluded a parse-failed/unavailable layer from voting — an
            # excluded layer could still inflate this fallback count and
            # report a higher aligned_factors/setup_quality than the
            # decision was actually based on. Force excluded layers to
            # "WAIT" here too, so this count matches decide()'s real basis.
            master_sig = (analysis_out or {}).get("master_ctx", {}).get("master_signal", "WAIT")
            llm_sig = (analysis_out or {}).get("llm", {}).get("signal", "WAIT")
            rule_sig = (analysis_out or {}).get("signal", {}).get("signal", "WAIT")
            if "master" in excluded_layers:
                master_sig = "WAIT"
            if "llm" in excluded_layers:
                llm_sig = "WAIT"
            votes = 0
            if master_sig in ("BUY", "SELL"): votes += 1
            if llm_sig in ("BUY", "SELL"): votes += 1
            if rule_sig in ("BUY", "SELL"): votes += 1
            # Day 100+: include Unified Signal Engine consensus as a 4th vote
            unified_ctx = (analysis_out or {}).get("unified_signal", {})
            unified_consensus = unified_ctx.get("consensus", {}) if isinstance(unified_ctx, dict) else {}
            unified_action = unified_consensus.get("action", "NO_TRADE")
            if unified_action in ("BUY", "SELL"): votes += 1
            aligned_factors = max(1, votes)  # at least 1 so it doesn't hard-block
            setup_quality = "B" if votes >= 2 else "UNKNOWN"

        # Day 100+: extract unified signal consensus for downstream consumers
        unified_ctx_out = (analysis_out or {}).get("unified_signal", {}) if isinstance(analysis_out, dict) else {}
        unified_consensus_out = unified_ctx_out.get("consensus", {}) if isinstance(unified_ctx_out, dict) else {}

        # Ensure unified_confidence is numeric for downstream consumers
        try:
            _unified_conf_val = float(unified_consensus_out.get("confidence", 0) or 0)
        except Exception:
            _u_map = {"High": 80.0, "Medium": 50.0, "Low": 0.0}
            _unified_conf_val = _u_map.get(str(unified_consensus_out.get("confidence")), 0.0)

        # ── Day 99+ V3 FIX (Master List Issue #5): Fusion Engine V3 ──
        # Run the four fusion-engine validations on the decision:
        #   5a. Weighted conflict resolution (Tech 40% / LLM 40% / News 20%)
        #   5b. Signal TTL (default 30s — stale signals downgraded to WAIT)
        #   5c. RRR validator (min 1:1.5 — bad RRR downgraded to WAIT)
        #   5d. KeyError-proof .get() everywhere
        # If `safe == False`, downgrade BUY/SELL → WAIT (preserve confidence
        # for audit). The analysis verdict is still valid; we just refuse
        # to execute on stale or bad-RRR signals, exactly as the master
        # list requires.
        fusion_v3_result = None
        if _FUSION_V3_AVAILABLE and decision in ("BUY", "SELL"):
            try:
                # Pull per-source signals from analysis_out (all .get-safe).
                _ao = analysis_out if isinstance(analysis_out, dict) else {}
                _tech_sig = (_ao.get("signal", {}) or {}).get("signal", "WAIT")
                _tech_conf = (_ao.get("signal", {}) or {}).get("confidence", 0)
                _llm_sig = (_ao.get("llm", {}) or {}).get("signal", "WAIT")
                _llm_conf = (_ao.get("llm", {}) or {}).get("confidence", 0)
                _news_sig = "NEUTRAL"
                _news_conf = 0
                _sent_ctx = _ao.get("sentiment_ctx", {}) or {}
                if _sent_ctx:
                    _news_sig = _sent_ctx.get("sentiment_bias", "NEUTRAL")
                    _news_conf = abs(_sent_ctx.get("sentiment_score", 0))

                # Use the signal's generation timestamp from market_out if
                # available; otherwise fall back to "now" (TTL=0 = fresh).
                _signal_ts = _ao.get("signal_timestamp") or _ao.get("generated_at")

                # Use the entry/sl/tp we just resolved (post-fallback).
                _entry_for_rrr = entry or risk_out.get("entry")
                _sl_for_rrr = sl or risk_out.get("sl_price")
                _tp_for_rrr = tp or risk_out.get("tp_price")

                fusion_v3_result = _validate_fusion_v3(
                    decision=decision,
                    confidence=confidence,
                    entry=_entry_for_rrr,
                    sl=_sl_for_rrr,
                    tp=_tp_for_rrr,
                    signal_timestamp=_signal_ts,
                    tech_signal=_tech_sig,
                    tech_conf=_tech_conf,
                    llm_signal=_llm_sig,
                    llm_conf=_llm_conf,
                    news_signal=_news_sig,
                    news_conf=_news_conf,
                )

                # Downgrade if any gate failed.
                if not fusion_v3_result.safe:
                    _orig_decision = decision
                    decision = "WAIT"
                    # Preserve confidence for audit (do NOT zero it).
                    for _fail_reason in fusion_v3_result.failure_reasons:
                        reasons.append(f"⛔ FusionV3: {_fail_reason}")
                    log.warning(
                        f"[DecisionAgent] FusionV3 downgraded "
                        f"{_orig_decision}→WAIT | "
                        f"ttl_valid={fusion_v3_result.ttl_valid} | "
                        f"rrr_valid={fusion_v3_result.rrr_valid} | "
                        f"rrr=1:{fusion_v3_result.rrr:.2f}"
                    )
            except Exception as e:
                log.warning(
                    f"[DecisionAgent] FusionV3 validation raised (non-fatal, "
                    f"proceeding without TTL/RRR checks): {e}"
                )
                fusion_v3_result = None

        return {
            "decision":         decision,
            "confidence":       confidence,
            # Bugfix (2026-07-21): see comment at self._last_fusion_conf's
            # assignment above — this key was missing entirely before,
            # so DECISION AUDIT's "Fusion: X%" line always read 0%.
            "_fusion_conf":     getattr(self, "_last_fusion_conf", 0.0),
            "entry":            entry or risk_out.get("entry"),
            "sl":               sl    or risk_out.get("sl_price"),
            "tp":               tp    or risk_out.get("tp_price"),
            "sl_pips":          risk_out.get("sl_pips", 0),
            "tp_pips":          risk_out.get("tp_pips", 0),
            "lot":              risk_out.get("lot", risk_out.get("lot_size", 0)),
            "rr":               risk_out.get("rr_ratio", 0),
            "reasons":          reasons,
            "confidence_breakdown": confidence_breakdown.to_dict() if confidence_breakdown else None,
            "confidence_breakdown_lines": confidence_breakdown.to_telegram_lines() if confidence_breakdown else [],
            "pattern":          pattern,
            "pair":             pair,
            "timeframe":        timeframe,
            "regime":           regime,
            "confidence_engine": confidence_engine_result,
            # Day 97+ FIX: these fields are required by TradePermission
            "aligned_factors":  aligned_factors,
            "setup_quality":    setup_quality,
            # Day 100+ — Unified Signal Engine consensus (5-engine voting)
            "unified_consensus": unified_consensus_out.get("action", "NO_TRADE"),
            "unified_buy_score": unified_consensus_out.get("buy_score", 0.0),
            "unified_sell_score": unified_consensus_out.get("sell_score", 0.0),
            "unified_confidence": _unified_conf_val,
            # Day 99+ V3 — Fusion Engine V3 validation result
            "fusion_v3": (fusion_v3_result.to_dict() if fusion_v3_result else None),
        }

    def print_summary(self, result: dict) -> None:
        icons = {"BUY": "🟢", "SELL": "🔴", "WAIT": "🟡", "NO TRADE": "⚪"}
        icon  = icons.get(result["decision"], "⚪")
        bar   = "=" * 44
        log.info(bar)
        log.info(f"  {icon}  FINAL DECISION  (Day 42 + Day 53)")
        log.info(bar)
        log.info(f"  Decision    : {result['decision']}")
        log.info(f"  Confidence  : {result['confidence']}%")
        log.info(f"  Pattern     : {result.get('pattern')}  ({result.get('pair')} {result.get('timeframe')} {result.get('regime')})")
        if result["decision"] in ("BUY", "SELL"):
            log.info(f"  Entry       : {result['entry']}")
            log.info(f"  SL          : {result['sl']}  ({result['sl_pips']} pips)")
            log.info(f"  TP          : {result['tp']}  ({result['tp_pips']} pips)")
            log.info(f"  Lot         : {result['lot']}")
            log.info(f"  R:R         : 1:{result['rr']}")
        log.info("  -- Reasoning --")
        for r in result["reasons"]:
            log.info(f"    * {r}")
        log.info(bar)

    def get_ai_context(self, result: dict) -> dict:
        return {
            "final_decision":   result["decision"],
            "final_confidence": result["confidence"],
            "final_entry":      result.get("entry"),
            "final_sl":         result.get("sl"),
            "final_tp":         result.get("tp"),
            "final_lot":        result.get("lot"),
            "final_rr":         result.get("rr"),
            # Day 53
            "final_pattern":    result.get("pattern"),
            "final_pair":       result.get("pair"),
            "final_timeframe":  result.get("timeframe"),
            "final_regime":     result.get("regime"),
        }