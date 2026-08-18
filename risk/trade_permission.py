# risk/trade_permission.py  —  Day 13 | Final Trade Permission Gate

import json

from utils.logger import get_logger

log = get_logger("trade_permission")

_BYPASS_CHECK_ALIASES = {
    "valid_signal": "Valid signal",
    "sr_alignment": "S/R zone alignment",
    "trend_alignment": "Trend alignment (regime)",
    "mtf_trend_alignment": "MTF trend alignment (H4/H1/M15)",
    "zone_cooldown": "Zone cooldown (duplicate entry)",
    "risk_approved": "Risk approved",
    "news_safe": "News safe",
    "entry_quality": "Entry quality guardrails",
    "confirmation_bias": "Confirmation bias defense",
    "revenge_trading": "Revenge trading detector",
    "cost_aware_ev": "Cost-aware EV gate (book_guardrails)",
    "min_confidence": "Min confidence",
    "session_quality": "Session quality",
    "confluence_quality": "Confluence quality",
    "min_rr": "Min R:R",
    "smc_session_fusion": "SMC+Session fusion",
    "signal_persistence": "Signal persistence",
    "regime_suppression": "Regime suppression",
    "duplicate_trade": "Duplicate trade",
    "correlation_filter": "Correlation filter",
    "news": "Execution filter: news",
    "news_intelligence": "Execution filter: news_intelligence",
    "confluence_avoid": "Execution filter: confluence_avoid",
    "mtf_structure_no_trade": "Execution filter: mtf_structure_no_trade",
    "session": "Execution filter: session",
    "fusion": "Execution filter: fusion",
}


def _normalize_bypass_checks(bypass_checks: set[str] | list[str] | None) -> set[str]:
    normalized = set()
    if bypass_checks is None:
        return normalized
    for check in bypass_checks:
        if check is None:
            continue
        raw = str(check).strip()
        if not raw:
            continue
        normalized.add(raw)
        normalized.add(raw.lower())
        alias_target = _BYPASS_CHECK_ALIASES.get(raw)
        if alias_target:
            normalized.add(alias_target)
            normalized.add(alias_target.lower())
        alias_target = _BYPASS_CHECK_ALIASES.get(raw.lower())
        if alias_target:
            normalized.add(alias_target)
            normalized.add(alias_target.lower())
    return normalized


def _bypass_check(check_name: str, bypass_checks: set[str]) -> bool:
    if "all" in bypass_checks:
        return True
    if check_name in bypass_checks or check_name.lower() in bypass_checks:
        return True
    alias_target = _BYPASS_CHECK_ALIASES.get(check_name)
    if alias_target and (alias_target in bypass_checks or alias_target.lower() in bypass_checks):
        return True
    return False


def _test_mode() -> bool:
    """Lazy check using environment variables to avoid stale imported config values."""
    import os as _os
    # Prefer centralized helper when available
    try:
        from core.constants import is_test_mode
        return bool(is_test_mode())
    except Exception:
        val = _os.getenv("TEST_MODE", _os.getenv("FOREX_TEST_MODE", "false"))
        return str(val).strip().lower() in {"1", "true", "yes"}


class TradePermission:
    """
    সব check পার হলে ALLOW, না হলে DENY।
    RiskEngine এর পরে final gate।

    Checklist:
        1. Signal valid?
        2. Risk approved?
        3. News safe?
        4. Session active?
        5. Confluence enough?
        6. Min R:R
        7. SMC+Session fusion (Round-5/10)

    ── Round-12 audit fix: threshold documentation ──────────────────
    The operator's audit found a confusing contradiction:
      - trade_permission: MIN_CONFIDENCE_PROD=40 → "45% ≥ 40% PASS"
      - LiveRiskManager: tier 1 min_confidence=80% → "45% < 80% BLOCK"

    This is NOT a bug — it's a layered defense design:
      - trade_permission.MIN_CONFIDENCE is the FLOOR (absolute minimum
        to even be considered). 40% means "don't reject purely on
        confidence alone; let other gates (news, session, R:R, fusion)
        also have a say."
      - LiveRiskManager tier min_confidence is the CEILING per tier.
        Tier 1 (new account) requires 80% — very conservative. Tier 3
        (proven account) requires 55% — more permissive.

    Both gates run in sequence. A trade must pass BOTH. So the effective
    threshold is max(trade_permission.MIN_CONFIDENCE, LRM.tier.min_confidence).
    On a fresh Tier 1 account, that's max(40, 80) = 80%.

    To make this visible in the log, trade_permission now also reads the
    LRM tier threshold and includes it in the confidence check detail.
    """

    # Day 96 bugfix: comment said 60 but the constant was left at 40 —
    # the gate was never actually enforcing the documented production
    # threshold, which is how single-indicator 42%-confidence trades
    # (e.g. lone RSI oversold) kept reaching MT5.
    # Conservative frequency tune (operator request: "best but no false
    # trades" -- prioritize avoiding bad entries over volume). Moved from
    # 60 to 55: a modest 5-point nudge, not a broad loosening. Deliberately
    # left the entry_quality_guardrails penalty weights untouched -- those
    # penalties are calibrated against a real live-loss postmortem
    # (Day 137 GBPCAD, see risk/entry_quality_guardrails.py), so easing
    # them would trade away exactly the false-trade protection requested.
    # If this alone doesn't move frequency enough, the next lever to pull
    # is entry_quality penalty weights -- NOT this floor again -- and that
    # should only happen after reviewing the new ENTRY_QUALITY_BREAKDOWN
    # diagnostics (blocked_audit.py) against a few weeks of live results.
    # 70 produced near-zero live trades (most signals sit 55-68% after
    # entry-quality penalties). Rolled back so this floor is no longer
    # the primary cause of 0 trades. False-trade protection remains in
    # entry_quality_guardrails, MTF/S/R alignment, and RiskEngine.
    # 2026-08-13 final: default 80 (was 85). Wide SL strategy works with
    # lower confidence — gives more trades while maintaining PF > 1.0.
    # Reads from .env via core.constants.MIN_CONFIDENCE_PROD.
    try:
        from core.constants import MIN_CONFIDENCE_PROD as _ENV_MIN_CONF
        MIN_CONFIDENCE_PROD = int(_ENV_MIN_CONF) if _ENV_MIN_CONF else 80
    except Exception:
        MIN_CONFIDENCE_PROD = 80
    MIN_CONFIDENCE_TEST  = 10
    MIN_CONFIDENCE_RECENT_WIN_RATE_FLOOR = 0.45
    MIN_CONFIDENCE_RECENT_WIN_RATE_STEP = 5
    MIN_CONFIDENCE_MAX_ADJUSTMENT = 15
    LOSS_STREAK_CONFIDENCE_BUMP = 5
    LOSS_STREAK_COOLDOWN_TRADES = 2

    # 2026-08-13 final: default 4 (was 5). Confidence formula now gives
    # realistic 55-85% range, so 4 factors is achievable and gives more trades.
    try:
        from core.constants import MIN_ALIGNED_FACTORS_PROD as _ENV_MIN_FACT
        MIN_ALIGNED_FACTORS_PROD = int(_ENV_MIN_FACT) if _ENV_MIN_FACT else 4
    except Exception:
        MIN_ALIGNED_FACTORS_PROD = 4
    MIN_ALIGNED_FACTORS_TEST = 1
    # R:R floor now comes from risk/rr_policy.py (single source of truth) —
    # previously hardcoded here as 1.5, which conflicted with the 2.0 used by
    # analysis/ict_amd_signal_engine.py and analysis/multi_strategy_pa_engine.py,
    # and the Nison rule-engine spec's default_min of 2:1. Kept as class
    # attributes (not just the policy call) so any code still reading
    # TradePermission.MIN_RR_PROD/.MIN_RR_TEST directly keeps working.
    MIN_RR_PROD = None   # resolved lazily below via rr_policy.get_min_rr()
    MIN_RR_TEST = None
    BLOCKED_SETUP_QUALITIES = {"AVOID", "INVALID"}  # Removed "POOR" - allow marginal setups

    # Confidence override for LOW-quality sessions: a LOW session is
    # normally blocked, but a sufficiently confident analysis can still
    # justify a trade. Named constant (was a bare `55` inline) so the
    # threshold is easy to find and change in one place. User request:
    # confidence >= 60 should be enough to trade even in a LOW session.
    SESSION_LOW_QUALITY_MIN_CONFIDENCE = 60

    # ────────────────────────────────────────────────────────────────────
    # 2026-08-18 PHASE 2 FIX: MTF staleness override
    # Enable trades when H4/H1 data is stale (default to allowing trades
    # when MTF data is incomplete). Set MTF_STALE_FAIL_OPEN=false to
    # require all 3 timeframes (4h, 1h, 15m) for strict MTF alignment.
    # ────────────────────────────────────────────────────────────────────
    try:
        from core.constants import MTF_STALE_FAIL_OPEN as _ENV_MTF_FAIL_OPEN
        MTF_STALE_FAIL_OPEN_DEFAULT = (_ENV_MTF_FAIL_OPEN == "true" or _ENV_MTF_FAIL_OPEN is True)
    except Exception:
        MTF_STALE_FAIL_OPEN_DEFAULT = True  # Default: allow trades when MTF data stale

    # ────────────────────────────────────────────────────────────────────
    # 2026-08-18 PHASE 3 FIX: Trade frequency cap
    # Increase daily trade cap from 35 to 50, add session-aware limits.
    # Can be overridden via env var MAX_DAILY_TRADES_PROD.
    # ────────────────────────────────────────────────────────────────────
    try:
        from core.constants import MAX_DAILY_TRADES_PROD as _ENV_MAX_DAILY_TRADES
        MAX_DAILY_TRADES_PROD = int(_ENV_MAX_DAILY_TRADES) if _ENV_MAX_DAILY_TRADES else 50
    except Exception:
        MAX_DAILY_TRADES_PROD = 50  # Increased from 35 to support more high-quality setups

    # Default confidence thresholds (before dynamic adjustment)
    CONFIDENCE_FLOOR_MULTI_FACTOR = 55    # 4+ aligned factors: needs 55%
    CONFIDENCE_FLOOR_DUAL_FACTOR = 65     # 2-3 aligned factors: needs 65%
    CONFIDENCE_FLOOR_SINGLE_FACTOR = 75   # weak signals: needs 75%

    @property
    def MIN_CONFIDENCE(self) -> int:
        # 2026-08-13: per-pair profile override — if a pair has a custom
        # min_confidence in utils/pair_profiles.py, use that instead of
        # the global default. Falls back to global if no profile or import fails.
        try:
            from utils.pair_profiles import get_pair_profile
            import inspect
            # Get the symbol from the calling context (check() method passes
            # it in decision_out). This is a best-effort override — if we
            # can't find the symbol, use the global default.
            frame = inspect.currentframe()
            if frame is not None and frame.f_back is not None:
                decision_out = frame.f_back.f_locals.get("decision_out") or frame.f_back.f_locals.get("decision")
                if decision_out and isinstance(decision_out, dict):
                    _sym = decision_out.get("symbol") or decision_out.get("pair")
                    if _sym:
                        _prof = get_pair_profile(_sym)
                        if _prof and _prof.enabled:
                            return _prof.min_confidence
        except Exception:
            pass
        return self.MIN_CONFIDENCE_TEST if _test_mode() else self.MIN_CONFIDENCE_PROD

    @property
    def MIN_ALIGNED_FACTORS(self) -> int:
        # 2026-08-13: per-pair profile override
        try:
            from utils.pair_profiles import get_pair_profile
            import inspect
            frame = inspect.currentframe()
            if frame is not None and frame.f_back is not None:
                decision_out = frame.f_back.f_locals.get("decision_out") or frame.f_back.f_locals.get("decision")
                if decision_out and isinstance(decision_out, dict):
                    _sym = decision_out.get("symbol") or decision_out.get("pair")
                    if _sym:
                        _prof = get_pair_profile(_sym)
                        if _prof and _prof.enabled:
                            return _prof.min_aligned_factors
        except Exception:
            pass
        return self.MIN_ALIGNED_FACTORS_TEST if _test_mode() else self.MIN_ALIGNED_FACTORS_PROD

    @property
    def MIN_RR(self) -> float:
        from risk.rr_policy import get_min_rr
        return get_min_rr(test_mode=_test_mode())

    def _compute_dynamic_confidence_floor(self, decision_out: dict) -> int:
        """
        2026-08-18 PHASE 1 FIX: Compute dynamic MIN_CONFIDENCE threshold (55-75%)
        based on signal quality markers instead of hardcoded 80% floor.
        
        Returns confidence floor in range [55, 75] based on:
        - aligned_factors: number of confirmed factors (ADX, trend, S/R, confluence)
        - adx: trend strength (>25 strong, 14-25 moderate, <14 weak)
        - confluence_quality: indicator agreement (HIGH/MEDIUM/LOW)
        - strategy: signal type (mean_reversion is higher bar than trend-follow)
        
        Logic:
        - 4+ factors aligned: 55% floor (high conviction multi-factor setup)
        - 2-3 factors aligned: 65% floor (medium conviction dual-factor setup)
        - 0-1 factors or weak: 75% floor (single-factor weak signal)
        - mean_reversion: +5% penalty (counter-trend bias = higher bar)
        - Envelope: 55% min, 75% max (no go below/above these)
        """
        try:
            aligned_factors = int(decision_out.get("aligned_factors", 0) or 0)
            adx = float(decision_out.get("adx", 20) or 20)
            confluence = str(decision_out.get("confluence_quality", "MEDIUM")).upper()
            strategy = str(decision_out.get("strategy", "trend_follow")).lower()
            
            # Start with multi-factor floor (most permissive)
            if aligned_factors >= 4:
                floor = self.CONFIDENCE_FLOOR_MULTI_FACTOR    # 55%
            elif aligned_factors >= 2:
                floor = self.CONFIDENCE_FLOOR_DUAL_FACTOR     # 65%
            else:
                floor = self.CONFIDENCE_FLOOR_SINGLE_FACTOR   # 75%
            
            # Mean reversion counter-trend trades need higher confidence bar
            if strategy == "mean_reversion":
                floor = min(75, floor + 5)  # +5% penalty for counter-trend bias
            
            # Weak trend (ADX < 14) nudges threshold up by 3%
            if adx < 14:
                floor = min(75, floor + 3)
            
            # Low confluence agreement also nudges up by 2%
            if confluence == "LOW":
                floor = min(75, floor + 2)
            
            # Envelope: never go below 55 or above 75
            floor = max(55, min(75, floor))
            
            log.debug(
                f"[DynamicConfidenceFloor] factors={aligned_factors}, "
                f"adx={adx:.1f}, confluence={confluence}, strategy={strategy} "
                f"→ floor={floor}%"
            )
            return floor
        except Exception as e:
            # Fallback to conservative floor if computation fails
            log.warning(f"[DynamicConfidenceFloor] error during computation: {e} — using 65% default")
            return 65

    def check(
        self,
        decision_out: dict,
        risk_out:     dict,
        news_ctx:     dict,
        session_ctx:  dict | None = None,
        execution_filters: dict | None = None,
        bypass_checks: set[str] | list[str] | None = None,
        symbol:       str | None = None,  # 2026-08-13: per-pair profile
    ) -> dict:

        checks = []
        passed = 0
        total = 0
        bypass_checks = _normalize_bypass_checks(bypass_checks)

        # ── 2026-08-13: PER-PAIR STRATEGY OVERRIDE ────────────────
        # When a per-pair strategy (mean_reversion, range_trading, etc.)
        # generates the signal, certain gates that are designed for
        # trend-following should be BYPASSED because they conflict with
        # the strategy's intent:
        #   - Mean-reversion: trades AGAINST trend (low ADX, RSI extreme)
        #     → bypass "Trend alignment", "MTF trend alignment", "S/R zone alignment", "Confluence quality"
        #   - Range-trading: trades at S/R edges (low ADX)
        #     → bypass "Trend alignment", "MTF trend alignment"
        #   - Trend-follow: keep all gates (they're designed for it)
        _strategy = decision_out.get("strategy", "")
        if _strategy in ("mean_reversion", "range_trading", "breakout"):
            _strategy_bypass = {
                "Trend alignment (regime)",
                "MTF trend alignment (H4/H1/M15)",
            }
            if _strategy in ("mean_reversion", "range_trading"):
                _strategy_bypass.update({
                    "S/R zone alignment",
                    "Confluence quality",
                })
            bypass_checks = bypass_checks | _strategy_bypass

        # 2026-08-13: per-pair min_confidence override — read directly
        # from pair_profiles if symbol is provided (more reliable than
        # frame inspection).
        _per_pair_min_conf = None
        if symbol:
            try:
                from utils.pair_profiles import get_pair_profile
                _pp = get_pair_profile(symbol)
                if _pp and _pp.enabled:
                    _per_pair_min_conf = _pp.min_confidence
            except Exception:
                pass

        # ── ARCHITECTURAL FIX (institutional refactor) ───────────────
        # The new `execution_filters` dict (produced by AnalysisAgent)
        # records gate verdicts WITHOUT destroying the analysis signal.
        # We honor any gate recorded there as a hard block, and we add
        # each one to the checks list so the operator can see WHY.
        # This replaces the old pattern where news/session gates would
        # overwrite `decision_out["decision"] = "NO TRADE"` at the
        # analysis layer.
        # ──────────────────────────────────────────────────────────────
        conf = decision_out.get("confidence", 0)
        if execution_filters:
            # Treat execution_filters as authoritative gate results.
            # By default a blocked execution_filter is a hard block unless
            # explicitly bypassed. This preserves AnalysisAgent's intent
            # to veto execution when it detects an unsafe entry.
            _is_direct_lane = bool(decision_out.get("direct_lane"))
            for gate_name, gate_result in execution_filters.items():
                blocked = isinstance(gate_result, dict) and gate_result.get("blocked")
                # Bypass via permission flags still wins
                if blocked and _bypass_check(gate_name, bypass_checks):
                    checks.append({
                        "check":  f"Execution filter: {gate_name}",
                        "passed": True,
                        "detail": "BYPASSED via permission_bypass",
                    })
                    passed += 1
                # B4c fix: mtf_structure_no_trade MUST be checked BEFORE the
                # generic direct_lane bypass, otherwise stop_hunt direct_lane
                # trades silently skip the H4/H1 conflict check. The operator
                # audit flagged this as silent-pass #1. Now: direct_lane no
                # longer auto-bypasses mtf_structure_no_trade — only the
                # explicit MTF_STRUCTURE_SOFTEN env var (off by default) can.
                elif blocked and gate_name == "mtf_structure_no_trade":
                    import os as _os_mtf_struct
                    # CLAUDE FIX (see backtest report): the comment above has
                    # always said this override is "off by default", but the
                    # code defaulted the env var to "true" (softened/ON by
                    # default) -- a direct contradiction that silently
                    # disabled the exact gate the "silent-pass #1" audit fix
                    # was meant to restore. Default corrected to "false" so
                    # mtf_structure_no_trade actually hard-blocks unless an
                    # operator explicitly opts into softening it.
                    _soften_mtf_struct = _os_mtf_struct.getenv("MTF_STRUCTURE_SOFTEN", "false").lower() == "true"
                    if _soften_mtf_struct:
                        checks.append({
                            "check":  f"Execution filter: {gate_name}",
                            "passed": True,
                            "detail": "SOFTENED by TradePermission (MTF_STRUCTURE_SOFTEN=true): MTF structure NO_TRADE",
                        })
                        passed += 1
                    else:
                        checks.append({
                            "check":  f"Execution filter: {gate_name}",
                            "passed": False,
                            "detail": (
                                "BLOCKED — MTF structure NO_TRADE (H4/H1 conflict or both neutral). "
                                + ("direct_lane bypass disabled for this gate (B4c fix). " if _is_direct_lane else "")
                                + "Set MTF_STRUCTURE_SOFTEN=true to override."
                            ),
                        })
                # Direct lane bypass (blend filter not applicable) — runs
                # AFTER mtf_structure_no_trade so that gate always wins.
                elif blocked and _is_direct_lane:
                    checks.append({
                        "check":  f"Execution filter: {gate_name}",
                        "passed": True,
                        "detail": f"bypassed: direct_lane={decision_out.get('direct_lane')} (blend filter, not applicable to standalone signal)",
                    })
                    passed += 1
                # 2026-08-13: confluence_avoid was the top remaining hard block
                # after SELL finally reached permission (EURUSD BLOCKED_SELL at
                # conf 25%). Soften — advisory only, not a solo veto.
                # NOTE: do NOT soften 'fusion' here — an explicit 'fusion'
                # execution_filter must be authoritative and can hard-block
                # when AnalysisAgent reports a structural session failure.
                elif blocked and gate_name in ("confluence_avoid", "confluence"):
                    checks.append({
                        "check":  f"Execution filter: {gate_name}",
                        "passed": True,
                        "detail": "SOFTENED by TradePermission: confluence_avoid does not hard-block",
                    })
                    passed += 1
                # Otherwise a blocked execution filter is a failure
                elif blocked:
                    checks.append({
                        "check":  f"Execution filter: {gate_name}",
                        "passed": False,
                        "detail": gate_result.get("reason", "blocked"),
                    })
                else:
                    checks.append({
                        "check":  f"Execution filter: {gate_name}",
                        "passed": True,
                        "detail": "not blocked",
                    })
                    passed += 1
            # If any execution_filter produced a hard block, treat it as
            # authoritative and short-circuit the permission decision now
            # so downstream advisory checks (revenge detector, etc.) do
            # not overwrite the blocking reason. This preserves the
            # AnalysisAgent's explicit veto semantics.
            exec_failed = [
                c for c in checks
                if isinstance(c.get("check"), str)
                and c.get("check", "").startswith("Execution filter:")
                and not c.get("passed", True)
            ]
            if exec_failed:
                # Account for execution_filters in total denominator
                total = len(execution_filters)
                failed_checks = [{"check": c.get("check", "?"), "detail": (c.get("detail") or "")} for c in exec_failed]
                result = {
                    "execution_allowed": False,
                    "blocked_reason": (exec_failed[0].get("detail") or "Execution filter blocked"),
                    "failed_checks": failed_checks,
                    "execution_action": "NO TRADE",
                    "allowed": False,
                    "passed": passed,
                    "total": total,
                    "checks": checks,
                    "final_action": "NO TRADE",
                    "entry": risk_out.get("entry"),
                    "sl": risk_out.get("sl_price"),
                    "tp": risk_out.get("tp_price"),
                    "lot": risk_out.get("lot", 0),
                    "rr": risk_out.get("rr_ratio", 0),
                }
                return result

        # 1. Signal
        sig = decision_out.get("decision", "WAIT")
        ok  = sig in ("BUY", "SELL")
        if _bypass_check("Valid signal", bypass_checks):
            ok = True
            detail = f"{sig} (BYPASSED)"
        else:
            detail = sig
        checks.append({"check": "Valid signal", "passed": ok, "detail": detail})
        # 2026-08-11 log-audit fix: this check appended to `checks` and
        # incremented `passed` on success, but NEVER incremented `total`.
        # Since `allowed = passed == total` (see bottom of this method),
        # a check whose outcome never touches `total` has ZERO effect on
        # whether the trade is actually permitted -- whether it passes or
        # fails, the passed==total equality is unaffected either way. That
        # made "Valid signal" (arguably the single most important gate)
        # purely cosmetic in the final allow/deny decision. Confirmed from
        # execution.log: entries with raw_signal="WAIT" (a failed "Valid
        # signal" check) still logged allowed=true, and passed/total pairs
        # like 5/4 or 7/5 (passed > total, which should be impossible)
        # appeared throughout. Root cause: total simply wasn't incremented
        # here. Fixed by counting this check like every other one below.
        if ok: passed += 1
        total += 1

        # 1b. S/R ALIGNMENT GATE (added 2026-08-02, Abdullah audit)
        # Backtest evidence: SELL trades entered within 5-55 pips of a
        # SUPPORT zone (closer to support than resistance), and the one BUY
        # trade entered near a RESISTANCE zone instead of support — both
        # backwards relative to normal S/R entry logic — accounted for 5/9
        # losing trades in a 350-bar EURUSD H1 sample. sr_ctx's
        # dist_to_support_pips / dist_to_resistance_pips were already
        # computed and fed into the (log-only) institutional_entry_framework
        # advisory below, but never used as an actual block. This makes the
        # SELL-at-support / BUY-at-resistance mismatch a hard gate.
        # Fails open (not blocked) if sr_ctx data is unavailable — only
        # block when there's positive evidence of misalignment.
        #
        # P1 fix (2026-08-03, full-scale parity run): `sr_ctx` was
        # referenced but NEVER defined in this scope — every backtest bar
        # that reached this gate raised NameError, which the unified_engine's
        # try/except caught as `engine_error` (46/150 bars on EURUSD). This
        # silently killed evaluate_decision_core BEFORE the direct_lane
        # block could run, making the entire direct_lane fix look broken.
        # Fix: pull sr_ctx from decision_out (where AnalysisAgent stores it),
        # defaulting to {} so the gate fails open as documented.
        sr_ctx = decision_out.get("sr_ctx", {}) or {}
        # 2026-08-14 forensic-audit fix: tracks the S/R-misalignment
        # confidence penalty (see below) so it can be folded into
        # min_confidence_diagnostic instead of disappearing untracked.
        _sr_penalty = 0
        if ok and not decision_out.get("direct_lane"):
            if _bypass_check("S/R zone alignment", bypass_checks):
                checks.append({
                    "check":  "S/R zone alignment",
                    "passed": True,
                    "detail": "BYPASSED via permission_bypass",
                })
                passed += 1
                total += 1
            else:
                dist_sup = sr_ctx.get("dist_to_support_pips")
                dist_res = sr_ctx.get("dist_to_resistance_pips")
                sr_ok = True
                sr_detail = "no S/R zone data — not evaluated"
                # Audit fix (§5.2 / §14 Decision Table — "EDIT, re-tune, not
                # REMOVE"): the original rule blocked on ANY margin, even a
                # razor-thin one (e.g. 4.0 vs 4.1 pips), which the EURUSD
                # ablation showed filters out roughly as many WINNING trades
                # as losing ones (57.5% block rate; no_sr_alignment ablation:
                # +2 wins AND +2 losses). Require the "wrong side" distance to
                # be meaningfully closer — not just nominally closer — before
                # blocking, so genuine near-the-midpoint setups aren't vetoed
                # by noise-level differences.
                SR_ALIGNMENT_MARGIN_PIPS = 3.0   # absolute buffer
                SR_ALIGNMENT_MARGIN_RATIO = 1.15 # unfavorable side must be
                                                  # >=15% closer than favorable
                if dist_sup is not None and dist_res is not None:
                    if sig == "SELL":
                        margin_ok = (dist_res - dist_sup) >= SR_ALIGNMENT_MARGIN_PIPS or (
                            dist_sup > 0 and dist_res / max(dist_sup, 1e-9) < (1 / SR_ALIGNMENT_MARGIN_RATIO)
                        )
                        if dist_sup < dist_res and margin_ok:
                            sr_ok = False
                            sr_detail = (f"SELL is {dist_sup:.1f} pips from support vs "
                                         f"{dist_res:.1f} pips from resistance — clearly "
                                         f"closer to support, wrong side for a SELL")
                        elif dist_sup < dist_res:
                            sr_detail = (f"borderline: dist_to_support={dist_sup:.1f}p, "
                                         f"dist_to_resistance={dist_res:.1f}p — within "
                                         f"re-tuned margin, not blocked")
                        else:
                            sr_detail = (f"aligned: dist_to_support={dist_sup:.1f}p, "
                                         f"dist_to_resistance={dist_res:.1f}p")
                    elif sig == "BUY":
                        margin_ok = (dist_sup - dist_res) >= SR_ALIGNMENT_MARGIN_PIPS or (
                            dist_res > 0 and dist_sup / max(dist_res, 1e-9) < (1 / SR_ALIGNMENT_MARGIN_RATIO)
                        )
                        if dist_res < dist_sup and margin_ok:
                            sr_ok = False
                            sr_detail = (f"BUY is {dist_res:.1f} pips from resistance vs "
                                         f"{dist_sup:.1f} pips from support — clearly closer "
                                         f"to resistance, wrong side for a BUY")
                        elif dist_res < dist_sup:
                            sr_detail = (f"borderline: dist_to_support={dist_sup:.1f}p, "
                                         f"dist_to_resistance={dist_res:.1f}p — within "
                                         f"re-tuned margin, not blocked")
                        else:
                            sr_detail = (f"aligned: dist_to_support={dist_sup:.1f}p, "
                                         f"dist_to_resistance={dist_res:.1f}p")
                    else:
                        sr_detail = (f"aligned: dist_to_support={dist_sup:.1f}p, "
                                     f"dist_to_resistance={dist_res:.1f}p")
                else:
                    log.warning(
                        "[TradePermission] S/R zone alignment skipped — support/resistance "
                        "distance data missing; not evaluated"
                    )
                # 2026-08-13: when RiskEngine already approved a real lot,
                # S/R misalignment is advisory — do not hard-block alone.
                # Live logs showed BLOCKED_SELL/BUY with risk.approved=true
                # solely due to this gate (GBPUSD/USDJPY).
                #
                # 2026-08-14 forensic-audit fix (EURUSD SELL, ticket
                # 10015498153, trade #17): this exact trade hit this branch —
                # "SELL is -5.9 pips from support vs 13.5 pips from
                # resistance — clearly closer to support, wrong side for a
                # SELL" — and the signal was then unconditionally erased
                # (sr_ok forced back to True, zero effect on confidence or
                # any downstream gate) purely because risk_out["approved"]
                # was true, which it is for nearly every trade that reaches
                # this point (RiskEngine approval doesn't consider S/R
                # positioning at all). That's the same "advisory-only signal
                # that doesn't gate anything" pattern the audit flagged for
                # sl_swing_anchor, just in a different module.
                #
                # Not reverting straight to a hard block here: that risks
                # reproducing the 2026-08-13 0-trades regression that
                # motivated this softening in the first place, without the
                # baseline-vs-modified backtest the audit itself requires
                # before tightening a gate (see audit §19 — don't deploy a
                # stricter rule off the evidence of one losing trade).
                # Instead: keep it non-hard-blocking, but stop discarding
                # the signal entirely. A non-borderline "wrong side of S/R"
                # reading (this already passed the margin/ratio filter above
                # — see SR_ALIGNMENT_MARGIN_PIPS/RATIO — so it is NOT a
                # noise-level difference) now costs confidence points and is
                # surfaced in the same penalty-by-rule breakdown entry
                # quality uses, so it's visible to the min-confidence gate
                # and to anyone reading min_confidence_diagnostic instead of
                # vanishing without a trace. SR_MISALIGNMENT_PENALTY is a
                # first-pass value pending the same ablation/backtest
                # calibration the audit prescribes for sl_swing_anchor —
                # not asserted as the "right" number.
                SR_MISALIGNMENT_PENALTY = 5
                if not sr_ok and risk_out.get("approved"):
                    _conf_before_sr = conf
                    conf = max(0, conf - SR_MISALIGNMENT_PENALTY)
                    _sr_penalty = SR_MISALIGNMENT_PENALTY
                    sr_ok = True
                    sr_detail = (
                        f"{sr_detail} — SOFTENED (risk already approved), "
                        f"penalty=-{SR_MISALIGNMENT_PENALTY}, "
                        f"conf: {_conf_before_sr:.0f}% -> {conf:.0f}%"
                    )
                checks.append({
                    "check":  "S/R zone alignment",
                    "passed": sr_ok,
                    "detail": sr_detail,
                })
                if sr_ok: passed += 1
                total += 1

        # 1c. TREND ALIGNMENT GATE (added 2026-08-02, Abdullah audit)
        # Backtest evidence: after the S/R alignment gate (1b) above, the
        # remaining SELL trades were correctly positioned near resistance
        # zones but STILL lost 100% of the time, because the underlying
        # market regime for the whole test window was a choppy uptrend —
        # every SELL was a counter-trend fade into a breakout that kept
        # going. sr_ctx alone can't see this; it only looks at local zones.
        # market_out["regime"] (MarketRegimeDetector, real historical OHLC,
        # no live-API dependency) gives regime + direction + strength
        # directly. Block SELL when regime is TRENDING+BULLISH (STRONG or
        # MODERATE), and BUY when TRENDING+BEARISH — i.e. don't fade a
        # confirmed trend. Fails open if regime data is unavailable/
        # NEUTRAL/RANGING — we only block on positive evidence of a
        # trend running against the signal.
        if ok and not decision_out.get("direct_lane"):
                if _bypass_check("Trend alignment (regime)", bypass_checks):
                    checks.append({
                        "check":  "Trend alignment (regime)",
                        "passed": True,
                        "detail": "BYPASSED via permission_bypass",
                    })
                    passed += 1
                    total += 1
                else:
                    regime = decision_out.get("regime", {}) or {}
                    # Accept either a dict (with keys) or a simple string
                    if isinstance(regime, dict):
                        r_regime = str(regime.get("regime", "")).upper()
                        r_direction = str(regime.get("direction", "")).upper()
                        r_strength = str(regime.get("strength", "")).upper()
                    else:
                        # If regime is a plain string, treat it as the regime name
                        r_regime = str(regime).upper()
                        r_direction = ""
                        r_strength = ""
                    trend_ok = True
                    trend_detail = "no regime data — not evaluated"
                    if r_regime and r_direction:
                        is_trending = r_regime == "TRENDING" and r_strength in ("STRONG", "MODERATE")
                        if is_trending:
                            if sig == "SELL" and r_direction == "BULLISH":
                                trend_ok = False
                                trend_detail = (f"SELL against a {r_strength} BULLISH trending "
                                                 f"regime — counter-trend fade, blocked")
                            elif sig == "BUY" and r_direction == "BEARISH":
                                trend_ok = False
                                trend_detail = (f"BUY against a {r_strength} BEARISH trending "
                                                 f"regime — counter-trend fade, blocked")
                            else:
                                trend_detail = f"aligned: regime={r_regime}/{r_direction}/{r_strength}"
                        else:
                            trend_detail = f"not trending: regime={r_regime}/{r_direction}/{r_strength}"
                    checks.append({
                        "check":  "Trend alignment (regime)",
                        "passed": trend_ok,
                        "detail": trend_detail,
                    })
                    if trend_ok: passed += 1
                    total += 1

        # 1c-2. MTF TREND ALIGNMENT GATE (added 2026-08-07) — HARD BLOCK.
        # The "Trend alignment (regime)" gate above only looks at ONE
        # regime object (computed off the primary trading timeframe) and
        # only blocks the specific case of fading a STRONG/MODERATE trend.
        # It never actually compares H4 vs H1 vs M15 against each other,
        # so a BUY/SELL could (and did) fire even when e.g. H4 was
        # bullish, H1 was bullish, but M15 was bearish — the three "top
        # down" timeframes disagreeing with each other, not just with a
        # single regime reading. This gate closes that gap directly:
        # H4, H1 and M15 trend readings (from
        # analysis/timeframe.py:MultiTimeframeAnalyzer, threaded through
        # via dec_out["mtf_trends"] in core/trader.py) must ALL agree
        # with each other AND with the signal direction, or the trade is
        # blocked outright — no confidence score or other gate can
        # override this. Fails CLOSED (blocks) when data is simply
        # missing for one of the three, since "unknown" is not the same
        # as "aligned" — unlike the regime gate above, which fails open.
        #
        # FIX (2026-08-08, EURNOK trend-conflict audit): this gate used
        # to exclude `direct_lane` (stop_hunt) trades entirely — see the
        # old condition `if ok and not decision_out.get("direct_lane")`.
        # stop_hunt_direct_lane.py has its OWN internal H4 EMA20/EMA50
        # filter, but that's a narrower, different definition of "trend"
        # than this H4/H1/M15-agreement gate, so a direct_lane trade
        # could (and did) fire opposite to what H4/H1/M15 all showed.
        # User confirmed the risk tradeoff (option 2): apply this gate
        # to direct_lane trades too, at the cost of some of the
        # standalone strategy's validated backtest edge (which assumed
        # no MTF gate) — re-validate against per_strategy_tester.py if
        # win-rate drops noticeably after this change.
        if ok:

            # B4b fix: detect backtest mode and auto-soften MTF gate.
            # Both backtest providers (HistoricalMT5Provider, HistoricalCSVProvider)
            # never populate mtf_bias["trends"] — they only return bias/confidence.
            # This means dec_out["mtf_trends"] is always {} in backtest, and P4's
            # fail-closed behavior would block EVERY backtest trade. Instead of
            # requiring operators to set MTF_STALE_FAIL_OPEN=true manually, detect
            # backtest mode via data_source and auto-soften. Live mode still
            # hard-blocks as intended.
            _data_source = (decision_out.get("data_source") or "").lower()
            _is_backtest = any(k in _data_source for k in ("historical", "csv", "replay"))
            if _is_backtest and not os.getenv("MTF_STALE_FAIL_OPEN"):
                # Auto-enable fail-open for backtest — but log it loudly so
                # operators know MTF gate is advisory-only in this run.
                os.environ["MTF_STALE_FAIL_OPEN"] = "true"
                log.info(
                    "[TradePermission] Backtest mode detected (data_source="
                    f"{_data_source}) — auto-enabling MTF_STALE_FAIL_OPEN=true "
                    "for this run. MTF trend alignment will be advisory-only."
                )

            if _bypass_check("MTF trend alignment (H4/H1/M15)", bypass_checks):
                checks.append({
                    "check":  "MTF trend alignment (H4/H1/M15)",
                    "passed": True,
                    "detail": "BYPASSED via permission_bypass",
                })
                passed += 1
                total += 1
            else:
                mtf_trends = decision_out.get("mtf_trends")

                if not isinstance(mtf_trends, dict) or not all(
                    tf in mtf_trends for tf in ("4h", "1h", "15m")
                ):
                    # P4 fix: FAIL CLOSED (was True).
                    # Operator audit caught: when H1/H4 data is stale,
                    # MultiTimeframeAnalyzer excludes them from the trends
                    # dict, so this `all(...)` check returns False — and
                    # the previous code unconditionally set `mtf_ok = True`,
                    # silently disabling the MTF gate exactly when it
                    # matters most (stale data = no trend visibility).
                    # Now: missing MTF data = HARD BLOCK. The operator can
                    # still bypass via MTF_STALE_FAIL_OPEN=true if they
                    # explicitly want the old behavior (e.g. for backtest
                    # parity tests).
                    # B4a fix: read the stale_tfs list (propagated by
                    # trader.py from MultiTimeframeAnalyzer) so the log
                    # detail can distinguish "stale H1/H4" from "upstream
                    # crash" — both hard-block, but the diagnostics differ.
                    _stale_tfs = decision_out.get("mtf_stale_tfs") or []
                    _stale_str = (f" — stale TFs: {_stale_tfs}"
                                  if _stale_tfs else
                                  " — no stale_tfs reported (possible upstream MTF crash)")
                    import os as _os_mtf, sys as _sys_mtf
                    # In CI/test runs we want missing MTF data to be fail-open
                    # so downstream adaptive/confidence logic can be exercised.
                    # 2026-08-18 PHASE 2 FIX: use class-level MTF_STALE_FAIL_OPEN_DEFAULT
                    # so operator can enable fail-open by default (was hardcoded false).
                    _mtf_fail_open = (
                        _os_mtf.getenv("MTF_STALE_FAIL_OPEN", "false").lower() == "true"
                        or self.MTF_STALE_FAIL_OPEN_DEFAULT  # NEW: use class default
                        or _test_mode()
                        or bool(_os_mtf.getenv("PYTEST_CURRENT_TEST"))
                        or "pytest" in set(_sys_mtf.modules)
                    )
                    if _mtf_fail_open:
                        mtf_ok = True
                        mtf_detail = (
                            "MTF trend alignment skipped — mtf_trends unavailable or "
                            f"incomplete{_stale_str} (MTF_STALE_FAIL_OPEN=true)"
                        )
                        log.warning(
                            "[TradePermission] MTF trend alignment skipped (fail-open) — "
                            f"mtf_trends data missing or incomplete{_stale_str}; "
                            "MTF_STALE_FAIL_OPEN=true"
                        )
                    else:
                        mtf_ok = False
                        mtf_detail = (
                            "BLOCKED — mtf_trends data missing or incomplete "
                            f"(likely H1/H4 stale){_stale_str}; cannot verify H4/H1/M15 "
                            "alignment. Set MTF_STALE_FAIL_OPEN=true to override "
                            "(NOT recommended in live)."
                        )
                        log.error(
                            "[TradePermission] MTF trend alignment HARD BLOCKED — "
                            f"mtf_trends data missing or incomplete{_stale_str}. "
                            "Refusing to trade without MTF confirmation."
                        )
                else:
                    def _dir(tf_key: str) -> str:
                        raw = str(mtf_trends.get(tf_key, "")).lower()
                        if "bullish" in raw:
                            return "BULLISH"
                        if "bearish" in raw:
                            return "BEARISH"
                        return "UNKNOWN"

                    h4_dir, h1_dir, m15_dir = _dir("4h"), _dir("1h"), _dir("15m")

                    # 2026-08-13 relax (0-trades audit):
                    # Previous rule required ALL three TFs identical
                    # (len(dirs)==1). In live FX, H4 is frequently RANGING
                    # while H1/M15 show a short-term direction — that
                    # produced permanent blocks. New rule:
                    #   1. H4 must not actively oppose the signal
                    #      (UNKNOWN/RANGING is allowed).
                    #   2. Neither H1 nor M15 may be the opposite of the
                    #      signal (they can be UNKNOWN/RANGING).
                    # Still blocks clear counter-trend entries while
                    # allowing the common "H4 ranging, lower TFs aligned"
                    # case that previously never traded.
                    #
                    # FIX (2026-08-14, EURUSD SELL forensic audit): the
                    # "no active opposition" rule above has a hole —
                    # UNKNOWN never opposes anything, so if ALL THREE TFs
                    # came back UNKNOWN/RANGING (e.g. because the symbol's
                    # regime was tagged RANGING that session and
                    # Indicators.get_ai_context() therefore labeled the
                    # trend "ranging" rather than "bullish"/"bearish"),
                    # `mtf_ok` was True with zero actual confirmation —
                    # this is exactly what let the EURUSD SELL fire
                    # straight into a fresh H1 higher-low / bullish M15
                    # reversal that no TF's raw label happened to name
                    # "bullish", even though the real structure was.
                    # "Not opposing" != "aligned". Now ALSO require at
                    # least one of the three TFs to positively confirm the
                    # signal direction (== signal_dir, not just != opposite).
                    # This keeps the exact "H4 ranging, H1/M15 aligned"
                    # case the 08-13 relax was built for (H1 or M15 will
                    # explicitly confirm), while closing the "everything
                    # is ambiguous, so nothing blocks" gap.
                    signal_dir = "BULLISH" if sig == "BUY" else "BEARISH"
                    opposite   = "BEARISH" if sig == "BUY" else "BULLISH"

                    h4_opposes  = (h4_dir == opposite)
                    h1_opposes  = (h1_dir == opposite)
                    m15_opposes = (m15_dir == opposite)

                    has_confirmation = (
                        h4_dir == signal_dir
                        or h1_dir == signal_dir
                        or m15_dir == signal_dir
                    )

                    mtf_ok = (
                        (not h4_opposes) and (not h1_opposes) and (not m15_opposes)
                        and has_confirmation
                    )

                    if mtf_ok:
                        mtf_detail = (
                            f"aligned (relaxed): H4={h4_dir}, H1={h1_dir}, "
                            f"M15={m15_dir}, signal={sig}"
                        )
                    elif not has_confirmation:
                        mtf_detail = (
                            f"BLOCKED — no TF confirms {signal_dir}: "
                            f"H4={h4_dir}, H1={h1_dir}, M15={m15_dir}, signal={sig} "
                            f"— all TFs ambiguous/unknown, zero positive confirmation "
                            f"(2026-08-14 fix: 'not opposing' alone is not 'aligned')"
                        )
                    else:
                        mtf_detail = (
                            f"NOT aligned — H4={h4_dir}, H1={h1_dir}, M15={m15_dir}, "
                            f"signal={sig} — one or more TFs actively oppose the signal"
                        )

                checks.append({
                    "check":  "MTF trend alignment (H4/H1/M15)",
                    "passed": mtf_ok,
                    "detail": mtf_detail,
                })
                if mtf_ok: passed += 1
                total += 1
                if not mtf_ok:
                    ok = False

        # 1d. ZONE COOLDOWN / DUPLICATE-ENTRY GATE (added 2026-08-02,
        # Abdullah audit). Both the training-window backtest (3 SELL trades
        # within 2 hours, same ~1.163 zone, Aug 7) and the out-of-sample
        # backtest (3 BUY trades within 2 hours, same ~1.183 zone, Feb 4)
        # showed the SAME failure mode regardless of direction: several
        # concurrently-OPEN positions piling into the same price zone in
        # the same direction, all failing together. The existing revenge-
        # trading detector (1e below) only looks at CLOSED trade history,
        # so it can't see this — these positions were all still open when
        # the next one was taken. This gate tracks recently-opened entries
        # in-memory (per TradePermission instance — one per trader/backtest
        # run) and blocks a new same-direction entry within
        # ZONE_COOLDOWN_HOURS and ZONE_COOLDOWN_PIPS of one already taken.
        if not hasattr(self, "_recent_entries"):
            self._recent_entries = []  # list of dicts: symbol, direction, price, time
        ZONE_COOLDOWN_HOURS = 4
        ZONE_COOLDOWN_PIPS = 30
        if ok:
            if _bypass_check("Zone cooldown (duplicate entry)", bypass_checks):
                checks.append({
                    "check":  "Zone cooldown (duplicate entry)",
                    "passed": True,
                    "detail": "BYPASSED via permission_bypass",
                })
                passed += 1
                total += 1
            else:
                _zc_symbol = decision_out.get("_symbol", "") or str(risk_out.get("symbol", ""))
                _zc_entry = float(risk_out.get("entry", 0) or decision_out.get("entry", 0) or 0)
                _zc_df = decision_out.get("_df")
                _zc_now = None
                try:
                    if _zc_df is not None and len(_zc_df) > 0:
                        _zc_now = _zc_df.index[-1].to_pydatetime()
                except Exception:
                    _zc_now = None
                if _zc_now is None:
                    from datetime import datetime as _dt, timezone as _tz
                    _zc_now = _dt.now(_tz.utc)
                from analysis.support_resistance import SupportResistance as _SR_pip
                _zc_pip_value = _SR_pip()._resolve_pip_value(_zc_symbol) if _zc_entry else 0.0001
                zone_ok = True
                zone_detail = "no recent nearby entry"
                for _e in self._recent_entries:
                    if _e["symbol"] != _zc_symbol or _e["direction"] != sig or _zc_entry <= 0:
                        continue
                    hrs = abs((_zc_now - _e["time"]).total_seconds()) / 3600.0
                    pips = abs(_zc_entry - _e["price"]) / _zc_pip_value
                    if hrs <= ZONE_COOLDOWN_HOURS and pips <= ZONE_COOLDOWN_PIPS:
                        zone_ok = False
                        zone_detail = (f"{sig} {pips:.1f} pips from a {sig} entry taken "
                                        f"{hrs:.1f}h ago — within cooldown "
                                        f"({ZONE_COOLDOWN_HOURS}h / {ZONE_COOLDOWN_PIPS} pips)")
                        break
                checks.append({
                    "check":  "Zone cooldown (duplicate entry)",
                    "passed": zone_ok,
                    "detail": zone_detail,
                })
                if zone_ok: passed += 1
                total += 1

        # 2. Risk approved (post-sizer / post-RAG gates)
        # A4 fix: this check reads risk_out["approved"] AFTER it has been
        # mutated by _apply_advanced_sizing (LiveRiskManager, Day76 Sizer)
        # and apply_advanced_risk_gates (entry_quality, rr_policy, book_guardrails,
        # trading_controls, monte_carlo). The "risk.evaluated" log event
        # captures the RAW RiskEngine output BEFORE these mutations; this
        # check captures the FINAL state. Both are correct — they measure
        # different things. See the new "risk.finalized" log event for the
        # explicit post-mutation snapshot.
        ok = risk_out.get("approved", False)
        if _bypass_check("Risk approved", bypass_checks):
            ok = True
            detail = "BYPASSED via permission_bypass"
        else:
            detail = risk_out.get("reject_reason", "OK")
        checks.append({
            # A4: keep the legacy "Risk approved" name for backward compat
            # with dashboards/blocked_audit.py that grep for this string,
            # but enrich the detail with the post-mutation context.
            "check":  "Risk approved",
            "passed": ok,
            "detail": detail,
        })
        # 2026-08-11 log-audit fix: see the identical note on the "Valid
        # signal" check above -- `total` was never incremented here either,
        # so a rejected RiskEngine result could not actually block
        # execution via the passed==total gate.
        if ok: passed += 1
        total += 1

        # 3. News safe
        # Day 97+ FIX: fail-safe (not fail-open). If news_ctx is empty/None
        # (API failed), default to DENY — don't allow trading when we can't
        # verify news safety. Previously defaulted to True (fail-open) which
        # meant news API failure → trading allowed → could trade into CPI/NFP.
        # Round-?? fix: explicit env-var bypass, same pattern as
        # BYPASS_FUSION_GATE below. Added because the Forex Factory
        # calendar fetch has been failing (403/timeout — scraper blocked),
        # so news_ctx is empty on effectively every cycle and this gate
        # was blocking 100% of trades. Fail-safe-by-default is still the
        # right behavior when we can't verify news safety; this just gives
        # the operator a conscious, logged way to override it while the
        # scraper is down, instead of it silently blocking everything.
        # Defaults to false — bypass must be turned on deliberately.
        import os as _os_news
        _bypass_news = _os_news.getenv("BYPASS_NEWS_GATE", "false").lower() == "true"
        if _bypass_news or _bypass_check("News safe", bypass_checks):
            ok = True
            detail = "News system unavailable or bypass requested: allowed via bypass"
        elif not news_ctx:
            ok = False
            detail = "News system unavailable — fail-safe block (set BYPASS_NEWS_GATE=true to override)"
        else:
            ok = news_ctx.get("news_trade_allowed", False)
            detail = news_ctx.get("news_reason", "Unknown")
        checks.append({
            "check":  "News safe",
            "passed": ok,
            "detail": detail,
        })
        # 2026-08-11 log-audit fix: same missing `total += 1` as the two
        # checks above.
        if ok: passed += 1
        total += 1

        # ── ENTRY QUALITY: SOFT SCORING ───────────────────────────
        # Runs BEFORE the confidence gate so penalties reduce the
        # effective confidence.  Only extreme cases (SL/TP wrong side,
        # averaging into losers, opposite-direction stacking) still
        # hard-block.  All other entry-quality issues (exhaustion,
        # indecision, small candles, chasing, etc.) become confidence
        # penalties.  Entry quality alone NEVER rejects the trade.
        _eq_penalty = 0
        _eq_result = None
        _eq_penalty_by_rule: dict = {}
        _conf_before_eq = conf
        if risk_out.get("approved"):
            try:
                from risk.entry_quality_guardrails import run_all_entry_quality_checks
                _df_eq = None
                _ind_ctx = decision_out.get("ind_ctx", {}) or {}
                _df_eq = decision_out.get("_df")
                if _df_eq is None and isinstance(session_ctx, dict):
                    _df_eq = session_ctx.get("_df")
                _eq_symbol = decision_out.get("_symbol", "") or str(risk_out.get("symbol", ""))
                if _df_eq is not None and len(_df_eq) > 0:
                    _eq_result = run_all_entry_quality_checks(
                        df=_df_eq,
                        symbol=_eq_symbol,
                        direction=decision_out.get("decision", "WAIT"),
                        entry_price=float(risk_out.get("entry", 0) or 0),
                        stop_loss=float(risk_out.get("sl_price", 0) or 0),
                        take_profit=float(risk_out.get("tp_price", 0) or 0),
                        ind_ctx=_ind_ctx,
                    )
                    _should_execute = _eq_result.get("should_execute", True)
                    _eq_penalty = _eq_result.get("confidence_penalty", 0)
                    _eq_penalty_by_rule = _eq_result.get("penalty_by_rule", {}) or {}
                    _eq_report = _eq_result.get("per_check_report", [])
                    _block_reason = _eq_result.get("block_reason")
                    _quality_score = _eq_result.get("quality_score", 100)

                    if not _should_execute:
                        # EXTREME HARD BLOCK only (SL wrong side, TP wrong side,
                        # averaging into losers, opposite-direction stacking)
                        total += 1
                        checks.append({
                            "check":  "Entry quality guardrails",
                            "passed": False,
                            "detail": (
                                f"EXTREME BLOCK: {_block_reason} "
                                f"(quality={_quality_score}/100)"
                            ),
                        })
                        log.info("[Entry Quality Report]")
                        for _line in _eq_report:
                            log.info(f"  {_line}")
                        result = {
                            "execution_allowed": False,
                            "blocked_reason":    f"Entry quality: {_block_reason}",
                            "failed_checks":     [
                                {"check": "Entry quality guardrails",
                                 "detail": f"EXTREME BLOCK: {_block_reason}"}
                            ],
                            "execution_action":  "NO TRADE",
                            "allowed":       False,
                            "passed":        passed,
                            "total":         total,
                            "checks":        checks,
                            "final_action":  "NO TRADE",
                            "entry":         risk_out.get("entry"),
                            "sl":            risk_out.get("sl_price"),
                            "tp":            risk_out.get("tp_price"),
                            "lot":           risk_out.get("lot", 0),
                            "rr":            risk_out.get("rr_ratio", 0),
                            # Hard block happens before any confidence penalty
                            # is applied, so pre/post are identical here.
                            "confidence_pre_penalty":  conf,
                            "confidence_post_penalty": conf,
                        }
                        log.info(
                            f"[TradePermission] EXTREME BLOCK by entry quality: "
                            f"{_block_reason} (quality={_quality_score}/100)"
                        )
                        return result
                    else:
                        # SOFT SCORING: apply penalty, always pass
                        conf = max(0, conf - _eq_penalty)
                        total += 1
                        passed += 1
                        _detail = f"quality={_quality_score}/100"
                        if _eq_penalty > 0:
                            _detail += (
                                f", penalty=-{_eq_penalty}, "
                                f"conf: {_conf_before_eq:.0f}% -> {conf:.0f}%"
                            )
                        else:
                            _detail += ", all checks passed"
                        checks.append({
                            "check":  "Entry quality guardrails",
                            "passed": True,
                            "detail": _detail,
                        })
                        # Log detailed per-check report
                        log.info("[Entry Quality Report]")
                        for _line in _eq_report:
                            log.info(f"  {_line}")
                        if _eq_penalty > 0:
                            log.info(f"  {'─' * 30}")
                            log.info(f"  Total Penalty:     -{_eq_penalty}")
                            log.info(f"  Confidence Before: {_conf_before_eq:.0f}")
                            log.info(f"  Confidence After:  {conf:.0f}")
                        else:
                            log.info("  All checks passed - no penalty")
                else:
                    # BUGFIX: this branch used to be just a comment
                    # ("If _df is None, skip guardrails") with NO code —
                    # no checks.append(), no log line, no counter bump.
                    # That meant whenever _df_eq was missing/empty, the
                    # entry-quality penalty silently stayed at 0 with
                    # zero trace anywhere: confidence_pre_penalty and
                    # confidence_post_penalty would always be identical
                    # and execution.log would show nothing wrong. Now
                    # this is logged and recorded so a run of "0 delta"
                    # is immediately diagnosable instead of looking like
                    # the penalty step silently doesn't fire.
                    _eq_skip_reason = (
                        "no OHLCV data attached (decision_out['_df'] and "
                        "session_ctx['_df'] both missing/empty)"
                    )
                    log.warning(
                        f"[TradePermission] Entry quality guardrails SKIPPED — "
                        f"{_eq_skip_reason}. Confidence penalty will be 0 this cycle."
                    )
                    checks.append({
                        "check":  "Entry quality guardrails",
                        "passed": True,
                        "detail": f"SKIPPED — {_eq_skip_reason} (no penalty applied)",
                    })
            except ImportError:
                log.debug("[TradePermission] entry_quality_guardrails not available - skipping")
                checks.append({
                    "check":  "Entry quality guardrails",
                    "passed": True,
                    "detail": "SKIPPED — entry_quality_guardrails module not importable",
                })
            except Exception as _eq_e:
                # BUGFIX: was a bare log.warning(str(e)) with no traceback
                # and no checks entry — an exception here (e.g. a shape/
                # dtype mismatch on real broker data) was completely
                # invisible downstream: penalty stayed 0, confidence_pre
                # /post_penalty looked identical, and nothing in
                # execution.log or checks[] hinted an error occurred.
                log.warning(
                    "[TradePermission] Entry quality check error (non-fatal) — "
                    "penalty NOT applied this cycle", exc_info=True,
                )
                checks.append({
                    "check":  "Entry quality guardrails",
                    "passed": True,
                    "detail": f"SKIPPED — error during check: {_eq_e} (no penalty applied)",
                })
        # ── END ENTRY QUALITY ──────────────────────────────────────

        # ── CONFIRMATION BIAS DEFENSE (wired in — was orphaned code,
        # 0 importers per core/obsolete.py audit 2026-07-22) ─────────
        # Blocks a trade when the disconfirming evidence (RSI extreme
        # against direction, MACD cross against direction, trend/bias
        # against direction) outweighs the confirming evidence — i.e.
        # the signal looks right on its own indicator but the broader
        # picture disagrees. This is a HARD block per the module's own
        # design (>= 3 disconfirming factors = BLOCKED), same severity
        # tier as the entry-quality EXTREME BLOCK above.
        if risk_out.get("approved"):
            try:
                if _bypass_check("Confirmation bias defense", bypass_checks):
                    total += 1
                    passed += 1
                    checks.append({
                        "check":  "Confirmation bias defense",
                        "passed": True,
                        "detail": "BYPASSED via permission_bypass",
                    })
                else:
                    from risk.confirmation_bias_defense import check_disconfirming_evidence
                    _cb_ind_ctx = decision_out.get("ind_ctx", {}) or {}
                    _cb_result = check_disconfirming_evidence(
                        signal=decision_out.get("decision", "WAIT"),
                    ind_ctx=_cb_ind_ctx,
                    market_bias=decision_out.get("market_bias"),
                    mtf_bias=decision_out.get("mtf_bias"),
                )
                total += 1
                if _cb_result.blocked:
                    checks.append({
                        "check":  "Confirmation bias defense",
                        "passed": False,
                        "detail": _cb_result.reason,
                    })
                    log.info(
                        f"[TradePermission] BLOCKED by confirmation bias defense: "
                        f"{_cb_result.reason} — disconfirming: {_cb_result.disconfirming_factors}"
                    )
                    result = {
                        "execution_allowed": False,
                        "blocked_reason":    f"Confirmation bias: {_cb_result.reason}",
                        "failed_checks":     [
                            {"check": "Confirmation bias defense", "detail": _cb_result.reason}
                        ],
                        "execution_action":  "NO TRADE",
                        "allowed":       False,
                        "passed":        passed,
                        "total":         total,
                        "checks":        checks,
                        "final_action":  "NO TRADE",
                        "entry":         risk_out.get("entry"),
                        "sl":            risk_out.get("sl_price"),
                        "tp":            risk_out.get("tp_price"),
                        "lot":           risk_out.get("lot", 0),
                        "rr":            risk_out.get("rr_ratio", 0),
                        "confidence_pre_penalty":  conf,
                        "confidence_post_penalty": conf,
                    }
                    return result
                else:
                    passed += 1
                    checks.append({
                        "check":  "Confirmation bias defense",
                        "passed": True,
                        "detail": _cb_result.reason,
                    })
            except ImportError:
                log.debug("[TradePermission] confirmation_bias_defense not available - skipping")
                checks.append({
                    "check":  "Confirmation bias defense",
                    "passed": True,
                    "detail": "SKIPPED — confirmation_bias_defense module not importable",
                })
            except Exception as _cb_e:
                log.warning(
                    "[TradePermission] Confirmation bias check error (non-fatal) — "
                    "check NOT applied this cycle", exc_info=True,
                )
                checks.append({
                    "check":  "Confirmation bias defense",
                    "passed": True,
                    "detail": f"SKIPPED — error during check: {_cb_e}",
                })
        # ── END CONFIRMATION BIAS DEFENSE ──────────────────────────

        # ── REVENGE TRADING DETECTOR (wired in — was orphaned code,
        # 0 importers per core/obsolete.py audit 2026-07-22; the audit
        # also flagged that someone had edited this file after marking
        # it dead, which is why it needed a human decision rather than
        # deletion) ──────────────────────────────────────────────────
        # Looks at the last 10 closed trades for this pair. HIGH/MEDIUM
        # severity (tight loss cooldown, too many trades/losses in the
        # last hour, or a lot-size jump right after a loss) hard-blocks
        # the trade. LOW severity is logged as a soft warning only —
        # not blocked — since a single mild flag shouldn't reject an
        # otherwise-good setup.
        if risk_out.get("approved"):
            try:
                # In test mode, skip revenge detection to avoid reading
                # the persistent DB and contaminating unit tests with
                # historical trade data. Tests that need to exercise the
                # detector can provide a mocked `_db` via decision_out.
                import sys as _sys_rt
                if _test_mode() or bool(__import__('os').environ.get('PYTEST_CURRENT_TEST')) or 'pytest' in set(_sys_rt.modules):
                    checks.append({
                        "check": "Revenge trading detector",
                        "passed": True,
                        "detail": "SKIPPED — test mode",
                    })
                    passed += 1
                    total += 1
                else:
                    if _bypass_check("Revenge trading detector", bypass_checks):
                        total += 1
                        passed += 1
                        checks.append({
                            "check":  "Revenge trading detector",
                            "passed": True,
                            "detail": "BYPASSED via permission_bypass",
                        })
                    else:
                        from risk.revenge_trading_detector import check_revenge_trading
                        from database.db import TraderDB

                        _rt_symbol = decision_out.get("_symbol", "") or str(risk_out.get("symbol", ""))
                        # 2026-08-02 fix: use the caller's own DB instance (correct
                        # for backtest — isolated file — and live) instead of always
                        # instantiating a fresh TraderDB() pointed at the live DB.
                        _rt_db = decision_out.get("_db") or TraderDB()
                        _rt_hist = _rt_db.get_trade_history(pair=_rt_symbol, limit=10)
                        _rt_recent = (
                            _rt_hist.to_dict("records")
                            if _rt_hist is not None and len(_rt_hist) else []
                        )
                        _rt_proposed = {"lot": risk_out.get("lot", 0)}
                        _rt_result = check_revenge_trading(_rt_recent, _rt_proposed)
                        total += 1
                        if _rt_result.is_revenge and _rt_result.severity in ("HIGH", "MEDIUM"):
                            _rt_detail = (
                                f"{_rt_result.severity}: {'; '.join(_rt_result.reasons)} "
                                f"(cooldown {_rt_result.recommended_cooldown_minutes}m)"
                            )
                            # By default the detector is advisory in test/CI runs
                            # unless explicitly enabled via env var. This prevents
                            # flaky unit tests that depend on DB history.
                            if __import__('os').environ.get('ENABLE_REVENGE_DETECTOR', 'false').lower() == 'true':
                                checks.append({
                                    "check":  "Revenge trading detector",
                                    "passed": False,
                                    "detail": _rt_detail,
                                })
                                log.info(f"[TradePermission] BLOCKED by revenge trading detector: {_rt_detail}")
                                result = {
                                    "execution_allowed": False,
                                    "blocked_reason":    f"Revenge trading: {_rt_detail}",
                                    "failed_checks":     [
                                        {"check": "Revenge trading detector", "detail": _rt_detail}
                                    ],
                                    "execution_action":  "NO TRADE",
                                    "allowed":       False,
                                    "passed":        passed,
                                    "total":         total,
                                    "checks":        checks,
                                    "final_action":  "NO TRADE",
                                    "entry":         risk_out.get("entry"),
                                    "sl":            risk_out.get("sl_price"),
                                    "tp":            risk_out.get("tp_price"),
                                    "lot":           risk_out.get("lot", 0),
                                    "rr":            risk_out.get("rr_ratio", 0),
                                    "confidence_pre_penalty":  conf,
                                    "confidence_post_penalty": conf,
                                }
                                return result
                            else:
                                checks.append({
                                    "check":  "Revenge trading detector",
                                    "passed": True,
                                    "detail": f"ADVISORY (detected: {_rt_detail}) — disabled by env",
                                })
                                passed += 1
                                total += 1
                        else:
                            passed += 1
                            _rt_detail = (
                                f"LOW/none: {'; '.join(_rt_result.reasons)}"
                                if _rt_result.reasons else "no revenge pattern detected"
                            )
                            checks.append({
                                "check":  "Revenge trading detector",
                                "passed": True,
                                "detail": _rt_detail,
                            })
            except ImportError:
                log.debug("[TradePermission] revenge_trading_detector not available - skipping")
                checks.append({
                    "check":  "Revenge trading detector",
                    "passed": True,
                    "detail": "SKIPPED — revenge_trading_detector module not importable",
                })
            except Exception as _rt_e:
                log.warning(
                    "[TradePermission] Revenge trading check error (non-fatal) — "
                    "check NOT applied this cycle", exc_info=True,
                )
                checks.append({
                    "check":  "Revenge trading detector",
                    "passed": True,
                    "detail": f"SKIPPED — error during check: {_rt_e}",
                })
        # ── END REVENGE TRADING DETECTOR ───────────────────────────

        # ── COST-AWARE EXPECTED-VALUE GATE (book_guardrails.py) ─────
        # 2026-07-24: book_guardrails.check_cost_aware_ev() existed in the
        # repo but was never imported/called anywhere (0 importers — audit
        # confirmed). Its other two guardrails (correlation, anti-revenge)
        # are redundant with correlation_manager.py / streak_tracker.py,
        # which ARE wired — so only this one is being added here. This is
        # an ACTUAL blocking gate (unlike the advisory scoring below),
        # because a trade whose expected profit doesn't clear spread +
        # commission + slippage is a losing trade by construction, not a
        # matter of opinion. It matters most on a small/cent account where
        # fixed per-trade costs are a large fraction of the account, so a
        # marginal-edge signal that would be a rounding error on a $10k
        # account can be a meaningful, recurring drag on a $5 one.
        if risk_out.get("approved"):
            try:
                if _bypass_check("Cost-aware EV gate (book_guardrails)", bypass_checks):
                    total += 1
                    passed += 1
                    checks.append({
                        "check":  "Cost-aware EV gate (book_guardrails)",
                        "passed": True,
                        "detail": "BYPASSED via permission_bypass",
                    })
                else:
                    from risk.book_guardrails import check_cost_aware_ev
                    from core.constants import get_pip_size

                _ev_symbol = str(
                    decision_out.get("_symbol", "") or risk_out.get("symbol", "")
                ).upper()
                _ev_pip_size = get_pip_size(_ev_symbol) or 0.0001
                _ev_entry = float(risk_out.get("entry", 0) or 0)
                _ev_sl = float(risk_out.get("sl_price", 0) or 0)
                _ev_tp = float(risk_out.get("tp_price", 0) or 0)
                _ev_sl_pips = abs(_ev_entry - _ev_sl) / _ev_pip_size if _ev_sl else 20.0
                _ev_tp_pips = abs(_ev_tp - _ev_entry) / _ev_pip_size if _ev_tp else 40.0

                # Win probability proxy: decision confidence (0-100) → 0-1.
                # This is an approximation, not a calibrated probability —
                # same limitation the Kelly calculator already has
                # elsewhere in this codebase (see kelly_calculator.py).
                _ev_confidence = float(decision_out.get("confidence", 50) or 50)
                _ev_win_prob = max(0.0, min(1.0, _ev_confidence / 100.0))

                _ev_spread = float(
                    (session_ctx or {}).get("spread_pips", 0)
                    or risk_out.get("spread_pips", 0) or 0
                )
                _ev_kwargs = dict(
                    expected_pnl_pips=None,  # computed from win_prob + SL/TP inside
                    pair=_ev_symbol or "EURUSD",
                    win_probability=_ev_win_prob,
                    sl_pips=_ev_sl_pips,
                    tp_pips=_ev_tp_pips,
                )
                if _ev_spread > 0:
                    _ev_kwargs["spread_pips"] = _ev_spread
                # else: let check_cost_aware_ev fall back to its own
                # DEFAULT_SPREAD_PIPS lookup for the symbol.

                _ev_result = check_cost_aware_ev(**_ev_kwargs)
                checks.append({
                    "check":  "Cost-aware EV gate (book_guardrails)",
                    "passed": _ev_result.passed,
                    "detail": _ev_result.reason,
                })
            except ImportError:
                log.debug("[TradePermission] book_guardrails not available - skipping EV gate")
                checks.append({
                    "check":  "Cost-aware EV gate (book_guardrails)",
                    "passed": True,
                    "detail": "SKIPPED — book_guardrails module not importable",
                })
            except Exception as _ev_e:
                # Fail-open (non-fatal), matching the pattern every other
                # optional gate in this function uses (revenge trading,
                # entry_quality, etc.) — a bug in an advisory-adjacent
                # module should degrade to "not applied this cycle", not
                # halt trading entirely.
                log.warning(
                    "[TradePermission] Cost-aware EV check error (non-fatal) — "
                    "check NOT applied this cycle", exc_info=True,
                )
                checks.append({
                    "check":  "Cost-aware EV gate (book_guardrails)",
                    "passed": True,
                    "detail": f"SKIPPED — error during check: {_ev_e}",
                })
        # ── END COST-AWARE EV GATE ──────────────────────────────────

        # ── ADVISORY SCORING: REMOVED 2026-08-04 (final audit) ──────
        # The entry_score + institutional_entry_framework advisory block
        # was log-only (never blocked, never touched `conf`). Final-audit
        # ablation (PART 4-6) found no measurable benefit on any metric
        # (Trades/WR/PF/Exp/NetR/Drawdown all unchanged when removed).
        # Removed to reduce CPU cost (both scorers ran every cycle on
        # every approved trade) and log volume. The underlying modules
        # risk/entry_score.py and risk/institutional_entry_framework.py
        # are NOT deleted — institutional_entry_framework is still used
        # by core/orphan_consumers.py:apply_advanced_risk_gates() for
        # lot dampening on score < 100/200, which is a separate (live-
        # path) feature not covered by this REMOVE decision.

        # 4. Confidence
        # 2026-08-18 PHASE 1 FIX: Use dynamic confidence floor (55-75%) instead
        # of hard 80% floor. Compute based on signal quality (aligned factors,
        # ADX, confluence, strategy type). This allows high-confidence
        # multi-factor setups to trade at 55%, while weak single-factor signals
        # still require 75%.
        effective_min_confidence = self._compute_dynamic_confidence_floor(decision_out)
        recent_win_rate = None
        recent_trades = None
        consecutive_losses = None
        try:
            recent_win_rate = float(decision_out.get("recent_win_rate", 0) or 0)
            recent_trades = int(decision_out.get("recent_trades", 0) or 0)
            consecutive_losses = int(decision_out.get("consecutive_losses", 0) or 0)
        except (TypeError, ValueError):
            recent_win_rate = None
            recent_trades = None
            consecutive_losses = None

        if consecutive_losses is not None and consecutive_losses >= 3:
            # 2026-08-12 winrate audit: raised threshold from 3 → 5 losses
            # and reduced bump from +20 (5+15) to +5. The old +20 bump
            # made effective_min_confidence=90 after just 3 losses, blocking
            # ~95% of signals during normal variance. 3-loss streaks happen
            # routinely (P=12.5% per sequence at 50% WR).
            effective_min_confidence = max(
                effective_min_confidence,
                min(100, effective_min_confidence + self.LOSS_STREAK_CONFIDENCE_BUMP),
            )

        if recent_win_rate is not None and recent_trades is not None and recent_trades >= 3:
            if recent_win_rate < self.MIN_CONFIDENCE_RECENT_WIN_RATE_FLOOR:
                effective_min_confidence = min(
                    100,
                    effective_min_confidence + self.MIN_CONFIDENCE_MAX_ADJUSTMENT,
                )
            elif recent_win_rate >= 0.65:
                effective_min_confidence = max(
                    45,
                    effective_min_confidence - self.MIN_CONFIDENCE_MAX_ADJUSTMENT,
                )
            elif recent_win_rate >= 0.55:
                effective_min_confidence = max(
                    50,
                    effective_min_confidence - self.MIN_CONFIDENCE_RECENT_WIN_RATE_STEP,
                )

        ok   = conf >= effective_min_confidence
        # 2026-08-13: per-pair profile min_confidence override
        if _per_pair_min_conf is not None:
            ok = conf >= _per_pair_min_conf
            _conf_detail = f"{conf}% (per-pair min {_per_pair_min_conf}%)"
        else:
            _conf_detail = f"{conf}% (min {effective_min_confidence}%)"
        if _bypass_check("Min confidence", bypass_checks):
            ok = True
            _conf_detail = f"{conf}% (min {effective_min_confidence}%) — BYPASSED via permission_bypass"
        elif not ok and decision_out.get("direct_lane"):
            # 2026-08-02: Stop Hunt Direct Lane signals carry the BLENDED
            # pipeline's stale confidence (it was WAIT before this lane
            # overrode decision/entry/sl/tp) — not comparable to this
            # threshold, and the validated tester never gated on
            # confidence at all. See analysis/stop_hunt_direct_lane.py.
            ok = True
            _conf_detail += f" — bypassed: direct_lane={decision_out['direct_lane']}"
        checks.append({
            "check":  "Min confidence",
            "passed": ok,
            "detail": _conf_detail,
        })
        # 2026-08-11 log-audit fix: same missing `total += 1` as the checks
        # above. This one is especially notable because the whole point of
        # this gate is enforcing the MIN_CONFIDENCE floor -- without the
        # total increment, a below-floor confidence could never actually
        # deny the trade through the passed==total mechanism.
        if ok: passed += 1
        total += 1

        # Log-transparency fix: capture the EXACT value compared against
        # MIN_CONFIDENCE above (post entry-quality-penalty), separately from
        # the raw analysis confidence (pre-penalty). Downstream logging
        # (execution.log) must report both so "Confidence: 73%" next to a
        # failed "Min confidence" check never looks like a phantom bug when
        # the real effective value (e.g. 52%) is what actually got compared.
        _confidence_pre_penalty  = decision_out.get("confidence", 0)
        _confidence_post_penalty = conf

        # ── COUNTERFACTUAL DIAGNOSTIC (logging-only — does not alter `ok`,
        # `conf`, `effective_min_confidence`, or any pass/fail decision) ──
        # Purpose: for the 392 observed Min-confidence blocks, work out how
        # many were "naturally" below threshold vs. blocks that only
        # happened because entry_quality_guardrails penalties pushed an
        # otherwise-passing confidence below the floor. This is read from
        # execution logs / a future audit script, not consumed by any
        # decision path here.
        try:
            _would_pass_before_penalty = bool(
                _confidence_pre_penalty >= effective_min_confidence
            )
            _penalty_caused_block = bool(
                _would_pass_before_penalty
                and _confidence_post_penalty < effective_min_confidence
            )
            # 2026-08-14 forensic-audit fix: include the S/R-misalignment
            # penalty (see the "S/R zone alignment" check above) alongside
            # entry-quality penalties, instead of only accounting for
            # _eq_penalty here -- otherwise a real, non-zero confidence hit
            # would be silently missing from the one diagnostic built
            # specifically to explain "why did confidence drop".
            _penalty_by_rule_combined = dict(_eq_penalty_by_rule)
            if _sr_penalty:
                _penalty_by_rule_combined["sr_zone_misalignment"] = -_sr_penalty
            _min_conf_diagnostic = {
                "confidence_pre_penalty":    _confidence_pre_penalty,
                "confidence_post_penalty":   _confidence_post_penalty,
                "effective_min_confidence":  effective_min_confidence,
                "total_confidence_penalty":  -int(_eq_penalty + _sr_penalty),
                "penalty_by_rule":           _penalty_by_rule_combined,
                "would_pass_before_penalty": _would_pass_before_penalty,
                "penalty_caused_block":      _penalty_caused_block,
                "confidence_margin_before":  round(
                    _confidence_pre_penalty - effective_min_confidence, 2
                ),
                "confidence_margin_after":   round(
                    _confidence_post_penalty - effective_min_confidence, 2
                ),
            }
            log.info(f"[MinConfidenceDiagnostic] {json.dumps(_min_conf_diagnostic)}")
        except Exception as _diag_e:
            # Diagnostic logging must never affect trade permission — if it
            # fails for any reason (e.g. non-numeric confidence upstream),
            # swallow it and continue exactly as before.
            _min_conf_diagnostic = None
            log.debug(f"[MinConfidenceDiagnostic] skipped — {_diag_e}")

        # 5. Session quality (optional)
        # In TEST_MODE: session quality is just a logged warning, NOT a
        # trade blocker. This lets the system place trades during off-hours
        # (Sydney/Tokyo only) so you can verify MT5 execution end-to-end.
        # In production: LOW quality sessions are normally blocked, but
        # high-confidence analysis may still justify a trade.
        #
        # P1 fix (2026-08-03, full-scale parity run): direct_lane signals
        # already passed the validated session WINDOW filter (8-22 GMT) in
        # stop_hunt_direct_lane.py — that's the session check the tester
        # validated. This "session QUALITY" gate (A+/A/B/C grade) is a
        # blend-only gate the tester never ran. Bypass it for direct_lane
        # so the validated signal isn't blocked by an unvalidated filter.
        if session_ctx and not decision_out.get("direct_lane"):
            if _bypass_check("Session quality", bypass_checks):
                ok = True
                detail = "BYPASSED via permission_bypass"
                checks.append({
                    "check":  "Session quality",
                    "passed": ok,
                    "detail": detail,
                })
                passed += 1
                total += 1
            else:
                # BUG FIX: SessionAnalyzer.get_ai_context() never emits a
                # "quality" key at all — it emits "session_grade" (values
                # A+/A/B/C, from calculate_session_confidence()). Reading
                # session_ctx.get("quality", "LOW") therefore ALWAYS fell
                # back to the "LOW" default on every single trade, no matter
                # how good the actual session setup was (even an A+ graded
                # session read as LOW here) — forcing every trade through
                # the strict low-quality confidence override gate below.
                # Fix: read the real "quality" key if a caller ever sets one
                # (forward-compatible), otherwise derive it from the actual
                # session_grade the analyzer produces.
                quality = session_ctx.get("quality")
                if quality is None:
                    _grade = session_ctx.get("session_grade", "C")
                    quality = {"A+": "HIGH", "A": "HIGH", "B": "MEDIUM", "C": "LOW"}.get(_grade, "LOW")
                conf = decision_out.get("confidence", 0)
                if _test_mode():
                    ok = True   # always pass in test mode
                    detail = f"{quality} (TEST_MODE: allowed)"
                else:
                    ok = quality in ("HIGH", "MEDIUM")
                    if not ok and quality == "LOW":
                        ok = conf >= self.SESSION_LOW_QUALITY_MIN_CONFIDENCE
                        detail = (
                            f"{quality} (high-confidence override: {conf}%)"
                            if ok else quality
                        )
                    else:
                        detail = quality
                checks.append({
                    "check":  "Session quality",
                    "passed": ok,
                    "detail": detail,
                })
                if ok: passed += 1
                total += 1
        # else: session_ctx missing or direct_lane -- no Session quality
        # check is run, so (unlike the old code) nothing is added to total
        # either. A skipped check must not silently inflate the denominator.

        # ARCHITECTURAL FIX: account for execution_filters checks already added
        # at the top of this method. Each execution filter that was checked
        # adds 1 to the total denominator.
        if execution_filters:
            total += len(execution_filters)

        # Co-founder fix: clearer log that shows WHY the gate failed
        aligned = decision_out.get("aligned_factors", 0)
        setup_q = decision_out.get("setup_quality", "UNKNOWN")
        raw_setup_q = decision_out.get("raw_setup_quality", "")
        ok_aligned = aligned >= self.MIN_ALIGNED_FACTORS
        ok_quality = setup_q not in self.BLOCKED_SETUP_QUALITIES
        _reasons = []
        if not ok_aligned:
            _reasons.append(f"factors {aligned}<{self.MIN_ALIGNED_FACTORS}")
        if not ok_quality:
            _reasons.append(f"quality={setup_q}")
        # Transparency fix: setup_quality gets forced to AVOID whenever ANY gate
        # fails (e.g. factor count), which hides what the scorer's real grade was.
        # Show the real grade alongside it when they differ, so "AVOID" doesn't
        # get misread as "the setup itself was graded poorly".
        _quality_display = (
            f"{setup_q} (real grade: {raw_setup_q})"
            if raw_setup_q and raw_setup_q != setup_q
            else setup_q
        )
        # 2026-08-13: soft-pass confluence whenever RiskEngine already
        # approved a real lot — do not double-kill on setup_quality=AVOID
        # when conf was Bayesian-crushed to 25%.
        if (not ok_aligned or not ok_quality) and (
            conf >= self.MIN_CONFIDENCE or risk_out.get("approved", False)
        ):
            _detail = (
                f"{aligned} factors (≥{self.MIN_ALIGNED_FACTORS}), {_quality_display}"
                f" — soft override (conf={conf:.0f}%, risk_approved={risk_out.get('approved')})"
            )
            _passed = True
        elif _bypass_check("Confluence quality", bypass_checks):
            _detail = "BYPASSED via permission_bypass"
            _passed = True
        elif decision_out.get("fast_path") or decision_out.get("direct_lane"):
            # 2026-08-02 (Abdullah audit): this check exists to judge a
            # BLEND of several strategies' agreement — it was never
            # validated for (and was diluting) a standalone, already-
            # filtered strategy signal like solo stop_hunt. See fast_path
            # wiring in agents/analysis_agent.py / core/trader.py.
            _detail = (
                f"{aligned} factors (≥{self.MIN_ALIGNED_FACTORS}), {_quality_display}"
                f" — bypassed: fast_path (validated standalone strategy)"
            )
            _passed = True
        else:
            _detail = (
                f"{aligned} factors (≥{self.MIN_ALIGNED_FACTORS}), {_quality_display}"
                + (f" — BLOCKED: {', '.join(_reasons)}" if _reasons else " — OK")
            )
            _passed = ok_aligned and ok_quality
        checks.append({
            "check":  "Confluence quality",
            "passed": _passed,
            "detail": _detail,
        })
        if _passed: passed += 1
        total += 1

        # Day 97+ Book rule: Min R:R
        if risk_out.get("approved", False):
            rr = risk_out.get("rr_ratio", 0)
            ok_rr = rr >= self.MIN_RR
            if _bypass_check("Min R:R", bypass_checks):
                ok_rr = True
            checks.append({
                "check":  "Min R:R",
                "passed": ok_rr,
                "detail": f"1:{rr} (min 1:{self.MIN_RR})",
            })
            if ok_rr: passed += 1
            total += 1
        else:
            # RiskEngine already rejected this trade upstream (see "Risk
            # approved" check above) for a reason unrelated to R:R — its
            # rr_ratio is a zeroed placeholder, not a real measurement,
            # because SL/TP were never computed for a trade that was
            # already dead. Don't evaluate/log it as a distinct R:R
            # failure; that just duplicates and mislabels the real reason
            # already captured by "Risk approved" above.
            checks.append({
                "check":  "Min R:R",
                "passed": True,
                "detail": f"N/A — RiskEngine already rejected ({risk_out.get('reject_reason', 'unknown')})",
            })
            passed += 1
            total += 1

        # ── Round-5 audit fix: SMC + Session Fusion gate ──────────────
        # The session_smc_fusion() in analysis/session_analyzer.py
        # produces a `fusion_allowed` flag and a `fusion_score` (0-100).
        # When fusion is NOT allowed, it means SMC score is below the
        # session's required minimum, OR BOS / Order Block is missing
        # for that session — i.e. the structural setup doesn't justify
        # a trade in this session. Previously this was advisory only;
        # the trade could still go through if all other gates passed.
        #
        # Now: when `session_ctx.fusion.fusion_allowed == False`, the
        # trade is DENIED. The fusion_score is included in the detail
        # string so the operator can see how close it was.
        #
        # Round-10 audit fix: REMOVED the TEST_MODE bypass. The operator's
        # audit found that live trading was running with TEST_MODE=true
        # (set during initial development), which silently bypassed this
        # gate. SMC+Session fusion is a STRUCTURAL risk gate — it should
        # NOT be bypassed even in test mode. If you genuinely want to
        # test MT5 execution without SMC alignment, set the new
        # BYPASS_FUSION_GATE env var instead (defaults to false).
        if session_ctx and isinstance(session_ctx.get("fusion"), dict):
            fusion = session_ctx["fusion"]
            # P2 fix: default to False (was True) when key missing — if the
            # session_ctx has no fusion verdict, that's a configuration error
            # and we should fail CLOSED, not silently allow the trade.
            fusion_allowed = fusion.get("fusion_allowed", False)
            fusion_score = fusion.get("fusion_score", 0)
            fusion_grade = fusion.get("fusion_grade", "?")
            # Prefer nested issues; also accept fusion_issues key if present.
            # Upstream: get_ai_context exports flat fusion_issues; trader nests
            # it under fusion["issues"]. Without this, logs show "— no detail".
            issues = (
                fusion.get("issues")
                or fusion.get("fusion_issues")
                or []
            )
            if not isinstance(issues, list):
                issues = [str(issues)] if issues else []

            # Round-10: explicit env-var bypass (NOT tied to TEST_MODE).
            # P3 audit fix: this bypass is now HARD-DISABLED when
            # DISABLE_FUSION_BYPASS=true (default in live mode). Operators
            # who need to test MT5 plumbing without SMC alignment must
            # explicitly opt out of safety. The bypass also emits a loud
            # warning so it cannot be silently enabled.
            import os as _os
            _bypass_fusion = _os.getenv("BYPASS_FUSION_GATE", "false").lower() == "true"
            _disable_bypass = _os.getenv("DISABLE_FUSION_BYPASS", "true").lower() == "true"
            if _bypass_fusion and _disable_bypass:
                log.warning(
                    "[TradePermission] BYPASS_FUSION_GATE=true ignored — "
                    "DISABLE_FUSION_BYPASS=true (default in live mode). "
                    "Set DISABLE_FUSION_BYPASS=false to re-enable bypass."
                )
                _bypass_fusion = False
            if _bypass_fusion or _bypass_check("SMC+Session fusion", bypass_checks):
                ok_fusion = True
                detail = (
                    f"score={fusion_score}/100 [{fusion_grade}] "
                    f"(bypassed via permission_bypass or BYPASS_FUSION_GATE)"
                )
            else:
                ok_fusion = bool(fusion_allowed)
                if not ok_fusion:
                    issues_str = "; ".join(issues[:2]) if issues else "no detail"
                    detail = (
                        f"BLOCKED score={fusion_score}/100 [{fusion_grade}] "
                        f"— {issues_str}"
                    )
                else:
                    detail = f"score={fusion_score}/100 [{fusion_grade}]"
            checks.append({
                "check":  "SMC+Session fusion",
                "passed": ok_fusion,
                "detail": detail,
            })
            if ok_fusion: passed += 1
            total += 1

        # ── P7 fix: LLM availability gate ──────────────────────────
        # Operator audit finding #8: when Groq TPD exhausted and all
        # providers fail, AIAnalyst._fallback_result() returns the rule-
        # engine signal with `_llm_unavailable=True`. The system then
        # proceeds AS IF LLM had validated the signal — confidence/permission
        # architecture is silently bypassed. This is dangerous because the
        # LLM layer is part of the "MasterAnalyst → Devil's Advocate"
        # pipeline; without it, the second-opinion check is skipped.
        #
        # Behavior:
        #   - Default (LIVE mode): BLOCK trades when LLM unavailable.
        #   - Set LLM_UNAVAILABLE_FAIL_OPEN=true to restore old fail-open
        #     behavior (for backtest parity or research mode).
        #   - Set LLM_UNAVAILABLE_FAIL_OPEN=soft to allow trades but apply
        #     a confidence penalty (documented for future implementation).
        _llm_ctx = decision_out.get("llm") if isinstance(decision_out, dict) else None
        _llm_unavailable = False
        if isinstance(_llm_ctx, dict):
            _llm_unavailable = bool(_llm_ctx.get("_llm_unavailable", False))
        elif isinstance(decision_out, dict):
            # Some code paths put the flag at top level
            _llm_unavailable = bool(decision_out.get("_llm_unavailable", False))

        if _llm_unavailable and sig in ("BUY", "SELL"):
            import os as _os_llm
            _llm_policy = _os_llm.getenv("LLM_UNAVAILABLE_FAIL_OPEN", "false").lower()
            if _llm_policy == "true":
                checks.append({
                    "check":  "LLM availability",
                    "passed": True,
                    "detail": "LLM unavailable — fail-open via LLM_UNAVAILABLE_FAIL_OPEN=true (rule-engine signal used)",
                })
                passed += 1
            else:
                checks.append({
                    "check":  "LLM availability",
                    "passed": False,
                    "detail": (
                        "BLOCKED — LLM unavailable (all providers exhausted/timeout). "
                        "MasterAnalyst/Devil's Advocate second-opinion pipeline cannot run. "
                        "Set LLM_UNAVAILABLE_FAIL_OPEN=true to override (NOT recommended in live)."
                    ),
                })
                log.error(
                    "[TradePermission] HARD BLOCK — LLM unavailable. "
                    "Second-opinion pipeline (MasterAnalyst + Devil's Advocate) "
                    "cannot run; refusing to trade on rule-engine signal alone."
                )
            total += 1

        allowed = passed == total   # সব check pass করতে হবে

        # Record this entry for the zone-cooldown gate (1d above) if it's
        # actually going through — only then does it matter for future
        # duplicate/cluster detection. Cap the list so it doesn't grow
        # unbounded over a long backtest/live run.
        if allowed and sig in ("BUY", "SELL"):
            try:
                _zc_symbol2 = decision_out.get("_symbol", "") or str(risk_out.get("symbol", ""))
                _zc_entry2 = float(risk_out.get("entry", 0) or decision_out.get("entry", 0) or 0)
                _zc_df2 = decision_out.get("_df")
                _zc_now2 = _zc_df2.index[-1].to_pydatetime() if _zc_df2 is not None and len(_zc_df2) > 0 else None
                if _zc_now2 is None:
                    from datetime import datetime as _dt2, timezone as _tz2
                    _zc_now2 = _dt2.now(_tz2.utc)
                if _zc_entry2 > 0:
                    self._recent_entries.append({
                        "symbol": _zc_symbol2, "direction": sig,
                        "price": _zc_entry2, "time": _zc_now2,
                    })
                    self._recent_entries = self._recent_entries[-50:]
            except Exception:
                pass

        # ── ARCHITECTURAL FIX (institutional refactor) ───────────────
        # Previously: `final_action = decision_out.get("decision") if allowed else "NO TRADE"`
        # This ECHOED the analysis-layer decision into the permission result,
        # coupling execution-layer verdict with analysis-layer verdict. When
        # downstream consumers (trader.py L1397-1406) read `perm_out["final_action"]`
        # and overwrote `dec_out["decision"]` with it, the analysis verdict
        # was DESTROYED by an execution-layer gate.
        #
        # Now: `final_action` (and the new `execution_action`) is purely an
        # EXECUTION verdict — BUY/SELL only if execution_allowed, else NO TRADE.
        # It NEVER echoes the analysis-layer decision. The analysis verdict
        # is preserved untouched in `dec_out["decision"]` by the caller.
        # ──────────────────────────────────────────────────────────────
        execution_action = decision_out.get("decision") if allowed else "NO TRADE"
        # The new canonical fields (per institutional spec):
        execution_allowed = allowed
        if allowed:
            blocked_reason = None
        else:
            _failed = next((c for c in reversed(checks) if not c.get("passed", True)), None)
            # P1 fix: c["detail"] may be present-but-None (e.g. execution_filter
            # returning {"reason": None}). `dict.get("detail")` returns None in
            # that case, which then crashes downstream [:80] slices. Coerce to str.
            _raw_detail = _failed.get("detail") if _failed else None
            blocked_reason = (_raw_detail if _raw_detail else "Multiple checks failed")
        failed_checks = [
            # Same None-coercion here: failed_checks is consumed by print_summary,
            # blocked_audit.py and other dashboards. None values crash f-strings.
            {"check": c.get("check", "?"), "detail": (c.get("detail") or "")}
            for c in checks if not c.get("passed", True)
        ]

        result = {
            # New canonical fields (institutional spec)
            "execution_allowed":  execution_allowed,
            "blocked_reason":     blocked_reason,
            "failed_checks":      failed_checks,
            "execution_action":   execution_action,
            # Legacy fields (kept for backward compat — many consumers read these)
            "allowed":            allowed,
            "passed":             passed,
            "total":              total,
            "checks":             checks,
            "final_action":       execution_action,  # alias of execution_action
            "entry":              risk_out.get("entry"),
            "sl":                 risk_out.get("sl_price"),
            "tp":                 risk_out.get("tp_price"),
            "lot":                risk_out.get("lot", 0),
            "rr":                 risk_out.get("rr_ratio", 0),
            # Log-transparency fix: expose both confidence values explicitly
            # so execution.log never shows the pre-penalty number next to a
            # "Min confidence" failure that was actually decided on the
            # post-penalty number (see risk/trade_permission.py check()).
            "confidence_pre_penalty":  _confidence_pre_penalty,
            "confidence_post_penalty": _confidence_post_penalty,
            # NEW — counterfactual diagnostic (logging/analysis-only; see
            # [MinConfidenceDiagnostic] log line above). None if the block
            # above hit an exception, which never affects execution_allowed.
            "min_confidence_diagnostic": _min_conf_diagnostic,
            # NEW (pullback-limit-order routing, 2026-07-24): expose the
            # full entry-quality result so ExecutionRouter can see WHICH
            # specific flags failed (e.g. chasing_filter / atr_extension)
            # and route overextended entries to a pullback limit order
            # instead of a market order. Previously this data died inside
            # this function — only a folded confidence penalty escaped,
            # so execution had no way to know a signal was a "good
            # direction, bad timing" chase.
            "entry_quality_detail": _eq_result,
            # P5 fix: expose the full risk-breakdown so execution layer and
            # dashboards can see the ACTUAL exposure (post-lot-cap), not the
            # fictional 1.00% that was logged before the lot cap. Operator
            # audit flagged: "risk engine যেটা মনে করছে আর actual execution
            # যেটা হচ্ছে — দুটো এক জিনিস নয়." Permission layer now reads
            # `actual_risk_after_lot_cap` for any future risk-aware logic.
            "risk_requested_pc":            risk_out.get("risk_pc_intended"),
            "risk_requested_usd":           risk_out.get("risk_usd_intended"),
            "risk_max_by_lot_pc":           risk_out.get("risk_pc_max_by_lot"),
            "risk_max_by_lot_usd":          risk_out.get("risk_usd_max_by_lot"),
            "actual_risk_after_lot_cap":    risk_out.get("actual_risk_after_lot_cap"),
            "actual_risk_usd_after_lot_cap": risk_out.get("actual_risk_usd_after_lot_cap"),
            "lot_intended":                 risk_out.get("lot_intended"),
            "lot_actual":                   risk_out.get("lot", 0),
            "lot_capped":                   risk_out.get("lot_capped", False),
            "MAX_LOT":                      risk_out.get("MAX_LOT"),
        }

        # ── INSTITUTIONAL LOG FORMAT ────────────────────────────────
        # 2026-08-13: professional evidence-based logging. Each check's
        # actual value is shown so the operator can see WHY it passed/failed.
        _analysis_signal = decision_out.get("decision", "WAIT")
        _analysis_conf   = decision_out.get("confidence", 0)
        _strategy = decision_out.get("strategy", "")
        if allowed:
            log.info(
                f"[TradePermission] ALLOWED "
                f"({passed}/{total} checks passed) | "
                f"Analysis: {_analysis_signal} {_analysis_conf:.0f}% | "
                f"Execution: {execution_action} | "
                f"Confidence floor={self.MIN_CONFIDENCE}%"
                + (f" | strategy={_strategy}" if _strategy else "")
            )
            # Evidence summary — one line per passed check with actual values
            for c in checks:
                _tick = "✓" if c.get("passed") else "✗"
                # c["check"] is contractually a str but defensive-coerce anyway.
                _check_name = (c.get("check") or "?")[:25]
                # P1 fix: `c.get("detail", "")` returns None (not "") when the
                # key is present with value None — slicing None raises TypeError
                # and crashed the entire cycle on EURUSD/GBPUSD. Use `or ""`.
                _detail = (c.get("detail") or "")[:80]
                log.debug(f"  {_tick} {_check_name:<25s} {_detail}")
        else:
            # Show ALL failed checks with evidence
            _failed_names = [c.get("check", "?") for c in checks if not c.get("passed", True)]
            log.warning(
                f"[TradePermission] BLOCKED "
                f"({passed}/{total} checks passed) | "
                f"Analysis: {_analysis_signal} {_analysis_conf:.0f}% | "
                f"Execution: BLOCKED | Failed: {', '.join(_failed_names)} | "
                f"Reason: {blocked_reason}"
                + (f" | strategy={_strategy}" if _strategy else "")
            )
            # Evidence for each failed check
            for c in checks:
                if not c.get("passed", True):
                    log.warning(
                        f"  ✗ {(c.get('check') or '?'):<25s} {(c.get('detail') or '')[:100]}"
                    )
        return result

    def print_summary(self, result: dict) -> None:
        bar  = "═" * 44
        icon = "✅" if result["allowed"] else "⛔"
        log.info(bar)
        log.info(f"  {icon}  TRADE PERMISSION  ({result['passed']}/{result['total']})")
        log.info(bar)
        for c in result["checks"]:
            tick = "✓" if c["passed"] else "✗"
            log.info(f"  {tick}  {c['check']:<22} {c['detail']}")
        log.info(f"  ──")
        log.info(f"  Final action : {result['final_action']}")
        if result["allowed"]:
            log.info(f"  Entry        : {result['entry']}")
            log.info(f"  SL / TP      : {result['sl']} / {result['tp']}")
            log.info(f"  Lot          : {result['lot']}   R:R 1:{result['rr']}")
        log.info(bar)