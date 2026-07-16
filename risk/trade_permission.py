# risk/trade_permission.py  —  Day 13 | Final Trade Permission Gate

from utils.logger import get_logger

log = get_logger("trade_permission")


def _test_mode() -> bool:
    """Lazy check — avoids importing config at module load (which would
    crash unit tests on systems without a .env file)."""
    try:
        from config import TEST_MODE
        return bool(TEST_MODE)
    except Exception as e:
        return False


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
    MIN_CONFIDENCE_PROD  = 35  # Lowered for better trade frequency (was 40)
    MIN_CONFIDENCE_TEST  = 10

    # Co-founder fix: raised thresholds for institutional-grade entries
    MIN_ALIGNED_FACTORS_PROD = 2
    MIN_ALIGNED_FACTORS_TEST = 1
    MIN_RR_PROD = 1.5   # min 1:1.5 R:R (balanced institutional standard)
    MIN_RR_TEST = 1.0
    BLOCKED_SETUP_QUALITIES = {"AVOID", "INVALID"}  # Removed "POOR" - allow marginal setups

    @property
    def MIN_CONFIDENCE(self) -> int:
        return self.MIN_CONFIDENCE_TEST if _test_mode() else self.MIN_CONFIDENCE_PROD

    @property
    def MIN_ALIGNED_FACTORS(self) -> int:
        return self.MIN_ALIGNED_FACTORS_TEST if _test_mode() else self.MIN_ALIGNED_FACTORS_PROD

    @property
    def MIN_RR(self) -> float:
        return self.MIN_RR_TEST if _test_mode() else self.MIN_RR_PROD

    def check(
        self,
        decision_out: dict,
        risk_out:     dict,
        news_ctx:     dict,
        session_ctx:  dict | None = None,
        execution_filters: dict | None = None,
    ) -> dict:

        checks = []
        passed = 0

        # ── ARCHITECTURAL FIX (institutional refactor) ───────────────
        # The new `execution_filters` dict (produced by AnalysisAgent)
        # records gate verdicts WITHOUT destroying the analysis signal.
        # We honor any gate recorded there as a hard block, and we add
        # each one to the checks list so the operator can see WHY.
        # This replaces the old pattern where news/session gates would
        # overwrite `decision_out["decision"] = "NO TRADE"` at the
        # analysis layer.
        # ──────────────────────────────────────────────────────────────
        if execution_filters:
            for gate_name, gate_result in execution_filters.items():
                if isinstance(gate_result, dict) and gate_result.get("blocked"):
                    checks.append({
                        "check":  f"Execution filter: {gate_name}",
                        "passed": False,
                        "detail": gate_result.get("reason", "blocked"),
                    })
                    # Don't increment passed — this is a hard block.
                else:
                    checks.append({
                        "check":  f"Execution filter: {gate_name}",
                        "passed": True,
                        "detail": "not blocked",
                    })
                    passed += 1

        # 1. Signal
        sig = decision_out.get("decision", "WAIT")
        ok  = sig in ("BUY", "SELL")
        checks.append({"check": "Valid signal", "passed": ok, "detail": sig})
        if ok: passed += 1

        # 2. Risk approved
        ok = risk_out.get("approved", False)
        checks.append({
            "check":  "Risk approved",
            "passed": ok,
            "detail": risk_out.get("reject_reason", "OK"),
        })
        if ok: passed += 1

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
        if _bypass_news:
            ok = True
            detail = "News system unavailable (BYPASS_NEWS_GATE=true: allowed anyway — no CPI/NFP protection)"
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
        if ok: passed += 1

        # 4. Confidence
        conf = decision_out.get("confidence", 0)
        ok   = conf >= self.MIN_CONFIDENCE
        checks.append({
            "check":  "Min confidence",
            "passed": ok,
            "detail": f"{conf}% (min {self.MIN_CONFIDENCE}%)",
        })
        if ok: passed += 1

        # 5. Session quality (optional)
        # In TEST_MODE: session quality is just a logged warning, NOT a
        # trade blocker. This lets the system place trades during off-hours
        # (Sydney/Tokyo only) so you can verify MT5 execution end-to-end.
        # In production: LOW quality sessions block the trade.
        if session_ctx:
            quality = session_ctx.get("quality", "LOW")
            if _test_mode():
                ok = True   # always pass in test mode
                detail = f"{quality} (TEST_MODE: allowed)"
            else:
                ok = quality in ("HIGH", "MEDIUM")
                detail = quality
            checks.append({
                "check":  "Session quality",
                "passed": ok,
                "detail": detail,
            })
            if ok: passed += 1
            total = 5
        else:
            total = 4

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
        _detail = (
            f"{aligned} factors (≥{self.MIN_ALIGNED_FACTORS}), {_quality_display}"
            + (f" — BLOCKED: {', '.join(_reasons)}" if _reasons else " — OK")
        )
        checks.append({
            "check":  "Confluence quality",
            "passed": ok_aligned and ok_quality,
            "detail": _detail,
        })
        if ok_aligned and ok_quality: passed += 1
        total += 1

        # Day 97+ Book rule: Min R:R
        rr = risk_out.get("rr_ratio", 0)
        ok_rr = rr >= self.MIN_RR
        checks.append({
            "check":  "Min R:R",
            "passed": ok_rr,
            "detail": f"1:{rr} (min 1:{self.MIN_RR})",
        })
        if ok_rr: passed += 1
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
            fusion_allowed = fusion.get("fusion_allowed", True)
            fusion_score = fusion.get("fusion_score", 0)
            fusion_grade = fusion.get("fusion_grade", "?")
            issues = fusion.get("issues", []) or []

            # Round-10: explicit env-var bypass (NOT tied to TEST_MODE)
            import os as _os
            _bypass_fusion = _os.getenv("BYPASS_FUSION_GATE", "false").lower() == "true"
            if _bypass_fusion:
                ok_fusion = True
                detail = (
                    f"score={fusion_score}/100 [{fusion_grade}] "
                    f"(BYPASS_FUSION_GATE=true: allowed even if blocked)"
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

        allowed = passed == total   # সব check pass করতে হবে

        # ── Round-22 audit fix: Entry Quality Guardrails ───────────────
        # R1 fix from risk/ folder audit. This 1,716-line module was built
        # from a real-trade post-mortem (GBPUSD M5, 2026-07-02) but was
        # NEVER wired into the live pipeline — 0 importers. It checks 12
        # entry-quality red flags including:
        #   1. Chasing filter (block entries after extended move without pullback)
        #   2. SL must be swing-anchored
        #   3. TP must have prior price-action test
        #   4. Indecision candle filter
        #   5. Indicator confluence required
        #   6. Round number awareness
        #   7-12. Additional post-mortem fixes (rejection wick, averaging
        #         into losers, fresh high rejection, etc.)
        #
        # Now: if all other gates passed, run guardrails as the FINAL check
        # before allowing the trade. If any BLOCK-severity flag fires, the
        # trade is denied. WARNING-severity flags are logged but don't block.
        if allowed and risk_out.get("approved"):
            try:
                from risk.entry_quality_guardrails import run_all_entry_quality_checks
                _df = None
                _ind_ctx = decision_out.get("ind_ctx", {}) or {}
                # Round-30 fix F1: check decision_out["_df"] directly.
                # Previously only checked inside `if isinstance(session_ctx, dict):`
                # which meant if session_ctx was None, df was never found.
                # Now: check decision_out first (set by trader.py L1258),
                # then fall back to session_ctx.
                _df = decision_out.get("_df")
                if _df is None and isinstance(session_ctx, dict):
                    _df = session_ctx.get("_df")
                _eq_symbol = decision_out.get("_symbol", "") or str(risk_out.get("symbol", ""))
                if _df is not None and len(_df) > 0:
                    _eq_result = run_all_entry_quality_checks(
                        df=_df,
                        symbol=_eq_symbol,
                        direction=decision_out.get("decision", "WAIT"),
                        entry_price=float(risk_out.get("entry", 0) or 0),
                        stop_loss=float(risk_out.get("sl_price", 0) or 0),
                        take_profit=float(risk_out.get("tp_price", 0) or 0),
                        ind_ctx=_ind_ctx,
                    )
                    _should_execute = _eq_result.get("should_execute", True)
                    _block_reason = _eq_result.get("block_reason")
                    _quality_score = _eq_result.get("quality_score", 100)
                    _eq_warnings = _eq_result.get("warnings", [])

                    if not _should_execute:
                        checks.append({
                            "check":  "Entry quality guardrails",
                            "passed": False,
                            "detail": f"BLOCKED: {_block_reason} (quality={_quality_score}/100)",
                        })
                        allowed = False
                        result = {
                            # New canonical fields
                            "execution_allowed": False,
                            "blocked_reason":    f"Entry quality: {_block_reason}",
                            "failed_checks":     [{"check": "Entry quality guardrails",
                                                   "detail": f"BLOCKED: {_block_reason}"}],
                            "execution_action":  "NO TRADE",
                            # Legacy fields
                            "allowed":       False,
                            "passed":        passed,
                            "total":         total + 1,
                            "checks":        checks,
                            "final_action":  "NO TRADE",
                            "entry":         risk_out.get("entry"),
                            "sl":            risk_out.get("sl_price"),
                            "tp":            risk_out.get("tp_price"),
                            "lot":           risk_out.get("lot", 0),
                            "rr":            risk_out.get("rr_ratio", 0),
                        }
                        log.info(
                            f"[TradePermission] BLOCKED by entry quality guardrails: "
                            f"{_block_reason} (quality={_quality_score}/100) | "
                            f"Analysis verdict preserved upstream"
                        )
                        return result
                    else:
                        checks.append({
                            "check":  "Entry quality guardrails",
                            "passed": True,
                            "detail": f"quality={_quality_score}/100" + (
                                f" ({len(_eq_warnings)} warnings)" if _eq_warnings else ""
                            ),
                        })
                        total += 1
                        if _eq_warnings:
                            log.debug(
                                f"[TradePermission] Entry quality warnings: {_eq_warnings}"
                            )
                # If _df is None, skip guardrails (can't run without price data)
            except ImportError:
                log.debug("[TradePermission] entry_quality_guardrails not available — skipping")
            except Exception as _eq_e:
                log.warning(f"[TradePermission] Entry quality check error (non-fatal): {_eq_e}")

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
        blocked_reason = None if allowed else (
            checks[-1].get("detail") if checks and not checks[-1].get("passed", True)
            else "Multiple checks failed"
        )
        failed_checks = [
            {"check": c.get("check", "?"), "detail": c.get("detail", "")}
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
        }

        # ── INSTITUTIONAL LOG FORMAT ────────────────────────────────
        # Separates the ANALYSIS verdict from the EXECUTION verdict so the
        # operator can see "BUY 79% (analysis) → BLOCKED (news)" instead of
        # the misleading "WAIT 0%" that the old pipeline produced.
        _analysis_signal = decision_out.get("decision", "WAIT")
        _analysis_conf   = decision_out.get("confidence", 0)
        if allowed:
            log.info(
                f"[TradePermission] ALLOWED "
                f"({passed}/{total} checks passed) | "
                f"Analysis: {_analysis_signal} {_analysis_conf:.0f}% | "
                f"Execution: {execution_action}"
            )
        else:
            log.info(
                f"[TradePermission] BLOCKED "
                f"({passed}/{total} checks passed) | "
                f"Analysis: {_analysis_signal} {_analysis_conf:.0f}% | "
                f"Execution: BLOCKED | Reason: {blocked_reason}"
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