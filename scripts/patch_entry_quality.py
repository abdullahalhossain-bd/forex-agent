#!/usr/bin/env python3
"""
Patch entry quality guardrails from HARD BLOCK to SOFT SCORING.

Changes:
  1. risk/entry_quality_guardrails.py — run_all_entry_quality_checks() now
     returns confidence_penalty + per_check_report instead of hard should_execute=False
     (except for extreme safety cases).
  2. risk/trade_permission.py — entry quality runs BEFORE confidence gate,
     penalties reduce effective confidence, detailed per-check logging.
"""
import re

BASE = "/home/z/my-project/forex-agent"


def patch_entry_quality_guardrails():
    path = f"{BASE}/risk/entry_quality_guardrails.py"
    with open(path, "r") as f:
        content = f.read()

    # ── Replace the aggregation logic ──
    old_agg = '''    passed_count = sum(1 for r in results if r.passed)
    block_count = sum(1 for r in results if not r.passed and r.severity == "BLOCK")
    warning_count = sum(1 for r in results if not r.passed and r.severity == "WARNING")
    all_passed = passed_count == len(results)
    should_execute = block_count == 0

    block_reason = next(
        (r.reason for r in results if not r.passed and r.severity == "BLOCK"), None
    )
    warnings = [r.reason for r in results if not r.passed and r.severity == "WARNING"]

    # Quality score: 100 base - (blocks \u00d7 25) - (warnings \u00d7 10), clamped [0, 100]
    quality_score = max(0, min(100, 100 - (block_count * 25) - (warning_count * 10)))

    return {
        "all_passed":      bool(all_passed),
        "passed_count":    int(passed_count),
        "total_count":     len(results),
        "block_count":     int(block_count),
        "warning_count":   int(warning_count),
        "should_execute":  bool(should_execute),
        "results":         [r.to_dict() for r in results],
        "block_reason":    block_reason,
        "warnings":        warnings,
        "quality_score":   int(quality_score),
    }'''

    new_agg = '''    # ── SOFT SCORING: penalties instead of hard blocks ────────────
    # Each failed entry-quality check contributes a confidence penalty
    # rather than a hard block.  Only extreme / safety cases (SL or TP
    # on the WRONG SIDE, averaging into losers, opposite-direction
    # stacking) still produce should_execute=False.
    #
    # Operator-specified penalty values:
    #   Momentum exhaustion   -> -5 ~ -10  (mid: -8)
    #   Small body candles    -> -3 ~ -8   (mid: -5)
    #   Weak consolidation    -> -5
    #   Low momentum          -> -5
    #   Weak breakout         -> -5
    #   Poor candle quality   -> -5

    _PENALTY_MAP = {
        "chasing_filter":             8,   # momentum exhaustion (mid -5 to -10)
        "indecision_candles":         5,   # small body candles (mid -3 to -8)
        "sl_swing_anchor":            3,   # WARNING: not structurally anchored
        "tp_structure_validation":    3,   # WARNING: unconfirmed territory
        "indicator_confluence":       5,   # low momentum / weak breakout
        "round_number_tp":            2,   # minor
        "rejection_wick_at_entry":    5,   # poor candle quality
        "fresh_high_rejection":       5,   # weak breakout
        "tp_above_unconfirmed_spike": 3,   # minor
        "exhaustion_filter":          8,   # momentum exhaustion (mid -5 to -10)
    }

    _DISPLAY_NAMES = {
        "chasing_filter":              "Chasing Filter",
        "sl_swing_anchor":             "SL Swing Anchor",
        "tp_structure_validation":     "TP Structure",
        "indecision_candles":          "Indecision",
        "indicator_confluence":        "Confluence",
        "round_number_tp":             "Round Number TP",
        "rejection_wick_at_entry":     "Rejection Wick",
        "averaging_into_losers":       "Averaging",
        "fresh_high_rejection":        "Fresh High Reject",
        "tp_above_unconfirmed_spike":  "Spike TP",
        "opposite_direction_stacking": "Opposite Stack",
        "exhaustion_filter":           "Exhaustion",
    }

    # Extreme hard-blocks: only these keep should_execute=False
    _EXTREME_FLAGS = {"averaging_into_losers", "opposite_direction_stacking"}
    _EXTREME_REASON_KW = {"WRONG SIDE"}

    passed_count = sum(1 for r in results if r.passed)
    confidence_penalty = 0
    per_check_report = []
    extreme_block_reason = None
    warning_reasons = []

    for r in results:
        display = _DISPLAY_NAMES.get(r.flag_name, r.flag_name)
        if r.passed:
            per_check_report.append(f"{display:<22} PASS")
        else:
            is_extreme = (
                r.flag_name in _EXTREME_FLAGS
                or any(kw in r.reason for kw in _EXTREME_REASON_KW)
            )
            if is_extreme:
                per_check_report.append(f"{display:<22} BLOCK (extreme)")
                extreme_block_reason = extreme_block_reason or r.reason
            else:
                penalty = _PENALTY_MAP.get(r.flag_name, 3)
                confidence_penalty += penalty
                per_check_report.append(f"{display:<22} FAIL (-{penalty})")
                warning_reasons.append(r.reason)

    block_count = 1 if extreme_block_reason else 0
    warning_count = len(warning_reasons)
    should_execute = extreme_block_reason is None
    block_reason = extreme_block_reason

    # Quality score: 100 - total penalty, clamped [0, 100]
    quality_score = max(0, min(100, 100 - confidence_penalty))

    return {
        "all_passed":           bool(passed_count == len(results)),
        "passed_count":         int(passed_count),
        "total_count":          len(results),
        "block_count":          int(block_count),
        "warning_count":        int(warning_count),
        "should_execute":       bool(should_execute),
        "results":              [r.to_dict() for r in results],
        "block_reason":         block_reason,
        "warnings":             warning_reasons,
        "quality_score":        int(quality_score),
        # NEW soft-scoring fields
        "confidence_penalty":   int(confidence_penalty),
        "per_check_report":     per_check_report,
    }'''

    if old_agg not in content:
        print("ERROR: Could not find aggregation block in entry_quality_guardrails.py")
        # Debug: show what we're looking for
        # Try to find the first line
        first_line = "    passed_count = sum(1 for r in results if r.passed)"
        if first_line in content:
            print(f"  First line found, but full block doesn't match")
            idx = content.index(first_line)
            print(f"  Context around match: ...{content[idx:idx+200]}...")
        else:
            print(f"  First line NOT found either")
        return False

    content = content.replace(old_agg, new_agg, 1)

    with open(path, "w") as f:
        f.write(content)
    print("OK: patched entry_quality_guardrails.py (aggregation -> soft scoring)")
    return True


def patch_trade_permission():
    path = f"{BASE}/risk/trade_permission.py"
    with open(path, "r") as f:
        content = f.read()

    # ── Edit 1: Initialize total = 0 alongside passed = 0 ──
    old1 = "        passed = 0\n"
    new1 = "        passed = 0\n        total = 0\n"
    if old1 not in content:
        print("ERROR: Could not find 'passed = 0' in trade_permission.py")
        return False
    content = content.replace(old1, new1, 1)

    # ── Edit 2: total = 5 -> total += 5 (inside session_ctx block) ──
    old2 = "            if ok: passed += 1\n            total = 5\n"
    new2 = "            if ok: passed += 1\n            total += 5\n"
    if old2 not in content:
        print("ERROR: Could not find 'total = 5' in trade_permission.py")
        return False
    content = content.replace(old2, new2, 1)

    # ── Edit 3: total = 4 -> total += 4 (else branch) ──
    old3 = "            total = 4\n"
    new3 = "            total += 4\n"
    if old3 not in content:
        print("ERROR: Could not find 'total = 4' in trade_permission.py")
        return False
    content = content.replace(old3, new3, 1)

    # ── Edit 4: Insert new entry quality block BEFORE confidence check ──
    old4 = """        if ok: passed += 1

        # 4. Confidence"""

    new_eq_block = """        if ok: passed += 1

        # ── ENTRY QUALITY: SOFT SCORING ───────────────────────────
        # Runs BEFORE the confidence gate so penalties reduce the
        # effective confidence.  Only extreme cases (SL/TP wrong side,
        # averaging into losers, opposite-direction stacking) still
        # hard-block.  All other entry-quality issues (exhaustion,
        # indecision, small candles, chasing, etc.) become confidence
        # penalties.  Entry quality alone NEVER rejects the trade.
        _eq_penalty = 0
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
                # If _df is None, skip guardrails (can't run without price data)
            except ImportError:
                log.debug("[TradePermission] entry_quality_guardrails not available - skipping")
            except Exception as _eq_e:
                log.warning(f"[TradePermission] Entry quality check error (non-fatal): {_eq_e}")
        # ── END ENTRY QUALITY ──────────────────────────────────────

        # 4. Confidence"""

    if old4 not in content:
        print("ERROR: Could not find insertion point (news -> confidence) in trade_permission.py")
        # Debug
        if "        # 4. Confidence" in content:
            print("  Found '# 4. Confidence' but context doesn't match")
            idx = content.index("        # 4. Confidence")
            print(f"  Context: ...{repr(content[idx-60:idx+30])}...")
        return False
    content = content.replace(old4, new_eq_block, 1)

    # ── Edit 5: Remove the OLD entry quality block ──
    old5_start = "        # ── Round-22 audit fix: Entry Quality Guardrails ───────────────\n"
    old5_end = '                log.warning(f"[TradePermission] Entry quality check error (non-fatal): {_eq_e}")\n'

    start_idx = content.find(old5_start)
    end_idx = content.find(old5_end)
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        print("ERROR: Could not find old entry quality block in trade_permission.py")
        print(f"  start_idx={start_idx}, end_idx={end_idx}")
        return False

    # Remove from start of comment to end of last except line (inclusive)
    remove_end = end_idx + len(old5_end)
    # Also remove trailing blank line if present
    if content[remove_end:remove_end + 1] == "\n":
        remove_end += 1
    content = content[:start_idx] + content[remove_end:]

    with open(path, "w") as f:
        f.write(content)
    print("OK: patched trade_permission.py (soft scoring + moved before confidence gate)")
    return True


def verify():
    """Quick sanity check: make sure the patched files parse without syntax errors."""
    import py_compile

    files = [
        f"{BASE}/risk/entry_quality_guardrails.py",
        f"{BASE}/risk/trade_permission.py",
    ]
    ok = True
    for path in files:
        try:
            py_compile.compile(path, doraise=True)
            print(f"OK: {path} compiles cleanly")
        except py_compile.PyCompileError as e:
            print(f"SYNTAX ERROR in {path}: {e}")
            ok = False
    return ok


if __name__ == "__main__":
    print("=" * 60)
    print("  Patching Entry Quality: HARD BLOCK -> SOFT SCORING")
    print("=" * 60)
    print()

    r1 = patch_entry_quality_guardrails()
    print()
    r2 = patch_trade_permission()
    print()

    if r1 and r2:
        print("=" * 60)
        print("  Verifying syntax...")
        print("=" * 60)
        if verify():
            print()
            print("ALL PATCHES APPLIED SUCCESSFULLY")
        else:
            print()
            print("PATCHES APPLIED BUT SYNTAX ERRORS DETECTED - manual fix needed")
    else:
        print("PATCHING FAILED - see errors above")