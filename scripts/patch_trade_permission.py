#!/usr/bin/env python3
"""
Patch trade_permission.py for entry quality soft scoring.
Reorders: remove old block FIRST, then insert new block.
"""
BASE = "/home/z/my-project/forex-agent"


def patch_trade_permission():
    path = f"{BASE}/risk/trade_permission.py"
    with open(path, "r") as f:
        content = f.read()

    # ── Step 1: Initialize total = 0 alongside passed = 0 ──
    old1 = "        passed = 0\n"
    new1 = "        passed = 0\n        total = 0\n"
    if old1 not in content:
        print("ERROR: Could not find 'passed = 0'")
        return False
    content = content.replace(old1, new1, 1)
    print("  Edit 1: total = 0 initialized")

    # ── Step 2: total = 5 -> total += 5 ──
    old2 = "            if ok: passed += 1\n            total = 5\n"
    new2 = "            if ok: passed += 1\n            total += 5\n"
    if old2 not in content:
        print("ERROR: Could not find 'total = 5'")
        return False
    content = content.replace(old2, new2, 1)
    print("  Edit 2: total = 5 -> total += 5")

    # ── Step 3: total = 4 -> total += 4 ──
    old3 = "            total = 4\n"
    new3 = "            total += 4\n"
    if old3 not in content:
        print("ERROR: Could not find 'total = 4'")
        return False
    content = content.replace(old3, new3, 1)
    print("  Edit 3: total = 4 -> total += 4")

    # ── Step 4: REMOVE old entry quality block FIRST ──
    # Find from the unique comment header to the last except line
    old5_start = "        # ── Round-22 audit fix: Entry Quality Guardrails ───────────────\n"
    old5_end_marker = '                log.warning(f"[TradePermission] Entry quality check error (non-fatal): {_eq_e}")\n'

    start_idx = content.find(old5_start)
    if start_idx == -1:
        print("ERROR: Could not find old entry quality block start")
        return False

    # Find old5_end_marker AFTER start_idx
    end_idx = content.find(old5_end_marker, start_idx)
    if end_idx == -1:
        print("ERROR: Could not find old entry quality block end")
        return False

    remove_end = end_idx + len(old5_end_marker)
    # Also remove trailing blank line if present
    if remove_end < len(content) and content[remove_end:remove_end + 1] == "\n":
        remove_end += 1

    print(f"  Edit 4: removing old entry quality block (chars {start_idx}-{remove_end})")
    content = content[:start_idx] + content[remove_end:]

    # ── Step 5: INSERT new entry quality block BEFORE confidence check ──
    old_anchor = """        if ok: passed += 1

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

    if old_anchor not in content:
        print("ERROR: Could not find insertion point (news -> confidence)")
        return False
    content = content.replace(old_anchor, new_eq_block, 1)
    print("  Edit 5: inserted new entry quality soft-scoring block before confidence gate")

    # ── Write ──
    with open(path, "w") as f:
        f.write(content)
    print("  Written to disk")
    return True


def verify():
    import py_compile
    path = f"{BASE}/risk/trade_permission.py"
    try:
        py_compile.compile(path, doraise=True)
        print(f"OK: {path} compiles cleanly")
        return True
    except py_compile.PyCompileError as e:
        print(f"SYNTAX ERROR in {path}: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("  Patching trade_permission.py")
    print("=" * 60)
    print()
    if patch_trade_permission():
        print()
        if verify():
            print("\nSUCCESS")
        else:
            print("\nSYNTAX ERROR - needs manual fix")
    else:
        print("\nFAILED")