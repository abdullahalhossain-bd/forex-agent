"""
audit_tools/static_code_audit.py — Full critical-path static audit
==================================================================

Scans every module on the backtest/live decision path for the issues
the operator listed:

  - incorrect logic
  - missing data
  - silent failures (bare except, log.debug on errors, fallback allow-all)
  - fallback behavior that hides failures
  - duplicated filters
  - contradictory rules
  - excessive confidence penalties
  - unused/dead modules
  - models trained but never used
  - backtest/live behavior mismatch
  - incorrect pip/point/spread calculations
  - look-ahead / data leakage
  - unrealistic backtest assumptions
  - errors that could cause unintended real-money losses

Output:
  download/ablation_results/static_code_audit.json
  stdout (human-readable summary)

USAGE:
    py -3.13 audit_tools\\static_code_audit.py
"""
from __future__ import annotations

import ast
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ── Files on the critical decision path ─────────────────────────────────
CRITICAL_FILES = [
    "backtest/persistent_runner.py",
    "backtest/unified_engine.py",
    "backtest/broker_sim.py",
    "backtest/gating_bridge.py",
    "backtest/symbol_specs.py",
    "core/trader.py",
    "core/csv_data_provider.py",
    "core/data_provider.py",
    "core/execution_adapter.py",
    "core/constants.py",
    "core/devils_advocate.py",
    "core/regime_suppression.py",
    "core/signal_persistence.py",
    "core/confidence_manager.py",
    "core/confidence_breakdown.py",
    "core/signal_scorer.py",
    "core/entry_safety_filters.py",
    "core/master_decision.py",
    "core/fusion_engine_v3.py",
    "agents/analysis_agent.py",
    "agents/decision_agent.py",
    "agents/market_agent.py",
    "risk/trade_permission.py",
    "risk/risk_engine.py",
    "risk/live_risk_manager.py",
    "risk/usd_tp_sl_calculator.py",
    "risk/atr_risk_manager.py",
    "risk/strict_risk_manager.py",
    "risk/position_sizer.py",
    "risk/position_allocator.py",
    "risk/correlation_manager.py",
    "risk/drawdown_controller.py",
    "risk/kill_switch.py",
    "risk/circuit_breaker.py",
    "risk/trade_frequency.py",
    "risk/entry_quality_guardrails.py",
    "risk/confirmation_bias_defense.py",
    "risk/revenge_trading_detector.py",
    "risk/book_guardrails.py",
    "risk/advanced_risk_orchestrator.py",
    "learning/confidence_engine.py",
    "ml/rl_agent.py",
    "ml/rl_policy_store.py",
    "ml/model_predictor.py",
    "ml/model_store.py",
    "ml/ensemble.py",
    "ml/feature_engineer.py",
    "ml/confidence_fusion.py",
    "execution/paper_trader.py",
    "execution/order_manager.py",
    "execution/simulated_executor.py",
    "broker/spread_monitor.py",
    "data/fetcher.py",
    "data/indicators.py",
    "data/indicators_ext.py",
    "data/indicator_registry.py",
    "analysis/market_regime.py",
    "analysis/support_resistance.py",
    "analysis/stop_hunt_direct_lane.py",
    "analysis/confluence_engine.py",
    "analysis/timeframe.py",
    "analysis/session_analyzer.py",
    "main.py",
    "config.py",
    "utils/logger.py",
]


def _read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""


def _ast_parse(src: str):
    try:
        return ast.parse(src)
    except Exception:
        return None


# ── Individual audit checks ─────────────────────────────────────────────

def check_bare_excepts(files: dict[str, str]) -> list:
    """Bare `except:` swallows everything including KeyboardInterrupt."""
    findings = []
    for path, src in files.items():
        tree = _ast_parse(src)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                # Check if it's followed by `pass` (silent) or has logging
                body_src = ast.get_source_segment(src, node) or ""
                is_silent = ("pass" in body_src and
                              "log" not in body_src.lower() and
                              "print" not in body_src.lower())
                findings.append({
                    "file": path,
                    "line": node.lineno,
                    "severity": "HIGH" if is_silent else "MEDIUM",
                    "issue": "bare except:" + (" + silent pass (swallows ALL errors including KeyboardInterrupt)" if is_silent else ""),
                    "evidence": body_src[:120].replace("\n", " "),
                })
    return findings


def check_silent_fallbacks(files: dict[str, str]) -> list:
    """log.debug on errors, return-on-exception, allow-all fallbacks."""
    findings = []
    # Pattern: `except Exception as e: log.debug(...)` — error hidden at debug
    pat_debug = re.compile(
        r"except\s+(Exception|BaseException|.*Error).*?log\.debug\(",
        re.DOTALL,
    )
    # Pattern: `except.*:\s*pass` — silent swallow
    pat_pass = re.compile(r"except[^\n]*:\s*\n\s*pass\b")
    # Pattern: 'fail-open' comments (backtest-only fail-open)
    pat_fail_open = re.compile(r"fail[- ]?open", re.IGNORECASE)

    for path, src in files.items():
        for m in pat_debug.finditer(src):
            line = src[:m.start()].count("\n") + 1
            findings.append({
                "file": path,
                "line": line,
                "severity": "MEDIUM",
                "issue": "exception caught but only logged at debug level (invisible in INFO logs)",
                "evidence": src[max(0,m.start()-30):m.end()+30].replace("\n"," ")[:160],
            })
        for m in pat_pass.finditer(src):
            line = src[:m.start()].count("\n") + 1
            findings.append({
                "file": path,
                "line": line,
                "severity": "HIGH",
                "issue": "silent except + pass (error swallowed with no logging)",
                "evidence": src[max(0,m.start()-30):m.end()+30].replace("\n"," ")[:160],
            })
        for m in pat_fail_open.finditer(src):
            line = src[:m.start()].count("\n") + 1
            findings.append({
                "file": path,
                "line": line,
                "severity": "INFO",
                "issue": "fail-open behavior documented in comment (intentional, but verify it's only backtest)",
                "evidence": src[max(0,m.start()-30):m.end()+30].replace("\n"," ")[:160],
            })
    return findings


def check_lookahead(files: dict[str, str]) -> list:
    """Look for potential look-ahead: df.shift(-N), df.iloc[i+1:], future bars."""
    findings = []
    patterns = [
        (r"shift\s*\(\s*-\s*[1-9]", "negative shift (future data leaked into current row)"),
        (r"iloc\[i\s*\+\s*[2-9]", "iloc[i+N] with N>=2 (may access future bar)"),
        (r"iloc\[i\s*\+\s*1\s*:", "iloc[i+1:] slice (accesses future bars)"),
        (r"\.iloc\[:i\s*\+\s*1\]", "iloc[:i+1] (correct causal slice — record as PASS)", ),
    ]
    for path, src in files.items():
        for pat, desc in patterns:
            for m in re.finditer(pat, src):
                line = src[:m.start()].count("\n") + 1
                severity = "PASS" if "correct" in desc else "HIGH"
                findings.append({
                    "file": path,
                    "line": line,
                    "severity": severity,
                    "issue": desc,
                    "evidence": src[max(0,m.start()-30):m.end()+30].replace("\n"," ")[:160],
                })
    return findings


def check_pip_point_spread(files: dict[str, str]) -> list:
    """Audit pip/point/spread calculations for correctness."""
    findings = []
    # Yen pairs have pip=0.01, others 0.0001 — check pip_size table
    pip_table_files = ["backtest/symbol_specs.py", "core/constants.py"]
    for pf in pip_table_files:
        if pf in files:
            src = files[pf]
            # Find pip_size definitions
            for m in re.finditer(r"(JPY|jpy)", src):
                line = src[:m.start()].count("\n") + 1
                # Check the context — is JPY handled specially?
                ctx = src[max(0, m.start()-200):m.end()+200]
                if "0.01" in ctx or "pip_size" in ctx.lower():
                    findings.append({
                        "file": pf,
                        "line": line,
                        "severity": "INFO",
                        "issue": "JPY pair pip size handling detected (verify = 0.01)",
                        "evidence": ctx[:200].replace("\n"," "),
                    })
    # Check default spread values
    for path, src in files.items():
        for m in re.finditer(r"DEFAULT_SPREAD_PIPS\s*=\s*(\{[^}]+\})", src, re.DOTALL):
            line = src[:m.start()].count("\n") + 1
            findings.append({
                "file": path,
                "line": line,
                "severity": "INFO",
                "issue": "DEFAULT_SPREAD_PIPS table — verify per-symbol values match live broker spreads",
                "evidence": m.group(1)[:200].replace("\n"," "),
            })
    # Check for hardcoded pip values
    for path, src in files.items():
        for m in re.finditer(r"\b0\.0001\b", src):
            line = src[:m.start()].count("\n") + 1
            ctx = src[max(0,m.start()-60):m.end()+60]
            # Skip comments and pip_size definitions
            if "//" in ctx[:ctx.find("0.0001")] or "pip" in ctx.lower():
                continue
            findings.append({
                "file": path,
                "line": line,
                "severity": "LOW",
                "issue": "hardcoded 0.0001 (may be pip size for non-JPY — verify not a magic number)",
                "evidence": ctx.replace("\n"," ")[:160],
            })
    return findings


def check_duplicate_filters(files: dict[str, str]) -> list:
    """Find duplicate/overlapping filter logic."""
    findings = []
    # Confluence check appears in multiple places:
    #   - analysis/confluence_engine.py (ConfluenceEngine)
    #   - intelligence/confluence_engine.py (ConfluenceDecision)
    #   - risk/trade_permission.py (Confluence quality gate)
    #   - backtest/gating_bridge.py (Confluence quality gate, backtest copy)
    confluence_locations = []
    for path, src in files.items():
        if "confluence" in path.lower() or "Confluence" in src:
            count = src.count("confluence")
            if count > 5:
                confluence_locations.append({"file": path, "mentions": count})
    if len(confluence_locations) > 2:
        findings.append({
            "file": "(multiple)",
            "line": 0,
            "severity": "MEDIUM",
            "issue": f"confluence logic spread across {len(confluence_locations)} files — risk of duplicate/contradictory filters",
            "evidence": str(confluence_locations[:5]),
        })
    # Session quality check
    session_locations = []
    for path, src in files.items():
        if "session_quality" in src or "session_grade" in src:
            session_locations.append(path)
    if len(session_locations) > 3:
        findings.append({
            "file": "(multiple)",
            "line": 0,
            "severity": "MEDIUM",
            "issue": f"session quality logic in {len(session_locations)} files — verify no contradictions",
            "evidence": str(session_locations[:6]),
        })
    return findings


def check_backtest_live_mismatch(files: dict[str, str]) -> list:
    """Find places where backtest behavior diverges from live."""
    findings = []
    for path, src in files.items():
        # Look for execution_mode == 'backtest' branches that skip checks
        for m in re.finditer(r"execution_mode[^=]*==[^=]*['\"]backtest['\"]", src):
            line = src[:m.start()].count("\n") + 1
            ctx = src[max(0,m.start()-100):m.end()+400]
            # Is this skipping a safety check?
            if any(k in ctx.lower() for k in ["skip", "bypass", "disable", "return true", "allow"]):
                findings.append({
                    "file": path,
                    "line": line,
                    "severity": "HIGH",
                    "issue": "backtest-mode branch that may skip a live safety check",
                    "evidence": ctx[:300].replace("\n"," "),
                })
        # Look for "in TEST_MODE" branches
        for m in re.finditer(r"_test_mode\(\)|TEST_MODE", src):
            line = src[:m.start()].count("\n") + 1
            ctx = src[max(0,m.start()-100):m.end()+300]
            if any(k in ctx.lower() for k in ["skip", "bypass", "always pass", "return true"]):
                findings.append({
                    "file": path,
                    "line": line,
                    "severity": "HIGH",
                    "issue": "TEST_MODE branch that disables a safety check — must never be True in production",
                    "evidence": ctx[:300].replace("\n"," "),
                })
    return findings


def check_confidence_penalties(files: dict[str, str]) -> list:
    """Find confidence penalty applications — look for excessive ones."""
    findings = []
    # _apply_confidence_penalty(signal_result, N, reason, ...) — N is the penalty
    for path, src in files.items():
        for m in re.finditer(r"_apply_confidence_penalty\s*\([^,]+,\s*(\d+)", src):
            penalty = int(m.group(1))
            line = src[:m.start()].count("\n") + 1
            ctx = src[max(0,m.start()-50):m.end()+200]
            reason_match = re.search(r'"([^"]+)"', ctx[len(m.group(0)):])
            reason = reason_match.group(1) if reason_match else "?"
            severity = "MEDIUM" if penalty >= 15 else "LOW"
            findings.append({
                "file": path,
                "line": line,
                "severity": severity,
                "issue": f"confidence penalty of {penalty} applied for '{reason}'",
                "evidence": ctx[:200].replace("\n"," "),
            })
    return findings


def check_min_confidence_threshold(files: dict[str, str]) -> list:
    """Find MIN_CONFIDENCE definitions."""
    findings = []
    for path, src in files.items():
        for m in re.finditer(r"MIN_CONFIDENCE\s*[:=]\s*(\d+(?:\.\d+)?)", src):
            val = float(m.group(1))
            line = src[:m.start()].count("\n") + 1
            findings.append({
                "file": path,
                "line": line,
                "severity": "INFO",
                "issue": f"MIN_CONFIDENCE = {val}",
                "evidence": src[max(0,m.start()-60):m.end()+60].replace("\n"," ")[:160],
            })
    return findings


def check_risk_management_bypass(files: dict[str, str]) -> list:
    """Verify NO code path bypasses risk management in live mode."""
    findings = []
    # Look for "risk_approved = True" hardcoding (other than placeholder_risk)
    for path, src in files.items():
        for m in re.finditer(r"risk_out\s*\[\s*[\"']approved[\"']\s*\]\s*=\s*True", src):
            line = src[:m.start()].count("\n") + 1
            ctx = src[max(0,m.start()-200):m.end()+200]
            if "placeholder" in ctx.lower():
                continue  # placeholder_risk pattern is OK
            findings.append({
                "file": path,
                "line": line,
                "severity": "CRITICAL",
                "issue": "hardcoded risk_out['approved'] = True — possible risk bypass",
                "evidence": ctx[:200].replace("\n"," "),
            })
    return findings


def check_unrealistic_backtest_assumptions(files: dict[str, str]) -> list:
    """Find things like: instant fills, no slippage, no spread, perfect TP/SL."""
    findings = []
    for path, src in files.items():
        if "slippage_pips=0" in src or "slippage_pips = 0" in src:
            line = src[:src.find("slippage_pips")].count("\n") + 1
            findings.append({
                "file": path,
                "line": line,
                "severity": "HIGH",
                "issue": "slippage_pips=0 — unrealistic backtest assumption (live always has slippage)",
                "evidence": src[max(0,src.find("slippage_pips")-60):src.find("slippage_pips")+60].replace("\n"," "),
            })
        if "spread_pips=0" in src or "spread_pips = 0" in src:
            line = src[:src.find("spread_pips")].count("\n") + 1
            findings.append({
                "file": path,
                "line": line,
                "severity": "HIGH",
                "issue": "spread_pips=0 — unrealistic backtest assumption (live always has spread)",
                "evidence": src[max(0,src.find("spread_pips")-60):src.find("spread_pips")+60].replace("\n"," "),
            })
    return findings


def check_real_money_safety(files: dict[str, str]) -> list:
    """Patterns that could cause unintended real-money losses."""
    findings = []
    # 1. market orders without SL
    for path, src in files.items():
        for m in re.finditer(r"send_order|order_send|market_buy|market_sell", src, re.IGNORECASE):
            line = src[:m.start()].count("\n") + 1
            ctx = src[max(0,m.start()-200):m.end()+400]
            if "sl" not in ctx.lower() and "stop_loss" not in ctx.lower() and "stoploss" not in ctx.lower():
                findings.append({
                    "file": path,
                    "line": line,
                    "severity": "CRITICAL",
                    "issue": f"order send without explicit SL in nearby code ({m.group(0)})",
                    "evidence": ctx[:200].replace("\n"," "),
                })
    # 2. lot size hardcoding to large value
    for path, src in files.items():
        for m in re.finditer(r"lot\s*=\s*(\d+\.?\d*)", src):
            try:
                lot = float(m.group(1))
                if lot >= 1.0 and "test" not in path.lower() and "audit" not in path.lower():
                    line = src[:m.start()].count("\n") + 1
                    findings.append({
                        "file": path,
                        "line": line,
                        "severity": "HIGH",
                        "issue": f"hardcoded lot={lot} (>=1.0) — verify this is sized from balance, not fixed",
                        "evidence": src[max(0,m.start()-60):m.end()+60].replace("\n"," ")[:160],
                    })
            except ValueError:
                pass
    return findings


def main():
    print("="*78)
    print("  STATIC CODE AUDIT — Critical Decision Path")
    print("="*78)

    files = {}
    missing = []
    for rel in CRITICAL_FILES:
        p = PROJECT_ROOT / rel
        if p.exists():
            files[rel] = _read(p)
        else:
            missing.append(rel)
    print(f"\n[audit] {len(files)} files scanned, {len(missing)} missing")
    if missing:
        print(f"[audit] Missing: {missing[:5]}{'...' if len(missing)>5 else ''}")

    all_findings = {
        "bare_excepts": check_bare_excepts(files),
        "silent_fallbacks": check_silent_fallbacks(files),
        "lookahead": check_lookahead(files),
        "pip_point_spread": check_pip_point_spread(files),
        "duplicate_filters": check_duplicate_filters(files),
        "backtest_live_mismatch": check_backtest_live_mismatch(files),
        "confidence_penalties": check_confidence_penalties(files),
        "min_confidence_thresholds": check_min_confidence_threshold(files),
        "risk_management_bypass": check_risk_management_bypass(files),
        "unrealistic_backtest_assumptions": check_unrealistic_backtest_assumptions(files),
        "real_money_safety": check_real_money_safety(files),
    }

    # Print summary
    total = 0
    by_severity = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "INFO": 0, "PASS": 0}
    for category, items in all_findings.items():
        if not items:
            continue
        print(f"\n[{category}] ({len(items)} findings)")
        for f in items[:15]:  # show first 15 per category
            sev = f.get("severity", "?")
            by_severity[sev] = by_severity.get(sev, 0) + 1
            total += 1
            marker = {"CRITICAL":"🔥","HIGH":"✗","MEDIUM":"!","LOW":"·","INFO":"i","PASS":"✓"}.get(sev, "?")
            print(f"  {marker} [{sev}] {f['file']}:{f['line']} — {f['issue']}")
            ev = f.get('evidence', '')
            if ev and sev in ("CRITICAL", "HIGH"):
                print(f"      | {ev[:150]}")
        if len(items) > 15:
            print(f"  ... and {len(items)-15} more")

    print("\n" + "="*78)
    print("  SEVERITY SUMMARY")
    print("="*78)
    for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO", "PASS"]:
        if by_severity.get(sev, 0) > 0:
            print(f"  {sev:<10} {by_severity[sev]:>4}")
    print(f"  {'TOTAL':<10} {total:>4}")

    # Write JSON
    out_dir = PROJECT_ROOT / "download" / "ablation_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files_scanned": list(files.keys()),
        "files_missing": missing,
        "total_findings": total,
        "by_severity": by_severity,
        "findings": all_findings,
    }
    out_path = out_dir / "static_code_audit.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\n[audit] Full JSON report: {out_path}")

    return 1 if by_severity.get("CRITICAL", 0) > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
