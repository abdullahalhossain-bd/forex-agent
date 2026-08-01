#!/usr/bin/env python3
"""
monitoring/trade_pipeline_tracer.py — Detailed Trade Condition Tracer (Day 81+)
==============================================================================

WHY THIS EXISTS:
    The existing `debug_project.py` shows the FINAL outcome of one
    run_cycle() per symbol — but it doesn't show WHICH condition
    inside each pipeline stage caused the rejection. When the bot
    doesn't trade, you need to see:

        [market_data]    OK       (300 candles, RSI=52, trend=UP)
        [circuit_brkr]   OK       (mode=NORMAL, trades_today=0)
        [analysis]       signal=BUY conf=72%  (SMC: BOS @ 1.0850)
        [decision]       WAIT     (consensus=2/3, master=BUY, llm=WAIT, rule=BUY)
        [risk_engine]    REJECT   (lot=0.0, reason="Kelly negative: win_rate=42%")
        [permission]     (skipped because risk_engine rejected)
        [execution]      (skipped)
        ─────────────────────────────────────────────
        BLOCKED_AT: risk_engine
        ROOT_CAUSE: Kelly negative (win_rate=42% < 50% threshold)

    This script produces that trace by reading the SignalDebugger's
    output + re-interrogating each stage's intermediate result.

USAGE:
    # Trace one symbol end-to-end
    python -m monitoring.trade_pipeline_tracer EURUSD

    # Trace multiple symbols
    python -m monitoring.trade_pipeline_tracer EURUSD,GBPUSD,XAUUSD

    # Trace all symbols from config.py
    python -m monitoring.trade_pipeline_tracer --all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Project setup ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Quiet down chatty loggers
import os
os.environ.setdefault("ENABLE_TELEGRAM", "false")
os.environ.setdefault("USE_SCANNER", "false")

import logging
for noisy in ("urllib3", "httpx", "httpcore", "chromadb",
              "sentence_transformers", "huggingface_hub"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


# ── Color helpers ──────────────────────────────────────────────────
_IS_TTY = sys.stdout.isatty()


def _c(text: str, color: str) -> str:
    if not _IS_TTY:
        return text
    codes = {
        "red": "31", "green": "32", "yellow": "33",
        "blue": "34", "magenta": "35", "cyan": "36",
        "gray": "90", "bold": "1",
    }
    return f"\033[{codes.get(color, '0')}m{text}\033[0m"


def _stage_icon(status: str) -> str:
    s = status.upper()
    if s in ("OK", "BUY", "SELL"):
        return _c("[OK]  ", "green")
    if s in ("WAIT", "NEUTRAL"):
        return _c("[WAIT]", "yellow")
    if s in ("REJECT", "BLOCK", "ERROR"):
        return _c("[FAIL]", "red")
    if s == "DETECTED":
        return _c("[DET] ", "cyan")
    return _c(f"[{s[:4]:<4}]", "gray")


# ── Trace one symbol ───────────────────────────────────────────────

def trace_symbol(symbol: str, timeframe: str = "15m") -> Dict[str, Any]:
    """Run one real cycle and return a detailed stage-by-stage trace.

    Returns:
        {
            "symbol": "EURUSD",
            "final_action": "NO_TRADE",
            "blocked_at": "risk_engine",
            "root_cause": "Kelly negative (win_rate=42%)",
            "stages": [
                {"name": "market_data", "status": "OK", "detail": "...", "raw": {...}},
                {"name": "circuit_breaker", "status": "OK", "detail": "...", "raw": {...}},
                ...
            ],
        }
    """
    out: Dict[str, Any] = {
        "symbol": symbol,
        "timeframe": timeframe,
        "final_action": None,
        "blocked_at": None,
        "root_cause": None,
        "stages": [],
        "error": None,
    }

    try:
        # Boot the runtime up to RUNTIME phase (builds AITrader + all deps)
        from core.runtime import get_runtime, register_default_phases
        from core.lifecycle import Phase
        from core.service_registry import get_registry
        from core.trader import AITrader
        from monitoring.signal_debugger import get_signal_debugger

        rt = get_runtime()
        register_default_phases(rt.lifecycle)
        rt.lifecycle.boot(until=Phase.RUNTIME)
        reg = get_registry()

        # Reset debugger so we start fresh
        debugger = get_signal_debugger()

        trader = AITrader(
            balance=10000.0,
            symbol=symbol,
            timeframe=timeframe,
            registry=reg,
            notifier=None,
        )

        # Run one full cycle (auto_paper_trade=True so all stages execute)
        result = trader.run_cycle(auto_paper_trade=True)

        out["final_action"] = result.get("final_action", result.get("decision", "WAIT"))

        # ── Extract stage verdicts from the debugger ────────────────
        # The trader writes them via debugger.record() at each checkpoint.
        cycle_debug = debugger._current
        if cycle_debug is not None:
            for verdict in cycle_debug.layers:
                out["stages"].append({
                    "name":   verdict.layer,
                    "status": verdict.status,
                    "detail": verdict.detail,
                })
            out["blocked_at"] = cycle_debug.blocked_at

        # ── Determine root cause ─────────────────────────────────────
        # The reject_reason field has the most specific explanation.
        reject_reason = result.get("reject_reason", "")
        if reject_reason:
            out["root_cause"] = reject_reason[:200]
        elif out["final_action"] in ("BUY", "SELL"):
            out["root_cause"] = None  # trade went through
        else:
            out["root_cause"] = "No specific reason recorded"

        # ── Enrich with raw intermediate values (for power users) ────
        out["raw_result"] = {
            k: result.get(k) for k in (
                "final_action", "decision", "confidence", "reject_reason",
                "trade_allowed", "rule_signal", "llm_signal", "entry",
                "sl", "tp", "lot", "rr", "session", "monitor_only", "error",
                "news_safe", "approval_mode",
            ) if k in result
        }

        # ── If trade went through, capture execution details ─────────
        if out["final_action"] in ("BUY", "SELL"):
            out["trade_details"] = {
                "entry":  result.get("entry"),
                "sl":     result.get("sl"),
                "tp":     result.get("tp"),
                "lot":    result.get("lot"),
                "rr":     result.get("rr"),
                "confidence": result.get("confidence"),
            }

    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        out["blocked_at"] = "exception"
        out["root_cause"] = traceback.format_exc().splitlines()[-1]
        out["stages"].append({
            "name":   "exception",
            "status": "ERROR",
            "detail": str(e)[:200],
        })

    return out


# ── Pretty-print the trace ─────────────────────────────────────────

def print_trace(trace: Dict[str, Any]) -> None:
    """Print a human-readable trace."""
    symbol = trace["symbol"]
    tf = trace["timeframe"]
    final = trace.get("final_action", "WAIT")
    blocked = trace.get("blocked_at")

    print()
    print(_c("═" * 78, "bold"))
    print(_c(f"  {symbol} {tf} — TRADE PIPELINE TRACE", "bold"))
    print(_c("═" * 78, "bold"))

    if trace.get("error"):
        print(_c(f"  ❌ EXCEPTION during run_cycle: {trace['error']}", "red"))
        print(_c(f"     {trace.get('root_cause', '')}", "red"))
        print()
        return

    # Print each stage
    for stage in trace["stages"]:
        name = stage["name"]
        status = stage["status"]
        detail = stage["detail"]
        print(f"  {_stage_icon(status)} {name:<18} {detail}")

    print(_c("  " + "─" * 76, "gray"))

    # Final outcome
    if final in ("BUY", "SELL"):
        td = trace.get("trade_details", {})
        print(_c(f"  🟢 FINAL: {final}", "green") +
              f"  entry={td.get('entry')}  sl={td.get('sl')}  tp={td.get('tp')}  "
              f"lot={td.get('lot')}  rr=1:{td.get('rr')}")
    else:
        print(_c(f"  ⛔ FINAL: {final}", "red"))

    # Blocked-at summary
    if blocked:
        print(_c(f"  🎯 BLOCKED_AT: {blocked}", "yellow"))

    # Root cause
    root = trace.get("root_cause")
    if root:
        print(_c(f"  💡 ROOT_CAUSE: {root}", "cyan"))

    print()


# ── Main ───────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detailed trade condition tracer — see EXACTLY where signals die."
    )
    parser.add_argument(
        "symbols", nargs="?",
        help="Comma-separated symbols (e.g. EURUSD,GBPUSD)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Trace all symbols from config.SYMBOLS",
    )
    parser.add_argument(
        "--timeframe", default="15m",
        help="Timeframe (default: 15m)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output JSON instead of human-readable text",
    )
    args = parser.parse_args()

    # Resolve symbol list
    if args.all:
        try:
            from config import SYMBOLS
            symbols = list(SYMBOLS)
        except Exception:
            symbols = ["EURUSD"]
    elif args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = ["EURUSD"]

    print()
    print(_c("=" * 78, "bold"))
    print(_c("  TRADE PIPELINE TRACER — Why isn't the bot trading?", "bold"))
    print(_c("=" * 78, "bold"))
    print(f"  Symbols   : {', '.join(symbols)}")
    print(f"  Timeframe : {args.timeframe}")
    print(f"  Time      : {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print()

    results = []
    for sym in symbols:
        print(_c(f"→ Tracing {sym} {args.timeframe}...", "magenta"))
        t0 = time.time()
        trace = trace_symbol(sym, args.timeframe)
        elapsed = time.time() - t0
        print(_c(f"  ({elapsed:.1f}s)", "gray"))

        if args.json:
            results.append(trace)
        else:
            print_trace(trace)

    if args.json:
        print(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "timeframe": args.timeframe,
            "traces": results,
        }, indent=2, default=str))

    # Summary
    print(_c("=" * 78, "bold"))
    print(_c("  SUMMARY", "bold"))
    print(_c("=" * 78, "bold"))
    traded = [r for r in results if r.get("final_action") in ("BUY", "SELL")]
    blocked_list = [r for r in results if r.get("blocked_at")]
    crashed = [r for r in results if r.get("error")]
    print(f"  Total     : {len(results)}")
    print(f"  Traded    : {len(traded)}  ({', '.join(r['symbol'] for r in traded) or 'none'})")
    print(f"  Blocked   : {len(blocked_list)}")
    print(f"  Crashed   : {len(crashed)}")
    if crashed:
        print()
        print(_c("  ❌ Crashed symbols:", "red"))
        for r in crashed:
            print(f"     {r['symbol']}: {r.get('error')}")
    if blocked_list:
        print()
        print(_c("  Blocker breakdown:", "yellow"))
        blockers: Dict[str, int] = {}
        for r in blocked_list:
            b = r.get("blocked_at", "unknown")
            blockers[b] = blockers.get(b, 0) + 1
        for layer, count in sorted(blockers.items(), key=lambda x: -x[1]):
            print(f"     {layer:<20} → {count} symbol(s)")
    print()

    return 0 if not crashed else 1


if __name__ == "__main__":
    sys.exit(main())
