#!/usr/bin/env python3
"""
debug_project.py — Whole-project error scanner for forex_ai
============================================================

Runs a comprehensive sweep across every Python file in the project
and reports any issue that would break `python main.py`. Use this
BEFORE running main.py to catch silent failures early.

Checks performed:
  1. Syntax  — every .py file parses cleanly
  2. Imports — every module imports without ImportError / ModuleNotFoundError
             (now includes core.approval_mode, core.trade_decision_log,
              core.orphan_cleanup, monitoring.signal_debugger,
              intelligence.confluence_engine, risk.trade_frequency, etc.)
  3. Missing dependencies (pip packages not installed)
  4. Critical config files exist (.env, config.py) + directories writable
  5. Config flags (.env production-safety) — TEST_MODE, SIMULATION_MODE,
             MAX_LOT, APPROVAL_MODE, MAX_OPEN_TRADES
  6. Stray files / common gotchas (.pyc outside cache, CRLF endings)
  7. Network reachability (Groq, Gemini, Telegram, yfinance)
  8. All 24 boot phases can complete (dry-run) — including trading_engine,
             telegram_notifier, correlation_filter, mistake_analyzer
  9. Day 76 PositionSizer functional test (happy path + reject + correlation)
 10. AITrader construction with registry — verifies position_sizer,
             live_risk_manager, drawdown_monitor, kill_switch,
             mistake_analyzer, circuit_breaker, approval, corr_filter

Usage:
    python debug_project.py            # full scan (10 checks)
    python debug_project.py --quick    # skip the heavy boot/AITrader tests (checks 1-7 only)
    python debug_project.py --json     # machine-readable output

Exit code:
    0 = all checks passed
    1 = at least one error found
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Project setup ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Quiet down the chatty loggers during the scan
os.environ.setdefault("EXECUTION_MODE", "mt5_demo")
os.environ.setdefault("APPROVAL_MODE", "3")
os.environ.setdefault("ENABLE_TELEGRAM", "false")
os.environ.setdefault("USE_SCANNER", "false")


# ── Color helpers (disabled if not a TTY) ───────────────────────────
_IS_TTY = sys.stdout.isatty()


def _c(text: str, color: str) -> str:
    if not _IS_TTY:
        return text
    codes = {
        "red":    "31",
        "green":  "32",
        "yellow": "33",
        "blue":   "34",
        "magenta":"35",
        "cyan":   "36",
        "gray":   "90",
        "bold":   "1",
    }
    return f"\033[{codes.get(color, '0')}m{text}\033[0m"


# ── Result containers ───────────────────────────────────────────────
class CheckResult:
    def __init__(self, name: str, category: str):
        self.name = name
        self.category = category
        self.ok: bool = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []

    def error(self, msg: str) -> None:
        self.ok = False
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
        }


ALL_RESULTS: List[CheckResult] = []


def _print_result(r: CheckResult, verbose: bool = False) -> None:
    if r.ok and not r.warnings and not verbose:
        print(f"  {_c('OK', 'green')}  {r.name}")
        return
    if r.ok:
        print(f"  {_c('OK', 'green')}  {r.name}")
    else:
        print(f"  {_c('FAIL', 'red')}  {r.name}")
    for w in r.warnings:
        print(f"       {_c('WARN', 'yellow')}  {w}")
    for e in r.errors:
        print(f"       {_c('ERR ', 'red')}  {e}")
    if verbose:
        for i in r.info:
            print(f"       {_c('info', 'gray')}  {i}")


# ── 1. Syntax check — every .py file ────────────────────────────────
def check_syntax() -> CheckResult:
    r = CheckResult("Python syntax for every .py file", "syntax")
    py_files = sorted(PROJECT_ROOT.rglob("*.py"))
    r.info.append(f"Scanned {len(py_files)} .py files")
    for f in py_files:
        # Skip hidden / cache / venv directories
        rel = f.relative_to(PROJECT_ROOT)
        if any(part in {"__pycache__", ".git", ".venv", "venv", "node_modules"}
               for part in rel.parts):
            continue
        try:
            with open(f, "r", encoding="utf-8", errors="replace") as fh:
                src = fh.read()
            ast.parse(src, filename=str(f))
        except SyntaxError as e:
            r.error(f"{rel}:{e.lineno}:{e.offset} — {e.msg}")
    ALL_RESULTS.append(r)
    return r


# ── 2. Imports check — every module imports cleanly ─────────────────
# Map of (module path, friendly name). We use Python module paths
# (dotted) so importlib actually executes the module top-level code.
CRITICAL_MODULES = [
    # Core infrastructure
    ("core.runtime", "core.runtime"),
    ("core.trader", "core.trader"),
    ("core.event_bus", "core.event_bus"),
    ("core.lifecycle", "core.lifecycle"),
    ("core.service_registry", "core.service_registry"),
    ("core.health_monitor", "core.health_monitor"),
    ("core.runtime_metrics", "core.runtime_metrics"),
    ("core.constants", "core.constants"),
    ("core.master_decision", "core.master_decision"),
    ("core.signal_fusion", "core.signal_fusion"),
    ("core.confidence_manager", "core.confidence_manager"),
    ("core.decision_validator", "core.decision_validator"),
    ("core.llm_key_manager", "core.llm_key_manager"),
    ("core.professional_tools", "core.professional_tools"),
    ("core.trading_engine", "core.trading_engine"),
    # AITrader direct deps (trader.py — used in run_cycle + __init__)
    ("core.approval_mode", "core.approval_mode"),
    ("core.execution_logger", "core.execution_logger (Day 81+)"),
    ("core.trade_decision_log", "core.trade_decision_log (Day 81+)"),
    ("core.orphan_cleanup", "core.orphan_cleanup (Day 81+)"),
    # Monitoring (Day 81+)
    ("monitoring.signal_debugger", "monitoring.signal_debugger (Day 81+)"),
    ("monitoring.trade_pipeline_tracer", "monitoring.trade_pipeline_tracer (Day 81+)"),
    # Intelligence (Day 66/67 — confluence + news alerts)
    ("intelligence.confluence_engine", "intelligence.confluence_engine (Day 67)"),
    ("intelligence.news_ai", "intelligence.news_ai (Day 66)"),
    # Risk — trade frequency controller (Day 84+)
    ("risk.trade_frequency", "risk.trade_frequency (Day 84+)"),
    # Learning — mistake analyzer (used in trader.py)
    ("learning.mistake_analyzer", "learning.mistake_analyzer"),
    # Config
    ("config", "config"),
    # Database + memory
    ("database.db", "database.db"),
    ("memory.trade_memory", "memory.trade_memory"),
    ("memory.learning", "memory.learning"),
    ("memory.history", "memory.history"),
    ("memory.knowledge_store", "memory.knowledge_store"),
    # Data + analysis
    ("data.fetcher", "data.fetcher"),
    ("data.indicators", "data.indicators"),
    ("data.validator", "data.validator"),
    ("analysis.intermarket", "analysis.intermarket"),
    ("analysis.session_analyzer", "analysis.session_analyzer"),
    # Agents + AI
    ("agents.market_agent", "agents.market_agent"),
    ("agents.analysis_agent", "agents.analysis_agent"),
    ("agents.decision_agent", "agents.decision_agent"),
    ("agents.learning_agent", "agents.learning_agent"),
    ("agents.risk_agent", "agents.risk_agent"),
    ("agents.master_analyst", "agents.master_analyst"),
    ("ai.ai_analyst", "ai.ai_analyst"),
    # Risk — Day 75 + Day 76
    ("risk.risk_engine", "risk.risk_engine"),
    ("risk.circuit_breaker", "risk.circuit_breaker"),
    ("risk.trade_permission", "risk.trade_permission"),
    ("risk.drawdown_controller", "risk.drawdown_controller"),
    ("risk.autonomous_risk", "risk.autonomous_risk"),
    ("risk.kill_switch", "risk.kill_switch (Day 75)"),
    ("risk.exposure_manager", "risk.exposure_manager (Day 75)"),
    ("risk.drawdown_monitor", "risk.drawdown_monitor (Day 75)"),
    ("risk.risk_reporter", "risk.risk_reporter (Day 75)"),
    ("risk.live_risk_manager", "risk.live_risk_manager (Day 75)"),
    ("risk.kelly_calculator", "risk.kelly_calculator (Day 76)"),
    ("risk.volatility_adjuster", "risk.volatility_adjuster (Day 76)"),
    ("risk.confidence_scaler", "risk.confidence_scaler (Day 76)"),
    ("risk.correlation_manager", "risk.correlation_manager (Day 76)"),
    ("risk.position_sizer", "risk.position_sizer (Day 76)"),
    # Scanner + strategies
    ("scanner.market_scanner", "scanner.market_scanner"),
    ("scanner.correlation_filter", "scanner.correlation_filter"),
    ("scanner.opportunity_ranker", "scanner.opportunity_ranker"),
    ("strategy.signal_engine", "strategy.signal_engine"),
    # Execution + broker
    ("execution.paper_trader", "execution.paper_trader"),
    ("execution.execution_router", "execution.execution_router"),
    # Alerts
    ("alerts.telegram_bot", "alerts.telegram_bot"),
    # Main entry
    ("main", "main"),
]


def check_imports() -> CheckResult:
    r = CheckResult("Critical module imports", "imports")
    r.info.append(f"Testing {len(CRITICAL_MODULES)} modules")
    for mod_path, friendly in CRITICAL_MODULES:
        try:
            importlib.import_module(mod_path)
        except ModuleNotFoundError as e:
            # Missing pip dependency
            missing = e.name or mod_path
            r.error(f"{friendly}: missing dependency '{missing}'  →  pip install {missing}")
        except ImportError as e:
            r.error(f"{friendly}: {e}")
        except Exception as e:
            # Other exceptions (e.g. config errors, runtime crashes at import time)
            short = traceback.format_exc().splitlines()[-1]
            r.error(f"{friendly}: {type(e).__name__}: {e}")
    ALL_RESULTS.append(r)
    return r


# ── 3. Missing pip dependencies (explicit list) ─────────────────────
REQUIRED_PACKAGES = [
    # Hard runtime deps
    "pandas", "numpy", "requests", "yfinance", "ta",
    "python-telegram-bot", "flask",
    "scikit-learn", "joblib",
    # Optional but commonly needed
    # MetaTrader5 is Windows-only — paper mode + broker phase skip cleanly
    ("MetaTrader5", "optional"),
    # ML / LLM (optional — degraded mode if missing)
    ("sentence-transformers", "optional"),
    ("chromadb", "optional"),
    ("groq", "optional"),
    ("google-genai", "optional"),
    ("stable-baselines3", "optional"),
    ("mlflow", "optional"),
    ("anthropic", "optional"),
]


def check_dependencies() -> CheckResult:
    r = CheckResult("Pip dependencies installed", "deps")
    # Map pip package name → actual import name when they differ.
    IMPORT_NAME = {
        "python-telegram-bot": "telegram",
        "scikit-learn": "sklearn",
        "sentence-transformers": "sentence_transformers",
        "google-genai": "google.genai",
        "stable-baselines3": "stable_baselines3",
    }
    for pkg in REQUIRED_PACKAGES:
        optional = False
        if isinstance(pkg, tuple):
            pkg, optional = pkg
        import_name = IMPORT_NAME.get(pkg, pkg.replace("-", "_"))
        try:
            importlib.import_module(import_name)
        except ImportError:
            if optional:
                r.warn(f"{pkg} not installed (optional — degraded mode)")
            else:
                r.error(f"{pkg} not installed  →  pip install {pkg}")
        else:
            r.info.append(f"{pkg} OK")
    ALL_RESULTS.append(r)
    return r


# ── 4. Config files + writable directories ──────────────────────────
def check_paths() -> CheckResult:
    r = CheckResult("Config files + writable directories", "paths")
    # Config files
    for f in (".env", "config.py", "main.py", "requirements.txt"):
        p = PROJECT_ROOT / f
        if not p.exists():
            r.error(f"Missing file: {f}")
        elif p.stat().st_size == 0 and f == ".env":
            r.warn(f".env is empty — copy .env.example and fill in keys")

    # Directories that must be writable
    for d in ("logs", "memory", "database", "backups", "reports"):
        p = PROJECT_ROOT / d
        try:
            p.mkdir(parents=True, exist_ok=True)
            test = p / ".write_test"
            test.write_text("ok")
            test.unlink()
        except Exception as e:
            r.error(f"Directory not writable: {d}/ ({e})")

    ALL_RESULTS.append(r)
    return r


# ── 5. Boot phases dry-run (heavy) ──────────────────────────────────
def check_boot() -> CheckResult:
    r = CheckResult("24-phase boot dry-run", "boot")
    try:
        from core.runtime import get_runtime, register_default_phases
        from core.lifecycle import Phase
        from core.service_registry import get_registry

        rt = get_runtime()
        register_default_phases(rt.lifecycle)

        # Boot all phases up to and including RUNTIME (the trader itself)
        rt.lifecycle.boot(until=Phase.RUNTIME)
        report = rt.lifecycle.report()

        ok_phases = sum(1 for p in report.get("phases", []) if p.get("ok"))
        failed_phases = [p for p in report.get("phases", []) if not p.get("ok")]
        skipped_phases = [p for p in report.get("phases", []) if p.get("skipped")]
        r.info.append(
            f"Boot complete: {ok_phases} OK, "
            f"{len(failed_phases)} failed, {len(skipped_phases)} skipped"
        )

        for p in failed_phases:
            r.error(f"Phase {p.get('phase')}: {p.get('error', 'unknown error')}")

        # Check critical services registered
        reg = get_registry()
        critical_services = [
            # Originals
            "config", "db", "trade_memory", "data_fetcher",
            "market_scanner", "circuit_breaker", "risk_engine_factory",
            "execution_router", "paper_trader", "trader",
            # Day 37+ runtime unification (main.py resolves these after boot)
            "trading_engine", "telegram_notifier", "correlation_filter",
            # Day 75
            "kill_switch", "live_risk_manager", "exposure_manager",
            "drawdown_monitor", "risk_reporter",
            # Day 76
            "position_sizer", "kelly_calculator", "volatility_adjuster",
            "confidence_scaler", "correlation_manager",
            # Day 81+ (AITrader pulls from registry in __init__)
            "mistake_analyzer",
        ]
        missing = [s for s in critical_services if not reg.try_resolve(s)]
        if missing:
            r.error(f"Missing services: {', '.join(missing)}")
        else:
            r.info.append(f"All {len(critical_services)} critical services registered")

    except Exception as e:
        r.error(f"Boot crashed: {type(e).__name__}: {e}")
        r.errors.append(traceback.format_exc().splitlines()[-1])
    ALL_RESULTS.append(r)
    return r


# ── 6. Day 76 risk modules functional test ──────────────────────────
def check_day76_functional() -> CheckResult:
    r = CheckResult("Day 76 PositionSizer functional test", "day76")
    try:
        from risk.position_sizer import get_position_sizer, PositionSizeResult, AdvancedPositionResult
        # Verify alias
        if PositionSizeResult is not AdvancedPositionResult:
            r.error("PositionSizeResult alias broken — live_risk_manager will fail to import")

        sizer = get_position_sizer()

        # Happy path
        result = sizer.calculate(
            balance=10000.0, risk_pct=0.01, sl_pips=20.0,
            pip_value_per_lot=10.0, confidence=78.0,
            atr=0.0010, atr_median=0.0010,
            consecutive_losses=0, tier_mult=1.0,
            win_rate=0.55, avg_win_r=1.5, avg_loss_r=1.0, trade_count=50,
            pair="EURUSD", direction="BUY", open_positions=[],
            current_drawdown_pct=0.02, is_new_equity_high=False, news_active=False,
        )
        if not result.approved:
            r.error(f"Happy-path sizing rejected: {result.reject_reason}")
        elif result.lot <= 0:
            r.error(f"Happy-path lot is 0: {result.to_dict()}")
        else:
            r.info.append(
                f"Happy path: base={result.base_lot:.2f} → final={result.lot:.2f} "
                f"(×{result.final_mult:.3f}), risk=${result.risk_amount_usd:.0f}"
            )

        # Reject path (low confidence)
        result2 = sizer.calculate(
            balance=10000.0, risk_pct=0.01, sl_pips=20.0,
            pip_value_per_lot=10.0, confidence=40.0,  # below 55% floor
            atr=0.0010, atr_median=0.0010,
            consecutive_losses=0, tier_mult=1.0,
            trade_count=0,
            pair="EURUSD", direction="BUY", open_positions=[],
        )
        if result2.approved:
            r.error("Reject path did not block low-confidence trade")
        else:
            r.info.append(f"Reject path OK: '{result2.reject_reason[:60]}'")

        # Portfolio heat check
        result3 = sizer.calculate(
            balance=10000.0, risk_pct=0.01, sl_pips=20.0,
            pip_value_per_lot=10.0, confidence=80.0,
            atr=0.0010, atr_median=0.0010,
            consecutive_losses=0, tier_mult=1.0, trade_count=50,
            win_rate=0.55, avg_win_r=1.5, avg_loss_r=1.0,
            pair="EURUSD", direction="BUY",
            # 3 correlated same-direction positions → must block
            open_positions=[
                {"pair": "GBPUSD", "direction": "BUY", "risk_usd": 100},
                {"pair": "AUDUSD", "direction": "BUY", "risk_usd": 100},
                {"pair": "NZDUSD", "direction": "BUY", "risk_usd": 100},
            ],
        )
        if result3.approved:
            r.warn("3 correlated positions did NOT block — check correlation_manager thresholds")
        else:
            r.info.append(f"Correlation block OK: '{result3.reject_reason[:60]}'")

    except Exception as e:
        r.error(f"Day 76 functional test crashed: {type(e).__name__}: {e}")
        r.errors.append(traceback.format_exc().splitlines()[-1])
    ALL_RESULTS.append(r)
    return r


# ── 7. AITrader construction test ───────────────────────────────────
def check_aitrader() -> CheckResult:
    r = CheckResult("AITrader construction with registry", "trader")
    try:
        from core.runtime import get_runtime, register_default_phases
        from core.lifecycle import Phase
        from core.service_registry import get_registry

        rt = get_runtime()
        register_default_phases(rt.lifecycle)
        rt.lifecycle.boot(until=Phase.RISK)

        from core.trader import AITrader
        trader = AITrader(
            balance=10000.0, symbol="EURUSD", timeframe="15m",
            registry=get_registry(),
        )
        if trader._position_sizer is None:
            r.error("AITrader did not pull position_sizer from registry")
        else:
            r.info.append("Day 76 sizer wired into AITrader")
        if trader._live_risk_manager is None:
            r.warn("AITrader did not pull live_risk_manager from registry")
        if trader._drawdown_monitor is None:
            r.warn("AITrader did not pull drawdown_monitor from registry")
        # Day 75 kill_switch pull (used for consecutive_losses in _apply_advanced_sizing)
        if trader._kill_switch is None:
            r.warn("AITrader did not pull kill_switch from registry — loss-streak penalty disabled")
        else:
            r.info.append("kill_switch wired into AITrader")
        # Day 81+ mistake_analyzer pull
        if trader._mistake_analyzer is None:
            r.warn("AITrader mistake_analyzer is None — closed-trade learning disabled")
        else:
            r.info.append("mistake_analyzer wired into AITrader")
        # Verify circuit_breaker + approval shared (not per-AITrader)
        if trader._circuit_breaker is None:
            r.error("AITrader _circuit_breaker is None — circuit breaker gate will crash in run_cycle")
        if trader._approval is None:
            r.error("AITrader _approval is None — approval gate will crash in run_cycle")
        # Verify Day 37 hotfix: _corr_filter pulled from registry
        if trader._corr_filter is None:
            r.error("AITrader _corr_filter is None — correlation check will crash in run_cycle")

        # Test the _apply_advanced_sizing helper
        fake_risk = {
            "approved": True, "signal": "BUY", "symbol": "EURUSD",
            "entry": 1.0850, "sl_price": 1.0830, "tp_price": 1.0880,
            "sl_pips": 20, "tp_pips": 30, "lot": 0.5,
            "risk_usd": 100.0, "risk_pc": 1.0, "rr_ratio": 1.5,
            "daily_loss_pc": 0.0, "open_trades": 0, "reject_reason": None,
        }
        fake_dec = {"decision": "BUY", "confidence": 82}
        fake_market = {
            "ind_ctx": {"close": 1.0850, "atr": 0.0010, "rsi": 55, "trend": "BULL"},
            "regime": {"regime": "TREND", "volatility": "NORMAL", "atr_median": 0.0010},
        }
        fake_analysis = {"news_ctx": {"trade_allowed": True}}

        result = trader._apply_advanced_sizing(fake_risk, fake_dec, fake_market, fake_analysis)
        if "position_sizing" not in result:
            r.error("_apply_advanced_sizing did not store breakdown")
        elif not result.get("approved"):
            r.warn(f"Sizing rejected happy path: {result.get('reject_reason')}")
        else:
            r.info.append(
                f"_apply_advanced_sizing OK: lot={result['lot']:.2f} "
                f"mult=×{result['position_sizing']['final_mult']:.3f}"
            )
    except Exception as e:
        r.error(f"AITrader test crashed: {type(e).__name__}: {e}")
        r.errors.append(traceback.format_exc().splitlines()[-1])
    ALL_RESULTS.append(r)
    return r


# ── 8. Stray files / common gotchas ─────────────────────────────────
def check_strays() -> CheckResult:
    r = CheckResult("Stray files + common gotchas", "strays")
    # Stale .pyc files in source dirs
    pyc_files = [
        f for f in PROJECT_ROOT.rglob("*.pyc")
        if not any(part in {"__pycache__", ".git", ".venv"} for part in f.parts)
    ]
    if pyc_files:
        r.warn(f"{len(pyc_files)} stray .pyc files outside __pycache__ (safe to delete)")

    # Backup copies of source dirs
    for d in PROJECT_ROOT.iterdir():
        if d.is_dir() and " - Copy" in d.name:
            r.warn(f"Backup directory found: {d.name} (safe to remove)")

    # CRLF line endings in .py files (causes git churn)
    crlf_files = []
    for f in PROJECT_ROOT.rglob("*.py"):
        if any(part in {"__pycache__", ".git", ".venv"} for part in f.parts):
            continue
        try:
            with open(f, "rb") as fh:
                head = fh.read(4096)
            if b"\r\n" in head:
                crlf_files.append(str(f.relative_to(PROJECT_ROOT)))
        except Exception:
            pass
        if len(crlf_files) >= 5:
            break
    if crlf_files:
        r.warn(
            f"{len(crlf_files)}+ .py files use CRLF line endings "
            f"(e.g. {crlf_files[0]}) — run: dos2unix or sed -i 's/\\r$//'"
        )
    ALL_RESULTS.append(r)
    return r


# ── 9. Config flag safety check ──────────────────────────────────────
def check_config_flags() -> CheckResult:
    """Verify .env / config.py production-safety flags.

    main.py prints a banner when TEST_MODE=true and SIMULATION_MODE=true —
    these are intentional dev-only bypasses that must never slip into
    production. This check surfaces them as warnings so they're visible
    in the debug scan output even when you don't read the boot banner.
    """
    r = CheckResult("Config flags (.env production-safety)", "config_flags")
    try:
        from config import (
            TEST_MODE, SIMULATION_MODE, EXECUTION_MODE,
            MAX_LOT, APPROVAL_MODE, MAX_OPEN_TRADES,
        )
        # TEST_MODE bypasses MasterAnalyst, Confluence, Ensemble, RL, TradePermission
        # confidence floor and PositionSizer rejects.  Never leave on in prod.
        if TEST_MODE:
            r.warn(
                "TEST_MODE=True — MasterAnalyst/Confluence/RL bypassed, "
                "TradePermission MIN_CONFIDENCE=10, PositionSizer force-approved. "
                "Set TEST_MODE=false in .env before live trading."
            )
        else:
            r.info.append("TEST_MODE=False (production-safe)")

        if SIMULATION_MODE:
            r.warn(
                "SIMULATION_MODE=True — no real MT5 orders will be placed. "
                "Set SIMULATION_MODE=false in .env for live trading."
            )
        else:
            r.info.append("SIMULATION_MODE=False (live orders enabled)")

        if MAX_LOT > 1.0:
            r.warn(f"MAX_LOT={MAX_LOT} — unusually large; double-check .env")
        else:
            r.info.append(f"MAX_LOT={MAX_LOT}")

        if APPROVAL_MODE not in (1, 2, 3):
            r.error(f"APPROVAL_MODE={APPROVAL_MODE} is invalid — must be 1, 2, or 3")
        else:
            r.info.append(f"APPROVAL_MODE={APPROVAL_MODE}")

        if MAX_OPEN_TRADES < 1 or MAX_OPEN_TRADES > 20:
            r.warn(f"MAX_OPEN_TRADES={MAX_OPEN_TRADES} — outside typical range 1-20")
        else:
            r.info.append(f"MAX_OPEN_TRADES={MAX_OPEN_TRADES}")

        r.info.append(f"EXECUTION_MODE={EXECUTION_MODE}")

    except ImportError as e:
        r.error(f"config.py missing required flag: {e}")
    except Exception as e:
        r.error(f"Config flag check crashed: {type(e).__name__}: {e}")
    ALL_RESULTS.append(r)
    return r


# ── 10. Network connectivity (LLM + Telegram reachability) ──────────
def check_network() -> CheckResult:
    """Check that we can reach Groq, Gemini, Telegram, yfinance etc.
    Without network access, LLM keys get disabled and Telegram alerts
    fail silently.  This is the #1 cause of "Confidence 0%" issues.
    """
    r = CheckResult("Network reachability (LLM + Telegram)", "network")

    # Try to import the check_network module we ship alongside.
    check_network_path = Path(__file__).resolve().parent / "check_network.py"
    if not check_network_path.exists():
        r.warn("check_network.py not found — skipping network test")
        ALL_RESULTS.append(r)
        return r

    # Run the network checks inline.
    import socket
    import ssl
    import urllib.request

    services = [
        # (label, host, port, path, expect_statuses, is_critical)
        ("Telegram API", "api.telegram.org", 443, "/bot000:getMe", (401, 403, 404), True),
        ("Groq LLM",     "api.groq.com", 443, "/openai/v1/models", (401, 403, 404), True),
        ("Gemini LLM",   "generativelanguage.googleapis.com", 443, "/v1beta/models", (401, 403, 404), True),
        ("yfinance",     "query1.finance.yahoo.com", 443, "/v8/finance/chart/AAPL", (200,), False),
        ("Forex Factory","www.forexfactory.com", 443, "/calendar", (200, 403), False),
    ]

    for label, host, port, path, expect, critical in services:
        # 1. DNS
        try:
            socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)
        except socket.gaierror:
            r.error(f"{label}: DNS lookup failed for {host} — service unreachable")
            continue
        # 2. TCP
        try:
            with socket.create_connection((host, port), timeout=5.0):
                pass
        except (socket.timeout, OSError) as e:
            msg = f"{label}: TCP connect to {host}:{port} failed — {type(e).__name__}"
            if critical:
                r.error(msg)
            else:
                r.warn(msg)
            continue
        # 3. HTTPS request
        url = f"https://{host}{path}"
        try:
            ctx = ssl.create_default_context()
            req = urllib.request.Request(url, headers={"User-Agent": "forex_ai/debug"})
            with urllib.request.urlopen(req, timeout=8.0, context=ctx) as resp:
                status = resp.status
        except urllib.error.HTTPError as e:
            # HTTP error = connection worked, server rejected request. OK.
            status = e.code
        except Exception as e:
            msg = f"{label}: HTTPS request failed — {type(e).__name__}: {str(e)[:60]}"
            if critical:
                r.error(msg)
            else:
                r.warn(msg)
            continue

        if expect and status not in expect:
            r.warn(f"{label}: got HTTP {status} (expected {expect})")
        else:
            r.info.append(f"{label}: reachable (HTTP {status})")

    if r.ok and not r.errors:
        r.info.append("All critical services reachable")
    ALL_RESULTS.append(r)
    return r


# ── Main ────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description="forex_ai project error scanner")
    parser.add_argument("--quick", action="store_true",
                        help="Skip heavy boot + AITrader tests")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON instead of human-readable text")
    parser.add_argument("--verbose", action="store_true",
                        help="Show info lines even on OK checks")
    parser.add_argument("--skip-trace", action="store_true",
                        help="Skip the detailed trade pipeline trace at the end")
    parser.add_argument("--trace-symbols", type=str, default="EURUSD",
                        help="Comma-separated symbols for the trace (default: EURUSD)")
    parser.add_argument("--timeframe", type=str, default="15m",
                        help="Timeframe for the trace (default: 15m)")
    args = parser.parse_args()

    print()
    print(_c("=" * 70, "bold"))
    print(_c("  forex_ai — Whole-Project Debug Scan", "bold"))
    print(_c("=" * 70, "bold"))
    print(f"  Project : {PROJECT_ROOT}")
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  Time    : {datetime.now().isoformat(timespec='seconds')}")
    print(f"  Mode    : {'quick' if args.quick else 'full'}")
    print()

    # Run all checks
    print(_c("[1/10] Syntax check — every .py file", "cyan"))
    _print_result(check_syntax(), args.verbose)
    print()

    print(_c("[2/10] Critical imports", "cyan"))
    _print_result(check_imports(), args.verbose)
    print()

    print(_c("[3/10] Pip dependencies", "cyan"))
    _print_result(check_dependencies(), args.verbose)
    print()

    print(_c("[4/10] Config + writable paths", "cyan"))
    _print_result(check_paths(), args.verbose)
    print()

    print(_c("[5/10] Config flags (.env production-safety)", "cyan"))
    _print_result(check_config_flags(), args.verbose)
    print()

    print(_c("[6/10] Stray files + gotchas", "cyan"))
    _print_result(check_strays(), args.verbose)
    print()

    print(_c("[7/10] Network reachability (LLM + Telegram)", "cyan"))
    _print_result(check_network(), args.verbose)
    print()

    if not args.quick:
        print(_c("[8/10] 24-phase boot dry-run (heavy)", "cyan"))
        _print_result(check_boot(), args.verbose)
        print()

        print(_c("[9/10] Day 76 PositionSizer functional test", "cyan"))
        _print_result(check_day76_functional(), args.verbose)
        print()

        print(_c("[10/10] AITrader construction with registry", "cyan"))
        _print_result(check_aitrader(), args.verbose)
        print()
    else:
        print(_c("[8-10] Skipped (--quick)", "gray"))
        print()

    # Summary
    total = len(ALL_RESULTS)
    passed = sum(1 for r in ALL_RESULTS if r.ok and not r.warnings)
    passed_with_warn = sum(1 for r in ALL_RESULTS if r.ok and r.warnings)
    failed = sum(1 for r in ALL_RESULTS if not r.ok)
    total_errors = sum(len(r.errors) for r in ALL_RESULTS)
    total_warns = sum(len(r.warnings) for r in ALL_RESULTS)

    print(_c("=" * 70, "bold"))
    print(_c("  SUMMARY", "bold"))
    print(_c("=" * 70, "bold"))
    print(f"  Checks    : {total} total  ({passed} passed, {passed_with_warn} with warnings, {failed} failed)")
    print(f"  Errors    : {total_errors}")
    print(f"  Warnings  : {total_warns}")
    print()

    if total_errors == 0:
        print(_c("  ✓ All checks passed. Safe to run: python main.py", "green"))
    else:
        print(_c("  ✗ Errors found. Fix them BEFORE running main.py:", "red"))
        for r in ALL_RESULTS:
            for e in r.errors:
                print(f"      [{r.category}] {e}")
    print()

    # ── Day 81+ — Detailed trade condition trace ───────────────────
    # This is the most important check for "why isn't my bot trading?".
    # It runs one real run_cycle() per symbol and prints EACH stage's
    # verdict (OK/WAIT/REJECT/BLOCK) so you can see EXACTLY which
    # condition killed the trade.
    if not args.quick and not args.skip_trace:
        print(_c("=" * 70, "bold"))
        print(_c("  DETAILED TRADE CONDITION TRACE", "bold"))
        print(_c("  (per-symbol stage-by-stage trace — see EXACTLY where signals die)", "cyan"))
        print(_c("=" * 70, "bold"))
        try:
            from monitoring.trade_pipeline_tracer import trace_symbol, print_trace
            trace_syms = args.trace_symbols.split(",") if args.trace_symbols else ["EURUSD"]
            trace_syms = [s.strip().upper() for s in trace_syms if s.strip()]
            for sym in trace_syms:
                print_trace(trace_symbol(sym, args.timeframe))
        except Exception as e:
            print(_c(f"  Trace failed: {type(e).__name__}: {e}", "red"))
        print()


    if args.json:
        print(_c("--- JSON output ---", "gray"))
        print(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "project": str(PROJECT_ROOT),
            "python": sys.version.split()[0],
            "summary": {
                "total": total,
                "passed": passed,
                "passed_with_warnings": passed_with_warn,
                "failed": failed,
                "errors": total_errors,
                "warnings": total_warns,
            },
            "results": [r.to_dict() for r in ALL_RESULTS],
        }, indent=2))

    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())