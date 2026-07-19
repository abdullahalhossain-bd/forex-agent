#!/usr/bin/env python3
"""
=====================================================
FOREX AI AUTONOMOUS TRADING SYSTEM
=====================================================
Main Entry Point — Central Controller (Day 37+ Runtime-Unified version)

This version of main.py replaces the ad-hoc ForexAISystem class with a
composition root that drives the new LifecycleManager + ServiceRegistry
infrastructure in core/runtime.py. Every runtime module is now brought up
in a strict phase order through a single boot path, and torn down in
reverse on shutdown.

Pipeline (single source of truth — see core/runtime.py:register_default_phases):

    System Bootstrap (config, paths, event bus, metrics, health monitor)
    → Persistence (DB, TradeMemory, LearningEngine, KnowledgeStore)
    → Data (Fetcher, Validator, Indicators, AutomatedUpdater)
    → Market (Scanner, CorrelationFilter, OpportunityRanker, MT5Connection)
    → Research (ResearchAgent, HypothesisEngine, ExperimentRunner, Reports)
    → Fundamental (NewsFilter, FundamentalSentimentScore)
    → Analysis (IntermarketEngine, SessionAnalyzer)
    → AI (AIAnalyst, MasterAnalyst, ModelVersionManager)
    → Agents (Market/Analysis/Decision/Learning/Risk agent classes)
    → Strategy (SignalEngine, strategies package)
    → Hybrid (FlowController — constructed, not actively driven)
    → Risk (RiskEngine, CircuitBreaker, TradePermission, Drawdown, AutonomousRisk)
    → Safety (SafetyGuard, SpreadMonitor)
    → Execution (PaperTrader, ExecutionRouter)
    → Broker (AccountManager, OrderManager, JournalBridge, EconomicCalendar)
    → Analytics (PerformanceAnalyzer, StrategyTracker, RankingEngine, PerformanceReport)
    → Reports (BacktestReport)
    → Learning (ConfidenceEngine, AutoOptimizer, LessonMemory, MemoryIntegration, MistakeAnalyzer)
    → Dashboard (Streamlit path + bus subscriptions)
    → Alerts (TelegramNotifier + bus subscribers for risk/broker/error events)
    → Automation (ErrorHandler, DailyReview, SystemHealth legacy)
    → Webhook (SignalPipeline, Flask app)
    → Orchestrator (TradingOrchestrator, DailyRoutine, Scheduler, AuditTrail, HumanOverride, MessageBus, SystemState)
    → Runtime (AutonomousTraderSystem / TradingEngine — the trader itself)

Usage:
    python main.py                      # Start autonomous trading (full boot)
    python main.py --mode init          # Initialize + verify, don't start loop
    python main.py --mode status        # Show system status (boot, then print)
    python main.py --mode backtest      # Run backtest
    python main.py --mode health        # Boot + print health snapshot
    python main.py --mode obsolete      # Print obsolete-module registry
    python main.py --pairs EURUSD,GBPUSD  # Override pairs
    python main.py --timeframe 1h       # Override timeframe
    python main.py --paper              # Force paper mode
    python main.py --no-telegram        # Disable Telegram
=====================================================
"""

import argparse
import json
import logging
import sys
import os
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure UTF-8 stdout/stderr (Windows console quirks)
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure HF cache directory is set to the repository cache to avoid
# runtime network HEAD/timeouts when fetching small files from the Hub.
# If the user or system already set HF_HOME, respect that.
if "HF_HOME" not in os.environ:
    _hf_cache_path = PROJECT_ROOT / "data" / "hf_cache"
    try:
        _hf_cache_path.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    os.environ["HF_HOME"] = str(_hf_cache_path)
    print(f"HF_HOME not set — using repo cache: {os.environ['HF_HOME']}")

from config import (
    EXECUTION_MODE,
    INITIAL_BALANCE,
    ENABLE_TELEGRAM, SYMBOLS, DEFAULT_TIMEFRAME,
)
from core.constants import clean_symbol, LOGS_DIR
from core.lifecycle import Phase
from core.runtime import boot_runtime, get_runtime


# ──────────────────────────────────────────────────────────────
# SYSTEM BANNER
# ──────────────────────────────────────────────────────────────

BANNER = r"""
=================================
  ____  _____ ___ _   _ ____
 |  _ \| ____|_ _| \ | |  _ \
 | | | |  _|  | ||  \| | | | |
 | |_| | |___ | || |\  | |_| |
 |____/|_____|___|_| \_|____/
                     _    _ _____
                    / \  | | ___|
                   / _ \ | |___ \
                  / ___ \| |___) |
                 /_/   \_\_|____/

  AUTONOMOUS TRADING SYSTEM
  Day 37+ Runtime-Unified
=================================
"""


# ──────────────────────────────────────────────────────────────
# LOGGING SETUP
# ──────────────────────────────────────────────────────────────

def setup_logging():
    """Configure comprehensive logging."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_level = logging.INFO

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(LOGS_DIR / "forex_ai.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    # Reduce verbosity of noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


# ──────────────────────────────────────────────────────────────
# MAIN SYSTEM CLASS (composition root)
# ──────────────────────────────────────────────────────────────

class ForexAISystem:
    """
    Central controller for the FOREX AI Autonomous Trading System.

    Day 37+ runtime-unified version: this class is now a thin wrapper around
    core.runtime.Runtime, which owns the ServiceRegistry + LifecycleManager +
    EventBus + HealthMonitor + RuntimeMetrics. The previous version wired
    ~9 of the ~24 required services inline; this version wires ALL of them
    through a single phase-ordered boot path.
    """

    def __init__(self, args=None):
        self.args = args or argparse.Namespace()
        self.runtime = get_runtime()
        self.status = SystemStatus()
        self.running = False
        self.start_time = None
        self._stop_requested = False

        # Resolve execution mode (MT5 demo only - paper trading removed)
        self.execution_mode = EXECUTION_MODE  # Always mt5_demo
        self.enable_telegram = ENABLE_TELEGRAM and not getattr(
            self.args, "no_telegram", False
        )
        self.symbols = self._resolve_symbols()
        self.timeframe = getattr(self.args, "timeframe", None) or DEFAULT_TIMEFRAME
        self.balance = getattr(self.args, "balance", None) or INITIAL_BALANCE
        self.max_cycles = getattr(self.args, "max_cycles", None)

        # The trader is constructed by the runtime's RUNTIME phase and
        # registered in the ServiceRegistry under "trader" / "trading_engine".
        # We resolve it after boot.
        self.trading_engine = None

    def _resolve_symbols(self) -> list[str]:
        pairs_arg = getattr(self.args, "pairs", None)
        if pairs_arg:
            return [
                clean_symbol(p.strip())
                for p in pairs_arg.split(",")
                if p.strip()  # drop empty tokens from trailing/double commas
            ]
        return [clean_symbol(s) for s in SYMBOLS]

    # ─────────────────────────────────────────────
    # INITIALIZATION
    # ─────────────────────────────────────────────

    def initialize(self) -> bool:
        """Boot the entire runtime through the LifecycleManager. Returns True
        if every critical phase succeeded."""
        print(BANNER)
        print("  Booting runtime (24 phases)...\n")

        # Day 81+ hotfix: TEST_MODE warning banner.
        # TEST_MODE bypasses MasterAnalyst, Confluence, Ensemble, RL, and
        # MasterDecision gates.  It also lowers TradePermission confidence
        # threshold to 10% and force-approves PositionSizer rejects with
        # lot=0.01.  This is fine for first-time MT5 verification, but
        # should NOT stay on in production.
        try:
            from config import TEST_MODE, SIMULATION_MODE, MAX_LOT, APPROVAL_MODE, MAX_OPEN_TRADES
            if TEST_MODE:
                print("=" * 60)
                print("  ⚠️  TEST_MODE = True  ⚠️")
                print("  All safety gates are PERMISSIVE:")
                print("    • MasterAnalyst/Confluence/Ensemble/RL bypassed")
                print("    • TradePermission MIN_CONFIDENCE = 10 (prod=60)")
                print("    • PositionSizer rejects force-approved lot=0.01")
                print("    • Session quality check disabled")
                print("  Set TEST_MODE=false in .env for production trading.")
                print("=" * 60)
                print()
            if SIMULATION_MODE:
                print("=" * 60)
                print("  🔬  SIMULATION_MODE = True  🔬")
                print("  No real MT5 orders will be placed.")
                print("  Set SIMULATION_MODE=false in .env for live trading.")
                print("=" * 60)
                print()
            print(f"  Config: MAX_LOT={MAX_LOT} | APPROVAL_MODE={APPROVAL_MODE} | MAX_OPEN_TRADES={MAX_OPEN_TRADES}")
            print()
        except Exception as e:
            # This banner reports whether permissive test gates (10% confidence
            # threshold, force-approved 0.01 lot) are active. Swallowing the
            # failure here would mean an operator starts the system with no
            # indication of whether TEST_MODE is on — that's a safety signal,
            # not cosmetic output, so surface it loudly instead of hiding it.
            print(f"  ⚠️  Could not display TEST_MODE/SIMULATION_MODE banner: {e}")
            logging.error(f"[System] Safety-mode banner failed to render: {e}", exc_info=True)

        # Override config-driven settings if CLI args were supplied.
        self._apply_cli_overrides()

        # Register a phase-complete callback so we get a live progress print.
        def _on_phase(result):
            icon = "OK" if result.ok else "!!"
            if result.skipped:
                icon = "--"
            svcs = ", ".join(result.services_registered) if result.services_registered else "(no services)"
            err = f"  ERR: {result.error}" if result.error else ""
            print(f"  [{icon}] {result.phase.value:<14} ({result.duration_sec}s) — {svcs}{err}")

        self.runtime.lifecycle.on_phase_complete(_on_phase)

        # Boot every phase in order.
        self.runtime.boot()

        # Resolve the trading engine from the registry.
        self.trading_engine = self.runtime.registry.try_resolve("trading_engine")
        if self.trading_engine is None:
            # Fallback: try the trader directly.
            self.trading_engine = self.runtime.registry.try_resolve("trader")

        # Day 81+ hotfix: auto-reconcile DB trades with MT5 live positions.
        # Closes orphan DB-OPEN trades that were closed externally (SL/TP hit,
        # manual close, restart) but never marked CLOSED in DB.  Without this,
        # stale open_pairs blocks new trades on correlated pairs forever.
        try:
            from core.orphan_cleanup import reconcile_open_positions
            mt5_conn = None
            try:
                router = self.runtime.registry.try_resolve("execution_router")
                if router and hasattr(router, "_mt5_conn"):
                    mt5_conn = router._mt5_conn
            except Exception:
                pass
            paper_trader = None
            try:
                paper_trader = self.runtime.registry.try_resolve("paper_trader")
            except Exception:
                pass
            reconciled = reconcile_open_positions(
                db=None, mt5_conn=mt5_conn, paper_trader=paper_trader,
            )
            if reconciled["closed"] > 0:
                print(f"  🧹  Orphan cleanup: {reconciled['closed']} stale DB-OPEN trades auto-closed")
                logging.info(f"[System] Orphan cleanup: {reconciled}")
            elif reconciled["kept"] > 0:
                print(f"  ✓  {reconciled['kept']} DB-OPEN trades verified against MT5 (all real)")
        except Exception as e:
            logging.warning(f"[System] Orphan cleanup skipped: {e}")

        print()
        self._print_boot_summary()

        # Critical phases that must succeed before trading can start.
        # RISK/SAFETY/EXECUTION added alongside the runtime.py fix that made
        # boot_risk/boot_safety/boot_execution/boot_broker actually report
        # ok=False on failure instead of always claiming success — without
        # checking them here too, that fix would have no effect: initialize()
        # would still return True and start_trading() would still run with
        # no risk engine, no safety guard, or no order path wired.
        for critical in (Phase.BOOTSTRAP, Phase.PERSISTENCE, Phase.RISK,
                         Phase.SAFETY, Phase.EXECUTION, Phase.BROKER):
            r = self.runtime.lifecycle.last_result(critical)
            if r is None or not r.ok:
                return False
        return True

    def _apply_cli_overrides(self) -> None:
        """If CLI args override config values, push them into the registry."""
        # CLI pairs override
        if hasattr(self.args, "pairs") and self.args.pairs:
            self.runtime.registry.register_instance("symbols", self.symbols)
        if self.execution_mode != EXECUTION_MODE:
            # Re-register execution_mode so boot_runtime_phase picks up the
            # CLI override instead of the .env value.
            self.runtime.registry.register_instance("execution_mode", self.execution_mode)

    def _print_boot_summary(self) -> None:
        """Print final boot summary."""
        report = self.runtime.lifecycle.report()
        phases = report["phases"]
        ok = sum(1 for p in phases if p["ok"] and not p["skipped"])
        failed = sum(1 for p in phases if not p["ok"])
        skipped = sum(1 for p in phases if p["skipped"])
        print("=" * 60)
        print(f"  Boot complete: {ok} phases OK, {failed} failed, {skipped} skipped")
        if failed:
            print(f"  FAILED phases:")
            for p in phases:
                if not p["ok"]:
                    print(f"    - {p['phase']}: {p.get('error', 'unknown')}")
        print(f"  Trader wired: {'yes' if self.trading_engine else 'NO'}")
        print(f"  Registry services: {len(self.runtime.registry.health())}")
        print("=" * 60)

    # ─────────────────────────────────────────────
    # MAIN TRADING LOOP
    # ─────────────────────────────────────────────

    def start_trading(self):
        """Start the autonomous trading loop.

        Day 37+ fix: This method now wraps the trader's run() in an
        auto-restart loop. If the trader exits for ANY reason (crash,
        unexpected exception, or graceful return), main.py waits 10
        seconds and relaunches it. The agent NEVER turns off unless the
        user presses Ctrl+C twice or sends /stop via Telegram.
        """
        if not self.trading_engine:
            logging.error("Trading engine not initialized — cannot start")
            return

        self.running = True
        self.start_time = datetime.now(timezone.utc)
        # Day 102+ hotfix: disambiguate the three different balances that
        # appear in startup logs. Previously the line just said
        # "Balance: $10000" which collided with the MT5 demo account
        # balance ($99147) and the paper trader balance ($9516),
        # confusing operators about which pool of money was at risk.
        #   - Risk Balance  : config INITIAL_BALANCE (used by risk math)
        #   - Paper Balance : PaperTrader's running simulated balance
        #   - MT5 Balance   : live account equity at the broker
        _mt5_balance_str = ""
        try:
            from utils.registry import ServiceRegistry
            _mt5_conn = ServiceRegistry.try_resolve("mt5_connection")
            if _mt5_conn is not None:
                _acct = _mt5_conn.account_info()
                if _acct is not None:
                    _mt5_balance_str = f" | MT5 Balance: ${getattr(_acct, 'balance', '?')}"
        except Exception as e:
            # Not fatal (paper/demo runs may not have a live MT5 connection
            # yet), but a broken MT5 lookup at startup is worth a debug trail
            # rather than disappearing entirely.
            logging.debug(f"[System] Could not read MT5 account balance at startup: {e}")
        logging.info(
            f"[System] Trading started | Mode: {self.execution_mode.upper()} | "
            f"Pairs: {self.symbols} | Risk Balance: ${self.balance} (config)"
            f"{_mt5_balance_str} | "
            f"Auto-restart: ON"
        )

        # KILL SHOT FIX: Wire MetricsExporter — start Prometheus exporter
        try:
            from monitoring.metrics_exporter import get_metrics_exporter
            self._metrics_exporter = get_metrics_exporter()
            self._metrics_exporter.start()
            logging.info("[System] Prometheus metrics exporter started on :9090")
        except Exception as e:
            logging.warning(f"[System] Metrics exporter failed (non-critical): {e}")
            self._metrics_exporter = None

        # Send startup notification
        notifier = self.runtime.registry.try_resolve("telegram_notifier")
        if notifier:
            self._notify_startup(notifier)

        # ANNIHILATION FIX: Install SIGINT/SIGTERM handlers for graceful shutdown.
        # Previously, Ctrl+C would kill the process immediately — potentially
        # leaving open positions unmanaged and state files half-written.
        import signal as _signal
        def _on_signal(signum, frame):
            logging.warning(f"[System] Signal {signum} received — requesting graceful stop")
            self._stop_requested = True
            if self.trading_engine:
                self.trading_engine.stop()
        try:
            _signal.signal(_signal.SIGINT, _on_signal)
            _signal.signal(_signal.SIGTERM, _on_signal)
            logging.info("[System] Signal handlers installed (SIGINT, SIGTERM)")
        except (ValueError, OSError) as e:
            logging.warning(f"[System] Could not install signal handlers: {e}")

        # Issue 7 fix: bounded exponential-backoff restart policy.
        #
        # The previous version restarted on ANY exit — forever, at a fixed
        # 10s/30s delay — with no upper bound on restart_count. A permanent
        # configuration failure (bad MT5 credentials, missing module, etc.)
        # would spin the process indefinitely, hammering the broker/API and
        # spamming Telegram, never surfacing that it was NOT a transient
        # problem.
        #
        # New behaviour:
        #   - delay doubles each consecutive restart: 10s, 20s, 40s, ... up
        #     to a hard cap of MAX_RESTART_DELAY_SEC (5 min).
        #   - a restart that survives at least MIN_STABLE_RUN_SEC resets the
        #     backoff back to the base delay (the failure was transient).
        #   - after MAX_RESTARTS consecutive failures without a stable run,
        #     the process stops restarting, notifies the operator, and
        #     exits non-zero so a process supervisor / on-call alert can
        #     take over instead of looping silently forever.
        # NOTE ON SEMANTICS: a clean/graceful return from trading_engine.run()
        # is counted toward consecutive_failures exactly like an exception.
        # This is intentional given this module's contract: the trader is
        # expected to loop internally forever, so ANY return — exception or
        # not — is treated as anomalous and restart-worthy. This assumption
        # only breaks if trading_engine.run() is ever changed to legitimately
        # return early and often (e.g. a designed periodic restart, or a
        # max_cycles-style batch completion faster than MIN_STABLE_RUN_SEC) —
        # in that case this loop would falsely declare "permanent failure"
        # and halt autonomous trading after MAX_RESTARTS clean cycles. If
        # trading_engine.run() semantics ever change, this loop must change
        # with it.
        BASE_RESTART_DELAY_SEC = 10
        MAX_RESTART_DELAY_SEC = 300
        MIN_STABLE_RUN_SEC = 300
        MAX_RESTARTS = int(os.getenv("MAX_RESTARTS", "10"))

        restart_count = 0
        consecutive_failures = 0
        permanent_failure = False
        try:
            while not self._stop_requested:
                run_started = time.time()
                try:
                    if self.max_cycles:
                        try:
                            report = self.trading_engine.run(max_cycles=self.max_cycles)
                        except TypeError:
                            logging.warning(
                                "[System] --max-cycles was set but trading_engine.run() "
                                "doesn't accept a max_cycles argument — ignoring override."
                            )
                            report = self.trading_engine.run()
                    else:
                        report = self.trading_engine.run()
                    self._write_final_report(report)
                    if self._stop_requested:
                        break
                    run_duration = time.time() - run_started
                    if run_duration >= MIN_STABLE_RUN_SEC:
                        consecutive_failures = 0
                    consecutive_failures += 1
                    restart_count += 1
                    if consecutive_failures > MAX_RESTARTS:
                        permanent_failure = True
                        logging.critical(
                            f"[System] Trader exited unexpectedly {consecutive_failures} "
                            f"times in a row without a stable run — stopping auto-restart. "
                            f"This looks like a permanent configuration failure, not a "
                            f"transient one."
                        )
                        self._notify_restart(restart_count, reason="permanent failure — auto-restart stopped")
                        break
                    delay = min(BASE_RESTART_DELAY_SEC * (2 ** (consecutive_failures - 1)),
                                MAX_RESTART_DELAY_SEC)
                    logging.warning(
                        f"[System] Trader exited unexpectedly (restart #{restart_count}, "
                        f"{consecutive_failures}/{MAX_RESTARTS} consecutive). "
                        f"Relaunching in {delay}s..."
                    )
                    self._notify_restart(restart_count, reason="unexpected exit")
                    # Day 81+ hotfix: record to crash log so operator can
                    # see exactly what happened.
                    try:
                        from core.trade_decision_log import log_cycle_error
                        log_cycle_error(symbol="SYSTEM", stage="trader_loop_exit",
                                        error=f"Trader exited unexpectedly (restart #{restart_count})")
                    except Exception: pass
                    time.sleep(delay)
                except KeyboardInterrupt:
                    logging.info("[System] Stop requested by user (Ctrl+C)")
                    self._stop_requested = True
                    break
                except Exception as e:
                    run_duration = time.time() - run_started
                    if run_duration >= MIN_STABLE_RUN_SEC:
                        consecutive_failures = 0
                    consecutive_failures += 1
                    restart_count += 1
                    # Day 81+ hotfix: capture exact error to crash log.
                    import traceback as _tb
                    _error_detail = f"{type(e).__name__}: {e}\n{_tb.format_exc()}"
                    logging.error(
                        f"[System] Fatal error in trading loop (restart #{restart_count}, "
                        f"{consecutive_failures}/{MAX_RESTARTS} consecutive): {e}",
                        exc_info=True,
                    )
                    try:
                        from core.trade_decision_log import log_cycle_error
                        log_cycle_error(symbol="SYSTEM", stage="trader_loop_crash",
                                        error=_error_detail[:2000])
                    except Exception: pass
                    # Publish system.error so bus subscribers (alerts) pick it up.
                    try:
                        from core.event_bus import get_bus
                        get_bus().publish("system.error", {
                            "channel": "fatal",
                            "reason": str(e),
                            "restart_count": restart_count,
                        }, source="main")
                    except Exception:
                        pass
                    if consecutive_failures > MAX_RESTARTS:
                        permanent_failure = True
                        logging.critical(
                            f"[System] {consecutive_failures} consecutive crashes without a "
                            f"stable run — stopping auto-restart (likely a permanent "
                            f"configuration failure): {e}"
                        )
                        self._notify_restart(restart_count, reason=f"permanent failure — auto-restart stopped: {str(e)[:150]}")
                        break
                    self._notify_restart(restart_count, reason=str(e)[:200])
                    if self._stop_requested:
                        break
                    delay = min(BASE_RESTART_DELAY_SEC * (2 ** (consecutive_failures - 1)),
                                MAX_RESTART_DELAY_SEC)
                    logging.info(f"[System] Relaunching trader in {delay}s...")
                    time.sleep(delay)
        finally:
            self.running = False
            if permanent_failure:
                logging.critical(
                    "[System] Exiting due to repeated unrecoverable failures. "
                    "Manual investigation required before restarting."
                )
                self._stop_requested = True
            self._shutdown()

    def stop_trading(self):
        """Request the trading loop to stop."""
        self._stop_requested = True
        if self.trading_engine:
            self.trading_engine.stop()
        logging.info("[System] Stop requested — shutting down gracefully")

    # ─────────────────────────────────────────────
    # SYSTEM STATUS
    # ─────────────────────────────────────────────

    def get_system_status(self) -> dict:
        """Get comprehensive system status."""
        uptime = (
            str(datetime.now(timezone.utc) - self.start_time)
            if self.start_time and self.running
            else None
        )
        return {
            "running": self.running,
            "uptime": uptime,
            "mode": self.execution_mode.upper(),
            "pairs": self.symbols,
            "timeframe": self.timeframe,
            "balance": self.balance,
            "boot_phases": self.runtime.lifecycle.report()["phases"],
            "registry_health": self.runtime.registry.health(),
            "trader_health": (
                self.trading_engine.health_status()
                if self.trading_engine and hasattr(self.trading_engine, "health_status")
                else None
            ),
            "runtime_metrics": self.runtime.metrics.build_report(),
        }

    def get_health_snapshot(self) -> dict:
        """Force a one-shot health check and return the snapshot."""
        return self.runtime.health.run_once().to_dict()

    # ─────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────

    def _notify_startup(self, notifier):
        """Send startup notification via Telegram."""
        try:
            msg = (
                f"🤖 FOREX AI System Started\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"Mode: {self.execution_mode.upper()}\n"
                f"Pairs: {len(self.symbols)} ({', '.join(self.symbols[:5])}...)\n"
                f"Timeframe: {self.timeframe}\n"
                f"Balance: ${self.balance}\n"
                f"Max Open: 5 | Max Daily Loss: 3%\n"
                f"Auto-restart: ON\n"
                f"━━━━━━━━━━━━━━━━━━━━━"
            )
            from utils.async_utils import run_coro_sync
            run_coro_sync(notifier.send_message(msg))
        except Exception as e:
            logging.warning(f"[System] Telegram startup notification failed: {e}")

    def _notify_restart(self, restart_count: int, reason: str = ""):
        """Send auto-restart notification via Telegram."""
        notifier = self.runtime.registry.try_resolve("telegram_notifier")
        if not notifier:
            return
        try:
            msg = (
                f"🔄 AUTO-RESTART #{restart_count}\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"Reason: {reason[:200] if reason else 'unexpected exit'}\n"
                f"Relaunching in 10-30s...\n"
                f"━━━━━━━━━━━━━━━━━━━━━"
            )
            from utils.async_utils import run_coro_sync
            run_coro_sync(notifier.send_message(msg))
        except Exception as e:
            logging.warning(f"[System] Telegram restart notification failed: {e}")

    def _write_final_report(self, report: dict):
        """Save final system report."""
        report_dir = PROJECT_ROOT / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / "latest_report.json"
        tmp_path = path.with_suffix(".json.tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            # Atomic on POSIX and Windows (Python 3.3+) — readers of
            # latest_report.json never observe a partially-written file.
            os.replace(tmp_path, path)
        except Exception as e:
            logging.error(f"[System] Failed to write final report to {path}: {e}")
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _shutdown(self):
        """Graceful shutdown sequence."""
        logging.info("[System] Shutting down...")

        # Co-founder fix: resolve notifier from registry (was undefined)
        notifier = None
        try:
            if hasattr(self, 'runtime') and self.runtime:
                notifier = self.runtime.registry.try_resolve("telegram_notifier")
        except Exception:
            pass

        # Wire GracefulShutdownManager — run cleanup callbacks
        try:
            from core.graceful_shutdown import GracefulShutdownManager
            shutdown_mgr = GracefulShutdownManager(timeout_seconds=30.0)

            # Register cleanup callbacks
            shutdown_mgr.register_cleanup_callback(lambda: self.runtime.shutdown())
            if notifier:
                try:
                    from utils.async_utils import run_coro_sync
                    shutdown_mgr.register_cleanup_callback(
                        lambda: run_coro_sync(
                            notifier.send_message("FOREX AI System Stopped")
                        )
                    )
                except Exception:
                    pass

            # Run cleanup
            shutdown_mgr.run_cleanup()
            logging.info("[System] GracefulShutdownManager cleanup complete")
        except Exception as e:
            logging.error(f"[System] GracefulShutdown failed, falling back: {e}")
            # Fallback: original shutdown
            try:
                self.runtime.shutdown()
            except Exception as e2:
                logging.error(f"[System] Runtime shutdown error: {e2}")

        logging.info("[System] Shutdown complete")


# ──────────────────────────────────────────────────────────────
# SYSTEM STATUS TRACKER (kept for backward compat with old code paths)
# ──────────────────────────────────────────────────────────────

class SystemStatus:
    """Tracks initialization status of all components."""

    def __init__(self):
        self.checks = {}
        self.errors = []

    def ok(self, component: str, detail: str = ""):
        self.checks[component] = {"status": "OK", "detail": detail}

    def fail(self, component: str, reason: str):
        self.checks[component] = {"status": "FAILED", "detail": reason}
        self.errors.append(f"{component}: {reason}")

    def warn(self, component: str, detail: str):
        self.checks[component] = {"status": "WARNING", "detail": detail}

    @property
    def all_ok(self) -> bool:
        return not self.errors


# ──────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────────────────────

def main():
    """Main entry point."""
    # Audit fix: config validation used to happen as an import-time side
    # effect (crashing before this function even started, with a raw
    # traceback). It's now explicit and handled here so a bad/missing MT5
    # credential produces one clear message and a clean exit instead of
    # a stack trace, and so it can't crash unrelated code that merely
    # imports `config` for a constant.
    try:
        from config import validate_all_config
        validate_all_config()
    except Exception as e:
        print(f"\n[FATAL] Configuration invalid — refusing to start: {e}\n"
              f"Fix your .env (or set MT5_FALLBACK_TO_SIMULATION=true) and retry.\n")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="FOREX AI — Autonomous Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Start autonomous trading
  python main.py --mode init              # Initialize system only
  python main.py --mode status            # Show system status
  python main.py --mode health            # Show health snapshot
  python main.py --mode obsolete          # Show obsolete module registry
  python main.py --paper                  # Force paper trading mode
  python main.py --pairs EURUSD,GBPUSD    # Trade specific pairs
  python main.py --timeframe 1h           # Use 1-hour timeframe
  python main.py --no-telegram            # Disable Telegram alerts
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["init", "start", "status", "stop", "backtest", "health", "obsolete", "diagnostic"],
        default="start",
        help="System mode",
    )
    parser.add_argument("--pairs", help="Comma-separated currency pairs (e.g., EURUSD,GBPUSD)")
    parser.add_argument("--timeframe", help="Trading timeframe (e.g., 15m, 1h, 4h). In --mode backtest, "
                                             "MT5-style aliases (M15, H1, H4, D1) are also accepted.")
    parser.add_argument("--paper", action="store_true", help="Force paper trading mode")
    parser.add_argument("--no-telegram", action="store_true", help="Disable Telegram notifications")
    parser.add_argument("--balance", type=float, help="Starting balance override")
    parser.add_argument("--max-cycles", type=int, help="Max trading cycles (for testing)")
    parser.add_argument("--disable-news-block", action="store_true", help="Allow trading even when the news filter would otherwise block")
    # --mode backtest options (delegated to run_backtest.run_pairs() — see _run_backtest()).
    parser.add_argument("--bars", type=int, default=500, help="[backtest mode] Number of historical bars to replay")
    parser.add_argument("--days", type=int, default=None, help="[backtest mode] Convenience alias for --bars, in days")
    parser.add_argument("--synthetic", action="store_true", help="[backtest mode] Use synthetic OHLCV data instead of MT5")

    args = parser.parse_args()

    if args.disable_news_block:
        os.environ["NEWS_BLOCK_ENABLED"] = "false"
        print("[main] News block disabled via CLI override")

    if getattr(args, "paper", False):
        print("[main] WARNING: --paper has no effect — legacy paper trading mode was "
              "removed from this system. EXECUTION_MODE is mt5_demo by default, or "
              "mt5_live if explicitly opted into (see config.py ALLOW_REAL_MONEY_TRADING).")

    setup_logging()
    logger = logging.getLogger("main")

    # Special modes that don't require a full boot
    if args.mode == "obsolete":
        _print_obsolete_registry()
        return

    if args.mode == "backtest":
        _run_backtest(args)
        return

    # Day 96 — Signal Diagnostic Mode
    # Runs one full analysis cycle per pair and prints where signals die.
    if args.mode == "diagnostic":
        _run_diagnostic(args)
        return

    # All other modes boot the runtime.
    system = ForexAISystem(args)

    try:
        if args.mode == "init":
            success = system.initialize()
            sys.exit(0 if success else 1)

        elif args.mode == "start":
            if system.initialize():
                system.start_trading()
            else:
                logger.error("System initialization failed — cannot start trading")
                sys.exit(1)

        elif args.mode == "status":
            if system.initialize():
                status = system.get_system_status()
                print(json.dumps(status, indent=2, default=str))

        elif args.mode == "health":
            if system.initialize():
                health = system.get_health_snapshot()
                print(json.dumps(health, indent=2, default=str))

        elif args.mode == "stop":
            logger.info("Stop command — system must be stopped via Ctrl+C or Telegram")

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt — stopping")
        system.stop_trading()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


def _run_backtest(args):
    """Run a backtest via the shared decision kernel.

    FIX (execution-parity audit §6.4 — Critical, "dead code"): this used
    to do `engine = BacktestEngine(); log.info("Backtest complete")` and
    return — it never called `run_strategy()`, so anyone using
    `main.py --mode backtest` (as opposed to the standalone
    `run_backtest.py` script) got a misleading "complete" log line and
    zero actual backtest activity, no error raised.

    Now this delegates to `run_backtest.run_pairs()` — the SAME function
    the standalone `python run_backtest.py` CLI uses, which drives
    `backtest.unified_engine.run_unified_backtest()` (the shared
    AnalysisAgent -> DecisionAgent -> RiskEngine -> PositionSizer kernel
    Demo/Real also use). There is now exactly one backtest implementation,
    reachable from two entry points, instead of three disconnected ones
    (this dead stub, the old UnifiedSignalEngine-only run_backtest.py
    loop, and the never-imported BacktestEngine class).
    """
    import run_backtest as _rb

    pairs_arg = args.pairs if getattr(args, "pairs", None) else None
    tf_raw = (args.timeframe or "H1").upper()
    # main.py's --timeframe accepts live-style aliases ("15m","1h"); the
    # shared backtest kernel/broker-sim use MT5-style ("M15","H1"). Accept
    # either so `--mode backtest --timeframe 1h` and `--tf H1` both work.
    TF_ALIAS = {"15M": "M15", "30M": "M30", "1H": "H1", "4H": "H4", "1D": "D1"}
    tf = TF_ALIAS.get(tf_raw, tf_raw)
    if tf not in ("M15", "M30", "H1", "H4", "D1"):
        logging.warning(f"[Backtest] Unrecognized timeframe '{args.timeframe}', defaulting to H1")
        tf = "H1"

    ns = argparse.Namespace(
        pair=(pairs_arg.split(",")[0].strip().upper() if pairs_arg else "EURUSD"),
        pairs=pairs_arg or "",
        tf=tf,
        bars=getattr(args, "bars", 500) or 500,
        days=getattr(args, "days", None),
        balance=getattr(args, "balance", None) or 10000.0,
        spread=None,
        commission=7.0,
        slippage=2.0,
        max_trades=3,
        max_hold=100,
        synthetic=getattr(args, "synthetic", False),
        verbose=True,
        json=False,
    )

    logging.info(f"[Backtest] Starting shared-kernel backtest: pairs={ns.pairs or ns.pair} "
                 f"tf={ns.tf} bars={ns.bars} synthetic={ns.synthetic}")
    results = _rb.run_pairs(ns)
    if not results:
        logging.error("[Backtest] No results produced — check data source "
                       "(pass --synthetic if MT5 is not available) and logs/ for errors.")
        return
    logging.info(f"[Backtest] Complete — {len(results)} pair(s) run. "
                 f"CSVs written to backtest/results_<PAIR>_<TF>.csv")


def _print_obsolete_registry():
    """Print the obsolete module registry."""
    from core.obsolete import OBSOLETE_MODULES, obsolete_summary

    print("\n" + "=" * 70)
    print("  OBSOLETE / ORPHAN MODULE REGISTRY")
    print("=" * 70)
    summary = obsolete_summary()
    print(f"\n  Total: {summary['total']} modules")
    for cat, count in sorted(summary.items()):
        if cat == "total":
            continue
        print(f"    {cat:<14} {count}")
    print()
    for entry in OBSOLETE_MODULES:
        print(f"  [{entry.category.value.upper():<12}] {entry.path}")
        print(f"    reason: {entry.reason}")
        print(f"    action: {entry.action}")
        print()


def _run_diagnostic(args):
    """Day 96 — Signal Diagnostic Mode.

    Runs one full analysis cycle per pair and prints a summary showing
    WHERE signals die — so you can understand why the bot isn't trading.
    """
    print("=" * 60)
    print("  🔍  SIGNAL DIAGNOSTIC MODE  (Day 96)")
    print("=" * 60)

    from config import SYMBOLS, DEFAULT_TIMEFRAME
    from agents.market_agent import MarketAgent
    from agents.analysis_agent import AnalysisAgent

    pairs = args.pairs.split(",") if args.pairs else SYMBOLS
    timeframe = args.timeframe or DEFAULT_TIMEFRAME

    print(f"\n  Pairs:      {', '.join(pairs)}")
    print(f"  Timeframe:  {timeframe}")
    print(f"  Mode:       SAFE (80% confidence threshold)")
    print()

    agent = AnalysisAgent()

    for pair in pairs:
        pair = pair.strip().upper()
        print(f"\n{'─' * 60}")
        print(f"  📊  {pair} {timeframe}")
        print(f"{'─' * 60}")

        # Step 1: Market data
        try:
            market = MarketAgent(pair, timeframe).run()
            if "error" in market:
                print(f"  ❌ Data:        FAIL — {market['error']}")
                continue
            print(f"  ✅ Data:        PASS — {len(market['df'])} candles from {market.get('data_source','?')}")
            print(f"     Price:       {market['ind_ctx'].get('price','?')}")
            print(f"     Trend:       {market['ind_ctx'].get('trend','?')}")
            print(f"     RSI:         {market['ind_ctx'].get('rsi','?')}")
            print(f"     ADX:         {market['ind_ctx'].get('adx','?')}")
            print(f"     Regime:      {market['regime'].get('regime','?')} {market['regime'].get('direction','?')}")
        except Exception as e:
            print(f"  ❌ Data:        FAIL — {e}")
            continue

        # Step 2: Full analysis pipeline
        try:
            # Fix: mtf_bias can be a string from MarketAgent, but AnalysisAgent
            # expects a dict. Convert string → dict for compatibility.
            if isinstance(market.get("mtf_bias"), str):
                market["mtf_bias"] = {"bias": market["mtf_bias"], "confidence": "MEDIUM"}
            analysis = agent.run(market, memory_ctx={})
            final = analysis.get("final_signal", "UNKNOWN")
            print(f"\n  ── Pipeline Results ──")
            print(f"  Signal:        {analysis.get('signal',{}).get('signal','?')} ({analysis.get('signal',{}).get('confidence',0)}%)")
            print(f"  SMC:           {analysis.get('smc_ctx',{}).get('smc_signal','?')}")
            print(f"  Strategy:      {analysis.get('strategy',{}).get('strategy','?')} ({analysis.get('strategy',{}).get('confidence',0)}%)")

            # Day 94/95 contexts
            fred = analysis.get("fred_ctx", {})
            if fred.get("fred_source") != "none":
                print(f"  FRED:          yield={fred.get('fred_yield_curve','?')} rates={fred.get('fred_rate_env','?')} CPI={fred.get('fred_cpi','?')}")

            sent = analysis.get("retail_sentiment_ctx", {})
            if sent.get("sentiment_source") not in ("fallback", "none", None):
                print(f"  Sentiment:     {sent.get('sentiment_contrarian','?')} ({sent.get('sentiment_strength','?')}) src={sent.get('sentiment_source','?')}")

            econ = analysis.get("econ_calendar_ctx", {})
            if econ.get("econcal_source") not in ("none", None):
                print(f"  Econ Cal:      {econ.get('econcal_event_count',0)} events, block={econ.get('econcal_trade_block',False)}")

            # Master decision
            md = analysis.get("master_decision", {})
            if md:
                print(f"  Master:        {md.get('final_signal','?')} ({md.get('master_confidence',0):.0f}%) pos={md.get('position_size','?')}")

            # Final verdict
            print(f"\n  ═══ FINAL: {final} ═══")
            if final in ("NO_TRADE", "WAIT"):
                # Find the blocker
                blocked_at = "unknown"
                if not analysis.get("session", {}).get("trade_allowed", True):
                    blocked_at = "session gate"
                elif analysis.get("news", {}).get("trade_allowed") is False:
                    blocked_at = "news block"
                elif analysis.get("master_decision", {}).get("strategy") == "WAIT":
                    blocked_at = "strategy WAIT"
                elif analysis.get("signal", {}).get("confidence", 0) < 80:
                    blocked_at = f"confidence {analysis.get('signal',{}).get('confidence',0)}% < 80%"
                print(f"  Blocked at:    {blocked_at}")
            elif final in ("BUY", "SELL"):
                print(f"  ✅ Trade signal generated!")
        except Exception as e:
            print(f"  ❌ Analysis:    FAIL — {e}")
            import traceback; traceback.print_exc()

    print(f"\n{'=' * 60}")
    print("  Diagnostic complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()