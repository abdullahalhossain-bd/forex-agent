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
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure UTF-8 stdout/stderr (Windows console quirks)
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

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
    """Configure comprehensive logging.

    2026-08-12 cleanup: unified logging to use utils/logger.py's
    RotatingFileHandler (logs/trader.log) instead of a separate
    forex_ai.log. This ensures all logs go through SecretRedactionFilter
    and benefit from log rotation. The legacy forex_ai.log is retained
    as a symlink alias for backwards compatibility.
    """
    from utils.logger import get_logger as _get_logger

    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    # Configure root logger to use utils/logger.py's handlers
    # (RotatingFileHandler at logs/trader.log + console)
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # If root already has handlers (from basicConfig in imported modules),
    # clear them to avoid duplicate output
    if root.handlers:
        root.handlers.clear()

    # Use the shared handlers from utils/logger.py
    try:
        from utils.logger import _get_shared_handlers
        for h in _get_shared_handlers():
            root.addHandler(h)
    except Exception:
        # Fallback: basic config
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(LOGS_DIR / "forex_ai.log", encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )

    return _get_logger("main")


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
        self.balance = INITIAL_BALANCE

        # The trader is constructed by the runtime's RUNTIME phase and
        # registered in the ServiceRegistry under "trader" / "trading_engine".
        # We resolve it after boot.
        self.trading_engine = None

    def _resolve_symbols(self) -> list[str]:
        pairs_arg = getattr(self.args, "pairs", None)
        if pairs_arg:
            return [clean_symbol(p.strip()) for p in pairs_arg.split(",")]
        return [clean_symbol(s) for s in SYMBOLS]

    # ─────────────────────────────────────────────
    # INITIALIZATION
    # ─────────────────────────────────────────────

    def initialize(self) -> bool:
        """Boot the entire runtime through the LifecycleManager. Returns True
        if every critical phase succeeded."""
        # 2026-08-12: replaced print() with logger for clean professional output
        log = logging.getLogger("main")
        log.info("=" * 60)
        log.info("  FOREX AI AUTONOMOUS TRADING SYSTEM — Booting")
        log.info("=" * 60)
        log.info("  Booting runtime (24 phases)...")

        # Day 81+ hotfix: TEST_MODE warning banner.
        # TEST_MODE bypasses MasterAnalyst, Confluence, Ensemble, RL, and
        # MasterDecision gates.  It also lowers TradePermission confidence
        # threshold to 10% and force-approves PositionSizer rejects with
        # lot=0.01.  This is fine for first-time MT5 verification, but
        # should NOT stay on in production.
        try:
            from config import TEST_MODE, SIMULATION_MODE, MAX_LOT, APPROVAL_MODE, MAX_OPEN_TRADES
            if TEST_MODE:
                log.warning("=" * 60)
                log.warning("  TEST_MODE = True — All safety gates are PERMISSIVE")
                log.warning("    - MasterAnalyst/Confluence/Ensemble/RL bypassed")
                log.warning("    - TradePermission MIN_CONFIDENCE = 10 (prod=70)")
                log.warning("    - PositionSizer rejects force-approved lot=0.01")
                log.warning("    - Session quality check disabled")
                log.warning("  Set TEST_MODE=false in .env for production trading.")
                log.warning("=" * 60)
            if SIMULATION_MODE:
                log.info("=" * 60)
                log.info("  SIMULATION_MODE = True — No real MT5 orders will be placed.")
                log.info("  Set SIMULATION_MODE=false in .env for live trading.")
                log.info("=" * 60)
            log.info(f"  Config: MAX_LOT={MAX_LOT} | APPROVAL_MODE={APPROVAL_MODE} | MAX_OPEN_TRADES={MAX_OPEN_TRADES}")
            print()
        except Exception:
            pass

        # Override config-driven settings if CLI args were supplied.
        self._apply_cli_overrides()

        # Register a phase-complete callback so we get a live progress log.
        _phase_log = logging.getLogger("main.boot")
        def _on_phase(result):
            icon = "OK" if result.ok else "!!"
            if result.skipped:
                icon = "--"
            svcs = ", ".join(result.services_registered) if result.services_registered else "(no services)"
            err = f"  ERR: {result.error}" if result.error else ""
            _phase_log.info(f"  [{icon}] {result.phase.value:<14} ({result.duration_sec:.3f}s) - {svcs}{err}")

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
                logging.getLogger("main").info(f"  Orphan cleanup: {reconciled['closed']} stale DB-OPEN trades auto-closed")
                logging.getLogger("main").info(f"  Orphan cleanup details: {reconciled}")
            elif reconciled["kept"] > 0:
                logging.getLogger("main").info(f"  {reconciled['kept']} DB-OPEN trades verified against MT5 (all real)")
        except Exception as e:
            logging.getLogger("main").warning(f"  Orphan cleanup skipped: {e}")

        self._print_boot_summary()

        # Critical phases: BOOTSTRAP and PERSISTENCE must succeed.
        for critical in (Phase.BOOTSTRAP, Phase.PERSISTENCE):
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
        """Print final boot summary via logger."""
        _log = logging.getLogger("main")
        report = self.runtime.lifecycle.report()
        phases = report["phases"]
        ok = sum(1 for p in phases if p["ok"] and not p["skipped"])
        failed = sum(1 for p in phases if not p["ok"])
        skipped = sum(1 for p in phases if p["skipped"])
        _log.info("=" * 60)
        _log.info(f"  Boot complete: {ok} phases OK, {failed} failed, {skipped} skipped")
        if failed:
            _log.warning(f"  FAILED phases:")
            for p in phases:
                if not p["ok"]:
                    _log.warning(f"    - {p['phase']}: {p.get('error', 'unknown')}")
        _log.info(f"  Trader wired: {'YES' if self.trading_engine else 'NO'}")
        _log.info(f"  Registry services: {len(self.runtime.registry.health())}")
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
        logging.info(
            f"[System] Trading started | Mode: {self.execution_mode.upper()} | "
            f"Pairs: {self.symbols} | Balance: ${self.balance} | "
            f"Auto-restart: ON"
        )

        # Send startup notification
        notifier = self.runtime.registry.try_resolve("telegram_notifier")
        if notifier:
            self._notify_startup(notifier)

        restart_count = 0
        try:
            while not self._stop_requested:
                try:
                    report = self.trading_engine.run()
                    self._write_final_report(report)
                    if self._stop_requested:
                        break
                    # Trader returned without being asked to stop — this is
                    # unexpected (the run() loop should run forever). Log +
                    # restart.
                    restart_count += 1
                    logging.warning(
                        f"[System] Trader exited unexpectedly (restart #{restart_count}). "
                        f"Relaunching in 10s..."
                    )
                    self._notify_restart(restart_count, reason="unexpected exit")
                    # Day 81+ hotfix: record to crash log so operator can
                    # see exactly what happened.
                    try:
                        from core.trade_decision_log import log_cycle_error
                        log_cycle_error(symbol="SYSTEM", stage="trader_loop_exit",
                                        error=f"Trader exited unexpectedly (restart #{restart_count})")
                    except Exception: pass
                    time.sleep(10)
                except KeyboardInterrupt:
                    logging.info("[System] Stop requested by user (Ctrl+C)")
                    self._stop_requested = True
                    break
                except Exception as e:
                    restart_count += 1
                    # Day 81+ hotfix: capture exact error to crash log.
                    import traceback as _tb
                    _error_detail = f"{type(e).__name__}: {e}\n{_tb.format_exc()}"
                    logging.error(
                        f"[System] Fatal error in trading loop (restart #{restart_count}): {e}",
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
                    self._notify_restart(restart_count, reason=str(e)[:200])
                    if self._stop_requested:
                        break
                    logging.info(f"[System] Relaunching trader in 30s...")
                    time.sleep(30)
        finally:
            self.running = False
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
            import asyncio

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
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(notifier.send_message(msg))
                else:
                    loop.run_until_complete(notifier.send_message(msg))
            except RuntimeError:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(notifier.send_message(msg))
                loop.close()
        except Exception as e:
            logging.warning(f"[System] Telegram startup notification failed: {e}")

    def _notify_restart(self, restart_count: int, reason: str = ""):
        """Send auto-restart notification via Telegram."""
        notifier = self.runtime.registry.try_resolve("telegram_notifier")
        if not notifier:
            return
        try:
            import asyncio
            msg = (
                f"🔄 AUTO-RESTART #{restart_count}\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"Reason: {reason[:200] if reason else 'unexpected exit'}\n"
                f"Relaunching in 10-30s...\n"
                f"━━━━━━━━━━━━━━━━━━━━━"
            )
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(notifier.send_message(msg))
                else:
                    loop.run_until_complete(notifier.send_message(msg))
            except RuntimeError:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(notifier.send_message(msg))
                loop.close()
        except Exception as e:
            logging.warning(f"[System] Telegram restart notification failed: {e}")

    def _write_final_report(self, report: dict):
        """Save final system report."""
        report_dir = PROJECT_ROOT / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / "latest_report.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

    def _shutdown(self):
        """Graceful shutdown sequence."""
        logging.info("[System] Shutting down...")
        notifier = self.runtime.registry.try_resolve("telegram_notifier")
        if notifier:
            try:
                import asyncio

                msg = "FOREX AI System Stopped"
                try:
                    loop = asyncio.get_event_loop()
                    if not loop.is_running():
                        loop.run_until_complete(notifier.send_message(msg))
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(notifier.send_message(msg))
                    loop.close()
            except Exception:
                pass

        # Tear down the runtime (stops health monitor, runs shutdown hooks).
        try:
            self.runtime.shutdown()
        except Exception as e:
            logging.error(f"[System] Runtime shutdown error: {e}", exc_info=True)

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
    parser.add_argument("--timeframe", help="Trading timeframe (e.g., 15m, 1h, 4h)")
    parser.add_argument("--paper", action="store_true", help="Force paper trading mode")
    parser.add_argument("--no-telegram", action="store_true", help="Disable Telegram notifications")
    parser.add_argument("--balance", type=float, help="Starting balance override")
    parser.add_argument("--max-cycles", type=int, help="Max trading cycles (for testing)")
    parser.add_argument("--days", type=int, help="Backtest: use only the last N days of data")
    parser.add_argument("--bars", type=int, help="Backtest: use only the last N bars/candles of data")
    parser.add_argument("--bypass-gates", help="Comma-separated TradePermission gates to bypass during backtest")

    args = parser.parse_args()

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
    """Run the unified backtest engine — the SAME decision core as live trading.

    This is the ONLY backtest entry point whose results predict live/demo
    behavior. It constructs a real AITrader (the same class live uses)
    and replays historical bars through its evaluate_decision_core() method.

    Every gate that blocks in live (confidence, R:R, cost-EV, news, session,
    fusion, entry quality, confirmation bias, revenge trading, correlation)
    runs here too, via the same code, with the same thresholds, in the same order.

    The ONLY differences from live:
      1. Data source: historical CSV instead of live MT5 feed
      2. Execution: BrokerSimulator (bar OHLC SL/TP touch) instead of MT5 order
      3. Account: isolated backtest DB, never the live trader.db
    """
    import pandas as pd
    from backtest.unified_engine import run_unified_backtest
    from backtest.data_loader import HistoricalDataLoader

    # Resolve parameters
    pairs = [clean_symbol(p.strip()) for p in (args.pairs or ",").split(",") if p.strip()] or ["EURUSD"]
    timeframe = (args.timeframe or DEFAULT_TIMEFRAME).upper().replace("M", "").replace("H", "") + \
               ("m" if (args.timeframe or DEFAULT_TIMEFRAME).startswith("M") else "H")
    # Normalize timeframe format (e.g., "1h" -> "H1", "15m" -> "M15")
    tf_raw = args.timeframe or DEFAULT_TIMEFRAME
    tf_map = {"1m": "M1", "5m": "M5", "15m": "M15", "30m": "M30",
              "1h": "H1", "4h": "H4", "1d": "D1"}
    timeframe = tf_map.get(tf_raw.lower(), tf_raw.upper())

    balance = args.balance if hasattr(args, 'balance') and args.balance else INITIAL_BALANCE

    logging.info(f"[Backtest] Unified engine — pairs={pairs} timeframe={timeframe} balance=${balance}")

    loader = HistoricalDataLoader()
    all_results = {}

    for symbol in pairs:
        csv_path = PROJECT_ROOT / "data" / f"{symbol}_{timeframe}.csv"
        if not csv_path.exists():
            logging.error(f"[Backtest] Data file not found: {csv_path}")
            print(f"  ERROR: {csv_path} not found. Available data files in data/")
            continue

        print(f"\n{'=' * 60}")
        print(f"  BACKTEST: {symbol} {timeframe}")
        print(f"  Data: {csv_path}")
        print(f"  Balance: ${balance}")
        print(f"  Engine: unified (shares live decision core)")
        print(f"{'=' * 60}")

        df = loader.load_csv(file_path=str(csv_path), pair=symbol, timeframe=timeframe)
        if df is None or len(df) == 0:
            logging.error(f"[Backtest] Failed to load {csv_path}")
            continue

        # --days / --bars: let the user run a quick short test or a full
        # multi-year backtest without touching the CSV files themselves.
        # --days filters by wall-clock time (works across any timeframe);
        # --bars just takes the last N rows. If both are given, --bars wins
        # since it's the more precise cut.
        if getattr(args, "days", None):
            cutoff = df.index.max() - pd.Timedelta(days=args.days)
            df = df[df.index >= cutoff]
        if getattr(args, "bars", None):
            df = df.tail(args.bars)

        if len(df) == 0:
            logging.error(f"[Backtest] No data left for {symbol} after applying --days/--bars filter")
            continue

        print(f"  Bars: {len(df)} | {df.index[0]} to {df.index[-1]}")
        print(f"  Running...")

        bypass_checks = []
        if getattr(args, "bypass_gates", None):
            bypass_checks = [gate.strip() for gate in args.bypass_gates.split(",") if gate.strip()]
            print(f"  Bypass gates: {bypass_checks}")

        result = run_unified_backtest(
            symbol=symbol,
            df=df,
            timeframe=timeframe,
            starting_balance=balance,
            warmup_bars=50,
            max_open_trades=3,
            max_hold_bars=100,
            db_path=f"backtest/backtest_run_{symbol}_{timeframe}.db",
            verbose=True,
            bypass_checks=bypass_checks,
        )

        if result.error:
            print(f"\n  BACKTEST FAILED: {result.error}")
            all_results[symbol] = {"error": result.error}
            continue

        m = result.metrics
        print(f"\n{'=' * 60}")
        print(f"  RESULTS: {symbol} {timeframe}")
        print(f"{'=' * 60}")
        print(f"  Total bars replayed:  {result.bars}")
        print(f"  Total trades:          {m.total_trades}")
        print(f"  Win rate:              {m.win_rate:.1f}%")
        print(f"  Profit factor:         {m.profit_factor:.2f}")
        print(f"  Total P&L:             ${m.total_pnl_usd:.2f}")
        avg_pnl = (m.total_pnl_usd / m.total_trades) if m.total_trades else 0.0
        print(f"  Avg P&L per trade:     ${avg_pnl:.2f}")
        print(f"  Max drawdown:          {m.max_drawdown_pct:.1f}%")
        print(f"  Sharpe (annualized):   {m.sharpe_ratio:.2f}")
        print(f"  Sortino (annualized):  {m.sortino_ratio:.2f}")
        print(f"  Avg win / Avg loss:    ${m.avg_win_usd:.2f} / ${m.avg_loss_usd:.2f}")
        print(f"\n  Rejection stats:")
        for k, v in result.rejection_stats.items():
            print(f"    {k:<25} {v}")
        print(f"{'=' * 60}")

        all_results[symbol] = {
            "symbol": symbol, "timeframe": timeframe,
            "trades": m.total_trades, "win_rate": round(m.win_rate, 2),
            "profit_factor": round(m.profit_factor, 3),
            "total_pnl_usd": round(m.total_pnl_usd, 2),
            "max_drawdown_pct": round(m.max_drawdown_pct, 2),
            "sharpe": round(m.sharpe_ratio, 3), "sortino": round(m.sortino_ratio, 3),
            "rejection_stats": result.rejection_stats,
        }

    # Save summary
    summary_path = PROJECT_ROOT / "backtest" / "results" / "unified_backtest_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Summary saved: {summary_path}")
    logging.info(f"[Backtest] Complete — summary at {summary_path}")


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