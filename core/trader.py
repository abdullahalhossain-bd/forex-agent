# core/trader.py  —  Day 37 | Full Integration (Week 3 + Day 31 + Day 36 wired in)
#
# Changes vs the Day 21 version:
#   - AITrader now routes every order through ExecutionRouter (paper / mt5_demo)
#     instead of calling PaperTrader directly, so EXECUTION_MODE in .env
#     actually switches backends without touching this file again.
#   - CircuitBreaker (kill switch) and ApprovalMode (Mode 1/2/3 human approval)
#     are real gates in run_cycle() now, not just standalone unused modules.
#   - CorrelationFilter is folded into the Safety Guard step alongside the
#     existing TradePermission checks (news/confidence/session/duplicate).
#   - AutonomousTraderSystem can pull its per-cycle pair list from
#     MarketScanner instead of a fixed SYMBOLS list (falls back safely if
#     the MT5 market-data adapter isn't wired yet).
#   - CircuitBreaker + ApprovalMode are created ONCE in AutonomousTraderSystem
#     and shared across every symbol's AITrader — both persist to a single
#     global state file (memory/circuit_breaker_state.json,
#     memory/pending_approvals.json), so per-symbol instances would silently
#     stomp on each other's state. Standalone AITrader usage still works:
#     if you don't pass one in, it creates its own.
#
#   - Day 37 hotfix: `vec_ctx` is now initialized to `{}` BEFORE the
#     `if memory_ctx["total_trades"] > 0:` block in run_cycle(). Previously
#     it was only assigned inside that block, so on a fresh symbol with no
#     trade history yet (total_trades == 0), `result["memory_context"] =
#     vec_ctx` further down raised UnboundLocalError and killed the whole
#     cycle for every symbol in the same run.
#
#   - Day 37+ runtime-unified hotfix (this revision): `_start_telegram_commands`
#     had a body indented at the SAME level as its `def` line (instead of one
#     level deeper), which is a syntax error in Python and also caused
#     `_notify_warning` to be swallowed as part of that broken block. Fixed
#     indentation restores both as proper, separate methods of
#     AutonomousTraderSystem.

import asyncio
import json
import os
import shutil
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from agents.analysis_agent import AnalysisAgent
from agents.decision_agent import DecisionAgent
from agents.learning_agent import LearningAgent
from agents.market_agent import MarketAgent
from config import EXECUTION_MODE
from core.approval_mode import ApprovalMode
from database.db import TraderDB
from execution.execution_router import ExecutionRouter
from execution.paper_trader import PaperTrader
from memory.history import AnalysisHistory
from memory.learning import LearningEngine
from memory.trade_memory import TradeMemory
from risk.circuit_breaker import CircuitBreaker
from risk.risk_engine import RiskEngine
from risk.trade_permission import TradePermission
from scanner.correlation_filter import CorrelationFilter
from utils.logger import get_logger
from analysis.session_analyzer import SessionAnalyzer
from visualization.chart import ChartEngine

# ── MT5 availability guard (Bug fix: MT5_AVAILABLE was not defined) ──
try:
    import MetaTrader5 as _mt5_module
    MT5_AVAILABLE = True
except (ImportError, OSError, Exception):
    MT5_AVAILABLE = False
    _mt5_module = None

# ── Runtime infrastructure (Day 37+ runtime unification) ─────────────
# These imports are soft — if the new runtime modules aren't available
# (e.g. during a partial deployment), the trader still works exactly as
# before. When they ARE available, the trader publishes events, records
# metrics, and accepts a service registry for dependency injection.
try:
    from core.event_bus import EventBus, get_bus
    from core.runtime_metrics import RuntimeMetrics, get_metrics
    from core.service_registry import ServiceRegistry
    _RUNTIME_INFRA_AVAILABLE = True
except Exception as e:
    _RUNTIME_INFRA_AVAILABLE = False
    EventBus = None
    get_bus = None
    RuntimeMetrics = None
    get_metrics = None
    ServiceRegistry = None

try:
    import alerts.telegram_bot as telegram_module
    from alerts.telegram_bot import TelegramNotifier, start_telegram_bot_polling
except Exception as e:
    telegram_module = None
    TelegramNotifier = None
    start_telegram_bot_polling = None

try:
    from learning.mistake_analyzer import AdvancedMistakeAnalyzer
except Exception as e:
    AdvancedMistakeAnalyzer = None

try:
    from scanner.market_scanner import MarketScanner
except Exception as e:
    MarketScanner = None

log = get_logger("ai_trader")


class AITrader:

    VERSION = "Week3-Day37-RuntimeUnified"

    def __init__(
        self,
        balance: float = 10000.0,
        symbol: str = "EURUSD",
        timeframe: str = "15m",
        seed_rules: bool = True,
        paper_balance: float = 10000.0,
        notifier=None,
        execution_mode: str = None,
        approval_mode: int = 3,
        circuit_breaker: CircuitBreaker = None,
        approval: ApprovalMode = None,
        registry: "Optional[ServiceRegistry]" = None,
        paper_trader: "Optional[PaperTrader]" = None,
        db: "Optional[TraderDB]" = None,
    ):
        self.balance = balance
        self.symbol = self._clean_symbol(symbol)
        self.timeframe = timeframe
        self.notifier = notifier
        self.execution_mode = (execution_mode or EXECUTION_MODE).lower()
        self._last_decision_candle = None

        # ── Day 131 fix: Risk/Paper/Live balance synchronization ──────
        # self.balance used to be a frozen snapshot taken once at boot and
        # never updated, so risk sizing (e.g. risk_usd = balance * pct)
        # silently drifted away from both the PaperTrader's evolving
        # balance and the real MT5 account balance as PnL accrued. See
        # _sync_balance() for the fix.
        self._balance_lock = threading.Lock()
        self._balance_source = "config"  # "config" | "paper" | "live"
        self._balance_deviation_warn_pct = 0.05  # log if >5% drift ignored

        # Day 131 fix: per-instance ticket-tracking dict for MT5 close
        # detection. Must be created fresh per AITrader (per symbol) —
        # see the comment above _detect_mt5_position_closes() for why a
        # shared/class-level dict here was a cross-symbol data bug.
        self._mt5_known_tickets: dict = {}

        # ── Runtime infrastructure (Day 37+ unification) ──────────────
        # The registry is optional — when supplied, AITrader pulls shared
        # singletons from it (TradeMemory, CircuitBreaker, etc.) instead of
        # constructing fresh copies, and publishes events / metrics to the
        # central bus. When absent, the trader behaves exactly as before.
        self._registry = registry
        self._bus = get_bus() if _RUNTIME_INFRA_AVAILABLE else None
        self._metrics = get_metrics() if _RUNTIME_INFRA_AVAILABLE else None

        # Audit fix: explicit MT5 sync/health tracking used by
        # _get_live_open_pairs() / _detect_mt5_position_closes() to fail
        # closed (block new entries) instead of silently assuming "no open
        # positions" or "no losses" when MT5 is unreachable.
        self._mt5_sync_ok = True
        self._mt5_disconnect_cycles = 0
        # Consecutive cycles MT5 may be unreachable before new entries are
        # blocked (existing positions are never force-closed by this).
        self._MT5_DISCONNECT_FAIL_CLOSED_THRESHOLD = int(
            os.getenv("MT5_DISCONNECT_FAIL_CLOSED_THRESHOLD", "3")
        )

        self._market = MarketAgent(self.symbol, timeframe)
        self._analysis = AnalysisAgent()
        self._decision = DecisionAgent()
        self._risk = RiskEngine(balance=balance, symbol=self.symbol)
        self._perm = TradePermission()
        self._learn = LearningAgent()
        # Prefer the registry's shared TradeMemory if available (so all
        # AITraders for different symbols share the same vector store).
        if registry is not None:
            shared_mem = registry.try_resolve("trade_memory")
            if shared_mem is not None:
                self._memory = shared_mem
            else:
                self._memory = TradeMemory(seed_rules=seed_rules)
        else:
            self._memory = TradeMemory(seed_rules=seed_rules)
        self._learning = LearningEngine()
        # Same for TraderDB — share the registry's connection if available.
        # EXECUTION-PARITY FIX: previously this ignored any `db` the caller
        # passed in and always fell back to TraderDB() (the hardcoded live
        # "database/trader.db" path) whenever no registry was supplied.
        # backtest/unified_engine.py's _make_backtest_trader() builds an
        # isolated backtest DB and passes it as `paper_trader`'s db, but
        # AITrader's own top-level self._db (used by ExecutionRouter,
        # decision logging, etc.) never received it — so a backtest run
        # was unconditionally opening a second connection to the LIVE
        # database. Now: explicit `db` argument wins, then registry, then
        # the live default (unchanged behavior for Demo/Real, which never
        # pass `db` explicitly).
        if db is not None:
            self._db = db
        elif registry is not None:
            shared_db = registry.try_resolve("db")
            self._db = shared_db if shared_db is not None else TraderDB()
        else:
            self._db = TraderDB()
        # Use a shared PaperTrader if one was passed in (e.g. from
        # AutonomousTraderSystem), otherwise create a private one.
        self._paper = paper_trader or PaperTrader(starting_balance=paper_balance, db=self._db)
        # Prefer the registry's shared mistake_analyzer.
        if registry is not None:
            shared_ma = registry.try_resolve("mistake_analyzer")
            self._mistake_analyzer = shared_ma if shared_ma is not None else (
                AdvancedMistakeAnalyzer() if AdvancedMistakeAnalyzer else None
            )
        else:
            self._mistake_analyzer = AdvancedMistakeAnalyzer() if AdvancedMistakeAnalyzer else None

        # Day 37 wiring — execution router shares THIS instance's PaperTrader
        # so paper-mode balance never drifts between router and trader.
        # Prefer the registry's shared router if available.
        if registry is not None:
            shared_router = registry.try_resolve("execution_router")
            self._router = shared_router if shared_router is not None else ExecutionRouter(
                mode=self.execution_mode, db=self._db, paper_trader=self._paper
            )
        else:
            self._router = ExecutionRouter(
                mode=self.execution_mode, db=self._db, paper_trader=self._paper
            )

        # Round-22 audit fix (B1): wire PositionManager for active trade
        # management (trailing stop, breakeven, partial close, Friday close).
        # Previously this 747-line module was completely dead — paper trades
        # got active management via PaperTrader.update_price(), but live MT5
        # trades had NO trailing/breakeven/partial-close/Friday-close at all.
        # Now: instantiate PositionManager in mt5_demo mode and call
        # poll_once() each cycle alongside the paper-trader update.
        self._position_manager = None
        if self.execution_mode in ("mt5_demo", "mt5_live"):
            try:
                from broker.position_manager import PositionManager
                # Round-30 fix N2: resolve shared OrderManager from registry
                # instead of constructing with wrong kwargs.
                # OrderManager.__init__ requires (connection, account_manager),
                # not (router=). The shared instance is already registered at
                # runtime.py:967 as "order_manager". Reuse it — never create
                # a duplicate.
                _om = None
                if self._registry is not None:
                    _om = self._registry.try_resolve("order_manager")
                if _om is None:
                    # Fallback: construct properly with connection + account_manager
                    _conn = self._resolve_mt5_connection()
                    if _conn is not None:
                        from broker.account_manager import AccountManager
                        from broker.order_manager import OrderManager
                        _acct = AccountManager(_conn)
                        _om = OrderManager(_conn, _acct)
                if _om is not None:
                    self._position_manager = PositionManager(
                        order_manager=_om,
                        journal_bridge=None,
                        on_closed=self._on_mt5_position_closed,
                        trade_memory=self._memory if hasattr(self, '_memory') else None,
                        risk_engine=self._risk if hasattr(self, '_risk') else None,
                    )
                    log.info(
                        f"[AITrader] {self.symbol} PositionManager wired for "
                        f"active management (trailing/breakeven/partial/Friday close)"
                    )
                else:
                    log.warning(
                        f"[AITrader] {self.symbol} PositionManager skipped — "
                        f"no OrderManager available (registry={self._registry is not None})"
                    )
            except Exception as _pm_e:
                log.warning(
                    f"[AITrader] {self.symbol} PositionManager init failed "
                    f"(live management unavailable): {_pm_e}"
                )
        # Circuit breaker / approval mode are global state (single JSON file
        # each) — accept a shared instance from AutonomousTraderSystem, or
        # make a private one if this AITrader is used standalone.
        self._circuit_breaker = circuit_breaker or CircuitBreaker(balance=balance)
        self._approval = approval or ApprovalMode(mode=approval_mode)
        # Correlation filter — prefer shared instance from registry.
        if registry is not None:
            shared_cf = registry.try_resolve("correlation_filter")
            self._corr_filter = shared_cf if shared_cf is not None else CorrelationFilter()
        else:
            self._corr_filter = CorrelationFilter()

        # ── Day 76 — Smart Capital Allocation Engine ──────────────────
        # The PositionSizer fuses Kelly × Volatility × Confidence ×
        # Correlation × Drawdown × Loss-streak into a single lot override
        # that runs AFTER RiskEngine picks SL/TP/entry.  When absent
        # (e.g. standalone AITrader without registry), the trader simply
        # uses RiskEngine's lot as before — fully backward-compatible.
        self._position_sizer = None
        self._live_risk_manager = None
        self._drawdown_monitor = None
        self._kill_switch = None
        if registry is not None:
            self._position_sizer = registry.try_resolve("position_sizer")
            self._live_risk_manager = registry.try_resolve("live_risk_manager")
            self._drawdown_monitor = registry.try_resolve("drawdown_monitor")
            self._kill_switch = registry.try_resolve("kill_switch")

        log.info(
            f"AITrader {self.VERSION} | {self.symbol} {timeframe} | "
            f"Mode: {self.execution_mode.upper()} | Approval: {self._approval.mode_name} | "
            f"Risk Balance: ${balance} | Paper Balance: ${self._paper.balance} | "
            f"Registry: {'yes' if registry else 'no'} | "
            f"Day76 Sizer: {'on' if self._position_sizer else 'off'}"
        )

    # ── Event / metrics helpers (Day 37+ runtime unification) ────────
    def _publish(self, channel: str, payload: dict) -> None:
        """Publish an event to the bus, if available."""
        if self._bus is not None:
            try:
                self._bus.publish(channel, payload, source=f"aitrader:{self.symbol}")
            except Exception as e:
                log.debug(f"event publish failed on {channel}: {e}")

    def _stage(self, name: str):
        """Return a timer context manager from runtime metrics, or a no-op."""
        if self._metrics is not None:
            return self._metrics.timer(name)
        # Fallback no-op context manager
        import contextlib

        class _NoOp:
            def __enter__(self):
                return None
            def __exit__(self, *a):
                return False
        return _NoOp()

    def _record_error(self, channel: str, reason: str) -> None:
        if self._metrics is not None:
            try:
                self._metrics.record_error(channel=channel)
            except Exception as e:
                log.warning(f"Suppressed exception at line 244: {e}")
                pass
        self._publish("system.error", {"channel": channel, "symbol": self.symbol, "reason": reason})

    # ── Co-founder fix: _reject helper for LiveRiskManager gate ──────
    def _reject(self, gate: str, reason: str = "", confidence: float = 0.0) -> dict:
        """Build a rejection result dict when a gate blocks the trade.

        Used by LiveRiskManager gate and other pre-trade checks that need
        to short-circuit the pipeline with a NO TRADE outcome.
        """
        log.info(f"[Trader] REJECTED by {gate}: {reason}")
        return {
            "trade_allowed": False,
            "reject_reason": f"{gate}: {reason}" if reason else gate,
            "final_action": "NO TRADE",
            "confidence": confidence,
            "gate": gate,
        }

    # ── Day 76 — Smart Capital Allocation override ────────────────────
    def _apply_advanced_sizing(
        self,
        risk_out: dict,
        dec_out: dict,
        market_out: dict,
        analysis_out: dict,
    ) -> dict:
        """Run the master PositionSizer on top of RiskEngine's base lot.

        Returns a (possibly modified) `risk_out` dict.  When the sizer is
        not wired or RiskEngine already rejected the trade, the dict is
        returned unchanged.  When the sizer rejects (Kelly negative,
        volatility too high, confidence below floor, portfolio heat
        exceeded, loss streak too long, etc.) the lot is set to 0 and
        reject_reason is filled.  When the sizer approves, lot/risk_usd/
        risk_pc are overridden with the sizer's output.

        The full breakdown (kelly/volatility/confidence/correlation/
        drawdown/streak multipliers + explanation lines) is stored under
        risk_out["position_sizing"] for downstream consumers (journal,
        telegram alerts, dashboard, audit_trail).
        """
        # Pass-through if no sizer, or RiskEngine already rejected.
        if self._position_sizer is None or not risk_out.get("approved"):
            return risk_out

        try:
            ind = market_out.get("ind_ctx", {}) or {}
            regime = market_out.get("regime", {}) or {}
            # ARCHITECTURAL FIX: risk_out no longer carries a `signal` field
            # (risk is an execution filter, not an analysis layer). Use the
            # analysis-layer decision as the authoritative direction. Falls
            # back to "WAIT" only if neither source has a direction.
            direction = dec_out.get("decision") or risk_out.get("signal") or "WAIT"
            confidence = float(dec_out.get("confidence", 0) or 0)

            # Pip value per lot — RiskEngine uses get_pip_value_usd(symbol).
            # We approximate with the same lookup so the sizer stays in sync.
            try:
                from core.constants import get_pip_value_usd
                pip_value = get_pip_value_usd(self.symbol)
            except Exception as e:
                pip_value = 10.0  # safe default for non-JPY majors

            # ATR + median ATR for volatility adjustment.
            atr = float(ind.get("atr", 0.0005) or 0.0005)
            # Median ATR — use a rolling estimate from the regime ctx if
            # available; otherwise fall back to the current ATR (ratio = 1.0
            # → NORMAL volatility, no boost/penalty).
            atr_median = float(regime.get("atr_median", atr) or atr)

            # Drawdown % — pull from the DrawdownMonitor if wired.
            current_dd = 0.0
            if self._drawdown_monitor is not None:
                try:
                    dd_status = self._drawdown_monitor.status()
                    current_dd = float(dd_status.get("current_drawdown_pct", 0.0) or 0.0)
                except Exception as e:
                    log.warning(f"Suppressed exception at line 303: {e}")
                    pass

            # Loss streak — pull from the kill switch state if available,
            # else default to 0 (no penalty).
            consecutive_losses = 0
            if self._kill_switch is not None:
                try:
                    ks_state = self._kill_switch.status() if hasattr(self._kill_switch, "status") else {}
                    consecutive_losses = int(ks_state.get("consecutive_losses", 0) or 0)
                except Exception as e:
                    log.warning(f"Suppressed exception at line 314: {e}")
                    pass

            # Open positions for correlation/heat check.
            # Audit fix (EX-2 / X-2): previously read self._paper.get_open_positions()
            # directly, which is NOT authoritative in mt5_demo mode (the bot
            # trades through MT5, not PaperTrader) — this silently blinded the
            # correlation/portfolio-heat check to real MT5 exposure. Now uses
            # _get_live_open_positions_detailed(), which shares the same
            # fail-closed MT5-vs-PaperTrader source logic as
            # _get_live_open_pairs() (used elsewhere for the same purpose).
            try:
                open_positions = self._get_live_open_positions_detailed()
            except Exception as e:
                log.warning(f"[Sizing] _get_live_open_positions_detailed() failed: {e}")
                open_positions = []

            # News window flag — if analysis flagged news as unsafe, treat
            # as news-active so volatility adjuster caps the size.
            news_ctx = analysis_out.get("news_ctx", {}) or {}
            news_active = not bool(news_ctx.get("trade_allowed", True))

            # Historical stats for Kelly — pull from TradeMemory if available.
            win_rate = None
            avg_win_r = None
            avg_loss_r = None
            trade_count = 0
            try:
                mem_ctx = self._memory.get_context_for_ai(self.symbol) if self._memory else {}
                trade_count = int(mem_ctx.get("total_trades", 0) or 0)
                wr_pct = float(mem_ctx.get("overall_win_rate", 0) or 0)
                if trade_count > 0 and wr_pct > 0:
                    win_rate = wr_pct / 100.0
                    # Default R multiples when memory doesn't expose them.
                    avg_win_r = float(mem_ctx.get("avg_win_r", 1.5) or 1.5)
                    avg_loss_r = float(mem_ctx.get("avg_loss_r", 1.0) or 1.0)
            except Exception as e:
                log.warning(f"Suppressed exception at line 350: {e}")
                pass

            # Base risk % from RiskEngine (typically 1.0%).
            base_risk_pct = float(risk_out.get("risk_pc", 1.0) or 1.0) / 100.0
            # Capital tier multiplier — Tier 3 = 1.0 by default; lower
            # tiers reduce.  We pull this from the live risk manager when
            # available, otherwise default to 1.0 (mature system).
            #
            # CRITICAL FIX: Also call check_trade_permission() — this was
            # previously dead code. The method runs 6 pre-trade checks
            # (drawdown, daily loss, consecutive losses, spread, margin,
            # correlation) and should gate every trade.
            tier_mult = 1.0
            if self._live_risk_manager is not None:
                try:
                    tier_mult = float(self._live_risk_manager.current_tier.tier_mult)
                    # Co-founder fix: corrected parameter names + all required params
                    lrm_check = self._live_risk_manager.check_trade_permission(
                        pair=self.symbol,
                        direction=direction,
                        confidence=confidence,
                        sl_pips=float(risk_out.get("sl_pips", 0) or 0),
                        tp_pips=float(risk_out.get("tp_pips", 0) or 0),
                        balance=float(self.balance),
                        atr=atr,
                        atr_median=atr_median,
                        spread_pips=float(ind.get("spread_pips", 1.5) or 1.5),
                        open_positions=self._paper.get_open_positions() if self._paper else [],
                        daily_pnl=float(getattr(self._risk, 'daily_pnl', 0.0) or 0.0),
                        weekly_pnl=float(getattr(self._risk, 'weekly_pnl', 0.0) or 0.0),
                    )
                    # Normalize return (TradePermission object or dict)
                    if hasattr(lrm_check, "allowed"):
                        _lrm_allowed = bool(lrm_check.allowed)
                        _lrm_reason = getattr(lrm_check, "reject_reason", "") or "blocked"
                    elif isinstance(lrm_check, dict):
                        _lrm_allowed = bool(lrm_check.get("allowed", True))
                        _lrm_reason = lrm_check.get("reason") or lrm_check.get("reject_reason", "blocked")
                    else:
                        _lrm_allowed = True
                        _lrm_reason = ""
                    if not _lrm_allowed:
                        # Round-6 audit fix: do NOT replace risk_out with a
                        # fresh dict from self._reject(). The old code did:
                        #     return self._reject("LiveRiskManager gate", ...)
                        # which returned a brand-new minimal dict containing
                        # only {trade_allowed, reject_reason, final_action,
                        # confidence, gate} — and LOST all the keys that
                        # RiskEngine had already computed:
                        #   - rr_ratio         (downstream trade_permission
                        #                        reads risk_out["rr_ratio"]
                        #                        and defaulted to 0 → "Min
                        #                        R:R 1:0" FAIL, blocking
                        #                        every LRM-rejected trade
                        #                        with a misleading reason)
                        #   - approved         (trade_permission reads
                        #                        risk_out["approved"] and
                        #                        defaulted to False → a
                        #                        second spurious FAIL)
                        #   - entry, sl_price, tp_price, lot, sl_pips,
                        #     tp_pips, signal, risk_usd, risk_pc, etc.
                        #
                        # The correct pattern (matching Day 76 Sizer reject
                        # at lines 518-531) is to UPDATE risk_out in place
                        # so downstream gates see the real reason and
                        # don't generate false-positive rejections.
                        log.info(
                            f"[Trader] LiveRiskManager BLOCKED trade: {_lrm_reason} "
                            f"(preserving risk_out keys: rr_ratio={risk_out.get('rr_ratio','?')}, "
                            f"approved will be set to False)"
                        )
                        risk_out["approved"] = False
                        risk_out["lot"] = 0.0
                        risk_out["risk_usd"] = 0.0
                        risk_out["risk_pc"] = 0.0
                        risk_out["reject_reason"] = f"LiveRiskManager gate: {_lrm_reason}"
                        risk_out["trade_allowed"] = False
                        risk_out["final_action"] = "NO TRADE"
                        # rr_ratio, entry, sl_price, tp_price, sl_pips,
                        # tp_pips, signal — all preserved from RiskEngine.
                        self._publish("risk.event", {
                            "kind": "live_risk_manager_reject",
                            "symbol": self.symbol,
                            "reason": _lrm_reason,
                        })
                        return risk_out
                except Exception as e:
                    log.warning(f"[Trader] LiveRiskManager check failed (non-fatal): {e}")

            # New equity high flag — paper trader balance vs starting.
            is_new_high = False
            try:
                is_new_high = float(self._paper.balance) >= float(self.balance) * 1.10
            except Exception as e:
                log.warning(f"Suppressed exception at line 391: {e}")
                pass

            # ── Run the master PositionSizer ──────────────────────────
            sizing = self._position_sizer.calculate(
                balance=float(self.balance),
                risk_pct=base_risk_pct,
                sl_pips=float(risk_out.get("sl_pips", 0) or 0),
                pip_value_per_lot=pip_value,
                confidence=confidence,
                atr=atr,
                atr_median=atr_median,
                consecutive_losses=consecutive_losses,
                tier_mult=tier_mult,
                win_rate=win_rate,
                avg_win_r=avg_win_r,
                avg_loss_r=avg_loss_r,
                trade_count=trade_count,
                pair=self.symbol,
                direction=direction,
                open_positions=open_positions,
                current_drawdown_pct=current_dd,
                is_new_equity_high=is_new_high,
                news_active=news_active,
            )

            # Persist the full breakdown for journal/telegram/dashboard.
            risk_out["position_sizing"] = sizing.to_dict()

            # Apply the sizer's verdict.
            # Day 81+ hotfix: In TEST_MODE, don't let PositionSizer reject
            # trades. The sizer checks Kelly criterion, volatility, drawdown,
            # correlation, etc. — all of which fail on a fresh account with
            # no trade history (Kelly needs win_rate, drawdown needs history,
            # etc.). In TEST_MODE, force-approve with minimum lot.
            _test_mode_sizer = False
            try:
                from config import TEST_MODE
                _test_mode_sizer = bool(TEST_MODE)
            except Exception as e:
                log.warning(f"Suppressed exception at line 431: {e}")
                pass

            if not sizing.approved and _test_mode_sizer:
                log.info(
                    f"[Day 76 Sizer] REJECTED {self.symbol} {direction} — {sizing.reject_reason} — "
                    f"BUT TEST_MODE=true, force-approving with lot=0.01"
                )
                risk_out["approved"] = True
                risk_out["lot"] = 0.01
                risk_out["risk_usd"] = round(self.balance * 0.01, 2)
                risk_out["risk_pc"] = 1.0
                risk_out["reject_reason"] = None
            elif not sizing.approved:
                log.info(
                    f"[Day 76 Sizer] REJECTED {self.symbol} {direction} — {sizing.reject_reason}"
                )
                risk_out["approved"] = False
                risk_out["lot"] = 0.0
                risk_out["risk_usd"] = 0.0
                risk_out["risk_pc"] = 0.0
                risk_out["reject_reason"] = f"Day76 Sizer: {sizing.reject_reason}"
                self._publish("risk.event", {
                    "kind": "position_sizer_reject",
                    "symbol": self.symbol,
                    "reason": sizing.reject_reason,
                })
            else:
                log.info(
                    f"[Day 76 Sizer] {self.symbol} {direction} | "
                    f"base_lot={sizing.base_lot:.2f} → final_lot={sizing.lot:.2f} | "
                    f"mult=×{sizing.final_mult:.3f} | "
                    f"risk=${sizing.risk_amount_usd:.0f} ({sizing.risk_pct:.2%})"
                )
                risk_out["lot"] = sizing.lot
                risk_out["risk_usd"] = sizing.risk_amount_usd
                risk_out["risk_pc"] = round(sizing.risk_pct * 100, 2)
                self._publish("analytics.metric", {
                    "symbol": self.symbol,
                    "metric": "position_size_multiplier",
                    "value": sizing.final_mult,
                })
        except Exception as e:
            # Audit fix (fail-open risk bug): this used to log and fall
            # through with the ORIGINAL risk_out untouched — meaning if
            # RiskEngine had already approved the trade, a crash anywhere
            # in the sizer/permission-check path silently skipped Kelly /
            # volatility / correlation / drawdown / loss-streak sizing and
            # let the trade proceed on RiskEngine's base lot alone. A risk
            # component that fails should never let a trade through it
            # never actually evaluated. We now fail CLOSED: the trade is
            # rejected this cycle and the operator sees why. This can
            # briefly reduce trade frequency if the sizer has a bug, but
            # that is the correct trade-off for a live-money risk gate.
            log.error(
                f"[Day 76 Sizer] {self.symbol} sizing failed — FAILING CLOSED "
                f"(trade rejected, not executed on unsized risk_out): {e}"
            )
            risk_out["approved"] = False
            risk_out["lot"] = 0.0
            risk_out["risk_usd"] = 0.0
            risk_out["risk_pc"] = 0.0
            risk_out["reject_reason"] = f"Day76 Sizer crashed (fail-closed): {e}"
            self._publish("risk.event", {
                "kind": "position_sizer_crash_fail_closed",
                "symbol": self.symbol,
                "error": str(e),
            })

        return risk_out

    def get_signal(self, show_chart: bool = False, auto_paper_trade: bool = True) -> dict:
        return self.run_cycle(show_chart=show_chart, auto_paper_trade=auto_paper_trade)

    def evaluate_decision_core(self, market_out: dict, session_ctx: dict, debugger=None) -> dict:
        """
        SHARED DECISION CORE — Backtest / Demo / Real execution parity.//

        This is the SINGLE implementation of Analysis -> Decision -> Risk ->
        Position Sizing -> Safety Guard (Permission + Correlation) used by
        EVERY execution mode. It is extracted verbatim from run_cycle() (the
        live/demo trading loop) so that a historical replay in backtest mode
        calls the exact same code object as live trading -- not a
        reimplementation, not an approximation.

        Per the project's execution-parity requirement, only these things
        may legitimately differ between modes, and NONE of them live in
        this method:
          - where `market_out` came from (live MT5 tick vs. a historical
            bar built from MT5 history) -- caller's responsibility
          - what happens to the resulting decision (real MT5 order vs.
            BrokerSimulator fill vs. paper trade) -- caller's responsibility
          - account balance bookkeeping (MT5 account vs. simulated ledger)

        Args:
            market_out: MarketAgentResult-shaped dict (df, ind_ctx, regime,
                regime_ctx, mtf_bias, ...). Backtest builds this from a
                historical slice using the SAME canonical indicator
                registry live uses (data/indicator_registry.py) -- see
                backtest/unified_engine.py.
            session_ctx: output of SessionAnalyzer().get_current_session()
                (or, in backtest, the historical-time equivalent).
            debugger: optional signal-pipeline debugger (live-cycle
                instrumentation); safe to leave None in backtest.

        Returns:
            dict with keys: analysis_out, dec_out, risk_out, perm_out,
            memory_ctx, pat_ctx, vec_ctx, entry, session_ctx.
        """
        log.info("[3/9] Analysis Agent...")
        analysis_out = self._analysis.run(market_out)
        if "error" in analysis_out:
            try:
                from core.trade_decision_log import log_decision
                log_decision(symbol=self.symbol, signal="NO TRADE",
                             reject_stage="analysis_agent",
                             reject_reason=f"Analysis error: {analysis_out.get('error')}")
            except Exception as e: pass
            # Return a dict with the SAME shape the success path returns
            # (see the `return {...}` at the bottom of this method) so
            # every caller can unpack it identically and just check
            # `"error" in core["analysis_out"]` — instead of returning
            # self._error_result()'s differently-shaped dict here, which
            # would break unpacking in run_cycle() / any backtest caller.
            return {
                "analysis_out": analysis_out, "dec_out": {}, "risk_out": {},
                "perm_out": {}, "memory_ctx": {}, "pat_ctx": {}, "vec_ctx": {},
                "entry": None, "session_ctx": session_ctx,
            }

        memory_ctx = self._memory.get_context_for_ai(self.symbol)
        pattern = self._extract_pattern(market_out)
        regime_str = market_out.get("regime", {}).get("regime", "")
        pat_ctx = self._memory.get_pattern_context(self.symbol, regime_str, pattern)
        # FIX (execution-parity audit — NameError found via backtest smoke
        # test, 2nd+ run against a DB with existing trade history): 'ind'
        # and 'latest_price' were both referenced below but never defined
        # in this method — only in run_cycle() (the method this was
        # extracted from), at market_out.get("ind_ctx", {}) / ind.get
        # ("close") or ind.get("price") respectively (see line ~1225).
        ind = market_out.get("ind_ctx", {}) or {}
        latest_price = ind.get("close") or ind.get("price")

        # Day 37 hotfix: initialize BEFORE the conditional so that a fresh
        # symbol with no trade history yet (total_trades == 0) never hits
        # the UnboundLocalError that used to crash every symbol's cycle.
        vec_ctx = {}

        if memory_ctx["total_trades"] > 0:
            log.info(
                f"[Memory] Trades: {memory_ctx['total_trades']} | "
                f"WR: {memory_ctx['overall_win_rate']}% | "
                f"Pattern wins: {pat_ctx.get('similar_wins', 0)} | "
                f"losses: {pat_ctx.get('similar_losses', 0)}"
            )
            if pat_ctx.get("warning"):
                log.info("[Memory] Warning: similar setups produced more losses than wins")

            vec_ctx = self._memory.get_pattern_context(
                pair=self.symbol,
                trend=ind.get("trend"),
                rsi=ind.get("rsi"),
                pattern=pattern,
                regime=regime_str,
            )

        log.info("[4/9] Decision Agent...")
        # Day 81+ hotfix: analysis_out may come from an early-return path
        # (dead zone, error, etc.) that doesn't include a "signal" key.
        # Use defensive `.get()` so we never raise KeyError here.
        # _build_result() already uses `.get("signal", {})` for the same
        # reason — this brings run_cycle() in line with that pattern.
        signal_data = analysis_out.get("signal") or {}
        # Day 81+ hotfix #2: when LLM is unavailable (all Groq keys
        # rate-limited, Gemini auth failed), master_ctx.master_entry
        # is None, so signal_data["entry"] is None. The fallback to
        # ind.get("close", 0) returns 0 if ind_ctx.close is missing,
        # which causes the RiskEngine to compute SL/TP around 0 (e.g.
        # SL=-0.00072, TP=0.00144) — a guaranteed instant stop-out.
        # Use latest_price (already extracted from market_out above)
        # as the second-tier fallback so entry is always a real price.
        entry = signal_data.get("entry") or ind.get("close") or ind.get("price") or latest_price or 0
        # Day 72 fix: normalize STRONG_BUY/STRONG_SELL to BUY/SELL for approved check
        _final_norm = analysis_out.get("final_signal", "WAIT")
        if "STRONG_BUY" in str(_final_norm):
            _final_norm = "BUY"
        elif "STRONG_SELL" in str(_final_norm):
            _final_norm = "SELL"
        placeholder_risk = {
            "approved": _final_norm in ("BUY", "SELL"),
            "lot": 0,
            "sl_pips": 0,
            "tp_pips": 0,
            "rr_ratio": 0,
            "reject_reason": None,
            # Audit fix: explicit marker so decision_agent.py doesn't have
            # to infer "placeholder vs. real rejection" from all-zero
            # fields (a real RiskEngine rejection can also be all-zero).
            "is_placeholder": True,
        }
        dec_out = self._decision.decide(market_out, analysis_out, placeholder_risk)
        self._decision.print_summary(dec_out)
        if debugger:
            debugger.record("decision",
                            dec_out.get("decision", "WAIT"),
                            f"conf={dec_out.get('confidence', 0):.0f}%")

        log.info("[5/9] Risk Engine...")
        # Day 81+ hotfix (Day 90 bugfix): sync live open positions into
        # RiskEngine so the correlation check uses authoritative PaperTrader
        # state instead of potentially-stale daily_risk.json open_pairs list.
        #
        # Day 90 bugfix history:
        #   Previously this was wrapped in a try/except that logged at DEBUG
        #   level — meaning any failure was silently swallowed and the
        #   operator never knew. Worse, RiskEngine had TWO duplicate
        #   sync_open_positions methods (Python kept the second one that
        #   never set _live_open_pairs), so even when this call "succeeded",
        #   the correlation check still fell back to the stale file state.
        #
        #   Both bugs are now fixed:
        #     1. risk_engine.py has ONE sync_open_positions method that sets
        #        BOTH _live_open_pairs (in-memory) and daily_risk.json (file)
        #     2. This call logs failures at WARNING level (visible in prod)
        #        and is no longer wrapped in a try/except that hides errors.
        #        If PaperTrader.get_open_positions() raises, we log + fall
        #        back to an empty list (no open positions = no correlation
        #        blocks), which is safer than silently using stale state.
        _live_open = self._get_live_open_pairs()

        if hasattr(self._risk, "sync_open_positions"):
            try:
                self._risk.sync_open_positions(_live_open)
            except Exception as e:
                # Day 90 bugfix: log at WARNING so silent failures are visible.
                # RiskEngine.sync_open_positions now has its own internal
                # try/except + counters, so this outer catch is just a
                # safety net for unexpected attribute errors etc.
                log.warning(
                    f"[Risk] sync_open_positions raised: {e} — "
                    f"correlation check may use stale daily_risk.json state"
                )
        else:
            # Day 90 bugfix: surface this — should never happen with the
            # merged RiskEngine, but if it does we need to know.
            log.warning(
                f"[Risk] self._risk ({type(self._risk).__name__}) has no "
                f"sync_open_positions method — correlation check will use "
                f"stale daily_risk.json state"
            )

        risk_out = self._risk.evaluate(
            signal=dec_out["decision"],
            entry=entry,
            atr=ind.get("atr", 0.0005),
            regime=market_out["regime"],
        )

        # Audit fix: fail CLOSED on new entries when we can't trust our
        # picture of live exposure/PnL. Two independent degraded-state
        # signals are checked:
        #   1. _mt5_sync_ok=False  → _get_live_open_pairs() couldn't reach
        #      MT5 this cycle, so the correlation check above may have run
        #      against an incomplete/empty position list.
        #   2. _mt5_disconnect_cycles beyond threshold → the close-detector
        #      hasn't been able to poll MT5 for several cycles in a row,
        #      so any losses that occurred during the outage haven't been
        #      folded into daily PnL — the circuit breaker is blind.
        # Existing open positions are left alone (we don't touch SL/TP or
        # force-close); only NEW entries are blocked.
        if (
            self.execution_mode == "mt5_demo"
            and risk_out.get("approved")
            and dec_out.get("decision") in ("BUY", "SELL")
        ):
            if not getattr(self, "_mt5_sync_ok", True):
                risk_out["approved"] = False
                risk_out["lot"] = 0.0
                risk_out["reject_reason"] = (
                    "MT5 position sync unavailable this cycle — refusing new "
                    "entry (fail-closed; existing positions unaffected)"
                )
                log.error(f"[Risk] {self.symbol} — {risk_out['reject_reason']}")
            elif getattr(self, "_mt5_disconnect_cycles", 0) >= self._MT5_DISCONNECT_FAIL_CLOSED_THRESHOLD:
                risk_out["approved"] = False
                risk_out["lot"] = 0.0
                risk_out["reject_reason"] = (
                    f"MT5 unreachable for {self._mt5_disconnect_cycles} consecutive "
                    f"cycles — daily PnL/circuit breaker may be stale, refusing new "
                    f"entry (fail-closed)"
                )
                log.error(f"[Risk] {self.symbol} — {risk_out['reject_reason']}")

        self._risk.print_summary(risk_out)
        if debugger:
            risk_status = "OK" if risk_out.get("approved") else "REJECT"
            # Day 81+ crash fix: reject_reason can be None (not a string),
            # so we can't do [:20] on it directly. Coerce to string first.
            _reject_reason = risk_out.get("reject_reason") or "approved"
            debugger.record("risk", risk_status,
                            f"lot={risk_out.get('lot', 0)} "
                            f"reason={str(_reject_reason)[:20]}")

        daily = self._risk.get_daily_summary()
        log.info(
            f"Daily PnL — Net: ${daily['net_usd']} | "
            f"Loss: {daily['daily_loss_pc']}% | "
            f"Limit left: {daily['limit_remaining_pc']}%"
        )

        # ── Day 76 — Smart Capital Allocation override ───────────────
        # Run the master PositionSizer on top of RiskEngine's base lot.
        # This applies Kelly × Volatility × Confidence × Correlation ×
        # Drawdown × Loss-streak multipliers and may shrink, grow, or
        # block the lot.  When the sizer is not wired, risk_out passes
        # through unchanged (fully backward-compatible).
        risk_out = self._apply_advanced_sizing(risk_out, dec_out, market_out, analysis_out)

        # Co-founder fix: SYNC dec_out with FINAL risk_out so final report
        # and learning agent see REAL lot/sl/tp/entry, not placeholder values.
        dec_out["lot"]    = risk_out.get("lot", dec_out.get("lot", 0))
        dec_out["entry"]  = risk_out.get("entry",  dec_out.get("entry"))
        dec_out["sl"]     = risk_out.get("sl_price", dec_out.get("sl"))
        dec_out["tp"]     = risk_out.get("tp_price", dec_out.get("tp"))
        dec_out["sl_pips"] = risk_out.get("sl_pips", dec_out.get("sl_pips", 0))
        dec_out["tp_pips"] = risk_out.get("tp_pips", dec_out.get("tp_pips", 0))
        dec_out["rr"]     = risk_out.get("rr_ratio", dec_out.get("rr", 0))

        # Round-30 fix F1: pass df into dec_out so trade_permission's
        # entry_quality_guardrails can access OHLCV data. Without this,
        # the guardrails code looks for decision_out.get("_df"), finds
        # None, and silently skips all 12 entry-quality checks.
        dec_out["_df"] = market_out.get("df")
        dec_out["_symbol"] = self.symbol

        log.info("[6/9] Safety Guard (Permission + Correlation)...")
        perm_out = self._perm.check(
            decision_out=dec_out,
            risk_out=risk_out,
            news_ctx=analysis_out.get("news_ctx", {}),
            session_ctx=self._session_permission_context(session_ctx),
            execution_filters=analysis_out.get("execution_filters", {}),
        )

        # Day 97+ Book Page 15: Signal Persistence Filter
        # Suppress entries when signal is flip-flopping (unstable)
        if perm_out["allowed"]:
            try:
                from core.signal_persistence import get_signal_persistence_filter
                spf = get_signal_persistence_filter()
                final_sig = perm_out.get("final_action", "WAIT")
                if not spf.is_stable(self.symbol, final_sig):
                    perm_out["allowed"] = False
                    perm_out["execution_allowed"] = False
                    perm_out["final_action"] = "NO TRADE"
                    perm_out["execution_action"] = "NO TRADE"
                    perm_out["blocked_reason"] = "Signal flip-flopping (unstable)"
                    perm_out["checks"].append({
                        "check": "Signal persistence",
                        "passed": False,
                        "detail": f"Signal flip-flopping (unstable)",
                    })
                    perm_out["total"] = perm_out.get("total", 0) + 1
                spf.record(self.symbol, final_sig, dec_out.get("confidence", 0))
            except Exception as e:
                log.warning(f"Suppressed exception at line 798: {e}")
                pass

        # Day 97+ Book Page 15: Regime Suppression
        # Suppress entries in known false-signal regimes
        if perm_out["allowed"]:
            try:
                from core.regime_suppression import get_regime_suppressor
                rs = get_regime_suppressor()
                regime_ctx = market_out.get("regime", {})
                ind_ctx = market_out.get("ind_ctx", {})
                suppress, reason = rs.should_suppress(
                    symbol=self.symbol,
                    regime=regime_ctx,
                    session=session_ctx,
                    news_ctx=analysis_out.get("news_ctx", {}),
                    ind_ctx=ind_ctx,
                )
                if suppress:
                    perm_out["allowed"] = False
                    perm_out["execution_allowed"] = False
                    perm_out["final_action"] = "NO TRADE"
                    perm_out["execution_action"] = "NO TRADE"
                    perm_out["blocked_reason"] = f"Regime suppression: {reason}"
                    perm_out["checks"].append({
                        "check": "Regime suppression",
                        "passed": False,
                        "detail": reason,
                    })
                    perm_out["total"] = perm_out.get("total", 0) + 1
            except Exception as e:
                log.warning(f"Suppressed exception at line 825: {e}")
                pass

        if self._paper.has_open_position(self.symbol, perm_out.get("final_action")):
            perm_out["allowed"] = False
            perm_out["execution_allowed"] = False
            perm_out["final_action"] = "NO TRADE"
            perm_out["execution_action"] = "NO TRADE"
            perm_out["blocked_reason"] = f"{self.symbol} {dec_out.get('decision')} already open"
            perm_out["checks"].append(
                {
                    "check": "Duplicate trade",
                    "passed": False,
                    "detail": f"{self.symbol} {dec_out.get('decision')} already open",
                }
            )
            perm_out["total"] = perm_out.get("total", 0) + 1

        # Correlation check — same underlying-risk group already has an open
        # position (e.g. EURUSD BUY blocks a fresh GBPUSD BUY). Lot size, SL
        # distance, and daily loss are already enforced inside RiskEngine
        # above; news/confidence/session/duplicate are TradePermission above;
        # this is the last piece of the Day 37 "Safety Guard" checklist.
        if perm_out["allowed"]:
            open_pairs = [t.get("pair") for t in self._paper.get_open_positions()]
            self._corr_filter.sync_open(open_pairs)
            still_allowed = self._corr_filter.allow(
                [{"symbol": self.symbol, "signal": perm_out["final_action"]}]
            )
            if not still_allowed:
                perm_out["allowed"] = False
                perm_out["execution_allowed"] = False
                perm_out["final_action"] = "NO TRADE"
                perm_out["execution_action"] = "NO TRADE"
                perm_out["blocked_reason"] = "Correlated pair group already has an open position"
                perm_out["checks"].append(
                    {
                        "check": "Correlation filter",
                        "passed": False,
                        "detail": "Correlated pair group already has an open position",
                    }
                )
                perm_out["total"] = perm_out.get("total", 0) + 1

        self._perm.print_summary(perm_out)
        if debugger:
            perm_status = "OK" if perm_out.get("allowed") else "BLOCK"
            debugger.record("permission", perm_status,
                            f"{perm_out.get('passed', 0)}/{perm_out.get('total', 0)} checks")

        # Day 81+ hotfix: log permission outcome to execution.log
        try:
            from core.execution_logger import log_permission_checked
            log_permission_checked(
                symbol=self.symbol,
                allowed=perm_out.get("allowed", False),
                passed=perm_out.get("passed", 0),
                total=perm_out.get("total", 0),
                failed_checks=[c.get("check", "?") for c in perm_out.get("checks", [])
                               if not c.get("passed", True)],
                decision=dec_out.get("decision"),
                confidence=dec_out.get("confidence", 0),
            )
        except Exception as e:
            log.warning(f"Suppressed exception at line 881: {e}")
            pass

        return {
            "analysis_out": analysis_out,
            "dec_out": dec_out,
            "risk_out": risk_out,
            "perm_out": perm_out,
            "memory_ctx": memory_ctx,
            "pat_ctx": pat_ctx,
            "vec_ctx": vec_ctx,
            "entry": entry,
            "session_ctx": session_ctx,
        }


    def run_cycle(self, show_chart: bool = False, auto_paper_trade: bool = True) -> dict:
        log.info("━" * 52)
        log.info(f"  AITrader {self.VERSION} — {self.symbol} {self.timeframe}")
        log.info("━" * 52)
        t0 = time.time()

        # Day 81+ hotfix: reset per-cycle LLM call counter so each
        # symbol cycle gets a fresh budget of MAX_LLM_CALLS_PER_CYCLE.
        # Without this, the counter would accumulate across cycles and
        # block all LLM calls after the first few symbols.
        try:
            from core.llm_key_manager import get_llm_key_manager
            get_llm_key_manager().reset_cycle_calls()
        except Exception as e:
            log.warning(f"Suppressed exception at line 497: {e}")
            pass

        # ── Day 81+ — Start signal pipeline debugger for this cycle ──
        try:
            from monitoring.signal_debugger import get_signal_debugger
            debugger = get_signal_debugger()
            debugger.start_cycle(self.symbol, self.timeframe)
        except Exception as e:
            debugger = None

        # ── Day 84+ — Trade frequency controller (early bail if daily cap hit) ──
        # PREMORTEM FIX: Equity stop — halt all trading if equity drops
        # below 90% of balance. This prevents catastrophic drawdown.
        #
        # Day 131 fix: this previously called `mt5.account_info()` directly
        # on the raw MetaTrader5 module, bypassing the shared MT5Connection
        # entirely. That created a second, unlocked access path into the
        # single process-wide MT5 terminal session — a thread-safety hazard
        # when multiple per-symbol AITraders run concurrently — and meant
        # the account snapshot was never fed back into risk sizing. Now
        # routed through the shared connection (with its lock) when
        # available, and used to keep self.balance (risk balance) in sync
        # with the real broker account instead of the frozen boot-time
        # config value. Falls back to a direct, still-guarded raw call
        # only if no shared connection is registered.
        try:
            if MT5_AVAILABLE and self.execution_mode == "mt5_demo":
                acct = None
                mt5_conn = self._resolve_mt5_connection()
                if mt5_conn is not None and hasattr(mt5_conn, "account_info"):
                    lock = getattr(mt5_conn, "MT5_LOCK", None)
                    if lock is not None:
                        with lock:
                            acct = mt5_conn.account_info()
                    else:
                        acct = mt5_conn.account_info()
                else:
                    import MetaTrader5 as mt5
                    acct = mt5.account_info()

                if acct and acct.balance > 0:
                    equity_ratio = acct.equity / acct.balance
                    # Keep the risk-sizing balance synced to the real,
                    # authoritative broker balance (see _sync_balance()).
                    self._sync_balance(live_balance=acct.balance)
                    if equity_ratio < 0.90:
                        log.critical(
                            f"[Trader] EQUITY STOP: equity=${acct.equity:.0f} "
                            f"< 90% of balance=${acct.balance:.0f} "
                            f"(ratio={equity_ratio:.3f}). HALTING all trading."
                        )
                        return {
                            "symbol": self.symbol,
                            "final_action": "WAIT",
                            "trade_allowed": False,
                            "reject_reason": f"EQUITY STOP: equity ratio {equity_ratio:.3f} < 0.90",
                            "error": "equity_stop_triggered",
                        }
            elif self.execution_mode != "mt5_demo":
                # Paper mode — PaperTrader is authoritative.
                self._sync_balance()
        except Exception as e:
            log.warning(f"[Trader] Equity check failed: {e}")

        try:
            from risk.trade_frequency import get_trade_frequency_controller
            freq_ctrl = get_trade_frequency_controller()
            if auto_paper_trade and not freq_ctrl.can_trade_now():
                if debugger:
                    debugger.record("frequency_ctrl", "BLOCK",
                                    f"Daily cap {freq_ctrl.trade_count_today()}/max")
                    debugger.record_final("NO_TRADE", "Daily trade cap reached")
                    debugger.log_cycle_summary()
                    debugger.save_to_file()
                log.warning("[Frequency] Daily trade cap hit — skipping cycle")
                try:
                    from core.trade_decision_log import log_decision
                    log_decision(symbol=self.symbol, signal="NO TRADE",
                                 reject_stage="frequency_cap",
                                 reject_reason="Daily trade cap reached")
                except Exception as e: pass
                return self._error_result("Daily trade cap reached")
        except Exception as e:
            freq_ctrl = None
            try:
                from core.trade_decision_log import log_cycle_error
                log_cycle_error(self.symbol, str(e), "frequency_ctrl_init")
            except Exception as e: pass

        session_ctx = SessionAnalyzer().get_current_session()
        latest_price = None

        log.info("[1/9] Market Agent...")
        with self._stage(f"aitrader.{self.symbol}.market"):
            market_out = self._market.run()
        if "error" in market_out:
            # Don't send Telegram alert for market fetch failures — they're
            # common (market closed, symbol temporarily unavailable) and
            # would spam the user. Just log locally.
            _is_unavailable = market_out.get("skipped") or "symbol_unavailable" in market_out.get("error", "")
            if _is_unavailable:
                log.debug(f"[Market] {self.symbol} skipped — not available on broker")
            else:
                log.warning(f"[Market] {self.symbol} data fetch failed — skipping this cycle")
            try:
                from core.trade_decision_log import log_decision
                log_decision(symbol=self.symbol, signal="NO TRADE",
                             reject_stage="market_data",
                             reject_reason=f"Market data fetch failed: {market_out.get('error')}")
            except Exception as e: pass
            if debugger:
                debugger.record("market_data", "ERROR", market_out.get("error", "fetch_failed"))
                debugger.record_final("NO_TRADE", "Market data fetch failed")
                debugger.log_cycle_summary()
                debugger.save_to_file()
            # Return WITHOUT "error" key for unavailable symbols — they must
            # NOT count as cycle errors for recovery pause logic.
            if _is_unavailable:
                return {
                    "symbol": self.symbol,
                    "timeframe": self.timeframe,
                    "version": self.VERSION,
                    "final_action": "NO TRADE",
                    "trade_allowed": False,
                    "skipped_unavailable": True,
                }
            return self._error_result(f"Market Agent: {market_out['error']}")

        # Record market data success
        if debugger:
            ind_ctx = market_out.get("ind_ctx", {}) or {}
            # Round-5 audit fix: Indicators.get_ai_context() uses key "price"
            # (NOT "close") — confirmed in data/indicators.py:89-90. The
            # old `ind_ctx.get('close', '?')` always returned '?' because
            # the key never exists. Now uses 'price' (the actual key),
            # falling back to 'close' for any caller that has a raw OHLCV
            # row instead of an ind_ctx.
            debugger.record("market_data", "OK",
                            f"price={ind_ctx.get('price', ind_ctx.get('close', '?'))} "
                            f"trend={ind_ctx.get('trend', '?')}")

        ind = market_out.get("ind_ctx", {})
        # Day 81+ hotfix: Indicators.get_ai_context() uses "price" key, not "close"
        latest_price = ind.get("close") or ind.get("price")

        # PREMORTEM FIX: Data quality validation — reject bad ticks.
        # If the latest price is zero/negative or NaN, skip this cycle.
        if latest_price is not None:
            try:
                price_val = float(latest_price)
                if price_val <= 0 or price_val != price_val:  # NaN check
                    log.warning(f"[Trader] BAD PRICE detected: {latest_price} — skipping cycle")
                    return {
                        "symbol": self.symbol,
                        "final_action": "WAIT",
                        "trade_allowed": False,
                        "reject_reason": f"Bad price: {latest_price}",
                    }
            except (ValueError, TypeError):
                log.warning(f"[Trader] INVALID PRICE type: {latest_price} — skipping cycle")
                return {
                    "symbol": self.symbol,
                    "final_action": "WAIT",
                    "trade_allowed": False,
                    "reject_reason": f"Invalid price: {latest_price}",
                }

        candle_time = self._extract_candle_time(market_out)
        closed_now = []

        if auto_paper_trade and latest_price:
            # Day 102+ hotfix: pass candle high/low so PaperTrader can
            # detect intra-candle SL/TP hits (matching real broker fills).
            # Previously only `close` was passed, so any TP hit during the
            # candle that finished below TP was missed — producing
            # optimistic paper P&L and corrupting ML labels.
            _candle_high = ind.get("high") or latest_price
            _candle_low = ind.get("low") or latest_price
            closed_now = self._paper.update_price(
                self.symbol, latest_price,
                high=_candle_high, low=_candle_low,
            )
        closed_processed = self._process_closed_trades(closed_now)

        # Round-22 audit fix (B1): poll PositionManager for active MT5 trade
        # management (trailing stop, breakeven, partial close, Friday close).
        # Only runs in mt5_demo/mt5_live mode — paper trades are already
        # managed by PaperTrader.update_price() above.
        if self._position_manager is not None:
            try:
                mt5_closed = self._position_manager.poll_once()
                if mt5_closed:
                    closed_processed.extend(mt5_closed)
                    log.info(
                        f"[AITrader] {self.symbol} PositionManager detected "
                        f"{len(mt5_closed)} MT5 close event(s)"
                    )
            except Exception as _pm_poll_e:
                log.debug(f"[AITrader] PositionManager poll skipped: {_pm_poll_e}")

        # [2/9] Circuit Breaker Gate — existing positions above still get
        # monitored (SL/TP/timeout) even while tripped; only NEW entries block.
        log.info("[2/9] Circuit Breaker Gate...")
        self._circuit_breaker.reset_daily()
        cb_check = self._circuit_breaker.allow_trade()
        if not cb_check["allowed"]:
            log.warning(f"[CircuitBreaker] {cb_check['mode']} — {cb_check['reason']}")
            if debugger:
                debugger.record("circuit_breaker", "BLOCK", f"{cb_check['mode']}: {cb_check['reason']}")
                debugger.record_final("NO_TRADE", f"CircuitBreaker: {cb_check['reason']}")
                debugger.log_cycle_summary()
                debugger.save_to_file()
            self._publish("risk.circuit_breaker", {
                "symbol": self.symbol, "mode": cb_check["mode"], "reason": cb_check["reason"],
            })
            self._publish("risk.event", {
                "kind": "circuit_breaker", "symbol": self.symbol,
                "mode": cb_check["mode"], "reason": cb_check["reason"],
            })
            result = self._monitor_only_result(
                price=latest_price,
                candle_time=candle_time,
                session_ctx=session_ctx,
                elapsed=round(time.time() - t0, 1),
                closed_trades=closed_processed,
            )
            result["reject_reason"] = f"Circuit breaker [{cb_check['mode']}]: {cb_check['reason']}"
            try:
                from core.trade_decision_log import log_decision
                log_decision(symbol=self.symbol, signal="NO TRADE",
                             reject_stage="circuit_breaker",
                             reject_reason=f"[{cb_check['mode']}] {cb_check['reason']}")
            except Exception as e: pass
            self._print_final(result)
            return result
        if debugger:
            debugger.record("circuit_breaker", "OK", "Trade allowed")

        # Day 72 fix: Removed candle dedup check that was blocking ALL pairs.
        # The old logic compared candle_time with _last_decision_candle and
        # skipped analysis if they matched. But since all pairs share the
        # same candle time (e.g. 17:30), the first pair's cycle would set
        # _last_decision_candle and ALL subsequent pairs would be skipped.
        # Now we always run the full analysis pipeline. Duplicate trade
        # prevention is handled by TradePermission + duplicate check in
        # the Safety Guard step.
        self._last_decision_candle = candle_time

        # ── Audit fix: data staleness + candle-close gates ────────────
        # core/production_hardening.py shipped check_data_staleness() and
        # is_candle_closed() specifically to catch two classic false-signal
        # sources (trading on stale ticks / trading on a still-forming
        # candle), but neither was ever called from the live pipeline.
        # Placed here — AFTER open-position monitoring (_paper.update_price
        # above) and the Circuit Breaker gate — so it follows the same
        # "existing positions still get monitored, only NEW entries block"
        # rule already documented for the Circuit Breaker Gate just above.
        # Each check is independently wrapped so a failure here can never
        # block trading; worst case it no-ops and behavior is unchanged
        # from before this fix.
        df_for_checks = market_out.get("df")
        if df_for_checks is not None and len(df_for_checks.index) > 0:
            # ── Round-3 audit fix: hoist tf_code normalization ABOVE the
            # staleness check so BOTH the staleness check and the
            # candle-close check below can use it. Previously _tf_map
            # was only computed inside the candle-close block, which
            # forced the staleness check above to use a hardcoded
            # `max_age_sec=120` (2 min) — catastrophically tight on M15
            # (candle updates every 900s) and triggered "STALE DATA"
            # on every single cycle, blocking all new-entry analysis.
            _tf_map = {
                "1M": "M1", "5M": "M5", "15M": "M15", "30M": "M30",
                "1H": "H1", "4H": "H4", "1D": "D1",
            }
            tf_code = _tf_map.get(self.timeframe.upper(), self.timeframe.upper())

            try:
                from core.production_hardening import (
                    check_data_staleness,
                    compute_staleness_threshold,
                )
                # ── Round-3 audit fix: timeframe-aware staleness threshold ──
                # Was: `max_age_sec=120` (hardcoded 2 min).
                # Now: `compute_staleness_threshold(tf_code)` returns
                # `max(tf_sec + 60s buffer, 120s floor)`:
                #   M15 → 960s  (16 min — fixes the 183s "stale" bug)
                #   H1  → 3660s (61 min)
                #   D1  → 86460s (~24h)
                # See compute_staleness_threshold() docstring for full table.
                staleness_max_age = compute_staleness_threshold(tf_code)
                staleness = check_data_staleness(df_for_checks, max_age_sec=staleness_max_age)
                if staleness.get("is_stale"):
                    log.warning(
                        f"[Trader] STALE DATA for {self.symbol}: "
                        f"{staleness.get('reason')} (threshold={staleness_max_age}s, tf={tf_code}) "
                        f"— skipping new-entry analysis"
                    )
                    try:
                        from core.trade_decision_log import log_decision
                        log_decision(symbol=self.symbol, signal="NO TRADE",
                                     reject_stage="data_staleness",
                                     reject_reason=staleness.get("reason", "Stale data"))
                    except Exception:
                        pass
                    result = self._monitor_only_result(
                        price=latest_price, candle_time=candle_time,
                        session_ctx=session_ctx, elapsed=round(time.time() - t0, 1),
                        closed_trades=closed_processed,
                    )
                    result["reject_reason"] = f"Stale data: {staleness.get('reason')}"
                    self._print_final(result)
                    return result
            except Exception as e:
                log.debug(f"[Trader] Data staleness check skipped (non-fatal): {e}")

            try:
                from core.production_hardening import (
                    is_candle_closed,
                    get_last_closed_bar_time,
                )
                # tf_code was already normalized above (before the
                # staleness check) so both checks share it. The
                # _tf_map block that used to live here has been
                # hoisted up so the staleness threshold could be
                # computed from the same resolved timeframe.
                #
                # ── Round-4 audit fix: use LAST CLOSED bar, not forming bar ──
                # MT5's copy_rates_from_pos returns N rows where the LAST
                # row (df.index[-1]) is the CURRENTLY-FORMING candle. Its
                # close_time is in the FUTURE, so is_candle_closed() on it
                # ALWAYS returns False — blocking all new-entry analysis
                # on every cycle, every pair.
                #
                # get_last_closed_bar_time() walks backward and returns
                # the first bar whose close_time <= now. That's the last
                # fully-closed bar — the one whose OHLCV is final and
                # safe to analyze.
                last_bar_time, _row_idx, _forming_bar = get_last_closed_bar_time(
                    df_for_checks, tf_code
                )
                if last_bar_time is None:
                    log.warning(
                        f"[Trader] {self.symbol} get_last_closed_bar_time returned "
                        f"None — skipping candle-close check (df empty?)"
                    )
                else:
                    if hasattr(last_bar_time, "to_pydatetime"):
                        last_bar_time = last_bar_time.to_pydatetime()

                    # ── P1 audit: diagnostic log of the exact timestamp ──
                    # we're about to feed into is_candle_closed(). This
                    # makes it trivial to spot the broker-tz bug: if
                    # `last_bar_time` shows a wall-clock 2-3 hours ahead
                    # of UTC `now`, the fetcher is mislabeling broker
                    # server time as UTC and the trader will block
                    # forever on "candle still forming".
                    from datetime import datetime as _dt, timezone as _tz
                    _now_utc = _dt.now(_tz.utc)
                    _bar_for_log = last_bar_time
                    if _bar_for_log.tzinfo is None:
                        _bar_for_log = _bar_for_log.replace(tzinfo=_tz.utc)
                    _delta_sec = (_now_utc - _bar_for_log).total_seconds()
                    log.info(
                        f"[Trader] {self.symbol} candle-close check | "
                        f"tf={tf_code} | "
                        f"last_closed_bar={last_bar_time.isoformat()} (row={_row_idx}) | "
                        f"forming_bar_skipped={_forming_bar.isoformat() if _forming_bar else 'None'} | "
                        f"tzinfo={'naive' if last_bar_time.tzinfo is None else str(last_bar_time.tzinfo)} | "
                        f"now_utc={_now_utc.isoformat()} | "
                        f"delta={_delta_sec:.0f}s "
                        f"({'FUTURE' if _delta_sec < 0 else 'past'})"
                    )
                    if _delta_sec < -60:
                        log.critical(
                            f"[Trader] {self.symbol} last_bar_time is "
                            f"{abs(_delta_sec):.0f}s in the FUTURE. The fetcher "
                            f"is returning broker server time mislabeled as UTC. "
                            f"Set MT5_BROKER_TZ_OFFSET_HOURS in .env to the "
                            f"broker's GMT offset (2 or 3)."
                        )

                    closed_check = is_candle_closed(tf_code, last_bar_time)
                    if not closed_check.get("is_closed", True):
                        log.info(
                            f"[Trader] {self.symbol} candle still forming "
                            f"({closed_check.get('reason')}) — skipping new-entry analysis"
                        )
                        try:
                            from core.trade_decision_log import log_decision
                            log_decision(symbol=self.symbol, signal="NO TRADE",
                                         reject_stage="candle_not_closed",
                                         reject_reason=closed_check.get("reason", "Candle forming"))
                        except Exception:
                            pass
                        result = self._monitor_only_result(
                            price=latest_price, candle_time=candle_time,
                            session_ctx=session_ctx, elapsed=round(time.time() - t0, 1),
                            closed_trades=closed_processed,
                        )
                        result["reject_reason"] = f"Candle not closed: {closed_check.get('reason')}"
                        self._print_final(result)
                        return result
            except Exception as e:
                log.debug(f"[Trader] Candle-close check skipped (non-fatal): {e}")

        log.info("[3/9] Analysis Agent...")
        _core = self.evaluate_decision_core(market_out, session_ctx, debugger)
        analysis_out = _core["analysis_out"]
        dec_out      = _core["dec_out"]
        risk_out     = _core["risk_out"]
        perm_out     = _core["perm_out"]
        memory_ctx   = _core["memory_ctx"]
        pat_ctx      = _core["pat_ctx"]
        vec_ctx      = _core["vec_ctx"]
        entry        = _core["entry"]
        if "error" in analysis_out:
            try:
                from core.trade_decision_log import log_decision
                log_decision(symbol=self.symbol, signal="NO TRADE",
                             reject_stage="analysis_agent",
                             reject_reason=f"Analysis error: {analysis_out.get('error')}")
            except Exception as e: pass
            return self._error_result(f"Analysis Agent: {analysis_out['error']}")

        log.info("[7/9] Learning Agent...")
        # ── ARCHITECTURAL FIX (institutional refactor) ───────────────
        # Previously: `dec_out["decision"] = "WAIT"` (or _final_action)
        # when permission was denied. This OVERWROTE the analysis-layer
        # verdict with the execution-layer verdict, destroying the audit
        # trail. The learning agent then "learned" from post-gate signals
        # instead of analysis signals — a fundamental data contamination.
        #
        # Now: `dec_out["decision"]` STAYS as the analysis-layer verdict
        # (BUY/SELL/WAIT — what the analysts actually said). A NEW field
        # `dec_out["execution_action"]` carries the post-gate verdict
        # (BUY/SELL/NO TRADE — what the system will actually do). The
        # learning agent, audit trail, and dashboard read BOTH fields
        # and can distinguish "analysis said X, execution did Y".
        # ──────────────────────────────────────────────────────────────
        _raw_signal = dec_out.get("decision", "WAIT")
        _final_action = perm_out.get("final_action", perm_out.get("execution_action", "WAIT"))
        dec_out["raw_signal"] = _raw_signal
        dec_out["execution_action"] = _final_action
        if _final_action in ("NO TRADE", "WAIT", None, ""):
            # DETAILED REJECTION LOGGING
            _perm_checks = perm_out.get("checks", [])
            for _chk in _perm_checks:
                if not _chk.get("passed", True):
                    log.warning(
                        f"[REJECTION] {_chk.get('check', 'Unknown')} | "
                        f"detail={_chk.get('detail', 'N/A')}"
                    )
            log.warning(
                f"[SIGNAL REJECTED] Analysis={_raw_signal} → Execution={_final_action} | "
                f"Fusion Score={dec_out.get('confidence', 0)}% | "
                f"Risk Approved={risk_out.get('approved', False)} | "
                f"R:R={risk_out.get('rr_ratio', 0)} | "
                f"Session={session_ctx.get('current_session', 'N/A') if session_ctx else 'N/A'} | "
                f"Blocked by={perm_out.get('blocked_reason', 'multiple checks')}"
            )
            # Execution is gated — but analysis verdict is PRESERVED in
            # dec_out["decision"]. Only execution_action reflects the block.
            dec_out.setdefault("gated_by_permission", True)
            dec_out.setdefault("blocked_reason", perm_out.get("blocked_reason"))
            log.info(
                f"[Learning] Signal sync: analysis={_raw_signal} → "
                f"execution={_final_action} (GATED — analysis verdict preserved)"
            )
        else:
            log.info(
                f"[Learning] Signal sync: analysis={_raw_signal} → "
                f"execution={_final_action} (ALLOWED)"
            )

        try:
            decision_id = self._learn.save_decision(dec_out, analysis_out, market_out)
        except Exception as e:
            log.warning(f"[Learning] save_decision failed: {e}")
            decision_id = None
        stats = self._learn.get_performance_stats()

        self._save_all(market_out, analysis_out, risk_out, dec_out, perm_out)

        elapsed = round(time.time() - t0, 1)
        result = self._build_result(
            market_out,
            analysis_out,
            dec_out,
            risk_out,
            perm_out,
            stats,
            elapsed,
            session_ctx=session_ctx,
            candle_time=candle_time,
            closed_trades=closed_processed,
        )

        trade_id = self._memory.on_signal_generated(result, market_out, analysis_out)
        if trade_id:
            result["trade_id"] = trade_id
            log.info(f"[Memory] Trade #{trade_id} saved")

        # Day 102+ hotfix: stash decision_id on the result so the execution
        # layer can embed it in the trade's context dict. When the trade
        # eventually closes, _process_closed_trades can use it to backfill
        # the LearningAgent JSON via update_outcome(id, ...) — instead of
        # falling back to the symbol-based search.
        if decision_id is not None:
            result["decision_id"] = decision_id

        result["memory_context"] = vec_ctx
        result["pattern_context"] = pat_ctx
        result["approval_mode"] = self._approval.mode_name

        log.info("[8/9] Approval Gate...")
        approved_to_execute = False
        if auto_paper_trade and result["trade_allowed"]:
            approval_out = self._approval.process(
                {
                    "symbol": self.symbol,
                    "final_action": result["final_action"],
                    "confidence": result["confidence"],
                    "entry": result["entry"],
                    "sl": result["sl"],
                    "tp": result["tp"],
                    "lot": result["lot"],
                    "rr": result["rr"],
                    "llm_analysis": result.get("llm_analysis", ""),
                }
            )
            approved_to_execute = approval_out["proceed"]

            # Day 81+ hotfix: log every approval decision to execution.log
            try:
                from core.execution_logger import log_approval_processed
                log_approval_processed(
                    symbol=self.symbol,
                    proceed=approved_to_execute,
                    mode=approval_out.get("mode", 0),
                    action=approval_out.get("action", "unknown"),
                    final_action=result["final_action"],
                    confidence=result["confidence"],
                )
            except Exception as e:
                log.warning(f"Suppressed exception at line 943: {e}")
                pass

            if approval_out["action"] == "WAIT_APPROVAL":
                result["pending_approval_id"] = approval_out.get("pending_id")
                # ApprovalMode.process() builds the human-readable summary but
                # can't safely send it itself (its telegram_bot.send_message()
                # call would be an un-awaited coroutine) — send it the same
                # async-safe way every other Telegram alert goes out below.
                if self.notifier:
                    self._run_async(self.notifier.send_message(approval_out["message"]))

            if not approved_to_execute:
                result["reject_reason"] = approval_out.get("message", result.get("reject_reason"))

        log.info("[9/9] Execution + Alerts...")
        if approved_to_execute:
            with self._stage(f"aitrader.{self.symbol}.execute"):
                trade = self._router.execute(
                    {
                        "decision": result["final_action"],
                        "symbol": self.symbol,
                        "entry": result["entry"],
                        "sl": result["sl"],
                        "tp": result["tp"],
                        "lot": result["lot"],
                        "confidence": result["confidence"],
                        "rr": result["rr"],
                        # Day 81+ hotfix: pass trade_allowed + risk_approved
                        # so ExecutionRouter can hard-block on permission bypass.
                        "trade_allowed": result["trade_allowed"],
                        "risk_approved": result.get("risk_approved", True),
                        "timeframe": self.timeframe,
                        # BUGFIX: these were already computed in `result` by
                        # _build_result() (trend/rsi/regime/session — same
                        # data that lands correctly in the `analysis` table
                        # every cycle) but never forwarded here. journal_bridge
                        # .log_mt5_open() reads exactly these keys off
                        # decision_result to fill the trades table's
                        # pattern/regime/trend/rsi/session columns — without
                        # them every real MT5 trade recorded NULL for all
                        # five, permanently losing the strategy context that
                        # produced that specific trade.
                        "trend": result.get("trend"),
                        "rsi": result.get("rsi"),
                        "regime": result.get("regime"),
                        "session": result.get("session"),
                        "pattern": (result.get("pattern_context") or {}).get("pattern")
                                   or result.get("rule_signal"),
                        "mtf_bias": result.get("mtf_bias"),
                        "llm_signal": result.get("llm_signal"),
                    }
                )
            if trade:
                result["paper_trade_id"] = trade.get("id")
                # Day 96 bugfix: the ExecutionRouter returns "ticket" in its
                # response dict for every successful fill (paper, simulated,
                # or real MT5), but this value was never copied onto `result`
                # — so result.get("ticket") was always None downstream,
                # which is why trade_decisions.jsonl showed "ticket: null"
                # for every trade including ones that filled successfully.
                result["ticket"] = trade.get("ticket")
                result["paper_balance"] = self._paper.balance
                self._risk.record_trade_open(self.symbol)
                self._notify_trade_open(trade, result, dec_out)

                # Round-30: Register MT5 ticket → DB trade_id in PositionManager.
                # This was NEVER called before — PositionManager._ticket_to_db_id
                # was always empty for MT5 trades, making close detection unable
                # to find the DB trade to update.
                _mt5_ticket = trade.get("ticket")
                _db_trade_id = trade.get("id")
                if _mt5_ticket is not None and _db_trade_id is not None and self._position_manager is not None:
                    try:
                        self._position_manager.register_open(int(_mt5_ticket), int(_db_trade_id))
                    except Exception as e:
                        # P1 fix: was `except Exception: pass` — silently lost ticket→DB mapping.
                        # Now logs + writes to orphan spool for reconcile_on_startup recovery.
                        log.warning(f"[AITrader] register_open failed for ticket={_mt5_ticket}, "
                                    f"db_trade_id={_db_trade_id}: {e}")
                        try:
                            import json as _json
                            import time as _time
                            import os as _os
                            _os.makedirs("memory", exist_ok=True)
                            with open("memory/orphan_trade_spool.jsonl", "a") as _f:
                                _f.write(_json.dumps({
                                    "ticket": _mt5_ticket,
                                    "db_trade_id": _db_trade_id,
                                    "symbol": self.symbol,
                                    "reason": "register_open_failed",
                                    "error": str(e),
                                    "ts": _time.time(),
                                }) + "\n")
                        except Exception:
                            pass  # best-effort spool; do not crash the trade path
                # ── Day 37+ runtime unification ─────────────────────────
                # Publish trade.execution + signal.generated events so the
                # bus subscribers (alerts, dashboard, webhook, audit_trail)
                # all see this trade without AITrader knowing about them.
                self._publish("trade.execution", {
                    "symbol": self.symbol,
                    "decision": result["final_action"],
                    "entry": result["entry"],
                    "sl": result["sl"],
                    "tp": result["tp"],
                    "lot": result["lot"],
                    "confidence": result["confidence"],
                    "rr": result["rr"],
                    "trade_id": trade.get("id"),
                    "execution_mode": self.execution_mode,
                    "timeframe": self.timeframe,
                })
                self._publish("signal.generated", {
                    "symbol": self.symbol,
                    "signal": result["final_action"],
                    "confidence": result["confidence"],
                    "source": "aitrader",
                })
                # ── Day 67: Confluence Engine Telegram alert ──────────────
                # If a confluence decision was computed by AnalysisAgent,
                # send the rich multi-factor signal alert to Telegram.
                try:
                    confluence_ctx = analysis_out.get("confluence") if isinstance(analysis_out, dict) else None
                    if confluence_ctx and confluence_ctx.get("should_trade"):
                        from intelligence.confluence_engine import ConfluenceDecision
                        decision = ConfluenceDecision(
                            pair=confluence_ctx.get("pair", self.symbol),
                            timeframe=confluence_ctx.get("timeframe", self.timeframe),
                            direction=confluence_ctx.get("direction", result["final_action"]),
                            confidence=confluence_ctx.get("confidence", result["confidence"]),
                            setup_quality=confluence_ctx.get("setup_quality", "A"),
                            aligned_factors=confluence_ctx.get("aligned_factors", 0),
                            total_factors=confluence_ctx.get("total_factors", 0),
                            buy_score=confluence_ctx.get("buy_score", 0),
                            sell_score=confluence_ctx.get("sell_score", 0),
                            net_score=confluence_ctx.get("net_score", 0),
                            factors=confluence_ctx.get("factors", []),
                            market_story=confluence_ctx.get("market_story", ""),
                            risks=confluence_ctx.get("risks", []),
                        )
                        alert_msg = decision.to_telegram_alert()
                        if alert_msg and self.notifier:
                            self._run_async(self.notifier.send_message(alert_msg))
                except Exception as e:
                    log.debug(f"[Day 67] confluence telegram alert failed: {e}")

                if self._metrics is not None:
                    try:
                        self._metrics.inc("trades.opened")
                        self._metrics.set_gauge(f"paper.balance.{self.symbol}", self._paper.balance)
                    except Exception as e:
                        log.warning(f"Suppressed exception at line 1046: {e}")
                        pass
            else:
                # Router returned None — broker failure or rejection.
                self._publish("broker.failure", {
                    "symbol": self.symbol,
                    "reason": "execution_router returned None",
                    "decision": result["final_action"],
                })
                self._record_error("broker", f"execution returned None for {self.symbol}")

        if show_chart:
            ChartEngine(self.symbol, self.timeframe).create_full_chart(
                df=market_out["df"],
                support_zones=analysis_out["sr_result"]["support_zones"],
                resistance_zones=analysis_out["sr_result"]["resistance_zones"],
                patterns_df=market_out["df"],
                show=True,
                save_html="data/chart.html",
            )

        self._print_final(result)

        # ── Phase 7: Structured per-decision audit log ──
        # Emits a fixed-format, grep-parseable block with all individual
        # confidence components (technical, ML, RL, LLM, master, SMC,
        # session, fusion, confluence) plus bonuses, penalties, and the
        # exact reason chain.  Pure logging — never mutates state.
        try:
            from utils.decision_logger import log_decision_block
            log_decision_block(result, analysis_out)
        except Exception as e:
            log.warning(f"DecisionLogger error (non-fatal): {e}")

        # ── Day 81+ — Record final outcome in signal debugger ──
        if debugger:
            final_action = result.get("final_action") or result.get("decision", "WAIT")
            reject_reason = result.get("reject_reason", "")
            debugger.record_final(final_action, reject_reason)
            debugger.log_cycle_summary()
            debugger.save_to_file()

        # ── Day 81+ hotfix: log EVERY trade decision to memory/trade_decisions.jsonl ──
        # This is the single source of truth for "why didn't the bot trade?".
        # The operator can tail this file to see exactly what happened in
        # each cycle: signal, confidence, taken/not-taken, reject stage+reason.
        try:
            from core.trade_decision_log import log_decision as _log_dec
            _final_action = result.get("final_action") or result.get("decision", "WAIT")
            _taken = bool(result.get("paper_trade_id") or result.get("ticket"))
            # Determine reject_stage from final_action + reject_reason
            _reject_stage = ""
            _reject_reason = result.get("reject_reason") or ""
            if not _taken and _reject_reason:
                if "Circuit breaker" in _reject_reason:
                    _reject_stage = "circuit_breaker"
                elif "Correlation" in _reject_reason:
                    _reject_stage = "risk_correlation"
                elif "Daily loss" in _reject_reason or "loss limit" in _reject_reason:
                    _reject_stage = "risk_daily_loss"
                elif "Max open" in _reject_reason:
                    _reject_stage = "risk_max_open"
                elif "Insufficient margin" in _reject_reason:
                    _reject_stage = "risk_margin"
                elif "Risk rejected" in _reject_reason or "Risk approved" in str(result.get("failed_checks", "")):
                    _reject_stage = "risk_engine"
                elif "News" in _reject_reason or "news" in _reject_reason:
                    _reject_stage = "news_filter"
                elif "Session" in _reject_reason or "session" in _reject_reason:
                    _reject_stage = "session"
                elif "WAIT_APPROVAL" in str(result.get("pending_approval_id")):
                    _reject_stage = "approval_mode_2"
                elif "execution" in _reject_reason.lower() or "router" in _reject_reason.lower():
                    _reject_stage = "execution_router"
                elif "market closed" in _reject_reason.lower() or "Hard Stop" in _reject_reason:
                    _reject_stage = "absolute_safety"
                elif _final_action in ("WAIT", "NO TRADE"):
                    _reject_stage = "decision_agent"
                else:
                    _reject_stage = "unknown"
            _log_dec(
                symbol=self.symbol,
                signal=_final_action,
                confidence=result.get("confidence", 0),
                timeframe=self.timeframe,
                taken=_taken,
                reject_stage=_reject_stage,
                reject_reason=_reject_reason,
                lot=result.get("lot"),
                entry=result.get("entry"),
                sl=result.get("sl"),
                tp=result.get("tp"),
                ticket=result.get("ticket"),
            )
        except Exception as e:
            log.warning(f"Suppressed exception at line 1130: {e}")
            pass

        # ── Day 84+ — Record trade in frequency controller ──
        if freq_ctrl:
            _final_for_freq = result.get("final_action")
            if _final_for_freq in ("BUY", "SELL"):
                try:
                    freq_ctrl.record_trade(symbol=self.symbol, direction=_final_for_freq)
                except Exception as e:
                    log.warning(f"Suppressed exception at line 1140: {e}")
                    pass

        # ── Day 37+ professional: log every decision to trade journal ──
        try:
            from core.professional_tools import get_trade_journal, JournalEntry
            journal = get_trade_journal()
            cycle = journal.next_cycle()
            entry = JournalEntry(
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                cycle=cycle,
                symbol=self.symbol,
                timeframe=self.timeframe,
                session=result.get("session") or "UNKNOWN",
                decision=result.get("final_action", "WAIT"),
                confidence=float(result.get("confidence", 0) or 0),
                entry=result.get("entry"),
                sl=result.get("sl"),
                tp1=result.get("tp"),
                tp2=None,
                lot=float(result.get("lot", 0) or 0),
                rr_ratio=float(result.get("rr", 0) or 0),
                risk_usd=float(result.get("risk_usd", 0) or 0),
                reason=str(result.get("reject_reason") or "")[:200],
                llm_analysis=str(result.get("llm_analysis", ""))[:500],
                master_analysis=str(result.get("master_analysis", ""))[:500],
            )
            journal.log_decision(entry)
        except Exception as e:
            log.debug(f"[Journal] log_decision failed: {e}")
        return result

    def monitor_open_trades(self, price: float = None) -> list[dict]:
        if price is None:
            market_out = self._market.run()
            price = market_out.get("ind_ctx", {}).get("close")
            if price is None:
                log.warning("[PaperTrader] No price available to check open trades")
                return []
        closed_now = self._paper.update_price(self.symbol, price)
        return self._process_closed_trades(closed_now)

    def check_open_paper_trades(self, price: float = None) -> list[dict]:
        return self.monitor_open_trades(price=price)

    def close_trade(self, trade_id: int, result: str, pnl: float):
        self._memory.on_trade_closed(trade_id, result, pnl)
        self._risk.record_trade_close(self.symbol, pnl)
        self._circuit_breaker.record_result(result, pnl)
        # Day 131 fix: re-sync risk balance immediately on close instead of
        # waiting for the next cycle's equity check, so a losing/winning
        # streak is reflected in position sizing right away.
        if self.execution_mode != "mt5_demo":
            self._sync_balance()
        log.info(f"Trade #{trade_id} closed: {result} | PnL: ${pnl}")

    def get_paper_dashboard(self) -> dict:
        return self._paper.get_dashboard()

    def print_paper_dashboard(self) -> None:
        self._paper.print_dashboard()

    def get_learning_report(self):
        self._learning.print_report()

    def get_memory_stats(self):
        self._memory.print_stats()

    def _resolve_mt5_connection(self):
        """Resolve the single shared MT5Connection from the DI registry.

        Day 131 fix: prefer the registry instance actually injected into
        this AITrader (self._registry, wired by AutonomousTraderSystem /
        core.runtime). If none was injected (e.g. AITrader constructed
        standalone), fall back to the process-wide ServiceRegistry
        singleton from core.service_registry — never a separate/duplicate
        registry module. This guarantees every caller resolves the exact
        same MT5Connection instance that core.runtime.boot_market()
        registered, eliminating the "No shared MT5Connection in registry"
        false negative caused by a previously-used, unrelated registry
        implementation.
        """
        reg = self._registry
        if reg is None:
            try:
                from core.service_registry import get_registry
                reg = get_registry()
            except Exception:
                return None
        try:
            return reg.try_resolve("mt5_connection")
        except Exception:
            return None

    def _sync_balance(self, live_balance: Optional[float] = None) -> None:
        """Keep self.balance (used for risk-% position sizing) synced to a
        single authoritative source instead of the frozen boot-time value.

        Day 131 fix (Risk / Paper / Live balance synchronization):
        previously `self.balance` was set once in __init__ and never
        touched again, while `self._paper.balance` moved with every paper
        trade and the real MT5 account balance moved with every live
        trade/market swing. Risk sizing therefore silently drifted from
        the account it was supposed to be sizing against.

        Precedence:
          * mt5_demo with a fresh live account snapshot -> live balance is
            authoritative (real broker money).
          * otherwise -> the shared PaperTrader's balance is authoritative.

        Thread-safe: several per-symbol AITraders may share one PaperTrader
        and call this concurrently.
        """
        with self._balance_lock:
            if live_balance is not None and live_balance > 0:
                new_balance, source = float(live_balance), "live"
            else:
                try:
                    new_balance = float(self._paper.balance)
                    source = "paper"
                except Exception:
                    return

            old_balance = self.balance
            if old_balance and old_balance > 0:
                drift = abs(new_balance - old_balance) / old_balance
                if drift > self._balance_deviation_warn_pct:
                    log.info(
                        f"[Balance] {self.symbol} risk balance re-synced "
                        f"${old_balance:,.2f} -> ${new_balance:,.2f} "
                        f"(source={source}, drift={drift:.1%})"
                    )
            self.balance = new_balance
            self._balance_source = source

    def _get_live_open_pairs(self) -> list:
        """
        Day 96 bugfix + Day 102 hardening: use shared MT5Connection
        instead of raw mt5.initialize()/shutdown() which killed the
        shared session and caused positions_get() to return None.

        Audit fix (correlation blind-spot): in mt5_demo mode, this used to
        silently fall back to PaperTrader.get_open_positions() whenever the
        shared MT5 connection was unavailable or positions_get() failed —
        and PaperTrader is empty when the bot actually trades through MT5,
        not paper. The correlation filter would then see "no open
        positions" and happily open correlated pairs (e.g. EURUSD +
        GBPUSD + AUDUSD together) while MT5 in fact already held exposure.

        This method now sets `self._mt5_sync_ok` to reflect whether the
        returned list is actually authoritative. Callers in mt5_demo mode
        must treat `_mt5_sync_ok is False` as "unknown open positions" —
        i.e. fail CLOSED on new entries — rather than "zero positions".
        """
        # Assume healthy until proven otherwise below; only meaningful in
        # mt5_demo mode (paper/backtest modes are never "degraded").
        self._mt5_sync_ok = True
        try:
            _sim_mode = bool(getattr(self._router, "_simulation_mode", False))
            if self.execution_mode == "mt5_demo" and not _sim_mode:
                from core.constants import MT5_MAGIC_NUMBER

                # Use the shared MT5Connection (registered in Phase 4 boot)
                # instead of calling mt5.initialize()/shutdown() directly.
                # Day 131 fix: this used to import a stale duplicate
                # `utils.registry.ServiceRegistry` and call try_resolve()
                # as an unbound class method — a different, unpopulated
                # registry than the one core.runtime actually wires
                # (core.service_registry). That always returned None,
                # which is the real root cause of the persistent
                # "No shared MT5Connection in registry" message even
                # when MT5 was connected and healthy. See
                # _resolve_mt5_connection() for the corrected lookup.
                mt5_conn = self._resolve_mt5_connection()

                if mt5_conn is not None:
                    # Shared connection available — use it
                    positions = mt5_conn.positions_get()
                    if positions is not None:
                        return [
                            p.symbol for p in positions
                            if getattr(p, "magic", MT5_MAGIC_NUMBER) == MT5_MAGIC_NUMBER
                        ]
                    # Shared connection returned None — this is a real issue
                    log.warning(
                        "[Risk] Shared MT5Connection.positions_get() returned None "
                        "(MT5 terminal may not be running or account issue) — "
                        "open positions unknown, failing closed for correlation"
                    )
                    self._mt5_sync_ok = False
                else:
                    # No shared connection — cannot verify live exposure.
                    self._mt5_sync_ok = False
                    if not getattr(self, '_mt5_unavailable_warned', False):
                        log.warning(
                            "[Risk] No shared MT5Connection in registry while in "
                            "mt5_demo mode — open positions unknown, correlation "
                            "check will fail closed until sync is restored"
                        )
                        self._mt5_unavailable_warned = True
        except ImportError:
            # MetaTrader5 package not installed — this only happens when
            # execution_mode isn't actually mt5_demo (e.g. Linux paper/dev
            # runs), so PaperTrader genuinely is authoritative here.
            if not getattr(self, '_mt5_unavailable_warned', False):
                log.info("[Risk] MetaTrader5 package not installed (Linux?) — "
                         "PaperTrader is the authoritative position source")
                self._mt5_unavailable_warned = True
        except Exception as e:
            log.warning(f"[Risk] Position detection failed: {e} — "
                        f"open positions unknown, failing closed for correlation")
            self._mt5_sync_ok = False

        # PaperTrader is the fallback source. In mt5_demo mode with a
        # degraded sync, this list is NOT authoritative — it's returned
        # only as a best-effort value for logging/UI; run_cycle() checks
        # `self._mt5_sync_ok` separately and fails closed on new entries
        # rather than trusting an empty list here as "no open positions".
        try:
            return [t.get("pair") for t in self._paper.get_open_positions() if t.get("pair")]
        except Exception as e:
            log.warning(
                f"[Risk] PaperTrader.get_open_positions() raised: {e} — "
                f"returning empty open-pairs list"
            )
            self._mt5_sync_ok = self._mt5_sync_ok and (self.execution_mode != "mt5_demo")
            return []

    def sync_risk_with_open_positions(self) -> None:
        open_pairs = self._get_live_open_pairs()
        self._risk.sync_open_positions(open_pairs)

    def _get_live_open_positions_detailed(self) -> list[dict]:
        """Audit fix (EX-2 / X-2): PositionSizer's correlation/portfolio-heat
        check (_apply_advanced_sizing) used to call self._paper.get_open_positions()
        directly — in mt5_demo mode that source is not authoritative (the bot
        trades through MT5, not PaperTrader), so the sizer would see "no open
        positions" even while MT5 held real correlated exposure, and would
        happily approve stacking more risk on top of it.

        This mirrors `_get_live_open_pairs()`'s authoritative-source logic but
        returns the richer {pair, direction, risk_usd} shape the sizer needs,
        keeping both call sites in sync on the same fail-closed semantics.

        Fail-closed contract: in mt5_demo mode, if MT5 positions can't be
        read, this returns an empty list AND leaves `self._mt5_sync_ok=False`
        (set by `_get_live_open_pairs()`, called first below) so callers that
        check that flag correctly treat "empty" as "unknown", not "flat".
        """
        # Piggyback on _get_live_open_pairs() for the authoritative-source
        # decision + self._mt5_sync_ok bookkeeping — avoids duplicating that
        # (already-hardened) fallback logic here.
        pairs_only = self._get_live_open_pairs()

        _sim_mode = bool(getattr(self._router, "_simulation_mode", False))
        if self.execution_mode == "mt5_demo" and not _sim_mode and self._mt5_sync_ok:
            try:
                from core.constants import MT5_MAGIC_NUMBER, get_pip_value_usd
                mt5_conn = self._resolve_mt5_connection()
                positions = mt5_conn.positions_get() if mt5_conn is not None else None
                if positions is not None:
                    detailed = []
                    for p in positions:
                        if getattr(p, "magic", MT5_MAGIC_NUMBER) != MT5_MAGIC_NUMBER:
                            continue
                        direction = "BUY" if getattr(p, "type", 0) == 0 else "SELL"
                        risk_usd = 0.0
                        try:
                            sl = float(getattr(p, "sl", 0) or 0)
                            entry = float(getattr(p, "price_open", 0) or 0)
                            volume = float(getattr(p, "volume", 0) or 0)
                            if sl and entry and volume:
                                import MetaTrader5 as mt5
                                info = mt5.symbol_info(p.symbol)
                                if info:
                                    pip_size = info.point * (10 if info.digits in (3, 5) else 1)
                                    pips = abs(entry - sl) / pip_size if pip_size else 0.0
                                    pip_value = get_pip_value_usd(p.symbol)
                                    risk_usd = round(pips * pip_value * volume, 2)
                        except Exception as e:
                            log.warning(f"[Sizing] risk_usd estimate failed for {p.symbol}: {e}")
                        detailed.append({
                            "pair": p.symbol,
                            "direction": direction,
                            "risk_usd": risk_usd,
                        })
                    return detailed
            except Exception as e:
                log.warning(f"[Sizing] MT5 detailed-position read failed: {e} — "
                            f"falling back to pair-only list with risk_usd=0")

            # MT5 detailed read failed after pairs_only already succeeded —
            # degrade gracefully to pair-only info rather than losing the
            # correlation signal entirely.
            return [{"pair": pair, "direction": "", "risk_usd": 0.0} for pair in pairs_only]

        if self.execution_mode == "mt5_demo" and not _sim_mode and not self._mt5_sync_ok:
            # Fail closed: unknown exposure must not be treated as "none".
            # Returning [] here means the sizer's correlation/heat check
            # sees nothing — callers that need the stronger guarantee
            # should also check self._mt5_sync_ok directly, same as
            # _get_live_open_pairs() callers do.
            return []

        # Paper/backtest modes — PaperTrader is authoritative.
        try:
            return [
                {
                    "pair": pos.get("pair", ""),
                    "direction": pos.get("signal") or pos.get("direction", ""),
                    "risk_usd": float(pos.get("risk_usd", 0) or 0),
                }
                for pos in self._paper.get_open_positions()
            ]
        except Exception as e:
            log.warning(f"[Sizing] PaperTrader.get_open_positions() raised: {e}")
            return []

    def _on_mt5_position_closed(self, symbol: str, result: str, pnl: float) -> None:
        """Round-22 audit fix (B1): callback for PositionManager close events.

        Called by broker/position_manager.py when it detects an MT5 position
        has been closed (SL/TP hit by broker, or scheduled exit like Friday
        close). This callback feeds the close into the same processing
        pipeline as paper-trade closes — circuit breaker, risk engine,
        trade memory, etc.
        """
        try:
            log.info(
                f"[AITrader] {symbol} MT5 position closed: {result} | PnL: ${pnl:.2f}"
            )
            # Feed into circuit breaker + risk engine for state updates
            self._risk.record_trade_close(symbol, pnl)
            self._circuit_breaker.record_result(result, pnl)
            # Sync balance after MT5 close
            if hasattr(self, '_sync_balance'):
                self._sync_balance()
        except Exception as e:
            log.warning(f"[AITrader] _on_mt5_position_closed error: {e}")

    def _process_closed_trades(self, closed_now: list[dict]) -> list[dict]:
        processed = []
        for trade in closed_now:
            context = trade.get("context") or {}
            memory_trade_id = context.get("memory_trade_id")
            rr_ratio = context.get("rr_ratio") or trade.get("rr_ratio", 0)
            trade["rr_ratio"] = rr_ratio

            self._risk.record_trade_close(trade["pair"], trade["pnl"])
            self._circuit_breaker.record_result(trade["result"], trade["pnl"])

            # Audit fix: core/confidence_manager.py's record_outcome() was
            # fully implemented but never called anywhere, so layer weights
            # never adjusted from real trade results. We wire a minimal,
            # correctly-derived signal here: for a directional trade, a WIN
            # means the predicted direction was right, a LOSS means the
            # opposite direction would have been right. This only records
            # the "decision_agent" layer (the one signal this codebase
            # actually tracks per-trade); true per-layer (rule/ml/rl/llm)
            # attribution needs those individual predictions persisted on
            # trade open, which is a larger change left for the
            # MasterDecisionEngine integration.
            try:
                predicted_signal = (context.get("decision") or trade.get("type") or "").upper()
                trade_result = trade.get("result")
                if predicted_signal in ("BUY", "SELL") and trade_result in ("WIN", "LOSS"):
                    actual_direction = predicted_signal if trade_result == "WIN" else (
                        "SELL" if predicted_signal == "BUY" else "BUY"
                    )
                    from core.confidence_manager import get_confidence_manager
                    get_confidence_manager().record_outcome(
                        "decision_agent", predicted_signal, actual_direction
                    )
            except Exception as e:
                log.debug(f"[ConfidenceManager] record_outcome failed: {e}")

            # ── Day 102+ hotfix: memory DB sync fallback ─────────────
            # Previously, if memory_trade_id was missing (context dict
            # lost on restart, MT5 position closed externally, etc.),
            # we'd skip the memory DB update entirely — leaving the
            # trade's `result` column stuck at 'OPEN' forever. This
            # silently broke win-rate stats and made every trade look
            # like the bot's "first ever" trade (no learning).
            #
            # Now: if memory_trade_id is missing, do a fallback lookup
            # by pair symbol in the trades table. If we find an OPEN
            # trade for that pair, close it with the result/pnl we
            # just got from the broker.
            if memory_trade_id:
                try:
                    self._memory.on_trade_closed(memory_trade_id, trade["result"], trade["pnl"])
                    if self._mistake_analyzer:
                        self._mistake_analyzer.analyze_closed_trade(memory_trade_id)
                except Exception as e:
                    log.warning(f"[Learning] Close sync failed for memory trade #{memory_trade_id}: {e}")
                # Day 102+ hotfix: also backfill the JSON-side LearningAgent
                # so get_performance_stats() sees the closed trade and the
                # "Memory: X decisions | WR: Y%" log line reflects reality.
                try:
                    decision_id = context.get("decision_id")
                    if decision_id is not None:
                        self._learn.update_outcome(decision_id, trade["result"], trade.get("pnl_pips", 0.0))
                    else:
                        # No decision_id stashed — use symbol-based fallback
                        self._learn.update_outcome_by_symbol(
                            trade["pair"], trade["result"], trade.get("pnl_pips", 0.0)
                        )
                except Exception as e:
                    log.debug(f"[Learning] LearningAgent outcome backfill failed: {e}")
            else:
                # Fallback: try to find an orphaned OPEN trade by pair
                try:
                    orphan_id = self._memory.db.close_orphaned_open_trade(
                        trade["pair"], trade["result"], trade["pnl"]
                    )
                    if orphan_id is not None:
                        log.warning(
                            f"[Learning] Synced orphaned trade #{orphan_id} "
                            f"({trade['pair']}) via fallback — memory_trade_id was missing"
                        )
                        memory_trade_id = orphan_id  # so downstream events carry it
                        # Mistake analysis still possible on the recovered id
                        if self._mistake_analyzer:
                            try:
                                self._mistake_analyzer.analyze_closed_trade(orphan_id)
                            except Exception as e:
                                log.debug(f"[Learning] Mistake analysis on orphan #{orphan_id} failed: {e}")
                    else:
                        log.warning(
                            f"[Learning] No OPEN trade found in memory for {trade['pair']} — "
                            f"close event dropped (result={trade['result']}, pnl=${trade['pnl']:.2f})"
                        )
                except Exception as e:
                    log.warning(
                        f"[Learning] Orphan-close fallback failed for {trade['pair']}: {e}"
                    )
                # Day 102+ hotfix: also backfill JSON-side LearningAgent
                # via symbol-based fallback (we never had a decision_id).
                try:
                    self._learn.update_outcome_by_symbol(
                        trade["pair"], trade["result"], trade.get("pnl_pips", 0.0)
                    )
                except Exception as e:
                    log.debug(f"[Learning] LearningAgent symbol-fallback backfill failed: {e}")

            self._notify_trade_close(trade)
            # ── Day 102+ hotfix: backfill the OTHER close handlers ─────
            # Three persistence layers had "open without close" bugs
            # identical to the trade_memory.json result=null issue we
            # already fixed. Their close methods existed but were never
            # called from production code:
            #   1. TradeJournal.log_close  (core/professional_tools.py:301)
            #   2. AnalysisHistory.update_result  (memory/history.py:62)
            # Without these calls, journal.csv close columns stayed blank
            # and analysis_history.json entries stayed result=null forever
            # — same silent gap pattern.
            try:
                from core.professional_tools import get_trade_journal
                journal = get_trade_journal()
                journal.log_close(
                    symbol=trade.get("pair", "?"),
                    cycle=int(context.get("cycle", 0)),
                    close_price=float(trade.get("exit_price", trade.get("close_price", 0)) or 0),
                    result=trade.get("result", "UNKNOWN"),
                    pnl_usd=float(trade.get("pnl", 0) or 0),
                    pnl_pips=float(trade.get("pnl_pips", 0) or 0),
                    lesson=trade.get("lesson", ""),
                )
            except Exception as e:
                log.debug(f"[Journal] log_close failed: {e}")

            try:
                # AnalysisHistory is indexed by append position, not by
                # trade_id. We use the decision_id stashed at open time
                # (minus 1 because update_result takes a 0-indexed position
                # while decision_id is 1-indexed).
                decision_id = context.get("decision_id")
                if decision_id is not None and decision_id > 0:
                    AnalysisHistory().update_result(
                        index=decision_id - 1,
                        result=trade.get("result", "UNKNOWN").lower(),
                        pnl=float(trade.get("pnl", 0) or 0),
                    )
            except Exception as e:
                log.debug(f"[History] update_result failed: {e}")

            # ── Day 37+ runtime unification ─────────────────────────
            # Publish trade.close + learning.feedback so bus subscribers
            # (analytics, dashboard, audit_trail, learning) all see the
            # closed trade without AITrader knowing about them.
            self._publish("trade.close", {
                "symbol": trade.get("pair"),
                "result": trade.get("result"),
                "pnl": trade.get("pnl"),
                "rr_ratio": rr_ratio,
                "trade_id": memory_trade_id,
            })
            self._publish("learning.feedback", {
                "symbol": trade.get("pair"),
                "result": trade.get("result"),
                "pnl": trade.get("pnl"),
                "rr_ratio": rr_ratio,
                "memory_trade_id": memory_trade_id,
            })
            if self._metrics is not None:
                try:
                    self._metrics.inc("trades.closed")
                    if trade.get("result") == "WIN":
                        self._metrics.inc("trades.wins")
                    elif trade.get("result") == "LOSS":
                        self._metrics.inc("trades.losses")
                except Exception as e:
                    log.warning(f"Suppressed exception at line 1297: {e}")
                    pass
            processed.append(trade)

        return processed

    def _monitor_only_result(
        self,
        price: float,
        candle_time: str | None,
        session_ctx: dict,
        elapsed: float,
        closed_trades: list[dict],
    ) -> dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "version": self.VERSION,
            "elapsed_sec": elapsed,
            "price": price,
            "trend": None,
            "rsi": None,
            "regime": None,
            "volatility": None,
            "mtf_bias": None,
            "rule_signal": None,
            "rule_conf": 0,
            "llm_signal": None,
            "llm_conf": 0,
            "llm_analysis": "",
            "llm_risk": "",
            "news_safe": True,
            "news_reason": "",
            "decision": "WAIT",
            "confidence": 0,
            "trade_allowed": False,
            "final_action": "WAIT",
            "entry": None,
            "sl": None,
            "tp": None,
            "sl_pips": 0,
            "tp_pips": 0,
            "lot": 0,
            "rr": 0,
            "risk_usd": 0,
            "reject_reason": "One decision already taken for this candle",
            "total_decisions": 0,
            "win_rate": "N/A",
            "session": self._format_session_label(session_ctx),
            "decision_candle": candle_time,
            "closed_trades": closed_trades,
            "monitor_only": True,
            "approval_mode": self._approval.mode_name,
        }

    @staticmethod
    def _rl_conf_100(rl_ctx: dict) -> float:
        """Convert RL confidence from 0-1 scale to 0-100 scale."""
        try:
            v = float(rl_ctx.get("confidence", 0) or 0)
            return min(99.0, v * 100) if v <= 1.0 else v
        except (TypeError, ValueError):
            return 0.0

    def _build_result(
        self,
        market_out,
        analysis_out,
        dec_out,
        risk_out,
        perm_out,
        stats,
        elapsed,
        session_ctx: dict | None = None,
        candle_time: str | None = None,
        closed_trades: list[dict] | None = None,
    ):
        ind = market_out["ind_ctx"]
        regime = market_out["regime"]
        signal = analysis_out.get("signal", {})
        llm = analysis_out.get("llm", {})
        news = analysis_out.get("news", {})
        
        # Day 81+ fix: extract fallback price from ind_ctx to ensure entry is never None
        fallback_price = ind.get("close") or ind.get("price") or 0

        # Co-founder fix: Price fallback chain so it's never None in report
        _price = (
            ind.get("close") or ind.get("price")
            or dec_out.get("entry") or risk_out.get("entry") or perm_out.get("entry")
        )
        if _price is None:
            try:
                _df = market_out.get("df")
                if _df is not None and len(_df) > 0 and "close" in _df.columns:
                    _price = float(_df["close"].iloc[-1])
            except Exception:
                pass
        if _price is None:
            _price = 0

        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "version": self.VERSION,
            "elapsed_sec": elapsed,
            "price": _price,
            "trend": ind.get("trend"),
            "rsi": ind.get("rsi"),
            "regime": regime.get("regime"),
            "volatility": regime.get("volatility"),
            "mtf_bias": market_out.get("mtf_bias"),
            "rule_signal": signal.get("signal"),
            "rule_conf": signal.get("confidence"),
            "llm_signal": llm.get("signal"),
            "llm_conf": llm.get("confidence"),
            "llm_analysis": llm.get("analysis", ""),
            "llm_risk": llm.get("key_risk", ""),
            "news_safe": news.get("trade_allowed", True),
            "news_reason": news.get("reason", ""),
            # ARCHITECTURAL FIX: `decision` is the ANALYSIS verdict
            # (BUY/SELL/WAIT — what the analysts said). `final_action` /
            # `execution_action` is the EXECUTION verdict (what the system
            # will actually do, after gates). Both are reported so the
            # operator can distinguish "analysis said X, execution did Y".
            "decision":           dec_out.get("decision"),
            "analysis_signal":    dec_out.get("decision"),  # alias, clearer name
            "execution_action":   dec_out.get("execution_action") or perm_out.get("final_action"),
            "confidence":         dec_out.get("confidence"),
            "trade_allowed":      perm_out["allowed"],
            "final_action":       perm_out["final_action"],
            "blocked_reason":     perm_out.get("blocked_reason"),
            "execution_filters":  analysis_out.get("execution_filters", {}),
            # Day 81+ hotfix: ensure entry is never None/0 — use dec_out's entry
            # (which has the ind_ctx fallback) if risk_out lost it
            "entry": risk_out.get("entry") or dec_out.get("entry") or fallback_price,
            "sl": risk_out.get("sl_price"),
            "tp": risk_out.get("tp_price"),
            "sl_pips": risk_out.get("sl_pips", 0),
            "tp_pips": risk_out.get("tp_pips", 0),
            "lot": risk_out.get("lot", 0),
            "rr": risk_out.get("rr_ratio", 0),
            "risk_usd": risk_out.get("risk_usd", 0),
            "reject_reason": risk_out.get("reject_reason")
            or (
                None
                if perm_out["allowed"]
                else next((c["detail"] for c in perm_out["checks"] if not c["passed"]), None)
            ),
            "total_decisions": stats.get("closed_trades", stats.get("total_decisions", 0)),
            "total_history":   stats.get("total_decisions", 0),
            "win_rate": stats.get("win_rate", "N/A"),
            "session": self._format_session_label(session_ctx),
            "decision_candle": candle_time,
            "closed_trades": closed_trades or [],
            # Day 76 — full sizer breakdown for journal/telegram/dashboard.
            "position_sizing": risk_out.get("position_sizing"),

            # ── Phase 6: Individual confidence source visibility ──
            # Previously _build_result() was the bottleneck that collapsed
            # all 10 confidence sources into 3 keys (rule_conf, llm_conf,
            # confidence).  Now each source is individually accessible in
            # the report dict so main.py, dashboards, and diagnostics can
            # see exactly what each module contributed.
            "master_confidence": (analysis_out.get("master_ctx") or {}).get("master_confidence", 0),
            "master_signal": (analysis_out.get("master_ctx") or {}).get("master_signal", "WAIT"),
            "ml_confidence": (analysis_out.get("ensemble") or {}).get("confidence", 0),
            "ml_available": (analysis_out.get("ensemble") or {}).get("ml_available", True),
            "rl_confidence": self._rl_conf_100(analysis_out.get("rl_agent") or {}),
            "smc_score": (analysis_out.get("smc_ctx") or {}).get("smc_score", 0),
            "smc_grade": (analysis_out.get("smc_ctx") or {}).get("smc_grade", "N/A"),
            "session_score": (analysis_out.get("session_ctx") or {}).get("session_score", 0),
            "session_confidence": (analysis_out.get("session_ctx") or {}).get("session_confidence", 0),
            "session_grade": (analysis_out.get("session_ctx") or {}).get("session_grade", "N/A"),
            "confluence_confidence": (analysis_out.get("confluence") or {}).get("confidence", 0),
            "confluence_quality": (analysis_out.get("confluence") or {}).get("setup_quality", "UNKNOWN"),
            "fusion_confidence": dec_out.get("_fusion_conf", 0),
            "per_source_confidence": dec_out.get("per_source_confidence", {}),
            # Raw aggregate confidence before voting adjustments (from decision_agent)
            "raw_confidence": (dec_out.get("per_source_confidence") or {}).get("aggregate_raw", dec_out.get("confidence", 0)),
            # Phase 7: confidence trace (full before/after audit trail)
            "confidence_trace": dec_out.get("confidence_trace", []),
            # Reasons list for the decision block logger
            "reasons": dec_out.get("reasons", []),
            # Pattern (from decision_agent extraction)
            "pattern": dec_out.get("pattern"),
        }

    def _print_final(self, r: dict) -> None:
        icons = {"BUY": "🟢", "SELL": "🔴", "WAIT": "🟡", "NO TRADE": "⚪"}
        icon = icons.get(r["final_action"], "⚪")
        bar = "═" * 52

        log.info(bar)
        log.info(f"  {icon}  AI TRADER FINAL REPORT — {r['symbol']}")
        log.info(bar)
        log.info(f"  Engine       : {self.execution_mode.upper()} | Approval: {r.get('approval_mode', 'N/A')}")
        log.info(f"  Price        : {r['price']}  |  Session: {r.get('session')}")
        if r.get("decision_candle"):
            log.info(f"  Candle       : {r['decision_candle']}")
        if r.get("monitor_only"):
            log.info(f"  Monitor only : {r['reject_reason']}")
        else:
            log.info(f"  Trend        : {r['trend']}  |  Regime: {r['regime']}")
            log.info(f"  RSI          : {r['rsi']}  |  Volatility: {r['volatility']}")
            log.info(f"  Rule signal  : {r['rule_signal']} ({r['rule_conf']}%)")
            log.info(f"  LLM signal   : {r['llm_signal']} ({r['llm_conf']}%)")
            # ── INSTITUTIONAL LOG FORMAT ──────────────────────────────
            # Separates the ANALYSIS verdict from the EXECUTION verdict so
            # the operator sees "BUY 79% (analysis) → BLOCKED (news)"
            # instead of the misleading "WAIT 0%" the old pipeline produced.
            _analysis_signal = r.get("analysis_signal") or r.get("decision", "WAIT")
            _execution_action = r.get("execution_action") or r.get("final_action", "WAIT")
            _analysis_conf = r.get("confidence", 0)
            log.info(f"  ──")
            log.info(f"  ANALYSIS     : {_analysis_signal} (confidence {_analysis_conf}%)")
            if _execution_action in ("BUY", "SELL") and r.get("trade_allowed"):
                log.info(f"  EXECUTION    : {_execution_action} (ALLOWED)")
            else:
                _blocked = r.get("blocked_reason") or r.get("reject_reason") or "gated"
                log.info(f"  EXECUTION    : BLOCKED — {_blocked}")
                # Show which execution filter blocked (if any)
                _filters = r.get("execution_filters") or {}
                for _gate, _info in _filters.items():
                    if isinstance(_info, dict) and _info.get("blocked"):
                        log.info(f"    └─ {_gate}: {_info.get('reason', 'blocked')}")
            log.info(f"  ──")

            if r.get("paper_trade_id"):
                log.info(f"  FINAL ACTION : {r['final_action']}")
                log.info(f"  Entry: {r['entry']} | SL: {r['sl']} | TP: {r['tp']}")
                log.info(f"  Lot: {r['lot']} | R:R 1:{r['rr']} | Risk: ${r['risk_usd']}")
                # Day 76 — show sizer breakdown if available
                ps = r.get("position_sizing")
                if ps and ps.get("approved"):
                    log.info(
                        f"  Sizer        : base={ps.get('base_lot', 0):.2f} → "
                        f"final={ps.get('lot', 0):.2f} (×{ps.get('final_mult', 0):.3f})"
                    )
                if r.get("trade_id"):
                    log.info(f"  Trade ID     : #{r['trade_id']}")
                log.info(
                    f"  Paper Trade  : #{r['paper_trade_id']}  |  "
                    f"Paper Balance: ${r.get('paper_balance')}"
                )
            elif r["trade_allowed"]:
                log.info(f"  FINAL ACTION : {r['final_action']} (not executed — {r.get('reject_reason', 'pending approval')})")
            else:
                log.info(f"  FINAL ACTION : NO TRADE — {r['reject_reason']}")

        if r.get("closed_trades"):
            log.info(f"  Closed now   : {len(r['closed_trades'])} trade(s) updated")
        # Day 102+ hotfix: log CLOSED decisions (with outcomes) separately
        # from TOTAL history entries. The previous "X decisions" line used
        # the total JSON history length, which always showed 0 because
        # outcomes were never backfilled — making it look like the bot
        # had no memory even after dozens of trades.
        _closed   = r.get("total_decisions", 0)
        _total    = r.get("total_history", _closed)
        _wr       = r.get("win_rate", "N/A")
        log.info(f"  Memory       : {_closed} closed / {_total} total | WR: {_wr}%")
        log.info(f"  Completed in : {r['elapsed_sec']}s")
        log.info(bar)

    def _save_all(self, market_out, analysis_out, risk_out, dec_out, perm_out):
        try:
            ind_ctx = market_out["ind_ctx"]
            combined = {
                **ind_ctx,
                **market_out.get("regime_ctx", {}),
                **analysis_out.get("pat_ctx", {}),
                **analysis_out.get("sr_ctx", {}),
                **analysis_out.get("bias_ctx", {}),
                **analysis_out.get("signal_ctx", {}),
                **analysis_out.get("llm_ctx", {}),
                **analysis_out.get("news_ctx", {}),
                **self._risk.get_ai_context(risk_out),
                **self._decision.get_ai_context(dec_out),
                "trade_allowed": perm_out["allowed"],
                "final_action": perm_out["final_action"],
            }
            db = self._db
            df = market_out["df"]
            db.save_candles(df, self.symbol, self.timeframe)
            db.save_indicators(df, self.symbol, self.timeframe)
            # 2026-07-20 fix: market_out["df"] never has the 'pattern' /
            # 'engulfing' / 'star_pattern' columns — PatternDetector.detect_all()
            # (agents/analysis_agent.py) makes its own df.copy() before adding
            # them, so those columns only exist on analysis_out["df"]. Calling
            # save_patterns() with market_out["df"] meant every row's
            # row.get('pattern','none') silently defaulted to 'none' and got
            # skipped — patterns table stayed at 0 rows forever even though
            # pattern detection was running correctly for live decisions.
            pattern_df = analysis_out.get("df", df)
            db.save_patterns(pattern_df, self.symbol, self.timeframe)
            db.save_analysis(
                self.symbol,
                self.timeframe,
                analysis_out["bias_result"]["net_score"],
                analysis_out["bias_result"]["bias"],
                combined,
            )
            # Co-founder fix: save the FINAL decision signal to history,
            # not the intermediate bias_ctx signal.
            _final_action = perm_out.get("final_action", "WAIT")
            _final_confidence = dec_out.get("confidence", 0)
            _final_bias_ctx = {
                "bias": _final_action,
                "confidence_pct": _final_confidence,
                "recommendation": f"Final decision: {_final_action}",
                "has_conflict": False,
            }
            AnalysisHistory().save(
                self.symbol,
                self.timeframe,
                _final_bias_ctx,
                ind_ctx,
            )
        except Exception as e:
            log.warning(f"DB save error (non-critical): {e}")

    def _extract_candle_time(self, market_out: dict) -> str | None:
        df = market_out.get("df")
        if df is None or len(df.index) == 0:
            return None
        latest = df.index[-1]
        try:
            return latest.to_pydatetime().isoformat()
        except Exception as e:
            return str(latest)

    def _extract_pattern(self, market_out: dict) -> str:
        df = market_out.get("df")
        if df is None:
            return "none"
        for key in ("pattern_name", "pattern", "engulfing", "star_pattern"):
            if key in df.columns:
                value = df.iloc[-1].get(key, "none")
                if value and value != "none":
                    return value
        return "none"

    def _format_session_label(self, session_ctx: dict | None) -> str | None:
        if not session_ctx:
            return None
        if session_ctx.get("overlap"):
            return session_ctx["overlap"]
        active = session_ctx.get("active_sessions") or []
        if active:
            return "/".join(s.replace("_", " ").title() for s in active)
        return "Closed"

    def _session_permission_context(self, session_ctx: dict | None) -> dict | None:
        if not session_ctx:
            return None
        trade_quality = (session_ctx.get("trade_quality") or "").upper()
        if "BEST" in trade_quality or "GOOD" in trade_quality:
            quality = "HIGH"
        elif "CAUTION" in trade_quality:
            quality = "MEDIUM"
        else:
            quality = "LOW"
        return {"quality": quality}

    def _notify_trade_open(self, trade: dict, result: dict, dec_out: dict) -> None:
        """Builds the Telegram payload from `result`, not `trade` — `trade`'s
        shape differs between paper mode (full PaperTrader record) and MT5
        demo mode (still a `PENDING_EXECUTOR` stub), but `result` always has
        symbol/final_action/entry/sl/tp/lot regardless of backend."""
        if not self.notifier:
            return
        payload = {
            "pair": result.get("symbol"),
            "signal": result.get("final_action"),
            "entry": result.get("entry"),
            "sl": result.get("sl"),
            "tp": result.get("tp"),
            "lot": result.get("lot"),
        }
        self._run_async(
            self.notifier.notify_trade_open(
                payload,
                result.get("confidence", 0),
                dec_out.get("reasons", []),
            )
        )

    def _notify_trade_close(self, trade: dict) -> None:
        if not self.notifier:
            return
        self._run_async(self.notifier.notify_trade_close(trade))

    def _run_async(self, coro) -> None:
        """Run an async coroutine safely (Bug fix: don't close event loop).

        Previous code used asyncio.run() which creates AND closes a loop
        each call. The python-telegram-bot Bot object caches the loop
        internally, so closing it caused 'Event loop is closed' errors
        on every subsequent call.

        Fix: delegate to utils.async_utils.run_coro_sync(), which reuses
        a persistent event loop (same fix as before) WITHOUT relying on
        the deprecated asyncio.get_event_loop() call outside a running
        loop context.
        """
        from utils.async_utils import run_coro_sync
        run_coro_sync(coro)

    def _clean_symbol(self, symbol: str) -> str:
        # Round-14 fix: see backtest/simulator.py — blanket "USDT"->"USD"
        # replace corrupted USDTRY/USDTHB (matched "USDT" mid-string, not
        # just as a Tether-quote suffix). Note: this method is also
        # defined again near line ~3370 in this same class — that later
        # definition wins at runtime (Python keeps the last one), so this
        # copy is currently dead code, but fixing both for consistency.
        cleaned = str(symbol).upper().replace("/", "").replace("=X", "").strip()
        if cleaned.endswith("USDT"):
            cleaned = cleaned[:-1]
        return cleaned

    def _error_result(self, reason: str) -> dict:
        log.error(f"Pipeline failed: {reason}")
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "version": self.VERSION,
            "final_action": "NO TRADE",
            "trade_allowed": False,
            "error": reason,
        }


class AutonomousTraderSystem:

    def __init__(
        self,
        symbols: list[str] | None = None,
        timeframe: str = "15m",
        balance: float = 10000.0,
        poll_seconds: int = 60,
        backup_interval_minutes: int = 30,
        cooldown_minutes: int = 5,
        max_cycles: int | None = None,
        enable_telegram: bool = True,
        use_scanner: bool = False,
        execution_mode: str = None,
        approval_mode: int = 3,
        registry: "Optional[ServiceRegistry]" = None,
    ):
        self.symbols = [self._clean_symbol(s) for s in (symbols or ["EURUSD", "GBPUSD", "USDJPY"])]
        self.timeframe = timeframe
        self.balance = balance
        self.poll_seconds = max(5, poll_seconds)
        self.backup_interval_minutes = max(5, backup_interval_minutes)
        self.cooldown_minutes = max(1, cooldown_minutes)
        self.max_cycles = max_cycles
        self.use_scanner = use_scanner
        self.execution_mode = (execution_mode or EXECUTION_MODE).lower()
        self.approval_mode = approval_mode
        self._stop_requested = False
        self._pause_until = None
        self._consecutive_error_cycles = 0
        self._bot_thread = None
        self._last_backup = None
        self._last_results: list[dict] = []

        # ── Day 37+ runtime unification ──────────────────────────────
        # Accept an optional ServiceRegistry. When provided, the system
        # pulls shared services (CircuitBreaker, ApprovalMode, scanner,
        # notifier, etc.) from it instead of constructing fresh copies,
        # and publishes events / metrics to the central bus.
        self._registry = registry
        self._bus = get_bus() if _RUNTIME_INFRA_AVAILABLE else None
        self._metrics = get_metrics() if _RUNTIME_INFRA_AVAILABLE else None

        # Day 36/37 — Market Scanner picks the day's Top-N tradeable pairs
        # each cycle instead of always scanning a fixed list. It needs a
        # market_data_manager (MT5 tick/candle bundle) to actually rank
        # pairs; that adapter isn't built yet, so until it is,
        # _select_cycle_symbols() below safely falls back to self.symbols.
        # If a registry is available, prefer its shared scanner.
        if registry is not None:
            shared_scanner = registry.try_resolve("market_scanner")
            if use_scanner and shared_scanner is not None:
                self.scanner = shared_scanner
            elif use_scanner and MarketScanner:
                self.scanner = MarketScanner(risk_engine=None)
            else:
                self.scanner = None
        else:
            self.scanner = MarketScanner(risk_engine=None) if (use_scanner and MarketScanner) else None

        # Circuit breaker + approval mode are global state (one shared JSON
        # file each) — created ONCE here and handed to every symbol's
        # AITrader so they don't overwrite each other's state.
        # Prefer the registry's shared instances if available.
        if registry is not None:
            shared_cb = registry.try_resolve("circuit_breaker")
            self.circuit_breaker = shared_cb if shared_cb is not None else CircuitBreaker(balance=balance)
        else:
            self.circuit_breaker = CircuitBreaker(balance=balance)
        self.approval = ApprovalMode(mode=approval_mode)

        if registry is not None:
            shared_notifier = registry.try_resolve("telegram_notifier")
            if shared_notifier is not None:
                self.notifier = shared_notifier
            else:
                self.notifier = TelegramNotifier() if enable_telegram and TelegramNotifier else None
        else:
            self.notifier = TelegramNotifier() if enable_telegram and TelegramNotifier else None

        # ── Shared PaperTrader ───────────────────────────────────────
        # One PaperTrader for ALL symbols so balance / open-position
        # tracking is unified.  Each AITrader previously created its own
        # PaperTrader, leading to 6 independent balances for 6 symbols.
        self._shared_paper = PaperTrader(starting_balance=balance)

        self.traders: dict[str, AITrader] = {
            symbol: self._build_trader(symbol) for symbol in self.symbols
        }
        self._sync_risk_state()

        # ── Webhook / external command subscription ──────────────────
        # Listen for webhook.command events and route them into the system.
        if self._bus is not None:
            try:
                self._bus.subscribe("webhook.command", self._on_webhook_command)
            except Exception as e:
                log.warning(f"[System] webhook.command subscription failed: {e}")

    def _on_webhook_command(self, evt) -> None:
        """Handle external commands received via webhook.
        Payload shape: {"action": "pause"|"resume"|"close_all"|"status", ...}"""
        if evt.payload is None:
            return
        action = (evt.payload.get("action") or "").lower() if isinstance(evt.payload, dict) else ""
        log.info(f"[System] Webhook command: {action} — {evt.payload}")
        if action == "pause":
            self._pause_until = datetime.now(timezone.utc) + timedelta(hours=1)
            log.info("[System] Paused via webhook for 1 hour")
        elif action == "resume":
            self._pause_until = None
            log.info("[System] Resumed via webhook")
        elif action == "stop":
            self.stop()
        elif action == "status":
            # The next cycle will write a fresh latest_report.json
            pass

    def _build_trader(self, symbol: str) -> AITrader:
        return AITrader(
            balance=self.balance,
            symbol=symbol,
            timeframe=self.timeframe,
            paper_balance=self.balance,
            notifier=self.notifier,
            execution_mode=self.execution_mode,
            approval_mode=self.approval_mode,
            circuit_breaker=self.circuit_breaker,
            approval=self.approval,
            registry=self._registry,
            paper_trader=self._shared_paper,
        )

    def run(self) -> dict:
        # Day 37+ fix: reset stop flag so run() can be called again after a
        # previous stop() / crash. This makes auto-restart from main.py work.
        self._stop_requested = False
        self._start_telegram_commands()
        self.backup_state(force=True)
        cycles = 0

        # ── Day 66: Send Telegram alert for upcoming high-impact news ──
        try:
            from intelligence.news_ai import get_news_intelligence
            news_ai = get_news_intelligence()
            news_ai.set_pairs(self.symbols)
            alert_msg = news_ai.format_telegram_alert()
            if alert_msg and self.notifier:
                # Issue 9 fix: route through the shared run_coro_sync() helper
                # instead of calling the deprecated asyncio.get_event_loop()
                # directly. run_coro_sync() reuses one persistent loop per
                # process, so we never create-and-abandon loops here.
                from utils.async_utils import run_coro_sync
                run_coro_sync(self.notifier.send_message(alert_msg))
        except Exception as e:
            log.debug(f"[System] Day 66 news alert failed: {e}")

        log.info(
            f"[System] Starting autonomous loop | Pairs={self.symbols} | "
            f"Timeframe={self.timeframe} | Mode={self.execution_mode.upper()} | "
            f"Scanner={'ON' if self.use_scanner else 'OFF'} | "
            f"Risk Balance=${self.balance} (config) | "
            f"Registry={'yes' if self._registry else 'no'}"
        )

        # Publish startup event
        if self._bus is not None:
            try:
                self._bus.publish("system.startup", {
                    "phase": "trader_loop",
                    "symbols": self.symbols,
                    "mode": self.execution_mode,
                    "balance": self.balance,
                }, source="autonomous_trader")
            except Exception as e:
                log.warning(f"Suppressed exception at line 1779: {e}")
                pass

        try:
            while not self._stop_requested:
                cycle_started = time.time()
                if self.max_cycles is not None and cycles >= self.max_cycles:
                    break

                if self._is_paused():
                    self._sleep_remaining(cycle_started)
                    cycles += 1
                    continue

                # BUG FIX: reset_fallback_throttle_cycle() (in
                # intelligence/sentiment_model.py) resets the dedicated
                # SentimentModel per-cycle LLM budget (default 5 calls)
                # plus the key_manager-absent fallback throttle. It was
                # defined but never called anywhere in the codebase —
                # confirmed via a full-repo grep, only the def line itself
                # matched. Result: the counter only ever incremented,
                # never reset, so once 5 sentiment LLM calls happened
                # (typically during the very first NewsIntelligence burst
                # at boot), EVERY subsequent sentiment.analyze() call for
                # the rest of the process's lifetime was silently skipped
                # with "sentiment cap: 5 calls/cycle reached" — headline
                # sentiment analysis was effectively permanently dead
                # after the first few seconds of runtime. Calling it here,
                # once at the top of every cycle (mirrors the cadence the
                # real _key_manager.check_cycle_throttle() resets on),
                # fixes that.
                try:
                    from intelligence.sentiment_model import reset_fallback_throttle_cycle
                    reset_fallback_throttle_cycle()
                except Exception as e:
                    log.debug(f"[System] sentiment cycle-throttle reset skipped (non-fatal): {e}")

                cycle_results = []
                cycle_errors = []
                active_symbols = self._select_cycle_symbols()
                if self._metrics is not None:
                    try:
                        self._metrics.record_cycle()
                        self._metrics.set_gauge("trader.active_symbols", len(active_symbols))
                    except Exception as e:
                        log.warning(f"Suppressed exception at line 1801: {e}")
                        pass

                # Co-founder fix: detect MT5 position closes BEFORE per-symbol
                # cycles run. Previously, MT5 SL/TP hits during runtime were
                # only reconciled at next boot via orphan_cleanup — meaning
                # the bot kept trading as if the position was still open,
                # and the close event (with its WIN/LOSS outcome) was lost
                # until restart. This polls MT5 once per cycle, compares to
                # the last snapshot, and for any disappeared ticket fires
                # the same close handlers that paper-trade closes go through.
                try:
                    self._detect_mt5_position_closes()
                except Exception as e:
                    log.debug(f"[System] MT5 close detection failed (non-fatal): {e}")

                # ── Audit fix: weekend gap-risk guard ──────────────────
                # core/production_hardening.py's should_close_for_weekend()
                # already correctly implements the Friday-late/Saturday/
                # Sunday-pre-open rules, but nothing in the live loop ever
                # called it — so weekend gap risk wasn't being actively
                # managed by this logic. Wired here, once per cycle, before
                # any per-symbol analysis runs.
                #
                # Scope kept deliberately conservative: this blocks NEW
                # entries (via the existing _pause_until mechanism already
                # used by webhook "pause" commands, checked at the top of
                # this loop) and, only for the hard Saturday/Friday-late
                # "close all" case, reuses the exact close pattern this
                # file already uses for the catastrophic-error emergency
                # close a few lines below (hasattr(t, '_order_manager') →
                # close_all_orders()) rather than inventing a new,
                # unverified close path. The "reduce 50%" Friday-evening
                # case is logged only — automatically resizing live
                # positions is a materially different (and riskier)
                # operation than blocking new entries, and isn't
                # implemented here without a verified position-resize API.
                try:
                    from core.production_hardening import should_close_for_weekend
                    _weekend_positions = []
                    for _t in self.traders.values():
                        try:
                            _weekend_positions.extend(_t._paper.get_open_positions())
                        except Exception:
                            pass
                    weekend_check = should_close_for_weekend(_weekend_positions)
                    if weekend_check.get("should_close_all"):
                        log.critical(f"[Weekend] {weekend_check.get('reason')} — "
                                     f"blocking new entries and attempting close-all")
                        self._pause_until = datetime.now(timezone.utc) + timedelta(hours=1)
                        for sym, t in self.traders.items():
                            if hasattr(t, "_order_manager"):
                                try:
                                    _results = t._order_manager.close_all_orders(
                                        reason=f"Weekend guard: {weekend_check.get('reason')}"
                                    )
                                    # P1 fix: was silently discarded; now check + alert.
                                    _failed = [r for r in (_results or []) if not r.get("success")]
                                    if _failed:
                                        log.error(f"[Weekend] {sym}: {len(_failed)} positions failed to close")
                                        # Best-effort Telegram alert
                                        try:
                                            if self._notifier is not None:
                                                self._notifier.send(
                                                    f"⚠️ Weekend close failed for {sym}: "
                                                    f"{len(_failed)} positions still open"
                                                )
                                        except Exception:
                                            pass
                                except Exception as e:
                                    log.warning(f"[Weekend] close_all_orders failed for {sym}: {e}")
                        if self._bus is not None:
                            try:
                                self._bus.publish("risk.event", {
                                    "kind": "weekend_guard_close_all",
                                    "reason": weekend_check.get("reason"),
                                }, source="autonomous_trader")
                            except Exception:
                                pass
                    elif weekend_check.get("should_reduce"):
                        log.warning(f"[Weekend] {weekend_check.get('reason')} — "
                                    f"consider manually reducing exposure "
                                    f"(auto-resize not implemented)")
                except Exception as e:
                    log.debug(f"[System] Weekend guard check skipped (non-fatal): {e}")

                for symbol in active_symbols:
                    # ── Skip symbols known to be unavailable on broker ──
                    # Avoids wasting cycle time on 30+ non-existent symbols
                    # (USOUSD, BTCUSD, etc.) that would all fail and
                    # trigger spurious recovery pauses.
                    try:
                        from data.fetcher import is_symbol_unavailable
                        if is_symbol_unavailable(symbol):
                            continue
                    except Exception:
                        pass

                    trader = self.traders.get(symbol) or self._spawn_trader(symbol)
                    try:
                        if self._manual_pause_active():
                            closed = trader.monitor_open_trades()
                            cycle_results.append(
                                {
                                    "symbol": symbol,
                                    "final_action": "WAIT",
                                    "trade_allowed": False,
                                    "closed_trades": closed,
                                    "reject_reason": "Trading paused from Telegram",
                                }
                            )
                            continue

                        result = trader.run_cycle(auto_paper_trade=True)
                        cycle_results.append(result)
                        if result.get("error"):
                            cycle_errors.append(f"{symbol}: {result['error']}")
                    except Exception as e:
                        msg = f"{symbol}: {e}"
                        cycle_errors.append(msg)
                        log.exception(f"[System] Symbol cycle failed — {msg}")

                self._last_results = cycle_results
                self._write_runtime_report()

                # ── Day 100+ — Feature drift detection (wire dead code) ──
                # Runs every 50 cycles to detect ML feature distribution shift.
                # If significant drift is found, logs a warning recommending retraining.
                if cycles % 50 == 0 and cycles > 0:
                    try:
                        from ml.feature_selector import get_feature_selector
                        import pandas as pd
                        fs = get_feature_selector()
                        # Use recent df from any trader as "current" window
                        for sym, t in self.traders.items():
                            if hasattr(t, '_df') and t._df is not None and len(t._df) > 100:
                                ref = t._df.iloc[-200:-50]  # reference window
                                cur = t._df.iloc[-50:]        # current window
                                if len(ref) > 10 and len(cur) > 10:
                                    drift_results = fs.detect_drift(ref, cur)
                                    sig = [d for d in drift_results if d.drift_level == "SIGNIFICANT"]
                                    if sig:
                                        log.warning(
                                            f"[System] ML DRIFT DETECTED on {sym}: "
                                            f"{len(sig)} features with significant drift — "
                                            f"retraining recommended. Top: "
                                            f"{sig[0].feature_name} (PSI={sig[0].psi:.3f})"
                                        )
                                break  # only check first available symbol
                    except Exception as e:
                        log.debug(f"[System] Drift check skipped: {e}")

                if cycle_errors:
                    self._handle_cycle_errors(cycle_errors)
                else:
                    self._consecutive_error_cycles = 0

                self.backup_state()

                # KILL SHOT FIX: DB backup every 100 cycles
                # If the single SQLite DB corrupts, we lose ALL trade history.
                # Back it up periodically so we can recover.
                if cycles % 100 == 0 and cycles > 0:
                    try:
                        import shutil
                        src = "database/trader.db"
                        dst = f"database/trader_backup_{cycles}.db"
                        shutil.copy2(src, dst)
                        # Keep only last 3 backups
                        import glob
                        backups = sorted(glob.glob("database/trader_backup_*.db"))
                        for old in backups[:-3]:
                            os.remove(old)
                        log.debug(f"[System] DB backed up to {dst}")
                    except Exception as e:
                        log.warning(f"[System] DB backup failed: {e}")

                # KILL SHOT FIX: Emergency close-all on catastrophic error streak
                # If 10 consecutive cycles ALL fail, close everything and halt.
                if self._consecutive_error_cycles >= 10:
                    log.critical(
                        f"[System] CATASTROPHIC: {self._consecutive_error_cycles} "
                        f"consecutive error cycles — EMERGENCY CLOSE ALL"
                    )
                    try:
                        for sym, t in self.traders.items():
                            if hasattr(t, '_order_manager'):
                                try:
                                    _results = t._order_manager.close_all_orders(
                                        reason="Emergency: 10 consecutive error cycles"
                                    )
                                    # P1 fix: check per-symbol results; was silently discarded.
                                    _failed = [r for r in (_results or []) if not r.get("success")]
                                    if _failed:
                                        log.critical(f"[System] {sym}: {len(_failed)} positions "
                                                     f"failed to close during emergency")
                                        try:
                                            if self._notifier is not None:
                                                self._notifier.send(
                                                    f"🚨 EMERGENCY close failed for {sym}: "
                                                    f"{len(_failed)} positions still open"
                                                )
                                        except Exception:
                                            pass
                                except Exception as _e:
                                    log.critical(f"[System] {sym} emergency close failed: {_e}")
                    except Exception as e:
                        log.critical(f"[System] Emergency close loop failed: {e}")
                    self._stop_requested = True
                    break

                self._sleep_remaining(cycle_started)
                cycles += 1

        except KeyboardInterrupt:
            log.info("[System] Stop requested by user")
        except Exception as e:
            # Day 37+ fix: any unexpected error in the outer loop used to
            # kill the trader entirely. Now we log + publish + keep the
            # function returning a report (main.py's auto-restart wrapper
            # will re-launch the loop).
            log.exception(f"[System] FATAL error in trading loop: {e}")
            try:
                if self._bus is not None:
                    self._bus.publish("system.error", {
                        "channel": "fatal_loop",
                        "reason": str(e),
                        "phase": "trader_loop",
                    }, source="autonomous_trader")
            except Exception as e:
                log.warning(f"Suppressed exception at line 1884: {e}")
                pass

        report = self._build_system_report()
        self._write_runtime_report(report)
        return report

    def stop(self) -> None:
        self._stop_requested = True

    # ──────────────────────────────────────────────────────────────
    # Co-founder fix: LIVE MT5 CLOSE DETECTION
    # ──────────────────────────────────────────────────────────────
    # Previously, MT5 SL/TP hits during runtime were only reconciled
    # at next boot via orphan_cleanup.reconcile_open_positions(). This
    # meant:
    #   1. The bot kept trading as if the position was still open
    #      (exposure/correlation risk still counted the closed trade)
    #   2. The close event (WIN/LOSS + pnl) was lost until restart
    #   3. CircuitBreaker, RiskEngine, and LearningAgent never saw
    #      the outcome — streak counters and win-rate stats were wrong
    #   4. If the bot crashed before next boot, the close was lost
    #      permanently
    #
    # This method runs once per cycle (before per-symbol trading),
    # snapshots MT5 positions, and for any ticket that disappeared
    # since the last snapshot, looks up the close in MT5 history and
    # fires the same close handlers that paper-trade closes go through.
    # ──────────────────────────────────────────────────────────────

    # Day 131 fix: `_mt5_known_tickets` used to be declared here as a
    # class-level mutable default (`_mt5_known_tickets: dict = {}`).
    # Because it's a dict (mutable), every AITrader instance — i.e. every
    # symbol — shared the *same* dict object until each instance's first
    # write. The guard below (`hasattr(self, ...)`) never caught this,
    # since `hasattr` is true for inherited class attributes too. In a
    # multi-symbol run, the very first read of this dict for a given
    # symbol could see ticket data left behind by a *different* symbol's
    # close-detection pass in the same cycle, producing false "closed
    # position" events (and the wrong PnL/close handlers firing) for the
    # wrong symbol. Now initialized per-instance in __init__ instead.

    def _detect_mt5_position_closes(self) -> list[dict]:
        """Poll MT5 once, detect closes, fire close handlers.

        Returns the list of close events detected this cycle. Safe to
        call when MT5 is unavailable (paper mode) — returns [].
        """
        # Skip entirely in paper mode or simulation
        if self.execution_mode != "mt5_demo":
            return []
        try:
            _sim_mode = bool(getattr(self._router, "_simulation_mode", False))
            if _sim_mode:
                return []
        except Exception:
            return []

        # Resolve shared MT5 connection (Day 131 fix — see
        # _resolve_mt5_connection() docstring for root cause).
        mt5_conn = self._resolve_mt5_connection()
        if mt5_conn is None:
            # Audit fix: this used to just `return []` silently every time
            # MT5 was unreachable, which meant a real broker-side close
            # (and its PnL) simply never got detected until the connection
            # came back — the daily-loss circuit breaker was blind for the
            # whole outage. We now count consecutive unreachable cycles so
            # run_cycle() can fail closed on NEW entries once the outage
            # has lasted long enough that "unknown" can't be treated as
            # "fine". Existing positions are never force-closed by this.
            self._mt5_disconnect_cycles = getattr(self, "_mt5_disconnect_cycles", 0) + 1
            if self._mt5_disconnect_cycles == 1 or self._mt5_disconnect_cycles % 5 == 0:
                log.warning(
                    f"[MT5CloseDetect] No shared MT5 connection — close "
                    f"detection skipped ({self._mt5_disconnect_cycles} "
                    f"consecutive cycles unreachable)"
                )
            return []

        # Snapshot current positions (our magic only)
        try:
            from core.constants import MT5_MAGIC_NUMBER
            positions = mt5_conn.positions_get()
            if positions is None:
                self._mt5_disconnect_cycles = getattr(self, "_mt5_disconnect_cycles", 0) + 1
                return []
            current = {
                p.ticket: p
                for p in positions
                if getattr(p, "magic", MT5_MAGIC_NUMBER) == MT5_MAGIC_NUMBER
            }
        except Exception as e:
            log.debug(f"[MT5CloseDetect] positions_get failed: {e}")
            self._mt5_disconnect_cycles = getattr(self, "_mt5_disconnect_cycles", 0) + 1
            return []

        # Successful poll — connection is healthy again.
        if getattr(self, "_mt5_disconnect_cycles", 0) > 0:
            log.info(
                f"[MT5CloseDetect] MT5 connection restored after "
                f"{self._mt5_disconnect_cycles} cycles — resuming normal "
                f"close detection"
            )
        self._mt5_disconnect_cycles = 0

        # Day 131 fix: _mt5_known_tickets is now always set per-instance
        # in __init__, so this is just a defensive belt-and-suspenders
        # check (e.g. for pre-existing pickled/older instances).
        if getattr(self, "_mt5_known_tickets", None) is None:
            self._mt5_known_tickets = {}

        # Find tickets that disappeared → closed
        closed_tickets = set(self._mt5_known_tickets.keys()) - set(current.keys())
        if not closed_tickets:
            self._mt5_known_tickets = current
            return []

        events = []
        for ticket in closed_tickets:
            last_pos = self._mt5_known_tickets[ticket]
            event = self._handle_mt5_close(mt5_conn, ticket, last_pos)
            if event:
                events.append(event)

        # Update snapshot
        self._mt5_known_tickets = current
        return events

    def _handle_mt5_close(self, mt5_conn, ticket: int, last_pos) -> dict | None:
        """Look up a closed MT5 position in history and fire close handlers.

        Uses mt5.history_get_by_ticket() (available in MT5 Python package)
        to retrieve the close price, close time, and profit. Then routes
        the close through _process_closed_trades so all the same handlers
        fire as for paper-trade closes (risk, circuit_breaker, memory,
        learning, journal).
        """
        try:
            import MetaTrader5 as mt5
            from datetime import datetime, timezone, timedelta
            from core.constants import MT5_MAGIC_NUMBER

            # history_get_by_ticket returns the deal that closed the position
            # Look back 24 hours for the close deal
            utc_to = datetime.now(timezone.utc)
            utc_from = utc_to - timedelta(hours=24)
            with mt5_conn.MT5_LOCK:
                deals = mt5.history_deals_get(utc_from, utc_to)
            if deals is None:
                log.warning(f"[MT5CloseDetect] No history deals for ticket {ticket}")
                return None

            # Find the closing deal for this position
            close_deal = None
            for d in deals:
                # position_id == ticket for the closing deal
                if getattr(d, "position_id", 0) == ticket:
                    close_deal = d
                    break

            symbol = getattr(last_pos, "symbol", "UNKNOWN")
            direction = "BUY" if getattr(last_pos, "type", 0) == 0 else "SELL"
            entry = float(getattr(last_pos, "price_open", 0))
            volume = float(getattr(last_pos, "volume", 0))
            sl = float(getattr(last_pos, "sl", 0))
            tp = float(getattr(last_pos, "tp", 0))

            if close_deal is not None:
                close_price = float(getattr(close_deal, "price", 0))
                pnl = float(getattr(close_deal, "profit", 0))
                close_reason = "SL HIT" if close_price <= sl and direction == "BUY" else \
                               "SL HIT" if close_price >= sl and direction == "SELL" else \
                               "TP HIT" if close_price >= tp and direction == "BUY" else \
                               "TP HIT" if close_price <= tp and direction == "SELL" else \
                               "MANUAL"
            else:
                # Can't find the close deal — best-effort estimate
                close_price = entry
                pnl = 0.0
                close_reason = "UNKNOWN (no history deal)"

            result = "WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "BREAKEVEN")

            # Build a close event in the same shape as PaperTrader.close_trade
            close_event = {
                "pair": symbol,
                "type": direction,
                "entry": entry,
                "exit_price": close_price,
                "lot": volume,
                "sl": sl,
                "tp": tp,
                "pnl": pnl,
                "pnl_pips": 0.0,  # would need pip_size to compute; non-critical
                "result": result,
                "reason": close_reason,
                "ticket": ticket,
                "close_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "context": {
                    "source": "mt5_demo",
                    "mt5_ticket": ticket,
                    "memory_trade_id": None,  # unknown — orphan fallback will handle
                    "close_reason": close_reason,
                },
            }

            log.info(
                f"[MT5CloseDetect] Close detected: {symbol} {direction} "
                f"ticket={ticket} → {result} ${pnl:.2f} ({close_reason})"
            )

            # Route through any AITrader's _process_closed_trades so all
            # close handlers fire (risk, CB, memory, learning, journal).
            # Pick the trader for this symbol, or the first available.
            target_trader = self.traders.get(symbol)
            if target_trader is None and self.traders:
                target_trader = next(iter(self.traders.values()))
            if target_trader is not None:
                try:
                    target_trader._process_closed_trades([close_event])
                    log.info(f"[MT5CloseDetect] Close event routed to AITrader({symbol}) for handler processing")
                except Exception as e:
                    log.warning(f"[MT5CloseDetect] Failed to route close to AITrader: {e}")

            return close_event

        except Exception as e:
            log.warning(f"[MT5CloseDetect] _handle_mt5_close failed for ticket {ticket}: {e}")
            return None

    def _select_cycle_symbols(self) -> list[str]:
        """Scanner-driven Top-N pairs when enabled and wired up; the static
        symbol list otherwise (or on any scanner failure).

        Day 37+ professional upgrade: when scanner is disabled, fall back to
        SessionAwarePairSelector instead of returning ALL 28 pairs every cycle.
        This makes each cycle focus on 8-12 pairs that are active in the
        current trading session (London/NY/Tokyo) instead of blindly scanning
        all 28 — faster cycles, higher-quality signals.
        """
        # 1. Scanner-driven mode (if enabled)
        if self.use_scanner and self.scanner:
            try:
                ranked = self.scanner.scan()
                top = self.scanner.get_top_opportunities(ranked)
                scanned = [opp["symbol"] for opp in top]
                if scanned:
                    return scanned
            except Exception as e:
                log.warning(f"[System] Scanner failed, falling back to session-aware selection: {e}")

        # 2. Session-aware fallback (Day 37+ professional default)
        try:
            from core.professional_tools import get_pair_selector
            selector = get_pair_selector(self.symbols)
            pairs, session = selector.select_with_session(top_n=len(self.symbols))
            if pairs:
                log.info(f"[System] Session-aware pair selection: {session} → {len(pairs)} pairs")
                return pairs
        except Exception as e:
            log.warning(f"[System] Session-aware selector failed: {e}")

        # 3. Final fallback: all symbols
        return self.symbols

    def _spawn_trader(self, symbol: str) -> AITrader:
        symbol = self._clean_symbol(symbol)
        trader = self._build_trader(symbol)
        self.traders[symbol] = trader
        return trader

    def backup_state(self, force: bool = False) -> Path | None:
        now = datetime.now(timezone.utc)
        if not force and self._last_backup:
            due_at = self._last_backup + timedelta(minutes=self.backup_interval_minutes)
            if now < due_at:
                return None

        timestamp = now.strftime("%Y%m%d_%H%M%S")
        backup_dir = Path("backups") / timestamp
        backup_dir.mkdir(parents=True, exist_ok=True)

        for relative_path in [
            "database/trader.db",
            "memory/trader.db",
            "memory/trade_memory.json",
            "memory/daily_risk.json",
            "memory/analysis_history.json",
            "memory/circuit_breaker_state.json",
            "memory/pending_approvals.json",
        ]:
            src = Path(relative_path)
            if src.exists():
                shutil.copy2(src, backup_dir / src.name)

        self._last_backup = now
        log.info(f"[System] Backup created: {backup_dir}")
        return backup_dir

    def _handle_cycle_errors(self, errors: list[str]) -> None:
        self._consecutive_error_cycles += 1
        self._pause_until = datetime.now(timezone.utc) + timedelta(minutes=self.cooldown_minutes)
        reason = "; ".join(errors[:3])
        log.error(f"[Recovery] Pausing trading after errors: {reason}")

        if self.notifier:
            self._notify_warning(
                f"System warning: trading paused for recovery. {reason}",
                f"{self.cooldown_minutes} minutes",
            )

    def _sync_risk_state(self) -> None:
        # Day 96 bugfix: was reading trader._paper directly (always empty
        # in mt5_demo mode) — now reuses AITrader._get_live_open_pairs(),
        # which reads real MT5 positions when execution_mode=mt5_demo.
        open_pairs = []
        for trader in self.traders.values():
            open_pairs.extend(trader._get_live_open_pairs())
        for trader in self.traders.values():
            trader._risk.sync_open_positions(open_pairs)

    def _start_telegram_commands(self) -> None:
        """
        Telegram polling boot_alerts phase এ already start হয়ে গেছে।
        এখানে আর start করার দরকার নেই — duplicate = 409 Conflict।
        শুধু log করি যে notifier available আছে কিনা।
        """
        if self.notifier:
            log.info("[System] Telegram notifier ready (polling already started by boot_alerts)")
        else:
            log.info("[System] Telegram notifier not available — skipping")

    def _notify_warning(self, event_name: str, time_remaining: str) -> None:
        if not self.notifier:
            return
        # Bug fix: use _run_async instead of asyncio.run (avoids event loop closure)
        self._run_async_safe(self.notifier.notify_news_warning(event_name, time_remaining))

    def _run_async_safe(self, coro) -> None:
        """Run an async coroutine safely.

        Issue 9 fix: delegate to utils.async_utils.run_coro_sync(), the same
        persistent-loop helper AITrader._run_async() uses, instead of a
        second, duplicated asyncio.get_event_loop()/new_event_loop() dance.
        Avoids the deprecated get_event_loop() call and the "Event loop is
        closed" errors that come from repeatedly creating/discarding loops.
        """
        try:
            from utils.async_utils import run_coro_sync
            run_coro_sync(coro)
        except Exception as e:
            log.warning(f"Telegram notify failed: {e}")

    def _manual_pause_active(self) -> bool:
        return bool(telegram_module and getattr(telegram_module, "IS_TRADING_PAUSED", False))

    def _is_paused(self) -> bool:
        if not self._pause_until:
            return False
        if datetime.now(timezone.utc) >= self._pause_until:
            log.info("[Recovery] Cooldown completed. Resuming trading loop safely.")
            self._pause_until = None
            return False
        return True

    def _sleep_remaining(self, cycle_started: float) -> None:
        elapsed = time.time() - cycle_started
        remaining = max(0, self.poll_seconds - elapsed)
        if remaining:
            time.sleep(remaining)

    def _write_runtime_report(self, report: dict | None = None) -> Path:
        report = report or self._build_system_report()
        report_dir = Path("reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / "latest_report.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return path

    def _build_system_report(self) -> dict:
        sample_trader = next(iter(self.traders.values()))
        stats = sample_trader._db.get_overall_stats(starting_balance=self.balance)
        recent = sample_trader._db.get_trade_history(limit=20)

        best_setup = "N/A"
        biggest_mistake = "N/A"
        if not recent.empty and "pattern" in recent.columns:
            wins = recent[recent["result"] == "WIN"]
            losses = recent[recent["result"] == "LOSS"]
            if not wins.empty:
                best_setup = str(wins["pattern"].fillna("unknown").mode().iloc[0])
            if not losses.empty:
                biggest_mistake = str(losses["pattern"].fillna("unknown").mode().iloc[0])

        avg_rr = 0
        closed_count = len(recent.index)
        if closed_count and {"entry", "sl", "tp"}.issubset(set(recent.columns)):
            rr_values = []
            for _, row in recent.iterrows():
                try:
                    risk = abs(float(row["entry"]) - float(row["sl"]))
                    reward = abs(float(row["tp"]) - float(row["entry"]))
                    rr_values.append(round(reward / risk, 2) if risk else 0)
                except Exception as e:
                    log.warning(f"Suppressed exception at line 2061: {e}")
                    continue
            if rr_values:
                avg_rr = round(sum(rr_values) / len(rr_values), 2)

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mode": self.execution_mode.upper(),
            "scanner": "ON" if self.use_scanner else "OFF",
            "pairs": self.symbols,
            "active_pairs": list(self.traders.keys()),
            "timeframe": self.timeframe,
            "balance": self.balance,
            "system_state": "PAUSED" if self._manual_pause_active() or self._is_paused() else "RUNNING",
            "circuit_breaker": self.circuit_breaker.get_status(),
            "summary": {
                "trades": stats.get("total", 0),
                "wins": stats.get("wins", 0),
                "losses": stats.get("losses", 0),
                "win_rate": stats.get("win_rate", 0),
                "profit": stats.get("total_pnl", 0),
                "balance": stats.get("balance", self.balance),
                "open_positions": stats.get("open_trades", 0),
                "average_rr": avg_rr,
                "best_setup": best_setup,
                "biggest_mistake": biggest_mistake,
            },
            "last_results": self._last_results[-len(self.symbols):],
        }

    def _clean_symbol(self, symbol: str) -> str:
        # Round-14 fix: see backtest/simulator.py — blanket "USDT"->"USD"
        # replace corrupted real FX codes containing "USDT" as a
        # substring: USDTRY (USD/Turkish Lira) -> USDRY, USDTHB
        # (USD/Thai Baht) -> USDHB. This is the definition that actually
        # runs (Python keeps the last of two same-named methods in this
        # class — see the now-dead duplicate near line 2533).
        cleaned = str(symbol).upper().replace("/", "").replace("=X", "").strip()
        if cleaned.endswith("USDT"):
            cleaned = cleaned[:-1]
        return cleaned

    # ── Day 37+ runtime unification: health & introspection ──────────
    def health_status(self) -> dict:
        """Return a snapshot of system health for the dashboard / health monitor.
        Aggregates circuit-breaker state, open positions, last cycle results,
        and any registered runtime metrics."""
        cb = self.circuit_breaker.get_status() if self.circuit_breaker else None
        open_positions = []
        for trader in self.traders.values():
            try:
                open_positions.extend(trader._paper.get_open_positions())
            except Exception as e:
                log.warning(f"Suppressed exception at line 2105: {e}")
                pass
        return {
            "running": not self._stop_requested,
            "paused": self._is_paused(),
            "manual_pause": self._manual_pause_active(),
            "execution_mode": self.execution_mode,
            "approval_mode": self.approval.mode_name if self.approval else "N/A",
            "symbols": self.symbols,
            "active_traders": list(self.traders.keys()),
            "circuit_breaker": cb,
            "open_positions": len(open_positions),
            "last_cycle_results": (self._last_results or [])[-len(self.symbols):],
            "consecutive_error_cycles": self._consecutive_error_cycles,
            "registry_wired": self._registry is not None,
            "bus_wired": self._bus is not None,
            "metrics_wired": self._metrics is not None,
        }

    def get_runtime_metrics(self) -> dict:
        """Return the runtime metrics report, if available."""
        if self._metrics is not None:
            try:
                return self._metrics.build_report()
            except Exception as e:
                return {"error": str(e)}
        return {"error": "runtime metrics not wired"}