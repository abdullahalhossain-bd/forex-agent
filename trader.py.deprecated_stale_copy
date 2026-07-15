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
import shutil
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
from utils.session import SessionAnalyzer
from visualization.chart import ChartEngine

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
except Exception:
    _RUNTIME_INFRA_AVAILABLE = False
    EventBus = None
    get_bus = None
    RuntimeMetrics = None
    get_metrics = None
    ServiceRegistry = None

try:
    import alerts.telegram_bot as telegram_module
    from alerts.telegram_bot import TelegramNotifier, start_telegram_bot_polling
except Exception:
    telegram_module = None
    TelegramNotifier = None
    start_telegram_bot_polling = None

try:
    from learning.mistake_analyzer import AdvancedMistakeAnalyzer
except Exception:
    AdvancedMistakeAnalyzer = None

try:
    from scanner.market_scanner import MarketScanner
except Exception:
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
    ):
        self.balance = balance
        self.symbol = self._clean_symbol(symbol)
        self.timeframe = timeframe
        self.notifier = notifier
        self.execution_mode = (execution_mode or EXECUTION_MODE).lower()
        self._last_decision_candle = None

        # ── Runtime infrastructure (Day 37+ unification) ──────────────
        # The registry is optional — when supplied, AITrader pulls shared
        # singletons from it (TradeMemory, CircuitBreaker, etc.) instead of
        # constructing fresh copies, and publishes events / metrics to the
        # central bus. When absent, the trader behaves exactly as before.
        self._registry = registry
        self._bus = get_bus() if _RUNTIME_INFRA_AVAILABLE else None
        self._metrics = get_metrics() if _RUNTIME_INFRA_AVAILABLE else None

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
        if registry is not None:
            shared_db = registry.try_resolve("db")
            self._db = shared_db if shared_db is not None else TraderDB()
        else:
            self._db = TraderDB()
        self._paper = PaperTrader(starting_balance=paper_balance, db=self._db)
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
            except Exception:
                pass
        self._publish("system.error", {"channel": channel, "symbol": self.symbol, "reason": reason})

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
            direction = risk_out.get("signal") or dec_out.get("decision") or "WAIT"
            confidence = float(dec_out.get("confidence", 0) or 0)

            # Pip value per lot — RiskEngine uses get_pip_value_usd(symbol).
            # We approximate with the same lookup so the sizer stays in sync.
            try:
                from core.constants import get_pip_value_usd
                pip_value = get_pip_value_usd(self.symbol)
            except Exception:
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
                except Exception:
                    pass

            # Loss streak — pull from the kill switch state if available,
            # else default to 0 (no penalty).
            consecutive_losses = 0
            if self._kill_switch is not None:
                try:
                    ks_state = self._kill_switch.status() if hasattr(self._kill_switch, "status") else {}
                    consecutive_losses = int(ks_state.get("consecutive_losses", 0) or 0)
                except Exception:
                    pass

            # Open positions for correlation/heat check.
            open_positions = []
            try:
                for pos in self._paper.get_open_positions():
                    open_positions.append({
                        "pair": pos.get("pair", ""),
                        "direction": pos.get("signal") or pos.get("direction", ""),
                        "risk_usd": float(pos.get("risk_usd", 0) or 0),
                    })
            except Exception:
                pass

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
            except Exception:
                pass

            # Base risk % from RiskEngine (typically 1.0%).
            base_risk_pct = float(risk_out.get("risk_pc", 1.0) or 1.0) / 100.0
            # Capital tier multiplier — Tier 3 = 1.0 by default; lower
            # tiers reduce.  We pull this from the live risk manager when
            # available, otherwise default to 1.0 (mature system).
            tier_mult = 1.0
            if self._live_risk_manager is not None:
                try:
                    tier_mult = float(self._live_risk_manager.current_tier.tier_mult)
                except Exception:
                    pass

            # New equity high flag — paper trader balance vs starting.
            is_new_high = False
            try:
                is_new_high = float(self._paper.balance) >= float(self.balance) * 1.10
            except Exception:
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
            except Exception:
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
            # The sizer must NEVER break the trading pipeline.  Log and
            # fall through with the original risk_out intact.
            log.warning(f"[Day 76 Sizer] {self.symbol} sizing failed — using RiskEngine base lot: {e}")

        return risk_out

    def get_signal(self, show_chart: bool = False, auto_paper_trade: bool = True) -> dict:
        return self.run_cycle(show_chart=show_chart, auto_paper_trade=auto_paper_trade)

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
        except Exception:
            pass

        # ── Day 81+ — Start signal pipeline debugger for this cycle ──
        try:
            from monitoring.signal_debugger import get_signal_debugger
            debugger = get_signal_debugger()
            debugger.start_cycle(self.symbol, self.timeframe)
        except Exception:
            debugger = None

        # ── Day 84+ — Trade frequency controller (early bail if daily cap hit) ──
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
                except Exception: pass
                return self._error_result("Daily trade cap reached")
        except Exception as e:
            freq_ctrl = None
            try:
                from core.trade_decision_log import log_cycle_error
                log_cycle_error(self.symbol, str(e), "frequency_ctrl_init")
            except Exception: pass

        session_ctx = SessionAnalyzer().get_current_session()
        latest_price = None

        log.info("[1/9] Market Agent...")
        with self._stage(f"aitrader.{self.symbol}.market"):
            market_out = self._market.run()
        if "error" in market_out:
            # Don't send Telegram alert for market fetch failures — they're
            # common (market closed, symbol temporarily unavailable) and
            # would spam the user. Just log locally.
            log.warning(f"[Market] {self.symbol} data fetch failed — skipping this cycle")
            try:
                from core.trade_decision_log import log_decision
                log_decision(symbol=self.symbol, signal="NO TRADE",
                             reject_stage="market_data",
                             reject_reason=f"Market data fetch failed: {market_out.get('error')}")
            except Exception: pass
            if debugger:
                debugger.record("market_data", "ERROR", market_out.get("error", "fetch_failed"))
                debugger.record_final("NO_TRADE", "Market data fetch failed")
                debugger.log_cycle_summary()
                debugger.save_to_file()
            return self._error_result(f"Market Agent: {market_out['error']}")

        # Record market data success
        if debugger:
            ind_ctx = market_out.get("ind_ctx", {}) or {}
            debugger.record("market_data", "OK",
                            f"price={ind_ctx.get('close', '?')} trend={ind_ctx.get('trend', '?')}")

        ind = market_out.get("ind_ctx", {})
        # Day 81+ hotfix: Indicators.get_ai_context() uses "price" key, not "close"
        latest_price = ind.get("close") or ind.get("price")
        candle_time = self._extract_candle_time(market_out)
        closed_now = []

        if auto_paper_trade and latest_price:
            closed_now = self._paper.update_price(self.symbol, latest_price)
        closed_processed = self._process_closed_trades(closed_now)

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
            except Exception: pass
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

        log.info("[3/9] Analysis Agent...")
        analysis_out = self._analysis.run(market_out)
        if "error" in analysis_out:
            try:
                from core.trade_decision_log import log_decision
                log_decision(symbol=self.symbol, signal="NO TRADE",
                             reject_stage="analysis_agent",
                             reject_reason=f"Analysis error: {analysis_out.get('error')}")
            except Exception: pass
            return self._error_result(f"Analysis Agent: {analysis_out['error']}")

        memory_ctx = self._memory.get_context_for_ai(self.symbol)
        pattern = self._extract_pattern(market_out)
        regime_str = market_out.get("regime", {}).get("regime", "")
        pat_ctx = self._memory.get_pattern_context(self.symbol, regime_str, pattern)

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
        try:
            _live_open = [t.get("pair") for t in self._paper.get_open_positions() if t.get("pair")]
        except Exception as e:
            # Day 90 bugfix: log at WARNING (not debug) — this should never
            # happen in normal operation, and if it does we need to know.
            log.warning(
                f"[Risk] PaperTrader.get_open_positions() raised: {e} — "
                f"using empty open-pairs list (correlation check will pass)"
            )
            _live_open = []

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

        log.info("[6/9] Safety Guard (Permission + Correlation)...")
        perm_out = self._perm.check(
            decision_out=dec_out,
            risk_out=risk_out,
            news_ctx=analysis_out.get("news_ctx", {}),
            session_ctx=self._session_permission_context(session_ctx),
        )

        if self._paper.has_open_position(self.symbol, perm_out.get("final_action")):
            perm_out["allowed"] = False
            perm_out["final_action"] = "NO TRADE"
            perm_out["checks"].append(
                {
                    "check": "Duplicate trade",
                    "passed": False,
                    "detail": f"{self.symbol} {dec_out.get('decision')} already open",
                }
            )

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
                perm_out["final_action"] = "NO TRADE"
                perm_out["checks"].append(
                    {
                        "check": "Correlation filter",
                        "passed": False,
                        "detail": "Correlated pair group already has an open position",
                    }
                )

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
        except Exception:
            pass

        log.info("[7/9] Learning Agent...")
        self._learn.save_decision(dec_out, analysis_out, market_out)
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
            except Exception:
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
                    }
                )
            if trade:
                result["paper_trade_id"] = trade.get("id")
                result["paper_balance"] = self._paper.balance
                self._risk.record_trade_open(self.symbol)
                self._notify_trade_open(trade, result, dec_out)
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
                    except Exception:
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
        except Exception:
            pass

        # ── Day 84+ — Record trade in frequency controller ──
        if freq_ctrl:
            _final_for_freq = result.get("final_action")
            if _final_for_freq in ("BUY", "SELL"):
                try:
                    freq_ctrl.record_trade(symbol=self.symbol, direction=_final_for_freq)
                except Exception:
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
        log.info(f"Trade #{trade_id} closed: {result} | PnL: ${pnl}")

    def get_paper_dashboard(self) -> dict:
        return self._paper.get_dashboard()

    def print_paper_dashboard(self) -> None:
        self._paper.print_dashboard()

    def get_learning_report(self):
        self._learning.print_report()

    def get_memory_stats(self):
        self._memory.print_stats()

    def sync_risk_with_open_positions(self) -> None:
        open_pairs = [trade.get("pair") for trade in self._paper.get_open_positions()]
        self._risk.sync_open_positions(open_pairs)

    def _process_closed_trades(self, closed_now: list[dict]) -> list[dict]:
        processed = []
        for trade in closed_now:
            context = trade.get("context") or {}
            memory_trade_id = context.get("memory_trade_id")
            rr_ratio = context.get("rr_ratio") or trade.get("rr_ratio", 0)
            trade["rr_ratio"] = rr_ratio

            self._risk.record_trade_close(trade["pair"], trade["pnl"])
            self._circuit_breaker.record_result(trade["result"], trade["pnl"])

            if memory_trade_id:
                try:
                    self._memory.on_trade_closed(memory_trade_id, trade["result"], trade["pnl"])
                    if self._mistake_analyzer:
                        self._mistake_analyzer.analyze_closed_trade(memory_trade_id)
                except Exception as e:
                    log.warning(f"[Learning] Close sync failed for memory trade #{memory_trade_id}: {e}")

            self._notify_trade_close(trade)
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
                except Exception:
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

        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "version": self.VERSION,
            "elapsed_sec": elapsed,
            "price": ind.get("close"),
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
            "decision": dec_out.get("decision"),
            "confidence": dec_out.get("confidence"),
            "trade_allowed": perm_out["allowed"],
            "final_action": perm_out["final_action"],
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
            "total_decisions": stats.get("total_decisions", 0),
            "win_rate": stats.get("win_rate", "N/A"),
            "session": self._format_session_label(session_ctx),
            "decision_candle": candle_time,
            "closed_trades": closed_trades or [],
            # Day 76 — full sizer breakdown for journal/telegram/dashboard.
            "position_sizing": risk_out.get("position_sizing"),
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
            log.info(f"  DECISION     : {r['decision']} ({r['confidence']}%)")
            log.info("  ──")

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
        log.info(f"  Memory       : {r['total_decisions']} decisions | WR: {r['win_rate']}%")
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
            db.save_patterns(df, self.symbol, self.timeframe)
            db.save_analysis(
                self.symbol,
                self.timeframe,
                analysis_out["bias_result"]["net_score"],
                analysis_out["bias_result"]["bias"],
                combined,
            )
            AnalysisHistory().save(
                self.symbol,
                self.timeframe,
                analysis_out["bias_ctx"],
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
        except Exception:
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
        try:
            asyncio.run(coro)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(coro)
            finally:
                loop.close()
        except Exception as e:
            log.warning(f"Telegram notify failed: {e}")

    def _clean_symbol(self, symbol: str) -> str:
        return str(symbol).upper().replace("/", "").replace("=X", "").replace("USDT", "USD").strip()

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
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.ensure_future(self.notifier.send_message(alert_msg))
                    else:
                        loop.run_until_complete(self.notifier.send_message(alert_msg))
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(self.notifier.send_message(alert_msg))
                    loop.close()
        except Exception as e:
            log.debug(f"[System] Day 66 news alert failed: {e}")

        log.info(
            f"[System] Starting autonomous loop | Pairs={self.symbols} | "
            f"Timeframe={self.timeframe} | Mode={self.execution_mode.upper()} | "
            f"Scanner={'ON' if self.use_scanner else 'OFF'} | Balance=${self.balance} | "
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
            except Exception:
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

                cycle_results = []
                cycle_errors = []
                active_symbols = self._select_cycle_symbols()
                if self._metrics is not None:
                    try:
                        self._metrics.record_cycle()
                        self._metrics.set_gauge("trader.active_symbols", len(active_symbols))
                    except Exception:
                        pass

                for symbol in active_symbols:
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

                if cycle_errors:
                    self._handle_cycle_errors(cycle_errors)
                else:
                    self._consecutive_error_cycles = 0

                self.backup_state()
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
            except Exception:
                pass

        report = self._build_system_report()
        self._write_runtime_report(report)
        return report

    def stop(self) -> None:
        self._stop_requested = True

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
            pairs, session = selector.select_with_session(top_n=12)
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
        open_pairs = []
        for trader in self.traders.values():
            open_pairs.extend([trade.get("pair") for trade in trader._paper.get_open_positions()])
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
        try:
            asyncio.run(self.notifier.notify_news_warning(event_name, time_remaining))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self.notifier.notify_news_warning(event_name, time_remaining))
            finally:
                loop.close()
        except Exception as e:
            log.warning(f"[System] Warning notify failed: {e}")

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
                except Exception:
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
        return str(symbol).upper().replace("/", "").replace("=X", "").replace("USDT", "USD").strip()

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
            except Exception:
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