# hybrid/flow_controller.py  —  Day 49 | Hybrid Execution Flow Controller ⭐
# ============================================================
# Doc-এর পুরো architecture একটাই runnable loop-এ:
#
#   MT5 Data Pipeline → Python Analysis Engine
#         ↓                      ↓
#   Quant Decision         Vision Check (TradingView + ChartReader)
#         ↓                      ↓
#         └──────────┬───────────┘
#                     ↓
#           DecisionValidator (Day 49 conflict gate)
#                     ↓
#              DecisionAgent (Day 42 weighted vote)
#                     ↓
#              RiskAgent (Day 12 risk check)
#                     ↓
#              ExecutionRouter (Day 49 mode/emergency gate)
#                     ↓
#              OrderManager/PaperTrader (Day 33 / pre-37)
#                     ↓
#              PositionManager (Day 33/35 management)
#                     ↓
#              LearningAgent (Day 12 record)
#
# এই controller কোনো analysis logic পুনরায় লেখে না — MarketAgent,
# AnalysisAgent, RiskAgent, DecisionAgent, OrderManager,
# PositionManager, LearningAgent — সব already-built agent reuse করে।
# এর একমাত্র দায়িত্ব: doc-এর "Hybrid" thesis বাস্তবায়ন করা —
#       Quant (fast, API) + Vision (confirmation, screen) আলাদা
#       ভূমিকায় থেকে, conflict হলে force trade না করে, mode/emergency
#       gate পার হয়ে, একসাথে কাজ করা।
# ============================================================

import os
import time
from datetime import datetime, timezone

from utils.logger import get_logger
from hybrid.decision_validator import DecisionValidator
from hybrid.execution_router import ExecutionRouter, ApprovalMode
from hybrid.confidence_calibrator import ConfidenceCalibrator
from core.constants import MEMORY_DIR

log = get_logger("flow_controller")

TRADE_LOG_DIR = str(MEMORY_DIR / "hybrid_trade_log")


class FlowController:
    """
    Usage:
        controller = FlowController(
            market_agent_factory=lambda symbol, tf: MarketAgent(symbol, tf),
            analysis_agent=AnalysisAgent(chart_reader=chart_reader),
            risk_agent=RiskAgent(account_balance=10000),
            decision_agent=DecisionAgent(),
            execution_router=ExecutionRouter(order_manager=om, execution_mode="paper"),
            learning_agent=LearningAgent(),
            position_manager=position_manager,   # optional — open trade-গুলো manage করার জন্য
        )

        result = controller.run_cycle("EURUSD", timeframe="15m")

        # বা continuous loop:
        controller.run_loop(symbols=["EURUSD", "GBPUSD"], interval_sec=60)
    """

    def __init__(
        self,
        market_agent_factory,
        analysis_agent,
        risk_agent,
        decision_agent,
        execution_router: ExecutionRouter,
        learning_agent=None,
        position_manager=None,
        decision_validator: DecisionValidator = None,
        confidence_calibrator: ConfidenceCalibrator = None,
        spread_monitor=None,
        news_calendar=None,
        memory_ctx_provider=None,
    ):
        """
        market_agent_factory : callable(symbol, timeframe) -> MarketAgent instance
                                (MarketAgent symbol/timeframe constructor-bound, তাই
                                factory pattern — প্রতি cycle/pair-এ নতুন instance)
        analysis_agent        : AnalysisAgent (Day 47, chart_reader inject করা)
        risk_agent             : RiskAgent (Day 12)
        decision_agent          : DecisionAgent (Day 42)
        execution_router        : ExecutionRouter (Day 49)
        learning_agent           : LearningAgent (Day 12) — optional কিন্তু strongly recommended
        position_manager          : PositionManager (Day 33/35) — open trade manage করার জন্য
        decision_validator         : DecisionValidator (Day 49) — না দিলে নিজে বানাবে
        confidence_calibrator        : ConfidenceCalibrator (Day 49) — না দিলে নিজে বানাবে
        spread_monitor                 : broker/spread_monitor.py SpreadMonitor (optional)
        news_calendar                   : broker/economic_calendar.py EconomicCalendar (optional)
        memory_ctx_provider               : callable() -> dict, MasterAnalyst-এর memory_ctx
                                             বানানোর জন্য (LearningAgent.get_performance_stats()
                                             wrap করে দিতে পারো)
        """
        self.market_agent_factory = market_agent_factory
        self.analysis_agent = analysis_agent
        self.risk_agent = risk_agent
        self.decision_agent = decision_agent
        self.execution_router = execution_router
        self.learning_agent = learning_agent
        self.position_manager = position_manager
        self.decision_validator = decision_validator or DecisionValidator()
        self.confidence_calibrator = confidence_calibrator or ConfidenceCalibrator()
        self.spread_monitor = spread_monitor
        self.news_calendar = news_calendar
        self.memory_ctx_provider = memory_ctx_provider

        os.makedirs(TRADE_LOG_DIR, exist_ok=True)

    # ═══════════════════════════════════════════════════════
    # MAIN CYCLE — ONE SYMBOL, ONE PASS THROUGH THE FULL PIPELINE
    # ═══════════════════════════════════════════════════════

    def run_cycle(self, symbol: str, timeframe: str = "15m") -> dict:
        """
        Doc-এর Step 1-8 — একটা সম্পূর্ণ pass। প্রতিটা ধাপে ব্যর্থতা/block
        হলে early-return করে, পরের ধাপে যায় না (no force trade)।
        """
        cycle_log = {"symbol": symbol, "timeframe": timeframe,
                     "started_at": datetime.now(timezone.utc).isoformat()}

        if self.execution_router.is_emergency_stopped():
            return self._finish(cycle_log, stage="EMERGENCY_STOP",
                                 reason="Router is in emergency-stopped state")

        # ── Step 1+2: MT5 data + Python analysis engine ──
        market_agent = self.market_agent_factory(symbol, timeframe)
        market_out = market_agent.run()
        # H-H6 fix: don't assume market_agent.run() always returns a dict —
        # every downstream field access below is a .get() call that would
        # raise AttributeError on a non-dict.
        if not isinstance(market_out, dict):
            return self._finish(cycle_log, stage="MARKET_DATA",
                                 reason=f"market_agent.run() returned {type(market_out).__name__}, expected dict")
        cycle_log["market_out_error"] = market_out.get("error")
        if "error" in market_out:
            return self._finish(cycle_log, stage="MARKET_DATA",
                                 reason=f"Market data error: {market_out['error']}")

        # H-C4/H-M18 fix: memory_ctx_provider is an arbitrary user-supplied
        # callable, and get_performance_stats() shape isn't guaranteed to
        # match what AnalysisAgent expects. Rather than assume the caller
        # got it right, ensure we always hand AnalysisAgent a dict (never
        # None, which would blow up the first .get() call inside it).
        try:
            memory_ctx = self.memory_ctx_provider() if self.memory_ctx_provider else (
                self.learning_agent.get_performance_stats() if self.learning_agent else {}
            )
        except Exception as e:
            log.warning(f"[FlowController] memory_ctx_provider failed ({e}) — using empty context")
            memory_ctx = {}
        if not isinstance(memory_ctx, dict):
            log.warning(f"[FlowController] memory_ctx_provider returned {type(memory_ctx).__name__}, not dict — using empty context")
            memory_ctx = {}

        analysis_out = self.analysis_agent.run(market_out, memory_ctx=memory_ctx)
        # H-H7 fix: same non-dict defense as market_out above — analysis_out
        # keys (master_ctx, vision_ctx, vision_fusion, vision, signal) are
        # all accessed via .get() a few lines down.
        if not isinstance(analysis_out, dict):
            return self._finish(cycle_log, stage="ANALYSIS",
                                 reason=f"analysis_agent.run() returned {type(analysis_out).__name__}, expected dict")
        if "error" in analysis_out:
            return self._finish(cycle_log, stage="ANALYSIS",
                                 reason=f"Analysis error: {analysis_out['error']}")

        # ── Step 3+4: Vision confirmation (AnalysisAgent.run() এর ভিতরেই
        #              Day 47 ChartReader call হয়ে গেছে — vision_ctx/vision_fusion
        #              বের করে নিচ্ছি, আলাদা করে আবার screen automation চালানো হচ্ছে না) ──
        vision_ctx = analysis_out.get("vision_ctx", {})
        vision_fusion = analysis_out.get("vision_fusion", {})
        vision_available = bool(analysis_out.get("vision")) and not analysis_out.get("vision", {}).get("error")

        quant_signal = analysis_out.get("master_ctx", {}).get("master_signal") \
            or analysis_out.get("signal", {}).get("signal", "WAIT")
        quant_confidence = analysis_out.get("master_ctx", {}).get("master_confidence") \
            or analysis_out.get("signal", {}).get("confidence", 0)

        vision_signal = vision_ctx.get("vision_trend", "UNKNOWN")
        # vision_trend "BULLISH"/"BEARISH"/"SIDEWAYS" → BUY/SELL/WAIT-এ map করো
        vision_signal_norm = {"BULLISH": "BUY", "BEARISH": "SELL"}.get(
            (vision_signal or "").upper(), "WAIT"
        )
        vision_confidence = vision_ctx.get("vision_confidence", 0)

        # ── Step 5: Decision Fusion — Day 49 hard conflict gate ──
        validator_verdict = self.decision_validator.validate(
            quant_signal=quant_signal,
            quant_confidence=quant_confidence,
            vision_signal=vision_signal_norm,
            vision_confidence=vision_confidence,
            vision_available=vision_available,
        )
        cycle_log["validator_verdict"] = validator_verdict

        if validator_verdict["has_hard_conflict"]:
            return self._finish(cycle_log, stage="CONFLICT_BLOCKED",
                                 reason=validator_verdict["reason"])

        if validator_verdict["final_signal"] == "NO TRADE":
            return self._finish(cycle_log, stage="NO_TRADE",
                                 reason=validator_verdict["reason"])

        # ── Confidence calibration (Bonus #2) — reality-check before risk sizing ──
        calibration = self.confidence_calibrator.calibrate(validator_verdict["final_score"])
        cycle_log["calibration"] = calibration

        # ── Step 6: Risk Validation ──
        entry_price = analysis_out.get("master_ctx", {}).get("master_entry") \
            or market_out.get("ind_ctx", {}).get("price", market_out.get("ind_ctx", {}).get("close"))

        risk_out = self.risk_agent.calculate(
            signal=validator_verdict["final_signal"],
            entry=entry_price,
            ind_ctx=market_out.get("ind_ctx", {}),
            regime=market_out.get("regime", {}),
            symbol=symbol,
        )
        # H-C7/H-M4 fix: don't assume calculate() always returns a dict —
        # treat an unexpected None/non-dict as an unapproved trade rather
        # than letting .get() raise AttributeError further down.
        if not isinstance(risk_out, dict):
            log.error(f"[FlowController] risk_agent.calculate() returned {type(risk_out).__name__}, expected dict — blocking trade")
            return self._finish(cycle_log, stage="RISK_REJECTED",
                                 reason="RiskAgent returned invalid result")
        cycle_log["risk_out"] = {"approved": risk_out.get("approved"), "reject_reason": risk_out.get("reject_reason")}

        if not risk_out.get("approved"):
            return self._finish(cycle_log, stage="RISK_REJECTED",
                                 reason=risk_out.get("reject_reason"))

        # ── Extra safety: spread + news (doc Step 6 checklist items) ──
        health_ctx = self._build_health_ctx(symbol, market_out, vision_available)
        if self.spread_monitor and health_ctx.get("spread_pips") is not None:
            spread_check = self.spread_monitor.check(
                symbol=symbol, current_spread_pips=health_ctx["spread_pips"],
                news_active=health_ctx.get("news_active", False),
            )
            if not spread_check["allowed"]:
                return self._finish(cycle_log, stage="SPREAD_BLOCKED", reason=spread_check["reason"])

        if self.news_calendar:
            news_check = self.news_calendar.check_news_window(symbol=symbol)
            if not news_check["trade_allowed"]:
                return self._finish(cycle_log, stage="NEWS_BLOCKED", reason=news_check["reason"])

        # ── DecisionAgent — Day 42 weighted-vote final decision ──
        decision_out = self.decision_agent.decide(market_out, analysis_out, risk_out)
        # H-H22 fix: decide() returning None would crash the next line
        # (decision_out["confidence"] = ...) with a TypeError — treat it as
        # a no-trade instead of propagating an unhandled exception.
        if not isinstance(decision_out, dict):
            log.error(f"[FlowController] decision_agent.decide() returned {type(decision_out).__name__}, expected dict — treating as WAIT")
            return self._finish(cycle_log, stage="DECISION_WAIT", reason="DecisionAgent returned invalid result")
        # Day 49 calibrated confidence override — calibration historical reality
        # দিয়ে adjust করা, raw model confidence-এর চেয়ে বেশি trustworthy
        #
        # ARCHITECTURAL FIX (institutional refactor): preserve the analysis-
        # layer confidence as `raw_confidence` so downstream consumers can
        # see what the analysts actually said. The calibrated value is
        # stored separately as `calibrated_confidence`. The legacy
        # `confidence` field is kept as the calibrated value for backward
        # compat with existing consumers (ExecutionRouter, learning, etc.).
        decision_out["raw_confidence"] = decision_out.get("confidence")
        decision_out["calibrated_confidence"] = calibration["calibrated_confidence"]
        decision_out["confidence"] = calibration["calibrated_confidence"]  # legacy alias
        cycle_log["decision_out"] = {
            "decision": decision_out.get("decision"),
            "confidence": decision_out.get("confidence"),
        }

        if decision_out.get("decision") not in ("BUY", "SELL"):
            return self._finish(cycle_log, stage="DECISION_WAIT", reason="DecisionAgent → WAIT/NO TRADE")

        # ── Step 7: Execution (Router handles mode gate + emergency stop) ──
        exec_result = self.execution_router.route(decision_out, symbol, health_ctx=health_ctx)
        cycle_log["execution"] = exec_result

        if exec_result.get("executed") and self.position_manager and exec_result.get("ticket"):
            db_id = self._log_to_learning(decision_out, analysis_out, market_out)
            if db_id:
                self.position_manager.register_open(exec_result["ticket"], db_id)
        elif self.learning_agent:
            self._log_to_learning(decision_out, analysis_out, market_out)

        return self._finish(
            cycle_log, stage="EXECUTED" if exec_result.get("executed") else exec_result.get("action", "BLOCKED"),
            reason=exec_result.get("reason"),
        )

    # ═══════════════════════════════════════════════════════
    # CONTINUOUS LOOP
    # ═══════════════════════════════════════════════════════

    def run_loop(self, symbols: list, timeframe: str = "15m",
                 interval_sec: int = 60, stop_flag=None, max_cycles: int = None) -> None:
        """
        Doc-এর "Autonomous Trading Loop"। Blocking — সাধারণত main.py/scheduler
        থেকে চালানো হয়। stop_flag: callable, True হলে loop থামবে।
        """
        log.info(
            f"[FlowController] 🔄 Starting hybrid loop | symbols={symbols} "
            f"tf={timeframe} interval={interval_sec}s"
        )
        cycles_done = 0
        try:
            while True:
                if stop_flag and stop_flag():
                    log.info("[FlowController] Stop flag set — exiting loop")
                    break
                if max_cycles is not None and cycles_done >= max_cycles:
                    log.info(f"[FlowController] Reached max_cycles={max_cycles} — exiting")
                    break

                if self.execution_router.is_emergency_stopped():
                    log.error("[FlowController] 🚨 Emergency stop active — loop paused, not auto-resuming")
                    break

                # PositionManager-এর poll প্রথমে — existing positions manage হোক analysis-এর আগে
                if self.position_manager:
                    try:
                        self.position_manager.poll_once()
                    except Exception as e:
                        log.error(f"[FlowController] PositionManager poll error: {e}")

                for symbol in symbols:
                    try:
                        result = self.run_cycle(symbol, timeframe)
                        log.info(
                            f"[FlowController] {symbol} → stage={result['stage']} "
                            f"reason={result.get('reason')}"
                        )
                    except Exception as e:
                        log.error(f"[FlowController] Cycle error for {symbol}: {e}", exc_info=True)

                    if self.execution_router.is_emergency_stopped():
                        break   # emergency triggered mid-symbol-loop — stop immediately

                cycles_done += 1
                time.sleep(interval_sec)
        except KeyboardInterrupt:
            # H-M24 fix: previously an uncaught KeyboardInterrupt (Ctrl+C,
            # or a supervisor sending SIGINT) would propagate straight out
            # mid-cycle with no chance to log a clean shutdown message.
            log.info("[FlowController] KeyboardInterrupt received — stopping loop cleanly")

    # ═══════════════════════════════════════════════════════
    # HEALTH CONTEXT BUILDER  (doc Bonus #4 inputs)
    # ═══════════════════════════════════════════════════════

    def _build_health_ctx(self, symbol: str, market_out: dict, vision_available: bool) -> dict:
        """ExecutionRouter._check_emergency_conditions()-এর জন্য সব signal জড়ো করো।"""
        mt5_connected = None
        # H-C13 fix: don't assume execution_router has execution_mode/
        # order_manager attributes with those exact names — use getattr
        # defaults so a missing/renamed attribute degrades to "unknown"
        # instead of raising AttributeError and killing the whole cycle.
        if getattr(self.execution_router, "execution_mode", None) == "mt5_demo":
            order_manager = getattr(self.execution_router, "order_manager", None)
            if order_manager is not None:
                conn = getattr(order_manager, "connection", None)
                mt5_connected = getattr(conn, "connected", None)

        return {
            "vision_available": vision_available,
            "require_vision": False,   # doc thesis: vision হলো confirmation, hard requirement না
            "mt5_connected": mt5_connected,
            "spread_pips": market_out.get("ind_ctx", {}).get("spread_pips"),
            "news_active": False,   # news_calendar আলাদা ভাবে check হয় run_cycle-এ
        }

    # ═══════════════════════════════════════════════════════
    # LEARNING + TRADE LOG  (doc Step 8)
    # ═══════════════════════════════════════════════════════

    def _log_to_learning(self, decision_out: dict, analysis_out: dict, market_out: dict):
        if not self.learning_agent:
            return None
        try:
            self.learning_agent.save_decision(decision_out, analysis_out, market_out)
        except Exception as e:
            log.warning(f"[FlowController] LearningAgent save error: {e}")
        return None

    def _finish(self, cycle_log: dict, stage: str, reason: str = None) -> dict:
        cycle_log["stage"] = stage
        cycle_log["reason"] = reason
        cycle_log["finished_at"] = datetime.now(timezone.utc).isoformat()
        self._save_cycle_log(cycle_log)
        icon = {
            "EXECUTED": "✅", "CONFLICT_BLOCKED": "🚫", "NO_TRADE": "⚪",
            "RISK_REJECTED": "⛔", "EMERGENCY_STOP": "🚨",
        }.get(stage, "🟡")
        log.info(f"[FlowController] {icon} {cycle_log['symbol']} → {stage} | {reason}")
        return cycle_log

    def _save_cycle_log(self, cycle_log: dict) -> None:
        """প্রতিটা cycle-এর সম্পূর্ণ audit trail (doc Step 8 — Complete Trade Log)।"""
        import json
        date_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
        path = os.path.join(TRADE_LOG_DIR, f"{date_tag}.jsonl")
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(cycle_log, default=str) + "\n")
        except Exception as e:
            log.warning(f"[FlowController] Could not write cycle log: {e}")

    # ═══════════════════════════════════════════════════════
    # PRINT SUMMARY
    # ═══════════════════════════════════════════════════════

    def print_cycle_summary(self, result: dict) -> None:
        bar = "═" * 58
        print(f"\n{bar}")
        print("  🤖  HYBRID FLOW CONTROLLER  (Day 49)")
        print(bar)
        print(f"  Symbol     : {result.get('symbol')}")
        print(f"  Stage      : {result.get('stage')}")
        print(f"  Reason     : {result.get('reason')}")
        if result.get("validator_verdict"):
            v = result["validator_verdict"]
            print(f"  Quant/Vision: {v['quant_signal']} / {v['vision_signal']} "
                  f"→ {v['final_signal']} ({v['final_score']}%)")
        if result.get("calibration"):
            c = result["calibration"]
            print(f"  Calibration : raw={c['raw_confidence']}% → "
                  f"calibrated={c['calibrated_confidence']}% ({c['adjustment']:+d})")
        if result.get("decision_out"):
            print(f"  Decision    : {result['decision_out']}")
        if result.get("execution"):
            print(f"  Execution   : {result['execution']}")
        print(bar + "\n")