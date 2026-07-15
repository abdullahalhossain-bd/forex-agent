# hybrid/execution_router.py  —  Day 49 | Execution Router ⭐⭐⭐⭐⭐
# ============================================================
# Doc Step 7 (MT5 Execution) + Bonus #3 (Human Override Switch) +
# Bonus #4 (Emergency Stop)।
#
# এই router DecisionAgent-এর final decision নিয়ে আসলে broker-এ
# পাঠানোর আগে শেষ gate — exactly doc-এর thesis অনুযায়ী:
#
#     "Screen automation দিয়ে trade করবে না, screen ব্যবহার করবে
#      intelligence confirmation-এর জন্য। MT5 API = Hands for execution."
#
# তাই এই router execution-এর জন্য সবসময় OrderManager (MT5) বা
# PaperTrader ব্যবহার করে — ChartDrawer (Day 48)/TradingViewAgent
# (Day 46) কখনো trade execute করার জন্য call করা হয় না, শুধু
# vision capture/confirmation-এর জন্যই ব্যবহৃত হয় (যা ইতিমধ্যে
# ChartReader করে)।
#
# Mode shape তোমার existing .env-এর সাথে consistent রাখা হয়েছে:
#     EXECUTION_MODE = "paper" | "mt5_demo"
#     APPROVAL_MODE  = 1 (analysis only) | 2 (supervised) | 3 (autonomous)
#
# Emergency Stop triggers (doc বোনাস #4):
#   - Vision unavailable
#   - MT5 disconnected
#   - Abnormal spread
#   (+ আমার যোগ করা: daily loss limit hit, repeated execution failures)
# ============================================================

import os
from datetime import datetime, timezone
from enum import IntEnum

from utils.logger import get_logger

log = get_logger("execution_router")


class ApprovalMode(IntEnum):
    """তোমার .env-এর APPROVAL_MODE-এর সাথে হুবহু সামঞ্জস্যপূর্ণ।"""
    ANALYSIS_ONLY = 1   # AI শুধু দেখে, কখনো trade করে না
    SUPERVISED    = 2   # AI suggest করে, human approve() call করতে হবে
    AUTONOMOUS    = 3   # human gate ছাড়াই নিজে নিজে trade করে
    SHADOW        = 4   # Day 51 ⭐ — AI পূর্ণ decision নেয় (entry/SL/TP/lot সব calculate
                         # করে) কিন্তু execute করে না; পরে actual market move-এর সাথে
                         # tুলনা করে "নিলে কী হতো" measure করা হয় — ANALYSIS_ONLY থেকে
                         # আলাদা কারণ shadow trade-এর hypothetical outcome track করা হয়,
                         # ANALYSIS_ONLY-তে তা হয় না


class EmergencyStopError(Exception):
    """Emergency stop ট্রিগার হলে raise হয় — caller (FlowController) এটা catch করে
    trading loop pause করবে, exception swallow করবে না।"""
    pass


class ExecutionRouter:
    """
    Usage:
        router = ExecutionRouter(
            order_manager=om,            # MT5 OrderManager (Day 33)
            paper_trader=pt,             # PaperTrader (pre-Day-37, যদি থাকে)
            execution_mode="paper",      # বা "mt5_demo"
            approval_mode=ApprovalMode.AUTONOMOUS,
            approval_callback=None,      # SUPERVISED mode-এ call হবে
        )

        result = router.route(
            decision={"decision": "BUY", "entry":..., "sl":..., "tp":..., "lot":...},
            symbol="EURUSD",
            health_ctx={"vision_available": True, "mt5_connected": True, "spread_pips": 1.2},
        )
    """

    MAX_NORMAL_SPREAD_PIPS = 3.0   # spread_monitor.py-এর DEFAULT-এর সাথে consistent fallback
    MAX_CONSECUTIVE_FAILURES = 3

    def __init__(
        self,
        order_manager=None,
        paper_trader=None,
        execution_mode: str = None,
        approval_mode: ApprovalMode = None,
        approval_callback=None,
        spread_monitor=None,
    ):
        """
        order_manager   : broker/order_manager.py এর OrderManager (mt5_demo mode-এর জন্য)
        paper_trader    : তোমার existing PaperTrader (paper mode-এর জন্য) — না দিলে
                           paper mode শুধু simulate করে log করবে, real fill করবে না
        execution_mode  : "paper" | "mt5_demo" — না দিলে .env-এর EXECUTION_MODE পড়বে
        approval_mode   : ApprovalMode — না দিলে .env-এর APPROVAL_MODE পড়বে
        approval_callback: SUPERVISED mode-এ call হবে — callback(decision)->bool
        spread_monitor  : broker/spread_monitor.py এর SpreadMonitor (থাকলে reuse)
        """
        self.order_manager = order_manager
        self.paper_trader = paper_trader
        self.execution_mode = execution_mode or os.environ.get("EXECUTION_MODE", "paper")
        self.approval_mode = approval_mode or ApprovalMode(
            int(os.environ.get("APPROVAL_MODE", ApprovalMode.AUTONOMOUS))
        )
        self.approval_callback = approval_callback
        self.spread_monitor = spread_monitor

        self._consecutive_failures = 0
        self._emergency_stopped = False
        self._stop_reason = None
        self._pending_approvals: list = []
        self._shadow_trades: list = []   # Day 51 — executed হয়নি কিন্তু hypothetically track হচ্ছে

    # ═══════════════════════════════════════════════════════
    # MAIN ENTRY
    # ═══════════════════════════════════════════════════════

    def route(self, decision: dict, symbol: str, health_ctx: dict = None) -> dict:
        """
        DecisionAgent.decide()-এর output নিয়ে execute/log/block করো।

        health_ctx (doc বোনাস #4-এর জন্য):
            {
                "vision_available": bool,
                "mt5_connected": bool,
                "spread_pips": float,
                "news_active": bool,
            }
        """
        health_ctx = health_ctx or {}

        # ── Step 0: Emergency stop active থাকলে সব route() ব্লক ──
        if self._emergency_stopped:
            return self._blocked_result(f"Emergency stop active: {self._stop_reason}")

        # ── Step 1: Emergency conditions নতুন করে চেক ──
        stop_check = self._check_emergency_conditions(health_ctx)
        if stop_check["should_stop"]:
            self.trigger_emergency_stop(stop_check["reason"])
            return self._blocked_result(stop_check["reason"])

        decision_action = decision.get("decision", "WAIT")

        if decision_action not in ("BUY", "SELL"):
            log.info(f"[ExecutionRouter] No action needed — decision={decision_action}")
            return {"executed": False, "reason": f"decision={decision_action}", "action": "NONE"}

        # ── Step 2: Approval Mode gate (Bonus #3) ──
        gate = self._approval_gate(decision, symbol)
        if not gate["approved"]:
            return gate

        # ── Step 3: Execute ──
        result = self._execute(decision, symbol)

        # ── Step 4: Failure tracking → emergency stop if repeated ──
        if not result.get("executed"):
            self._consecutive_failures += 1
            log.warning(
                f"[ExecutionRouter] Execution failed ({self._consecutive_failures}/"
                f"{self.MAX_CONSECUTIVE_FAILURES}): {result.get('reason')}"
            )
            if self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                self.trigger_emergency_stop(
                    f"{self._consecutive_failures} consecutive execution failures"
                )
        else:
            self._consecutive_failures = 0

        return result

    # ═══════════════════════════════════════════════════════
    # BONUS #3 — HUMAN OVERRIDE SWITCH
    # ═══════════════════════════════════════════════════════

    def _approval_gate(self, decision: dict, symbol: str) -> dict:
        """
        Mode 1 (ANALYSIS_ONLY) → কখনো execute হবে না, শুধু log।
        Mode 2 (SUPERVISED)    → approval_callback() লাগবে, না দিলে pending queue।
        Mode 3 (AUTONOMOUS)    → সরাসরি এগিয়ে যায়।
        """
        if self.approval_mode == ApprovalMode.ANALYSIS_ONLY:
            log.info(
                f"[ExecutionRouter] 👁️ ANALYSIS_ONLY mode — {decision.get('decision')} "
                f"{symbol} logged but NOT executed"
            )
            return {
                "approved": False, "executed": False, "action": "LOGGED_ONLY",
                "reason": "APPROVAL_MODE=1 (analysis only) — execution skipped",
                "decision": decision,
            }

        if self.approval_mode == ApprovalMode.SHADOW:
            shadow_entry = self._record_shadow_trade(decision, symbol)
            log.info(
                f"[ExecutionRouter] 👻 SHADOW mode — {decision.get('decision')} {symbol} "
                f"conf={decision.get('confidence')}% — recorded, NOT executed"
            )
            return {
                "approved": False, "executed": False, "action": "SHADOW_RECORDED",
                "reason": "APPROVAL_MODE=4 (shadow) — decision recorded for later comparison",
                "decision": decision, "shadow_id": shadow_entry["id"],
            }

        if self.approval_mode == ApprovalMode.SUPERVISED:
            approved = False
            if self.approval_callback:
                try:
                    approved = bool(self.approval_callback(decision))
                except Exception as e:
                    log.warning(f"[ExecutionRouter] approval_callback error: {e}")

            if not approved:
                self._pending_approvals.append({"decision": decision, "symbol": symbol})
                log.info(
                    f"[ExecutionRouter] ⏸️  SUPERVISED mode — awaiting human approval: "
                    f"{decision.get('decision')} {symbol}"
                )
                return {
                    "approved": False, "executed": False, "action": "AWAITING_APPROVAL",
                    "reason": "APPROVAL_MODE=2 (supervised) — human approval required",
                    "decision": decision,
                }

            log.info(f"[ExecutionRouter] ✅ Human approved: {decision.get('decision')} {symbol}")

        # AUTONOMOUS, or SUPERVISED-with-approval
        return {"approved": True}

    def get_pending_approvals(self) -> list:
        return list(self._pending_approvals)

    def clear_pending_approval(self, index: int = 0) -> None:
        if 0 <= index < len(self._pending_approvals):
            self._pending_approvals.pop(index)

    # ═══════════════════════════════════════════════════════
    # SHADOW MODE  (Day 51 Bonus #1 — "নিলে কী হতো")
    # ═══════════════════════════════════════════════════════

    def _record_shadow_trade(self, decision: dict, symbol: str) -> dict:
        """Decision-টা পুরোপুরি capture করে রাখো (entry/SL/TP/lot সহ) যাতে
        পরে actual price move দেখে hypothetical outcome resolve করা যায়।"""
        entry = {
            "id": len(self._shadow_trades) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "decision": decision.get("decision"),
            "confidence": decision.get("confidence"),
            "entry": decision.get("entry"),
            "sl": decision.get("sl"),
            "tp": decision.get("tp"),
            "lot": decision.get("lot"),
            "resolved": False,
            "outcome": None,   # পরে resolve_shadow_trade() দিয়ে "WIN"/"LOSS"/"OPEN" সেট হবে
            "pnl_pips": None,
        }
        self._shadow_trades.append(entry)
        return entry

    def resolve_shadow_trade(self, shadow_id: int, current_price: float) -> dict:
        """
        পরবর্তী cycle-এ (বা ব্যাচ-ভিত্তিক) call করে দেখা হয় shadow trade-টা
        আসলে SL/TP hit করেছে কিনা current_price অনুযায়ী — সরাসরি broker
        ছাড়াই hypothetical resolution, কোনো real position এখানে নেই।
        """
        trade = next((t for t in self._shadow_trades if t["id"] == shadow_id), None)
        if not trade or trade["resolved"]:
            return trade or {}

        direction = trade["decision"]
        sl, tp = trade.get("sl"), trade.get("tp")
        entry = trade.get("entry")

        if sl is None or tp is None or entry is None:
            return trade

        hit_tp = (direction == "BUY" and current_price >= tp) or \
                 (direction == "SELL" and current_price <= tp)
        hit_sl = (direction == "BUY" and current_price <= sl) or \
                 (direction == "SELL" and current_price >= sl)

        if hit_tp:
            trade["resolved"] = True
            trade["outcome"] = "WIN"
            trade["pnl_pips"] = round(abs(tp - entry), 5)
        elif hit_sl:
            trade["resolved"] = True
            trade["outcome"] = "LOSS"
            trade["pnl_pips"] = -round(abs(entry - sl), 5)

        if trade["resolved"]:
            log.info(
                f"[ExecutionRouter] 👻 Shadow trade #{shadow_id} resolved → "
                f"{trade['outcome']} ({trade['pnl_pips']})"
            )
        return trade

    def get_shadow_trades(self, resolved_only: bool = False) -> list:
        if resolved_only:
            return [t for t in self._shadow_trades if t["resolved"]]
        return list(self._shadow_trades)

    def get_shadow_stats(self) -> dict:
        """doc-এর 'নিলে কী হতো' compare করার জন্য summary।"""
        resolved = self.get_shadow_trades(resolved_only=True)
        if not resolved:
            return {"total": len(self._shadow_trades), "resolved": 0, "win_rate": None}

        wins = sum(1 for t in resolved if t["outcome"] == "WIN")
        total_pnl = sum(t["pnl_pips"] for t in resolved)
        return {
            "total": len(self._shadow_trades),
            "resolved": len(resolved),
            "wins": wins,
            "losses": len(resolved) - wins,
            "win_rate": round(wins / len(resolved) * 100, 1),
            "total_pnl_pips": round(total_pnl, 5),
        }

    def set_approval_mode(self, mode: ApprovalMode) -> None:
        self.approval_mode = mode
        log.info(f"[ExecutionRouter] Approval mode switched → {mode.name}")

    # ═══════════════════════════════════════════════════════
    # BONUS #4 — EMERGENCY STOP
    # ═══════════════════════════════════════════════════════

    def _check_emergency_conditions(self, health_ctx: dict) -> dict:
        """
        Doc-এর তিনটা trigger + ২টা আমার যোগ করা (failures, daily loss) —
        সব মিলিয়ে check করে। প্রথম matched condition-এই থামে।
        """
        if self.execution_mode == "mt5_demo" and health_ctx.get("mt5_connected") is False:
            return {"should_stop": True, "reason": "MT5 disconnected"}

        if health_ctx.get("vision_available") is False and health_ctx.get("require_vision", False):
            return {"should_stop": True, "reason": "Vision system unavailable (required)"}

        spread = health_ctx.get("spread_pips")
        if spread is not None:
            max_spread = self._max_spread_for(health_ctx)
            if spread > max_spread:
                return {
                    "should_stop": True,
                    "reason": f"Abnormal spread: {spread} pips > {max_spread} pips max",
                }

        if health_ctx.get("daily_loss_limit_hit"):
            return {"should_stop": True, "reason": "Daily loss limit hit"}

        return {"should_stop": False, "reason": None}

    def _max_spread_for(self, health_ctx: dict) -> float:
        if self.spread_monitor:
            # SpreadMonitor.check() নিজেই symbol-aware threshold দেয় — এখানে শুধু
            # generic fallback, আসল check FlowController-এ SpreadMonitor.check()
            # সরাসরি করা উচিত এবং তার ফলাফল health_ctx-এ "spread_ok" হিসেবে পাঠানো উচিত
            pass
        return health_ctx.get("max_spread_pips", self.MAX_NORMAL_SPREAD_PIPS)

    def trigger_emergency_stop(self, reason: str) -> None:
        """
        Doc: "Trading paused"। এটা router-level pause — caller (FlowController/
        main loop) এই flag চেক করে পুরো autonomous loop-ই থামিয়ে দিতে পারে।
        """
        self._emergency_stopped = True
        self._stop_reason = reason
        log.error(f"[ExecutionRouter] 🚨 EMERGENCY STOP TRIGGERED — {reason}")

    def resume_after_emergency(self, confirmed_by: str = "operator") -> None:
        """Human manually resume করলে call করো — silent auto-resume কখনো না।"""
        log.info(f"[ExecutionRouter] ▶️ Resuming after emergency stop (confirmed by {confirmed_by})")
        self._emergency_stopped = False
        self._stop_reason = None
        self._consecutive_failures = 0

    def is_emergency_stopped(self) -> bool:
        return self._emergency_stopped

    # ═══════════════════════════════════════════════════════
    # EXECUTION  (paper vs mt5_demo router)
    # ═══════════════════════════════════════════════════════

    def _execute(self, decision: dict, symbol: str) -> dict:
        if self.execution_mode == "mt5_demo":
            return self._execute_mt5(decision, symbol)
        return self._execute_paper(decision, symbol)

    def _execute_mt5(self, decision: dict, symbol: str) -> dict:
        if not self.order_manager:
            return {
                "executed": False, "action": "PENDING_EXECUTOR",
                "reason": "EXECUTION_MODE=mt5_demo but no OrderManager wired in",
            }

        result = self.order_manager.place_market_order(
            symbol=symbol,
            direction=decision["decision"],
            lot=decision.get("lot", 0.01),
            sl=decision.get("sl"),
            tp=decision.get("tp"),
            comment="ai_trader_day49",
        )

        return {
            "executed": result.get("success", False),
            "action": "MT5_ORDER",
            "broker_result": result,
            "reason": result.get("reason") if not result.get("success") else None,
            "ticket": result.get("ticket"),
        }

    def _execute_paper(self, decision: dict, symbol: str) -> dict:
        if self.paper_trader:
            try:
                # PaperTrader uses open_trade_from_signal(), not open_trade()
                adapted = {
                    "final_action": decision.get("decision"),
                    "symbol": symbol,
                    "entry": decision.get("entry"),
                    "sl": decision.get("sl"),
                    "tp": decision.get("tp"),
                    "lot": decision.get("lot", 0.01),
                    "confidence": decision.get("confidence", 0),
                    "rr": decision.get("rr", 0),
                    "timeframe": decision.get("timeframe", "15m"),
                }
                paper_result = self.paper_trader.open_trade_from_signal(adapted)
                return {
                    "executed": True, "action": "PAPER_ORDER",
                    "paper_result": paper_result,
                }
            except Exception as e:
                log.error(f"[ExecutionRouter] PaperTrader error: {e}")
                return {"executed": False, "action": "PAPER_ORDER", "reason": str(e)}

        # PaperTrader না wired থাকলেও simulate-log করে (visibility-এর জন্য)
        log.info(
            f"[ExecutionRouter] 📝 PAPER (simulated, no PaperTrader wired) — "
            f"{decision['decision']} {symbol} @ {decision.get('entry')} "
            f"SL={decision.get('sl')} TP={decision.get('tp')} lot={decision.get('lot')}"
        )
        return {
            "executed": True, "action": "PAPER_SIMULATED",
            "reason": "No PaperTrader instance wired — logged only",
            "simulated_at": datetime.now(timezone.utc).isoformat(),
        }

    # ═══════════════════════════════════════════════════════
    # UTIL
    # ═══════════════════════════════════════════════════════

    def _blocked_result(self, reason: str) -> dict:
        return {"executed": False, "action": "BLOCKED", "reason": reason}

    def print_status(self) -> None:
        bar = "═" * 50
        print(f"\n{bar}")
        print("  🚦  EXECUTION ROUTER  (Day 49 + Day 51 Shadow)")
        print(bar)
        print(f"  Execution Mode  : {self.execution_mode}")
        print(f"  Approval Mode   : {self.approval_mode.name} ({int(self.approval_mode)})")
        print(f"  Emergency Stop  : {'🚨 ACTIVE — ' + str(self._stop_reason) if self._emergency_stopped else '✅ clear'}")
        print(f"  Pending approvals: {len(self._pending_approvals)}")
        print(f"  Consecutive fails: {self._consecutive_failures}/{self.MAX_CONSECUTIVE_FAILURES}")
        if self._shadow_trades:
            stats = self.get_shadow_stats()
            print(f"  Shadow trades    : {stats}")
        print(bar + "\n")