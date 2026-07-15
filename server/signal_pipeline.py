# server/signal_pipeline.py  —  Day 32-33 | Webhook → Full Decision Pipeline
# ============================================================
# এই module-টা TradingView থেকে আসা raw candle/indicator data নিয়ে
# পুরো existing stack চালায়:
#
#   webhook payload
#       ↓
#   rule signal       [এখনো placeholder — rule_engine.py uploaded হয়নি]
#       ↓
#   AIAnalyst.analyze()           [ai_analyst.py — already built, unchanged]
#       ↓
#   RiskEngine.evaluate()         [risk_engine.py — already built, REAL, wired]
#       ↓
#   DecisionAgent.decide()        [decision_agent.py — already built, unchanged]
#       ↓
#   CircuitBreaker.allow_trade()  [circuit_breaker.py — already built, unchanged]
#       ↓
#   ExecutionRouter.execute()     [execution_router.py — Day 31, unchanged]
#       ↓
#   TraderDB                       [db.py — already built, unchanged]
#
# কোনো existing file পরিবর্তন করা হয়নি — শুধু এখানে সঠিক ক্রমে call
# করা হয়েছে। `_placeholder_rule_signal()` এখনো বাকি — তোমার আসল
# rule_engine.py দিলে সেটা replace হবে।
# ============================================================

from utils.logger import get_logger
from database.db import TraderDB
from ai.ai_analyst import AIAnalyst
from agents.decision_agent import DecisionAgent
from risk.circuit_breaker import CircuitBreaker
from risk.risk_engine import RiskEngine
from execution.execution_router import ExecutionRouter

log = get_logger("signal_pipeline")


class SignalPipeline:
    """
    Singleton — Flask-এর প্রতি request-এ নতুন AIAnalyst/DecisionAgent
    বানালে unnecessary client re-init হবে (Groq/Gemini client সহ)।
    """

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, starting_balance: float = 10000.0):
        self.db = TraderDB()
        self.analyst = AIAnalyst()
        self.decision_agent = DecisionAgent()
        self.circuit_breaker = CircuitBreaker(balance=starting_balance)
        self.router = ExecutionRouter(db=self.db)
        self.starting_balance = starting_balance
        # Symbol-ভিত্তিক RiskEngine cache — প্রতিটা symbol-এর নিজের
        # daily-state file আছে (memory/daily_risk.json একটাই, কিন্তু
        # correlation/open_trades সব symbol শেয়ার করে), তাই একই
        # balance দিয়ে নতুন instance বানালেও state ফাইল থেকেই আসে।
        self._risk_engines: dict[str, RiskEngine] = {}
        log.info("[SignalPipeline] Initialized — all modules wired")

    def _get_risk_engine(self, symbol: str) -> RiskEngine:
        """প্রতি symbol-এর জন্য balance-synced RiskEngine — current account balance ব্যবহার করে।"""
        current_balance = self.db.get_account_stats(
            starting_balance=self.starting_balance
        )["balance"]
        if symbol not in self._risk_engines:
            self._risk_engines[symbol] = RiskEngine(balance=current_balance, symbol=symbol)
        else:
            # balance বদলে থাকতে পারে — sync করো (RiskEngine নিজে balance cache করে)
            self._risk_engines[symbol].balance = current_balance
        return self._risk_engines[symbol]

    # ─────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ─────────────────────────────────────────────

    def process(self, payload: dict) -> dict:
        """
        payload হলো ai_trader_webhook.pine-এর jsonMsg — এতে থাকে:
        symbol, timeframe, close, open, high, low, rsi, ema_fast,
        ema_slow, atr, event

        Pine Script-এ patterns/regime/S-R নেই — শুধু basic indicator।
        তাই এখানে rule engine context-টা payload থেকে build করা হয়
        (production-এ তোমার বিদ্যমান pattern detector/regime classifier/
        S-R module থাকলে candle history fetch করে সেগুলো এখানে যোগ করো —
        এই pipeline শুধু wiring দেখাচ্ছে, ওই module গুলো uploaded না)।
        """
        # Step 0 — Circuit breaker gate (সবার আগে, যাতে অকারণে LLM call না হয়)
        cb_check = self.circuit_breaker.allow_trade()
        if not cb_check["allowed"]:
            log.warning(f"[Pipeline] Blocked by circuit breaker: {cb_check['reason']}")
            return {"action": "BLOCKED", "reason": cb_check["reason"], "mode": cb_check["mode"]}

        symbol = payload.get("symbol", "EURUSD")

        # Step 1 — payload থেকে indicator context বানানো
        ind_ctx = self._build_indicator_context(payload)

        # Step 2 — rule engine signal (এখনো PLACEHOLDER — rule_engine.py
        # uploaded হয়নি। তোমার আসল rule engine দিলে এখানে সেটা বসবে:
        # rule_signal = my_rule_engine.evaluate(ind_ctx))
        rule_signal = self._placeholder_rule_signal(ind_ctx)

        # Step 3 — LLM second opinion (ai_analyst.py — already built, unchanged)
        llm_result = self.analyst.analyze(
            ind_ctx=ind_ctx,
            pat_ctx={"recent_patterns": [], "pattern_signal": "N/A"},
            sr_ctx={"location": "N/A"},
            regime={"regime": "N/A"},
            signal=rule_signal,
            symbol=symbol,
        )
        self.analyst.print_summary(llm_result)

        # Step 4 — Final decision (decision_agent.py — already built, unchanged)
        analysis_out = {
            "final_signal": rule_signal["signal"],
            "signal": rule_signal,
            "llm": llm_result,
            "news": {"trade_allowed": True},   # news filter uploaded না, default allow
        }

        risk_engine = self._get_risk_engine(symbol)
        risk_out = risk_engine.evaluate(
            signal=rule_signal["signal"],
            entry=ind_ctx.get("close"),
            atr=ind_ctx.get("atr") or 0.0,
            regime={"volatility": "NORMAL"},   # regime classifier uploaded না — default
        )
        risk_engine.print_summary(risk_out)

        decision = self.decision_agent.decide(
            market_out={}, analysis_out=analysis_out, risk_out=risk_out
        )
        self.decision_agent.print_summary(decision)

        # Step 5 — Execution (execution_router.py — Day 31, unchanged)
        decision["symbol"] = symbol
        decision["timeframe"] = payload.get("timeframe", "15M")
        trade = self.router.execute(decision)

        # Step 6 — Risk engine-কে নতুন open trade-এর কথা জানাও (daily state ও
        # correlation/open_trades count সঠিক রাখার জন্য — শুধু trade আসলে open হলে)
        if trade and decision["decision"] in ("BUY", "SELL"):
            risk_engine.record_trade_open(symbol)

        # NOTE: trade close হওয়ার event আলাদা জায়গা থেকে আসবে
        # (PaperTrader.update_price() বা MT5 position-close listener)।
        # তখন risk_engine.record_trade_close(symbol, pnl_usd) এবং
        # circuit_breaker.record_result("WIN"/"LOSS", pnl_usd) — দুটোই
        # call করতে হবে trade close handler-এ (নিচে on_trade_closed() দেখো)।

        return {
            "action": decision["decision"],
            "rule_signal": rule_signal["signal"],
            "llm_signal": llm_result.get("signal"),
            "confidence": decision["confidence"],
            "risk_approved": risk_out["approved"],
            "trade": trade,
        }

    # ─────────────────────────────────────────────
    # TRADE CLOSE HOOK  (PaperTrader.update_price() বা MT5 listener থেকে call করো)
    # ─────────────────────────────────────────────

    def on_trade_closed(self, symbol: str, result: str, pnl_usd: float) -> None:
        """
        একটা trade close হওয়ার পরে call করো (SL HIT / TP HIT / TIMEOUT)।
        এটা risk_engine-এর daily state আর circuit_breaker-এর consecutive-loss
        counter দুটোই update করে — দুটো module আলাদা জিনিস ট্র্যাক করে,
        তাই দুটোতেই জানানো জরুরি।
        """
        risk_engine = self._get_risk_engine(symbol)
        risk_engine.record_trade_close(symbol, pnl_usd)
        self.circuit_breaker.record_result(result, pnl_usd)
        log.info(
            f"[Pipeline] Trade closed → {symbol} {result} (${pnl_usd}) "
            f"— RiskEngine + CircuitBreaker updated"
        )

    # ─────────────────────────────────────────────
    # HELPERS  (rule signal এখনো placeholder — rule_engine.py uploaded হয়নি;
    # RiskEngine real — উপরে wired)
    # ─────────────────────────────────────────────

    def _build_indicator_context(self, payload: dict) -> dict:
        return {
            "close":  payload.get("close"),
            "trend":  "UP" if payload.get("ema_fast", 0) > payload.get("ema_slow", 0) else "DOWN",
            "ema9":   payload.get("ema_fast"),
            "sma20":  payload.get("ema_slow"),
            "rsi":    payload.get("rsi"),
            "atr":    payload.get("atr"),
            "macd_signal": "N/A",
            "macd":   "N/A",
            "bb_position": "N/A",
        }

    def _placeholder_rule_signal(self, ind_ctx: dict) -> dict:
        """
        সাধারণ EMA+RSI rule — তোমার আসল rule_engine.py uploaded করলে
        এই method পুরোপুরি বদলে দিও। এটা শুধু pipeline demonstrate
        করার জন্য একটা minimal বাস্তব rule।
        """
        rsi = ind_ctx.get("rsi") or 50
        trend = ind_ctx.get("trend")

        if trend == "UP" and rsi < 70:
            return {"signal": "BUY", "confidence": 55, "entry": ind_ctx.get("close"),
                     "blocked_by": "None", "reasons": ["EMA fast > slow", f"RSI {rsi} not overbought"]}
        if trend == "DOWN" and rsi > 30:
            return {"signal": "SELL", "confidence": 55, "entry": ind_ctx.get("close"),
                     "blocked_by": "None", "reasons": ["EMA fast < slow", f"RSI {rsi} not oversold"]}
        return {"signal": "NO TRADE", "confidence": 0, "entry": None,
                 "blocked_by": "No clear setup", "reasons": ["Conditions not met"]}

    # ─────────────────────────────────────────────