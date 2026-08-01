# learning/memory_integration.py  —  Day 52 | Memory + Decision Integration ⭐⭐⭐⭐⭐
# ============================================================
# DecisionAgent + MasterAnalyst-এ lesson memory inject করার bridge।
#
# Day 52 upgrade: এখন AI শুধু indicator দেখে না —
#   Current Setup + Past Mistakes + Lessons → Decision
#
# Example:
#   BUY setup detected
#   BUT: previous 15 similar trades: 12 losses
#   Decision: WAIT
# ============================================================

from utils.logger import get_logger
from learning.lesson_memory import LessonMemory
from learning.rule_updater import RuleUpdater
from learning.performance_feedback import PerformanceFeedback
from learning.deep_analyzer import DeepMistakeAnalyzer

log = get_logger("learning.memory_integration")


class MemoryIntegration:
    """
    Trading pipeline-এর সাথে learning system-এর bridge।

    Usage (in trading_engine or analysis_agent):
        memory = MemoryIntegration()

        # Before trading:
        ctx = memory.get_pre_trade_context(
            pattern="Bullish Engulfing",
            regime="RANGING",
            timeframe="M15",
        )
        if ctx["recommendation"] == "AVOID":
            signal = "NO TRADE"

        # After trade closes:
        memory.record_trade_outcome(
            trade_dict=closed_trade,
            outcome="LOSS",
        )
    """

    def __init__(self):
        self.lesson_memory   = LessonMemory()
        self.rule_updater    = RuleUpdater()
        self.feedback        = PerformanceFeedback()
        self.deep_analyzer   = DeepMistakeAnalyzer()

    # ──────────────────────────────────────────────────────────
    # PRE-TRADE MEMORY CHECK (DecisionAgent-এ call করো)
    # ──────────────────────────────────────────────────────────

    def get_pre_trade_context(
        self,
        pattern: str = None,
        regime: str = None,
        timeframe: str = None,
        pair: str = None,
    ) -> dict:
        """
        Trade নেওয়ার আগে memory দেখো।

        Returns:
        {
            "recommendation": "AVOID | CAUTION | PROCEED",
            "confidence_adjustment": -20,
            "memory_summary": "গত 10 বার ranging + engulfing = 7 loss → Avoid",
            "relevant_lessons": [...],
            "active_rule": {...},
            "should_override": True/False,
        }
        """
        # 1. Lesson memory recall
        recall = self.lesson_memory.recall(pattern=pattern, condition=regime)

        # 2. Active rule check
        rule_ctx = self.rule_updater.get_confidence_adjustment(
            pattern=pattern or "unknown",
            condition=regime or "unknown",
        )

        # 3. Performance feedback
        perf_ctx = self.feedback.get_master_context()

        # 4. Determine recommendation
        recommendation = recall.get("recommendation", "PROCEED")
        confidence_adj = rule_ctx.get("adjustment", 0)

        # Memory override: যদি loss rate খুব বেশি
        loss_rate = recall.get("loss_rate_pct", 0)
        should_override = loss_rate >= 70 and recall.get("total", 0) >= 5

        # Worst regime check
        if perf_ctx.get("avoid_regime") == regime:
            recommendation = "AVOID"
            confidence_adj = min(confidence_adj, -20)

        # Build context
        lessons_text = [l.get("new_rule", l.get("lesson", "")) for l in recall.get("lessons", [])[-3:]]

        ctx = {
            "recommendation":      recommendation,
            "confidence_adjustment": confidence_adj,
            "memory_summary":      recall.get("summary", "No past experience"),
            "relevant_lessons":    lessons_text,
            "active_rule":         rule_ctx if rule_ctx.get("has_rule") else None,
            "should_override":     should_override,
            "loss_rate_pct":       loss_rate,
            "total_similar_trades": recall.get("total", 0),
            "best_timeframe":      perf_ctx.get("best_timeframe"),
            "worst_regime":        perf_ctx.get("worst_regime"),
        }

        if should_override:
            log.warning(
                f"[MemoryIntegration] ⚠️ MEMORY OVERRIDE — "
                f"{pattern} in {regime}: {loss_rate}% loss rate over {recall.get('total')} trades"
            )
        else:
            log.info(
                f"[MemoryIntegration] Memory check — {pattern}/{regime}: "
                f"recommendation={recommendation}, adj={confidence_adj}%"
            )

        return ctx

    # ──────────────────────────────────────────────────────────
    # POST-TRADE RECORDING (close হওয়ার পর call করো)
    # ──────────────────────────────────────────────────────────

    def record_trade_outcome(self, trade_dict: dict, outcome: str) -> dict:
        """
        Trade close হওয়ার পর সব memory systems update করো।

        trade_dict: DecisionAgent-এর output + market context
        outcome:    "WIN" | "LOSS" | "BE"
        """
        pnl       = trade_dict.get("pnl", 0)
        pattern   = trade_dict.get("pattern", trade_dict.get("patterns", [None])[0] if trade_dict.get("patterns") else None)
        regime    = trade_dict.get("regime")
        timeframe = trade_dict.get("timeframe")
        pair      = trade_dict.get("symbol", trade_dict.get("pair"))

        # 1. Performance feedback record
        self.feedback.record_trade(
            outcome=outcome,
            pnl=pnl,
            pattern=pattern,
            regime=regime,
            timeframe=timeframe,
            pair=pair,
            rr=trade_dict.get("rr"),
            confidence=trade_dict.get("confidence"),
        )

        result = {"recorded": True}

        # 2. LOSS হলে deep analysis চালাও
        if outcome == "LOSS":
            log.info(f"[MemoryIntegration] LOSS detected — triggering deep analysis")

            trade_context = {
                "pair":                 pair,
                "timeframe":            timeframe,
                "entry":                trade_dict.get("entry"),
                "exit":                 trade_dict.get("exit", trade_dict.get("sl")),
                "pattern":              pattern,
                "rsi":                  trade_dict.get("rsi"),
                "macd":                 trade_dict.get("macd_cross"),
                "market_regime":        regime,
                "news":                 "none",
                "spread":               trade_dict.get("spread", 1.5),
                "atr":                  trade_dict.get("atr", 0.0035),
                "sl_pips":              trade_dict.get("sl_pips", 20),
                "h4_trend":             trade_dict.get("h4_trend", "neutral"),
                "confidence_at_entry":  trade_dict.get("confidence", 70),
                "pnl":                  pnl,
            }

            analysis = self.deep_analyzer.analyze_loss(trade_context)

            # Lesson memory-তেও add করো
            if analysis.get("lesson"):
                self.lesson_memory.add_lesson(
                    pattern=pattern or "unknown",
                    market_condition=regime or "UNKNOWN",
                    mistake=analysis.get("what_happened", ""),
                    new_rule=analysis.get("lesson"),
                    pair=pair,
                    timeframe=timeframe,
                    pnl=pnl,
                    confidence_at_entry=trade_dict.get("confidence"),
                    source="deep_analyzer_day52",
                )

            result["deep_analysis"] = analysis

        return result

    # ──────────────────────────────────────────────────────────
    # MASTER ANALYST MEMORY CONTEXT
    # ──────────────────────────────────────────────────────────

    def get_memory_for_master_analyst(self, pattern: str = None, regime: str = None) -> dict:
        """
        MasterAnalyst._build_context()-এ inject করার জন্য।
        Day 52 upgrade: memory context এখন master analyst পায়।
        """
        pre_trade = self.get_pre_trade_context(pattern=pattern, regime=regime)
        perf      = self.feedback.get_master_context()

        return {
            "memory_recommendation":   pre_trade.get("recommendation"),
            "memory_confidence_adj":   pre_trade.get("confidence_adjustment", 0),
            "memory_summary":          pre_trade.get("memory_summary"),
            "active_rule":             pre_trade.get("active_rule"),
            "should_memory_override":  pre_trade.get("should_override", False),
            "key_lessons":             pre_trade.get("relevant_lessons", []),
            "overall_win_rate":        perf.get("overall_win_rate", 50),
            "total_trades":            perf.get("total_trades", 0),
            "best_pattern":            perf.get("best_pattern"),
            "worst_pattern":           perf.get("worst_pattern"),
            "avoid_regime":            perf.get("avoid_regime"),
        }