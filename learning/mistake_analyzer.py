# learning/mistake_analyzer.py
# ============================================================
# Day 19 | Advanced AI Self-Learning Loop & Mistake Analyzer
# Production-hardened: fixed broken vector memory references
# ============================================================

import json
from datetime import datetime, timezone
from memory.trade_memory import TradeMemory
from utils.logger import get_logger

log = get_logger(__name__)


class AdvancedMistakeAnalyzer:
    """
    LLM এবং ভেক্টর মেমোরির সমন্বয়ে গঠিত ক্লোজড ট্রেড অ্যানালাইসিস লুপ।
    এটি প্রতিটি ট্রেডের গভীরে গিয়ে ভুল এবং সাফল্যের মূল কারণ অনুসন্ধান করে।
    """

    def __init__(self, llm_client=None):
        self.memory = TradeMemory(seed_rules=False)
        self.llm = llm_client  # আপনার প্রজেক্টের LLM/Gemini ক্লায়েন্ট এখানে পাস হবে

    def _has_vector_memory(self) -> bool:
        """Check if vector memory (sentence-transformers) is available."""
        return hasattr(self.memory, '_model') and self.memory._model is not None

    def _vector_search(self, query: str, limit: int = 2):
        """Safely search vector memory if available."""
        if not self._has_vector_memory():
            return []
        try:
            return self.memory.find_similar(query, limit=limit)
        except Exception as e:
            log.warning(f"[MistakeAnalyzer] Vector search failed: {e}")
            return []

    def _vector_add_lesson(self, text: str, pair: str = ""):
        """Safely add a lesson to vector memory if available."""
        if not self._has_vector_memory():
            return
        try:
            self.memory.add_vector_lesson(text, pair=pair)
        except Exception as e:
            log.warning(f"[MistakeAnalyzer] Vector add failed: {e}")

    def analyze_closed_trade(self, trade_id: int):
        """ট্রেড ক্লোজ হওয়ার পর সেলফ-লার্নিং লুপ ট্রিগার করার মেইন মেথড।"""
        trade = self.memory.db.get_trade_by_id(trade_id)
        if not trade:
            log.error(f"[MistakeAnalyzer] Trade #{trade_id} not found in database.")
            return

        result = trade.get("result")
        pnl = trade.get("pnl", 0)

        if result == "LOSS":
            log.info(f"[MistakeAnalyzer] Analyzing LOSS for Trade #{trade_id}...")
            self._process_loss_trade(trade, pnl)
        elif result == "WIN" and pnl > 0:
            log.info(f"[MistakeAnalyzer] Analyzing WIN for Trade #{trade_id}...")
            self._process_win_trade(trade, pnl)

    def _process_loss_trade(self, trade: dict, pnl: float):
        """লস ট্রেডের রুট কজ এবং ভেক্টর মেমোরি ম্যাচিং অ্যানালাইসিস।"""
        trade_snapshot = (
            json.loads(trade.get("chart_snapshot", "{}"))
            if isinstance(trade.get("chart_snapshot"), str)
            else trade.get("chart_snapshot", {})
        )

        # ১. ভেক্টর মেমোরি থেকে একই ধরণের অতীতের লস খোঁজা
        similar_past_failures = ""
        query_str = (
            f"{trade.get('pair')} LOSS "
            f"{trade_snapshot.get('trend', 'unknown')} trend "
            f"RSI {trade_snapshot.get('rsi', 50)} "
            f"pattern {trade_snapshot.get('pattern', 'none')}"
        )
        similar_memories = self._vector_search(query_str, limit=2)
        if similar_memories:
            lines = []
            for m in similar_memories:
                if isinstance(m, dict):
                    lines.append(f"- Past Lesson: {m.get('memory', m.get('text', str(m)))}")
                else:
                    lines.append(f"- Past Lesson: {m}")
            similar_past_failures = "\n".join(lines)

        # ২. LLM এর জন্য প্রম্পট রেডি করা (রুট কজ বের করতে)
        prompt = f"""
        You are the Post-Trade Audit Engine of an AI Trading Bot.
        Analyze this LOSS trade and determine the structural mistake or market context that caused it.

        [TRADE DETAILS]
        Pair: {trade.get('pair')}
        Signal: {trade.get('signal')}
        Entry: {trade.get('entry')} | SL: {trade.get('sl')} | TP: {trade.get('tp')}
        Risk-Reward: 1:{trade.get('rr_ratio')}
        Bot Confidence: {trade.get('confidence')}%
        PnL: {pnl}

        [MARKET CONTEXT AT ENTRY]
        Trend: {trade_snapshot.get('trend')}
        Regime: {trade_snapshot.get('regime')}
        RSI: {trade_snapshot.get('rsi')}
        Pattern: {trade_snapshot.get('pattern')}

        [SIMILAR PAST LESSONS FOUND]
        {similar_past_failures if similar_past_failures else "No repetitive pattern found yet."}

        Provide a structured breakdown in JSON format only:
        {{
            "error_type": "Short label of the mistake",
            "what_happened": "Detailed explanation",
            "lesson": "Actionable rule to prevent this",
            "confidence_adjustment": -5
        }}
        """

        # ৩. LLM এক্সিকিউশন এবং মেমোরি আপডেট
        try:
            if self.llm and hasattr(self.llm, 'generate'):
                response = self.llm.generate(prompt)
                try:
                    analysis = json.loads(response)
                except json.JSONDecodeError:
                    log.warning("[MistakeAnalyzer] LLM response not valid JSON, using fallback")
                    analysis = None
            else:
                analysis = None

            if not analysis:
                analysis = {
                    "error_type": "Market Variance",
                    "what_happened": f"Trade executed with {trade.get('confidence')}% confidence but market invalidated the setup.",
                    "lesson": "Maintain system discipline. Review higher timeframe structure next time.",
                    "confidence_adjustment": -2
                }

            # SQLite ও প্যাটার্ন মেমোরিতে সেভ করা
            mistake_data = {
                "trade_id": trade.get("id"),
                "pair": trade.get("pair"),
                "error_type": analysis.get("error_type"),
                "what_happened": analysis.get("what_happened"),
                "lesson": analysis.get("lesson")
            }
            self.memory.db.save_mistake(mistake_data)

            # Pattern memory ও ভেক্টরে পুশ
            try:
                self.memory.pattern.add_losing_pattern({
                    "pair": trade.get('pair'),
                    "signal": trade.get('signal'),
                    "pattern": trade_snapshot.get('pattern'),
                    "regime": trade_snapshot.get('regime'),
                    "rsi": trade_snapshot.get('rsi'),
                    "pnl": pnl
                }, lesson=analysis.get("lesson"))
            except Exception as e:
                log.warning(f"[MistakeAnalyzer] Pattern memory update failed: {e}")

            self._vector_add_lesson(
                f"CRITICAL LESSON for {trade.get('pair')} [{analysis.get('error_type')}]: {analysis.get('lesson')}",
                pair=trade.get("pair")
            )

            log.info(f"[MistakeAnalyzer] Lesson Learned: {analysis.get('lesson')}")

        except Exception as e:
            log.error(f"[MistakeAnalyzer] Failed to run LLM Mistake Audit: {e}")

    def _process_win_trade(self, trade: dict, pnl: float):
        """সফল ট্রেডগুলোর পজিটিভ রিইনফোর্সমেন্ট অ্যানালাইসিস।"""
        trade_snapshot = (
            json.loads(trade.get("chart_snapshot", "{}"))
            if isinstance(trade.get("chart_snapshot"), str)
            else trade.get("chart_snapshot", {})
        )

        positive_lesson = (
            f"Successful {trade.get('signal')} trade on {trade.get('pair')} "
            f"during {trade_snapshot.get('regime')} market with "
            f"{trade_snapshot.get('pattern')} pattern. R:R was 1:{trade.get('rr_ratio')}."
        )

        try:
            self.memory.pattern.add_winning_pattern({
                "pair": trade.get('pair'),
                "signal": trade.get('signal'),
                "pattern": trade_snapshot.get('pattern'),
                "regime": trade_snapshot.get('regime'),
                "rsi": trade_snapshot.get('rsi'),
                "pnl": pnl,
                "rr": trade.get('rr_ratio')
            })
        except Exception as e:
            log.warning(f"[MistakeAnalyzer] Pattern memory update failed: {e}")

        self._vector_add_lesson(
            f"VALIDATED SETUP: {positive_lesson} Keep replication high when these alpha factors align.",
            pair=trade.get("pair")
        )
        log.info(f"[MistakeAnalyzer] Win reinforcement logged for Trade #{trade.get('id')}")
