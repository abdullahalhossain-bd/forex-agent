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


class _LLMKeyManagerAdapter:
    """Adapts core.llm_key_manager.LLMKeyManager (Groq primary, Gemini
    fallback) to the simple `.generate(prompt) -> str` interface this
    module expects. Mirrors the exact provider-fallback pattern already
    used by core/devils_advocate.py, so loss analysis benefits from the
    same multi-key rotation/failover the rest of the stack relies on.
    """

    def __init__(self, timeout_sec: int = 8):
        self.timeout_sec = timeout_sec

    def generate(self, prompt: str) -> str:
        import time
        from core.llm_key_manager import get_llm_key_manager, log_llm_call_failure
        manager = get_llm_key_manager()
        deadline = time.monotonic() + self.timeout_sec
        last_error = None

        for provider in ("groq", "gemini"):
            if time.monotonic() >= deadline:
                break
            try:
                if provider == "groq":
                    client = manager.get_groq_client()
                    if client is None:
                        continue
                    resp = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        max_tokens=400,
                        response_format={"type": "json_object"},
                    )
                    usage = getattr(resp, "usage", None)
                    tokens = ((getattr(usage, "prompt_tokens", 0) or 0)
                              + (getattr(usage, "completion_tokens", 0) or 0)) if usage else 0
                    try:
                        manager.mark_groq_success(tokens_used=tokens, client=client)
                    except Exception:
                        pass
                    return resp.choices[0].message.content
                else:  # gemini
                    if (deadline - time.monotonic()) < 10.0:
                        continue  # Gemini rejects sub-10s deadlines
                    client = manager.get_gemini_client()
                    if client is None:
                        continue
                    resp = client.models.generate_content(
                        model="gemini-flash-lite-latest", contents=prompt
                    )
                    try:
                        manager.mark_gemini_success(client=client)
                    except Exception:
                        pass
                    return resp.text
            except Exception as exc:
                last_error = exc
                log_llm_call_failure(log, provider, "llama-3.1-8b-instant", 0, 1, exc)
                continue

        raise last_error or RuntimeError("MistakeAnalyzer: no LLM provider available")


class AdvancedMistakeAnalyzer:
    """
    LLM এবং ভেক্টর মেমোরির সমন্বয়ে গঠিত ক্লোজড ট্রেড অ্যানালাইসিস লুপ।
    এটি প্রতিটি ট্রেডের গভীরে গিয়ে ভুল এবং সাফল্যের মূল কারণ অনুসন্ধান করে।
    """

    def __init__(self, llm_client=None):
        self.memory = TradeMemory(seed_rules=False)
        # Audit fix (2026-08-09): this was always constructed with
        # llm_client=None at the only real call site (core/trader.py),
        # which silently made `if self.llm and hasattr(self.llm,
        # 'generate')` false on every single loss — the LLM root-cause
        # analysis this class is named/designed for NEVER actually ran;
        # every closed loss got the same generic templated fallback
        # ("Market Variance ... market invalidated the setup") with no
        # log line indicating a real LLM was never consulted. Default to
        # a working adapter (same Groq/Gemini rotation devils_advocate.py
        # uses) unless the caller explicitly passes False/a client to
        # keep it disabled (e.g. in unit tests).
        if llm_client is None:
            try:
                self.llm = _LLMKeyManagerAdapter()
            except Exception as e:
                log.warning(
                    f"[MistakeAnalyzer] Could not construct default LLM "
                    f"adapter — falling back to heuristic analysis only: {e}"
                )
                self.llm = None
        elif llm_client is False:
            self.llm = None
        else:
            self.llm = llm_client

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

        # Audit fix (2026-08-09): log the full decision context for every
        # loss up front, regardless of whether the LLM call below succeeds
        # — this is the trace the user's audit asked for (signal, market
        # conditions, entry, SL/TP, spread, decision factors) and must not
        # depend on LLM availability.
        spread = trade.get("spread", trade_snapshot.get("spread"))
        log.info(
            f"[MistakeAnalyzer] LOSS context for Trade #{trade.get('id')} "
            f"{trade.get('pair')}: signal={trade.get('signal')} "
            f"entry={trade.get('entry')} sl={trade.get('sl')} tp={trade.get('tp')} "
            f"rr=1:{trade.get('rr_ratio')} spread={spread} "
            f"confidence_at_entry={trade.get('confidence')}% pnl={pnl} | "
            f"trend={trade_snapshot.get('trend')} regime={trade_snapshot.get('regime')} "
            f"rsi={trade_snapshot.get('rsi')} pattern={trade_snapshot.get('pattern')}"
        )

        # ২. LLM এর জন্য প্রম্পট রেডি করা (রুট কজ বের করতে)
        prompt = f"""
        You are the Post-Trade Audit Engine of an AI Trading Bot.
        Analyze this LOSS trade and determine the structural mistake or market context that caused it.

        [TRADE DETAILS]
        Pair: {trade.get('pair')}
        Signal: {trade.get('signal')}
        Entry: {trade.get('entry')} | SL: {trade.get('sl')} | TP: {trade.get('tp')}
        Spread at entry: {spread}
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
            analysis = None
            if self.llm and hasattr(self.llm, 'generate'):
                try:
                    response = self.llm.generate(prompt)
                    try:
                        analysis = json.loads(response)
                    except json.JSONDecodeError:
                        log.warning(
                            "WARNING: LLM loss analysis skipped/reduced to fallback "
                            "because the LLM response was not valid JSON\n"
                            "SOURCE: learning/mistake_analyzer.py:_process_loss_trade\n"
                            "IMPACT: using generic heuristic explanation instead of "
                            "model-generated root cause"
                        )
                        analysis = None
                except Exception as llm_exc:
                    log.warning(
                        f"WARNING: LLM loss analysis skipped/reduced to fallback "
                        f"because the LLM call failed ({llm_exc})\n"
                        f"SOURCE: learning/mistake_analyzer.py:_process_loss_trade\n"
                        f"IMPACT: using generic heuristic explanation instead of "
                        f"model-generated root cause"
                    )
                    analysis = None
            else:
                log.warning(
                    "WARNING: LLM loss analysis skipped/reduced to fallback "
                    "because no LLM client is available\n"
                    "SOURCE: learning/mistake_analyzer.py:_process_loss_trade\n"
                    "IMPACT: using generic heuristic explanation instead of "
                    "model-generated root cause"
                )

            if not analysis:
                analysis = {
                    "error_type": "Market Variance",
                    "what_happened": f"Trade executed with {trade.get('confidence')}% confidence but market invalidated the setup.",
                    "lesson": "Maintain system discipline. Review higher timeframe structure next time.",
                    "confidence_adjustment": -2
                }

            # SQLite + memory/mistakes.json এ সেভ করা (full context সহ, যাতে
            # ডাউনস্ট্রিম অ্যানালাইসিস — যেমন devil's advocate আপডেট —
            # শুধু error_type/lesson না, আসল ট্রেড ডেটার উপরও ভিত্তি করতে পারে)
            mistake_data = {
                "trade_id": trade.get("id"),
                "pair": trade.get("pair"),
                "error_type": analysis.get("error_type"),
                "what_happened": analysis.get("what_happened"),
                "lesson": analysis.get("lesson"),
                "confidence_adjustment": analysis.get("confidence_adjustment"),
                "signal": trade.get("signal"),
                "entry": trade.get("entry"),
                "sl": trade.get("sl"),
                "tp": trade.get("tp"),
                "spread": spread,
                "rr_ratio": trade.get("rr_ratio"),
                "confidence": trade.get("confidence"),
                "pnl": pnl,
                "trend": trade_snapshot.get("trend"),
                "regime": trade_snapshot.get("regime"),
                "rsi": trade_snapshot.get("rsi"),
                "pattern": trade_snapshot.get("pattern"),
            }
            self.memory.db.save_mistake(mistake_data)

            # Pattern memory ও ভেক্টরে পুশ
            # BUGFIX (2026-08-25 audit): PatternMemory.add_losing_pattern()'s
            # real signature is (symbol, regime, pattern, pnl_pips) — this
            # call was passing a single dict as `symbol` plus an unexpected
            # `lesson=` kwarg, which raised TypeError on every closed loss.
            # Caught silently by the except below, so pattern_memory.json
            # never actually recorded a single loss. Call it correctly.
            try:
                self.memory.pattern.add_losing_pattern(
                    symbol=trade.get('pair'),
                    regime=trade_snapshot.get('regime'),
                    pattern=trade_snapshot.get('pattern'),
                    pnl_pips=trade.get('pnl_pips', pnl),
                )
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

        # BUGFIX (2026-08-25 audit): same signature mismatch as the loss
        # path above — a single dict was passed where (symbol, regime,
        # pattern, pnl_pips) positional args are required, raising
        # TypeError on every single WIN, silently swallowed below.
        try:
            self.memory.pattern.add_winning_pattern(
                symbol=trade.get('pair'),
                regime=trade_snapshot.get('regime'),
                pattern=trade_snapshot.get('pattern'),
                pnl_pips=trade.get('pnl_pips', pnl),
            )
        except Exception as e:
            log.warning(f"[MistakeAnalyzer] Pattern memory update failed: {e}")

        self._vector_add_lesson(
            f"VALIDATED SETUP: {positive_lesson} Keep replication high when these alpha factors align.",
            pair=trade.get("pair")
        )
        log.info(f"[MistakeAnalyzer] Win reinforcement logged for Trade #{trade.get('id')}")