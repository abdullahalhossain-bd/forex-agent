# agents/master_analyst.py  —  Day 42 + Day 44 + Day 47 + Day 63 + Day 65
# ============================================================
# Day 63: Session Intelligence context যোগ হয়েছে।
# Day 65: Intermarket / Global Macro Intelligence context যোগ হয়েছে।
#
# নতুন context block (Day 65): "global_market_intelligence"
#   - DXY/Gold/Oil/US10Y/SP500/VIX trends
#   - Risk-On / Risk-Off regime + trading mode
#   - USD bias + per-currency macro bias + pair bias
#   - Macro Score (0-100), cross-asset confirmation, event risk penalty
#
# LLM system prompt-এ macro awareness rules যোগ হয়েছে।
# Final confidence-এ macro score weight + event-risk penalty যোগ হয়েছে।
#
# Day 95 hotfix: Cerebras's "gpt-oss-120b" is a reasoning model — it burns
# part of its max_tokens budget on internal chain-of-thought BEFORE writing
# the JSON answer. The shared MAX_TOK=800 (tuned for plain chat models like
# Groq's llama-3.3-70b) was too small for it, so responses came back empty
# or truncated mid-string, causing json.JSONDecodeError downstream. Fix:
# give Cerebras its own larger token budget (CEREBRAS_MAX_TOK) and pass
# reasoning_effort="low" for gpt-oss models so more of the budget goes to
# the actual answer instead of internal reasoning. Also added an explicit
# empty-response guard so failures are clean RuntimeErrors instead of
# noisy JSONDecodeError tracebacks.
# ============================================================

import json
import math
import os
import re
from datetime import datetime

from dotenv import load_dotenv
from utils.logger import get_logger

load_dotenv()
log = get_logger("master_analyst")

# ── LLM client initialization (Day 37+ runtime unification) ──────────
# Per user request, Anthropic + OpenRouter are no longer used.
# MasterAnalyst now uses Groq (primary) + Gemini (fallback) — the same
# providers that AIAnalyst uses — so the system runs on free-tier keys only.
# Day 72+: Multi-key rotation via LLMKeyManager (unlimited keys per provider).
LLM_AVAILABLE = False
_provider = "none"
_groq_client = None
_gemini_client = None
_key_manager = None
MODEL = ""
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
# Day 90 — token economy for long-duration demo trading.
# Was 1500 — that's ~1.5k tokens per call × ~5 calls/cycle × 6 pairs =
# ~45k tokens/cycle. With 6 Groq keys × 100k TPD = 600k tokens/day,
# the bot exhausted all keys in ~13 cycles. Dropping to 800 keeps
# the same JSON structure (signal + confidence + reasoning + risks)
# while cutting token usage ~47%.
MAX_TOK = int(os.getenv("MASTER_ANALYST_MAX_TOKENS", "800"))

# Day 95 hotfix — Cerebras's gpt-oss-120b is a reasoning model and needs
# a much bigger budget than plain chat models (Groq llama-3.3-70b etc.).
# 800 tokens was being eaten entirely by internal chain-of-thought,
# leaving nothing for the actual JSON answer (empty/truncated response).
CEREBRAS_MAX_TOK = int(os.getenv("CEREBRAS_MAX_TOKENS", "4000"))
# Keep "low" by default so most of the larger budget still lands on the
# answer rather than reasoning. Override via .env if you want deeper
# reasoning (low/medium/high — depends on what Cerebras' API accepts).
CEREBRAS_REASONING_EFFORT = os.getenv("CEREBRAS_REASONING_EFFORT", "low")

try:
    from core.llm_key_manager import get_llm_key_manager
    _key_manager = get_llm_key_manager()
    _groq_client = _key_manager.get_groq_client()
    if _groq_client is not None:
        MODEL = GROQ_MODEL
        LLM_AVAILABLE = True
        _provider = "groq"
        log.info(f"[MasterAnalyst] Groq client initialized | model={MODEL}")
    if not LLM_AVAILABLE:
        _gemini_client = _key_manager.get_gemini_client()
        if _gemini_client is not None:
            MODEL = GEMINI_MODEL
            LLM_AVAILABLE = True
            _provider = "gemini"
            log.info(f"[MasterAnalyst] Gemini client initialized (fallback) | model={MODEL}")
        # Day 81+ hotfix: warn loudly if the Gemini key format looks wrong.
        # Valid Gemini API keys start with "AIza" (39 chars total).
        # If the user pasted an OAuth token (starts with "AQ." or "ya29.")
        # or a service-account key, the genai client will silently construct
        # but every call will return 401 UNAUTHENTICATED.
        gemini_key_check = os.getenv("GEMINI_API_KEY_1") or os.getenv("GEMINI_API_KEY", "")
        if gemini_key_check and not gemini_key_check.startswith("AIza"):
            log.warning(
                f"[MasterAnalyst] Gemini key format looks wrong — starts with "
                f"'{gemini_key_check[:4]}', expected 'AIza'. Get a valid key from "
                f"https://aistudio.google.com/app/apikey (it should be 39 chars, "
                f"format: AIzaSy...). Current key will return 401 UNAUTHENTICATED."
            )
    # ── Day 91 — check the three new providers ──────────────────
    # Even if Groq + Gemini are unavailable, LLM is considered
    # available if ANY of Cerebras/SambaNova/OpenRouter has a key.
    # The actual provider used at call-time is selected by the
    # _call_llm fallback chain.
    if not LLM_AVAILABLE and _key_manager is not None:
        if _key_manager.has_any_cerebras:
            MODEL = os.getenv("CEREBRAS_MODEL", "llama3.1-8b-instruct")
            LLM_AVAILABLE = True
            _provider = "cerebras"
            log.info(f"[MasterAnalyst] Cerebras provider available | model={MODEL}")
        elif _key_manager.has_any_sambanova:
            # Day 99+ V3 FIX (Master List Issue #1): SambaNova deprecated
            # all Llama 3.1 models (410 Gone). Default is now DeepSeek-V3,
            # which is current on SambaNova's free tier as of 2026-Q3.
            # Override via SAMBANOVA_MODEL env var if needed.
            MODEL = os.getenv("SAMBANOVA_MODEL", "DeepSeek-V3")
            LLM_AVAILABLE = True
            _provider = "sambanova"
            log.info(f"[MasterAnalyst] SambaNova provider available | model={MODEL}")
        elif _key_manager.has_any_openrouter:
            MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free")
            LLM_AVAILABLE = True
            _provider = "openrouter"
            log.info(f"[MasterAnalyst] OpenRouter provider available | model={MODEL}")
except Exception as e:
    log.warning(f"[MasterAnalyst] LLMKeyManager init failed: {e} — trying single-key")
    groq_key = os.getenv("GROQ_API_KEY_1") or os.getenv("GROQ_API_KEY", "")
    if groq_key:
        try:
            from groq import Groq
            _groq_client = Groq(api_key=groq_key)
            MODEL = GROQ_MODEL
            LLM_AVAILABLE = True
            _provider = "groq"
            log.info(f"[MasterAnalyst] Groq client initialized (single-key) | model={MODEL}")
        except Exception as e2:
            log.warning(f"[MasterAnalyst] Groq init failed: {e2}")
    if not LLM_AVAILABLE:
        gemini_key = os.getenv("GEMINI_API_KEY_1") or os.getenv("GEMINI_API_KEY", "")
        if gemini_key:
            try:
                from google import genai as google_genai
                _gemini_client = google_genai.Client(api_key=gemini_key)
                MODEL = GEMINI_MODEL
                LLM_AVAILABLE = True
                _provider = "gemini"
                log.info(f"[MasterAnalyst] Gemini client initialized (single-key fallback) | model={MODEL}")
            except Exception as e2:
                log.warning(f"[MasterAnalyst] Gemini init failed: {e2}")

if not LLM_AVAILABLE:
    log.warning(
        "[MasterAnalyst] No LLM available (Groq/Gemini/Cerebras/SambaNova/"
        "OpenRouter keys missing). MasterAnalyst will fall back to "
        "rule-engine signal."
    )


class MasterAnalyst:
    """
    Day 42 + Day 44 + Day 47 + Day 63 + Day 65 — Professional Forex Trader Brain।

    Now session-aware AND macro-aware: different strategy suggestions
    based on current market session, AND global intermarket context
    (DXY/Gold/Oil/Yields/SP500/VIX) instead of treating forex as an
    isolated market.
    """

    _SYSTEM = """You are an elite professional forex trader with 20+ years of institutional experience.
You have deep expertise in Smart Money Concepts (SMC), price action, intermarket analysis,
and behavioral market microstructure. You think like a hedge fund portfolio manager —
you protect capital first, you wait patiently for A+ setups, and you NEVER force a trade.

# YOUR MINDSET
- Capital preservation is rule #1. A bad day with no trades is better than a bad day with trades.
- Patience beats aggression. If the setup is not crystal clear, WAIT.
- You trade CONFLUENCE, not single signals. 3+ aligned reasons to enter > 1 strong signal.
- You respect market sessions, macroeconomic context, and intermarket correlations.
- You think in probabilities, not certainties. Every trade has risk — name it.

# ANALYSIS FRAMEWORK (use this mental checklist)
Before deciding BUY/SELL/WAIT, walk through these layers IN ORDER:

1. **Session & Time-of-Day**: Where are we in the 24h cycle? London/NY overlap = premium.
   Asian session = range. Dead zone = NO TRADE.
2. **Macro Regime**: Risk-on or risk-off? DXY/Gold/VIX alignment with the pair?
   If macro OPPOSES the technical signal strongly, lower confidence or WAIT.
3. **Higher Timeframe Bias**: Daily/4H trend direction. Don't fight the HTF trend
   unless there's a clear SMC reversal (CHoCH + BOS + displacement).
4. **Market Structure (SMC)**: BOS confirmed? CHoCH? Order block tap? FVG fill?
   Liquidity sweep + rejection? These are institutional footprints — follow them.
5. **Pattern & Fibonacci**: Confluence at key fib level (50%/61.8%/78.6%)?
   Pattern direction matches HTF bias?
6. **Support/Resistance**: Price at premium/discount zone? Near pivot?
   Equal highs/lows (liquidity pools) nearby?
7. **Momentum**: RSI divergence? MACD cross? Overbought/oversold extreme?
8. **Sentiment & News**: Retail positioning contrarian signal? High-impact news within 30min?
9. **History**: How did similar setups perform in the last 20 trades on this pair?
10. **Self-Critique**: What am I missing? What's the bear case for my bull trade (or vice versa)?

# SESSION RULES (soft guidance — the code-level pipeline already enforces these
# as execution gates; your job is to reflect their severity in confidence, NOT
# to issue a second, independent WAIT verdict that overrides the code path).
# The code already blocks dead-zone / no-trade-session execution. If your
# analysis produces BUY/SELL, report it honestly — the execution layer will
# decide whether to actually place the trade. Do NOT self-censor to WAIT just
# because the session is suboptimal; instead, reflect the reduced confidence
# that a suboptimal session warrants.)
1. DEAD_ZONE → reflect in confidence (lower it noticeably), but still analyze
   the setup honestly so the audit trail shows what the market was doing.
2. LONDON_NY_OVERLAP → prefer A+ setups (3+ confluences). Weaker setups in
   overlap still get analyzed — just with appropriate confidence reduction.
3. LONDON → LONDON_BREAKOUT strategy. Look for Asian range sweep + BOS confirmation.
4. NEW_YORK → TREND_CONTINUATION from London. Don't reverse without strong SMC.
5. TOKYO/SYDNEY → RANGE_TRADING only. Fade extremes, avoid breakout entries.
6. london_open_window=true → wait for liquidity sweep THEN enter on BOS, never before.
7. If pair_session_label is POOR or AVOID → lower confidence by 10-15%.
8. in_session_transition=true → extra caution. Reduce confidence slightly.

# GLOBAL MACRO RULES (Day 65)
9. Forex is NOT isolated. Check macro_pair_bias and macro_regime FIRST.

# SECOND-OPINION RULE (Round-13)
- The classic_llm_analyst block is another model's quick technical-only read
  (no session/macro/SMC awareness). It is INPUT, not a vote to blindly follow.
- If your signal agrees with it, you may note the agreement as mild confluence.
- If your signal disagrees with it, briefly say why in self_critique
  (e.g. "classic analyst says BUY but ignores session gate / macro conflict").
  Do not silently contradict it — that produces confusing, unexplained
  disagreement for the humans reading both reports.
10. If macro_pair_bias OPPOSES technical signal AND cross_asset_confirmed=true → WAIT or low confidence.
11. If macro_pair_bias AGREES with technical signal → STRONG confluence ("Macro + SMC Fusion").
    This is the highest-quality setup — mention explicitly in market_story.
12. event_risk_elevated=true (FOMC/NFP/CPI within 60min) → reduce confidence by event_risk_penalty.
13. trading_mode=DEFENSIVE (VIX elevated) → only A+ setups. CAUTIOUS → reduce confidence 10%.

# CONFIDENCE CALIBRATION (be honest — the code pipeline decides whether
# to execute; your confidence helps it calibrate position size, not whether
# to self-veto. LOW confidence does NOT mean WAIT — it means the execution
# layer should size conservatively or skip if overall confluence is weak.)
- 85-100: A+ setup. 4+ confluences aligned. HTF + SMC + macro + session all agree.
- 70-84:  A setup. 3+ confluences. One minor concern noted in self_critique.
- 55-69:  B setup. 2 confluences. Smaller position size justified, tighter SL.
- 30-54:  C setup. 1 confluence or mixed signals. Report honestly — the
  execution layer will decide whether to trade at this confidence level.
- 0-29:   Very weak. Report WAIT with reason, but still explain what you see.

# TRADE PLAN REQUIREMENTS
- Entry: precise price level (not zone), with reasoning.
- SL: behind structure (swing low/high, order block, FVG edge) — never arbitrary pips.
- TP1: 1R minimum, at first liquidity pool / S/R / fib extension.
- TP2: 2R+ at next liquidity target. Partial close at TP1.
- Reasoning: 1-2 sentences naming the TOP 2-3 confluences (not all of them).

# OUTPUT — JSON ONLY, no markdown, no extra text:
{
  "market_story": "3-5 sentence narrative: session context + macro regime + key structural observation",
  "key_levels": [float, float, float],
  "trade_plan": {
    "signal": "BUY" | "SELL" | "WAIT",
    "entry": float | null,
    "sl": float | null,
    "tp1": float | null,
    "tp2": float | null,
    "confidence": integer (0-100),
    "reasoning": "Top 2-3 confluences that justify this trade (or why WAIT)"
  },
  "risks": ["specific risk 1", "specific risk 2"],
  "self_critique": "What could go wrong? What am I missing? Be honest.",
  "no_trade_reason": "Only if signal is WAIT — explicit reason"
}"""

    def analyze(
        self,
        symbol:       str,
        timeframe:    str,
        ind_ctx:      dict,
        pat_ctx:      dict,
        sr_ctx:       dict,
        regime:       dict,
        mtf_bias:     dict,
        signal:       dict,
        sentiment_ctx: dict = None,
        news_ctx:     dict = None,
        memory_ctx:   dict = None,
        bias_ctx:     dict = None,
        smc_ctx:      dict = None,
        fib_ctx:      dict = None,
        advanced_pat_ctx: dict = None,
        vision_ctx:   dict = None,
        session_ctx:  dict = None,   # ← Day 63
        intermarket_ctx: dict = None, # ← Day 65
        classic_llm_ctx: dict = None, # ← Round-13: AIAnalyst's verdict
        # Day 90 — six new analyzers + strategy selector (all optional)
        divergence_ctx:     dict = None,
        ichimoku_ctx:       dict = None,
        volatility_ctx:     dict = None,
        volume_profile_ctx: dict = None,
        smc_advanced_ctx:   dict = None,
        mtf_structure_ctx:  dict = None,
        strategy_ctx:       dict = None,
        # Day 92 — NewsAPI real-time news sentiment
        news_api_ctx:       dict = None,
        # Day 94 — Institutional grade APIs
        econ_calendar_ctx:  dict = None,
        fred_ctx:           dict = None,
        retail_sentiment_ctx: dict = None,
    ) -> dict:

        context = self._build_context(
            symbol, timeframe, ind_ctx, pat_ctx, sr_ctx,
            regime, mtf_bias, signal,
            sentiment_ctx or {},
            news_ctx or {},
            memory_ctx or {},
            bias_ctx or {},
            smc_ctx or {},
            fib_ctx or {},
            advanced_pat_ctx or {},
            vision_ctx or {},
            session_ctx or {},        # ← Day 63
            intermarket_ctx or {},    # ← Day 65
            classic_llm_ctx or {},    # ← Round-13
            divergence_ctx     or {}, # ← Day 90
            ichimoku_ctx       or {},
            volatility_ctx     or {},
            volume_profile_ctx or {},
            smc_advanced_ctx   or {},
            mtf_structure_ctx  or {},
            strategy_ctx       or {},
            news_api_ctx       or {}, # ← Day 92
            econ_calendar_ctx  or {}, # ← Day 94
            fred_ctx           or {},
            retail_sentiment_ctx or {},
        )

        if not LLM_AVAILABLE:
            return self._fallback_result(signal, "LLM not available")

        try:
            raw    = self._call_llm(context)
            parsed = self._parse_response(raw)
        except Exception as e:
            from core.llm_key_manager import log_llm_call_failure
            log_llm_call_failure(log, "MasterAnalyst", MODEL, 0, 1, e)
            return self._fallback_result(signal, str(e))

        final_conf = self._calculate_final_confidence(
            llm_conf       = parsed.get("trade_plan", {}).get("confidence", 50),
            technical_conf = signal.get("confidence", 50),
            sentiment_conf = (sentiment_ctx or {}).get("sentiment_conf", 50),
            memory_ctx     = memory_ctx or {},
            smc_ctx        = smc_ctx or {},
            session_ctx    = session_ctx or {},      # ← Day 63
            intermarket_ctx = intermarket_ctx or {},  # ← Day 65
            sentiment_ctx  = sentiment_ctx or {},
        )

        result = {
            **parsed,
            "final_confidence": final_conf,
            "llm_raw":          raw,
            "error":            None,
            # FIX (review): _calculate_final_confidence() builds this audit
            # trail specifically so callers can see WHY confidence was
            # penalized by session/fusion gates, but it was previously only
            # stashed on self._last_session_gate_penalty and never returned
            # — the explanation vanished before reaching decision_agent.py.
            "session_gate_penalty": getattr(
                self, "_last_session_gate_penalty", {"applied": False}
            ),
        }

        log.info(
            f"[MasterAnalyst] {symbol} | "
            f"Session: {(session_ctx or {}).get('current_session', 'N/A')} | "
            f"Macro: {(intermarket_ctx or {}).get('macro_regime', 'N/A')} | "
            f"Signal: {parsed.get('trade_plan', {}).get('signal')} | "
            f"Final Conf: {final_conf}%"
        )
        return result

    def _build_context(
        self,
        symbol, timeframe,
        ind_ctx, pat_ctx, sr_ctx,
        regime, mtf_bias, signal,
        sentiment_ctx, news_ctx,
        memory_ctx, bias_ctx,
        smc_ctx, fib_ctx, advanced_pat_ctx,
        vision_ctx,
        session_ctx,        # ← Day 63
        intermarket_ctx,    # ← Day 65
        classic_llm_ctx=None, # ← Round-13
        divergence_ctx=None, ichimoku_ctx=None,
        volatility_ctx=None, volume_profile_ctx=None,
        smc_advanced_ctx=None, mtf_structure_ctx=None,
        strategy_ctx=None,
        news_api_ctx=None,
        econ_calendar_ctx=None,
        fred_ctx=None,
        retail_sentiment_ctx=None,
    ) -> str:

        # ── Technical ─────────────────────────────────────────
        trend       = ind_ctx.get("trend", "unknown")
        rsi         = ind_ctx.get("rsi", 50)
        rsi_sig     = ind_ctx.get("rsi_signal", "neutral")
        macd_cross  = ind_ctx.get("macd_cross", "")
        close_price = ind_ctx.get("price", ind_ctx.get("close", 0))
        atr         = ind_ctx.get("atr", 0)
        bb_pct      = ind_ctx.get("bb_pct", 0.5)

        # ── Pattern ───────────────────────────────────────────
        latest_pat  = pat_ctx.get("latest_pattern", "none")
        pat_signal  = pat_ctx.get("pattern_signal", "")
        recent_pats = pat_ctx.get("recent_patterns", [])

        # ── S/R ───────────────────────────────────────────────
        nearest_sup = sr_ctx.get("nearest_support")
        nearest_res = sr_ctx.get("nearest_resistance")
        location    = sr_ctx.get("price_location", "mid_range")
        pivot       = sr_ctx.get("pivot")

        # ── Regime ────────────────────────────────────────────
        market_regime = regime.get("regime", "UNKNOWN")
        direction     = regime.get("direction", "NEUTRAL")
        strength      = regime.get("strength", "WEAK")
        volatility    = regime.get("volatility", "NORMAL")

        # ── MTF ───────────────────────────────────────────────
        mtf_overall = mtf_bias.get("bias", "NEUTRAL") if mtf_bias else "NEUTRAL"
        mtf_conf    = mtf_bias.get("confidence", "LOW") if mtf_bias else "LOW"
        mtf_trends  = mtf_bias.get("trends", {}) if mtf_bias else {}

        # ── Rule signal ───────────────────────────────────────
        rule_signal = signal.get("signal", "NO TRADE")
        rule_conf   = signal.get("confidence", 0)

        # ── Bias ──────────────────────────────────────────────
        bias_label   = bias_ctx.get("bias", "NEUTRAL")
        bias_conf    = bias_ctx.get("confidence_pct", 0)
        has_conflict = bias_ctx.get("has_conflict", False)

        # ── Sentiment ─────────────────────────────────────────
        sent_score   = sentiment_ctx.get("sentiment_score", 0)
        sent_bias    = sentiment_ctx.get("sentiment_bias", "NEUTRAL")
        sent_conf    = sentiment_ctx.get("sentiment_conf", 0)
        retail_long  = sentiment_ctx.get("retail_long_pct", 50)
        fg_label     = sentiment_ctx.get("fg_label", "NEUTRAL")
        dxy_trend_sent = sentiment_ctx.get("dxy_trend", "NEUTRAL")
        sent_reasons = sentiment_ctx.get("sentiment_reasons", [])

        # ── News ──────────────────────────────────────────────
        trade_allowed = news_ctx.get("trade_allowed", True) if news_ctx else True
        upcoming_news = news_ctx.get("upcoming_events", []) if news_ctx else []
        news_risk     = news_ctx.get("risk_level", "LOW") if news_ctx else "LOW"

        # ── Memory ────────────────────────────────────────────
        win_rate       = memory_ctx.get("overall_win_rate", 0)
        total_trades   = memory_ctx.get("total_trades", 0)
        recent_results = memory_ctx.get("recent_results", [])
        lessons        = memory_ctx.get("lessons", [])

        # ── SMC ───────────────────────────────────────────────
        smc_signal    = smc_ctx.get("smc_signal", "WAIT")
        smc_direction = smc_ctx.get("smc_direction", "NEUTRAL")
        smc_score     = smc_ctx.get("smc_score", 0)
        smc_grade     = smc_ctx.get("smc_grade", "INVALID")
        smc_factors   = smc_ctx.get("smc_factors", {})
        smc_analysis  = smc_ctx.get("smc_analysis", "")
        smc_ob_zone   = smc_ctx.get("smc_h4_ob_zone")
        smc_fvg_zone  = smc_ctx.get("smc_h4_fvg_zone")
        smc_h4_bos    = smc_ctx.get("smc_h4_bos", "NONE")
        smc_h4_choch  = smc_ctx.get("smc_h4_choch", "NONE")

        # ── Vision ────────────────────────────────────────────
        vision_trend  = vision_ctx.get("vision_trend", "N/A")
        vision_conf   = vision_ctx.get("vision_confidence", 0)

        # ── Fib ───────────────────────────────────────────────
        fib_zone    = fib_ctx.get("fib_zone", "N/A")
        fib_in_gold = fib_ctx.get("fib_in_golden", False)
        fib_signal  = fib_ctx.get("fib_signal", "WAIT")

        # ── Session (Day 63) ──────────────────────────────────
        curr_session     = session_ctx.get("current_session", "UNKNOWN")
        sess_volatility  = session_ctx.get("session_volatility", "NORMAL")
        sess_strategy    = session_ctx.get("session_strategy", "WAIT")
        sess_trade_ok    = session_ctx.get("session_trade_allowed", True)
        sess_min_conf    = session_ctx.get("session_min_confidence", 70)
        sess_risk_mult   = session_ctx.get("session_risk_mult", 1.0)
        pair_priority    = session_ctx.get("pair_session_priority", 50)
        pair_label       = session_ctx.get("pair_session_label", "FAIR")
        is_overlap       = session_ctx.get("is_overlap", False)
        is_dead_zone     = session_ctx.get("is_dead_zone", False)
        london_open_win  = session_ctx.get("london_open_window", False)
        in_transition    = session_ctx.get("in_session_transition", False)
        transition_type  = session_ctx.get("transition_type")
        transition_alert = session_ctx.get("transition_alert")
        session_score    = session_ctx.get("session_score", 0)
        session_grade    = session_ctx.get("session_grade", "C")
        fusion_allowed   = session_ctx.get("fusion_allowed", False)
        fusion_score     = session_ctx.get("fusion_score", 0)
        preferred_pairs  = session_ctx.get("preferred_pairs", [])
        gmt_time         = session_ctx.get("gmt_time", "N/A")

        # Day 81+ hotfix: In TEST_MODE, hide the dead_zone flag from the LLM
        # so it actually produces a tradeable signal during off-hours.
        # Without this, the LLM sees is_dead_zone=true (or current_session
        # = "DEAD_ZONE") and returns WAIT regardless of the technical
        # analysis — it follows the "DEAD_ZONE → WAIT" rule in its prompt.
        if is_dead_zone or curr_session == "DEAD_ZONE":
            try:
                from config import TEST_MODE
                if TEST_MODE:
                    is_dead_zone = False
                    curr_session = "TOKYO"  # relabel so LLM doesn't see DEAD_ZONE
                    sess_trade_ok = True
                    sess_strategy = "RANGE_TRADING"
                    # Lower minimum confidence in test mode so signal passes
                    sess_min_conf = min(sess_min_conf, 30)
            except Exception:
                pass

        # ── Intermarket / Macro (Day 65) ───────────────────────
        dxy_trend         = intermarket_ctx.get("dxy_trend", "NEUTRAL")
        dxy_change        = intermarket_ctx.get("dxy_change_pct", 0)
        gold_trend        = intermarket_ctx.get("gold_trend", "NEUTRAL")
        oil_trend         = intermarket_ctx.get("oil_trend", "NEUTRAL")
        us10y_trend       = intermarket_ctx.get("us10y_trend", "NEUTRAL")
        sp500_trend       = intermarket_ctx.get("sp500_trend", "NEUTRAL")
        vix_value         = intermarket_ctx.get("vix_value")
        vix_trend         = intermarket_ctx.get("vix_trend", "NEUTRAL")
        macro_regime      = intermarket_ctx.get("macro_regime", "NEUTRAL")
        macro_regime_conf = intermarket_ctx.get("macro_regime_confidence", 0)
        trading_mode      = intermarket_ctx.get("trading_mode", "NORMAL")
        usd_bias          = intermarket_ctx.get("usd_bias", "NEUTRAL")
        usd_confirmations = intermarket_ctx.get("usd_confirmations", [])
        macro_pair_bias   = intermarket_ctx.get("macro_pair_bias", "NEUTRAL")
        macro_currency_bias = intermarket_ctx.get("macro_currency_bias", {})
        macro_score       = intermarket_ctx.get("macro_score", 0)
        cross_asset_conf  = intermarket_ctx.get("cross_asset_confirmed", False)
        cross_asset_note  = intermarket_ctx.get("cross_asset_note", "")
        event_risk_elev   = intermarket_ctx.get("event_risk_elevated", False)
        event_risk_pen    = intermarket_ctx.get("event_risk_penalty", 0)
        macro_corr        = intermarket_ctx.get("macro_correlations", {})

        ctx = {
            "pair":      symbol,
            "timeframe": timeframe,
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),

            # ── Day 63: Session Intelligence ──
            "session_intelligence": {
                "current_session":       curr_session,
                "gmt_time":              gmt_time,
                "session_volatility":    sess_volatility,
                "session_strategy":      sess_strategy,
                "session_trade_allowed": sess_trade_ok,
                "minimum_confidence":    sess_min_conf,
                "risk_multiplier":       sess_risk_mult,
                "pair_priority_score":   pair_priority,
                "pair_session_label":    pair_label,
                "is_overlap_session":    is_overlap,
                "is_dead_zone":          is_dead_zone,
                "london_open_window":    london_open_win,
                "in_session_transition": in_transition,
                "transition_type":       transition_type,
                "transition_alert":      transition_alert,
                "session_score":         session_score,
                "session_grade":         session_grade,
                "smc_session_fusion_allowed": fusion_allowed,
                "smc_session_fusion_score":   fusion_score,
                "preferred_pairs":       preferred_pairs[:5],
            },

            # ── Day 65: Global Market / Intermarket Intelligence ──
            "global_market_intelligence": {
                "dxy_trend":               dxy_trend,
                "dxy_change_pct":          dxy_change,
                "gold_trend":              gold_trend,
                "oil_trend":               oil_trend,
                "us10y_yield_trend":       us10y_trend,
                "sp500_trend":             sp500_trend,
                "vix_value":               vix_value,
                "vix_trend":               vix_trend,
                "macro_regime":            macro_regime,          # RISK_ON / RISK_OFF / NEUTRAL
                "macro_regime_confidence": macro_regime_conf,
                "trading_mode":            trading_mode,          # NORMAL / CAUTIOUS / DEFENSIVE
                "usd_bias":                usd_bias,              # STRONG / MODERATE / NEUTRAL
                "usd_confirmations":       usd_confirmations,
                "macro_pair_bias":         macro_pair_bias,       # BUY / SELL / NEUTRAL for THIS pair
                "macro_currency_bias":     macro_currency_bias,
                "macro_score":             macro_score,           # 0-100
                "cross_asset_confirmed":   cross_asset_conf,
                "cross_asset_note":        cross_asset_note,
                "event_risk_elevated":     event_risk_elev,
                "event_risk_penalty":      event_risk_pen,
                "intermarket_correlations": macro_corr,
            },

            "price_action": {
                "current_price":   close_price,
                "trend":           trend,
                "rsi":             round(rsi, 1),
                "rsi_signal":      rsi_sig,
                "macd_cross":      macd_cross,
                "atr":             round(atr, 5),
                "bb_position_pct": round(bb_pct * 100, 1),
            },

            "patterns": {
                "latest_pattern": latest_pat,
                "pattern_signal": pat_signal,
                "recent":         recent_pats[-3:] if recent_pats else [],
            },

            "support_resistance": {
                "nearest_support":    nearest_sup,
                "nearest_resistance": nearest_res,
                "price_location":     location,
                "pivot":              pivot,
            },

            "market_regime": {
                "regime":     market_regime,
                "direction":  direction,
                "strength":   strength,
                "volatility": volatility,
            },

            "multi_timeframe": {
                "overall_bias": mtf_overall,
                "confidence":   mtf_conf,
                "timeframes":   mtf_trends,
            },

            "market_bias_engine": {
                "bias":         bias_label,
                "confidence":   bias_conf,
                "has_conflict": has_conflict,
            },

            "smart_money_concepts": {
                "signal":               smc_signal,
                "direction":            smc_direction,
                "confluence_score":     smc_score,
                "grade":                smc_grade,
                "factors_present":      [k for k, v in smc_factors.items() if v],
                "h4_order_block_zone":  smc_ob_zone,
                "h4_fvg_zone":          smc_fvg_zone,
                "h4_bos":               smc_h4_bos,
                "h4_choch":             smc_h4_choch,
                "summary":              smc_analysis,
            },

            "fibonacci": {
                "zone":         fib_zone,
                "in_golden":    fib_in_gold,
                "signal":       fib_signal,
            },

            "vision_ai": {
                "trend":      vision_trend,
                "confidence": vision_conf,
            },

            "sentiment": {
                "score":           sent_score,
                "bias":            sent_bias,
                "confidence":      sent_conf,
                "retail_long_pct": retail_long,
                "fear_greed":      fg_label,
                "dxy_trend":       dxy_trend_sent,
                "key_reasons":     sent_reasons[:3],
            },

            "news": {
                "trade_allowed":   trade_allowed,
                "risk_level":      news_risk,
                "upcoming_events": upcoming_news[:3],
            },

            "rule_engine": {
                "signal":     rule_signal,
                "confidence": rule_conf,
            },

            # Round-13 fix: previously AIAnalyst ("Classic LLM Analyst",
            # step 10 of the pipeline) ran a full independent LLM call
            # and its BUY/SELL/confidence was silently discarded —
            # MasterAnalyst never saw it and made its own independent
            # LLM call from scratch. Result: two LLM opinions per cycle
            # per symbol (2x token/rate-limit cost) that could — and
            # regularly did — disagree with no reconciliation, e.g.
            # ai_analyst=BUY 80% vs master_analyst's own LLM=WAIT 0%
            # in the same cycle. Now the classic analyst's verdict is
            # passed in as an input the Master must explicitly weigh
            # (agree/override with reasoning), not a second independent
            # coin flip the voting system silently double-counts.
            "classic_llm_analyst": {
                "signal":      (classic_llm_ctx or {}).get("llm_signal", "WAIT"),
                "confidence":  (classic_llm_ctx or {}).get("llm_confidence", 0),
                "reasoning":   ((classic_llm_ctx or {}).get("llm_reasoning") or "")[:200],
                "key_risk":    (classic_llm_ctx or {}).get("llm_key_risk", ""),
                "note": (
                    "This is a separate LLM's quick technical read (no "
                    "session/macro/SMC context). Treat it as one more "
                    "input, not ground truth — if you disagree, say why "
                    "in self_critique rather than silently ignoring it."
                ),
            },

            "trade_history": {
                "total_trades":   total_trades,
                "win_rate_pct":   win_rate,
                "recent_results": recent_results[-5:],
                "key_lessons":    lessons[:3],
            },

            # ── Day 90 — Six new analyzers + strategy selector ─────
            "divergence": {
                "valid":          (divergence_ctx or {}).get("divergence_valid", False),
                "type":           (divergence_ctx or {}).get("divergence_type", "NONE"),
                "signal":         (divergence_ctx or {}).get("divergence_signal", "NONE"),
                "score":          (divergence_ctx or {}).get("divergence_score", 0),
                "reversal_risk":  (divergence_ctx or {}).get("divergence_reversal_risk", "LOW"),
                "trend_cont":     (divergence_ctx or {}).get("divergence_trend_cont", "NONE"),
            },
            "ichimoku": {
                "valid":          (ichimoku_ctx or {}).get("ichimoku_valid", False),
                "trend":          (ichimoku_ctx or {}).get("ichimoku_trend", "NEUTRAL"),
                "cloud":          (ichimoku_ctx or {}).get("ichimoku_cloud", "UNKNOWN"),
                "cloud_color":    (ichimoku_ctx or {}).get("ichimoku_cloud_color", "UNKNOWN"),
                "strength":       (ichimoku_ctx or {}).get("ichimoku_strength", 0),
                "signal":         (ichimoku_ctx or {}).get("ichimoku_signal", "WAIT"),
                "tk_cross":       (ichimoku_ctx or {}).get("ichimoku_tk_cross", "NONE"),
                "chikou_clear":   (ichimoku_ctx or {}).get("ichimoku_chikou_clear", False),
            },
            "volatility": {
                "valid":            (volatility_ctx or {}).get("volatility_valid", False),
                "squeeze_on":       (volatility_ctx or {}).get("squeeze_on", False),
                "squeeze_strength": (volatility_ctx or {}).get("squeeze_strength", "NONE"),
                "bb_width_pct":     (volatility_ctx or {}).get("bb_width_pct", 0),
                "atr_regime":       (volatility_ctx or {}).get("atr_regime", "NORMAL"),
                "expansion_prob":   (volatility_ctx or {}).get("expansion_prob", 0),
                "release":          (volatility_ctx or {}).get("volatility_release", "NONE"),
                "signal":           (volatility_ctx or {}).get("volatility_signal", "WAIT"),
            },
            "volume_profile": {
                "valid":             (volume_profile_ctx or {}).get("volume_profile_valid", False),
                "poc":               (volume_profile_ctx or {}).get("vp_poc"),
                "value_area_high":   (volume_profile_ctx or {}).get("vp_value_area_high"),
                "value_area_low":    (volume_profile_ctx or {}).get("vp_value_area_low"),
                "price_position":    (volume_profile_ctx or {}).get("vp_price_position", "UNKNOWN"),
                "bias":              (volume_profile_ctx or {}).get("vp_bias", "NEUTRAL"),
                "signal":            (volume_profile_ctx or {}).get("vp_signal", "WAIT"),
                "hvn_count":         (volume_profile_ctx or {}).get("vp_hvn_count", 0),
                "lvn_count":         (volume_profile_ctx or {}).get("vp_lvn_count", 0),
            },
            "smc_advanced": {
                "valid":            (smc_advanced_ctx or {}).get("smc_adv_valid", False),
                "bias":             (smc_advanced_ctx or {}).get("smc_adv_bias", "NEUTRAL"),
                "signal":           (smc_advanced_ctx or {}).get("smc_adv_signal", "WAIT"),
                "active_count":     (smc_advanced_ctx or {}).get("smc_adv_active_count", 0),
                "active_signals":   (smc_advanced_ctx or {}).get("smc_adv_active_signals", []),
                "mitigation_count": (smc_advanced_ctx or {}).get("smc_adv_mitigation_count", 0),
                "inducement_count": (smc_advanced_ctx or {}).get("smc_adv_inducement_count", 0),
                "has_active_retest":(smc_advanced_ctx or {}).get("smc_adv_has_active_retest", False),
            },
            "mtf_structure": {
                "valid":            (mtf_structure_ctx or {}).get("mtf_structure_valid", False),
                "combined_bias":    (mtf_structure_ctx or {}).get("mtf_combined_bias", "NEUTRAL"),
                "alignment":        (mtf_structure_ctx or {}).get("mtf_alignment", "INCOMPLETE"),
                "conflict":         (mtf_structure_ctx or {}).get("mtf_conflict", False),
                "trade_permission": (mtf_structure_ctx or {}).get("mtf_trade_permission", "NO_TRADE"),
                "external_bias":    (mtf_structure_ctx or {}).get("mtf_external_bias", "UNKNOWN"),
                "internal_bias":    (mtf_structure_ctx or {}).get("mtf_internal_bias", "UNKNOWN"),
                "external_bos":     (mtf_structure_ctx or {}).get("mtf_external_bos", "NONE"),
                "internal_choch":   (mtf_structure_ctx or {}).get("mtf_internal_choch", "NONE"),
            },
            "strategy": {
                "family":          (strategy_ctx or {}).get("strategy", "WAIT"),
                "confidence":      (strategy_ctx or {}).get("confidence", 0),
                "risk_mult":       (strategy_ctx or {}).get("risk_mult", 0.0),
                "position_mult":   (strategy_ctx or {}).get("position_mult", 0.0),
                "active_modules":  (strategy_ctx or {}).get("active_modules", []),
                "avoid":           (strategy_ctx or {}).get("avoid", []),
                "reason":          (strategy_ctx or {}).get("reason", ""),
            },
            # Day 92 — NewsAPI real-time news sentiment
            "news_api": {
                "bias":             (news_api_ctx or {}).get("newsapi_bias", "NEUTRAL"),
                "score":            (news_api_ctx or {}).get("newsapi_score", 0),
                "headline_count":   (news_api_ctx or {}).get("newsapi_headlines", 0),
                "top_headlines":    (news_api_ctx or {}).get("newsapi_top", []),
                "source":           (news_api_ctx or {}).get("newsapi_source", "unknown"),
            },
            # Day 94 — Institutional grade APIs
            "economic_calendar": {
                "source":         (econ_calendar_ctx or {}).get("econcal_source", "none"),
                "event_count":    (econ_calendar_ctx or {}).get("econcal_event_count", 0),
                "high_impact":    (econ_calendar_ctx or {}).get("econcal_high_impact", 0),
                "trade_block":    (econ_calendar_ctx or {}).get("econcal_trade_block", False),
                "block_reason":   (econ_calendar_ctx or {}).get("econcal_block_reason", ""),
                "next_event":     (econ_calendar_ctx or {}).get("econcal_next_event"),
            },
            "fred_macro": {
                "source":          (fred_ctx or {}).get("fred_source", "none"),
                "yield_curve":     (fred_ctx or {}).get("fred_yield_curve", "unknown"),
                "inflation_trend": (fred_ctx or {}).get("fred_inflation_trend", "stable"),
                "rate_environment":(fred_ctx or {}).get("fred_rate_env", "neutral"),
                "cpi":             (fred_ctx or {}).get("fred_cpi"),
                "unemployment":    (fred_ctx or {}).get("fred_unemployment"),
                "fed_rate":        (fred_ctx or {}).get("fred_fed_rate"),
                "treasury_10y":    (fred_ctx or {}).get("fred_10y_yield"),
                "vix":             (fred_ctx or {}).get("fred_vix"),
            },
            "retail_sentiment": {
                "source":         (retail_sentiment_ctx or {}).get("sentiment_source", "fallback"),
                "retail_long_pct":(retail_sentiment_ctx or {}).get("sentiment_retail_long", 50),
                "retail_short_pct":(retail_sentiment_ctx or {}).get("sentiment_retail_short", 50),
                "sentiment_label":(retail_sentiment_ctx or {}).get("sentiment_label", "NEUTRAL"),
                "contrarian":     (retail_sentiment_ctx or {}).get("sentiment_contrarian", "NEUTRAL"),
                "strength":       (retail_sentiment_ctx or {}).get("sentiment_strength", "WEAK"),
                "trade_bias":     (retail_sentiment_ctx or {}).get("sentiment_bias", "NEUTRAL"),
                "confidence":     (retail_sentiment_ctx or {}).get("sentiment_confidence", 0),
                "stop_cluster":   (retail_sentiment_ctx or {}).get("sentiment_stop_cluster"),
            },
        }

        return json.dumps(ctx, indent=2, default=str)

    def _call_llm(self, context: str) -> str:
        """Call LLM with Groq (primary) → Gemini (fallback) chain.
        Multi-key rotation: if one key fails, automatically tries next.
        Anthropic + OpenRouter are intentionally NOT used per user request.

        Day 81+ hotfix: per-cycle LLM throttle caps total calls per
        symbol cycle to MAX_LLM_CALLS_PER_CYCLE (default 5).  Also
        enforces LLM_CALL_INTERVAL_SEC between calls (default 1.0s)
        to prevent the Groq free-tier 429 storm.

        Day 90: LLM response cache — if the same prompt was asked in
        the last 5 minutes (same symbol/timeframe/regime), return the
        cached response without calling the API. This is the single
        biggest token-saver for long-duration demo trading where the
        market state doesn't change much between cycles.

        Day 95 hotfix: Cerebras's gpt-oss-120b reasoning model gets its
        own larger max_tokens budget (CEREBRAS_MAX_TOK) plus a low
        reasoning_effort, instead of reusing the Groq-tuned MAX_TOK=800.
        An explicit empty-response check turns a silently truncated
        answer into a clean RuntimeError instead of a downstream
        JSONDecodeError with a noisy traceback.
        """
        # ── Day 90 — cache lookup ──────────────────────────────
        # Day 101 hotfix: the cache key used to be hardcoded to
        # ("groq", GROQ_MODEL, context) — a misleading label. In reality
        # the value stored under that key can come from ANY of the 5
        # fallback providers (Groq/Gemini/Cerebras/SambaNova/OpenRouter),
        # since every provider's success path wrote into the SAME key.
        # Concretely this meant: Groq exhausts → Cerebras (gpt-oss-120b,
        # a reasoning model with a totally different output profile)
        # answers → gets cached under the "groq" label → next cycle,
        # even after Groq keys recover, the stale Cerebras response is
        # served back as if it were a fresh Groq cache hit. Two
        # completely different LLMs were being silently conflated under
        # one label.
        #
        # Fix: the cache key itself is now provider-neutral (it's keyed
        # on the market context, which is the actual thing we want to
        # avoid re-querying for — any provider's answer is a valid reuse
        # for the same context). Provenance (which LLM really produced
        # the cached text) is tracked separately in `_prov_cache_key` so
        # cache hits are logged accurately instead of implying "groq".
        _cache = None
        _cache_key = None
        _prov_cache_key = None
        try:
            from core.llm_cache import get_llm_cache
            _cache = get_llm_cache()
            _cache_key = _cache.make_key("master_analyst", "multi-provider", context)
            _prov_cache_key = _cache_key + "::provider"
            _cached = _cache.get(_cache_key)
            if _cached is not None:
                try:
                    _cached_provider = _cache.get(_prov_cache_key) or "unknown-provider"
                except Exception:
                    _cached_provider = "unknown-provider"
                log.debug(
                    f"[MasterAnalyst] LLM cache HIT (originally produced by "
                    f"{_cached_provider}) — skipping API call"
                )
                return _cached
        except Exception as _e:
            log.debug(f"[MasterAnalyst] cache lookup failed: {_e}")

        def _store_cache(_response: str, _provider_label: str, _token_estimate: int) -> None:
            """Day 101: single helper so every provider's success path
            writes to the SAME provider-neutral key AND records its real
            provenance — instead of each call site re-hardcoding 'groq'."""
            if _cache is None or _cache_key is None:
                return
            try:
                _cache.set(_cache_key, _response, token_estimate=_token_estimate)
                _cache.set(_prov_cache_key, _provider_label, token_estimate=0)
            except Exception:
                pass

        # Per-cycle throttle check
        if _key_manager is not None:
            allowed, reason = _key_manager.check_cycle_throttle()
            if not allowed:
                log.info(f"[MasterAnalyst] LLM skipped — {reason}")
                raise RuntimeError(f"LLM throttle: {reason}")

        user_prompt = (
            "Here is the complete market intelligence package (session-aware AND "
            "macro/intermarket-aware) for analysis:\n\n"
            f"{context}\n\n"
            "IMPORTANT: Check session_intelligence block first, then "
            "global_market_intelligence block. Follow session rules and macro rules strictly.\n"
            "Provide your professional trade decision as JSON."
        )

        import time as _time
        from core.llm_key_manager import log_llm_call_failure

        # Primary: Groq (with multi-key retry)
        max_retries = 3
        for attempt in range(max_retries):
            client = _groq_client
            if client is None and _key_manager is not None:
                client = _key_manager.get_groq_client()
            if client is None and _key_manager is not None:
                # Round-16 audit fix: NON-BLOCKING check instead of blocking wait.
                #
                # Previously (Round-12): called wait_for_any_groq(max_wait=30)
                # which BLOCKED for up to 30 seconds. This consumed most of
                # the analyze() timeout budget, leaving no time for fallback
                # providers (Cerebras/SambaNova/OpenRouter/Gemini).
                #
                # The operator's audit confirmed: "Groq exhaust হলেও Gemini
                # fallback call হচ্ছে না" — the blocking wait ate the entire
                # budget before the fallback chain could fire.
                #
                # Now: use a NON-BLOCKING has_any_groq check. If no keys
                # are available RIGHT NOW, immediately fall through to the
                # next provider. Groq will recover in the background.
                if not _key_manager.has_any_groq:
                    log.info(
                        "[MasterAnalyst] All Groq keys exhausted — skipping "
                        "to fallback providers (non-blocking)"
                    )
                    break
            if client is None:
                log.info(
                    "[MasterAnalyst] No Groq client available — "
                    "falling back to next provider"
                )
                break
            try:
                resp = client.chat.completions.create(
                    model=GROQ_MODEL,
                    max_tokens=MAX_TOK,
                    temperature=0.2,
                    messages=[
                        {"role": "system", "content": self._SYSTEM},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                if _key_manager is not None:
                    usage = getattr(resp, "usage", None)
                    tokens_used = 0
                    if usage is not None:
                        tokens_used = getattr(usage, "total_tokens", 0) or 0
                    _key_manager.mark_groq_success(tokens_used=tokens_used, client=client)
                _response = resp.choices[0].message.content.strip()
                # ── Day 90 — cache store (Day 101: correctly labeled) ──
                _store_cache(_response, f"groq:{GROQ_MODEL}", MAX_TOK)
                return _response
            except Exception as e:
                info = log_llm_call_failure(
                    log, "Groq", GROQ_MODEL, attempt, max_retries, e
                )
                if _key_manager is not None:
                    _key_manager.mark_groq_failure(
                        info["error_str"], info["rate_limited"], client=client
                    )
                    # Get fresh client with different key
                    import sys
                    current_module = sys.modules[__name__]
                    current_module._groq_client = _key_manager.get_groq_client()
                if attempt < max_retries - 1:
                    _time.sleep(1)

        # ── PRIMARY FALLBACK: Gemini (moved up from last position) ──
        # Gemini is the most reliable free-tier provider after Groq.
        # Cerebras (Cloudflare 403) and SambaNova (410 Gone) are known
        # broken, so we try Gemini BEFORE them to avoid wasted latency.
        # Requires: pip install google-genai (added to requirements.txt)
        # Fallback: Gemini (with multi-key retry)
        for attempt in range(max_retries):
            client = _gemini_client
            if client is None and _key_manager is not None:
                client = _key_manager.get_gemini_client()
            if client is None:
                log.error("[MasterAnalyst] No Gemini client available (keys exhausted or missing)")
                break
            try:
                full_prompt = f"{self._SYSTEM}\n\n{user_prompt}"
                resp = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=full_prompt,
                )
                if _key_manager is not None:
                    usage = getattr(resp, "usage_metadata", None)
                    tokens_used = 0
                    if usage is not None:
                        tokens_used = getattr(usage, "total_token_count", 0) or 0
                    _key_manager.mark_gemini_success(tokens_used=tokens_used, client=client)
                _response = resp.text.strip()
                # Day 101: Gemini's success path previously never cached
                # its response at all (only Groq did) — fixed as part of
                # the same mislabeled-cache cleanup.
                _store_cache(_response, f"gemini:{GEMINI_MODEL}", MAX_TOK)
                return _response
            except Exception as e:
                info = log_llm_call_failure(
                    log, "Gemini", GEMINI_MODEL, attempt, max_retries, e
                )
                if _key_manager is not None:
                    _key_manager.mark_gemini_failure(
                        info["error_str"], info["rate_limited"], client=client
                    )
                    import sys
                    current_module = sys.modules[__name__]
                    current_module._gemini_client = _key_manager.get_gemini_client()
                if attempt < max_retries - 1:
                    _time.sleep(1)


        # ── Day 91 — Cerebras fallback (OpenAI-compatible) ────────
        # Cerebras is fastest (dedicated Llama inference on wafer-scale
        # engine) but currently behind Cloudflare bot-filter on Linux
        # VPS. Will work transparently when that's resolved.
        #
        # Day 95 hotfix: gpt-oss-120b is a *reasoning* model — it spends
        # part of its max_tokens budget on internal chain-of-thought
        # before emitting the JSON answer. Reusing the Groq-tuned
        # MAX_TOK=800 here exhausted the budget mid-reasoning, so the
        # response came back empty ("Expecting value: line 1 column 1
        # (char 0)") or cut off mid-string ("Unterminated string...").
        # Fix: dedicated CEREBRAS_MAX_TOK + reasoning_effort="low" for
        # gpt-oss models, plus an explicit empty-content guard.
        if _key_manager is not None and _key_manager.has_any_cerebras:
            try:
                cb_client = _key_manager.get_cerebras_client()
                if cb_client is not None:
                    cb_model = os.getenv("CEREBRAS_MODEL", "llama3.1-8b-instruct")
                    cb_kwargs = dict(
                        model=cb_model,
                        max_tokens=CEREBRAS_MAX_TOK,
                        temperature=0.2,
                        messages=[
                            {"role": "system", "content": self._SYSTEM},
                            {"role": "user", "content": user_prompt},
                        ],
                    )
                    # gpt-oss reasoning models support reasoning_effort —
                    # keep it low so most of the (larger) token budget
                    # lands on the actual JSON answer, not chain-of-thought.
                    if "gpt-oss" in cb_model.lower():
                        cb_kwargs["reasoning_effort"] = CEREBRAS_REASONING_EFFORT

                    resp = cb_client.chat.completions.create(**cb_kwargs)
                    _key_manager.mark_cerebras_success()
                    _response = (resp.choices[0].message.content or "").strip()

                    if not _response:
                        # Reasoning model still came back empty (e.g. hit
                        # the token cap mid-thought) — fail cleanly instead
                        # of letting an empty string reach json.loads and
                        # produce a noisy JSONDecodeError traceback.
                        finish_reason = getattr(
                            resp.choices[0], "finish_reason", "unknown"
                        )
                        raise RuntimeError(
                            f"Cerebras ({cb_model}) returned empty content "
                            f"after reasoning — finish_reason={finish_reason}. "
                            f"Try raising CEREBRAS_MAX_TOKENS (current="
                            f"{CEREBRAS_MAX_TOK}) or lowering "
                            f"CEREBRAS_REASONING_EFFORT (current="
                            f"{CEREBRAS_REASONING_EFFORT})."
                        )

                    log.info(f"[MasterAnalyst] Cerebras OK | model={cb_model}")
                    _store_cache(_response, f"cerebras:{cb_model}", CEREBRAS_MAX_TOK)
                    return _response
            except Exception as e:
                info = log_llm_call_failure(log, "Cerebras", "cerebras-model", 0, 1, e)
                _key_manager.mark_cerebras_failure(info["error_str"], info["rate_limited"])

        # ── Day 91 — SambaNova fallback (OpenAI-compatible) ───────
        # Day 99+ V3 FIX (Master List Issue #1): SambaNova deprecated all
        # Llama 3.1 models. Default is now DeepSeek-V3 (current as of
        # 2026-Q3). Override via SAMBANOVA_MODEL env var if needed.
        if _key_manager is not None and _key_manager.has_any_sambanova:
            try:
                sn_client = _key_manager.get_sambanova_client()
                if sn_client is not None:
                    sn_model = os.getenv("SAMBANOVA_MODEL", "DeepSeek-V3")
                    resp = sn_client.chat.completions.create(
                        model=sn_model,
                        max_tokens=MAX_TOK,
                        temperature=0.2,
                        messages=[
                            {"role": "system", "content": self._SYSTEM},
                            {"role": "user", "content": user_prompt},
                        ],
                    )
                    _key_manager.mark_sambanova_success()
                    _response = resp.choices[0].message.content.strip()
                    log.info(f"[MasterAnalyst] SambaNova OK | model={sn_model}")
                    _store_cache(_response, f"sambanova:{sn_model}", MAX_TOK)
                    return _response
            except Exception as e:
                info = log_llm_call_failure(log, "SambaNova", "sambanova-model", 0, 1, e)
                _key_manager.mark_sambanova_failure(info["error_str"], info["rate_limited"])

        # ── Day 91 — OpenRouter fallback (OpenAI-compatible) ──────
        # OpenRouter is the most reliable free-tier option — it routes
        # to many providers (Google, Meta, Nvidia, Qwen, etc.) and we
        # verified multiple free models work (gemma-4-26b, nemotron-30b,
        # liquid-1.2b). If primary model 429s, try fallback models.
        if _key_manager is not None and _key_manager.has_any_openrouter:
            or_models = [os.getenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free")]
            fb1 = os.getenv("OPENROUTER_MODEL_FALLBACK_1", "")
            fb2 = os.getenv("OPENROUTER_MODEL_FALLBACK_2", "")
            if fb1: or_models.append(fb1)
            if fb2: or_models.append(fb2)

            for or_model in or_models:
                try:
                    or_client = _key_manager.get_openrouter_client()
                    if or_client is None:
                        break
                    resp = or_client.chat.completions.create(
                        model=or_model,
                        max_tokens=MAX_TOK,
                        temperature=0.2,
                        messages=[
                            {"role": "system", "content": self._SYSTEM},
                            {"role": "user", "content": user_prompt},
                        ],
                    )
                    _key_manager.mark_openrouter_success()
                    _response = resp.choices[0].message.content.strip()
                    log.info(f"[MasterAnalyst] OpenRouter OK | model={or_model}")
                    _store_cache(_response, f"openrouter:{or_model}", MAX_TOK)
                    return _response
                except Exception as e:
                    info = log_llm_call_failure(log, "OpenRouter", or_model, 0, 1, e)
                    _key_manager.mark_openrouter_failure(info["error_str"], info["rate_limited"])
                    # If this model 429s, try the next fallback model
                    continue

        raise RuntimeError("[MasterAnalyst] No LLM client available (all keys failed)")

    def _parse_response(self, raw: str) -> dict:
        text = raw.strip() if raw else ""

        # Day 99+ FIX (Issue #1): delegate to utils.llm_json.parse_llm_json
        # for robust parsing. Previously this method had inline regex
        # that only stripped ```json fences at the very start / end of
        # the string. If the LLM emitted any prose BEFORE the fence
        # (e.g. "Sure, here is the JSON:\n```json\n{...}\n```") the
        # leading-prose path meant the fence-strip regex didn't match
        # and json.loads raised JSONDecodeError, crashing the cycle.
        # The shared helper handles leading prose, trailing prose,
        # smart quotes, trailing commas, and embedded code blocks.
        from utils.llm_json import parse_llm_json

        try:
            data = parse_llm_json(text)
        except json.JSONDecodeError as e:
            log.error(
                f"[MasterAnalyst] JSON parse error after cleaning: {e.msg}"
            )
            raise

        data.setdefault("market_story", "Market analysis pending.")
        data.setdefault("key_levels", [])
        data.setdefault("trade_plan", {
            "signal": "WAIT", "entry": None, "sl": None,
            "tp1": None, "tp2": None, "confidence": 0,
            "reasoning": "Insufficient data."
        })
        data.setdefault("risks", [])
        data.setdefault("self_critique", "")
        data.setdefault("no_trade_reason", "")

        sig = data["trade_plan"].get("signal", "WAIT").upper()
        if sig not in ("BUY", "SELL", "WAIT"):
            sig = "WAIT"
        data["trade_plan"]["signal"] = sig
        return data

    def _calculate_final_confidence(
        self,
        llm_conf:       int,
        technical_conf: int,
        sentiment_conf: int,
        memory_ctx:     dict,
        smc_ctx:        dict = None,
        session_ctx:    dict = None,       # ← Day 63
        intermarket_ctx: dict = None,       # ← Day 65
        sentiment_ctx:  dict = None,
    ) -> int:
        """
        Weighted average:
            LLM opinion          : 30%
            Technical signals    : 20%
            Sentiment             : 10%
            Historical success    : 8%
            SMC confluence        : 10%
            Session score         : 12%
            Macro score (Day 65)  : 10%   ← new

        Total = 100%
        """
        smc_ctx         = smc_ctx or {}
        session_ctx      = session_ctx or {}
        intermarket_ctx  = intermarket_ctx or {}

        # FIX (agents-folder audit): exclude factors whose context was empty/missing
        # from the weighted average and re-normalize. Previously all 7 factors
        # defaulted to 50 (neutral) when context was unavailable, which diluted
        # the confidence — a missing engine silently contributed a "meh" score
        # instead of being excluded. Now only factors with real data participate.
        available = {}  # factor_name -> (weight_fraction, value)
        available["llm"]       = (0.30, llm_conf)
        available["technical"] = (0.20, technical_conf)
        available["sentiment"] = (0.10, sentiment_conf) if sentiment_ctx else None
        available["history"]  = (0.08, memory_ctx.get("overall_win_rate", 50)) if memory_ctx else None
        available["smc"]       = (0.10, smc_ctx.get("smc_score", 50)) if smc_ctx and smc_ctx.get("smc_score") is not None else None
        available["session"]   = (0.12, session_ctx.get("session_score", 50)) if session_ctx and session_ctx.get("session_score") is not None else None
        available["macro"]     = (0.10, intermarket_ctx.get("macro_score", 50)) if intermarket_ctx and intermarket_ctx.get("macro_score") is not None else None

        # Filter out None entries (missing data) and normalize weights
        # (filter BEFORE unpacking — available[k] is None, not a (w, v) tuple,
        # for any excluded factor, so unpacking in the `for` clause itself
        # would crash before the `if` filter ever runs)
        present = {k: val for k, val in available.items() if val is not None}
        total_w = sum(w for w, _ in present.values())

        if total_w > 0:
            weighted = sum(v * w for w, v in present.values()) / total_w
        else:
            # No context data at all — fall back to LLM + technical only
            weighted = (llm_conf * 0.60 + technical_conf * 0.40)

        # Session risk multiplier adjustment
        sess_risk = session_ctx.get("session_risk_mult", 1.0)
        if sess_risk < 1.0:
            weighted *= sess_risk

        # Recent trades momentum
        recent = memory_ctx.get("recent_results", [])
        if recent:
            last_5      = recent[-5:]
            win_streak  = sum(1 for r in last_5 if r == "WIN")
            loss_streak = sum(1 for r in last_5 if r == "LOSS")
            if win_streak >= 3:
                weighted += 3
            if loss_streak >= 3:
                weighted -= 5

        # SMC grade bonus
        if smc_ctx.get("smc_grade") in ("A+", "A"):
            weighted += 3

        # Session overlap bonus
        if session_ctx.get("is_overlap"):
            weighted += 2

        # Day 65 — Macro alignment bonus / event risk penalty
        if intermarket_ctx.get("cross_asset_confirmed"):
            weighted += 3
        if intermarket_ctx.get("event_risk_elevated"):
            weighted -= intermarket_ctx.get("event_risk_penalty", 0)
        if intermarket_ctx.get("trading_mode") == "DEFENSIVE":
            weighted -= 8
        elif intermarket_ctx.get("trading_mode") == "CAUTIOUS":
            weighted -= 4

        # ── ARCHITECTURAL FIX (institutional refactor) ───────────────
        # Previously: hard-zeroed `weighted = 0` when session/fusion/dead-zone
        # gate failed. This DESTROYED the analysis-layer confidence, causing
        # downstream consumers (decision_agent.py, signal fusion) to see
        # 0% confidence and treat the trade as if no analysis ever existed.
        #
        # Now: MasterAnalyst is an ANALYSIS layer. Session/fusion/dead-zone
        # are EXECUTION gates. The analyst reports its full analysis verdict
        # + computed confidence, and the execution layer (TradePermission)
        # decides whether to block. We apply a heavy penalty (×0.3) so the
        # confidence reflects "analysis is valid but session is unfavorable",
        # NOT zero. We also set a `session_gate_penalty` flag in the returned
        # context so downstream consumers can see WHY confidence was reduced.
        # ──────────────────────────────────────────────────────────────
        try:
            from config import TEST_MODE
            _ma_test_mode = bool(TEST_MODE)
        except Exception:
            _ma_test_mode = False

        _session_gate_penalty_applied = False
        _session_gate_reasons = []  # track ALL reasons, not just the last one
        _session_gate_multipliers = []  # track each multiplier applied

        if not _ma_test_mode:
            if session_ctx.get("is_dead_zone"):
                _multiplier = 0.85
                weighted *= _multiplier
                _session_gate_penalty_applied = True
                _session_gate_reasons.append(
                    f"dead_zone ({session_ctx.get('current_session', '?')})"
                )
                _session_gate_multipliers.append(("dead_zone", _multiplier))

            if not session_ctx.get("session_trade_allowed", True):
                _multiplier = 0.9
                weighted *= _multiplier
                _session_gate_penalty_applied = True
                _session_gate_reasons.append(
                    f"session_trade_allowed=False ({session_ctx.get('current_session', '?')})"
                )
                _session_gate_multipliers.append(("session_trade_allowed", _multiplier))

            if not session_ctx.get("fusion_allowed", True):
                _multiplier = 0.95  # mild penalty — SMC alignment is one factor
                weighted *= _multiplier
                _session_gate_penalty_applied = True
                _session_gate_reasons.append(
                    f"fusion_allowed=False "
                    f"(score={session_ctx.get('fusion_score', '?')}, "
                    f"grade={session_ctx.get('fusion_grade', '?')})"
                )
                _session_gate_multipliers.append(("fusion_allowed", _multiplier))

        # Stash the penalty info on the function's return path via instance
        # attribute so analyze() can pick it up and add it to master_ctx.
        # FIX (agents-folder audit): stores ALL reasons and multipliers,
        # not just the last one.  Old code used fragile substring matching
        # on a single reason string, so when two+ penalties applied
        # (common: dead zone + session_trade_allowed=False), only the
        # last was recorded in the audit trail.
        try:
            # FIX (review): `weighted` above is reduced multiplicatively,
            # one gate at a time (weighted *= _multiplier). The reported
            # combined_multiplier previously used an additive approximation
            # (1 - sum(1 - m)) that does NOT equal the actual compounded
            # effect and drifts further off as more gates stack — e.g.
            # dead_zone(0.85) + session_trade_allowed=False(0.9):
            #   real effect    = 0.85 * 0.90 = 0.765
            #   old (wrong)    = 1 - (0.15 + 0.10) = 0.750
            # Use math.prod so this field actually matches what happened
            # to the confidence score.
            _combined = (
                math.prod(m for _, m in _session_gate_multipliers)
                if _session_gate_multipliers else 1.0
            )
            self._last_session_gate_penalty = {
                "applied": _session_gate_penalty_applied,
                "reasons": _session_gate_reasons,
                "multipliers": _session_gate_multipliers,
                "combined_multiplier": _combined,
                "reason": "; ".join(_session_gate_reasons) if _session_gate_reasons else "",
            }
        except Exception:
            pass

        return max(0, min(99, round(weighted)))

    def _fallback_result(self, signal: dict, reason: str) -> dict:
        sig  = signal.get("signal", "WAIT")
        conf = signal.get("confidence", 0)
        # Day 81+ hotfix: when LLM is unavailable (rate-limited, auth failed),
        # the MasterAnalyst should NOT return WAIT if the rule engine has a
        # strong BUY/SELL signal. Use the rule signal directly with its
        # confidence. This is the "rule-engine fallback" path — the rule
        # engine already did all the technical analysis, so its signal is
        # valid even without LLM confirmation.
        # Also extract entry/sl/tp from the rule signal if available.
        entry = signal.get("entry")
        sl    = signal.get("sl")
        tp    = signal.get("tp")
        # P0 fix (audit C7): mirror ai_analyst.py:656 by setting _llm_parse_failed
        # / _llm_unavailable flags so decision_agent.py can zero the master vote
        # instead of treating the rule signal as a 3-weight master verdict.
        _reason_lower = (reason or "").lower()
        _parse_failed = "parse" in _reason_lower or "json" in _reason_lower
        _unavailable  = "no llm" in _reason_lower or "unavailable" in _reason_lower \
                        or "rate" in _reason_lower or "auth" in _reason_lower \
                        or "timeout" in _reason_lower or "429" in _reason_lower \
                        or "401" in _reason_lower or "403" in _reason_lower
        return {
            "market_story":     f"LLM unavailable — using rule engine signal: {sig} ({conf}%)",
            "key_levels":       [],
            "trade_plan": {
                "signal":     sig,
                "entry":      entry,
                "sl":         sl,
                "tp1":        tp,
                "tp2":        None,
                "confidence": conf,
                "reasoning":  f"Fallback — {reason}. Rule engine signal used as-is.",
            },
            "risks":            ["LLM analysis unavailable — rule engine signal only"],
            "self_critique":    "",
            "no_trade_reason":  "" if sig != "WAIT" else reason,
            "final_confidence": conf,
            "llm_raw":          "",
            "error":            reason,
            "_llm_parse_failed": _parse_failed,
            "_llm_unavailable":  _unavailable,
        }

    def get_ai_context(self, result: dict) -> dict:
        plan = result.get("trade_plan", {})
        return {
            "master_signal":     plan.get("signal", "WAIT"),
            "master_entry":      plan.get("entry"),
            "master_sl":         plan.get("sl"),
            "master_tp1":        plan.get("tp1"),
            "master_tp2":        plan.get("tp2"),
            "master_confidence": result.get("final_confidence", 0),
            "master_story":      result.get("market_story", ""),
            "master_risks":      result.get("risks", []),
            "master_critique":   result.get("self_critique", ""),
            # Bug #8 fix: these flags are set by _fallback_result() when the
            # LLM call fails (parse error / unavailable) and are required by
            # decision_agent.py's vote-exclusion logic (master_ctx.get(...)).
            # They were previously dropped here, so an excluded master vote
            # never actually got excluded downstream.
            "_llm_parse_failed": result.get("_llm_parse_failed", False),
            "_llm_unavailable":  result.get("_llm_unavailable", False),
            # FIX (review): was computed but never exposed to downstream
            # consumers — see analyze()'s result dict for details.
            "session_gate_penalty": result.get(
                "session_gate_penalty", {"applied": False}
            ),
        }

    def print_summary(self, result: dict) -> None:
        plan = result.get("trade_plan", {})
        sig  = plan.get("signal", "WAIT")
        icons = {"BUY": "🟢", "SELL": "🔴", "WAIT": "🟡"}
        icon  = icons.get(sig, "⚪")
        bar   = "═" * 56

        print(f"\n{bar}")
        print(f"  🧠  MASTER ANALYST  (Day 42 + 44 + 47 + 63 + 65)")
        print(bar)
        print(f"  Signal          : {icon}  {sig}")
        print(f"  Final Confidence: {result.get('final_confidence', 0)}%")
        print(f"  LLM Confidence  : {plan.get('confidence', 0)}%")
        if sig in ("BUY", "SELL"):
            print(f"  Entry           : {plan.get('entry')}")
            print(f"  SL              : {plan.get('sl')}")
            print(f"  TP1             : {plan.get('tp1')}")
            print(f"  TP2             : {plan.get('tp2')}")
        print()
        print(f"  ── Market Story ──")
        story = result.get("market_story", "")
        words = story.split()
        line  = "  "
        for word in words:
            if len(line) + len(word) > 54:
                print(line)
                line = "  " + word + " "
            else:
                line += word + " "
        if line.strip():
            print(line)
        print()

        key_levels = result.get("key_levels", [])
        if key_levels:
            print(f"  ── Key Levels ──")
            print(f"  {key_levels}")
            print()

        risks = result.get("risks", [])
        if risks:
            print(f"  ── Risks ──")
            for r in risks:
                print(f"  ⚠  {r}")
            print()

        critique = result.get("self_critique", "")
        if critique:
            print(f"  ── Self Critique ──")
            print(f"  {critique}")
            print()

        reasoning = plan.get("reasoning", "")
        if reasoning:
            print(f"  ── Reasoning ──")
            print(f"  {reasoning}")

        if result.get("error"):
            print(f"\n  ⚠  Error: {result['error']}")

        print(bar + "\n")