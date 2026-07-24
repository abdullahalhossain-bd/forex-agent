# ai/ai_analyst.py  —  Day 10 | LLM Analyst Brain
# Primary: Groq (fast)  |  Fallback: Gemini Flash
#
# Day 37: GROQ_MODEL / GEMINI_MODEL are now read from .env (with the
# original hardcoded values as defaults), so you can swap reasoning models
# without touching code — e.g. drop in a bigger Groq model or a different
# Gemini tier for trade reasoning.

import os
import json
import re
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dotenv import load_dotenv
from utils.logger import get_logger

load_dotenv()
log = get_logger("ai_analyst")


class AIAnalyst:
    """
    LLM-powered market analyst।
    Rule engine এর পর second opinion দেয়।

    Flow:
        Technical data → Context builder → LLM → JSON report
    """

    # 2026-07-20 fix: match config.py's move to the cheaper Groq model
    # (llama-3.3-70b-versatile was exhausting the 100k TPD quota), and use
    # `os.getenv(key) or default` since os.getenv's default doesn't kick in
    # for a present-but-empty env var (e.g. "GEMINI_MODEL=" in .env), which
    # was sending model="" to Gemini and crashing every fallback call.
    GROQ_MODEL   = os.getenv("GROQ_MODEL") or "llama-3.1-8b-instant"
    GEMINI_MODEL = os.getenv("GEMINI_MODEL") or "gemini-flash-lite-latest"

    # Local Ollama — user's own laptop, zero cost, zero rate limits, no API
    # key needed. Tried FIRST (see analyze()) so cloud providers (several of
    # which are known broken — see analyze()'s Cerebras/SambaNova comment)
    # are only reached if the local model is unreachable.
    OLLAMA_HOST    = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL   = os.getenv("OLLAMA_ANALYST_MODEL") or os.getenv("OLLAMA_MODEL") or "qwen3:4b"
    OLLAMA_ENABLED = os.getenv("OLLAMA_ANALYST_ENABLED", "true").lower() in ("true", "1", "yes")

    # Rough per-1K-token USD prices used only for cost *estimation* /
    # observability, not billing. Overridable via env if pricing changes.
    _TOKEN_COST_PER_1K = {
        "groq": float(os.getenv("GROQ_COST_PER_1K_TOKENS", "0.0")),
        "gemini": float(os.getenv("GEMINI_COST_PER_1K_TOKENS", "0.0")),
        "cerebras": float(os.getenv("CEREBRAS_COST_PER_1K_TOKENS", "0.0")),
        "sambanova": float(os.getenv("SAMBANOVA_COST_PER_1K_TOKENS", "0.0")),
        "openrouter": float(os.getenv("OPENROUTER_COST_PER_1K_TOKENS", "0.0")),
    }

    def __init__(self):
        self._groq_client   = None
        self._gemini_client = None  # google.genai Client object
        self._ollama_client = None  # ollama.Client object (lazy-init)
        self._init_clients()

        # ── Day 92 — token usage / cost tracking ────────────────────
        # BUGFIX (audit follow-up): analyze() previously had no visibility
        # into how many tokens/dollars each call cost. On a system polling
        # 6 symbols every 60s, a silent runaway (e.g. a provider ignoring
        # max_tokens, or a fallback chain firing every cycle) could burn
        # budget for days before anyone noticed. This is an in-process
        # running counter; a persistent/metrics-backed version can layer
        # on top later without changing the call sites below.
        self._usage_lock = threading.Lock()
        self._usage_totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
            "calls": 0,
        }

        # BUGFIX (audit follow-up): analyze() had no timeout protection —
        # a hung HTTP call to any provider could block the trading loop
        # indefinitely (compounded by wait_for_any_groq's own up-to-5-
        # minute block, fixed separately below). A tiny single-worker pool
        # lets us wrap each blocking provider call with a hard deadline.
        self._call_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ai_analyst_llm")

    def _record_usage(self, provider: str, prompt_tokens: int = 0, completion_tokens: int = 0):
        """Accumulate token usage/cost totals. Best-effort — never raises."""
        try:
            total = (prompt_tokens or 0) + (completion_tokens or 0)
            cost_per_1k = self._TOKEN_COST_PER_1K.get(provider, 0.0)
            cost = (total / 1000.0) * cost_per_1k
            with self._usage_lock:
                self._usage_totals["prompt_tokens"] += prompt_tokens or 0
                self._usage_totals["completion_tokens"] += completion_tokens or 0
                self._usage_totals["total_tokens"] += total
                self._usage_totals["estimated_cost_usd"] += cost
                self._usage_totals["calls"] += 1
        except Exception:
            pass

    def get_usage_summary(self) -> dict:
        """Snapshot of cumulative token usage/cost since process start."""
        with self._usage_lock:
            return dict(self._usage_totals)

    def shutdown(self, wait: bool = False) -> None:
        """Release the internal LLM-call thread pool.

        BUGFIX (audit follow-up): __init__ creates a ThreadPoolExecutor
        but nothing ever closed it. In long-running production use this
        is harmless (one AIAnalyst instance for the process lifetime),
        but if the app ever recreates AIAnalyst (hot-reload, per-request
        instantiation, tests), the old executor's worker threads leak
        indefinitely. Call this on teardown; safe to call multiple times.
        `wait=False` by default so shutdown never blocks on an in-flight
        provider call — matching this class's "never block the trading
        loop" design elsewhere (see _call_with_timeout).
        """
        try:
            self._call_executor.shutdown(wait=wait, cancel_futures=True)
        except TypeError:
            # cancel_futures kwarg added in Python 3.9; degrade gracefully
            # on older runtimes rather than raising.
            self._call_executor.shutdown(wait=wait)
        except Exception:
            pass

    def __del__(self):
        # Best-effort safety net only — explicit shutdown() is preferred.
        try:
            self.shutdown(wait=False)
        except Exception:
            pass

    def _call_with_timeout(self, fn, *args, timeout: float, **kwargs):
        """Run a blocking provider call with a hard wall-clock deadline.

        Returns the call's result, or None if it raised or exceeded
        `timeout` (matching the existing "None means try the next
        fallback" convention used throughout analyze()).

        BUGFIX (audit follow-up): ``future.result(timeout=...)`` only stops
        *waiting* on the future — it does NOT cancel the underlying thread.
        If ``fn`` is mid-retry (e.g. ``_call_groq``'s own 3-attempt loop
        with exponential backoff), that thread keeps running in the
        background after this method returns, silently occupying one of
        the 4 executor workers and still burning real API/rate-limit
        budget on a result nobody will use. Across 6 symbols polled every
        cycle, repeated timeouts could eventually exhaust the pool and
        make *future* calls queue for real (compounding the very problem
        this wrapper exists to prevent).

        A true hard-cancel of an in-flight blocking HTTP call isn't
        possible with ``ThreadPoolExecutor`` alone. This is mitigated two
        ways: (1) callers now pass an absolute ``deadline`` through to the
        provider call so its internal retry loop stops scheduling *new*
        attempts once the caller has already given up (see ``_call_groq``,
        ``_call_gemini``, ``_call_openai_compat``), and (2) each provider
        call now also passes a request-level timeout to the SDK itself
        where supported, so the in-flight socket call is aborted at the
        transport layer rather than relying on this wrapper alone.
        """
        future = self._call_executor.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError:
            log.warning(f"[AIAnalyst] {getattr(fn, '__name__', 'LLM call')} timed out after {timeout}s")
            return None
        except Exception as e:
            log.warning(f"[AIAnalyst] {getattr(fn, '__name__', 'LLM call')} raised: {e}")
            return None

    # ── Public read-only accessors ──────────────────────────────────
    # Added so external callers (main.py status check, health monitor) can
    # introspect which LLM is wired without poking at underscore-prefixed
    # attributes (which previously caused AttributeError in main.py:279).
    @property
    def groq_client(self):
        return self._groq_client

    @property
    def gemini_client(self):
        return self._gemini_client

    @property
    def groq_model(self) -> str:
        return self.GROQ_MODEL

    @property
    def gemini_model(self) -> str:
        return self.GEMINI_MODEL

    @property
    def active_provider(self) -> str:
        """Return 'groq', 'gemini', or 'none' depending on which client is wired."""
        if self._groq_client is not None:
            return "groq"
        if self._gemini_client is not None:
            return "gemini"
        return "none"

    def _init_clients(self):
        """Initialize LLM clients using LLMKeyManager (multi-key rotation)."""
        try:
            from core.llm_key_manager import get_llm_key_manager
            manager = get_llm_key_manager()
            self._key_manager = manager
            self._groq_client = manager.get_groq_client()
            if self._groq_client is not None:
                log.info(f"Groq client initialized | model={self.GROQ_MODEL}")
            self._gemini_client = manager.get_gemini_client()
            if self._gemini_client is not None:
                log.info(f"Gemini client initialized (fallback ready) | model={self.GEMINI_MODEL}")
            if self._groq_client is None and self._gemini_client is None:
                log.warning("No LLM client available (Groq + Gemini both failed)")
        except Exception as e:
            log.warning(f"LLMKeyManager init failed, falling back to single-key: {e}")
            self._key_manager = None
            # Fallback: single-key mode (backwards compat)
            groq_key = os.getenv("GROQ_API_KEY_1") or os.getenv("GROQ_API_KEY", "")
            if groq_key:
                try:
                    from groq import Groq
                    self._groq_client = Groq(api_key=groq_key)
                    log.info(f"Groq client initialized (single-key fallback) | model={self.GROQ_MODEL}")
                except Exception as e2:
                    log.warning(f"Groq init failed: {e2}")
            gemini_key = os.getenv("GEMINI_API_KEY_1") or os.getenv("GEMINI_API_KEY", "")
            if gemini_key:
                try:
                    from google import genai as google_genai
                    self._gemini_client = google_genai.Client(api_key=gemini_key)
                    log.info(f"Gemini client initialized (single-key fallback) | model={self.GEMINI_MODEL}")
                except Exception as e2:
                    log.warning(f"Gemini init failed: {e2}")

    # ── Public method ──────────────────────────────────────────
    def analyze(
        self,
        ind_ctx:    dict,
        pat_ctx:    dict,
        sr_ctx:     dict,
        regime:     dict,
        signal:     dict,
        mtf_bias:   str = "NEUTRAL",
        symbol:     str = "EURUSD",
        advanced_pat_ctx: dict = None,
        timeframe:  str = "15M",
        **kwargs,
    ) -> dict:
        """
        সব technical context নিয়ে LLM analyst এর opinion নেয়।
        Returns structured dict।
        """
        # BUGFIX (audit follow-up): TIMEFRAME was previously hardcoded to
        # "15M" inside _build_context regardless of what timeframe this
        # analyst was actually being run on. The prompt explicitly asks the
        # LLM to weigh session/timing context (rule 6 in _build_prompt) —
        # a silently wrong timeframe label would mislead that judgment if
        # this analyst is ever invoked on a different chart. Default
        # preserves existing behavior; callers can now pass the real value.
        context = self._build_context(
            ind_ctx, pat_ctx, sr_ctx, regime, signal, mtf_bias, symbol, advanced_pat_ctx, timeframe
        )
        prompt  = self._build_prompt(context)

        # ── Day 90 — LLM cache lookup ──────────────────────────
        # Same prompt within 5 min → return cached response.
        # This is the BIGGEST token saver because AIAnalyst gets
        # called once per symbol per cycle, and 6 symbols × 60s loop
        # = lots of redundant calls when market is quiet.
        try:
            from core.llm_cache import get_llm_cache
            _cache = get_llm_cache()
            # BUGFIX: cache key is intentionally provider/model-agnostic.
            # The cache's purpose is "same prompt within 5 min → reuse the
            # answer, whichever backend produced it" (see comment above).
            # Hardcoding "groq"/GROQ_MODEL here mislabeled every cache
            # entry as a Groq response even when Gemini/Cerebras/SambaNova/
            # OpenRouter actually served it (see cache-store call below),
            # which silently broke model-versioning: if GROQ_MODEL was ever
            # changed, stale non-Groq answers cached under the old key
            # would still be returned as if freshly generated.
            _cache_key = _cache.make_key("ai_analyst", "any", prompt)
            _cached = _cache.get(_cache_key)
            if _cached is not None:
                log.debug(f"[AIAnalyst] LLM cache HIT — skipping API call")
                result = self._parse_response(_cached, rule_signal=signal)
                result["_cache_hit"] = True
                return result
        except Exception:
            pass

        # BUGFIX (audit follow-up): analyze() previously had no timeout
        # protection at all — a hung provider call could block the whole
        # trading loop indefinitely (this is on top of wait_for_any_groq's
        # own separately-fixed 5-minute worst case). Every provider call
        # below is now wrapped with a shared wall-clock budget: once the
        # budget is spent, remaining providers are skipped and analyze()
        # falls back to the rule-engine signal rather than blocking.
        total_timeout = float(os.getenv("AI_ANALYST_TOTAL_TIMEOUT_SEC", "20"))
        deadline = time.monotonic() + total_timeout

        def _remaining():
            return max(0.5, deadline - time.monotonic())

        # ── PRIMARY: local Ollama (user's own laptop) ──────────
        # Tried first: no API key, no daily token budget, no rate limit,
        # so it can't be the reason this layer silently degrades to
        # "always excluded" the way a missing/exhausted cloud key can.
        # Falls through to Groq/Gemini/etc. automatically if Ollama isn't
        # running or the model isn't pulled — see _call_ollama().
        raw = None
        if self.OLLAMA_ENABLED and time.monotonic() < deadline:
            raw = self._call_with_timeout(self._call_ollama, prompt, timeout=_remaining(), deadline=deadline)
            if raw is not None:
                log.info(f"[AIAnalyst] Ollama ({self.OLLAMA_MODEL}) served this analysis")

        # Fallback: Groq
        if raw is None and self._groq_client and time.monotonic() < deadline:
            raw = self._call_with_timeout(self._call_groq, prompt, timeout=_remaining(), deadline=deadline)

        # ── PRIMARY FALLBACK: Gemini (moved up — most reliable after Groq) ──
        # Cerebras (Cloudflare 403) and SambaNova (410 Gone) are known broken.
        # Try Gemini BEFORE them to avoid wasted latency.
        # Requires: pip install google-genai
        if raw is None and time.monotonic() < deadline:
            # Try to get a Gemini client from key manager if not pre-initialized
            _gem_client = self._gemini_client
            if _gem_client is None and self._key_manager is not None:
                _gem_client = self._key_manager.get_gemini_client()
            if _gem_client is not None:
                raw = self._call_with_timeout(self._call_gemini, prompt, timeout=_remaining(), deadline=deadline)

        # ── Day 91 — Cerebras / SambaNova / OpenRouter fallback ──
        # All three are OpenAI-compatible; reuse _call_openai_compat
        # helper that takes a client + model + manager hooks.
        if raw is None and self._key_manager is not None and time.monotonic() < deadline:
            # Try Cerebras (currently blocked by Cloudflare on Linux VPS,
            # but harmless to attempt — adds <100ms when key unavailable)
            if self._key_manager.has_any_cerebras:
                raw = self._call_with_timeout(
                    self._call_openai_compat, timeout=_remaining(),
                    provider_name="Cerebras",
                    client_getter=self._key_manager.get_cerebras_client,
                    success_marker=self._key_manager.mark_cerebras_success,
                    failure_marker=self._key_manager.mark_cerebras_failure,
                    model_env="CEREBRAS_MODEL",
                    default_model="llama3.1-8b-instruct",
                    prompt=prompt,
                    deadline=deadline,
                )
        if raw is None and self._key_manager is not None and time.monotonic() < deadline:
            if self._key_manager.has_any_sambanova:
                raw = self._call_with_timeout(
                    self._call_openai_compat, timeout=_remaining(),
                    provider_name="SambaNova",
                    client_getter=self._key_manager.get_sambanova_client,
                    success_marker=self._key_manager.mark_sambanova_success,
                    failure_marker=self._key_manager.mark_sambanova_failure,
                    model_env="SAMBANOVA_MODEL",
                    # Day 99+ V3 FIX (Master List Issue #1): SambaNova
                    # deprecated all Llama 3.1 models (410 Gone). Default
                    # is now DeepSeek-V3 (current on free tier as of
                    # 2026-Q3). Override via SAMBANOVA_MODEL env var.
                    default_model="DeepSeek-V3",
                    prompt=prompt,
                    deadline=deadline,
                )
        if raw is None and self._key_manager is not None and time.monotonic() < deadline:
            if self._key_manager.has_any_openrouter:
                # OpenRouter has multiple free models — try fallback chain
                or_models = [os.getenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free")]
                fb1 = os.getenv("OPENROUTER_MODEL_FALLBACK_1", "")
                fb2 = os.getenv("OPENROUTER_MODEL_FALLBACK_2", "")
                if fb1: or_models.append(fb1)
                if fb2: or_models.append(fb2)
                for or_model in or_models:
                    if time.monotonic() >= deadline:
                        break
                    raw = self._call_with_timeout(
                        self._call_openai_compat, timeout=_remaining(),
                        provider_name=f"OpenRouter({or_model})",
                        client_getter=self._key_manager.get_openrouter_client,
                        success_marker=self._key_manager.mark_openrouter_success,
                        failure_marker=self._key_manager.mark_openrouter_failure,
                        model_env=None,  # use explicit model below
                        default_model=or_model,
                        prompt=prompt,
                        deadline=deadline,
                    )
                    if raw is not None:
                        break

        if raw is None:
            # BUGFIX (audit follow-up): fall back to the rule engine's own
            # signal instead of a hardcoded WAIT/0% — see _fallback_result.
            return self._fallback_result("No LLM available (or all providers timed out)", rule_signal=signal)

        # ── Day 90 — cache store ───────────────────────────────
        # Stored under the provider-agnostic key regardless of which
        # backend produced `raw` — see the lookup-side comment above.
        try:
            _cache.set(_cache_key, raw, token_estimate=400)
        except Exception:
            pass

        result = self._parse_response(raw, rule_signal=signal)
        log.info(
            f"LLM -> Signal: {result.get('signal')} | "
            f"Confidence: {result.get('confidence')}%"
        )
        return result

    # ── Context builder ────────────────────────────────────────
    def _build_context(
        self, ind, pat, sr, regime, signal, mtf_bias, symbol, advanced_pat=None, timeframe="15M"
    ) -> str:
        adv_patterns_str = "None"
        if advanced_pat and isinstance(advanced_pat, dict):
            adv_patterns_str = str(advanced_pat.get('recent_patterns', advanced_pat))

        return f"""
SYMBOL        : {symbol}
TIMEFRAME     : {timeframe}

-- PRICE & TREND --
Close         : {ind.get('close', 'N/A')}
Trend         : {ind.get('trend', 'N/A')}
EMA9          : {ind.get('ema9', 'N/A')}
SMA20         : {ind.get('sma20', 'N/A')}

-- MOMENTUM --
RSI (14)      : {ind.get('rsi', 'N/A')}
MACD Signal   : {ind.get('macd_signal', 'N/A')}
MACD Value    : {ind.get('macd', 'N/A')}

-- VOLATILITY --
ATR           : {ind.get('atr', 'N/A')}
BB Position   : {ind.get('bb_position', 'N/A')}

-- PATTERNS --
Recent        : {pat.get('recent_patterns', [])}
Advanced Pat  : {adv_patterns_str}
Signal        : {pat.get('pattern_signal', 'N/A')}

-- SUPPORT / RESISTANCE --
Location      : {sr.get('location', 'N/A')}
Nearest S     : {sr.get('nearest_support', 'N/A')}
Nearest R     : {sr.get('nearest_resistance', 'N/A')}
Pivot PP      : {sr.get('pivot_pp', 'N/A')}

-- MARKET REGIME --
Regime        : {regime.get('regime', 'N/A')}
Direction     : {regime.get('direction', 'N/A')}
Strength      : {regime.get('strength', 'N/A')}
Volatility    : {regime.get('volatility', 'N/A')}
ADX           : {regime.get('adx', 'N/A')}

-- RULE ENGINE SIGNAL --
Signal        : {signal.get('signal', 'N/A')}
Confidence    : {signal.get('confidence', 0)}%
Entry         : {signal.get('entry', 'N/A')}
Blocked by    : {signal.get('blocked_by', 'None')}
Reasons       : {signal.get('reasons', [])}

-- MULTI-TIMEFRAME --
MTF Bias      : {mtf_bias}
""".strip()

    # ── Prompt ────────────────────────────────────────────────
    def _build_prompt(self, context: str) -> str:
        return f"""You are an elite professional forex trader and market analyst with 20 years of experience.
You specialize in Smart Money Concepts (SMC), institutional order flow, and price action analysis.

Analyze the following market data carefully and provide a structured trade decision.

{context}

ANALYSIS RULES:
1. Combine ALL signals — do not rely on one indicator alone
2. Respect market regime — in strong trends, counter-trend trades are extremely risky
3. If signals conflict, recommend WAIT — capital preservation is paramount
4. Consider confluence — multiple confirming factors increase conviction
5. Always explain WHY — your reasoning must be transparent and verifiable
6. Consider the session context — London/NY overlap has different dynamics than Asian session
7. Evaluate risk/reward — only recommend trades with R:R >= 2:1

OUTPUT FORMAT — Return ONLY valid JSON, no extra text:

{{
  "analysis": "2-3 sentence market summary explaining the current state",
  "signal": "BUY or SELL or WAIT",
  "confidence": 0-100,
  "reasoning": "Detailed explanation: WHY this direction, what confirms it, what are the confluences",
  "key_risk": "The single most important risk that could invalidate this trade",
  "invalidation": "Specific price level or condition that would invalidate this signal",
  "market_condition": "TRENDING_UP or TRENDING_DOWN or RANGING or VOLATILE",
  "risk_warning": "Any additional risk warning for the trader"
}}"""

    # ── Local Ollama (primary — no key, no rate limit) ──────────
    def _get_ollama_client(self):
        """Lazy-init the Ollama client (same pattern as core/ollama_validator.py)."""
        if self._ollama_client is not None:
            return self._ollama_client
        try:
            from ollama import Client
            self._ollama_client = Client(host=self.OLLAMA_HOST)
            return self._ollama_client
        except ImportError:
            log.warning(
                "[AIAnalyst] 'ollama' package not installed — "
                "install with: pip install ollama"
            )
            return None

    def _call_ollama(self, prompt: str, deadline: float | None = None) -> str | None:
        """Call the local Ollama server. Single attempt, no multi-key retry
        needed since there's no key/quota — just a reachability check.

        Fails silently (returns None) if Ollama isn't running or the model
        isn't pulled, so analyze() falls through to the cloud cascade
        exactly as before. This must never raise.
        """
        if not self.OLLAMA_ENABLED:
            return None
        client = self._get_ollama_client()
        if client is None:
            return None
        try:
            remaining = None
            if deadline is not None:
                remaining = max(1.0, deadline - time.monotonic())
            response = client.chat(
                model=self.OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": "You are a senior Forex market analyst. Return ONLY valid JSON, no extra text, no markdown fences, no <think> blocks."},
                    {"role": "user", "content": prompt},
                ],
                options={
                    "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.2")),
                    "num_predict": int(os.getenv("OLLAMA_NUM_PREDICT", "512")),
                },
                # ollama-python's Client.chat doesn't take a wall-clock
                # timeout directly; the shared _call_with_timeout() wrapper
                # in analyze() already bounds this call from the outside.
            )
            raw_content = response.get("message", {}).get("content", "") if isinstance(response, dict) else getattr(response.message, "content", "")
            if not raw_content:
                return None
            self._record_usage("ollama", prompt_tokens=0, completion_tokens=0)  # local — no token cost
            return self._strip_ollama_thinking(raw_content)
        except Exception as e:
            log.info(f"[AIAnalyst] Ollama unreachable/failed ({self.OLLAMA_HOST}, model={self.OLLAMA_MODEL}): {e}")
            return None

    @staticmethod
    def _strip_ollama_thinking(text: str) -> str:
        """Remove <think>...</think> blocks and markdown fences some local
        models (e.g. Qwen3) emit before/around the JSON payload."""
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        cleaned = re.sub(r"^```json\s*", "", cleaned.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned.strip())
        return cleaned.strip()

    # ── LLM callers (multi-key retry) ───────────────────────────
    def _call_groq(self, prompt: str, deadline: float | None = None) -> str | None:
        """Call Groq with multi-key retry. If current key fails, tries next.

        If ALL keys are exhausted (e.g. Groq free-tier TPD hit), waits
        for the soonest-recovering key instead of bailing immediately —
        this prevents the 429 storm + supervisor restart loop seen in
        production logs.

        Day 81+ hotfix: per-cycle throttle caps total LLM calls per
        symbol cycle to MAX_LLM_CALLS_PER_CYCLE (default 5).  Also
        enforces LLM_CALL_INTERVAL_SEC between calls (default 1.0s)
        to prevent the Groq free-tier rate-limit storm.

        Args:
            deadline: absolute ``time.monotonic()`` cutoff, forwarded by
                ``analyze()`` via ``_call_with_timeout``. BUGFIX (audit
                follow-up): this method's own 3-attempt retry loop
                (up to ~7s of backoff sleep alone, plus 3x API latency)
                previously ran with no awareness of the caller's overall
                budget. If the outer wrapper's ``future.result(timeout=...)``
                already gave up, this loop kept running anyway in the
                background thread — see ``_call_with_timeout`` docstring.
                Checking the deadline before each retry stops it from
                scheduling further attempts once the caller has moved on.
        """
        # Per-cycle throttle check
        if hasattr(self, '_key_manager') and self._key_manager:
            allowed, reason = self._key_manager.check_cycle_throttle()
            if not allowed:
                log.info(f"[AIAnalyst] Groq skipped — {reason}")
                return None

        max_retries = 3
        for attempt in range(max_retries):
            if deadline is not None and time.monotonic() >= deadline:
                log.debug("[AIAnalyst] Groq: deadline already elapsed — abandoning remaining retries")
                return None
            client = self._groq_client
            if client is None and hasattr(self, '_key_manager') and self._key_manager:
                client = self._key_manager.get_groq_client()
            if client is None and hasattr(self, '_key_manager') and self._key_manager:
                # Round-16 audit fix: NON-BLOCKING check instead of blocking wait.
                #
                # Previously: called wait_for_any_groq(max_wait=15) which
                # BLOCKED for up to 15 seconds waiting for a key to recover.
                # Since analyze()'s total timeout budget is only 20s
                # (AI_ANALYST_TOTAL_TIMEOUT_SEC), this wait consumed the
                # entire budget — by the time _call_groq() returned None,
                # the deadline had passed and all fallback providers
                # (Cerebras/SambaNova/OpenRouter/Gemini) were skipped.
                #
                # The operator's audit confirmed this: "Groq exhaust হলেও
                # Gemini fallback call হচ্ছে না" — the code never reached
                # the Gemini fallback because the deadline was already gone.
                #
                # Now: use a NON-BLOCKING has_any_groq check. If no keys
                # are available RIGHT NOW, immediately return None and let
                # the fallback chain (Cerebras → SambaNova → OpenRouter →
                # Gemini) take over. Groq will naturally recover in the
                # background and be tried again on the next cycle.
                if not self._key_manager.has_any_groq:
                    log.info(
                        "[AIAnalyst] All Groq keys exhausted — skipping to "
                        "fallback providers (non-blocking)"
                    )
                    return None
            if client is None:
                log.warning("No Groq client available — falling back")
                return None
            try:
                create_kwargs = dict(
                    model=self.GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    # Day 90 — token economy: 600→400 (saves ~33% per call).
                    # The classic AIAnalyst's response is much shorter than
                    # MasterAnalyst's (just signal + reasoning, no full plan).
                    #
                    # Day 137 safety fix (real-money loss postmortem — GBPCAD
                    # 2026-07-20): 400 tokens was too aggressive — production
                    # logs showed repeated `[llm_json] could not decode JSON
                    # object ... Unterminated string` errors because the
                    # model's free-text "analysis" field routinely ran past
                    # 400 tokens before the JSON could close. Every truncated
                    # response silently degrades to "Signal: WAIT | Confidence:
                    # 30%" (see _parse_response's parse-failure fallback) which
                    # then gets excluded from voting — i.e. this budget being
                    # too tight was quietly throwing away real analysis on a
                    # meaningful fraction of cycles, not just wasting tokens.
                    # 700 gives headroom for a full analysis string without
                    # materially undoing the Day 90 cost savings.
                    max_tokens=int(os.getenv("AI_ANALYST_MAX_TOKENS", "700")),
                )
                if deadline is not None:
                    # BUGFIX (audit follow-up): request-level timeout so a
                    # hung/slow socket call is actually aborted by the SDK's
                    # transport layer, instead of relying solely on the
                    # outer ThreadPoolExecutor wrapper (which can only stop
                    # *waiting*, not cancel the in-flight call — see
                    # _call_with_timeout). The groq SDK (httpx-based) mirrors
                    # the OpenAI client's per-call `timeout` kwarg; guarded
                    # with a fallback in case an older SDK version doesn't
                    # accept it.
                    create_kwargs["timeout"] = max(0.5, deadline - time.monotonic())
                try:
                    resp = client.chat.completions.create(**create_kwargs)
                except TypeError:
                    create_kwargs.pop("timeout", None)
                    resp = client.chat.completions.create(**create_kwargs)
                # Success — mark key as healthy and record usage (single
                # lookup, reused for both — was computed twice before).
                usage = getattr(resp, "usage", None)
                prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
                if hasattr(self, '_key_manager') and self._key_manager:
                    self._key_manager.mark_groq_success(
                        tokens_used=(prompt_tokens or 0) + (completion_tokens or 0), client=client
                    )
                self._record_usage("groq", prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
                return resp.choices[0].message.content
            except Exception as e:
                from core.llm_key_manager import log_llm_call_failure
                info = log_llm_call_failure(
                    log, "Groq", self.GROQ_MODEL, attempt, max_retries, e
                )
                if hasattr(self, '_key_manager') and self._key_manager:
                    self._key_manager.mark_groq_failure(
                        info["error_str"], info["rate_limited"], client=client
                    )
                    # Get a fresh client with a different key
                    self._groq_client = self._key_manager.get_groq_client()
                if attempt < max_retries - 1:
                    # BUGFIX (audit follow-up): flat 1s sleep between
                    # retries hits a rate-limited endpoint at a fixed
                    # cadence, which is exactly the pattern that keeps
                    # tripping the same limit. Exponential backoff with
                    # jitter (1s, 2s, 4s, ± up to 250ms) spreads retries
                    # out and backs off harder specifically on 429s.
                    base_delay = 2 ** attempt
                    if info.get("rate_limited"):
                        base_delay *= 2
                    delay = base_delay + random.uniform(0, 0.25)
                    if deadline is not None:
                        delay = min(delay, max(0.0, deadline - time.monotonic()))
                    if delay > 0:
                        time.sleep(delay)
        return None

    # ── Day 91 — OpenAI-compatible fallback (Cerebras / SambaNova / OpenRouter)
    def _call_openai_compat(
        self,
        *,
        provider_name: str,
        client_getter,
        success_marker,
        failure_marker,
        model_env: str | None,
        default_model: str,
        prompt: str,
        deadline: float | None = None,
    ) -> str | None:
        """Generic OpenAI-compatible chat completion call.

        All three new providers (Cerebras, SambaNova, OpenRouter) expose
        the same /v1/chat/completions endpoint with .chat.completions.
        create() surface. This helper avoids duplicating the call+retry
        boilerplate across three near-identical blocks.

        Args:
            deadline: see ``_call_groq`` — bounds retries/backoff to the
                caller's remaining budget (audit follow-up).

        Returns:
            str response text on success, None on failure (caller should
            try the next fallback).
        """
        # BUGFIX (audit follow-up): this previously made exactly one
        # attempt — any transient error, and especially a 429 rate-limit
        # (which these free/low-tier fallback providers hit often), meant
        # an immediate fall-through to the next provider with no retry at
        # all. Add the same bounded retry + exponential-backoff-with-
        # jitter pattern used for Groq/Gemini, specifically extending the
        # delay on detected rate limits.
        # Day 137 safety fix: 400 truncated JSON responses mid-string on a
        # regular basis (see matching comment at the Groq call site above) —
        # raised to 700 for the fallback providers too, for consistency.
        max_tokens = int(os.getenv("AI_ANALYST_MAX_TOKENS", "700"))
        max_retries = 2
        model = default_model
        if model_env:
            model = os.getenv(model_env, default_model)

        for attempt in range(max_retries):
            if deadline is not None and time.monotonic() >= deadline:
                log.debug(f"[AIAnalyst] {provider_name}: deadline already elapsed — abandoning remaining retries")
                return None
            try:
                client = client_getter()
                if client is None:
                    return None
                create_kwargs = dict(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=max_tokens,
                )
                if deadline is not None:
                    create_kwargs["timeout"] = max(0.5, deadline - time.monotonic())
                try:
                    resp = client.chat.completions.create(**create_kwargs)
                except TypeError:
                    create_kwargs.pop("timeout", None)
                    resp = client.chat.completions.create(**create_kwargs)
                success_marker()
                usage = getattr(resp, "usage", None)
                self._record_usage(
                    provider_name.split("(")[0].strip().lower(),
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
                )
                text = resp.choices[0].message.content
                log.info(f"[AIAnalyst] {provider_name} OK | model={model}")
                return text.strip() if text else None
            except Exception as e:
                from core.llm_key_manager import log_llm_call_failure
                info = log_llm_call_failure(
                    log, provider_name, model, attempt, max_retries, e
                )
                failure_marker(info["error_str"], info["rate_limited"])
                if attempt < max_retries - 1:
                    base_delay = 2 ** attempt
                    if info.get("rate_limited"):
                        base_delay *= 2
                    delay = base_delay + random.uniform(0, 0.25)
                    if deadline is not None:
                        delay = min(delay, max(0.0, deadline - time.monotonic()))
                    if delay > 0:
                        time.sleep(delay)
        return None

    def _call_gemini(self, prompt: str, deadline: float | None = None) -> str | None:
        """Call Gemini with multi-key retry.

        Args:
            deadline: see ``_call_groq`` — bounds retries/backoff to the
                caller's remaining budget (audit follow-up).
        """
        max_retries = 3
        for attempt in range(max_retries):
            if deadline is not None and time.monotonic() >= deadline:
                log.debug("[AIAnalyst] Gemini: deadline already elapsed — abandoning remaining retries")
                return None
            client = self._gemini_client
            if client is None and hasattr(self, '_key_manager') and self._key_manager:
                client = self._key_manager.get_gemini_client()
            if client is None:
                log.warning("No Gemini client available")
                return None
            try:
                generate_kwargs = dict(model=self.GEMINI_MODEL, contents=prompt)
                if deadline is not None:
                    # Best-effort request-level timeout (audit follow-up —
                    # same rationale as _call_groq). google-genai's
                    # HttpOptions.timeout is milliseconds; wrapped in a
                    # broad except since SDK versions vary and this must
                    # never break a call that would otherwise succeed.
                    try:
                        from google.genai import types as _genai_types
                        remaining_ms = max(500, int((deadline - time.monotonic()) * 1000))
                        generate_kwargs["config"] = _genai_types.GenerateContentConfig(
                            http_options=_genai_types.HttpOptions(timeout=remaining_ms)
                        )
                    except Exception:
                        pass
                resp = client.models.generate_content(**generate_kwargs)
                if hasattr(self, '_key_manager') and self._key_manager:
                    usage = getattr(resp, "usage_metadata", None)
                    total_tokens = 0
                    if usage is not None:
                        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
                        completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
                        total_tokens = prompt_tokens + completion_tokens
                    self._key_manager.mark_gemini_success(tokens_used=total_tokens, client=client)
                usage = getattr(resp, "usage_metadata", None)
                self._record_usage(
                    "gemini",
                    prompt_tokens=getattr(usage, "prompt_token_count", 0) if usage else 0,
                    completion_tokens=getattr(usage, "candidates_token_count", 0) if usage else 0,
                )
                return resp.text
            except Exception as e:
                from core.llm_key_manager import log_llm_call_failure
                info = log_llm_call_failure(
                    log, "Gemini", self.GEMINI_MODEL, attempt, max_retries, e
                )
                if hasattr(self, '_key_manager') and self._key_manager:
                    self._key_manager.mark_gemini_failure(
                        info["error_str"], info["rate_limited"], client=client
                    )
                    self._gemini_client = self._key_manager.get_gemini_client()
                if attempt < max_retries - 1:
                    # BUGFIX (audit follow-up): exponential backoff with
                    # jitter instead of a flat 1s sleep — see _call_groq
                    # for the same fix and rationale.
                    base_delay = 2 ** attempt
                    if info.get("rate_limited"):
                        base_delay *= 2
                    delay = base_delay + random.uniform(0, 0.25)
                    if deadline is not None:
                        delay = min(delay, max(0.0, deadline - time.monotonic()))
                    if delay > 0:
                        time.sleep(delay)
        return None

    # ── Response parser ────────────────────────────────────────
    def _parse_response(self, raw: str, rule_signal: dict = None) -> dict:
        """Parse the LLM's JSON response.

        Args:
            rule_signal: the rule engine's own signal dict, passed through
                from analyze(). BUGFIX (audit follow-up): a malformed/
                non-JSON LLM response previously fell back to a hardcoded
                WAIT/0%-confidence result, which silently discarded
                whatever the rule engine had already decided — the one
                system that *did* produce a valid signal this cycle. Now
                falls back to the rule engine's own signal/confidence
                (still clearly labeled as an LLM-parse-failure, not an
                LLM opinion) so a JSON hiccup doesn't quietly override a
                good rule-engine call with a forced WAIT.

        Day 99+ FIX (Issue #1): delegate to utils.llm_json.parse_llm_json
        for robust parsing. The previous inline `raw.find("{")` worked
        only when the JSON object started on its own with no leading
        Markdown fence. If the LLM wrapped its response in
        ```json\n{...}\n``` (which Cerebras, Groq, and OpenRouter all
        do frequently), the '{' of the fence's language tag was found
        first, raw_decode tried to parse starting from there, and
        raised JSONDecodeError — silently discarding a perfectly valid
        JSON payload that was just wrapped in Markdown. The shared
        helper strips the fence first, then extracts the JSON.
        """
        from utils.llm_json import parse_llm_json

        try:
            parsed = parse_llm_json(raw)
            if isinstance(parsed, dict):
                # BUGFIX (audit follow-up): confidence used to be parsed
                # inline with `int(parsed.get("confidence", 0))` inside this
                # same try block. If the LLM returned a non-numeric value
                # for just that one field (e.g. "high", null, "N/A" —
                # observed from lower-tier fallback models under load),
                # the raised ValueError/TypeError was caught by the except
                # below and discarded the ENTIRE otherwise-valid response —
                # a good analysis/reasoning/signal thrown away because of
                # one malformed field. Parse confidence defensively so a
                # bad value degrades to 0 instead of nuking the whole parse.
                raw_confidence = parsed.get("confidence", 0)
                try:
                    confidence = min(99, max(0, int(raw_confidence)))
                except (TypeError, ValueError):
                    log.debug(f"[AIAnalyst] Non-numeric confidence field ({raw_confidence!r}) — defaulting to 0")
                    confidence = 0
                return {
                    "analysis":         parsed.get("analysis", "No analysis provided"),
                    "signal":           parsed.get("signal", "WAIT"),
                    "confidence":       confidence,
                    "reasoning":        parsed.get("reasoning", ""),
                    "key_risk":         parsed.get("key_risk", "Unknown"),
                    "invalidation":     parsed.get("invalidation", "Unknown"),
                    "market_condition": parsed.get("market_condition", "UNKNOWN"),
                    "risk_warning":     parsed.get("risk_warning", ""),
                }
        except (json.JSONDecodeError, AttributeError, ValueError, TypeError):
            pass

        log.warning("Could not parse LLM JSON — deferring to rule-engine signal")
        rule_signal = rule_signal or {}
        return {
            "analysis":         raw[:200] if raw else "Parse error",
            "signal":           rule_signal.get("signal", "WAIT"),
            "confidence":       rule_signal.get("confidence", 0),
            "reasoning":        "LLM JSON parse failed — deferred to rule engine signal",
            "key_risk":         "Unknown",
            "invalidation":     "Unknown",
            "market_condition": "UNKNOWN",
            "risk_warning":     "LLM response could not be parsed; rule-engine signal used instead",
            "_llm_parse_failed": True,
        }

    def _fallback_result(self, reason: str, rule_signal: dict = None) -> dict:
        """Used when no LLM provider is reachable at all.

        BUGFIX (audit follow-up): previously always returned WAIT/0%
        regardless of what the rule engine decided, meaning any provider
        outage forced every symbol to WAIT even when the rule engine had
        a confident, valid signal. Defer to the rule engine's own signal
        instead, clearly flagged as LLM-unavailable.
        """
        rule_signal = rule_signal or {}
        return {
            "analysis":         reason,
            "signal":           rule_signal.get("signal", "WAIT"),
            "confidence":       rule_signal.get("confidence", 0),
            "reasoning":        "LLM unavailable — using rule engine signal",
            "key_risk":         "N/A",
            "invalidation":     "N/A",
            "market_condition": "UNKNOWN",
            "risk_warning":     "AI analysis unavailable; rule-engine signal used instead",
            "_llm_unavailable": True,
        }

    # ── Print ──────────────────────────────────────────────────
    def print_summary(self, result: dict) -> None:
        icons = {"BUY": "[BUY]", "SELL": "[SELL]", "WAIT": "[WAIT]"}
        icon  = icons.get(result.get("signal", "WAIT"), "[WAIT]")
        bar   = "=" * 44

        log.info(bar)
        log.info(f"   {icon}  LLM ANALYST REPORT")
        log.info(bar)
        log.info(f"   Signal      : {result.get('signal')}")
        log.info(f"   Confidence  : {result.get('confidence')}%")
        log.info(f"   Analysis    : {result.get('analysis', '')[:80]}")
        log.info(f"   Reasoning   : {result.get('reasoning', '')[:100]}")
        log.info(f"   Key risk    : {result.get('key_risk', '')}")
        log.info(f"   Invalidation: {result.get('invalidation', '')}")
        log.info(bar)

    def get_ai_context(self, result: dict) -> dict:
        return {
            "llm_signal":           result.get("signal", "WAIT"),
            "llm_confidence":       result.get("confidence", 0),
            "llm_analysis":         result.get("analysis", ""),
            "llm_reasoning":        result.get("reasoning", ""),
            "llm_key_risk":         result.get("key_risk", ""),
            "llm_market_condition": result.get("market_condition", "UNKNOWN"),
            "llm_risk_warning":     result.get("risk_warning", ""),
        }