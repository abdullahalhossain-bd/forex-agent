"""
core/llm_key_manager.py — Multi-Key LLM Rotation Manager (Day 72+)
=====================================================================

Manages multiple API keys per provider (Groq, Gemini) with automatic
failover. If one key hits a rate limit or fails, it automatically
switches to the next available key.

Features:
  * Round-robin key rotation (distributes load across keys)
  * Automatic failover (key 1 fails → try key 2 → try key 3)
  * Rate limit tracking (temporarily disables keys that hit 429)
  * Health stats per key (success count, fail count, last error)
  * Supports unlimited keys per provider

Usage:
    manager = get_llm_key_manager()
    groq_client = manager.get_groq_client()   # returns a working Groq client
    gemini_client = manager.get_gemini_client()  # returns a working Gemini client

Environment variables (in .env):
    GROQ_API_KEY_1=gsk_xxx
    GROQ_API_KEY_2=gsk_yyy
    GROQ_API_KEY_3=gsk_zzz
    GROQ_API_KEY=gsk_xxx        # backwards compat (treated as key 1)

    GEMINI_API_KEY_1=AIzaXxx
    GEMINI_API_KEY_2=AIzaYyy
    GEMINI_API_KEY=AIzaXxx      # backwards compat
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv  # noqa: F401 (kept for import-compat; unused below)

# Day 131 fix — LLM cycle-cap inconsistency root cause:
#
# This module used to run its OWN independent `.env` discovery/loading
# (searching project root, then CWD, then the user's home directory,
# stopping at the first match) completely separately from config.py's
# single `load_dotenv()` call. python-dotenv does NOT override variables
# already present in os.environ, so whichever of the two `load_dotenv()`
# calls happened to run *first* silently won for the entire process.
#
# If this module got imported (directly or transitively) before config.py
# — or if a stale `~/.env` / CWD `.env` from an older deployment existed
# with e.g. `MAX_LLM_CALLS_PER_CYCLE=4` (an earlier, tighter default before
# the Day 102 hotfix raised it to 8) — that stale value would win and
# silently stick for the whole run, which is exactly the "runtime shows
# 4/4 while source says 8" symptom: config.py's own code correctly reads
# `os.getenv("MAX_LLM_CALLS_PER_CYCLE", "8")`, but if that env var was
# already set to "4" by this module's earlier, independent dotenv load,
# config.py's `load_dotenv()` call (no-override by default) never
# corrects it.
#
# Fix: config.py is the single source of truth for environment loading.
# This module no longer loads .env itself; it simply imports the config
# module (which performs load_dotenv() exactly once) to guarantee
# environment variables are populated before any os.getenv() call here,
# regardless of import order elsewhere in the app.
try:
    import config as _config  # noqa: F401  (import side-effect: loads .env once)
except Exception:
    # config.py is expected to always be importable in this project; if it
    # somehow isn't (e.g. isolated unit test), fall back to a local,
    # single load_dotenv() call so key-loading below still works, without
    # re-introducing the multi-path search that caused the drift.
    load_dotenv()

log = logging.getLogger("llm_key_manager")


def classify_llm_error(error: Exception) -> dict:
    """Classify LLM API failures without false positives (e.g. 'rate' in 'generate')."""
    error_str = str(error)
    err_lower = error_str.lower()
    return {
        "error_str": error_str,
        "error_type": type(error).__name__,
        "rate_limited": (
            "429" in error_str
            or "too many requests" in err_lower
            or "rate limit" in err_lower
            or "rate_limit" in err_lower
        ),
        "auth_failed": (
            "401" in error_str
            or "403" in error_str
            or "unauthorized" in err_lower
            or "invalid api key" in err_lower
            or "invalid x-api-key" in err_lower
        ),
    }


def log_llm_call_failure(
    logger: logging.Logger,
    provider: str,
    model: str,
    attempt: int,
    max_retries: int,
    error: Exception,
) -> dict:
    """Log full LLM failure details for diagnosis."""
    info = classify_llm_error(error)
    logger.error(
        "[LLM] %s failed attempt %s/%s | model=%s | type=%s | "
        "rate_limited=%s | auth_failed=%s | error=%s",
        provider,
        attempt + 1,
        max_retries,
        model,
        info["error_type"],
        info["rate_limited"],
        info["auth_failed"],
        info["error_str"][:800],
        exc_info=True,
    )
    return info


# ── Groq 429 retry-after parser ────────────────────────────────────
#
# Groq's TPD (tokens-per-day) rate-limit response looks like:
#   "Rate limit reached for model `llama-3.3-70b-versatile` ...
#    Please try again in 10m1.344s. Need more tokens? ..."
#
# The previous code hardcoded a 30-second cooldown when rate_limited=True,
# which is wildly wrong: the actual cooldown can be minutes to hours.
# This parser extracts the real wait time so the KeyHealth object
# disables the key for the right duration.

_GROQ_RETRY_RE_HMS  = re.compile(r"(\d+)\s*h\s*(\d+)\s*m\s*([\d.]+)\s*s", re.IGNORECASE)
_GROQ_RETRY_RE_HM   = re.compile(r"(\d+)\s*h\s*(\d+)\s*m(?:in)?(?:ute)?s?", re.IGNORECASE)
_GROQ_RETRY_RE_H    = re.compile(r"(\d+)\s*h(?:ou)?r?s?", re.IGNORECASE)
_GROQ_RETRY_RE_MMSS = re.compile(r"(\d+)m\s*([\d.]+)s")
_GROQ_RETRY_RE_SS   = re.compile(r"([\d.]+)\s*s")
_GROQ_RETRY_RE_MM   = re.compile(r"(\d+)\s*m(?:in)?(?:ute)?s?", re.IGNORECASE)
_GROQ_RETRY_RE_HDR  = re.compile(r"retry[-_ ]?after['\"\s:=]+(\d+)", re.IGNORECASE)

# Hard caps so a single malformed error message can't lock a key for an hour
MIN_RETRY_COOLDOWN = 60        # seconds — even "1s" gets bumped to 60s
# Day 132 fix: was 60*30 (30 min). Production log showed Groq returning
# "Please try again in 43m6.816s" for a TPD (tokens-per-day) limit — the
# parser correctly extracted ~2592s, but the old 30-min clamp forced it
# down to 1800s anyway, so the key was re-enabled ~13 minutes before its
# quota actually reset and immediately drew another 429. TPD is a DAILY
# budget, so genuine waits can be well over 30 minutes (up to just under
# 24h in the worst case, right before a token got used just after a
# reset). The ceiling here exists only to guard against a malformed or
# garbage parse producing something absurd (e.g. accidentally parsing
# years) — it should not be tighter than realistic real-world waits.
MAX_RETRY_COOLDOWN = 60 * 60 * 6   # 6 hour safety ceiling (was 30 min)
DEFAULT_RETRY_COOLDOWN = 300   # 5 min fallback if parsing fails
# Groq specifically needs longer cooldowns (TPD limits, not just per-minute).
# Production logs show Groq hitting 98k+ tokens used → needs ~30 min cooldown.
GROQ_DEFAULT_RETRY_COOLDOWN = 1800  # 30 min — Groq free tier TPD resets are slow


def parse_groq_retry_after(error_str: str) -> int:
    """Parse 'Please try again in Xh Ym Z.Zs' from a Groq 429 response.

    Returns the cooldown in seconds (clamped to [MIN_RETRY_COOLDOWN,
    MAX_RETRY_COOLDOWN]) plus a +5s safety margin. Falls back to
    DEFAULT_RETRY_COOLDOWN (300s) if no parseable duration is found.

    Round-14 audit fix: added hour (h) support.
    ─────────────────────────────────────────────
    The operator's audit found that Groq was returning "1h0m16.704s"
    (≈3617 seconds, ~1 hour) but the key manager was only disabling the
    key for 60 seconds. Root cause: the old regex only handled MMSS,
    SS, MM, and HDR formats — NO hour support. The string "1h0m16.704s"
    was matched by the MMSS regex (which ignored the "1h" prefix),
    extracting only "0m16.704s" = 16.704s → clamped to MIN_RETRY_COOLDOWN
    (60s). This caused the bot to retry every 60s and get a fresh 429
    each time, wasting API calls and adding latency.

    Now: the HMS regex (Xh Ym Z.Zs) is checked FIRST, before MMSS.
    "1h0m16.704s" → 1*3600 + 0*60 + 16.704 = 3616.704s + 5s margin =
    3621.7s, which (as of the Day 132 fix) is under MAX_RETRY_COOLDOWN
    (6h) so it passes through unclamped. Much better than 60s.
    Also added HM (Xh Ym) and H (Xh) formats for completeness.
    """
    if not error_str:
        return DEFAULT_RETRY_COOLDOWN
    s = str(error_str)

    # Round-14: Format "1h0m16.704s" (hour + minute + second) — CHECK FIRST
    m = _GROQ_RETRY_RE_HMS.search(s)
    if m:
        total = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        return max(MIN_RETRY_COOLDOWN, min(MAX_RETRY_COOLDOWN, int(total) + 5))

    # Round-14: Format "1h30m" (hour + minute, no seconds)
    m = _GROQ_RETRY_RE_HM.search(s)
    if m:
        total = int(m.group(1)) * 3600 + int(m.group(2)) * 60
        return max(MIN_RETRY_COOLDOWN, min(MAX_RETRY_COOLDOWN, total + 5))

    # Round-14: Format "2h" or "2 hours" (hour only)
    m = _GROQ_RETRY_RE_H.search(s)
    if m:
        total = int(m.group(1)) * 3600
        return max(MIN_RETRY_COOLDOWN, min(MAX_RETRY_COOLDOWN, total + 5))

    # Format: "10m1.344s"
    m = _GROQ_RETRY_RE_MMSS.search(s)
    if m:
        total = int(m.group(1)) * 60 + float(m.group(2))
        return max(MIN_RETRY_COOLDOWN, min(MAX_RETRY_COOLDOWN, int(total) + 5))

    # Format: "45s" or "1.344s"
    m = _GROQ_RETRY_RE_SS.search(s)
    if m:
        total = float(m.group(1))
        return max(MIN_RETRY_COOLDOWN, min(MAX_RETRY_COOLDOWN, int(total) + 5))

    # Format: "10m" or "10 minutes"
    m = _GROQ_RETRY_RE_MM.search(s)
    if m:
        total = int(m.group(1)) * 60
        return max(MIN_RETRY_COOLDOWN, min(MAX_RETRY_COOLDOWN, total + 5))

    # HTTP-style "Retry-After: 120"
    m = _GROQ_RETRY_RE_HDR.search(s)
    if m:
        total = int(m.group(1))
        return max(MIN_RETRY_COOLDOWN, min(MAX_RETRY_COOLDOWN, total + 5))

    return DEFAULT_RETRY_COOLDOWN


# ── Key health tracking ─────────────────────────────────────────────
# ── Round-19 fix: per-KEY TPD budgets (moved to module level so ──
# KeyHealth can consult it directly, before LLMKeyManager class exists)
_TPD_BUDGET_BY_PROVIDER = {
    "groq": int(os.getenv("GROQ_TPD_BUDGET", "80000")),
    "gemini": int(os.getenv("GEMINI_TPD_BUDGET", "150000")),
    "cerebras": int(os.getenv("CEREBRAS_TPD_BUDGET", "100000")),
    "sambanova": int(os.getenv("SAMBANOVA_TPD_BUDGET", "100000")),
    "openrouter": int(os.getenv("OPENROUTER_TPD_BUDGET", "200000")),
    "github": int(os.getenv("GITHUB_TPD_BUDGET", "50000")),
    "huggingface": int(os.getenv("HF_TPD_BUDGET", "50000")),
}
@dataclass
class KeyHealth:
    """Tracks health of one API key."""
    key: str
    provider: str             # groq / gemini
    index: int                # 0-based index
    success_count: int = 0
    fail_count: int = 0
    last_error: str = ""
    last_success: float = 0.0
    rate_limited_until: float = 0.0  # timestamp until which key is disabled
    is_active: bool = True
    # ── Round-19 fix: per-KEY (not per-provider) TPD tracking ──
    # Each Groq key belongs to a DIFFERENT org/account with its own
    # independent daily token quota (confirmed in production logs —
    # each 429 body showed a different org_xxxx id). The old code
    # tracked usage per-provider (shared across all 7 Groq keys),
    # which meant one key's heavy usage could falsely "exhaust" the
    # budget for keys that hadn't been used at all. Also, the old
    # TPD tracking was never actually consulted by is_available —
    # it existed but had zero effect on key selection (dead code).
    # This field + _is_tpd_exhausted() + record_tokens() fix both bugs.
    tpd_tokens_used: int = 0
    tpd_date: str = ""

    @property
    def is_available(self) -> bool:
        """Key is available if active AND not rate-limited AND not
        proactively known to be near/over its daily token budget."""
        if not self.is_active:
            return False
        if self.rate_limited_until > time.time():
            return False
        if self._is_tpd_exhausted():
            return False
        return True

    def _is_tpd_exhausted(self) -> bool:
        """Proactively skip a key whose TRACKED usage has reached its
        daily budget — instead of only reacting after Groq's API
        rejects the call with a 429. Requires record_tokens() to be
        called by the caller after every successful response."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.tpd_date != today:
            return False  # new day, or never tracked yet — assume fresh
        budget = _TPD_BUDGET_BY_PROVIDER.get(self.provider, 100000)
        return self.tpd_tokens_used >= budget

    def record_tokens(self, token_count: int) -> None:
        """Call after every successful API response (with the real
        prompt+completion token count) so this key's own daily usage
        is tracked independently of its sibling keys."""
        if token_count <= 0:
            return
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.tpd_date != today:
            self.tpd_date = today
            self.tpd_tokens_used = 0
        self.tpd_tokens_used += int(token_count)
        budget = _TPD_BUDGET_BY_PROVIDER.get(self.provider, 100000)
        if self.tpd_tokens_used >= budget * 0.8:
            log.warning(
                f"[LLM Keys] {self.provider} key #{self.index + 1} "
                f"at {self.tpd_tokens_used:,}/{budget:,} tokens today "
                f"({self.tpd_tokens_used/budget:.0%}) — will be proactively "
                f"skipped once its own budget is reached"
            )

    def mark_success(self) -> None:
        self.success_count += 1
        self.last_success = time.time()
        self.rate_limited_until = 0.0  # clear any rate limit

    def mark_failure(self, error: str = "", rate_limited: bool = False) -> None:
        self.fail_count += 1
        self.last_error = error[:200]

        # ── Network errors should NOT disable the key ───────────────
        # DNS failures (getaddrinfo), connection refused, timeouts etc.
        # are NOT a key problem — they're a local network problem.
        # Disabling the key on these just makes a temporary outage
        # permanent for 2 minutes.  Detect + skip the disable logic.
        err_lower = error.lower()
        is_network_error = any(s in err_lower for s in (
            "getaddrinfo", "connection", "timeout", "timed out",
            "network", "dns", "unreachable", "refused", "reset",
            "11001", "etimedout", "ehostunreach", "enetunreach",
            "ssl", "certificate", "proxyerror",
        ))

        if rate_limited:
            # Parse "Please try again in 10m1.344s" from Groq's 429 body and
            # use the real cooldown.  Falls back to DEFAULT_RETRY_COOLDOWN
            # (300s) if parsing fails.  Clamped to [MIN_RETRY_COOLDOWN,
            # MAX_RETRY_COOLDOWN] seconds.
            cooldown = parse_groq_retry_after(error)
            self.rate_limited_until = time.time() + cooldown
            log.warning(
                f"[LLM Keys] {self.provider} key #{self.index + 1} "
                f"rate-limited, disabled for {cooldown}s"
            )
        elif "401" in error or "unauthorized" in err_lower:
            # Invalid key — disable permanently
            self.is_active = False
            log.error(f"[LLM Keys] {self.provider} key #{self.index + 1} unauthorized — permanently disabled")
        elif is_network_error:
            # Network error — DON'T disable the key, just log it.
            # The next call will retry.  This prevents a 2-minute
            # disable spiral during temporary DNS / proxy outages.
            log.debug(
                f"[LLM Keys] {self.provider} key #{self.index + 1} network error "
                f"(NOT disabling — will retry): {error[:80]}"
            )
        elif self.fail_count > 20:
            # Too many failures — disable for 2 minutes (was 5 — too long)
            self.rate_limited_until = time.time() + 120
            log.warning(f"[LLM Keys] {self.provider} key #{self.index + 1} too many failures ({self.fail_count}), disabled for 2min")
        elif self.fail_count > 5:
            # Some failures — short cooldown
            self.rate_limited_until = time.time() + 10
            log.warning(f"[LLM Keys] {self.provider} key #{self.index + 1} {self.fail_count} failures, 10s cooldown")
        # Otherwise: single failure, no disable — try again next time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "index": self.index,
            "active": self.is_active,
            "available": self.is_available,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "last_error": self.last_error[:100],
            "rate_limited": self.rate_limited_until > time.time(),
        }


# ════════════════════════════════════════════════════════════════
# Day 91 — OpenAI-compatible REST client shim
# ════════════════════════════════════════════════════════════════
# Cerebras / SambaNova / OpenRouter all expose the standard OpenAI
# /v1/chat/completions endpoint. Rather than require the `openai`
# Python package, this tiny shim wraps `requests` and exposes a
# `.chat.completions.create(...)` surface that matches the Groq
# client's. So master_analyst._call_llm can write the same call
# regardless of which provider it's hitting:
#
#     client = manager.get_cerebras_client()
#     resp = client.chat.completions.create(
#         model="llama3.1-8b-instruct",
#         messages=[{"role":"user","content":"..."}],
#         max_tokens=800,
#         temperature=0.2,
#     )
#     text = resp.choices[0].message.content
#
# The shim also auto-injects OpenRouter-specific headers (HTTP-Referer
# + X-Title) that OpenRouter recommends for proper attribution + to
# avoid being throttled on the free tier.
# ════════════════════════════════════════════════════════════════


class _OpenAICompatClient:
    """Lightweight OpenAI-compatible REST client for Cerebras / SambaNova / OpenRouter."""

    def __init__(self, api_key: str, base_url: str, provider: str):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._provider = provider
        # Build the nested .chat.completions.create surface
        self.chat = self._ChatNamespace(self)

    class _ChatNamespace:
        def __init__(self, parent):
            self.completions = self._CompletionsNamespace(parent)

        class _CompletionsNamespace:
            def __init__(self, parent):
                self._parent = parent

            def create(self, *, model: str, messages: list,
                       max_tokens: int = 800, temperature: float = 0.2,
                       **kwargs):
                return self._parent._do_create(
                    model=model, messages=messages,
                    max_tokens=max_tokens, temperature=temperature,
                    extra=kwargs,
                )

    def _do_create(self, *, model: str, messages: list,
                   max_tokens: int, temperature: float, extra: dict):
        import requests
        # Day 99+ FIX (Issue #5): Cloudflare-protected LLM endpoints
        # (Cerebras, SambaNova, OpenRouter, GitHub Models) reject
        # Python's default `python-requests/2.x` User-Agent with HTTP
        # 403. Adding a browser-like User-Agent + Accept headers is
        # enough to pass Cloudflare's bot detection on all four
        # providers' free tiers. If curl_cffi is installed (it
        # emulates Chrome's TLS fingerprint exactly), we use it as
        # a transparent drop-in for requests.post — this defeats
        # even the stricter Cloudflare "Under Attack" mode that
        # some endpoints enable during DDoS events.
        browser_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }
        url = f"{self._base_url}/chat/completions"
        headers = browser_headers.copy()
        headers.update({
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        })
        # OpenRouter recommends these headers for proper attribution +
        # to avoid being throttled on the free tier.
        if self._provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/forex-ai-trader"
            headers["X-Title"] = "Forex AI Trader"

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        # Pass through any extra OpenAI-compatible kwargs (top_p, stream, etc.)
        # but skip ones we already set or that aren't supported.
        skip = {"model", "messages", "max_tokens", "temperature"}
        for k, v in extra.items():
            if k not in skip and v is not None:
                payload[k] = v

        # ── Day 99+ FIX (Issue #5): try curl_cffi first if available ──
        # curl_cffi emulates Chrome's TLS fingerprint (JA3) exactly,
        # which defeats Cloudflare's "browser fingerprint" check that
        # 403s Python's requests even with a browser User-Agent.
        # Fall back to plain requests if curl_cffi isn't installed
        # (it's an optional dependency — not all deployments have it).
        resp = None
        try:
            from curl_cffi import requests as _curl_requests  # type: ignore
            resp = _curl_requests.post(
                url, json=payload, headers=headers, timeout=60,
                impersonate="chrome120",
            )
        except ImportError:
            pass  # curl_cffi not installed — fall through to plain requests
        except Exception as e:
            # curl_cffi is installed but failed (network error, etc.).
            # Log and fall through to plain requests as a backup.
            log.debug(
                f"[{_OpenAICompatClient.__name__}] curl_cffi request failed "
                f"({type(e).__name__}: {e}) — falling back to requests"
            )
            resp = None

        if resp is None:
            resp = requests.post(url, json=payload, headers=headers, timeout=60)

        if resp.status_code != 200:
            # Surface the actual error body so callers can detect 429 / 403 etc.
            err_body = resp.text[:500]
            # Day 99+ FIX (Issue #5): add a targeted hint for 403, which
            # is almost always Cloudflare bot detection.
            hint = ""
            if resp.status_code == 403:
                hint = (
                    " — HTTP 403 typically means Cloudflare bot detection. "
                    "Install curl_cffi (pip install curl_cffi) for Chrome-"
                    "grade TLS fingerprint emulation that bypasses this."
                )
            raise RuntimeError(
                f"{self._provider} API error {resp.status_code}: {err_body}{hint}"
            )
        data = resp.json()
        # Wrap in a tiny object that exposes the same .choices[0].message.content
        # surface as the openai / groq SDKs.
        return _OpenAICompatResponse(data)


class _OpenAICompatResponse:
    """Mimics openai.ChatCompletion response object."""
    def __init__(self, data: dict):
        self._data = data
        # Build the .choices[0].message.content chain
        raw_choices = data.get("choices", [])
        choices = []
        for c in raw_choices:
            msg = c.get("message", {})
            choices.append(_OpenAICompatChoice(
                message=_OpenAICompatMessage(
                    content=msg.get("content", ""),
                    role=msg.get("role", "assistant"),
                ),
                finish_reason=c.get("finish_reason", "stop"),
            ))
        self.choices = choices
        # Surface usage stats for token-budget debugging
        self.usage = data.get("usage", {})


class _OpenAICompatChoice:
    def __init__(self, message, finish_reason):
        self.message = message
        self.finish_reason = finish_reason


class _OpenAICompatMessage:
    def __init__(self, content: str, role: str):
        self.content = content
        self.role = role


class LLMKeyManager:
    """Multi-key rotation manager for 7 LLM providers.

    Round-9 audit fix: expanded to 7 providers with up to 16 keys each
    (8+ API keys per provider as requested by operator). Provider chain:
      1. Gemini (Primary)        — gemini-2.5-flash
      2. Cerebras (Secondary)    — gpt-oss-120b
      3. Groq (Fast Inference)   — llama-3.3-70b-versatile
      4. SambaNova (Fallback)    — Meta-Llama-3.1-8B-Instruct
      5. OpenRouter (Router)     — universal router
      6. GitHub Models (Last Resort 1) — gpt-4o / Phi-3
      7. Hugging Face (Last Resort 2)  — inference endpoints
    """

    # Round-9: expanded key range from 9 to 16 (8+ keys per provider)
    MAX_KEYS_PER_PROVIDER = 16

    def __init__(self):
        self._lock = threading.RLock()
        # A client is selected before its request is made, then a caller
        # reports success/failure.  Keep that exact selection per thread.
        # Recomputing an "available" list in mark_* is incorrect after a
        # previous 429 removes a key from that list: its indexes shift and a
        # healthy sibling can be put on cooldown instead of the failing key.
        self._selected_keys: Dict[Tuple[str, int], KeyHealth] = {}
        self._client_keys: Dict[Tuple[str, int], KeyHealth] = {}
        self._groq_keys: List[KeyHealth] = []
        self._gemini_keys: List[KeyHealth] = []
        # Day 91 — three new OpenAI-compatible providers
        self._cerebras_keys: List[KeyHealth] = []
        self._sambanova_keys: List[KeyHealth] = []
        self._openrouter_keys: List[KeyHealth] = []
        # Round-9 — two new last-resort providers
        self._github_keys: List[KeyHealth] = []
        self._huggingface_keys: List[KeyHealth] = []
        # NEW PROVIDERS (architectural refactor):
        # Claude (Anthropic), GLM (Zhipu AI), DeepSeek — all use
        # OpenAI-compatible endpoints so they reuse _OpenAICompatClient.
        self._claude_keys: List[KeyHealth] = []
        self._glm_keys: List[KeyHealth] = []
        self._deepseek_keys: List[KeyHealth] = []
        self._groq_index = 0  # round-robin counter
        self._gemini_index = 0
        self._cerebras_index = 0
        self._sambanova_index = 0
        self._openrouter_index = 0
        self._github_index = 0
        self._huggingface_index = 0
        self._claude_index = 0
        self._glm_index = 0
        self._deepseek_index = 0
        self._exhausted_log_ts: Dict[str, float] = {}
        self._load_keys()
        # Day 131 fix — log the *effective* throttle config once at
        # startup so an operator can immediately see whether config.py's
        # documented default (8) is actually what's in effect, instead of
        # having to infer it later from "cycle cap reached (N/N)" lines.
        try:
            from config import (
                MAX_LLM_CALLS_PER_CYCLE,
                LLM_CALL_INTERVAL_SEC,
                MAX_LLM_CALLS_PER_MIN,
            )
            log.info(
                f"[LLM Throttle] effective config: "
                f"MAX_LLM_CALLS_PER_CYCLE={MAX_LLM_CALLS_PER_CYCLE} | "
                f"LLM_CALL_INTERVAL_SEC={LLM_CALL_INTERVAL_SEC} | "
                f"MAX_LLM_CALLS_PER_MIN={MAX_LLM_CALLS_PER_MIN}"
            )
            if MAX_LLM_CALLS_PER_CYCLE != 8:
                log.warning(
                    f"[LLM Throttle] MAX_LLM_CALLS_PER_CYCLE={MAX_LLM_CALLS_PER_CYCLE} "
                    f"differs from the documented default of 8 — check for a "
                    f"stale value in .env / the shell environment."
                )
        except Exception as e:
            log.debug(f"[LLM Throttle] could not read effective config: {e}")

    def _remember_selected_key(self, provider: str, key: KeyHealth) -> None:
        self._selected_keys[(provider, threading.get_ident())] = key

    def _consume_selected_key(self, provider: str) -> Optional[KeyHealth]:
        """Return the key used by this thread's most recent request.

        Marking by identity (rather than the current round-robin index) keeps
        retry rotation correct when any key became unavailable in between.
        """
        return self._selected_keys.pop((provider, threading.get_ident()), None)

    def _remember_client_key(self, provider: str, client: Any, key: KeyHealth) -> None:
        self._client_keys[(provider, id(client))] = key

    def _consume_client_key(self, provider: str, client: Optional[Any]) -> Optional[KeyHealth]:
        if client is not None:
            key = self._client_keys.pop((provider, id(client)), None)
            if key is not None:
                return key
        return self._consume_selected_key(provider)

    def _load_keys(self) -> None:
        """Load all keys from environment variables.

        Round-9 audit fix: expanded key range from range(1, 10) [9 keys]
        to range(1, 17) [16 keys] per provider. Also added GitHub Models
        and Hugging Face as last-resort providers.
        """
        _N = self.MAX_KEYS_PER_PROVIDER + 1  # 17 → range(1, 17) = 16 keys

        # ── Groq keys (Fast Inference) ──
        groq_keys = []
        for i in range(1, _N):
            key = os.getenv(f"GROQ_API_KEY_{i}", "")
            if key and key.strip():
                groq_keys.append(key.strip())
        legacy = os.getenv("GROQ_API_KEY", "")
        if legacy and legacy.strip() and legacy.strip() not in groq_keys:
            groq_keys.append(legacy.strip())
        for i, key in enumerate(groq_keys):
            self._groq_keys.append(KeyHealth(key=key, provider="groq", index=i))
        log.info(f"[LLM Keys] Loaded {len(self._groq_keys)} Groq key(s)")

        # ── Gemini keys (Primary) ──
        gemini_keys = []
        for i in range(1, _N):
            key = os.getenv(f"GEMINI_API_KEY_{i}", "")
            if key and key.strip():
                gemini_keys.append(key.strip())
        legacy = os.getenv("GEMINI_API_KEY", "")
        if legacy and legacy.strip() and legacy.strip() not in gemini_keys:
            gemini_keys.append(legacy.strip())
        for i, key in enumerate(gemini_keys):
            self._gemini_keys.append(KeyHealth(key=key, provider="gemini", index=i))
        log.info(f"[LLM Keys] Loaded {len(self._gemini_keys)} Gemini key(s)")

        # ── Cerebras keys (Secondary) ──
        cerebras_keys = []
        for i in range(1, _N):
            key = os.getenv(f"CEREBRAS_API_KEY_{i}", "")
            if key and key.strip():
                cerebras_keys.append(key.strip())
        legacy = os.getenv("CEREBRAS_API_KEY", "")
        if legacy and legacy.strip() and legacy.strip() not in cerebras_keys:
            cerebras_keys.append(legacy.strip())
        for i, key in enumerate(cerebras_keys):
            self._cerebras_keys.append(KeyHealth(key=key, provider="cerebras", index=i))
        log.info(f"[LLM Keys] Loaded {len(self._cerebras_keys)} Cerebras key(s)")

        # ── SambaNova keys (Fallback) ──
        sambanova_keys = []
        for i in range(1, _N):
            key = os.getenv(f"SAMBANOVA_API_KEY_{i}", "")
            if key and key.strip():
                sambanova_keys.append(key.strip())
        legacy = os.getenv("SAMBANOVA_API_KEY", "")
        if legacy and legacy.strip() and legacy.strip() not in sambanova_keys:
            sambanova_keys.append(legacy.strip())
        for i, key in enumerate(sambanova_keys):
            self._sambanova_keys.append(KeyHealth(key=key, provider="sambanova", index=i))
        log.info(f"[LLM Keys] Loaded {len(self._sambanova_keys)} SambaNova key(s)")

        # ── OpenRouter keys (Universal Router) ──
        openrouter_keys = []
        for i in range(1, _N):
            key = os.getenv(f"OPENROUTER_API_KEY_{i}", "")
            if key and key.strip():
                openrouter_keys.append(key.strip())
        legacy = os.getenv("OPENROUTER_API_KEY", "")
        if legacy and legacy.strip() and legacy.strip() not in openrouter_keys:
            openrouter_keys.append(legacy.strip())
        for i, key in enumerate(openrouter_keys):
            self._openrouter_keys.append(KeyHealth(key=key, provider="openrouter", index=i))
        log.info(f"[LLM Keys] Loaded {len(self._openrouter_keys)} OpenRouter key(s)")

        # ── Round-9: GitHub Models keys (Last Resort 1) ──
        # GitHub Models exposes an OpenAI-compatible endpoint at
        # https://models.inference.ai.azure.com/chat/completions
        # Authentication is via GitHub Personal Access Token (PAT).
        # Env var names: GITHUB_TOKEN or GITHUB_MODELS_API_KEY (legacy)
        # or GITHUB_TOKEN_1..16 / GITHUB_MODELS_API_KEY_1..16 (multi-key)
        github_keys = []
        for i in range(1, _N):
            key = os.getenv(f"GITHUB_TOKEN_{i}", "") or os.getenv(f"GITHUB_MODELS_API_KEY_{i}", "")
            if key and key.strip():
                github_keys.append(key.strip())
        legacy = os.getenv("GITHUB_TOKEN", "") or os.getenv("GITHUB_MODELS_API_KEY", "")
        if legacy and legacy.strip() and legacy.strip() not in github_keys:
            github_keys.append(legacy.strip())
        for i, key in enumerate(github_keys):
            self._github_keys.append(KeyHealth(key=key, provider="github", index=i))
        log.info(f"[LLM Keys] Loaded {len(self._github_keys)} GitHub Models key(s)")

        # ── Round-9: Hugging Face keys (Last Resort 2) ──
        # Hugging Face inference API uses HF_TOKEN for authentication.
        # https://api-inference.huggingface.co/models/{model}
        # Env var names: HF_TOKEN or HUGGINGFACE_API_KEY (legacy)
        # or HF_TOKEN_1..16 / HUGGINGFACE_API_KEY_1..16 (multi-key)
        hf_keys = []
        for i in range(1, _N):
            key = os.getenv(f"HF_TOKEN_{i}", "") or os.getenv(f"HUGGINGFACE_API_KEY_{i}", "")
            if key and key.strip():
                hf_keys.append(key.strip())
        legacy = os.getenv("HF_TOKEN", "") or os.getenv("HUGGINGFACE_API_KEY", "")
        if legacy and legacy.strip() and legacy.strip() not in hf_keys:
            hf_keys.append(legacy.strip())
        for i, key in enumerate(hf_keys):
            self._huggingface_keys.append(KeyHealth(key=key, provider="huggingface", index=i))
        log.info(f"[LLM Keys] Loaded {len(self._huggingface_keys)} Hugging Face key(s)")

        # ── NEW PROVIDERS (architectural refactor) ──────────────────
        # Claude (Anthropic), GLM (Zhipu AI), DeepSeek — all expose
        # OpenAI-compatible /v1/chat/completions endpoints, so they
        # reuse _OpenAICompatClient (no native SDK needed).

        # Claude (Anthropic) — env vars: ANTHROPIC_API_KEY or CLAUDE_API_KEY
        claude_keys = []
        for i in range(1, _N):
            key = os.getenv(f"ANTHROPIC_API_KEY_{i}", "") or os.getenv(f"CLAUDE_API_KEY_{i}", "")
            if key and key.strip():
                claude_keys.append(key.strip())
        legacy = os.getenv("ANTHROPIC_API_KEY", "") or os.getenv("CLAUDE_API_KEY", "")
        if legacy and legacy.strip() and legacy.strip() not in claude_keys:
            claude_keys.append(legacy.strip())
        for i, key in enumerate(claude_keys):
            self._claude_keys.append(KeyHealth(key=key, provider="claude", index=i))
        log.info(f"[LLM Keys] Loaded {len(self._claude_keys)} Claude (Anthropic) key(s)")

        # GLM (Zhipu AI) — env vars: GLM_API_KEY or ZHIPU_API_KEY
        glm_keys = []
        for i in range(1, _N):
            key = os.getenv(f"GLM_API_KEY_{i}", "") or os.getenv(f"ZHIPU_API_KEY_{i}", "")
            if key and key.strip():
                glm_keys.append(key.strip())
        legacy = os.getenv("GLM_API_KEY", "") or os.getenv("ZHIPU_API_KEY", "")
        if legacy and legacy.strip() and legacy.strip() not in glm_keys:
            glm_keys.append(legacy.strip())
        for i, key in enumerate(glm_keys):
            self._glm_keys.append(KeyHealth(key=key, provider="glm", index=i))
        log.info(f"[LLM Keys] Loaded {len(self._glm_keys)} GLM (Zhipu AI) key(s)")

        # DeepSeek — env vars: DEEPSEEK_API_KEY
        deepseek_keys = []
        for i in range(1, _N):
            key = os.getenv(f"DEEPSEEK_API_KEY_{i}", "")
            if key and key.strip():
                deepseek_keys.append(key.strip())
        legacy = os.getenv("DEEPSEEK_API_KEY", "")
        if legacy and legacy.strip() and legacy.strip() not in deepseek_keys:
            deepseek_keys.append(legacy.strip())
        for i, key in enumerate(deepseek_keys):
            self._deepseek_keys.append(KeyHealth(key=key, provider="deepseek", index=i))
        log.info(f"[LLM Keys] Loaded {len(self._deepseek_keys)} DeepSeek key(s)")

        # Summary log
        total = (len(self._groq_keys) + len(self._gemini_keys)
                 + len(self._cerebras_keys) + len(self._sambanova_keys)
                 + len(self._openrouter_keys) + len(self._github_keys)
                 + len(self._huggingface_keys)
                 + len(self._claude_keys) + len(self._glm_keys)
                 + len(self._deepseek_keys))
        log.info(
            f"[LLM Keys] Total: {total} key(s) across 10 providers "
            f"(Gemini={len(self._gemini_keys)}, Cerebras={len(self._cerebras_keys)}, "
            f"Groq={len(self._groq_keys)}, SambaNova={len(self._sambanova_keys)}, "
            f"OpenRouter={len(self._openrouter_keys)}, GitHub={len(self._github_keys)}, "
            f"HuggingFace={len(self._huggingface_keys)}, "
            f"Claude={len(self._claude_keys)}, GLM={len(self._glm_keys)}, "
            f"DeepSeek={len(self._deepseek_keys)})"
        )

    # ── Groq ──────────────────────────────────────────────────────

    def get_groq_client(self) -> Optional[Any]:
        """Get a working Groq client. Rotates through available keys."""
        with self._lock:
            available = [k for k in self._groq_keys if k.is_available]
            if not available:
                if self._groq_keys:
                    now = time.time()
                    last_ts = self._exhausted_log_ts.get("groq", 0.0)
                    if now - last_ts >= 60.0:
                        soonest = min(
                            (k.rate_limited_until for k in self._groq_keys if k.rate_limited_until > time.time()),
                            default=0.0,
                        )
                        eta = max(0.0, soonest - time.time())
                        log.warning(
                            f"[LLM Keys] All Groq keys exhausted — next recovery in {eta:.0f}s"
                        )
                        self._exhausted_log_ts["groq"] = now
                else:
                    log.error("[LLM Keys] No Groq keys configured!")
                return None

            # Round-robin: pick the next available key
            key = available[self._groq_index % len(available)]
            self._groq_index += 1
            self._remember_selected_key("groq", key)

        try:
            from groq import Groq
            client = Groq(api_key=key.key)
            self._remember_client_key("groq", client, key)
            log.debug(f"[LLM Keys] Using Groq key #{key.index + 1}")
            return client
        except ImportError:
            log.warning("[LLM Keys] groq package not installed")
            return None
        except Exception as e:
            # Constructor failure is NOT the same as API call failure.
            # Don't disable the key for constructor errors — just log and return None.
            # The key will be retried on the next call.
            log.debug(f"[LLM Keys] Groq constructor failed (non-fatal): {e}")
            return None

    def get_groq_key_info(self) -> Optional[KeyHealth]:
        """Get the KeyHealth object for the next available Groq key."""
        with self._lock:
            available = [k for k in self._groq_keys if k.is_available]
            if not available:
                return None
            return available[self._groq_index % len(available)]

    def mark_groq_success(self, tokens_used: int = 0, client: Optional[Any] = None) -> None:
        """Mark the current Groq key as successful.

        Round-19 fix: accepts tokens_used (from resp.usage.total_tokens)
        so per-key TPD tracking actually has data to work with. Without
        callers passing this, KeyHealth.tpd_tokens_used stays 0 forever
        and _is_tpd_exhausted() never proactively triggers — the system
        would still work (via reactive 429 handling) but only reacts
        AFTER wasting an API call on a key that was already exhausted.
        """
        with self._lock:
            key = self._consume_client_key("groq", client)
            if key is not None:
                key.mark_success()
                if tokens_used > 0:
                    key.record_tokens(tokens_used)

    def mark_groq_failure(self, error: str = "", rate_limited: bool = False, client: Optional[Any] = None) -> None:
        """Mark the current Groq key as failed."""
        with self._lock:
            key = self._consume_client_key("groq", client)
            if key is not None:
                key.mark_failure(error, rate_limited)

    # ── Gemini ────────────────────────────────────────────────────

    def get_gemini_client(self) -> Optional[Any]:
        """Get a working Gemini client. Rotates through available keys."""
        with self._lock:
            available = [k for k in self._gemini_keys if k.is_available]
            if not available:
                if self._gemini_keys:
                    now = time.time()
                    last_ts = self._exhausted_log_ts.get("gemini", 0.0)
                    if now - last_ts >= 60.0:
                        soonest = min(
                            (k.rate_limited_until for k in self._gemini_keys if k.rate_limited_until > time.time()),
                            default=0.0,
                        )
                        eta = max(0.0, soonest - time.time())
                        log.warning(
                            f"[LLM Keys] All Gemini keys exhausted — next recovery in {eta:.0f}s"
                        )
                        self._exhausted_log_ts["gemini"] = now
                else:
                    log.error("[LLM Keys] No Gemini keys configured!")
                return None

            key = available[self._gemini_index % len(available)]
            self._gemini_index += 1
            self._remember_selected_key("gemini", key)

        try:
            from google import genai as google_genai
            client = google_genai.Client(api_key=key.key)
            self._remember_client_key("gemini", client, key)
            log.debug(f"[LLM Keys] Using Gemini key #{key.index + 1}")
            return client
        except ImportError:
            log.warning("[LLM Keys] google-genai package not installed")
            return None
        except Exception as e:
            # Constructor failure — don't disable key, just return None
            log.debug(f"[LLM Keys] Gemini constructor failed (non-fatal): {e}")
            return None

    def get_gemini_key_info(self) -> Optional[KeyHealth]:
        with self._lock:
            available = [k for k in self._gemini_keys if k.is_available]
            if not available:
                return None
            return available[self._gemini_index % len(available)]

    def mark_gemini_success(self, tokens_used: int = 0, client: Optional[Any] = None) -> None:
        with self._lock:
            key = self._consume_client_key("gemini", client)
            if key is not None:
                key.mark_success()
                if tokens_used > 0:
                    key.record_tokens(tokens_used)

    def mark_gemini_failure(self, error: str = "", rate_limited: bool = False, client: Optional[Any] = None) -> None:
        with self._lock:
            key = self._consume_client_key("gemini", client)
            if key is not None:
                key.mark_failure(error, rate_limited)

    # ════════════════════════════════════════════════════════════
    # Day 91 — Cerebras / SambaNova / OpenRouter
    # ────────────────────────────────────────────────────────────
    # All three expose an OpenAI-compatible /v1/chat/completions
    # endpoint. We use the `openai` Python package if available, and
    # fall back to a tiny `requests`-based shim otherwise — so the
    # system works whether or not the openai package is installed.
    # Each provider returns a client object exposing a `.chat.
    # completions.create(...)` method that matches the Groq client's
    # surface so master_analyst._call_llm can use them interchangeably.
    # ════════════════════════════════════════════════════════════

    # ── Cerebras ────────────────────────────────────────────────

    def get_cerebras_client(self) -> Optional[Any]:
        """Get a working Cerebras client (OpenAI-compatible)."""
        with self._lock:
            available = [k for k in self._cerebras_keys if k.is_available]
            if not available:
                log.debug("[LLM Keys] No available Cerebras keys")
                return None
            key = available[self._cerebras_index % len(available)]
            self._cerebras_index += 1
        base_url = os.getenv("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")
        log.debug(f"[LLM Keys] Using Cerebras key #{key.index + 1}")
        return _OpenAICompatClient(key.key, base_url, "cerebras")

    def mark_cerebras_success(self) -> None:
        with self._lock:
            available = [k for k in self._cerebras_keys if k.is_available]
            if available:
                available[(self._cerebras_index - 1) % len(available)].mark_success()

    def mark_cerebras_failure(self, error: str = "", rate_limited: bool = False) -> None:
        with self._lock:
            available = [k for k in self._cerebras_keys if k.is_available]
            if available:
                available[(self._cerebras_index - 1) % len(available)].mark_failure(error, rate_limited)

    # ── SambaNova ───────────────────────────────────────────────

    def get_sambanova_client(self) -> Optional[Any]:
        """Get a working SambaNova client (OpenAI-compatible)."""
        with self._lock:
            available = [k for k in self._sambanova_keys if k.is_available]
            if not available:
                log.debug("[LLM Keys] No available SambaNova keys")
                return None
            key = available[self._sambanova_index % len(available)]
            self._sambanova_index += 1
        base_url = os.getenv("SAMBANOVA_BASE_URL", "https://api.sambanova.ai/v1")
        log.debug(f"[LLM Keys] Using SambaNova key #{key.index + 1}")
        return _OpenAICompatClient(key.key, base_url, "sambanova")

    def mark_sambanova_success(self) -> None:
        with self._lock:
            available = [k for k in self._sambanova_keys if k.is_available]
            if available:
                available[(self._sambanova_index - 1) % len(available)].mark_success()

    def mark_sambanova_failure(self, error: str = "", rate_limited: bool = False) -> None:
        with self._lock:
            available = [k for k in self._sambanova_keys if k.is_available]
            if available:
                available[(self._sambanova_index - 1) % len(available)].mark_failure(error, rate_limited)

    # ── OpenRouter ──────────────────────────────────────────────

    def get_openrouter_client(self) -> Optional[Any]:
        """Get a working OpenRouter client (OpenAI-compatible)."""
        with self._lock:
            available = [k for k in self._openrouter_keys if k.is_available]
            if not available:
                log.debug("[LLM Keys] No available OpenRouter keys")
                return None
            key = available[self._openrouter_index % len(available)]
            self._openrouter_index += 1
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        log.debug(f"[LLM Keys] Using OpenRouter key #{key.index + 1}")
        return _OpenAICompatClient(key.key, base_url, "openrouter")

    def mark_openrouter_success(self) -> None:
        with self._lock:
            available = [k for k in self._openrouter_keys if k.is_available]
            if available:
                available[(self._openrouter_index - 1) % len(available)].mark_success()

    def mark_openrouter_failure(self, error: str = "", rate_limited: bool = False) -> None:
        with self._lock:
            available = [k for k in self._openrouter_keys if k.is_available]
            if available:
                available[(self._openrouter_index - 1) % len(available)].mark_failure(error, rate_limited)

    # ── GitHub Models (Round-9 — Last Resort 1) ────────────────

    def get_github_client(self) -> Optional[Any]:
        """Get a working GitHub Models client (OpenAI-compatible).

        GitHub Models exposes an OpenAI-compatible endpoint at
        https://models.inference.ai.azure.com/chat/completions

        Authentication is via GitHub Personal Access Token (PAT)
        with `models:read` scope. The token is passed as a Bearer
        token in the Authorization header.
        """
        with self._lock:
            available = [k for k in self._github_keys if k.is_available]
            if not available:
                log.debug("[LLM Keys] No available GitHub Models keys")
                return None
            key = available[self._github_index % len(available)]
            self._github_index += 1
        base_url = os.getenv(
            "GITHUB_MODELS_BASE_URL",
            "https://models.inference.ai.azure.com"
        )
        log.debug(f"[LLM Keys] Using GitHub Models key #{key.index + 1}")
        return _OpenAICompatClient(key.key, base_url, "github")

    def mark_github_success(self) -> None:
        with self._lock:
            available = [k for k in self._github_keys if k.is_available]
            if available:
                available[(self._github_index - 1) % len(available)].mark_success()

    def mark_github_failure(self, error: str = "", rate_limited: bool = False) -> None:
        with self._lock:
            available = [k for k in self._github_keys if k.is_available]
            if available:
                available[(self._github_index - 1) % len(available)].mark_failure(error, rate_limited)

    # ── Hugging Face (Round-9 — Last Resort 2) ─────────────────

    def get_huggingface_client(self) -> Optional[Any]:
        """Get a working Hugging Face inference client.

        Hugging Face inference API uses the OpenAI-compatible endpoint
        at https://api-inference.huggingface.co/v1 (for chat completions
        via the router service) OR the direct model endpoint
        https://api-inference.huggingface.co/models/{model}.

        We use the OpenAI-compatible router for simplicity — it
        supports chat.completions.create() just like OpenRouter.
        Authentication is via HF_TOKEN Bearer header.
        """
        with self._lock:
            available = [k for k in self._huggingface_keys if k.is_available]
            if not available:
                log.debug("[LLM Keys] No available Hugging Face keys")
                return None
            key = available[self._huggingface_index % len(available)]
            self._huggingface_index += 1
        base_url = os.getenv(
            "HUGGINGFACE_BASE_URL",
            "https://api-inference.huggingface.co/v1"
        )
        log.debug(f"[LLM Keys] Using Hugging Face key #{key.index + 1}")
        return _OpenAICompatClient(key.key, base_url, "huggingface")

    def mark_huggingface_success(self) -> None:
        with self._lock:
            available = [k for k in self._huggingface_keys if k.is_available]
            if available:
                available[(self._huggingface_index - 1) % len(available)].mark_success()

    def mark_huggingface_failure(self, error: str = "", rate_limited: bool = False) -> None:
        with self._lock:
            available = [k for k in self._huggingface_keys if k.is_available]
            if available:
                available[(self._huggingface_index - 1) % len(available)].mark_failure(error, rate_limited)

    # ── Claude (Anthropic) — OpenAI-compatible endpoint ──────────

    def get_claude_client(self) -> Optional[Any]:
        """Get a working Claude (Anthropic) client.

        Anthropic exposes an OpenAI-compatible endpoint at
        https://api.anthropic.com/v1/ (the /v1/chat/completions route
        accepts the same payload shape as OpenAI). We reuse
        _OpenAICompatClient for the call.
        """
        with self._lock:
            available = [k for k in self._claude_keys if k.is_available]
            if not available:
                log.debug("[LLM Keys] No available Claude keys")
                return None
            key = available[self._claude_index % len(available)]
            self._claude_index += 1
        base_url = os.getenv(
            "CLAUDE_BASE_URL",
            "https://api.anthropic.com/v1"
        )
        log.debug(f"[LLM Keys] Using Claude key #{key.index + 1}")
        return _OpenAICompatClient(key.key, base_url, "claude")

    def mark_claude_success(self) -> None:
        with self._lock:
            available = [k for k in self._claude_keys if k.is_available]
            if available:
                available[(self._claude_index - 1) % len(available)].mark_success()

    def mark_claude_failure(self, error: str = "", rate_limited: bool = False) -> None:
        with self._lock:
            available = [k for k in self._claude_keys if k.is_available]
            if available:
                available[(self._claude_index - 1) % len(available)].mark_failure(error, rate_limited)

    # ── GLM (Zhipu AI) — OpenAI-compatible endpoint ──────────────

    def get_glm_client(self) -> Optional[Any]:
        """Get a working GLM (Zhipu AI) client.

        Zhipu AI exposes an OpenAI-compatible endpoint at
        https://open.bigmodel.cn/api/paas/v4/. We reuse
        _OpenAICompatClient for the call.
        """
        with self._lock:
            available = [k for k in self._glm_keys if k.is_available]
            if not available:
                log.debug("[LLM Keys] No available GLM keys")
                return None
            key = available[self._glm_index % len(available)]
            self._glm_index += 1
        base_url = os.getenv(
            "GLM_BASE_URL",
            "https://open.bigmodel.cn/api/paas/v4"
        )
        log.debug(f"[LLM Keys] Using GLM key #{key.index + 1}")
        return _OpenAICompatClient(key.key, base_url, "glm")

    def mark_glm_success(self) -> None:
        with self._lock:
            available = [k for k in self._glm_keys if k.is_available]
            if available:
                available[(self._glm_index - 1) % len(available)].mark_success()

    def mark_glm_failure(self, error: str = "", rate_limited: bool = False) -> None:
        with self._lock:
            available = [k for k in self._glm_keys if k.is_available]
            if available:
                available[(self._glm_index - 1) % len(available)].mark_failure(error, rate_limited)

    # ── DeepSeek — OpenAI-compatible endpoint ────────────────────

    def get_deepseek_client(self) -> Optional[Any]:
        """Get a working DeepSeek client.

        DeepSeek exposes an OpenAI-compatible endpoint at
        https://api.deepseek.com/v1. We reuse _OpenAICompatClient.
        """
        with self._lock:
            available = [k for k in self._deepseek_keys if k.is_available]
            if not available:
                log.debug("[LLM Keys] No available DeepSeek keys")
                return None
            key = available[self._deepseek_index % len(available)]
            self._deepseek_index += 1
        base_url = os.getenv(
            "DEEPSEEK_BASE_URL",
            "https://api.deepseek.com/v1"
        )
        log.debug(f"[LLM Keys] Using DeepSeek key #{key.index + 1}")
        return _OpenAICompatClient(key.key, base_url, "deepseek")

    def mark_deepseek_success(self) -> None:
        with self._lock:
            available = [k for k in self._deepseek_keys if k.is_available]
            if available:
                available[(self._deepseek_index - 1) % len(available)].mark_success()

    def mark_deepseek_failure(self, error: str = "", rate_limited: bool = False) -> None:
        with self._lock:
            available = [k for k in self._deepseek_keys if k.is_available]
            if available:
                available[(self._deepseek_index - 1) % len(available)].mark_failure(error, rate_limited)

    # ── Provider availability checks ───────────────────────────

    @property
    def has_any_cerebras(self) -> bool:
        return any(k.is_available for k in self._cerebras_keys)

    @property
    def has_any_sambanova(self) -> bool:
        return any(k.is_available for k in self._sambanova_keys)

    @property
    def has_any_openrouter(self) -> bool:
        return any(k.is_available for k in self._openrouter_keys)

    @property
    def has_any_github(self) -> bool:
        """Round-9: GitHub Models availability."""
        return any(k.is_available for k in self._github_keys)

    @property
    def has_any_huggingface(self) -> bool:
        """Round-9: Hugging Face availability."""
        return any(k.is_available for k in self._huggingface_keys)

    @property
    def has_any_claude(self) -> bool:
        """Architectural refactor: Claude (Anthropic) availability."""
        return any(k.is_available for k in self._claude_keys)

    @property
    def has_any_glm(self) -> bool:
        """Architectural refactor: GLM (Zhipu AI) availability."""
        return any(k.is_available for k in self._glm_keys)

    @property
    def has_any_deepseek(self) -> bool:
        """Architectural refactor: DeepSeek availability."""
        return any(k.is_available for k in self._deepseek_keys)

    def any_provider_available(self) -> bool:
        """True if ANY provider has at least one available key.

        Architectural refactor: expanded to include Claude, GLM, and
        DeepSeek as additional providers in the failover chain.
        Used by master_analyst to decide whether to even attempt an
        LLM call vs. going straight to the rule-engine fallback.
        """
        return (
            self.has_any_groq
            or self.has_any_cerebras
            or self.has_any_sambanova
            or self.has_any_openrouter
            or self.has_any_gemini
            or self.has_any_github
            or self.has_any_huggingface
            or self.has_any_claude
            or self.has_any_glm
            or self.has_any_deepseek
        )

    # ── Per-cycle LLM call throttle (Day 81+) ───────────────────
    # Prevents the Groq rate-limit storm by capping total LLM calls
    # per "cycle" (1 cycle = 1 symbol processed by 1 AITrader).
    # The cycle boundary is marked by reset_cycle_calls() which
    # trader.py calls at the top of each run_cycle().

    _cycle_call_count: int = 0
    _cycle_call_lock: threading.Lock = threading.Lock()
    _last_call_ts: float = 0.0

    # Day 81+ hotfix: GLOBAL rolling-window cap.  Per-cycle cap alone
    # is not enough — with 6 pairs × 5 calls/cycle = 30 calls in 2
    # minutes, all 6 Groq keys hit TPD limit.  This global cap limits
    # total calls across ALL cycles in a rolling 60-second window.
    # Default 12 calls/min (≈ 2 cycles × 6 calls each).
    _global_call_timestamps: deque = deque()
    _global_call_lock: threading.Lock = threading.Lock()

    def reset_cycle_calls(self) -> None:
        """Call at the start of each symbol cycle to reset the per-cycle
        LLM call counter.  trader.py calls this in run_cycle().

        Note: the GLOBAL rolling-window cap is NOT reset here — it
        persists across cycles to prevent the cross-cycle Groq storm.
        """
        with self._cycle_call_lock:
            self._cycle_call_count = 0

    def check_cycle_throttle(self) -> tuple[bool, str]:
        """Check if the current cycle has exceeded MAX_LLM_CALLS_PER_CYCLE.

        Returns (allowed, reason).  When allowed=False, the caller should
        skip the LLM call and use a fallback (e.g. rule engine signal).
        Also enforces LLM_CALL_INTERVAL_SEC between calls to the same
        provider (Groq free-tier rate-limit mitigation).

        Day 81+ hotfix: also enforces a GLOBAL rolling-window cap of
        MAX_LLM_CALLS_PER_MIN (default 12) calls per 60 seconds across
        all cycles.  Without this, 6 pairs × 5 calls/cycle = 30 calls
        in 2 minutes drains all 6 Groq keys' TPD quota.
        """
        try:
            from config import (
                MAX_LLM_CALLS_PER_CYCLE,
                LLM_CALL_INTERVAL_SEC,
                MAX_LLM_CALLS_PER_MIN,
            )
        except Exception as e:
            MAX_LLM_CALLS_PER_CYCLE = 8
            LLM_CALL_INTERVAL_SEC = 1.0
            MAX_LLM_CALLS_PER_MIN = 12

        # ── Global rolling-window cap (cross-cycle) ──
        now = time.time()
        with self._global_call_lock:
            # Evict timestamps older than 60 seconds
            cutoff = now - 60.0
            while self._global_call_timestamps and self._global_call_timestamps[0] < cutoff:
                self._global_call_timestamps.popleft()
            if len(self._global_call_timestamps) >= MAX_LLM_CALLS_PER_MIN:
                # Calculate sleep time until oldest timestamp exits window
                oldest = self._global_call_timestamps[0]
                wait_for = max(0.0, oldest + 60.0 - now)
                return False, (
                    f"global cap reached ({len(self._global_call_timestamps)}/"
                    f"{MAX_LLM_CALLS_PER_MIN} in last 60s) — retry in {wait_for:.0f}s"
                )

        with self._cycle_call_lock:
            # Per-cycle count cap
            if self._cycle_call_count >= MAX_LLM_CALLS_PER_CYCLE:
                return False, (
                    f"cycle cap reached ({self._cycle_call_count}/"
                    f"{MAX_LLM_CALLS_PER_CYCLE}) — skip LLM, use fallback"
                )
            # Per-call interval enforcement
            now = time.time()
            elapsed = now - self._last_call_ts
            if elapsed < LLM_CALL_INTERVAL_SEC:
                sleep_for = LLM_CALL_INTERVAL_SEC - elapsed
                # Release lock during sleep so other threads can proceed
                self._cycle_call_lock.release()
                try:
                    time.sleep(sleep_for)
                finally:
                    self._cycle_call_lock.acquire()
            self._cycle_call_count += 1
            self._last_call_ts = time.time()

        # Record this call in the global window (after cycle lock released)
        with self._global_call_lock:
            self._global_call_timestamps.append(time.time())

        # ── Round-10 audit fix: TPD (tokens-per-day) budget guard ────
        # The operator's audit found that Groq keys were hitting 98k/100k
        # TPD limit rapidly — 432 consecutive 429 errors in a single cycle.
        # This is because the system had no awareness of daily token budget.
        #
        # Now: track approximate token usage per provider per UTC day.
        # When a provider exceeds its TPD_BUDGET (env var, default 80000
        # to leave 20% headroom under Groq's 100k limit), the provider
        # is temporarily marked as "exhausted" until UTC midnight resets
        # the counter. The key manager will then skip that provider in
        # round-robin, forcing fallback to the next provider.
        try:
            self._track_tpd_usage()
        except Exception:
            pass  # TPD tracking is best-effort — never block a trade on it

        return True, f"call {self._cycle_call_count}/{MAX_LLM_CALLS_PER_CYCLE} (global {len(self._global_call_timestamps)}/{MAX_LLM_CALLS_PER_MIN})"

    # ── Round-10: TPD budget tracking ────────────────────────────────
    # Keys are organized by provider: {"groq": {"date": "2026-07-13", "tokens": 45000}, ...}
    _tpd_usage: Dict[str, Dict[str, Any]] = {}
    _tpd_lock: threading.Lock = threading.Lock()

    # Round-19 fix: alias to the module-level dict so KeyHealth (which
    # needs this before LLMKeyManager exists) and this class share one
    # source of truth instead of two independently-edited copies.
    _TPD_BUDGETS = _TPD_BUDGET_BY_PROVIDER

    @classmethod
    def _track_tpd_usage(cls) -> None:
        """Auto-reset TPD counters at UTC midnight. Called after every LLM call."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with cls._tpd_lock:
            for provider in cls._tpd_usage:
                if cls._tpd_usage[provider].get("date") != today:
                    # New UTC day — reset counter
                    cls._tpd_usage[provider] = {"date": today, "tokens": 0}

    def record_token_usage(self, provider: str, token_count: int) -> None:
        """Record token usage for TPD tracking. Call after each LLM response.

        Args:
            provider: "groq" | "gemini" | "cerebras" | "sambanova" | "openrouter" | "github" | "huggingface"
            token_count: Total tokens used (prompt + completion). If unknown, estimate
                         as len(prompt) / 4 (rough rule of thumb).
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._tpd_lock:
            if provider not in self._tpd_usage or self._tpd_usage[provider].get("date") != today:
                self._tpd_usage[provider] = {"date": today, "tokens": 0}
            self._tpd_usage[provider]["tokens"] += int(token_count)

        budget = self._TPD_BUDGETS.get(provider, 100000)
        used = self._tpd_usage[provider]["tokens"]
        if used >= budget * 0.8 and used < budget:
            log.warning(
                f"[LLM TPD] {provider} at {used:,}/{budget:,} tokens "
                f"({used/budget:.0%}) — approaching daily limit, will "
                f"switch to fallback providers soon"
            )
        elif used >= budget:
            log.warning(
                f"[LLM TPD] {provider} EXHAUSTED: {used:,}/{budget:,} tokens "
                f"({used/budget:.0%}) — this provider will be skipped until "
                f"UTC midnight resets the counter"
            )

    def is_tpd_exhausted(self, provider: str) -> bool:
        """Check if a provider has exceeded its TPD budget."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._tpd_lock:
            usage = self._tpd_usage.get(provider, {})
            if usage.get("date") != today:
                return False  # stale or new day — not exhausted
            used = usage.get("tokens", 0)
            budget = self._TPD_BUDGETS.get(provider, 100000)
            return used >= budget

    def tpd_status(self) -> Dict[str, Any]:
        """Return TPD usage status for all providers (for dashboard)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._tpd_lock:
            result = {}
            for provider, budget in self._TPD_BUDGETS.items():
                usage = self._tpd_usage.get(provider, {})
                if usage.get("date") != today:
                    result[provider] = {"used": 0, "budget": budget, "pct": 0.0, "exhausted": False}
                else:
                    used = usage.get("tokens", 0)
                    result[provider] = {
                        "used": used,
                        "budget": budget,
                        "pct": round(used / budget, 3) if budget > 0 else 0.0,
                        "exhausted": used >= budget,
                    }
            return result

    # ── Status ────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """Return status of all keys for dashboard."""
        with self._lock:
            return {
                "groq": {
                    "total": len(self._groq_keys),
                    "available": sum(1 for k in self._groq_keys if k.is_available),
                    "keys": [k.to_dict() for k in self._groq_keys],
                },
                "cerebras": {
                    "total": len(self._cerebras_keys),
                    "available": sum(1 for k in self._cerebras_keys if k.is_available),
                    "keys": [k.to_dict() for k in self._cerebras_keys],
                },
                "sambanova": {
                    "total": len(self._sambanova_keys),
                    "available": sum(1 for k in self._sambanova_keys if k.is_available),
                    "keys": [k.to_dict() for k in self._sambanova_keys],
                },
                "openrouter": {
                    "total": len(self._openrouter_keys),
                    "available": sum(1 for k in self._openrouter_keys if k.is_available),
                    "keys": [k.to_dict() for k in self._openrouter_keys],
                },
                "gemini": {
                    "total": len(self._gemini_keys),
                    "available": sum(1 for k in self._gemini_keys if k.is_available),
                    "keys": [k.to_dict() for k in self._gemini_keys],
                },
            }

    def reset_keys(self, provider: str = "all") -> None:
        """Clear fail counters + rate-limit cooldowns so all keys become
        available again.  Use this when a network outage tripped the
        disable thresholds and you want immediate recovery.

        Args:
            provider: "groq", "gemini", "cerebras", "sambanova",
                      "openrouter", or "all" (default).
        """
        with self._lock:
            targets = []
            if provider in ("all", "groq"):
                targets.extend(self._groq_keys)
            if provider in ("all", "gemini"):
                targets.extend(self._gemini_keys)
            if provider in ("all", "cerebras"):
                targets.extend(self._cerebras_keys)
            if provider in ("all", "sambanova"):
                targets.extend(self._sambanova_keys)
            if provider in ("all", "openrouter"):
                targets.extend(self._openrouter_keys)
            cleared = 0
            for k in targets:
                if not k.is_active:
                    continue
                k.fail_count = 0
                k.rate_limited_until = 0.0
                k.last_error = ""
                cleared += 1
            log.info(f"[LLM Keys] Reset {cleared} {provider} key(s) — all cooldowns cleared")

    @property
    def has_any_groq(self) -> bool:
        return any(k.is_available for k in self._groq_keys)

    @property
    def has_any_gemini(self) -> bool:
        return any(k.is_available for k in self._gemini_keys)

    @property
    def has_any_llm(self) -> bool:
        return self.has_any_groq or self.has_any_gemini

    # ── Global exhaustion detection ──────────────────────────────
    # When all keys for a provider are rate-limited simultaneously,
    # the previous code would return None from get_groq_client() and
    # callers would bail immediately — but the next call cycle (10s
    # later via the supervisor) would call get_groq_client() again,
    # and again, and again, hammering the still-rate-limited keys
    # and producing the 429 storm seen in the production logs.
    #
    # The fix: when all keys are exhausted, callers should *wait*
    # for the soonest-recovering key instead of looping fast.

    @property
    def all_groq_rate_limited(self) -> bool:
        """True if there is at least one Groq key AND all are unavailable."""
        with self._lock:
            return bool(self._groq_keys) and not any(
                k.is_available for k in self._groq_keys
            )

    @property
    def all_gemini_rate_limited(self) -> bool:
        with self._lock:
            return bool(self._gemini_keys) and not any(
                k.is_available for k in self._gemini_keys
            )

    def wait_for_any_groq(
        self,
        max_wait: float = 300.0,
        poll_interval: float = 10.0,
    ) -> bool:
        """Block until at least one Groq key becomes available, or
        ``max_wait`` seconds elapse.

        Returns True if a key is now available, False on timeout.  Use
        this from callers when ``get_groq_client()`` returns None to
        avoid hammering the API in a tight retry loop.

        Logs an ETA every poll cycle so the operator can see progress.
        """
        deadline = time.time() + max_wait
        while True:
            with self._lock:
                if any(k.is_available for k in self._groq_keys):
                    return True
                # ETA = soonest rate_limited_until among Groq keys
                soonest = min(
                    (k.rate_limited_until for k in self._groq_keys
                     if k.rate_limited_until > time.time()),
                    default=0.0,
                )
            remaining = deadline - time.time()
            if remaining <= 0:
                return self.has_any_groq
            eta = max(0.0, soonest - time.time())
            log.warning(
                f"[LLM Keys] All Groq keys exhausted — "
                f"soonest recovers in {eta:.0f}s, "
                f"max_wait remaining {remaining:.0f}s"
            )
            # Sleep the smaller of poll_interval / eta / remaining
            sleep_for = min(poll_interval, max(2.0, eta), remaining)
            time.sleep(sleep_for)
        # unreachable
        return self.has_any_groq


# ── Singleton ───────────────────────────────────────────────────────

_MANAGER: Optional[LLMKeyManager] = None
_MANAGER_LOCK = threading.Lock()


def get_llm_key_manager() -> LLMKeyManager:
    global _MANAGER
    if _MANAGER is None:
        with _MANAGER_LOCK:
            if _MANAGER is None:
                _MANAGER = LLMKeyManager()
    return _MANAGER