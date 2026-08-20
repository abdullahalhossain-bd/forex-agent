"""
core/llm_key_manager.py — Multi-Key LLM Rotation Manager (Day 72+)
=====================================================================

Manages multiple API keys per provider (Groq, Gemini) with automatic
failover. If one key hits a rate limit or fails, it automatically
switches to the next available key.
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
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from dotenv import load_dotenv

try:
    import config as _config  
except Exception:
    load_dotenv()

log = logging.getLogger("llm_key_manager")


def classify_llm_error(error: Exception) -> dict:
    """Classify LLM API failures without false positives.

    FIX (rate-limit granularity — audit follow-up): a 429 is not always
    the same problem. Groq's free tier enforces several independent
    limits — TPD (tokens per day), RPM (requests per minute), and TPM
    (tokens per minute) — and they need very different recovery
    strategies:
      - TPD exhausted   → this key/account is dead for the rest of the
        day. The right move is an immediate switch to the next account,
        not a short retry on the same key.
      - RPM/TPM limited → transient, resets in seconds to a couple of
        minutes. A short cooldown (or immediate key switch) is enough.
    Treating every 429 the same way (as before) meant a TPD-exhausted
    key could still get a short cooldown and be retried a few minutes
    later — burning a call that was guaranteed to fail again until UTC
    midnight. See KeyHealth.mark_failure for how these flags are used.
    """
    error_str = str(error)
    err_lower = error_str.lower()
    # ── FIX: 503 UNAVAILABLE, 502 Bad Gateway, 504 Gateway Timeout are
    # TRANSIENT server-side errors (the provider is overloaded / restarting).
    # Treating them as auth-failure-adjacent is wrong — they shouldn't burn
    # the consecutive_auth_failures counter. We classify them as transient
    # so they only get a short cooldown, not a permanent disable.
    is_transient_server = (
        "503" in error_str
        or "502" in error_str
        or "504" in error_str
        or "unavailable" in err_lower
        or "high demand" in err_lower
        or "temporarily" in err_lower
        or "service unavailable" in err_lower
        or "bad gateway" in err_lower
        or "gateway timeout" in err_lower
    )
    is_rate_limited = (
        "429" in error_str
        or "too many requests" in err_lower
        or "rate limit" in err_lower
        or "rate_limit" in err_lower
    )
    # TPD (daily token budget) — Groq's message reads e.g. "Rate limit
    # reached for ... tokens per day (TPD) ... Please try again in
    # 2h59m58s". Also treat plan/quota exhaustion wording the same way
    # (same recovery: switch account, don't retry soon).
    is_tpd = is_rate_limited and (
        "tpd" in err_lower
        or "tokens per day" in err_lower
        or "daily limit" in err_lower
        or "quota" in err_lower
        or "exceeded your current quota" in err_lower
    )
    # RPM/TPM — short-lived, resets within seconds to a couple of minutes.
    is_rpm = is_rate_limited and not is_tpd and (
        "rpm" in err_lower
        or "tpm" in err_lower
        or "requests per minute" in err_lower
        or "tokens per minute" in err_lower
    )
    is_model_unavailable = (
        "model_not_found" in err_lower
        or "does not exist" in err_lower
        or "has been decommissioned" in err_lower
        or "decommissioned" in err_lower
        or ("model" in err_lower and "not found" in err_lower)
        # BUGFIX (log audit — 279 OpenRouter 404 occurrences for
        # 'liquid/lfm-2.5-1.2b-instruct:free'): the 404 error message
        # from _OpenAICompatClient._do_create() is literally
        # "openrouter API error 404: model '...' not found (...)" —
        # the word "model" is followed by a quoted string then "not
        # found", so the existing `("model" in err_lower and "not
        # found" in err_lower)` check matches, but the wording
        # "API error 404" should also be treated as a hard
        # model-availability signal so callers can permanently skip
        # the offending model id instead of retrying it every cycle.
        or ("api error 404" in err_lower and "model" in err_lower)
    )
    # BUGFIX (log audit — 558 OpenRouter 429 occurrences): the OpenRouter
    # rate-limit body uses the phrase "Rate limit exceeded:
    # free-models-per-day" with `limit_source` set to
    # "openrouter_free_tier_daily". The existing 429/rate-limit
    # matcher catches it, but the TPD classifier missed the
    # "free-models-per-day" / "openrouter_free_tier_daily" wording,
    # so OpenRouter 429s were treated as generic RPM limits with a
    # 60s cooldown instead of a multi-hour daily-limit cooldown —
    # causing the same free-tier-exhausted key to be retried every
    # 60s for hours, generating hundreds of duplicate error lines.
    is_openrouter_free_tier_daily = (
        "free-models-per-day" in err_lower
        or "openrouter_free_tier_daily" in err_lower
        or "free tier daily" in err_lower
    )
    if is_openrouter_free_tier_daily:
        # Promote to TPD so it gets the multi-hour cooldown below.
        is_tpd = True
        is_rate_limited = True
    return {
        "error_str": error_str,
        "error_type": type(error).__name__,
        "rate_limited": is_rate_limited,
        "tpd_exhausted": is_tpd,
        "rpm_limited": is_rpm,
        "auth_failed": (
            "401" in error_str
            or "403" in error_str
            or "unauthorized" in err_lower
            or "invalid api key" in err_lower
            or "invalid x-api-key" in err_lower
        ),
        "transient_server": is_transient_server,
        "model_unavailable": is_model_unavailable,
        # Surface this so callers (master_analyst / ai_analyst) can
        # permanently skip a dead model id instead of just rotating
        # keys — the same dead model would otherwise fail on every
        # key in the rotation.
        "openrouter_free_tier_daily": is_openrouter_free_tier_daily,
    }


def log_llm_call_failure(
    logger: logging.Logger,
    provider: str,
    model: str,
    attempt: int,
    max_retries: int,
    error: Exception,
) -> dict:
    """Log full LLM failure details for diagnosis.

    Log level policy:
      * Rate-limit (429) and auth (401/403) failures are logged at
        WARNING, not ERROR — the system has a multi-provider fallback
        chain (Groq → Gemini → OpenRouter → Cerebras) that almost
        always recovers successfully, so logging these as ERROR was
        producing scary tracebacks in trader.log for a benign, handled
        condition. A real ERROR is reserved for genuine failures with
        no recovery path (e.g. all providers down, programming errors).
      * Genuine server errors (500/502/503), model_unavailable, and
        unknown exceptions are still logged at ERROR.
    """
    info = classify_llm_error(error)
    _is_recoverable = bool(
        info.get("rate_limited")
        or info.get("auth_failed")
        or info.get("transient_server")
    )
    _log_fn = logger.warning if _is_recoverable else logger.error
    _log_fn(
        "[LLM] %s failed attempt %s/%s | model=%s | type=%s | "
        "rate_limited=%s (tpd=%s, rpm=%s) | auth_failed=%s | "
        "transient_server=%s | model_unavailable=%s | error=%s",
        provider,
        attempt + 1,
        max_retries,
        model,
        info["error_type"],
        info["rate_limited"],
        info.get("tpd_exhausted"),
        info.get("rpm_limited"),
        info["auth_failed"],
        info.get("transient_server"),
        info.get("model_unavailable"),
        info["error_str"][:800],
        # Only attach the traceback for genuine errors — rate-limit
        # tracebacks are 100% noise (the same Groq SDK frames every time).
        exc_info=not _is_recoverable,
    )
    return info


# ── Groq 429 retry-after parser ────────────────────────────────────
_GROQ_RETRY_RE_HMS  = re.compile(r"(\d+)\s*h\s*(\d+)\s*m\s*([\d.]+)\s*s", re.IGNORECASE)
_GROQ_RETRY_RE_HM   = re.compile(r"(\d+)\s*h\s*(\d+)\s*m(?:in)?(?:ute)?s?", re.IGNORECASE)
_GROQ_RETRY_RE_H    = re.compile(r"(\d+)\s*h(?:ou)?r?s?", re.IGNORECASE)
_GROQ_RETRY_RE_MMSS = re.compile(r"(\d+)m\s*([\d.]+)s")
_GROQ_RETRY_RE_SS   = re.compile(r"([\d.]+)\s*s")
_GROQ_RETRY_RE_MM   = re.compile(r"(\d+)\s*m(?:in)?(?:ute)?s?", re.IGNORECASE)
_GROQ_RETRY_RE_HDR  = re.compile(r"retry[-_ ]?after['\"\s:=]+(\d+)", re.IGNORECASE)
# FIX (audit follow-up): used to scope duration-parsing to the actual
# "try again in ..." clause instead of the whole error string — see
# parse_groq_retry_after for why this matters (org-id false positives).
# The lookahead `(?!\d)` on the terminator makes sure we stop at a
# sentence-ending period ("...58s. Need more...") but NOT at a decimal
# point inside the duration itself ("...try again in 21.298s" — a plain
# `[^.]+` would truncate this to "21", silently dropping the seconds).
_TRY_AGAIN_CLAUSE_RE = re.compile(r"try again in\s+(.+?)(?:\.(?!\d)|$)", re.IGNORECASE)

MIN_RETRY_COOLDOWN = 60        
MAX_RETRY_COOLDOWN = 60 * 60 * 6   
DEFAULT_RETRY_COOLDOWN = 300   
GROQ_DEFAULT_RETRY_COOLDOWN = 1800  

# FIX (TPD vs RPM cooldown split — audit follow-up): previously every
# 429 fell back to the same DEFAULT_RETRY_COOLDOWN (5min) whenever the
# "Please try again in ..." text couldn't be parsed. That's fine for a
# transient RPM/TPM limit, but wrong for a TPD (daily budget) exhaustion
# — a key that hit its daily cap won't recover in 5 minutes, so it just
# gets retried (and fails) every 5 minutes for the rest of the day,
# wasting calls that were guaranteed to fail. When we can positively
# identify TPD exhaustion but can't parse an exact retry-after, fall
# back to a multi-hour cooldown instead.
TPD_FALLBACK_COOLDOWN = 60 * 60 * 3     # 3h — used only when TPD is detected but unparsable
RPM_MAX_COOLDOWN = 120                  # RPM/TPM limits reset fast; cap the cooldown so the key isn't held out too long


def parse_groq_retry_after(error_str: str) -> int:
    """Parse 'Please try again in Xh Ym Z.Zs' from a Groq 429 response.

    FIX (audit follow-up — confirmed in production log): the original
    version searched the ENTIRE error string with all duration patterns,
    including Groq's organization id (e.g.
    "org_01kxe63htqejksjmghz9n9fbh0"), which can contain a digit-then-'h'
    substring purely by chance (e.g. "...e63h..."). The hour-only pattern
    (_GROQ_RETRY_RE_H) matched that "63h" and turned a real
    "try again in 24m15.84s" (~24 min) into a bogus 63-hour parse,
    clamped to MAX_RETRY_COOLDOWN (6h) — silently disabling a key for
    6 hours when it would have recovered in 24 minutes.

    Fix: isolate the actual "try again in ..." clause first and run the
    duration patterns ONLY against that substring, so unrelated digits
    elsewhere in the message (org ids, limits, token counts) can't be
    mistaken for a duration. Full-string scanning is kept only as a
    last resort, and only with the narrow/safe patterns (explicit
    Retry-After header, bare seconds) that are far less likely to
    false-positive on incidental text.
    """
    if not error_str:
        return DEFAULT_RETRY_COOLDOWN
    s = str(error_str)

    clause_match = _TRY_AGAIN_CLAUSE_RE.search(s)
    if clause_match:
        search_text = clause_match.group(1)

        m = _GROQ_RETRY_RE_HMS.search(search_text)
        if m:
            total = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
            return max(MIN_RETRY_COOLDOWN, min(MAX_RETRY_COOLDOWN, int(total) + 5))

        m = _GROQ_RETRY_RE_HM.search(search_text)
        if m:
            total = int(m.group(1)) * 3600 + int(m.group(2)) * 60
            return max(MIN_RETRY_COOLDOWN, min(MAX_RETRY_COOLDOWN, total + 5))

        m = _GROQ_RETRY_RE_H.search(search_text)
        if m:
            total = int(m.group(1)) * 3600
            return max(MIN_RETRY_COOLDOWN, min(MAX_RETRY_COOLDOWN, total + 5))

        m = _GROQ_RETRY_RE_MMSS.search(search_text)
        if m:
            total = int(m.group(1)) * 60 + float(m.group(2))
            return max(MIN_RETRY_COOLDOWN, min(MAX_RETRY_COOLDOWN, int(total) + 5))

        m = _GROQ_RETRY_RE_SS.search(search_text)
        if m:
            total = float(m.group(1))
            return max(MIN_RETRY_COOLDOWN, min(MAX_RETRY_COOLDOWN, int(total) + 5))

        m = _GROQ_RETRY_RE_MM.search(search_text)
        if m:
            total = int(m.group(1)) * 60
            return max(MIN_RETRY_COOLDOWN, min(MAX_RETRY_COOLDOWN, total + 5))

        # Clause was found but didn't match any known duration shape —
        # fall through to DEFAULT rather than risk scanning the whole
        # string (which is exactly what caused the false-positive).
        return DEFAULT_RETRY_COOLDOWN

    # No "try again in ..." clause at all (e.g. a Retry-After header on
    # its own, or a plain "N seconds" message). These two patterns are
    # narrow enough (explicit header keyword, or a lone number+unit) to
    # be safe against the whole string.
    m = _GROQ_RETRY_RE_HDR.search(s)
    if m:
        total = int(m.group(1))
        return max(MIN_RETRY_COOLDOWN, min(MAX_RETRY_COOLDOWN, total + 5))

    # BUGFIX (log audit): OpenRouter returns 429 with an
    # "X-RateLimit-Reset" header (epoch-millis) embedded in the JSON
    # body, e.g. `"X-RateLimit-Reset":"1786147200000"`. Parse it and
    # compute the cooldown as (reset - now) so we don't waste cycles
    # retrying a key that the provider has already told us won't
    # recover until midnight UTC.
    or_reset_match = re.search(
        r'["\']X-RateLimit-Reset["\']\s*:\s*["\']?(\d{10,13})["\']?',
        s,
    )
    if or_reset_match:
        try:
            reset_ms = int(or_reset_match.group(1))
            # Normalize to seconds if it looks like millis
            if reset_ms > 10_000_000_000:  # > year 2286 in seconds
                reset_sec = reset_ms / 1000.0
            else:
                reset_sec = float(reset_ms)
            now_sec = time.time()
            cooldown = max(MIN_RETRY_COOLDOWN, int(reset_sec - now_sec) + 5)
            return min(MAX_RETRY_COOLDOWN, cooldown)
        except (ValueError, OSError):
            pass

    return DEFAULT_RETRY_COOLDOWN


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

    # Consecutive 401/403 failures allowed before permanent disable.
    # 2 = one "benefit of the doubt" cooldown cycle (30min), then dead
    # keys stop being retried. Configurable via env for ops flexibility.
    # NOTE: must be typing.ClassVar, not a plain annotated field — a
    # dataclass field with a default cannot precede `key`/`provider`/
    # `index` (which have no defaults); ClassVar opts it out of the
    # dataclass's generated __init__ entirely, which is what we want
    # for a shared constant anyway.
    AUTH_FAIL_PERMANENT_THRESHOLD: ClassVar[int] = int(os.getenv("LLM_KEY_AUTH_FAIL_THRESHOLD", "2"))

    key: str
    provider: str             
    index: int                
    success_count: int = 0
    fail_count: int = 0
    last_error: str = ""
    last_success: float = 0.0
    rate_limited_until: float = 0.0  
    is_active: bool = True
    tpd_tokens_used: int = 0
    tpd_date: str = ""
    # Day 140+ FIX (Bug#4 — infinite 401 retry loop): tracks how many
    # *consecutive* auth failures (401/403) this key has produced since
    # its last success. A single 401 could be a transient auth-service
    # blip, so it still only gets a 30min cooldown. But a key that keeps
    # failing 401 after every cooldown expires is provably dead (invalid/
    # revoked credential — cooldown time never fixes that), and retrying
    # it forever every 30min wastes a call + full round-trip per key per
    # cycle for no possible benefit. After AUTH_FAIL_PERMANENT_THRESHOLD
    # consecutive 401s we permanently disable it (is_active=False) so it
    # stops being selected at all. Reset on any success, or manually via
    # reset_keys(provider, force=True) once the credential is fixed.
    consecutive_auth_failures: int = 0

    @property
    def is_available(self) -> bool:
        if not self.is_active:
            return False
        if self.rate_limited_until > time.time():
            return False
        if self._is_tpd_exhausted():
            return False
        return True

    def _is_tpd_exhausted(self) -> bool:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.tpd_date != today:
            return False  
        budget = _TPD_BUDGET_BY_PROVIDER.get(self.provider, 100000)
        return self.tpd_tokens_used >= budget

    def record_tokens(self, token_count: int) -> None:
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
                f"({self.tpd_tokens_used/budget:.0%}) — will be proactively skipped"
            )

    def mark_success(self) -> None:
        self.success_count += 1
        self.last_success = time.time()
        self.rate_limited_until = 0.0  
        self.consecutive_auth_failures = 0  

    def mark_failure(self, error: str = "", rate_limited: bool = False) -> None:
        self.fail_count += 1
        self.last_error = error[:200]

        err_lower = error.lower()
        is_network_error = any(s in err_lower for s in (
            "getaddrinfo", "connection", "timeout", "timed out",
            "network", "dns", "unreachable", "refused", "reset",
            "11001", "etimedout", "ehostunreach", "enetunreach",
            "ssl", "certificate", "proxyerror",
        ))

        # FIX (audit follow-up): check transient-server (503/502/504/
        # "high demand") FIRST and unconditionally — regardless of
        # whether a 401/403 also appears in the same message. Previously
        # this check only ran *inside* the 401/403 branch, so a pure
        # Gemini "503 UNAVAILABLE — model overloaded" (no 401/403 at all)
        # fell through to the generic fail_count-based cooldown, which
        # only reacts after 5+ failures. A transient server error should
        # get a short cooldown immediately so the key rotates back in
        # quickly without needing several failed calls first.
        is_transient_server = any(s in err_lower for s in (
            "503", "502", "504", "unavailable", "high demand",
            "temporarily", "service unavailable", "bad gateway",
            "gateway timeout",
        ))
        if is_transient_server:
            self.rate_limited_until = time.time() + 30
            log.warning(
                f"[LLM Keys] {self.provider} key #{self.index + 1} "
                f"transient server error (503/502/504/high-demand) — 30s cooldown"
            )
            return

        if rate_limited:
            # FIX (TPD vs RPM — audit follow-up): don't apply the same
            # cooldown to every 429. Positively identify TPD (daily
            # budget) exhaustion vs a transient RPM/TPM limit and pick
            # a cooldown that actually matches how soon the limit clears.
            # BUGFIX (log audit): also detect OpenRouter's free-tier
            # daily limit ("free-models-per-day" / "openrouter_free_tier_daily")
            # — it's a per-day limit just like Groq's TPD, but the
            # wording didn't match any of the existing TPD detectors,
            # so the key was retried every 60s for hours (558 dup errors).
            is_tpd = (
                "tpd" in err_lower
                or "tokens per day" in err_lower
                or "daily limit" in err_lower
                or "quota" in err_lower
                or "free-models-per-day" in err_lower
                or "openrouter_free_tier_daily" in err_lower
                or "free tier daily" in err_lower
            )
            is_rpm = (not is_tpd) and (
                "rpm" in err_lower or "tpm" in err_lower
                or "requests per minute" in err_lower or "tokens per minute" in err_lower
            )
            parsed = parse_groq_retry_after(error)
            if is_tpd:
                # If the provider gave us an exact "try again in Xh Ym"
                # duration, trust it. Otherwise assume a multi-hour wait
                # rather than the generic 5-minute default — a TPD-capped
                # key will not recover in 5 minutes.
                cooldown = parsed if parsed != DEFAULT_RETRY_COOLDOWN else TPD_FALLBACK_COOLDOWN
                self.rate_limited_until = time.time() + cooldown
                log.warning(
                    f"[LLM Keys] {self.provider} key #{self.index + 1} "
                    f"TPD (daily token budget) EXHAUSTED — disabled for {cooldown}s "
                    f"(~{cooldown/3600:.1f}h). Rotating to next account."
                )
            elif is_rpm:
                cooldown = min(parsed, RPM_MAX_COOLDOWN)
                self.rate_limited_until = time.time() + cooldown
                log.warning(
                    f"[LLM Keys] {self.provider} key #{self.index + 1} "
                    f"RPM/TPM (short-lived) rate limit — {cooldown}s cooldown"
                )
            else:
                # Generic/unclassified 429 — keep prior behavior.
                cooldown = parsed
                self.rate_limited_until = time.time() + cooldown
                log.warning(
                    f"[LLM Keys] {self.provider} key #{self.index + 1} "
                    f"rate-limited, disabled for {cooldown}s"
                )
        # P4b FIX (Bug#2): also handle 403 Forbidden — previously only
        # 401 triggered action. 403 (common from Cerebras/Cloudflare) was
        # classified as auth_failed but never acted upon, so the key kept
        # getting selected and wasting API calls every cycle.
        elif "401" in error or "403" in error or "unauthorized" in err_lower:
            self.consecutive_auth_failures += 1
            # Day 140+ FIX (Bug#4): P4b's move away from permanent-disable
            # (30min cooldown "in case it's transient") had no upper bound —
            # confirmed in production logs: 7/7 Groq keys invalid, every one
            # retried every 30min forever, burning a call + round-trip per
            # key per cycle with zero chance of success (an invalid/revoked
            # API key does not become valid by waiting). We keep the
            # transient-friendly behavior for the FIRST failure (could be a
            # momentary auth-service blip), but a key that is STILL failing
            # 401/403 after its cooldown has already expired and it was
            # retried is provably dead — permanently disable it so it stops
            # being selected. `mark_success()` resets this counter, and
            # `reset_keys(provider, force=True)` remains available to
            # manually revive a key once the operator fixes the credential.
            if self.consecutive_auth_failures >= self.AUTH_FAIL_PERMANENT_THRESHOLD:
                self.is_active = False
                self.rate_limited_until = 0.0
                log.error(
                    f"[LLM Keys] {self.provider} key #{self.index + 1} PERMANENTLY "
                    f"DISABLED — {self.consecutive_auth_failures} consecutive auth "
                    f"failures ({'401' if '401' in error else '403'}). Key is invalid/"
                    f"revoked; retrying will not help. Fix the credential and call "
                    f"reset_keys(provider='{self.provider}', force=True) to re-enable."
                )
            else:
                self.rate_limited_until = time.time() + 1800
                log.error(
                    f"[LLM Keys] {self.provider} key #{self.index + 1} auth failure "
                    f"({'401' if '401' in error else '403'}) — 30min cooldown "
                    f"({self.consecutive_auth_failures}/{self.AUTH_FAIL_PERMANENT_THRESHOLD} "
                    f"before permanent disable)"
                )
        elif is_network_error:
            log.debug(
                f"[LLM Keys] {self.provider} key #{self.index + 1} network error "
                f"(NOT disabling — will retry): {error[:80]}"
            )
        elif self.fail_count > 20:
            self.rate_limited_until = time.time() + 120
            log.warning(f"[LLM Keys] {self.provider} key #{self.index + 1} too many failures ({self.fail_count}), disabled for 2min")
        elif self.fail_count > 5:
            self.rate_limited_until = time.time() + 10
            log.warning(f"[LLM Keys] {self.provider} key #{self.index + 1} {self.fail_count} failures, 10s cooldown")

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


class _OpenAICompatClient:
    """Lightweight OpenAI-compatible REST client for Cerebras / SambaNova / OpenRouter."""

    def __init__(self, api_key: str, base_url: str, provider: str):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._provider = provider
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
        # Default OpenAI-compatible chat completions path used by many
        # providers. Hugging Face inference API is not OpenAI-compatible
        # — call its /models/{model} endpoint instead and normalise.
        if self._provider == "huggingface":
            url = f"{self._base_url.rstrip('/')}/models/{model}"
        else:
            url = f"{self._base_url}/chat/completions"
        headers = browser_headers.copy()
        headers.update({
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        })
        if self._provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/forex-ai-trader"
            headers["X-Title"] = "Forex AI Trader"

        # Build payload. For Hugging Face we send a simple `inputs` payload
        # (their inference API expects the model id in the URL).
        if self._provider == "huggingface":
            # Prefer the last user message as the input for HF models.
            last_msg = None
            for m in reversed(messages or []):
                if isinstance(m, dict) and m.get("role") in ("user", "system", "assistant"):
                    last_msg = m.get("content")
                    break
            if last_msg is None:
                last_msg = ""
            payload = {"inputs": last_msg}
        else:
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        
        skip = {"model", "messages", "max_tokens", "temperature", "timeout"}
        http_timeout = extra.get("timeout") or 60
        for k, v in extra.items():
            if k not in skip and v is not None:
                payload[k] = v

        resp = None
        try:
            from curl_cffi import requests as _curl_requests  # type: ignore
            resp = _curl_requests.post(
                url, json=payload, headers=headers, timeout=http_timeout,
                impersonate="chrome120",
            )
        except ImportError:
            pass  
        except Exception as e:
            log.debug(f"[{_OpenAICompatClient.__name__}] curl_cffi failed, falling back to requests: {e}")
            resp = None

        if resp is None:
            resp = requests.post(url, json=payload, headers=headers, timeout=http_timeout)

        if resp.status_code != 200:
            err_body = resp.text[:500]
            # Convert 404 into a clearer error for missing model/repo.
            if resp.status_code == 404:
                raise RuntimeError(f"{self._provider} API error 404: model '{model}' not found ({url})")
            hint = ""
            if resp.status_code == 403:
                hint = " — HTTP 403 typically means Cloudflare bot detection. Install curl_cffi."
            raise RuntimeError(f"{self._provider} API error {resp.status_code}: {err_body}{hint}")

        # Normalise provider responses into the small OpenAI-like dict
        # structure the rest of the code expects (`choices[0].message.content`).
        if self._provider == "huggingface":
            # HF may return text, a list, or a dict with generated_text.
            try:
                data = resp.json()
            except Exception:
                data = resp.text
            text = ""
            if isinstance(data, dict):
                text = data.get("generated_text") or data.get("text") or str(data)
            elif isinstance(data, list) and data:
                first = data[0]
                if isinstance(first, dict):
                    text = first.get("generated_text") or first.get("text") or str(first)
                else:
                    text = str(first)
            else:
                text = str(data)
            norm = {"choices": [{"message": {"content": text}}]}
            return _OpenAICompatResponse(norm)

        data = resp.json()
        return _OpenAICompatResponse(data)


class _OpenAICompatResponse:
    def __init__(self, data: dict):
        self._data = data
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
    """Multi-key rotation manager for 10 LLM providers."""

    MAX_KEYS_PER_PROVIDER = 16

    def __init__(self):
        self._lock = threading.RLock()
        self._selected_keys: Dict[Tuple[str, int], KeyHealth] = {}
        self._client_keys: Dict[Tuple[str, int], KeyHealth] = {}
        self._groq_keys: List[KeyHealth] = []
        self._gemini_keys: List[KeyHealth] = []
        self._cerebras_keys: List[KeyHealth] = []
        self._sambanova_keys: List[KeyHealth] = []
        self._openrouter_keys: List[KeyHealth] = []
        self._github_keys: List[KeyHealth] = []
        self._huggingface_keys: List[KeyHealth] = []
        self._claude_keys: List[KeyHealth] = []
        self._glm_keys: List[KeyHealth] = []
        self._deepseek_keys: List[KeyHealth] = []
        
        self._groq_index = 0  
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
        # FIX: per-provider auto-revive timestamps. When all keys for a
        # provider are permanently disabled (is_active=False), the manager
        # revives them once per 30 minutes in case credentials were rotated.
        self._auto_revive_ts: Dict[str, float] = {}
        self._load_keys()
        
        try:
            from config import MAX_LLM_CALLS_PER_CYCLE, LLM_CALL_INTERVAL_SEC, MAX_LLM_CALLS_PER_MIN
            log.info(f"[LLM Throttle] Loaded config parameters successfully.")
        except Exception as e:
            log.debug(f"[LLM Throttle] could not read effective config: {e}")

    def _remember_selected_key(self, provider: str, key: KeyHealth) -> None:
        self._selected_keys[(provider, threading.get_ident())] = key

    def _consume_selected_key(self, provider: str) -> Optional[KeyHealth]:
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
        _N = self.MAX_KEYS_PER_PROVIDER + 1  

        # ── Groq keys ──
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

        # ── Gemini keys ──
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

        # ── Cerebras keys ──
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

        # ── SambaNova keys ──
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

        # ── OpenRouter keys ──
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

        # ── GitHub Models ──
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

        # ── Hugging Face ──
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

        # ── Claude (Anthropic) ──
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
        log.info(f"[LLM Keys] Loaded {len(self._claude_keys)} Claude key(s)")

        # ── GLM (Zhipu AI) ──
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
        log.info(f"[LLM Keys] Loaded {len(self._glm_keys)} GLM key(s)")

        # ── DeepSeek ──
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

    # ── Groq ──────────────────────────────────────────────────────

    def get_groq_client(self) -> Optional[Any]:
        """Get a working Groq client. Rotates through available keys with anti-storm cooling."""
        with self._lock:
            available = [k for k in self._groq_keys if k.is_available]
            if not available:
                # ── FIX: auto-recovery from total exhaustion.
                # If ALL keys are permanently disabled (is_active=False),
                # nothing will ever bring them back without operator action.
                # But credentials sometimes get re-rotated silently (e.g.
                # rotated env vars on redeploy). Once per 30 minutes, if
                # every single key is dead, give them all one more shot.
                if self._groq_keys and all(not k.is_active for k in self._groq_keys):
                    now = time.time()
                    last_revive = self._auto_revive_ts.get("groq", 0.0)
                    if now - last_revive >= 1800.0:  # 30 min
                        revived = 0
                        for k in self._groq_keys:
                            k.is_active = True
                            k.consecutive_auth_failures = 0
                            k.fail_count = 0
                            k.rate_limited_until = 0.0
                            revived += 1
                        self._auto_revive_ts["groq"] = now
                        log.info(
                            f"[LLM Keys] AUTO-REVIVE: re-enabled {revived} Groq keys "
                            f"(all were permanently disabled — possibly credentials rotated). "
                            f"Next auto-revive in 30 min if all still fail."
                        )
                        # Try selection again with revived keys
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
                            log.warning(f"[LLM Keys] All Groq keys exhausted — next recovery in {eta:.0f}s")
                            self._exhausted_log_ts["groq"] = now
                    return None

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
            log.debug(f"[LLM Keys] Groq constructor failed: {e}")
            return None

    def get_groq_key_info(self) -> Optional[KeyHealth]:
        with self._lock:
            available = [k for k in self._groq_keys if k.is_available]
            if not available:
                return None
            return available[self._groq_index % len(available)]

    def mark_groq_success(self, tokens_used: int = 0, client: Optional[Any] = None) -> None:
        with self._lock:
            key = self._consume_client_key("groq", client)
            if key is not None:
                key.mark_success()
                if tokens_used > 0:
                    key.record_tokens(tokens_used)

    def mark_groq_failure(self, error: str = "", rate_limited: bool = False, client: Optional[Any] = None) -> None:
        """Mark the current Groq key as failed. 
        Enforces a small cool-off delay if rate-limited to prevent immediate multi-key IP banning.
        """
        with self._lock:
            key = self._consume_client_key("groq", client)
            if key is not None:
                key.mark_failure(error, rate_limited)
        
        # Anti-storm cool-off delay outside the instance lock to protect the shared IP resource
        if rate_limited:
            log.info("[Anti-Storm] Groq 429 caught. Cooling down IP for 4 seconds before next key rotation...")
            time.sleep(4.0)

    # ── Gemini ────────────────────────────────────────────────────

    def get_gemini_client(self) -> Optional[Any]:
        with self._lock:
            available = [k for k in self._gemini_keys if k.is_available]
            if not available:
                # FIX: auto-revive (same as Groq)
                if self._gemini_keys and all(not k.is_active for k in self._gemini_keys):
                    now = time.time()
                    last_revive = self._auto_revive_ts.get("gemini", 0.0)
                    if now - last_revive >= 1800.0:
                        revived = 0
                        for k in self._gemini_keys:
                            k.is_active = True
                            k.consecutive_auth_failures = 0
                            k.fail_count = 0
                            k.rate_limited_until = 0.0
                            revived += 1
                        self._auto_revive_ts["gemini"] = now
                        log.info(
                            f"[LLM Keys] AUTO-REVIVE: re-enabled {revived} Gemini keys "
                            f"(all were permanently disabled). Next auto-revive in 30 min."
                        )
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
                            log.warning(f"[LLM Keys] All Gemini keys exhausted — next recovery in {eta:.0f}s")
                            self._exhausted_log_ts["gemini"] = now
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
        if rate_limited:
            time.sleep(2.0)

    # ── OpenAI Compatible Custom Shims (Cerebras / SambaNova / OpenRouter / Last resorts) ──

    # P4b FIX (Bug#1): All 8 non-Groq/Gemini providers now use the same
    # _remember_client_key / _consume_client_key pattern that Groq and
    # Gemini use.  Previously, mark_xxx_success/failure used broken
    # (self._xxx_index - 1) % len(available) arithmetic which marked the
    # WRONG key under concurrent access or when key availability changed
    # between get and mark calls.

    def get_cerebras_client(self) -> Optional[Any]:
        with self._lock:
            available = [k for k in self._cerebras_keys if k.is_available]
            if not available:
                if self._cerebras_keys:
                    log.debug(
                        f"[LLM Keys] No available Cerebras keys "
                        f"({len(self._cerebras_keys)} configured, all cooling down/rate-limited/disabled)"
                    )
                else:
                    log.debug("[LLM Keys] No Cerebras keys configured")
                return None
            key = available[self._cerebras_index % len(available)]
            self._cerebras_index += 1
        base_url = os.getenv("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")
        try:
            client = _OpenAICompatClient(key.key, base_url, "cerebras")
            self._remember_client_key("cerebras", client, key)
            log.debug(f"[LLM Keys] Using Cerebras key #{key.index + 1}")
            return client
        except Exception as e:
            log.debug(f"[LLM Keys] Cerebras constructor failed: {e}")
            return None

    def mark_cerebras_success(self, client: Optional[Any] = None) -> None:
        with self._lock:
            key = self._consume_client_key("cerebras", client)
            if key is not None: key.mark_success()

    def mark_cerebras_failure(self, error: str = "", rate_limited: bool = False, client: Optional[Any] = None) -> None:
        with self._lock:
            key = self._consume_client_key("cerebras", client)
            if key is not None:
                key.mark_failure(error, rate_limited)
                log.debug(
                    f"[LLM Keys] Cerebras key #{key.index + 1} failed "
                    f"(rate_limited={rate_limited}, fail_count={key.fail_count}): {error[:120]}"
                )
            else:
                log.debug(f"[LLM Keys] Cerebras failure reported but no key could be matched to consume (client={client is not None})")

    def get_sambanova_client(self) -> Optional[Any]:
        with self._lock:
            available = [k for k in self._sambanova_keys if k.is_available]
            if not available:
                if self._sambanova_keys:
                    log.debug(
                        f"[LLM Keys] No available SambaNova keys "
                        f"({len(self._sambanova_keys)} configured, all cooling down/rate-limited/disabled)"
                    )
                else:
                    log.debug("[LLM Keys] No SambaNova keys configured")
                return None
            key = available[self._sambanova_index % len(available)]
            self._sambanova_index += 1
        base_url = os.getenv("SAMBANOVA_BASE_URL", "https://api.sambanova.ai/v1")
        try:
            client = _OpenAICompatClient(key.key, base_url, "sambanova")
            self._remember_client_key("sambanova", client, key)
            log.debug(f"[LLM Keys] Using SambaNova key #{key.index + 1}")
            return client
        except Exception as e:
            log.debug(f"[LLM Keys] SambaNova constructor failed: {e}")
            return None

    def mark_sambanova_success(self, client: Optional[Any] = None) -> None:
        with self._lock:
            key = self._consume_client_key("sambanova", client)
            if key is not None: key.mark_success()

    def mark_sambanova_failure(self, error: str = "", rate_limited: bool = False, client: Optional[Any] = None) -> None:
        with self._lock:
            key = self._consume_client_key("sambanova", client)
            if key is not None:
                key.mark_failure(error, rate_limited)
                log.debug(
                    f"[LLM Keys] SambaNova key #{key.index + 1} failed "
                    f"(rate_limited={rate_limited}, fail_count={key.fail_count}): {error[:120]}"
                )
            else:
                log.debug(f"[LLM Keys] SambaNova failure reported but no key could be matched to consume (client={client is not None})")

    def get_openrouter_client(self) -> Optional[Any]:
        with self._lock:
            available = [k for k in self._openrouter_keys if k.is_available]
            if not available:
                if self._openrouter_keys:
                    log.debug(
                        f"[LLM Keys] No available OpenRouter keys "
                        f"({len(self._openrouter_keys)} configured, all cooling down/rate-limited/disabled)"
                    )
                else:
                    log.debug("[LLM Keys] No OpenRouter keys configured")
                return None
            key = available[self._openrouter_index % len(available)]
            self._openrouter_index += 1
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        try:
            client = _OpenAICompatClient(key.key, base_url, "openrouter")
            self._remember_client_key("openrouter", client, key)
            log.debug(f"[LLM Keys] Using OpenRouter key #{key.index + 1}")
            return client
        except Exception as e:
            log.debug(f"[LLM Keys] OpenRouter constructor failed: {e}")
            return None

    def mark_openrouter_success(self, client: Optional[Any] = None) -> None:
        with self._lock:
            key = self._consume_client_key("openrouter", client)
            if key is not None: key.mark_success()

    def mark_openrouter_failure(self, error: str = "", rate_limited: bool = False, client: Optional[Any] = None) -> None:
        with self._lock:
            key = self._consume_client_key("openrouter", client)
            if key is not None:
                key.mark_failure(error, rate_limited)
                log.debug(
                    f"[LLM Keys] OpenRouter key #{key.index + 1} failed "
                    f"(rate_limited={rate_limited}, fail_count={key.fail_count}): {error[:120]}"
                )
            else:
                log.debug(f"[LLM Keys] OpenRouter failure reported but no key could be matched to consume (client={client is not None})")

    def get_github_client(self) -> Optional[Any]:
        with self._lock:
            available = [k for k in self._github_keys if k.is_available]
            if not available:
                if self._github_keys:
                    log.debug(
                        f"[LLM Keys] No available GitHub Models keys "
                        f"({len(self._github_keys)} configured, all cooling down/rate-limited/disabled)"
                    )
                else:
                    log.debug("[LLM Keys] No GitHub Models keys configured")
                return None
            key = available[self._github_index % len(available)]
            self._github_index += 1
        base_url = os.getenv("GITHUB_MODELS_BASE_URL", "https://models.inference.ai.azure.com")
        try:
            client = _OpenAICompatClient(key.key, base_url, "github")
            self._remember_client_key("github", client, key)
            log.debug(f"[LLM Keys] Using GitHub Models key #{key.index + 1}")
            return client
        except Exception as e:
            log.debug(f"[LLM Keys] GitHub Models constructor failed: {e}")
            return None

    def mark_github_success(self, client: Optional[Any] = None) -> None:
        with self._lock:
            key = self._consume_client_key("github", client)
            if key is not None: key.mark_success()

    def mark_github_failure(self, error: str = "", rate_limited: bool = False, client: Optional[Any] = None) -> None:
        with self._lock:
            key = self._consume_client_key("github", client)
            if key is not None:
                key.mark_failure(error, rate_limited)
                log.debug(
                    f"[LLM Keys] GitHub Models key #{key.index + 1} failed "
                    f"(rate_limited={rate_limited}, fail_count={key.fail_count}): {error[:120]}"
                )
            else:
                log.debug(f"[LLM Keys] GitHub Models failure reported but no key could be matched to consume (client={client is not None})")

    def get_huggingface_client(self) -> Optional[Any]:
        with self._lock:
            available = [k for k in self._huggingface_keys if k.is_available]
            if not available:
                if self._huggingface_keys:
                    log.debug(
                        f"[LLM Keys] No available Hugging Face keys "
                        f"({len(self._huggingface_keys)} configured, all cooling down/rate-limited/disabled)"
                    )
                else:
                    log.debug("[LLM Keys] No Hugging Face keys configured")
                return None
            key = available[self._huggingface_index % len(available)]
            self._huggingface_index += 1
        base_url = os.getenv("HUGGINGFACE_BASE_URL", "https://api-inference.huggingface.co/v1")
        try:
            client = _OpenAICompatClient(key.key, base_url, "huggingface")
            self._remember_client_key("huggingface", client, key)
            log.debug(f"[LLM Keys] Using Hugging Face key #{key.index + 1}")
            return client
        except Exception as e:
            log.debug(f"[LLM Keys] Hugging Face constructor failed: {e}")
            return None

    def mark_huggingface_success(self, client: Optional[Any] = None) -> None:
        with self._lock:
            key = self._consume_client_key("huggingface", client)
            if key is not None: key.mark_success()

    def mark_huggingface_failure(self, error: str = "", rate_limited: bool = False, client: Optional[Any] = None) -> None:
        with self._lock:
            key = self._consume_client_key("huggingface", client)
            if key is not None:
                key.mark_failure(error, rate_limited)
                log.debug(
                    f"[LLM Keys] Hugging Face key #{key.index + 1} failed "
                    f"(rate_limited={rate_limited}, fail_count={key.fail_count}): {error[:120]}"
                )
            else:
                log.debug(f"[LLM Keys] Hugging Face failure reported but no key could be matched to consume (client={client is not None})")

    def get_claude_client(self) -> Optional[Any]:
        with self._lock:
            available = [k for k in self._claude_keys if k.is_available]
            if not available:
                if self._claude_keys:
                    log.debug(
                        f"[LLM Keys] No available Claude keys "
                        f"({len(self._claude_keys)} configured, all cooling down/rate-limited/disabled)"
                    )
                else:
                    log.debug("[LLM Keys] No Claude keys configured")
                return None
            key = available[self._claude_index % len(available)]
            self._claude_index += 1
        base_url = os.getenv("CLAUDE_BASE_URL", "https://api.anthropic.com/v1")
        try:
            client = _OpenAICompatClient(key.key, base_url, "claude")
            self._remember_client_key("claude", client, key)
            log.debug(f"[LLM Keys] Using Claude key #{key.index + 1}")
            return client
        except Exception as e:
            log.debug(f"[LLM Keys] Claude constructor failed: {e}")
            return None

    def mark_claude_success(self, client: Optional[Any] = None) -> None:
        with self._lock:
            key = self._consume_client_key("claude", client)
            if key is not None: key.mark_success()

    def mark_claude_failure(self, error: str = "", rate_limited: bool = False, client: Optional[Any] = None) -> None:
        with self._lock:
            key = self._consume_client_key("claude", client)
            if key is not None:
                key.mark_failure(error, rate_limited)
                log.debug(
                    f"[LLM Keys] Claude key #{key.index + 1} failed "
                    f"(rate_limited={rate_limited}, fail_count={key.fail_count}): {error[:120]}"
                )
            else:
                log.debug(f"[LLM Keys] Claude failure reported but no key could be matched to consume (client={client is not None})")

    def get_glm_client(self) -> Optional[Any]:
        with self._lock:
            available = [k for k in self._glm_keys if k.is_available]
            if not available:
                if self._glm_keys:
                    log.debug(
                        f"[LLM Keys] No available GLM keys "
                        f"({len(self._glm_keys)} configured, all cooling down/rate-limited/disabled)"
                    )
                else:
                    log.debug("[LLM Keys] No GLM keys configured")
                return None
            key = available[self._glm_index % len(available)]
            self._glm_index += 1
        base_url = os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
        try:
            client = _OpenAICompatClient(key.key, base_url, "glm")
            self._remember_client_key("glm", client, key)
            log.debug(f"[LLM Keys] Using GLM key #{key.index + 1}")
            return client
        except Exception as e:
            log.debug(f"[LLM Keys] GLM constructor failed: {e}")
            return None

    def mark_glm_success(self, client: Optional[Any] = None) -> None:
        with self._lock:
            key = self._consume_client_key("glm", client)
            if key is not None: key.mark_success()

    def mark_glm_failure(self, error: str = "", rate_limited: bool = False, client: Optional[Any] = None) -> None:
        with self._lock:
            key = self._consume_client_key("glm", client)
            if key is not None:
                key.mark_failure(error, rate_limited)
                log.debug(
                    f"[LLM Keys] GLM key #{key.index + 1} failed "
                    f"(rate_limited={rate_limited}, fail_count={key.fail_count}): {error[:120]}"
                )
            else:
                log.debug(f"[LLM Keys] GLM failure reported but no key could be matched to consume (client={client is not None})")

    def get_deepseek_client(self) -> Optional[Any]:
        with self._lock:
            available = [k for k in self._deepseek_keys if k.is_available]
            if not available:
                if self._deepseek_keys:
                    log.debug(
                        f"[LLM Keys] No available DeepSeek keys "
                        f"({len(self._deepseek_keys)} configured, all cooling down/rate-limited/disabled)"
                    )
                else:
                    log.debug("[LLM Keys] No DeepSeek keys configured")
                return None
            key = available[self._deepseek_index % len(available)]
            self._deepseek_index += 1
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        try:
            client = _OpenAICompatClient(key.key, base_url, "deepseek")
            self._remember_client_key("deepseek", client, key)
            log.debug(f"[LLM Keys] Using DeepSeek key #{key.index + 1}")
            return client
        except Exception as e:
            log.debug(f"[LLM Keys] DeepSeek constructor failed: {e}")
            return None

    def mark_deepseek_success(self, client: Optional[Any] = None) -> None:
        with self._lock:
            key = self._consume_client_key("deepseek", client)
            if key is not None: key.mark_success()

    def mark_deepseek_failure(self, error: str = "", rate_limited: bool = False, client: Optional[Any] = None) -> None:
        with self._lock:
            key = self._consume_client_key("deepseek", client)
            if key is not None:
                key.mark_failure(error, rate_limited)
                log.debug(
                    f"[LLM Keys] DeepSeek key #{key.index + 1} failed "
                    f"(rate_limited={rate_limited}, fail_count={key.fail_count}): {error[:120]}"
                )
            else:
                log.debug(f"[LLM Keys] DeepSeek failure reported but no key could be matched to consume (client={client is not None})")

    # ── Property Checkers ──

    @property
    def has_any_groq(self) -> bool: return any(k.is_available for k in self._groq_keys)
    @property
    def has_any_gemini(self) -> bool: return any(k.is_available for k in self._gemini_keys)
    @property
    def has_any_cerebras(self) -> bool: return any(k.is_available for k in self._cerebras_keys)
    @property
    def has_any_sambanova(self) -> bool: return any(k.is_available for k in self._sambanova_keys)
    @property
    def has_any_openrouter(self) -> bool: return any(k.is_available for k in self._openrouter_keys)
    @property
    def has_any_github(self) -> bool: return any(k.is_available for k in self._github_keys)
    @property
    def has_any_huggingface(self) -> bool: return any(k.is_available for k in self._huggingface_keys)
    @property
    def has_any_claude(self) -> bool: return any(k.is_available for k in self._claude_keys)
    @property
    def has_any_glm(self) -> bool: return any(k.is_available for k in self._glm_keys)
    @property
    def has_any_deepseek(self) -> bool: return any(k.is_available for k in self._deepseek_keys)
    @property
    def has_any_llm(self) -> bool:
        # BUGFIX (audit): this previously only checked groq/gemini, so if
        # both were TPD-exhausted but a viable fallback (cerebras,
        # sambanova, openrouter, github, huggingface, claude, glm,
        # deepseek) still had budget, callers relying on has_any_llm would
        # wrongly conclude no LLM was available. No caller currently uses
        # this property (verified via repo-wide search), so this was
        # latent rather than actively causing the WAIT/NO_TRADE runs seen
        # in production — but it's fixed now so it's correct if/when
        # something starts relying on it. Delegates to
        # any_provider_available() so there's a single source of truth.
        return self.any_provider_available()

    def any_provider_available(self) -> bool:
        return (
            self.has_any_groq or self.has_any_cerebras or self.has_any_sambanova
            or self.has_any_openrouter or self.has_any_gemini or self.has_any_github
            or self.has_any_huggingface or self.has_any_claude or self.has_any_glm
            or self.has_any_deepseek
        )

    # ── Per-cycle LLM call throttle ──

    _cycle_call_count: int = 0
    _cycle_call_lock: threading.Lock = threading.Lock()
    _last_call_ts: float = 0.0
    _global_call_timestamps: deque = deque()
    _global_call_lock: threading.Lock = threading.Lock()

    def reset_cycle_calls(self) -> None:
        with self._cycle_call_lock:
            self._cycle_call_count = 0

    def check_cycle_throttle(self) -> tuple[bool, str]:
        try:
            from config import MAX_LLM_CALLS_PER_CYCLE, LLM_CALL_INTERVAL_SEC, MAX_LLM_CALLS_PER_MIN
        except Exception:
            MAX_LLM_CALLS_PER_CYCLE = 8
            LLM_CALL_INTERVAL_SEC = 2.0
            MAX_LLM_CALLS_PER_MIN = 12

        now = time.time()
        with self._global_call_lock:
            cutoff = now - 60.0
            while self._global_call_timestamps and self._global_call_timestamps[0] < cutoff:
                self._global_call_timestamps.popleft()
            if len(self._global_call_timestamps) >= MAX_LLM_CALLS_PER_MIN:
                oldest = self._global_call_timestamps[0]
                wait_for = max(0.0, oldest + 60.0 - now)
                return False, f"global cap reached — retry in {wait_for:.0f}s"

        with self._cycle_call_lock:
            if self._cycle_call_count >= MAX_LLM_CALLS_PER_CYCLE:
                return False, f"cycle cap reached ({self._cycle_call_count}/{MAX_LLM_CALLS_PER_CYCLE})"
            
            elapsed = now - self._last_call_ts
            if elapsed < LLM_CALL_INTERVAL_SEC:
                sleep_for = LLM_CALL_INTERVAL_SEC - elapsed
                self._cycle_call_lock.release()
                try:
                    time.sleep(sleep_for)
                finally:
                    self._cycle_call_lock.acquire()
            self._cycle_call_count += 1
            self._last_call_ts = time.time()

        with self._global_call_lock:
            self._global_call_timestamps.append(time.time())

        try:
            self._track_tpd_usage()
        except Exception:
            pass

        return True, f"call allowed {self._cycle_call_count}/{MAX_LLM_CALLS_PER_CYCLE}"

    _tpd_usage: Dict[str, Dict[str, Any]] = {}
    _tpd_lock: threading.Lock = threading.Lock()
    _TPD_BUDGETS = _TPD_BUDGET_BY_PROVIDER

    @classmethod
    def _track_tpd_usage(cls) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with cls._tpd_lock:
            for provider in cls._tpd_usage:
                if cls._tpd_usage[provider].get("date") != today:
                    cls._tpd_usage[provider] = {"date": today, "tokens": 0}

    def record_token_usage(self, provider: str, token_count: int) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._tpd_lock:
            if provider not in self._tpd_usage or self._tpd_usage[provider].get("date") != today:
                self._tpd_usage[provider] = {"date": today, "tokens": 0}
            self._tpd_usage[provider]["tokens"] += int(token_count)

    def is_tpd_exhausted(self, provider: str) -> bool:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._tpd_lock:
            usage = self._tpd_usage.get(provider, {})
            if usage.get("date") != today:
                return False
            return usage.get("tokens", 0) >= self._TPD_BUDGETS.get(provider, 100000)

    def tpd_status(self) -> Dict[str, Any]:
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

    def status(self) -> Dict[str, Any]:
        # P4b FIX (Bug#7): report all 10 providers, not just Groq/Gemini
        with self._lock:
            def _pstat(keys, name):
                return {"total": len(keys), "available": sum(1 for k in keys if k.is_available)}
            return {
                "groq": _pstat(self._groq_keys, "groq"),
                "gemini": _pstat(self._gemini_keys, "gemini"),
                "cerebras": _pstat(self._cerebras_keys, "cerebras"),
                "sambanova": _pstat(self._sambanova_keys, "sambanova"),
                "openrouter": _pstat(self._openrouter_keys, "openrouter"),
                "github": _pstat(self._github_keys, "github"),
                "huggingface": _pstat(self._huggingface_keys, "huggingface"),
                "claude": _pstat(self._claude_keys, "claude"),
                "glm": _pstat(self._glm_keys, "glm"),
                "deepseek": _pstat(self._deepseek_keys, "deepseek"),
            }

    def reset_keys(self, provider: str = "all", force: bool = False) -> None:
        """Reset cooldowns and failure counts for the specified provider(s).

        P4b FIX (Bug#4): now resets ALL 10 providers when provider="all",
        not just Groq and Gemini.

        P4b FIX (Bug#5): with force=True, also re-enables keys that were
        permanently disabled (is_active=False) due to 401/403 auth failures.
        Without force, those keys remain disabled (they were in cooldown).
        """
        _ALL_PROVIDERS = {
            "groq": self._groq_keys, "gemini": self._gemini_keys,
            "cerebras": self._cerebras_keys, "sambanova": self._sambanova_keys,
            "openrouter": self._openrouter_keys, "github": self._github_keys,
            "huggingface": self._huggingface_keys, "claude": self._claude_keys,
            "glm": self._glm_keys, "deepseek": self._deepseek_keys,
        }
        with self._lock:
            cleared = 0
            for prov_name, key_list in _ALL_PROVIDERS.items():
                if provider not in ("all", prov_name):
                    continue
                for k in key_list:
                    if not k.is_active and not force:
                        continue  # skip permanently disabled keys unless force
                    if force:
                        k.is_active = True
                    k.fail_count = 0
                    k.rate_limited_until = 0.0
                    k.last_error = ""
                    cleared += 1
            log.info(f"[LLM Keys] Reset {cleared} keys (provider={provider}, force={force})")


_MANAGER: Optional[LLMKeyManager] = None
_MANAGER_LOCK = threading.Lock()

def get_llm_key_manager() -> LLMKeyManager:
    global _MANAGER
    if _MANAGER is None:
        with _MANAGER_LOCK:
            if _MANAGER is None:
                _MANAGER = LLMKeyManager()
    return _MANAGER