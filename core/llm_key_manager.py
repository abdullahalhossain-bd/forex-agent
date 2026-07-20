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
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

try:
    import config as _config  
except Exception:
    load_dotenv()

log = logging.getLogger("llm_key_manager")


def classify_llm_error(error: Exception) -> dict:
    """Classify LLM API failures without false positives."""
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
_GROQ_RETRY_RE_HMS  = re.compile(r"(\d+)\s*h\s*(\d+)\s*m\s*([\d.]+)\s*s", re.IGNORECASE)
_GROQ_RETRY_RE_HM   = re.compile(r"(\d+)\s*h\s*(\d+)\s*m(?:in)?(?:ute)?s?", re.IGNORECASE)
_GROQ_RETRY_RE_H    = re.compile(r"(\d+)\s*h(?:ou)?r?s?", re.IGNORECASE)
_GROQ_RETRY_RE_MMSS = re.compile(r"(\d+)m\s*([\d.]+)s")
_GROQ_RETRY_RE_SS   = re.compile(r"([\d.]+)\s*s")
_GROQ_RETRY_RE_MM   = re.compile(r"(\d+)\s*m(?:in)?(?:ute)?s?", re.IGNORECASE)
_GROQ_RETRY_RE_HDR  = re.compile(r"retry[-_ ]?after['\"\s:=]+(\d+)", re.IGNORECASE)

MIN_RETRY_COOLDOWN = 60        
MAX_RETRY_COOLDOWN = 60 * 60 * 6   
DEFAULT_RETRY_COOLDOWN = 300   
GROQ_DEFAULT_RETRY_COOLDOWN = 1800  


def parse_groq_retry_after(error_str: str) -> int:
    """Parse 'Please try again in Xh Ym Z.Zs' from a Groq 429 response."""
    if not error_str:
        return DEFAULT_RETRY_COOLDOWN
    s = str(error_str)

    m = _GROQ_RETRY_RE_HMS.search(s)
    if m:
        total = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        return max(MIN_RETRY_COOLDOWN, min(MAX_RETRY_COOLDOWN, int(total) + 5))

    m = _GROQ_RETRY_RE_HM.search(s)
    if m:
        total = int(m.group(1)) * 3600 + int(m.group(2)) * 60
        return max(MIN_RETRY_COOLDOWN, min(MAX_RETRY_COOLDOWN, total + 5))

    m = _GROQ_RETRY_RE_H.search(s)
    if m:
        total = int(m.group(1)) * 3600
        return max(MIN_RETRY_COOLDOWN, min(MAX_RETRY_COOLDOWN, total + 5))

    m = _GROQ_RETRY_RE_MMSS.search(s)
    if m:
        total = int(m.group(1)) * 60 + float(m.group(2))
        return max(MIN_RETRY_COOLDOWN, min(MAX_RETRY_COOLDOWN, int(total) + 5))

    m = _GROQ_RETRY_RE_SS.search(s)
    if m:
        total = float(m.group(1))
        return max(MIN_RETRY_COOLDOWN, min(MAX_RETRY_COOLDOWN, int(total) + 5))

    m = _GROQ_RETRY_RE_MM.search(s)
    if m:
        total = int(m.group(1)) * 60
        return max(MIN_RETRY_COOLDOWN, min(MAX_RETRY_COOLDOWN, total + 5))

    m = _GROQ_RETRY_RE_HDR.search(s)
    if m:
        total = int(m.group(1))
        return max(MIN_RETRY_COOLDOWN, min(MAX_RETRY_COOLDOWN, total + 5))

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

        if rate_limited:
            cooldown = parse_groq_retry_after(error)
            self.rate_limited_until = time.time() + cooldown
            log.warning(
                f"[LLM Keys] {self.provider} key #{self.index + 1} "
                f"rate-limited, disabled for {cooldown}s"
            )
        elif "401" in error or "unauthorized" in err_lower:
            self.is_active = False
            log.error(f"[LLM Keys] {self.provider} key #{self.index + 1} unauthorized — permanently disabled")
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
        url = f"{self._base_url}/chat/completions"
        headers = browser_headers.copy()
        headers.update({
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        })
        if self._provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/forex-ai-trader"
            headers["X-Title"] = "Forex AI Trader"

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
            hint = ""
            if resp.status_code == 403:
                hint = " — HTTP 403 typically means Cloudflare bot detection. Install curl_cffi."
            raise RuntimeError(f"{self._provider} API error {resp.status_code}: {err_body}{hint}")
            
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

    def get_cerebras_client(self) -> Optional[Any]:
        with self._lock:
            available = [k for k in self._cerebras_keys if k.is_available]
            if not available: return None
            key = available[self._cerebras_index % len(available)]
            self._cerebras_index += 1
        base_url = os.getenv("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")
        return _OpenAICompatClient(key.key, base_url, "cerebras")

    def mark_cerebras_success(self) -> None:
        with self._lock:
            available = [k for k in self._cerebras_keys if k.is_available]
            if available: available[(self._cerebras_index - 1) % len(available)].mark_success()

    def mark_cerebras_failure(self, error: str = "", rate_limited: bool = False) -> None:
        with self._lock:
            available = [k for k in self._cerebras_keys if k.is_available]
            if available: available[(self._cerebras_index - 1) % len(available)].mark_failure(error, rate_limited)

    def get_sambanova_client(self) -> Optional[Any]:
        with self._lock:
            available = [k for k in self._sambanova_keys if k.is_available]
            if not available: return None
            key = available[self._sambanova_index % len(available)]
            self._sambanova_index += 1
        base_url = os.getenv("SAMBANOVA_BASE_URL", "https://api.sambanova.ai/v1")
        return _OpenAICompatClient(key.key, base_url, "sambanova")

    def mark_sambanova_success(self) -> None:
        with self._lock:
            available = [k for k in self._sambanova_keys if k.is_available]
            if available: available[(self._sambanova_index - 1) % len(available)].mark_success()

    def mark_sambanova_failure(self, error: str = "", rate_limited: bool = False) -> None:
        with self._lock:
            available = [k for k in self._sambanova_keys if k.is_available]
            if available: available[(self._sambanova_index - 1) % len(available)].mark_failure(error, rate_limited)

    def get_openrouter_client(self) -> Optional[Any]:
        with self._lock:
            available = [k for k in self._openrouter_keys if k.is_available]
            if not available: return None
            key = available[self._openrouter_index % len(available)]
            self._openrouter_index += 1
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        return _OpenAICompatClient(key.key, base_url, "openrouter")

    def mark_openrouter_success(self) -> None:
        with self._lock:
            available = [k for k in self._openrouter_keys if k.is_available]
            if available: available[(self._openrouter_index - 1) % len(available)].mark_success()

    def mark_openrouter_failure(self, error: str = "", rate_limited: bool = False) -> None:
        with self._lock:
            available = [k for k in self._openrouter_keys if k.is_available]
            if available: available[(self._openrouter_index - 1) % len(available)].mark_failure(error, rate_limited)

    def get_github_client(self) -> Optional[Any]:
        with self._lock:
            available = [k for k in self._github_keys if k.is_available]
            if not available: return None
            key = available[self._github_index % len(available)]
            self._github_index += 1
        base_url = os.getenv("GITHUB_MODELS_BASE_URL", "https://models.inference.ai.azure.com")
        return _OpenAICompatClient(key.key, base_url, "github")

    def mark_github_success(self) -> None:
        with self._lock:
            available = [k for k in self._github_keys if k.is_available]
            if available: available[(self._github_index - 1) % len(available)].mark_success()

    def mark_github_failure(self, error: str = "", rate_limited: bool = False) -> None:
        with self._lock:
            available = [k for k in self._github_keys if k.is_available]
            if available: available[(self._github_index - 1) % len(available)].mark_failure(error, rate_limited)

    def get_huggingface_client(self) -> Optional[Any]:
        with self._lock:
            available = [k for k in self._huggingface_keys if k.is_available]
            if not available: return None
            key = available[self._huggingface_index % len(available)]
            self._huggingface_index += 1
        base_url = os.getenv("HUGGINGFACE_BASE_URL", "https://api-inference.huggingface.co/v1")
        return _OpenAICompatClient(key.key, base_url, "huggingface")

    def mark_huggingface_success(self) -> None:
        with self._lock:
            available = [k for k in self._huggingface_keys if k.is_available]
            if available: available[(self._huggingface_index - 1) % len(available)].mark_success()

    def mark_huggingface_failure(self, error: str = "", rate_limited: bool = False) -> None:
        with self._lock:
            available = [k for k in self._huggingface_keys if k.is_available]
            if available: available[(self._huggingface_index - 1) % len(available)].mark_failure(error, rate_limited)

    def get_claude_client(self) -> Optional[Any]:
        with self._lock:
            available = [k for k in self._claude_keys if k.is_available]
            if not available: return None
            key = available[self._claude_index % len(available)]
            self._claude_index += 1
        base_url = os.getenv("CLAUDE_BASE_URL", "https://api.anthropic.com/v1")
        return _OpenAICompatClient(key.key, base_url, "claude")

    def mark_claude_success(self) -> None:
        with self._lock:
            available = [k for k in self._claude_keys if k.is_available]
            if available: available[(self._claude_index - 1) % len(available)].mark_success()

    def mark_claude_failure(self, error: str = "", rate_limited: bool = False) -> None:
        with self._lock:
            available = [k for k in self._claude_keys if k.is_available]
            if available: available[(self._claude_index - 1) % len(available)].mark_failure(error, rate_limited)

    def get_glm_client(self) -> Optional[Any]:
        with self._lock:
            available = [k for k in self._glm_keys if k.is_available]
            if not available: return None
            key = available[self._glm_index % len(available)]
            self._glm_index += 1
        base_url = os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
        return _OpenAICompatClient(key.key, base_url, "glm")

    def mark_glm_success(self) -> None:
        with self._lock:
            available = [k for k in self._glm_keys if k.is_available]
            if available: available[(self._glm_index - 1) % len(available)].mark_success()

    def mark_glm_failure(self, error: str = "", rate_limited: bool = False) -> None:
        with self._lock:
            available = [k for k in self._glm_keys if k.is_available]
            if available: available[(self._glm_index - 1) % len(available)].mark_failure(error, rate_limited)

    def get_deepseek_client(self) -> Optional[Any]:
        with self._lock:
            available = [k for k in self._deepseek_keys if k.is_available]
            if not available: return None
            key = available[self._deepseek_index % len(available)]
            self._deepseek_index += 1
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        return _OpenAICompatClient(key.key, base_url, "deepseek")

    def mark_deepseek_success(self) -> None:
        with self._lock:
            available = [k for k in self._deepseek_keys if k.is_available]
            if available: available[(self._deepseek_index - 1) % len(available)].mark_success()

    def mark_deepseek_failure(self, error: str = "", rate_limited: bool = False) -> None:
        with self._lock:
            available = [k for k in self._deepseek_keys if k.is_available]
            if available: available[(self._deepseek_index - 1) % len(available)].mark_failure(error, rate_limited)

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
    def has_any_llm(self) -> bool: return self.has_any_groq or self.has_any_gemini

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
        with self._lock:
            return {
                "groq": {"total": len(self._groq_keys), "available": sum(1 for k in self._groq_keys if k.is_available)},
                "gemini": {"total": len(self._gemini_keys), "available": sum(1 for k in self._gemini_keys if k.is_available)},
            }

    def reset_keys(self, provider: str = "all") -> None:
        with self._lock:
            targets = []
            if provider in ("all", "groq"): targets.extend(self._groq_keys)
            if provider in ("all", "gemini"): targets.extend(self._gemini_keys)
            cleared = 0
            for k in targets:
                if not k.is_active: continue
                k.fail_count = 0
                k.rate_limited_until = 0.0
                k.last_error = ""
                cleared += 1
            log.info(f"[LLM Keys] Reset {cleared} keys — all cooldowns cleared")


_MANAGER: Optional[LLMKeyManager] = None
_MANAGER_LOCK = threading.Lock()

def get_llm_key_manager() -> LLMKeyManager:
    global _MANAGER
    if _MANAGER is None:
        with _MANAGER_LOCK:
            if _MANAGER is None:
                _MANAGER = LLMKeyManager()
    return _MANAGER