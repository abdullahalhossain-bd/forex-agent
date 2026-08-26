"""Provider-neutral gateway for the live LLM backend switch."""

from __future__ import annotations

import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from utils.llm_json import parse_llm_json
from utils.logger import get_logger

log = get_logger("llm_gateway")


class LLMGatewayError(RuntimeError):
    """Raised when the selected remote LLM cannot return valid JSON."""


def local_llm_enabled() -> bool:
    """Return the centralized local-backend switch."""
    from config import LLM_LOCAL

    return bool(LLM_LOCAL)


def backend_info() -> dict:
    """Return the effective backend configuration without exposing secrets."""
    from config import LLM_LOCAL, LLM_MODEL, LLM_REMOTE_URL, LLM_TIMEOUT

    return {
        "backend": "remote_ollama" if LLM_LOCAL else "legacy_provider_cascade",
        "model": LLM_MODEL if LLM_LOCAL else "configured_legacy_models",
        "url": ollama_api_url(LLM_REMOTE_URL) if LLM_LOCAL else None,
        "timeout": LLM_TIMEOUT,
    }


def ollama_api_url(base_url: str) -> str:
    """Normalize an Ollama base URL to exactly one ``/api/chat`` suffix."""
    base = (base_url or "").strip().rstrip("/")
    if base.endswith("/api/chat"):
        return base
    if base.endswith("/api"):
        return f"{base}/chat"
    return f"{base}/api/chat"


def call_remote_ollama(messages, *, model=None, timeout=None, retries=2) -> str:
    """Call remote Ollama and return only validated ``message.content``."""
    from config import LLM_MODEL, LLM_REMOTE_URL, LLM_TIMEOUT

    payload = {
        "model": model or LLM_MODEL,
        "messages": messages,
        "think": False,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }
    request = Request(
        ollama_api_url(LLM_REMOTE_URL),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    log.info(
        "[LLMGateway] selected backend=remote_ollama model=%s url=%s "
        "think=false stream=false format=json temperature=0 prompt_messages=%d",
        payload["model"], request.full_url, len(messages),
    )
    selected_timeout = float(timeout if timeout is not None else LLM_TIMEOUT)
    last_error = None
    for attempt in range(max(1, retries + 1)):
        try:
            with urlopen(request, timeout=selected_timeout) as response:
                decoded = json.loads(response.read().decode("utf-8"))
            content = decoded.get("message", {}).get("content") if isinstance(decoded, dict) else None
            if not isinstance(content, str) or not content.strip():
                raise LLMGatewayError("Ollama response did not contain message.content")
            parse_llm_json(content)
            log.info("[LLMGateway] validated JSON from message.content; thinking ignored")
            return content
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, TypeError, LLMGatewayError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2 ** attempt, 2))
    raise LLMGatewayError(
        f"Remote Ollama failed after {max(1, retries + 1)} attempt(s): {last_error}"
    ) from last_error


__all__ = ["LLMGatewayError", "backend_info", "call_remote_ollama", "local_llm_enabled", "ollama_api_url"]