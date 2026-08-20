"""
groq_check.py v3 — Dynamic model discovery + plain/JSON tests.

Fixes vs v2:
  - Classifies each key as VALID / INVALID (bad key, 401) / RATE_LIMITED / ERROR
    instead of lumping every models.list() failure into "no models listed".
  - Retries on 429 (rate limit) with backoff instead of giving up immediately.
  - Prints a clean per-key verdict table at the end so you can see at a glance
    which keys work and which models are valid on each.

Usage:
    python groq_check.py

1) Loads GROQ_API_KEY_1..N from .env
2) Lists models available to THAT key (GET /models)
3) Tests chat models only (skips whisper/tts/guard)
4) Plain completion + JSON mode (Devil's Advocate path)
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from dotenv import load_dotenv

load_dotenv()

try:
    from groq import Groq
    import groq as groq_sdk
except ImportError:
    print("pip install groq")
    raise SystemExit(1)

# groq-python raises these on 4xx/5xx; fall back gracefully if the SDK
# version in use doesn't expose one of them under this name.
AuthenticationError = getattr(groq_sdk, "AuthenticationError", Exception)
RateLimitError = getattr(groq_sdk, "RateLimitError", Exception)
APIStatusError = getattr(groq_sdk, "APIStatusError", Exception)
APIConnectionError = getattr(groq_sdk, "APIConnectionError", Exception)


def collect_keys() -> List[str]:
    keys: List[str] = []
    for i in range(1, 32):
        v = (os.getenv(f"GROQ_API_KEY_{i}") or "").strip()
        if v:
            keys.append(v)
    for name in ("GROQ_API_KEY", "GROQ_KEY"):
        v = (os.getenv(name) or "").strip()
        if v and v not in keys:
            keys.append(v)
    return keys


def mask(key: str) -> str:
    if len(key) <= 12:
        return "***"
    return f"{key[:7]}...{key[-4:]}"


SKIP_SUBSTR = (
    "whisper",
    "tts",
    "orpheus",
    "guard",
    "prompt-guard",
    "safeguard",
)


def is_chat_model(model_id: str) -> bool:
    m = model_id.lower()
    if any(s in m for s in SKIP_SUBSTR):
        return False
    return True


@dataclass
class KeyResult:
    idx: int
    masked: str
    status: str = "UNKNOWN"       # VALID / INVALID / RATE_LIMITED / ERROR
    detail: str = ""
    models_listed: List[str] = field(default_factory=list)
    plain_ok: List[str] = field(default_factory=list)
    json_ok: List[str] = field(default_factory=list)


def with_retry(fn, *, retries: int = 3, base_delay: float = 2.0):
    """Run fn(); retry on 429 with exponential backoff. Re-raises other errors."""
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except RateLimitError as e:
            last_exc = e
            if attempt == retries:
                raise
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)
        except APIStatusError as e:
            status_code = getattr(e, "status_code", None)
            if status_code == 429 and attempt < retries:
                last_exc = e
                time.sleep(base_delay * (2 ** attempt))
                continue
            raise
    raise last_exc  # pragma: no cover


def list_models(client: Groq, kr: KeyResult) -> None:
    """Populate kr.status / kr.detail / kr.models_listed."""
    try:
        page = with_retry(lambda: client.models.list())
        data = getattr(page, "data", None) or []
        ids = []
        for item in data:
            mid = getattr(item, "id", None) or (item.get("id") if isinstance(item, dict) else None)
            if mid and is_chat_model(str(mid)):
                ids.append(str(mid))
        kr.models_listed = sorted(set(ids))
        kr.status = "VALID" if kr.models_listed else "VALID_NO_CHAT_MODELS"
        kr.detail = f"{len(kr.models_listed)} chat model(s) visible"
    except AuthenticationError as e:
        kr.status = "INVALID"
        kr.detail = f"auth failed (bad/revoked key): {str(e)[:120]}"
    except RateLimitError as e:
        kr.status = "RATE_LIMITED"
        kr.detail = f"rate limited even after retries: {str(e)[:120]}"
    except APIConnectionError as e:
        kr.status = "ERROR"
        kr.detail = f"connection error: {str(e)[:120]}"
    except APIStatusError as e:
        status_code = getattr(e, "status_code", "?")
        kr.status = "INVALID" if status_code == 401 else "ERROR"
        kr.detail = f"HTTP {status_code}: {str(e)[:120]}"
    except Exception as e:
        kr.status = "ERROR"
        kr.detail = str(e)[:140]


def test_plain(client: Groq, model: str) -> Tuple[bool, str]:
    try:
        resp = with_retry(lambda: client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly the word OK and nothing else."}],
            temperature=0,
            max_tokens=32,
        ))
        msg = resp.choices[0].message
        text = (getattr(msg, "content", None) or "").strip()
        if not text:
            text = (getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None) or "").strip()[:40]
        if text:
            return True, text[:50].replace("\n", " ")
        return False, "empty content (finish=" + str(getattr(resp.choices[0], "finish_reason", "?")) + ")"
    except Exception as e:
        return False, str(e)[:140]


def test_json(client: Groq, model: str) -> Tuple[bool, str]:
    try:
        resp = with_retry(lambda: client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Return ONLY a JSON object with keys decision, confidence, reason. "
                        'Example: {"decision":"REJECT","confidence":0.1,"reason":"test"}'
                    ),
                }
            ],
            temperature=0,
            max_tokens=120,
            response_format={"type": "json_object"},
        ))
        raw = (resp.choices[0].message.content or "").strip()
        if not raw:
            return False, "empty generation (json_validate risk)"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            return False, f"bad JSON: {e} raw={raw[:50]!r}"
        if not isinstance(data, dict):
            return False, "not a dict"
        return True, "keys=" + str(list(data.keys())[:6])
    except Exception as e:
        err = str(e)
        if "json_validate" in err.lower():
            return False, "JSON_VALIDATE_FAILED: " + err[:90]
        return False, err[:140]


def prioritize(models: List[str]) -> List[str]:
    preferred = [
        os.getenv("DEVILS_ADVOCATE_MODEL", "").strip(),
        os.getenv("GROQ_MODEL", "").strip(),
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "llama-3.1-8b-instant",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "qwen/qwen3-32b",
        "moonshotai/kimi-k2-instruct",
        "moonshotai/kimi-k2-instruct-0905",
    ]
    ordered: List[str] = []
    seen: Set[str] = set()
    for m in preferred:
        if m and m in models and m not in seen:
            ordered.append(m)
            seen.add(m)
    for m in models:
        if m not in seen:
            ordered.append(m)
            seen.add(m)
    max_n = int(os.getenv("GROQ_CHECK_MAX_MODELS", "12"))
    return ordered[:max_n]


def main() -> None:
    keys = collect_keys()
    print(f"Groq keys from .env: {len(keys)}")
    if not keys:
        print("No GROQ_API_KEY_* found")
        raise SystemExit(2)

    plain_ok: Set[str] = set()
    json_ok: Set[str] = set()
    all_listed: Set[str] = set()
    results: List[KeyResult] = []

    for idx, key in enumerate(keys, 1):
        kr = KeyResult(idx=idx, masked=mask(key))
        results.append(kr)

        print("=" * 70)
        print(f"Key #{idx} {kr.masked}")
        if not key.startswith("gsk_"):
            print("   warning: expected gsk_ prefix")

        try:
            client = Groq(api_key=key)
        except Exception as e:
            kr.status = "ERROR"
            kr.detail = f"client init: {e}"
            print(f"   client init: {e}")
            continue

        list_models(client, kr)
        print(f"   status: {kr.status} — {kr.detail}")

        if kr.status not in ("VALID",):
            continue  # bad/rate-limited/errored key — nothing more to test

        all_listed.update(kr.models_listed)
        print(f"      {', '.join(kr.models_listed[:15])}{' ...' if len(kr.models_listed) > 15 else ''}")

        to_test = prioritize(kr.models_listed)
        print(f"   testing {len(to_test)} model(s)...")

        for model in to_test:
            time.sleep(0.4)
            ok, msg = test_plain(client, model)
            print(f"   {model}")
            if not ok:
                print(f"      plain -> FAIL {msg}")
                continue
            print(f"      plain -> OK {msg}")
            plain_ok.add(model)
            kr.plain_ok.append(model)

            time.sleep(0.4)
            jok, jmsg = test_json(client, model)
            if jok:
                print(f"      json  -> OK {jmsg}")
                json_ok.add(model)
                kr.json_ok.append(model)
            else:
                print(f"      json  -> FAIL {jmsg}")

    print("=" * 70)
    print("Per-key verdict")
    print("-" * 70)
    for kr in results:
        print(
            f"Key #{kr.idx:<2} {kr.masked:<16} {kr.status:<20} "
            f"plain_ok={len(kr.plain_ok)} json_ok={len(kr.json_ok)}"
        )
    print("=" * 70)

    working_keys = sum(1 for kr in results if kr.plain_ok)
    invalid_keys = [kr for kr in results if kr.status == "INVALID"]
    rate_limited_keys = [kr for kr in results if kr.status == "RATE_LIMITED"]

    print("Done\n")
    print(f"Working keys (>=1 model succeeded) : {working_keys}/{len(keys)}")
    if invalid_keys:
        print(f"Invalid keys (bad/revoked)          : {[k.masked for k in invalid_keys]}")
    if rate_limited_keys:
        print(f"Rate-limited keys (retry later)     : {[k.masked for k in rate_limited_keys]}")
    print(f"Union of listed models               : {len(all_listed)}")
    if all_listed:
        print(f"   {', '.join(sorted(all_listed))}")
    print(f"Plain OK (any key)                    : {sorted(plain_ok) or '-'}")
    print(f"JSON OK (for DA)                      : {sorted(json_ok) or '-'}")

    best: Optional[str] = None
    for m in prioritize(list(json_ok) or list(plain_ok)):
        if m in json_ok:
            best = m
            break
    if best is None and plain_ok:
        best = sorted(plain_ok)[0]

    print()
    if best and best in json_ok:
        print(f"Recommend for Devil's Advocate:")
        print(f"   DEVILS_ADVOCATE_MODEL={best}")
        print("   DEVILS_ADVOCATE_TIMEOUT_SEC=20")
    elif best:
        print(f"Only plain works: {best} (JSON failed — DA still risky)")
        print(f"   GROQ_MODEL={best}")
    else:
        print("No working chat completion.")
        print("   Check Groq console: keys active, limits, model access.")
        print("   https://console.groq.com/docs/models")

    da = os.getenv("DEVILS_ADVOCATE_MODEL", "")
    print(f"\nCurrent DEVILS_ADVOCATE_MODEL={da or '(unset)'}")
    if da in json_ok:
        print("   -> JSON OK")
    elif da in plain_ok:
        print("   -> plain only — production json_validate_failed risk")
    elif da:
        print("   -> not in successful set")
    print("=" * 70)


if __name__ == "__main__":
    main()