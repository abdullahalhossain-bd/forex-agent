#!/usr/bin/env python3
"""Quick OpenRouter connectivity smoke test.

Usage:
    python scripts/test_openrouter.py

This script reads all needed settings from the repository .env file.
"""

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"

if load_dotenv is not None:
    load_dotenv(ENV_PATH)

API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free").strip()
BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
URL = f"{BASE_URL}/chat/completions"


if not API_KEY:
    print("[TEST] OPENROUTER_API_KEY is not set in .env.")
    print(f"[TEST] Expected key in: {ENV_PATH}")
    sys.exit(1)

payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "ping"}],
    "max_tokens": 5,
}

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/your-bot",
    "X-Title": "ForexBot",
}

req = urllib.request.Request(
    URL,
    data=json.dumps(payload).encode("utf-8"),
    headers=headers,
    method="POST",
)

print(f"[TEST] Calling OpenRouter with model: {MODEL}")
print(f"[TEST] URL: {URL}")

try:
    with urllib.request.urlopen(req, timeout=25) as resp:
        body = resp.read().decode("utf-8", errors="ignore")
        print(f"[TEST] SUCCESS: HTTP {resp.status}")
        print(body[:500])
        sys.exit(0)
except urllib.error.HTTPError as e:
    body = e.read().decode("utf-8", errors="ignore")
    print(f"[TEST] FAILED: HTTP {e.code}")
    print(body[:500])
    sys.exit(1)
except Exception as e:
    print(f"[TEST] FAILED: {type(e).__name__}: {e}")
    sys.exit(1)
