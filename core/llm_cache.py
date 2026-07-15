"""
core/llm_cache.py — Day 90 LLM Response Cache
================================================
Caches LLM responses by (provider, model, prompt-hash) for a short TTL.
This is critical for long-duration demo trading where the same symbol+
timeframe+regime combination will produce nearly-identical LLM prompts
across consecutive cycles — re-calling the LLM just burns tokens.

Strategy:
  - Cache key = sha256(provider + model + prompt)[:16]
  - TTL = 5 minutes (300s) by default — short enough to react to new
    candles, long enough to deduplicate rapid retries
  - In-memory only (no persistence) — we want fresh start per session
  - Thread-safe via RLock
  - Hit-rate stats logged every N calls

Usage:
    from core.llm_cache import get_llm_cache
    cache = get_llm_cache()

    cache_key = cache.make_key("groq", "llama-3.3-70b-versatile", prompt)
    cached = cache.get(cache_key)
    if cached:
        return cached   # skip API call
    raw = call_llm_api(prompt)
    cache.set(cache_key, raw)
    return raw
"""
from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class CacheEntry:
    response: str
    timestamp: float
    token_estimate: int = 0


class LLMCache:
    def __init__(self, ttl_sec: int = 300, max_entries: int = 200):
        self.ttl_sec = ttl_sec
        self.max_entries = max_entries
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        # Stats
        self.hits = 0
        self.misses = 0
        self.tokens_saved = 0

    @staticmethod
    def make_key(provider: str, model: str, prompt: str) -> str:
        """Build a cache key from provider+model+prompt."""
        h = hashlib.sha256()
        h.update(provider.encode("utf-8", errors="ignore"))
        h.update(b"|")
        h.update(model.encode("utf-8", errors="ignore"))
        h.update(b"|")
        h.update(prompt.encode("utf-8", errors="ignore"))
        return h.hexdigest()[:16]

    def get(self, key: str) -> Optional[str]:
        """Return cached response if present and not expired. None otherwise."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self.misses += 1
                return None
            age = time.time() - entry.timestamp
            if age > self.ttl_sec:
                # Expired
                del self._cache[key]
                self.misses += 1
                return None
            self.hits += 1
            self.tokens_saved += entry.token_estimate
            return entry.response

    def set(self, key: str, response: str, token_estimate: int = 0) -> None:
        """Store a response. Evicts oldest entries when at capacity."""
        with self._lock:
            # Evict expired + enforce max size
            now = time.time()
            if len(self._cache) >= self.max_entries:
                # Remove oldest entries first
                sorted_items = sorted(
                    self._cache.items(), key=lambda kv: kv[1].timestamp
                )
                # Remove oldest 20%
                for k, _ in sorted_items[: max(1, len(sorted_items) // 5)]:
                    del self._cache[k]
            self._cache[key] = CacheEntry(
                response=response,
                timestamp=now,
                token_estimate=token_estimate,
            )

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0
            self.tokens_saved = 0

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                "entries": len(self._cache),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 3) if total else 0.0,
                "tokens_saved_est": self.tokens_saved,
                "ttl_sec": self.ttl_sec,
            }


# ── Singleton ───────────────────────────────────────────────────────

_CACHE: Optional[LLMCache] = None
_CACHE_LOCK = threading.Lock()


def get_llm_cache() -> LLMCache:
    global _CACHE
    if _CACHE is None:
        with _CACHE_LOCK:
            if _CACHE is None:
                _CACHE = LLMCache(ttl_sec=300, max_entries=200)
    return _CACHE
