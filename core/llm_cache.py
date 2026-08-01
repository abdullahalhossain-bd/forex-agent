"""
core/llm_cache.py — Day 90 LLM Response Cache (Optimized & High Performance)
=============================================================================
Caches LLM responses by (provider, model, prompt-hash) for a short TTL.[cite: 5]
This is critical for long-duration demo trading where the same symbol+[cite: 5]
timeframe+regime combination will produce nearly-identical LLM prompts[cite: 5]
across consecutive cycles — re-calling the LLM just burns tokens.[cite: 5]

Optimizations:
  - Replaced O(N log N) sorting eviction with O(1) OrderedDict queue logic.
  - Reduced thread lock contention overhead during capacity overflow.
  - Maintained complete backward compatibility with identical APIs.

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
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class CacheEntry:
    response: str
    timestamp: float
    token_estimate: int = 0


class LLMCache:
    def __init__(self, ttl_sec: int = 300, max_entries: int = 200):
        self.ttl_sec = ttl_sec
        self.max_entries = max_entries
        
        # High-performance insertion-ordered tracking for O(1) eviction
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Performance Tracking Stats[cite: 5]
        self.hits = 0
        self.misses = 0
        self.tokens_saved = 0

    @staticmethod
    def make_key(provider: str, model: str, prompt: str) -> str:
        """Build a secure, collision-resistant cache key from provider+model+prompt."""
        h = hashlib.sha256()
        h.update(provider.encode("utf-8", errors="ignore"))
        h.update(b"|")
        h.update(model.encode("utf-8", errors="ignore"))
        h.update(b"|")
        h.update(prompt.encode("utf-8", errors="ignore"))
        return h.hexdigest()[:16]

    def get(self, key: str) -> Optional[str]:
        """Return cached response if present and not expired. None otherwise.[cite: 5]"""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self.misses += 1
                return None
            
            age = time.time() - entry.timestamp
            if age > self.ttl_sec:
                # Expired — Remove immediately to reclaim space[cite: 5]
                del self._cache[key]
                self.misses += 1
                return None
            
            # Cache Hit: Refresh position in OrderedDict to signify recent usage (LRU policy)
            self._cache.move_to_end(key)
            self.hits += 1
            self.tokens_saved += entry.token_estimate
            return entry.response

    def set(self, key: str, response: str, token_estimate: int = 0) -> None:
        """Store a response. Proactively evicts oldest entries at capacity without sorting overhead."""
        now = time.time()
        with self._lock:
            # If entry already exists, remove it first to handle rewrite properly
            if key in self._cache:
                del self._cache[key]
            
            # Enforce max size limit reactively[cite: 5]
            if len(self._cache) >= self.max_entries:
                # Proactively clean out expired items first to avoid deleting fresh data
                expired_keys = [
                    k for k, entry in self._cache.items() 
                    if now - entry.timestamp > self.ttl_sec
                ]
                for k in expired_keys:
                    del self._cache[k]
                
                # If still over-capacity, drop the oldest 20% of entries using efficient O(1) pop
                if len(self._cache) >= self.max_entries:
                    evict_count = max(1, len(self._cache) // 5)
                    for _ in range(evict_count):
                        if self._cache:
                            self._cache.popitem(last=False)  # Pops the oldest FIFO item in O(1)
            
            # Write new entry
            self._cache[key] = CacheEntry(
                response=response,
                timestamp=now,
                token_estimate=token_estimate,
            )

    def clear(self) -> None:
        """Completely flushes the memory cache and resets performance monitors."""
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0
            self.tokens_saved = 0

    def stats(self) -> Dict[str, Any]:
        """Return real-time cryptographic caching operational insights."""
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


# ── Thread-Safe Singleton Instance Manager ─────────────────────────

_CACHE: Optional[LLMCache] = None
_CACHE_LOCK = threading.Lock()


def get_llm_cache() -> LLMCache:
    """Thread-safe global singleton constructor for the memory registry."""
    global _CACHE
    if _CACHE is None:
        with _CACHE_LOCK:
            if _CACHE is None:
                _CACHE = LLMCache(ttl_sec=300, max_entries=200)
    return _CACHE