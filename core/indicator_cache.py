# core/indicator_cache.py
import hashlib
import time
from typing import Any, Dict, Optional
import pandas as pd

class IndicatorCache:
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._timestamps: Dict[str, float] = {}
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0

    def _generate_key(self, symbol: str, timeframe: str, indicator_name: str, 
                      params: Dict, data_hash: str) -> str:
        param_str = str(sorted(params.items()))
        raw_key = f"{symbol}:{timeframe}:{indicator_name}:{param_str}:{data_hash}"
        return hashlib.md5(raw_key.encode()).hexdigest()

    def get(self, symbol: str, timeframe: str, indicator_name: str, 
            params: Dict, data_hash: str) -> Optional[Any]:
        key = self._generate_key(symbol, timeframe, indicator_name, params, data_hash)
        
        if key in self._cache:
            if time.time() - self._timestamps.get(key, 0) > self._ttl:
                del self._cache[key]
                del self._timestamps[key]
                return None
            
            self._hits += 1
            return self._cache[key]
        
        self._misses += 1
        return None

    def set(self, symbol: str, timeframe: str, indicator_name: str, 
            params: Dict, data_hash: str, value: Any):
        key = self._generate_key(symbol, timeframe, indicator_name, params, data_hash)
        
        if len(self._cache) >= self._max_size:
            oldest_key = min(self._timestamps, key=self._timestamps.get)
            del self._cache[oldest_key]
            del self._timestamps[oldest_key]
        
        self._cache[key] = value
        self._timestamps[key] = time.time()

    def invalidate(self, symbol: str, timeframe: str):
        keys_to_delete = [k for k in self._cache.keys() 
                         if k.startswith(f"{symbol}:{timeframe}")]
        for key in keys_to_delete:
            del self._cache[key]
            del self._timestamps[key]

    def get_stats(self) -> Dict:
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_percent": round(hit_rate, 2),
            "cache_size": len(self._cache)
        }

# গ্লোবাল ইন্সট্যান্স
global_indicator_cache = IndicatorCache()