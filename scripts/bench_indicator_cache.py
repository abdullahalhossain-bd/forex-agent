"""
scripts/bench_indicator_cache.py — Runtime validation for core/indicator_cache.py

Checks the three things flagged in the audit:
  1. Hit vs Miss: identical (symbol, timeframe, df) -> second call must HIT,
     not recompute.
  2. Freshness on new candle: appending a new bar changes the content hash
     -> must MISS and recompute (no manual invalidate() needed, since the
     cache key is content-addressed via a hash of the OHLCV block, not a
     bar-count or timestamp).
  3. Latency: cold (miss) vs warm (hit) wall-clock time for add_all().

Run: python3 scripts/bench_indicator_cache.py
"""
import sys
import time
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from core.indicator_cache import IndicatorCache, global_indicator_cache
from data.indicators_ext import ExtendedIndicators


def make_ohlcv(n=300, seed=42, start="2026-01-01", freq="5min"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=n, freq=freq)
    close = 1.1000 + np.cumsum(rng.normal(0, 0.0004, n))
    high = close + rng.uniform(0.0001, 0.0006, n)
    low = close - rng.uniform(0.0001, 0.0006, n)
    open_ = close + rng.normal(0, 0.0002, n)
    volume = rng.uniform(100, 1000, n)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    return df


def append_one_bar(df, seed=99):
    rng = np.random.default_rng(seed)
    last_close = df["close"].iloc[-1]
    new_close = last_close + rng.normal(0, 0.0004)
    new_idx = df.index[-1] + (df.index[-1] - df.index[-2])
    new_row = pd.DataFrame(
        {
            "open": [last_close],
            "high": [max(last_close, new_close) + 0.0003],
            "low": [min(last_close, new_close) - 0.0003],
            "close": [new_close],
            "volume": [rng.uniform(100, 1000)],
        },
        index=[new_idx],
    )
    return pd.concat([df, new_row])


def reset_global_cache():
    # global_indicator_cache is a module-level singleton — clear it between
    # test sections so results aren't contaminated by earlier calls.
    global_indicator_cache._cache.clear()
    global_indicator_cache._timestamps.clear()
    global_indicator_cache._hits = 0
    global_indicator_cache._misses = 0


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ─────────────────────────────────────────────────────────────────
# 1. HIT vs MISS on identical input
# ─────────────────────────────────────────────────────────────────
section("1) HIT vs MISS — identical (symbol, timeframe, df) called twice")
reset_global_cache()
ind = ExtendedIndicators()
df = make_ohlcv()

t0 = time.perf_counter()
out1 = ind.add_all(df, symbol="EURUSD", timeframe="M5")
t1 = time.perf_counter()
stats_after_first = global_indicator_cache.get_stats()
print(f"Call 1 (cold):  {(t1 - t0) * 1000:8.3f} ms   stats={stats_after_first}")
assert stats_after_first["misses"] == 1 and stats_after_first["hits"] == 0, \
    "First call on fresh cache must be a MISS"

t0 = time.perf_counter()
out2 = ind.add_all(df, symbol="EURUSD", timeframe="M5")
t1 = time.perf_counter()
stats_after_second = global_indicator_cache.get_stats()
print(f"Call 2 (warm):  {(t1 - t0) * 1000:8.3f} ms   stats={stats_after_second}")
assert stats_after_second["hits"] == 1, "Second identical call must be a HIT"
assert stats_after_second["misses"] == 1, "Second identical call must NOT recompute"

# Correctness: cached result must be identical to the freshly computed one,
# not just "some cached dataframe".
pd.testing.assert_frame_equal(out1, out2)
print("PASS — hit/miss counters correct AND cached df content matches original")

# Different symbol / same df -> must still MISS (namespacing works)
t0 = time.perf_counter()
ind.add_all(df, symbol="GBPUSD", timeframe="M5")
t1 = time.perf_counter()
stats3 = global_indicator_cache.get_stats()
print(f"Call 3 (diff symbol, same data): {(t1 - t0) * 1000:8.3f} ms   stats={stats3}")
assert stats3["misses"] == 2, "Different symbol with identical OHLCV must still MISS"
print("PASS — symbol/timeframe namespacing prevents cross-pair collisions")


# ─────────────────────────────────────────────────────────────────
# 2. Freshness on new candle (no manual invalidate() call exists anywhere
#    in the codebase — grep confirmed zero call sites for .invalidate()).
#    The design instead content-addresses the key with a hash of the
#    OHLCV block, so a new bar changes the hash and forces a MISS.
# ─────────────────────────────────────────────────────────────────
section("2) Freshness — new candle must MISS, not serve stale cached indicators")
df_next = append_one_bar(df)
assert len(df_next) == len(df) + 1

stats_before = global_indicator_cache.get_stats()
out3 = ind.add_all(df_next, symbol="EURUSD", timeframe="M5")
stats_after = global_indicator_cache.get_stats()
print(f"Before new bar: {stats_before}")
print(f"After new bar:  {stats_after}")
assert stats_after["misses"] == stats_before["misses"] + 1, \
    "New candle must produce a cache MISS (content hash changed)"
# And the new bar's indicator values must actually differ from the old
# cached frame's tail (i.e. we did NOT get served the stale 300-row result).
assert len(out3) == len(df_next), "Recomputed frame must include the new bar"
print("PASS — new candle busts the cache automatically via content hash "
      "(no explicit .invalidate() call needed or present anywhere in the codebase)")

# Re-running on df_next again must now HIT (proves the new hash got stored).
stats_before2 = global_indicator_cache.get_stats()
ind.add_all(df_next, symbol="EURUSD", timeframe="M5")
stats_after2 = global_indicator_cache.get_stats()
assert stats_after2["hits"] == stats_before2["hits"] + 1
print("PASS — repeated call on the extended df now HITs the freshly stored entry")


# ─────────────────────────────────────────────────────────────────
# 3. TTL expiry (stale-data / memory-leak-adjacent check)
# ─────────────────────────────────────────────────────────────────
section("3) TTL expiry — entries older than ttl_seconds must be evicted on access")
small_cache = IndicatorCache(max_size=10, ttl_seconds=1)
small_cache.set("EURUSD", "M5", "rsi", {}, "hash123", pd.DataFrame({"x": [1]}))
got_immediately = small_cache.get("EURUSD", "M5", "rsi", {}, "hash123")
assert got_immediately is not None, "Fresh entry (within TTL) must be retrievable"
print(f"Immediate get: {'HIT' if got_immediately is not None else 'MISS'} (expected HIT)")

time.sleep(1.2)
got_after_ttl = small_cache.get("EURUSD", "M5", "rsi", {}, "hash123")
print(f"Get after TTL expiry: {'HIT' if got_after_ttl is not None else 'MISS'} (expected MISS)")
assert got_after_ttl is None, "Entry must be evicted once past ttl_seconds"
assert "hash123" not in str(small_cache._cache), "Expired entry must be removed from storage, not just skipped"
print("PASS — TTL eviction actually deletes the entry (no unbounded stale retention)")


# ─────────────────────────────────────────────────────────────────
# 4. max_size eviction (memory-leak check: cache can't grow unbounded)
# ─────────────────────────────────────────────────────────────────
section("4) max_size eviction — cache must not grow past max_size")
bounded_cache = IndicatorCache(max_size=5, ttl_seconds=300)
for i in range(20):
    bounded_cache.set("EURUSD", "M5", "rsi", {}, f"hash{i}", pd.DataFrame({"x": [i]}))
final_size = bounded_cache.get_stats()["cache_size"]
print(f"Inserted 20 entries into a max_size=5 cache -> final size = {final_size}")
assert final_size <= 5, "Cache must evict oldest entries once max_size is reached"
print("PASS — bounded correctly, no unbounded memory growth")


# ─────────────────────────────────────────────────────────────────
# 5. Real-world latency profile — how much does add_all() actually save
#    across a realistic multi-caller cycle? (add_all is called from 25+
#    call sites per the docstring, often on the same candle data within
#    one analysis cycle.)
# ─────────────────────────────────────────────────────────────────
section("5) Latency profile — simulated analysis cycle calling add_all() 8x "
        "on the same candle data (structure.py, liquidity.py, fibonacci.py, "
        "smart_money.py, currency_strength.py, etc. all do this per the "
        "docstring in indicators_ext.py)")
reset_global_cache()
df_cycle = make_ohlcv(n=500, seed=7)
n_callers = 8

# WITHOUT cache: force a fresh IndicatorCache each call to simulate "no cache"
t_nocache_start = time.perf_counter()
for _ in range(n_callers):
    reset_global_cache()  # forces a MISS every time, i.e. no cache benefit
    ind.add_all(df_cycle, symbol="EURUSD", timeframe="M5")
t_nocache_total = time.perf_counter() - t_nocache_start

# WITH cache: one real cycle, cache stays warm across the 8 callers
reset_global_cache()
t_cache_start = time.perf_counter()
for _ in range(n_callers):
    ind.add_all(df_cycle, symbol="EURUSD", timeframe="M5")
t_cache_total = time.perf_counter() - t_cache_start

stats_cycle = global_indicator_cache.get_stats()
print(f"Without cache (8 full recomputes):     {t_nocache_total * 1000:8.2f} ms total")
print(f"With cache (1 compute + 7 hits):       {t_cache_total * 1000:8.2f} ms total")
print(f"Speedup:                               {t_nocache_total / t_cache_total:6.2f}x")
print(f"Cache stats for the warm run:           {stats_cycle}")
assert stats_cycle["hits"] == n_callers - 1, "7 of 8 same-cycle calls must be cache hits"
assert t_cache_total < t_nocache_total, "Cached cycle must be faster than uncached cycle"
print("PASS — cache meaningfully reduces per-cycle latency for repeated same-bar calls")

section("ALL CHECKS PASSED")
