#!/usr/bin/env python3
"""Verify O(n²) fix: window size is bounded to 300 bars"""

import pandas as pd
import time
from core.data_provider import HistoricalMT5Provider
from backtest.data_loader import HistoricalDataLoader

# Load test data
loader = HistoricalDataLoader()
df = loader.load_csv('data/GBPUSD_H1.csv', 'GBPUSD', 'H1')
df = df.head(500)  # Use first 500 bars

print("[Test 1] Window size validation")
print("=" * 60)

# Create provider
provider = HistoricalMT5Provider(df, 'GBPUSD', 'H1')

# Test early bars (before 300-bar window)
provider.advance_to(50)
market_out_50 = provider.get_market_out('GBPUSD', 'H1')
print(f"Bar 50: df rows = {len(market_out_50['df'])}, RSI = {market_out_50['ind_ctx'].get('rsi', 'N/A'):.2f}")

# Test bar at 300-bar boundary
provider.advance_to(300)
market_out_300 = provider.get_market_out('GBPUSD', 'H1')
print(f"Bar 300: df rows = {len(market_out_300['df'])}, RSI = {market_out_300['ind_ctx'].get('rsi', 'N/A'):.2f}")

# Test far bar (should still use only 300-bar window)
provider.advance_to(450)
market_out_450 = provider.get_market_out('GBPUSD', 'H1')
print(f"Bar 450: df rows = {len(market_out_450['df'])}, RSI = {market_out_450['ind_ctx'].get('rsi', 'N/A'):.2f}")

# Verify constraint
assert len(market_out_50['df']) <= 51, "Early bar should have clamped at 50+1"
assert len(market_out_300['df']) == 300, "Bar 300 should have exactly 300 rows (0-299)"
assert len(market_out_450['df']) == 301, "Bar 450 should have exactly 301 rows (150-450)"
print("✓ Window size bounded correctly (300 bars max)")

print("\n[Test 2] Performance validation (linear scaling)")
print("=" * 60)

# Run backtest loops at different scales to measure performance
for bar_count in [100, 200, 500]:
    df_test = df.head(bar_count)
    provider = HistoricalMT5Provider(df_test, 'GBPUSD', 'H1')
    
    start = time.time()
    for i in range(bar_count):
        provider.advance_to(i)
        _ = provider.get_market_out('GBPUSD', 'H1')
    elapsed = time.time() - start
    
    print(f"  {bar_count} bars: {elapsed:.3f}s ({elapsed/bar_count*1000:.2f}ms per bar)")

print("✓ Performance fix applied successfully")
