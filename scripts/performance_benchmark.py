"""Performance benchmark: _enrich / run_full_detection before/after."""
import sys, os, time, json
sys.path.insert(0, '/home/z/my-project/forex-agent')
os.chdir('/home/z/my-project/forex-agent')
import logging
logging.getLogger().setLevel(logging.ERROR)

import pandas as pd
import numpy as np
from analysis.patterns import PatternDetector

# Load full EURUSD H1 data
df_full = pd.read_csv('data/EURUSD_H1.csv', parse_dates=['datetime_utc']).set_index('datetime_utc').rename(columns={'tick_volume': 'volume'})
print(f'Full data: {len(df_full)} bars')

# Also load M15 (larger dataset) for stress test
df_m15 = pd.read_csv('data/EURUSD_M15.csv', parse_dates=['datetime_utc']).set_index('datetime_utc').rename(columns={'tick_volume': 'volume'})
print(f'M15 data: {len(df_m15)} bars')

results = {}
detector = PatternDetector()

for label, df_source, sizes in [
    ('EURUSD_H1', df_full, [100, 500, 1000, 5000, len(df_full)]),
    ('EURUSD_M15', df_m15, [100, 500, 1000, 5000, len(df_m15)]),
]:
    print(f'\n=== {label} ===')
    for n_bars in sizes:
        if n_bars > len(df_source):
            n_bars = len(df_source)
        df = df_source.tail(n_bars).copy()
        print(f'  {n_bars} bars...')

        # Vectorized timing
        t0 = time.time()
        df_vec = detector._run_full_detection_vectorized(df.copy())
        t_vec = time.time() - t0

        # Legacy timing (only for smaller sizes to avoid 60+ second runs)
        if n_bars <= 1000:
            t0 = time.time()
            df_legacy = detector._run_full_detection_legacy(df.copy())
            t_legacy = time.time() - t0
            speedup = t_legacy / t_vec if t_vec > 0 else float('inf')
        else:
            t_legacy = None
            speedup = None

        # Count detected patterns
        pattern_cols = ['pattern', 'engulfing', 'star_pattern', 'three_bar_cont', 'three_bar_rev',
                        'breakout_candle', 'piercing_line', 'harami', 'three_soldiers_crows',
                        'context_pattern', 'dark_cloud_cover', 'doji_variant', 'three_methods',
                        'tweezers', 'ib_false_breakout', 'engulfing_context', 'doji_context', 'harami_context']
        total_patterns = 0
        for col in pattern_cols:
            if col in df_vec.columns:
                total_patterns += (df_vec[col].fillna('none').astype(str) != 'none').sum()

        results[f'{label}_{n_bars}'] = {
            'bars': n_bars,
            'vectorized_seconds': round(t_vec, 3),
            'legacy_seconds': round(t_legacy, 3) if t_legacy is not None else None,
            'speedup': round(speedup, 1) if speedup else None,
            'total_patterns_detected': int(total_patterns),
        }
        print(f'    vectorized: {t_vec:.3f}s | legacy: {t_legacy:.3f}s | speedup: {speedup:.1f}x' if t_legacy else f'    vectorized: {t_vec:.3f}s (legacy skipped)')
        print(f'    patterns detected: {total_patterns}')

# Save
with open('/home/z/my-project/download/performance_benchmark.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\n\n{"="*80}')
print(f'  PERFORMANCE BENCHMARK SUMMARY')
print(f'{"="*80}')
print(f'  {"Test":<25} {"Bars":>8} {"Vectorized":>12} {"Legacy":>12} {"Speedup":>10}')
print(f'  {"-"*25} {"-"*8} {"-"*12} {"-"*12} {"-"*10}')
for k, v in results.items():
    name, n = k.rsplit('_', 1)
    leg = f'{v["legacy_seconds"]:.2f}s' if v['legacy_seconds'] else 'N/A'
    sp = f'{v["speedup"]}x' if v['speedup'] else 'N/A'
    print(f'  {name:<25} {v["bars"]:>8} {v["vectorized_seconds"]:>10.3f}s {leg:>12} {sp:>10}')
print(f'{"="*80}')
print(f'\nSaved: /home/z/my-project/download/performance_benchmark.json')
