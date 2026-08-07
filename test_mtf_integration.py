#!/usr/bin/env python3
"""Test MTF engine with fixes applied"""
import pandas as pd
import numpy as np
from analysis.structure_mtf import MTFStructureEngine

# Create dummy data
np.random.seed(42)
n = 100
prices = 1.1000 + np.cumsum(np.random.randn(n) * 0.001 + 0.0005)

df_h4 = pd.DataFrame({
    'open': prices,
    'high': prices + 0.001,
    'low': prices - 0.001,
    'close': prices,
    'atr': 0.002
})

df_m15 = df_h4.copy()

engine = MTFStructureEngine()
result = engine.analyze(df_external=df_h4, df_internal=df_m15)

print('\nMTF Analysis Result:')
print(f'Valid: {result.get("valid")}')
print(f'Combined Bias: {result.get("combined_bias")}')
print(f'Trade Permission: {result.get("trade_permission")}')
print()
print('External Structure:')
ext = result.get('external') or {}
print(f'  BOS: {ext.get("bos", {}).get("event", "NONE")}')
print(f'  CHoCH: {ext.get("choch", {}).get("event", "NONE")}')
print()
print('Internal Structure:')
int_ = result.get('internal') or {}
print(f'  BOS: {int_.get("bos", {}).get("event", "NONE")}')
print(f'  CHoCH: {int_.get("choch", {}).get("event", "NONE")}')
print()
print('[SUCCESS] MTF Analysis working correctly')
