#!/usr/bin/env python3
"""
Test script to verify CHoCH and BOS detection improvements.
"""

import numpy as np
import pandas as pd
from analysis.structure import MarketStructureEngine

def create_bullish_trend_with_reversal():
    """Create a dataset: bullish trend then reversal"""
    n = 250
    prices = 1.1000 + np.cumsum(np.random.randn(n) * 0.001 + 0.0005)
    prices[-30:] = prices[-31] - np.cumsum(np.random.randn(30) * 0.001 - 0.0002)
    
    df = pd.DataFrame({
        "open":  prices,
        "high":  prices + 0.0015,
        "low":   prices - 0.0015,
        "close": prices,
        "atr":   0.002,
    })
    return df

def create_bearish_structure_with_bos():
    """Create a dataset with bearish BOS"""
    n = 200
    prices = 1.1000 - np.cumsum(np.random.randn(n) * 0.001 + 0.0003)
    prices[-1] = prices[-2] - 0.003
    
    df = pd.DataFrame({
        "open":  prices,
        "high":  prices + 0.001,
        "low":   prices - 0.001,
        "close": prices,
        "atr":   0.002,
    })
    return df

def main():
    print("\n" + "="*60)
    print("STRUCTURE ENGINE FIX VALIDATION")
    print("Testing improved CHoCH and BOS detection")
    print("="*60 + "\n")
    
    # Test 1: CHoCH
    print("TEST 1: CHoCH Detection")
    print("-"*60)
    df1 = create_bullish_trend_with_reversal()
    engine1 = MarketStructureEngine(swing_window=5)
    result1 = engine1.analyze(df1)
    
    if result1.get("valid"):
        choch = result1.get("choch", {})
        print(f"Structure: {result1.get('structure')}")
        print(f"CHoCH Event: {choch.get('event')}")
        print(f"CHoCH Confidence: {choch.get('confidence')}")
        print(f"Note: {choch.get('note')}")
        if choch.get('event') != 'NONE':
            print("[PASS] CHoCH detected\n")
        else:
            print("[INFO] CHoCH is NONE\n")
    
    # Test 2: BOS
    print("TEST 2: BOS Detection")
    print("-"*60)
    df2 = create_bearish_structure_with_bos()
    engine2 = MarketStructureEngine(swing_window=5)
    result2 = engine2.analyze(df2)
    
    if result2.get("valid"):
        bos = result2.get("bos", {})
        print(f"Structure: {result2.get('structure')}")
        print(f"BOS Event: {bos.get('event')}")
        print(f"BOS Level: {bos.get('level')}")
        print(f"BOS Confidence: {bos.get('confidence')}")
        print(f"Note: {bos.get('note')}")
        if bos.get('event') != 'NONE':
            print("[PASS] BOS detected\n")
        else:
            print("[INFO] BOS is NONE\n")
    
    # Test 3: AI Context
    print("TEST 3: AI Context Extraction")
    print("-"*60)
    ctx = engine1.get_ai_context(result1)
    none_count = sum(1 for v in ctx.values() if v == "NONE")
    print(f"NONE values in context: {none_count}")
    print(f"Expected: 2 or less (displacement_dir, and possibly structure_bos)")
    
    print("\n" + "="*60)
    print("Validation complete - check results above")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
