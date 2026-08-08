#!/usr/bin/env python3
"""
Quick diagnostic: how many labeled ML training samples exist per pair/timeframe.

Copy this file into your project root (D:\\Projects\\forex\\) and run:
    python check_feature_store.py
"""
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from ml.feature_store import get_feature_store

store = get_feature_store()
stats = store.stats()

print("=" * 60)
print("  Feature Store Diagnostic")
print("=" * 60)
print(f"  Total feature rows : {stats.get('total_features')}")
print(f"  Total labeled rows : {stats.get('total_labels')}  (label_binary IS NOT NULL)")
print(f"  Total with outcome : {stats.get('total_outcomes')}  (trade actually closed)")
print(f"  Wins / Losses      : {stats.get('wins')} / {stats.get('losses')}")
print()
print("  By pair:")
for pair, count in stats.get("by_pair", []):
    print(f"    {pair:10s} : {count}")
print("=" * 60)
print()
print("Note: training needs >= MIN_TRAINING_SAMPLES (default 100) LABELED rows")
print("for the SPECIFIC pair + timeframe you pass to train_models.py.")
print("A 'WAIT' decision is never labeled (label=None), so only cycles where")
print("the final_signal was BUY or SELL count toward this total.")