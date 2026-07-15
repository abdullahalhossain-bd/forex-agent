#!/usr/bin/env python3
"""
ML Model Training Script — Forex AI Trading System

Trains XGBoost + Random Forest models for all configured pairs using
historical data from the FeatureStore.  Replaces the synthetic seed
models with real trained models.

Usage:
    python scripts/train_models.py                  # train all pairs
    python scripts/train_models.py --pair EURUSD    # train one pair
    python scripts/train_models.py --min-samples 200  # require more data
"""
import argparse
import sys
import os
import time

# Ensure project root is on path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Train ML models for Forex AI bot")
    parser.add_argument("--pair", type=str, default=None,
                        help="Train only this pair (e.g. EURUSD). Default: all pairs.")
    parser.add_argument("--timeframe", type=str, default="15m",
                        help="Timeframe (default: 15m)")
    parser.add_argument("--min-samples", type=int, default=100,
                        help="Minimum samples required to train (default: 100)")
    args = parser.parse_args()

    print("=" * 60)
    print("  ML Model Training — Forex AI Trading System")
    print("=" * 60)
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Timeframe    : {args.timeframe}")
    print(f"  Min samples  : {args.min_samples}")
    print()

    # Import after path setup
    try:
        from ml.model_trainer import ModelTrainer
        from utils.logger import get_logger
    except ImportError as e:
        print(f"FAIL: Cannot import ML modules: {e}")
        print("Make sure you're running from the project root.")
        sys.exit(1)

    log = get_logger("train_models")
    trainer = ModelTrainer()

    # Determine pairs to train
    if args.pair:
        pairs = [args.pair.upper()]
    else:
        try:
            from config import SYMBOLS
            pairs = [s.upper() for s in SYMBOLS]
        except Exception:
            pairs = ["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD", "XAUUSD"]

    print(f"  Pairs to train: {', '.join(pairs)}")
    print()

    # Train each pair
    total_start = time.time()
    success_count = 0
    fail_count = 0

    for pair in pairs:
        print("-" * 60)
        print(f"  Training {pair} {args.timeframe}...")
        t0 = time.time()
        try:
            result = trainer.train_all(
                pair=pair,
                timeframe=args.timeframe,
                min_samples=args.min_samples,
            )
            elapsed = time.time() - t0

            if result.errors:
                print(f"  ERRORS for {pair}:")
                for err in result.errors:
                    print(f"    - {err}")
                fail_count += 1
            elif result.models_trained:
                print(f"  OK: {pair} trained in {elapsed:.1f}s")
                print(f"    Models: {', '.join(result.models_trained)}")
                for model_name, metrics in result.metrics.items():
                    acc = metrics.get("accuracy", 0)
                    n = metrics.get("n_samples", 0)
                    print(f"    {model_name}: accuracy={acc:.1%} (n={n})")
                success_count += 1
            else:
                print(f"  SKIP: {pair} — no models trained (insufficient data?)")
                fail_count += 1
        except Exception as e:
            print(f"  FAIL: {pair} — {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1

    # Summary
    total_elapsed = time.time() - total_start
    print()
    print("=" * 60)
    print(f"  Training Complete in {total_elapsed:.1f}s")
    print(f"  Success: {success_count} | Failed: {fail_count}")
    print("=" * 60)

    if fail_count > 0:
        print()
        print("Tips for failed pairs:")
        print("  1. Ensure MT5 is connected so historical data can be fetched")
        print("  2. Run the bot once to populate FeatureStore with candle data")
        print("  3. Lower --min-samples if dataset is small")
        print("  4. Check logs/training.log for detailed errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
