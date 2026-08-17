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
    python scripts/train_models.py --no-bootstrap   # fail instead of
                                                      # padding with synthetic
                                                      # seed rows when real
                                                      # data is insufficient
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
    parser.add_argument("--timeframe", "--tf", type=str, default="15m",
                        help="Timeframe (default: 15m). --tf is an alias.")
    parser.add_argument("--min-samples", type=int, default=100,
                        help="Minimum samples required to train (default: 100)")
    parser.add_argument("--labeling-method", type=str, default="fixed_horizon",
                        choices=["fixed_horizon", "triple_barrier"],
                        help="fixed_horizon (default, UNCHANGED behavior — trains on "
                             "real/bootstrapped closed-trade outcomes from FeatureStore) "
                             "| triple_barrier (uses the per-pair ATR-barrier config "
                             "validated by walk-forward backtest — see "
                             "config/barrier_config.py and ML_SYSTEM_WINRATE_REPORT.md "
                             "for the numbers behind each pair's width). Only 6 pairs "
                             "have a tuned entry; any other pair falls back to an "
                             "untested 1.5x ATR default.")
    parser.add_argument("--use-purged-split", action="store_true",
                        help="Purge/embargo train-val-test boundary leakage (recommended "
                             "with --labeling-method=triple_barrier; see ml/cv_splitter.py)")
    parser.add_argument("--no-bootstrap", action="store_true",
                        help="Disable synthetic seed-row bootstrap. If real market data "
                             "is below --min-samples, the pair fails instead of being "
                             "padded with synthetic rows. Use this for any run whose "
                             "reported accuracy you intend to trust -- bootstrap rows "
                             "are placeholder noise for first-run/dev use only, not real "
                             "market data, and mixing them into a training run inflates "
                             "reported accuracy without meaning anything.")
    args = parser.parse_args()

    print("=" * 60)
    print("  ML Model Training — Forex AI Trading System")
    print("=" * 60)
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Timeframe    : {args.timeframe}")
    print(f"  Min samples  : {args.min_samples}")
    if args.no_bootstrap:
        print(f"  Bootstrap    : disabled (real data only)")
    print()

    # Import after path setup
    try:
        from ml.data_bootstrap import bootstrap_feature_store_if_needed
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
            boot = bootstrap_feature_store_if_needed(
                pair=pair,
                timeframe=args.timeframe,
                min_samples=args.min_samples,
                rows_per_pair=max(200, args.min_samples * 2),
                enabled=not args.no_bootstrap,
            )
            if boot["bootstrapped"]:
                print(f"    Bootstrap: created {boot['seeded']} seed rows for {pair} {args.timeframe} "
                      f"(NOTE: synthetic placeholder data, not real market history)")
            elif boot.get("skipped_reason"):
                print(f"    Bootstrap skipped: {boot['skipped_reason']} "
                      f"(have {boot['rows_before']} real rows, need {args.min_samples})")
            # Only include bootstrap rows in the actual training set if this
            # run just used them to pad insufficient real data. If real data
            # was already sufficient, train real-only even though old
            # bootstrap rows might still exist in the store from other runs.
            result = trainer.train_all(
                pair=pair,
                timeframe=args.timeframe,
                min_samples=args.min_samples,
                labeling_method=args.labeling_method,
                use_purged_split=args.use_purged_split,
                include_bootstrap=bool(boot.get("bootstrapped", False)),
            )
            if boot.get("bootstrapped"):
                print(f"    NOTE: this run's accuracy numbers include synthetic "
                      f"placeholder data — do not treat them as real performance.")
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
        if args.no_bootstrap:
            print("  3. --no-bootstrap is active: real data is below --min-samples for "
                  "this pair. Either fetch more history, or lower --min-samples, or drop "
                  "--no-bootstrap to pad with synthetic placeholder rows (accuracy numbers "
                  "from a bootstrapped run should not be trusted as real performance).")
        else:
            print("  3. If no data has been generated yet, the trainer will now bootstrap a minimal seed dataset")
        print("  4. Lower --min-samples if dataset is small")
        print("  5. Check logs/training.log for detailed errors")
        sys.exit(1)


if __name__ == "__main__":
    main()