"""
ml/ensemble_train.py — Retrain base models + report ensemble state
====================================================================

FIX (2026-07-15): `python -m ml.ensemble_train` used to fail with
`No module named ml.ensemble_train` — this module simply didn't exist.

Honesty note: this codebase does NOT have a batch "ensemble trainer" in
the sense of fitting new ensemble weights offline. Looking at
ml/ensemble_store.py and ml/confidence_fusion.py, ensemble voting weights
come from two places instead:
  1. `ml/model_weights.json` — static starting weights per model/voter.
  2. `EnsembleStore.update_model_performance()` — weights are nudged over
     time from *live* trade outcomes recorded during actual trading, not
     from a training script.

So "ensemble retrain" for this system really means: (a) retrain the
underlying XGBoost/RandomForest/LSTM models per pair via ModelTrainer,
and (b) print the current ensemble weights + live performance stats so
you can see what the ensemble is actually doing. That's what this script
does — it does not fabricate a fake "ensemble training" step that isn't
part of the real pipeline.

Usage:
    python -m ml.ensemble_train                 # retrain all configured pairs
    python -m ml.ensemble_train --pair EURUSD    # retrain a single pair
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from utils.logger import get_logger

log = get_logger("ensemble_train")


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrain base ML models and report ensemble weights")
    parser.add_argument("--pair", type=str, default=None, help="Train only this pair (default: all configured pairs)")
    parser.add_argument("--timeframe", type=str, default="15m")
    parser.add_argument("--min-samples", type=int, default=100)
    args = parser.parse_args()

    from ml.model_trainer import train_all

    print("=" * 60)
    print("  Retraining base models")
    print("=" * 60)
    results = train_all(pair=args.pair, timeframe=args.timeframe, min_samples=args.min_samples)

    ok, failed = [], []
    for pair, result in results.items():
        if getattr(result, "errors", None):
            failed.append(pair)
            print(f"  FAIL {pair}: {result.errors}")
        else:
            ok.append(pair)
            print(f"  OK   {pair}: models={getattr(result, 'models_trained', [])}")

    print()
    print("=" * 60)
    print("  Current ensemble weights (ml/model_weights.json)")
    print("=" * 60)
    weights_path = Path(__file__).resolve().parent / "model_weights.json"
    if weights_path.exists():
        print(json.dumps(json.loads(weights_path.read_text()), indent=2))
    else:
        print(f"  (no {weights_path} found)")

    print()
    print("=" * 60)
    print("  Live ensemble performance (ml/ensemble_store.py)")
    print("=" * 60)
    try:
        from ml.ensemble_store import get_ensemble_store
        store = get_ensemble_store()
        print(json.dumps(store.stats(), indent=2, default=str))
        print(json.dumps(store.get_model_performance(), indent=2, default=str))
    except Exception as e:
        print(f"  (could not load ensemble store: {e})")

    print()
    print(f"  Trained OK: {len(ok)}  |  Failed/insufficient data: {len(failed)}")
    if failed:
        print(f"  Failed pairs: {', '.join(failed)}")
        print("  (Weights are NOT retrained by this script — they update live "
              "from real trade outcomes. Failed pairs usually mean not enough "
              "historical feature data exists yet.)")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
