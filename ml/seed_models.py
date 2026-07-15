"""
ml/seed_models.py — Bootstrap initial ML models for all symbols (Day 102)
=========================================================================
Generates lightweight XGBoost classifier models seeded with synthetic data
so the ModelStore ensemble pipeline has models to load from day one.

Run once:  python -m ml.seed_models
The models are NOT production-grade — they're placeholders that let the
ensemble pipeline exercise its full code path. Real models are trained
by the LearningAgent after sufficient trade history accumulates.
"""

from __future__ import annotations

import sys
import numpy as np
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import SYMBOLS, DEFAULT_TIMEFRAME
from ml.model_store import get_model_store

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


def seed_xgboost_models() -> dict:
    """Generate and save a seeded XGBoost model for each symbol."""
    if not XGB_AVAILABLE:
        print("[seed_models] xgboost not installed — skipping (pip install xgboost)")
        return {"error": "xgboost not installed"}

    store = get_model_store()
    results = {}

    for symbol in SYMBOLS:
        np.random.seed(hash(symbol) % 2**31)

        # Generate synthetic training data (500 samples, 20 features)
        n_samples = 500
        n_features = 20
        X = np.random.randn(n_samples, n_features)
        # Slightly imbalanced classes (60% DOWN, 40% UP) — realistic for forex
        y = np.random.choice([0, 1], size=n_samples, p=[0.55, 0.45])

        model = XGBClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
        )
        model.fit(X, y)

        # Training metrics on synthetic data
        train_acc = float(np.mean(model.predict(X) == y))

        version = store.save_model(
            model=model,
            pair=symbol,
            timeframe=DEFAULT_TIMEFRAME,
            model_type="xgboost",
            metrics={
                "accuracy": round(train_acc, 4),
                "n_samples": n_samples,
                "n_features": n_features,
                "seeded": True,
                "note": "Synthetic seed — replace with trained model after real data",
            },
        )
        results[symbol] = {"version": version, "accuracy": round(train_acc, 4)}
        print(f"  [OK] {symbol}_{DEFAULT_TIMEFRAME} xgboost {version} (acc={train_acc:.1%})")

    return results


if __name__ == "__main__":
    print(f"[seed_models] Seeding models for {len(SYMBOLS)} symbols...")
    results = seed_xgboost_models()

    if "error" not in results:
        print(f"\n[seed_models] Done — {len(results)} models saved to ModelStore")
        store = get_model_store()
        for m in store.list_models():
            print(f"  {m['key']} → {m['latest']} ({m['versions']} version(s))")
    else:
        print(f"\n[seed_models] Nothing to do: {results.get('error')}")