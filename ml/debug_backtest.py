"""
debug_backtest.py — inspect why catboost / random_forest backtest fails
with 'float' object is not subscriptable.

Run from D:\\Projects\\forex :
    python debug_backtest.py
"""
from utils.safe_pickle import safe_pickle_load
import numpy as np
import pickle
from pathlib import Path

SYMBOL = "EURUSD"

# Try common cache locations for the phase6 dataset split.
# Adjust CACHE_CANDIDATES if none of these exist on your machine.
CACHE_CANDIDATES = [
    Path(f"data/cache/{SYMBOL}_dataset.pkl"),
    Path(f"data/cache/{SYMBOL}/dataset.pkl"),
    Path(f"data/datasets/{SYMBOL}.pkl"),
    Path(f"data/processed/{SYMBOL}_dataset.pkl"),
]

split_path = None
for c in CACHE_CANDIDATES:
    if c.exists():
        split_path = c
        break

if split_path is None:
    print("Could not auto-find the phase6 dataset cache. Searched:")
    for c in CACHE_CANDIDATES:
        print(" -", c.resolve())
    print("\nSearching data/ for likely candidates instead...")
    for p in Path("data").rglob("*.pkl"):
        if SYMBOL.lower() in p.name.lower():
            print(" found:", p)
    raise SystemExit(1)

print(f"Using dataset cache: {split_path}")
with open(split_path, "rb") as f:
    split = pickle.load(f)

X_test = np.nan_to_num(split.test[split.feature_columns].values, nan=0.0)
print(f"X_test shape: {X_test.shape}")

for model_type in ["catboost", "random_forest", "xgboost", "lightgbm"]:
    model_path = Path(f"data/trained_models/{SYMBOL}/{model_type}/model.pkl")
    if not model_path.exists():
        print(f"\n[{model_type}] model.pkl not found at {model_path.resolve()}")
        continue

    model = safe_pickle_load(str(model_path))
    predictions = model.predict(X_test)

    print(f"\n[{model_type}]")
    print("  raw type:", type(predictions))
    print("  raw dtype:", getattr(predictions, "dtype", "N/A"))
    print("  raw shape:", getattr(predictions, "shape", "N/A"))
    print("  sample (first 5):", predictions[:5])

    # This mimics what phase11_backtest.py does:
    flat = np.atleast_1d(np.asarray(predictions, dtype=int)).ravel()
    print("  after atleast_1d+ravel shape:", flat.shape)
    print("  after flatten sample:", flat[:5])