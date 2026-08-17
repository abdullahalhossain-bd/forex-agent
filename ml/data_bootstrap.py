from __future__ import annotations

import json
import math
import random
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ml.feature_store import FeatureStore

# Fixed seed so bootstrap runs are reproducible across calls, but the seed
# itself is NOT derived from `index`, so it can't leak into any feature.
_BOOTSTRAP_SEED = 42


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def _make_seed_features(rng: random.Random) -> Dict[str, float]:
    """Generate a synthetic OHLC-like row.

    IMPORTANT: every value here must come from `rng` (independent randomness),
    never from a deterministic function of the row index. If a feature is a
    deterministic function of index and the label is also a deterministic
    function of index, the two end up correlated with each other even though
    neither is a real cause of the other -- that's a label leak, and it lets
    a model reach unrealistically high accuracy on the synthetic rows without
    learning anything about real market behavior.
    """
    close = 1.0 + rng.uniform(-0.01, 0.01)
    open_ = close + rng.uniform(-0.0005, 0.0005)
    high = max(open_, close) + abs(rng.uniform(0.0, 0.0004))
    low = min(open_, close) - abs(rng.uniform(0.0, 0.0004))

    return {
        "price_open": open_,
        "price_high": high,
        "price_low": low,
        "price_close": close,
        "price_volume": rng.uniform(500.0, 5000.0),
        "change_1": rng.uniform(-0.0005, 0.0005),
        "change_3": rng.uniform(-0.0008, 0.0008),
        "change_5": rng.uniform(-0.0012, 0.0012),
        "rsi_14": rng.uniform(20.0, 80.0),
        "sma_20": close + rng.uniform(-0.0006, 0.0006),
        "sma_50": close + rng.uniform(-0.001, 0.001),
        "distance_ma20": rng.uniform(-0.0006, 0.0006),
        "distance_ma50": rng.uniform(-0.001, 0.001),
        "atr_14": rng.uniform(0.00015, 0.00045),
        "macd": rng.uniform(-0.0003, 0.0003),
        "macd_signal": rng.uniform(-0.00025, 0.00025),
        "trend_bias": rng.choice([-1.0, 1.0]),
        "volatility": rng.uniform(0.0002, 0.0006),
        "session_hour": float(rng.randint(0, 23)),
        "day_of_week": float(rng.randint(0, 6)),
        "pair_strength": rng.uniform(-1.0, 1.0),
    }


def bootstrap_feature_store_if_needed(
    pair: str,
    timeframe: str = "15m",
    min_samples: int = 100,
    rows_per_pair: int = 200,
    store: Optional[FeatureStore] = None,
    enabled: bool = True,
) -> Dict[str, Any]:
    """Create a minimal training dataset in the FeatureStore when it is empty.

    This avoids instantly failing the CLI with a generic "Insufficient data"
    message during a first run or in a clean environment. The synthetic rows
    are intentionally *pure noise*: every feature and the label are drawn
    independently at random, so a model cannot learn any real pattern from
    them. This is a bootstrap fallback for first-run/dev use only -- it is
    not a replacement for real market data, and should be disabled
    (`enabled=False`, or `--no-bootstrap` on the CLI) for any run whose
    accuracy numbers you intend to trust.

    Args:
        enabled: if False, this is a no-op -- returns immediately without
            writing any synthetic rows, even if the store is under
            min_samples. Callers that care about real-data-only training
            should pass enabled=False and handle the resulting "insufficient
            data" error explicitly, rather than silently training on noise.
    """
    if store is None:
        store = FeatureStore()

    # Only count real ('live') rows when deciding whether to bootstrap —
    # if this counted bootstrap rows too, one bootstrap run would
    # permanently satisfy min_samples for that pair/timeframe forever,
    # even with zero real data, and no one would notice.
    rows = store.load_training_data(pair=pair, timeframe=timeframe, min_samples=0, include_bootstrap=False)
    if len(rows) >= max(1, min_samples):
        return {"bootstrapped": False, "rows_before": len(rows), "rows_after": len(rows)}

    if not enabled:
        return {
            "bootstrapped": False,
            "rows_before": len(rows),
            "rows_after": len(rows),
            "skipped_reason": "bootstrap disabled (--no-bootstrap); real data insufficient",
        }

    # Deterministic-but-independent RNG: same seed every run for reproducibility,
    # but NOT keyed off pair/timeframe/index, so nothing downstream can be
    # reconstructed from row position.
    rng = random.Random(_BOOTSTRAP_SEED)

    seeded = 0
    for _ in range(rows_per_pair):
        # Label is an independent coin flip -- unrelated to any feature value.
        label = rng.choice([0, 1])

        features = _make_seed_features(rng)
        features["high"] = features["price_high"]
        features["low"] = features["price_low"]
        features["close"] = features["price_close"]
        features["open"] = features["price_open"]
        store.save_features(
            pair=pair,
            timeframe=timeframe,
            features=features,
            label=int(label),
            forward_pips=_safe_float(features["price_close"] - features["price_open"]) * 10000.0,
            labeling_method="fixed_horizon",
            sample_weight=1.0,
            source="bootstrap",
        )
        seeded += 1

    rows_after = store.load_training_data(pair=pair, timeframe=timeframe, min_samples=0, include_bootstrap=True)
    return {
        "bootstrapped": True,
        "rows_before": len(rows),
        "rows_after": len(rows_after),
        "seeded": seeded,
    }