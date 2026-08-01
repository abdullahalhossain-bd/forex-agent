"""
Integration tests for ml.dataset_builder.DatasetBuilder's Priority #1
(purged split + triple-barrier labeling) wiring.

Run with: pytest tests/test_dataset_builder_purged.py -v
"""
import numpy as np
import pandas as pd
import pytest

from ml.dataset_builder import DatasetBuilder


def _synthetic_df(n=500, seed=0, with_label=True):
    rng = np.random.default_rng(seed)
    close = 1.10 + np.cumsum(rng.normal(0, 0.0004, n))
    high = close + rng.uniform(0.0001, 0.0006, n)
    low = close - rng.uniform(0.0001, 0.0006, n)
    df = pd.DataFrame({
        "high": high, "low": low, "close": close,
        "feat1": rng.normal(size=n), "feat2": rng.normal(size=n),
    })
    if with_label:
        label = (rng.random(n) > 0.5).astype(float)
        label[-5:] = np.nan  # NaN tail, matching LabelGenerator's convention
        df["label"] = label
    return df


def test_old_path_unchanged():
    """Default params must reproduce the exact original DatasetBuilder
    behavior — no breaking change for any existing caller."""
    df = _synthetic_df(seed=1)
    builder = DatasetBuilder(train_pct=0.70, val_pct=0.15)
    ds = builder.build_from_dataframe(df, pair="EURUSD", timeframe="15m", min_samples=50)

    assert ds is not None
    assert ds.labeling_method == "fixed_horizon"
    assert ds.cv_method == "naive_chronological"
    assert ds.sample_weight is None
    assert ds.purge_stats is None
    # Split percentages apply AFTER NaN-label rows are dropped (the last 5
    # rows have label=NaN by construction in _synthetic_df), not against
    # the raw dataframe length.
    n_valid = df["label"].notna().sum()
    assert ds.train_size == int(n_valid * 0.70)


def test_purged_split_removes_boundary_rows():
    df = _synthetic_df(seed=2)
    builder = DatasetBuilder(train_pct=0.70, val_pct=0.15)

    ds_old = builder.build_from_dataframe(df.copy(), pair="EURUSD", timeframe="15m", min_samples=50)
    ds_purged = builder.build_from_dataframe(
        df.copy(), pair="EURUSD", timeframe="15m", min_samples=50,
        use_purged_split=True, label_horizon=20,
    )

    assert ds_purged.cv_method == "purged_embargoed"
    assert ds_purged.train_size < ds_old.train_size
    assert ds_purged.purge_stats["rows_purged"] > 0
    # No NaN leakage: purged X_train's max positional index + horizon must
    # not reach into the val split boundary.
    assert ds_purged.train_size == ds_old.train_size - 20


def test_triple_barrier_labeling_end_to_end():
    df = _synthetic_df(seed=3, with_label=False)  # no pre-existing label col
    builder = DatasetBuilder(train_pct=0.70, val_pct=0.15)
    ds = builder.build_from_dataframe(
        df, pair="EURUSD", timeframe="15m", min_samples=50,
        labeling_method="triple_barrier", use_purged_split=True, label_horizon=15,
    )
    assert ds is not None
    assert ds.labeling_method == "triple_barrier"
    assert ds.cv_method == "purged_embargoed"
    assert ds.sample_weight is not None
    # sample_weight must be aligned to X_train's index (not just same length)
    assert list(ds.sample_weight.index) == list(ds.X_train.index)


def test_triple_barrier_requires_ohlc_columns():
    df = pd.DataFrame({"feat1": np.random.randn(200), "feat2": np.random.randn(200)})
    builder = DatasetBuilder()
    ds = builder.build_from_dataframe(
        df, pair="EURUSD", timeframe="15m", min_samples=50, labeling_method="triple_barrier",
    )
    assert ds is None  # must fail loudly (via log.error), not silently proceed


def test_purge_never_produces_empty_training_set_silently():
    """A label_horizon so large it would purge all training rows must
    return None (caller can detect and skip), never a Dataset with 0 rows."""
    df = _synthetic_df(n=60, seed=4)  # tiny dataset
    builder = DatasetBuilder(train_pct=0.70, val_pct=0.15)
    ds = builder.build_from_dataframe(
        df, pair="EURUSD", timeframe="15m", min_samples=10,
        use_purged_split=True, label_horizon=100,  # far exceeds train size
    )
    assert ds is None
