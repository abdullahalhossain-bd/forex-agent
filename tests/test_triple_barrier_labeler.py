"""
Unit tests for ml.triple_barrier_labels.TripleBarrierLabeler (Priority #1
— leakage audit: activating the previously-dead triple-barrier module).

Run with: pytest tests/test_triple_barrier_labeler.py -v
"""
import numpy as np
import pandas as pd

from ml.triple_barrier_labels import (
    TripleBarrierLabeler,
    compute_label_uniqueness,
    triple_barrier_labels,
)


def _flat_price_df(n=100, price=1.1000, noise=0.00002, seed=0):
    rng = np.random.default_rng(seed)
    close = price + rng.normal(0, noise, n)
    high = close + noise
    low = close - noise
    return pd.DataFrame({"high": high, "low": low, "close": close})


def test_label_uniqueness_isolated_sample():
    """A single non-overlapping sample should get uniqueness ≈ 1.0."""
    # One isolated event surrounded by NaN (no other overlapping windows)
    labels = pd.Series([np.nan] * 20 + [1] + [np.nan] * 20)
    weights = compute_label_uniqueness(labels, holding_period=5)
    isolated_weight = weights.iloc[20]
    assert isolated_weight == pytest_approx(1.0)


def test_label_uniqueness_fully_overlapping():
    """N samples with identical/heavily-overlapping windows should each get
    a low weight roughly proportional to 1/concurrency."""
    n = 50
    labels = pd.Series(np.ones(n))  # every row is a valid label -> max overlap
    weights = compute_label_uniqueness(labels, holding_period=10)
    # Every weight should be small and roughly uniform (heavy overlap)
    assert (weights <= 1.0).all()
    assert weights.std() < 0.3  # roughly uniform under constant density


def test_triple_barrier_stops_at_sl():
    """A price path that dips through SL before recovering must be
    labeled as a loss, not as if it eventually recovered (this is exactly
    the leak fixed_horizon labeling has and triple-barrier does not)."""
    n = 30
    close = np.full(n, 1.1000)
    high = close.copy()
    low = close.copy()
    # Bar 2: crash through SL (use_atr=False -> width is an absolute price
    # offset; stop_loss_width=0.0025 means the barrier sits at 1.1000-0.0025)
    low[2] = 1.0970
    close[2] = 1.0972
    # Later: price recovers well above TP (would fool a fixed-horizon label
    # that only checks price at t+horizon, ignoring the path taken there)
    close[10:] = 1.1100
    high[10:] = 1.1105

    df = pd.DataFrame({"high": high, "low": low, "close": close})
    labels = triple_barrier_labels(
        df, holding_period=15, take_profit_width=0.0025, stop_loss_width=0.0025,
        use_atr=False,
    )
    # The label at bar 0 must reflect the SL hit (-1), not the later recovery
    assert labels.iloc[0] == -1

    # Sanity cross-check: a flat/quiet path where neither barrier is
    # touched before timeout should agree with a simple "price didn't move"
    # expectation (label 0, timeout) — this is the fixed_horizon-agreement
    # sanity check from the design doc's test list.
    quiet = pd.DataFrame({
        "high": np.full(20, 1.1000), "low": np.full(20, 1.1000), "close": np.full(20, 1.1000),
    })
    quiet_labels = triple_barrier_labels(
        quiet, holding_period=5, take_profit_width=0.01, stop_loss_width=0.01, use_atr=False,
    )
    assert (quiet_labels.dropna() == 0).all()


def test_triple_barrier_labeler_class_matches_functions():
    """TripleBarrierLabeler.label_dataframe() must be consistent with the
    underlying triple_barrier_labels()/compute_label_uniqueness() functions
    it wraps — this is the class ml.dataset_builder.DatasetBuilder actually
    calls, so it must not silently diverge from the tested primitives."""
    df = _flat_price_df(n=150, seed=1)
    labeler = TripleBarrierLabeler(holding_period=10, take_profit_width=2.0, stop_loss_width=2.0)
    out = labeler.label_dataframe(df, pair="EURUSD")

    raw_ternary = triple_barrier_labels(
        df, holding_period=10, take_profit_width=2.0, stop_loss_width=2.0,
    )
    pd.testing.assert_series_equal(
        out["label_ternary"].reset_index(drop=True),
        raw_ternary.reset_index(drop=True),
        check_names=False,
    )

    # Binary convention: 1 only where ternary == 1, 0 for -1/0, NaN preserved
    valid = out["label_ternary"].notna()
    assert ((out.loc[valid, "label"] == 1) == (out.loc[valid, "label_ternary"] == 1)).all()

    # NaN tail matches holding_period
    assert out["label"].isna().sum() == 10

    # sample_weight is present, non-negative, and <=1 on valid-labeled rows
    # (NaN-tail rows may carry 0 weight, since they have no label at all)
    sw_valid = out.loc[valid, "sample_weight"]
    assert (sw_valid > 0).all() and (sw_valid <= 1.0).all()


def pytest_approx(x, rel=1e-6):
    class _Approx:
        def __eq__(self, other):
            return abs(other - x) <= rel * max(abs(x), 1e-9)
    return _Approx()
