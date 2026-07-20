"""
Unit tests for ml.cv_splitter.PurgedEmbargoedSplitter (Priority #1 —
leakage audit / purged & embargoed cross-validation).

Run with: pytest tests/test_cv_splitter.py -v
"""
import numpy as np
import pytest

from ml.cv_splitter import PurgedEmbargoedSplitter


def test_purged_split_backward_compat():
    """label_horizon=0 must reproduce the exact current iloc-slice output."""
    n, train_end, val_end = 1000, 700, 850
    splitter = PurgedEmbargoedSplitter(label_horizon=0)
    train_idx, val_idx, test_idx, stats = splitter.purge_train_val_test(n, train_end, val_end)

    assert list(train_idx) == list(range(0, train_end))
    assert list(val_idx) == list(range(train_end, val_end))
    assert list(test_idx) == list(range(val_end, n))
    assert stats.rows_purged == 0
    assert stats.purge_ratio == 0.0


def test_purged_split_no_overlap():
    """No training-sample label window should extend into the following split."""
    n, train_end, val_end, h = 1000, 700, 850, 48
    splitter = PurgedEmbargoedSplitter(label_horizon=h)
    train_idx, val_idx, test_idx, stats = splitter.purge_train_val_test(n, train_end, val_end)

    assert (train_idx + h < train_end).all()
    assert (val_idx + h < val_end).all()
    assert stats.rows_purged == h  # only the trailing h rows should be dropped
    assert stats.purged_train_size == stats.original_train_size - h


def test_purge_train_val_test_empty_training_set_is_loud():
    """A label_horizon larger than the split itself must not fail silently —
    it should return an empty array (caller decides whether to abort), not
    raise an unguarded exception or silently truncate data elsewhere."""
    splitter = PurgedEmbargoedSplitter(label_horizon=48)
    train_idx, val_idx, test_idx, stats = splitter.purge_train_val_test(n=100, train_end=40, val_end=70)
    assert len(train_idx) == 0
    assert stats.purge_ratio == 1.0


def test_purge_expanding_fold_purges_trailing_rows():
    """Expanding-window fold: purge trims the LAST h rows of train, not
    interior rows (folds are contiguous: train_end == test_start)."""
    splitter = PurgedEmbargoedSplitter(label_horizon=48)
    train_end, test_start, stats = splitter.purge_expanding_fold(
        train_end=500, test_start=500, test_end=550, n=1000, embargo_pct=0.0,
    )
    assert train_end == 500 - 48
    assert stats.rows_purged == 48


def test_purge_expanding_fold_embargo_delays_test_start():
    splitter = PurgedEmbargoedSplitter(label_horizon=0)
    train_end, test_start, stats = splitter.purge_expanding_fold(
        train_end=500, test_start=500, test_end=550, n=1000, embargo_pct=0.01,
    )
    assert test_start == 500 + 10  # 0.01 * 1000
    assert stats.embargo_rows == 10
    assert train_end == 500  # no purge requested (label_horizon=0)


def test_purge_expanding_fold_zero_params_are_noop():
    splitter = PurgedEmbargoedSplitter()
    train_end, test_start, stats = splitter.purge_expanding_fold(
        train_end=500, test_start=500, test_end=550, n=1000,
    )
    assert train_end == 500
    assert test_start == 500
    assert stats.rows_purged == 0
    assert stats.embargo_rows == 0


def test_invalid_embargo_pct_rejected():
    with pytest.raises(ValueError):
        PurgedEmbargoedSplitter(embargo_pct=0.9)


def test_invalid_n_splits_rejected():
    with pytest.raises(ValueError):
        PurgedEmbargoedSplitter(n_splits=0)


def test_generic_kfold_split_respects_embargo():
    splitter = PurgedEmbargoedSplitter(n_splits=4, label_horizon=10, embargo_pct=0.02)
    n = 400
    folds = list(splitter.split(n))
    assert len(folds) == 4
    for train_idx, test_idx in folds:
        # no train index should fall within [test_start - horizon, test_end + embargo)
        test_start, test_end = test_idx.min(), test_idx.max() + 1
        embargo = int(round(0.02 * n))
        forbidden = set(range(max(0, test_start - 10), min(n, test_end + embargo)))
        assert not (set(train_idx.tolist()) & forbidden - set(test_idx.tolist()))
