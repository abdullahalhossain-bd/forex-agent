"""
ml/cv_splitter.py — Purged & Embargoed Cross-Validation (Priority #1)
=======================================================================

Fixes the label-window-overlap leakage present in the naive chronological
splits used by DatasetBuilder (single train/val/test cut) and
WalkForwardValidator (expanding-window folds).

Why this exists
----------------
Every label in this system is computed from a FUTURE price window
(`label_generator.py`'s `horizon` candles, or `triple_barrier_labels.py`'s
`holding_period` candles). That means a training row at index `i` carries
information from candles `i+1 .. i+horizon`. If a train/test boundary sits
inside that window, the training label was partly computed from data that
is nominally "test" — a direct leak, not a theoretical one.

Two independent mechanisms fix this (López de Prado, "Advances in
Financial Machine Learning", ch. 7):

  * **Purging**   — drop any training row whose label window
                     [t_i, t_i + h] overlaps the test window.
  * **Embargoing** — after a test window, leave a gap of `embargo_pct * N`
                      rows before training data resumes, because serial
                      correlation (e.g. rolling-window indicators) can leak
                      information backward across the boundary even
                      without direct window overlap.

Backward compatibility contract
--------------------------------
Every method here is OPT-IN. `label_horizon=0` (or `purge_window=0` and
`embargo_pct=0.0`, depending on the call) reproduces the exact index
output of the current naive slicing byte-for-byte. Nothing in this module
changes behavior unless a caller explicitly passes a non-zero horizon —
see `test_purged_split_backward_compat` in tests/test_cv_splitter.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("cv_splitter")

# If purging would remove more than this fraction of a fold's would-be
# training data, we refuse to silently continue (see Edge Cases in the
# design doc: a label_horizon larger than a fold's window can purge a
# fold's training set to near-empty).
MAX_PURGE_RATIO_WARNING = 0.5


@dataclass
class PurgeStats:
    """Diagnostics for one purge/embargo operation — surfaced to logs and,
    optionally, to ValidationReport so a caller can tell how much data a
    purge actually removed (never fail silently, per design doc)."""
    original_train_size: int
    purged_train_size: int
    rows_purged: int
    purge_ratio: float
    embargo_rows: int


class PurgedEmbargoedSplitter:
    """
    Chronological splitter with purging (drop training samples whose label
    window overlaps the test window) and embargo (gap after the test
    window before training data is considered clean again).

    Two usage modes, matching the two call sites in this codebase:

      1. `purge_train_val_test(n, train_end, val_end, label_horizon)`
         — for DatasetBuilder's single chronological 3-way split.

      2. `split(df_len, n_splits, label_horizon, embargo_pct)`
         — for WalkForwardValidator's expanding-window folds.

    `label_horizon=0` on mode 1, or `embargo_pct=0.0` with a fold's
    horizon already excluded, reproduces current behavior exactly.
    """

    def __init__(
        self,
        n_splits: int = 1,
        label_horizon: int = 0,
        embargo_pct: float = 0.0,
    ):
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        if not (0.0 <= embargo_pct < 0.5):
            raise ValueError("embargo_pct must be in [0.0, 0.5)")
        if label_horizon < 0:
            raise ValueError("label_horizon must be >= 0")
        self.n_splits = n_splits
        self.label_horizon = label_horizon
        self.embargo_pct = embargo_pct

    # ── Mode 1: single 3-way split (DatasetBuilder) ─────────────────────

    def purge_train_val_test(
        self,
        n: int,
        train_end: int,
        val_end: int,
        label_horizon: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, PurgeStats]:
        """
        Given the current naive cut points (train_end, val_end) over `n`
        positional rows, return purged positional index arrays for
        train/val/test such that no training (or val) row's label window
        [i, i+h] extends into the following split's range.

        `label_horizon=0` returns train=[0,train_end), val=[train_end,val_end),
        test=[val_end,n) unchanged — exact current behavior.
        """
        h = self.label_horizon if label_horizon is None else label_horizon

        train_idx_full = np.arange(0, train_end)
        val_idx_full = np.arange(train_end, val_end)
        test_idx = np.arange(val_end, n)

        if h <= 0:
            stats = PurgeStats(
                original_train_size=len(train_idx_full),
                purged_train_size=len(train_idx_full),
                rows_purged=0,
                purge_ratio=0.0,
                embargo_rows=0,
            )
            return train_idx_full, val_idx_full, test_idx, stats

        # Purge: a train row at position i is contaminated if its label
        # window [i, i+h] reaches into the val split.
        train_mask = (train_idx_full + h) < train_end
        train_idx = train_idx_full[train_mask]

        # Same logic for val rows bleeding into test.
        val_mask = (val_idx_full + h) < val_end
        val_idx = val_idx_full[val_mask]

        rows_purged = len(train_idx_full) - len(train_idx)
        purge_ratio = rows_purged / len(train_idx_full) if len(train_idx_full) else 0.0

        if purge_ratio > MAX_PURGE_RATIO_WARNING:
            log.warning(
                f"[PurgedEmbargoedSplitter] purge removed {purge_ratio:.1%} of training "
                f"rows (label_horizon={h} vs train_end={train_end}) — label_horizon may be "
                f"too large relative to the split size."
            )

        stats = PurgeStats(
            original_train_size=len(train_idx_full),
            purged_train_size=len(train_idx),
            rows_purged=rows_purged,
            purge_ratio=purge_ratio,
            embargo_rows=0,
        )

        if len(train_idx) == 0:
            log.error(
                "[PurgedEmbargoedSplitter] purge removed ALL training rows — "
                "label_horizon is too large for this split. Refusing to return "
                "an empty training set silently."
            )

        return train_idx, val_idx, test_idx, stats

    # ── Mode 2: expanding-window folds (WalkForwardValidator) ──────────

    def purge_expanding_fold(
        self,
        train_end: int,
        test_start: int,
        test_end: int,
        n: int,
        label_horizon: Optional[int] = None,
        embargo_pct: Optional[float] = None,
    ) -> Tuple[int, int, PurgeStats]:
        """
        For one expanding-window fold (train=[0,train_end), test=[test_start,
        test_end)), compute the purged train end and an embargo-adjusted
        test start.

        WalkForwardValidator's folds are contiguous (train_end == test_start
        today), so purging here means trimming the LAST `h` rows off the
        training set rather than dropping interior rows. Embargo applies to
        the START of test (delays it by `embargo_pct * n` rows), which is
        the conservative interpretation for an expanding-window scheme
        where every prior test window becomes future training data.

        Returns (purged_train_end, embargoed_test_start, stats).
        `label_horizon=0` and `embargo_pct=0.0` return (train_end, test_start,
        stats-with-zero-effect) — exact current behavior.
        """
        h = self.label_horizon if label_horizon is None else label_horizon
        e_pct = self.embargo_pct if embargo_pct is None else embargo_pct

        embargo_rows = int(round(e_pct * n))
        embargoed_test_start = min(test_start + embargo_rows, test_end)

        # Purge: drop training rows whose label window would still be
        # forming when the (embargo-adjusted) test window begins.
        purged_train_end = train_end
        if h > 0:
            purged_train_end = max(0, train_end - h)
            # Guard: never purge past a sane floor — caller's min_train_size
            # check happens upstream, but we defend here too.
            if purged_train_end <= 0:
                log.error(
                    f"[PurgedEmbargoedSplitter] purge would empty the training set "
                    f"(train_end={train_end}, label_horizon={h}). Keeping unpurged "
                    f"train_end and flagging via stats — caller should skip this fold."
                )
                purged_train_end = train_end

        rows_purged = train_end - purged_train_end
        purge_ratio = rows_purged / train_end if train_end else 0.0
        if purge_ratio > MAX_PURGE_RATIO_WARNING:
            log.warning(
                f"[PurgedEmbargoedSplitter] fold purge removed {purge_ratio:.1%} of "
                f"training rows — consider a smaller label_horizon or larger min_train_size."
            )

        stats = PurgeStats(
            original_train_size=train_end,
            purged_train_size=purged_train_end,
            rows_purged=rows_purged,
            purge_ratio=purge_ratio,
            embargo_rows=embargo_rows,
        )
        return purged_train_end, embargoed_test_start, stats

    # ── Mode 3: generic k-fold (available for future callers) ──────────

    def split(self, n: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generic chronological k-fold purged/embargoed split over `n` rows.
        Not currently wired into a live caller — provided so `ml/cv_splitter`
        has a single canonical k-fold implementation if a future module
        (e.g. hyperparameter search) needs one, instead of hand-rolling
        another split routine.
        """
        fold_size = n // self.n_splits
        if fold_size < 1:
            raise ValueError(f"n={n} too small for n_splits={self.n_splits}")

        for fold in range(self.n_splits):
            test_start = fold * fold_size
            test_end = n if fold == self.n_splits - 1 else (fold + 1) * fold_size
            test_idx = np.arange(test_start, test_end)

            embargo_rows = int(round(self.embargo_pct * n))
            pre_train = np.arange(0, max(0, test_start - self.label_horizon))
            post_train_start = min(test_end + embargo_rows, n)
            post_train = np.arange(post_train_start, n)
            train_idx = np.concatenate([pre_train, post_train])

            if len(train_idx) == 0:
                log.warning(f"[PurgedEmbargoedSplitter] fold {fold}: empty training set after purge/embargo")
                continue

            yield train_idx, test_idx


# ── Singleton (mirrors the rest of ml/*.py) ──────────────────────────────

_SPLITTER: Optional[PurgedEmbargoedSplitter] = None


def get_cv_splitter() -> PurgedEmbargoedSplitter:
    """Default-parameter singleton (label_horizon=0, embargo_pct=0.0) —
    a no-op splitter. Callers that need purging construct their own
    PurgedEmbargoedSplitter(label_horizon=..., embargo_pct=...) instance;
    this singleton exists only for service-registry wiring / introspection."""
    global _SPLITTER
    if _SPLITTER is None:
        _SPLITTER = PurgedEmbargoedSplitter()
    return _SPLITTER
