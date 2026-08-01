"""
ml/dataset_builder.py — Training dataset assembler (Day 69)
=============================================================

Assembles ML-ready training datasets from the FeatureStore + historical
market data. Handles:
  1. Loading features from the store
  2. Generating labels via LabelGenerator (if not already labeled)
  3. Chronological train/validation/test split (70/15/15)
  4. Returning clean DataFrames ready for model training

CRITICAL: All splits are chronological (no shuffle) to prevent future leakage.
The most recent 15% of data is ALWAYS the test set — never used in training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import os

from config import LOG_DIR, MIN_TRAINING_SAMPLES

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("dataset_builder")


@dataclass
class Dataset:
    """A chronologically-split ML dataset."""
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    feature_names: List[str]
    pair: str
    timeframe: str
    train_size: int
    val_size: int
    test_size: int
    label_distribution: Dict[str, Any]
    # NEW (Priority #1 — leakage audit): both fields default to values that
    # reproduce current behavior for every existing caller.
    #   sample_weight: None unless labeling_method="triple_barrier", in
    #     which case it holds compute_label_uniqueness() weights aligned to
    #     X_train.index. model_trainer.py passes this to fit() only when
    #     it is not None — XGBoost/RandomForest with sample_weight=None is
    #     identical to not passing the kwarg at all.
    #   labeling_method / cv_method: tags persisted alongside the model so
    #     ModelStore/ValidationEngine can tell which pipeline produced a
    #     given model (see ml/feature_store.py, ml/validation.py changes).
    sample_weight: Optional[pd.Series] = None
    labeling_method: str = "fixed_horizon"
    cv_method: str = "naive_chronological"
    purge_stats: Optional[Dict[str, Any]] = None

    def summary(self) -> Dict[str, Any]:
        return {
            "pair": self.pair,
            "timeframe": self.timeframe,
            "train_size": self.train_size,
            "val_size": self.val_size,
            "test_size": self.test_size,
            "n_features": len(self.feature_names),
            "label_distribution": self.label_distribution,
            "labeling_method": self.labeling_method,
            "cv_method": self.cv_method,
            "purge_stats": self.purge_stats,
        }


class DatasetBuilder:
    """Builds chronologically-split training datasets."""

    def __init__(self, train_pct: float = 0.70, val_pct: float = 0.15):
        self.train_pct = train_pct
        self.val_pct = val_pct
        # test_pct = 1 - train_pct - val_pct = 0.15

    def build_from_store(
        self,
        pair: str,
        timeframe: str = "15m",
        min_samples: int = None,
        labeling_method: str = "fixed_horizon",
        use_purged_split: bool = False,
        label_horizon: int = 0,
    ) -> Optional[Dataset]:
        """Load features + labels from the FeatureStore and split.

        New optional params (Priority #1, all default to current behavior):
          labeling_method: "fixed_horizon" (current, default) | "triple_barrier"
          use_purged_split: if True, purge boundary-overlapping rows using
            `label_horizon`. Default False reproduces the exact current
            iloc-slice output.
          label_horizon: bars a label's window looks forward. Required
            (non-zero) when use_purged_split=True; ignored otherwise.
        """
        try:
            from ml.feature_store import get_feature_store
            store = get_feature_store()
            # If min_samples not provided, fall back to global config
            min_samples_call = min_samples if min_samples is not None else MIN_TRAINING_SAMPLES
            df = store.load_training_data(pair=pair, timeframe=timeframe, min_samples=min_samples_call)
        except Exception as e:
            log.error(f"[DatasetBuilder] FeatureStore load failed: {e}")
            return None

        if df.empty:
            log.warning(f"[DatasetBuilder] No data for {pair} {timeframe}")
            return None

        return self.build_from_dataframe(
            df, pair=pair, timeframe=timeframe, min_samples=min_samples,
            labeling_method=labeling_method, use_purged_split=use_purged_split,
            label_horizon=label_horizon,
        )

    def build_from_dataframe(
        self,
        df: pd.DataFrame,
        pair: str = "EURUSD",
        timeframe: str = "15m",
        min_samples: int = None,
        labeling_method: str = "fixed_horizon",
        use_purged_split: bool = False,
        label_horizon: int = 0,
    ) -> Optional[Dataset]:
        """Split a feature dataframe into train/val/test.

        labeling_method="fixed_horizon", use_purged_split=False (the
        defaults) reproduce the exact current behavior — zero breaking
        change for any existing caller. Set labeling_method="triple_barrier"
        to relabel `df` via ml.triple_barrier_labels.TripleBarrierLabeler
        first (df must have high/low/close columns in that case). Set
        use_purged_split=True + label_horizon>0 to purge label-window-
        overlap leakage at the train/val and val/test boundaries.
        """
        if labeling_method not in ("fixed_horizon", "triple_barrier"):
            log.error(f"[DatasetBuilder] unknown labeling_method={labeling_method!r}")
            return None

        if labeling_method == "triple_barrier":
            required_cols = {"high", "low", "close"}
            if not required_cols.issubset(df.columns):
                log.error(
                    f"[DatasetBuilder] labeling_method='triple_barrier' requires "
                    f"{required_cols} columns, missing: {required_cols - set(df.columns)}"
                )
                return None
            from ml.triple_barrier_labels import get_triple_barrier_labeler
            labeler = get_triple_barrier_labeler()
            # Pass label_horizon as a per-call override (NOT a mutation of
            # labeler.holding_period) — get_triple_barrier_labeler() is a
            # shared singleton, and mutating shared state here would race
            # under concurrent training of multiple pairs/timeframes.
            df = labeler.label_dataframe(df, pair=pair, holding_period=label_horizon or None)
            if label_horizon <= 0:
                label_horizon = labeler.holding_period

        if "label" not in df.columns:
            log.error("[DatasetBuilder] no 'label' column in dataframe")
            return None

        # sample_weight (present only when labeling_method="triple_barrier")
        # must be captured before meta-column stripping and excluded from
        # the feature matrix — it's a per-row weight, not a predictor.
        sample_weight_full = df.get("sample_weight")

        # Drop meta columns + gather diagnostics before removing NaNs
        meta_cols = [c for c in df.columns if c.startswith("_") or c in
                     ("outcome", "pnl_usd", "forward_pips", "label_ternary",
                      "label_forward_return", "label_forward_pips",
                      "label_mae_pips", "label_mfe_pips", "label_r_multiple",
                      "sample_weight")]
        feature_df_pre = df.drop(columns=meta_cols, errors="ignore").copy()

        total_rows = len(feature_df_pre)
        # Replace inf with NaN for diagnostics
        inf_mask = feature_df_pre.replace([np.inf, -np.inf], np.nan)
        missing_counts = inf_mask.isna().sum().to_dict()
        missing_ratio = {k: float(v) / total_rows if total_rows > 0 else 0.0 for k, v in missing_counts.items()}
        features_all_nan = [k for k, v in missing_counts.items() if v == total_rows]
        features_missing_over_80 = [k for k, v in missing_ratio.items() if v >= 0.8]

        # Separate label from feature columns
        labels_full = df.get("label")
        label_present_count = int(labels_full.notna().sum()) if labels_full is not None else 0

        # Drop features with too many missing values (>=80%) to avoid destroying usable rows
        cols_to_drop = [c for c in features_missing_over_80]
        features_only = feature_df_pre.drop(columns=["label"], errors="ignore")
        if cols_to_drop:
            features_only = features_only.drop(columns=cols_to_drop, errors="ignore")

        # Now keep only rows that have labels (can't train without labels)
        if labels_full is None:
            log.warning("[DatasetBuilder] 'label' column missing after meta drop — cannot build dataset")
            return None
        idx_label_present = labels_full.notna()

        # Drop remaining rows with any NaN in the retained feature columns
        feature_df = features_only.loc[idx_label_present].replace([np.inf, -np.inf], np.nan).dropna()

        # Attach labels aligned to feature_df index
        labels = labels_full.loc[feature_df.index].astype(int)

        # Duplicates and zero-variance diagnostics
        duplicate_count = int(feature_df.duplicated().sum()) if len(feature_df) > 0 else 0
        zero_variance = [c for c in feature_df.columns if feature_df[c].std(ddof=0) == 0]

        # Determine min_samples threshold
        min_samples_use = min_samples if min_samples is not None else MIN_TRAINING_SAMPLES
        if len(feature_df) < min_samples_use:
            log.warning(f"[DatasetBuilder] only {len(feature_df)} usable samples — need ≥{min_samples_use}")
            # Write diagnostics to logs for inspection
            try:
                ml_log_dir = Path(LOG_DIR) / "ml"
                ml_log_dir.mkdir(parents=True, exist_ok=True)
                report = {
                    "pair": pair,
                    "timeframe": timeframe,
                    "total_rows": int(total_rows),
                    "label_present_count": int(label_present_count),
                    "usable_rows": int(len(feature_df)),
                    "min_samples_required": int(min_samples_use),
                    "features_all_nan": features_all_nan,
                    "features_missing_over_80pct": features_missing_over_80,
                    "missing_counts": {k: int(v) for k, v in missing_counts.items()},
                    "dropped_columns": cols_to_drop,
                    "duplicate_rows": duplicate_count,
                    "zero_variance_columns": zero_variance,
                }
                fname = ml_log_dir / f"ml_dataset_report_{pair}_{timeframe}.json"
                with open(fname, "w", encoding="utf8") as f:
                    json.dump(report, f, indent=2)
                txt = ml_log_dir / f"ml_dataset_report_{pair}_{timeframe}.txt"
                with open(txt, "w", encoding="utf8") as f:
                    f.write(json.dumps(report, indent=2))
            except Exception:
                log.exception("[DatasetBuilder] failed to write ML dataset diagnostics")
            return None

        n = len(feature_df)
        train_end = int(n * self.train_pct)
        val_end = int(n * (self.train_pct + self.val_pct))

        cv_method = "naive_chronological"
        purge_stats_dict: Optional[Dict[str, Any]] = None

        if use_purged_split:
            if label_horizon <= 0:
                log.warning(
                    "[DatasetBuilder] use_purged_split=True but label_horizon<=0 — "
                    "no forward-looking window is known, so purging is a no-op. "
                    "Pass the label horizon (e.g. label_horizon=48) to purge correctly."
                )
            from ml.cv_splitter import PurgedEmbargoedSplitter
            splitter = PurgedEmbargoedSplitter(label_horizon=label_horizon)
            train_idx, val_idx, test_idx, purge_stats = splitter.purge_train_val_test(
                n=n, train_end=train_end, val_end=val_end,
            )
            if len(train_idx) == 0:
                log.error(
                    f"[DatasetBuilder] purged split left an EMPTY training set for "
                    f"{pair} {timeframe} (label_horizon={label_horizon}). Refusing to "
                    f"return a dataset with zero training rows."
                )
                return None

            X_train = feature_df.iloc[train_idx]
            X_val = feature_df.iloc[val_idx]
            X_test = feature_df.iloc[test_idx]
            y_train = labels.iloc[train_idx]
            y_val = labels.iloc[val_idx]
            y_test = labels.iloc[test_idx]
            cv_method = "purged_embargoed"
            purge_stats_dict = {
                "original_train_size": purge_stats.original_train_size,
                "purged_train_size": purge_stats.purged_train_size,
                "rows_purged": purge_stats.rows_purged,
                "purge_ratio": round(purge_stats.purge_ratio, 4),
                "label_horizon": label_horizon,
            }
            log.info(
                f"[DatasetBuilder] purged split: dropped {purge_stats.rows_purged} "
                f"boundary-overlapping rows ({purge_stats.purge_ratio:.1%} of naive train size)"
            )
        else:
            X_train = feature_df.iloc[:train_end]
            X_val = feature_df.iloc[train_end:val_end]
            X_test = feature_df.iloc[val_end:]
            y_train = labels.iloc[:train_end]
            y_val = labels.iloc[train_end:val_end]
            y_test = labels.iloc[val_end:]

        sample_weight_train: Optional[pd.Series] = None
        if sample_weight_full is not None:
            sample_weight_train = sample_weight_full.reindex(X_train.index).fillna(1.0)

        # Label distribution
        def _dist(y):
            vc = y.value_counts().to_dict()
            return {str(int(k)): int(v) for k, v in vc.items()}

        label_dist = {
            "train": _dist(y_train),
            "val": _dist(y_val),
            "test": _dist(y_test),
        }

        log.info(
            f"[DatasetBuilder] {pair} {timeframe}: "
            f"train={len(X_train)}, val={len(X_val)}, test={len(X_test)}, "
            f"features={len(feature_df.columns)}"
        )

        return Dataset(
            X_train=X_train, X_val=X_val, X_test=X_test,
            y_train=y_train, y_val=y_val, y_test=y_test,
            feature_names=list(feature_df.columns),
            pair=pair, timeframe=timeframe,
            train_size=len(X_train), val_size=len(X_val), test_size=len(X_test),
            label_distribution=label_dist,
            sample_weight=sample_weight_train,
            labeling_method=labeling_method,
            cv_method=cv_method,
            purge_stats=purge_stats_dict,
        )


# ── Singleton ───────────────────────────────────────────────────────

_BUILDER: Optional[DatasetBuilder] = None


def get_dataset_builder() -> DatasetBuilder:
    global _BUILDER
    if _BUILDER is None:
        _BUILDER = DatasetBuilder()
    return _BUILDER
