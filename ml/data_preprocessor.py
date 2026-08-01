"""
ml/data_preprocessor.py — Data preprocessor (Day 68)
======================================================

Prepares feature matrices for ML training:
  1. **Leakage prevention** — drops rows with NaN labels, ensures no future
     data in features
  2. **Train/test split** — chronological (no shuffle) to respect time order
  3. **Normalization** — StandardScaler (z-score) per feature, fit on train only
  4. **Outlier handling** — clip features to ±3 std
  5. **Feature matrix export** — saves X_train, X_test, y_train, y_test to disk

CRITICAL RULE: scaler is fit ONLY on training data, then applied to test.
Never fit on the full dataset — that leaks test statistics into training.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger
from utils.safe_pickle import safe_pickle_load as _safe_load
from core.constants import MEMORY_DIR

log = get_logger("data_preprocessor")

PROCESSED_DIR = MEMORY_DIR / "ml_processed"


@dataclass
class ProcessedDataset:
    """Result of preprocessing a feature matrix."""
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: List[str]
    train_size: int
    test_size: int
    scaler_path: Optional[str] = None

    def summary(self) -> Dict[str, Any]:
        return {
            "train_size": self.train_size,
            "test_size": self.test_size,
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names[:20],  # preview
            "y_train_distribution": self.y_train.value_counts().to_dict() if hasattr(self.y_train, "value_counts") else {},
            "y_test_distribution": self.y_test.value_counts().to_dict() if hasattr(self.y_test, "value_counts") else {},
        }


class DataPreprocessor:
    """Prepares data for ML training with strict leakage prevention."""

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = None
        self.feature_means: Dict[str, float] = {}
        self.feature_stds: Dict[str, float] = {}

    def drop_invalid(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 1a: Replace inf with NaN, drop rows with any NaN.

        Safe to run BEFORE the train/test split — this only removes rows,
        it doesn't compute any statistic over the data, so there's no
        train/test information to leak.
        """
        df = df.copy()
        df = df.replace([np.inf, -np.inf], np.nan)
        before = len(df)
        df = df.dropna()
        after = len(df)
        if before > after:
            log.info(f"[Preprocessor] dropped {before - after} NaN rows ({after} remaining)")
        return df

    def fit_clip_bounds(self, X_train: pd.DataFrame) -> None:
        """Step 1b: Fit ±3-std outlier-clip bounds on TRAINING data only.

        BUG FIX (leakage): this used to be computed inside `clean_features`
        with mean()/std() over the FULL X (train+test combined) *before*
        the chronological split — silently leaking test-set distribution
        into the bounds applied to training data. It must be fit the same
        way the scaler already correctly is: train-only, then applied
        (not refit) to test.
        """
        self._clip_lower: Dict[str, float] = {}
        self._clip_upper: Dict[str, float] = {}
        for col in X_train.select_dtypes(include=[np.number]).columns:
            mean = X_train[col].mean()
            std = X_train[col].std()
            if std > 0:
                self._clip_lower[col] = mean - 3 * std
                self._clip_upper[col] = mean + 3 * std
        log.info(f"[Preprocessor] outlier-clip bounds fit on {len(self._clip_lower)} features (train only)")

    def apply_clip(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the train-fit ±3-std clip bounds to any split (train or test)."""
        df = df.copy()
        lower = getattr(self, "_clip_lower", {})
        upper = getattr(self, "_clip_upper", {})
        for col, lo in lower.items():
            if col in df.columns:
                df[col] = df[col].clip(lo, upper[col])
        return df

    def clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """DEPRECATED: drops NaN rows AND clips outliers using stats from
        whatever `df` is passed in. Kept only for backward compatibility
        with any external caller that used it directly on a single,
        already-final dataset (no train/test split involved). `process()`
        below no longer calls this — it uses `drop_invalid` +
        `fit_clip_bounds`/`apply_clip` instead to avoid fitting clip
        bounds on data that includes the test set.
        """
        df = self.drop_invalid(df)
        for col in df.select_dtypes(include=[np.number]).columns:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[col] = df[col].clip(mean - 3 * std, mean + 3 * std)
        return df

    def chronological_split(
        self, X: pd.DataFrame, y: pd.Series, test_size: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Step 2: Chronological train/test split (no shuffle — preserves time order)."""
        test_size = test_size or self.test_size
        n = len(X)
        split_idx = int(n * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        log.info(f"[Preprocessor] chronological split: train={len(X_train)}, test={len(X_test)}")
        return X_train, X_test, y_train, y_test

    def fit_scaler(self, X_train: pd.DataFrame) -> None:
        """Step 3: Fit StandardScaler on TRAINING data only (no leakage)."""
        for col in X_train.columns:
            self.feature_means[col] = float(X_train[col].mean())
            self.feature_stds[col] = float(X_train[col].std())
        log.info(f"[Preprocessor] scaler fit on {len(self.feature_means)} features")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply z-score normalization using fitted means/stds."""
        X_norm = X.copy()
        for col in X_norm.columns:
            mean = self.feature_means.get(col, 0.0)
            std = self.feature_stds.get(col, 1.0)
            if std > 0:
                X_norm[col] = (X_norm[col] - mean) / std
            else:
                X_norm[col] = 0.0
        return X_norm

    def save_scaler(self, path: Optional[Path] = None) -> Path:
        """Persist the scaler (means + stds) for later inference."""
        path = path or (PROCESSED_DIR / "scaler.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({
                "means": self.feature_means,
                "stds": self.feature_stds,
            }, f)
        log.info(f"[Preprocessor] scaler saved to {path}")
        return path

    def load_scaler(self, path: Path) -> None:
        """Load a previously saved scaler."""
        with path.open("rb") as f:
            data = _safe_load(f)
        self.feature_means = data["means"]
        self.feature_stds = data["stds"]
        log.info(f"[Preprocessor] scaler loaded from {path}")

    def process(
        self,
        features_df: pd.DataFrame,
        labels: pd.Series,
        test_size: Optional[float] = None,
        save: bool = True,
    ) -> ProcessedDataset:
        """Full pipeline: drop-invalid → split → fit clip+scaler on TRAIN → transform → return."""
        # Align indices
        common_idx = features_df.index.intersection(labels.index)
        X = features_df.loc[common_idx].copy()
        y = labels.loc[common_idx].copy()

        # Step 1: only drop NaN/inf rows here (no statistics computed —
        # safe pre-split). Outlier clipping is now fit AFTER the split,
        # train-only, same as the scaler (see fit_clip_bounds docstring).
        X = self.drop_invalid(X)
        # Re-align y after cleaning (X may have dropped rows)
        y = y.loc[X.index]

        # Split (chronological — no shuffle for time series)
        X_train, X_test, y_train, y_test = self.chronological_split(X, y, test_size)

        # Fit outlier-clip bounds + scaler on TRAIN ONLY (leakage prevention)
        self.fit_clip_bounds(X_train)
        X_train = self.apply_clip(X_train)
        X_test = self.apply_clip(X_test)  # same train-fit bounds, not refit

        self.fit_scaler(X_train)
        X_train_norm = self.transform(X_train)
        X_test_norm = self.transform(X_test)

        scaler_path = None
        if save:
            scaler_path = str(self.save_scaler())

        return ProcessedDataset(
            X_train=X_train_norm,
            X_test=X_test_norm,
            y_train=y_train,
            y_test=y_test,
            feature_names=list(X_train.columns),
            train_size=len(X_train),
            test_size=len(X_test),
            scaler_path=scaler_path,
        )


# ── Singleton ───────────────────────────────────────────────────────

_PREPROCESSOR: Optional[DataPreprocessor] = None


def get_preprocessor() -> DataPreprocessor:
    global _PREPROCESSOR
    if _PREPROCESSOR is None:
        _PREPROCESSOR = DataPreprocessor()
    return _PREPROCESSOR