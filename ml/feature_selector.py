"""
ml/feature_selector.py — Feature selection + drift detection (Day 68)
=======================================================================

Two professional ML features:

1. **Feature Importance Tracking** — uses a fast tree-based estimator
   (LightGBM if available, else RandomForest, else simple variance threshold)
   to rank features by predictive power. Returns the top-K most important
   features for the current target.

2. **Feature Drift Detection** — compares the distribution of each feature
   between a reference window (e.g. last 30 days) and a current window
   (e.g. last 7 days). If a feature's mean or std has drifted significantly
   (PSI > 0.2), it flags the feature as drifted → retraining recommended.

   PSI (Population Stability Index):
     PSI < 0.1   → no significant drift
     PSI 0.1-0.2 → moderate drift, monitor
     PSI > 0.2   → significant drift, retrain

3. **Multi-Timeframe Feature Aggregation** — given feature vectors at
   M15, H1, H4, D1, returns a single combined vector with suffixed names
   (e.g. rsi_14_m15, rsi_14_h1, rsi_14_h4, rsi_14_d1).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("feature_selector")


@dataclass
class ImportanceResult:
    """Feature importance ranking."""
    feature_names: List[str]
    importances: List[float]
    top_k: List[str] = field(default_factory=list)
    method: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DriftResult:
    """Per-feature drift report."""
    feature: str
    psi: float
    ref_mean: float
    cur_mean: float
    ref_std: float
    cur_std: float
    drift_level: str  # NONE / MODERATE / SIGNIFICANT

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FeatureSelector:
    """Feature importance + drift detection."""

    # ── Feature importance ─────────────────────────────────────────

    def compute_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = "auto",
        top_k: int = 20,
    ) -> ImportanceResult:
        """Compute feature importance using the best available method.

        method:
          "auto"     — try LightGBM → RandomForest → variance
          "lightgbm" — LightGBM only (fast, accurate)
          "forest"   — RandomForest only
          "variance" — variance threshold (no model)
        """
        method = method.lower()
        if method == "auto":
            for m in ("lightgbm", "forest", "variance"):
                try:
                    return self._compute_with_method(X, y, m, top_k)
                except Exception as e:
                    log.debug(f"[Selector] {m} failed: {e} — trying next")
            # Last resort
            return self._variance_importance(X, y, top_k)
        return self._compute_with_method(X, y, method, top_k)

    def _compute_with_method(self, X, y, method, top_k) -> ImportanceResult:
        if method == "lightgbm":
            return self._lightgbm_importance(X, y, top_k)
        elif method == "forest":
            return self._forest_importance(X, y, top_k)
        elif method == "variance":
            return self._variance_importance(X, y, top_k)
        raise ValueError(f"unknown method: {method}")

    def _lightgbm_importance(self, X, y, top_k) -> ImportanceResult:
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm not installed")
        # Detect classification vs regression
        is_classif = y.nunique() <= 5
        model = lgb.LGBMClassifier(n_estimators=100, max_depth=5, verbose=-1) if is_classif \
                else lgb.LGBMRegressor(n_estimators=100, max_depth=5, verbose=-1)
        model.fit(X, y)
        importances = model.feature_importances_
        names = list(X.columns)
        # Sort by importance descending
        sorted_pairs = sorted(zip(names, importances), key=lambda x: -x[1])
        sorted_names = [p[0] for p in sorted_pairs]
        sorted_imp = [float(p[1]) for p in sorted_pairs]
        return ImportanceResult(
            feature_names=sorted_names,
            importances=sorted_imp,
            top_k=sorted_names[:top_k],
            method="lightgbm",
        )

    def _forest_importance(self, X, y, top_k) -> ImportanceResult:
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        except ImportError:
            raise ImportError("sklearn not installed")
        is_classif = y.nunique() <= 5
        model = RandomForestClassifier(n_estimators=80, max_depth=6, random_state=42) if is_classif \
                else RandomForestRegressor(n_estimators=80, max_depth=6, random_state=42)
        model.fit(X, y)
        importances = model.feature_importances_
        names = list(X.columns)
        sorted_pairs = sorted(zip(names, importances), key=lambda x: -x[1])
        sorted_names = [p[0] for p in sorted_pairs]
        sorted_imp = [float(p[1]) for p in sorted_pairs]
        return ImportanceResult(
            feature_names=sorted_names,
            importances=sorted_imp,
            top_k=sorted_names[:top_k],
            method="forest",
        )

    def _variance_importance(self, X, y, top_k) -> ImportanceResult:
        """Fallback: rank features by absolute correlation with target."""
        if not isinstance(y, pd.Series):
            y = pd.Series(y, index=X.index)
        # Compute absolute correlation per feature
        corrs = {}
        for col in X.columns:
            try:
                corr = abs(X[col].corr(y))
                corrs[col] = float(corr) if not math.isnan(corr) else 0.0
            except Exception:
                corrs[col] = 0.0
        sorted_pairs = sorted(corrs.items(), key=lambda x: -x[1])
        sorted_names = [p[0] for p in sorted_pairs]
        sorted_imp = [p[1] for p in sorted_pairs]
        return ImportanceResult(
            feature_names=sorted_names,
            importances=sorted_imp,
            top_k=sorted_names[:top_k],
            method="variance",
        )

    # ── Drift detection ────────────────────────────────────────────

    def detect_drift(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        psi_threshold_moderate: float = 0.1,
        psi_threshold_significant: float = 0.2,
    ) -> List[DriftResult]:
        """Compare feature distributions between reference and current windows.

        Returns a DriftResult per feature, sorted by PSI (most drifted first).
        """
        results: List[DriftResult] = []
        common_cols = [c for c in reference.columns if c in current.columns]
        for col in common_cols:
            ref = reference[col].dropna()
            cur = current[col].dropna()
            if len(ref) < 10 or len(cur) < 10:
                continue
            psi = self._psi(ref, cur)
            ref_mean = float(ref.mean())
            cur_mean = float(cur.mean())
            ref_std = float(ref.std())
            cur_std = float(cur.std())
            if psi < psi_threshold_moderate:
                level = "NONE"
            elif psi < psi_threshold_significant:
                level = "MODERATE"
            else:
                level = "SIGNIFICANT"
            results.append(DriftResult(
                feature=col, psi=round(psi, 4),
                ref_mean=round(ref_mean, 4), cur_mean=round(cur_mean, 4),
                ref_std=round(ref_std, 4), cur_std=round(cur_std, 4),
                drift_level=level,
            ))
        # Sort by PSI descending
        results.sort(key=lambda r: -r.psi)
        return results

    def _psi(self, ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
        """Compute Population Stability Index between two distributions."""
        try:
            # Use quantile bins from reference
            _, edges = pd.cut(ref, bins=bins, retbins=True, duplicates="drop")
            edges[0] = -np.inf
            edges[-1] = np.inf
            ref_pct = pd.cut(ref, bins=edges).value_counts(normalize=True).sort_index()
            cur_pct = pd.cut(cur, bins=edges).value_counts(normalize=True).sort_index()
            # Avoid division by zero
            ref_pct = ref_pct.replace(0, 0.0001)
            cur_pct = cur_pct.replace(0, 0.0001)
            psi = float(((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)).sum())
            return psi if not math.isnan(psi) else 0.0
        except Exception as e:
            log.debug(f"[Selector] PSI calc failed: {e}")
            return 0.0

    # ── Multi-timeframe aggregation ────────────────────────────────

    def aggregate_multi_timeframe(
        self,
        feature_vectors: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        """Combine feature vectors from multiple timeframes into one.

        Args:
            feature_vectors: {"m15": {...}, "h1": {...}, "h4": {...}, "d1": {...}}

        Returns:
            {"rsi_14_m15": ..., "rsi_14_h1": ..., "rsi_14_h4": ..., "rsi_14_d1": ...}
        """
        combined: Dict[str, float] = {}
        for tf, feats in feature_vectors.items():
            tf = tf.lower()
            for name, val in feats.items():
                combined[f"{name}_{tf}"] = float(val) if val is not None else 0.0
        return combined


# ── Singleton ───────────────────────────────────────────────────────

_SELECTOR: Optional[FeatureSelector] = None


def get_feature_selector() -> FeatureSelector:
    global _SELECTOR
    if _SELECTOR is None:
        _SELECTOR = FeatureSelector()
    return _SELECTOR
