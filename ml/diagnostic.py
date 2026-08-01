"""
ml/diagnostic.py — ML Diagnostic Engine (Day 74)
===================================================

Diagnoses ML model issues when performance is poor:
  1. Feature importance ranking — which features contribute most
  2. Bad feature detection — features with <1% importance
  3. Hyperparameter tuning suggestions
  4. Data quality check — sufficient samples?

Usage:
    from ml.diagnostic import MLDiagnostic
    diag = MLDiagnostic()
    report = diag.diagnose(X, y, model)
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

log = get_logger("ml_diagnostic")
from config import MIN_TRAINING_SAMPLES


@dataclass
class DiagnosticReport:
    """ML diagnostic report."""
    total_features: int = 0
    top_features: List[Dict[str, Any]] = field(default_factory=list)
    bad_features: List[str] = field(default_factory=list)
    data_sufficient: bool = False
    sample_count: int = 0
    recommendations: List[str] = field(default_factory=list)
    hyperparameter_suggestions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MLDiagnostic:
    """Diagnoses ML model and data quality issues."""

    def diagnose(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model: Optional[Any] = None,
    ) -> DiagnosticReport:
        """Run full diagnostic on the ML model + data.

        Args:
            X: Feature matrix.
            y: Labels.
            model: Trained model (with feature_importances_ or coef_).

        Returns:
            DiagnosticReport with findings + recommendations.
        """
        report = DiagnosticReport()
        report.total_features = len(X.columns)
        report.sample_count = len(X)

        # Check data sufficiency
        report.data_sufficient = len(X) >= MIN_TRAINING_SAMPLES
        if not report.data_sufficient:
            report.recommendations.append(
                f"Insufficient data: {len(X)} samples (need ≥{MIN_TRAINING_SAMPLES}). "
                f"Collect more data before training."
            )

        # Feature importance analysis
        if model is not None:
            importances = self._get_feature_importance(model, X)
            if importances is not None:
                report.top_features = self._rank_features(X.columns, importances, top_k=10)
                report.bad_features = self._find_bad_features(X.columns, importances)

                if report.bad_features:
                    report.recommendations.append(
                        f"Remove {len(report.bad_features)} low-importance features: "
                        f"{report.bad_features[:5]}..."
                    )

        # Hyperparameter suggestions
        report.hyperparameter_suggestions = self._suggest_hyperparameters(X, y)
        report.recommendations.extend(self._general_recommendations(X, y))

        log.info(
            f"[Diagnostic] {report.total_features} features | "
            f"{report.sample_count} samples | "
            f"{len(report.bad_features)} bad features | "
            f"{len(report.recommendations)} recommendations"
        )
        return report

    def _get_feature_importance(self, model: Any, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract feature importance from model."""
        try:
            if hasattr(model, "feature_importances_"):
                return model.feature_importances_
            elif hasattr(model, "coef_"):
                return np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
        except Exception as e:
            log.warning(f"[Diagnostic] Could not extract feature importance: {e}")
        return None

    def _rank_features(self, columns, importances, top_k: int = 10) -> List[Dict[str, Any]]:
        """Rank features by importance."""
        pairs = list(zip(columns, importances))
        pairs.sort(key=lambda x: -x[1])
        return [
            {"feature": str(f), "importance": round(float(imp), 4)}
            for f, imp in pairs[:top_k]
        ]

    def _find_bad_features(self, columns, importances, threshold: float = 0.005) -> List[str]:
        """Find features with very low importance (<0.5%)."""
        bad = []
        for col, imp in zip(columns, importances):
            if float(imp) < threshold:
                bad.append(str(col))
        return bad

    def _suggest_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> List[Dict[str, Any]]:
        """Suggest hyperparameter tuning configurations."""
        suggestions = []

        # XGBoost suggestions
        suggestions.append({
            "model": "xgboost",
            "configs": [
                {"max_depth": 3, "learning_rate": 0.01, "n_estimators": 200},
                {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 300},
                {"max_depth": 7, "learning_rate": 0.1, "n_estimators": 500},
            ],
        })

        # Random Forest suggestions
        suggestions.append({
            "model": "random_forest",
            "configs": [
                {"n_estimators": 100, "max_depth": 6},
                {"n_estimators": 300, "max_depth": 8},
                {"n_estimators": 500, "max_depth": 10},
            ],
        })

        return suggestions

    def _general_recommendations(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Generate general ML recommendations."""
        recs = []

        # Class balance check
        if hasattr(y, "value_counts"):
            vc = y.value_counts()
            if len(vc) == 2:
                ratio = vc.min() / vc.max()
                if ratio < 0.3:
                    recs.append(f"Class imbalance detected (ratio {ratio:.2f}). Consider oversampling or class weights.")

        # Feature count check
        if len(X.columns) > 200:
            recs.append(f"High feature count ({len(X.columns)}). Consider dimensionality reduction (PCA).")

        # Missing values
        missing = X.isnull().sum().sum()
        if missing > 0:
            recs.append(f"Missing values: {missing}. Fill or drop before training.")

        return recs


# ── Singleton ───────────────────────────────────────────────────────

_DIAG: Optional[MLDiagnostic] = None


def get_ml_diagnostic() -> MLDiagnostic:
    global _DIAG
    if _DIAG is None:
        _DIAG = MLDiagnostic()
    return _DIAG
