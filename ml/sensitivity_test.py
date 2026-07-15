"""
ml/sensitivity_test.py — Parameter Sensitivity + Leakage Detection (Day 72)
============================================================================

Two tests:

1. **Parameter Sensitivity Test** — perturbs key features by ±10% and
   checks if predictions remain stable. An overfit model is extremely
   sensitive to small changes. A robust model is stable.

2. **Data Leakage Detector** — checks feature vectors for columns that
   might contain future information (e.g. "tomorrow_high", "future_close",
   "label_*" columns that shouldn't be in features).

Output:
    {
        "sensitivity": {
            "perturbed_accuracy": 0.61,
            "original_accuracy": 0.63,
            "drop": 0.02,
            "status": "ROBUST",
            "score": 88,
            "passed": True,
        },
        "leakage": {
            "suspicious_columns": [],
            "status": "CLEAN",
            "passed": True,
        }
    }
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("sensitivity_test")


# ── Suspicious column patterns (potential future leakage) ───────────
LEAKAGE_PATTERNS = [
    r"future", r"tomorrow", r"next_", r"forward_",
    r"label_", r"target", r"outcome", r"actual_",
    r"y_true", r"y_pred", r"result_",
]


@dataclass
class SensitivityResult:
    """Parameter sensitivity test result."""
    original_accuracy: float = 0.0
    perturbed_accuracy: float = 0.0
    accuracy_drop: float = 0.0
    status: str = "UNKNOWN"     # ROBUST / SENSITIVE / OVERFIT
    score: float = 0.0
    passed: bool = False
    perturbation_details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in asdict(self).items()}


@dataclass
class LeakageResult:
    """Data leakage detection result."""
    suspicious_columns: List[str] = field(default_factory=list)
    status: str = "CLEAN"       # CLEAN / SUSPICIOUS
    passed: bool = True
    total_columns_checked: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SensitivityTester:
    """Tests model robustness to parameter perturbation + detects data leakage."""

    def test_sensitivity(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        perturbation_pct: float = 0.10,
    ) -> SensitivityResult:
        """Perturb all numeric features by ±10% and check prediction stability.

        An overfit model's accuracy will drop sharply. A robust model stays stable.
        """
        result = SensitivityResult()

        try:
            # Original accuracy
            y_pred_orig = model.predict(X)
            result.original_accuracy = float(np.mean(y_pred_orig == np.array(y).astype(int)))

            # Perturb: add ±10% noise to each feature
            rng = np.random.RandomState(42)
            X_perturbed = X.copy()
            for col in X_perturbed.select_dtypes(include=[np.number]).columns:
                noise = rng.normal(0, perturbation_pct, len(X_perturbed))
                X_perturbed[col] = X_perturbed[col] * (1 + noise)

            y_pred_pert = model.predict(X_perturbed)
            result.perturbed_accuracy = float(np.mean(y_pred_pert == np.array(y).astype(int)))
            result.accuracy_drop = result.original_accuracy - result.perturbed_accuracy

            # Classify
            if result.accuracy_drop < 0.03:
                result.status = "ROBUST"
            elif result.accuracy_drop < 0.08:
                result.status = "SENSITIVE"
            else:
                result.status = "OVERFIT"

            # Score: less drop = higher score
            result.score = max(0, min(100, 100 - result.accuracy_drop * 500))
            result.passed = result.accuracy_drop < 0.08

            log.info(
                f"[Sensitivity] orig_acc={result.original_accuracy:.1%} "
                f"pert_acc={result.perturbed_accuracy:.1%} "
                f"drop={result.accuracy_drop:.1%} → {result.status} "
                f"(score={result.score:.1f})"
            )
        except Exception as e:
            log.warning(f"[Sensitivity] test failed: {e}")
            result.status = "ERROR"

        return result

    def detect_leakage(self, X: pd.DataFrame) -> LeakageResult:
        """Check feature columns for potential future-data leakage."""
        result = LeakageResult(total_columns_checked=len(X.columns))
        suspicious: List[str] = []

        for col in X.columns:
            col_lower = str(col).lower()
            for pattern in LEAKAGE_PATTERNS:
                if re.search(pattern, col_lower):
                    suspicious.append(col)
                    break

        result.suspicious_columns = suspicious
        if suspicious:
            result.status = "SUSPICIOUS"
            result.passed = False
            log.warning(f"[Leakage] suspicious columns: {suspicious}")
        else:
            result.status = "CLEAN"
            result.passed = True
            log.info(f"[Leakage] {len(X.columns)} columns checked — CLEAN")

        return result


# ── Singleton ───────────────────────────────────────────────────────

_TESTER: Optional[SensitivityTester] = None


def get_sensitivity_tester() -> SensitivityTester:
    global _TESTER
    if _TESTER is None:
        _TESTER = SensitivityTester()
    return _TESTER
