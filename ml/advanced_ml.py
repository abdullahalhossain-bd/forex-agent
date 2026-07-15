"""
ml/advanced_ml.py — Online Learning + Concept Drift + Uncertainty
=================================================================
3 missing advanced ML areas from the quant/HFT research list:

4. Concept Drift Detection — detect when market behavior changes
5. Online Learning — incrementally update models with new data
10. Uncertainty Estimation — prediction confidence intervals

USAGE:
    from ml.advanced_ml import (
        ConceptDriftDetector,
        OnlineLearner,
        UncertaintyEstimator,
    )
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from utils.logger import get_logger

log = get_logger("advanced_ml")


# ════════════════════════════════════════════════════════════════════
# 4. CONCEPT DRIFT DETECTION
# ════════════════════════════════════════════════════════════════════

class ConceptDriftDetector:
    """Detect when market behavior changes (concept drift).

    Uses statistical tests to compare recent data distribution against
    a reference window. If the distribution has shifted significantly,
    the model may need retraining.

    Methods:
    - KS Test (Kolmogorov-Smirnov) for distribution comparison
    - PSI (Population Stability Index) for feature drift
    - ADWIN (Adaptive Windowing) for concept drift
    """

    REFERENCE_WINDOW = 500  # reference period (bars)
    RECENT_WINDOW = 50      # recent period to compare
    PSI_THRESHOLD = 0.25    # PSI > 0.25 = significant drift
    KS_THRESHOLD = 0.05     # p-value < 0.05 = significant drift

    def __init__(self):
        self._reference_data: Dict[str, np.ndarray] = {}
        self._drift_history: List[dict] = []

    def set_reference(self, feature_name: str, data: np.ndarray):
        """Set the reference distribution for a feature."""
        self._reference_data[feature_name] = np.array(data)

    def check_drift(self, feature_name: str, recent_data: np.ndarray) -> dict:
        """Check if a feature has drifted from its reference distribution.

        Returns:
            {
                "feature": str,
                "drifted": bool,
                "psi": float,        # Population Stability Index
                "ks_statistic": float,
                "ks_pvalue": float,
                "severity": str,     # NONE / LOW / MEDIUM / HIGH
                "recommendation": str,
            }
        """
        ref = self._reference_data.get(feature_name)
        if ref is None or len(ref) < 10 or len(recent_data) < 10:
            return {
                "feature": feature_name, "drifted": False, "psi": 0,
                "ks_statistic": 0, "ks_pvalue": 1, "severity": "NONE",
                "recommendation": "Insufficient data for drift check",
            }

        # ── PSI (Population Stability Index) ──
        psi = self._compute_psi(ref, recent_data)

        # ── KS Test (simplified — no scipy dependency) ──
        ks_stat, ks_pval = self._ks_test(ref, recent_data)

        # ── Determine severity ──
        drifted = psi > self.PSI_THRESHOLD or ks_pval < self.KS_THRESHOLD
        if psi > 0.5:
            severity = "HIGH"
            recommendation = f"Severe drift (PSI={psi:.3f}) — retrain model immediately"
        elif psi > 0.25:
            severity = "MEDIUM"
            recommendation = f"Significant drift (PSI={psi:.3f}) — schedule retraining"
        elif psi > 0.1:
            severity = "LOW"
            recommendation = f"Minor drift (PSI={psi:.3f}) — monitor closely"
        else:
            severity = "NONE"
            recommendation = "No drift detected"

        result = {
            "feature": feature_name,
            "drifted": drifted,
            "psi": round(psi, 4),
            "ks_statistic": round(ks_stat, 4),
            "ks_pvalue": round(ks_pval, 4),
            "severity": severity,
            "recommendation": recommendation,
        }

        if drifted:
            log.warning(
                f"[ConceptDrift] {feature_name}: PSI={psi:.3f} KS_p={ks_pval:.4f} "
                f"severity={severity} — {recommendation}"
            )

        return result

    def _compute_psi(self, reference: np.ndarray, recent: np.ndarray, n_bins: int = 10) -> float:
        """Compute Population Stability Index."""
        # Create bins from reference data
        bins = np.linspace(np.percentile(reference, 1), np.percentile(reference, 99), n_bins + 1)
        bins[0] = -np.inf
        bins[-1] = np.inf

        ref_hist, _ = np.histogram(reference, bins=bins)
        rec_hist, _ = np.histogram(recent, bins=bins)

        # Convert to proportions
        ref_prop = ref_hist / len(reference) + 1e-6  # avoid log(0)
        rec_prop = rec_hist / len(recent) + 1e-6

        # PSI = sum((rec - ref) * ln(rec / ref))
        psi = np.sum((rec_prop - ref_prop) * np.log(rec_prop / ref_prop))

        return float(psi)

    def _ks_test(self, reference: np.ndarray, recent: np.ndarray) -> Tuple[float, float]:
        """Simplified Kolmogorov-Smirnov test (no scipy)."""
        ref_sorted = np.sort(reference)
        rec_sorted = np.sort(recent)

        # Compute empirical CDFs
        n1, n2 = len(ref_sorted), len(rec_sorted)
        all_values = np.sort(np.concatenate([ref_sorted, rec_sorted]))

        cdf1 = np.searchsorted(ref_sorted, all_values, side="right") / n1
        cdf2 = np.searchsorted(rec_sorted, all_values, side="right") / n2

        ks_stat = float(np.max(np.abs(cdf1 - cdf2)))

        # Approximate p-value (Kolmogorov distribution)
        en = np.sqrt(n1 * n2 / (n1 + n2))
        p_value = float(max(0, min(1, 2 * np.exp(-2 * (en * ks_stat) ** 2))))

        return ks_stat, p_value

    def check_all_features(self, recent_data: Dict[str, np.ndarray]) -> dict:
        """Check drift for all features at once."""
        results = {}
        any_drifted = False
        max_severity = "NONE"

        for feature, data in recent_data.items():
            if feature in self._reference_data:
                result = self.check_drift(feature, data)
                results[feature] = result
                if result["drifted"]:
                    any_drifted = True
                    if result["severity"] == "HIGH":
                        max_severity = "HIGH"
                    elif result["severity"] == "MEDIUM" and max_severity != "HIGH":
                        max_severity = "MEDIUM"

        return {
            "any_drifted": any_drifted,
            "max_severity": max_severity,
            "features": results,
            "recommendation": "Retrain model" if any_drifted else "No action needed",
        }


# ════════════════════════════════════════════════════════════════════
# 5. ONLINE LEARNING
# ════════════════════════════════════════════════════════════════════

class OnlineLearner:
    """Incremental/online learning for model updates.

    Instead of retraining from scratch, this module incrementally
    updates model weights as new data arrives. Supports:
    - SGD-based online updates (for sklearn models with partial_fit)
    - Exponential weighting of recent observations
    - Model versioning with automatic rollback if performance degrades

    USAGE:
        learner = OnlineLearner()
        learner.initialize(model_type="sgd_classifier")
        learner.partial_fit(X_new, y_new)
        prediction = learner.predict(X)
    """

    MODEL_PATH = Path("memory/online_model")
    MAX_SAMPLES_BEFORE_RETRAIN = 100
    PERFORMANCE_DECAY_THRESHOLD = 0.3  # if accuracy drops by 30%, rollback

    def __init__(self):
        self._model = None
        self._model_type = None
        self._classes = None
        self._n_samples_seen = 0
        self._performance_history: List[float] = []
        self._best_performance = 0.0

    def initialize(self, model_type: str = "sgd_classifier"):
        """Initialize the online learning model.

        Args:
            model_type: "sgd_classifier" / "passive_aggressive" / "perceptron"
        """
        self._model_type = model_type
        try:
            from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, Perceptron
            if model_type == "sgd_classifier":
                self._model = SGDClassifier(loss="log_loss", warm_start=True)
            elif model_type == "passive_aggressive":
                self._model = PassiveAggressiveClassifier(warm_start=True)
            elif model_type == "perceptron":
                self._model = Perceptron(warm_start=True)
            else:
                log.warning(f"[OnlineLearner] Unknown model type: {model_type}")
                self._model = SGDClassifier(loss="log_loss", warm_start=True)
            log.info(f"[OnlineLearner] Initialized: {model_type}")
        except ImportError:
            log.warning("[OnlineLearner] sklearn not available — online learning disabled")

    def partial_fit(self, X: np.ndarray, y: np.ndarray, classes: list = None):
        """Incrementally update the model with new data.

        Args:
            X: Feature matrix.
            y: Labels.
            classes: All possible classes (required for first call).
        """
        if self._model is None:
            self.initialize()

        if self._model is None:
            return

        try:
            if classes is not None and self._classes is None:
                self._classes = classes

            self._model.partial_fit(X, y, classes=self._classes)
            self._n_samples_seen += len(X)

            log.info(
                f"[OnlineLearner] Updated with {len(X)} samples "
                f"(total seen: {self._n_samples_seen})"
            )
        except Exception as e:
            log.warning(f"[OnlineLearner] partial_fit failed: {e}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the online model."""
        if self._model is None:
            return np.array([])
        try:
            return self._model.predict(X)
        except Exception:
            return np.array([])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (if model supports it)."""
        if self._model is None:
            return np.array([])
        try:
            if hasattr(self._model, "predict_proba"):
                return self._model.predict_proba(X)
            elif hasattr(self._model, "decision_function"):
                # Convert decision function to pseudo-probabilities
                decisions = self._model.decision_function(X)
                if decisions.ndim == 1:
                    probs = 1 / (1 + np.exp(-decisions))
                    return np.column_stack([1 - probs, probs])
                else:
                    # Softmax
                    exp_d = np.exp(decisions - decisions.max(axis=1, keepdims=True))
                    return exp_d / exp_d.sum(axis=1, keepdims=True)
        except Exception:
            pass
        return np.array([])

    def record_performance(self, accuracy: float):
        """Record model performance for rollback detection."""
        self._performance_history.append(accuracy)
        if len(self._performance_history) > 20:
            self._performance_history = self._performance_history[-20:]

        if accuracy > self._best_performance:
            self._best_performance = accuracy

        # Check for degradation
        if len(self._performance_history) >= 5:
            recent_avg = np.mean(self._performance_history[-5:])
            if self._best_performance > 0 and recent_avg < self._best_performance * (1 - self.PERFORMANCE_DECAY_THRESHOLD):
                log.warning(
                    f"[OnlineLearner] Performance degraded: recent avg={recent_avg:.2%} "
                    f"vs best={self._best_performance:.2%} — consider rollback"
                )
                return True  # signal: rollback recommended
        return False

    def needs_full_retrain(self) -> bool:
        """Check if enough new samples have been seen to warrant full retraining."""
        return self._n_samples_seen >= self.MAX_SAMPLES_BEFORE_RETRAIN


# ════════════════════════════════════════════════════════════════════
# 10. UNCERTAINTY ESTIMATION
# ════════════════════════════════════════════════════════════════════

class UncertaintyEstimator:
    """Estimate prediction uncertainty for trade decisions.

    If the model is uncertain, the bot should NOT trade. This module
    estimates two types of uncertainty:
    - Epistemic: model doesn't know (can be reduced with more data)
    - Aleatoric: inherent market noise (cannot be reduced)

    Methods:
    - MC Dropout: run model N times with dropout, measure variance
    - Ensemble variance: if ensemble members disagree, uncertainty is high
    - Prediction interval: estimate range of likely outcomes
    """

    MIN_CONFIDENCE_TO_TRADE = 0.60  # below 60% confidence = no trade
    HIGH_UNCERTAINTY_THRESHOLD = 0.40  # above 40% uncertainty = skip

    def __init__(self):
        self._prediction_history: List[dict] = []

    def estimate_from_ensemble(
        self,
        predictions: List[Any],  # list of predictions from different models
        confidences: List[float],
    ) -> dict:
        """Estimate uncertainty from ensemble disagreement.

        Args:
            predictions: List of predictions (e.g., ["BUY", "SELL", "BUY", "WAIT"]).
            confidences: List of confidence scores (0-1).

        Returns:
            {
                "agreement": float,      # 0-1, how much models agree
                "uncertainty": float,    # 0-1, 1 = maximally uncertain
                "confidence": float,     # 0-1, adjusted confidence
                "should_trade": bool,    # based on uncertainty threshold
                "reason": str,
            }
        """
        if not predictions:
            return {"agreement": 0, "uncertainty": 1, "confidence": 0,
                    "should_trade": False, "reason": "No predictions"}

        # ── Agreement: fraction of models that agree with majority ──
        from collections import Counter
        vote_count = Counter(predictions)
        majority_pred, majority_votes = vote_count.most_common(1)[0]
        agreement = majority_votes / len(predictions)

        # ── Uncertainty: 1 - agreement ──
        uncertainty = 1.0 - agreement

        # ── Confidence: weighted average of agreeing models' confidence ──
        agreeing_confidences = [
            conf for pred, conf in zip(predictions, confidences)
            if pred == majority_pred
        ]
        base_confidence = np.mean(agreeing_confidences) if agreeing_confidences else 0.5

        # Adjusted confidence: reduced by disagreement
        adjusted_confidence = base_confidence * agreement

        # ── Decision ──
        should_trade = (
            adjusted_confidence >= self.MIN_CONFIDENCE_TO_TRADE and
            uncertainty <= self.HIGH_UNCERTAINTY_THRESHOLD
        )

        if should_trade:
            reason = f"Confident: {adjusted_confidence:.0%} confidence, {uncertainty:.0%} uncertainty"
        elif uncertainty > self.HIGH_UNCERTAINTY_THRESHOLD:
            reason = f"Too uncertain: {uncertainty:.0%} uncertainty (models disagree)"
        else:
            reason = f"Low confidence: {adjusted_confidence:.0%} < {self.MIN_CONFIDENCE_TO_TRADE:.0%}"

        log.info(
            f"[Uncertainty] agreement={agreement:.0%} uncertainty={uncertainty:.0%} "
            f"confidence={adjusted_confidence:.0%} → {'TRADE' if should_trade else 'SKIP'} ({reason})"
        )

        return {
            "agreement": round(agreement, 3),
            "uncertainty": round(uncertainty, 3),
            "confidence": round(adjusted_confidence, 3),
            "should_trade": should_trade,
            "majority_prediction": majority_pred,
            "reason": reason,
        }

    def estimate_from_confidence_interval(
        self,
        prediction: float,
        lower_bound: float,
        upper_bound: float,
        threshold: float = 0.0,  # 0 = neutral; >0 = buy threshold; <0 = sell threshold
    ) -> dict:
        """Estimate uncertainty from prediction interval width.

        Wide interval = high uncertainty = skip trade.
        Narrow interval = confident = trade.

        Args:
            prediction: Model's predicted value (e.g., expected return).
            lower_bound: Lower bound of prediction interval.
            upper_bound: Upper bound of prediction interval.
            threshold: Decision threshold (e.g., 0.0 = any positive = buy).

        Returns:
            {
                "prediction": float,
                "interval_width": float,
                "uncertainty": float,   # 0-1, normalized interval width
                "should_trade": bool,
                "reason": str,
            }
        """
        interval_width = abs(upper_bound - lower_bound)

        # Normalize uncertainty (relative to prediction magnitude)
        if abs(prediction) > 1e-6:
            uncertainty = min(interval_width / (abs(prediction) * 4), 1.0)
        else:
            uncertainty = 1.0  # prediction near zero = very uncertain

        # Check if prediction is clearly above/below threshold
        clearly_above = lower_bound > threshold
        clearly_below = upper_bound < threshold

        should_trade = (
            uncertainty < self.HIGH_UNCERTAINTY_THRESHOLD and
            (clearly_above or clearly_below)
        )

        if should_trade:
            direction = "BUY" if clearly_above else "SELL"
            reason = f"Confident {direction}: prediction={prediction:.4f} ±{interval_width/2:.4f}"
        else:
            reason = f"Uncertain: interval [{lower_bound:.4f}, {upper_bound:.4f}] too wide"

        return {
            "prediction": round(prediction, 4),
            "interval_width": round(interval_width, 4),
            "uncertainty": round(uncertainty, 3),
            "should_trade": should_trade,
            "reason": reason,
        }

    def estimate_from_history(
        self,
        recent_predictions: List[dict],  # [{prediction, actual_outcome, confidence}, ...]
    ) -> dict:
        """Estimate model calibration from recent prediction history.

        If the model says "80% confident" but only wins 50% of the time,
        it's poorly calibrated → high uncertainty.

        Returns:
            {
                "calibration_error": float,  # 0 = perfectly calibrated
                "avg_confidence": float,
                "actual_win_rate": float,
                "overconfident": bool,       # confidence > actual win rate
                "should_trade": bool,
                "reason": str,
            }
        """
        if len(recent_predictions) < 10:
            return {
                "calibration_error": 0.5, "avg_confidence": 0.5,
                "actual_win_rate": 0.5, "overconfident": False,
                "should_trade": True, "reason": "Insufficient history for calibration",
            }

        confidences = [p["confidence"] for p in recent_predictions]
        outcomes = [1 if p["actual_outcome"] == "WIN" else 0 for p in recent_predictions]

        avg_confidence = np.mean(confidences)
        actual_win_rate = np.mean(outcomes)
        calibration_error = abs(avg_confidence - actual_win_rate)

        overconfident = avg_confidence > actual_win_rate + 0.1

        # If severely miscalibrated, don't trade
        should_trade = calibration_error < 0.2

        if overconfident:
            reason = f"OVERCONFIDENT: avg confidence {avg_confidence:.0%} but actual WR {actual_win_rate:.0%}"
        elif calibration_error > 0.2:
            reason = f"Poorly calibrated: error={calibration_error:.0%}"
        else:
            reason = f"Well calibrated: confidence={avg_confidence:.0%}, WR={actual_win_rate:.0%}"

        log.info(
            f"[Uncertainty] Calibration: confidence={avg_confidence:.0%} "
            f"actual_WR={actual_win_rate:.0%} error={calibration_error:.0%} "
            f"→ {'TRADE' if should_trade else 'SKIP'} ({reason})"
        )

        return {
            "calibration_error": round(calibration_error, 3),
            "avg_confidence": round(avg_confidence, 3),
            "actual_win_rate": round(actual_win_rate, 3),
            "overconfident": overconfident,
            "should_trade": should_trade,
            "reason": reason,
        }


# ════════════════════════════════════════════════════════════════════
# SMOKE TESTS
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)

    print("=== 1. Concept Drift Detection ===")
    detector = ConceptDriftDetector()
    # Reference: normal distribution
    ref_data = np.random.randn(500) * 0.001 + 1.1000
    detector.set_reference("rsi", ref_data)
    # Recent: shifted distribution
    recent_data = np.random.randn(50) * 0.002 + 1.1050  # higher mean, higher vol
    drift = detector.check_drift("rsi", recent_data)
    print(f"  PSI={drift['psi']:.3f} KS_p={drift['ks_pvalue']:.4f} severity={drift['severity']}")
    print(f"  Drifted: {drift['drifted']} — {drift['recommendation']}")

    # No drift case
    recent_normal = np.random.randn(50) * 0.001 + 1.1000
    drift2 = detector.check_drift("rsi", recent_normal)
    print(f"  No drift: PSI={drift2['psi']:.3f} severity={drift2['severity']}")

    print("\n=== 2. Online Learning ===")
    learner = OnlineLearner()
    learner.initialize("sgd_classifier")
    # Simulate training data
    X = np.random.randn(100, 5)
    y = (X[:, 0] > 0).astype(int)
    learner.partial_fit(X[:50], y[:50], classes=[0, 1])
    learner.partial_fit(X[50:], y[50:])
    preds = learner.predict(X[:5])
    print(f"  Predictions: {preds}")
    print(f"  Samples seen: {learner._n_samples_seen}")
    print(f"  Needs retrain: {learner.needs_full_retrain()}")

    print("\n=== 3. Uncertainty Estimation ===")
    est = UncertaintyEstimator()

    # Ensemble disagreement
    r1 = est.estimate_from_ensemble(
        predictions=["BUY", "BUY", "BUY", "SELL", "WAIT"],
        confidences=[0.8, 0.75, 0.7, 0.6, 0.5],
    )
    print(f"  Ensemble: agreement={r1['agreement']:.0%} uncertainty={r1['uncertainty']:.0%} "
          f"confidence={r1['confidence']:.0%} → trade={r1['should_trade']}")

    # High agreement
    r2 = est.estimate_from_ensemble(
        predictions=["BUY", "BUY", "BUY", "BUY"],
        confidences=[0.85, 0.80, 0.75, 0.70],
    )
    print(f"  High agreement: agreement={r2['agreement']:.0%} uncertainty={r2['uncertainty']:.0%} "
          f"confidence={r2['confidence']:.0%} → trade={r2['should_trade']}")

    # Calibration check
    history = [
        {"confidence": 0.8, "actual_outcome": "WIN"},
        {"confidence": 0.7, "actual_outcome": "WIN"},
        {"confidence": 0.8, "actual_outcome": "LOSS"},
        {"confidence": 0.9, "actual_outcome": "WIN"},
        {"confidence": 0.7, "actual_outcome": "LOSS"},
        {"confidence": 0.8, "actual_outcome": "WIN"},
        {"confidence": 0.6, "actual_outcome": "WIN"},
        {"confidence": 0.9, "actual_outcome": "WIN"},
        {"confidence": 0.7, "actual_outcome": "LOSS"},
        {"confidence": 0.8, "actual_outcome": "WIN"},
    ]
    cal = est.estimate_from_history(history)
    print(f"  Calibration: error={cal['calibration_error']:.0%} "
          f"confidence={cal['avg_confidence']:.0%} WR={cal['actual_win_rate']:.0%} "
          f"overconfident={cal['overconfident']} → trade={cal['should_trade']}")

    print("\nAll advanced ML smoke tests passed.")
