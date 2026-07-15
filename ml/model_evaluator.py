"""
ml/model_evaluator.py — Model evaluation (Day 69)
===================================================

Evaluates trained ML models with both standard ML metrics AND trading-
specific metrics:

Standard:
  - Accuracy
  - Precision (per class)
  - Recall (per class)
  - F1 score
  - AUC-ROC
  - Confusion matrix

Trading-specific:
  - Win rate (predicted BUY → actual up)
  - Profit factor (gross profit / gross loss from signals)
  - Max drawdown
  - Sharpe ratio (simplified)
  - Overfitting detector (train vs test gap)

Walk-forward:
  - run_walk_forward() — rolling train/test with expanding window
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger
from config import WALK_FORWARD_MIN_TRAIN_SIZE, WALK_FORWARD_STEP_SIZE

log = get_logger("model_evaluator")


@dataclass
class ModelMetrics:
    """Comprehensive evaluation metrics for one model."""
    model_name: str
    # Standard ML metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    auc_roc: float = 0.0
    # Trading metrics
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    # Overfitting detection
    train_accuracy: float = 0.0
    test_accuracy: float = 0.0
    overfitting_score: float = 0.0  # train - test
    is_overfit: bool = False
    # Confusion matrix
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def summary_line(self) -> str:
        return (
            f"{self.model_name}: acc={self.accuracy:.1%} | AUC={self.auc_roc:.3f} | "
            f"WR={self.win_rate:.1%} | PF={self.profit_factor:.2f} | "
            f"overfit={self.overfitting_score:.1%}{' ⚠️' if self.is_overfit else ''}"
        )


class ModelEvaluator:
    """Evaluates classification models with ML + trading metrics."""

    def evaluate(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "model",
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.Series] = None,
    ) -> ModelMetrics:
        """Full evaluation of a trained model."""
        metrics = ModelMetrics(model_name=model_name)

        # Predictions
        try:
            y_pred = model.predict(X_test)
        except Exception as e:
            log.error(f"[Evaluator] predict failed: {e}")
            return metrics

        # Probabilities (if available)
        y_proba = None
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            try:
                y_proba = model.predict(X_test).ravel()
            except Exception:
                pass

        y_test_arr = np.array(y_test).astype(int)
        y_pred_arr = np.array(y_pred).astype(int)

        # Confusion matrix
        metrics.tp = int(np.sum((y_pred_arr == 1) & (y_test_arr == 1)))
        metrics.fp = int(np.sum((y_pred_arr == 1) & (y_test_arr == 0)))
        metrics.tn = int(np.sum((y_pred_arr == 0) & (y_test_arr == 0)))
        metrics.fn = int(np.sum((y_pred_arr == 0) & (y_test_arr == 1)))

        # Accuracy
        metrics.accuracy = float(np.mean(y_pred_arr == y_test_arr))

        # Precision / Recall / F1
        precision_denom = metrics.tp + metrics.fp
        metrics.precision = metrics.tp / precision_denom if precision_denom > 0 else 0.0
        recall_denom = metrics.tp + metrics.fn
        metrics.recall = metrics.tp / recall_denom if recall_denom > 0 else 0.0
        f1_denom = metrics.precision + metrics.recall
        metrics.f1 = (2 * metrics.precision * metrics.recall / f1_denom) if f1_denom > 0 else 0.0

        # AUC-ROC
        if y_proba is not None:
            try:
                from sklearn.metrics import roc_auc_score
                metrics.auc_roc = float(roc_auc_score(y_test_arr, y_proba))
            except Exception:
                metrics.auc_roc = 0.5

        # Trading metrics: treat each BUY prediction as a trade
        # Win = predicted 1 AND actual 1
        # Loss = predicted 1 AND actual 0
        wins = metrics.tp
        losses = metrics.fp
        total_trades = wins + losses
        metrics.win_rate = wins / total_trades if total_trades > 0 else 0.0

        # Profit factor (assume 1:1 R:R for simplicity; 1 unit per win, -1 per loss)
        gross_profit = float(wins)
        gross_loss = float(losses)
        metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

        # Max drawdown (simplified: consecutive losses)
        if total_trades > 0:
            equity = [0.0]
            for i in range(len(y_pred_arr)):
                if y_pred_arr[i] == 1:
                    equity.append(equity[-1] + (1 if y_test_arr[i] == 1 else -1))
            peak = equity[0]
            max_dd = 0.0
            for v in equity:
                if v > peak:
                    peak = v
                dd = peak - v
                if dd > max_dd:
                    max_dd = dd
            metrics.max_drawdown = max_dd

            # Sharpe ratio (simplified: mean / std of per-trade returns)
            returns = []
            for i in range(len(y_pred_arr)):
                if y_pred_arr[i] == 1:
                    returns.append(1.0 if y_test_arr[i] == 1 else -1.0)
            if returns:
                mean_r = np.mean(returns)
                std_r = np.std(returns)
                metrics.sharpe_ratio = float(mean_r / std_r) if std_r > 0 else 0.0

        # Overfitting detection
        if X_train is not None and y_train is not None:
            try:
                train_pred = model.predict(X_train)
                metrics.train_accuracy = float(np.mean(train_pred == np.array(y_train).astype(int)))
                metrics.test_accuracy = metrics.accuracy
                metrics.overfitting_score = metrics.train_accuracy - metrics.test_accuracy
                metrics.is_overfit = metrics.overfitting_score > 0.15  # >15% gap = overfit
            except Exception:
                pass

        return metrics

    def compare_models(self, metrics_list: List[ModelMetrics]) -> Dict[str, Any]:
        """Compare multiple models side by side."""
        if not metrics_list:
            return {}
        best_by = {
            "accuracy": max(metrics_list, key=lambda m: m.accuracy),
            "auc_roc": max(metrics_list, key=lambda m: m.auc_roc),
            "win_rate": max(metrics_list, key=lambda m: m.win_rate),
            "profit_factor": max(metrics_list, key=lambda m: m.profit_factor if m.profit_factor != float("inf") else 999),
            "sharpe": max(metrics_list, key=lambda m: m.sharpe_ratio),
        }
        return {
            "models": [m.to_dict() for m in metrics_list],
            "best": {
                "accuracy": best_by["accuracy"].model_name,
                "auc_roc": best_by["auc_roc"].model_name,
                "win_rate": best_by["win_rate"].model_name,
                "profit_factor": best_by["profit_factor"].model_name,
                "sharpe": best_by["sharpe"].model_name,
            },
            "summary_lines": [m.summary_line for m in metrics_list],
        }


# ── Walk-forward validation ─────────────────────────────────────────

class WalkForwardValidator:
    """Rolling window validation for time-series models."""

    def __init__(self, min_train_size: int = None, step_size: int = None):
        self.min_train_size = min_train_size if min_train_size is not None else WALK_FORWARD_MIN_TRAIN_SIZE
        self.step_size = step_size if step_size is not None else WALK_FORWARD_STEP_SIZE

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_fn,
        predict_fn,
    ) -> List[Dict[str, Any]]:
        """Run walk-forward validation.

        train_fn(X_train, y_train) -> model
        predict_fn(model, X_test) -> (y_pred, y_proba)

        Returns a list of per-fold results.
        """
        results: List[Dict[str, Any]] = []
        n = len(X)
        if n < self.min_train_size + self.step_size:
            log.warning("[WalkForward] not enough data")
            return results

        fold = 0
        start = self.min_train_size
        while start + self.step_size <= n:
            X_train = X.iloc[:start]
            y_train = y.iloc[:start]
            X_test = X.iloc[start:start + self.step_size]
            y_test = y.iloc[start:start + self.step_size]

            try:
                model = train_fn(X_train, y_train)
                y_pred, y_proba = predict_fn(model, X_test)
                acc = float(np.mean(np.array(y_pred).astype(int) == np.array(y_test).astype(int)))
                results.append({
                    "fold": fold,
                    "train_size": len(X_train),
                    "test_size": len(X_test),
                    "accuracy": acc,
                })
                log.info(f"[WalkForward] fold {fold}: train={len(X_train)}, test={len(X_test)}, acc={acc:.1%}")
            except Exception as e:
                log.warning(f"[WalkForward] fold {fold} failed: {e}")

            fold += 1
            start += self.step_size

        if results:
            avg_acc = np.mean([r["accuracy"] for r in results])
            log.info(f"[WalkForward] {fold} folds, avg accuracy: {avg_acc:.1%}")
        return results


# ── Singletons ──────────────────────────────────────────────────────

_EVALUATOR: Optional[ModelEvaluator] = None
_WF_VALIDATOR: Optional[WalkForwardValidator] = None


def get_evaluator() -> ModelEvaluator:
    global _EVALUATOR
    if _EVALUATOR is None:
        _EVALUATOR = ModelEvaluator()
    return _EVALUATOR


def get_walk_forward_validator() -> WalkForwardValidator:
    global _WF_VALIDATOR
    if _WF_VALIDATOR is None:
        _WF_VALIDATOR = WalkForwardValidator()
    return _WF_VALIDATOR
