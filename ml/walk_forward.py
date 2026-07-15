"""
ml/walk_forward.py — Walk-Forward Validation (Day 72)
======================================================

Rolling-window validation that respects chronological order. Instead of
one train/test split, it slides a window across the data:

  Window 1: train [Jan-Jun] → test [Jul]
  Window 2: train [Jan-Jul] → test [Aug]
  Window 3: train [Jan-Aug] → test [Sep]
  ...

Each window produces its own profit factor, win rate, and max drawdown.
The final score is the AVERAGE across all windows — a model that only
works in one window will score poorly.

This catches overfitting: a model trained on 2024 might ace 2024 test
data but fail on 2025. Walk-forward exposes this.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("walk_forward")


@dataclass
class WalkForwardFold:
    """One fold of walk-forward validation."""
    fold: int
    train_size: int
    test_size: int
    accuracy: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward results."""
    folds: List[WalkForwardFold] = field(default_factory=list)
    avg_accuracy: float = 0.0
    avg_win_rate: float = 0.0
    avg_profit_factor: float = 0.0
    avg_max_drawdown: float = 0.0
    avg_sharpe: float = 0.0
    consistency: float = 0.0    # std dev of profit factors (lower = more consistent)
    score: float = 0.0          # 0-100 final walk-forward score
    passed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "folds": [f.to_dict() for f in self.folds],
            "avg_accuracy": round(self.avg_accuracy, 4),
            "avg_win_rate": round(self.avg_win_rate, 4),
            "avg_profit_factor": round(self.avg_profit_factor, 2),
            "avg_max_drawdown": round(self.avg_max_drawdown, 4),
            "avg_sharpe": round(self.avg_sharpe, 3),
            "consistency": round(self.consistency, 3),
            "score": round(self.score, 1),
            "passed": self.passed,
        }


class WalkForwardValidator:
    """Rolling-window validation for time-series models."""

    def __init__(self, min_train_size: int = 200, step_size: int = 50, min_folds: int = 3):
        self.min_train_size = min_train_size
        self.step_size = step_size
        self.min_folds = min_folds

    def validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_fn,
        predict_fn,
    ) -> WalkForwardResult:
        """Run walk-forward validation.

        Args:
            X: Full feature matrix (chronological order).
            y: Full label series (chronological order).
            train_fn: callable(X_train, y_train) → model
            predict_fn: callable(model, X_test) → (y_pred, y_proba)

        Returns:
            WalkForwardResult with per-fold + aggregated metrics.
        """
        result = WalkForwardResult()
        n = len(X)

        if n < self.min_train_size + self.step_size:
            log.warning(f"[WalkForward] not enough data: {n} < {self.min_train_size + self.step_size}")
            return result

        pfs: List[float] = []
        wrs: List[float] = []
        accs: List[float] = []
        dds: List[float] = []
        sharpes: List[float] = []

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
                y_pred_arr = np.array(y_pred).astype(int)
                y_test_arr = np.array(y_test).astype(int)

                # Metrics
                acc = float(np.mean(y_pred_arr == y_test_arr))
                tp = int(np.sum((y_pred_arr == 1) & (y_test_arr == 1)))
                fp = int(np.sum((y_pred_arr == 1) & (y_test_arr == 0)))
                wr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                pf = tp / fp if fp > 0 else (float("inf") if tp > 0 else 0.0)
                pf = min(pf, 10.0)  # cap for averaging

                # Max drawdown (simplified)
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

                # Sharpe
                returns = [1 if y_test_arr[i] == 1 else -1 for i in range(len(y_pred_arr)) if y_pred_arr[i] == 1]
                sharpe = float(np.mean(returns) / np.std(returns)) if returns and np.std(returns) > 0 else 0.0

                fold_result = WalkForwardFold(
                    fold=fold, train_size=len(X_train), test_size=len(X_test),
                    accuracy=acc, win_rate=wr, profit_factor=pf,
                    max_drawdown=max_dd, sharpe=sharpe,
                )
                result.folds.append(fold_result)

                pfs.append(pf)
                wrs.append(wr)
                accs.append(acc)
                dds.append(max_dd)
                sharpes.append(sharpe)

                log.info(f"[WalkForward] fold {fold}: acc={acc:.1%} PF={pf:.2f} WR={wr:.1%} DD={max_dd}")
                fold += 1
            except Exception as e:
                log.warning(f"[WalkForward] fold {fold} failed: {e}")

            start += self.step_size

        if not result.folds:
            return result

        # Aggregate
        result.avg_accuracy = float(np.mean(accs))
        result.avg_win_rate = float(np.mean(wrs))
        result.avg_profit_factor = float(np.mean(pfs))
        result.avg_max_drawdown = float(np.mean(dds))
        result.avg_sharpe = float(np.mean(sharpes))
        result.consistency = float(np.std(pfs))  # lower = more stable

        # Score: combination of PF, WR, consistency
        # PF > 1.5 = good, WR > 55% = good, consistency < 0.5 = stable
        pf_score = min(100, result.avg_profit_factor * 40)  # PF 2.5 → 100
        wr_score = min(100, max(0, (result.avg_win_rate - 0.40) * 250))  # WR 40%→0, 80%→100
        consistency_score = max(0, 100 - result.consistency * 100)
        result.score = (pf_score * 0.4 + wr_score * 0.35 + consistency_score * 0.25)
        result.passed = result.score >= 60 and len(result.folds) >= self.min_folds

        log.info(
            f"[WalkForward] {len(result.folds)} folds | "
            f"avg PF={result.avg_profit_factor:.2f} WR={result.avg_win_rate:.1%} | "
            f"score={result.score:.1f} passed={result.passed}"
        )
        return result


# ── Singleton ───────────────────────────────────────────────────────

_VALIDATOR: Optional[WalkForwardValidator] = None


def get_walk_forward_validator() -> WalkForwardValidator:
    global _VALIDATOR
    if _VALIDATOR is None:
        _VALIDATOR = WalkForwardValidator()
    return _VALIDATOR
