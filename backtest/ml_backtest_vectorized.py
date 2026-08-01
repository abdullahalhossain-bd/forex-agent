# backtest/ml_backtest_vectorized.py — ML-based vectorized backtests
# =============================================================================
# Ported from FXBot (https://github.com/trentstauff/FXBot):
#   - backtesting/MLClassificationBacktest.py    → MLClassificationBacktest
#   - backtesting/MultipleRegressionModelPredictor.py → MLRegressionBacktest
# Original author: Trent Stauffner — MIT license (inferred)
#
# Both classes predict the DIRECTION of next-bar returns from lagged returns:
#   - Classification: LogisticRegression predicts sign(returns)
#   - Regression:     LinearRegression predicts returns, then takes sign
#
# The Classification variant is the more standard approach. The Regression
# variant is included for completeness — its sign-prediction is a noisier
# signal but can capture magnitude information.
#
# Train/test split is by time (no shuffling — that would leak future info).
# Default split: 70% train, 30% test.
# =============================================================================

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from backtest.vectorized_base import VectorizedBacktester
from utils.logger import get_logger

log = get_logger("ml_backtest")

try:
    from sklearn.linear_model import LogisticRegression, LinearRegression
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    log.warning("scikit-learn not available — ML backtests will not work. "
                "Install with: pip install scikit-learn")


# ── ML Classification (Logistic Regression on lagged returns) ────────────────

class MLClassificationBacktest(VectorizedBacktester):
    """
    Predicts direction of next-bar return using LogisticRegression on lagged
    returns. Position = sign(prediction).

    Pipeline:
        1. Compute log returns.
        2. Build lag features: lag_1, lag_2, ..., lag_N.
        3. Split into train (e.g., 70%) and test (30%) by TIME (not random).
        4. Fit LogisticRegression on train, predicting sign(returns).
        5. Predict on test set; position = sign(prediction).
        6. Compute strategy returns, cumulative returns, hit ratio.

    Notes
    -----
    - LogisticRegression with C=1e6 (very low regularization) — same as FXBot.
    - Hit ratio = fraction of bars where sign(prediction) == sign(returns).
    - This is a vectorized, single-fit model. For walk-forward validation,
      use backtest/walk_forward.py with this strategy inside.
    """

    def __init__(self, *args, lags: int = 5, train_ratio: float = 0.7,
                 C: float = 1e6, max_iter: int = 100000, **kwargs):
        if not _HAS_SKLEARN:
            raise ImportError("MLClassificationBacktest requires scikit-learn. "
                              "Install with: pip install scikit-learn")
        self.lags = lags
        self.train_ratio = train_ratio
        self.C = C
        self.max_iter = max_iter
        self._model = LogisticRegression(C=C, max_iter=max_iter)
        self._hitratio: float | None = None
        self._feature_columns: list[str] = []
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return (f"MLClassificationBacktest(symbol={self.symbol}, lags={self.lags}, "
                f"train_ratio={self.train_ratio}, granularity={self.granularity})")

    def prepare_data(self) -> pd.DataFrame:
        """Add lag_1 .. lag_N columns."""
        df = self._data.copy()
        for lag in range(1, self.lags + 1):
            df[f"lag_{lag}"] = df["returns"].shift(lag)
        self._feature_columns = [f"lag_{lag}" for lag in range(1, self.lags + 1)]
        return df

    def get_hitratio(self) -> float:
        if self._hitratio is None:
            raise RuntimeError("Call .test() first.")
        return self._hitratio

    def test(self, mute: bool = False) -> Tuple[float, float]:
        if not mute:
            log.info(f"Testing ML Classification with lags={self.lags}, "
                     f"train_ratio={self.train_ratio} ...")
        data = self._data.copy().dropna(subset=self._feature_columns + ["returns"])
        if data.empty:
            log.warning("No data after lag dropna — check lags vs data length.")
            return 1.0, 0.0

        # Time-based split
        split_idx = int(len(data) * self.train_ratio)
        train = data.iloc[:split_idx]
        test = data.iloc[split_idx:]
        if len(train) < 50 or len(test) < 10:
            log.warning(f"Train ({len(train)}) or test ({len(test)}) set too small.")
            return 1.0, 0.0

        # Fit
        X_train = train[self._feature_columns]
        y_train = np.sign(train["returns"])
        self._model.fit(X_train, y_train)

        # Predict on test
        X_test = test[self._feature_columns]
        test = test.copy()
        test["prediction"] = self._model.predict(X_test)
        test["position"] = test["prediction"]
        test["strategy"] = test["position"] * test["returns"]

        # Hit ratio
        hits = np.sign(test["returns"] * test["prediction"]).value_counts()
        if 1.0 in hits.index and -1.0 in hits.index:
            self._hitratio = float(hits[1.0] / (hits[1.0] + hits[-1.0]))
        elif 1.0 in hits.index:
            self._hitratio = 1.0
        else:
            self._hitratio = 0.0

        # Compute performance
        self._data = test
        perf, outperf, n_trades = self._compute_performance(test)
        if not mute:
            log.info(f"Return: {(perf-1)*100:.2f}%, OutPerformance: {outperf*100:.2f}%, "
                     f"Hit Ratio: {self._hitratio:.3f}, Trades: {n_trades}")
        return perf, outperf

    # ML strategies don't have a meaningful grid-search — skip optimize()


# ── ML Regression (Linear Regression on lagged returns, take sign) ───────────

class MLRegressionBacktest(VectorizedBacktester):
    """
    Predicts next-bar return using LinearRegression on lagged returns.
    Position = sign(prediction).

    Same shape as MLClassificationBacktest but with a continuous target.
    The sign-of-continuous-prediction is noisier than direct classification,
    but can capture direction-magnitude tradeoffs.
    """

    def __init__(self, *args, lags: int = 3, train_ratio: float = 0.7, **kwargs):
        if not _HAS_SKLEARN:
            raise ImportError("MLRegressionBacktest requires scikit-learn.")
        self.lags = lags
        self.train_ratio = train_ratio
        self._model = LinearRegression(fit_intercept=True)
        self._hitratio: float | None = None
        self._feature_columns: list[str] = []
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return (f"MLRegressionBacktest(symbol={self.symbol}, lags={self.lags}, "
                f"train_ratio={self.train_ratio}, granularity={self.granularity})")

    def prepare_data(self) -> pd.DataFrame:
        df = self._data.copy()
        for lag in range(1, self.lags + 1):
            df[f"lag_{lag}"] = df["returns"].shift(lag)
        self._feature_columns = [f"lag_{lag}" for lag in range(1, self.lags + 1)]
        return df

    def get_hitratio(self) -> float:
        if self._hitratio is None:
            raise RuntimeError("Call .test() first.")
        return self._hitratio

    def test(self, mute: bool = False) -> Tuple[float, float]:
        if not mute:
            log.info(f"Testing ML Regression with lags={self.lags}, "
                     f"train_ratio={self.train_ratio} ...")
        data = self._data.copy().dropna(subset=self._feature_columns + ["returns"])
        if data.empty:
            log.warning("No data after lag dropna.")
            return 1.0, 0.0

        split_idx = int(len(data) * self.train_ratio)
        train = data.iloc[:split_idx]
        test = data.iloc[split_idx:]
        if len(train) < 50 or len(test) < 10:
            log.warning(f"Train ({len(train)}) or test ({len(test)}) too small.")
            return 1.0, 0.0

        # Fit on continuous returns
        X_train = train[self._feature_columns]
        y_train = train["returns"]
        self._model.fit(X_train, y_train)

        # Predict and take sign
        X_test = test[self._feature_columns]
        test = test.copy()
        test["prediction"] = np.sign(self._model.predict(X_test))
        test["position"] = test["prediction"]
        test["strategy"] = test["position"] * test["returns"]

        # Hit ratio
        hits = np.sign(test["returns"] * test["prediction"]).value_counts()
        if 1.0 in hits.index and -1.0 in hits.index:
            self._hitratio = float(hits[1.0] / (hits[1.0] + hits[-1.0]))
        elif 1.0 in hits.index:
            self._hitratio = 1.0
        else:
            self._hitratio = 0.0

        self._data = test
        perf, outperf, n_trades = self._compute_performance(test)
        if not mute:
            log.info(f"Return: {(perf-1)*100:.2f}%, OutPerformance: {outperf*100:.2f}%, "
                     f"Hit Ratio: {self._hitratio:.3f}, Trades: {n_trades}")
        return perf, outperf
