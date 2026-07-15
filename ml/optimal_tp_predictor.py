# ml/optimal_tp_predictor.py — ML regressor that predicts optimal TP distance
# =============================================================================
# Ported from: https://github.com/MaxwellMendenhall/ml_backtest
# Original: ml_backtest/machine_learning/wrapper.py + ml_backtest/models/rfr.py
# Original author: Maxwell Mendenhall — MIT license
#
# Trains a regressor (RandomForestRegressor by default) on past trade outcomes
# to predict the optimal take-profit distance for each new pattern occurrence.
#
# Pipeline
# --------
#   1. Run a Backtest with a Strategy that uses a FIXED take-profit.
#   2. Collect trades: each has entry_price, exit_price, target = high - entry.
#   3. For each trade, extract pattern features (ml/pattern_features.py) from
#      the metadata stored at entry.
#   4. Train: X = pattern features, y = target (how far price moved favorably).
#   5. Save the model. At trade time, call model.predict(features) to get
#      a per-trade TP instead of using a fixed value.
#
# Why this is valuable
# --------------------
# A Hammer with a long lower shadow (3x body) typically produces a larger
# follow-through than a Hammer with a 2.1x lower shadow. A fixed TP leaves
# money on the table for the strong setups and gets stopped out on the weak
# ones. An ML-predicted TP captures this geometry → higher win rate + larger
# average winner.
#
# Differences from the original Mendenhall implementation:
#   - Decoupled from the ml_backtest MachineLearningInterface class hierarchy.
#     Any sklearn regressor works (RandomForest, GradientBoosting, etc.).
#   - Optional target engineering (clip extreme outliers).
#   - Optional feature engineering hook (add EMA/RSI/MACD columns).
#   - Persists model via joblib (same as original).
# =============================================================================

from __future__ import annotations

import os
from typing import Callable, Optional, List

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("optimal_tp_predictor")

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from joblib import dump, load
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    log.info("scikit-learn not installed — ml.optimal_tp_predictor will not work. "
             "Install with: pip install scikit-learn joblib")


if _HAS_SKLEARN:

    class OptimalTPPredictor:
        """
        Train a regressor to predict optimal take-profit distance for
        candlestick-pattern entries.

        Parameters
        ----------
        trades : DataFrame
            Output of backtest.ml_engine.MLBacktest.get_trades(). Must have:
            - 'entry time' (Unix timestamp or convertible)
            - 'target' (favorable excursion: high - entry for longs)
            - 'metadata' (dict of OHLC values at entry, used by feature_calculator)
        feature_calculator : callable
            One of the functions from ml.pattern_features (e.g.,
            hammer_features, engulfing_features). Called as
            feature_calculator(**trade_metadata) → 1-D np.ndarray.
        model : sklearn regressor, optional
            Defaults to RandomForestRegressor(n_estimators=100, random_state=42).
        test_size : float
            Fraction of data for the test set (default 0.2).
        random_state : int
            For reproducibility.
        """

        def __init__(
            self,
            trades: pd.DataFrame,
            feature_calculator: Callable[..., np.ndarray],
            *,
            model=None,
            test_size: float = 0.2,
            random_state: int = 42,
        ):
            if trades.empty:
                raise ValueError("trades DataFrame is empty — run a backtest first")
            if 'metadata' not in trades.columns:
                raise ValueError("trades must have a 'metadata' column "
                                 "(set when calling strategy.buy/sell)")
            if 'target' not in trades.columns:
                raise ValueError("trades must have a 'target' column "
                                 "(set by the backtest engine from highest-high)")

            self.trades = trades
            self.feature_calculator = feature_calculator
            self.model = model or RandomForestRegressor(
                n_estimators=100, random_state=random_state
            )
            self.test_size = test_size
            self.random_state = random_state

            self._X: Optional[np.ndarray] = None
            self._y: Optional[np.ndarray] = None
            self._predictions: Optional[np.ndarray] = None
            self._mse: Optional[float] = None
            self._feature_length: Optional[int] = None

        # ── Feature extraction ──────────────────────────────────────────────

        def extract_features(self) -> tuple[np.ndarray, np.ndarray]:
            """
            Build X (features) and y (targets) from trades.

            X is built by calling self.feature_calculator(**trade.metadata)
            for each trade. y is the 'target' column (favorable excursion).
            """
            features_list = []
            targets = []
            for _, trade in self.trades.iterrows():
                metadata = trade['metadata']
                if not isinstance(metadata, dict):
                    continue
                try:
                    features = self.feature_calculator(**metadata)
                    features_list.append(features)
                    targets.append(trade['target'])
                except (TypeError, ValueError) as e:
                    log.warning(f"Skipping trade {trade.name}: feature calc failed: {e}")
                    continue

            if not features_list:
                raise ValueError("No valid features extracted — check that "
                                 "feature_calculator matches the trade metadata")

            X = np.vstack(features_list)
            y = np.array(targets, dtype=float)
            self._X = X
            self._y = y
            self._feature_length = X.shape[1]
            log.info(f"Extracted {X.shape[0]} samples × {X.shape[1]} features")
            return X, y

        # ── Optional target engineering ──────────────────────────────────────

        def clip_targets(self, lower: Optional[float] = None,
                         upper: Optional[float] = None) -> None:
            """
            Clip extreme target values to reduce outlier influence.
            Call after extract_features(), before train().
            """
            if self._y is None:
                raise RuntimeError("Call extract_features() first")
            if lower is not None:
                self._y = np.maximum(self._y, lower)
            if upper is not None:
                self._y = np.minimum(self._y, upper)
            log.info(f"Clipped targets to [{lower}, {upper}]")

        # ── Training ─────────────────────────────────────────────────────────

        def train(self) -> dict:
            """
            Train the model. Returns a dict with MSE and train/test sizes.
            """
            if self._X is None:
                self.extract_features()

            X = np.around(self._X, decimals=4)
            y = self._y

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            self.model.fit(X_train, y_train)
            self._predictions = self.model.predict(X_test)
            self._mse = float(mean_squared_error(y_test, self._predictions))

            log.info(f"Trained {type(self.model).__name__}: MSE={self._mse:.6f}, "
                     f"train_size={len(X_train)}, test_size={len(X_test)}")
            return {
                'mse': self._mse,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'feature_count': X.shape[1],
            }

        # ── Prediction (single trade) ────────────────────────────────────────

        def predict(self, **trade_metadata) -> float:
            """
            Predict optimal TP distance for a new trade given its OHLC metadata.

            Example:
                >>> predictor.predict(current_open=1.0, current_close=1.02,
                ...                   current_high=1.03, current_low=0.95)
                0.0273  # → set TP at entry + 0.0273
            """
            if self.model is None or not hasattr(self.model, 'predict'):
                raise RuntimeError("Model not trained — call train() first")
            features = self.feature_calculator(**trade_metadata)
            return float(self.model.predict(features.reshape(1, -1))[0])

        # ── Persistence ──────────────────────────────────────────────────────

        def save(self, path: str) -> str:
            """Save the trained model to a joblib file."""
            if self.model is None:
                raise RuntimeError("Model not trained — call train() first")
            full_path = path if path.endswith('.joblib') else path + '.joblib'
            dump(self.model, full_path)
            log.info(f"Model saved to {full_path}")
            return full_path

        @classmethod
        def load(cls, path: str, feature_calculator: Callable[..., np.ndarray]) -> "OptimalTPPredictor":
            """
            Load a saved model. Returns a predictor instance ready for .predict().

            Note: the returned instance has trades=None and untrained internal
            state — only .predict() is usable.
            """
            full_path = path if path.endswith('.joblib') else path + '.joblib'
            model = load(full_path)
            # Create a "fake" instance bypassing __init__
            instance = cls.__new__(cls)
            instance.trades = None
            instance.feature_calculator = feature_calculator
            instance.model = model
            instance.test_size = 0.2
            instance.random_state = 42
            instance._X = instance._y = instance._predictions = None
            instance._mse = None
            instance._feature_length = None
            log.info(f"Model loaded from {full_path}")
            return instance

        # ── Introspection ────────────────────────────────────────────────────

        def feature_importances(self) -> Optional[dict]:
            """Return feature importances if the model supports them."""
            if not hasattr(self.model, 'feature_importances_'):
                return None
            return {f"feature_{i}": imp
                    for i, imp in enumerate(self.model.feature_importances_)}

        def get_mse(self) -> Optional[float]:
            return self._mse


# ── Convenience: train a TP predictor from a backtest's trades ───────────────

def train_tp_predictor_from_trades(
    trades: pd.DataFrame,
    pattern_name: str,
    *,
    save_path: Optional[str] = None,
) -> "OptimalTPPredictor":
    """
    End-to-end helper: build a TP predictor for a given pattern from
    a backtest's trades.

    Parameters
    ----------
    trades : DataFrame from MLBacktest.get_trades()
    pattern_name : one of the keys in ml.pattern_features.PATTERN_FEATURE_CALCULATORS
    save_path : if given, save the trained model to this path (without extension)

    Returns
    -------
    Trained OptimalTPPredictor instance.
    """
    if not _HAS_SKLEARN:
        raise ImportError("scikit-learn not installed")

    from ml.pattern_features import get_feature_calculator, PATTERN_FEATURE_CALCULATORS
    calc = get_feature_calculator(pattern_name)
    if calc is None:
        raise ValueError(
            f"No feature calculator for pattern {pattern_name!r}. "
            f"Available: {list(PATTERN_FEATURE_CALCULATORS.keys())}"
        )

    predictor = OptimalTPPredictor(trades, calc)
    predictor.extract_features()
    predictor.train()

    if save_path:
        predictor.save(save_path)

    return predictor


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not _HAS_SKLEARN:
        print("Skipping smoke test — scikit-learn not installed")
    else:
        # Build synthetic trades: 100 Hammer entries with random metadata
        rng = np.random.default_rng(42)
        n = 100
        trades_data = []
        for i in range(n):
            open_ = 1.0 + rng.normal(0, 0.001)
            close = open_ + rng.uniform(0.001, 0.01)  # bullish
            high = close + rng.uniform(0.001, 0.005)
            low = open_ - rng.uniform(0.001, 0.01)
            target = high - open_  # favorable excursion
            trades_data.append({
                'entry time': i,
                'entry price': open_,
                'target': target,
                'metadata': {
                    'current_open': open_,
                    'current_close': close,
                    'current_high': high,
                    'current_low': low,
                },
            })
        trades = pd.DataFrame(trades_data)

        from ml.pattern_features import hammer_features
        predictor = OptimalTPPredictor(trades, hammer_features)
        X, y = predictor.extract_features()
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        result = predictor.train()
        print(f"Training result: {result}")
        print(f"Feature importances: {predictor.feature_importances()}")

        # Predict on a new trade
        pred = predictor.predict(current_open=1.0, current_close=1.005,
                                  current_high=1.008, current_low=0.995)
        print(f"Predicted TP distance for new trade: {pred:.6f}")

        # Save and reload
        predictor.save('/tmp/test_tp_predictor')
        loaded = OptimalTPPredictor.load('/tmp/test_tp_predictor', hammer_features)
        pred2 = loaded.predict(current_open=1.0, current_close=1.005,
                                current_high=1.008, current_low=0.995)
        assert abs(pred - pred2) < 1e-9, "loaded model should give same prediction"
        print(f"Reloaded model gives same prediction: {pred2:.6f}")

        print("\nOptimalTPPredictor smoke test passed.")
