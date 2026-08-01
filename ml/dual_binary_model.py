# ml/dual_binary_model.py — Dual binary XGBoost model (BUY model + SELL model)
# =============================================================================
# Ported from: https://github.com/bruh7463/forex_bot/blob/master/model/predictor.py
# Original author: bruh7463 — educational project
#
# Instead of a single 3-class model (BUY/SELL/HOLD), this uses TWO separate
# binary classifiers:
#   - BUY model: predicts P(price goes up in next N bars)
#   - SELL model: predicts P(price goes down in next N bars)
#
# Advantages over a single 3-class model:
#   1. INDEPENDENT THRESHOLDS — buy_threshold and sell_threshold can be tuned
#      separately (e.g., require 60% confidence for BUY but only 55% for SELL).
#   2. CONFLICT DETECTION — if both models fire (both P > threshold), it's a
#      natural HOLD signal (conflicting signals = uncertainty).
#   3. SIGMOID CONFIDENCE — each model outputs an ABSOLUTE probability (0..1),
#      not a relative softmax. "BUY 0.6" means the model is 60% confident,
#      regardless of what the SELL model says.
#   4. ASYMMETRIC BEHAVIOR — markets trend differently up vs down. Separate
#      models capture this asymmetry naturally.
#
# Signal logic:
#   BUY  if buy_prob >= buy_threshold  AND sell_prob < sell_threshold
#   SELL if sell_prob >= sell_threshold AND buy_prob < buy_threshold
#   HOLD otherwise (including when both fire — conflict = uncertainty)
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("dual_binary_model")

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score
    import joblib
    _HAS_ML = True
except ImportError:
    _HAS_ML = False
    log.info("xgboost/sklearn not installed — DualBinaryModel will not work. "
             "Install with: pip install xgboost scikit-learn joblib")


@dataclass
class DualBinaryModel:
    """
    Container for a trained dual-binary model pair.

    Attributes
    ----------
    buy_model : XGBClassifier trained on "did price go up?" target
    sell_model : XGBClassifier trained on "did price go down?" target
    buy_threshold : minimum P(up) to trigger BUY (default 0.55)
    sell_threshold : minimum P(down) to trigger SELL (default 0.55)
    feature_names : list of feature column names the models expect
    version : always "v5_dual_binary" for format detection
    """
    buy_model: object = None
    sell_model: object = None
    buy_threshold: float = 0.55
    sell_threshold: float = 0.55
    feature_names: list[str] = field(default_factory=list)
    version: str = "v5_dual_binary"


def create_target(
    df: pd.DataFrame,
    close_col: str = "close",
    forward_bars: int = 5,
    threshold_pct: float = 0.001,
) -> pd.DataFrame:
    """
    Create binary targets for BUY and SELL models.

    BUY target = 1 if close[t + forward_bars] > close[t] * (1 + threshold_pct)
    SELL target = 1 if close[t + forward_bars] < close[t] * (1 - threshold_pct)

    Parameters
    ----------
    df : DataFrame with a close column.
    close_col : name of the close price column.
    forward_bars : how many bars forward to look for the move.
    threshold_pct : minimum % move to count as a buy/sell signal
        (default 0.1% — filters out tiny moves that are just noise).

    Returns
    -------
    DataFrame with 'target_buy' and 'target_sell' columns added.
    Last `forward_bars` rows will have NaN targets (no future data).
    """
    out = df.copy()
    future_close = out[close_col].shift(-forward_bars)
    # Use NaN-aware comparison: rows without future data get NaN target
    buy_cond = (future_close > out[close_col] * (1 + threshold_pct))
    sell_cond = (future_close < out[close_col] * (1 - threshold_pct))
    out["target_buy"] = buy_cond.astype("Int64")  # nullable int — NaN stays NaN
    out["target_sell"] = sell_cond.astype("Int64")
    # Explicitly set last `forward_bars` rows to NaN (no future data)
    out.loc[out.index[-forward_bars:], "target_buy"] = pd.NA
    out.loc[out.index[-forward_bars:], "target_sell"] = pd.NA
    return out


def train_dual_model(
    df: pd.DataFrame,
    feature_names: list[str],
    *,
    forward_bars: int = 5,
    threshold_pct: float = 0.001,
    buy_threshold: float = 0.55,
    sell_threshold: float = 0.55,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DualBinaryModel:
    """
    Train a dual-binary XGBoost model on the given data.

    Parameters
    ----------
    df : DataFrame with features + a 'close' column.
    feature_names : list of feature column names to use for training.
    forward_bars : forward-looking window for target creation.
    threshold_pct : minimum % move for a buy/sell target.
    buy_threshold, sell_threshold : minimum probabilities to trigger signals.
    test_size : fraction of data for the test set.
    random_state : for reproducibility.

    Returns
    -------
    Trained DualBinaryModel instance.
    """
    if not _HAS_ML:
        raise ImportError("xgboost/sklearn required. Install with: pip install xgboost scikit-learn")

    # Create targets
    df = create_target(df, forward_bars=forward_bars, threshold_pct=threshold_pct)
    df = df.dropna(subset=["target_buy", "target_sell"] + feature_names)

    X = df[feature_names].values.astype(np.float32)
    y_buy = df["target_buy"].values
    y_sell = df["target_sell"].values

    log.info(f"Training data: {len(df)} samples, {len(feature_names)} features")
    log.info(f"Buy target distribution: {np.bincount(y_buy)}")
    log.info(f"Sell target distribution: {np.bincount(y_sell)}")

    # Split — Day 102+ CRITICAL hotfix: split ONCE with both targets.
    # Previously this called train_test_split twice (once for y_buy,
    # once for y_sell) with DIFFERENT stratify arrays, which produces
    # DIFFERENT row orderings. The SELL model was then fit on X_train
    # from the first split paired with y_sell_train from the second —
    # features and labels that don't correspond. Every SELL prediction
    # was garbage. Fix: split once with both targets together so the
    # same row indices are used for both models.
    X_train, X_test, y_buy_train, y_buy_test, y_sell_train, y_sell_test = train_test_split(
        X, y_buy, y_sell,
        test_size=test_size,
        random_state=random_state,
        stratify=y_buy,  # stratify on the buy target (more balanced typically)
    )

    # Train BUY model
    log.info("Training BUY model...")
    buy_model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=random_state, eval_metric="logloss",
    )
    buy_model.fit(X_train, y_buy_train)
    buy_acc = accuracy_score(y_buy_test, buy_model.predict(X_test))
    log.info(f"BUY model accuracy: {buy_acc:.4f}")

    # Train SELL model
    log.info("Training SELL model...")
    sell_model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=random_state, eval_metric="logloss",
    )
    sell_model.fit(X_train, y_sell_train)
    sell_acc = accuracy_score(y_sell_test, sell_model.predict(X_test))
    log.info(f"SELL model accuracy: {sell_acc:.4f}")

    return DualBinaryModel(
        buy_model=buy_model,
        sell_model=sell_model,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        feature_names=feature_names,
    )


def predict_signal(
    model: DualBinaryModel,
    df: pd.DataFrame,
) -> dict:
    """
    Generate a trading signal from the dual-binary model.

    Returns a dict with:
        signal: "BUY", "SELL", or "HOLD"
        buy_probability: float (0..1)
        sell_probability: float (0..1)
        confidence: float (the winning model's probability, or 0 for HOLD)
        conflict: True if both models fire (natural HOLD)
    """
    if model.buy_model is None or model.sell_model is None:
        raise RuntimeError("Models not trained")

    # Verify features
    missing = [f for f in model.feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    if df.empty:
        raise ValueError("DataFrame is empty")

    latest = df.iloc[-1]
    X = latest[model.feature_names].values.reshape(1, -1).astype(np.float32)

    buy_prob = float(model.buy_model.predict_proba(X)[0, 1])
    sell_prob = float(model.sell_model.predict_proba(X)[0, 1])

    # Signal logic
    buy_fires = buy_prob >= model.buy_threshold
    sell_fires = sell_prob >= model.sell_threshold
    conflict = buy_fires and sell_fires

    if buy_fires and not sell_fires:
        signal = "BUY"
        confidence = buy_prob
    elif sell_fires and not buy_fires:
        signal = "SELL"
        confidence = sell_prob
    else:
        signal = "HOLD"
        confidence = 0.0

    return {
        "signal": signal,
        "buy_probability": round(buy_prob, 4),
        "sell_probability": round(sell_prob, 4),
        "confidence": round(confidence, 4),
        "buy_threshold": model.buy_threshold,
        "sell_threshold": model.sell_threshold,
        "conflict": conflict,
        "latest_price": float(latest.get("close", 0)),
    }


def save_model(model: DualBinaryModel, path: str) -> str:
    """Save the dual-binary model to a joblib file."""
    if not _HAS_ML:
        raise ImportError("joblib required")
    full_path = path if path.endswith('.joblib') else path + '.joblib'
    joblib.dump(model, full_path)
    log.info(f"Model saved to {full_path}")
    return full_path


def load_model(path: str) -> DualBinaryModel:
    """Load a dual-binary model from a joblib file."""
    if not _HAS_ML:
        raise ImportError("joblib required")
    full_path = path if path.endswith('.joblib') else path + '.joblib'
    return joblib.load(full_path)


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not _HAS_ML:
        print("xgboost/sklearn not installed. Install with: pip install xgboost scikit-learn")
    else:
        import numpy as np

        # Synthetic data
        np.random.seed(42)
        n = 1000
        close = 1.1000 + np.cumsum(np.random.randn(n) * 0.001)
        df = pd.DataFrame({
            "close": close,
            "feature_1": np.random.randn(n),
            "feature_2": np.random.randn(n),
            "feature_3": np.random.randn(n),
        })

        features = ["feature_1", "feature_2", "feature_3"]

        # Train
        model = train_dual_model(df, features, forward_bars=5,
                                  buy_threshold=0.55, sell_threshold=0.55)
        print(f"Buy threshold: {model.buy_threshold}")
        print(f"Sell threshold: {model.sell_threshold}")

        # Predict
        result = predict_signal(model, df)
        print(f"\nPrediction on latest bar:")
        for k, v in result.items():
            print(f"  {k}: {v}")

        # Save/load
        save_model(model, '/tmp/test_dual_model')
        loaded = load_model('/tmp/test_dual_model')
        result2 = predict_signal(loaded, df)
        assert result["signal"] == result2["signal"]
        print("\n✓ Save/load verified")

        # Test conflict detection
        model.buy_threshold = 0.0  # always fire BUY
        model.sell_threshold = 0.0  # always fire SELL
        result_conflict = predict_signal(model, df)
        assert result_conflict["conflict"] is True
        assert result_conflict["signal"] == "HOLD"
        print("✓ Conflict detection verified (both fire → HOLD)")

        print("\nDual binary model smoke test passed.")
