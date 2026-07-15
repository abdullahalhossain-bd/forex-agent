# ml/pattern_features.py — Pattern-specific ML feature engineering
# =============================================================================
# Ported from: https://github.com/MaxwellMendenhall/ml_backtest
# Original: ml_backtest/machine_learning/data.py → CandleStickDataProcessing
# Original author: Maxwell Mendenhall — MIT license
#
# Per-pattern feature calculators that turn candlestick pattern OHLC data into
# ML-friendly numeric features. Each returns a 1-D numpy array of floats.
#
# These features feed into ml/optimal_tp_predictor.py to train a regressor that
# predicts the optimal take-profit distance for each new pattern occurrence,
# instead of using a fixed pip-based or USD-based TP.
#
# Why this is valuable
# --------------------
# Most ML-on-price work uses generic features (lagged returns, MACD, RSI).
# This module uses PATTERN-SPECIFIC features:
#   - For an Engulfing pattern: the engulfing_ratio (current body / prev body)
#   - For a Hammer: upper_to_body_ratio + body_to_total_ratio
#   - For a Morning Star: body_length_ratio (3rd/1st candle) + middle_range_to_body
#
# These features capture the GEOMETRY of the pattern, which is what predicts
# how far the resulting move will go. Generic features miss this.
#
# Faithful to the original Mendenhall implementation, with two changes:
#   1. Each calculator is a standalone function (not a static method on a class).
#   2. Added small epsilon (0.001) to all denominators to avoid ZeroDivisionError
#      on edge-case candles (already in the original).
# =============================================================================

from __future__ import annotations

import numpy as np


# ── 1-bar pattern features ───────────────────────────────────────────────────

def basic_features(current_open: float, current_close: float,
                current_high: float, current_low: float) -> np.ndarray:
    """
    Generic 4-feature vector for any single candle.
    Used by Hammer, Inverted Hammer, Dragonfly Doji, and as a building block.
    """
    body_length = abs(current_open - current_close)
    upper_shadow = current_high - max(current_open, current_close)
    lower_shadow = min(current_open, current_close) - current_low
    candlestick_length = current_high - current_low
    return np.array([body_length, upper_shadow, lower_shadow, candlestick_length],
                    dtype=float)


def hammer_features(current_open: float, current_close: float,
                    current_high: float, current_low: float) -> np.ndarray:
    """
    Features for Hammer / Hanging Man patterns.
    Returns: [lower_to_body_ratio, body_to_total_ratio]
    """
    body_length = abs(current_open - current_close)
    lower_shadow = min(current_open, current_close) - current_low
    total_length = current_high - current_low
    lower_to_body_ratio = lower_shadow / (body_length + 0.001)
    body_to_total_ratio = body_length / (total_length + 0.001)
    return np.array([lower_to_body_ratio, body_to_total_ratio], dtype=float)


def inverted_hammer_features(current_open: float, current_close: float,
                             current_high: float, current_low: float) -> np.ndarray:
    """
    Features for Inverted Hammer / Shooting Star patterns.
    Returns: [upper_to_body_ratio, body_to_total_ratio]
    """
    body_length = abs(current_open - current_close)
    upper_shadow = current_high - max(current_open, current_close)
    total_length = current_high - current_low
    upper_to_body_ratio = upper_shadow / (body_length + 0.001)
    body_to_total_ratio = body_length / (total_length + 0.001)
    return np.array([upper_to_body_ratio, body_to_total_ratio], dtype=float)


def dragonfly_doji_features(current_open: float, current_close: float,
                            current_high: float, current_low: float) -> np.ndarray:
    """
    Features for Dragonfly Doji / Gravestone Doji patterns.
    Same shape as hammer_features (lower-to-body + body-to-total).
    """
    body_length = abs(current_open - current_close)
    lower_shadow = min(current_open, current_close) - current_low
    total_length = current_high - current_low
    lower_to_body_ratio = lower_shadow / (body_length + 0.001)
    body_to_total_ratio = body_length / (total_length + 0.001)
    return np.array([lower_to_body_ratio, body_to_total_ratio], dtype=float)


# ── 2-bar pattern features ───────────────────────────────────────────────────

def engulfing_features(
    current_open: float, current_close: float,
    prev_open: float, prev_close: float,
) -> np.ndarray:
    """
    Features for Bullish/Bearish Engulfing patterns.
    Returns: [engulfing_ratio]  (current body / previous body)
    """
    current_body = abs(current_close - current_open)
    previous_body = abs(prev_close - prev_open)
    engulfing_ratio = current_body / previous_body if previous_body else 0.0
    return np.array([engulfing_ratio], dtype=float)


def harami_features(
    current_open: float, current_close: float,
    prev_open: float, prev_close: float,
) -> np.ndarray:
    """
    Features for Bullish/Bearish Harami patterns.
    Returns: [body_length_ratio]  (current body / previous body)
    """
    current_body_length = abs(current_close - current_open)
    previous_body_length = abs(prev_close - prev_open)
    body_length_ratio = current_body_length / (previous_body_length + 0.001)
    return np.array([body_length_ratio], dtype=float)


def piercing_pattern_features(
    prev_open: float, prev_close: float,
    current_open: float, current_close: float,
) -> np.ndarray:
    """
    Features for Piercing Line / Dark Cloud Cover patterns.
    Returns: [body_length_ratio, penetration_ratio]
    penetration_ratio = (current_close - prev_close) / (prev_open - prev_close)
    """
    previous_body_length = abs(prev_close - prev_open)
    current_body_length = abs(current_close - current_open)
    body_length_ratio = current_body_length / (previous_body_length + 0.001)
    penetration_ratio = (current_close - prev_close) / (prev_open - prev_close + 0.001)
    return np.array([body_length_ratio, penetration_ratio], dtype=float)


# ── 3-bar pattern features ───────────────────────────────────────────────────

def morning_star_features(
    b_prev_open: float, b_prev_close: float,    # 1st candle (large)
    prev_open: float, prev_close: float,        # 2nd candle (star)
    current_open: float, current_close: float,  # 3rd candle (large)
) -> np.ndarray:
    """
    Features for Morning Star / Evening Star patterns.
    Returns: [body_length_ratio, middle_range_to_body_ratio]
      - body_length_ratio = 3rd body / 1st body
      - middle_range_to_body_ratio = middle range / 3rd body
    """
    first_candle_body_length = abs(b_prev_close - b_prev_open)
    third_candle_body_length = abs(current_close - current_open)
    middle_candle_range = abs(prev_close - prev_open)
    body_length_ratio = third_candle_body_length / (first_candle_body_length + 0.001)
    middle_range_to_body_ratio = middle_candle_range / (third_candle_body_length + 0.001)
    return np.array([body_length_ratio, middle_range_to_body_ratio], dtype=float)


def morning_star_doji_features(
    b_prev_open: float, b_prev_close: float,    # 1st candle
    prev_high: float, prev_low: float,          # 2nd candle (doji) — only H/L needed
    current_open: float, current_close: float,  # 3rd candle
) -> np.ndarray:
    """
    Features for Morning Doji Star / Evening Doji Star patterns.
    Returns: [first_to_third_body_ratio, doji_to_body_ratio]
    """
    doji_range = prev_high - prev_low
    third_candle_body_length = abs(current_close - current_open)
    first_to_third_body_ratio = abs(b_prev_close - b_prev_open) / (third_candle_body_length + 0.001)
    doji_to_body_ratio = doji_range / (third_candle_body_length + 0.001)
    return np.array([first_to_third_body_ratio, doji_to_body_ratio], dtype=float)


# ── Registry: pattern name → feature calculator ─────────────────────────────

PATTERN_FEATURE_CALCULATORS = {
    "Hammer":              hammer_features,
    "Inverted Hammer":     inverted_hammer_features,
    "Dragonfly Doji":      dragonfly_doji_features,
    "Gravestone Doji":     inverted_hammer_features,   # mirror of Dragonfly
    "Bullish Engulfing":   engulfing_features,
    "Bearish Engulfing":   engulfing_features,
    "Bullish Harami":      harami_features,
    "Bearish Harami":      harami_features,
    "Piercing Line":       piercing_pattern_features,
    "Dark Cloud Cover":    piercing_pattern_features,
    "Morning Star":        morning_star_features,
    "Evening Star":        morning_star_features,
    "Morning Doji Star":   morning_star_doji_features,
    "Evening Doji Star":   morning_star_doji_features,
}


def get_feature_calculator(pattern_name: str):
    """Look up the feature calculator for a pattern. Returns None if unknown."""
    return PATTERN_FEATURE_CALCULATORS.get(pattern_name)


def feature_vector_length(pattern_name: str) -> int:
    """Return the number of features produced by the calculator for this pattern."""
    fn = get_feature_calculator(pattern_name)
    if fn is None:
        return 0
    # Call with dummy args to measure output length
    # (all calculators take the same shape per category — 1-bar, 2-bar, or 3-bar)
    dummy_1bar = fn(1.0, 1.0, 1.0, 1.0) if pattern_name in {
        "Hammer", "Inverted Hammer", "Dragonfly Doji", "Gravestone Doji"
    } else None
    if dummy_1bar is not None:
        return len(dummy_1bar)
    if pattern_name in {"Bullish Engulfing", "Bearish Engulfing",
                        "Bullish Harami", "Bearish Harami",
                        "Piercing Line", "Dark Cloud Cover"}:
        return len(fn(1.0, 1.0, 1.0, 1.0))
    # 3-bar
    if pattern_name in {"Morning Star", "Evening Star"}:
        return len(fn(1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    if pattern_name in {"Morning Doji Star", "Evening Doji Star"}:
        return len(fn(1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    return 0


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Hammer features
    f = hammer_features(current_open=1.0, current_close=1.02,
                        current_high=1.03, current_low=0.95)
    print(f"Hammer features: {f}  (len={len(f)})")
    assert len(f) == 2

    # Engulfing features
    f = engulfing_features(current_open=1.0, current_close=1.10, prev_open=1.05, prev_close=0.95)
    print(f"Engulfing features: {f}  (len={len(f)})")
    assert len(f) == 1

    # Piercing features
    f = piercing_pattern_features(prev_open=1.10, prev_close=0.90,
                                   current_open=0.85, current_close=1.05)
    print(f"Piercing features: {f}  (len={len(f)})")
    assert len(f) == 2

    # Morning Star features
    f = morning_star_features(b_prev_open=1.10, b_prev_close=0.90,
                              prev_open=0.90, prev_close=0.91,
                              current_open=0.91, current_close=1.15)
    print(f"Morning Star features: {f}  (len={len(f)})")
    assert len(f) == 2

    # Registry check
    assert get_feature_calculator("Hammer") is not None
    assert get_feature_calculator("Unknown Pattern") is None
    assert feature_vector_length("Hammer") == 2
    assert feature_vector_length("Bullish Engulfing") == 1
    assert feature_vector_length("Morning Star") == 2

    print("\nAll pattern features smoke tests passed.")
