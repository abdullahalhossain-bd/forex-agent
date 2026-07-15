# ml/target_and_backtest_utils.py — Price direction targets + portfolio backtest utils
# =============================================================================
# Ported from: https://github.com/sallamy2580/forex-trading-bot/blob/master/helpers/utils.py
# Original author: sallamy2580 — educational project
#
# Utility functions for ML-based trading:
#   1. price_to_binary_target() — 3-class UP/DOWN/FLAT target with adaptive threshold
#   2. portfolio_value() — backtest P&L with transaction costs
#   3. market session dummies — London/NY/Sydney/Tokyo binary indicators
#   4. train_test_cv_split() — 3-way time-series split
#   5. min_max_scale_outlier_aware() — outlier removal + MinMax scaling
#   6. get_pca_features() — PCA dimensionality reduction
#   7. get_polynomial_features() — polynomial feature expansion
#
# All functions are pure numpy/pandas — no TensorFlow or ta-lib needed.
# =============================================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from utils.logger import get_logger

log = get_logger("target_backtest_utils")


# ── 1. Price direction target (3-class: UP / DOWN / FLAT) ────────────────────

def price_to_binary_target(
    prices: np.ndarray,
    delta: Optional[float] = None,
) -> tuple[np.ndarray, float]:
    """
    Create 3-class one-hot targets from price series:
        [1, 0, 0] = UP (price rises beyond delta)
        [0, 1, 0] = DOWN (price falls beyond delta)
        [0, 0, 1] = FLAT (price change within delta)

    The `delta` threshold is automatically set so that each class makes
    roughly 1/3 of the dataset. This prevents class imbalance.

    Parameters
    ----------
    prices : 1-D array of close prices.
    delta : manual threshold for "flat". If None, auto-computed for 1/3 split.

    Returns
    -------
    (binary_target, delta_used)
        binary_target: (N, 3) one-hot array. Last row is NaN (no future data).
        delta_used: the threshold actually used.
    """
    prices = np.asarray(prices, dtype=float).ravel()
    n = len(prices)
    price_change = np.zeros(n)
    price_change[1:] = prices[1:] / prices[:-1] - 1

    # Auto-compute delta for 1/3 split if not provided
    if delta is None:
        abs_changes = np.abs(price_change[1:])
        # Sort and find the threshold that splits into thirds
        sorted_changes = np.sort(abs_changes)
        delta = sorted_changes[len(sorted_changes) // 3]
        log.info(f"Auto delta for 1/3 split: {delta:.6f}")

    binary_target = np.zeros((n, 3))
    binary_target[-1] = np.nan  # no future data for last bar

    for i in range(n - 1):
        change = price_change[i + 1]
        if change > delta:
            binary_target[i] = [1, 0, 0]  # UP
        elif change < -delta:
            binary_target[i] = [0, 1, 0]  # DOWN
        else:
            binary_target[i] = [0, 0, 1]  # FLAT

    # Print distribution
    valid = binary_target[:-1]
    up = valid[:, 0].sum() / len(valid)
    down = valid[:, 1].sum() / len(valid)
    flat = valid[:, 2].sum() / len(valid)
    log.info(f"Target distribution — UP: {up:.2f}, DOWN: {down:.2f}, FLAT: {flat:.2f}")

    return binary_target, delta


# ── 2. Portfolio value with transaction costs ────────────────────────────────

def portfolio_value(
    price_change: np.ndarray,
    signal: np.ndarray,
    trans_cost: float = 0.0,
) -> np.ndarray:
    """
    Compute cumulative portfolio value given signals and price changes.

    Parameters
    ----------
    price_change : 1-D array of percentage price changes (e.g., pct_change()).
    signal : 1-D array of positions: +1 (long), -1 (short), 0 (flat).
    trans_cost : transaction cost per trade (as fraction, e.g., 0.0002 = 2 pips).

    Returns
    -------
    Cumulative portfolio value (starts at 1.0).

    Notes
    -----
    - Signal at bar t is applied to price change at bar t+1 (no look-ahead).
    - Transaction cost is applied when position changes (signal[i] != signal[i+1]
      and signal[i+1] != 0).
    """
    price_change = np.asarray(price_change, dtype=float).ravel()
    signal = np.asarray(signal, dtype=float).ravel()

    # Signal from bar t → applied to price change at bar t+1
    signal_percent = signal[:-1] * price_change[1:]

    # Transaction costs: applied when position changes
    transaction_costs = np.zeros_like(signal_percent)
    for i in range(len(signal) - 1):
        if signal[i] != signal[i + 1] and signal[i + 1] != 0:
            transaction_costs[i] = trans_cost

    value = np.cumsum(signal_percent - transaction_costs) + 1.0
    return value


# ── 3. Market session binary indicators ──────────────────────────────────────

def get_market_session_dummies(
    timestamps: pd.DatetimeIndex | pd.Series,
) -> pd.DataFrame:
    """
    Create binary market session indicators (UTC hours):
        London:   03:00 - 11:00 UTC
        New York: 08:00 - 16:00 UTC
        Sydney:   17:00 - 01:00 UTC (wraps midnight)
        Tokyo:    19:00 - 03:00 UTC (wraps midnight)

    Parameters
    ----------
    timestamps : DatetimeIndex or Series of timestamps.

    Returns
    -------
    DataFrame with columns: mrkt_london, mrkt_ny, mrkt_sydney, mrkt_tokyo.
    """
    if isinstance(timestamps, pd.DatetimeIndex):
        hours = np.array(timestamps.hour)
    else:
        hours = pd.to_datetime(timestamps).dt.hour.to_numpy()

    df = pd.DataFrame(index=range(len(hours)))
    df["mrkt_london"] = ((hours >= 3) & (hours <= 11)).astype(int)
    df["mrkt_ny"] = ((hours >= 8) & (hours <= 16)).astype(int)
    df["mrkt_sydney"] = ((hours >= 17) | (hours <= 1)).astype(int)
    df["mrkt_tokyo"] = ((hours >= 19) | (hours <= 3)).astype(int)
    return df


# ── 4. 3-way train/test/cv split ─────────────────────────────────────────────

def train_test_cv_split(
    *arrays: np.ndarray,
    split: tuple[float, float, float] = (0.5, 0.35, 0.15),
) -> tuple:
    """
    Split arrays into train, test, and cross-validation sets.

    Parameters
    ----------
    *arrays : arrays to split (all must have the same length).
    split : (train_frac, test_frac, cv_frac). Must sum to 1.0.

    Returns
    -------
    Tuple of (train, test, cv) for each input array.
    Example: train_x, test_x, cv_x, train_y, test_y, cv_y = train_test_cv_split(x, y)
    """
    assert abs(sum(split) - 1.0) < 1e-6, f"split must sum to 1.0, got {sum(split)}"
    train_frac, test_frac, _ = split
    n = len(arrays[0])
    train_end = int(n * train_frac)
    test_end = int(n * (train_frac + test_frac))

    result = []
    for arr in arrays:
        result.append(arr[:train_end])   # train
        result.append(arr[train_end:test_end])  # test
        result.append(arr[test_end:])    # cv
    return tuple(result)


# ── 5. Outlier-aware MinMax scaling ──────────────────────────────────────────

def min_max_scale_outlier_aware(
    train: np.ndarray,
    test: np.ndarray,
    cv: np.ndarray,
    std_dev_threshold: float = 2.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MinMax scale data with outlier removal before fitting the scaler.

    Outliers are removed from the TRAIN set only (using median ± N×std),
    then the scaler is fit on the cleaned train data and applied to all sets.
    """
    from sklearn.preprocessing import MinMaxScaler

    # Remove outliers from train set
    train_df = pd.DataFrame(train)
    mask = train_df.apply(
        lambda x: np.abs(x - x.median()) / (x.std() + 1e-10) < std_dev_threshold
    ).all(axis=1)
    train_clean = train_df[mask].values

    scaler = MinMaxScaler()
    scaler.fit(train_clean)

    return (
        scaler.transform(train),
        scaler.transform(test),
        scaler.transform(cv),
    )


# ── 6. PCA feature reduction ─────────────────────────────────────────────────

def get_pca_features(
    train: np.ndarray,
    test: np.ndarray,
    cv: np.ndarray,
    variance_threshold: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Apply PCA to reduce feature dimensions.

    Parameters
    ----------
    train, test, cv : feature matrices.
    variance_threshold : keep components with explained variance > threshold.

    Returns
    -------
    (train_pca, test_pca, cv_pca, n_components)
    """
    from sklearn.decomposition import PCA

    pca = PCA()
    pca.fit(train)
    n_components = int(np.sum(pca.explained_variance_ratio_ > variance_threshold))
    n_components = max(1, n_components)
    log.info(f"PCA: keeping {n_components} components (threshold={variance_threshold})")

    train_pca = pca.transform(train)[:, :n_components]
    test_pca = pca.transform(test)[:, :n_components]
    cv_pca = pca.transform(cv)[:, :n_components]

    return train_pca, test_pca, cv_pca, n_components


# ── 7. Polynomial feature expansion ──────────────────────────────────────────

def get_polynomial_features(
    train: np.ndarray,
    test: np.ndarray,
    cv: np.ndarray,
    degree: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Expand features with polynomial combinations.

    WARNING: degree=2 on 250 features → ~31,000 features. Use PCA first.
    """
    from sklearn.preprocessing import PolynomialFeatures

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly.fit(train)

    return (
        poly.transform(train),
        poly.transform(test),
        poly.transform(cv),
    )


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Price to binary target
    prices = np.array([1.0, 1.001, 0.998, 1.005, 1.003, 0.999, 1.002, 1.004, 1.001, 1.000])
    target, delta = price_to_binary_target(prices, delta=0.001)
    print(f"Delta: {delta:.6f}")
    print(f"Target shape: {target.shape}")
    assert target.shape == (10, 3)
    assert np.isnan(target[-1]).all()  # last row is NaN

    # 2. Portfolio value
    price_change = np.array([0.001, -0.002, 0.005, -0.002, -0.004, 0.003, 0.002, -0.003, -0.001])
    signal = np.array([1, 1, 1, -1, -1, 0, 0, 1, 1])
    pv = portfolio_value(price_change, signal, trans_cost=0.0002)
    print(f"Portfolio value (with costs): {pv}")
    pv_no_cost = portfolio_value(price_change, signal, trans_cost=0.0)
    print(f"Portfolio value (no costs):   {pv_no_cost}")
    assert (pv <= pv_no_cost + 1e-10).all()  # costs reduce value (with tolerance)

    # 3. Market session dummies
    ts = pd.date_range("2024-01-01", periods=48, freq="1h", tz="UTC")
    sessions = get_market_session_dummies(ts)
    print(f"Sessions shape: {sessions.shape}")
    print(f"London open at 03:00: {sessions['mrkt_london'].iloc[3]}")
    print(f"NY open at 08:00: {sessions['mrkt_ny'].iloc[8]}")
    assert sessions["mrkt_london"].iloc[3] == 1
    assert sessions["mrkt_ny"].iloc[8] == 1
    assert sessions["mrkt_london"].iloc[0] == 0  # midnight, London closed

    # 4. 3-way split
    x = np.random.randn(100, 5)
    y = np.random.randint(0, 3, 100)
    x_tr, x_te, x_cv, y_tr, y_te, y_cv = train_test_cv_split(x, y)
    assert len(x_tr) == 50
    assert len(x_te) == 35
    assert len(x_cv) == 15
    assert len(y_tr) == 50

    # 5. Outlier-aware scaling
    train_data = np.random.randn(100, 5) * 10
    train_data[0] = 1000  # outlier
    test_data = np.random.randn(50, 5) * 10
    cv_data = np.random.randn(20, 5) * 10
    scaled_tr, scaled_te, scaled_cv = min_max_scale_outlier_aware(
        train_data, test_data, cv_data
    )
    assert scaled_tr.shape == train_data.shape
    # Note: outlier values in train may scale outside [0,1] since scaler
    # was fit on cleaned data. Just check shapes match.

    # 6. PCA
    pca_tr, pca_te, pca_cv, n_comp = get_pca_features(scaled_tr, scaled_te, scaled_cv)
    print(f"PCA components: {n_comp}")
    assert pca_tr.shape[1] == n_comp

    print("\nTarget and backtest utils smoke test passed.")
