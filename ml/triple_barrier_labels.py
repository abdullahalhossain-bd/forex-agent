# ml/triple_barrier_labels.py — Triple-barrier labeling for ML trading
# =============================================================================
# Ported from: https://github.com/stefan-jansen/machine-learning-for-trading
# Original: 07_defining_the_learning_task/03_label_methods.py
# Original author: Stefan Jansen — MIT license (book companion code)
#
# Triple-barrier method (Marcos López de Prado, "Advances in Financial Machine Learning"):
#
#   For each bar, set three barriers:
#     1. Upper barrier: entry + take_profit_width
#     2. Lower barrier: entry - stop_loss_width
#     3. Vertical barrier: entry + holding_period bars
#
#   The label is determined by which barrier is hit FIRST:
#     +1 (long)  → upper barrier hit first (take profit)
#     -1 (short) → lower barrier hit first (stop loss)
#      0 (hold)  → vertical barrier hit first (time expired)
#
#   Barrier widths can be ATR-based (adaptive to volatility) or fixed.
#
# This is the GOLD STANDARD for ML trade labeling because:
#   - It captures PATH-DEPENDENT outcomes (not just end-of-period return)
#   - It's naturally aligned with how trades actually work (TP/SL/timeout)
#   - ATR-based widths adapt to volatility regimes
#   - It avoids the look-ahead bias of fixed-horizon labels
#
# Also includes:
#   - fixed_horizon_labels() — simple forward-return labels for comparison
#   - meta_labels() — secondary model labels (bet size / confidence)
#   - compute_label_uniqueness() — weights for non-overlapping samples
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from typing import Optional

from utils.logger import get_logger

log = get_logger("triple_barrier")


def _atr(
    high: pd.Series, low: pd.Series, close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Compute Average True Range (Wilder's smoothing)."""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def triple_barrier_labels(
    df: pd.DataFrame,
    *,
    holding_period: int = 10,
    take_profit_width: float = 2.0,    # in ATR multiples
    stop_loss_width: float = 2.0,      # in ATR multiples
    atr_period: int = 14,
    use_atr: bool = True,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.Series:
    """
    Compute triple-barrier labels for ML training.

    Parameters
    ----------
    df : OHLC DataFrame.
    holding_period : number of bars before vertical barrier (time expiry).
    take_profit_width : upper barrier distance from entry (ATR multiples if
        use_atr=True, else absolute price units).
    stop_loss_width : lower barrier distance from entry.
    atr_period : ATR calculation period (only used if use_atr=True).
    use_atr : if True, barrier widths are ATR × multiplier (adaptive).

    Returns
    -------
    pd.Series of labels: +1 (TP hit), -1 (SL hit), 0 (timeout), NaN (insufficient data).
    """
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    n = len(df)

    if use_atr:
        atr = _atr(high, low, close, atr_period)
    else:
        atr = pd.Series(1.0, index=df.index)  # fixed widths

    labels = pd.Series(np.nan, index=df.index, dtype=float)

    for i in range(n - holding_period):
        entry_price = close.iloc[i]
        atr_val = atr.iloc[i] if not np.isnan(atr.iloc[i]) else 0.001

        upper = entry_price + take_profit_width * atr_val
        lower = entry_price - stop_loss_width * atr_val

        # Check barriers in order: first hit wins
        for j in range(1, holding_period + 1):
            idx = i + j
            if idx >= n:
                break

            # Upper barrier hit (take profit)
            if high.iloc[idx] >= upper:
                labels.iloc[i] = 1
                break
            # Lower barrier hit (stop loss)
            if low.iloc[idx] <= lower:
                labels.iloc[i] = -1
                break
            # Vertical barrier (time expired)
            if j == holding_period:
                labels.iloc[i] = 0

    return labels


def fixed_horizon_labels(
    df: pd.DataFrame,
    *,
    horizon: int = 10,
    threshold: float = 0.0,
    close_col: str = "close",
) -> pd.Series:
    """
    Simple fixed-horizon labels: compare forward return to threshold.

    +1 if return > threshold, -1 if return < -threshold, else 0.
    """
    close = df[close_col]
    forward_return = close.shift(-horizon) / close - 1
    labels = pd.Series(0, index=df.index, dtype=float)
    labels[forward_return > threshold] = 1
    labels[forward_return < -threshold] = -1
    labels[forward_return.isna()] = np.nan
    return labels


def meta_labels(
    primary_signal: pd.Series,
    returns: pd.Series,
    *,
    horizon: int = 10,
    threshold: float = 0.0,
) -> pd.Series:
    """
    Meta-labeling: given a primary signal (e.g., from a strategy), compute
    whether the signal was correct (profitable) within `horizon` bars.

    Returns: 1 (signal was correct), 0 (signal was wrong).
    Used to train a secondary model that filters the primary signal.
    """
    forward_return = returns.shift(-horizon)
    meta = pd.Series(0, index=returns.index, dtype=float)

    # For long signals (primary=1): correct if forward return > threshold
    long_mask = (primary_signal == 1) & (forward_return > threshold)
    meta[long_mask] = 1

    # For short signals (primary=-1): correct if forward return < -threshold
    short_mask = (primary_signal == -1) & (forward_return < -threshold)
    meta[short_mask] = 1

    meta[forward_return.isna()] = np.nan
    return meta


def compute_label_uniqueness(
    labels: pd.Series,
    holding_period: int = 10,
) -> pd.Series:
    """
    Compute uniqueness weight for each labeled bar.
    Bars whose holding periods overlap with other bars get lower weight.
    This prevents overlapping samples from dominating training.
    """
    n = len(labels)
    uniqueness = pd.Series(1.0, index=labels.index)

    for i in range(n):
        if np.isnan(labels.iloc[i]):
            uniqueness.iloc[i] = 0
            continue

        overlap = 0
        for j in range(max(0, i - holding_period), min(n, i + holding_period + 1)):
            if j == i or np.isnan(labels.iloc[j]):
                continue
            overlap += 1

        uniqueness.iloc[i] = 1.0 / (1.0 + overlap)

    return uniqueness


def mfe_mae(
    df: pd.DataFrame,
    *,
    holding_period: int = 10,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.DataFrame:
    """
    Compute Maximum Favorable Excursion (MFE) and Maximum Adverse Excursion (MAE)
    for each bar, over the next `holding_period` bars.

    MFE = max(high[i+1..i+H] / close[i]) - 1  (best upside)
    MAE = 1 - min(low[i+1..i+H] / close[i])    (worst downside)

    Returns DataFrame with 'mfe' and 'mae' columns.
    Used to calibrate triple-barrier widths empirically.
    """
    high = df[high_col].values
    low = df[low_col].values
    close = df[close_col].values
    n = len(df)

    mfe = np.full(n, np.nan)
    mae = np.full(n, np.nan)

    for i in range(n - holding_period):
        entry = close[i]
        future_highs = high[i+1:i+1+holding_period]
        future_lows = low[i+1:i+1+holding_period]

        mfe[i] = (future_highs.max() / entry) - 1
        mae[i] = 1 - (future_lows.min() / entry)

    return pd.DataFrame({"mfe": mfe, "mae": mae}, index=df.index)


# ── Live wiring: class-based interface (Priority #1) ──────────────────────
#
# Everything above this point is the original, unmodified standalone
# module (functions only, reachable so far only from its own __main__
# smoke test below). This class is the ONLY new code in this file — a
# thin wrapper with the same public shape as ml.label_generator.LabelGenerator
# (a `label_dataframe(df, pair)` method returning label + weight columns)
# so ml.dataset_builder.DatasetBuilder can select between
# labeling_method="fixed_horizon" | "triple_barrier" without any branching
# logic outside this module. No existing function's behavior changes.

@dataclass
class TripleBarrierResult:
    """Summary of one label_dataframe() call, mirroring LabelResult's role
    for LabelGenerator — used for logging/diagnostics, not required by
    DatasetBuilder (which reads columns off the returned DataFrame)."""
    total_labeled: int
    tp_count: int
    sl_count: int
    timeout_count: int
    holding_period: int
    take_profit_width: float
    stop_loss_width: float


class TripleBarrierLabeler:
    """
    Class-based interface over triple_barrier_labels() / compute_label_uniqueness(),
    matching LabelGenerator's public shape so DatasetBuilder can swap labeling
    methods interchangeably.

    label_dataframe() adds these columns (mirrors LabelGenerator's
    "label_*" naming so downstream meta-column filtering in DatasetBuilder
    needs only one additional entry, not a parallel code path):
        label            — {-1, 0, 1} triple-barrier outcome, int
                            (0 is remapped to the model's existing binary
                            convention downstream by DatasetBuilder, same
                            as fixed_horizon's label_binary)
        label_ternary    — raw {-1, 0, 1}, unchanged
        sample_weight    — from compute_label_uniqueness(), in (0, 1]
        label_mfe_pips / label_mae_pips — reused from mfe_mae(), same
                            column names LabelGenerator already produces,
                            so ValidationEngine/regime tests that read
                            these columns keep working unmodified.
    """

    def __init__(
        self,
        holding_period: int = 10,
        take_profit_width: float = 2.0,
        stop_loss_width: float = 2.0,
        atr_period: int = 14,
        use_atr: bool = True,
    ):
        self.holding_period = holding_period
        self.take_profit_width = take_profit_width
        self.stop_loss_width = stop_loss_width
        self.atr_period = atr_period
        self.use_atr = use_atr

    def label_dataframe(self, df: pd.DataFrame, pair: str = "EURUSD",
                         holding_period: Optional[int] = None) -> pd.DataFrame:
        """Add triple-barrier label + sample_weight columns to a copy of df.

        Rows where the barrier system found no valid outcome (insufficient
        forward data, i.e. the last `holding_period` rows) get NaN labels,
        matching LabelGenerator.label_dataframe()'s NaN-tail convention so
        DatasetBuilder's existing `labels_full.notna()` filter drops them
        the same way it already drops fixed-horizon's NaN tail.

        `holding_period`, if given, overrides self.holding_period for this
        call only — it does NOT mutate instance state. This matters because
        `get_triple_barrier_labeler()` returns a shared singleton: mutating
        `self.holding_period` from a caller would race under concurrent
        training (e.g. two pairs trained on separate threads with different
        horizons stepping on each other's setting mid-call).
        """
        h = holding_period if holding_period is not None else self.holding_period
        result = df.copy()

        ternary = triple_barrier_labels(
            df,
            holding_period=h,
            take_profit_width=self.take_profit_width,
            stop_loss_width=self.stop_loss_width,
            atr_period=self.atr_period,
            use_atr=self.use_atr,
        )
        result["label_ternary"] = ternary

        # Binary convention matching LabelGenerator: 1 = profitable long
        # setup (TP hit), 0 = everything else (SL hit or timeout). This
        # keeps the champion-approval pipeline's binary-classification
        # metrics (precision/recall/AUC on a 0/1 target) unchanged whether
        # labeling_method is "fixed_horizon" or "triple_barrier".
        result["label"] = np.where(ternary == 1, 1, np.where(ternary.isna(), np.nan, 0))

        weights = compute_label_uniqueness(ternary, holding_period=h)
        result["sample_weight"] = weights

        excursions = mfe_mae(
            df, holding_period=h,
            high_col="high", low_col="low", close_col="close",
        )
        result["label_mfe_pips"] = excursions["mfe"]
        result["label_mae_pips"] = excursions["mae"]

        return result

    def compute_sample_weights(self, labels_df: pd.DataFrame, holding_period: Optional[int] = None) -> pd.Series:
        """Thin wrapper around compute_label_uniqueness(), for callers that
        already have a labels_df (e.g. from a cached run) and just need
        weights recomputed — avoids re-running the O(n·h) barrier scan."""
        h = holding_period if holding_period is not None else self.holding_period
        series = labels_df["label_ternary"] if "label_ternary" in labels_df.columns else labels_df["label"]
        return compute_label_uniqueness(series, holding_period=h)

    def summary(self, labeled_df: pd.DataFrame) -> TripleBarrierResult:
        valid = labeled_df["label_ternary"].dropna()
        return TripleBarrierResult(
            total_labeled=int(len(valid)),
            tp_count=int((valid == 1).sum()),
            sl_count=int((valid == -1).sum()),
            timeout_count=int((valid == 0).sum()),
            holding_period=self.holding_period,
            take_profit_width=self.take_profit_width,
            stop_loss_width=self.stop_loss_width,
        )


# ── Singleton (mirrors LabelGenerator's get_label_generator()) ────────────

_LABELER: Optional[TripleBarrierLabeler] = None


def get_triple_barrier_labeler() -> TripleBarrierLabeler:
    global _LABELER
    if _LABELER is None:
        _LABELER = TripleBarrierLabeler()
    return _LABELER


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Synthetic data
    np.random.seed(42)
    n = 200
    close = 1.0850 + np.cumsum(np.random.randn(n) * 0.0005)
    high = close + np.random.uniform(0.0001, 0.0005, n)
    low = close - np.random.uniform(0.0001, 0.0005, n)
    df = pd.DataFrame({"high": high, "low": low, "close": close},
                      index=pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"))

    # Triple-barrier labels
    labels = triple_barrier_labels(df, holding_period=10, take_profit_width=2.0,
                                    stop_loss_width=2.0, use_atr=True)
    valid = labels.dropna()
    print(f"Triple-barrier labels: {len(valid)} valid out of {n}")
    print(f"  TP (+1): {int((valid == 1).sum())}")
    print(f"  SL (-1): {int((valid == -1).sum())}")
    print(f"  Timeout (0): {int((valid == 0).sum())}")

    assert valid.shape[0] > 0, "expected some valid labels"
    assert set(valid.unique()).issubset({-1, 0, 1})

    # Fixed-horizon labels
    fh_labels = fixed_horizon_labels(df, horizon=5, threshold=0.0002)
    fh_valid = fh_labels.dropna()
    print(f"\nFixed-horizon labels: {len(fh_valid)} valid")
    assert set(fh_valid.unique()).issubset({-1, 0, 1})

    # Meta-labels
    primary = pd.Series(np.random.choice([-1, 0, 1], n), index=df.index)
    returns = df["close"].pct_change()
    meta = meta_labels(primary, returns, horizon=5)
    print(f"Meta-labels: {int(meta.sum())} correct signals out of {len(meta.dropna())}")

    # Uniqueness
    uniq = compute_label_uniqueness(labels, holding_period=10)
    print(f"Uniqueness range: [{uniq.dropna().min():.3f}, {uniq.dropna().max():.3f}]")

    # MFE/MAE
    excursions = mfe_mae(df, holding_period=10)
    print(f"\nMFE: mean={excursions['mfe'].mean():.6f}, std={excursions['mfe'].std():.6f}")
    print(f"MAE: mean={excursions['mae'].mean():.6f}, std={excursions['mae'].std():.6f}")
    # MFE/MAE can be slightly negative if entry is at local extreme
    assert excursions['mfe'].dropna().shape[0] > 0, "MFE should have valid values"
    assert excursions['mae'].dropna().shape[0] > 0, "MAE should have valid values"

    print("\nTriple-barrier labels + MFE/MAE smoke test passed.")
