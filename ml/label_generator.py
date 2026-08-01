"""
ml/label_generator.py — Target variable generator (Day 69)
============================================================

Generates ML training labels from historical price data. The label
represents "what happened next" — whether a BUY, SELL, or WAIT signal
would have been the right call over a given horizon.

Label: multi-class signal
  * **BUY**  ( 1) — forward move exceeds the dynamic threshold, upward
  * **SELL** (-1) — forward move exceeds the dynamic threshold, downward
  * **WAIT** ( 0) — forward move stays inside the threshold band (no edge)

Plus continuous companions for regression / auxiliary heads:
  * **forward_return**       — (future_price - current) / current
  * **forward_pips**         — pips moved over the horizon
  * **mae_pips**              — max adverse excursion before horizon end (risk proxy)
  * **mfe_pips**              — max favorable excursion before horizon end
  * **r_multiple**            — mfe / abs(mae) — reward/risk ratio

Threshold — ATR-based, dynamic (NOT a fixed pip count):
  threshold_pips = max(ATR_pips(atr_period) * atr_multiplier, min_threshold_pips)
  ATR is Wilder-smoothed and computed using ONLY candles up to and
  including the current row — it never looks forward, so it's safe to
  use as a volatility-scaled classification boundary. This makes the
  BUY/SELL/WAIT split regime-aware: quiet markets get a tighter band,
  volatile markets get a wider one, instead of one fixed pip count that
  is too tight in high-ATR regimes (noise labeled as signal) and too
  wide in low-ATR regimes (real moves labeled WAIT).

Horizon — configurable, one of:
  * 4  candles ahead (1 hour on M15)
  * 8  candles ahead (2 hours on M15)
  * 16 candles ahead (4 hours on M15)
  `label_dataframe` accepts any subset of these and produces one column
  set per horizon (suffix `_h{n}`), so a single labeling pass can feed
  multiple prediction horizons.

CRITICAL — look-ahead boundary:
  - forward_return / forward_pips / mae_pips / mfe_pips / signal class use
    ONLY future candles relative to the feature row (row_idx+1 .. row_idx+horizon).
    This is the only place future data is allowed — and only for creating
    training labels, never for inference features.
  - ATR (used to set the threshold) uses ONLY candles up to and including
    the current row. It does not touch row_idx+1 onward. Conflating these
    two windows (e.g. computing ATR from a centered or future-inclusive
    window) would leak volatility information from the horizon into the
    threshold and bias the label distribution — see CO-FOUNDER FIX note
    in `label_dataframe` for a previous bug in the same spirit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from enum import IntEnum
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("label_generator")

# Only these horizons are supported — keeps every consumer (training
# pipeline, backtester, live inference) aligned on the same candle counts.
VALID_HORIZONS = (4, 8, 16)


class SignalClass(IntEnum):
    """Multi-class label. Values double as the numeric label stored in
    dataframes/model targets — don't renumber without a migration plan
    for anything that has already trained on these integers."""
    SELL = -1
    WAIT = 0
    BUY = 1

    @property
    def label(self) -> str:
        return {SignalClass.SELL: "SELL", SignalClass.WAIT: "WAIT", SignalClass.BUY: "BUY"}[self]


@dataclass
class LabelResult:
    """Labels for a single row."""
    signal_class: int = 0             # -1 SELL, 0 WAIT, 1 BUY
    signal_label: str = "WAIT"
    forward_return: float = 0.0       # (future - current) / current
    forward_pips: float = 0.0         # pips moved
    mae_pips: float = 0.0             # max adverse excursion (negative = drawdown)
    mfe_pips: float = 0.0             # max favorable excursion (positive = profit)
    r_multiple: float = 0.0           # mfe / abs(mae) — reward/risk ratio
    horizon_candles: int = 4
    atr_pips: float = 0.0             # ATR at decision time, in pips
    threshold_pips: float = 0.0       # dynamic: max(atr_pips * multiplier, floor)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder-smoothed ATR, STRICTLY backward-looking.

    ATR[i] is derived only from true-range values at candles
    [i-period+1 .. i] — i.e. only information available AT or BEFORE row
    i. Safe to use as a volatility-scaled label threshold because it
    never touches row i+1 onward (which is where the label's own
    forward-looking window lives). The first `period - 1` rows are NaN
    (not enough history yet); callers must treat those as "no label".
    """
    tr = _true_range(df)
    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return atr


class LabelGenerator:
    """Generates forward-looking, ATR-scaled, multi-class labels for ML training."""

    def __init__(
        self,
        default_horizon: int = 4,
        atr_period: int = 14,
        default_atr_multiplier: float = 1.5,
        default_min_threshold_pips: float = 3.0,
    ):
        if default_horizon not in VALID_HORIZONS:
            raise ValueError(f"default_horizon must be one of {VALID_HORIZONS}, got {default_horizon}")
        self.default_horizon = default_horizon
        self.atr_period = atr_period
        self.default_atr_multiplier = default_atr_multiplier
        self.default_min_threshold_pips = default_min_threshold_pips

    # ── Single-row API ──────────────────────────────────────────────

    def label_for_row(
        self,
        df: pd.DataFrame,
        row_idx: int,
        pair: str = "EURUSD",
        horizon: Optional[int] = None,
        atr_multiplier: Optional[float] = None,
        min_threshold_pips: Optional[float] = None,
        atr_series: Optional[pd.Series] = None,
    ) -> Optional[LabelResult]:
        """Generate the label for a single row at `row_idx`.

        Returns None if there aren't enough future candles for the
        horizon, or not enough history yet for a stable ATR.

        `atr_series` (optional): pass a precomputed `compute_atr(df, ...)`
        series when labeling many rows in a loop — recomputing ATR from
        scratch on every call is O(n) per call and gets expensive fast.
        `label_dataframe` already does this correctly and vectorized;
        prefer it for bulk labeling.
        """
        horizon = horizon or self.default_horizon
        if horizon not in VALID_HORIZONS:
            raise ValueError(f"horizon must be one of {VALID_HORIZONS}, got {horizon}")
        atr_multiplier = atr_multiplier if atr_multiplier is not None else self.default_atr_multiplier
        min_threshold_pips = (
            min_threshold_pips if min_threshold_pips is not None else self.default_min_threshold_pips
        )

        if row_idx + horizon >= len(df):
            return None
        if row_idx < self.atr_period - 1:
            return None  # not enough history for a stable ATR yet

        pair = pair.upper()
        pip_size = 0.01 if pair.endswith("JPY") else 0.0001

        if atr_series is None:
            # Only ever slice up to row_idx+1 (inclusive of the current
            # row) — this is what keeps ATR backward-looking when called
            # standalone.
            atr_series = compute_atr(df.iloc[: row_idx + 1], period=self.atr_period)
        atr_value = float(atr_series.iloc[row_idx])
        if pd.isna(atr_value):
            return None
        atr_pips = atr_value / pip_size
        threshold_pips = max(atr_pips * atr_multiplier, min_threshold_pips)

        current_close = float(df.iloc[row_idx]["close"])
        future_close = float(df.iloc[row_idx + horizon]["close"])
        future_window = df.iloc[row_idx + 1: row_idx + horizon + 1]

        forward_return = (future_close - current_close) / current_close if current_close > 0 else 0.0
        forward_pips = (future_close - current_close) / pip_size

        if len(future_window) > 0:
            mae_pips = (future_window["low"].min() - current_close) / pip_size  # negative
            mfe_pips = (future_window["high"].max() - current_close) / pip_size  # positive
        else:
            mae_pips = 0.0
            mfe_pips = 0.0

        r_multiple = (mfe_pips / abs(mae_pips)) if mae_pips < 0 else (mfe_pips if mfe_pips > 0 else 0.0)

        if forward_pips > threshold_pips:
            signal = SignalClass.BUY
        elif forward_pips < -threshold_pips:
            signal = SignalClass.SELL
        else:
            signal = SignalClass.WAIT

        return LabelResult(
            signal_class=int(signal),
            signal_label=signal.label,
            forward_return=forward_return,
            forward_pips=forward_pips,
            mae_pips=mae_pips,
            mfe_pips=mfe_pips,
            r_multiple=r_multiple,
            horizon_candles=horizon,
            atr_pips=atr_pips,
            threshold_pips=threshold_pips,
        )

    # ── Bulk / vectorized API ───────────────────────────────────────

    def label_dataframe(
        self,
        df: pd.DataFrame,
        pair: str = "EURUSD",
        horizons: Optional[Sequence[int]] = None,
        atr_multiplier: Optional[float] = None,
        min_threshold_pips: Optional[float] = None,
    ) -> pd.DataFrame:
        """Add label columns to a dataframe for one or more horizons.
        Returns a copy with new columns; does not mutate `df`.

        Shared columns (independent of horizon):
          atr_pips, label_threshold_pips

        Per-horizon columns (suffix `_h{horizon}`, e.g. `_h4`):
          label_class, label_name, label_forward_return, label_forward_pips,
          label_mae_pips, label_mfe_pips, label_r_multiple

        `label_class` is nullable Int64 (-1/0/1); rows with no valid ATR
        yet or not enough future candles are left as <NA>, NOT defaulted
        to WAIT — defaulting them would silently bias the class balance.
        """
        horizons = tuple(horizons) if horizons is not None else (self.default_horizon,)
        bad = [h for h in horizons if h not in VALID_HORIZONS]
        if bad:
            raise ValueError(f"horizons must be a subset of {VALID_HORIZONS}, got {bad}")

        atr_multiplier = atr_multiplier if atr_multiplier is not None else self.default_atr_multiplier
        min_threshold_pips = (
            min_threshold_pips if min_threshold_pips is not None else self.default_min_threshold_pips
        )

        pair = pair.upper()
        pip_size = 0.01 if pair.endswith("JPY") else 0.0001

        result = df.copy()

        # ATR — backward-looking only (see compute_atr docstring). Computed
        # once and shared across all horizons: the volatility regime at row
        # i doesn't depend on which horizon we're labeling.
        atr = compute_atr(df, period=self.atr_period)
        atr_pips = atr / pip_size
        threshold_pips = (atr_pips * atr_multiplier).clip(lower=min_threshold_pips)
        result["atr_pips"] = atr_pips
        result["label_threshold_pips"] = threshold_pips

        for horizon in horizons:
            suffix = f"_h{horizon}"
            future_close = df["close"].shift(-horizon)
            fwd_return = (future_close - df["close"]) / df["close"]
            fwd_pips = (future_close - df["close"]) / pip_size

            # Forward MAE / MFE over (row_idx+1 .. row_idx+horizon).
            #
            # CO-FOUNDER FIX (audit finding, still applies here): the naive
            #   df["high"].shift(-horizon).rolling(horizon).max().shift(horizon)
            # nets out to max(high[i-horizon+1 .. i]) — i.e. it silently
            # reads PAST candles, not future ones. Fix: shift by 1 first
            # (window starts at the NEXT candle, matching
            # label_for_row's df.iloc[row_idx+1 : row_idx+horizon+1]),
            # then use a reverse-rolling max/min so the window looks
            # forward from i+1 through i+horizon.
            future_high = df["high"].shift(-1)
            future_high_max = future_high[::-1].rolling(window=horizon, min_periods=horizon).max()[::-1]
            future_low = df["low"].shift(-1)
            future_low_min = future_low[::-1].rolling(window=horizon, min_periods=horizon).min()[::-1]
            mae_pips = (future_low_min - df["close"]) / pip_size
            mfe_pips = (future_high_max - df["close"]) / pip_size
            r_multiple = np.where(
                mae_pips < 0,
                mfe_pips / mae_pips.abs(),
                np.where(mfe_pips > 0, mfe_pips, 0.0),
            )

            signal_codes = np.where(
                fwd_pips > threshold_pips, int(SignalClass.BUY),
                np.where(fwd_pips < -threshold_pips, int(SignalClass.SELL), int(SignalClass.WAIT)),
            )
            signal_class = pd.array(signal_codes, dtype="Int64")

            # Rows with no valid ATR (warm-up period) or no valid forward
            # window (tail of the dataframe) must not get a class — leaving
            # them WAIT by default would corrupt the class balance.
            invalid = threshold_pips.isna().to_numpy() | fwd_pips.isna().to_numpy()
            signal_class[invalid] = pd.NA

            label_name = pd.Series(pd.NA, index=df.index, dtype="object")
            valid_mask = ~invalid
            label_name.loc[valid_mask] = [
                SignalClass(v).label for v in signal_class[valid_mask].astype(int)
            ]

            result[f"label_forward_return{suffix}"] = fwd_return
            result[f"label_forward_pips{suffix}"] = fwd_pips
            result[f"label_mae_pips{suffix}"] = mae_pips
            result[f"label_mfe_pips{suffix}"] = mfe_pips
            result[f"label_r_multiple{suffix}"] = r_multiple
            result[f"label_class{suffix}"] = signal_class
            result[f"label_name{suffix}"] = label_name

        return result

    # ── Diagnostics ──────────────────────────────────────────────────

    def class_balance(self, df: pd.DataFrame, horizon: Optional[int] = None) -> Dict[str, Any]:
        """Class balance for one horizon's labels. Always inspect this
        before training — a WAIT-dominated split (very common with FX,
        most bars have no real edge) will make plain accuracy meaningless
        and needs class weights / resampling / a P&L-based metric instead
        (see AI/ML standards: class imbalance under classification-style
        trade signals)."""
        horizon = horizon or self.default_horizon
        col = f"label_class_h{horizon}"
        if col not in df.columns:
            return {"error": f"'{col}' not found — run label_dataframe(..., horizons=[{horizon}, ...]) first"}

        valid = df[col].dropna()
        total = len(valid)
        if total == 0:
            return {"error": "no labeled rows"}

        counts = valid.astype(int).value_counts()
        buy = int(counts.get(int(SignalClass.BUY), 0))
        sell = int(counts.get(int(SignalClass.SELL), 0))
        wait = int(counts.get(int(SignalClass.WAIT), 0))

        def pct(n: int) -> float:
            return round(n / total * 100, 2)

        majority_pct = max(pct(buy), pct(sell), pct(wait))
        severe = majority_pct > 90.0

        return {
            "horizon": horizon,
            "total_labeled": total,
            "buy": buy, "sell": sell, "wait": wait,
            "buy_pct": pct(buy), "sell_pct": pct(sell), "wait_pct": pct(wait),
            "majority_class_pct": majority_pct,
            "is_severely_imbalanced": severe,
            "imbalance_warning": (
                "Majority class exceeds 90% of labeled rows. If it's WAIT: "
                "the ATR multiplier may be too wide, or this segment is "
                "genuinely low-signal — either is fine, but plain accuracy "
                "will be misleading; use class weights, resampling, or a "
                "precision/recall or P&L-based metric instead. If it's "
                "BUY/SELL: check for a trending regime in this sample "
                "rather than a real, generalizable edge."
                if severe else None
            ),
        }

    def class_balance_all_horizons(
        self, df: pd.DataFrame, horizons: Optional[Sequence[int]] = None
    ) -> Dict[int, Dict[str, Any]]:
        """Class balance across every labeled horizon present in `df`."""
        horizons = tuple(horizons) if horizons is not None else VALID_HORIZONS
        return {
            h: self.class_balance(df, h)
            for h in horizons
            if f"label_class_h{h}" in df.columns
        }

    def label_summary(self, df: pd.DataFrame, horizon: Optional[int] = None) -> Dict[str, Any]:
        """Return summary statistics (class balance + continuous-label
        averages) for one horizon's labels in a labeled dataframe."""
        horizon = horizon or self.default_horizon
        suffix = f"_h{horizon}"
        col = f"label_class{suffix}"
        if col not in df.columns:
            return {"error": f"dataframe not labeled for horizon={horizon}"}

        valid = df.dropna(subset=[col])
        total = len(valid)
        if total == 0:
            return {"error": "no labeled rows"}

        balance = self.class_balance(df, horizon)
        return {
            **balance,
            "avg_forward_pips": round(valid[f"label_forward_pips{suffix}"].mean(), 2),
            "avg_mae_pips": round(valid[f"label_mae_pips{suffix}"].mean(), 2),
            "avg_mfe_pips": round(valid[f"label_mfe_pips{suffix}"].mean(), 2),
            "avg_r_multiple": round(valid[f"label_r_multiple{suffix}"].mean(), 2),
            "avg_atr_pips": round(valid["atr_pips"].mean(), 2),
            "avg_threshold_pips": round(valid["label_threshold_pips"].mean(), 2),
        }


# ── Singleton ───────────────────────────────────────────────────────

_GENERATOR: Optional[LabelGenerator] = None


def get_label_generator() -> LabelGenerator:
    global _GENERATOR
    if _GENERATOR is None:
        _GENERATOR = LabelGenerator()
    return _GENERATOR