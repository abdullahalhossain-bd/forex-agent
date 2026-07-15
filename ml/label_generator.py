"""
ml/label_generator.py — Target variable generator (Day 68)
============================================================

Generates ML training labels from historical price data. The label
represents "what happened next" — whether a BUY or SELL signal would
have been profitable over a given horizon.

Label types:
  * **binary_directional**  — 1 if price moved up >threshold, 0 otherwise
  * **ternary**              — 1 (up), -1 (down), 0 (neutral)
  * **forward_return**       — continuous: (future_price - current) / current
  * **forward_pips**         — continuous: pips moved over horizon
  * **max_adverse_excursion** — worst drawdown before horizon end (risk proxy)
  * **max_favorable_excursion** — best profit before horizon end

CRITICAL: labels use ONLY future candles relative to the feature row.
This is the only place where future data is allowed — and only for
creating training labels, never for inference features.

Horizon options:
  * "next_1"   — 1 candle ahead
  * "next_4"   — 4 candles ahead (1 hour on M15)
  * "next_16"  — 16 candles ahead (4 hours on M15)
  * "next_48"  — 48 candles ahead (12 hours on M15)

Threshold (pips): default 10 for majors, 8 for JPY pairs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("label_generator")


@dataclass
class LabelResult:
    """Labels for a single row."""
    binary_directional: int = 0       # 1 if up >threshold, 0 otherwise
    ternary: int = 0                  # 1 (up), -1 (down), 0 (neutral)
    forward_return: float = 0.0       # (future - current) / current
    forward_pips: float = 0.0         # pips moved
    mae_pips: float = 0.0             # max adverse excursion (negative = drawdown)
    mfe_pips: float = 0.0             # max favorable excursion (positive = profit)
    r_multiple: float = 0.0           # mfe / abs(mae) — reward/risk ratio
    horizon_candles: int = 4
    threshold_pips: float = 10.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LabelGenerator:
    """Generates forward-looking labels for ML training."""

    def __init__(self, default_horizon: int = 4, default_threshold_pips: float = 10.0):
        self.default_horizon = default_horizon
        self.default_threshold_pips = default_threshold_pips

    def label_for_row(
        self,
        df: pd.DataFrame,
        row_idx: int,
        pair: str = "EURUSD",
        horizon: Optional[int] = None,
        threshold_pips: Optional[float] = None,
    ) -> Optional[LabelResult]:
        """Generate labels for a single row at `row_idx`.

        Returns None if not enough future candles.
        """
        horizon = horizon or self.default_horizon
        threshold_pips = threshold_pips or self.default_threshold_pips

        if row_idx + horizon >= len(df):
            return None

        # Adjust threshold for JPY pairs
        pair = pair.upper()
        if pair.endswith("JPY") and threshold_pips == 10.0:
            threshold_pips = 8.0

        pip_size = 0.01 if pair.endswith("JPY") else 0.0001

        current_close = float(df.iloc[row_idx]["close"])
        future_close = float(df.iloc[row_idx + horizon]["close"])
        future_window = df.iloc[row_idx + 1: row_idx + horizon + 1]

        # Forward return
        forward_return = (future_close - current_close) / current_close if current_close > 0 else 0.0
        forward_pips = (future_close - current_close) / pip_size

        # Max adverse / favorable excursion over the horizon
        if len(future_window) > 0:
            mae_pips = (future_window["low"].min() - current_close) / pip_size  # negative
            mfe_pips = (future_window["high"].max() - current_close) / pip_size  # positive
        else:
            mae_pips = 0.0
            mfe_pips = 0.0

        # R-multiple (reward / risk)
        r_multiple = (mfe_pips / abs(mae_pips)) if mae_pips < 0 else (mfe_pips if mfe_pips > 0 else 0.0)

        # Binary directional label
        binary = 1 if forward_pips > threshold_pips else 0

        # Ternary label
        if forward_pips > threshold_pips:
            ternary = 1
        elif forward_pips < -threshold_pips:
            ternary = -1
        else:
            ternary = 0

        return LabelResult(
            binary_directional=binary,
            ternary=ternary,
            forward_return=forward_return,
            forward_pips=forward_pips,
            mae_pips=mae_pips,
            mfe_pips=mfe_pips,
            r_multiple=r_multiple,
            horizon_candles=horizon,
            threshold_pips=threshold_pips,
        )

    def label_dataframe(
        self,
        df: pd.DataFrame,
        pair: str = "EURUSD",
        horizon: Optional[int] = None,
        threshold_pips: Optional[float] = None,
    ) -> pd.DataFrame:
        """Add label columns to a dataframe. Returns a copy with new columns.

        New columns: label_binary, label_ternary, label_forward_return,
                     label_forward_pips, label_mae_pips, label_mfe_pips, label_r_multiple
        """
        horizon = horizon or self.default_horizon
        threshold_pips = threshold_pips or self.default_threshold_pips
        if pair.endswith("JPY") and threshold_pips == 10.0:
            threshold_pips = 8.0

        pip_size = 0.01 if pair.endswith("JPY") else 0.0001
        result = df.copy()

        # Vectorized computation for speed
        future_close = df["close"].shift(-horizon)
        result["label_forward_return"] = (future_close - df["close"]) / df["close"]
        result["label_forward_pips"] = (future_close - df["close"]) / pip_size

        # Rolling min/max for MAE/MFE (forward-looking)
        #
        # CO-FOUNDER FIX (audit finding): the original
        #   df["high"].shift(-horizon).rolling(horizon).max().shift(horizon)
        # nets out to max(high[i-horizon+1 .. i]) — i.e. it silently reads
        # PAST candles, not future ones (verified numerically). Every
        # mae_pips / mfe_pips / r_multiple value computed by this bulk
        # method was therefore backward-looking noise instead of the
        # forward risk/reward the label is documented to represent, while
        # label_for_row() (the per-row method above) does this correctly.
        # Fix: shift by 1 first (window starts at the NEXT candle, matching
        # label_for_row's df.iloc[row_idx+1 : row_idx+horizon+1]), then use
        # a reverse-rolling-max/min so the window looks forward.
        future_high = df["high"].shift(-1)
        future_high_max = future_high[::-1].rolling(window=horizon, min_periods=horizon).max()[::-1]
        future_low = df["low"].shift(-1)
        future_low_min = future_low[::-1].rolling(window=horizon, min_periods=horizon).min()[::-1]
        result["label_mae_pips"] = (future_low_min - df["close"]) / pip_size
        result["label_mfe_pips"] = (future_high_max - df["close"]) / pip_size
        result["label_r_multiple"] = np.where(
            result["label_mae_pips"] < 0,
            result["label_mfe_pips"] / abs(result["label_mae_pips"]),
            result["label_mfe_pips"],
        )

        # Binary + ternary
        result["label_binary"] = (result["label_forward_pips"] > threshold_pips).astype(int)
        result["label_ternary"] = np.where(
            result["label_forward_pips"] > threshold_pips, 1,
            np.where(result["label_forward_pips"] < -threshold_pips, -1, 0),
        )

        return result

    def label_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Return summary statistics of labels in a labeled dataframe."""
        if "label_binary" not in df.columns:
            return {"error": "dataframe not labeled"}
        total = len(df.dropna(subset=["label_binary"]))
        if total == 0:
            return {"error": "no labeled rows"}
        wins = int(df["label_binary"].sum())
        losses = total - wins
        return {
            "total_labeled": total,
            "wins": wins,
            "losses": losses,
            "win_rate_pct": round((wins / total) * 100, 1),
            "avg_forward_pips": round(df["label_forward_pips"].mean(), 2),
            "avg_mae_pips": round(df["label_mae_pips"].mean(), 2),
            "avg_mfe_pips": round(df["label_mfe_pips"].mean(), 2),
            "avg_r_multiple": round(df["label_r_multiple"].mean(), 2),
        }


# ── Singleton ───────────────────────────────────────────────────────

_GENERATOR: Optional[LabelGenerator] = None


def get_label_generator() -> LabelGenerator:
    global _GENERATOR
    if _GENERATOR is None:
        _GENERATOR = LabelGenerator()
    return _GENERATOR