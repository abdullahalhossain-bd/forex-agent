# analysis/global_filters.py — Cross-cutting rule-spec filters
# =============================================================================
# Two global filters from the rule spec that had no home anywhere in the
# existing codebase:
#
#   1. doji_density_filter — if a chart region already has many doji/small-
#      real-body candles, a NEW doji carries reduced significance (noise).
#      Track rolling doji_count in a lookback window and scale doji_weight
#      inversely.
#   2. market_type_gap_flexibility — for forex/index/intraday charts, star-
#      pattern gap requirements (open ~= prior close) may be relaxed, since
#      true opening gaps are rare intraday. For daily equity/futures charts,
#      keep the strict gap requirement.
#
# These are pure filters/weights — they never generate a signal on their own.
# Callers (candlestick pattern modules, the confluence engine) should import
# and apply them when scoring a doji-based or star-based pattern.
# =============================================================================

from __future__ import annotations

import numpy as np
import pandas as pd

RELAXED_GAP_INSTRUMENT_TYPES = {"forex", "index", "intraday"}


def doji_density(
    is_doji: pd.Series,
    *,
    lookback: int = 20,
) -> pd.Series:
    """
    Rolling count of doji/small-real-body candles in the trailing `lookback`
    window, per bar. Higher density = choppier region = noisier doji signals.
    """
    return is_doji.astype(int).rolling(lookback, min_periods=1).sum()


def doji_weight(
    is_doji: pd.Series,
    *,
    lookback: int = 20,
    max_density_for_full_weight: int = 2,
) -> pd.Series:
    """
    Scale a doji's significance inversely with how many other doji/small-body
    candles have recently appeared. 0 or 1 recent doji -> full weight (1.0).
    As density rises toward `lookback`, weight decays toward a floor so a doji
    in a chronically choppy/box-range market never contributes zero but also
    never counts as much as an isolated doji in a clean trend.
    """
    density = doji_density(is_doji, lookback=lookback)
    floor = 0.15
    weight = np.where(
        density <= max_density_for_full_weight,
        1.0,
        np.maximum(floor, 1.0 - (density - max_density_for_full_weight) / lookback),
    )
    return pd.Series(weight, index=is_doji.index)


def gap_required_for_star_patterns(instrument_type: str) -> bool:
    """
    True (strict) for daily equity/futures charts — a morning/evening star's
    middle candle must show a real gap from the first candle.
    False (relaxed) for forex/index/intraday charts, where open ~= prior
    close is normal and a strict gap requirement would suppress valid stars.
    """
    return instrument_type.lower() not in RELAXED_GAP_INSTRUMENT_TYPES


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    is_doji = pd.Series([False, True, False, True, True, False, True, True, True, False])

    density = doji_density(is_doji, lookback=5)
    weight = doji_weight(is_doji, lookback=5)
    print("density:\n", density.tolist())
    print("weight:\n", [round(w, 2) for w in weight.tolist()])

    assert weight.iloc[1] == 1.0, "first doji in a clean region should get full weight"
    assert weight.iloc[8] < 1.0, "doji in a dense choppy region should be down-weighted"

    assert gap_required_for_star_patterns("forex") is False
    assert gap_required_for_star_patterns("equity") is True

    print("\nGlobal filters module smoke test passed.")