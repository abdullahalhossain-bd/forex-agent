# analysis/adx_trend_filter.py — ADX-based trend filter
# =============================================================================
# Ported from: https://github.com/bruh7463/forex_bot
# Original: config/settings.py → MIN_ADX = 20, and bot/runner.py ADX check
# Original author: bruh7463 — educational project
#
# ADX (Average Directional Index) measures trend STRENGTH (not direction).
#   ADX < 20  → market is ranging / choppy (no clear trend)
#   ADX 20-25 → trend is forming
#   ADX 25-50 → strong trend
#   ADX 50-75 → very strong trend (rare, late stage)
#   ADX > 75 → extremely strong (parabolic, likely to reverse)
#
# This filter blocks trades when ADX < threshold (default 20), so the bot
# only trades in trending markets. This dramatically reduces false signals
# from mean-reversion strategies that get chopped up in ranges.
#
# The filter also provides DIRECTIONAL context:
#   - +DI > -DI → bullish trend (even if ADX is low)
#   - -DI > +DI → bearish trend
#   - ADX rising → trend is gaining strength
#   - ADX falling → trend is weakening
#
# Output columns added to the DataFrame:
#   adx, adx_pos (+DI), adx_neg (-DI), adx_trend_strength ("weak"/"forming"/
#   "strong"/"very_strong"), adx_direction ("bullish"/"bearish"/"neutral"),
#   adx_filter_pass (True if ADX >= threshold)
# =============================================================================

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("adx_trend_filter")

try:
    import ta
    _HAS_TA = True
except ImportError:
    _HAS_TA = False


def compute(
    df: pd.DataFrame,
    *,
    period: int = 14,
    min_adx: float = 20.0,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.DataFrame:
    """
    Compute ADX trend filter.

    Parameters
    ----------
    df : DataFrame with high, low, close columns.
    period : ADX calculation period (default 14 — standard).
    min_adx : minimum ADX value to pass the filter (default 20).
    high_col, low_col, close_col : column names.

    Returns
    -------
    DataFrame with adx, adx_pos, adx_neg, adx_trend_strength,
    adx_direction, adx_filter_pass columns added.
    """
    if not _HAS_TA:
        raise ImportError("`ta` library required. Install with: pip install ta")

    out = df.copy()
    h, l, c = out[high_col], out[low_col], out[close_col]

    adx_ind = ta.trend.ADXIndicator(high=h, low=l, close=c, window=period)
    out["adx"] = adx_ind.adx()
    out["adx_pos"] = adx_ind.adx_pos()  # +DI
    out["adx_neg"] = adx_ind.adx_neg()  # -DI

    # Trend strength category
    def _strength(adx_val):
        if pd.isna(adx_val):
            return "unknown"
        if adx_val < 20:
            return "weak"
        elif adx_val < 25:
            return "forming"
        elif adx_val < 50:
            return "strong"
        elif adx_val < 75:
            return "very_strong"
        else:
            return "extreme"

    out["adx_trend_strength"] = out["adx"].apply(_strength)

    # Directional bias
    out["adx_direction"] = np.where(
        out["adx_pos"] > out["adx_neg"], "bullish",
        np.where(out["adx_pos"] < out["adx_neg"], "bearish", "neutral")
    )

    # Filter pass: ADX >= min_adx
    out["adx_filter_pass"] = out["adx"] >= min_adx

    return out


def should_trade(
    df: pd.DataFrame,
    *,
    min_adx: float = 20.0,
    adx_col: str = "adx",
) -> bool:
    """
    Quick check: should we trade based on the latest bar's ADX?

    Returns True if ADX on the latest bar >= min_adx.
    """
    if adx_col not in df.columns:
        raise ValueError(f"Column '{adx_col}' not found. Run compute() first.")
    if df.empty:
        return False
    latest_adx = df[adx_col].iloc[-1]
    if pd.isna(latest_adx):
        return False
    return latest_adx >= min_adx


def get_trend_context(df: pd.DataFrame, *, adx_col: str = "adx",
                      pos_col: str = "adx_pos", neg_col: str = "adx_neg") -> dict:
    """
    Return a dict with the latest bar's ADX context.
    """
    if df.empty or adx_col not in df.columns:
        return {}
    latest = df.iloc[-1]
    adx = float(latest[adx_col]) if not pd.isna(latest[adx_col]) else 0.0
    pos_di = float(latest[pos_col]) if not pd.isna(latest[pos_col]) else 0.0
    neg_di = float(latest[neg_col]) if not pd.isna(latest[neg_col]) else 0.0

    # ADX slope (is trend gaining or losing strength?)
    if len(df) >= 2 and not pd.isna(df[adx_col].iloc[-2]):
        adx_prev = float(df[adx_col].iloc[-2])
        adx_slope = adx - adx_prev
    else:
        adx_slope = 0.0

    return {
        "adx": round(adx, 2),
        "plus_di": round(pos_di, 2),
        "minus_di": round(neg_di, 2),
        "direction": "bullish" if pos_di > neg_di else "bearish" if pos_di < neg_di else "neutral",
        "adx_slope": round(adx_slope, 2),
        "trend_gaining": adx_slope > 0,
        "trend_strength": (
            "weak" if adx < 20 else
            "forming" if adx < 25 else
            "strong" if adx < 50 else
            "very_strong" if adx < 75 else
            "extreme"
        ),
    }


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not _HAS_TA:
        print("`ta` not installed. Install with: pip install ta")
    else:
        np.random.seed(42)
        n = 300
        # Trending data (should produce high ADX)
        t = np.arange(n)
        close = 1.1000 + 0.0002 * t + np.random.randn(n) * 0.0005
        high = close + np.random.uniform(0.0001, 0.0005, n)
        low = close - np.random.uniform(0.0001, 0.0005, n)
        df = pd.DataFrame({"high": high, "low": low, "close": close},
                          index=pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"))

        out = compute(df, min_adx=20)
        print(f"Rows: {len(out)}")
        print(f"ADX NaN (warmup): {out['adx'].isna().sum()}")
        print(f"ADX range: [{out['adx'].min():.1f}, {out['adx'].max():.1f}]")
        print(f"Filter pass rate: {out['adx_filter_pass'].mean():.1%}")
        print(f"Trend strength distribution:")
        print(out["adx_trend_strength"].value_counts())

        # Latest bar context
        ctx = get_trend_context(out)
        print(f"\nLatest bar context: {ctx}")

        # should_trade check
        print(f"Should trade: {should_trade(out, min_adx=20)}")
        print(f"Should trade (min_adx=50): {should_trade(out, min_adx=50)}")

        print("\nADX trend filter smoke test passed.")
