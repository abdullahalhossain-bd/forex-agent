# analysis/optimal_trading_time.py — Find optimal hour to trade a given instrument
# =============================================================================
# Ported from FXBot (https://github.com/trentstauff/FXBot/blob/master/helpers/helpers.py)
# Original: helpers.find_optimal_trading_time()
# Original author: Trent Stauffner — MIT license (inferred)
#
# For each hour of the day (UTC), computes the fraction of bars where the
# price change exceeds the spread. High coverage = trading costs are
# recoverable in that hour; low coverage = trading costs eat your edge.
#
# Use this to pick the BEST trading session for a strategy:
#   - For SMA crossover, you want stable trends → trade hours with
#     high coverage (often London/NY overlap, 13:00-17:00 UTC).
#   - For mean-reversion, you want choppy ranges → trade hours with
#     lower coverage but tight spreads.
#
# Differences from FXBot:
#   - FXBot hard-codes tpqoa (OANDA). We accept ANY DataFrame with
#     bid/ask columns (or a single 'close' column → synthetic spread).
#   - Returns a DataFrame in addition to plotting (so callers can
#     programmatically pick the best hour).
#   - Plotting is optional and respects the global matplotlib backend.
# =============================================================================

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("optimal_trading_time")

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def find_optimal_trading_time(
    df: pd.DataFrame,
    *,
    bid_col: str = "bid",
    ask_col: str = "ask",
    mid_col: Optional[str] = None,
    granularity: str = "M5",
    plot: bool = True,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Find the UTC hours where trading costs are most recoverable.

    Parameters
    ----------
    df : DataFrame with bid/ask columns, OR a single 'mid_col' column.
        Must have a tz-aware DatetimeIndex (we extract hour in UTC).
    bid_col, ask_col : column names for bid/ask prices.
    mid_col : if bid/ask aren't available, use this column and a
        synthetic spread of 0 (costs always "covered"). Useful for
        testing with close-only data.
    granularity : the bar timeframe of `df` (for the plot title only).
    plot : if True (and matplotlib available), plot the coverage %.
    save_path : if given, save the plot to this path instead of showing.

    Returns
    -------
    DataFrame indexed by UTC hour (0-23) with column 'coverage_pct':
        the fraction of bars where |price_change| > spread.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex")

    # Ensure UTC for hour extraction
    if df.index.tz is None:
        log.warning("Index is tz-naive; assuming UTC.")
        idx_utc = df.index.tz_localize("UTC")
    else:
        idx_utc = df.index.tz_convert("UTC")

    out = df.copy()
    out.index = idx_utc

    # Compute mid price, spread
    if bid_col in out.columns and ask_col in out.columns:
        out["mid_price"] = (out[bid_col] + out[ask_col]) / 2.0
        out["spread"] = out[ask_col] - out[bid_col]
    elif mid_col is not None and mid_col in out.columns:
        out["mid_price"] = out[mid_col]
        out["spread"] = 0.0
        log.info("Using synthetic spread=0 (no bid/ask columns).")
    else:
        raise ValueError(
            f"Need either ({bid_col}, {ask_col}) or {mid_col!r} columns; "
            f"got {list(out.columns)}"
        )

    out["hour"] = out.index.hour
    out["price_change"] = out["mid_price"].diff().abs()
    out["covered_costs"] = out["price_change"] > out["spread"]

    hourly = out.groupby("hour")["covered_costs"].mean().to_frame("coverage_pct")
    hourly["coverage_pct"] = hourly["coverage_pct"] * 100.0  # to %

    if plot and _HAS_MPL:
        fig, ax = plt.subplots(figsize=(12, 8))
        hourly["coverage_pct"].plot(kind="bar", ax=ax, color="steelblue")
        ax.set_xlabel("UTC Hour")
        ax.set_ylabel("Percentage of Bars Where Costs Covered (%)")
        ax.set_title(f"Cost Coverage by Hour — granularity={granularity}")
        ax.grid(True, alpha=0.3, axis="y")
        if save_path:
            fig.savefig(save_path, dpi=100, bbox_inches="tight")
            log.info(f"Saved plot to {save_path}")
        else:
            plt.show()
        plt.close(fig)

    return hourly


def best_trading_hours(
    df: pd.DataFrame,
    *,
    top_n: int = 3,
    **kwargs,
) -> list[int]:
    """
    Return the top-N UTC hours with the highest cost-coverage %.
    """
    hourly = find_optimal_trading_time(df, plot=False, **kwargs)
    return hourly.sort_values("coverage_pct", ascending=False).head(top_n).index.tolist()


# ── CLI ──────────────────────────────────────────────────────────────────────

def _cli():
    """Quick CLI: python -m analysis.optimal_trading_time <parquet_file>"""
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m analysis.optimal_trading_time <parquet_file> [bid_col] [ask_col]")
        sys.exit(1)
    path = sys.argv[1]
    bid_col = sys.argv[2] if len(sys.argv) > 2 else "bid"
    ask_col = sys.argv[3] if len(sys.argv) > 3 else "ask"
    df = pd.read_parquet(path)
    if bid_col not in df.columns or ask_col not in df.columns:
        print(f"Available columns: {list(df.columns)}")
        print(f"Specified bid/ask not found; using 'close' as mid with synthetic spread=0")
        hourly = find_optimal_trading_time(df, mid_col="close", plot=False)
    else:
        hourly = find_optimal_trading_time(df, bid_col=bid_col, ask_col=ask_col, plot=False)
    print("\nCost coverage by UTC hour:")
    print(hourly.to_string())
    print(f"\nTop 3 hours: {best_trading_hours(df, top_n=3)}")


if __name__ == "__main__":
    _cli()
