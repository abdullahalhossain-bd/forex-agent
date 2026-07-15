# backtest/vectorized_base.py — Vectorized backtesting framework
# =============================================================================
# Ported from: https://github.com/trentstauff/FXBot/blob/master/backtesting/Backtester.py
# Original author: Trent Stauffner — MIT license (inferred from repo)
#
# Vectorized backtesting base class. Each strategy subclass:
#   1. Calls super().__init__(symbol, start, end, granularity, trading_cost)
#      which loads historical data and computes log returns.
#   2. Overrides prepare_data() to add strategy-specific columns.
#   3. Overrides test() to compute position + strategy returns.
#   4. Optionally overrides optimize() to grid-search parameters.
#   5. Calls plot_results() to visualize cumulative returns vs buy & hold.
#
# Differences from FXBot:
#   - Data source: FXBot hard-codes tpqoa (OANDA). We accept ANY DataFrame
#     with a 'close' column, OR a callable `data_loader(symbol, start, end,
#     granularity) -> DataFrame`. This lets the class work with our existing
#     MT5 backtest cache, yfinance, or any other data source.
#   - We use mid-price by default (FXBot used close). Pass `price_col="bid"`
#     or `"ask"` to override.
#   - All numerical edge cases (empty data, all-NaN, single bar) are
#     handled explicitly — FXBot would crash.
#
# This module is INDEPENDENT of our existing `backtest/engine.py` (which is
# event-driven). Both can coexist:
#   - `backtest/engine.py` — event-driven, supports our `strategies/` classes.
#   - `backtest/vectorized_base.py` — vectorized, supports the simpler
#     FXBot-style strategies. Faster for parameter optimization.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("vectorized_backtest")


# ── Optional plotting (matplotlib is a hard dep of the project anyway) ───────
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# ── Default data loader: reads from our MT5 backtest cache ───────────────────
def _default_data_loader(symbol: str, start: str, end: str,
                         granularity: str) -> pd.DataFrame:
    """
    Default data loader. Reads parquet from data/backtest_cache/.

    Expects files named like: EURUSD_H1_3000.parquet
    Returns a DataFrame with at least a 'close' column and a DatetimeIndex.
    """
    from pathlib import Path
    from core.constants import DATA_DIR

    cache_dir = DATA_DIR / "backtest_cache"
    # Try to find a matching file
    pattern = f"{symbol}_{granularity}_*.parquet"
    matches = sorted(cache_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No cached data for {symbol} {granularity} in {cache_dir}. "
            f"Run `python -m backtest.mt5_bulk_fetcher --symbol {symbol} "
            f"--timeframe {granularity}` first, or pass a custom data_loader."
        )
    df = pd.read_parquet(matches[-1])
    if "close" not in df.columns:
        raise ValueError(f"Cached file {matches[-1]} has no 'close' column")
    # Filter by date range
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Cached file must have a DatetimeIndex")
    df = df.loc[start:end]
    if df.empty:
        raise ValueError(f"No data in range {start} to {end}")
    return df


# ── Result container ─────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    """Holds the result of a single backtest run."""
    performance: float          # cumulative strategy return (1.0 = breakeven)
    out_performance: float      # strategy - buy_and_hold
    n_trades: int
    n_bars: int
    hit_ratio: Optional[float] = None  # for ML strategies
    results_df: Optional[pd.DataFrame] = None
    best_params: Optional[dict] = None  # for optimized runs


# ── Base Backtester ──────────────────────────────────────────────────────────

class VectorizedBacktester:
    """
    Vectorized backtesting framework. Subclass and override `prepare_data()`
    and `test()`.

    Example
    -------
    >>> from backtest.vectorized_strategies import SMABacktest
    >>> bt = SMABacktest("EURUSD", "2024-01-01", "2024-06-01",
    ...                  "H1", smas=10, smal=20, trading_cost=0.0001)
    >>> result = bt.test()
    >>> bt.plot_results()
    >>> best = bt.optimize()  # grid-search smas, smal
    """

    def __init__(
        self,
        symbol: str,
        start: str,
        end: str,
        granularity: str = "D",
        trading_cost: float = 0.0,
        *,
        price_col: str = "close",
        data_loader: Optional[Callable[[str, str, str, str], pd.DataFrame]] = None,
        data: Optional[pd.DataFrame] = None,
    ):
        """
        Parameters
        ----------
        symbol : instrument ticker, e.g., "EURUSD"
        start, end : date strings "YYYY-MM-DD"
        granularity : bar timeframe, e.g., "D", "H1", "M15"
        trading_cost : per-trade cost subtracted from strategy returns
        price_col : which column to use as price (default "close")
        data_loader : callable(symbol, start, end, granularity) → DataFrame.
            If None, uses the default MT5 cache loader.
        data : pre-loaded DataFrame. If given, `data_loader` is ignored.
        """
        self.symbol = symbol
        self.start = start
        self.end = end
        self.granularity = granularity
        self.trading_cost = trading_cost
        self.price_col = price_col

        self._results: Optional[pd.DataFrame] = None

        if data is not None:
            self._data = self._prepare_initial(data)
        else:
            loader = data_loader or _default_data_loader
            raw = loader(symbol, start, end, granularity)
            self._data = self._prepare_initial(raw)

        # Strategy subclass can add columns in prepare_data()
        self._data = self.prepare_data()

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _prepare_initial(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute log returns. Returns a copy with a 'price' and 'returns' column."""
        if self.price_col not in df.columns:
            raise ValueError(f"DataFrame must have a {self.price_col!r} column; got {list(df.columns)}")
        out = df.copy()
        out["price"] = out[self.price_col]
        out["returns"] = np.log(out["price"].div(out["price"].shift(1)))
        out.dropna(subset=["price"], inplace=True)
        return out

    # ── To override ──────────────────────────────────────────────────────────

    def prepare_data(self) -> pd.DataFrame:
        """Override to add strategy-specific columns. Default: no-op."""
        return self._data.copy()

    def test(self, **kwargs) -> Tuple[float, float]:
        """Override. Must set self._results and return (performance, out_performance)."""
        raise NotImplementedError

    def optimize(self, **kwargs):
        """Override to grid-search parameters. Default: not implemented."""
        raise NotImplementedError

    # ── Common API ───────────────────────────────────────────────────────────

    def get_data(self) -> pd.DataFrame:
        return self._data

    def get_results(self) -> Optional[pd.DataFrame]:
        if self._results is None:
            log.warning("No results yet — call .test() first.")
        return self._results

    def plot_results(self, columns=("creturns", "cstrategy"), save_path: Optional[str] = None):
        """
        Plot cumulative returns (strategy vs buy & hold).
        If `save_path` is given, save PNG instead of showing.
        """
        if self._results is None:
            log.warning("No results to plot — call .test() first.")
            return
        if not _HAS_MPL:
            log.warning("matplotlib not available — cannot plot.")
            return

        fig, ax = plt.subplots(figsize=(12, 8))
        # Only plot columns that exist
        cols = [c for c in columns if c in self._results.columns]
        if not cols:
            log.warning(f"None of {columns} exist in results. Available: {list(self._results.columns)}")
            return
        self._results[cols].plot(ax=ax, title=self.symbol)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return")
        ax.grid(True, alpha=0.3)
        if save_path:
            fig.savefig(save_path, dpi=100, bbox_inches="tight")
            log.info(f"Saved plot to {save_path}")
        else:
            plt.show()
        plt.close(fig)

    def _compute_performance(self, data: pd.DataFrame) -> Tuple[float, float, int]:
        """
        Common performance computation. Expects `data` to have:
          - 'returns' : log returns
          - 'strategy': strategy log returns (after trading costs)
          - 'creturns': cumulative log returns (buy & hold)
          - 'cstrategy': cumulative log returns (strategy)
        Returns (performance, out_performance, n_trades).
        """
        if "strategy" not in data.columns:
            raise ValueError("test() must add a 'strategy' column before calling _compute_performance()")
        data = data.copy()
        data["trades"] = data.get("position", pd.Series(0, index=data.index)).diff().fillna(0).abs()
        data["strategy"] = data["strategy"] - data["trades"] * self.trading_cost
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self._results = data

        performance = float(data["cstrategy"].iloc[-1])
        out_performance = performance - float(data["creturns"].iloc[-1])
        n_trades = int(data["trades"].sum())
        return performance, out_performance, n_trades
