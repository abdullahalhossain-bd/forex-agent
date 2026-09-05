"""Strict Live-Trading-Mirror Backtest facade.

Historical inputs are validated and all external/current-data providers are
forced into historical-safe mode. The actual replay is delegated to
``backtest.live_mirror_engine`` so execution uses HistoricalExecutionRouter
and CanonicalPositionState rather than the legacy adapter shortcut.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Optional
import pandas as pd

@dataclass(frozen=True)
class ReplayValidation:
    rows: int
    start: pd.Timestamp
    end: pd.Timestamp

def validate_historical_ohlcv(df: pd.DataFrame, *, min_rows: int = 2) -> ReplayValidation:
    if not isinstance(df, pd.DataFrame): raise TypeError("Historical replay requires a pandas DataFrame")
    if len(df) < min_rows: raise ValueError(f"Historical replay requires at least {min_rows} rows; got {len(df)}")
    missing = {"open", "high", "low", "close"}.difference(df.columns)
    if missing: raise ValueError(f"Historical replay missing OHLC columns: {sorted(missing)}")
    if not isinstance(df.index, pd.DatetimeIndex): raise TypeError("Historical replay index must be a pandas DatetimeIndex")
    if df.index.tz is None: raise ValueError("Historical replay timestamps must be timezone-aware UTC")
    if str(df.index.tz) not in ("UTC", "UTC+00:00"): raise ValueError(f"Historical replay index must be UTC; got {df.index.tz}")
    if not df.index.is_monotonic_increasing: raise ValueError("Historical replay timestamps must be monotonically increasing")
    if df.index.has_duplicates: raise ValueError("Historical replay contains duplicate timestamps")
    numeric = df[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any(): raise ValueError("Historical replay contains non-numeric/NaN OHLC values")
    if (numeric <= 0).any().any(): raise ValueError("Historical replay contains non-positive OHLC values")
    if (numeric["high"] < numeric[["open", "close"]].max(axis=1)).any(): raise ValueError("Historical replay contains high below open/close")
    if (numeric["low"] > numeric[["open", "close"]].min(axis=1)).any(): raise ValueError("Historical replay contains low above open/close")
    return ReplayValidation(len(df), df.index[0], df.index[-1])

def _neutral_sentiment(pair: str) -> dict:
    return {"pair": pair, "retail_long_pct": 50.0, "retail_source": "historical_unavailable_neutral", "fg_index": 50.0, "fg_label": "Neutral", "fg_source": "historical_unavailable_neutral", "currency_strengths": {}, "strength_source": "historical_unavailable_neutral", "dxy_trend": "NEUTRAL", "dxy_change_pct": 0.0, "dxy_source": "historical_unavailable_neutral", "source": "historical_unavailable_neutral"}

@contextmanager
def _strict_replay_environment() -> Iterator[None]:
    import config
    from core.constants import is_backtest_mode, set_backtest_mode
    old_test_mode = getattr(config, "TEST_MODE", False)
    old_simulation_mode = getattr(config, "SIMULATION_MODE", False)
    old_backtest_mode = is_backtest_mode()
    config.TEST_MODE = False
    config.SIMULATION_MODE = True
    set_backtest_mode(True)
    sentiment_cls = None; original_get_all = None
    try:
        from analysis.sentiment_data import SentimentDataProvider
        sentiment_cls = SentimentDataProvider; original_get_all = sentiment_cls.get_all
        sentiment_cls.get_all = lambda self, pair: _neutral_sentiment(pair)
        yield
    finally:
        if sentiment_cls is not None and original_get_all is not None: sentiment_cls.get_all = original_get_all
        config.TEST_MODE = old_test_mode
        config.SIMULATION_MODE = old_simulation_mode
        set_backtest_mode(old_backtest_mode)

def run_live_mirror_backtest(*, symbol: str, df: pd.DataFrame, timeframe: str = "H1", starting_balance: float = 10000.0, warmup_bars: int = 300, max_open_trades: Optional[int] = None, max_hold_bars: int = 100, spread_pips: Optional[float] = None, commission_per_lot: Optional[float] = None, slippage_pips: Optional[float] = None, db_path: str = "backtest/live_mirror.db", verbose: bool = False, save_forensics: bool = True, forensics_path: Optional[str] = None, bypass_checks: Optional[set[str] | list[str]] = None) -> Any:
    validation = validate_historical_ohlcv(df)
    replay_df = df.copy(deep=True)
    replay_df.attrs["live_mirror_validation"] = {"rows": validation.rows, "start": str(validation.start), "end": str(validation.end), "timezone": "UTC"}
    from core.clock import ReplayClock
    replay_clock = ReplayClock(validation.start.to_pydatetime())
    with _strict_replay_environment():
        from backtest.live_mirror_engine import run_live_mirror_engine
        return run_live_mirror_engine(symbol=symbol, df=replay_df, timeframe=timeframe,
            starting_balance=starting_balance, warmup_bars=warmup_bars,
            max_open_trades=max_open_trades, max_hold_bars=max_hold_bars,
            spread_pips=spread_pips, commission_per_lot=commission_per_lot,
            slippage_pips=slippage_pips, clock=replay_clock, verbose=verbose)

__all__ = ["ReplayValidation", "validate_historical_ohlcv", "run_live_mirror_backtest"]
