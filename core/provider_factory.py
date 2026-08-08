"""
core/provider_factory.py — Factory for selecting the right DataProvider.

Decides which provider to use based on:
  1. Explicit `provider_type` argument ("csv", "mt5", "auto")
  2. Environment (MT5 installed? CSVs available?)
  3. Backtest mode flag (CSV preferred in backtest mode if CSVs exist)

Usage:
    from core.provider_factory import make_backtest_provider

    provider = make_backtest_provider(
        symbol="EURUSD",
        timeframe="H1",
        df=optional_df,  # only used for HistoricalMT5Provider
        prefer="csv",     # "csv", "mt5", or "auto"
    )
    provider.advance_to(500)
    market_out = provider.get_market_out("EURUSD", "H1")
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger("provider_factory")


def _csv_available(symbol: str, timeframe: str, data_dir: Optional[Path] = None) -> bool:
    """Check if a CSV file exists for (symbol, timeframe) in either layout."""
    from core.csv_data_provider import _find_csv
    return _find_csv(symbol, timeframe, data_dir) is not None


def make_backtest_provider(
    symbol: str,
    timeframe: str,
    df: Optional[pd.DataFrame] = None,
    prefer: str = "auto",
    data_dir: Optional[Path] = None,
    **kwargs,
):
    """Construct the right backtest DataProvider.

    Args:
        symbol: e.g. "EURUSD"
        timeframe: e.g. "H1"
        df: optional pre-loaded DataFrame (only used by HistoricalMT5Provider)
        prefer: "csv" (force CSV), "mt5" (force HistoricalMT5Provider), or "auto"
            (default — uses CSV if available, else falls back to df-based
            HistoricalMT5Provider)
        data_dir: where to look for CSVs (default: PROJECT_ROOT/data)
        **kwargs: passed to the provider constructor

    Returns:
        A DataProvider instance ready for advance_to() / get_market_out()

    Raises:
        FileNotFoundError if prefer="csv" but no CSV exists
        ValueError if prefer="mt5" but df is None
    """
    prefer = prefer.lower()

    if prefer == "csv":
        from core.csv_data_provider import HistoricalCSVDataProvider
        if not _csv_available(symbol, timeframe, data_dir):
            raise FileNotFoundError(
                f"prefer='csv' but no CSV found for {symbol} {timeframe}. "
                f"Run: python scripts/download_historical_data.py "
                f"--symbols {symbol} --timeframes {timeframe} "
                f"--start 2025-07-01 --end 2026-08-08"
            )
        log.info(f"[provider_factory] Using HistoricalCSVProvider for {symbol} {timeframe}")
        return HistoricalCSVDataProvider(symbol, timeframe, data_dir=data_dir, **kwargs)

    if prefer == "mt5":
        from core.data_provider import HistoricalMT5Provider
        if df is None:
            raise ValueError("prefer='mt5' requires a pre-loaded df argument")
        log.info(f"[provider_factory] Using HistoricalMT5Provider for {symbol} {timeframe}")
        return HistoricalMT5Provider(df, symbol, timeframe, **kwargs)

    # auto: prefer CSV if available, else HistoricalMT5Provider
    if _csv_available(symbol, timeframe, data_dir):
        from core.csv_data_provider import HistoricalCSVDataProvider
        log.info(f"[provider_factory] Auto-selected HistoricalCSVProvider for {symbol} {timeframe}")
        return HistoricalCSVDataProvider(symbol, timeframe, data_dir=data_dir, **kwargs)

    if df is not None:
        from core.data_provider import HistoricalMT5Provider
        log.info(f"[provider_factory] Auto-selected HistoricalMT5Provider for {symbol} {timeframe} (no CSV)")
        return HistoricalMT5Provider(df, symbol, timeframe, **kwargs)

    raise FileNotFoundError(
        f"No data source available for {symbol} {timeframe}. "
        f"Either: (a) download CSV via scripts/download_historical_data.py, "
        f"or (b) pass a pre-loaded df argument."
    )
