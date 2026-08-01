# broker/mt5_historical_fetcher.py — Chunked MT5 historical data fetcher
# =============================================================================
# Ported from: https://github.com/JimmyAreaFiscal/MercadoFinanceiro/blob/main/conexao.py
# Original author: JimmyAreaFiscal — license not specified (public repo)
# Original function: importar_dados_historicos()
#
# Fetches large date ranges of MT5 historical data in MONTHLY CHUNKS.
#
# Why this is valuable
# --------------------
# MT5's `copy_rates_range()` has an internal limit on how many bars it returns
# in a single call (varies by broker/server, typically ~65k bars). For multi-
# year backtests on M1 or M5 timeframes, a single call silently truncates.
#
# This fetcher loops month-by-month, concatenating results, so you always get
# the full requested range. It also adds:
#   - `ticker` column (for multi-symbol concatenation)
#   - `data` (date) and `horario` (time) columns (Brazilian book convention)
#   - Graceful handling of MT5 not being available (Linux / CI)
#
# Faithful to the original, with these additions:
#   - English function name + Portuguese alias
#   - Optional timeframe parameter (default M1, same as original)
#   - Type hints + docstring
#   - Returns empty DataFrame (not None) on failure
# =============================================================================

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from utils.logger import get_logger

log = get_logger("mt5_historical_fetcher")

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False


# ── Timeframe mapping (MT5 constants) ────────────────────────────────────────
TIMEFRAMES = {
    "M1":  mt5.TIMEFRAME_M1 if MT5_AVAILABLE else 1,
    "M5":  mt5.TIMEFRAME_M5 if MT5_AVAILABLE else 5,
    "M15": mt5.TIMEFRAME_M15 if MT5_AVAILABLE else 15,
    "M30": mt5.TIMEFRAME_M30 if MT5_AVAILABLE else 30,
    "H1":  mt5.TIMEFRAME_H1 if MT5_AVAILABLE else 60,
    "H4":  mt5.TIMEFRAME_H4 if MT5_AVAILABLE else 240,
    "D1":  mt5.TIMEFRAME_D1 if MT5_AVAILABLE else 1440,
    "W1":  mt5.TIMEFRAME_W1 if MT5_AVAILABLE else 10080,
    "MN1": mt5.TIMEFRAME_MN1 if MT5_AVAILABLE else 43200,
}


def fetch_historical_data(
    ticker: str,
    start: datetime,
    end: datetime,
    timeframe: int = None,
) -> pd.DataFrame:
    """
    Fetch MT5 historical bars for `ticker` from `start` to `end`, in monthly
    chunks. Returns a DataFrame with columns:
        time, open, high, low, close, tick_volume, real_volume, spread,
        ticker, data (date), horario (time)

    Parameters
    ----------
    ticker : MT5 symbol, e.g. "EURUSD"
    start, end : datetime objects defining the date range
    timeframe : MT5 timeframe constant (e.g., mt5.TIMEFRAME_H1).
        If None, defaults to mt5.TIMEFRAME_M1 (same as original).

    Returns
    -------
    pd.DataFrame indexed by datetime. Empty if MT5 unavailable or no data.
    """
    if not MT5_AVAILABLE:
        log.warning("MetaTrader5 not available — returning empty DataFrame. "
                    "MT5 only works on Windows with the terminal running.")
        return pd.DataFrame()

    if timeframe is None:
        timeframe = mt5.TIMEFRAME_M1

    from dateutil.relativedelta import relativedelta

    all_data = pd.DataFrame()
    start_loop = start

    while start_loop <= end:
        end_loop = start_loop + relativedelta(months=1) - relativedelta(days=1)
        if end_loop > end:
            end_loop = end

        log.debug(f"Fetching {ticker} {start_loop.date()} → {end_loop.date()}")
        rates = mt5.copy_rates_range(ticker, timeframe, start_loop, end_loop)

        if rates is not None and len(rates) > 0:
            chunk = pd.DataFrame(rates)
            chunk["ticker"] = str(ticker)
            chunk.index = pd.to_datetime(chunk["time"], unit="s")
            chunk["data"] = chunk.index.date
            chunk["horario"] = chunk.index.time
            all_data = pd.concat([all_data, chunk])

        start_loop += relativedelta(months=1)

    if all_data.empty:
        log.warning(f"No data fetched for {ticker} {start} → {end}")

    return all_data


# ── Portuguese alias (backwards compatibility) ───────────────────────────────

def importar_dados_historicos(ticker, start, end, timeframe=None):
    """Portuguese alias for fetch_historical_data (matches original notebook)."""
    if timeframe is None and MT5_AVAILABLE:
        timeframe = mt5.TIMEFRAME_M1
    return fetch_historical_data(ticker, start, end, timeframe)


# ── Convenience: fetch + cache to parquet ────────────────────────────────────

def fetch_and_cache(
    ticker: str,
    start: datetime,
    end: datetime,
    timeframe: str = "H1",
    cache_dir: str = "data/backtest_cache",
) -> pd.DataFrame:
    """
    Fetch historical data and cache it as parquet for later use.
    Filename: <ticker>_<timeframe>_<start>_<end>.parquet

    If a cached file already exists for this exact range, loads it instead
    of re-fetching from MT5.
    """
    from pathlib import Path

    tf = TIMEFRAMES.get(timeframe.upper())
    if tf is None:
        raise ValueError(f"Unknown timeframe {timeframe!r}. "
                         f"Supported: {list(TIMEFRAMES.keys())}")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    filename = f"{ticker}_{timeframe}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.parquet"
    cache_file = cache_path / filename

    if cache_file.exists():
        log.info(f"Loading cached data from {cache_file}")
        return pd.read_parquet(cache_file)

    log.info(f"Fetching {ticker} {timeframe} {start} → {end} from MT5...")
    df = fetch_historical_data(ticker, start, end, tf)
    if not df.empty:
        df.to_parquet(cache_file)
        log.info(f"Cached {len(df)} bars to {cache_file}")
    return df


# ── Smoke test (skipped if MT5 unavailable) ──────────────────────────────────
if __name__ == "__main__":
    if not MT5_AVAILABLE:
        print("MetaTrader5 not available — smoke test skipped (expected on Linux/CI).")
        print("To test: run on Windows with MT5 terminal running.")
    else:
        from datetime import datetime, timedelta
        end = datetime.now()
        start = end - timedelta(days=30)
        df = fetch_historical_data("EURUSD", start, end, mt5.TIMEFRAME_H1)
        print(f"Fetched {len(df)} bars of EURUSD H1")
        if not df.empty:
            print(df.head())
            print(df.tail())
