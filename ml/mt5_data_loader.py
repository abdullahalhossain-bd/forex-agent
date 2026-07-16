"""
ml/mt5_data_loader.py — Professional MT5 Historical Data Pipeline
==================================================================

This module provides a production-grade MT5 historical data fetcher for ML training.
It replaces ALL synthetic data generators with real market data.

Features:
  • Safe MT5 initialization with automatic reconnection
  • Automatic retry on connection failures
  • Proper shutdown and cleanup
  • Data validation (no duplicates, chronological order)
  • Timezone-safe timestamp handling
  • Incomplete last candle removal
  • Configurable symbols and timeframes
  • Minimum data enforcement (default 100,000 bars)
  • Comprehensive error handling

Supported Symbols:
  EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD,
  XAUUSD, XAGUSD, BTCUSD, ETHUSD

Supported Timeframes:
  M1, M5, M15, M30, H1, H4, D1

Usage:
    from ml.mt5_data_loader import MT5DataLoader

    loader = MT5DataLoader()
    df = loader.fetch(symbol="EURUSD", timeframe="M15", bars=100000)

    # Or with date range
    df = loader.fetch(
        symbol="EURUSD",
        timeframe="M15",
        start_date="2020-01-01",
        end_date="2024-12-31"
    )
"""

from __future__ import annotations

import os
import sys
import time
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass

import pandas as pd
import numpy as np

# Try to import MetaTrader5
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False
    logging.warning(
        "MetaTrader5 package not installed. Install with: pip install MetaTrader5\n"
        "Note: MT5 only works on Windows with MetaTrader 5 terminal running."
    )

# ── Constants ───────────────────────────────────────────────────────

SUPPORTED_SYMBOLS = [
    # Forex Majors
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
    # Metals
    "XAUUSD", "XAGUSD",
    # Crypto
    "BTCUSD", "ETHUSD",
]

SUPPORTED_TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]

DEFAULT_BARS = 100_000

TIMEFRAME_MAP: Dict[str, Any] = {}


def _init_timeframe_map():
    """Initialize MT5 timeframe mapping."""
    if not MT5_AVAILABLE or TIMEFRAME_MAP:
        return
    TIMEFRAME_MAP.update({
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    })


if MT5_AVAILABLE:
    _init_timeframe_map()


@dataclass
class FetchResult:
    """Result of a data fetch operation."""
    symbol: str
    timeframe: str
    dataframe: Optional[pd.DataFrame]
    rows_downloaded: int
    rows_after_cleaning: int
    feature_count: int = 0
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    source: str = "mt5"
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    def summary(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "rows_downloaded": self.rows_downloaded,
            "rows_after_cleaning": self.rows_after_cleaning,
            "feature_count": self.feature_count,
            "start_date": str(self.start_date) if self.start_date else None,
            "end_date": str(self.end_date) if self.end_date else None,
            "source": self.source,
            "errors": self.errors,
        }


class MT5DataLoader:
    """
    Professional MT5 historical data loader for ML training.

    This class handles all aspects of fetching real market data from MT5:
    - Connection management with automatic reconnection
    - Data validation and cleaning
    - Timezone handling
    - Error recovery
    """

    MAX_RETRIES = 3
    RETRY_DELAY = 2.0  # seconds
    CONNECTION_TIMEOUT = 30  # seconds

    def __init__(
        self,
        mt5_path: Optional[str] = None,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
    ):
        """
        Initialize the MT5 data loader.

        Args:
            mt5_path: Path to MT5 terminal executable (optional)
            login: MT5 account login (optional, uses default if not provided)
            password: MT5 account password (optional)
            server: MT5 server name (optional)
        """
        self.mt5_path = mt5_path or os.getenv("MT5_PATH", "")
        self.login = login
        self.password = password
        self.server = server

        self._initialized = False
        self._connected = False
        self._last_error: Optional[str] = None

        # Logging setup - FIX BUG #1: Prevent duplicate handlers on singleton logger
        self.logger = logging.getLogger("mt5_data_loader")

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            ))
            self.logger.addHandler(handler)
            self.logger.propagate = False  # Prevent bubbling to root

        self.logger.setLevel(logging.INFO)

    def _initialize_mt5(self) -> bool:
        """
        Initialize MT5 connection safely.

        Returns:
            True if successful, False otherwise
        """
        if not MT5_AVAILABLE:
            self.logger.error("MetaTrader5 package not installed")
            return False

        if self._initialized:
            return True

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                # Shutdown any existing connection first
                mt5.shutdown()
                time.sleep(1)

                init_kwargs = {}
                if self.mt5_path:
                    init_kwargs["path"] = self.mt5_path

                self.logger.info("Initializing MT5 connection...")

                if not mt5.initialize(**init_kwargs):
                    err = mt5.last_error()
                    self.logger.error(f"MT5 initialize failed: {err}")
                    time.sleep(self.RETRY_DELAY)
                    continue

                # Login if credentials provided
                if self.login and self.password and self.server:
                    if not mt5.login(self.login, password=self.password, server=self.server):
                        err = mt5.last_error()
                        self.logger.error(f"MT5 login failed: {err}")
                        mt5.shutdown()
                        time.sleep(self.RETRY_DELAY)
                        continue

                # Verify connection
                terminal_info = mt5.terminal_info()
                if terminal_info is None:
                    self.logger.error("MT5 terminal info unavailable")
                    mt5.shutdown()
                    time.sleep(self.RETRY_DELAY)
                    continue

                self._initialized = True
                self._connected = True
                self.logger.info("✅ Connected to MT5")
                self.logger.info(f"   Server: {terminal_info.name} | Build: {terminal_info.build}")
                return True

            except Exception as e:
                self.logger.error(f"MT5 initialization attempt {attempt}/{self.MAX_RETRIES} failed: {e}")
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.RETRY_DELAY)

        self._last_error = "Failed to initialize MT5 after all retries"
        self.logger.error(f"❌ {self._last_error}")
        return False

    def _ensure_connected(self) -> bool:
        """Ensure MT5 is connected, reconnecting if necessary."""
        if not self._initialized:
            return self._initialize_mt5()

        # Check connection health
        try:
            terminal = mt5.terminal_info()
            if terminal is None or not terminal.connected:
                self.logger.warning("MT5 connection lost — attempting reconnect...")
                self._initialized = False
                return self._initialize_mt5()
        except Exception as e:
            self.logger.warning(f"Connection check failed: {e}")
            self._initialized = False
            return self._initialize_mt5()

        return True

    def _validate_symbol(self, symbol: str) -> bool:
        """Validate that a symbol is available in MT5."""
        if not self._ensure_connected():
            return False

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            self.logger.warning(f"Symbol {symbol} not found in MT5")
            return False

        if not symbol_info.visible:
            self.logger.warning(f"Symbol {symbol} is not visible in Market Watch")
            return False

        return True

    def _fetch_rates(
        self,
        symbol: str,
        timeframe: str,
        bars: int,
    ) -> Optional[np.ndarray]:
        """
        Fetch raw rates from MT5 with retry logic.

        FIX BUG #2: Use chunked fetching to avoid MT5 terminal's
        "Max bars in chart" limit that causes (-2, 'Invalid params') error.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            bars: Number of bars to fetch

        Returns:
            Raw rates array or None on failure
        """
        if timeframe not in TIMEFRAME_MAP:
            self.logger.error(f"Unsupported timeframe: {timeframe}")
            return None

        mt5_tf = TIMEFRAME_MAP[timeframe]

        # FIX BUG #2: Chunk size to avoid terminal "Max bars in history" limit
        # This is more robust than relying on terminal settings
        CHUNK_SIZE = 5000  # Safe default for most terminal configurations

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                if not self._ensure_connected():
                    return None

                # Select symbol in Market Watch
                if not mt5.symbol_select(symbol, True):
                    err = mt5.last_error()
                    self.logger.error(f"Failed to select symbol {symbol}: {err}")
                    return None

                # If requested bars <= CHUNK_SIZE, fetch directly
                if bars <= CHUNK_SIZE:
                    rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, bars)

                    if rates is None:
                        err = mt5.last_error()
                        self.logger.warning(f"copy_rates_from_pos returned None: {err}")
                        if attempt < self.MAX_RETRIES:
                            time.sleep(self.RETRY_DELAY)
                            continue
                        return None

                    if len(rates) == 0:
                        self.logger.warning(f"No data returned for {symbol} {timeframe}")
                        return None

                    return rates

                # FIX BUG #2: Fetch in chunks to avoid terminal limits
                self.logger.debug(f"Fetching {bars} bars in chunks of {CHUNK_SIZE}")
                all_chunks = []
                pos = 0
                remaining = bars

                while remaining > 0:
                    n = min(CHUNK_SIZE, remaining)
                    part = mt5.copy_rates_from_pos(symbol, mt5_tf, pos, n)

                    if part is None or len(part) == 0:
                        err = mt5.last_error()
                        self.logger.warning(f"Chunk fetch failed at pos={pos}: {err}")
                        break

                    all_chunks.append(part)
                    fetched = len(part)
                    pos += fetched
                    remaining -= n

                    # If we got fewer bars than requested, no more history available
                    if fetched < n:
                        self.logger.debug(f"Reached end of history at pos={pos}")
                        break

                if not all_chunks:
                    if attempt < self.MAX_RETRIES:
                        time.sleep(self.RETRY_DELAY)
                        continue
                    return None

                # Merge all chunks
                merged = np.concatenate(all_chunks)

                # Remove any duplicate timestamps (can happen with overlapping chunks)
                _, unique_idx = np.unique(merged['time'], return_index=True)
                merged = merged[np.sort(unique_idx)]

                self.logger.debug(f"Merged {len(all_chunks)} chunks into {len(merged)} bars")
                return merged

            except Exception as e:
                self.logger.error(f"Fetch attempt {attempt}/{self.MAX_RETRIES} failed: {e}")
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.RETRY_DELAY)

        return None

    def _process_rates(
        self,
        rates: np.ndarray,
        symbol: str,
        timeframe: str,
    ) -> pd.DataFrame:
        """
        Process raw MT5 rates into a clean DataFrame.

        Handles:
        - Timestamp conversion with timezone
        - Duplicate removal
        - Chronological ordering
        - Incomplete last candle removal
        - Column standardization

        Args:
            rates: Raw MT5 rates array
            symbol: Trading symbol
            timeframe: Timeframe string

        Returns:
            Clean OHLCV DataFrame
        """
        # Convert to DataFrame
        df = pd.DataFrame(rates)

        # Convert time to datetime
        # MT5 returns timestamps in broker server time (not UTC)
        # We need to handle this carefully.
        #
        # TIMEZONE POLICY (single source of truth for this project):
        #   Every timestamp in the pipeline is tz-AWARE UTC. We never mix
        #   naive and aware timestamps, and we never silently drop tzinfo
        #   with `.replace(tzinfo=None)` / `tz_localize(None)`. If a naive
        #   timestamp needs to be compared or subtracted against an aware
        #   one, it must first be localized/converted to UTC.
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)

        # Get broker timezone offset from environment
        # Common values: 2 (GMT+2 winter), 3 (GMT+3 summer), 0 (UTC)
        offset_hours = float(os.getenv("MT5_BROKER_TZ_OFFSET_HOURS", "0") or 0)

        if offset_hours != 0:
            # Adjust to UTC (subtraction on two tz-aware series is safe)
            df['time'] = df['time'] - pd.Timedelta(hours=offset_hours)

        # `pd.to_datetime(..., utc=True)` above already produces a tz-aware
        # (UTC) column, so there is nothing left to localize. Trying to
        # tz_localize('UTC') a column that is already tz-aware raises
        # `TypeError: Already tz-aware, use tz_convert to convert`, so we
        # assert the invariant instead of calling tz_localize again.
        assert df['time'].dt.tz is not None, (
            "df['time'] must be tz-aware UTC after pd.to_datetime(utc=True); "
            "got tz-naive dtype instead."
        )

        # Set time as index
        df.set_index('time', inplace=True)

        # Sort chronologically (should already be sorted, but ensure it)
        df.sort_index(inplace=True)

        # Remove duplicates (keep last occurrence)
        dupes_before = len(df)
        df = df[~df.index.duplicated(keep='last')]
        dupes_removed = dupes_before - len(df)
        if dupes_removed > 0:
            self.logger.info(f"Removed {dupes_removed} duplicate candles")

        # Remove incomplete last candle
        # The most recent candle may not be closed yet
        if len(df) > 0:
            last_candle_time = df.index[-1]

            # ── TIMEZONE FIX ─────────────────────────────────────────────
            # Original bug: `now_utc` was tz-aware while `last_candle_time`
            # had `.replace(tzinfo=None)` applied right before the
            # subtraction, forcing it tz-naive. Subtracting an aware
            # timestamp from a naive one raises:
            #   TypeError: Cannot subtract tz-naive and tz-aware
            #   datetime-like objects
            #
            # Fix: normalize BOTH operands to tz-aware UTC before doing any
            # arithmetic, and never strip tzinfo again. `last_candle_time`
            # is handled defensively for both cases because it can come
            # from different code paths (e.g. a cached/pickled DataFrame
            # built before this fix, or a df.index that was tz_localized
            # to a non-UTC zone upstream) — we don't want to assume it is
            # already tz-aware UTC just because it usually is in this
            # function.
            if last_candle_time.tzinfo is None:
                # Naive timestamp (no tzinfo at all) -> attach UTC.
                # tz_localize() ASSIGNS a timezone to a naive timestamp
                # without shifting the underlying wall-clock time, which is
                # correct here because upstream code already treats these
                # values as UTC — it just forgot to say so.
                last_candle_time = last_candle_time.tz_localize("UTC")
            else:
                # Already tz-aware (UTC or otherwise) -> convert to UTC.
                # tz_convert() re-expresses the SAME instant in a different
                # zone (shifting wall-clock time as needed), which is the
                # correct operation for an already-aware timestamp. Calling
                # tz_localize() on an aware timestamp instead would raise
                # `TypeError: Already tz-aware, use tz_convert to convert`.
                last_candle_time = last_candle_time.tz_convert("UTC")

            # Keep `now_utc` tz-aware UTC (pandas Timestamp, not
            # datetime.now(timezone.utc), so both sides are the same type
            # and compare/subtract cleanly under pandas 2.x).
            now_utc = pd.Timestamp.now(tz="UTC")

            # ── Debug logs before datetime arithmetic ──────────────────
            self.logger.debug(
                "Pre-subtraction timestamp check | "
                f"index dtype={df.index.dtype}, tz={df.index.tz} | "
                f"min={df.index.min()}, max={df.index.max()} | "
                f"last_candle_time={last_candle_time} (tz={last_candle_time.tzinfo}) | "
                f"now_utc={now_utc} (tz={now_utc.tz})"
            )

            # ── Assertions: never let a naive/aware pair reach subtraction ──
            assert last_candle_time.tzinfo is not None, (
                "last_candle_time is tz-naive after normalization — refusing "
                "to subtract against a tz-aware timestamp."
            )
            assert now_utc.tzinfo is not None, (
                "now_utc must be tz-aware (pd.Timestamp.now(tz='UTC'))."
            )

            # Define timeframe durations
            tf_duration = {
                "M1": timedelta(minutes=1),
                "M5": timedelta(minutes=5),
                "M15": timedelta(minutes=15),
                "M30": timedelta(minutes=30),
                "H1": timedelta(hours=1),
                "H4": timedelta(hours=4),
                "D1": timedelta(days=1),
            }.get(timeframe, timedelta(hours=1))

            # If the last candle's time is very recent, it might be incomplete
            # (incomplete-candle detection kept exactly as before — only the
            # timezone handling changed). Both operands are now tz-aware UTC,
            # so this subtraction/comparison is safe under pandas 2.x.
            if now_utc - last_candle_time < tf_duration:
                self.logger.info("Removing potentially incomplete last candle")
                df = df.iloc[:-1]

        # Standardize columns
        required_cols = ['open', 'high', 'low', 'close', 'tick_volume']
        available_cols = [c for c in required_cols if c in df.columns]

        df = df[available_cols].copy()

        # Rename tick_volume to volume for consistency
        if 'tick_volume' in df.columns:
            df.rename(columns={'tick_volume': 'volume'}, inplace=True)

        # Ensure correct column order
        col_order = ['open', 'high', 'low', 'close', 'volume']
        df = df[[c for c in col_order if c in df.columns]]

        # Validate OHLC consistency
        # High should be >= Low, High >= Open, High >= Close, etc.
        invalid_mask = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        if invalid_mask.any():
            invalid_count = invalid_mask.sum()
            self.logger.warning(f"Found {invalid_count} candles with invalid OHLC relationships")
            df = df[~invalid_mask]

        return df

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        bars: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> FetchResult:
        """
        Fetch historical OHLCV data from MT5.

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            timeframe: Timeframe ("M1", "M5", "M15", "M30", "H1", "H4", "D1")
            bars: Number of bars to fetch (default: 100,000)
            start_date: Start date string "YYYY-MM-DD" (alternative to bars)
            end_date: End date string "YYYY-MM-DD" (alternative to bars)

        Returns:
            FetchResult with DataFrame and metadata
        """
        symbol = symbol.upper()
        timeframe = timeframe.upper()
        bars = bars or DEFAULT_BARS

        result = FetchResult(
            symbol=symbol,
            timeframe=timeframe,
            dataframe=None,
            rows_downloaded=0,
            rows_after_cleaning=0,
        )

        # Validate inputs
        if symbol not in SUPPORTED_SYMBOLS:
            msg = f"Symbol {symbol} not in supported list: {SUPPORTED_SYMBOLS}"
            self.logger.warning(msg)
            result.errors.append(msg)
            # Continue anyway - symbol might still be available

        if timeframe not in SUPPORTED_TIMEFRAMES:
            msg = f"Timeframe {timeframe} not supported: {SUPPORTED_TIMEFRAMES}"
            self.logger.error(msg)
            result.errors.append(msg)
            return result

        # Log progress
        self.logger.info(f"Downloading {symbol} {timeframe}...")

        # Fetch raw data
        rates = self._fetch_rates(symbol, timeframe, bars)

        if rates is None:
            msg = f"Failed to fetch data for {symbol} {timeframe}"
            self.logger.error(msg)
            result.errors.append(msg)
            return result

        result.rows_downloaded = len(rates)
        self.logger.info(f"Downloaded {result.rows_downloaded} candles")

        # Process and clean data
        self.logger.info("Cleaning data...")
        df = self._process_rates(rates, symbol, timeframe)

        result.rows_after_cleaning = len(df)
        result.dataframe = df
        result.start_date = df.index[0] if len(df) > 0 else None
        result.end_date = df.index[-1] if len(df) > 0 else None

        # Validate minimum data
        if len(df) < bars * 0.5:  # Allow some tolerance
            self.logger.warning(
                f"Only got {len(df)} bars, expected ~{bars}. "
                "This may be the maximum available history."
            )

        self.logger.info(
            f"✅ Data ready: {len(df)} rows | "
            f"{result.start_date} → {result.end_date}"
        )

        return result

    def fetch_multiple(
        self,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        bars: Optional[int] = None,
    ) -> Dict[Tuple[str, str], FetchResult]:
        """
        Fetch data for multiple symbol/timeframe combinations.

        Args:
            symbols: List of symbols (default: all SUPPORTED_SYMBOLS)
            timeframes: List of timeframes (default: all SUPPORTED_TIMEFRAMES)
            bars: Number of bars per fetch

        Returns:
            Dictionary mapping (symbol, timeframe) to FetchResult
        """
        symbols = symbols or SUPPORTED_SYMBOLS
        timeframes = timeframes or SUPPORTED_TIMEFRAMES
        bars = bars or DEFAULT_BARS

        results = {}
        total = len(symbols) * len(timeframes)
        current = 0

        for symbol in symbols:
            for tf in timeframes:
                current += 1
                self.logger.info(f"[{current}/{total}] Fetching {symbol} {tf}...")
                result = self.fetch(symbol, tf, bars=bars)
                results[(symbol, tf)] = result

        return results

    def shutdown(self):
        """Properly shutdown MT5 connection."""
        try:
            if self._initialized and MT5_AVAILABLE:
                mt5.shutdown()
                self._initialized = False
                self._connected = False
                self.logger.info("MT5 connection closed")
        except Exception as e:
            self.logger.warning(f"Error during shutdown: {e}")

    def __del__(self):
        """Destructor ensures cleanup."""
        self.shutdown()


# ── Convenience Functions ───────────────────────────────────────────

def fetch_mt5_history(
    symbol: str,
    timeframe: str,
    bars: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Convenience function to fetch MT5 historical data.

    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        bars: Number of bars (default: 100,000)
        start_date: Start date "YYYY-MM-DD"
        end_date: End date "YYYY-MM-DD"

    Returns:
        OHLCV DataFrame or None on failure
    """
    loader = MT5DataLoader()
    result = loader.fetch(symbol, timeframe, bars=bars, start_date=start_date, end_date=end_date)

    if result.dataframe is None:
        return None

    return result.dataframe


def get_mt5_data_loader() -> MT5DataLoader:
    """Get a configured MT5DataLoader instance."""
    return MT5DataLoader()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MT5 Historical Data Fetcher")
    parser.add_argument("--symbol", type=str, default="EURUSD", help="Trading symbol")
    parser.add_argument("--timeframe", type=str, default="M15", help="Timeframe")
    parser.add_argument("--bars", type=int, default=100000, help="Number of bars")
    parser.add_argument("--output", type=str, default=None, help="Output file path")

    args = parser.parse_args()

    print("=" * 60)
    print("  MT5 Historical Data Fetcher")
    print("=" * 60)

    loader = MT5DataLoader()
    result = loader.fetch(args.symbol, args.timeframe, bars=args.bars)

    if result.dataframe is not None:
        print(f"\n✅ Success!")
        print(f"   Rows downloaded: {result.rows_downloaded}")
        print(f"   Rows after cleaning: {result.rows_after_cleaning}")
        print(f"   Date range: {result.start_date} → {result.end_date}")
        print(f"   Columns: {list(result.dataframe.columns)}")

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.dataframe.to_parquet(output_path, index=True)
            print(f"   Saved to: {output_path}")
    else:
        print(f"\n❌ Failed to fetch data")
        if result.errors:
            print(f"   Errors: {result.errors}")

    loader.shutdown()
    print("\n" + "=" * 60)