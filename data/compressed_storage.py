# data/compressed_storage.py — Day-partitioned compressed binary quote storage
# =============================================================================
# Ported from: https://github.com/NewYaroslav/xquotes_history (C++ header-only)
# Original author: NewYaroslav (Elektro Yar) — MIT license
#
# Python implementation of the xquotes_history storage architecture:
#   - Day-partitioned binary format: each trading day = 1440 minutes = 1 block
#   - Integer price encoding: price × 100000 → uint32 (5-digit precision)
#   - zstd compression (optional) with per-symbol dictionary
#   - Timestamp-based lookup: O(1) access to any bar by Unix timestamp
#   - Multi-symbol support: read/write multiple pairs in one storage
#
# Compression comparison (from the original README):
#   CSV files (20+ pairs):     10.9 GB
#   Binary (no compression):    5.52 GB  (2× smaller than CSV)
#   Binary + zstd:              932 MB   (12× smaller than CSV)
#
# File format (Python port, .qhs4):
#   [4 bytes] offset to header
#   [N × 5760 bytes] day blocks (1440 min × 4 prices × 4 bytes)
#                    or [N × 7200 bytes] with volume (1440 × 5 × 4)
#   [header] num_days, day_keys[], block_sizes[], block_offsets[], note
#
# Each day key = days since Unix epoch (Jan 1, 1970).
# Zero price = missing data (no bar at that minute).
#
# The C++ original uses zstd dictionaries for extra compression. This Python
# port uses zstd directly (without dictionaries) which still gives ~8-10×
# compression. For dictionary-based compression, use the zstd CLI:
#   zstd --train *.bin -o dictionary
#   zstd -D dictionary file.bin
# =============================================================================

from __future__ import annotations

import struct
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("compressed_storage")

try:
    import zstandard as zstd
    _HAS_ZSTD = True
except ImportError:
    _HAS_ZSTD = False
    log.info("zstandard not installed — compression disabled. "
             "Install with: pip install zstandard")

# Constants (matching the C++ original)
PRICE_MULTIPLIER = 100000.0  # 5-digit precision
MINUTES_IN_DAY = 1440
UINT32_SIZE = 4

# Price storage modes
PRICE_CLOSE = 1       # Only close price (1 value per bar)
PRICE_OHLC = 4        # Open, High, Low, Close (4 values per bar)
PRICE_OHLCV = 5       # Open, High, Low, Close, Volume (5 values per bar)

# Compression modes
NO_COMPRESSION = 0
USE_COMPRESSION = 1


def _price_to_uint(price: float) -> int:
    """Convert float price to uint32 (price × 100000, rounded)."""
    return int(round(price * PRICE_MULTIPLIER))


def _uint_to_price(val: int) -> float:
    """Convert uint32 back to float price."""
    return val / PRICE_MULTIPLIER


def _timestamp_to_day_key(ts: int) -> int:
    """Convert Unix timestamp to day key (days since epoch)."""
    return ts // 86400


def _day_key_to_timestamp(day_key: int) -> int:
    """Convert day key back to Unix timestamp (start of day, 00:00 UTC)."""
    return day_key * 86400


def _timestamp_to_minute_of_day(ts: int) -> int:
    """Convert Unix timestamp to minute-of-day (0-1439)."""
    return (ts % 86400) // 60


# ── Single-symbol storage ────────────────────────────────────────────────────

class CompressedQuoteStorage:
    """
    Day-partitioned binary storage for forex quotes.

    Parameters
    ----------
    filepath : path to the .qhs4 file.
    price_mode : PRICE_CLOSE, PRICE_OHLC, or PRICE_OHLCV.
    compression : NO_COMPRESSION or USE_COMPRESSION.
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        price_mode: int = PRICE_OHLC,
        compression: int = USE_COMPRESSION if _HAS_ZSTD else NO_COMPRESSION,
    ):
        self.filepath = Path(filepath)
        self.price_mode = price_mode
        self.compression = compression
        self._values_per_bar = price_mode  # 1, 4, or 5
        self._block_size = MINUTES_IN_DAY * self._values_per_bar * UINT32_SIZE

        # In-memory index: day_key → (offset, size)
        self._day_index: dict[int, tuple[int, int]] = {}
        self._file_size = 0

        if self.filepath.exists() and self.filepath.stat().st_size > 0:
            self._load_header()

    def _load_header(self):
        """Load the day index from the file header."""
        with open(self.filepath, 'rb') as f:
            # Read header offset (first 4 bytes)
            header_offset = struct.unpack('<I', f.read(4))[0]
            if header_offset == 0 or header_offset >= self.filepath.stat().st_size:
                return  # empty/corrupt file

            f.seek(header_offset)
            num_days = struct.unpack('<I', f.read(4))[0]
            for _ in range(num_days):
                day_key = struct.unpack('<H', f.read(2))[0]
                block_size = struct.unpack('<I', f.read(4))[0]
                block_offset = struct.unpack('<I', f.read(4))[0]
                self._day_index[day_key] = (block_offset, block_size)

            self._file_size = self.filepath.stat().st_size
            log.debug(f"Loaded {num_days} days from {self.filepath.name}")

    def _compress(self, data: bytes) -> bytes:
        if self.compression == USE_COMPRESSION and _HAS_ZSTD:
            return zstd.compress(data)
        return data

    def _decompress(self, data: bytes) -> Optional[bytes]:
        if self.compression == USE_COMPRESSION and _HAS_ZSTD:
            try:
                return zstd.decompress(data)
            except Exception:
                log.warning("[CompressedQuoteStorage] Decompression failed — returning None (callers must handle)")
                return None
        return data

    # ── Write ────────────────────────────────────────────────────────────────

    def write_day(
        self,
        day_timestamp: int,
        candles: pd.DataFrame,
    ) -> None:
        """
        Write one day of candle data.

        Parameters
        ----------
        day_timestamp : Unix timestamp at start of the day (00:00 UTC).
        candles : DataFrame with columns matching the price_mode:
            PRICE_OHLC: open, high, low, close
            PRICE_OHLCV: open, high, low, close, volume
            PRICE_CLOSE: close
            Must have a DatetimeIndex or a 'time' column.
        """
        day_key = _timestamp_to_day_key(day_timestamp)

        # Build the binary block: 1440 minutes × values_per_bar × uint32
        block = np.zeros(MINUTES_IN_DAY * self._values_per_bar, dtype=np.uint32)

        for _, row in candles.iterrows():
            ts = row.name if isinstance(candles.index, pd.DatetimeIndex) else row.get('time')
            if isinstance(ts, pd.Timestamp):
                ts = int(ts.timestamp())
            minute = _timestamp_to_minute_of_day(ts)
            idx = minute * self._values_per_bar

            if self.price_mode == PRICE_CLOSE:
                block[minute] = _price_to_uint(row['close'])
            elif self.price_mode == PRICE_OHLC:
                block[idx] = _price_to_uint(row['open'])
                block[idx + 1] = _price_to_uint(row['high'])
                block[idx + 2] = _price_to_uint(row['low'])
                block[idx + 3] = _price_to_uint(row['close'])
            elif self.price_mode == PRICE_OHLCV:
                block[idx] = _price_to_uint(row['open'])
                block[idx + 1] = _price_to_uint(row['high'])
                block[idx + 2] = _price_to_uint(row['low'])
                block[idx + 3] = _price_to_uint(row['close'])
                block[idx + 4] = int(row.get('volume', 0))

        raw_data = block.tobytes()
        compressed = self._compress(raw_data)

        # Append to file
        with open(self.filepath, 'ab') as f:
            offset = f.tell() if f.tell() > 0 else 4  # skip header offset space
            if offset == 4 and not self._day_index:
                # First write — reserve 4 bytes for header offset
                f.seek(0)
                f.write(struct.pack('<I', 0))  # placeholder
                offset = 4
            f.seek(0, 2)  # seek to end
            offset = f.tell()
            f.write(compressed)
            self._day_index[day_key] = (offset, len(compressed))

        self._write_header()

    def _write_header(self):
        """Write the day index header at the end of the file."""
        with open(self.filepath, 'r+b') as f:
            f.seek(0, 2)  # end of file
            header_offset = f.tell()

            f.write(struct.pack('<I', len(self._day_index)))
            for day_key, (offset, size) in sorted(self._day_index.items()):
                f.write(struct.pack('<H', day_key))
                f.write(struct.pack('<I', size))
                f.write(struct.pack('<I', offset))

            # Write header offset at the beginning
            f.seek(0)
            f.write(struct.pack('<I', header_offset))

        self._file_size = self.filepath.stat().st_size

    # ── Read ─────────────────────────────────────────────────────────────────

    def get_candle(self, timestamp: int) -> Optional[dict]:
        """
        Get a single candle by Unix timestamp.
        Returns {"open", "high", "low", "close", "volume", "timestamp"} or None.
        """
        day_key = _timestamp_to_day_key(timestamp)
        if day_key not in self._day_index:
            return None

        offset, size = self._day_index[day_key]
        with open(self.filepath, 'rb') as f:
            f.seek(offset)
            data = f.read(size)
            data = self._decompress(data)
            if data is None:
                return None
            block = np.frombuffer(data, dtype=np.uint32)

        minute = _timestamp_to_minute_of_day(timestamp)
        idx = minute * self._values_per_bar

        if self.price_mode == PRICE_CLOSE:
            val = block[minute]
            if val == 0:
                return None
            return {"close": _uint_to_price(val), "timestamp": timestamp}

        if self.price_mode >= PRICE_OHLC:
            o, h, l, c = block[idx], block[idx+1], block[idx+2], block[idx+3]
            if o == 0 and c == 0:
                return None
            result = {
                "open": _uint_to_price(o),
                "high": _uint_to_price(h),
                "low": _uint_to_price(l),
                "close": _uint_to_price(c),
                "timestamp": timestamp,
            }
            if self.price_mode == PRICE_OHLCV:
                result["volume"] = int(block[idx + 4])
            return result

    def get_day(self, day_timestamp: int) -> Optional[pd.DataFrame]:
        """
        Get all candles for a specific day.
        Returns a DataFrame indexed by timestamp, or None if day not found.
        """
        day_key = _timestamp_to_day_key(day_timestamp)
        if day_key not in self._day_index:
            return None

        offset, size = self._day_index[day_key]
        with open(self.filepath, 'rb') as f:
            f.seek(offset)
            data = f.read(size)
            data = self._decompress(data)
            if data is None:
                return None
            block = np.frombuffer(data, dtype=np.uint32)

        day_start = _day_key_to_timestamp(day_key)
        records = []
        for minute in range(MINUTES_IN_DAY):
            idx = minute * self._values_per_bar
            ts = day_start + minute * 60

            if self.price_mode == PRICE_CLOSE:
                val = block[minute]
                if val == 0:
                    continue
                records.append({"timestamp": ts, "close": _uint_to_price(val)})
            elif self.price_mode >= PRICE_OHLC:
                o, h, l, c = block[idx], block[idx+1], block[idx+2], block[idx+3]
                if o == 0 and c == 0:
                    continue
                rec = {
                    "timestamp": ts,
                    "open": _uint_to_price(o),
                    "high": _uint_to_price(h),
                    "low": _uint_to_price(l),
                    "close": _uint_to_price(c),
                }
                if self.price_mode == PRICE_OHLCV:
                    rec["volume"] = int(block[idx + 4])
                records.append(rec)

        if not records:
            return None
        df = pd.DataFrame(records)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        df = df.set_index('datetime')
        return df

    def get_min_max_day_timestamp(self) -> tuple[int, int]:
        """Return (min_day_timestamp, max_day_timestamp) across all stored days."""
        if not self._day_index:
            return (0, 0)
        keys = sorted(self._day_index.keys())
        return (_day_key_to_timestamp(keys[0]), _day_key_to_timestamp(keys[-1]))

    def get_num_days(self) -> int:
        return len(self._day_index)

    def get_file_size(self) -> int:
        return self._file_size


# ── Multi-symbol storage ─────────────────────────────────────────────────────

class MultiSymbolStorage:
    """
    Manage multiple CompressedQuoteStorage instances for multi-pair backtesting.
    """

    def __init__(self, storage_dir: Union[str, Path], price_mode: int = PRICE_OHLC,
                 compression: int = USE_COMPRESSION if _HAS_ZSTD else NO_COMPRESSION):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.price_mode = price_mode
        self.compression = compression
        self._storages: dict[str, CompressedQuoteStorage] = {}

    def _get_storage(self, symbol: str) -> CompressedQuoteStorage:
        if symbol not in self._storages:
            ext = ".qhs4" if self.price_mode == PRICE_OHLC else ".qhs5" if self.price_mode == PRICE_OHLCV else ".qhs"
            filepath = self.storage_dir / f"{symbol}{ext}"
            self._storages[symbol] = CompressedQuoteStorage(filepath, self.price_mode, self.compression)
        return self._storages[symbol]

    def write_day(self, symbol: str, day_timestamp: int, candles: pd.DataFrame) -> None:
        self._get_storage(symbol).write_day(day_timestamp, candles)

    def get_candle(self, symbol: str, timestamp: int) -> Optional[dict]:
        return self._get_storage(symbol).get_candle(timestamp)

    def get_day(self, symbol: str, day_timestamp: int) -> Optional[pd.DataFrame]:
        return self._get_storage(symbol).get_day(day_timestamp)

    def get_symbols(self) -> list[str]:
        return list(self._storages.keys())

    def get_min_max_day_timestamp(self) -> tuple[int, int]:
        """Get the overall min/max day range across all symbols."""
        all_min, all_max = [], []
        for storage in self._storages.values():
            mn, mx = storage.get_min_max_day_timestamp()
            if mn > 0:
                all_min.append(mn)
                all_max.append(mx)
        if not all_min:
            return (0, 0)
        return (min(all_min), max(all_max))


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        filepath = Path(tmp) / "EURUSD.qhs4"
        storage = CompressedQuoteStorage(filepath, PRICE_OHLC, USE_COMPRESSION)

        # Generate synthetic data for one day
        day_ts = int(datetime(2024, 1, 15, tzinfo=timezone.utc).timestamp())
        rng = np.random.default_rng(42)
        minutes = 1440
        timestamps = [day_ts + m * 60 for m in range(minutes)]
        closes = 1.0850 + rng.normal(0, 0.0005, minutes)
        opens = closes + rng.normal(0, 0.0001, minutes)
        highs = np.maximum(opens, closes) + rng.uniform(0.0001, 0.0003, minutes)
        lows = np.minimum(opens, closes) - rng.uniform(0.0001, 0.0003, minutes)

        df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows, "close": closes,
        }, index=pd.to_datetime(timestamps, unit='s', utc=True))

        # Write
        storage.write_day(day_ts, df)
        file_size = storage.get_file_size()
        print(f"Written {len(df)} candles, file size: {file_size} bytes")
        print(f"Days stored: {storage.get_num_days()}")

        # Read back
        candle = storage.get_candle(day_ts + 12 * 60 * 60)  # noon
        print(f"\nCandle at noon: {candle}")

        day_df = storage.get_day(day_ts)
        print(f"Day DataFrame: {len(day_df)} rows")
        print(day_df.head())

        # Verify round-trip
        assert candle is not None
        original_close = closes[720]  # noon = minute 720
        assert abs(candle['close'] - original_close) < 1e-5, \
            f"close mismatch: {candle['close']} vs {original_close}"

        # Multi-symbol
        multi = MultiSymbolStorage(Path(tmp) / "multi", PRICE_OHLC, USE_COMPRESSION)
        multi.write_day("EURUSD", day_ts, df)
        multi.write_day("GBPUSD", day_ts, df * 1.2)
        c1 = multi.get_candle("EURUSD", day_ts + 3600)
        c2 = multi.get_candle("GBPUSD", day_ts + 3600)
        print(f"\nEURUSD: {c1['close']:.5f}")
        print(f"GBPUSD: {c2['close']:.5f}")
        print(f"Symbols: {multi.get_symbols()}")
        mn, mx = multi.get_min_max_day_timestamp()
        print(f"Date range: {mn} - {mx}")

        print("\nCompressed quote storage smoke test passed.")
