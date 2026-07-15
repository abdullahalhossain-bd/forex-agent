"""
backtest/mt5_bulk_fetcher.py — MT5 Auto-Pair Discovery + Bulk Data Fetcher
==========================================================================

Automatically discovers ALL available pairs from MT5 and fetches their
historical data across multiple timeframes. Designed for the comprehensive
backtest system.

Features:
  • Auto-discovers all currency pairs, metals, indices, crypto from MT5
  • Fetches data for every available timeframe
  • Caches fetched data to disk (parquet/CSV) for fast re-runs
  • Falls back to synthetic data if MT5 not available (for development)
  • Provides a unified iterator: (pair, timeframe, df) tuples

Usage:
    from backtest.mt5_bulk_fetcher import MT5BulkFetcher
    fetcher = MT5BulkFetcher()
    pairs = fetcher.discover_pairs()  # auto-detect from MT5
    for pair, tf, df in fetcher.iter_all_data(pairs, timeframes=['M15','H1','H4']):
        print(f"{pair} {tf}: {len(df)} candles")

MT5 Connection:
    Requires MetaTrader5 installed and terminal running on Windows.
    Set environment variable MT5_PATH if MT5 is installed in non-default location.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import numpy as np

from utils.logger import get_logger

log = get_logger("mt5_fetcher")

# ── MT5 availability check (graceful) ────────────────────────
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False


# ════════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════════

# All timeframes we want to test (Book 5 Chapter 12 recommends 3-TF trio)
ALL_TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"]

# Timeframe mapping (resolved lazily when MT5 is available)
TF_MAP: Dict[str, Any] = {}

def _resolve_tf_map():
    global TF_MAP
    if TF_MAP or not MT5_AVAILABLE:
        return
    TF_MAP = {
        "M1":  mt5.TIMEFRAME_M1,
        "M5":  mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1":  mt5.TIMEFRAME_H1,
        "H4":  mt5.TIMEFRAME_H4,
        "D1":  mt5.TIMEFRAME_D1,
        "W1":  mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1,
    }

# Default cache directory — portable across OS
# Uses a "data/backtest_cache" folder next to the forex_ai project
_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # forex_ai/
CACHE_DIR = _PROJECT_ROOT / "data" / "backtest_cache"

# Pair filtering — focus on liquid tradable instruments
DEFAULT_PAIR_FILTERS = {
    "forex_majors": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD",
                     "USDCAD", "NZDUSD"],
    "forex_crosses": ["EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "EURAUD",
                      "GBPAUD", "CADJPY", "CHFJPY", "NZDJPY", "EURCAD"],
    "metals": ["XAUUSD", "XAGUSD", "XPTUSD", "XPDUSD"],
    "indices": ["US30", "NAS100", "SPX500", "UK100", "GER40", "JPN225"],
    "crypto": ["BTCUSD", "ETHUSD", "LTCUSD", "XRPUSD"],
    "energy": ["USOIL", "UKOIL", "XNGUSD"],
}


# ════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ════════════════════════════════════════════════════════════════

@dataclass
class PairInfo:
    """Metadata about a discovered trading pair."""
    symbol: str
    description: str = ""
    category: str = "unknown"     # forex_majors/forex_crosses/metals/indices/crypto
    digits: int = 5
    point: float = 0.00001
    trade_contract_size: float = 100_000.0
    visible: bool = True
    available: bool = True


@dataclass
class FetchResult:
    """Result of fetching data for one (pair, timeframe)."""
    pair: str
    timeframe: str
    df: Optional[pd.DataFrame] = None
    n_candles: int = 0
    error: str = ""
    source: str = "mt5"          # "mt5" | "cache" | "synthetic"
    fetched_at: str = ""


# ════════════════════════════════════════════════════════════════
#  BULK FETCHER
# ════════════════════════════════════════════════════════════════

class MT5BulkFetcher:
    """
    Auto-discovers all tradable pairs from MT5 and fetches their
    historical data across multiple timeframes.

    Falls back to synthetic data generation when MT5 is unavailable
    (for development/testing on non-Windows systems).
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_candles_per_fetch: int = 100_000,
        mt5_path: Optional[str] = None,
    ):
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_candles = max_candles_per_fetch
        self.mt5_path = mt5_path or os.getenv("MT5_PATH", "")
        self._initialized = False
        self._pairs_cache: List[PairInfo] = []

    # ══════════════════════════════════════════════════════════
    #  MT5 INITIALIZATION
    # ══════════════════════════════════════════════════════════

    def _init_mt5(self) -> bool:
        """Initialize MT5 connection."""
        if not MT5_AVAILABLE:
            return False
        if self._initialized:
            return True
        try:
            if self.mt5_path:
                ok = mt5.initialize(self.mt5_path)
            else:
                ok = mt5.initialize()
            if not ok:
                log.error(f"[MT5] initialize failed: {mt5.last_error()}")
                return False
            _resolve_tf_map()
            self._initialized = True
            log.info("[MT5] Connected successfully")
            return True
        except Exception as e:
            log.error(f"[MT5] init error: {e}")
            return False

    # ══════════════════════════════════════════════════════════
    #  PAIR DISCOVERY
    # ══════════════════════════════════════════════════════════

    def discover_pairs(self, force_refresh: bool = False) -> List[PairInfo]:
        """
        Auto-discover all available trading pairs from MT5.

        If MT5 is unavailable, returns a curated default list from
        DEFAULT_PAIR_FILTERS (so the backtest can still run on synthetic
        data for development purposes).

        Returns:
            List of PairInfo objects for all available pairs.
        """
        if self._pairs_cache and not force_refresh:
            return self._pairs_cache

        if not self._init_mt5():
            log.warning("[MT5] MT5 unavailable — using default pair list")
            pairs = self._default_pairs()
            self._pairs_cache = pairs
            return pairs

        # Live MT5 discovery
        try:
            all_symbols = mt5.symbols_get()
            if all_symbols is None:
                log.error("[MT5] symbols_get() returned None")
                pairs = self._default_pairs()
                self._pairs_cache = pairs
                return pairs

            pairs = []
            for sym in all_symbols:
                # Filter: only tradable, visible symbols
                if not sym.visible:
                    continue
                # Skip non-trading symbols (e.g., session markers)
                if sym.trade_mode == 0:  # TRADE_MODE_DISABLED
                    continue

                category = self._categorize_symbol(sym.name)
                pairs.append(PairInfo(
                    symbol=sym.name,
                    description=sym.description or "",
                    category=category,
                    digits=sym.digits,
                    point=sym.point,
                    trade_contract_size=sym.trade_contract_size,
                    visible=sym.visible,
                    available=True,
                ))

            log.info(f"[MT5] Discovered {len(pairs)} tradable pairs")
            if not pairs:
                log.warning("[MT5] No pairs discovered — falling back to defaults")
                pairs = self._default_pairs()

            self._pairs_cache = pairs
            return pairs

        except Exception as e:
            log.error(f"[MT5] discover_pairs error: {e}")
            pairs = self._default_pairs()
            self._pairs_cache = pairs
            return pairs

    @staticmethod
    def _categorize_symbol(symbol: str) -> str:
        """Categorize a symbol by name (forex/metals/indices/crypto/etc)."""
        s = symbol.upper()
        # Metals
        if any(s.startswith(m) or s.endswith(m) for m in ["XAU", "XAG", "XPT", "XPD", "GOLD", "SILVER"]):
            return "metals"
        # Crypto
        if any(c in s for c in ["BTC", "ETH", "LTC", "XRP", "DOGE", "SOL"]):
            return "crypto"
        # Indices
        if any(s.startswith(idx) or s == idx for idx in
               ["US30", "NAS100", "SPX500", "UK100", "GER40", "JPN225",
                "FRA40", "AUS200", "HK50", "US500", "US100", "GER30"]):
            return "indices"
        # Energy
        if any(e in s for e in ["USOIL", "UKOIL", "XNG", "WTI", "BRENT"]):
            return "energy"
        # Forex — 6-character all-letter symbol
        if len(s) == 6 and s.isalpha():
            majors = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD",
                      "USDCAD", "NZDUSD"]
            if s in majors:
                return "forex_majors"
            return "forex_crosses"
        return "other"

    def _default_pairs(self) -> List[PairInfo]:
        """Return curated default pair list (used when MT5 unavailable)."""
        pairs = []
        for category, symbols in DEFAULT_PAIR_FILTERS.items():
            for sym in symbols:
                # Set typical digits/point per category
                if category == "forex_majors" or category == "forex_crosses":
                    digits = 5
                    point = 0.00001
                elif "JPY" in sym:
                    digits = 3
                    point = 0.001
                elif category == "metals":
                    digits = 2
                    point = 0.01
                else:
                    digits = 2
                    point = 0.01
                pairs.append(PairInfo(
                    symbol=sym, category=category,
                    digits=digits, point=point,
                ))
        return pairs

    def filter_pairs(
        self,
        pairs: List[PairInfo],
        categories: Optional[List[str]] = None,
        symbols: Optional[List[str]] = None,
    ) -> List[PairInfo]:
        """Filter the pair list by category and/or explicit symbol list."""
        result = pairs
        if categories:
            result = [p for p in result if p.category in categories]
        if symbols:
            sym_set = {s.upper() for s in symbols}
            result = [p for p in result if p.symbol.upper() in sym_set]
        return result

    # ══════════════════════════════════════════════════════════
    #  DATA FETCHING
    # ══════════════════════════════════════════════════════════

    def fetch(
        self,
        pair: str,
        timeframe: str,
        n_candles: Optional[int] = None,
        use_cache: bool = True,
    ) -> FetchResult:
        """
        Fetch OHLCV data for one (pair, timeframe).

        Args:
            pair       : symbol name (e.g., "EURUSD")
            timeframe  : "M1" / "M5" / "M15" / "H1" / "H4" / "D1" / etc.
            n_candles  : how many candles to fetch (default: self.max_candles)
            use_cache  : if True, try loading from disk cache first

        Returns:
            FetchResult
        """
        n_candles = n_candles or self.max_candles
        cache_path = self._cache_path(pair, timeframe, n_candles)

        # Try cache first
        if use_cache and cache_path.exists():
            try:
                df = pd.read_parquet(cache_path) if cache_path.suffix == ".parquet" \
                    else pd.read_csv(cache_path, parse_dates=["time"], index_col="time")
                if len(df) > 0:
                    log.debug(f"[Cache] {pair} {timeframe}: {len(df)} candles")
                    return FetchResult(
                        pair=pair, timeframe=timeframe, df=df,
                        n_candles=len(df), source="cache",
                        fetched_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                    )
            except Exception as e:
                log.warning(f"[Cache] load failed for {pair} {timeframe}: {e}")

        # Try MT5
        if self._init_mt5():
            df = self._fetch_mt5(pair, timeframe, n_candles)
            if df is not None and len(df) > 0:
                # Save to cache
                try:
                    df.to_parquet(cache_path.with_suffix(".parquet"))
                except Exception as e:
                    df.to_csv(cache_path.with_suffix(".csv"))
                return FetchResult(
                    pair=pair, timeframe=timeframe, df=df,
                    n_candles=len(df), source="mt5",
                    fetched_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                )

        # Fallback: synthetic data
        log.info(f"[Synthetic] Generating synthetic data for {pair} {timeframe}")
        df = self._generate_synthetic(pair, timeframe, n_candles)
        return FetchResult(
            pair=pair, timeframe=timeframe, df=df,
            n_candles=len(df), source="synthetic",
            fetched_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

    def _fetch_mt5(self, pair: str, timeframe: str, n_candles: int) -> Optional[pd.DataFrame]:
        """Fetch data from MT5."""
        if not MT5_AVAILABLE or timeframe not in TF_MAP:
            return None
        try:
            rates = mt5.copy_rates_from_pos(pair, TF_MAP[timeframe], 0, n_candles)
            if rates is None or len(rates) == 0:
                return None
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df = df.set_index("time")
            # Rename columns to lowercase
            df = df.rename(columns={
                "open": "open", "high": "high", "low": "low",
                "close": "close", "tick_volume": "volume",
            })
            # Keep only OHLCV
            cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            df = df[cols]
            return df
        except Exception as e:
            log.error(f"[MT5] fetch {pair} {timeframe} failed: {e}")
            return None

    def _generate_synthetic(self, pair: str, timeframe: str, n_candles: int) -> pd.DataFrame:
        """
        Generate realistic synthetic OHLCV data (for dev/testing when MT5 unavailable).
        Uses geometric brownian motion with regime switches to simulate trends/ranges.
        """
        # Round-21 audit fix: use hashlib.md5 instead of hash().
        # hash() of str is randomized per-process (PYTHONHASHSEED=random),
        # so synthetic data differed on every run. hashlib.md5 is stable.
        import hashlib
        seed = int(hashlib.md5((pair + timeframe).encode()).hexdigest()[:8], 16)
        np.random.seed(seed)

        # Base price per category
        if "JPY" in pair:
            base = 110.0 if "USDJPY" in pair else 130.0
            pip = 0.01
        elif "XAU" in pair:
            base = 2000.0
            pip = 0.1
        elif "XAG" in pair:
            base = 25.0
            pip = 0.01
        elif any(idx in pair for idx in ["US30", "NAS100", "SPX500"]):
            base = 18000.0 if "US30" in pair else (20000.0 if "NAS" in pair else 4500.0)
            pip = 1.0
        elif "BTC" in pair:
            base = 65000.0
            pip = 1.0
        elif "ETH" in pair:
            base = 3500.0
            pip = 0.1
        else:
            base = 1.1000  # typical forex
            pip = 0.0001

        # Volatility per timeframe
        vol_map = {"M1": 2, "M5": 5, "M15": 10, "M30": 15, "H1": 25,
                   "H4": 60, "D1": 120, "W1": 250, "MN1": 500}
        vol = vol_map.get(timeframe, 25) * pip

        # Timeframe in minutes
        tf_minutes = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60,
                      "H4": 240, "D1": 1440, "W1": 10080, "MN1": 43200}
        minutes = tf_minutes.get(timeframe, 60)

        # Generate with regime switches (trending vs ranging)
        dates = pd.date_range("2023-01-01", periods=n_candles, freq=f"{minutes}min")

        # Returns: mix of trend + noise
        returns = np.zeros(n_candles)
        regime = 1  # 1=trend up, -1=trend down, 0=range
        regime_len = 0
        trend_strength = 0.0
        for i in range(n_candles):
            if regime_len <= 0:
                regime = np.random.choice([1, -1, 0], p=[0.4, 0.4, 0.2])
                regime_len = np.random.randint(50, 200)
                trend_strength = np.random.uniform(0.2, 0.8) * vol
            regime_len -= 1
            noise = np.random.randn() * vol
            returns[i] = regime * trend_strength + noise

        # Build price series
        prices = base * np.exp(np.cumsum(returns))

        # Build OHLC from close prices
        df = pd.DataFrame(index=dates)
        df["close"] = prices
        df["open"] = df["close"].shift(1).fillna(base)
        # High/low: add random intrabar range
        intrabar = np.abs(np.random.randn(n_candles)) * vol * 0.5
        df["high"] = df[["open", "close"]].max(axis=1) + intrabar
        df["low"] = df[["open", "close"]].min(axis=1) - intrabar
        df["volume"] = np.random.randint(100, 10000, n_candles)

        return df

    def _cache_path(self, pair: str, timeframe: str, n_candles: int) -> Path:
        """Get the cache file path for a (pair, timeframe, n_candles).

        Round-21 audit fix: previously always returned .parquet path, but
        the save fallback (L358-361) writes .csv when parquet fails.
        The cache-load branch only checked the parquet path, so a saved
        CSV was invisible on subsequent fetches — MT5 was re-hit every
        time, defeating the cache.

        Now: check for both .parquet and .csv, return whichever exists.
        Prefers .parquet if both exist (faster to load).
        """
        parquet_path = self.cache_dir / f"{pair}_{timeframe}_{n_candles}.parquet"
        if parquet_path.exists():
            return parquet_path
        csv_path = self.cache_dir / f"{pair}_{timeframe}_{n_candles}.csv"
        if csv_path.exists():
            return csv_path
        # Default: return parquet path (will be created on save)
        return parquet_path

    # ══════════════════════════════════════════════════════════
    #  BULK ITERATOR
    # ══════════════════════════════════════════════════════════

    def iter_all_data(
        self,
        pairs: Optional[List[PairInfo]] = None,
        timeframes: Optional[List[str]] = None,
        n_candles: Optional[int] = None,
        use_cache: bool = True,
        progress: bool = True,
    ) -> Iterable[Tuple[PairInfo, str, pd.DataFrame]]:
        """
        Iterate over all (pair, timeframe) combinations.

        Yields:
            (PairInfo, timeframe_str, DataFrame) tuples
        """
        pairs = pairs or self.discover_pairs()
        timeframes = timeframes or ["M15", "H1", "H4", "D1"]  # sensible default

        total = len(pairs) * len(timeframes)
        done = 0

        for pair in pairs:
            for tf in timeframes:
                done += 1
                if progress:
                    log.info(f"[{done}/{total}] Fetching {pair.symbol} {tf}...")
                result = self.fetch(pair.symbol, tf, n_candles=n_candles,
                                    use_cache=use_cache)
                if result.df is not None and len(result.df) > 50:
                    yield pair, tf, result.df
                else:
                    log.warning(f"[{done}/{total}] {pair.symbol} {tf}: insufficient data")

    # ══════════════════════════════════════════════════════════
    #  CONNECTION CLEANUP
    # ══════════════════════════════════════════════════════════

    def shutdown(self):
        """Close MT5 connection."""
        if self._initialized and MT5_AVAILABLE:
            try:
                mt5.shutdown()
                self._initialized = False
                log.info("[MT5] Disconnected")
            except Exception as e:
                log.warning(f"Suppressed exception at line 524: {e}")
                pass

    def __del__(self):
        self.shutdown()


# ════════════════════════════════════════════════════════════════
#  CLI ENTRY
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 64)
    print("  MT5 Bulk Fetcher — Pair Discovery + Data Fetch")
    print("=" * 64)
    print(f"\n  MT5 available: {MT5_AVAILABLE}")

    fetcher = MT5BulkFetcher()
    pairs = fetcher.discover_pairs()

    print(f"\n  Discovered {len(pairs)} pairs:")

    # Group by category
    by_cat: Dict[str, List[PairInfo]] = {}
    for p in pairs:
        by_cat.setdefault(p.category, []).append(p)

    for cat, items in sorted(by_cat.items()):
        print(f"\n  {cat} ({len(items)} pairs):")
        for p in items[:5]:
            print(f"    {p.symbol:<10} digits={p.digits} point={p.point}")
        if len(items) > 5:
            print(f"    ... and {len(items)-5} more")

    # Test fetch one pair
    if pairs:
        test_pair = pairs[0]
        print(f"\n  Test fetch: {test_pair.symbol} M15 (500 candles)")
        result = fetcher.fetch(test_pair.symbol, "M15", n_candles=500)
        print(f"  Result: {result.n_candles} candles from {result.source}")
        if result.df is not None:
            print(f"  Range: {result.df.index[0]} → {result.df.index[-1]}")
            print(f"  Last close: {result.df['close'].iloc[-1]:.5f}")

    fetcher.shutdown()
    print("\n" + "=" * 64)
    print("  Done.")
    print("=" * 64)
