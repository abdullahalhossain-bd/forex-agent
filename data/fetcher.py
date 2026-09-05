# data/fetcher.py
# ============================================================
# Multi-Source Data Fetcher (MT5-first)
# Primary Source: MetaTrader5 (native forex data)
# Fallback Source: TradingView via tvdatafeed
# ============================================================

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional
from utils.logger import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────
# BROKER SYMBOL SUFFIX (case-sensitive!) — e.g. Exness "m" accounts
# ─────────────────────────────────────────────────────────────
# 2026-08-20 fix: trader.log showed every MT5 fetch failing —
# "AUDJPYM", "EURUSDM", etc. — because MT5 symbol names are
# case-sensitive and this broker's real symbols are lowercase-suffixed
# ("AUDJPYm", "EURUSDm"). The old _normalize_symbol() called
# .upper() on the RAW symbol (suffix included), turning a valid
# broker symbol into one that doesn't exist. Same env var as
# config.py's BROKER_SYMBOL_SUFFIX, so both stay in sync.
_MT5_BROKER_SUFFIX = os.getenv("MT5_SYMBOL_SUFFIX", "m")


def _strip_mt5_suffix(symbol: str) -> str:
    """Remove a trailing broker suffix (case-insensitive match) to get
    the bare pair name, e.g. "EURUSDm" -> "EURUSD". No-op if the
    suffix isn't configured or isn't present."""
    s = str(symbol).strip()
    suffix = _MT5_BROKER_SUFFIX
    if suffix and len(s) > len(suffix) and s[-len(suffix):].lower() == suffix.lower():
        return s[: -len(suffix)]
    return s


def _to_mt5_symbol(symbol: str) -> str:
    """Append the broker suffix in the EXACT case the broker expects
    (e.g. "m", not "M"). Only call this right before an actual MT5
    call (symbol_select, copy_rates_from_pos, symbol_info_tick, ...).
    Idempotent — safe to call on a symbol that already has the
    correctly-cased suffix."""
    s = _strip_mt5_suffix(symbol)
    suffix = _MT5_BROKER_SUFFIX
    return f"{s}{suffix}" if suffix else s


# ─────────────────────────────────────────────────────────────
# LOG SPAM GUARD — repeated per-symbol config errors
# ─────────────────────────────────────────────────────────────
# Bugfix: "[Finnhub] API key not set" (and same for AlphaVantage/
# TwelveData/Polygon) was logged at ERROR level on every symbol, every
# cycle — e.g. 7x per cycle for a 7-symbol watchlist — even though it's
# not a per-symbol condition, it's a one-time missing-config fact that
# never changes until the operator sets the env var. Log it loudly once,
# then drop to DEBUG so it doesn't drown out real per-symbol errors in
# the log.
import threading as _threading
_missing_key_lock = _threading.Lock()
_missing_key_warned: set = set()


def _log_missing_api_key(provider: str) -> None:
    """Log 'API key not set' at ERROR the first time for this provider
    in this process, DEBUG every time after."""
    with _missing_key_lock:
        first_time = provider not in _missing_key_warned
        _missing_key_warned.add(provider)
    if first_time:
        log.error(
            f"[{provider}] API key not set — set the corresponding "
            f"env var (see .env.example) to enable this fallback source. "
            f"(further occurrences this run are logged at DEBUG level)"
        )
    else:
        log.debug(f"[{provider}] API key not set (already warned this run)")


# MetaTrader5 package is Windows-only. On Linux/Mac the import
# would crash the whole project at module-load time. We guard it
# here so DataFetcher still imports cleanly and falls back to
# tvdatafeed / "unavailable" mode when MT5 isn't installed.
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False
    log.info(
        "MetaTrader5 package not installed — DataFetcher will use "
        "tvdatafeed as fallback. Install MetaTrader5 on Windows with "
        "MetaTrader 5 terminal running to enable MT5 data source."
    )

# ─────────────────────────────────────────────────────────────
# UNAVAILABLE SYMBOL TRACKING
# ─────────────────────────────────────────────────────────────
# Symbols that the broker doesn't support are remembered here
# so downstream code can skip them silently instead of
# triggering recovery pauses every cycle.
_UNAVAILABLE_SYMBOLS: set = set()

# Per-symbol consecutive fetch failure counter.
# After FETCH_FAIL_THRESHOLD consecutive failures, the symbol
# is auto-marked as unavailable to stop triggering recovery pauses.
_FETCH_FAILURE_COUNTS: dict = {}
FETCH_FAIL_THRESHOLD = 3

# ─────────────────────────────────────────────────────────────
# MT5 COLD-START SYNC WINDOW
# ─────────────────────────────────────────────────────────────
# Right after mt5.initialize()/login, the terminal must sync
# history from the broker.  Real-world observation (trader.log):
#   100% of fetches failed with KeyError: 'time' from 20:12:08
#   to 20:16:10, then 100% succeeded from 20:16:43 onward.
# That's a ~242-second cold-start window.  Failures during this
# window are TRANSIENT and must NOT count toward FETCH_FAIL_THRESHOLD,
# otherwise every symbol gets permanently marked unavailable on
# the very first cycle after startup.
#
# The window is measured from the first successful mt5 connection.
# After the window expires, failures are treated as real/genuine.
_MT5_COLD_START_DEADLINE_SEC = 300  # 5 minutes — covers the observed ~4 min + safety margin
_MT5_FIRST_CONNECTED_AT: Optional[float] = None  # epoch seconds, set once


def set_mt5_first_connected_at(epoch_seconds: float) -> None:
    """Record the epoch time of the first MT5 connection.

    Called by MT5Connection._try_connect on first successful connect.
    Write-once: subsequent reconnects do NOT reset the deadline,
    because the cold-start sync only happens after the VERY FIRST
    connection to the terminal (not after a mid-session reconnect
    which reuses the already-synced terminal state).
    """
    global _MT5_FIRST_CONNECTED_AT
    if _MT5_FIRST_CONNECTED_AT is None:
        _MT5_FIRST_CONNECTED_AT = epoch_seconds


def _is_within_cold_start_window() -> bool:
    """Return True if we are still inside the MT5 cold-start sync window.

    During this window, fetch failures caused by missing 'time' column
    are considered TRANSIENT and should NOT increment the per-symbol
    failure counter — they are the expected symptom of history sync.
    """
    if _MT5_FIRST_CONNECTED_AT is None:
        return False
    return (time.time() - _MT5_FIRST_CONNECTED_AT) < _MT5_COLD_START_DEADLINE_SEC


def mark_symbol_unavailable(symbol: str) -> None:
    """Record that a symbol is not available on the current broker."""
    _UNAVAILABLE_SYMBOLS.add(symbol.upper())


def is_symbol_unavailable(symbol: str) -> bool:
    """Check if a symbol has been confirmed unavailable on the broker."""
    return symbol.upper() in _UNAVAILABLE_SYMBOLS


def get_unavailable_symbols() -> set:
    """Return the full set of unavailable symbols (for diagnostics)."""
    return set(_UNAVAILABLE_SYMBOLS)


def record_fetch_failure(symbol: str, *, cold_start: bool = False) -> bool:
    """Record a fetch failure for a symbol.

    Args:
        symbol: The symbol that failed.
        cold_start: If True, the failure happened during MT5's cold-start
            sync window (missing 'time' column).  Such failures are
            NOT counted against FETCH_FAIL_THRESHOLD because they are
            transient — the data will appear once sync finishes.

    Returns:
        True if the symbol has now exceeded the failure threshold and
        should be marked as unavailable.
    """
    if cold_start:
        log.debug(
            f"[DataFetcher] {symbol}: failure during MT5 cold-start sync "
            f"window — NOT counting against unavailable threshold"
        )
        return False
    key = symbol.upper()
    count = _FETCH_FAILURE_COUNTS.get(key, 0) + 1
    _FETCH_FAILURE_COUNTS[key] = count
    if count >= FETCH_FAIL_THRESHOLD:
        return True
    return False


def record_fetch_success(symbol: str) -> None:
    """Reset the failure counter for a symbol after a successful fetch."""
    key = symbol.upper()
    _FETCH_FAILURE_COUNTS.pop(key, None)
    # If a previously-unavailable symbol starts working again, unmark it
    # so it gets re-tried. This handles broker symbol list changes.
    _UNAVAILABLE_SYMBOLS.discard(key)


# ─────────────────────────────────────────────────────────────
# MT5 TIMEFRAME MAPPING
# ─────────────────────────────────────────────────────────────
# Built lazily — only resolved when MT5 is available, so importing
# this module on Linux/Mac (where MetaTrader5 is unavailable) doesn't
# raise AttributeError on `mt5.TIMEFRAME_*`.
TIMEFRAME_MAP = {}

def _build_timeframe_map():
    """Populate TIMEFRAME_MAP from live mt5 constants (called once, lazily)."""
    if not MT5_AVAILABLE or TIMEFRAME_MAP:
        return
    TIMEFRAME_MAP.update({
        "M1":   mt5.TIMEFRAME_M1,       # 1 minute (MT5 User Guide Page 18)
        "M5":   mt5.TIMEFRAME_M5,       # 5 minutes
        "M15":  mt5.TIMEFRAME_M15,      # 15 minutes
        "M30":  mt5.TIMEFRAME_M30,      # 30 minutes
        "H1":   mt5.TIMEFRAME_H1,       # 1 hour
        "H4":   mt5.TIMEFRAME_H4,       # 4 hours
        "D1":   mt5.TIMEFRAME_D1,       # 1 day
        "W1":   mt5.TIMEFRAME_W1,       # 1 week
        "MN1":  mt5.TIMEFRAME_MN1,      # 1 month
        # Aliases for backward compatibility
        "1m":   mt5.TIMEFRAME_M1,
        "5m":   mt5.TIMEFRAME_M5,
        "15m":  mt5.TIMEFRAME_M15,
        "30m":  mt5.TIMEFRAME_M30,
        "1h":   mt5.TIMEFRAME_H1,
        "4h":   mt5.TIMEFRAME_H4,
        "1d":   mt5.TIMEFRAME_D1,
    })

# Populate immediately if MT5 is available; otherwise TIMEFRAME_MAP
# stays empty and the fetcher will report "no data source available".
_build_timeframe_map()

# ─────────────────────────────────────────────────────────────
# CANONICAL TIMEFRAME REGISTRY (source-independent)
# ─────────────────────────────────────────────────────────────
# This is the single source of truth for "what timeframes does this
# project support" and is used by _normalize_timeframe() regardless
# of which data source is active (MT5, yfinance, Alpha Vantage, ...).
# Keys are the canonical internal representation used everywhere
# downstream (e.g. by the Decision Layer for multi-timeframe logic).
CANONICAL_TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1")

# Symbol normalization — internal style to MT5 style
# MT5 symbols are typically EURUSD, GBPUSD (no =X suffix)
SYMBOL_MAP = {
    # Forex majors
    "EURUSD":      "EURUSD",
    "GBPUSD":      "GBPUSD",
    "USDJPY":      "USDJPY",
    "AUDUSD":      "AUDUSD",
    "USDCHF":      "USDCHF",
    "USDCAD":      "USDCAD",
    "NZDUSD":      "NZDUSD",
    # Forex crosses
    "EURGBP":      "EURGBP",
    "EURJPY":      "EURJPY",
    "EURCHF":      "EURCHF",
    "EURAUD":      "EURAUD",
    "EURCAD":      "EURCAD",
    "EURNZD":      "EURNZD",
    "GBPJPY":      "GBPJPY",
    "GBPCHF":      "GBPCHF",
    "GBPAUD":      "GBPAUD",
    "GBPCAD":      "GBPCAD",
    "GBPNZD":      "GBPNZD",
    "AUDJPY":      "AUDJPY",
    "AUDCHF":      "AUDCHF",
    "AUDCAD":      "AUDCAD",
    "AUDNZD":      "AUDNZD",
    "NZDJPY":      "NZDJPY",
    "NZDCHF":      "NZDCHF",
    "NZDCAD":      "NZDCAD",
    "CADJPY":      "CADJPY",
    "CADCHF":      "CADCHF",
    "CHFJPY":      "CHFJPY",
    # Metals
    "XAUUSD":      "XAUUSD",
    "XAGUSD":      "XAGUSD",
    # Legacy/alternative formats
    "EUR/USD":     "EURUSD",
    "GBP/USD":     "GBPUSD",
    "USD/JPY":     "USDJPY",
    "AUD/USD":     "AUDUSD",
    "USD/CHF":     "USDCHF",
    "USD/CAD":     "USDCAD",
    "EUR/USDT":    "EURUSD",
    "GBP/USDT":    "GBPUSD",
    "EURUSD=X":    "EURUSD",
    "GBPUSD=X":    "GBPUSD",
    "USDJPY=X":    "USDJPY",
}


def _is_forex_market_expected_open(now_utc: "pd.Timestamp") -> bool:
    """
    Lightweight forex market-hours check — no external deps, safe to call
    from the low-level fetch path.

    Standard forex convention: the market is closed from Friday ~21:00 UTC
    (NY close) through Sunday ~21:00 UTC (Sydney open). Outside that window
    it's a normal trading day. This intentionally does NOT try to model
    bank holidays (that's a much fuzzier, broker-dependent calendar) — it
    only answers "is this an ordinary weekend closure", which is exactly
    the gap that was getting confused with genuine stale-data conditions.

    Returns:
        True  -> market SHOULD be open right now (a stale bar here is a
                 real data problem).
        False -> market is in its expected weekend closure (a stale bar
                 here is normal and not a feed issue).
    """
    try:
        weekday = now_utc.weekday()  # Monday=0 .. Sunday=6
        hour = now_utc.hour
        if weekday == 5:  # Saturday — always closed
            return False
        if weekday == 6 and hour < 21:  # Sunday before ~21:00 UTC open
            return False
        if weekday == 4 and hour >= 21:  # Friday after ~21:00 UTC close
            return False
        return True
    except Exception as e:
        # If anything about `now_utc` is unexpected, don't assert either
        # way — default to "market open" so genuine staleness still gets
        # flagged rather than silently waved through as a weekend gap.
        log.debug(f"[DataFetcher] is_forex_market_open() check failed on now_utc={now_utc!r}: {type(e).__name__}: {e} — defaulting to OPEN")
        return True


class DataFetcher:
    """
    MT5-first data fetcher.
    
    Uses MetaTrader5 to fetch OHLCV data for forex/metals.
    Fallback to tvdatafeed if MT5 is unavailable.
    """

    def __init__(self, mt5_conn=None):
        """
        Args:
            mt5_conn: Optional, already-connected broker.mt5_connection.MT5Connection
                instance (mirrors execution_router.ExecutionRouter's `mt5_conn`
                injection pattern). When provided, _fetch_mt5() routes every MT5
                call through this shared, locked connection instead of calling
                mt5.initialize()/copy_rates_from_pos() directly against the
                global MetaTrader5 module.

                P1 fix (institutional audit §3.1): previously this class called
                mt5.initialize() directly in _fetch_mt5(), racing
                MT5Connection's own initialize()/shutdown() cycle in
                mt5_connection.py and execution_router.py — the exact class of
                bug the "Day 90+ hotfix" fixed everywhere except here. A
                concurrent, unlocked mt5.initialize() from this fetcher could
                invalidate an authenticated session mid-order in
                execution_router.py. Session ownership now belongs to exactly
                one MT5Connection instance, shared across fetch + execution.

                If not injected, DataFetcher builds its own MT5Connection from
                config (backward compatible with pre-fix callers that don't
                inject one) instead of touching the mt5 module directly.
        """
        self.source = self._detect_source()
        self._mt5_conn = None
        self._owns_mt5_conn = False
        # Dynamic broker UTC-offset cache (see _get_broker_utc_offset_hours).
        # Re-checked periodically instead of once at startup so a broker's
        # DST flip mid-session (most brokers change GMT+2 <-> GMT+3 on a
        # different date than the local PC's DST, since MT5 servers often
        # follow US or EU DST calendars) is picked up without a restart.
        self._broker_offset_cache: Optional[float] = None
        self._broker_offset_cache_at: float = 0.0
        self._BROKER_OFFSET_CACHE_TTL_SEC = 1800  # 30 min
        # FIX: per-symbol last-log timestamp for IPC-connection (-10004)
        # errors. Used to debounce the spam we saw in trader.log (200+
        # identical errors per second when MT5 terminal briefly restarted).
        self._last_ipc_log_ts: dict = {}
        if self.source == "mt5":
            self._init_mt5_connection(mt5_conn)
        log.info(f"[OK] DataFetcher initialized | source: {self.source}")

    def _init_mt5_connection(self, mt5_conn) -> None:
        """Wire up the shared MT5Connection (injected or self-built)."""
        try:
            from broker.mt5_connection import get_mt5_connection
        except Exception as e:
            log.warning(
                f"[DataFetcher] broker.mt5_connection unavailable ({e}) — "
                f"_fetch_mt5 will not be able to fetch until this is fixed"
            )
            return

        if mt5_conn is not None:
            self._mt5_conn = mt5_conn
            if not getattr(self._mt5_conn, "connected", False):
                self._mt5_conn.connect()
            log.info("[DataFetcher] Using shared/injected MT5Connection")
            return

        try:
            from config import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH
        except Exception as e:
            log.warning(
                f"[DataFetcher] No MT5 credentials in config ({e}) — "
                f"cannot build a fallback MT5Connection; inject one via "
                f"get_data_fetcher(mt5_conn=...) instead"
            )
            return

        # Bug fix: was `MT5Connection(...)` — built its own independent
        # session (separate mt5.initialize()+login()) instead of reusing
        # whatever the rest of the process already has open, which is what
        # produced duplicate connection banners in the logs. Route through
        # the singleton factory instead so it reuses (or creates once and
        # shares) the connection for this (login, server).
        self._mt5_conn = get_mt5_connection(
            login=MT5_LOGIN, password=MT5_PASSWORD,
            server=MT5_SERVER, path=MT5_PATH or None,
            auto_connect=True,
        )
        if self._mt5_conn.connected:
            self._owns_mt5_conn = True
        else:
            log.error("[DataFetcher] Fallback MT5Connection failed to connect")

    def _detect_source(self):
        """Detect available data source.

        Day 81+ architecture change: MT5 is now the SINGLE SOURCE OF TRUTH.
        TradingView (tvdatafeed) fallback is intentionally disabled because
        trading on data from source A while executing on broker B causes
        data/execution mismatch (different spreads, tick timing, liquidity).

        If MT5 is unavailable, the fetcher returns "unavailable" and the
        trading cycle aborts — this is by design.  Do NOT re-enable the
        TradingView fallback without a corresponding execution-side
        fallback (i.e. paper trading).

        Day 90 addition: yfinance fallback for Linux VPS / dev environments
        where MT5 is unavailable. Yahoo Finance exposes forex pairs as
        EURUSD=X, GBPUSD=X etc. and is free + keyless. Use ONLY for
        demo / paper trading — production should still use MT5 for
        data/execution consistency.

        Day 103 fix (institutional review): priority order corrected.
        Previously yfinance (free, delayed, keyless) was checked BEFORE
        any paid/professional API key (Alpha Vantage, Polygon, Finnhub,
        Twelve Data) unless PREFERRED_DATA_SOURCE was explicitly set.
        That meant an operator who configured a paid key to get better
        data would silently get Yahoo's delayed data instead, with no
        warning. Paid providers are now checked first; yfinance is the
        last-resort fallback for demo/dev environments with no keys
        configured at all.
        """
        if MT5_AVAILABLE:
            try:
                import MetaTrader5 as _mt5_check
                # Day 102: avoid init/shutdown cycle that kills shared connection.
                # If MT5 package is importable, assume mt5 source is available.
                # Actual connection health is verified at fetch time.
                return "mt5"
            except Exception as e:
                log.warning(
                    f"[DataFetcher] MT5_AVAILABLE=True but `import MetaTrader5` "
                    f"failed at source-selection time: {type(e).__name__}: {e} "
                    f"— falling through to API providers"
                )

        # TradingView fallback DISABLED — see docstring above.
        # try:
        #     from tvdatafeed import TvDatafeed  # noqa: F401
        #     return "tvdatafeed"
        # except ImportError:
        #     log.debug("tvdatafeed not available")

        # ── Day 92 — Preferred source override (highest priority) ──
        # If the operator explicitly set PREFERRED_DATA_SOURCE in .env,
        # use it without falling through to auto-detect.
        preferred = os.getenv("PREFERRED_DATA_SOURCE", "").lower().strip()
        candidates = [
            ("alpha_vantage", "ALPHA_VANTAGE_API_KEY"),
            ("polygon",       "POLYGON_API_KEY"),
            ("finnhub",       "FINNHUB_API_KEY"),
            ("twelve_data",   "TWELVE_DATA_API_KEY"),
        ]
        if preferred:
            if preferred == "yfinance":
                try:
                    import yfinance  # noqa: F401
                    log.info("[DataFetcher] yfinance selected (PREFERRED_DATA_SOURCE)")
                    return "yfinance"
                except ImportError:
                    log.warning(
                        "[DataFetcher] PREFERRED_DATA_SOURCE=yfinance but the "
                        "package is not installed — falling through to auto-detect"
                    )
            for name, env in candidates:
                if name == preferred and os.getenv(env, "").strip():
                    log.info(f"[DataFetcher] {name} selected (PREFERRED_DATA_SOURCE)")
                    return name
            if preferred != "yfinance":
                log.warning(
                    f"[DataFetcher] PREFERRED_DATA_SOURCE={preferred!r} but its API "
                    f"key is missing — falling through to auto-detect"
                )

        # ── Day 103 — Paid/professional API keys take priority over yfinance ──
        # A configured API key is an explicit signal of operator intent;
        # yfinance (free, delayed, keyless) should only be used when
        # nothing else is configured.
        for name, env in candidates:
            if os.getenv(env, "").strip():
                log.info(f"[DataFetcher] {name} selected (key found in env)")
                return name

        # ── Day 90 — yfinance fallback (Linux VPS / demo only, last resort) ──
        try:
            import yfinance  # noqa: F401
            log.info(
                "[DataFetcher] yfinance available — using as demo data source "
                "(no paid API key configured). Set SIMULATION_MODE=true for "
                "execution-side matching."
            )
            return "yfinance"
        except ImportError:
            pass

        log.warning(
            "[DataFetcher] MT5 unavailable and TradingView fallback is disabled. "
            "Install MetaTrader5 on Windows with MT5 terminal running to enable data."
        )
        return "unavailable"

    def fetch_ohlcv_mt5(self, symbol="EURUSD", timeframe="M15", limit=300):
        """Fetch closed historical bars from MT5 without fallback sources."""
        if self.source != "mt5":
            raise RuntimeError(f"MT5 source unavailable; selected source is {self.source!r}")
        normalized_symbol = self._normalize_symbol(symbol)
        normalized_timeframe = self._normalize_timeframe(timeframe)
        if normalized_timeframe is None:
            raise ValueError(f"Unrecognized timeframe: {timeframe}")
        result = self._fetch_mt5(normalized_symbol, normalized_timeframe, int(limit))
        if result is None or result.empty:
            raise RuntimeError(
                f"MT5 returned no closed bars for {normalized_symbol} {normalized_timeframe}"
            )
        return result

    def fetch_ohlcv(self, symbol="EURUSD", timeframe="M15", limit=300, periods=None):
        """
        Fetch OHLCV data from the available source.
        
        Args:
            symbol (str):     Trading pair (e.g., "EURUSD", "EUR/USD", "EURUSD=X")
            timeframe (str):   Timeframe (e.g., "M5", "M15", "H1", "15m", "1h")
            limit (int):      Number of candles to fetch (default 300)
            periods (int):    Alias for limit (backward compatibility)
        
        Returns:
            pd.DataFrame: OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']
                         and datetime index. Returns None on failure or on an
                         unrecognized timeframe (never silently substitutes a
                         different timeframe than the one requested).
        """
        # Backward compatibility: periods → limit
        if periods is not None:
            limit = periods

        symbol = self._normalize_symbol(symbol)
        norm_timeframe = self._normalize_timeframe(timeframe)
        if norm_timeframe is None:
            log.error(
                f"[DataFetcher] Unrecognized timeframe '{timeframe}' — refusing to "
                f"fetch. Supported: {CANONICAL_TIMEFRAMES}"
            )
            return None
        timeframe = norm_timeframe

        # PERF FIX: this had no is_backtest_mode() check, so every backtest
        # bar that needed MTF data (e.g. H4 bias) tried to reach a live MT5
        # terminal — connect, health-check (4 retries), then disconnect —
        # before finally failing over. That retry/backoff sequence (~3s+ per
        # attempt per the trader.log evidence) was one of the dominant
        # per-bar costs, on top of being pointless: a live terminal can't
        # serve historical candles for a date the backtest is replaying
        # anyway. Backtests must get MTF data from the CSV provider (see
        # csv_data_provider.py) — fail fast here instead of retrying network
        # I/O that was never going to succeed usefully.
        try:
            from core.constants import is_backtest_mode
            if is_backtest_mode():                # Iteration-3: serve registered historical series (M15 + resampled
                # H1/H4) so SMCEngine / MTF paths work without live MT5.
                # Returns None if nothing registered or asof not set — same
                # fail-closed behaviour as before for unregistered symbols.
                try:
                    from data.backtest_ohlcv_cache import get_ohlcv as _bt_get
                    _bt_df = _bt_get(symbol, timeframe, limit=limit)
                    if _bt_df is not None and not _bt_df.empty:
                        log.debug(
                            f"[DataFetcher] backtest cache hit {symbol} {timeframe} "
                            f"bars={len(_bt_df)}"
                        )
                        return _bt_df
                except Exception as _bt_e:
                    log.debug(f"[DataFetcher] backtest cache miss: {_bt_e}")
                log.debug(
                    f"[DataFetcher] backtest mode — no cache for {symbol} {timeframe}"
                )
                return None
        except Exception as _bt_mode_e:
            log.debug(f"[DataFetcher] is_backtest_mode() check failed, assuming live mode: {_bt_mode_e}")

        log.debug(f"Fetching {symbol} | {timeframe} | {limit} candles...")

        # P4c FIX: per-fetch fallback chain instead of single-source dispatch.
        # Previously, self.source was chosen once at init and never changed.
        # If MT5 was selected (package importable) but the terminal was
        # down, every fetch returned None with no fallback — the entire
        # trading cycle was skipped for all symbols.
        #
        # Now: try the primary source first; if it fails, try the next
        # available source in priority order.  Log a WARNING on fallback
        # so the operator knows data quality has degraded.
        _FALLBACK_ORDER = [
            ("mt5",          self._fetch_mt5),
            ("alpha_vantage", self._fetch_alpha_vantage),
            ("polygon",      self._fetch_polygon),
            ("finnhub",      self._fetch_finnhub),
            ("twelve_data",  self._fetch_twelve_data),
            ("yfinance",     self._fetch_yfinance),
        ]

        # Operator fix (2026-07-23): MT5_ONLY_MODE — the external fallback
        # chain was observed producing data worse than having none at all
        # (self-contradictory XAUUSD prices between consecutive requests,
        # 3-candle H1 windows where 200 were requested, a permanently
        # misconfigured Finnhub key, TwelveData rate-limited on every
        # attempt). Default: MT5 only. On MT5 failure, return None and let
        # the existing "no data this cycle" fail-safe path handle it
        # (skip, log, retry next cycle) instead of silently trading on
        normalized_symbol = self._normalize_symbol(symbol)
        # old multi-provider fallback.
        try:
            from config import MT5_ONLY_MODE
        except Exception as e:
            log.debug(f"[DataFetcher] MT5_ONLY_MODE not in config ({e}), defaulting to True (fail-safe)")
            MT5_ONLY_MODE = True  # fail-safe default if config import fails
        if MT5_ONLY_MODE:
            _FALLBACK_ORDER = [("mt5", self._fetch_mt5)]

        # Determine start index: try the configured primary source first,
        # but also allow starting from further down if the primary is known
        # to be unavailable.
        start_idx = 0
        for i, (src_name, _) in enumerate(_FALLBACK_ORDER):
            if src_name == self.source:
                start_idx = i
                break

        result = None
        tried_sources = []
        for src_name, fetch_fn in _FALLBACK_ORDER[start_idx:]:
            tried_sources.append(src_name)
            try:
                result = fetch_fn(symbol, timeframe, limit)
                if result is not None and len(result) > 0:
                    if src_name != self.source:
                        log.warning(
                            f"[DataFetcher] {symbol}: primary source '{self.source}' failed, "
                            f"fell back to '{src_name}' ({len(result)} candles). "
                            f"Data quality may differ from production MT5."
                        )
                        self.source = src_name  # remember for next fetch
                    break
            except Exception as e:
                log.debug(f"[DataFetcher] {symbol} {src_name} fetch error: {e}")
                result = None

        # Track fetch success/failure for auto-unavailable marking
        if result is not None and len(result) > 0:
            record_fetch_success(symbol)
        else:
            if MT5_ONLY_MODE:
                log.warning(
                    f"[DataFetcher] {symbol} {timeframe}: MT5 fetch failed and "
                    f"MT5_ONLY_MODE=true — no external fallback attempted "
                    f"(by design; see config.MT5_ONLY_MODE). Skipping this cycle."
                )
            # Pass cold_start flag so sync-window failures don't
            # permanently mark symbols unavailable.
            _cold = _is_within_cold_start_window()
            if record_fetch_failure(symbol, cold_start=_cold):
                mark_symbol_unavailable(symbol)
                log.warning(
                    f"[DataFetcher] {symbol} failed {FETCH_FAIL_THRESHOLD}x consecutively — "
                    f"auto-marked unavailable. It will be skipped on future cycles."
                )

        return result

    # ─────────────────────────────────────────────
    # SOURCE 1: MetaTrader5 (PRIMARY)
    # ─────────────────────────────────────────────

    def _get_broker_utc_offset_hours(self, symbol: str = "EURUSD") -> float:
        """Dynamically detect the broker's current UTC offset by comparing
        a LIVE tick timestamp (mt5.symbol_info_tick) against the PC's real
        UTC clock — not a hardcoded/manual guess, and not dependent on
        MT5_BROKER_TZ_OFFSET_HOURS being set correctly by a human.

        Why tick time instead of bar time: a tick is "right now" by
        definition, so the comparison against datetime.now(timezone.utc)
        is a direct, same-instant measurement. Bar-based detection (see
        detect_broker_tz_offset() below) has to assume the last bar is
        recent, which isn't true when the market is closed or a symbol is
        thin.

        Self-correcting across DST: cached for
        _BROKER_OFFSET_CACHE_TTL_SEC (30 min) then re-measured, so if the
        broker flips GMT+2 <-> GMT+3 mid-session the fetcher picks up the
        new offset on its own within 30 minutes — no manual
        MT5_BROKER_TZ_OFFSET_HOURS edit, no restart, and it never "breaks"
        on the DST boundary the way a hardcoded 2 or 3 would.

        MT5_BROKER_TZ_OFFSET_HOURS is still honored as an explicit manual
        override (e.g. for a broker whose tick time is itself unreliable)
        — if the env var is set, it wins and detection is skipped.
        """
        env_override = os.getenv("MT5_BROKER_TZ_OFFSET_HOURS")
        if env_override not in (None, ""):
            return float(env_override)

        now = time.time()
        if (
            self._broker_offset_cache is not None
            and (now - self._broker_offset_cache_at) < self._BROKER_OFFSET_CACHE_TTL_SEC
        ):
            return self._broker_offset_cache

        if not MT5_AVAILABLE or self._mt5_conn is None:
            return self._broker_offset_cache or 0.0

        try:
            # 2026-08-20 fix: normalize whatever form of symbol was passed
            # in (bare "EURUSD", the default param, or an already-suffixed
            # "EURUSDm") to the correctly-cased broker symbol before
            # calling MT5's get_tick — same case-sensitivity issue as
            # _fetch_mt5's copy_rates_from_pos/symbol_select calls.
            mt5_symbol = _to_mt5_symbol(symbol)
            tick = self._mt5_conn.get_tick(mt5_symbol)
            if tick is None or not getattr(tick, "time", None):
                log.debug(
                    f"[MT5] Dynamic tz offset: no live tick for {mt5_symbol} — "
                    f"keeping previous offset ({self._broker_offset_cache or 0.0}h)"
                )
                return self._broker_offset_cache or 0.0

            broker_now = datetime.fromtimestamp(tick.time, tz=timezone.utc)
            utc_now = datetime.now(timezone.utc)
            offset_hours = round((broker_now - utc_now).total_seconds() / 3600)

            if self._broker_offset_cache is not None and offset_hours != self._broker_offset_cache:
                log.info(
                    f"[MT5] Broker UTC offset changed: "
                    f"{self._broker_offset_cache:+.0f}h -> {offset_hours:+.0f}h "
                    f"(likely a DST flip) — picked up automatically"
                )
            self._broker_offset_cache = float(offset_hours)
            self._broker_offset_cache_at = now
            return self._broker_offset_cache
        except Exception as e:
            log.warning(f"[MT5] Dynamic tz offset detection failed: {e} — "
                        f"keeping previous offset ({self._broker_offset_cache or 0.0}h)")
            return self._broker_offset_cache or 0.0

    def _fetch_mt5(self, symbol, timeframe, limit):
        """
        Fetch OHLCV data from MetaTrader5.

        Args:
            symbol (str):     MT5 symbol name (e.g., "EURUSD")
            timeframe (str):  Timeframe key (e.g., "M15")
            limit (int):      Number of candles to fetch

        Returns:
            pd.DataFrame: OHLCV data, or None on error

        P1 fix (institutional audit §3.1): this used to call
        mt5.initialize()/mt5.symbol_select()/mt5.copy_rates_from_pos()
        directly against the global MetaTrader5 module — unlocked, and
        completely independent of the MT5Connection instance +
        MT5_LOCK that mt5_connection.py/execution_router.py were
        specifically hardened to enforce (the "Day 90+ hotfix"). That
        meant this fetch path could call mt5.initialize() concurrently
        with an in-flight, lock-protected order in execution_router.py
        and invalidate the shared session mid-order. Now every MT5 call
        goes through self._mt5_conn, which owns the lock.
        """
        if not MT5_AVAILABLE:
            log.error("[MT5] MetaTrader5 package not installed — cannot fetch")
            return None
        if self._mt5_conn is None:
            log.error(
                "[MT5] No MT5Connection wired up (see DataFetcher.__init__) — "
                "cannot fetch without racing the shared MT5 session"
            )
            return None
        try:
            # Ensure MT5 is initialized — via the shared, locked connection,
            # NOT a direct mt5.initialize() call.
            if not self._mt5_conn.ensure_connected():
                log.error("[MT5] Shared MT5Connection could not be established")
                return None

            # Map timeframe string to MT5 constant
            if timeframe not in TIMEFRAME_MAP:
                log.error(f"[MT5] Unknown timeframe: {timeframe}")
                return None

            mt5_timeframe = TIMEFRAME_MAP[timeframe]

            # 2026-08-20 fix: `symbol` here is the bare canonical form
            # (e.g. "EURUSD") from _normalize_symbol(). MT5 needs the
            # broker's real, case-sensitive symbol name (e.g. "EURUSDm"
            # on Exness) — computed here, used ONLY for the actual MT5
            # calls below. `symbol` stays bare for logging/tracking so
            # it matches the keys used by mark_symbol_unavailable() /
            # record_fetch_failure() elsewhere in this module.
            mt5_symbol = _to_mt5_symbol(symbol)

            # Activate symbol in Market Watch — via the shared connection's
            # locked wrapper, not a direct mt5.symbol_select() call.
            if not self._mt5_conn.symbol_select(mt5_symbol, True):
                error_code, error_msg = mt5.last_error()
                # code=-1 means symbol doesn't exist on this broker.
                # Mark it so the system can skip it silently on future cycles
                # instead of triggering recovery pauses.
                if error_code == -1:
                    mark_symbol_unavailable(symbol)
                    log.info(
                        f"[MT5] Symbol '{mt5_symbol}' not available on broker "
                        f"(code=-1) — marked unavailable, will be skipped"
                    )
                # ── FIX: -10004 = "No IPC connection" means MT5 terminal
                # dropped its IPC channel. Logging it once per symbol per
                # cycle spammed the log (200+ identical errors in trader.log
                # all within the same second, when MT5 was momentarily down).
                # Debounce: only log once per 60s window for this error code,
                # and downgrade to warning (it's transient, not a code bug).
                elif error_code == -10004:
                    now = time.time()
                    last = self._last_ipc_log_ts.get(symbol, 0.0)
                    if now - last >= 60.0:
                        log.warning(
                            f"[MT5] IPC connection lost (code=-10004) — "
                            f"MT5 terminal may be restarting. Will retry next cycle. "
                            f"(debounced, next log in 60s)"
                        )
                        self._last_ipc_log_ts[symbol] = now
                    # BUG FIX (2026-08-21): previously this only logged and
                    # hoped the next cycle would work by itself. But -10004
                    # doesn't flip is_alive()'s health-check flag (that
                    # checks terminal_info(), a separate MT5 API context
                    # that can stay "up" while symbol_select/copy_rates
                    # keep failing) — so nothing ever forced a reconnect,
                    # and this could repeat for 30+ consecutive cycles
                    # until the catastrophic-error-cycle threshold killed
                    # and restarted the whole process. Actively escalate
                    # to a forced reconnect after a few consecutive hits.
                    self._mt5_conn.note_ipc_failure()
                    # Don't mark symbol unavailable — it's a connection issue,
                    # not a symbol issue. Next cycle may succeed.
                else:
                    log.error(
                        f"[MT5] Failed to select symbol '{mt5_symbol}': "
                        f"code={error_code}, msg={error_msg}"
                    )
                return None

            log.debug(f"[MT5] Symbol selected: {mt5_symbol}")
            self._mt5_conn.note_ipc_success()

            # Fetch candles from position 0 (most recent) backward — via the
            # shared connection's locked wrapper.
            #
            # BUGFIX (KeyError: 'time' at startup): right after
            # mt5.initialize()/login, the MT5 terminal has NOT finished
            # syncing this symbol's history from the broker yet — this
            # is normal and can take a few minutes for a large watchlist
            # (confirmed from trader.log: 100% of fetches failed with
            # KeyError: 'time' from 20:12:08 to 20:16:10, then 100%
            # succeeded from 20:16:43 onward — a pure cold-start/sync
            # window, not a data or symbol problem). During that window
            # copy_rates_from_pos() can return a malformed/incomplete
            # structured array that, once converted to a DataFrame,
            # has no 'time' field — so df['time'] blew up further down
            # with no retry, permanently skipping that cycle (and, with
            # MT5_ONLY_MODE=true, no fallback source either) instead of
            # just waiting the extra second or two for MT5 to catch up.
            #
            # Fix: retry with EXPONENTIAL BACKOFF until a WALL-CLOCK
            # DEADLINE is reached (default 5 min, configurable via
            # MT5_COLD_START_RETRY_DEADLINE_SEC env var). This replaces
            # the old fixed 4 × 1.5 s = 4.5 s retry budget, which was
            # ~50× too small for the observed ~242 s sync window.
            #
            # The deadline is only enforced for 'time'-column-missing
            # retries (i.e., cold-start sync failures).  Hard MT5
            # errors (candles is None or empty) still return immediately.
            _COLD_START_RETRY_DEADLINE = float(
                os.getenv("MT5_COLD_START_RETRY_DEADLINE_SEC", "300")
            )
            _COLD_START_BACKOFF_INITIAL = 1.0   # seconds
            _COLD_START_BACKOFF_MAX = 10.0       # seconds
            _retry_deadline = time.monotonic() + _COLD_START_RETRY_DEADLINE
            _backoff = _COLD_START_BACKOFF_INITIAL
            candles = None
            df = None
            _sync_attempt = 0
            while True:
                _sync_attempt += 1
                candles = self._mt5_conn.copy_rates_from_pos(mt5_symbol, mt5_timeframe, 0, limit)

                if candles is None:
                    error_code, error_msg = mt5.last_error()
                    log.error(
                        f"[MT5] copy_rates_from_pos failed for {mt5_symbol} {timeframe}: "
                        f"code={error_code}, msg={error_msg}"
                    )
                    return None

                if len(candles) == 0:
                    log.warning(f"[MT5] No candles returned for {mt5_symbol} {timeframe}")
                    return None

                # Convert numpy structured array → pandas DataFrame
                df = pd.DataFrame(candles)

                if 'time' in df.columns:
                    symbol_info = mt5.symbol_info(mt5_symbol)
                    if symbol_info is not None:
                        df.attrs['mt5_digits'] = int(symbol_info.digits)
                    break  # good data — proceed normally below

                # Malformed/incomplete data — almost always the MT5
                # cold-start history-sync window.
                remaining = _retry_deadline - time.monotonic()
                if remaining <= 0:
                    log.error(
                        f"[MT5] {mt5_symbol} {timeframe}: still missing 'time' column "
                        f"after {_sync_attempt} attempts over "
                        f"{_COLD_START_RETRY_DEADLINE}s — MT5 history "
                        f"sync may be stuck. Skipping this cycle for real."
                    )
                    return None

                log.warning(
                    f"[MT5] {mt5_symbol} {timeframe}: fetched candles missing "
                    f"'time' column (got columns={list(df.columns)}) — "
                    f"likely MT5 still syncing history after startup "
                    f"(attempt {_sync_attempt}, {remaining:.0f}s remaining). "
                    f"Retrying in {_backoff:.1f}s..."
                )
                df = None
                time.sleep(_backoff)
                _backoff = min(_backoff * 2, _COLD_START_BACKOFF_MAX)

            # ── Timezone handling (audit P1 fix) ──────────────────────
            # MT5's `time` field is documented as Unix-epoch seconds (UTC
            # absolute). HOWEVER, in practice many brokers configure the
            # MT5 server to return bar OPEN time in BROKER SERVER TIME
            # (commonly GMT+2 in winter / GMT+3 in summer, the so-called
            # "FX broker time" used by IC Markets, Pepperstone, Exness,
            # FXTM, etc.). When that happens, `pd.to_datetime(unit='s')`
            # silently treats the broker wall-clock as UTC, producing
            # timestamps that are 2-3 hours in the FUTURE relative to
            # true UTC — which is exactly the "11126s left on M15" bug
            # reported by the operator.
            #
            # We now auto-detect this by comparing a live tick timestamp
            # against the PC's real UTC clock every fetch (cached 30 min —
            # see _get_broker_utc_offset_hours), so this self-corrects
            # across a broker's DST flip instead of needing a human to
            # notice and edit an env var:
            #
            #   MT5_BROKER_TZ_OFFSET_HOURS=2   (manual override, optional)
            #   MT5_BROKER_TZ_OFFSET_HOURS=3   (manual override, optional)
            #
            # When non-zero, we SUBTRACT the offset from the parsed time
            # to convert broker wall-clock → true UTC, then attach
            # tzinfo=timezone.utc so downstream code (is_candle_closed,
            # check_data_staleness) can rely on the tz tag.
            broker_offset_hours = self._get_broker_utc_offset_hours(mt5_symbol)

            # Convert 'time' from Unix seconds to datetime.
            # `pd.to_datetime(unit='s')` returns NAIVE UTC by default.
            df['time'] = pd.to_datetime(df['time'], unit='s', utc=False)

            if broker_offset_hours != 0:
                # Broker returned server wall-clock mislabeled as epoch.
                # Subtract the offset to recover true UTC.
                df['time'] = df['time'] - pd.Timedelta(hours=broker_offset_hours)
                log.info(
                    f"[MT5] Applied broker tz offset: -{broker_offset_hours}h "
                    f"(MT5_BROKER_TZ_OFFSET_HOURS={broker_offset_hours}). "
                    f"Bar timestamps are now true UTC."
                )

            # Attach explicit UTC tzinfo so downstream is_candle_closed()
            # no longer needs the dangerous `replace(tzinfo=utc)` fallback.
            df['time'] = df['time'].dt.tz_localize('UTC')

            # Normalize the adapter output so every consumer sees
            # chronological bars, regardless of MT5 row ordering.
            df.sort_values('time', inplace=True)
            df.drop_duplicates(subset='time', keep='last', inplace=True)

            # Diagnostic log: show first & last bar timestamps in UTC.
            # This makes the broker-tz bug visible at fetch time instead
            # of being silently propagated to the trader.
            log.info(
                f"[MT5] Bar timestamps (UTC): "
                f"first={df['time'].iloc[0].isoformat()} | "
                f"last={df['time'].iloc[-1].isoformat()} | "
                f"broker_offset={broker_offset_hours}h"
            )

            # Set datetime as index
            df.set_index('time', inplace=True)

            # Keep only OHLCV columns, standardize to lowercase.
            # NOTE: MT5's 'tick_volume' is the number of price ticks in the
            # bar, not consolidated traded volume — forex is decentralized
            # and there is no true consolidated volume figure. We keep the
            # column named 'volume' for downstream compatibility, but this
            # is documented here and in _fetch_yfinance/others so anyone
            # weighting signal confidence by "volume" knows it's a tick
            # activity proxy, not real traded volume.
            _ohlcv_columns = ['open', 'high', 'low', 'close', 'tick_volume']
            if 'spread' in df.columns:
                _ohlcv_columns.append('spread')
            df = df[_ohlcv_columns].copy()
            df.rename(columns={'tick_volume': 'volume'}, inplace=True)

            # Ensure correct column order
            _ordered_columns = ['open', 'high', 'low', 'close', 'volume']
            if 'spread' in df.columns:
                _ordered_columns.append('spread')
            df = df[_ordered_columns]

            # ── Forming-candle guard (audit fix) ───────────────────────
            # copy_rates_from_pos(symbol, tf, 0, limit) starts at position 0,
            # which the MT5 docs define as the CURRENT bar — i.e. it can
            # still be forming when this call happens mid-bar. Nothing
            # upstream of here ever dropped that row.
            #
            # _detect_bos()'s own docstring assumes the DataFrame's last
            # row is a fully CLOSED candle. Feeding it a still-forming bar
            # means BOS/CHoCH on H4/H1 can silently repaint as that bar's
            # high/low/close keep moving — this is not specific to one
            # trade, it's every call to fetch_ohlcv() via MT5.
            #
            # Fix: compute the bar's implied close time (open + timeframe
            # duration) and drop the last row if that close time is still
            # in the future, so every downstream consumer (not just
            # _detect_bos) transparently only ever sees closed candles.
            _tf_seconds = {
                "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
                "H1": 3600, "H4": 14400, "D1": 86400,
            }.get(timeframe.upper())
            if _tf_seconds and len(df) > 1:
                _last_open = df.index[-1]
                _implied_close = _last_open + pd.Timedelta(seconds=_tf_seconds)
                _now_utc = pd.Timestamp.now(tz='UTC')
                if _implied_close > _now_utc:
                    log.debug(
                        f"[MT5] {symbol} {timeframe}: dropping still-forming "
                        f"last bar (open={_last_open.isoformat()}, implied "
                        f"close={_implied_close.isoformat()} is still in the "
                        f"future) — structural detectors (BOS/CHoCH) require "
                        f"closed candles only."
                    )
                    df = df.iloc[:-1].copy()

            log.info(
                f"[OK] Got {len(df)} candles for {symbol} {timeframe} via MT5 | "
                f"Latest: {df.index.max()}"
            )

            # ── P1 audit: verify latest bar is not in the future ─────
            # If the DataFrame index carries tzinfo (we tag it UTC above),
            # compare against true UTC now. If it's naive, the broker-tz
            # bug may be present — flag for the operator.
            try:
                _last_ts = df.index.max()
                _now_utc = pd.Timestamp.now(tz='UTC')
                if hasattr(_last_ts, 'tzinfo') and _last_ts.tzinfo:
                    # BUGFIX (H1/H4/D1 false STALE): `_last_ts` is the bar's
                    # OPEN time (MT5/pandas convention — index = bar open),
                    # and the forming-candle guard above already drops the
                    # in-progress bar, so `df.index.max()` is always the most
                    # recent CLOSED bar's open time. Measuring `_delta` from
                    # open time means a perfectly healthy feed shows an age
                    # that climbs to nearly 2x the timeframe just before the
                    # NEXT bar closes (e.g. an H1 bar that opened at :00 is
                    # correctly still "the latest closed bar" right up until
                    # :59 of the FOLLOWING hour, i.e. up to ~7200s old, not
                    # ~3600s). The 1.5x multiplier below was sized as if
                    # `_delta` were measured from CLOSE time, so H1/H4/D1
                    # were false-flagging STALE for the last portion of every
                    # period. Fix: measure from the bar's implied CLOSE time
                    # (open + tf_seconds) instead, so a healthy bar's age
                    # tops out at ~1x the timeframe, matching what the
                    # multiplier was actually designed for.
                    _tf_sec_for_close = {
                        "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
                        "H1": 3600, "H4": 14400, "D1": 86400,
                    }.get(timeframe.upper(), 3600)
                    _last_close_ts = _last_ts.to_pydatetime() + timedelta(seconds=_tf_sec_for_close)
                    _delta = (_now_utc - _last_close_ts).total_seconds()
                    if _delta < -60:
                        log.critical(
                            f"[MT5] Latest bar is {_delta:.0f}s in the FUTURE "
                            f"relative to UTC now — broker timezone is being "
                            f"mislabeled as UTC. Set MT5_BROKER_TZ_OFFSET_HOURS "
                            f"to the broker's GMT offset (e.g. 2 or 3)."
                        )
                    elif _delta > 3600:
                        # Stale data warning — but be timeframe-aware so
                        # we don't cry wolf on D1/H4 bars that legitimately
                        # only update once a day / every 4h.
                        _tf_sec = _tf_sec_for_close
                        # Warn only if the bar is older than 1.5× its
                        # timeframe interval (allows for weekend gaps
                        # and the brief moment after a bar closes before
                        # the next one prints). D1 on Sat/Sun is normal
                        # at up to ~60h old; H4 on Sun is normal at up
                        # to ~12h old.
                        #
                        # The 1.5× multiplier is configurable via env var
                        # MT5_STALE_BAR_MULTIPLIER (default 1.5). Some
                        # brokers print H1 bars 2-3 minutes after the
                        # hour mark, and on low-liquidity pairs (e.g.
                        # USDTRY, USDZAR) the H1 bar can be 10-15 minutes
                        # late without indicating a feed problem.
                        # Setting MT5_STALE_BAR_MULTIPLIER=3.0 raises the
                        # H1 threshold from 5400s (90 min) to 10800s
                        # (3 hours), eliminating false-positive stale-data
                        # errors during normal broker latency.
                        try:
                            _stale_mult = float(os.getenv("MT5_STALE_BAR_MULTIPLIER", "1.5"))
                        except (TypeError, ValueError):
                            _stale_mult = 1.5
                        _stale_threshold = max(_tf_sec * _stale_mult, 3600)
                        if _delta > _stale_threshold:
                            # BUGFIX: this used to just log.warning and
                            # move on — "stale data or market closed" was
                            # logged as a single ambiguous message with no
                            # actual market-hours check behind it, and the
                            # returned df carried no signal that anything
                            # was wrong. Downstream (AnalysisAgent) would
                            # happily generate a BUY/SELL signal off a
                            # 6-8hr-old H1 bar during a normal weekend
                            # close, because nothing ever distinguished
                            # "market's closed, this gap is expected" from
                            # "market's open and our data feed is broken".
                            # Now: check forex market hours explicitly and
                            # tag the df via .attrs so AnalysisAgent can
                            # gate on it instead of guessing from a log line.
                            _market_open_expected = _is_forex_market_expected_open(_now_utc)
                            if _market_open_expected:
                                log.error(
                                    f"[MT5] Latest {timeframe} bar is "
                                    f"{_delta:.0f}s old (>{_stale_threshold:.0f}s "
                                    f"threshold) while the forex market is "
                                    f"expected to be OPEN — this is a genuine "
                                    f"stale-data condition (feed/connection "
                                    f"problem), NOT a weekend gap."
                                )
                                df.attrs["stale_data"] = True
                                df.attrs["stale_reason"] = (
                                    f"{timeframe} bar {_delta:.0f}s old "
                                    f"(>{_stale_threshold:.0f}s) during expected "
                                    f"open market hours"
                                )
                                # ── Forced reconnect on genuine staleness ──
                                # Root cause (fixed in mt5_connection.py
                                # is_alive()): terminal_info() returning a
                                # non-None struct only proves the terminal
                                # PROCESS is reachable via IPC — it does not
                                # mean the terminal is connected to the
                                # broker's trade server. is_alive() used to
                                # ignore terminal_info().connected entirely,
                                # so ensure_connected() kept reporting healthy
                                # while the feed was frozen. With that fixed,
                                # is_alive() will now correctly report False
                                # here and _require_connected() will already
                                # have auto-reconnected via reconnect().
                                # This call is now mostly a belt-and-braces
                                # nudge in case ensure_connected() above ran
                                # before is_alive()'s cache window expired —
                                # debounced so we don't hammer the broker
                                # across every symbol in the same cycle.
                                _now_mono = time.monotonic()
                                _last_forced = getattr(
                                    self, "_last_forced_reconnect_at", 0.0
                                )
                                _RECONNECT_DEBOUNCE_SEC = float(
                                    os.getenv("MT5_STALE_RECONNECT_DEBOUNCE_SEC", "60")
                                )
                                if _now_mono - _last_forced >= _RECONNECT_DEBOUNCE_SEC:
                                    self._last_forced_reconnect_at = _now_mono
                                    log.warning(
                                        f"[MT5] Feed staleness detected — "
                                        f"forcing reconnect."
                                    )
                                    reconnected = self._mt5_conn.reconnect()
                                    if reconnected:
                                        log.warning(
                                            "[MT5] Reconnect succeeded — next "
                                            "cycle should see fresh bars."
                                        )
                                    else:
                                        log.error(
                                            "[MT5] Reconnect FAILED — feed "
                                            "will remain stale until terminal/"
                                            "broker connectivity is restored."
                                        )
                            else:
                                log.info(
                                    f"[MT5] Latest {timeframe} bar is "
                                    f"{_delta:.0f}s old (>{_stale_threshold:.0f}s "
                                    f"threshold) — expected weekend/holiday "
                                    f"market-closed gap, not a data problem."
                                )
                                # Still mark stale (the bar genuinely is old
                                # and shouldn't be traded on), but with a
                                # benign reason so it's not confused with a
                                # feed outage.
                                df.attrs["stale_data"] = True
                                df.attrs["stale_reason"] = "market_closed_gap"
                            df.attrs["bar_age_seconds"] = _delta
                            df.attrs["market_expected_open"] = _market_open_expected
                        else:
                            # Within tolerance for this timeframe — log
                            # at DEBUG only so we don't spam INFO logs.
                            log.debug(
                                f"[MT5] Latest {timeframe} bar is "
                                f"{_delta:.0f}s old (within tolerance)."
                            )
                            df.attrs["stale_data"] = False
                            df.attrs["bar_age_seconds"] = _delta
            except Exception as _diag_e:
                log.debug(f"[MT5] future-bar diagnostic skipped: {_diag_e}")

            return df

        except Exception as e:
            log.error(f"[MT5] Exception during fetch: {type(e).__name__}: {e}")
            return None
        finally:
            # Keep MT5 initialized for subsequent calls (don't shutdown)
            pass

    def detect_broker_tz_offset(self, symbol: str = "EURUSD",
                                 timeframe: str = "M15") -> Optional[int]:
        """
        Audit P1 helper: auto-detect the broker's GMT offset by comparing
        the latest MT5 bar timestamp against true UTC now.

        This is a DIAGNOSTIC method — call it once at startup (or from
        a CLI helper) to figure out what value to put in
        MT5_BROKER_TZ_OFFSET_HOURS. It does NOT modify the env var.

        Algorithm:
          1. Fetch a small slice of M15 candles WITHOUT applying any
             offset (we monkey-patch the env var to "0" locally).
          2. Read the last bar's timestamp.
          3. Compare to datetime.now(timezone.utc).
          4. Round the delta (in hours) to the nearest integer — that's
             the broker's GMT offset.

        Returns:
          int: suggested MT5_BROKER_TZ_OFFSET_HOURS value (0, 1, 2, 3, ...)
          None: if detection failed (MT5 unavailable, no data, etc.)

        Example log output:
          [MT5] Broker tz detection: last_bar=2026-07-13T14:00:00+00:00
                  now_utc=2026-07-13T11:00:00+00:00
                  delta_hours=3.0
          [MT5] Suggested MT5_BROKER_TZ_OFFSET_HOURS=3
                  (add this line to .env to fix FUTURE_BAR warnings)
        """
        if not MT5_AVAILABLE or self._mt5_conn is None:
            log.warning("[MT5] detect_broker_tz_offset: MT5 unavailable")
            return None

        # Temporarily force offset=0 so we see the RAW broker time.
        original = os.environ.get("MT5_BROKER_TZ_OFFSET_HOURS")
        os.environ["MT5_BROKER_TZ_OFFSET_HOURS"] = "0"
        try:
            df = self._fetch_mt5(symbol, timeframe, limit=5)
        finally:
            # Restore the original env value
            if original is None:
                os.environ.pop("MT5_BROKER_TZ_OFFSET_HOURS", None)
            else:
                os.environ["MT5_BROKER_TZ_OFFSET_HOURS"] = original

        if df is None or len(df) == 0:
            log.warning("[MT5] detect_broker_tz_offset: no data returned")
            return None

        try:
            last_bar = df.index.max()
            if hasattr(last_bar, "to_pydatetime"):
                last_bar = last_bar.to_pydatetime()
            if last_bar.tzinfo is None:
                last_bar = last_bar.replace(tzinfo=timezone.utc)
            now_utc = datetime.now(timezone.utc)
            delta_sec = (last_bar - now_utc).total_seconds()
            delta_hours = delta_sec / 3600.0

            # Round to nearest integer hour (broker offsets are whole hours)
            # Only positive offsets make sense (broker ahead of UTC).
            suggested = max(0, round(delta_hours))

            log.info(
                f"[MT5] Broker tz detection: "
                f"last_bar={last_bar.isoformat()} | "
                f"now_utc={now_utc.isoformat()} | "
                f"delta_hours={delta_hours:.2f}"
            )
            if suggested > 0:
                log.info(
                    f"[MT5] Suggested MT5_BROKER_TZ_OFFSET_HOURS={suggested} "
                    f"(add this line to .env to fix FUTURE_BAR warnings)"
                )
            else:
                log.info(
                    "[MT5] Broker tz offset appears to be 0 — broker IS "
                    "UTC, no env var change needed."
                )
            return suggested
        except Exception as e:
            log.error(f"[MT5] detect_broker_tz_offset failed: {e}")
            return None

    # ─────────────────────────────────────────────
    # SOURCE 2: TradingView (FALLBACK)
    # ─────────────────────────────────────────────

    def _fetch_tvdatafeed(self, symbol, timeframe, limit):
        """
        Fetch OHLCV data from TradingView (fallback).
        
        Args:
            symbol (str):     Trading pair (e.g., "EURUSD")
            timeframe (str):  Timeframe (e.g., "M15", "15m")
            limit (int):      Number of candles
        
        Returns:
            pd.DataFrame: OHLCV data, or None on error
        """
        try:
            from tvdatafeed import TvDatafeed, Interval

            tf_map = {
                'M5':   Interval.in_5_minute,
                'M15':  Interval.in_15_minute,
                'M30':  Interval.in_30_minute,
                'H1':   Interval.in_1_hour,
                'H4':   Interval.in_4_hour,
                'D1':   Interval.in_daily,
            }

            tv_timeframe = tf_map.get(timeframe, Interval.in_15_minute)

            tv = TvDatafeed()
            raw = tv.get_hist(
                symbol=symbol,
                exchange='FX',
                interval=tv_timeframe,
                n_bars=limit,
            )

            if raw is None or raw.empty:
                log.error(f"[TVDatafeed] No data returned for {symbol}")
                return None

            df = raw[['open', 'high', 'low', 'close', 'volume']]
            log.info(
                f"[OK] Got {len(df)} candles for {symbol} {timeframe} via TradingView | "
                f"Latest: {df.index[-1]}"
            )
            return df

        except Exception as e:
            log.error(f"[TVDatafeed] Exception: {type(e).__name__}: {e}")
            return None

    # ─────────────────────────────────────────────
    # UTILITY METHODS
    # ─────────────────────────────────────────────

    # ── Day 90 — yfinance fallback (Linux VPS / demo) ──
    @staticmethod
    def _resample_h1_to_h4(df_h1: "pd.DataFrame", limit: int) -> Optional["pd.DataFrame"]:
        """Aggregate H1 candles into H4 bars.

        Bugfix: yfinance and Alpha Vantage have no native 4-hour interval
        on their free tiers ("unsupported timeframe: H4"), which was
        cascading into MTF confluence failures ("Could not fetch 4h").
        Rather than failing outright, resample from H1 — the standard
        technique for synthesizing a timeframe a provider doesn't expose
        natively. Uses the MT5/forex session convention of 4h bars
        starting at 00:00 UTC (00-04, 04-08, ... 20-24) via origin="epoch",
        so bars line up with MT5's own H4 candles instead of drifting to
        whatever hour the fetched H1 series happens to start on.
        """
        if df_h1 is None or len(df_h1) == 0:
            return None
        try:
            agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
            if "volume" in df_h1.columns:
                agg["volume"] = "sum"
            df_h4 = df_h1.resample("4h", origin="epoch").agg(agg)
            df_h4 = df_h4.dropna(subset=["open", "high", "low", "close"])
            if len(df_h4) == 0:
                return None
            return df_h4.tail(limit)
        except Exception as e:
            log.warning(f"[resample] H1→H4 aggregation failed: {e}")
            return None

    def _fetch_yfinance(self, symbol, timeframe, limit):
        """
        Fetch OHLCV data from Yahoo Finance via yfinance.

        Yahoo exposes forex pairs as EURUSD=X, GBPUSD=X, USDJPY=X etc.
        Metals: GC=F (gold), SI=F (silver). Indexes: ^GSPC (S&P 500).

        Limitations:
          - Yahoo's forex data is delayed 15-20 min.
          - Intraday history is limited to last 60 days for 5m/15m.
          - 'volume' for FX tickers from Yahoo is frequently 0 or unreliable;
            treat it the same as MT5 tick_volume — an activity proxy, not
            true consolidated volume.
          - Use ONLY for demo / paper trading, never production.

        Returns DataFrame with columns ['open','high','low','close','volume']
        and datetime index, or None on failure.
        """
        try:
            import yfinance as yf
        except ImportError:
            log.error("[yfinance] package not installed — run: pip install yfinance")
            return None

        # Map symbol to Yahoo format
        yf_symbol = self._to_yahoo_symbol(symbol)
        # Metals (XAUUSD/XAGUSD/XPDUSD/XPTUSD) and any other symbol Yahoo
        # doesn't support return None — short-circuit here so the caller
        # (DataFetcher.fetch) can fall back to MT5 / other sources instead
        # of producing a noisy "possibly delisted" error from yfinance.
        if yf_symbol is None:
            log.info(
                f"[yfinance] {symbol} has no Yahoo Finance ticker — "
                f"deferring to MT5 / other data sources."
            )
            return None
        # Map timeframe to yfinance interval
        interval = self._tf_to_yfinance_interval(timeframe)
        if interval is None:
            # Bugfix: H4 has no native yfinance interval. Instead of
            # failing outright (the old behavior — correct for avoiding a
            # *silent* substitution, but left H4 permanently unusable from
            # this source), fetch H1 and resample. Still explicit/logged,
            # just no longer a dead end.
            if timeframe.upper() == "H4":
                log.info("[yfinance] H4 not natively supported — fetching H1 and resampling")
                df_h1 = self._fetch_yfinance(symbol, "H1", limit=limit * 4 + 20)
                return self._resample_h1_to_h4(df_h1, limit)
            log.error(f"[yfinance] unsupported timeframe: {timeframe}")
            return None

        # Compute period — yfinance doesn't take a candle count.
        # Use a generous lookback; the tail(limit) truncates later.
        period = "60d" if interval in ("5m", "15m", "30m") else "1y"

        try:
            log.debug(f"[yfinance] Fetching {yf_symbol} interval={interval} period={period}")
            df = yf.download(
                yf_symbol,
                interval=interval,
                period=period,
                progress=False,
                auto_adjust=False,
            )
        except Exception as e:
            log.error(f"[yfinance] download failed for {yf_symbol}: {e}")
            return None

        if df is None or len(df) == 0:
            log.error(f"[yfinance] no data returned for {yf_symbol}")
            return None

        # Normalize columns
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })
        # If multi-level columns (yfinance sometimes returns DataFrame
        # with MultiIndex columns when single ticker), flatten.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Keep only OHLCV
        keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
        df = df[keep].copy()

        # Truncate to limit
        df = df.tail(limit)

        # Ensure tz-naive (some pipelines expect naive index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        log.info(
            f"[yfinance] {symbol} ({yf_symbol}) | {timeframe} | "
            f"{len(df)} candles | last close: {df['close'].iloc[-1]:.5f}"
        )
        return df

    @staticmethod
    def _to_yahoo_symbol(symbol: str) -> str:
        """Convert internal symbol to Yahoo Finance format."""
        s = symbol.upper().replace("/", "").replace("=", "")
        # Forex majors — Yahoo uses EURUSD=X format
        forex_pairs = {
            "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
            "USDCHF", "NZDUSD", "EURGBP", "EURJPY", "EURCHF",
            "EURAUD", "EURCAD", "EURNZD", "GBPJPY", "GBPCHF",
            "GBPAUD", "GBPCAD", "GBPNZD", "AUDJPY", "AUDCHF",
            "AUDCAD", "AUDNZD", "NZDJPY", "NZDCHF", "NZDCAD",
            "CADJPY", "CADCHF", "CHFJPY",
        }
        if s in forex_pairs:
            return f"{s}=X"
        # Metals — Yahoo has REMOVED free tickers for all four FX-metals.
        # Day 82 fix: GC=F (Gold futures) is delisted from yfinance.
        # Day 121 fix: SI=F (Silver) still works but is unreliable for
        # short periods; XPDUSD (palladium) and XPTUSD (platinum) have
        # NEVER had a working free Yahoo ticker (returns "possibly
        # delisted; no price data found" on every call — see error log).
        # For ALL metals, return None so the caller falls back to MT5 /
        # other sources instead of producing noisy delisted-symbol errors.
        if s in ("XAUUSD", "XAGUSD", "XPDUSD", "XPTUSD"):
            return None  # Let MT5 or other sources handle metal prices
        # Indices
        if s == "SPX500":
            return "^GSPC"
        if s == "US30":
            return "^DJI"
        if s == "NAS100":
            return "^NDX"
        if s == "VIX":
            return "^VIX"
        # Default — assume it's already a Yahoo ticker (e.g. AAPL)
        return s

    @staticmethod
    def _tf_to_yfinance_interval(timeframe: str):
        """Map internal (canonical, prefix-style) timeframe to a yfinance
        interval string.

        Day 103 fix (institutional review): the previous implementation
        did string-replace on the whole timeframe token (e.g. stripping
        "M"/"H" characters) which does not correctly invert the prefix-style
        internal convention ("H1", "H4", "D1", ...). It happened to work
        for a couple of cases by coincidence but was not a reliable inverse
        of _normalize_timeframe. This version takes an already-canonicalized
        timeframe (see _normalize_timeframe) and maps it via a direct,
        unambiguous lookup table — no string surgery.

        4H has no native yfinance interval; it is intentionally NOT
        silently downgraded to 1h here (that would be the same silent
        substitution bug this review flagged for MT5). Callers requesting
        H4 against the yfinance source get None and an explicit log error.
        """
        mapping = {
            "M5":  "5m",
            "M15": "15m",
            "M30": "30m",
            "H1":  "1h",
            "D1":  "1d",
        }
        return mapping.get(timeframe)

    # ════════════════════════════════════════════════════════════
    # Day 92 — Professional free-tier API providers
    # ════════════════════════════════════════════════════════════
    # Each provider has slightly different symbol formats + interval
    # conventions. We normalize them all to our internal format
    # (EURUSD / M15) so downstream code doesn't care which source
    # produced the data.
    # ════════════════════════════════════════════════════════════

    # ── SOURCE: Alpha Vantage ────────────────────────────────────
    # Free tier: 25 requests/day, 5 req/min. Good for live forex +
    # pre-built technical indicators (RSI, MACD, SMA) without us
    # having to compute them ourselves.
    # Docs: https://www.alphavantage.co/documentation/
    #
    # Day 103 note (institutional review): this endpoint is called with
    # outputsize="full" on every fetch, which can return years of data
    # and will exhaust the 25-req/day free quota almost immediately in
    # an automated polling loop. There is no caching layer in this file.
    # A caching/rate-limit layer should sit in front of this method
    # before it is used in a live automated cycle — see review notes.

    def _fetch_alpha_vantage(self, symbol: str, timeframe: str, limit: int):
        """Fetch OHLCV from Alpha Vantage FX_INTRADAY / FX_DAILY endpoint."""
        import requests
        # Day 99+ FIX (Issue #2): route through rate_limited_get to
        # respect the 5 req/min free-tier limit + retry on 429.
        from utils.api_rate_limiter import rate_limited_get
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
        if not api_key:
            _log_missing_api_key("AlphaVantage")
            return None

        # AV uses EUR/USD format (with slash)
        av_symbol = self._to_av_symbol(symbol)
        av_interval = self._tf_to_av_interval(timeframe)
        if av_interval is None:
            # Bugfix: Alpha Vantage's FX_INTRADAY only offers 1/5/15/30/60min
            # buckets — no native 4h. Resample from the 60min series instead
            # of returning "unsupported timeframe" (which was the direct
            # cause of MTF's "Could not fetch 4h").
            if timeframe.upper() == "H4":
                log.info("[AlphaVantage] H4 not natively supported — fetching H1 and resampling")
                df_h1 = self._fetch_alpha_vantage(symbol, "H1", limit=limit * 4 + 20)
                return self._resample_h1_to_h4(df_h1, limit)
            log.error(f"[AlphaVantage] unsupported timeframe: {timeframe}")
            return None

        # FX_INTRADAY for intraday, FX_DAILY for daily
        if av_interval == "daily":
            function = "FX_DAILY"
            params = {
                "function": function,
                "from_symbol": symbol[:3],
                "to_symbol": symbol[3:6],
                "outputsize": "full",
                "apikey": api_key,
            }
        else:
            function = "FX_INTRADAY"
            params = {
                "function": function,
                "from_symbol": symbol[:3],
                "to_symbol": symbol[3:6],
                "interval": av_interval,
                "outputsize": "full",
                "apikey": api_key,
            }

        try:
            url = os.getenv("ALPHA_VANTAGE_BASE_URL", "https://www.alphavantage.co/query")
            log.debug(f"[AlphaVantage] {function} {symbol} interval={av_interval}")
            resp = rate_limited_get(
                url, provider="alpha_vantage",
                params=params, timeout=15,
            )
            if resp is None:
                log.error("[AlphaVantage] request failed after retries")
                return None
            try:
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                log.error(f"[AlphaVantage] response parse failed: {e}")
                return None
        except Exception as e:
            log.error(f"[AlphaVantage] fetch failed: {e}")
            return None

        # Parse the time series
        ts_key = next((k for k in data if k.startswith("Time Series")), None)
        if not ts_key:
            err = data.get("Note") or data.get("Error Message") or "unknown"
            log.warning(f"[AlphaVantage] no time series in response: {err}")
            return None

        ts = data[ts_key]
        rows = []
        skipped = 0
        for ts_str, ohlc in ts.items():
            try:
                # Day 99+ FIX (Issue #3): attach UTC tzinfo so downstream
                # is_candle_closed() / check_data_staleness() don't have
                # to fall back to the dangerous naive-replace path that
                # produces false FUTURE_BAR warnings when the broker
                # offset is non-zero. Alpha Vantage timestamps are
                # already UTC (their server runs UTC), we just tag them.
                if " " in ts_str:
                    dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                else:
                    dt = datetime.strptime(ts_str, "%Y-%m-%d")
                dt = dt.replace(tzinfo=timezone.utc)
                rows.append({
                    "datetime": dt,
                    "open":  float(ohlc["1. open"]),
                    "high":  float(ohlc["2. high"]),
                    "low":   float(ohlc["3. low"]),
                    "close": float(ohlc["4. close"]),
                    "volume": 0.0,
                })
            except Exception as e:
                skipped += 1
                if skipped <= 3:
                    log.debug(f"[AlphaVantage] row {ts_str!r} parse failed: {type(e).__name__}: {e}")
                continue
        if skipped:
            log.warning(f"[AlphaVantage] skipped {skipped} malformed row(s) for {symbol}")

        if not rows:
            log.warning(f"[AlphaVantage] parsed 0 rows for {symbol}")
            return None

        df = pd.DataFrame(rows).sort_values("datetime").tail(limit).reset_index(drop=True)
        df = df.set_index("datetime")
        df.index.name = None
        log.info(
            f"[AlphaVantage] {symbol} | {timeframe} | "
            f"{len(df)} candles | last close: {df['close'].iloc[-1]:.5f}"
        )
        return df

    @staticmethod
    def _to_av_symbol(symbol: str) -> str:
        """Convert EURUSD → EUR/USD (Alpha Vantage format)."""
        s = symbol.upper().replace("/", "").replace("=X", "")
        if len(s) >= 6:
            return f"{s[:3]}/{s[3:6]}"
        return s

    @staticmethod
    def _tf_to_av_interval(timeframe: str):
        """Map internal timeframe to Alpha Vantage interval."""
        tf = timeframe.upper()
        return {
            "M5":  "5min", "5M": "5min",
            "M15": "15min", "15M": "15min",
            "M30": "30min", "30M": "30min",
            "H1":  "60min", "1H": "60min",
            "D1":  "daily", "1D": "daily",
        }.get(tf)

    # ── SOURCE: Polygon.io ──────────────────────────────────────
    # Free tier: 5 requests/min, end-of-day data only (no real-time).
    # Good for backtesting + historical analysis. Real-time needs paid.
    # Docs: https://polygon.io/docs/forex
    def _fetch_polygon(self, symbol: str, timeframe: str, limit: int):
        """Fetch OHLCV from Polygon.io forex aggregates endpoint."""
        import requests
        # Day 99+ FIX (Issue #2): route through rate_limited_get to
        # respect the 5 req/min free-tier limit + retry on 429.
        from utils.api_rate_limiter import rate_limited_get
        api_key = os.getenv("POLYGON_API_KEY", "")
        if not api_key:
            _log_missing_api_key("Polygon")
            return None

        # Polygon uses C:EURUSD format
        poly_symbol = f"C:{symbol.upper().replace('/', '').replace('=X', '')}"
        poly_mult, poly_timespan = self._tf_to_polygon(timeframe)
        if poly_mult is None:
            log.error(f"[Polygon] unsupported timeframe: {timeframe}")
            return None

        # Compute date range (Polygon needs explicit from/to)
        end = datetime.now(timezone.utc)
        # Generous lookback (limit * interval minutes, in days)
        lookback_days = max(30, limit * poly_mult // (60 * 24) + 30)
        start = end - timedelta(days=lookback_days)

        url = f"https://api.polygon.io/v2/aggs/ticker/{poly_symbol}/range/{poly_mult}/{poly_timespan}/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
        params = {"adjusted": "true", "sort": "asc", "limit": min(limit, 50000), "apiKey": api_key}

        try:
            log.debug(f"[Polygon] {poly_symbol} {poly_mult}{poly_timespan}")
            resp = rate_limited_get(
                url, provider="polygon",
                params=params, timeout=15,
            )
            if resp is None:
                log.error("[Polygon] request failed after retries")
                return None
            try:
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                log.error(f"[Polygon] response parse failed: {e}")
                return None
        except Exception as e:
            log.error(f"[Polygon] fetch failed: {e}")
            return None

        results = data.get("results", [])
        if not results:
            log.warning(f"[Polygon] no results for {symbol}")
            return None

        rows = []
        skipped = 0
        for r in results:
            try:
                # Polygon timestamp is in milliseconds. P1 audit fix:
                # KEEP the tzinfo=UTC (the original `.replace(tzinfo=None)`
                # stripped it, producing naive timestamps that downstream
                # is_candle_closed() had to re-tag with the dangerous
                # `replace(tzinfo=utc)` fallback).
                dt = datetime.fromtimestamp(r["t"] / 1000, tz=timezone.utc)
                rows.append({
                    "datetime": dt,
                    "open":  float(r["o"]),
                    "high":  float(r["h"]),
                    "low":   float(r["l"]),
                    "close": float(r["c"]),
                    "volume": float(r.get("v", 0)),
                })
            except Exception as e:
                skipped += 1
                if skipped <= 3:
                    log.debug(f"[Polygon] row {r!r} parse failed: {type(e).__name__}: {e}")
                continue
        if skipped:
            log.warning(f"[Polygon] skipped {skipped} malformed row(s) for {symbol}")

        df = pd.DataFrame(rows).tail(limit).reset_index(drop=True)
        df = df.set_index("datetime")
        df.index.name = None
        log.info(
            f"[Polygon] {symbol} | {timeframe} | "
            f"{len(df)} candles | last close: {df['close'].iloc[-1]:.5f}"
        )
        return df

    @staticmethod
    def _tf_to_polygon(timeframe: str):
        """Map internal timeframe to (multiplier, timespan) for Polygon."""
        tf = timeframe.upper()
        return {
            "M5":  (5, "minute"),  "5M":  (5, "minute"),
            "M15": (15, "minute"), "15M": (15, "minute"),
            "M30": (30, "minute"), "30M": (30, "minute"),
            "H1":  (1, "hour"),    "1H":  (1, "hour"),
            "H4":  (4, "hour"),    "4H":  (4, "hour"),
            "D1":  (1, "day"),     "1D":  (1, "day"),
        }.get(tf, (None, None))

    # ── SOURCE: Finnhub ─────────────────────────────────────────
    # Free tier: 60 req/min, forex candles endpoint.
    # Docs: https://finnhub.io/docs/api/forex-candles
    def _fetch_finnhub(self, symbol: str, timeframe: str, limit: int):
        """Fetch OHLCV from Finnhub forex candle endpoint."""
        import requests
        # Day 99+ FIX (Issue #2): route through rate_limited_get to
        # respect the 60 req/min free-tier limit + retry on 429.
        from utils.api_rate_limiter import rate_limited_get
        api_key = os.getenv("FINNHUB_API_KEY", "")
        if not api_key:
            _log_missing_api_key("Finnhub")
            return None

        # Finnhub uses OANDA:EUR_USD format
        finn_symbol = f"OANDA:{symbol[:3]}_{symbol[3:6]}"
        finn_res = self._tf_to_finnhub(timeframe)
        if finn_res is None:
            log.error(f"[Finnhub] unsupported timeframe: {timeframe}")
            return None

        end = int(datetime.now(timezone.utc).timestamp())
        # Generous lookback
        start = end - 30 * 86400  # 30 days

        url = os.getenv("FINNHUB_BASE_URL", "https://finnhub.io/api/v1") + "/forex/candle"
        params = {"symbol": finn_symbol, "resolution": finn_res,
                  "from": start, "to": end, "token": api_key}

        try:
            log.debug(f"[Finnhub] {finn_symbol} res={finn_res}")
            resp = rate_limited_get(
                url, provider="finnhub",
                params=params, timeout=15,
            )
            if resp is None:
                log.error("[Finnhub] request failed after retries")
                return None
            try:
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                log.error(f"[Finnhub] response parse failed: {e}")
                return None
        except Exception as e:
            log.error(f"[Finnhub] fetch failed: {e}")
            return None

        if data.get("s") != "ok":
            log.warning(f"[Finnhub] response not ok: {data}")
            return None

        rows = []
        skipped = 0
        for i, ts in enumerate(data["t"]):
            try:
                # P1 audit fix: keep tzinfo=UTC (was stripped before).
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                rows.append({
                    "datetime": dt,
                    "open":  float(data["o"][i]),
                    "high":  float(data["h"][i]),
                    "low":   float(data["l"][i]),
                    "close": float(data["c"][i]),
                    "volume": float(data["v"][i]) if i < len(data.get("v", [])) else 0,
                })
            except Exception as e:
                skipped += 1
                if skipped <= 3:
                    log.debug(f"[Finnhub] row index={i} ts={ts} parse failed: {type(e).__name__}: {e}")
                continue
        if skipped:
            log.warning(f"[Finnhub] skipped {skipped} malformed row(s) for {symbol}")

        df = pd.DataFrame(rows).tail(limit).reset_index(drop=True)
        df = df.set_index("datetime")
        df.index.name = None
        log.info(
            f"[Finnhub] {symbol} | {timeframe} | "
            f"{len(df)} candles | last close: {df['close'].iloc[-1]:.5f}"
        )
        return df

    @staticmethod
    def _tf_to_finnhub(timeframe: str):
        """Map internal timeframe to Finnhub resolution."""
        tf = timeframe.upper()
        return {
            "M5":  "5",  "5M":  "5",
            "M15": "15", "15M": "15",
            "M30": "30", "30M": "30",
            "H1":  "60", "1H":  "60",
            "H4":  "240","4H":  "240",
            "D1":  "D",  "1D":  "D",
        }.get(tf)

    # ── SOURCE: Twelve Data ─────────────────────────────────────
    # Free tier: 800 req/day, 8 req/min, 5-year historical.
    # Docs: https://twelvedata.com/docs#time-series
    def _fetch_twelve_data(self, symbol: str, timeframe: str, limit: int):
        """Fetch OHLCV from Twelve Data time_series endpoint."""
        import requests
        # Day 99+ FIX (Issue #2): route through rate_limited_get to
        # respect the 8 req/min free-tier limit + retry on 429.
        from utils.api_rate_limiter import rate_limited_get
        api_key = os.getenv("TWELVE_DATA_API_KEY", "")
        if not api_key:
            _log_missing_api_key("TwelveData")
            return None

        # Twelve Data uses EUR/USD format
        td_symbol = self._to_av_symbol(symbol)  # same format
        td_interval = self._tf_to_twelve_data(timeframe)
        if td_interval is None:
            log.error(f"[TwelveData] unsupported timeframe: {timeframe}")
            return None

        url = os.getenv("TWELVE_DATA_BASE_URL", "https://api.twelvedata.com") + "/time_series"
        params = {
            "symbol": td_symbol,
            "interval": td_interval,
            "outputsize": min(limit, 5000),
            "apikey": api_key,
            "format": "JSON",
        }

        try:
            log.debug(f"[TwelveData] {td_symbol} interval={td_interval}")
            resp = rate_limited_get(
                url, provider="twelve_data",
                params=params, timeout=15,
            )
            if resp is None:
                log.error("[TwelveData] request failed after retries")
                return None
            try:
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                log.error(f"[TwelveData] response parse failed: {e}")
                return None
        except Exception as e:
            log.error(f"[TwelveData] fetch failed: {e}")
            return None

        values = data.get("values", [])
        if not values:
            log.warning(f"[TwelveData] no values: {data.get('message', 'unknown')}")
            return None

        rows = []
        skipped = 0
        for v in values:
            try:
                # Day 99+ FIX (Issue #3): attach UTC tzinfo so downstream
                # is_candle_closed() doesn't have to fall back to the
                # dangerous naive-replace path. Twelve Data timestamps
                # are already UTC (their API server runs UTC).
                dt = datetime.strptime(v["datetime"], "%Y-%m-%d %H:%M:%S")
                dt = dt.replace(tzinfo=timezone.utc)
                rows.append({
                    "datetime": dt,
                    "open":  float(v["open"]),
                    "high":  float(v["high"]),
                    "low":   float(v["low"]),
                    "close": float(v["close"]),
                    "volume": 0.0,
                })
            except Exception as e:
                skipped += 1
                if skipped <= 3:
                    log.debug(f"[TwelveData] row {v!r} parse failed: {type(e).__name__}: {e}")
                continue
        if skipped:
            log.warning(f"[TwelveData] skipped {skipped} malformed row(s) for {symbol}")

        # Twelve Data returns newest-first; reverse for chronological order
        rows.reverse()
        df = pd.DataFrame(rows).tail(limit).reset_index(drop=True)
        df = df.set_index("datetime")
        df.index.name = None
        log.info(
            f"[TwelveData] {symbol} | {timeframe} | "
            f"{len(df)} candles | last close: {df['close'].iloc[-1]:.5f}"
        )
        return df

    @staticmethod
    def _tf_to_twelve_data(timeframe: str):
        """Map internal timeframe to Twelve Data interval."""
        tf = timeframe.upper()
        return {
            "M5":  "5min",  "5M":  "5min",
            "M15": "15min", "15M": "15min",
            "M30": "30min", "30M": "30min",
            "H1":  "1h",    "1H":  "1h",
            "H4":  "4h",    "4H":  "4h",
            "D1":  "1day",  "1D":  "1day",
        }.get(tf)

    def _normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol to bare canonical MT5-style form (e.g., "EURUSD").
        
        Converts:
          - "EUR/USD" → "EURUSD"
          - "EURUSD=X" → "EURUSD"
          - "EUR/USDT" → "EURUSD"
          - "EURUSD" → "EURUSD"
          - "EURUSDm" → "EURUSD"  (broker suffix stripped)

        2026-08-20 fix: this used to call .upper() on the RAW symbol
        BEFORE stripping the broker suffix, so a live symbol like
        "EURUSDm" (Exness "m" accounts) became "EURUSDM" — invalid on
        the broker (MT5 symbol names are case-sensitive), and also not
        a match in SYMBOL_MAP or the yfinance/Alpha Vantage pair sets
        below, breaking every fallback source too. Fix: strip the
        broker suffix FIRST, then clean/uppercase the bare pair name
        as before. The suffix is re-applied — in the correct case —
        only at the point of the actual MT5 call, via _to_mt5_symbol()
        in _fetch_mt5().
        """
        symbol = _strip_mt5_suffix(str(symbol).strip()).upper()
        # Use mapping if available
        if symbol in SYMBOL_MAP:
            return SYMBOL_MAP[symbol]
        # Otherwise, clean it manually
        # Round-14 fix: .replace("USDT", "USD") matched "USDT" ANYWHERE
        # in the string, not just as a trailing Tether-quote suffix like
        # "BTCUSDT" — it silently corrupted real forex codes that contain
        # "USDT" as a substring: "USDTRY" (USD/Turkish Lira) -> "USDRY",
        # "USDTHB" (USD/Thai Baht) -> "USDHB". Since this is the live
        # MT5 symbol-lookup path, that meant MT5 was being asked for a
        # nonexistent symbol ("USDRY"/"USDHB") every cycle for those
        # pairs — likely failing silently rather than trading correctly.
        # Fix: only strip the trailing "T" when USDT is genuinely a
        # Tether-quote SUFFIX (i.e. the string ends with it).
        symbol = (
            symbol
            .replace("=X", "")
            .replace("/", "")
        )
        if symbol.endswith("USDT"):
            symbol = symbol[:-1]
        return symbol

    def _normalize_timeframe(self, timeframe: str):
        """
        Normalize a timeframe string to its canonical internal form
        (e.g., "M15", "H1", "H4", "D1") — regardless of which data
        source is active.

        Day 103 fix (institutional review — CRITICAL):
        The previous implementation only worked correctly when MT5 was
        available, because it checked membership in TIMEFRAME_MAP, which
        is populated lazily and stays EMPTY whenever MT5 is not installed
        (every non-MT5 fallback path: yfinance, Alpha Vantage, Polygon,
        Finnhub, Twelve Data). On those paths the fallback logic used
        `.endswith("M"/"H"/"D")`, but this project's own internal
        convention is PREFIX-style ("H1", "H4", "D1", "MN1"), not
        suffix-style — none of those tokens end in M/H/D, so every call
        silently fell through to `return "M15"`.

        Concretely, this meant any request for H4 or D1 data on a non-MT5
        source was silently served as M15 data. In a Decision Layer doing
        multi-timeframe confirmation, that produces false confluence
        (e.g. comparing M15 against M15 while believing it's M15 vs H4)
        without any visible error — a mispriced/miscompared signal is far
        more dangerous than an explicit failure.

        This version:
          1. Accepts both suffix-style aliases ("15m", "1h", "1d") and the
             canonical prefix-style form ("M15", "H1", "D1") as input.
          2. Parses by regex (leading letters + trailing digits) instead
             of naive suffix matching, so it works identically whether or
             not MT5/TIMEFRAME_MAP is populated.
          3. Returns None — instead of silently defaulting to "M15" — for
             anything it cannot confidently resolve. Callers (fetch_ohlcv)
             now treat None as a hard failure and refuse to fetch, rather
             than silently substituting the wrong timeframe.
        """
        import re

        raw = str(timeframe).strip()
        tf = raw.upper()

        # Already canonical form, e.g. "M15", "H1", "H4", "D1", "W1", "MN1"
        if tf in CANONICAL_TIMEFRAMES:
            return tf

        # Suffix-style alias, e.g. "15m", "1h", "1d", "4h"
        m = re.fullmatch(r"(\d+)([MHD])", tf)
        if m:
            num, unit = m.group(1), m.group(2)
            candidate = f"{unit}{num}"
            if candidate in CANONICAL_TIMEFRAMES:
                return candidate

        # Prefix-style but not an exact canonical match, e.g. "m15" already
        # upper-cased above; also tolerate stray whitespace already stripped.
        m = re.fullmatch(r"([A-Z]+)(\d+)", tf)
        if m:
            unit, num = m.group(1), m.group(2)
            candidate = f"{unit}{num}"
            if candidate in CANONICAL_TIMEFRAMES:
                return candidate

        log.error(
            f"[DataFetcher] Unrecognized timeframe format: '{raw}'. "
            f"Supported: {CANONICAL_TIMEFRAMES} (or suffix aliases like "
            f"'15m', '1h', '1d'). Refusing to guess — no timeframe will "
            f"be silently substituted."
        )
        return None


# ── Singleton ───────────────────────────────────────────────────

_FETCHER: Optional["DataFetcher"] = None


def get_data_fetcher(mt5_conn=None) -> "DataFetcher":
    """Return a shared DataFetcher instance (singleton).
    
    Avoids repeated MT5 initialize/shutdown cycles when multiple
    modules create their own DataFetcher.  The singleton is lazily
    created on first call.

    Args:
        mt5_conn: Optional shared MT5Connection to inject on first
            creation (see DataFetcher.__init__). Ignored on subsequent
            calls once the singleton already exists — pass it on the
            first call made during app startup (e.g. from core.runtime,
            the same place execution_router.py's mt5_conn comes from).
    """
    global _FETCHER
    if _FETCHER is None:
        _FETCHER = DataFetcher(mt5_conn=mt5_conn)
    return _FETCHER