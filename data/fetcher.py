# data/fetcher.py
# ============================================================
# Multi-Source Data Fetcher (MT5-first)
# Primary Source: MetaTrader5 (native forex data)
# Fallback Source: TradingView via tvdatafeed
# ============================================================

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional
from utils.logger import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────
# MT5 AVAILABILITY GUARD
# ─────────────────────────────────────────────────────────────
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


def mark_symbol_unavailable(symbol: str) -> None:
    """Record that a symbol is not available on the current broker."""
    _UNAVAILABLE_SYMBOLS.add(symbol.upper())


def is_symbol_unavailable(symbol: str) -> bool:
    """Check if a symbol has been confirmed unavailable on the broker."""
    return symbol.upper() in _UNAVAILABLE_SYMBOLS


def get_unavailable_symbols() -> set:
    """Return the full set of unavailable symbols (for diagnostics)."""
    return set(_UNAVAILABLE_SYMBOLS)


def record_fetch_failure(symbol: str) -> bool:
    """Record a fetch failure for a symbol.

    Returns True if the symbol has now exceeded the failure threshold
    and should be marked as unavailable (caller should call
    mark_symbol_unavailable after this returns True).
    """
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
            except Exception:
                pass

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

        log.info(f"Fetching {symbol} | {timeframe} | {limit} candles...")

        result = None
        if self.source == "mt5":
            result = self._fetch_mt5(symbol, timeframe, limit)
        elif self.source == "tvdatafeed":
            result = self._fetch_tvdatafeed(symbol, timeframe, limit)
        elif self.source == "yfinance":
            result = self._fetch_yfinance(symbol, timeframe, limit)
        elif self.source == "alpha_vantage":
            result = self._fetch_alpha_vantage(symbol, timeframe, limit)
        elif self.source == "polygon":
            result = self._fetch_polygon(symbol, timeframe, limit)
        elif self.source == "finnhub":
            result = self._fetch_finnhub(symbol, timeframe, limit)
        elif self.source == "twelve_data":
            result = self._fetch_twelve_data(symbol, timeframe, limit)
        else:
            log.error("No data source available (MT5 not connected, tvdatafeed not installed)")

        # Track fetch success/failure for auto-unavailable marking
        if result is not None and len(result) > 0:
            record_fetch_success(symbol)
        else:
            if record_fetch_failure(symbol):
                mark_symbol_unavailable(symbol)
                log.warning(
                    f"[DataFetcher] {symbol} failed {FETCH_FAIL_THRESHOLD}x consecutively — "
                    f"auto-marked unavailable. It will be skipped on future cycles."
                )

        return result

    # ─────────────────────────────────────────────
    # SOURCE 1: MetaTrader5 (PRIMARY)
    # ─────────────────────────────────────────────

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

            # Activate symbol in Market Watch — via the shared connection's
            # locked wrapper, not a direct mt5.symbol_select() call.
            if not self._mt5_conn.symbol_select(symbol, True):
                error_code, error_msg = mt5.last_error()
                # code=-1 means symbol doesn't exist on this broker.
                # Mark it so the system can skip it silently on future cycles
                # instead of triggering recovery pauses.
                if error_code == -1:
                    mark_symbol_unavailable(symbol)
                    log.info(
                        f"[MT5] Symbol '{symbol}' not available on broker "
                        f"(code=-1) — marked unavailable, will be skipped"
                    )
                else:
                    log.error(
                        f"[MT5] Failed to select symbol '{symbol}': "
                        f"code={error_code}, msg={error_msg}"
                    )
                return None

            log.debug(f"[MT5] Symbol selected: {symbol}")

            # Fetch candles from position 0 (most recent) backward — via the
            # shared connection's locked wrapper.
            candles = self._mt5_conn.copy_rates_from_pos(symbol, mt5_timeframe, 0, limit)

            if candles is None:
                error_code, error_msg = mt5.last_error()
                log.error(
                    f"[MT5] copy_rates_from_pos failed for {symbol} {timeframe}: "
                    f"code={error_code}, msg={error_msg}"
                )
                return None

            if len(candles) == 0:
                log.warning(f"[MT5] No candles returned for {symbol} {timeframe}")
                return None

            # Convert numpy structured array → pandas DataFrame
            df = pd.DataFrame(candles)

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
            # We now expose this as an explicit env-var so the operator
            # can compensate:
            #
            #   MT5_BROKER_TZ_OFFSET_HOURS=2   (winter,  GMT+2 broker)
            #   MT5_BROKER_TZ_OFFSET_HOURS=3   (summer,  GMT+3 broker)
            #   MT5_BROKER_TZ_OFFSET_HOURS=0   (default, broker is UTC)
            #
            # When non-zero, we SUBTRACT the offset from the parsed time
            # to convert broker wall-clock → true UTC, then attach
            # tzinfo=timezone.utc so downstream code (is_candle_closed,
            # check_data_staleness) can rely on the tz tag.
            broker_offset_hours = float(os.getenv("MT5_BROKER_TZ_OFFSET_HOURS", "0") or 0)

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
            df = df[['open', 'high', 'low', 'close', 'tick_volume']].copy()
            df.rename(columns={'tick_volume': 'volume'}, inplace=True)

            # Ensure correct column order
            df = df[['open', 'high', 'low', 'close', 'volume']]

            log.info(
                f"[OK] Got {len(df)} candles for {symbol} {timeframe} via MT5 | "
                f"Latest: {df.index[-1]}"
            )

            # ── P1 audit: verify latest bar is not in the future ─────
            # If the DataFrame index carries tzinfo (we tag it UTC above),
            # compare against true UTC now. If it's naive, the broker-tz
            # bug may be present — flag for the operator.
            try:
                _last_ts = df.index[-1]
                _now_utc = pd.Timestamp.now(tz='UTC')
                if hasattr(_last_ts, 'tzinfo') and _last_ts.tzinfo:
                    _delta = (_now_utc - _last_ts.to_pydatetime()).total_seconds()
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
                        _tf_sec = {
                            "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
                            "H1": 3600, "H4": 14400, "D1": 86400,
                        }.get(timeframe.upper(), 3600)
                        # Warn only if the bar is older than 1.5× its
                        # timeframe interval (allows for weekend gaps
                        # and the brief moment after a bar closes before
                        # the next one prints). D1 on Sat/Sun is normal
                        # at up to ~60h old; H4 on Sun is normal at up
                        # to ~12h old.
                        _stale_threshold = max(_tf_sec * 1.5, 3600)
                        if _delta > _stale_threshold:
                            log.warning(
                                f"[MT5] Latest {timeframe} bar is "
                                f"{_delta:.0f}s old (>{_stale_threshold:.0f}s "
                                f"threshold) — stale data or market closed."
                            )
                        else:
                            # Within tolerance for this timeframe — log
                            # at DEBUG only so we don't spam INFO logs.
                            log.debug(
                                f"[MT5] Latest {timeframe} bar is "
                                f"{_delta:.0f}s old (within tolerance)."
                            )
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
            last_bar = df.index[-1]
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
        # Map timeframe to yfinance interval
        interval = self._tf_to_yfinance_interval(timeframe)
        if interval is None:
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
        # Metals — Yahoo uses futures tickers
        # Day 82 fix: GC=F (Gold futures) is delisted from yfinance.
        # For XAUUSD, rely on MT5 data instead (if trading on MT5).
        # If yfinance fetch is needed, return None so caller can handle gracefully.
        if s == "XAUUSD":
            # return "GC=F"   # DELISTED
            return None  # Let MT5 or other sources handle gold prices
        if s == "XAGUSD":
            return "SI=F"   # Silver futures
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
            log.error("[AlphaVantage] API key not set")
            return None

        # AV uses EUR/USD format (with slash)
        av_symbol = self._to_av_symbol(symbol)
        av_interval = self._tf_to_av_interval(timeframe)
        if av_interval is None:
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
            except Exception:
                skipped += 1
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
            log.error("[Polygon] API key not set")
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
            except Exception:
                skipped += 1
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
            log.error("[Finnhub] API key not set")
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
            except Exception:
                skipped += 1
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
            log.error("[TwelveData] API key not set")
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
            except Exception:
                skipped += 1
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
        Normalize symbol to MT5 format (e.g., "EURUSD").
        
        Converts:
          - "EUR/USD" → "EURUSD"
          - "EURUSD=X" → "EURUSD"
          - "EUR/USDT" → "EURUSD"
          - "EURUSD" → "EURUSD"
        """
        symbol = str(symbol).upper().strip()
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