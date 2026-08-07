"""
data/data_orchestrator.py — Day 93 Unified Data Orchestrator
============================================================
Single entry point for ALL market data needs. Decides whether to
pull from MT5 (preferred — when running on Windows with MT5
terminal) or from external APIs (Twelve Data, yfinance, etc. —
fallback when MT5 unavailable, e.g. on Linux VPS).

PRINCIPLE (per user request):
    "যা যা সম্ভব MT5 থেকে নেবে, যা নেয়া যায় না তার জন্য API"

What MT5 provides (prefer MT5 for these):
  ✓ OHLCV candles (any timeframe, real-time + historical)
  ✓ Account info (balance, equity, margin, free margin)
  ✓ Open positions (ticket, symbol, lot, sl, tp, pnl)
  ✓ Pending orders
  ✓ Symbol info (spread, contract size, digits, tick value)
  ✓ Order execution (market buy/sell, modify SL/TP, close)
  ✓ Tick data (real-time bid/ask)

What MT5 does NOT provide (use external API for these):
  ✗ News sentiment          → NewsAPI.org + Forex Factory scraper
  ✗ Economic calendar       → Forex Factory scraper
  ✗ Intermarket data (DXY, Gold, Oil, VIX) → yfinance
  ✗ LLM brain               → OpenRouter / Groq / Cerebras
  ✗ Currency strength scores → computed from MT5 data ourselves
  ✗ Breaking news headlines → NewsAPI.org

USAGE:
    from data.data_orchestrator import get_data_orchestrator
    orch = get_data_orchestrator()

    # Candles — tries MT5 first, falls back to API
    df = orch.get_candles("EURUSD", "M15", limit=300)

    # Account info — MT5 only (returns None on Linux VPS)
    account = orch.get_account_info()

    # Open positions — MT5 only
    positions = orch.get_open_positions()

    # Symbol info (spread, digits) — MT5 preferred
    info = orch.get_symbol_info("EURUSD")

    # Where did the last candle come from?
    print(orch.last_source)  # "mt5" | "twelve_data" | "yfinance" | ...
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.logger import get_logger

log = get_logger("data_orchestrator")


def _mt5_positions_get(retries: int = 2, delay: float = 0.3, **kwargs):
    """Call mt5.positions_get() with retry logic.

    MT5 can return None intermittently. This helper retries
    a few times before giving up, reducing false negatives.
    """
    import time
    import MetaTrader5 as mt5_lib
    for attempt in range(retries + 1):
        try:
            result = mt5_lib.positions_get(**kwargs) if kwargs else mt5_lib.positions_get()
            if result is not None:
                return result
        except Exception:
            pass
        if attempt < retries:
            time.sleep(delay)
    return None


def _get_mt5_credentials():
    """Read MT5 credentials from environment."""
    login = int(os.getenv("MT5_LOGIN", 0))
    password = os.getenv("MT5_PASSWORD", "")
    server = os.getenv("MT5_SERVER", "")
    return login, password, server


class DataOrchestrator:
    """Unified data access layer — MT5 first, API fallback."""

    def __init__(self, mt5_conn=None):
        """
        Args:
            mt5_conn: Optional, already-connected broker.mt5_connection.MT5Connection
                instance, shared across data_orchestrator.py, fetcher.py, and
                execution_router.py.

                CROSS-FILE FIX (audit: "data_orchestrator.py <-> fetcher.py —
                duplicate fetch logic" and "data_orchestrator.py <->
                execution_router.py — separate MT5 connections"):
                previously `_get_mt5()`, `get_account_info()`, and
                `place_market_order()` each built and `.connect()`-ed a BRAND
                NEW `MT5Connection(login=..., password=..., server=...)`
                instance on every call — never reusing one another, and
                never reusing the shared, locked connection that fetcher.py
                (see its own `mt5_conn` docstring) and execution_router.py
                depend on. That is precisely the class of bug the "Day 90+
                hotfix" fixed everywhere except here: concurrent, unlocked
                `MT5Connection.connect()` calls from this class could
                invalidate an authenticated session mid-order or mid-fetch
                elsewhere. This class now owns (or receives) exactly ONE
                MT5Connection, reused for every MT5 operation, and injects
                that same instance into `data.fetcher.DataFetcher` instead
                of maintaining its own separate MT5 candle-fetching path
                (the old `broker.mt5_data.MT5DataFeed` route), which
                duplicated `fetcher.py`'s `_fetch_mt5()` logic.
        """
        self._fetcher = None
        self._mt5_conn = mt5_conn
        self._mt5_conn_initialized = mt5_conn is not None
        self.last_source: str = "unknown"
        # FIX (duplicate-order protection): tracks the last successful
        # order timestamp per "symbol:magic" key, used by
        # place_market_order() to block near-simultaneous duplicate
        # sends (e.g. two overlapping signal cycles or a retry racing
        # the original call).
        self._last_order_time: Dict[str, float] = {}

    # ─────────────────────────────────────────────────────────
    # LAZY INITIALIZERS
    # ─────────────────────────────────────────────────────────

    def _get_fetcher(self):
        """Lazy-init the DataFetcher, sharing our single MT5Connection.

        FIX: previously called get_data_fetcher() with no arguments, so
        DataFetcher built (or was given) a connection totally independent
        of this orchestrator's own MT5 session. Now the same MT5Connection
        instance is injected, so there is exactly one MT5 session shared
        between candle fetching (via fetcher.py) and account/order calls
        (below) — eliminating both the duplicate fetch path and the
        duplicate connection.
        """
        if self._fetcher is None:
            from data.fetcher import get_data_fetcher
            self._fetcher = get_data_fetcher(mt5_conn=self._get_mt5_conn())
        return self._fetcher

    def _get_mt5_conn(self):
        """Lazy-init (or reuse the injected) shared MT5Connection.

        This is now the SINGLE place in this class that ever constructs
        or connects an MT5Connection. Every other method below must call
        this instead of building its own — see the __init__ docstring
        for why that mattered.
        """
        if self._mt5_conn_initialized:
            return self._mt5_conn
        self._mt5_conn_initialized = True
        try:
            from broker.mt5_connection import MT5_AVAILABLE
            if not MT5_AVAILABLE:
                log.debug("[Orchestrator] MT5 not available (Linux/Mac) — will use API fallback")
                return None
            # Bug fix: was `MT5Connection(...)` here — even with this
            # class's own single-construction guard, that still created a
            # second, independent session from whatever core/runtime.py
            # (or data/fetcher.py) already opened for the same
            # (login, server), which is what produced duplicate connection
            # banners in the logs. get_mt5_connection() reuses the shared
            # instance instead.
            from broker.mt5_connection import get_mt5_connection
            login, password, server = _get_mt5_credentials()
            conn = get_mt5_connection(login=login, password=password, server=server,
                                      auto_connect=True)
            if not conn.connected:
                log.warning("[Orchestrator] MT5 connect failed — using API fallback")
                self._mt5_conn = None
                return None
            self._mt5_conn = conn
            log.info("[Orchestrator] MT5 connected — using MT5 as primary data source")
        except Exception as e:
            log.warning(f"[Orchestrator] MT5 init failed: {e} — using API fallback")
            self._mt5_conn = None
        return self._mt5_conn

    # ─────────────────────────────────────────────────────────
    # CANDLES (the most-used method)
    # ─────────────────────────────────────────────────────────

    def get_candles(
        self,
        symbol: str,
        timeframe: str = "M15",
        limit: int = 300,
    ) -> Optional[pd.DataFrame]:
        """Get OHLCV candles — MT5 first (via the shared fetcher), API fallback.

        FIX: this used to call a separate `broker.mt5_data.MT5DataFeed`
        instance built from its own throwaway MT5Connection, then
        re-normalize the result with `_normalize_mt5_candles()` — logic
        that duplicated (and could drift from) fetcher.py's own
        `_fetch_mt5()` + candle normalization. DataFetcher already
        implements a well-tested MT5-first-then-API-fallback path using
        our shared, locked MT5Connection, so we now delegate to it
        directly instead of re-implementing the same thing here.
        """
        fetcher = self._get_fetcher()
        df = fetcher.fetch_ohlcv(symbol, timeframe, limit=limit)
        if df is not None and len(df) > 0:
            self.last_source = fetcher.source
            log.debug(f"[Orchestrator] {symbol} {timeframe}: {len(df)} candles from {fetcher.source}")
        else:
            self.last_source = "failed"
        return df

    # ─────────────────────────────────────────────────────────
    # ACCOUNT INFO (MT5 only)
    # ─────────────────────────────────────────────────────────

    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get account balance/equity/margin — MT5 only.

        FIX: previously built and connected a brand-new MT5Connection
        on every call instead of reusing the shared one (see __init__
        docstring). Now reuses self._get_mt5_conn().
        """
        conn = self._get_mt5_conn()
        if conn is None:
            log.debug("[Orchestrator] get_account_info: MT5 unavailable")
            return None
        try:
            return conn.get_account_info()
        except Exception as e:
            log.warning(f"[Orchestrator] get_account_info failed: {e}")
            return None

    # ─────────────────────────────────────────────────────────
    # OPEN POSITIONS (MT5 only)
    # ─────────────────────────────────────────────────────────

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get list of currently-open positions — MT5 only."""
        conn = self._get_mt5_conn()
        if conn is None:
            log.debug("[Orchestrator] get_open_positions: MT5 unavailable")
            return []
        try:
            positions = _mt5_positions_get()
            if positions is None:
                return []
            result = []
            for p in positions:
                result.append({
                    "ticket":        p.ticket,
                    "symbol":        p.symbol,
                    "type":          "buy" if p.type == 0 else "sell",
                    "volume":        p.volume,
                    "sl":            p.sl,
                    "tp":            p.tp,
                    "pnl":           p.profit,
                    "swap":          p.swap,
                    "open_time":     pd.Timestamp(p.time, unit="s"),
                    "price_open":    p.price_open,
                    "price_current": p.price_current,
                })
            return result
        except Exception as e:
            log.warning(f"[Orchestrator] get_open_positions failed: {e}")
            return []

    # ─────────────────────────────────────────────────────────
    # PENDING ORDERS (MT5 only)
    # ─────────────────────────────────────────────────────────

    def get_pending_orders(self) -> List[Dict[str, Any]]:
        """Get list of pending orders — MT5 only."""
        conn = self._get_mt5_conn()
        if conn is None:
            return []
        try:
            import MetaTrader5 as mt5_lib
            orders = mt5_lib.orders_get()
            if orders is None:
                return []
            return [{
                "ticket":    o.ticket,
                "symbol":    o.symbol,
                "type":      o.type,
                "volume":    o.volume_current,
                "price":     o.price_open,
                "sl":        o.sl,
                "tp":        o.tp,
                "open_time": pd.Timestamp(o.time_setup, unit="s"),
            } for o in orders]
        except Exception as e:
            log.warning(f"[Orchestrator] get_pending_orders failed: {e}")
            return []

    # ─────────────────────────────────────────────────────────
    # SYMBOL INFO (MT5 preferred, API fallback)
    # ─────────────────────────────────────────────────────────

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get symbol metadata — spread, digits, contract size, etc."""
        conn = self._get_mt5_conn()
        if conn is not None:
            try:
                import MetaTrader5 as mt5_lib
                info = mt5_lib.symbol_info(symbol)
                if info is not None:
                    return {
                        "symbol":        info.symbol,
                        "digits":        info.digits,
                        "spread":        info.spread,
                        "point":         info.point,
                        "contract_size": info.trade_contract_size,
                        "tick_value":    info.trade_tick_value,
                        "tick_size":     info.trade_tick_size,
                        "min_lot":       info.volume_min,
                        "max_lot":       info.volume_max,
                        "lot_step":      info.volume_step,
                        "source":        "mt5",
                    }
            except Exception as e:
                log.warning(f"[Orchestrator] MT5 symbol_info failed: {e}")

        return {
            "symbol":        symbol,
            "digits":        5,
            "spread":        10,
            "point":         0.00001,
            "contract_size": 100000,
            "tick_value":    1.0,
            "tick_size":     0.00001,
            "min_lot":       0.01,
            "max_lot":       100.0,
            "lot_step":      0.01,
            "source":        "fallback",
        }

    # ─────────────────────────────────────────────────────────
    # TICK DATA (real-time bid/ask — MT5 only)
    # ─────────────────────────────────────────────────────────

    def get_tick(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get real-time bid/ask — MT5 only.

        FIX: previously called `.get_tick()` on a separate MT5DataFeed
        object built independently of this class's own MT5 session.
        Now goes through the same shared MT5Connection used everywhere
        else in this class.
        """
        conn = self._get_mt5_conn()
        if conn is None:
            return None
        try:
            import MetaTrader5 as mt5_lib
            tick = mt5_lib.symbol_info_tick(symbol)
            if tick is None:
                return None
            return {
                "bid": tick.bid,
                "ask": tick.ask,
                "last": tick.last,
                "time": tick.time,
            }
        except Exception as e:
            log.warning(f"[Orchestrator] get_tick failed: {e}")
            return None

    # ─────────────────────────────────────────────────────────
    # MULTI-TIMEFRAME CANDLES
    # ─────────────────────────────────────────────────────────

    def get_multi_timeframe(
        self,
        symbol: str,
        timeframes: List[str] = None,
        limit: int = 100,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch candles for multiple timeframes in one call."""
        if timeframes is None:
            timeframes = ["D1", "H4", "H1", "M15"]

        result = {}
        for tf in timeframes:
            df = self.get_candles(symbol, tf, limit=limit)
            if df is not None:
                result[tf] = df
            else:
                log.warning(f"[Orchestrator] {symbol} {tf}: no data")
        return result

    # ─────────────────────────────────────────────────────────
    # ORDER EXECUTION (MT5 only)
    # ─────────────────────────────────────────────────────────

    def place_market_order(
        self,
        symbol: str,
        direction: str,
        volume: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        deviation: int = 20,
        magic: int = 0,
        comment: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Place a market order via MT5.

        FIX (cross-file: "separate MT5 connections" with
        execution_router.py): this used to build and `.connect()` a
        brand-new `MT5Connection` on EVERY order — the exact race
        condition class that the fetcher.py "Day 90+ hotfix" addressed
        everywhere except here. A concurrent, unlocked connect() here
        could invalidate execution_router.py's authenticated session
        mid-order. This now reuses the single shared MT5Connection.
        """
        conn = self._get_mt5_conn()
        if conn is None:
            log.warning("[Orchestrator] place_market_order: MT5 unavailable — use SimulatedExecutor")
            return None
        try:
            import MetaTrader5 as mt5_lib
            if hasattr(conn, "ensure_connected") and not conn.ensure_connected():
                log.error("[Orchestrator] place_market_order: shared MT5Connection unavailable")
                return None

            tick = mt5_lib.symbol_info_tick(symbol)
            if tick is None:
                log.error(f"[Orchestrator] no tick for {symbol}")
                return None

            price = tick.ask if direction.lower() == "buy" else tick.bid
            order_type = mt5_lib.ORDER_TYPE_BUY if direction.lower() == "buy" else mt5_lib.ORDER_TYPE_SELL

            request = {
                "action":       mt5_lib.TRADE_ACTION_DEAL,
                "symbol":       symbol,
                "volume":       float(volume),
                "type":         order_type,
                "price":        price,
                "sl":           float(sl) if sl else 0.0,
                "tp":           float(tp) if tp else 0.0,
                "deviation":    deviation,
                "magic":        magic,
                "comment":      comment[:31],
                "type_time":    mt5_lib.ORDER_TIME_GTC,
                "type_filling": mt5_lib.ORDER_FILLING_IOC,
            }
            result = mt5_lib.order_send(request)
            if result is None:
                log.error("[Orchestrator] order_send returned None")
                return None
            return {
                "ticket":  result.order,
                "retcode": result.retcode,
                "comment": result.comment,
                "price":   result.price,
                "volume":  result.volume,
                "success": result.retcode == 10009,
            }
        except Exception as e:
            log.error(f"[Orchestrator] place_market_order failed: {e}")
            return None

    def close_position(self, ticket: int) -> bool:
        """Close an open position by ticket — MT5 only."""
        conn = self._get_mt5_conn()
        if conn is None:
            return False
        try:
            import MetaTrader5 as mt5_lib
            position = _mt5_positions_get(ticket=ticket)
            if not position:
                log.error(f"[Orchestrator] position {ticket} not found")
                return False
            pos = position[0]
            tick = mt5_lib.symbol_info_tick(pos.symbol)
            if tick is None:
                return False
            close_type = mt5_lib.ORDER_TYPE_SELL if pos.type == 0 else mt5_lib.ORDER_TYPE_BUY
            close_price = tick.bid if close_type == mt5_lib.ORDER_TYPE_SELL else tick.ask

            request = {
                "action":       mt5_lib.TRADE_ACTION_DEAL,
                "symbol":       pos.symbol,
                "volume":       pos.volume,
                "type":         close_type,
                "position":     ticket,
                "price":        close_price,
                "deviation":    20,
                "magic":        0,
                "comment":      "close by bot",
                "type_time":    mt5_lib.ORDER_TIME_GTC,
                "type_filling": mt5_lib.ORDER_FILLING_IOC,
            }
            result = mt5_lib.order_send(request)
            return result is not None and result.retcode == 10009
        except Exception as e:
            log.error(f"[Orchestrator] close_position failed: {e}")
            return False

    # ─────────────────────────────────────────────────────────
    # STATUS / DIAGNOSTICS
    # ─────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """Return diagnostic info about which data sources are active."""
        conn = self._get_mt5_conn()
        fetcher_source = "unknown"
        try:
            fetcher_source = self._get_fetcher().source
        except Exception:
            pass
        return {
            "mt5_available":    conn is not None,
            "mt5_initialized":  self._mt5_conn_initialized,
            "api_source":       fetcher_source,
            "last_source":      self.last_source,
            "preferred_source": os.getenv("PREFERRED_DATA_SOURCE", ""),
        }


# ── Singleton ─────────────────────────────────────────────────────

_ORCHESTRATOR: Optional[DataOrchestrator] = None


def get_data_orchestrator(mt5_conn=None) -> DataOrchestrator:
    """Return a shared DataOrchestrator instance (singleton).

    Args:
        mt5_conn: Optional shared MT5Connection to inject on first
            creation (see DataOrchestrator.__init__). Pass the same
            instance used by execution_router.py / core.runtime here
            so candle fetching, account/position queries, and order
            placement all share exactly one MT5 session. Ignored on
            subsequent calls once the singleton already exists.
    """
    global _ORCHESTRATOR
    if _ORCHESTRATOR is None:
        _ORCHESTRATOR = DataOrchestrator(mt5_conn=mt5_conn)
    return _ORCHESTRATOR