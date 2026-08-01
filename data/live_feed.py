"""
data/live_feed.py — MT5 Live Tick Intelligence (Day 81+)
=========================================================

Single source of truth for real-time MT5 market data:
  - Bid/Ask/Spread
  - Tick velocity (ticks per second — proxy for market activity)
  - Tick direction pressure (buyers vs sellers over last N ticks)
  - Spread explosion detection (current spread vs N-period median)
  - Liquidity condition classification (NORMAL / THIN / EXPLOSIVE)

FIX (Day 82+):
  - spread_pips == 0 আর CLOSED মানে না — MT5 cached/off-hours data-তে
    spread=0 আসতে পারে কিন্তু market open থাকে। তাই THIN দেওয়া হয়েছে।
  - is_safe_to_trade() এ TEST_MODE-এ শুধু trade_mode==0 চেক করে।
    velocity/spread দিয়ে block হয় না।
  - ABSOLUTE_SAFETY=false হলে সব check skip হয়।
"""
from __future__ import annotations

import os
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Dict, Optional

from config import TEST_MODE
from utils.logger import get_logger

log = get_logger("live_feed")

# No-op context manager used when no shared/locked MT5Connection has
# been injected into LiveFeed — see LiveFeed._mt5_lock().
_NULL_CONTEXT = nullcontext()

# Guard MT5 import so this module loads on Linux/Mac (for unit tests)
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False


# ── Per-symbol spread thresholds (pips) ───────────────────────
# These are used only in PRODUCTION mode (TEST_MODE=false).
# Raise a symbol's threshold if you see false EXPLOSIVE blocks.
SPREAD_LIMITS_PIPS: Dict[str, float] = {
    "EURUSD": 3.0,
    "GBPUSD": 4.0,
    "USDJPY": 3.0,
    "AUDUSD": 3.0,
    "USDCAD": 4.0,
    "XAUUSD": 50.0,   # Gold — wide spread is normal
    "XAGUSD": 10.0,
    "DEFAULT": 10.0,
}


def _get_spread_limit(symbol: str) -> float:
    return SPREAD_LIMITS_PIPS.get(symbol.upper(), SPREAD_LIMITS_PIPS["DEFAULT"])


# ── Tick snapshot ─────────────────────────────────────────────

@dataclass
class TickSnapshot:
    """One moment-in-time view of a symbol's live market state."""
    symbol: str
    bid: float
    ask: float
    spread_pips: float
    last_price: float
    timestamp: str  # ISO 8601 UTC

    # Velocity — ticks per second over the last 60s window
    tick_velocity: float = 0.0
    # Direction pressure — +1.0 = all buyers, -1.0 = all sellers, 0 = balanced
    direction_pressure: float = 0.0
    # Liquidity classification
    liquidity: str = "NORMAL"  # NORMAL / THIN / EXPLOSIVE / CLOSED
    # Spread vs median (multiple, e.g. 2.5x = current spread is 2.5x normal)
    spread_multiple: float = 1.0

    # Internal: raw spread history for median calc (kept out of __repr__)
    _raw_spread_history: tuple = field(default_factory=tuple, repr=False)

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def is_tradeable(self) -> bool:
        """Quick check — should we even consider trading this symbol right now?"""
        return self.liquidity in ("NORMAL", "THIN") and self.spread_multiple < 10.0

    def to_dict(self) -> Dict:
        return {
            "symbol":              self.symbol,
            "bid":                 self.bid,
            "ask":                 self.ask,
            "spread_pips":         self.spread_pips,
            "last_price":          self.last_price,
            "timestamp":           self.timestamp,
            "tick_velocity":       round(self.tick_velocity, 2),
            "direction_pressure":  round(self.direction_pressure, 2),
            "liquidity":           self.liquidity,
            "spread_multiple":     round(self.spread_multiple, 2),
            "is_tradeable":        self.is_tradeable,
        }


# ── Live feed ─────────────────────────────────────────────────

class LiveFeed:
    """
    Real-time MT5 tick intelligence layer.

    Maintains a per-symbol rolling buffer of the last N ticks so we can
    compute velocity, pressure, and spread-median without re-fetching.
    """

    BUFFER_SIZE = 120  # ~2 minutes of ticks at 1/sec

    def __init__(self, buffer_size: int = None, mt5_conn=None):
        """
        Args:
            mt5_conn: Optional, already-connected broker.mt5_connection.MT5Connection
                instance, shared with data.fetcher.DataFetcher and
                data.data_orchestrator.DataOrchestrator.

                CROSS-FILE FIX (audit: "live_feed.py <-> mt5_connection.py —
                MT5 lock bypass"): this class used to call
                mt5.symbol_info_tick()/mt5.symbol_info() directly against the
                global MetaTrader5 module on every get_snapshot() — completely
                bypassing the lock that MT5Connection/execution_router.py were
                hardened to enforce (the same "Day 90+ hotfix" fetcher.py and
                data_orchestrator.py were fixed for). A tick poll here could
                race an in-flight, lock-protected order elsewhere. When
                mt5_conn is provided and exposes a `.lock`, every raw MT5 call
                below now happens inside that lock. If not provided, this
                falls back to the old unlocked behavior (so existing callers
                don't break) but logs a one-time warning so the gap is visible
                instead of silent.
        """
        self._buffers: Dict[str, Deque[Dict]] = {}
        self._size = buffer_size or self.BUFFER_SIZE
        self._last_fetch: Dict[str, float] = {}  # symbol → epoch
        self._mt5_conn = mt5_conn
        self._warned_unlocked = False

    # ── Internal: locked MT5 access ─────────────────────────────

    def _mt5_lock(self):
        """Return the shared connection's lock as a context manager, or a
        no-op context manager if no shared connection was injected."""
        lock = getattr(self._mt5_conn, "lock", None) if self._mt5_conn is not None else None
        if lock is not None:
            return lock
        if not self._warned_unlocked:
            self._warned_unlocked = True
            log.warning(
                "[LiveFeed] No shared MT5Connection injected — tick polling is "
                "UNLOCKED and can race order execution elsewhere. Pass "
                "mt5_conn=... (see get_live_feed()) to share the locked "
                "connection used by fetcher.py / data_orchestrator.py."
            )
        return _NULL_CONTEXT

    # ── Public API ─────────────────────────────────────────────

    def get_snapshot(self, symbol: str) -> Optional[TickSnapshot]:
        """Fetch a fresh tick + compute intelligence metrics. None if MT5 down."""
        if not MT5_AVAILABLE:
            log.debug("[LiveFeed] MetaTrader5 not installed — returning None")
            return None

        if self._mt5_conn is not None and hasattr(self._mt5_conn, "ensure_connected"):
            if not self._mt5_conn.ensure_connected():
                log.debug(f"[LiveFeed] shared MT5Connection unavailable for {symbol}")
                return None

        with self._mt5_lock():
            tick = mt5.symbol_info_tick(symbol)
            info = mt5.symbol_info(symbol) if tick is not None else None

        if tick is None or tick.time == 0:
            log.debug(f"[LiveFeed] No tick for {symbol}")
            return None

        digits = info.digits if info else 5
        spread_points = tick.ask - tick.bid
        spread_pips = round(spread_points * (10 ** (digits - 1)), 2) if digits else 0

        # Push to rolling buffer
        now = time.time()
        record = {
            "time":   now,
            "bid":    tick.bid,
            "ask":    tick.ask,
            "last":   tick.last if tick.last else (tick.bid + tick.ask) / 2,
            "spread": spread_pips,
        }
        buf = self._buffers.setdefault(symbol, deque(maxlen=self._size))
        buf.append(record)
        self._last_fetch[symbol] = now

        # Compute intelligence metrics from the buffer
        velocity       = self._compute_velocity(buf, now)
        pressure       = self._compute_pressure(buf)
        spread_median  = self._compute_spread_median(buf)
        spread_multiple = (spread_pips / spread_median) if spread_median > 0 else 1.0
        liquidity      = self._classify_liquidity(spread_pips, spread_multiple, velocity)

        return TickSnapshot(
            symbol=symbol,
            bid=tick.bid,
            ask=tick.ask,
            spread_pips=spread_pips,
            last_price=record["last"],
            timestamp=datetime.fromtimestamp(tick.time, tz=timezone.utc).isoformat(),
            tick_velocity=velocity,
            direction_pressure=pressure,
            liquidity=liquidity,
            spread_multiple=spread_multiple,
        )

    def get_multi_snapshot(self, symbols: list[str]) -> Dict[str, TickSnapshot]:
        """Fetch snapshots for many symbols at once — used by the scanner."""
        out = {}
        for sym in symbols:
            snap = self.get_snapshot(sym)
            if snap is not None:
                out[sym] = snap
        return out

    # ── Intelligence calculators ───────────────────────────────

    @staticmethod
    def _compute_velocity(buf: Deque[Dict], now: float, window_sec: float = 60.0) -> float:
        """Ticks per second over the last `window_sec` seconds."""
        if len(buf) < 2:
            return 0.0
        cutoff = now - window_sec
        recent = [r for r in buf if r["time"] >= cutoff]
        if len(recent) < 2:
            return 0.0
        elapsed = recent[-1]["time"] - recent[0]["time"]
        if elapsed <= 0:
            return 0.0
        return len(recent) / elapsed

    @staticmethod
    def _compute_pressure(buf: Deque[Dict], lookback: int = 20) -> float:
        """Direction pressure over last N ticks. +1 = all buyers, -1 = all sellers."""
        if len(buf) < 2:
            return 0.0
        recent = list(buf)[-lookback:]
        buyers = 0
        sellers = 0
        for i in range(1, len(recent)):
            if recent[i]["last"] > recent[i - 1]["last"]:
                buyers += 1
            elif recent[i]["last"] < recent[i - 1]["last"]:
                sellers += 1
        total = buyers + sellers
        if total == 0:
            return 0.0
        return (buyers - sellers) / total

    @staticmethod
    def _compute_spread_median(buf: Deque[Dict]) -> float:
        """Median spread over the buffer (used to detect spread explosions)."""
        if not buf:
            return 0.0
        spreads = sorted(r["spread"] for r in buf)
        n = len(spreads)
        if n % 2 == 1:
            return spreads[n // 2]
        return (spreads[n // 2 - 1] + spreads[n // 2]) / 2

    @staticmethod
    def _classify_liquidity(spread_pips: float, spread_multiple: float, velocity: float) -> str:
        """Classify current liquidity condition.

        DAY 82+ FIX:
          spread_pips == 0 আর CLOSED না। MT5 off-hours-এ cached zero spread
          দিতে পারে কিন্তু market আসলে open থাকে। তাই THIN দেওয়া হলো —
          is_safe_to_trade() trade_mode দিয়ে truly closed যাচাই করবে।

        NORMAL    — typical spread, normal tick activity
        THIN      — low velocity OR zero spread (off-hours / light session)
        EXPLOSIVE — spread blew up >5x normal (news just hit) → DO NOT TRADE
        """
        if spread_multiple >= 5.0:
            return "EXPLOSIVE"
        if spread_pips == 0 or velocity < 0.1:
            return "THIN"
        return "NORMAL"

    # ── Hard safety gates (used by ABSOLUTE_SAFETY) ────────────

    def is_safe_to_trade(
        self,
        symbol: str,
        max_spread_multiple: float = None,
    ) -> tuple[bool, str]:
        """
        ABSOLUTE_SAFETY gate — returns (safe, reason).

        DAY 82+ FIX:
          TEST_MODE=true のとき: trade_mode==0 だけ block、
          spread/velocity/liquidity は無視する。
          これにより off-hours でも verification trade が通る。

          PRODUCTION (TEST_MODE=false): symbol のデフォルト spread 閾値を
          使用し、EXPLOSIVE と over-limit spread をブロックする。
        """
        if max_spread_multiple is None:
            max_spread_multiple = 10.0

        _test_mode = TEST_MODE

        # ── TEST_MODE: trade_mode チェックのみ ──────────────────
        if _test_mode:
            if not MT5_AVAILABLE:
                # MT5 なければ pass — paper/simulation で動作
                return True, "OK (TEST_MODE, MT5 unavailable)"
            try:
                with self._mt5_lock():
                    info = mt5.symbol_info(symbol)
                if info is not None and info.trade_mode == 0:
                    log.warning(
                        f"[ABSOLUTE_SAFETY] {symbol} trade_mode=0 — "
                        f"truly disabled by broker"
                    )
                    return False, f"{symbol} trade_mode=0 (disabled by broker)"
                # trade_mode=4 (Full) বা অন্য যেকোনো — allow
                return True, f"OK (TEST_MODE, trade_mode={getattr(info, 'trade_mode', '?')})"
            except Exception as e:
                log.warning(f"[ABSOLUTE_SAFETY] trade_mode check failed: {e} — allowing")
                return True, f"OK (TEST_MODE, trade_mode check error: {e})"

        # ── PRODUCTION MODE ─────────────────────────────────────
        snap = self.get_snapshot(symbol)
        if snap is None:
            return False, "MT5 unavailable or no tick data"

        if snap.liquidity == "EXPLOSIVE":
            return False, (
                f"{symbol} spread exploded "
                f"({snap.spread_multiple:.1f}x normal)"
            )

        sym_spread_limit = _get_spread_limit(symbol)
        if snap.spread_pips > sym_spread_limit:
            return False, (
                f"{symbol} spread {snap.spread_pips:.1f} pips "
                f"exceeds limit {sym_spread_limit:.1f} pips"
            )

        return True, "OK"


# ── Singleton accessor ────────────────────────────────────────

_FEED: Optional[LiveFeed] = None


def get_live_feed(mt5_conn=None) -> LiveFeed:
    """Return a shared LiveFeed instance (singleton).

    Args:
        mt5_conn: Optional shared MT5Connection (see LiveFeed.__init__)
            to inject on first creation — pass the same instance used
            by data.fetcher.get_data_fetcher() / data_orchestrator.py /
            execution_router.py so tick polling shares their lock
            instead of racing it. Ignored on subsequent calls once the
            singleton already exists.
    """
    global _FEED
    if _FEED is None:
        _FEED = LiveFeed(mt5_conn=mt5_conn)
    return _FEED