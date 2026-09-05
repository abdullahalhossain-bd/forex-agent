"""Trade frequency controller with injectable live/replay clock."""
from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Deque, Dict, List, Optional

from utils.logger import get_logger

log = get_logger("trade_frequency")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


DEFAULT_MIN_DAILY_TRADES = 3
DEFAULT_MAX_DAILY_TRADES = 50


@dataclass
class TradeRecord:
    timestamp: float
    symbol: str
    direction: str


class TradeFrequencyController:
    """Tracks trade frequency using an injected clock.

    Live callers use LiveClock implicitly. Historical callers pass ReplayClock;
    no wall-clock value is consulted for day/session/cutoff decisions.
    """

    def __init__(self, clock=None, *, min_daily: Optional[int] = None,
                 max_daily: Optional[int] = None):
        self.clock = clock
        self._trades: Deque[TradeRecord] = deque(maxlen=500)
        self._min_daily = _env_int("MIN_DAILY_TRADES", DEFAULT_MIN_DAILY_TRADES) if min_daily is None else int(min_daily)
        self._max_daily = _env_int("MAX_DAILY_TRADES", DEFAULT_MAX_DAILY_TRADES) if max_daily is None else int(max_daily)
        self._last_status_check: Optional[datetime] = None

    def _now(self) -> datetime:
        if self.clock is not None:
            return self.clock.now()
        return datetime.now(timezone.utc)

    def record_trade(self, symbol: str, direction: str, ts: float = None) -> None:
        now = self._now()
        timestamp = float(ts) if ts is not None else now.timestamp()
        self._trades.append(TradeRecord(timestamp=timestamp, symbol=symbol, direction=direction))
        cutoff = now.timestamp() - timedelta(hours=48).total_seconds()
        while self._trades and self._trades[0].timestamp < cutoff:
            self._trades.popleft()

    def trades_today(self, tz: str = "UTC") -> List[TradeRecord]:
        # `tz` is retained for API compatibility; replay is always UTC.
        today = self._now().date()
        return [t for t in self._trades
                if datetime.fromtimestamp(t.timestamp, tz=timezone.utc).date() == today]

    def trade_count_today(self) -> int:
        return len(self.trades_today())

    def _get_session_aware_max_trades(self) -> int:
        hour = self._now().hour
        if 8 <= hour < 21:
            return min(self._max_daily, 50)
        if 0 <= hour < 8:
            return min(self._max_daily, 30)
        return min(self._max_daily, 35)

    def can_trade_now(self) -> bool:
        count = self.trade_count_today()
        session_max = self._get_session_aware_max_trades()
        if count >= session_max:
            log.warning(f"[TradeFrequency] BLOCKED — {count}/{session_max} trades today")
            return False
        return True

    def status(self) -> Dict:
        count = self.trade_count_today()
        session_max = self._get_session_aware_max_trades()
        now = self._now()
        if count >= session_max:
            status, recommendation = "AT_MAX", "block_new_trades"
        elif count < self._min_daily:
            day_progress = (now.hour * 60 + now.minute) / (24 * 60)
            if day_progress > 0.5:
                status, recommendation = "BELOW_MIN_LATE", "lower_threshold_aggressive"
            else:
                status, recommendation = "BELOW_MIN_EARLY", "lower_threshold_gentle"
        else:
            status, recommendation = "IN_RANGE", "hold"
        return {
            "trades_today": count,
            "min_required": self._min_daily,
            "max_allowed": session_max,
            "status": status,
            "recommendation": recommendation,
            "remaining_trades": max(0, session_max - count),
        }

    def daily_summary(self) -> Dict:
        s = self.status()
        s["trade_log"] = [
            {"time": datetime.fromtimestamp(t.timestamp, tz=timezone.utc).isoformat(),
             "symbol": t.symbol, "direction": t.direction}
            for t in self.trades_today()
        ]
        return s

    def threshold_adjustment_hint(self) -> int:
        return {
            "BELOW_MIN_LATE": -10,
            "BELOW_MIN_EARLY": -5,
            "IN_RANGE": 0,
            "AT_MAX": +10,
        }.get(self.status()["status"], 0)


_CTRL: Optional[TradeFrequencyController] = None


def get_trade_frequency_controller(clock=None) -> TradeFrequencyController:
    """Return the singleton; a supplied clock replaces it when necessary."""
    global _CTRL
    if _CTRL is None or (clock is not None and _CTRL.clock is not clock):
        _CTRL = TradeFrequencyController(clock=clock)
    return _CTRL
