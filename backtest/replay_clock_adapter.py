"""Replay-clock adapter for time-based risk gates.

Provides a tiny compatibility layer so legacy risk controllers can consume
historical replay time without reading the wall clock.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from core.clock import ReplayClock


class ReplayTradeFrequencyController:
    """Deterministic trade-frequency state machine for historical replay."""

    def __init__(self, clock: ReplayClock, max_daily: int = 50, min_daily: int = 3):
        self.clock = clock
        self.max_daily = int(max_daily)
        self.min_daily = int(min_daily)
        self._trades: list[tuple[datetime, str, str]] = []

    def record_trade(self, symbol: str, direction: str, ts: Optional[datetime] = None) -> None:
        t = ts or self.clock.now()
        self._trades.append((t, symbol, direction))

    def _today(self):
        day = self.clock.current_date()
        return [t for t in self._trades if t[0].date() == day]

    def trade_count_today(self) -> int:
        return len(self._today())

    def can_trade_now(self) -> bool:
        return self.trade_count_today() < self.max_daily

    def status(self) -> dict:
        count = self.trade_count_today()
        progress = (self.clock.now().hour * 60 + self.clock.now().minute) / 1440.0
        if count >= self.max_daily:
            status, recommendation = "AT_MAX", "block_new_trades"
        elif count < self.min_daily:
            status = "BELOW_MIN_LATE" if progress > 0.5 else "BELOW_MIN_EARLY"
            recommendation = "lower_threshold_aggressive" if progress > 0.5 else "lower_threshold_gentle"
        else:
            status, recommendation = "IN_RANGE", "hold"
        return {
            "trades_today": count,
            "min_required": self.min_daily,
            "max_allowed": self.max_daily,
            "status": status,
            "recommendation": recommendation,
            "remaining_trades": max(0, self.max_daily - count),
        }

    def threshold_adjustment_hint(self) -> int:
        return {
            "BELOW_MIN_LATE": -10,
            "BELOW_MIN_EARLY": -5,
            "IN_RANGE": 0,
            "AT_MAX": 10,
        }.get(self.status()["status"], 0)
