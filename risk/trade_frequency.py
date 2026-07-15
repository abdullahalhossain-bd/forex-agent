"""
risk/trade_frequency.py — Trade Frequency Controller (Day 84+)
================================================================

WHY THIS EXISTS:
    The system can be either too conservative (0 trades/day) or too
    aggressive (50 trades/day = churn).  This controller enforces
    MIN_DAILY_TRADES / MAX_DAILY_TRADES bounds via two mechanisms:

    1. MAX cap — hard limit, no new trades once hit (avoids over-trading)
    2. MIN floor — diagnostic warning + adaptive threshold relaxation
       (if the system is falling short of MIN_DAILY_TRADES, the
       SignalScorer's threshold gets lowered automatically)

USAGE:
    from risk.trade_frequency import get_trade_frequency_controller

    ctrl = get_trade_frequency_controller()

    # Before placing a trade:
    if not ctrl.can_trade_now():
        return  # daily cap hit

    # After placing a trade:
    ctrl.record_trade(symbol="EURUSD")

    # End of day summary:
    summary = ctrl.daily_summary()
    # → {"trades_today": 3, "min_required": 5, "max_allowed": 10,
    #    "status": "BELOW_MIN", "recommendation": "lower_threshold"}
"""
from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Deque, Dict, List, Optional

from utils.logger import get_logger

log = get_logger("trade_frequency")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception as e:
        return default


# Defaults — overridable via .env
DEFAULT_MIN_DAILY_TRADES = 3
DEFAULT_MAX_DAILY_TRADES = 15


@dataclass
class TradeRecord:
    timestamp: float
    symbol: str
    direction: str


class TradeFrequencyController:
    """
    Tracks trades placed today and enforces min/max bounds.
    """

    def __init__(self):
        self._trades: Deque[TradeRecord] = deque(maxlen=500)
        self._min_daily = _env_int("MIN_DAILY_TRADES", DEFAULT_MIN_DAILY_TRADES)
        self._max_daily = _env_int("MAX_DAILY_TRADES", DEFAULT_MAX_DAILY_TRADES)
        self._last_status_check: Optional[datetime] = None
        log.info(
            f"[TradeFrequency] bounds: MIN={self._min_daily} MAX={self._max_daily} trades/day"
        )

    # ── Trade recording ────────────────────────────────────────

    def record_trade(self, symbol: str, direction: str, ts: float = None) -> None:
        self._trades.append(TradeRecord(
            timestamp=ts or datetime.now(timezone.utc).timestamp(),
            symbol=symbol,
            direction=direction,
        ))
        # Prune trades older than 48h (we keep a 24h rolling window + 24h buffer)
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=48)).timestamp()
        while self._trades and self._trades[0].timestamp < cutoff:
            self._trades.popleft()

    # ── Bounds checking ────────────────────────────────────────

    def trades_today(self, tz: str = "UTC") -> List[TradeRecord]:
        """Return trades placed since 00:00 today (in given tz)."""
        today = datetime.now(timezone.utc).date()
        return [t for t in self._trades
                if datetime.fromtimestamp(t.timestamp, tz=timezone.utc).date() == today]

    def trade_count_today(self) -> int:
        return len(self.trades_today())

    def can_trade_now(self) -> bool:
        """True if we haven't hit the daily max yet."""
        count = self.trade_count_today()
        if count >= self._max_daily:
            log.warning(
                f"[TradeFrequency] BLOCKED — {count}/{self._max_daily} trades today (daily cap hit)"
            )
            return False
        return True

    # ── Diagnostic status ──────────────────────────────────────

    def status(self) -> Dict:
        """Current status — used by dashboard and adaptive threshold logic."""
        count = self.trade_count_today()
        if count >= self._max_daily:
            status = "AT_MAX"
            recommendation = "block_new_trades"
        elif count < self._min_daily:
            # How far into the day are we?
            now = datetime.now(timezone.utc)
            day_progress = (now.hour * 60 + now.minute) / (24 * 60)
            # If >50% through the day and still below MIN, suggest lowering threshold
            if day_progress > 0.5:
                status = "BELOW_MIN_LATE"
                recommendation = "lower_threshold_aggressive"
            else:
                status = "BELOW_MIN_EARLY"
                recommendation = "lower_threshold_gentle"
        else:
            status = "IN_RANGE"
            recommendation = "hold"

        return {
            "trades_today":     count,
            "min_required":     self._min_daily,
            "max_allowed":      self._max_daily,
            "status":           status,
            "recommendation":   recommendation,
            "remaining_trades": max(0, self._max_daily - count),
        }

    def daily_summary(self) -> Dict:
        """End-of-day summary — for the daily review report."""
        s = self.status()
        s["trade_log"] = [
            {
                "time":      datetime.fromtimestamp(t.timestamp, tz=timezone.utc).isoformat(),
                "symbol":    t.symbol,
                "direction": t.direction,
            }
            for t in self.trades_today()
        ]
        return s

    # ── Adaptive threshold hint ────────────────────────────────

    def threshold_adjustment_hint(self) -> int:
        """Returns a suggested threshold delta for the SignalScorer.

        Returns:
            -10  if BELOW_MIN_LATE   (significantly under min, day more than half over)
            -5   if BELOW_MIN_EARLY  (under min, but day still has time)
            0    if IN_RANGE
            +10  if AT_MAX           (over-trading guard — raise bar)
        """
        s = self.status()
        return {
            "BELOW_MIN_LATE":   -10,
            "BELOW_MIN_EARLY":  -5,
            "IN_RANGE":          0,
            "AT_MAX":           +10,
        }.get(s["status"], 0)


# ── Singleton ─────────────────────────────────────────────────

_CTRL: Optional[TradeFrequencyController] = None


def get_trade_frequency_controller() -> TradeFrequencyController:
    global _CTRL
    if _CTRL is None:
        _CTRL = TradeFrequencyController()
    return _CTRL
