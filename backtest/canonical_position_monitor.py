"""Historical position monitoring for live-mirroring replay.

Only the execution-side lifecycle lives here. The caller supplies one
historical bar at a time, so no future observation is consulted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from backtest.canonical_execution import PositionLifecycle, resolve_intrabar, mark_close


@dataclass(frozen=True)
class Bar:
    timestamp: str
    high: float
    low: float
    close: float
    bid: Optional[float] = None
    spread_pips: Optional[float] = None


class HistoricalPositionMonitor:
    """Deterministic close detector and P/L calculator."""

    def __init__(self, *, pip_size: float, intrabar_policy: str = "AMBIGUOUS_INTRABAR",
                 commission_per_lot: float = 0.0):
        if pip_size <= 0:
            raise ValueError("pip_size must be positive")
        self.pip_size = float(pip_size)
        self.intrabar_policy = intrabar_policy
        self.commission_per_lot = float(commission_per_lot)

    def check_bar(self, position: PositionLifecycle, bar: Bar) -> Optional[PositionLifecycle]:
        """Check only this historical bar for SL/TP touches."""
        if position.status != "OPEN":
            return position
        reason = resolve_intrabar(
            direction=position.direction, bar_high=float(bar.high), bar_low=float(bar.low),
            stop_loss=float(position.stop_loss), take_profit=float(position.take_profit),
            policy=self.intrabar_policy,
        )
        if reason is None:
            return None
        if reason == "AMBIGUOUS_INTRABAR":
            raise ValueError(
                f"AMBIGUOUS_INTRABAR: trade_id={position.trade_id} timestamp={bar.timestamp}"
            )
        exit_price = float(position.stop_loss if reason == "SL" else position.take_profit)
        return self._close(position, bar.timestamp, exit_price, reason)

    def replay(self, position: PositionLifecycle, bars: Iterable[Bar]) -> Optional[PositionLifecycle]:
        for bar in bars:
            closed = self.check_bar(position, bar)
            if closed is not None:
                return closed
        return None

    def close_at_market(self, position: PositionLifecycle, *, timestamp: str,
                        bid: float, spread_pips: float = 0.0,
                        slippage_pips: float = 0.0,
                        reason: str = "MARKET_CLOSE") -> PositionLifecycle:
        """Close using the supplied historical bid, never wall-clock/current data."""
        if position.direction.upper() == "BUY":
            exit_price = float(bid - (spread_pips + slippage_pips) * self.pip_size)
        elif position.direction.upper() == "SELL":
            exit_price = float(bid + (spread_pips + slippage_pips) * self.pip_size)
        else:
            raise ValueError(f"Unsupported direction: {position.direction}")
        return self._close(position, timestamp, exit_price, reason)

    def _close(self, position: PositionLifecycle, timestamp: str,
               exit_price: float, reason: str) -> PositionLifecycle:
        pnl = self._pnl_before_costs(position, exit_price)
        commission = float(position.filled_lot) * self.commission_per_lot
        position.commission_usd += commission
        return mark_close(position, timestamp, exit_price, reason, pnl - commission)

    @staticmethod
    def _pnl_before_costs(position: PositionLifecycle, exit_price: float) -> float:
        if position.direction.upper() == "BUY":
            return (exit_price - position.fill_price) * position.filled_lot * position.pnl_multiplier
        if position.direction.upper() == "SELL":
            return (position.fill_price - exit_price) * position.filled_lot * position.pnl_multiplier
        raise ValueError(f"Unsupported direction: {position.direction}")
