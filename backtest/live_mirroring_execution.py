"""Live-mirroring execution bridge.

Flow:
    live decision/risk/permission payload
        -> canonical replay execution adapter
        -> open PositionLifecycle
        -> historical position monitor
        -> deterministic close + P/L

This bridge intentionally contains no strategy logic.
"""
from __future__ import annotations

from backtest.canonical_execution import CanonicalHistoricalExecutionAdapter, PositionLifecycle
from backtest.canonical_position_monitor import Bar, HistoricalPositionMonitor


class LiveMirroringExecutionBridge:
    """Research-only replacement for the MT5 execution backend."""

    def __init__(self, adapter: CanonicalHistoricalExecutionAdapter,
                 monitor: HistoricalPositionMonitor):
        self.adapter = adapter
        self.monitor = monitor

    def execute_decision(self, *, decision_result: dict, signal_time: str,
                         entry_time: str, historical_bid: float,
                         pnl_multiplier: float) -> PositionLifecycle:
        """Pass the complete live decision result through unchanged."""
        return self.adapter.open_trade(
            decision_result=decision_result,
            signal_time=signal_time,
            entry_time=entry_time,
            historical_bid=historical_bid,
            pnl_multiplier=pnl_multiplier,
        )

    def advance_position(self, trade_id: int, bar: Bar):
        """Feed exactly one historical observation to close detection."""
        position = self.adapter.open_positions.get(int(trade_id))
        if position is None:
            raise KeyError(f"unknown open trade_id={trade_id}")
        closed = self.monitor.check_bar(position, bar)
        if closed is not None:
            self.adapter.open_positions.pop(int(trade_id), None)
        return closed

    def force_market_close(self, trade_id: int, *, timestamp: str, bid: float,
                           spread_pips: float = 0.0, slippage_pips: float = 0.0,
                           reason: str = "MARKET_CLOSE"):
        position = self.adapter.open_positions.get(int(trade_id))
        if position is None:
            raise KeyError(f"unknown open trade_id={trade_id}")
        closed = self.monitor.close_at_market(
            position, timestamp=timestamp, bid=bid,
            spread_pips=spread_pips, slippage_pips=slippage_pips, reason=reason,
        )
        self.adapter.open_positions.pop(int(trade_id), None)
        return closed
