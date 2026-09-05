"""Execution boundary used by live and historical replay.

Live execution remains `execution.execution_router.ExecutionRouter`.
Historical replay may replace only that broker-facing side with the canonical
adapter below; analysis, decision, risk and permission payloads remain live
pipeline outputs.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class ExecutionAdapter(ABC):
    @abstractmethod
    def open_trade(self, *, symbol: str, direction: str, entry_price: float,
                   sl: float, tp: float, lot: float, confidence: int,
                   **kwargs) -> dict:
        ...

    @abstractmethod
    def get_balance(self) -> float:
        ...


class MT5ExecutionAdapter(ExecutionAdapter):
    """Thin wrapper around the existing live/demo ExecutionRouter."""

    def __init__(self, execution_router):
        self._router = execution_router

    def open_trade(self, *, symbol: str, direction: str, entry_price: float,
                   sl: float, tp: float, lot: float, confidence: int,
                   **kwargs) -> dict:
        decision_result = {
            "symbol": symbol, "decision": direction, "entry": entry_price,
            "sl": sl, "tp": tp, "lot": lot, "confidence": confidence,
            **kwargs,
        }
        return self._router.execute(decision_result)

    def get_balance(self) -> float:
        raise NotImplementedError(
            "Live balance comes from AITrader._sync_balance()/MT5 account_info()."
        )


class HistoricalExecutionAdapter(ExecutionAdapter):
    """Legacy bar simulator adapter retained for existing backtests."""

    def __init__(self, broker_simulator):
        self._broker = broker_simulator

    def open_trade(self, *, symbol: str, direction: str, entry_price: float,
                   sl: float, tp: float, lot: float, confidence: int,
                   bar_time=None, **kwargs) -> dict:
        return self._broker.open_trade(
            symbol=symbol, direction=direction, entry_price=entry_price,
            sl=sl, tp=tp, lot=lot, bar_time=bar_time,
            confidence=confidence, **kwargs,
        )

    def check_exit(self, trade, high: float, low: float, close: float, bar_time):
        return self._broker.check_exit(trade, high, low, close, bar_time)

    def get_balance(self) -> float:
        return self._broker.get_balance()


class CanonicalReplayExecutionAdapter:
    """Named adapter for the live-mirroring execution boundary.

    It deliberately does not subclass the legacy `ExecutionAdapter` because
    the canonical replay contract requires the complete live `decision_result`
    plus replay timestamps and explicit market/cost assumptions. This prevents
    accidental use of the simplified legacy signature in research runs.
    """

    def __init__(self, canonical_adapter):
        from backtest.canonical_execution import CanonicalHistoricalExecutionAdapter
        if not isinstance(canonical_adapter, CanonicalHistoricalExecutionAdapter):
            raise TypeError("canonical_adapter must be CanonicalHistoricalExecutionAdapter")
        self._adapter = canonical_adapter

    def open_trade(self, *, decision_result: dict, signal_time: str,
                   entry_time: str, historical_bid: float,
                   pnl_multiplier: float):
        return self._adapter.open_trade(
            decision_result=decision_result,
            signal_time=signal_time,
            entry_time=entry_time,
            historical_bid=historical_bid,
            pnl_multiplier=pnl_multiplier,
        )

    def close_trade(self, trade_id: int, *, exit_time: str, exit_price: float,
                    reason: str):
        return self._adapter.close_trade(
            trade_id, exit_time=exit_time, exit_price=exit_price, reason=reason
        )

    @property
    def open_positions(self):
        return self._adapter.open_positions
