"""
core/execution_adapter.py — ExecutionAdapter abstraction (execution-parity refactor).

The trading engine (AITrader.evaluate_decision_core) produces a TradeDecision-
shaped dict (dec_out/risk_out/perm_out). What happens to that decision — a
real MT5 order, a demo MT5 order, or a simulated fill against historical
OHLC — is the ONLY thing allowed to differ between modes, and this module
is where that boundary lives.

- MT5ExecutionAdapter wraps the EXISTING execution.execution_router.
  ExecutionRouter, which is already mode-agnostic between mt5_demo and
  mt5_live (single class, `self.mode` flag, confirmed by reading the code:
  ExecutionRouter.__init__ branches on self.mode == "mt5_live" only for
  the extra real-money safety gate, not for a different order-placement
  code path). This class does not reimplement order placement — it is a
  thin named wrapper so the abstraction the architecture calls for
  actually exists as a class, not just a convention.

- HistoricalExecutionAdapter wraps backtest.broker_sim.BrokerSimulator,
  the only component in the repo that can replay bar high/low SL/TP
  touches against historical OHLC. It is NOT execution.simulated_executor.
  SimulatedExecutor (that one is a live-pipeline dry-run smoke test, fills
  instantly at a fabricated price, no OHLC awareness — wrong tool for
  historical replay, kept for its own purpose, not merged in here).
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class ExecutionAdapter(ABC):
    """Contract: given a TradeDecision (action, entry, sl, tp, lot,
    confidence), return a fill/rejection result. Never inspects mode."""

    @abstractmethod
    def open_trade(self, *, symbol: str, direction: str, entry_price: float,
                   sl: float, tp: float, lot: float, confidence: int,
                   **kwargs) -> dict:
        ...

    @abstractmethod
    def get_balance(self) -> float:
        ...


class MT5ExecutionAdapter(ExecutionAdapter):
    """Wraps the existing ExecutionRouter (mode='mt5_demo' or 'mt5_live').
    Demo and Real are NOT two adapters — they are the same adapter with a
    different `.mode` string, exactly as ExecutionRouter already
    implements it. Do not create a separate DemoExecutionAdapter /
    RealExecutionAdapter — that would reintroduce the duplication this
    refactor removes.
    """

    def __init__(self, execution_router):
        self._router = execution_router

    def open_trade(self, *, symbol: str, direction: str, entry_price: float,
                   sl: float, tp: float, lot: float, confidence: int,
                   **kwargs) -> dict:
        # BUGFIX (execution-parity wiring): ExecutionRouter.execute() reads
        # decision_result.get("decision") for BUY/SELL — NOT "action". This
        # key was wrong since the adapter was first written, which is why
        # it was never wired anywhere: had it been wired with "action", the
        # router's hard gate (`decision_result.get("decision") not in
        # (BUY, SELL)`) would have silently treated every trade as
        # "no action" and never executed a single order.
        decision_result = {
            "symbol": symbol, "decision": direction, "entry": entry_price,
            "sl": sl, "tp": tp, "lot": lot, "confidence": confidence,
            **kwargs,
        }
        return self._router.execute(decision_result)

    def get_balance(self) -> float:
        # ExecutionRouter delegates balance to the live/demo MT5 account
        # (see core/trader.py._sync_balance) — not tracked here directly.
        raise NotImplementedError(
            "Live balance comes from AITrader._sync_balance() / MT5 "
            "account_info(), not from the execution adapter. Call "
            "trader.balance instead."
        )


class HistoricalExecutionAdapter(ExecutionAdapter):
    """Wraps backtest.broker_sim.BrokerSimulator — bar-based SL/TP touch
    detection against historical OHLC. This is the ONLY execution-side
    difference the architecture permits for backtest mode."""

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
        """Historical-only: sweep the current bar's high/low against an
        open trade's SL/TP. Live/Real never call this — MT5 itself detects
        SL/TP touches server-side, which is why this method is NOT part
        of the shared ExecutionAdapter contract, only on this subclass."""
        return self._broker.check_exit(trade, high, low, close, bar_time)

    def get_balance(self) -> float:
        return self._broker.get_balance()
