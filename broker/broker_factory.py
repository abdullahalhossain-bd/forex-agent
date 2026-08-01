# broker/broker_factory.py — Abstract broker interface + factory pattern
# =============================================================================
# Ported from: https://github.com/bruh7463/forex_bot/blob/master/broker/
# Original: broker/base.py (BrokerBase) + broker/executor.py (get_executor factory)
# Original author: bruh7463 — educational project
#
# Abstract broker interface that all broker executors must implement, plus a
# factory function that returns the correct executor based on config.
#
# This is a CLEANER version of the existing broker/ module's implicit interface.
# The existing project has broker/mt5_connection.py, broker/order_manager.py,
# etc. but no formal abstract base class. This adds one, so future broker
# integrations (e.g., a paper-trading broker, a backtest broker) can be
# dropped in without modifying the orchestrator.
#
# The factory pattern lets you switch brokers by changing one config setting:
#     from broker.broker_factory import get_executor
#     executor = get_executor("mt5")  # or "paper", "simulated", etc.
# =============================================================================

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from utils.logger import get_logger

log = get_logger("broker_factory")


class BrokerBase(ABC):
    """
    Abstract base class for broker integrations.
    All broker executors must implement these methods.
    """

    @abstractmethod
    def get_account_balance(self) -> float:
        """Return current account balance in account currency."""
        pass

    @abstractmethod
    def get_open_trades(self) -> list[dict[str, Any]]:
        """
        Return list of open trades. Each trade dict has:
            trade_id, pair, side ("BUY"/"SELL"), size, entry_price,
            current_price, stop_loss, take_profit, profit_loss, open_time
        """
        pass

    @abstractmethod
    def place_order(
        self,
        pair: str,
        signal: str,
        size: float,
        stop_loss: float,
        take_profit: float,
    ) -> dict[str, Any]:
        """
        Place a market order. Returns dict with:
            order_id, trade_id, status ("filled"/"pending"/"rejected"),
            filled_price, message
        """
        pass

    @abstractmethod
    def close_trade(self, trade_id: str) -> bool:
        """Close a trade by ID. Returns True on success."""
        pass

    # ── Helper methods (non-abstract) ────────────────────────────────────────

    def get_open_trade_for_pair(self, pair: str) -> Optional[dict]:
        """Get the open trade for a specific pair, or None."""
        for trade in self.get_open_trades():
            if trade.get("pair") == pair:
                return trade
        return None

    def has_open_trade(self, pair: str) -> bool:
        """Check if there's an open trade for a pair."""
        return self.get_open_trade_for_pair(pair) is not None

    def get_total_profit_loss(self) -> float:
        """Get total unrealized P/L across all open trades."""
        return sum(t.get("profit_loss", 0) for t in self.get_open_trades())

    def close_all_trades(self) -> list[str]:
        """Close ALL open trades. Returns list of closed trade IDs."""
        closed = []
        for trade in list(self.get_open_trades()):
            trade_id = trade.get("trade_id")
            if trade_id and self.close_trade(trade_id):
                closed.append(trade_id)
        log.info(f"Closed {len(closed)} trades: {closed}")
        return closed


# ── Paper trading broker (always available — no MT5 needed) ──────────────────

class PaperBroker(BrokerBase):
    """
    Paper trading broker — simulates order fills at the current price.
    No real broker connection. Useful for testing and CI.
    """

    def __init__(self, initial_balance: float = 10000.0):
        self._balance = initial_balance
        self._initial_balance = initial_balance
        self._trades: list[dict] = []
        self._trade_counter = 0

    def get_account_balance(self) -> float:
        # Balance = initial + realized P/L from closed trades
        realized = sum(t.get("realized_pnl", 0) for t in self._trades if t.get("status") == "closed")
        return self._initial_balance + realized

    def get_open_trades(self) -> list[dict]:
        return [t for t in self._trades if t.get("status") == "open"]

    def place_order(self, pair, signal, size, stop_loss, take_profit) -> dict:
        self._trade_counter += 1
        trade_id = f"PAPER-{self._trade_counter}"
        # Simulate fill at a reasonable price (caller should pass current price as size context)
        # For paper trading, we don't have a live price feed — use 1.0 as placeholder
        trade = {
            "trade_id": trade_id,
            "pair": pair,
            "side": signal,
            "size": size,
            "entry_price": 1.0,  # caller should update with real price
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "profit_loss": 0.0,
            "open_time": None,
            "status": "open",
        }
        self._trades.append(trade)
        return {
            "order_id": f"ORD-{self._trade_counter}",
            "trade_id": trade_id,
            "status": "filled",
            "filled_price": 1.0,
            "message": "Paper order filled",
        }

    def close_trade(self, trade_id: str) -> bool:
        for t in self._trades:
            if t["trade_id"] == trade_id and t["status"] == "open":
                t["status"] = "closed"
                t["realized_pnl"] = 0.0  # paper — no real P/L
                return True
        return False


# ── Factory ──────────────────────────────────────────────────────────────────

_BROKER_REGISTRY: dict[str, type[BrokerBase]] = {
    "paper": PaperBroker,
}


def register_broker(name: str, broker_class: type[BrokerBase]) -> None:
    """Register a new broker type in the factory."""
    _BROKER_REGISTRY[name.lower()] = broker_class
    log.info(f"Registered broker: {name}")


def get_executor(broker: str = "paper", **kwargs) -> BrokerBase:
    """
    Factory: return a broker executor instance.

    Parameters
    ----------
    broker : broker type name. Built-in: "paper". The project's existing
        MT5 broker can be registered via register_broker("mt5", MT5BrokerAdapter).
    **kwargs : passed to the broker's __init__.

    Returns
    -------
    BrokerBase instance.

    Raises
    ------
    ValueError if broker type is unknown.
    """
    broker_lower = broker.lower().strip()
    if broker_lower not in _BROKER_REGISTRY:
        available = list(_BROKER_REGISTRY.keys())
        raise ValueError(
            f"Unknown broker: {broker!r}. Available: {available}. "
            f"Use register_broker() to add new broker types."
        )
    return _BROKER_REGISTRY[broker_lower](**kwargs)


def list_available_brokers() -> list[str]:
    """Return a list of registered broker type names."""
    return list(_BROKER_REGISTRY.keys())


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Paper broker
    broker = get_executor("paper", initial_balance=10000.0)
    print(f"Broker: {type(broker).__name__}")
    print(f"Balance: ${broker.get_account_balance():.2f}")
    print(f"Open trades: {len(broker.get_open_trades())}")

    # Place an order
    result = broker.place_order("EURUSD", "BUY", 0.1, 1.09800, 1.10500)
    print(f"\nOrder: {result}")
    print(f"Open trades: {len(broker.get_open_trades())}")
    print(f"Has EURUSD: {broker.has_open_trade('EURUSD')}")

    # Close
    assert broker.close_trade(result["trade_id"])
    print(f"After close: {len(broker.get_open_trades())} open")

    # List brokers
    print(f"\nAvailable brokers: {list_available_brokers()}")

    # Test unknown broker
    try:
        get_executor("unknown")
        assert False
    except ValueError as e:
        print(f"Unknown broker error: {e}")

    # Register a custom broker
    class TestBroker(BrokerBase):
        def get_account_balance(self): return 999.0
        def get_open_trades(self): return []
        def place_order(self, **kw): return {"status": "filled"}
        def close_trade(self, trade_id): return True

    register_broker("test", TestBroker)
    test_broker = get_executor("test")
    print(f"\nCustom broker balance: ${test_broker.get_account_balance():.2f}")
    print(f"Available brokers: {list_available_brokers()}")

    print("\nBroker factory smoke test passed.")
