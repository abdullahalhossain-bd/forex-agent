"""
execution/simulated_executor.py — Dry-run order executor.

Mirrors the public interface of broker.order_manager.OrderManager but
does NOT contact MT5.  Every "order" is logged to logs/execution.log
as a broker.order_send event with retcode=10009 (TRADE_RETCODE_DONE).

Activated by config.SIMULATION_MODE=True.  ExecutionRouter picks this
executor instead of constructing an MT5Connection + OrderManager.

This lets the operator verify the full signal → risk → approval →
router → "broker" chain end-to-end without a live MT5 terminal,
which is the fastest way to diagnose "trades not being placed"
when the bug is upstream of MT5.
"""
from __future__ import annotations

import time
import random
from typing import Any

from utils.logger import get_logger
from core.execution_logger import log_broker_order_send, log_router_success

log = get_logger("simulated_executor")


class SimulatedExecutor:
    """Drop-in replacement for OrderManager.place_market_order().

    Returns the same dict shape:
        {"success": True, "ticket": int, "retcode": 10009,
         "price": float, "volume": float}
    """

    RETCODE_DONE = 10009

    def __init__(self, db=None):
        self._db = db
        self._ticket_counter = int(time.time()) % 1_000_000  # pseudo-unique
        log.info("[SimulatedExecutor] SIMULATION_MODE active — no real MT5 orders will be placed")

    def place_market_order(
        self,
        symbol: str,
        direction: str,
        lot: float,
        sl: float | None = None,
        tp: float | None = None,
        comment: str = "ai_trader_sim",
    ) -> dict[str, Any]:
        """Simulate a market order. Always succeeds."""
        # Use the supplied entry if available, otherwise pick a plausible price.
        # In simulation we don't have a tick, so derive a fake price from SL/TP.
        if sl and tp:
            entry = (sl + tp) / 2.0
        elif sl:
            entry = sl * 1.001 if direction == "BUY" else sl * 0.999
        elif tp:
            entry = tp * 0.999 if direction == "BUY" else tp * 1.001
        else:
            entry = 1.1000  # arbitrary fallback

        # Simulate tiny slippage (±0.5 pip on a 5-digit symbol)
        slippage = random.uniform(-0.00005, 0.00005)
        filled_price = round(entry + slippage, 5)

        self._ticket_counter += 1
        ticket = self._ticket_counter

        result = {
            "success": True,
            "ticket": ticket,
            "retcode": self.RETCODE_DONE,
            "price": filled_price,
            "volume": lot,
        }

        log_broker_order_send(
            symbol=symbol,
            retcode=self.RETCODE_DONE,
            comment="SIMULATED — no broker contact",
            price=filled_price,
            volume=lot,
            ticket=ticket,
            simulated=True,
        )

        log.info(
            f"[SimulatedExecutor] ✅ SIMULATED {direction} {symbol} "
            f"lot={lot} @ {filled_price} (ticket={ticket}, retcode=10009)"
        )

        log_router_success(
            symbol=symbol,
            ticket=ticket,
            price=filled_price,
            lot=lot,
            simulated=True,
        )

        return result

    def get_open_positions(self, symbol: str | None = None) -> list[dict]:
        """Return empty list — simulator doesn't track positions."""
        return []

    def close_order(self, ticket: int) -> dict[str, Any]:
        """Simulate closing a position."""
        log_broker_order_send(
            symbol="unknown",
            retcode=self.RETCODE_DONE,
            comment="SIMULATED CLOSE",
            price=None,
            volume=None,
            ticket=ticket,
            simulated=True,
            action="close",
        )
        return {"success": True, "ticket": ticket, "retcode": self.RETCODE_DONE}
