"""Historical execution router: live-router-shaped policy over OHLC.

This module is deliberately strategy-agnostic. It only models execution state:
market orders fill on the next bar open; limit orders remain pending until their
price is touched; expired/unfilled orders never become trades. BrokerSimulator
continues to own the actual fill accounting and SL/TP touch/P&L math.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class PendingLimit:
    order_id: int
    symbol: str
    direction: str
    requested_entry: float
    sl: float
    tp: float
    lot: float
    confidence: int
    submitted_at: Any
    submitted_bar: int
    expiry_bars: int
    kwargs: dict


class HistoricalExecutionRouter:
    """Execution boundary for historical replay.

    The router never manufactures a fill from the signal-bar close. A market
    request is scheduled for the next bar open. A limit request is pending and
    is evaluated against subsequent OHLC bars. All state is instance-local,
    making replay deterministic and safe for multi-symbol harnesses.
    """

    def __init__(self, broker, *, max_lot: Optional[float] = None,
                 default_limit_expiry_bars: int = 3):
        self.broker = broker
        self.max_lot = max_lot
        self.default_limit_expiry_bars = max(1, int(default_limit_expiry_bars))
        self._next_order_id = 1
        self._pending: dict[int, PendingLimit] = {}
        self._market_queue: list[dict] = []
        self.telemetry: list[dict] = []

    @property
    def pending_orders(self) -> list[PendingLimit]:
        return list(self._pending.values())

    def _cap_lot(self, lot: float) -> float:
        value = float(lot)
        if self.max_lot is None:
            try:
                from config import MAX_LOT
                value = min(value, float(MAX_LOT))
            except Exception:
                pass
        else:
            value = min(value, float(self.max_lot))
        return value

    @staticmethod
    def _is_limit(kwargs: dict) -> bool:
        order_type = str(kwargs.get("order_type", kwargs.get("execution_type", ""))).upper()
        return order_type in {"LIMIT", "BUY_LIMIT", "SELL_LIMIT", "PENDING_LIMIT"} or bool(kwargs.get("pending_limit"))

    def submit(self, *, symbol: str, direction: str, entry_price: float,
               sl: float, tp: float, lot: float, confidence: int,
               bar_index: int, bar_time=None, **kwargs) -> dict:
        lot = self._cap_lot(lot)
        if lot <= 0:
            return {"status": "REJECTED", "reason": "invalid_lot"}
        direction = str(direction).upper()
        if direction not in {"BUY", "SELL"}:
            return {"status": "REJECTED", "reason": "invalid_direction"}
        if self._is_limit(kwargs):
            oid = self._next_order_id
            self._next_order_id += 1
            expiry = int(kwargs.get("limit_expiry_bars", self.default_limit_expiry_bars))
            order = PendingLimit(oid, symbol, direction, float(entry_price),
                                 float(sl), float(tp), lot, int(confidence),
                                 bar_time, int(bar_index), max(1, expiry),
                                 dict(kwargs))
            self._pending[oid] = order
            event = {"event": "LIMIT_PENDING", "order_id": oid,
                     "symbol": symbol, "direction": direction,
                     "bar_index": bar_index, "entry": float(entry_price)}
            self.telemetry.append(event)
            return {"status": "PENDING", "order_id": oid, "pending": order}

        self._market_queue.append({
            "symbol": symbol, "direction": direction, "entry_price": float(entry_price),
            "sl": float(sl), "tp": float(tp), "lot": lot,
            "confidence": int(confidence), "submitted_bar": int(bar_index),
            "submitted_at": bar_time, "kwargs": dict(kwargs),
        })
        self.telemetry.append({"event": "MARKET_QUEUED", "symbol": symbol,
                               "direction": direction, "bar_index": bar_index})
        return {"status": "QUEUED", "submitted_bar": bar_index}

    def advance(self, *, bar_index: int, bar_open: float, bar_high: float,
                bar_low: float, bar_close: float, bar_time=None) -> list[dict]:
        """Advance execution state to a new historical bar.

        Returns newly filled trades. Market orders submitted on bar N are
        eligible only at N+1. Pending limits are evaluated from N+1 onward.
        """
        fills: list[dict] = []

        queued = self._market_queue
        self._market_queue = []
        for req in queued:
            if int(req["submitted_bar"]) >= int(bar_index):
                # Never allow same-bar execution; retain until a later bar.
                self._market_queue.append(req)
                continue
            trade = self.broker.open_trade(
                symbol=req["symbol"], direction=req["direction"],
                entry_price=float(bar_open), sl=req["sl"], tp=req["tp"],
                lot=req["lot"], bar_time=bar_time,
                confidence=req["confidence"], **req["kwargs"],
            )
            if trade is not None:
                fills.append(trade)
                self.telemetry.append({"event": "MARKET_FILLED", "bar_index": bar_index,
                                       "trade_id": getattr(trade, "trade_id", None),
                                       "entry_price": getattr(trade, "entry_price", None)})
            else:
                self.telemetry.append({"event": "MARKET_REJECTED", "bar_index": bar_index})

        for oid, order in list(self._pending.items()):
            age = int(bar_index) - order.submitted_bar
            if age <= 0:
                continue
            if age > order.expiry_bars:
                del self._pending[oid]
                self.telemetry.append({"event": "LIMIT_EXPIRED", "order_id": oid,
                                       "bar_index": bar_index})
                continue

            touched = float(bar_low) <= order.requested_entry <= float(bar_high)
            if not touched:
                continue
            trade = self.broker.open_trade(
                symbol=order.symbol, direction=order.direction,
                entry_price=order.requested_entry, sl=order.sl, tp=order.tp,
                lot=order.lot, bar_time=bar_time, confidence=order.confidence,
                **order.kwargs,
            )
            del self._pending[oid]
            if trade is not None:
                fills.append(trade)
                self.telemetry.append({"event": "LIMIT_FILLED", "order_id": oid,
                                       "bar_index": bar_index,
                                       "trade_id": getattr(trade, "trade_id", None)})
            else:
                self.telemetry.append({"event": "LIMIT_REJECTED", "order_id": oid,
                                       "bar_index": bar_index})
        return fills

    def cancel_all(self, reason: str = "replay_end") -> None:
        for oid in list(self._pending):
            self.telemetry.append({"event": "LIMIT_CANCELLED", "order_id": oid, "reason": reason})
        self._pending.clear()
        self._market_queue.clear()
