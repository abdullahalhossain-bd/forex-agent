"""Historical execution router: live-router-shaped policy over OHLC."""
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
    """Canonical historical execution boundary.

    BrokerSimulator owns accounting. CanonicalPositionState owns replay
    exposure. Market orders are queued and can fill only on a later bar;
    limits remain pending until a later OHLC bar touches the requested price.
    """
    def __init__(self, broker, *, max_lot: Optional[float] = None,
                 max_open_positions: Optional[int] = None, position_state=None,
                 default_limit_expiry_bars: int = 3):
        self.broker = broker
        self.max_lot = max_lot
        self.max_open_positions = max_open_positions
        self.default_limit_expiry_bars = max(1, int(default_limit_expiry_bars))
        self._next_order_id = 1
        self._pending = {}
        self._market_queue = []
        self.telemetry = []
        if position_state is None:
            from backtest.position_state import CanonicalPositionState
            position_state = CanonicalPositionState()
        self.position_state = position_state

    @property
    def pending_orders(self):
        return list(self._pending.values())

    def _cap_lot(self, lot):
        value = float(lot)
        try:
            cap = self.max_lot
            if cap is None:
                from config import MAX_LOT
                cap = MAX_LOT
            value = min(value, float(cap))
        except Exception:
            pass
        return value

    @staticmethod
    def _is_limit(kwargs):
        order_type = str(kwargs.get("order_type", kwargs.get("execution_type", ""))).upper()
        return order_type in {"LIMIT", "BUY_LIMIT", "SELL_LIMIT", "PENDING_LIMIT"} or bool(kwargs.get("pending_limit"))

    def _exposure_reason(self, symbol, direction):
        if self.max_open_positions is not None and self.position_state.count() >= int(self.max_open_positions):
            return "max_open_positions"
        if self.position_state.has_symbol_direction(symbol, direction):
            return "duplicate_symbol_direction"
        return None

    def submit(self, *, symbol, direction, entry_price, sl, tp, lot, confidence,
               bar_index, bar_time=None, **kwargs):
        direction = str(direction).upper()
        if direction not in {"BUY", "SELL"}:
            return {"status": "REJECTED", "reason": "invalid_direction"}
        lot = self._cap_lot(lot)
        if lot <= 0:
            return {"status": "REJECTED", "reason": "invalid_lot"}
        reason = self._exposure_reason(symbol, direction)
        if reason:
            self.telemetry.append({"event": "ORDER_REJECTED", "reason": reason, "bar_index": bar_index})
            return {"status": "REJECTED", "reason": reason}
        if self._is_limit(kwargs):
            oid = self._next_order_id
            self._next_order_id += 1
            expiry = max(1, int(kwargs.get("limit_expiry_bars", self.default_limit_expiry_bars)))
            order = PendingLimit(oid, symbol, direction, float(entry_price), float(sl), float(tp),
                                 lot, int(confidence), bar_time, int(bar_index), expiry, dict(kwargs))
            self._pending[oid] = order
            self.telemetry.append({"event": "LIMIT_PENDING", "order_id": oid, "bar_index": bar_index})
            return {"status": "PENDING", "order_id": oid, "pending": order}
        self._market_queue.append({"symbol": symbol, "direction": direction,
            "sl": float(sl), "tp": float(tp), "lot": lot, "confidence": int(confidence),
            "submitted_bar": int(bar_index), "submitted_at": bar_time, "kwargs": dict(kwargs)})
        self.telemetry.append({"event": "MARKET_QUEUED", "bar_index": bar_index})
        return {"status": "QUEUED", "submitted_bar": bar_index}

    def advance(self, *, bar_index, bar_open, bar_high, bar_low, bar_close, bar_time=None):
        fills = []
        queued, self._market_queue = self._market_queue, []
        for req in queued:
            if req["submitted_bar"] >= int(bar_index):
                self._market_queue.append(req)
                continue
            trade = self.broker.open_trade(symbol=req["symbol"], direction=req["direction"],
                entry_price=float(bar_open), sl=req["sl"], tp=req["tp"], lot=req["lot"],
                bar_time=bar_time, confidence=req["confidence"], **req["kwargs"])
            if trade is not None:
                fills.append(trade)
                self.position_state.register(trade)
                self.telemetry.append({"event": "MARKET_FILLED", "bar_index": bar_index, "trade_id": getattr(trade, "trade_id", None)})
            else:
                self.telemetry.append({"event": "MARKET_REJECTED", "bar_index": bar_index})
        for oid, order in list(self._pending.items()):
            age = int(bar_index) - order.submitted_bar
            if age <= 0:
                continue
            if age > order.expiry_bars:
                del self._pending[oid]
                self.telemetry.append({"event": "LIMIT_EXPIRED", "order_id": oid, "bar_index": bar_index})
                continue
            if not (float(bar_low) <= order.requested_entry <= float(bar_high)):
                continue
            trade = self.broker.open_trade(symbol=order.symbol, direction=order.direction,
                entry_price=order.requested_entry, sl=order.sl, tp=order.tp, lot=order.lot,
                bar_time=bar_time, confidence=order.confidence, **order.kwargs)
            del self._pending[oid]
            if trade is not None:
                fills.append(trade)
                self.position_state.register(trade)
                self.telemetry.append({"event": "LIMIT_FILLED", "order_id": oid, "bar_index": bar_index, "trade_id": getattr(trade, "trade_id", None)})
            else:
                self.telemetry.append({"event": "LIMIT_REJECTED", "order_id": oid, "bar_index": bar_index})
        return fills

    def on_close(self, trade, *, reason="closed"):
        self.position_state.close(trade, reason=reason)

    def cancel_all(self, reason="replay_end"):
        for oid in list(self._pending):
            self.telemetry.append({"event": "LIMIT_CANCELLED", "order_id": oid, "reason": reason})
        self._pending.clear()
        self._market_queue.clear()
