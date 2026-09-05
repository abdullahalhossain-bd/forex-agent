"""Canonical position state shared by historical execution and risk gates."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReplayPosition:
    trade_id: Any
    symbol: str
    direction: str
    lot: float
    entry_price: float
    opened_at: Any
    metadata: dict = field(default_factory=dict)


class CanonicalPositionState:
    """Single source of truth for replay exposure.

    Risk/permission code can query this state instead of maintaining a second
    synthetic PaperTrader position list while BrokerSimulator owns fills.
    """

    def __init__(self):
        self._positions: dict[Any, ReplayPosition] = {}
        self.events: list[dict] = []

    def register(self, trade) -> ReplayPosition:
        position = ReplayPosition(
            trade_id=getattr(trade, "trade_id", id(trade)),
            symbol=str(trade.symbol).upper(),
            direction=str(trade.direction).upper(),
            lot=float(getattr(trade, "lot_size", 0.0)),
            entry_price=float(trade.entry_price),
            opened_at=getattr(trade, "entry_time", None),
        )
        self._positions[position.trade_id] = position
        self.events.append({"event": "OPEN", "trade_id": position.trade_id,
                            "symbol": position.symbol, "direction": position.direction})
        return position

    def close(self, trade_or_id, *, reason: str = "closed") -> ReplayPosition | None:
        tid = trade_or_id if trade_or_id in self._positions else getattr(trade_or_id, "trade_id", None)
        position = self._positions.pop(tid, None)
        if position is not None:
            self.events.append({"event": "CLOSE", "trade_id": tid, "reason": reason})
        return position

    def open_positions(self) -> list[ReplayPosition]:
        return list(self._positions.values())

    def count(self) -> int:
        return len(self._positions)

    def count_symbol(self, symbol: str) -> int:
        s = str(symbol).upper()
        return sum(p.symbol == s for p in self._positions.values())

    def has_symbol_direction(self, symbol: str, direction: str) -> bool:
        s, d = str(symbol).upper(), str(direction).upper()
        return any(p.symbol == s and p.direction == d for p in self._positions.values())

    def exposure(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for p in self._positions.values():
            out[p.symbol] = out.get(p.symbol, 0.0) + p.lot
        return out

    def clear(self) -> None:
        self._positions.clear()
        self.events.clear()
