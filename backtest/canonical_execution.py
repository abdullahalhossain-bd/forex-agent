"""Canonical historical execution adapter for live-mirroring replay."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class FillPolicy:
    spread_pips: float
    slippage_pips: float = 0.0
    commission_per_lot: float = 0.0
    intrabar_policy: str = "AMBIGUOUS_INTRABAR"


@dataclass
class PositionLifecycle:
    trade_id: int
    symbol: str
    direction: str
    signal_time: str
    entry_time: str
    requested_entry: float
    fill_price: float
    requested_lot: float
    filled_lot: float
    stop_loss: float
    take_profit: float
    pnl_multiplier: float
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    commission_usd: float = 0.0
    pnl_usd: Optional[float] = None
    status: str = "OPEN"

    def to_dict(self):
        return asdict(self)


def _require_timestamp(value, field: str) -> datetime:
    if value is None:
        raise ValueError(f"{field} is required")
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    return dt


def ensure_execution_order(signal_time, entry_time):
    signal = _require_timestamp(signal_time, "signal_time")
    entry = _require_timestamp(entry_time, "entry_time")
    if entry < signal:
        raise ValueError(
            f"LOOK_AHEAD_EXECUTION: entry_time={entry.isoformat()} "
            f"precedes signal_time={signal.isoformat()}"
        )


def compute_fill(direction: str, requested_price: float, spread_pips: float,
                pip_size: float, slippage_pips: float = 0.0,
                historical_ask: Optional[float] = None) -> float:
    """Compute deterministic executable fill from historical bid/ask data.

    BUY uses historical ASK when available. If only BID plus a known spread is
    available, ASK is reconstructed as BID + full spread. SELL uses BID.
    Slippage is applied in the adverse direction. No random component exists.
    """
    if pip_size <= 0:
        raise ValueError("pip_size must be positive")
    if spread_pips < 0 or slippage_pips < 0:
        raise ValueError("spread/slippage cannot be negative")
    slip = slippage_pips * pip_size
    if direction.upper() == "BUY":
        ask = float(historical_ask) if historical_ask is not None else float(requested_price) + spread_pips * pip_size
        return ask + slip
    if direction.upper() == "SELL":
        return float(requested_price) - slip
    raise ValueError(f"Unsupported direction: {direction}")


def resolve_intrabar(*, direction: str, bar_high: float, bar_low: float,
                     stop_loss: float, take_profit: float,
                     policy: str = "AMBIGUOUS_INTRABAR") -> Optional[str]:
    if direction.upper() == "BUY":
        sl_hit = bar_low <= stop_loss
        tp_hit = bar_high >= take_profit
    elif direction.upper() == "SELL":
        sl_hit = bar_high >= stop_loss
        tp_hit = bar_low <= take_profit
    else:
        raise ValueError(f"Unsupported direction: {direction}")
    if not sl_hit and not tp_hit:
        return None
    if sl_hit and tp_hit:
        if policy in {"WORST_CASE", "CONSERVATIVE"}:
            return "SL"
        if policy == "BEST_CASE":
            return "TP"
        if policy in {"OHLC_ASSUMPTION", "LOWER_TF_REPLAY"}:
            raise ValueError("AMBIGUOUS_INTRABAR: both SL and TP touched; ordering source required")
        return "AMBIGUOUS_INTRABAR"
    return "SL" if sl_hit else "TP"


def mark_close(position: PositionLifecycle, exit_time, exit_price: float,
               reason: str, pnl_usd: float):
    close_dt = _require_timestamp(exit_time, "exit_time")
    entry_dt = _require_timestamp(position.entry_time, "entry_time")
    if close_dt < entry_dt:
        raise ValueError("EXIT_BEFORE_ENTRY")
    position.exit_time = close_dt.isoformat()
    position.exit_price = float(exit_price)
    position.exit_reason = reason
    position.pnl_usd = float(pnl_usd)
    position.status = "CLOSED"
    return position


class CanonicalHistoricalExecutionAdapter:
    """Broker-facing replay backend at the live execution boundary."""

    def __init__(self, *, pip_size: float, fill_policy: FillPolicy):
        if pip_size <= 0:
            raise ValueError("pip_size must be positive")
        self.pip_size = float(pip_size)
        self.fill_policy = fill_policy
        self._next_trade_id = 1
        self.open_positions: dict[int, PositionLifecycle] = {}

    def open_trade(self, *, decision_result: dict, signal_time: str,
                   entry_time: str, historical_bid: float,
                   pnl_multiplier: float,
                   historical_ask: Optional[float] = None) -> PositionLifecycle:
        ensure_execution_order(signal_time, entry_time)
        direction = str(decision_result.get("decision", "")).upper()
        if direction not in {"BUY", "SELL"}:
            raise ValueError("EXECUTION_REJECTED: decision is not BUY/SELL")
        missing = [k for k in ("entry", "sl", "tp", "lot") if decision_result.get(k) is None]
        if missing:
            raise ValueError(f"EXECUTION_REJECTED: missing {missing}")
        if pnl_multiplier <= 0:
            raise ValueError("pnl_multiplier must be positive and explicitly supplied")
        lot = float(decision_result["lot"])
        if lot <= 0:
            raise ValueError("EXECUTION_REJECTED: lot must be positive")
        requested_entry = float(decision_result["entry"])
        fill = compute_fill(
            direction, float(historical_bid), self.fill_policy.spread_pips,
            self.pip_size, self.fill_policy.slippage_pips,
            historical_ask=historical_ask,
        )
        trade = PositionLifecycle(
            trade_id=self._next_trade_id,
            symbol=str(decision_result.get("symbol", "")), direction=direction,
            signal_time=_require_timestamp(signal_time, "signal_time").isoformat(),
            entry_time=_require_timestamp(entry_time, "entry_time").isoformat(),
            requested_entry=requested_entry, fill_price=fill,
            requested_lot=lot, filled_lot=lot,
            stop_loss=float(decision_result["sl"]), take_profit=float(decision_result["tp"]),
            pnl_multiplier=float(pnl_multiplier),
        )
        self._next_trade_id += 1
        self.open_positions[trade.trade_id] = trade
        return trade

    def close_trade(self, trade_id: int, *, exit_time: str, exit_price: float,
                    reason: str) -> PositionLifecycle:
        trade = self.open_positions.get(int(trade_id))
        if trade is None:
            raise KeyError(f"unknown open trade_id={trade_id}")
        pnl = self._pnl_before_costs(trade, float(exit_price))
        commission = trade.filled_lot * self.fill_policy.commission_per_lot
        trade.commission_usd += commission
        closed = mark_close(trade, exit_time, exit_price, reason, pnl - commission)
        self.open_positions.pop(trade.trade_id, None)
        return closed

    @staticmethod
    def _pnl_before_costs(trade: PositionLifecycle, exit_price: float) -> float:
        if trade.direction == "BUY":
            return (exit_price - trade.fill_price) * trade.filled_lot * trade.pnl_multiplier
        if trade.direction == "SELL":
            return (trade.fill_price - exit_price) * trade.filled_lot * trade.pnl_multiplier
        raise ValueError(f"Unsupported direction: {trade.direction}")
