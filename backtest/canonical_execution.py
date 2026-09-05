"""Canonical historical execution adapter for live-mirroring replay.

Purpose
-------
Keep live decision/risk/permission code authoritative while replacing only
MT5's execution backend.  This adapter is deterministic and timestamp-bound.
It never reads wall-clock time and never invents a fill from a future bar.

Important: this is an execution backend, not a strategy or sizing engine.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class FillPolicy:
    """Explicit assumptions required when historical bid/ask is unavailable."""
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
    """Reject a fill that occurs before the signal decision."""
    signal = _require_timestamp(signal_time, "signal_time")
    entry = _require_timestamp(entry_time, "entry_time")
    if entry < signal:
        raise ValueError(
            f"LOOK_AHEAD_EXECUTION: entry_time={entry.isoformat()} "
            f"precedes signal_time={signal.isoformat()}"
        )


def compute_fill(direction: str, requested_price: float, spread_pips: float,
                pip_size: float, slippage_pips: float = 0.0) -> float:
    """Compute a deterministic bid/ask fill from a historical mid/bid price.

    Convention matches the replay contract: input is the historical BID.
    BUY executes at ASK (plus adverse slippage); SELL executes at BID
    (minus adverse slippage).  No random component is permitted.
    """
    if pip_size <= 0:
        raise ValueError("pip_size must be positive")
    if spread_pips < 0 or slippage_pips < 0:
        raise ValueError("spread/slippage cannot be negative")
    half = spread_pips * pip_size / 2.0
    slip = slippage_pips * pip_size
    if direction.upper() == "BUY":
        return requested_price + half + slip
    if direction.upper() == "SELL":
        return requested_price - slip
    raise ValueError(f"Unsupported direction: {direction}")


def resolve_intrabar(*, direction: str, bar_high: float, bar_low: float,
                     stop_loss: float, take_profit: float,
                     policy: str = "AMBIGUOUS_INTRABAR") -> Optional[str]:
    """Resolve SL/TP touch without claiming unavailable tick ordering."""
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
            raise ValueError(
                "AMBIGUOUS_INTRABAR: both SL and TP touched; "
                f"policy={policy} requires an ordering source"
            )
        return "AMBIGUOUS_INTRABAR"
    return "SL" if sl_hit else "TP"


def mark_close(position: PositionLifecycle, exit_time, exit_price: float,
               reason: str, pnl_usd: float):
    """Close an existing replay position without touching global/live state."""
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
