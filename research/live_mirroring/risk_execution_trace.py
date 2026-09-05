"""Canonical, side-effect-free trace helpers for live-mirroring replay.

This module deliberately does NOT implement risk, sizing, permission, or
execution policy. The live implementations remain the source of truth.
It only normalizes their already-produced outputs into an auditable record.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Mapping


@dataclass
class GateTrace:
    stage: str
    name: str
    status: str  # PASS | FAIL | SKIPPED | UNKNOWN
    reason: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    output: Any = None


@dataclass
class RiskExecutionTrace:
    replay_timestamp: str
    symbol: str
    direction: str
    entry_requested: float | None = None

    # RiskEngine output
    risk_approved: bool | None = None
    risk_reason: str = ""
    risk_pc_intended: float | None = None
    risk_usd_intended: float | None = None
    balance_before: float | None = None
    sl_raw: float | None = None
    sl_final: float | None = None
    tp_raw: float | None = None
    tp_final: float | None = None
    sl_pips: float | None = None
    tp_pips: float | None = None
    rr_final: float | None = None

    # PositionSizer output — all values copied from the live result
    base_lot_raw: float | None = None
    base_lot_normalized: float | None = None
    final_lot_raw: float | None = None
    final_lot_normalized: float | None = None
    actual_risk_usd: float | None = None
    actual_risk_pct: float | None = None
    sizing_breakdown: dict[str, Any] = field(default_factory=dict)

    # Permission / DA
    permission_allowed: bool | None = None
    permission_reason: str = ""
    da_allowed: bool | None = None
    da_reason: str = ""

    # Execution
    execution_requested: bool = False
    execution_accepted: bool = False
    execution_reason: str = ""
    requested_lot: float | None = None
    filled_lot: float | None = None
    requested_entry: float | None = None
    filled_entry: float | None = None
    spread_pips: float | None = None
    slippage_pips: float | None = None
    commission_usd: float | None = None
    swap_usd: float | None = None
    ticket: str | int | None = None

    # Lifecycle
    lifecycle_status: str = "NOT_OPENED"
    opened_at: str | None = None
    closed_at: str | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    pnl_usd: float | None = None
    pnl_pips: float | None = None
    pnl_r: float | None = None
    mae: float | None = None
    mfe: float | None = None
    time_to_mae: str | None = None
    time_to_mfe: str | None = None

    gates: list[GateTrace] = field(default_factory=list)
    assumptions: dict[str, Any] = field(default_factory=dict)

    def add_gate(self, stage: str, name: str, status: str, reason: str = "", *,
                 inputs: Mapping[str, Any] | None = None, output: Any = None) -> None:
        status = str(status).upper()
        if status not in {"PASS", "FAIL", "SKIPPED", "UNKNOWN"}:
            raise ValueError(f"Invalid gate status: {status}")
        self.gates.append(GateTrace(
            stage=stage,
            name=name,
            status=status,
            reason=reason,
            inputs=dict(inputs or {}),
            output=output,
        ))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def trace_live_risk_output(trace: RiskExecutionTrace, risk_out: Mapping[str, Any]) -> None:
    """Copy the live RiskEngine result without recalculating anything."""
    trace.risk_approved = bool(risk_out.get("approved"))
    trace.risk_reason = str(risk_out.get("reject_reason") or risk_out.get("reason") or "")
    trace.risk_pc_intended = _num(risk_out.get("risk_pc_intended", risk_out.get("risk_pc")))
    trace.risk_usd_intended = _num(risk_out.get("risk_usd_intended", risk_out.get("risk_usd")))
    trace.sl_raw = _num(risk_out.get("sl_raw", risk_out.get("raw_sl", risk_out.get("stop_loss"))))
    trace.sl_final = _num(risk_out.get("sl_price", risk_out.get("sl")))
    trace.tp_raw = _num(risk_out.get("tp_raw", risk_out.get("raw_tp", risk_out.get("take_profit"))))
    trace.tp_final = _num(risk_out.get("tp_price", risk_out.get("tp")))
    trace.sl_pips = _num(risk_out.get("sl_pips"))
    trace.tp_pips = _num(risk_out.get("tp_pips"))
    trace.rr_final = _num(risk_out.get("rr", risk_out.get("risk_reward")))
    trace.requested_lot = _num(risk_out.get("lot"))

    sizing = risk_out.get("position_sizing")
    if isinstance(sizing, Mapping):
        trace.sizing_breakdown = dict(sizing)
        trace.base_lot_raw = _num(sizing.get("base_lot_raw", sizing.get("base_lot")))
        trace.base_lot_normalized = _num(sizing.get("base_lot"))
        trace.final_lot_raw = _num(sizing.get("final_lot_raw"))
        trace.final_lot_normalized = _num(sizing.get("lot"))
        trace.actual_risk_usd = _num(sizing.get("risk_amount_usd"))
        trace.actual_risk_pct = _num(sizing.get("risk_pct"))


def trace_permission_output(trace: RiskExecutionTrace, permission_out: Mapping[str, Any]) -> None:
    trace.permission_allowed = bool(permission_out.get("allowed"))
    trace.permission_reason = str(
        permission_out.get("reason")
        or permission_out.get("reject_reason")
        or ""
    )


def trace_execution_output(trace: RiskExecutionTrace, result: Mapping[str, Any] | None) -> None:
    trace.execution_requested = True
    if not result:
        trace.execution_accepted = False
        trace.execution_reason = "Execution returned no result"
        return
    trace.execution_accepted = bool(result.get("success", result.get("accepted", False)))
    trace.execution_reason = str(result.get("reason") or result.get("retcode") or "")
    trace.ticket = result.get("ticket")
    trace.filled_entry = _num(result.get("price", result.get("filled_price")))
    trace.filled_lot = _num(result.get("volume", result.get("filled_lot", result.get("lot"))))


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
