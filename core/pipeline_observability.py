"""
core/pipeline_observability.py — Structured Trade Pipeline Observability
========================================================================

OBSERVABILITY ONLY. Does not change trading decisions, thresholds, or order params.

Every trading cycle gets a unique evaluation_id (reused from AITrader when present).
Each meaningful pipeline stage emits structured events so an operator can answer:
  - What symbol / TF / signal?
  - Did each layer run?
  - What value vs threshold did each gate evaluate?
  - Which exact layer blocked the trade?
  - Which layers were NOT_REACHED?
  - Was MT5 order_send attempted / succeeded / failed?

Usage (inside AITrader.run_cycle):
    from core.pipeline_observability import get_pipeline_trace, PipelineStage, StageStatus

    trace = get_pipeline_trace()
    trace.start_cycle(evaluation_id=self._current_evaluation_id,
                      symbol=self.symbol, timeframe=self.timeframe)
    trace.stage_entered(PipelineStage.MARKET_DATA)
    ...
    trace.stage_result(PipelineStage.CONFIDENCE, StageStatus.BLOCKED,
                       reason_code="CONFIDENCE_TOO_LOW",
                       reason="Confidence below threshold",
                       value=48.0, threshold=50.0)
    ...
    trace.emit_summary()  # always call once at end of cycle
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("pipeline_obs")


class PipelineStage(str, Enum):
    HUMAN_OVERRIDE = "HUMAN_OVERRIDE"
    FREQUENCY_CAP = "FREQUENCY_CAP"
    MARKET_DATA = "MARKET_DATA"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"
    INDICATORS = "INDICATORS"
    ANALYSIS = "ANALYSIS"
    SIGNAL = "SIGNAL"
    CONSENSUS = "CONSENSUS"
    CONFIDENCE = "CONFIDENCE"
    ENTRY_GATE = "ENTRY_GATE"
    FILTERS = "FILTERS"
    RISK = "RISK"
    POSITION_SIZE = "POSITION_SIZE"
    SLTP = "SLTP"
    TRADE_PERMISSION = "TRADE_PERMISSION"
    FINAL_DECISION = "FINAL_DECISION"
    DEVILS_ADVOCATE = "DEVILS_ADVOCATE"
    APPROVAL = "APPROVAL"
    MT5_CONNECTION = "MT5_CONNECTION"
    MT5_ORDER_SEND = "MT5_ORDER_SEND"
    EXECUTION_RESULT = "EXECUTION_RESULT"


_STAGE_ORDER: List[str] = [
    PipelineStage.HUMAN_OVERRIDE.value,
    PipelineStage.FREQUENCY_CAP.value,
    PipelineStage.MARKET_DATA.value,
    PipelineStage.CIRCUIT_BREAKER.value,
    PipelineStage.INDICATORS.value,
    PipelineStage.ANALYSIS.value,
    PipelineStage.SIGNAL.value,
    PipelineStage.CONSENSUS.value,
    PipelineStage.CONFIDENCE.value,
    PipelineStage.ENTRY_GATE.value,
    PipelineStage.FILTERS.value,
    PipelineStage.RISK.value,
    PipelineStage.POSITION_SIZE.value,
    PipelineStage.SLTP.value,
    PipelineStage.TRADE_PERMISSION.value,
    PipelineStage.FINAL_DECISION.value,
    PipelineStage.DEVILS_ADVOCATE.value,
    PipelineStage.APPROVAL.value,
    PipelineStage.MT5_CONNECTION.value,
    PipelineStage.MT5_ORDER_SEND.value,
    PipelineStage.EXECUTION_RESULT.value,
]


class StageStatus(str, Enum):
    ENTERED = "ENTERED"
    PASS = "PASS"
    BLOCKED = "BLOCKED"
    REJECTED = "REJECTED"
    SKIPPED = "SKIPPED"
    NOT_REACHED = "NOT_REACHED"
    ERROR = "ERROR"
    EXECUTION_ATTEMPTED = "EXECUTION_ATTEMPTED"
    EXECUTION_SUCCESS = "EXECUTION_SUCCESS"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    WAIT = "WAIT"


class ReasonCode(str, Enum):
    NO_SIGNAL = "NO_SIGNAL"
    WAIT_SIGNAL = "WAIT_SIGNAL"
    CONSENSUS_FAILED = "CONSENSUS_FAILED"
    CONFIDENCE_TOO_LOW = "CONFIDENCE_TOO_LOW"
    SPREAD_TOO_HIGH = "SPREAD_TOO_HIGH"
    LOW_LIQUIDITY = "LOW_LIQUIDITY"
    CHOPPY_MARKET = "CHOPPY_MARKET"
    NEWS_BLOCK = "NEWS_BLOCK"
    SESSION_BLOCK = "SESSION_BLOCK"
    COOLDOWN_ACTIVE = "COOLDOWN_ACTIVE"
    MAX_OPEN_TRADES = "MAX_OPEN_TRADES"
    RISK_LIMIT = "RISK_LIMIT"
    DUPLICATE_POSITION = "DUPLICATE_POSITION"
    INVALID_SYMBOL = "INVALID_SYMBOL"
    INVALID_LOT = "INVALID_LOT"
    INVALID_SL = "INVALID_SL"
    INVALID_TP = "INVALID_TP"
    INVALID_RR = "INVALID_RR"
    TRADE_PERMISSION_DENIED = "TRADE_PERMISSION_DENIED"
    DEVILS_ADVOCATE_REJECT = "DEVILS_ADVOCATE_REJECT"
    MT5_NOT_CONNECTED = "MT5_NOT_CONNECTED"
    MT5_ORDER_FAILED = "MT5_ORDER_FAILED"
    HUMAN_OVERRIDE = "HUMAN_OVERRIDE"
    FREQUENCY_CAP = "FREQUENCY_CAP"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"
    MARKET_DATA_FAILED = "MARKET_DATA_FAILED"
    ENTRY_TOO_FAR = "ENTRY_TOO_FAR"
    EQUITY_STOP = "EQUITY_STOP"
    EXCEPTION = "EXCEPTION"
    UNKNOWN = "UNKNOWN"
    OK = "OK"


_BLOCKING_STATUSES = {
    StageStatus.BLOCKED.value,
    StageStatus.REJECTED.value,
    StageStatus.ERROR.value,
    StageStatus.EXECUTION_FAILED.value,
    StageStatus.WAIT.value,
}


@dataclass
class StageEvent:
    stage: str
    status: str
    reason_code: str = ""
    reason: str = ""
    value: Any = None
    threshold: Any = None
    duration_ms: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "stage": self.stage,
            "status": self.status,
            "ts": self.ts,
        }
        if self.reason_code:
            d["reason_code"] = self.reason_code
        if self.reason:
            d["reason"] = self.reason
        if self.value is not None:
            d["value"] = self.value
        if self.threshold is not None:
            d["threshold"] = self.threshold
        if self.duration_ms is not None:
            d["duration_ms"] = round(self.duration_ms, 2)
        if self.extra:
            d.update(self.extra)
        return d


@dataclass
class CycleTrace:
    evaluation_id: str
    symbol: str
    timeframe: str
    started_at: float = field(default_factory=time.time)
    signal: str = ""
    events: List[StageEvent] = field(default_factory=list)
    stage_results: Dict[str, StageEvent] = field(default_factory=dict)
    _stage_enter_ts: Dict[str, float] = field(default_factory=dict)
    ticket: Optional[int] = None
    lot: Optional[float] = None
    entry: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    summary_emitted: bool = False

    def stage_entered(self, stage: str) -> None:
        self._stage_enter_ts[stage] = time.time()
        self.events.append(StageEvent(stage=stage, status=StageStatus.ENTERED.value))

    def stage_result(
        self,
        stage: str,
        status: str,
        *,
        reason_code: str = "",
        reason: str = "",
        value: Any = None,
        threshold: Any = None,
        **extra: Any,
    ) -> None:
        duration_ms = None
        t0 = self._stage_enter_ts.get(stage)
        if t0 is not None:
            duration_ms = (time.time() - t0) * 1000.0
        ev = StageEvent(
            stage=stage,
            status=status,
            reason_code=reason_code or "",
            reason=reason or "",
            value=value,
            threshold=threshold,
            duration_ms=duration_ms,
            extra={k: v for k, v in extra.items() if v is not None},
        )
        self.events.append(ev)
        if status != StageStatus.ENTERED.value:
            self.stage_results[stage] = ev

    def first_blocker(self) -> Optional[StageEvent]:
        for ev in self.events:
            if ev.status in _BLOCKING_STATUSES and ev.status != StageStatus.ENTERED.value:
                return ev
        return None

    def not_reached_stages(self) -> List[str]:
        reached = set(self.stage_results.keys()) | set(self._stage_enter_ts.keys())
        blocker = self.first_blocker()
        if blocker is None:
            return [s for s in _STAGE_ORDER if s not in reached]
        try:
            bi = _STAGE_ORDER.index(blocker.stage)
        except ValueError:
            bi = -1
        out = []
        for i, s in enumerate(_STAGE_ORDER):
            if s in reached:
                continue
            if bi >= 0 and i > bi:
                out.append(s)
            elif bi < 0 and s not in reached:
                out.append(s)
        return out

    def build_summary(self) -> Dict[str, Any]:
        blocker = self.first_blocker()
        not_reached = self.not_reached_stages()

        for s in not_reached:
            if s not in self.stage_results:
                self.stage_results[s] = StageEvent(
                    stage=s, status=StageStatus.NOT_REACHED.value
                )

        final_status = "PASS"
        execution = "NOT_REACHED"
        if blocker:
            final_status = blocker.status
            if blocker.stage in (
                PipelineStage.MT5_ORDER_SEND.value,
                PipelineStage.EXECUTION_RESULT.value,
            ):
                execution = blocker.status
        else:
            exec_ev = self.stage_results.get(PipelineStage.EXECUTION_RESULT.value)
            send_ev = self.stage_results.get(PipelineStage.MT5_ORDER_SEND.value)
            if exec_ev and exec_ev.status == StageStatus.EXECUTION_SUCCESS.value:
                final_status = StageStatus.EXECUTION_SUCCESS.value
                execution = "SUCCESS"
            elif send_ev and send_ev.status == StageStatus.EXECUTION_FAILED.value:
                final_status = StageStatus.EXECUTION_FAILED.value
                execution = "FAILED"
            elif send_ev and send_ev.status == StageStatus.EXECUTION_ATTEMPTED.value:
                execution = "ATTEMPTED"
            elif exec_ev:
                execution = exec_ev.status
            elif send_ev:
                execution = send_ev.status

        stages_snapshot = {
            s: (self.stage_results[s].to_dict() if s in self.stage_results else {
                "stage": s, "status": StageStatus.NOT_REACHED.value
            })
            for s in _STAGE_ORDER
            if s in self.stage_results or s in not_reached or s in self._stage_enter_ts
        }

        summary: Dict[str, Any] = {
            "event": "trade.pipeline_summary",
            "evaluation_id": self.evaluation_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "signal": self.signal or "",
            "final_status": final_status,
            "blocked_stage": blocker.stage if blocker else None,
            "reason_code": (blocker.reason_code if blocker else None) or None,
            "reason": (blocker.reason if blocker else None) or None,
            "execution": execution,
            "duration_ms": round((time.time() - self.started_at) * 1000.0, 2),
            "stages_reached": sorted(set(self._stage_enter_ts.keys()) | set(self.stage_results.keys())),
            "stages_not_reached": not_reached,
            "stages": stages_snapshot,
        }
        if self.ticket is not None:
            summary["ticket"] = self.ticket
        if self.lot is not None:
            summary["lot"] = self.lot
        if self.entry is not None:
            summary["entry"] = self.entry
        if self.sl is not None:
            summary["sl"] = self.sl
        if self.tp is not None:
            summary["tp"] = self.tp
        return summary


class PipelineObservability:
    def __init__(self) -> None:
        self._local = threading.local()
        self._lock = threading.Lock()
        self._log_path = self._resolve_log_path()

    @staticmethod
    def _resolve_log_path() -> Path:
        try:
            from core.constants import MEMORY_DIR
            base = MEMORY_DIR
        except Exception:
            base = Path(os.getenv("MEMORY_DIR", "memory"))
        base = Path(base)
        try:
            base.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return base / "pipeline_trace.jsonl"

    def _current(self) -> Optional[CycleTrace]:
        return getattr(self._local, "cycle", None)

    def start_cycle(
        self,
        evaluation_id: str,
        symbol: str,
        timeframe: str,
    ) -> CycleTrace:
        cycle = CycleTrace(
            evaluation_id=evaluation_id or f"eval_{int(time.time())}_{symbol}",
            symbol=symbol,
            timeframe=timeframe,
        )
        self._local.cycle = cycle
        self._emit_event({
            "event": "trade.cycle_start",
            "evaluation_id": cycle.evaluation_id,
            "symbol": symbol,
            "timeframe": timeframe,
            "status": StageStatus.ENTERED.value,
        })
        return cycle

    def stage_entered(self, stage: str) -> None:
        cycle = self._current()
        if cycle is None:
            return
        try:
            cycle.stage_entered(stage)
            self._emit_event({
                "event": "trade.stage",
                "evaluation_id": cycle.evaluation_id,
                "symbol": cycle.symbol,
                "timeframe": cycle.timeframe,
                "stage": stage,
                "status": StageStatus.ENTERED.value,
            })
        except Exception:
            pass

    def stage_result(
        self,
        stage: str,
        status: str,
        *,
        reason_code: str = "",
        reason: str = "",
        value: Any = None,
        threshold: Any = None,
        signal: Optional[str] = None,
        **extra: Any,
    ) -> None:
        cycle = self._current()
        if cycle is None:
            return
        try:
            if signal:
                cycle.signal = signal
            cycle.stage_result(
                stage, status,
                reason_code=reason_code,
                reason=reason,
                value=value,
                threshold=threshold,
                **extra,
            )
            payload: Dict[str, Any] = {
                "event": "trade.gate" if status in _BLOCKING_STATUSES else "trade.stage",
                "evaluation_id": cycle.evaluation_id,
                "symbol": cycle.symbol,
                "timeframe": cycle.timeframe,
                "stage": stage,
                "status": status,
            }
            if reason_code:
                payload["reason_code"] = reason_code
            if reason:
                payload["reason"] = reason
            if value is not None:
                payload["value"] = value
            if threshold is not None:
                payload["threshold"] = threshold
            if signal:
                payload["signal"] = signal
            for k, v in extra.items():
                if v is not None and k not in payload:
                    payload[k] = v
            self._emit_event(payload)
        except Exception:
            pass

    def set_trade_params(
        self,
        *,
        lot: Optional[float] = None,
        entry: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        ticket: Optional[int] = None,
        signal: Optional[str] = None,
    ) -> None:
        cycle = self._current()
        if cycle is None:
            return
        if lot is not None:
            cycle.lot = lot
        if entry is not None:
            cycle.entry = entry
        if sl is not None:
            cycle.sl = sl
        if tp is not None:
            cycle.tp = tp
        if ticket is not None:
            cycle.ticket = ticket
        if signal:
            cycle.signal = signal

    def emit_summary(self, *, force: bool = False) -> Optional[Dict[str, Any]]:
        cycle = self._current()
        if cycle is None:
            return None
        if cycle.summary_emitted and not force:
            return None
        try:
            summary = cycle.build_summary()
            cycle.summary_emitted = True
            self._emit_event(summary)
            blocked = summary.get("blocked_stage")
            if blocked:
                log.info(
                    "[Pipeline] %s %s eval=%s signal=%s FINAL=%s blocked_at=%s "
                    "reason_code=%s reason=%s execution=%s",
                    summary.get("symbol"),
                    summary.get("timeframe"),
                    summary.get("evaluation_id"),
                    summary.get("signal"),
                    summary.get("final_status"),
                    blocked,
                    summary.get("reason_code"),
                    (summary.get("reason") or "")[:120],
                    summary.get("execution"),
                )
            else:
                log.info(
                    "[Pipeline] %s %s eval=%s signal=%s FINAL=%s execution=%s ticket=%s",
                    summary.get("symbol"),
                    summary.get("timeframe"),
                    summary.get("evaluation_id"),
                    summary.get("signal"),
                    summary.get("final_status"),
                    summary.get("execution"),
                    summary.get("ticket"),
                )
            return summary
        except Exception as e:
            log.debug("[Pipeline] emit_summary failed: %s", e)
            return None

    def end_cycle(self) -> Optional[Dict[str, Any]]:
        summary = self.emit_summary()
        self._local.cycle = None
        return summary

    def current_evaluation_id(self) -> Optional[str]:
        cycle = self._current()
        return cycle.evaluation_id if cycle else None

    def _emit_event(self, payload: Dict[str, Any]) -> None:
        try:
            if "ts" not in payload:
                payload["ts"] = datetime.now(timezone.utc).isoformat()
            line = json.dumps(payload, default=str, ensure_ascii=False)
            with self._lock:
                try:
                    with open(self._log_path, "a", encoding="utf-8") as f:
                        f.write(line + "\n")
                except Exception:
                    pass
            log.debug("%s", line)
        except Exception:
            pass


_OBS: Optional[PipelineObservability] = None
_OBS_LOCK = threading.Lock()


def get_pipeline_trace() -> PipelineObservability:
    global _OBS
    if _OBS is None:
        with _OBS_LOCK:
            if _OBS is None:
                _OBS = PipelineObservability()
    return _OBS


def map_reject_reason_to_code(stage: str, reason: str) -> str:
    r = (reason or "").lower()
    s = (stage or "").lower()
    if "human override" in r or s == "human_override":
        return ReasonCode.HUMAN_OVERRIDE.value
    if "daily trade cap" in r or "frequency" in s:
        return ReasonCode.FREQUENCY_CAP.value
    if "market data" in r or "fetch failed" in r:
        return ReasonCode.MARKET_DATA_FAILED.value
    if "circuit" in r or "kill switch" in r:
        return ReasonCode.CIRCUIT_BREAKER.value
    if "confidence" in r and ("low" in r or "below" in r or "<" in r):
        return ReasonCode.CONFIDENCE_TOO_LOW.value
    if "spread" in r:
        return ReasonCode.SPREAD_TOO_HIGH.value
    if "liquidity" in r or "thin" in r or "explosive" in r:
        return ReasonCode.LOW_LIQUIDITY.value
    if "news" in r:
        return ReasonCode.NEWS_BLOCK.value
    if "session" in r:
        return ReasonCode.SESSION_BLOCK.value
    if "cooldown" in r:
        return ReasonCode.COOLDOWN_ACTIVE.value
    if "max open" in r or "max_open" in r:
        return ReasonCode.MAX_OPEN_TRADES.value
    if "duplicate" in r or "already open" in r:
        return ReasonCode.DUPLICATE_POSITION.value
    if "devil" in r or "da_" in s or "devils" in s:
        return ReasonCode.DEVILS_ADVOCATE_REJECT.value
    if "permission" in r or "permission" in s:
        return ReasonCode.TRADE_PERMISSION_DENIED.value
    if "mt5" in r and ("disconnect" in r or "not connected" in r or "unavailable" in r):
        return ReasonCode.MT5_NOT_CONNECTED.value
    if "order" in r and ("fail" in r or "reject" in r or "retcode" in r):
        return ReasonCode.MT5_ORDER_FAILED.value
    if "lot" in r and ("invalid" in r or "zero" in r or "0.0" in r):
        return ReasonCode.INVALID_LOT.value
    if "rr" in r or "risk:reward" in r or "r:r" in r:
        return ReasonCode.INVALID_RR.value
    if "sl" in r and "invalid" in r:
        return ReasonCode.INVALID_SL.value
    if "tp" in r and "invalid" in r:
        return ReasonCode.INVALID_TP.value
    if "entry" in r and ("far" in r or "chase" in r or "retest" in r):
        return ReasonCode.ENTRY_TOO_FAR.value
    if "equity stop" in r:
        return ReasonCode.EQUITY_STOP.value
    if "wait" in r or r.strip() in ("wait", "no trade", "no_trade"):
        return ReasonCode.WAIT_SIGNAL.value
    if "choppy" in r or "adx" in r:
        return ReasonCode.CHOPPY_MARKET.value
    if "risk" in s or "kelly" in r or "drawdown" in r:
        return ReasonCode.RISK_LIMIT.value
    if "exception" in r or "error" in r:
        return ReasonCode.EXCEPTION.value
    if "consensus" in r:
        return ReasonCode.CONSENSUS_FAILED.value
    return ReasonCode.UNKNOWN.value
