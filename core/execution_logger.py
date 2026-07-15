"""
core/execution_logger.py — Structured execution logger for the trade path.

Writes one JSONL line per execution event to logs/execution.log so the
entire signal → risk → approval → router → broker chain is observable
end-to-end.  Plug-in callers use ``log_event(...)`` at each stage;
no other code changes needed.

Events emitted:
  - signal.generated       — analysis_agent produced a BUY/SELL
  - decision.resolved      — decision_agent returned final decision
  - risk.evaluated         — risk_engine approved/rejected
  - sizer.evaluated        — Day 76 position sizer verdict
  - permission.checked     — trade_permission 5-check result
  - approval.processed     — approval_mode.process() result
  - router.execute.start   — ExecutionRouter.execute() entered
  - router.execute.success — order filled
  - router.execute.fail    — order rejected / failed
  - broker.order_send      — mt5.order_send() result (with retcode)
  - broker.last_error      — mt5.last_error() snapshot on failure
  - orphan.position        — DB journal failed after broker fill

Each line is a JSON object with:
  ts, event, symbol, decision, confidence, lot, sl, tp, retcode,
  ticket, reason, extra{}

Usage from trader.py / execution_router.py:
    from core.execution_logger import log_event
    log_event("router.execute.start", symbol="EURUSD", decision="BUY",
              lot=0.01, sl=1.0850, tp=1.0950)
"""
from __future__ import annotations

import gzip
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Lock so concurrent AITrader threads (one per symbol) don't interleave writes.
_LOCK = threading.Lock()

# Resolve logs directory. Mirrors core/constants.py:LOGS_DIR but does not
# import it (avoids circular import risk during early boot).
_LOGS_DIR = Path(os.getenv("LOGS_DIR", "logs"))
_LOG_PATH = _LOGS_DIR / "execution.log"

# ── Issue 11: log rotation (same policy as core/trade_decision_log.py) ──
_MAX_LOG_BYTES = int(os.getenv("EXEC_LOG_MAX_BYTES", str(10 * 1024 * 1024)))  # 10MB
_MAX_BACKUPS = int(os.getenv("EXEC_LOG_MAX_BACKUPS", "5"))


def _rotate_if_needed(path: Path, max_bytes: int = _MAX_LOG_BYTES,
                       max_backups: int = _MAX_BACKUPS) -> None:
    """Rotate `path` to `path.1.gz`, `path.2.gz`, ... once it exceeds
    max_bytes. Must be called while holding _LOCK. Never raises.

    Day 99+ V4 FIX (Audit Issue #2): the original code used
    `path.with_suffix(path.suffix + f".{i}.gz")` which works for
    `execution.log` (produces `execution.log.1.gz`) but is brittle:
      - On paths with NO extension, `path.suffix == ""` and the result
        is `name.1.gz` (loses any type information).
      - On paths with MULTIPLE dots (e.g. `execution.log.gz`),
        `path.suffix` only returns the LAST extension (`.gz`), so the
        rotation produces `execution.log.gz.1.gz` (double `.gz`).
      - `with_suffix()` raises `ValueError` if the new suffix starts
        with `.` AND contains another `.` mid-string on some Python
        versions, silently breaking rotation.
    The new helper `_backup_path(path, i)` builds the backup name by
    concatenating the FULL original filename (preserving all extensions)
    with `.{i}.gz`, producing predictable names like:
      execution.log      → execution.log.1.gz, execution.log.2.gz
      trade_decisions    → trade_decisions.1.gz, trade_decisions.2.gz
      execution.log.gz   → execution.log.gz.1.gz, execution.log.gz.2.gz
    """
    def _backup_path(p: Path, i: int) -> Path:
        """Build backup name as `<full_original_name>.<i>.gz`."""
        return p.with_name(f"{p.name}.{i}.gz")

    try:
        if not path.exists() or path.stat().st_size < max_bytes:
            return
        for i in range(max_backups - 1, 0, -1):
            src = _backup_path(path, i)
            dst = _backup_path(path, i + 1)
            if src.exists():
                if i + 1 > max_backups:
                    src.unlink(missing_ok=True)
                else:
                    src.replace(dst)
        backup_path = _backup_path(path, 1)
        with open(path, "rb") as f_in, gzip.open(backup_path, "wb") as f_out:
            f_out.writelines(f_in)
        path.write_text("", encoding="utf-8")
    except Exception:
        pass  # best-effort — rotation failure must never block the write


def _ensure_log_dir() -> None:
    try:
        _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass  # best-effort; if this fails the write below will fail too


def log_event(event: str, **fields: Any) -> None:
    """Write one JSONL line to logs/execution.log.

    Never raises — logging failures are silently dropped to avoid
    crashing the trade path.  If you want to detect logging failures,
    check the file's existence after the call.
    """
    try:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
        }
        record.update(fields)
        line = json.dumps(record, default=str)  # default=str handles Decimal/datetime
        with _LOCK:
            _ensure_log_dir()
            _rotate_if_needed(_LOG_PATH)
            with open(_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as e:
        # Never let logging crash the trade path.
        pass


def log_signal_generated(symbol: str, signal: str, confidence: float,
                         source: str = "analysis_agent", **extra) -> None:
    log_event("signal.generated", symbol=symbol, decision=signal,
              confidence=confidence, source=source, **extra)


def log_decision_resolved(symbol: str, decision: str, confidence: float,
                          reasons: list[str] | None = None, **extra) -> None:
    log_event("decision.resolved", symbol=symbol, decision=decision,
              confidence=confidence, reasons=reasons or [], **extra)


def log_risk_evaluated(symbol: str, approved: bool, lot: float,
                       sl: float | None = None, tp: float | None = None,
                       reject_reason: str | None = None, **extra) -> None:
    log_event("risk.evaluated", symbol=symbol, approved=approved, lot=lot,
              sl=sl, tp=tp, reject_reason=reject_reason, **extra)


def log_permission_checked(symbol: str, allowed: bool, passed: int, total: int,
                           failed_checks: list[str] | None = None, **extra) -> None:
    log_event("permission.checked", symbol=symbol, allowed=allowed,
              passed=passed, total=total, failed_checks=failed_checks or [],
              **extra)


def log_approval_processed(symbol: str, proceed: bool, mode: int,
                           action: str, **extra) -> None:
    log_event("approval.processed", symbol=symbol, proceed=proceed,
              mode=mode, action=action, **extra)


def log_router_start(symbol: str, decision: str, lot: float,
                     sl: float | None = None, tp: float | None = None,
                     **extra) -> None:
    log_event("router.execute.start", symbol=symbol, decision=decision,
              lot=lot, sl=sl, tp=tp, **extra)


def log_router_success(symbol: str, ticket: int | None, price: float,
                       lot: float, trade_id: int | None = None, **extra) -> None:
    log_event("router.execute.success", symbol=symbol, ticket=ticket,
              price=price, lot=lot, trade_id=trade_id, **extra)


def log_router_fail(symbol: str, reason: str, stage: str = "unknown",
                    **extra) -> None:
    log_event("router.execute.fail", symbol=symbol, reason=reason,
              stage=stage, **extra)


def log_broker_order_send(symbol: str, retcode: int | None,
                          comment: str | None, price: float | None,
                          volume: float | None, ticket: int | None = None,
                          **extra) -> None:
    log_event("broker.order_send", symbol=symbol, retcode=retcode,
              comment=comment, price=price, volume=volume, ticket=ticket,
              **extra)


def log_broker_last_error(symbol: str, error: Any, **extra) -> None:
    """Capture mt5.last_error() snapshot when an MT5 call fails."""
    log_event("broker.last_error", symbol=symbol, error=str(error), **extra)


def log_orphan_position(symbol: str, ticket: int | None, reason: str,
                        **extra) -> None:
    """DB journal failed AFTER broker fill — broker has a position,
    bot has no record.  Operator must reconcile manually."""
    log_event("orphan.position", symbol=symbol, ticket=ticket,
              reason=reason, **extra)