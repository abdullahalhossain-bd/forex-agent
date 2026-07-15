"""
core/trade_decision_log.py — Records every trade decision (taken or not).

Writes one JSONL line per symbol cycle to memory/trade_decisions.jsonl
so the operator can see EXACTLY why each trade was or wasn't placed.

Each record contains:
  ts           — ISO timestamp
  symbol       — pair symbol
  timeframe    — chart timeframe
  signal       — final signal (BUY/SELL/NO TRADE/WAIT)
  confidence   — decision confidence %
  decision     — what the system decided to do
  taken        — was a trade placed? (True/False)
  reject_stage — where the trade was blocked (if not taken)
  reject_reason— human-readable reason (if not taken)
  lot          — final lot size (if taken)
  entry/sl/tp  — trade parameters (if taken)
  ticket       — MT5 ticket number (if taken)
  cycle_errors — list of errors that occurred during this cycle

Usage:
    from core.trade_decision_log import log_decision
    log_decision(symbol="EURUSD", signal="BUY", confidence=65,
                 taken=False, reject_stage="risk",
                 reject_reason="Correlation conflict with AUDUSD")
"""
from __future__ import annotations

import gzip
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_LOCK = threading.Lock()
_LOG_PATH = Path("memory/trade_decisions.jsonl")

# ── Issue 11: log rotation ────────────────────────────────────────
# Without this, trade_decisions.jsonl grows forever — on a busy multi-symbol
# bot running for months this becomes a multi-GB file that's slow to read
# (get_recent_decisions/get_summary read the WHOLE file) and can eventually
# fill the disk, which would take down the whole process.
_MAX_LOG_BYTES = int(os.getenv("TRADE_LOG_MAX_BYTES", str(10 * 1024 * 1024)))  # 10MB
_MAX_BACKUPS = int(os.getenv("TRADE_LOG_MAX_BACKUPS", "5"))


def _rotate_if_needed(path: Path, max_bytes: int = _MAX_LOG_BYTES,
                       max_backups: int = _MAX_BACKUPS) -> None:
    """Rotate `path` to `path.1.gz`, `path.2.gz`, ... if it has grown past
    max_bytes. Must be called while holding _LOCK. Never raises — a
    rotation failure should not block writing the log line itself.

    Day 99+ V4 FIX (Audit Issue #3): the original code used
    `path.with_suffix(path.suffix + f".{i}.gz")` which works for files
    with a single `.log` extension but is brittle for paths with no
    extension or multiple extensions. New `_backup_path` helper builds
    the backup name as `<full_original_name>.<i>.gz` for predictable
    naming across all extension shapes.
    """
    def _backup_path(p: Path, i: int) -> Path:
        """Build backup name as `<full_original_name>.<i>.gz`."""
        return p.with_name(f"{p.name}.{i}.gz")

    try:
        if not path.exists() or path.stat().st_size < max_bytes:
            return
        # Shift existing backups up: .4.gz -> .5.gz (dropped), .3.gz -> .4.gz, ...
        for i in range(max_backups - 1, 0, -1):
            src = _backup_path(path, i)
            dst = _backup_path(path, i + 1)
            if src.exists():
                if i + 1 > max_backups:
                    src.unlink(missing_ok=True)
                else:
                    src.replace(dst)
        # Compress the current log into .1.gz, then truncate it.
        backup_path = _backup_path(path, 1)
        with open(path, "rb") as f_in, gzip.open(backup_path, "wb") as f_out:
            f_out.writelines(f_in)
        path.write_text("", encoding="utf-8")
    except Exception:
        # Best-effort — never let rotation crash the trade path.
        pass


def log_decision(
    symbol: str,
    signal: str = "NO TRADE",
    confidence: float = 0,
    timeframe: str = "15m",
    decision: str = "",
    taken: bool = False,
    reject_stage: str = "",
    reject_reason: str = "",
    lot: Optional[float] = None,
    entry: Optional[float] = None,
    sl: Optional[float] = None,
    tp: Optional[float] = None,
    ticket: Optional[int] = None,
    cycle_errors: Optional[list] = None,
    **extra,
) -> None:
    """Write one decision record to memory/trade_decisions.jsonl.

    Never raises — logging failures are silently dropped.
    """
    try:
        record = {
            "ts":             datetime.now(timezone.utc).isoformat(),
            "symbol":         symbol,
            "timeframe":      timeframe,
            "signal":         signal,
            "confidence":     confidence,
            "decision":       decision or signal,
            "taken":          taken,
            "reject_stage":   reject_stage,
            "reject_reason":  reject_reason,
            "lot":            lot,
            "entry":          entry,
            "sl":             sl,
            "tp":             tp,
            "ticket":         ticket,
            "cycle_errors":   cycle_errors or [],
        }
        record.update(extra)
        line = json.dumps(record, default=str)
        with _LOCK:
            _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            _rotate_if_needed(_LOG_PATH)
            with open(_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as e:
        # Never let logging crash the trade path.
        pass


def log_cycle_error(symbol: str, error: str, stage: str = "unknown") -> None:
    """Log a non-fatal error that occurred during a symbol cycle.

    These are accumulated and included in the next log_decision() call's
    cycle_errors list.
    """
    try:
        # Write to a separate errors log so they're easy to find
        err_path = Path("memory/cycle_errors.jsonl")
        record = {
            "ts":      datetime.now(timezone.utc).isoformat(),
            "symbol":  symbol,
            "stage":   stage,
            "error":   error,
        }
        with _LOCK:
            err_path.parent.mkdir(parents=True, exist_ok=True)
            _rotate_if_needed(err_path)
            with open(err_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        # Never let logging crash the trade path. (Previously referenced an
        # undefined `log` object here, which would itself raise NameError.)
        pass


def get_recent_decisions(limit: int = 20) -> list[dict]:
    """Read the most recent N decision records (for dashboard/debugging)."""
    try:
        if not _LOG_PATH.exists():
            return []
        lines = _LOG_PATH.read_text().strip().split("\n")
        records = []
        for line in lines[-limit:]:
            try:
                records.append(json.loads(line))
            except Exception:
                # Skip malformed lines; never let a bad line break the read.
                pass
        return records
    except Exception:
        return []


def get_summary() -> dict:
    """Get summary stats of all decisions in the log."""
    try:
        if not _LOG_PATH.exists():
            return {"total": 0, "taken": 0, "rejected": 0, "by_stage": {}}
        lines = _LOG_PATH.read_text().strip().split("\n")
        total = 0
        taken = 0
        rejected = 0
        by_stage: dict[str, int] = {}
        for line in lines:
            try:
                rec = json.loads(line)
                total += 1
                if rec.get("taken"):
                    taken += 1
                else:
                    rejected += 1
                    stage = rec.get("reject_stage", "unknown")
                    by_stage[stage] = by_stage.get(stage, 0) + 1
            except Exception:
                # Skip malformed lines; never let a bad line break the summary.
                pass
        return {"total": total, "taken": taken, "rejected": rejected, "by_stage": by_stage}
    except Exception as e:
        return {"total": 0, "taken": 0, "rejected": 0, "by_stage": {}}