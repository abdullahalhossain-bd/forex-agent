# utils/structured_logger.py — Structured (JSON) logging adapter
# ============================================================
# Drop-in replacement for utils.logger.get_logger that emits
# structured JSON logs when FOREX_LOG_FORMAT=json, and falls back
# to the existing human-readable text format otherwise.
#
# Why structured logs?
#   - Machine-parseable by log aggregators (ELK, Loki, Datadog)
#   - Fields like 'pair', 'pnl', 'session_id' become queryable
#   - Easier to correlate trades → log lines → outcomes
#
# Usage — exactly the same as the existing logger:
#
#     from utils.structured_logger import get_logger
#     log = get_logger("my_module")
#     log.info("Trade opened", extra={"pair": "EURUSD", "lot": 0.05})
#
# When FOREX_LOG_FORMAT=json, the above emits:
#   {"ts":"2026-07-25T10:30:00Z","level":"info","logger":"my_module",
#    "msg":"Trade opened","pair":"EURUSD","lot":0.05}
#
# When FOREX_LOG_FORMAT=text (default), it emits the existing
# human-readable format — full backwards compatibility.
# ============================================================

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Try to import structlog — it's optional. If unavailable we use a
# stdlib-only JSON formatter that produces the same shape.
try:
    import structlog  # type: ignore
    _HAS_STRUCTLOG = True
except ImportError:
    _HAS_STRUCTLOG = False


# ── JSON formatter (stdlib-only, no structlog dep) ───────────

class _JsonFormatter(logging.Formatter):
    """
    Emit log records as one JSON object per line.

    Always includes: ts, level, logger, msg.
    Merges record.__dict__ extras (anything passed via extra={...})
    into the top-level JSON object so it's queryable downstream.
    """

    # Standard LogRecord attributes we don't want leaking into the JSON.
    _RESERVED = frozenset({
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "taskName",
    })

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()

        payload: Dict[str, Any] = {
            "ts":     ts,
            "level":  record.levelname.lower(),
            "logger": record.name,
            "msg":    record.getMessage(),
        }

        # Pull extras (any attribute added via extra={...})
        for key, value in record.__dict__.items():
            if key in self._RESERVED or key.startswith("_"):
                continue
            try:
                # Force JSON-serializable — best-effort coercion
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = repr(value)

        # Add exception info if present
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


# ── Public API ───────────────────────────────────────────────

def _resolve_format() -> str:
    """Read FOREX_LOG_FORMAT lazily so tests can change it at runtime."""
    return (os.getenv("FOREX_LOG_FORMAT") or "text").strip().lower()


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Return a configured logger.

    - FOREX_LOG_FORMAT=json → JSON lines (structured, for log aggregators)
    - FOREX_LOG_FORMAT=text (default) → existing human-readable format

    The returned logger is a stdlib logging.Logger; structlog is used
    internally only for its processor chain if available, but the API
    is plain logging so existing call sites don't need to change.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured — don't add duplicate handlers
        return logger

    logger.setLevel(os.getenv("FOREX_LOG_LEVEL", "INFO").upper())
    logger.propagate = False

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    if _resolve_format() == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        # Fall back to a simple human-readable format that matches
        # the existing utils.logger style closely enough for development.
        handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))

    logger.addHandler(handler)
    return logger


def log_event(
    logger: logging.Logger,
    level: str,
    event: str,
    **fields: Any,
) -> None:
    """
    Convenience helper for emitting a structured event.

    Example:
        log_event(log, "info", "trade_opened", pair="EURUSD", lot=0.05, sl=1.0950)

    In JSON mode this produces:
        {"msg":"trade_opened","pair":"EURUSD","lot":0.05,"sl":1.0950,...}

    In text mode it produces:
        "trade_opened pair=EURUSD lot=0.05 sl=1.0950"
    """
    if _resolve_format() == "json" or _HAS_STRUCTLOG:
        # In JSON mode, pass fields as extras so _JsonFormatter picks them up
        getattr(logger, level.lower(), logger.info)(event, extra=fields)
    else:
        # In text mode, append key=value pairs to the message
        kv = " ".join(f"{k}={v}" for k, v in fields.items())
        msg = f"{event} {kv}".strip()
        getattr(logger, level.lower(), logger.info)(msg)


def is_structured_mode() -> bool:
    """True if the logger is currently emitting JSON."""
    return _resolve_format() == "json"
