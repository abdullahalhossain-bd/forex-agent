"""
utils/confidence_trace.py — Shared confidence trace helper
==========================================================

Provides a lightweight, structured confidence trace that threads through
the decision pipeline.  Each module appends a before/after entry so
operators can see exactly where confidence was penalised and why.

Usage (in any pipeline module):
    from utils.confidence_trace import confidence_trace

    confidence_trace.record(
        module="smart_money",
        before=72,
        after=72,
        reason="BOS(25) + OB(25) + FVG(20) = 70, no cutoff applied",
    )

The trace is a module-level list that is reset at the start of each
trading cycle by the calling orchestrator (trader.py / decision_agent).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class TraceEntry:
    """One confidence-modification step."""
    module: str          # e.g. "smart_money", "session_analyzer"
    before: float        # confidence before this step
    after: float         # confidence after this step
    reason: str          # human-readable explanation
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConfidenceTrace:
    """Thread-safe, per-cycle confidence audit trail."""

    def __init__(self):
        self._entries: List[TraceEntry] = []
        self._lock = threading.Lock()

    def reset(self) -> None:
        """Clear all entries (call at the start of each trading cycle)."""
        with self._lock:
            self._entries.clear()

    def record(
        self,
        module: str,
        before: float,
        after: float,
        reason: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            self._entries.append(TraceEntry(
                module=module,
                before=round(before, 2),
                after=round(after, 2),
                reason=reason,
                details=details or {},
            ))

    @property
    def entries(self) -> List[TraceEntry]:
        with self._lock:
            return list(self._entries)

    def to_list(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self.entries]

    def summary(self) -> str:
        """Compact single-line summary for log output."""
        parts = [f"{e.module}:{e.before}->{e.after}" for e in self._entries]
        return " | ".join(parts) if parts else "(no trace)"


# ── Module-level singleton ─────────────────────────────────────────────
confidence_trace = ConfidenceTrace()