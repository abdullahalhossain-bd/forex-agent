"""
orchestrator/self_healing.py — Minimal stub (Day 60 placeholder)
=================================================================

This file exists to satisfy the import in `orchestrator/trading_orchestrator.py`:

    from orchestrator.self_healing import SelfHealingSystem

The full SelfHealingSystem logic was never implemented in the upstream repo.
This stub provides the API surface so the orchestrator can import cleanly.

Day 102+ CRITICAL hotfix: aligned constructor signature + added the
three missing methods the orchestrator calls. Previously the orchestrator
called `SelfHealingSystem(self.bus, self.state_mgr)` but the stub's
__init__ took no args — any attempt to instantiate TradingOrchestrator
crashed with TypeError. The orchestrator also calls `register_healers()`,
`on_error(msg)`, and `heal(error)` — all of which were missing. Added
no-op implementations so the orchestrator at least boots cleanly.

Marked LEGACY_STUB in core/obsolete.py.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

log = logging.getLogger(__name__)


class SelfHealingSystem:
    """Detects recurring runtime errors and applies automatic remediation.

    Currently a no-op stub that records issues for future pattern analysis.
    Extension points:
      * restart crashed sub-systems
      * rotate log files
      * reconnect MT5
      * rebuild corrupted DB indexes
    """

    def __init__(self, bus=None, state_mgr=None):
        """
        Day 102+ hotfix: accept (bus, state_mgr) args the orchestrator
        passes. They're stored for future use but currently unused —
        remediation is still a no-op.
        """
        self._bus = bus
        self._state_mgr = state_mgr
        self._issues: List[Dict[str, Any]] = []
        self._remediations: List[Dict[str, Any]] = []
        self._healers: List[Dict[str, Any]] = []

    def register_healers(self) -> None:
        """
        Day 102+ hotfix: orchestrator calls this at boot (line 362).
        Future: register per-subsystem healers (MT5 reconnect, DB
        vacuum, log rotation, etc.). Currently a no-op that logs so
        the orchestrator's _init_self_healing has something to consume.
        """
        log.info("[SelfHealing] register_healers() called — no healers registered (stub)")
        # Future: self._healers.append({"kind": "mt5_disconnect", "fn": ...})

    def on_error(self, msg: Any) -> None:
        """
        Day 102+ hotfix: orchestrator subscribes this to the bus's
        'error' topic (line 430). Records the error for pattern
        analysis. Currently does NOT auto-heal — just records.
        """
        try:
            kind = getattr(msg, "kind", None) or (msg.get("kind") if isinstance(msg, dict) else "unknown")
            detail = getattr(msg, "detail", None) or (msg.get("detail") if isinstance(msg, dict) else str(msg))
        except Exception:
            kind, detail = "unknown", str(msg)
        self.record_issue(kind=kind, detail=detail)

    def heal(self, error: Any) -> bool:
        """
        Day 102+ hotfix: orchestrator calls this when an error needs
        immediate remediation (line 817). Currently a no-op that
        records the attempt. Returns False to indicate no healing
        was actually performed — the orchestrator should fall back
        to its own recovery logic.
        """
        self._remediations.append({
            "ts": time.time(),
            "kind": "heal",
            "error": str(error),
            "action": "noop",
        })
        log.warning("[SelfHealing] heal() called for error: %s — no healer available (stub)", error)
        return False

    def record_issue(self, kind: str, detail: str) -> None:
        entry = {"ts": time.time(), "kind": kind, "detail": detail}
        self._issues.append(entry)
        log.warning("SelfHealing issue: %s — %s", kind, detail)
        # Try a no-op remediation so the orchestrator's _check_self_healing
        # has something to consume.
        self._try_remediate(kind, detail)

    def _try_remediate(self, kind: str, detail: str) -> bool:
        """Future: dispatch on `kind` to apply fixes. Currently a no-op."""
        self._remediations.append({"ts": time.time(), "kind": kind, "action": "noop"})
        return False

    def get_recent_issues(self, limit: int = 20) -> List[Dict[str, Any]]:
        return list(self._issues[-limit:])

    def status(self) -> Dict[str, Any]:
        return {
            "issues_recorded": len(self._issues),
            "remediations_attempted": len(self._remediations),
            "healers_registered": len(self._healers),
        }
