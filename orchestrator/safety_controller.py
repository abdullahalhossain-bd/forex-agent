"""
orchestrator/safety_controller.py — Minimal stub (Day 60 placeholder)
=====================================================================

This file exists to satisfy the import in `orchestrator/trading_orchestrator.py`:

    from orchestrator.safety_controller import SafetyController

The full SafetyController logic was never implemented in the upstream repo.
This stub provides the API surface so the orchestrator can import cleanly,
and so future development can extend SafetyController without touching the
orchestrator's imports.

Marked LEGACY_STUB in core/obsolete.py.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


class SafetyController:
    """Centralized safety gate. Currently a thin wrapper around the existing
    risk.circuit_breaker.CircuitBreaker + risk.trade_permission.TradePermission.

    Future extension points:
      * cross-symbol exposure checks
      * news-blackout enforcement
      * drawdown-tier risk scaling
    """

    def __init__(self, circuit_breaker=None, trade_permission=None):
        self.circuit_breaker = circuit_breaker
        self.trade_permission = trade_permission
        self._emergency_stop = False
        self._stop_reason: Optional[str] = None

    def check_pre_trade(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Return {'allowed': bool, 'reason': str}."""
        if self._emergency_stop:
            return {"allowed": False, "reason": f"emergency stop: {self._stop_reason}"}
        if self.circuit_breaker is not None:
            cb = self.circuit_breaker.allow_trade()
            if not cb.get("allowed", False):
                return {"allowed": False, "reason": f"circuit_breaker: {cb.get('reason')}"}
        return {"allowed": True, "reason": "ok"}

    def trigger_emergency_stop(self, reason: str = "manual") -> None:
        self._emergency_stop = True
        self._stop_reason = reason
        log.warning("SafetyController emergency stop: %s", reason)

    def clear_emergency_stop(self) -> None:
        self._emergency_stop = False
        self._stop_reason = None
        log.info("SafetyController emergency stop cleared")

    def status(self) -> Dict[str, Any]:
        return {
            "emergency_stop": self._emergency_stop,
            "stop_reason": self._stop_reason,
            "circuit_breaker_wired": self.circuit_breaker is not None,
            "trade_permission_wired": self.trade_permission is not None,
        }
