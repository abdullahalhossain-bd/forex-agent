"""
orchestrator/mode_manager.py — Minimal stub (Day 60 placeholder)
=================================================================

This file exists to satisfy the import in `orchestrator/trading_orchestrator.py`:

    from orchestrator.mode_manager import ModeManager

The full ModeManager logic was never implemented in the upstream repo.
This stub provides the API surface so the orchestrator can import cleanly.
The live mode state continues to live in `core/approval_mode.py` (the
canonical implementation).

Marked LEGACY_STUB in core/obsolete.py.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

log = logging.getLogger(__name__)


class ModeManager:
    """Thin shim around `core.approval_mode.ApprovalMode`.

    The canonical mode logic (1=analysis, 2=supervised, 3=autonomous) lives
    in core/approval_mode.py. This class exists because TradingOrchestrator
    expects a separate ModeManager object. It just delegates to ApprovalMode.
    """

    def __init__(self, approval_mode=None):
        # approval_mode: a core.approval_mode.ApprovalMode instance
        self.approval_mode = approval_mode
        self._analysis_only = False
        self._autonomous = True

    @property
    def mode_name(self) -> str:
        if self.approval_mode is not None:
            return self.approval_mode.mode_name
        return "AUTONOMOUS" if self._autonomous else "ANALYSIS_ONLY"

    def set_analysis_only(self, on: bool = True) -> None:
        self._analysis_only = on
        self._autonomous = not on
        log.info("ModeManager: analysis_only=%s", on)

    def set_autonomous(self, on: bool = True) -> None:
        self._autonomous = on
        self._analysis_only = not on
        log.info("ModeManager: autonomous=%s", on)

    def is_trading_allowed(self) -> bool:
        return self._autonomous

    def status(self) -> Dict[str, Any]:
        return {
            "mode_name": self.mode_name,
            "analysis_only": self._analysis_only,
            "autonomous": self._autonomous,
            "approval_mode_wired": self.approval_mode is not None,
        }
