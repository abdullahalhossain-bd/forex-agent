"""
risk/drawdown_monitor.py — Drawdown Monitoring (Day 75)
=========================================================

Tracks account drawdown and activates Capital Preservation Mode
when drawdown exceeds thresholds.

Capital Preservation Mode:
  - Drawdown > 8%  → DEFENSIVE (only 85%+ confidence trades, half lot)
  - Drawdown > 12% → PROTECTIVE (only A+ setups, quarter lot)
  - Drawdown > 15% → EMERGENCY (no new trades — kill switch Level 3)

Also tracks peak balance and recovery progress.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from utils.logger import get_logger

log = get_logger("drawdown_monitor")


@dataclass
class DrawdownStatus:
    """Current drawdown status."""
    current_drawdown_pct: float = 0.0
    peak_balance: float = 0.0
    current_balance: float = 0.0
    mode: str = "NORMAL"          # NORMAL / DEFENSIVE / PROTECTIVE / EMERGENCY
    min_confidence_required: float = 50.0
    position_multiplier: float = 1.0
    recovery_pct: float = 0.0     # how much recovered from peak DD

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DrawdownMonitor:
    """Monitors drawdown and activates protection modes."""

    # Thresholds
    DEFENSIVE_THRESHOLD = 0.08     # 8% drawdown
    PROTECTIVE_THRESHOLD = 0.12    # 12% drawdown
    EMERGENCY_THRESHOLD = 0.15     # 15% drawdown

    # Mode settings
    MODE_SETTINGS = {
        "NORMAL": {
            "min_confidence": 50.0,
            "position_mult": 1.0,
        },
        "DEFENSIVE": {
            "min_confidence": 70.0,
            "position_mult": 0.5,
        },
        "PROTECTIVE": {
            "min_confidence": 85.0,
            "position_mult": 0.25,
        },
        "EMERGENCY": {
            "min_confidence": 100.0,  # effectively blocks all trades
            "position_mult": 0.0,
        },
    }

    def __init__(self):
        self._lock = threading.RLock()
        self._peak_balance = 0.0
        self._max_drawdown_seen = 0.0
        self._current_mode = "NORMAL"

    def update(self, balance: float, initial_balance: float) -> DrawdownStatus:
        """Update drawdown tracking with current balance.

        Args:
            balance: Current account balance.
            initial_balance: Starting balance.

        Returns:
            DrawdownStatus with current mode + settings.
        """
        with self._lock:
            if balance > self._peak_balance:
                self._peak_balance = balance

            if self._peak_balance > 0:
                drawdown = (self._peak_balance - balance) / self._peak_balance
            else:
                drawdown = 0.0

            if drawdown > self._max_drawdown_seen:
                self._max_drawdown_seen = drawdown

            # Determine mode
            if drawdown >= self.EMERGENCY_THRESHOLD:
                mode = "EMERGENCY"
            elif drawdown >= self.PROTECTIVE_THRESHOLD:
                mode = "PROTECTIVE"
            elif drawdown >= self.DEFENSIVE_THRESHOLD:
                mode = "DEFENSIVE"
            else:
                mode = "NORMAL"

            if mode != self._current_mode:
                log.warning(
                    f"[DrawdownMonitor] Mode changed: {self._current_mode} → {mode} "
                    f"(DD={drawdown:.1%})"
                )
                self._current_mode = mode

            settings = self.MODE_SETTINGS.get(mode, self.MODE_SETTINGS["NORMAL"])

            # Recovery progress
            recovery = 0.0
            if self._max_drawdown_seen > 0:
                recovery = 1.0 - (drawdown / self._max_drawdown_seen) if self._max_drawdown_seen > 0 else 0
                recovery = max(0.0, min(1.0, recovery))

            return DrawdownStatus(
                current_drawdown_pct=drawdown,
                peak_balance=self._peak_balance,
                current_balance=balance,
                mode=mode,
                min_confidence_required=settings["min_confidence"],
                position_multiplier=settings["position_mult"],
                recovery_pct=recovery,
            )

    def reset(self, balance: float) -> None:
        """Reset peak balance (e.g. after manual deposit/withdrawal)."""
        with self._lock:
            self._peak_balance = balance
            self._max_drawdown_seen = 0.0
            self._current_mode = "NORMAL"

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "mode": self._current_mode,
                "peak_balance": self._peak_balance,
                "max_drawdown_seen": self._max_drawdown_seen,
                "thresholds": {
                    "defensive": self.DEFENSIVE_THRESHOLD,
                    "protective": self.PROTECTIVE_THRESHOLD,
                    "emergency": self.EMERGENCY_THRESHOLD,
                },
            }


# ── Singleton ───────────────────────────────────────────────────────

_MONITOR: Optional[DrawdownMonitor] = None


def get_drawdown_monitor() -> DrawdownMonitor:
    global _MONITOR
    if _MONITOR is None:
        _MONITOR = DrawdownMonitor()
    return _MONITOR
