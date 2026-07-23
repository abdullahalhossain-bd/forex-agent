"""
risk/streak_tracker.py — Single source of truth for consecutive-loss streak
=============================================================================

Co-founder fix: CONSOLIDATE STREAK COUNTERS.

Previously, FIVE separate modules maintained their own consecutive-loss
counter, each with different semantics and only ONE (CircuitBreaker)
surviving restart:

  1. risk/circuit_breaker.py        — consecutive_losses (PERSISTED) ✅
  2. risk/live_risk_manager.py      — _consecutive_losses (in-memory)
  3. risk/strict_risk_manager.py    — consecutive_losses (in-memory, dead code)
  4. risk/autonomous_risk.py        — current_streak (in-memory)
  5. risk/position_sizer.py         — consecutive_losses (input param only)

The in-memory counters drift from the persisted one after every restart
(CB's counter survives, the others reset to 0). This means LiveRiskManager
might think there have been 0 consecutive losses (allowing a trade) while
CircuitBreaker knows there have been 2 (about to trip on the 3rd).

This module provides a singleton `StreakTracker` that reads directly from
CircuitBreaker's persisted state file (memory/circuit_breaker_state.json).
All other modules should call `StreakTracker.get_consecutive_losses()`
instead of maintaining their own counter.

This is a READ-ONLY service — writes still go through CircuitBreaker
(record_result), which is the single writer. We just consolidate the
READS so every module sees the same number.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

from utils.logger import get_logger
from core.constants import MEMORY_DIR

log = get_logger("streak_tracker")

# Read from the same path CircuitBreaker uses
_CB_STATE_PATH = MEMORY_DIR / "circuit_breaker_state.json"


class StreakTracker:
    """
    Singleton that provides read-only access to the authoritative
    consecutive-loss count (sourced from CircuitBreaker's persisted state).

    Usage:
        from risk.streak_tracker import StreakTracker
        losses = StreakTracker.get_consecutive_losses()  # → int
        recent = StreakTracker.get_recent_results(limit=10)  # → list[str]
    """

    _instance: Optional["StreakTracker"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._cache_ttl_sec = 2.0  # cache reads for 2 seconds to avoid disk hammering
        self._cache: dict | None = None
        self._cache_ts: float = 0.0

    @classmethod
    def get_instance(cls) -> "StreakTracker":
        """Singleton accessor."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _read_state(self) -> dict:
        """Read CircuitBreaker's persisted state (with 2s cache)."""
        import time
        now = time.time()
        if self._cache is not None and (now - self._cache_ts) < self._cache_ttl_sec:
            return self._cache

        try:
            if _CB_STATE_PATH.exists():
                state = json.loads(_CB_STATE_PATH.read_text(encoding="utf-8"))
                self._cache = state
                self._cache_ts = now
                return state
        except Exception as e:
            log.debug(f"[StreakTracker] read failed (will retry): {e}")

        # On any failure, return a safe default (0 losses = permissive)
        # — fail-OPEN here is fine because CircuitBreaker itself fails CLOSED.
        # If we returned 99, we'd block all trading just because we couldn't
        # read the streak; that's CircuitBreaker's job, not ours.
        default = {"consecutive_losses": 0, "recent_results": []}
        self._cache = default
        self._cache_ts = now
        return default

    @classmethod
    def get_consecutive_losses(cls) -> int:
        """Return the authoritative consecutive-loss count.

        Reads from CircuitBreaker's persisted state. Returns 0 on any
        read failure (fail-open — CB itself fails closed separately).
        """
        return int(cls.get_instance()._read_state().get("consecutive_losses", 0))

    @classmethod
    def get_recent_results(cls, limit: int = 10) -> list[str]:
        """Return the last N trade results (WIN/LOSS) from CB state.

        Useful for modules that need to compute their own win-rate
        without maintaining a separate counter.
        """
        results = cls.get_instance()._read_state().get("recent_results", [])
        return list(results[-limit:]) if results else []

    @classmethod
    def get_win_rate(cls, lookback: int = 10) -> float:
        """Return the win rate (%) over the last `lookback` trades.

        Returns 0.0 if fewer than 5 trades recorded (insufficient data).
        """
        results = cls.get_recent_results(lookback)
        if len(results) < 5:
            return 0.0
        wins = results.count("WIN")
        return round(wins / len(results) * 100, 1)

    @classmethod
    def invalidate_cache(cls) -> None:
        """Force the next read to hit disk. Call this after a known write
        to CircuitBreaker if you need immediate consistency."""
        inst = cls.get_instance()
        inst._cache = None
        inst._cache_ts = 0.0


# Convenience module-level functions (so callers don't need to type
# StreakTracker.get_instance().get_consecutive_losses())
def get_consecutive_losses() -> int:
    return StreakTracker.get_consecutive_losses()


def get_recent_results(limit: int = 10) -> list[str]:
    return StreakTracker.get_recent_results(limit)


def get_win_rate(lookback: int = 10) -> float:
    return StreakTracker.get_win_rate(lookback)


# ── Smoke test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Consecutive losses: {get_consecutive_losses()}")
    print(f"Recent results (last 10): {get_recent_results(10)}")
    print(f"Win rate (last 10): {get_win_rate(10)}%")
    print("StreakTracker smoke test passed.")
