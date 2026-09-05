"""Clock abstractions used by live trading and deterministic replay."""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional


class LiveClock:
    """Wall-clock implementation used by live operation."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def current_date(self) -> date:
        return self.now().date()

    def current_session(self) -> str:
        return _session_for(self.now())


class ReplayClock:
    """Deterministic clock advanced explicitly by the replay driver."""

    def __init__(self, initial: Optional[datetime] = None) -> None:
        self._current: Optional[datetime] = None
        if initial is not None:
            self.advance(initial)

    def now(self) -> datetime:
        if self._current is None:
            raise RuntimeError("ReplayClock has not been initialized")
        return self._current

    def advance(self, timestamp: datetime) -> datetime:
        normalized = _as_utc(timestamp)
        if self._current is not None and normalized < self._current:
            raise ValueError("ReplayClock cannot move backwards")
        self._current = normalized
        return normalized

    def current_date(self) -> date:
        return self.now().date()

    def current_session(self) -> str:
        return _session_for(self.now())


def _as_utc(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _session_for(timestamp: datetime) -> str:
    """Return the coarse UTC session used by replay-safe callers."""
    hour = _as_utc(timestamp).hour
    if 0 <= hour < 8:
        return "ASIA"
    if 8 <= hour < 13:
        return "LONDON"
    if 13 <= hour < 21:
        return "NEW_YORK"
    return "OFF_HOURS"