# orchestrator/trading_sessions.py — Time-based trading session management
# =============================================================================
# Ported from: https://github.com/Ichinga-Samuel/aiomql/blob/master/src/aiomql/lib/sessions.py
# Original author: Ichinga-Samuel — MIT license
#
# Trading sessions define time windows when trading is allowed. At session
# boundaries, automatic actions can be triggered:
#   - close_all: close all open positions
#   - close_win: close only winning positions
#   - close_loss: close only losing positions
#   - custom: call a user-defined callback
#
# Example:
#   London session: 08:00-16:00 UTC, close_all on end
#   NY session: 13:00-21:00 UTC, close_loss on end
#   Tokyo session: 00:00-08:00 UTC, no auto-action
#
# The Sessions container manages multiple sessions, finds the current/next
# session, and can be used as a context manager.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from typing import Callable, Optional, Literal, NamedTuple

from utils.logger import get_logger

log = get_logger("trading_sessions")


class Duration(NamedTuple):
    """Session duration breakdown."""
    hours: int
    minutes: int
    seconds: int


def _delta(t: time) -> timedelta:
    """Convert datetime.time to timedelta from midnight."""
    return timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)


@dataclass
class Session:
    """
    A trading session representing a time window in UTC.

    Attributes
    ----------
    start : UTC start time (datetime.time or int hour).
    end : UTC end time (datetime.time or int hour).
    on_start : action at session start: "close_all", "close_win", "close_loss", or None.
    on_end : action at session end: "close_all", "close_win", "close_loss", or None.
    name : human-readable name.
    """

    start: time
    end: time
    on_start: Optional[str] = None  # "close_all" | "close_win" | "close_loss" | None
    on_end: Optional[str] = None
    name: str = ""
    custom_start: Optional[Callable] = None
    custom_end: Optional[Callable] = None

    def __post_init__(self):
        # Allow int hour input
        if isinstance(self.start, int):
            self.start = time(hour=self.start, tzinfo=timezone.utc)
        if isinstance(self.end, int):
            self.end = time(hour=self.end, tzinfo=timezone.utc)
        # Ensure UTC
        if self.start.tzinfo is None:
            self.start = self.start.replace(tzinfo=timezone.utc)
        if self.end.tzinfo is None:
            self.end = self.end.replace(tzinfo=timezone.utc)
        if not self.name:
            self.name = f"{self.start.strftime('%H:%M')}→{self.end.strftime('%H:%M')}"

    def __contains__(self, moment: time) -> bool:
        """Check if a time falls within this session."""
        span = (_delta(self.end) - _delta(self.start)).seconds
        item_span = (_delta(self.end) - _delta(moment)).seconds
        return 0 <= item_span <= span

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"Session({self.name})"

    def __len__(self) -> int:
        """Duration in seconds."""
        return int((_delta(self.end) - _delta(self.start)).seconds)

    def in_session(self) -> bool:
        """True if current UTC time is within this session."""
        now = datetime.now(timezone.utc).time()
        return now in self

    def duration(self) -> Duration:
        """Get duration as (hours, minutes, seconds)."""
        total = len(self)
        hours, seconds = divmod(total, 3600)
        minutes, seconds = divmod(seconds, 60)
        return Duration(hours=hours, minutes=minutes, seconds=seconds)

    def until_start(self) -> int:
        """Seconds until session starts (negative if already started)."""
        secs = (_delta(self.start) - _delta(datetime.now(timezone.utc).time())).seconds
        return secs

    def until_end(self) -> int:
        """Seconds until session ends (negative if already ended)."""
        secs = (_delta(self.end) - _delta(datetime.now(timezone.utc).time())).seconds
        return secs

    def should_trigger_start(self) -> bool:
        """True if on_start action should fire (just entered session)."""
        return self.in_session() and self.on_start is not None

    def should_trigger_end(self) -> bool:
        """True if on_end action should fire (just left session)."""
        return not self.in_session() and self.on_end is not None


class Sessions:
    """
    Container for multiple trading sessions with automatic management.

    Sessions are sorted by start time. The container can find the current
    session, the next session, and check if trading is currently allowed.

    Usage
    -----
        sessions = Sessions()
        sessions.add(Session(start=8, end=16, on_end="close_all", name="London"))
        sessions.add(Session(start=13, end=21, on_end="close_loss", name="NY"))

        if sessions.is_trading_time():
            # Execute trades
            pass
        else:
            next_session = sessions.find_next()
            print(f"Next session: {next_session.name} in {next_session.until_start()}s")
    """

    def __init__(self, sessions: Optional[list[Session]] = None):
        self.sessions: list[Session] = sessions or []
        self._sort()
        self._was_in_session: dict[str, bool] = {}

    def add(self, session: Session) -> None:
        """Add a session to the collection."""
        self.sessions.append(session)
        self._sort()

    def _sort(self) -> None:
        self.sessions.sort(key=lambda s: (s.start.hour, s.start.minute))

    def find_current(self) -> Optional[Session]:
        """Find the session containing the current UTC time, or None."""
        now = datetime.now(timezone.utc).time()
        for s in self.sessions:
            if now in s:
                return s
        return None

    def find_next(self) -> Optional[Session]:
        """Find the next session after current time. Wraps to first if at end of day."""
        if not self.sessions:
            return None
        now = datetime.now(timezone.utc).time()
        for s in self.sessions:
            if _delta(now) < _delta(s.start):
                return s
        return self.sessions[0]  # wrap to first session

    def is_trading_time(self) -> bool:
        """True if current time is within any session."""
        return self.find_current() is not None

    def get_pending_actions(self) -> list[tuple[Session, str]]:
        """
        Check for session boundary transitions and return pending actions.

        Returns list of (session, action_type) tuples:
            ("start", "close_all") — session just started, run on_start action
            ("end", "close_loss") — session just ended, run on_end action
        """
        actions = []
        for s in self.sessions:
            key = s.name
            was_in = self._was_in_session.get(key, False)
            now_in = s.in_session()

            if now_in and not was_in and s.on_start:
                actions.append((s, "start"))
            elif not now_in and was_in and s.on_end:
                actions.append((s, "end"))

            self._was_in_session[key] = now_in

        return actions

    def list_sessions(self) -> list[dict]:
        """Return a summary of all sessions."""
        result = []
        for s in self.sessions:
            result.append({
                "name": s.name,
                "start": s.start.strftime("%H:%M"),
                "end": s.end.strftime("%H:%M"),
                "duration": str(s.duration()),
                "on_start": s.on_start,
                "on_end": s.on_end,
                "in_session": s.in_session(),
                "until_start": s.until_start(),
                "until_end": s.until_end(),
            })
        return result

    def __len__(self) -> int:
        return len(self.sessions)

    def __contains__(self, session: Session) -> bool:
        return session in self.sessions


# ── Preset sessions ──────────────────────────────────────────────────────────

def forex_sessions() -> Sessions:
    """Create the standard forex trading sessions (UTC)."""
    return Sessions([
        Session(start=time(0, 0, tzinfo=timezone.utc),
                end=time(8, 0, tzinfo=timezone.utc),
                name="Tokyo"),
        Session(start=time(8, 0, tzinfo=timezone.utc),
                end=time(16, 0, tzinfo=timezone.utc),
                on_end="close_all",
                name="London"),
        Session(start=time(13, 0, tzinfo=timezone.utc),
                end=time(21, 0, tzinfo=timezone.utc),
                on_end="close_loss",
                name="New York"),
        Session(start=time(21, 0, tzinfo=timezone.utc),
                end=time(23, 59, tzinfo=timezone.utc),
                name="Sydney"),
    ])


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Create sessions
    sessions = Sessions()
    sessions.add(Session(start=8, end=16, on_end="close_all", name="London"))
    sessions.add(Session(start=13, end=21, on_end="close_loss", name="NY"))
    sessions.add(Session(start=0, end=8, name="Tokyo"))

    print(f"Sessions: {len(sessions)}")
    for s in sessions.list_sessions():
        print(f"  {s['name']:10s} {s['start']}→{s['end']} "
              f"in_session={s['in_session']} until_start={s['until_start']}s")

    # Find current
    current = sessions.find_current()
    print(f"\nCurrent session: {current}")
    print(f"Trading time: {sessions.is_trading_time()}")

    # Find next
    nxt = sessions.find_next()
    print(f"Next session: {nxt}")

    # Pending actions (first call initializes state — no transitions)
    actions = sessions.get_pending_actions()
    print(f"Pending actions (initial): {actions}")

    # Check boundary detection
    # Simulate: call again, no transitions expected
    actions2 = sessions.get_pending_actions()
    print(f"Pending actions (no change): {actions2}")

    # Preset forex sessions
    fx = forex_sessions()
    print(f"\nForex sessions: {len(fx)}")
    for s in fx.list_sessions():
        print(f"  {s['name']:10s} {s['start']}→{s['end']} on_end={s['on_end']}")

    # Duration
    london = sessions.sessions[0]
    dur = london.duration()
    print(f"\nLondon duration: {dur.hours}h {dur.minutes}m {dur.seconds}s")
    assert dur.hours == 8

    print("\nTrading sessions smoke test passed.")
