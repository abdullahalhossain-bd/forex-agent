"""
core/event_bus.py — Unified cross-module event bus
==================================================

A simple, thread-safe in-process pub/sub bus that every runtime module can
publish to and subscribe to. This replaces the broken
`orchestrator/communication_bus.py` (which depends on the missing
orchestrator sub-modules) and the various ad-hoc notifier callbacks scattered
across the codebase.

Channels (canonical names — keep these stable):
  * system.error       — any unhandled exception in a runtime component
  * system.warning     — non-fatal degradation (e.g. data fetch failed)
  * system.startup     — fired once per service during boot
  * system.shutdown    — fired once per service during graceful stop
  * risk.event         — risk-gate triggered (circuit breaker, daily-loss, etc.)
  * risk.circuit_breaker
  * trade.execution    — order placed (paper or mt5)
  * trade.close        — position closed
  * broker.failure     — MT5 disconnect, order rejection, etc.
  * broker.reconnect
  * signal.generated   — AI/scanner produced a trading signal
  * webhook.command    — external command received via webhook
  * learning.feedback  — closed-trade outcome routed back to learning system
  * analytics.metric   — periodic metric snapshot (consumed by dashboard)
  * health.report      — periodic health snapshot (consumed by dashboard)

Public API:
  * `EventBus` — the bus class.
  * `Event` — the event payload dataclass.
  * `get_bus()` — module-level singleton accessor.
  * `publish(channel, payload)` — convenience wrapper around get_bus().
  * `subscribe(channel, handler)` — convenience wrapper.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Deque, Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class Event:
    """A single event on the bus."""
    channel: str
    payload: Any = None
    timestamp: float = field(default_factory=time.time)
    source: Optional[str] = None
    sequence: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel": self.channel,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "source": self.source,
            "sequence": self.sequence,
        }


Handler = Callable[[Event], None]


class EventBus:
    """Thread-safe in-process pub/sub bus with bounded history."""

    def __init__(self, history_size: int = 500):
        self._subscribers: Dict[str, List[Handler]] = defaultdict(list)
        self._wildcard_subscribers: List[Handler] = []
        self._history: Deque[Event] = deque(maxlen=history_size)
        self._lock = threading.RLock()
        self._sequence = 0
        self._history_size = history_size

    def subscribe(self, channel: str, handler: Handler) -> Callable[[], None]:
        """Subscribe `handler` to `channel`. Returns an unsubscribe callable.
        Use channel='*' to subscribe to every channel."""
        with self._lock:
            if channel == "*":
                self._wildcard_subscribers.append(handler)
            else:
                self._subscribers[channel].append(handler)
        def _unsubscribe():
            with self._lock:
                if channel == "*":
                    if handler in self._wildcard_subscribers:
                        self._wildcard_subscribers.remove(handler)
                elif handler in self._subscribers.get(channel, []):
                    self._subscribers[channel].remove(handler)
        return _unsubscribe

    def publish(self, channel: str, payload: Any = None, source: Optional[str] = None) -> Event:
        """Publish an event. Handlers run synchronously on the caller's thread.
        Any handler exception is caught and logged so a bad listener can't
        take down the publisher."""
        with self._lock:
            self._sequence += 1
            evt = Event(
                channel=channel,
                payload=payload,
                source=source,
                sequence=self._sequence,
            )
            self._history.append(evt)
            handlers = list(self._subscribers.get(channel, [])) + list(self._wildcard_subscribers)
        for h in handlers:
            try:
                h(evt)
            except Exception as e:
                log.error("EventBus handler %s failed on channel '%s': %s",
                          getattr(h, "__name__", h), channel, e, exc_info=True)
        log.debug("bus: %s <- %s", channel, source or "?")
        return evt

    def history(self, channel: Optional[str] = None, limit: int = 100) -> List[Event]:
        with self._lock:
            events = list(self._history)
        if channel:
            events = [e for e in events if e.channel == channel]
        return events[-limit:]

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()

    def subscriber_count(self, channel: Optional[str] = None) -> int:
        with self._lock:
            if channel is None:
                return sum(len(v) for v in self._subscribers.values()) + len(self._wildcard_subscribers)
            return len(self._subscribers.get(channel, []))

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            per_channel = {ch: len(v) for ch, v in self._subscribers.items()}
            return {
                "total_subscribers": sum(per_channel.values()) + len(self._wildcard_subscribers),
                "wildcard_subscribers": len(self._wildcard_subscribers),
                "per_channel": per_channel,
                "events_recorded": len(self._history),
                "history_capacity": self._history_size,
                "last_sequence": self._sequence,
            }


# ── module-level singleton ──────────────────────────────────────────

_BUS: Optional[EventBus] = None
_BUS_LOCK = threading.Lock()


def get_bus() -> EventBus:
    global _BUS
    if _BUS is None:
        with _BUS_LOCK:
            if _BUS is None:
                _BUS = EventBus()
    return _BUS


def publish(channel: str, payload: Any = None, source: Optional[str] = None) -> Event:
    return get_bus().publish(channel, payload, source)


def subscribe(channel: str, handler: Handler) -> Callable[[], None]:
    return get_bus().subscribe(channel, handler)
