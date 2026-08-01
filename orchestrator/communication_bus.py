# orchestrator/communication_bus.py — Day 60 | Agent Communication Bus
# ============================================================
# Decoupled agent communication layer. Agents never call each other
# directly — all communication flows through this message bus.
#
# Benefits:
#   - Less dependency between agents
#   - Easy debugging (all messages logged)
#   - Scalable architecture
#   - Message history for audit trail
#   - Async-capable message queue
#
# Message Types:
#   market_analysis, smc_analysis, pattern_analysis, sentiment_analysis,
#   news_filter, decision, risk_check, execution, learning, research,
#   system_event, error, warning, heartbeat
# ============================================================

import json
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Optional
from pathlib import Path

from utils.logger import get_logger

log = get_logger("communication_bus")

from core.constants import MEMORY_DIR
BUS_HISTORY_PATH = MEMORY_DIR / "message_bus_history.json"
MAX_HISTORY = 5000  # Keep last 5000 messages


class AgentMessage:
    """
    Single message on the bus.
    
    Schema:
        {
            "id": "msg_1700000000_123",
            "source": "market_agent",
            "type": "market_analysis",
            "data": {"trend": "bullish", "regime": "trending"},
            "timestamp": "2024-01-01T00:00:00Z",
            "cycle_id": "cycle_1700000000",
            "correlation_id": null,
            "priority": "normal",
            "metadata": {}
        }
    """

    _counter = 0

    def __init__(
        self,
        source: str,
        msg_type: str,
        data: dict,
        priority: str = "normal",
        cycle_id: str = None,
        correlation_id: str = None,
        metadata: dict = None,
    ):
        AgentMessage._counter += 1
        self.id = f"msg_{int(time.time())}_{AgentMessage._counter}"
        self.source = source
        self.type = msg_type
        self.data = data
        self.timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.cycle_id = cycle_id or f"cycle_{int(time.time())}"
        self.correlation_id = correlation_id
        self.priority = priority  # low, normal, high, critical
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "type": self.type,
            "data": self.data,
            "timestamp": self.timestamp,
            "cycle_id": self.cycle_id,
            "correlation_id": self.correlation_id,
            "priority": self.priority,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AgentMessage":
        msg = cls(
            source=d["source"],
            msg_type=d["type"],
            data=d["data"],
            priority=d.get("priority", "normal"),
            cycle_id=d.get("cycle_id"),
            correlation_id=d.get("correlation_id"),
            metadata=d.get("metadata"),
        )
        msg.id = d["id"]
        msg.timestamp = d["timestamp"]
        return msg

    def __repr__(self) -> str:
        return f"<AgentMessage {self.source}→{self.type} [{self.priority}]>"


class AgentMessageBus:
    """
    Central communication bus for all agent interactions.
    
    Usage:
        bus = AgentMessageBus()
        bus.subscribe("decision", my_callback)
        bus.publish(AgentMessage("market_agent", "market_analysis", {...}))
        bus.get_history(source="risk_agent", msg_type="risk_check")
    """

    PRIORITY_ORDER = {"critical": 0, "high": 1, "normal": 2, "low": 3}

    def __init__(self, max_history: int = MAX_HISTORY):
        self._subscribers: dict[str, list[Callable]] = {}
        self._wildcard_subscribers: list[Callable] = []
        self._history: deque = deque(maxlen=max_history)
        self._pending: deque = deque()
        self._error_count = 0
        self._message_count = 0
        self._load_history()

    def publish(self, message: AgentMessage) -> bool:
        """
        Publish a message. All matching subscribers are notified.
        Returns True if at least one subscriber was notified.
        """
        self._message_count += 1
        self._history.append(message.to_dict())

        notified = False

        # Type-specific subscribers
        subscribers = self._subscribers.get(message.type, [])
        for callback in subscribers:
            try:
                callback(message)
                notified = True
            except Exception as e:
                self._error_count += 1
                log.error(
                    f"[Bus] Subscriber error for {message.type}: {e}", exc_info=True
                )

        # Wildcard subscribers (receive ALL messages)
        for callback in self._wildcard_subscribers:
            try:
                callback(message)
                notified = True
            except Exception as e:
                self._error_count += 1
                log.error(f"[Bus] Wildcard subscriber error: {e}", exc_info=True)

        log.debug(f"[Bus] {message.source} → {message.type} (notified={notified})")
        return notified

    def subscribe(self, msg_type: str, callback: Callable) -> None:
        """Subscribe to a specific message type."""
        if msg_type not in self._subscribers:
            self._subscribers[msg_type] = []
        self._subscribers[msg_type].append(callback)
        log.debug(f"[Bus] {callback.__name__} subscribed to '{msg_type}'")

    def subscribe_all(self, callback: Callable) -> None:
        """Subscribe to ALL messages (wildcard)."""
        self._wildcard_subscribers.append(callback)
        log.debug(f"[Bus] {callback.__name__} subscribed to ALL messages")

    def unsubscribe(self, msg_type: str, callback: Callable) -> None:
        """Remove a subscriber."""
        if msg_type in self._subscribers:
            self._subscribers[msg_type] = [
                cb for cb in self._subscribers[msg_type] if cb != callback
            ]

    def get_history(
        self,
        source: str = None,
        msg_type: str = None,
        cycle_id: str = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query message history with optional filters."""
        messages = list(self._history)

        if source:
            messages = [m for m in messages if m["source"] == source]
        if msg_type:
            messages = [m for m in messages if m["type"] == msg_type]
        if cycle_id:
            messages = [m for m in messages if m.get("cycle_id") == cycle_id]

        return messages[-limit:]

    def get_last_message(self, source: str = None, msg_type: str = None) -> Optional[dict]:
        """Get the most recent message matching filters."""
        history = self.get_history(source=source, msg_type=msg_type, limit=1)
        return history[0] if history else None

    def get_cycle_messages(self, cycle_id: str) -> list[dict]:
        """Get all messages from a specific trading cycle."""
        return self.get_history(cycle_id=cycle_id, limit=1000)

    def clear_history(self) -> None:
        """Clear message history."""
        self._history.clear()

    def get_stats(self) -> dict:
        """Get bus statistics."""
        return {
            "total_messages": self._message_count,
            "history_size": len(self._history),
            "error_count": self._error_count,
            "subscriber_types": len(self._subscribers),
            "wildcard_subscribers": len(self._wildcard_subscribers),
        }

    def _load_history(self) -> None:
        """Load persisted message history from disk."""
        try:
            if BUS_HISTORY_PATH.exists():
                with open(BUS_HISTORY_PATH, "r") as f:
                    saved = json.load(f)
                for msg_dict in saved[-MAX_HISTORY:]:
                    self._history.append(msg_dict)
                log.debug(f"[Bus] Loaded {len(self._history)} historical messages")
        except Exception as e:
            log.warning(f"[Bus] Could not load history: {e}")

    def save_history(self) -> None:
        """Persist message history to disk."""
        try:
            MEMORY_DIR.mkdir(parents=True, exist_ok=True)
            with open(BUS_HISTORY_PATH, "w") as f:
                json.dump(list(self._history)[-MAX_HISTORY:], f, indent=2)
        except Exception as e:
            log.warning(f"[Bus] Could not save history: {e}")

    def print_summary(self) -> None:
        """Print bus summary to log."""
        stats = self.get_stats()
        bar = "=" * 50
        log.info(bar)
        log.info("  AGENT COMMUNICATION BUS")
        log.info(bar)
        log.info(f"  Total Messages : {stats['total_messages']}")
        log.info(f"  History Size  : {stats['history_size']}")
        log.info(f"  Subscriptions : {stats['subscriber_types']} type(s)")
        log.info(f"  Wildcard Subs : {stats['wildcard_subscribers']}")
        log.info(f"  Errors        : {stats['error_count']}")
        log.info(bar)
