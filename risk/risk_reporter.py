"""
risk/risk_reporter.py — Risk Event Reporter (Day 75)
=====================================================

Records risk events to SQLite and sends Telegram alerts when
critical risk thresholds are hit.

Events tracked:
  - DAILY_LIMIT_HIT
  - WEEKLY_LIMIT_HIT
  - MAX_DRAWDOWN_HIT
  - LOSS_STREAK_WARNING
  - EXPOSURE_REJECTED
  - CAPITAL_PRESERVATION_ACTIVATED
  - KILL_SWITCH_TRIGGERED
  - HUMAN_OVERRIDE
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from utils.logger import get_logger

log = get_logger("risk_reporter")

DB_PATH = Path("memory/risk_events.db")


class RiskReporter:
    """Records risk events + sends Telegram alerts."""

    def __init__(self):
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self) -> None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(DB_PATH)) as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS risk_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    trigger_value TEXT,
                    action_taken TEXT,
                    details TEXT,
                    timestamp TEXT NOT NULL
                )
            """)
            c.commit()

    def record_event(
        self,
        event_type: str,
        trigger_value: str = "",
        action_taken: str = "",
        details: str = "",
        send_telegram: bool = True,
    ) -> int:
        """Record a risk event and optionally send Telegram alert.

        Args:
            event_type: DAILY_LIMIT_HIT / WEEKLY_LIMIT_HIT / etc.
            trigger_value: What value triggered the event.
            action_taken: What the system did.
            details: Additional context.
            send_telegram: Whether to send Telegram alert.

        Returns:
            Event ID from DB.
        """
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock, sqlite3.connect(str(DB_PATH)) as c:
            cur = c.execute(
                "INSERT INTO risk_events (event_type, trigger_value, action_taken, details, timestamp) VALUES (?, ?, ?, ?, ?)",
                (event_type, trigger_value, action_taken, details, ts),
            )
            c.commit()
            event_id = cur.lastrowid

        log.warning(f"[RiskReporter] EVENT: {event_type} | {trigger_value} | {action_taken}")

        if send_telegram:
            self._send_telegram(event_type, trigger_value, action_taken)

        return event_id

    def _send_telegram(self, event_type: str, trigger_value: str, action: str) -> None:
        """Send Telegram alert for critical risk events."""
        try:
            from core.service_registry import get_registry
            registry = get_registry()
            notifier = registry.try_resolve("telegram_notifier")
            if not notifier:
                return

            emoji_map = {
                "DAILY_LIMIT_HIT": "⚠️",
                "WEEKLY_LIMIT_HIT": "🟠",
                "MAX_DRAWDOWN_HIT": "🔴",
                "LOSS_STREAK_WARNING": "⚠️",
                "EXPOSURE_REJECTED": "🟡",
                "CAPITAL_PRESERVATION_ACTIVATED": "🛡",
                "KILL_SWITCH_TRIGGERED": "🚨",
                "HUMAN_OVERRIDE": "👤",
            }
            emoji = emoji_map.get(event_type, "⚠️")

            msg = (
                f"{emoji} FOREX AI RISK ALERT\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"Event: {event_type}\n"
                f"Trigger: {trigger_value}\n"
                f"Action: {action}\n"
                f"━━━━━━━━━━━━━━━━━━━━━"
            )

            from utils.async_utils import run_coro_sync
            run_coro_sync(notifier.send_message(msg))
        except Exception as e:
            log.debug(f"[RiskReporter] Telegram send failed: {e}")

    def get_recent_events(self, limit: int = 20) -> list:
        """Get recent risk events."""
        try:
            with sqlite3.connect(str(DB_PATH)) as c:
                rows = c.execute(
                    "SELECT * FROM risk_events ORDER BY timestamp DESC LIMIT ?", (limit,)
                ).fetchall()
            cols = ["id", "event_type", "trigger_value", "action_taken", "details", "timestamp"]
            return [dict(zip(cols, row)) for row in rows]
        except Exception as e:
            log.warning(f"Suppressed exception at line 146: {e}")
            return []

    def stats(self) -> Dict[str, Any]:
        """Return risk event statistics."""
        try:
            with sqlite3.connect(str(DB_PATH)) as c:
                total = c.execute("SELECT COUNT(*) FROM risk_events").fetchone()[0]
                by_type = c.execute(
                    "SELECT event_type, COUNT(*) FROM risk_events GROUP BY event_type ORDER BY COUNT(*) DESC"
                ).fetchall()
            return {
                "total_events": total,
                "by_type": dict(by_type),
            }
        except Exception as e:
            return {"total_events": 0, "by_type": {}}


# ── Singleton ───────────────────────────────────────────────────────

_REPORTER: Optional[RiskReporter] = None


def get_risk_reporter() -> RiskReporter:
    global _REPORTER
    if _REPORTER is None:
        _REPORTER = RiskReporter()
    return _REPORTER