# orchestrator/audit_trail.py — Day 60 | Complete Audit Trail
# ============================================================
# সবকিছু save — কেন trade নিল, কেন reject করল, কেন risk কমাল,
# কেন strategy বদলাল। Full compliance and debugging trail.
#
# Event Types:
#   trade_opened, trade_closed, trade_rejected, decision_made,
#   risk_rejection, strategy_change, system_start, system_stop,
#   error, warning, safety_trigger, human_override, config_change
# ============================================================

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

log = get_logger("audit_trail")

from core.constants import MEMORY_DIR
AUDIT_TRAIL_PATH = MEMORY_DIR / "audit_trail.json"
MAX_AUDIT_ENTRIES = 10000


class AuditTrail:
    """
    Complete audit trail for all system events.
    Every significant action is recorded with full context.
    """

    def __init__(self, trail_path: Path = None):
        self._path = trail_path or AUDIT_TRAIL_PATH
        self._entries: list[dict] = []
        self._load()

    def log_event(self, event_type: str, data: dict, source: str = "system") -> dict:
        """
        Log a system event to the audit trail.
        
        Args:
            event_type: Type of event (e.g., "trade_opened", "risk_rejection")
            data: Event-specific data
            source: Source module that triggered the event
        """
        entry = {
            "id": len(self._entries) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "event_type": event_type,
            "source": source,
            "data": data,
        }

        self._entries.append(entry)

        # Log critical events
        if event_type in ("trade_opened", "trade_closed", "safety_trigger", "human_override"):
            log.info(f"[Audit] {event_type}: {data}")

        return entry

    def log_message(self, msg) -> dict:
        """Log a bus message to the audit trail."""
        return self.log_event(
            f"message:{msg.type}",
            msg.data,
            source=msg.source,
        )

    def get_events(
        self,
        event_type: str = None,
        source: str = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query audit events with filters."""
        entries = self._entries

        if event_type:
            entries = [e for e in entries if e.get("event_type") == event_type]
        if source:
            entries = [e for e in entries if e.get("source") == source]

        return entries[-limit:]

    def get_trade_history(self, limit: int = 50) -> list[dict]:
        """Get all trade-related events."""
        return self.get_events(event_type="trade_opened", limit=limit)

    def get_rejection_history(self, limit: int = 50) -> list[dict]:
        """Get all rejection events."""
        rejections = [
            e for e in self._entries
            if "rejection" in e.get("event_type", "") or "rejected" in str(e.get("data", {}))
        ]
        return rejections[-limit:]

    def get_safety_history(self, limit: int = 50) -> list[dict]:
        """Get all safety-related events."""
        safety = [
            e for e in self._entries
            if "safety" in e.get("event_type", "") or "emergency" in str(e.get("data", {}).get("event", ""))
        ]
        return safety[-limit:]

    def get_stats(self) -> dict:
        """Get audit trail statistics."""
        event_counts = {}
        for entry in self._entries:
            et = entry["event_type"]
            event_counts[et] = event_counts.get(et, 0) + 1

        return {
            "total_events": len(self._entries),
            "event_types": len(event_counts),
            "event_counts": dict(sorted(event_counts.items(), key=lambda x: x[1], reverse=True)[:20]),
            "sources": len(set(e["source"] for e in self._entries)),
        }

    def _load(self) -> None:
        try:
            if self._path.exists():
                with open(self._path, "r") as f:
                    self._entries = json.load(f)
                log.debug(f"[AuditTrail] Loaded {len(self._entries)} entries")
        except Exception as e:
            log.warning(f"[AuditTrail] Load error: {e}")
            self._entries = []

    def _save(self) -> None:
        try:
            MEMORY_DIR.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w") as f:
                json.dump(self._entries[-MAX_AUDIT_ENTRIES:], f, indent=2)
        except Exception as e:
            log.warning(f"[AuditTrail] Save error: {e}")

    def save(self) -> None:
        """Public save method."""
        self._save()

    def print_summary(self) -> None:
        """Print audit trail summary."""
        stats = self.get_stats()
        bar = "=" * 55
        log.info(bar)
        log.info("  AUDIT TRAIL SUMMARY")
        log.info(bar)
        log.info(f"  Total Events : {stats['total_events']}")
        log.info(f"  Event Types  : {stats['event_types']}")
        log.info(f"  Sources      : {stats['sources']}")
        log.info("  Top Events:")
        for et, count in list(stats["event_counts"].items())[:10]:
            log.info(f"    {et:<30} {count}")
        log.info(bar)
