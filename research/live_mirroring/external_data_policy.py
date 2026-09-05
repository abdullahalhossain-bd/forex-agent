"""P0 historical replay external-data policy.

This module is deliberately small: it does not alter live behavior. Replay
code can use it to make live-only dependencies explicit instead of silently
falling back to current data.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class ExternalStatus(str, Enum):
    DISABLED_BACKTEST = "DISABLED_BACKTEST"
    HISTORICAL = "HISTORICAL"
    ASSUMED = "DEFAULT_ASSUMPTION"
    OBSERVED = "OBSERVED"


@dataclass(frozen=True)
class ExternalValue:
    name: str
    status: ExternalStatus
    value: Any = None
    source_timestamp: Optional[str] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "value": self.value,
            "source_timestamp": self.source_timestamp,
            "reason": self.reason,
        }


LIVE_ONLY = {
    "economic_calendar": "DISABLED_BACKTEST",
    "live_news": "DISABLED_BACKTEST",
    "current_sentiment": "DISABLED_BACKTEST",
    "external_macro": "DISABLED_BACKTEST",
    "network_state": "DISABLED_BACKTEST",
    "live_microstructure": "DISABLED_BACKTEST",
    "live_institutional_flow": "DISABLED_BACKTEST",
}


def disabled(name: str, reason: str = "live/current external source is not historical") -> ExternalValue:
    """Return an explicit disabled marker; never substitute current data."""
    return ExternalValue(
        name=name,
        status=ExternalStatus.DISABLED_BACKTEST,
        value=None,
        reason=reason,
    )


def require_timestamped(
    name: str,
    value: Any,
    source_timestamp: Optional[str],
    replay_timestamp: str,
) -> ExternalValue:
    """Accept an external value only when it has an explicit source timestamp.

    Callers must additionally validate that the source timestamp is <= replay
    timestamp. This function intentionally does not guess timestamps.
    """
    if value is None or not source_timestamp:
        return disabled(name, "missing timestamped historical value")
    if source_timestamp > replay_timestamp:
        raise ValueError(
            f"look-ahead external data: {name} source_timestamp={source_timestamp} "
            f"> replay_timestamp={replay_timestamp}"
        )
    return ExternalValue(
        name=name,
        status=ExternalStatus.HISTORICAL,
        value=value,
        source_timestamp=source_timestamp,
        reason="timestamped historical source",
    )
