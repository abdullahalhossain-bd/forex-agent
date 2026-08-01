"""
core/runtime_metrics.py — Runtime metrics collector
====================================================

A single place to record runtime metrics (cycle counts, stage timings,
error counts, trade counts). Existing `automation/runtime_metrics.py` has
the same idea but is never wired in. This module is the canonical runtime
replacement — the legacy one is marked obsolete in `core/obsolete.py`.

Public API:
  * `RuntimeMetrics` — the collector class.
  * `get_metrics()` — singleton accessor.
  * `metric(name, value)` — convenience publish.
  * `timer(name)` — context manager for stage timing.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional

from core.event_bus import get_bus

log = logging.getLogger(__name__)


@dataclass
class StageStat:
    name: str
    count: int = 0
    total_sec: float = 0.0
    min_sec: float = float("inf")
    max_sec: float = 0.0
    last_sec: float = 0.0
    errors: int = 0

    def record(self, elapsed: float, error: bool = False) -> None:
        self.count += 1
        self.total_sec += elapsed
        self.min_sec = min(self.min_sec, elapsed)
        self.max_sec = max(self.max_sec, elapsed)
        self.last_sec = elapsed
        if error:
            self.errors += 1

    def to_dict(self) -> Dict[str, Any]:
        avg = self.total_sec / self.count if self.count else 0.0
        return {
            "count": self.count,
            "total_sec": round(self.total_sec, 3),
            "avg_sec": round(avg, 3),
            "min_sec": round(self.min_sec, 3) if self.count else 0.0,
            "max_sec": round(self.max_sec, 3),
            "last_sec": round(self.last_sec, 3),
            "errors": self.errors,
        }


class RuntimeMetrics:
    """Thread-safe collector of runtime metrics with rolling history."""

    def __init__(self, history_size: int = 500):
        self._lock = threading.RLock()
        self._stages: Dict[str, StageStat] = {}
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._events: Deque[Dict[str, Any]] = deque(maxlen=history_size)
        self._started_at = time.time()
        self._cycle_count = 0
        self._error_count = 0
        self._reconnect_count = 0
        self._bus = get_bus()

    # ── counters & gauges ───────────────────────────────────────────

    def inc(self, name: str, amount: float = 1.0) -> None:
        with self._lock:
            self._counters[name] += amount

    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def get_counter(self, name: str) -> float:
        with self._lock:
            return self._counters.get(name, 0.0)

    def get_gauge(self, name: str) -> Optional[float]:
        with self._lock:
            return self._gauges.get(name)

    # ── cycle & error helpers ───────────────────────────────────────

    def record_cycle(self) -> None:
        with self._lock:
            self._cycle_count += 1

    def record_error(self, channel: str = "runtime") -> None:
        with self._lock:
            self._error_count += 1
        self._bus.publish("system.error", {"channel": channel}, source="runtime_metrics")
        self.inc(f"errors.{channel}")

    def record_reconnect(self) -> None:
        with self._lock:
            self._reconnect_count += 1
        self._bus.publish("broker.reconnect", {}, source="runtime_metrics")

    # ── stage timing ────────────────────────────────────────────────

    @contextmanager
    def timer(self, stage_name: str) -> Iterator[StageStat]:
        t0 = time.time()
        err = False
        try:
            with self._lock:
                stat = self._stages.setdefault(stage_name, StageStat(name=stage_name))
            yield stat
        except Exception as e:
            err = True
            raise
        finally:
            elapsed = time.time() - t0
            with self._lock:
                stat = self._stages.get(stage_name)
                if stat is None:
                    stat = StageStat(name=stage_name)
                    self._stages[stage_name] = stat
                stat.record(elapsed, error=err)

    def record_stage(self, stage_name: str, elapsed: float, error: bool = False) -> None:
        with self._lock:
            stat = self._stages.setdefault(stage_name, StageStat(name=stage_name))
            stat.record(elapsed, error=error)

    # ── events ──────────────────────────────────────────────────────

    def log_event(self, kind: str, detail: Dict[str, Any]) -> None:
        evt = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "kind": kind,
            "detail": detail,
        }
        with self._lock:
            self._events.append(evt)

    def recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._events)[-limit:]

    # ── snapshots ───────────────────────────────────────────────────

    def build_report(self) -> Dict[str, Any]:
        with self._lock:
            stages = {n: s.to_dict() for n, s in self._stages.items()}
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            uptime_sec = time.time() - self._started_at
            cycles = self._cycle_count
            errors = self._error_count
            reconnects = self._reconnect_count
        avg_cycle = uptime_sec / cycles if cycles else 0.0
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "uptime_sec": round(uptime_sec, 1),
            "cycle_count": cycles,
            "avg_cycle_sec": round(avg_cycle, 2),
            "error_count": errors,
            "reconnect_count": reconnects,
            "stages": stages,
            "counters": counters,
            "gauges": gauges,
        }

    def snapshot_to_bus(self) -> None:
        """Push a metrics snapshot to the event bus (consumed by dashboard)."""
        self._bus.publish("analytics.metric", self.build_report(), source="runtime_metrics")


# ── singleton ───────────────────────────────────────────────────────

_METRICS: Optional[RuntimeMetrics] = None


def get_metrics() -> RuntimeMetrics:
    global _METRICS
    if _METRICS is None:
        _METRICS = RuntimeMetrics()
    return _METRICS


def metric(name: str, value: float) -> None:
    get_metrics().set_gauge(name, value)
