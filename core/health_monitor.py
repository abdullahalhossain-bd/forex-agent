
from __future__ import annotations

import logging
import platform
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from core.event_bus import EventBus, get_bus
from core.service_registry import (
    ServiceRegistry,
    ServiceStatus,
    get_registry,
)

log = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    name: str
    fn: Callable[[], Dict[str, Any]]
    critical: bool = False
    last_result: Optional[Dict[str, Any]] = None
    last_run: Optional[float] = None


@dataclass
class HealthSnapshot:
    timestamp: float
    overall: HealthStatus
    services: Dict[str, Dict[str, Any]]
    checks: Dict[str, Dict[str, Any]]
    system: Dict[str, Any]

    def to_dict(self):
        return {
            "timestamp":
                datetime.fromtimestamp(
                    self.timestamp,
                    tz=timezone.utc,
                ).isoformat(),
            "overall": self.overall.value,
            "services": self.services,
            "checks": self.checks,
            "system": self.system,
        }


class HealthMonitor:
    def __init__(
        self,
        registry: Optional[ServiceRegistry] = None,
        bus: Optional[EventBus] = None,
        interval_sec: float = 30.0,
    ):
        self.registry = registry or get_registry()
        self.bus = bus or get_bus()

        # minimum 15 sec
        self.interval_sec = max(15.0, interval_sec)

        self._checks: Dict[str, HealthCheck] = {}
        self._history: List[HealthSnapshot] = []
        self._history_max = 100

        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.RLock()

        self._last_snapshot = None

    # --------------------------------------------------

    def register_check(
        self,
        name: str,
        fn: Callable[[], Dict[str, Any]],
        critical: bool = False,
    ):
        with self._lock:
            self._checks[name] = HealthCheck(
                name=name,
                fn=fn,
                critical=critical,
            )

    # --------------------------------------------------

    def start(self):
        if self._thread and self._thread.is_alive():
            return

        self._stop.clear()

        self._thread = threading.Thread(
            target=self._loop,
            name="health-monitor",
            daemon=True,
        )

        self._thread.start()

        log.info(
            "HealthMonitor started (interval=%ss)",
            self.interval_sec,
        )

    # --------------------------------------------------

    def stop(self, timeout: float = 5.0):
        self._stop.set()

        if self._thread:
            self._thread.join(timeout=timeout)

        log.info("HealthMonitor stopped")

    # --------------------------------------------------

    def _loop(self):
        while not self._stop.is_set():
            try:
                snap = self.run_once()

                self.bus.publish(
                    "health.report",
                    snap.to_dict(),
                    source="health_monitor",
                )

            except Exception as e:
                log.exception(
                    "HealthMonitor loop error"
                )

            self._stop.wait(self.interval_sec)

    # --------------------------------------------------

    def run_once(self):
        services = self.registry.health()

        checks = {}

        with self._lock:
            check_items = list(
                self._checks.items()
            )

        for name, chk in check_items:
            t0 = time.time()

            try:
                result = chk.fn() or {}

                if "ok" not in result:
                    result["ok"] = True

            except Exception as e:
                result = {
                    "ok": False,
                    "error": str(e),
                }

            result["duration_ms"] = round(
                (time.time() - t0) * 1000,
                1,
            )

            chk.last_result = result
            chk.last_run = time.time()

            checks[name] = result

        system = self._collect_system_metrics()

        overall = self._compute_overall(
            services,
            checks,
        )

        snap = HealthSnapshot(
            timestamp=time.time(),
            overall=overall,
            services=services,
            checks=checks,
            system=system,
        )

        with self._lock:
            self._history.append(snap)

            if len(self._history) > self._history_max:
                self._history.pop(0)

            self._last_snapshot = snap

        return snap

    # --------------------------------------------------

    def _compute_overall(
        self,
        services,
        checks,
    ):
        failed = False
        degraded = False

        for _, info in services.items():
            st = info.get(
                "status",
                "registered",
            )

            if st == ServiceStatus.FAILED.value:
                failed = True

            elif st in (
                ServiceStatus.DEGRADED.value,
                ServiceStatus.SHUTDOWN.value,
            ):
                degraded = True

        for name, chk in checks.items():
            if not chk.get("ok", False):
                rec = self._checks.get(name)

                if rec and rec.critical:
                    failed = True
                else:
                    degraded = True

        if failed:
            return HealthStatus.UNHEALTHY

        if degraded:
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY

    # --------------------------------------------------

    def _collect_system_metrics(self):
        metrics = {
            "platform":
                platform.platform(),
            "python":
                platform.python_version(),
            "time_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(
                    timespec="seconds"
                ),
        }

        try:
            import psutil

            vm = psutil.virtual_memory()

            metrics["cpu_pct"] = (
                psutil.cpu_percent(
                    interval=0.1
                )
            )

            metrics["mem_used_pct"] = (
                vm.percent
            )

            metrics[
                "mem_available_mb"
            ] = round(
                vm.available /
                (1024 * 1024),
                1,
            )

        except Exception as e:
            log.warning(f"Suppressed exception at line 293: {e}")
            pass

        return metrics

    # --------------------------------------------------

    def latest(self):
        with self._lock:
            return self._last_snapshot

    def history(self, limit=20):
        with self._lock:
            return list(
                self._history[-limit:]
            )

    def status(self):
        snap = self.latest()

        if snap is None:
            snap = self.run_once()

        return snap.to_dict()


_MONITOR = None


def get_health_monitor():
    global _MONITOR

    if _MONITOR is None:
        _MONITOR = HealthMonitor()

    return _MONITOR


# =====================================================
# MT5 HEALTH CHECK
# =====================================================

def register_mt5_health(
    monitor,
    mt5_connection,
):
    def mt5_check():
        try:
            alive = (
                mt5_connection.is_alive()
            )

            return {
                "ok": alive,
                "connected": alive,
                "detail":
                    "MT5 connected"
                    if alive
                    else "MT5 disconnected",
            }

        except Exception as e:
            return {
                "ok": False,
                "detail": str(e),
            }

    monitor.register_check(
        "mt5",
        mt5_check,
        critical=True,
    )