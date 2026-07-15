"""
system/network_monitor.py — Day 97 Network & Latency Monitor
=============================================================
Tracks internet latency, MT5 broker ping, and execution delay.

When latency is high (>500ms), scalping is disabled and only swing
trades are allowed — because on a slow connection, M1/M5 entries
suffer unacceptable slippage.

Checks (every 30s):
  1. Internet ping (to Google DNS 8.8.8.8)
  2. MT5 broker server ping (if connected)
  3. Last execution latency (from ExecutionQualityMonitor)
  4. Overall network status classification

Status levels:
  GOOD    — ping < 100ms, no execution issues → all strategies allowed
  OK      — ping 100-300ms → scalping allowed with caution
  SLOW    — ping 300-500ms → scalping disabled, swing only
  BAD     — ping > 500ms → new trades blocked entirely

Usage:
    from system.network_monitor import get_network_monitor
    nm = get_network_monitor()
    nm.start()                    # background thread
    status = nm.get_status()      # {"ping": 38, "status": "GOOD", ...}
    if not nm.scaling_allowed():  # True/False
        # skip scalping strategy
"""
from __future__ import annotations

import os
import platform
import subprocess
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, Optional

from utils.logger import get_logger

log = get_logger("network_monitor")


class NetworkMonitor:
    """Background network latency tracker."""

    CHECK_INTERVAL_SEC = 30       # check every 30s
    PING_TARGETS = ["8.8.8.8"]    # Google DNS (reliable global target)
    PING_TIMEOUT_SEC = 5
    HISTORY_SIZE = 60             # keep last 60 measurements (30 min)

    # Latency thresholds (ms)
    LATENCY_GOOD = 100
    LATENCY_OK = 300
    LATENCY_SLOW = 500

    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.RLock()

        self._last_ping_ms: Optional[float] = None
        self._last_mt5_ping_ms: Optional[float] = None
        self._last_execution_latency_ms: Optional[int] = None
        self._broker_server: str = ""
        self._status: str = "UNKNOWN"
        self._history: Deque[Dict] = deque(maxlen=self.HISTORY_SIZE)
        self._consecutive_failures = 0

    # ─────────────────────────────────────────────────────────
    # LIFECYCLE
    # ─────────────────────────────────────────────────────────

    def start(self):
        """Start background monitoring thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="network_monitor")
        self._thread.start()
        log.info("[NetMonitor] started — checking every 30s")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _run_loop(self):
        while self._running:
            try:
                self.check_now()
            except Exception as e:
                log.error(f"[NetMonitor] check raised: {e}")
            for _ in range(self.CHECK_INTERVAL_SEC):
                if not self._running:
                    break
                time.sleep(1)

    # ─────────────────────────────────────────────────────────
    # CHECK
    # ─────────────────────────────────────────────────────────

    def check_now(self) -> Dict[str, Any]:
        """Run a single latency check.

        Returns: dict with ping, mt5_ping, status, etc.
        """
        # 1. Internet ping
        ping_ms = self._ping_host(self.PING_TARGETS[0])

        # 2. MT5 broker ping (if available)
        mt5_ping_ms = self._ping_mt5()

        # 3. Get last execution latency from ExecutionQualityMonitor
        exec_latency = self._get_execution_latency()

        # 4. Classify status
        worst_latency = max(
            ping_ms or 0,
            mt5_ping_ms or 0,
            exec_latency or 0,
        )

        if worst_latency == 0:
            status = "UNKNOWN"
        elif worst_latency <= self.LATENCY_GOOD:
            status = "GOOD"
            self._consecutive_failures = 0
        elif worst_latency <= self.LATENCY_OK:
            status = "OK"
            self._consecutive_failures = 0
        elif worst_latency <= self.LATENCY_SLOW:
            status = "SLOW"
            log.warning(f"[NetMonitor] SLOW — ping={ping_ms}ms mt5={mt5_ping_ms}ms exec={exec_latency}ms")
        else:
            status = "BAD"
            self._consecutive_failures += 1
            log.error(f"[NetMonitor] BAD — ping={ping_ms}ms mt5={mt5_ping_ms}ms exec={exec_latency}ms")

        result = {
            "timestamp":           datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "ping_ms":             ping_ms,
            "mt5_ping_ms":         mt5_ping_ms,
            "execution_latency_ms": exec_latency,
            "broker_server":       self._broker_server,
            "status":              status,
            "worst_latency_ms":    worst_latency,
            "scalping_allowed":    status in ("GOOD", "OK"),
            "trading_allowed":     status != "BAD",
            "consecutive_failures": self._consecutive_failures,
        }

        with self._lock:
            self._last_ping_ms = ping_ms
            self._last_mt5_ping_ms = mt5_ping_ms
            self._last_execution_latency_ms = exec_latency
            self._status = status
            self._history.append(result)

        return result

    # ─────────────────────────────────────────────────────────
    # PING METHODS
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _ping_host(host: str) -> Optional[float]:
        """Ping a host and return latency in ms. None on failure."""
        try:
            system = platform.system().lower()
            if system == "windows":
                cmd = ["ping", "-n", "1", "-w", str(NetworkMonitor.PING_TIMEOUT_SEC * 1000), host]
            else:
                cmd = ["ping", "-c", "1", "-W", str(NetworkMonitor.PING_TIMEOUT_SEC), host]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode != 0:
                return None

            # Parse latency from output
            output = result.stdout
            # Linux: "time=38.2 ms"
            # Windows: "time=38ms" or "time<1ms"
            import re
            match = re.search(r"time[=<](\d+\.?\d*)\s*ms", output, re.IGNORECASE)
            if match:
                return float(match.group(1))
            return None
        except Exception:
            return None

    def _ping_mt5(self) -> Optional[float]:
        """Ping MT5 broker server (if connected)."""
        try:
            from broker.mt5_connection import MT5_AVAILABLE
            if not MT5_AVAILABLE:
                return None

            import MetaTrader5 as mt5
            if not mt5.initialize():
                return None

            # Get terminal info (includes ping/connection info)
            info = mt5.terminal_info()
            mt5.shutdown()

            if info is None:
                return None

            # Store broker server name
            self._broker_server = getattr(info, "name", "unknown")

            # MT5 terminal_info doesn't directly expose ping, but
            # ping_ms is available in some versions via 'ping' attribute
            ping = getattr(info, "ping", None)
            if ping is not None and ping > 0:
                return float(ping)

            # Fallback: measure round-trip time for a simple API call
            start = time.time()
            mt5.initialize()
            mt5.symbols_total()
            mt5.shutdown()
            elapsed_ms = (time.time() - start) * 1000
            # Subtract overhead (~20ms) for a rough estimate
            return max(0, elapsed_ms - 20)
        except Exception:
            return None

    @staticmethod
    def _get_execution_latency() -> Optional[int]:
        """Get last execution latency from ExecutionQualityMonitor."""
        try:
            from monitoring.execution_quality import get_execution_quality_monitor
            eqm = get_execution_quality_monitor()
            report = eqm.get_quality_report()
            latency = report.get("avg_latency_ms", 0)
            return int(latency) if latency > 0 else None
        except Exception:
            return None

    # ─────────────────────────────────────────────────────────
    # PUBLIC QUERIES
    # ─────────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Return current network status."""
        with self._lock:
            return {
                "ping_ms":             self._last_ping_ms,
                "mt5_ping_ms":         self._last_mt5_ping_ms,
                "execution_latency_ms": self._last_execution_latency_ms,
                "broker_server":       self._broker_server,
                "status":              self._status,
                "scalping_allowed":    self._status in ("GOOD", "OK"),
                "trading_allowed":     self._status != "BAD",
                "consecutive_failures": self._consecutive_failures,
                "history_count":       len(self._history),
            }

    def scalping_allowed(self) -> bool:
        """True if latency is low enough for scalping (M1/M5)."""
        with self._lock:
            return self._status in ("GOOD", "OK")

    def trading_allowed(self) -> bool:
        """True if any trading is allowed (swing or scalping)."""
        with self._lock:
            return self._status != "BAD"

    def get_ai_context(self) -> Dict[str, Any]:
        """Compact context for MasterAnalyst / decision engine."""
        with self._lock:
            return {
                "net_status":          self._status,
                "net_ping_ms":         self._last_ping_ms,
                "net_mt5_ping_ms":     self._last_mt5_ping_ms,
                "net_scalping_allowed": self._status in ("GOOD", "OK"),
                "net_trading_allowed": self._status != "BAD",
            }

    def print_summary(self, status: Dict[str, Any] = None):
        """Print network status summary."""
        if status is None:
            status = self.get_status()
        bar = "═" * 50
        log.info(bar)
        log.info("  🌐  NETWORK MONITOR  (Day 97)")
        log.info(bar)
        log.info(f"  Status           : {status.get('status','?')}")
        log.info(f"  Internet ping    : {status.get('ping_ms','?')} ms")
        log.info(f"  MT5 ping         : {status.get('mt5_ping_ms','?')} ms")
        log.info(f"  Exec latency     : {status.get('execution_latency_ms','?')} ms")
        log.info(f"  Broker           : {status.get('broker_server','?')}")
        log.info(f"  Scalping allowed : {'✅' if status.get('scalping_allowed') else '❌'}")
        log.info(f"  Trading allowed  : {'✅' if status.get('trading_allowed') else '❌'}")
        log.info(bar)


# ── Singleton ────────────────────────────────────────────────────

_MONITOR: Optional[NetworkMonitor] = None
_LOCK = threading.Lock()

def get_network_monitor() -> NetworkMonitor:
    global _MONITOR
    if _MONITOR is None:
        with _LOCK:
            if _MONITOR is None:
                _MONITOR = NetworkMonitor()
    return _MONITOR
