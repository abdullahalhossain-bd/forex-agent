# automation/system_health.py  —  Day 51 | System Health Monitor ⭐⭐⭐⭐⭐
# ============================================================
# Doc Section 4 — AI নিজেকে monitor করবে:
#     MT5 connection ✅ | Database ✅ | Vision API ✅ | Internet ✅ | Memory usage ✅
#
# নাম ইচ্ছাকৃতভাবে "system_health.py" — তোমার broker/health_monitor.py
# (Day 31) আগেই MT5-connection-specific health check করে। এই module
# সেটাকে replace করে না, বরং wrap করে — broker/health_monitor.py এখানে
# একটা sub-check, পাশে আরো চারটা subsystem check যুক্ত হয়েছে।
#
# AutonomousRunner প্রতি N cycle-এ (বা সময়ে) এটা call করবে এবং doc-এর
# output format-এ print করবে:
#     SYSTEM HEALTH
#     Broker: OK | AI Brain: OK | Vision: OK | Execution: OK
# ============================================================

import os
import socket
import time

from utils.logger import get_logger

log = get_logger("system_health")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except Exception:
    PSUTIL_AVAILABLE = False
    log.warning("[SystemHealth] psutil unavailable — memory check will be skipped")

MEMORY_WARN_THRESHOLD_PCT = 85.0


class SystemHealth:
    """
    Usage:
        health = SystemHealth(
            mt5_health_monitor=broker_health_monitor,   # Day 31 HealthMonitor, optional
            db=trader_db,                                # database/db.py TraderDB, optional
            vision_analyzer=vision_analyzer,             # Day 47 VisionAnalyzer, optional
        )
        snapshot = health.check_all()
        health.print_status(snapshot)
    """

    def __init__(self, mt5_health_monitor=None, db=None, vision_analyzer=None,
                 internet_check_host: str = "8.8.8.8", internet_check_port: int = 53):
        self.mt5_health_monitor = mt5_health_monitor
        self.db = db
        self.vision_analyzer = vision_analyzer
        self.internet_check_host = internet_check_host
        self.internet_check_port = internet_check_port

    # ═══════════════════════════════════════════════════════
    # MAIN — সব subsystem একসাথে চেক
    # ═══════════════════════════════════════════════════════

    def check_all(self) -> dict:
        checks = {
            "broker":   self._check_broker(),
            "database": self._check_database(),
            "vision":   self._check_vision(),
            "internet": self._check_internet(),
            "memory":   self._check_memory(),
        }
        overall_ok = all(c["ok"] for c in checks.values() if c["ok"] is not None)
        return {"checks": checks, "overall_ok": overall_ok}

    # ═══════════════════════════════════════════════════════
    # 1. BROKER / MT5  (Day 31 broker/health_monitor.py reuse)
    # ═══════════════════════════════════════════════════════

    def _check_broker(self) -> dict:
        if not self.mt5_health_monitor:
            return {"ok": None, "detail": "No HealthMonitor wired — skipped"}
        try:
            status = self.mt5_health_monitor.get_status()
            return {"ok": status.get("healthy", False), "detail": status}
        except Exception as e:
            return {"ok": False, "detail": f"Health check error: {e}"}

    # ═══════════════════════════════════════════════════════
    # 2. DATABASE
    # ═══════════════════════════════════════════════════════

    def _check_database(self) -> dict:
        if not self.db:
            return {"ok": None, "detail": "No DB instance wired — skipped"}
        try:
            # Lightest possible query — তোমার TraderDB-তে যে method available
            # থাকুক, একটা cheap read করে দেখাই যথেষ্ট প্রমাণ যে connection ঠিক আছে
            if hasattr(self.db, "get_trade_history"):
                self.db.get_trade_history(limit=1)
            return {"ok": True, "detail": "Query succeeded"}
        except Exception as e:
            return {"ok": False, "detail": f"DB query failed: {e}"}

    # ═══════════════════════════════════════════════════════
    # 3. VISION API
    # ═══════════════════════════════════════════════════════

    def _check_vision(self) -> dict:
        try:
            from computer_use.vision_analyzer import LLM_AVAILABLE
            if not LLM_AVAILABLE:
                return {"ok": False, "detail": "Anthropic client not initialized (API key missing?)"}
            return {"ok": True, "detail": "Vision client available"}
        except Exception as e:
            return {"ok": False, "detail": f"Vision check error: {e}"}

    # ═══════════════════════════════════════════════════════
    # 4. INTERNET CONNECTIVITY
    # ═══════════════════════════════════════════════════════

    def _check_internet(self, timeout: float = 3.0) -> dict:
        """CRITICAL FIX: use context manager to ensure socket is closed.
        Previous code created socket without closing — file descriptor leak.
        Over days of running, this exhausts file descriptors and crashes.
        """
        try:
            start = time.time()
            # Use with statement to ensure socket is closed
            with socket.create_connection(
                (self.internet_check_host, self.internet_check_port),
                timeout=timeout
            ):
                latency_ms = round((time.time() - start) * 1000, 1)
                return {"ok": True, "detail": f"Connected ({latency_ms}ms)"}
        except Exception as e:
            return {"ok": False, "detail": f"No internet: {e}"}

    # ═══════════════════════════════════════════════════════
    # 5. MEMORY USAGE
    # ═══════════════════════════════════════════════════════

    def _check_memory(self) -> dict:
        if not PSUTIL_AVAILABLE:
            return {"ok": None, "detail": "psutil not installed — skipped"}
        try:
            vm = psutil.virtual_memory()
            ok = vm.percent < MEMORY_WARN_THRESHOLD_PCT
            return {"ok": ok, "detail": f"{vm.percent}% used"}
        except Exception as e:
            return {"ok": False, "detail": f"Memory check error: {e}"}

    # ═══════════════════════════════════════════════════════
    # PRINT  (doc-এর exact output format)
    # ═══════════════════════════════════════════════════════

    def print_status(self, snapshot: dict = None) -> None:
        snapshot = snapshot or self.check_all()
        checks = snapshot["checks"]

        def icon(ok):
            return "✅" if ok else ("➖" if ok is None else "❌")

        bar = "═" * 40
        print(f"\n{bar}")
        print("  🩺  SYSTEM HEALTH  (Day 51)")
        print(bar)
        print(f"  Broker (MT5)  : {icon(checks['broker']['ok'])}  {checks['broker']['detail']}")
        print(f"  Database      : {icon(checks['database']['ok'])}  {checks['database']['detail']}")
        print(f"  Vision API    : {icon(checks['vision']['ok'])}  {checks['vision']['detail']}")
        print(f"  Internet      : {icon(checks['internet']['ok'])}  {checks['internet']['detail']}")
        print(f"  Memory        : {icon(checks['memory']['ok'])}  {checks['memory']['detail']}")
        print(bar)
        print(f"  Overall       : {'✅ HEALTHY' if snapshot['overall_ok'] else '⚠️ DEGRADED'}")
        print(bar + "\n")

    def get_summary_line(self, snapshot: dict = None) -> str:
        """doc-এর সংক্ষিপ্ত format: 'Broker: OK | AI Brain: OK | Vision: OK | Execution: OK'"""
        snapshot = snapshot or self.check_all()
        c = snapshot["checks"]
        parts = []
        for label, key in [("Broker", "broker"), ("DB", "database"),
                            ("Vision", "vision"), ("Internet", "internet"),
                            ("Memory", "memory")]:
            ok = c[key]["ok"]
            parts.append(f"{label}: {'OK' if ok else ('N/A' if ok is None else 'FAIL')}")
        return " | ".join(parts)