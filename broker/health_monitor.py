# broker/health_monitor.py  —  Day 31 Part 3 & 4 | Connection Health + Auto Reconnect
# ============================================================
# Autonomous trading system-এ connection loss মানেই danger:
#   - Open position থাকতে থাকতে disconnect হলে SL/TP monitor বন্ধ হয়ে যায়
#   - তাই health check + auto-reconnect একটা critical safety layer
# ============================================================

import time
from datetime import datetime
from utils.logger import get_logger
from broker.mt5_connection import MT5Connection, MT5_AVAILABLE

log = get_logger("health_monitor")

if MT5_AVAILABLE:
    import MetaTrader5 as mt5


class HealthMonitor:
    """
    MT5 connection-এর health track করে এবং disconnect হলে
    নিজে নিজে reconnect attempt করে।

    Flow:
        Connection lost
            ↓
        Pause trading (callback দিয়ে external system-কে জানায়)
            ↓
        Reconnect attempt (max N বার)
            ↓
        Success → Resume trading
        Fail    → Alert + Shutdown (external system নিজে decide করবে কী করবে)

    Usage:
        monitor = HealthMonitor(conn, on_disconnect=pause_fn, on_reconnect=resume_fn)
        monitor.run_loop()   # blocking — সাধারণত আলাদা thread/process-এ চালাও
        # বা একবারের জন্য:
        monitor.check_once()
    """

    CHECK_INTERVAL_SEC = 30
    MAX_RECONNECT_ATTEMPTS = 5
    RECONNECT_BACKOFF_SEC = 10   # প্রতি attempt-এ বাড়বে (10, 20, 30...)

    def __init__(
        self,
        connection: MT5Connection,
        on_disconnect=None,
        on_reconnect=None,
        on_fatal=None,
    ):
        self.connection = connection
        self.on_disconnect = on_disconnect   # callback: trading pause করার জন্য
        self.on_reconnect = on_reconnect      # callback: trading resume করার জন্য
        self.on_fatal = on_fatal              # callback: সব reconnect fail হলে alert

        self._healthy = True
        self._last_check: datetime | None = None
        self._consecutive_failures = 0

    # ─────────────────────────────────────────────
    # SINGLE CHECK
    # ─────────────────────────────────────────────

    def check_once(self) -> bool:
        """
        একবার connection health check করো।
        Returns True if healthy, False otherwise. Disconnect ধরা পড়লে
        নিজে নিজে reconnect-এর চেষ্টা করবে।
        """
        self._last_check = datetime.utcnow()
        healthy = self._is_connection_ok()

        if healthy:
            if not self._healthy:
                log.info("[HealthMonitor] 🟢 Connection recovered")
            self._healthy = True
            self._consecutive_failures = 0
            return True

        # Connection broken
        self._healthy = False
        log.warning("[HealthMonitor] 🔴 Connection lost — initiating recovery")
        if self.on_disconnect:
            self.on_disconnect("MT5 connection lost — trading paused")

        recovered = self._attempt_reconnect()
        if recovered:
            self._healthy = True
            self._consecutive_failures = 0
            if self.on_reconnect:
                self.on_reconnect("MT5 connection restored — trading resumed")
            return True

        # সব reconnect attempt fail
        self._consecutive_failures += 1
        log.error(
            f"[HealthMonitor] ⛔ Reconnect failed after "
            f"{self.MAX_RECONNECT_ATTEMPTS} attempts"
        )
        if self.on_fatal:
            self.on_fatal(
                "MT5 reconnect failed — manual intervention দরকার। Shutdown করা হলো।"
            )
        return False

    # ─────────────────────────────────────────────
    # CONTINUOUS LOOP  (Day 31 pseudo-code-এর real version)
    # ─────────────────────────────────────────────

    def run_loop(self, stop_flag=None):
        """
        Blocking loop — প্রতি CHECK_INTERVAL_SEC সেকেন্ডে health check করে।
        stop_flag: callable যা True হলে loop বন্ধ হয়ে যাবে (graceful shutdown-এর জন্য)।
        সাধারণত এটা একটা আলাদা background thread-এ চালানো হয়, main trading
        loop block করে না।
        """
        log.info(
            f"[HealthMonitor] Starting health loop "
            f"(every {self.CHECK_INTERVAL_SEC}s)"
        )
        while True:
            if stop_flag and stop_flag():
                log.info("[HealthMonitor] Stop flag set — exiting loop")
                break

            ok = self.check_once()
            if not ok and self.on_fatal:
                # Fatal হলে loop বন্ধ — external system shutdown handle করবে
                break

            time.sleep(self.CHECK_INTERVAL_SEC)

    # ─────────────────────────────────────────────
    # INTERNAL
    # ─────────────────────────────────────────────

    def _is_connection_ok(self) -> bool:
        if not MT5_AVAILABLE:
            return False

        term_info = mt5.terminal_info()
        if term_info is None:
            return False

        if not term_info.connected:
            return False

        # Account access ও double-check করো (terminal connected হলেও
        # broker session expire হতে পারে)
        account = mt5.account_info()
        if account is None:
            return False

        return True

    def _attempt_reconnect(self) -> bool:
        for attempt in range(1, self.MAX_RECONNECT_ATTEMPTS + 1):
            wait = self.RECONNECT_BACKOFF_SEC * attempt
            log.info(
                f"[HealthMonitor] Reconnect attempt {attempt}/"
                f"{self.MAX_RECONNECT_ATTEMPTS} (waiting {wait}s first)"
            )
            time.sleep(wait)

            self.connection.disconnect()
            if self.connection.connect():
                log.info(f"[HealthMonitor] ✅ Reconnected on attempt {attempt}")
                return True

        return False

    def get_status(self) -> dict:
        return {
            "healthy": self._healthy,
            "last_check": self._last_check.isoformat() if self._last_check else None,
            "consecutive_failures": self._consecutive_failures,
        }

    def print_status(self) -> None:
        s = self.get_status()
        icon = "🟢 Healthy" if s["healthy"] else "🔴 Lost"
        log.info(f"MT5 Connection: {icon}")
        if s["last_check"]:
            log.info(f"  Last check: {s['last_check']}")
        if s["consecutive_failures"]:
            log.info(f"  Consecutive failures: {s['consecutive_failures']}")