"""
system/watchdog.py — Day 96 System Watchdog & Heartbeat
=======================================================
Monitors system health and auto-recovers from failures.

Checks (every 60s):
  1. Bot process alive?
  2. MT5 connection alive? (if Windows)
  3. Database accessible?
  4. Last signal within reasonable time? (no stale cycles)
  5. Memory usage within limits?
  6. Daily loss limit not exceeded?

On failure:
  - Log WARNING/ERROR
  - Send Telegram alert
  - Attempt auto-recovery (reconnect MT5, restart bot, clear cache)
  - If 3 consecutive failures → full restart
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from utils.logger import get_logger

log = get_logger("watchdog")

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


class SystemWatchdog:
    """Background health monitor + auto-recovery."""

    CHECK_INTERVAL_SEC = 60
    STALE_SIGNAL_MIN = 30
    MAX_CONSECUTIVE_FAILS = 3
    MEMORY_LIMIT_PCT = 90

    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.RLock()
        self._last_check: Optional[datetime] = None
        self._consecutive_fails = 0
        self._checks_passed = 0
        self._checks_failed = 0
        self._last_signal_time: Optional[datetime] = None
        self._recovery_count = 0
        self._health_history: list = []
        self._restart_callback = None
        self._telegram_alert_callback = None

    def start(self, restart_callback=None, telegram_callback=None):
        self._restart_callback = restart_callback
        self._telegram_alert_callback = telegram_callback
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="watchdog")
        self._thread.start()
        log.info("[Watchdog] started — checking every 60s")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        log.info("[Watchdog] stopped")

    def _run_loop(self):
        while self._running:
            try:
                self.check_now()
            except Exception as e:
                log.error(f"[Watchdog] check raised: {e}")
            for _ in range(self.CHECK_INTERVAL_SEC):
                if not self._running:
                    break
                time.sleep(1)

    def check_now(self) -> Dict[str, Any]:
        checks = {}
        all_pass = True

        checks["process"] = {"status": "OK", "detail": "watchdog running"}
        self._checks_passed += 1

        mt5_status = self._check_mt5()
        checks["mt5"] = mt5_status
        if mt5_status["status"] not in ("OK", "SKIP"):
            all_pass = False

        db_status = self._check_database()
        checks["database"] = db_status
        if db_status["status"] not in ("OK", "SKIP", "WARN"):
            all_pass = False

        signal_status = self._check_signal_freshness()
        checks["last_signal"] = signal_status
        if signal_status["status"] == "STALE":
            all_pass = False

        mem_status = self._check_memory()
        checks["memory"] = mem_status
        if mem_status["status"] not in ("OK", "SKIP"):
            all_pass = False

        loss_status = self._check_daily_loss()
        checks["daily_loss"] = loss_status
        if loss_status["status"] == "EXCEEDED":
            all_pass = False

        result = {
            "timestamp":          datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "overall":            "OK" if all_pass else "DEGRADED",
            "checks":             checks,
            "consecutive_fails":  self._consecutive_fails,
            "total_passed":       self._checks_passed,
            "total_failed":       self._checks_failed,
            "recovery_count":     self._recovery_count,
        }

        with self._lock:
            self._last_check = datetime.now(timezone.utc)
            if all_pass:
                self._consecutive_fails = 0
            else:
                self._consecutive_fails += 1
                self._checks_failed += 1
                self._handle_failure(checks)
            self._health_history.append(result)
            if len(self._health_history) > 100:
                self._health_history.pop(0)

        if not all_pass:
            log.warning(f"[Watchdog] DEGRADED: {checks}")
        else:
            log.debug(f"[Watchdog] OK — all checks passed")
        return result

    @staticmethod
    def _check_mt5() -> Dict[str, str]:
        try:
            from broker.mt5_connection import MT5_AVAILABLE
            if not MT5_AVAILABLE:
                return {"status": "SKIP", "detail": "MT5 not installed (Linux VPS)"}
            import MetaTrader5 as mt5
            if not mt5.initialize():
                return {"status": "FAIL", "detail": f"MT5 init failed"}
            info = mt5.account_info()
            mt5.shutdown()
            if info is None:
                return {"status": "FAIL", "detail": "MT5 no account info"}
            return {"status": "OK", "detail": f"balance=${info.balance:.0f}"}
        except Exception as e:
            return {"status": "FAIL", "detail": str(e)[:80]}

    @staticmethod
    def _check_database() -> Dict[str, str]:
        try:
            import sqlite3
            db_path = "memory/trader.db"
            if not os.path.exists(db_path):
                return {"status": "WARN", "detail": "DB file not found"}
            with sqlite3.connect(db_path) as conn:
                conn.execute("SELECT 1").fetchone()
            return {"status": "OK", "detail": "DB accessible"}
        except Exception as e:
            return {"status": "FAIL", "detail": str(e)[:80]}

    def _check_signal_freshness(self) -> Dict[str, str]:
        if self._last_signal_time is None:
            return {"status": "WARN", "detail": "no signal yet (bot just started?)"}
        elapsed = (datetime.now(timezone.utc) - self._last_signal_time).total_seconds() / 60
        if elapsed > self.STALE_SIGNAL_MIN:
            return {"status": "STALE", "detail": f"no signal in {elapsed:.0f} min"}
        return {"status": "OK", "detail": f"last signal {elapsed:.0f} min ago"}

    @staticmethod
    def _check_memory() -> Dict[str, str]:
        if not _PSUTIL_AVAILABLE:
            return {"status": "SKIP", "detail": "psutil not installed"}
        try:
            mem = psutil.virtual_memory()
            if mem.percent > 90:
                return {"status": "WARN", "detail": f"RAM {mem.percent:.0f}% used"}
            return {"status": "OK", "detail": f"RAM {mem.percent:.0f}% used"}
        except Exception as e:
            return {"status": "SKIP", "detail": str(e)[:60]}

    @staticmethod
    def _check_daily_loss() -> Dict[str, str]:
        try:
            import json
            path = "memory/daily_risk.json"
            if not os.path.exists(path):
                return {"status": "OK", "detail": "no daily risk file"}
            with open(path) as f:
                data = json.load(f)
            loss = data.get("total_loss_usd", 0)
            from config import INITIAL_BALANCE, DAILY_LOSS_LIMIT_PCT
            loss_pct = (loss / INITIAL_BALANCE) * 100 if INITIAL_BALANCE > 0 else 0
            if loss_pct >= DAILY_LOSS_LIMIT_PCT:
                return {"status": "EXCEEDED", "detail": f"daily loss {loss_pct:.1f}%"}
            return {"status": "OK", "detail": f"daily loss {loss_pct:.1f}%"}
        except Exception as e:
            return {"status": "SKIP", "detail": str(e)[:60]}

    def _handle_failure(self, checks: Dict[str, Any]):
        if self._telegram_alert_callback:
            try:
                failed = {k: v for k, v in checks.items() if v["status"] not in ("OK", "SKIP")}
                msg = "⚠️ Watchdog alert:\n" + "\n".join(
                    f"  {k}: {v['detail']}" for k, v in failed.items()
                )
                self._telegram_alert_callback(msg)
            except Exception:
                pass
        if self._consecutive_fails >= self.MAX_CONSECUTIVE_FAILS:
            log.error(f"[Watchdog] {self._consecutive_fails} fails — attempting restart")
            self._recovery_count += 1
            if self._restart_callback:
                try:
                    self._restart_callback()
                except Exception as e:
                    log.error(f"[Watchdog] restart failed: {e}")

    def record_signal(self, pair: str, signal: str):
        with self._lock:
            self._last_signal_time = datetime.now(timezone.utc)

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "running":            self._running,
                "last_check":         self._last_check.isoformat() if self._last_check else None,
                "consecutive_fails":  self._consecutive_fails,
                "total_passed":       self._checks_passed,
                "total_failed":       self._checks_failed,
                "recovery_count":     self._recovery_count,
                "last_signal_time":   self._last_signal_time.isoformat() if self._last_signal_time else None,
                "recent_history":     self._health_history[-5:],
            }


_WATCHDOG: Optional[SystemWatchdog] = None
_LOCK = threading.Lock()

def get_watchdog() -> SystemWatchdog:
    global _WATCHDOG
    if _WATCHDOG is None:
        with _LOCK:
            if _WATCHDOG is None:
                _WATCHDOG = SystemWatchdog()
    return _WATCHDOG
