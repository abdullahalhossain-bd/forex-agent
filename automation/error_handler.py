# automation/error_handler.py  —  Day 51 | Error Logging System ⭐⭐⭐⭐⭐
# ============================================================
# Doc Section 3 — logs/system.log।
#
# এটা computer_use/stability_manager.py-এর logs/stability.log-এর
# replacement না — সেটা শুধু browser/screen-automation-specific issue
# (popup, session, chart-load, screenshot quality) record করে।
#
# logs/system.log বরং পুরো system-এর সব ধরনের error ধরে: API timeout,
# MT5 disconnect, vision failure, browser issue, order rejection —
# doc-এর exact format অনুসরণ করে (timestamp, error, action, result)।
#
# AutonomousRunner প্রতিটা pipeline stage-এর around try/except-এ এই
# handler call করবে, RuntimeMetrics.record_error()-ও সাথে সাথে call
# হয়ে যাবে যাতে রিপোর্টে error count সঠিক থাকে।
# ============================================================

import json
import os
import time
from datetime import datetime, timezone

from utils.logger import get_logger

log = get_logger("error_handler")

SYSTEM_LOG_PATH = "logs/system.log"

# Error category অনুযায়ী default retry policy — doc Section 7
# (Automatic Recovery System)-এর "Retry 3 times" pattern generalize করা
DEFAULT_RETRY_POLICY = {
    "API_TIMEOUT":      {"max_retries": 3, "backoff_sec": 5},
    "MT5_DISCONNECT":   {"max_retries": 3, "backoff_sec": 10},
    "VISION_FAILURE":   {"max_retries": 2, "backoff_sec": 3},
    "BROWSER_ISSUE":    {"max_retries": 2, "backoff_sec": 5},
    "ORDER_REJECTION":  {"max_retries": 1, "backoff_sec": 2},
    "DATABASE_ERROR":   {"max_retries": 2, "backoff_sec": 3},
    "DEFAULT":          {"max_retries": 1, "backoff_sec": 3},
}


class ErrorHandler:
    """
    Usage:
        handler = ErrorHandler(metrics=runtime_metrics)

        # সরাসরি log করা:
        handler.log_error("VISION_API_TIMEOUT", action="Retry", result="Success")

        # retry wrapper দিয়ে — fail করলে policy অনুযায়ী retry করবে:
        result = handler.with_retry(
            fn=lambda: vision_client.analyze(image),
            error_category="VISION_FAILURE",
        )
    """

    def __init__(self, metrics=None, log_path: str = SYSTEM_LOG_PATH):
        self.metrics = metrics   # RuntimeMetrics instance — record_error() সাথে call হবে
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self._error_counts: dict = {}   # category -> count, repeated-failure detection-এর জন্য

    # ═══════════════════════════════════════════════════════
    # DIRECT LOGGING  (doc-এর exact format)
    # ═══════════════════════════════════════════════════════

    def log_error(self, error: str, action: str = None, result: str = None,
                  severity: str = "ERROR") -> dict:
        """
        doc format:
            [10:45]
            Vision API timeout
            Action: Retry
            Result: Success
        """
        ts = datetime.now(timezone.utc)
        entry = {
            "timestamp": ts.isoformat(timespec="seconds"),
            "error": error, "action": action, "result": result,
            "severity": severity,
        }

        self._error_counts[error] = self._error_counts.get(error, 0) + 1

        line = (
            f"[{ts.strftime('%H:%M')}] {error}"
            f"{' | Action: ' + action if action else ''}"
            f"{' | Result: ' + result if result else ''}"
        )
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:
            log.warning(f"[ErrorHandler] Could not write system.log: {e}")

        log_fn = log.error if severity == "ERROR" else log.warning
        log_fn(f"[ErrorHandler] {line}")

        if self.metrics:
            self.metrics.record_error(error, action=action, result=result)

        return entry

    def log_crash(self, component: str, exception: Exception) -> dict:
        """Unrecoverable exception — RuntimeMetrics-এর crash counter বাড়াবে (system_status
        নির্ধারণে এটা সবচেয়ে গুরুত্বপূর্ণ — একটাও crash থাকলে report FAIL হবে)।"""
        return self.log_error(
            error=f"CRASH",
            action=f"{component}: {type(exception).__name__}: {exception}",
            result="unrecovered",
            severity="ERROR",
        )

    # ═══════════════════════════════════════════════════════
    # RETRY WRAPPER  (doc Section 7 — Automatic Recovery pattern)
    # ═══════════════════════════════════════════════════════

    def with_retry(self, fn, error_category: str = "DEFAULT", on_exhausted=None):
        """
        doc flow:
            Connection lost → Retry 3 times → Reconnect → Continue
            (exhausted হলে on_exhausted callback — যেমন "safe mode/no trade")

        fn: callable, exception তুললে retry করা হবে। success হলে fn()-এর
            return value সরাসরি ফেরত আসে।
        """
        policy = DEFAULT_RETRY_POLICY.get(error_category, DEFAULT_RETRY_POLICY["DEFAULT"])
        max_retries = policy["max_retries"]
        backoff = policy["backoff_sec"]

        last_exception = None
        for attempt in range(1, max_retries + 2):
            try:
                result = fn()
                if attempt > 1:
                    self.log_error(error_category, action=f"Retry (attempt {attempt})", result="Success")
                return result
            except Exception as e:
                last_exception = e
                if attempt <= max_retries:
                    self.log_error(
                        error_category, action=f"Retry {attempt}/{max_retries} in {backoff}s",
                        result=f"failed: {e}", severity="WARNING",
                    )
                    time.sleep(backoff)

        # সব retry শেষ
        self.log_error(error_category, action="Retries exhausted", result=str(last_exception))
        if on_exhausted:
            return on_exhausted(last_exception)
        raise last_exception

    # ═══════════════════════════════════════════════════════
    # ANALYSIS / SUMMARY
    # ═══════════════════════════════════════════════════════

    def get_error_summary(self) -> dict:
        return dict(self._error_counts)

    def get_recent_errors(self, limit: int = 20) -> list:
        if not os.path.exists(self.log_path):
            return []
        try:
            with open(self.log_path, encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            return lines[-limit:]
        except Exception:
            return []

    def print_summary(self) -> None:
        bar = "═" * 48
        print(f"\n{bar}")
        print("  📋  ERROR LOG SUMMARY  (Day 51)")
        print(bar)
        for err_type, count in self._error_counts.items():
            print(f"  {err_type:<28} ×{count}")
        print(bar + "\n")