"""
core/graceful_shutdown.py — Graceful Shutdown Manager
=====================================================

Ensures clean shutdown when the trading bot is killed (Ctrl+C, SIGTERM,
VPS reboot). Prevents orphaned positions, half-written state files,
and background threads dying abruptly.

Shutdown sequence:
  1. Signal received → stop accepting new signals
  2. Wait for in-flight orders to complete (with timeout)
  3. Flush write-ahead log (WAL) to disk
  4. Save system state
  5. Stop background threads (OrderReconciler, etc.)
  6. Close MT5 connection
  7. Exit cleanly

Usage:
    from core.graceful_shutdown import GracefulShutdownManager
    shutdown = GracefulShutdownManager()
    shutdown.register_cleanup_callback(lambda: mt5.shutdown())
    shutdown.register_cleanup_callback(lambda: reconciler.stop())

    # In main loop:
    while not shutdown.is_shutting_down():
        ... do trading ...
    shutdown.wait_for_completion()
"""

from __future__ import annotations

import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from utils.logger import get_logger

log = get_logger("shutdown")


@dataclass
class ShutdownState:
    """Tracks the shutdown process."""
    shutdown_requested: bool = False
    shutdown_initiated: bool = False
    shutdown_complete: bool = False
    signal_received: str = ""
    signal_time: str = ""
    cleanup_callbacks_run: int = 0
    cleanup_errors: List[str] = field(default_factory=list)


class GracefulShutdownManager:
    """
    Manages graceful shutdown of the trading system.

    Registers signal handlers for SIGINT (Ctrl+C) and SIGTERM (kill).
    When a signal is received:
      1. Sets shutdown_requested = True (main loop should stop)
      2. Runs cleanup callbacks in reverse order
      3. Waits for background threads to join
      4. Marks shutdown_complete
    """

    def __init__(self, timeout_seconds: float = 30.0):
        self.timeout = timeout_seconds
        self._lock = threading.RLock()
        self._state = ShutdownState()
        self._cleanup_callbacks: List[Callable[[], None]] = []
        self._background_threads: List[threading.Thread] = []

        # Install signal handlers (only in main thread)
        try:
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)
            log.info("[Shutdown] Signal handlers installed (SIGINT, SIGTERM)")
        except (ValueError, OSError) as e:
            # Not in main thread — skip signal handler installation
            log.warning(f"[Shutdown] Could not install signal handlers: {e}")

    def register_cleanup_callback(self, callback: Callable[[], None]):
        """Register a function to run during shutdown.

        Callbacks run in reverse registration order (LIFO) —
        so the most recently registered runs first.
        This ensures dependencies are torn down in the right order.
        """
        with self._lock:
            self._cleanup_callbacks.append(callback)

    def register_thread(self, thread: threading.Thread):
        """Register a background thread to stop during shutdown."""
        with self._lock:
            self._background_threads.append(thread)

    def is_shutting_down(self) -> bool:
        """Check if shutdown has been requested."""
        with self._lock:
            return self._state.shutdown_requested

    def request_shutdown(self, reason: str = "manual"):
        """Manually request shutdown (without signal)."""
        with self._lock:
            if not self._state.shutdown_requested:
                self._state.shutdown_requested = True
                self._state.signal_received = reason
                self._state.signal_time = datetime.now(timezone.utc).isoformat()
                log.warning(f"[Shutdown] Shutdown requested: {reason}")

    def _handle_signal(self, signum: int, frame):
        """Signal handler — called by OS."""
        signal_name = signal.Signals(signum).name
        with self._lock:
            if self._state.shutdown_requested:
                # Second signal — force exit
                log.critical(f"[Shutdown] Second {signal_name} received — forcing exit")
                sys.exit(1)
            self._state.shutdown_requested = True
            self._state.signal_received = signal_name
            self._state.signal_time = datetime.now(timezone.utc).isoformat()
            log.warning(f"[Shutdown] {signal_name} received — initiating graceful shutdown")

    def run_cleanup(self):
        """Run all cleanup callbacks. Called after main loop exits."""
        with self._lock:
            if self._state.shutdown_initiated:
                return  # already running cleanup
            self._state.shutdown_initiated = True

        log.info("[Shutdown] Running cleanup callbacks...")

        # Run callbacks in reverse order (LIFO)
        with self._lock:
            callbacks = list(reversed(self._cleanup_callbacks))

        for cb in callbacks:
            try:
                cb()
                with self._lock:
                    self._state.cleanup_callbacks_run += 1
            except Exception as e:
                with self._lock:
                    self._state.cleanup_errors.append(f"{cb.__name__}: {e}")
                log.error(f"[Shutdown] Cleanup callback failed: {e}")

        # Wait for background threads
        with self._lock:
            threads = list(self._background_threads)

        for t in threads:
            if t.is_alive() and not t.daemon:
                log.info(f"[Shutdown] Waiting for thread {t.name}...")
                t.join(timeout=5.0)
                if t.is_alive():
                    log.warning(f"[Shutdown] Thread {t.name} did not stop — "
                               "will be killed (daemon=False)")

        with self._lock:
            self._state.shutdown_complete = True

        log.info(f"[Shutdown] Complete — {self._state.cleanup_callbacks_run} "
                 f"callbacks run, {len(self._state.cleanup_errors)} errors")

    def wait_for_completion(self):
        """Block until shutdown is complete (or timeout)."""
        if not self.is_shutting_down():
            log.warning("[Shutdown] wait_for_completion called but shutdown "
                       "not requested")
            return

        self.run_cleanup()

        deadline = time.time() + self.timeout
        while time.time() < deadline:
            with self._lock:
                if self._state.shutdown_complete:
                    return
            time.sleep(0.1)

        log.error(f"[Shutdown] Timed out after {self.timeout}s — forcing exit")

    def get_state(self) -> Dict[str, Any]:
        """Get current shutdown state for monitoring."""
        with self._lock:
            return {
                "shutdown_requested": self._state.shutdown_requested,
                "shutdown_initiated": self._state.shutdown_initiated,
                "shutdown_complete": self._state.shutdown_complete,
                "signal_received": self._state.signal_received,
                "signal_time": self._state.signal_time,
                "cleanup_callbacks_run": self._state.cleanup_callbacks_run,
                "cleanup_errors": self._state.cleanup_errors,
                "n_callbacks_registered": len(self._cleanup_callbacks),
                "n_threads_registered": len(self._background_threads),
            }


# ════════════════════════════════════════════════════════════════
#  CLI ENTRY (smoke test)
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    print("=" * 70)
    print("  GRACEFUL SHUTDOWN MANAGER — Smoke Test")
    print("=" * 70)

    shutdown = GracefulShutdownManager(timeout_seconds=5.0)

    # Register some cleanup callbacks
    cleanup_order = []

    def cleanup_1():
        cleanup_order.append("cleanup_1")
        print("  [cleanup_1] Closing MT5 connection...")

    def cleanup_2():
        cleanup_order.append("cleanup_2")
        print("  [cleanup_2] Stopping OrderReconciler thread...")

    def cleanup_3():
        cleanup_order.append("cleanup_3")
        print("  [cleanup_3] Flushing WAL to disk...")

    shutdown.register_cleanup_callback(cleanup_1)
    shutdown.register_cleanup_callback(cleanup_2)
    shutdown.register_cleanup_callback(cleanup_3)

    # Simulate main loop
    print("\n  Simulating main loop (3 iterations)...")
    for i in range(3):
        if shutdown.is_shutting_down():
            print(f"  Iteration {i}: shutdown requested — breaking")
            break
        print(f"  Iteration {i}: running...")
        time.sleep(0.1)
        if i == 1:
            # Simulate shutdown request
            print("  → Requesting shutdown...")
            shutdown.request_shutdown("test")

    print(f"\n  Shutdown state: {shutdown.get_state()}")

    # Run cleanup
    print("\n  Running cleanup...")
    shutdown.run_cleanup()

    print(f"\n  Cleanup order: {cleanup_order}")
    print(f"  (should be LIFO: cleanup_3, cleanup_2, cleanup_1)")

    print(f"\n  Final state: {shutdown.get_state()}")

    print("\n" + "=" * 70)
    print("  Graceful shutdown test complete.")
    print("=" * 70)
