"""
monitoring/metrics_exporter.py — Prometheus Metrics Exporter
=============================================================

Exposes trading system metrics on :9090/metrics for Prometheus scraping.

Metrics:
  forex_ai_balance           — current account balance
  forex_ai_equity            — current equity
  forex_ai_drawdown_pct      — current drawdown %
  forex_ai_daily_loss_pct    — today's loss %
  forex_ai_open_positions    — number of open positions
  forex_ai_win_rate          — rolling win rate
  forex_ai_trades_total      — total trades taken
  forex_ai_risk_halt         — 1 if trading halted, 0 if running
  forex_ai_latency_ms        — last cycle latency in ms
  forex_ai_memory_mb         — process memory usage in MB
  forex_ai_cpu_pct           — process CPU usage %

Usage:
    from monitoring.metrics_exporter import MetricsExporter
    exporter = MetricsExporter()
    exporter.start()  # runs HTTP server on :9090

    # Update metrics from trading loop:
    exporter.update_balance(10500.0)
    exporter.update_drawdown(2.3)
    exporter.update_open_positions(3)
"""

from __future__ import annotations

import os
import time
import threading
import psutil
from typing import Optional

from utils.logger import get_logger

log = get_logger("metrics")


class MetricsExporter:
    """
    Prometheus-compatible metrics exporter.
    Uses plain text format (no prometheus_client dependency needed).
    """

    def __init__(self, port: int = 9090):
        self.port = port
        self._metrics: dict[str, float] = {}
        self._server_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        # Initialize defaults
        self._metrics["forex_ai_balance"] = 0.0
        self._metrics["forex_ai_equity"] = 0.0
        self._metrics["forex_ai_drawdown_pct"] = 0.0
        self._metrics["forex_ai_daily_loss_pct"] = 0.0
        self._metrics["forex_ai_open_positions"] = 0.0
        self._metrics["forex_ai_win_rate"] = 0.0
        self._metrics["forex_ai_trades_total"] = 0.0
        self._metrics["forex_ai_risk_halt"] = 0.0
        self._metrics["forex_ai_latency_ms"] = 0.0
        self._metrics["forex_ai_memory_mb"] = 0.0
        self._metrics["forex_ai_cpu_pct"] = 0.0

    def start(self):
        """Start the HTTP metrics server in background thread."""
        if self._running:
            return
        self._running = True
        self._server_thread = threading.Thread(target=self._serve, daemon=True)
        self._server_thread.start()
        log.info(f"[Metrics] Prometheus exporter started on :{self.port}/metrics")

    def stop(self):
        """Stop the metrics server."""
        self._running = False

    def update_balance(self, balance: float):
        with self._lock:
            self._metrics["forex_ai_balance"] = balance

    def update_equity(self, equity: float):
        with self._lock:
            self._metrics["forex_ai_equity"] = equity

    def update_drawdown(self, pct: float):
        with self._lock:
            self._metrics["forex_ai_drawdown_pct"] = pct

    def update_daily_loss(self, pct: float):
        with self._lock:
            self._metrics["forex_ai_daily_loss_pct"] = pct

    def update_open_positions(self, count: int):
        with self._lock:
            self._metrics["forex_ai_open_positions"] = float(count)

    def update_win_rate(self, wr: float):
        with self._lock:
            self._metrics["forex_ai_win_rate"] = wr

    def update_trades_total(self, total: int):
        with self._lock:
            self._metrics["forex_ai_trades_total"] = float(total)

    def update_risk_halt(self, halted: bool):
        with self._lock:
            self._metrics["forex_ai_risk_halt"] = 1.0 if halted else 0.0

    def update_latency(self, ms: float):
        with self._lock:
            self._metrics["forex_ai_latency_ms"] = ms

    def update_system_stats(self):
        """Update memory and CPU usage."""
        try:
            proc = psutil.Process()
            with self._lock:
                self._metrics["forex_ai_memory_mb"] = proc.memory_info().rss / 1024 / 1024
                self._metrics["forex_ai_cpu_pct"] = proc.cpu_percent(interval=0.1)
        except Exception:
            pass

    def _serve(self):
        """Simple HTTP server that returns metrics in Prometheus format."""
        from http.server import HTTPServer, BaseHTTPRequestHandler

        # Capture in closure to avoid broken class-attribute binding
        # (self.update_system_stats would receive handler instance as first arg)
        _update_stats = self.update_system_stats
        _lock = self._lock
        _metrics = self._metrics

        class MetricsHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/metrics":
                    _update_stats()
                    with _lock:
                        lines = []
                        for key, val in _metrics.items():
                            lines.append(f"{key} {val}")
                    body = "\n".join(lines) + "\n"
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(body.encode())
                elif self.path == "/health":
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"OK")
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, *args):
                pass  # suppress access logs

        try:
            server = HTTPServer(("127.0.0.1", self.port), MetricsHandler)
            while self._running:
                server.handle_request()
        except Exception as e:
            log.error(f"[Metrics] Server error: {e}")


# Singleton
_exporter: Optional[MetricsExporter] = None


def get_metrics_exporter() -> MetricsExporter:
    global _exporter
    if _exporter is None:
        _exporter = MetricsExporter()
    return _exporter
