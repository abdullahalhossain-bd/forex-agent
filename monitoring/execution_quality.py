"""
monitoring/execution_quality.py — Day 96 Execution Quality Monitor
===================================================================
Tracks slippage between requested and executed prices to detect
broker issues, spread widening, or market conditions that degrade
execution quality.

Metrics tracked:
  - Slippage per trade (requested vs filled price)
  - Average slippage over last N trades
  - Spread at execution time
  - Execution latency (order sent → filled)
  - Requote count

If average slippage exceeds threshold → reduce trading frequency
or position size.

Usage:
    from monitoring.execution_quality import ExecutionQualityMonitor
    eqm = ExecutionQualityMonitor()
    eqm.record_trade(ticket=123, pair="EURUSD", requested=1.08500,
                     executed=1.08508, spread_pips=1.2, latency_ms=450)
    quality = eqm.get_quality_report()
    # quality = {"avg_slippage_pips": 0.8, "status": "GOOD", "recommendation": "normal"}
"""
from __future__ import annotations

import os
import json
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, Optional

from utils.logger import get_logger

log = get_logger("execution_quality")


class ExecutionQualityMonitor:
    """Tracks execution quality metrics + recommends adjustments."""

    MAX_HISTORY = 50          # keep last 50 trades
    SLIPPAGE_WARN_PIPS = 2.0  # warn if avg slippage > 2 pips
    SLIPPAGE_BAD_PIPS  = 5.0  # bad if avg slippage > 5 pips
    LATENCY_WARN_MS    = 1000 # warn if avg latency > 1000ms

    def __init__(self):
        self._trades: Deque[Dict] = deque(maxlen=self.MAX_HISTORY)
        self._load_history()

    # ─────────────────────────────────────────────────────────
    # RECORD TRADE
    # ─────────────────────────────────────────────────────────

    def record_trade(
        self,
        ticket: int,
        pair: str,
        requested: float,
        executed: float,
        spread_pips: float = 0,
        latency_ms: int = 0,
        direction: str = "BUY",
    ):
        """Record a single trade execution.

        Args:
            ticket:       MT5 ticket number
            pair:         e.g. "EURUSD"
            requested:    price the AI wanted
            executed:     price actually filled
            spread_pips:  spread at execution time
            latency_ms:   order-sent to fill latency
            direction:    "BUY" or "SELL"
        """
        # Calculate slippage
        # For BUY: positive slippage = filled higher than requested (bad)
        # For SELL: positive slippage = filled lower than requested (bad)
        if direction.upper() == "BUY":
            slippage = executed - requested
        else:
            slippage = requested - executed

        # Convert to pips (rough — 0.0001 = 1 pip for most pairs)
        pip_size = 0.01 if "JPY" in pair else 0.0001
        slippage_pips = slippage / pip_size if pip_size > 0 else 0

        trade_record = {
            "ticket":         ticket,
            "pair":           pair,
            "direction":      direction,
            "requested":      requested,
            "executed":       executed,
            "slippage":       round(slippage, 6),
            "slippage_pips":  round(slippage_pips, 2),
            "spread_pips":    spread_pips,
            "latency_ms":     latency_ms,
            "timestamp":      datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

        self._trades.append(trade_record)
        self._save_history()

        # Log if slippage is significant
        if abs(slippage_pips) > self.SLIPPAGE_WARN_PIPS:
            log.warning(
                f"[ExecQuality] {pair} ticket={ticket} | "
                f"slippage={slippage_pips:+.1f} pips | "
                f"req={requested} exec={executed} | spread={spread_pips}p"
            )
        else:
            log.info(
                f"[ExecQuality] {pair} ticket={ticket} | "
                f"slippage={slippage_pips:+.1f} pips | spread={spread_pips}p"
            )

    # ─────────────────────────────────────────────────────────
    # QUALITY REPORT
    # ─────────────────────────────────────────────────────────

    def get_quality_report(self) -> Dict[str, Any]:
        """Get execution quality summary over recent trades.

        Returns: dict with avg_slippage, status, recommendation, etc.
        """
        if not self._trades:
            return {
                "status":         "NO_DATA",
                "trade_count":    0,
                "avg_slippage_pips": 0,
                "avg_spread_pips": 0,
                "avg_latency_ms": 0,
                "recommendation": "no data yet",
            }

        trades = list(self._trades)
        n = len(trades)

        slippages = [abs(t["slippage_pips"]) for t in trades]
        spreads = [t["spread_pips"] for t in trades if t["spread_pips"] > 0]
        latencies = [t["latency_ms"] for t in trades if t["latency_ms"] > 0]

        avg_slip = sum(slippages) / n if n > 0 else 0
        avg_spread = sum(spreads) / len(spreads) if spreads else 0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        # Status classification
        if avg_slip >= self.SLIPPAGE_BAD_PIPS:
            status = "BAD"
            recommendation = "reduce position size 50% — execution quality degraded"
        elif avg_slip >= self.SLIPPAGE_WARN_PIPS:
            status = "WARN"
            recommendation = "monitor closely — slippage above normal"
        elif avg_latency >= self.LATENCY_WARN_MS:
            status = "SLOW"
            recommendation = "latency high — check network/broker connection"
        else:
            status = "GOOD"
            recommendation = "normal execution quality"

        # Worst trade
        worst = max(trades, key=lambda t: abs(t["slippage_pips"]))

        return {
            "status":             status,
            "trade_count":        n,
            "avg_slippage_pips":  round(avg_slip, 2),
            "avg_spread_pips":    round(avg_spread, 2),
            "avg_latency_ms":     round(avg_latency, 0),
            "worst_slippage_pips": round(worst["slippage_pips"], 2),
            "worst_trade_pair":   worst["pair"],
            "recommendation":     recommendation,
        }

    # ─────────────────────────────────────────────────────────
    # PERSISTENCE
    # ─────────────────────────────────────────────────────────

    def _save_history(self):
        """Save trade history to disk (survives restarts)."""
        try:
            os.makedirs("memory", exist_ok=True)
            path = "memory/execution_quality.json"
            with open(path, "w") as f:
                json.dump(list(self._trades)[-50:], f, indent=2)
        except Exception as e:
            log.debug(f"[ExecQuality] save failed: {e}")

    def _load_history(self):
        """Load trade history from disk."""
        try:
            path = "memory/execution_quality.json"
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                for record in data[-50:]:
                    self._trades.append(record)
                log.info(f"[ExecQuality] loaded {len(self._trades)} historical trades")
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────
    # AI CONTEXT
    # ─────────────────────────────────────────────────────────

    def get_ai_context(self, report: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "exec_status":       report.get("status", "NO_DATA"),
            "exec_avg_slippage": report.get("avg_slippage_pips", 0),
            "exec_avg_spread":   report.get("avg_spread_pips", 0),
            "exec_recommendation": report.get("recommendation", ""),
        }

    def print_summary(self, report: Dict[str, Any]) -> None:
        bar = "═" * 50
        log.info(bar)
        log.info("  ⚡  EXECUTION QUALITY  (Day 96)")
        log.info(bar)
        log.info(f"  Status          : {report.get('status','?')}")
        log.info(f"  Trades tracked  : {report.get('trade_count',0)}")
        log.info(f"  Avg slippage    : {report.get('avg_slippage_pips',0):.2f} pips")
        log.info(f"  Avg spread      : {report.get('avg_spread_pips',0):.2f} pips")
        log.info(f"  Avg latency     : {report.get('avg_latency_ms',0):.0f} ms")
        log.info(f"  Worst slippage  : {report.get('worst_slippage_pips',0):.2f} pips ({report.get('worst_trade_pair','')})")
        log.info(f"  Recommendation  : {report.get('recommendation','')}")
        log.info(bar)


# ── Singleton ────────────────────────────────────────────────────

_INSTANCE: Optional[ExecutionQualityMonitor] = None

def get_execution_quality_monitor() -> ExecutionQualityMonitor:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = ExecutionQualityMonitor()
    return _INSTANCE
