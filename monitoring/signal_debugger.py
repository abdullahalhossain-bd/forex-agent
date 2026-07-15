"""
monitoring/signal_debugger.py — Signal Pipeline Debugger (Day 81+)
                                  + Day 91 hotfix (root-cause tracking)
====================================================================

WHY THIS EXISTS:
    The system has 9+ pipeline layers (Market → Indicators → SMC →
    Liquidity → Pattern → Sentiment → ML → LLM → Risk → Execution).
    Each layer can independently reject a trade. When the system runs
    24h without a single trade, there's no way to know WHICH layer
    is killing the signal — until now.

WHAT IT DOES:
    Every cycle, the trader calls `debugger.start_cycle(symbol)` then
    records each layer's verdict as it happens:

        debugger.record("market_data", "OK", "300 candles fetched")
        debugger.record("smc",         "BUY", "BOS confirmed @ 1.0850")
        debugger.record("ml",          "WAIT", "confidence 45% < 50% threshold")
        debugger.record("risk",        "REJECTED", "lot 0 — Kelly negative")
        debugger.record_final("NO_TRADE", "Risk rejected: Kelly negative")

    At end of cycle, debugger logs a one-block summary so you can
    instantly see WHERE signals die:

        ┌─ EURUSD M15 cycle ─────────────────────────┐
        │ market_data    OK       300 candles        │
        │ indicators     OK       RSI=52 trend=UP    │
        │ smc            BUY      BOS @ 1.0850       │
        │ liquidity      DETECTED 1.0820 sweep       │
        │ pattern        NEUTRAL  no signal          │
        │ sentiment      BULLISH  +8 boost           │
        │ ml             BUY      68% confidence     │
        │ llm            WAIT     45% (below 50%)   │
        │ risk           REJECT   Kelly negative     │
        │ permission     BLOCK    3/5 checks passed  │
        │ ────────────────────────────────────────── │
        │ FINAL: NO_TRADE — Risk rejected            │
        │ ────────────────────────────────────────── │
        │ FIRST_BLOCKED_AT: risk                     │
        │ BLOCKED_AT:       permission                │
        └────────────────────────────────────────────┘

    Day 91 hotfix notes (why this changed):
        - The old code only recorded the LAST hard-block (REJECT/BLOCK/
          ERROR) in the chain as `blocked_at`. In practice this meant a
          downstream confirmation gate (e.g. "permission") would silently
          overwrite the layer that actually rejected first (e.g. "risk"),
          making it look like "permission" was the root cause when it
          was really just re-confirming a decision "risk" already made.
        - Fix: track BOTH `first_blocked_at` (root cause — the first hard
          block in the chain) and `blocked_at` (kept for backward
          compatibility — still the LAST hard block, same as before).
          Dashboards / stats that want root-cause analysis should use
          `first_blocked_at`; anything reading `blocked_at` keeps working
          unchanged.
        - The ASCII summary's `detail` column used to silently truncate
          long messages (e.g. multi-line LLM-key errors) with no
          indication that anything was cut. The on-screen box still
          truncates for layout (it's a fixed-width box), but it now
          shows an ellipsis when it does, and the JSONL file written by
          save_to_file() always stores the FULL untruncated detail text
          — only the printed box is shortened.
        - block_stats()/recent_blocks()/recent_trades() used to read only
          from the in-memory deque, so a process restart silently reset
          all aggregate stats to zero even though memory/signal_debug.jsonl
          still had the full history on disk. They now lazily backfill
          from disk on first use if the in-memory history is empty.

USAGE:
    from monitoring.signal_debugger import get_signal_debugger

    debugger = get_signal_debugger()
    debugger.start_cycle("EURUSD", "M15")

    # ... in market_agent.py:
    debugger.record("market_data", "OK", f"{len(df)} candles")

    # ... in smc_engine.py:
    debugger.record("smc", signal, f"{structure} @ {level}")

    # ... at end of cycle in trader.py:
    debugger.record_final(final_action, reject_reason)
    debugger.log_cycle_summary()
    debugger.save_to_file()  # for dashboard
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

from utils.logger import get_logger

log = get_logger("signal_debugger")

# Where the latest cycle's full debug log lives (dashboard reads this)
DEBUG_FILE = "memory/signal_debug.jsonl"
DEBUG_HISTORY_LIMIT = 500  # keep last N cycles in JSONL (in-memory deque cap)

# Hard-block statuses — a layer with one of these explicitly refused to proceed.
HARD_BLOCK_STATUSES = ("REJECT", "BLOCK", "ERROR")

# Width used for the `detail` column in the printed ASCII box. This is a
# display-only constraint — the JSONL file always keeps the full text.
_DETAIL_DISPLAY_WIDTH = 22


# ── Per-layer verdict ─────────────────────────────────────────

@dataclass
class LayerVerdict:
    """One pipeline layer's verdict for one cycle."""
    layer: str           # e.g. "market_data", "smc", "ml", "llm", "risk"
    status: str          # OK / BUY / SELL / WAIT / REJECT / BLOCK / ERROR
    detail: str = ""     # short human-readable reason (full text, never truncated here)
    timestamp: float = field(default_factory=time.time)

    @property
    def icon(self) -> str:
        return {
            "OK":       "[OK]",
            "BUY":      "[BUY]",
            "SELL":     "[SELL]",
            "WAIT":     "[WAIT]",
            "REJECT":   "[REJ]",
            "BLOCK":    "[BLK]",
            "ERROR":    "[ERR]",
            "NEUTRAL":  "[---]",
            "DETECTED": "[DET]",
        }.get(self.status, f"[{self.status[:3].upper()}]")

    def display_detail(self, width: int = _DETAIL_DISPLAY_WIDTH) -> str:
        """Detail text truncated for the fixed-width ASCII box, with a
        visible ellipsis when truncation actually happens (the old code
        truncated silently with no indication anything was cut)."""
        text = self.detail or ""
        if len(text) <= width:
            return text
        if width <= 1:
            return text[:width]
        return text[: width - 1] + "…"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer":      self.layer,
            "status":     self.status,
            "detail":     self.detail,   # full, untruncated — JSONL keeps everything
            "timestamp":  self.timestamp,
        }


# ── Per-cycle record ──────────────────────────────────────────

@dataclass
class CycleDebug:
    """All layer verdicts + final outcome for one cycle."""
    symbol: str
    timeframe: str
    started_at: float
    layers: List[LayerVerdict] = field(default_factory=list)
    final_action: str = ""
    final_reason: str = ""
    blocked_at: Optional[str] = None        # LAST hard-block layer (kept for back-compat)
    first_blocked_at: Optional[str] = None  # Day 91: FIRST hard-block layer — root cause

    def record(self, layer: str, status: str, detail: str = "") -> None:
        self.layers.append(LayerVerdict(layer=layer, status=status, detail=detail))

    def record_final(self, action: str, reason: str = "") -> None:
        self.final_action = action
        self.final_reason = reason

        # Day 91: find BOTH the first and last hard-block in the chain.
        # `first_blocked_at` is the root cause — the earliest layer that
        # explicitly refused to proceed. `blocked_at` is kept exactly as
        # before (the LAST hard-block) so any existing code/dashboards
        # reading `blocked_at` keep working unchanged.
        hard_blocks = [v.layer for v in self.layers if v.status in HARD_BLOCK_STATUSES]
        if hard_blocks:
            self.first_blocked_at = hard_blocks[0]
            self.blocked_at = hard_blocks[-1]
        else:
            # No hard-block but final is NO_TRADE/WAIT — the killer is the
            # last layer that emitted WAIT (no signal). Same for first/last
            # since there's no distinct "root cause vs confirmation" chain
            # in the soft-block case.
            if action in ("NO_TRADE", "WAIT"):
                for v in reversed(self.layers):
                    if v.status == "WAIT":
                        self.blocked_at = v.layer
                        self.first_blocked_at = v.layer
                        break

    def summary_block(self) -> str:
        """One-box ASCII summary for the log file."""
        lines = [
            f"┌─ {self.symbol} {self.timeframe} cycle ─────────────────────────┐",
        ]
        for v in self.layers:
            lines.append(f"│ {v.layer:<14} {v.icon:<7} {v.display_detail():<22} │")
        lines.append("│ ────────────────────────────────────────── │")
        action_icon = {
            "BUY":     "🟢 BUY",
            "SELL":    "🔴 SELL",
            "WAIT":    "⏳ WAIT",
            "NO_TRADE":"⛔ NO_TRADE",
        }.get(self.final_action, self.final_action)
        # Day 81+ hotfix: self.final_reason can be None (when trade
        # succeeds and no rejection reason was set).  Coerce to str
        # before subscripting — otherwise 'NoneType' object is not
        # subscriptable crash after successful fills.
        reason = (self.final_reason or "")[:24]
        lines.append(f"│ FINAL: {action_icon:<9} — {reason:<24} │")
        if self.blocked_at:
            lines.append(f"│ ────────────────────────────────────────── │")
            if self.first_blocked_at and self.first_blocked_at != self.blocked_at:
                # Day 91: show both when they differ, so it's clear which
                # layer is the root cause vs. which one just confirmed it.
                lines.append(f"│ FIRST_BLOCKED_AT: {self.first_blocked_at:<25} │")
                lines.append(f"│ BLOCKED_AT:       {self.blocked_at:<25} │")
            else:
                lines.append(f"│ BLOCKED_AT: {self.blocked_at:<31} │")
        lines.append("└────────────────────────────────────────────┘")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol":           self.symbol,
            "timeframe":        self.timeframe,
            "started_at":       datetime.fromtimestamp(self.started_at, tz=timezone.utc).isoformat(),
            "layers":           [v.to_dict() for v in self.layers],
            "final_action":     self.final_action,
            "final_reason":     self.final_reason,
            "blocked_at":       self.blocked_at,        # last hard-block (back-compat)
            "first_blocked_at": self.first_blocked_at,  # Day 91: root cause
        }


# ── Singleton debugger ────────────────────────────────────────

class SignalDebugger:
    """
    Accumulates layer verdicts across the current cycle and across
    historical cycles (for trend analysis).
    """

    def __init__(self, history_limit: int = DEBUG_HISTORY_LIMIT):
        self._current: Optional[CycleDebug] = None
        self._history: Deque[Dict[str, Any]] = deque(maxlen=history_limit)
        # Aggregate stats — how often is each layer the (root-cause) blocker?
        self._block_counts: Dict[str, int] = {}
        # Day 91: track whether we've already tried to backfill stats from
        # disk this process lifetime, so we only do it once.
        self._backfilled_from_disk = False

    # ── Per-cycle API ──────────────────────────────────────────

    def start_cycle(self, symbol: str, timeframe: str) -> None:
        self._current = CycleDebug(
            symbol=symbol,
            timeframe=timeframe,
            started_at=time.time(),
        )

    def record(self, layer: str, status: str, detail: str = "") -> None:
        """Record one layer's verdict. No-op if start_cycle wasn't called."""
        if self._current is None:
            return
        self._current.record(layer, status, detail)

    def record_final(self, action: str, reason: str = "") -> None:
        if self._current is None:
            return
        self._current.record_final(action, reason)

    def log_cycle_summary(self) -> None:
        """Print the one-box summary to the logger."""
        if self._current is None:
            return
        log.info("\n" + self._current.summary_block())

    def save_to_file(self) -> None:
        """Persist this cycle to JSONL for dashboard / trend analysis."""
        if self._current is None:
            return
        os.makedirs(os.path.dirname(DEBUG_FILE), exist_ok=True)
        record = self._current.to_dict()
        self._history.append(record)
        # Day 91: aggregate stats now key off `first_blocked_at` (root
        # cause) instead of `blocked_at` (last/downstream confirmation).
        root_cause = record.get("first_blocked_at") or record.get("blocked_at")
        if root_cause:
            self._block_counts[root_cause] = self._block_counts.get(root_cause, 0) + 1
        try:
            with open(DEBUG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            log.debug(f"signal_debugger save failed: {e}")

    # ── Day 91: disk backfill ──────────────────────────────────

    def _ensure_backfilled(self) -> None:
        """
        If the in-memory history/stats are empty (e.g. right after a
        process restart) but memory/signal_debug.jsonl has prior cycles
        on disk, load the most recent ones so block_stats()/recent_blocks()/
        recent_trades() don't silently report empty results after a restart.
        Runs at most once per process lifetime.
        """
        if self._backfilled_from_disk:
            return
        self._backfilled_from_disk = True

        if self._history or not os.path.exists(DEBUG_FILE):
            return

        try:
            with open(DEBUG_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            log.debug(f"signal_debugger backfill read failed: {e}")
            return

        # Only need the most recent DEBUG_HISTORY_LIMIT records.
        for line in lines[-DEBUG_HISTORY_LIMIT:]:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            self._history.append(record)
            root_cause = record.get("first_blocked_at") or record.get("blocked_at")
            if root_cause:
                self._block_counts[root_cause] = self._block_counts.get(root_cause, 0) + 1

        log.info(f"signal_debugger: backfilled {len(self._history)} cycles from disk")

    # ── Aggregate stats ────────────────────────────────────────

    def block_stats(self) -> Dict[str, Any]:
        """Summary of which layers blocked trades over the recent history.
        Keyed off root-cause (first hard-block), not the downstream
        confirmation layer. Backfills from disk if memory is empty
        (e.g. right after a restart)."""
        self._ensure_backfilled()
        total = sum(self._block_counts.values())
        return {
            "total_blocked_cycles": total,
            "by_layer":             dict(self._block_counts),
            "most_common_blocker":  max(self._block_counts,
                                        key=self._block_counts.get) if self._block_counts else None,
        }

    def recent_blocks(self, last_n: int = 10) -> List[Dict[str, Any]]:
        """Last N cycles that ended in NO_TRADE — for dashboard."""
        self._ensure_backfilled()
        return [r for r in self._history if r.get("final_action") in ("NO_TRADE", "WAIT")][-last_n:]

    def recent_trades(self, last_n: int = 10) -> List[Dict[str, Any]]:
        """Last N cycles that ended in a trade — for dashboard."""
        self._ensure_backfilled()
        return [r for r in self._history if r.get("final_action") in ("BUY", "SELL")][-last_n:]


# ── Singleton accessor ────────────────────────────────────────

_DEBUGGER: Optional[SignalDebugger] = None


def get_signal_debugger() -> SignalDebugger:
    global _DEBUGGER
    if _DEBUGGER is None:
        _DEBUGGER = SignalDebugger()
    return _DEBUGGER
