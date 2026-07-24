"""
orchestrator/self_healing.py — Real remediation (2026-07 rewrite).

Previously a no-op stub (Day 60 placeholder, Day 102+ hotfix only added
the API surface). `heal()` always returned False and did nothing;
`on_error()` couldn't even read the real error out of the message it was
given (it looked for `msg.kind`/`msg.detail`, but the AgentMessage objects
actually published on the bus use `.msg_type`/`.data` — see
orchestrator/communication_bus.py's AgentMessage — so every recorded issue
was mis-tagged "unknown" with the message's repr() as the detail).

This rewrite does two honest things, and deliberately does NOT pretend to
do a third:
  1. FIXED: on_error() now reads the real (stage, symbol, error) out of
     msg.data, so issues are recorded with actual, useful information.
  2. REAL HEALER — MT5 reconnect: if an error's text matches
     connection/timeout patterns, look up the already-connected
     MT5Connection singleton(s) (broker/mt5_connection.py) and call their
     existing, tested `.reconnect()` (exponential backoff already built
     in there). This is a real fix for a real, common failure mode.
  3. REAL CONTAINMENT — symbol quarantine: self-healing cannot know how
     to fix an arbitrary, unclassified exception — no amount of pattern
     matching turns "the analysis stage raised ValueError" into a
     targeted fix. What it CAN safely do is notice when the SAME
     (symbol, stage) fails repeatedly in a short window and stop the
     orchestrator from re-entering that failure on every single cycle
     (which burns time/API budget and spams logs/errors for no benefit).
     Quarantine is time-boxed (auto-expires) and orchestrator/
     trading_orchestrator.py's per-symbol loop now checks
     `is_quarantined()` before running a symbol's cycle.

  NOT implemented, on purpose: rebuilding corrupted DB indexes, rotating
  log files, restarting arbitrary sub-systems. Those need per-subsystem
  knowledge this class doesn't have; faking them would be worse than the
  honest `return False` this module used to give for everything.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# Error-text patterns that indicate an MT5/broker connectivity problem
# (as opposed to a logic bug elsewhere in the pipeline). Matched
# case-insensitively against the error string.
_CONNECTION_ERROR_PATTERNS = re.compile(
    r"(mt5|metatrader|not connected|connection|disconnect|timed?\s*out|"
    r"ipc\s*timeout|terminal.*not.*found|no\s*route\s*to\s*host|"
    r"broken\s*pipe|socket)",
    re.IGNORECASE,
)

# How many failures for the same (symbol, stage) within the window below
# triggers quarantine.
_QUARANTINE_FAILURE_THRESHOLD = 3
_QUARANTINE_WINDOW_SECONDS = 15 * 60     # 15 minutes
_QUARANTINE_DURATION_SECONDS = 30 * 60   # 30 minutes, then auto-expires

# Don't hammer MT5.reconnect() more than once per this many seconds even
# if multiple connection-flavored errors arrive back to back.
_MT5_RECONNECT_COOLDOWN_SECONDS = 20


class SelfHealingSystem:
    """Detects recurring runtime errors and applies automatic remediation
    where a real, safe remediation exists; otherwise contains repeat
    failures via time-boxed symbol quarantine rather than pretending to
    fix them.
    """

    def __init__(self, bus=None, state_mgr=None):
        self._bus = bus
        self._state_mgr = state_mgr
        self._issues: List[Dict[str, Any]] = []
        self._remediations: List[Dict[str, Any]] = []
        self._healers: List[Dict[str, Any]] = []

        # (symbol, stage) -> list of failure timestamps (for quarantine)
        self._failure_history: Dict[tuple, List[float]] = {}
        # symbol -> quarantine-expiry unix timestamp
        self._quarantined: Dict[str, float] = {}
        self._last_mt5_reconnect_attempt: float = 0.0

    # ── Boot ─────────────────────────────────────────────────

    def register_healers(self) -> None:
        self._healers = [
            {"kind": "mt5_reconnect", "matches": "connection-pattern errors"},
            {"kind": "symbol_quarantine", "matches": "repeated same-symbol failures"},
        ]
        log.info(
            "[SelfHealing] register_healers() — 2 real healers active: "
            "mt5_reconnect, symbol_quarantine"
        )

    # ── Bus hook ─────────────────────────────────────────────

    def on_error(self, msg: Any) -> None:
        """Subscribed to the bus's 'error' topic. AgentMessage carries the
        real error info in `.data` (see communication_bus.py), not
        `.kind`/`.detail` — this used to look in the wrong place and
        record every issue as kind='unknown'.
        """
        data: Dict[str, Any] = {}
        try:
            if isinstance(msg, dict):
                data = msg.get("data", {}) or {}
            else:
                data = getattr(msg, "data", {}) or {}
        except Exception:
            pass

        stage = data.get("stage", "unknown")
        symbol = data.get("symbol")
        error_text = data.get("error", str(msg))
        kind = f"stage:{stage}" if stage != "unknown" else "unknown"

        self.record_issue(kind=kind, detail=error_text, symbol=symbol, stage=stage)

    # ── Core remediation entry point ────────────────────────

    def heal(self, error: Any, symbol: Optional[str] = None, stage: str = "unknown") -> bool:
        """Called by the orchestrator when an error needs immediate
        remediation. Tries real healers in order; returns True only if
        one of them genuinely acted (not just logged).
        """
        error_text = str(error)
        healed = False
        action = "none"

        if _CONNECTION_ERROR_PATTERNS.search(error_text):
            if self._try_mt5_reconnect():
                healed = True
                action = "mt5_reconnect"

        if not healed and symbol:
            if self._register_failure_and_maybe_quarantine(symbol, stage):
                healed = True
                action = "symbol_quarantine"

        self._remediations.append({
            "ts": time.time(), "kind": "heal", "error": error_text,
            "symbol": symbol, "stage": stage, "action": action, "healed": healed,
        })
        if healed:
            log.warning(f"[SelfHealing] heal() action='{action}' for {symbol or '?'}/{stage}: {error_text}")
        else:
            log.warning(
                f"[SelfHealing] heal() called for {symbol or '?'}/{stage} — no known "
                f"remediation for this error pattern (recorded, not fixed): {error_text}"
            )
        return healed

    # ── Healer #1: MT5 reconnect ────────────────────────────

    def _try_mt5_reconnect(self) -> bool:
        now = time.time()
        if now - self._last_mt5_reconnect_attempt < _MT5_RECONNECT_COOLDOWN_SECONDS:
            log.info("[SelfHealing] mt5_reconnect skipped — cooldown active")
            return False
        self._last_mt5_reconnect_attempt = now

        try:
            from broker.mt5_connection import MT5Connection, MT5_AVAILABLE
        except Exception as e:
            log.info(f"[SelfHealing] mt5_reconnect unavailable (import failed): {e}")
            return False

        if not MT5_AVAILABLE:
            log.info("[SelfHealing] mt5_reconnect skipped — MetaTrader5 package not installed")
            return False

        instances = list(getattr(MT5Connection, "_instances", {}).values())
        if not instances:
            log.info("[SelfHealing] mt5_reconnect skipped — no MT5Connection instance exists yet")
            return False

        any_success = False
        for inst in instances:
            try:
                if not getattr(inst, "connected", False):
                    ok = inst.reconnect()
                    any_success = any_success or ok
            except Exception as e:
                log.warning(f"[SelfHealing] mt5_reconnect attempt raised: {e}")
        return any_success

    # ── Healer #2: symbol quarantine (containment) ──────────

    def _register_failure_and_maybe_quarantine(self, symbol: str, stage: str) -> bool:
        key = (symbol, stage)
        now = time.time()
        hist = self._failure_history.setdefault(key, [])
        hist.append(now)
        # Drop entries outside the rolling window
        cutoff = now - _QUARANTINE_WINDOW_SECONDS
        hist[:] = [t for t in hist if t >= cutoff]

        if len(hist) >= _QUARANTINE_FAILURE_THRESHOLD and symbol not in self._quarantined:
            self._quarantined[symbol] = now + _QUARANTINE_DURATION_SECONDS
            log.error(
                f"[SelfHealing] QUARANTINE: {symbol} failed {len(hist)}x on stage "
                f"'{stage}' within {_QUARANTINE_WINDOW_SECONDS//60}min — pausing this "
                f"symbol for {_QUARANTINE_DURATION_SECONDS//60}min. This does not fix "
                f"the underlying error; it stops the same failure from repeating every "
                f"cycle. Check logs for the root cause before it re-enters rotation."
            )
            return True
        return False

    def is_quarantined(self, symbol: str) -> bool:
        """Called by the orchestrator's per-symbol loop before processing
        a symbol. Auto-expires quarantine once the duration elapses."""
        expiry = self._quarantined.get(symbol)
        if expiry is None:
            return False
        if time.time() >= expiry:
            del self._quarantined[symbol]
            log.info(f"[SelfHealing] Quarantine expired for {symbol} — re-entering rotation")
            return False
        return True

    def get_quarantined_symbols(self) -> Dict[str, float]:
        """Returns {symbol: seconds_remaining} for currently-quarantined symbols."""
        now = time.time()
        return {s: round(exp - now) for s, exp in self._quarantined.items() if exp > now}

    # ── Issue log ────────────────────────────────────────────

    def record_issue(self, kind: str, detail: str, symbol: Optional[str] = None, stage: str = "unknown") -> None:
        entry = {"ts": time.time(), "kind": kind, "detail": detail, "symbol": symbol, "stage": stage}
        self._issues.append(entry)
        log.warning(f"[SelfHealing] issue: {kind} | symbol={symbol} stage={stage} | {detail}")
        self.heal(detail, symbol=symbol, stage=stage)

    def get_recent_issues(self, limit: int = 20) -> List[Dict[str, Any]]:
        return list(self._issues[-limit:])

    def status(self) -> Dict[str, Any]:
        return {
            "issues_recorded": len(self._issues),
            "remediations_attempted": len(self._remediations),
            "remediations_that_actually_healed": sum(1 for r in self._remediations if r.get("healed")),
            "healers_registered": len(self._healers),
            "quarantined_symbols": self.get_quarantined_symbols(),
        }
