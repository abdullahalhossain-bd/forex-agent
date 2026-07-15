# orchestrator/system_state.py — Day 60 | System State Manager
# ============================================================
# AI সবসময় জানবে system-এর current state।
#
# Tracks:
#   - Operating mode (RESEARCH / PAPER / DEMO / LIVE)
#   - Market status (OPEN / CLOSED / WEEKEND)
#   - Risk mode (AGGRESSIVE / NORMAL / DEFENSIVE / EMERGENCY)
#   - Active trade count and daily P&L
#   - System health (MT5 connected, API status, etc.)
#   - Current cycle info
#
# Example state:
#   {
#     "mode": "LIVE",
#     "market": "OPEN",
#     "risk_mode": "NORMAL",
#     "active_trades": 2,
#     "daily_loss": 0.8,
#     "system_health": "HEALTHY"
#   }
# ============================================================

import json
import time
from datetime import datetime, timezone
from typing import Any, Optional
from pathlib import Path

from utils.logger import get_logger

log = get_logger("system_state")

from core.constants import MEMORY_DIR
SYSTEM_STATE_PATH = MEMORY_DIR / "system_state.json"

# Market session times (UTC)
FOREX_OPEN_HOUR = 0  # Sunday 22:00 UTC = Monday 00:00 server
FOREX_CLOSE_HOUR = 21  # Friday 21:00 UTC


class SystemState:
    """
    Immutable snapshot of the entire system state.
    """

    def __init__(self, **kwargs):
        self.timestamp = kwargs.get("timestamp", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        self.mode = kwargs.get("mode", "PAPER")  # RESEARCH, PAPER, DEMO, LIVE
        self.market_status = kwargs.get("market_status", "CLOSED")  # OPEN, CLOSED, WEEKEND
        self.risk_mode = kwargs.get("risk_mode", "NORMAL")  # AGGRESSIVE, NORMAL, DEFENSIVE, EMERGENCY
        self.active_trades = kwargs.get("active_trades", 0)
        self.daily_pnl_pct = kwargs.get("daily_pnl_pct", 0.0)
        self.daily_pnl_usd = kwargs.get("daily_pnl_usd", 0.0)
        self.weekly_pnl_pct = kwargs.get("weekly_pnl_pct", 0.0)
        self.balance = kwargs.get("balance", 0.0)
        self.starting_balance = kwargs.get("starting_balance", 0.0)
        self.system_health = kwargs.get("system_health", "INITIALIZING")  # HEALTHY, DEGRADED, DOWN
        self.current_task = kwargs.get("current_task", "STARTUP")
        self.cycle_count = kwargs.get("cycle_count", 0)
        self.current_cycle_id = kwargs.get("current_cycle_id", None)
        self.last_cycle_time = kwargs.get("last_cycle_time", None)
        self.mt5_connected = kwargs.get("mt5_connected", False)
        self.paper_trader_active = kwargs.get("paper_trader_active", True)
        self.research_active = kwargs.get("research_active", False)
        self.risk_manager_active = kwargs.get("risk_manager_active", False)
        self.dashboard_running = kwargs.get("dashboard_running", False)
        self.telegram_active = kwargs.get("telegram_active", False)
        self.scanner_active = kwargs.get("scanner_active", False)
        self.human_override = kwargs.get("human_override", None)  # None, PAUSED, STOPPED
        self.errors_today = kwargs.get("errors_today", 0)
        self.uptime_seconds = kwargs.get("uptime_seconds", 0)
        self.current_pair = kwargs.get("current_pair", None)
        self.current_decision = kwargs.get("current_decision", None)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "mode": self.mode,
            "market_status": self.market_status,
            "risk_mode": self.risk_mode,
            "active_trades": self.active_trades,
            "daily_pnl_pct": round(self.daily_pnl_pct, 4),
            "daily_pnl_usd": round(self.daily_pnl_usd, 2),
            "weekly_pnl_pct": round(self.weekly_pnl_pct, 4),
            "balance": round(self.balance, 2),
            "starting_balance": round(self.starting_balance, 2),
            "system_health": self.system_health,
            "current_task": self.current_task,
            "cycle_count": self.cycle_count,
            "current_cycle_id": self.current_cycle_id,
            "last_cycle_time": self.last_cycle_time,
            "mt5_connected": self.mt5_connected,
            "paper_trader_active": self.paper_trader_active,
            "research_active": self.research_active,
            "risk_manager_active": self.risk_manager_active,
            "dashboard_running": self.dashboard_running,
            "telegram_active": self.telegram_active,
            "scanner_active": self.scanner_active,
            "human_override": self.human_override,
            "errors_today": self.errors_today,
            "uptime_seconds": self.uptime_seconds,
            "current_pair": self.current_pair,
            "current_decision": self.current_decision,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SystemState":
        return cls(**d)

    def is_trading_allowed(self) -> bool:
        """Check if trading is currently allowed."""
        if self.human_override in ("PAUSED", "STOPPED"):
            return False
        if self.risk_mode == "EMERGENCY":
            return False
        if self.market_status != "OPEN":
            return False
        if self.mode == "RESEARCH":
            return False
        if self.system_health == "DOWN":
            return False
        return True

    def is_analysis_allowed(self) -> bool:
        """Check if analysis (non-execution) is allowed."""
        if self.system_health == "DOWN":
            return False
        if self.human_override == "STOPPED":
            return False
        return True

    def get_summary_line(self) -> str:
        """Get a one-line status summary."""
        icons = {
            "HEALTHY": "green",
            "DEGRADED": "yellow",
            "DOWN": "red",
        }
        health_icon = {"HEALTHY": "OK", "DEGRADED": "WARN", "DOWN": "FAIL"}.get(
            self.system_health, "?"
        )
        return (
            f"Mode={self.mode} | Market={self.market_status} | "
            f"Risk={self.risk_mode} | Trades={self.active_trades} | "
            f"Daily P&L={self.daily_pnl_pct:+.2f}% | Health={health_icon} | "
            f"Task={self.current_task}"
        )


class SystemStateManager:
    """
    Manages the global system state. Thread-safe state transitions.
    Persists state to disk for crash recovery.
    """

    def __init__(self):
        self._state = SystemState()
        self._start_time = time.time()
        self._listeners: list = []
        self._load_state()

    @property
    def state(self) -> SystemState:
        """Get current immutable state snapshot."""
        return self._state

    def get_state(self) -> dict:
        """Get state as dict."""
        return self._state.to_dict()

    def update(self, **kwargs) -> SystemState:
        """
        Update state fields. Returns new state snapshot.
        Notifies all listeners of the change.
        """
        current = self._state.to_dict()
        current.update(kwargs)
        current["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        current["uptime_seconds"] = int(time.time() - self._start_time)
        self._state = SystemState(**current)
        self._save_state()
        self._notify_listeners()
        return self._state

    def update_market_status(self) -> str:
        """
        Auto-detect forex market status based on UTC day/hour.
        Returns the detected status.
        """
        now = datetime.now(timezone.utc)
        weekday = now.weekday()  # 0=Monday, 6=Sunday

        if weekday == 5:  # Saturday
            status = "WEEKEND"
        elif weekday == 6 and now.hour < 22:  # Sunday before 22:00 UTC
            status = "WEEKEND"
        elif weekday == 4 and now.hour >= 21:  # Friday after 21:00 UTC
            status = "CLOSED"
        else:
            status = "OPEN"

        self.update(market_status=status)
        log.debug(f"[SystemState] Market status: {status}")
        return status

    def on_state_change(self, callback) -> None:
        """Register a state change listener."""
        self._listeners.append(callback)

    def _notify_listeners(self) -> None:
        for callback in self._listeners:
            try:
                callback(self._state)
            except Exception as e:
                log.error(f"[SystemState] Listener error: {e}", exc_info=True)

    def _load_state(self) -> None:
        """Load state from disk (crash recovery)."""
        try:
            if SYSTEM_STATE_PATH.exists():
                with open(SYSTEM_STATE_PATH, "r") as f:
                    data = json.load(f)
                self._state = SystemState.from_dict(data)
                log.info(f"[SystemState] Restored state from disk")
        except Exception as e:
            log.warning(f"[SystemState] Could not load state: {e}")

    def _save_state(self) -> None:
        """Persist state to disk."""
        try:
            MEMORY_DIR.mkdir(parents=True, exist_ok=True)
            with open(SYSTEM_STATE_PATH, "w") as f:
                json.dump(self._state.to_dict(), f, indent=2)
        except Exception as e:
            log.warning(f"[SystemState] Could not save state: {e}")

    def print_dashboard(self) -> None:
        """Print a dashboard-style state summary."""
        s = self._state
        bar = "=" * 55
        log.info(bar)
        log.info("  AI TRADER SYSTEM STATE")
        log.info(bar)
        
        mode_icons = {"RESEARCH": "🔬", "PAPER": "📝", "DEMO": "🧪", "LIVE": "💰"}
        market_icons = {"OPEN": "🟢", "CLOSED": "🔴", "WEEKEND": "⏸️"}
        risk_icons = {"AGGRESSIVE": "🔥", "NORMAL": "⚖️", "DEFENSIVE": "🛡️", "EMERGENCY": "🚨"}
        health_icons = {"HEALTHY": "✅", "DEGRADED": "⚠️", "DOWN": "❌"}

        log.info(f"  Mode          : {mode_icons.get(s.mode, '?')} {s.mode}")
        log.info(f"  Market        : {market_icons.get(s.market_status, '?')} {s.market_status}")
        log.info(f"  Risk Mode     : {risk_icons.get(s.risk_mode, '?')} {s.risk_mode}")
        log.info(f"  Health        : {health_icons.get(s.system_health, '?')} {s.system_health}")
        log.info(f"  Current Task  : {s.current_task}")
        log.info(f"  Active Trades : {s.active_trades}")
        log.info(f"  Daily P&L     : {s.daily_pnl_pct:+.2f}% (${s.daily_pnl_usd:+.2f})")
        log.info(f"  Balance       : ${s.balance:,.2f}")
        log.info(f"  Cycles        : {s.cycle_count}")
        log.info(f"  Uptime        : {s.uptime_seconds // 3600}h {(s.uptime_seconds % 3600) // 60}m")
        if s.human_override:
            log.info(f"  HUMAN OVERRIDE : {s.human_override}")
        log.info(bar)
