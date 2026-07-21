"""
risk/kill_switch.py — Emergency Kill Switch (Day 75)
=====================================================

3-level emergency brake system:

Level 1 — Daily Loss Limit (default 3%)
  Action: Stop trading for the rest of the day.

Level 2 — Weekly Loss Limit (default 8%)
  Action: Pause trading for 7 days. System review required.

Level 3 — Maximum Drawdown (default 15%)
  Action: FULL STOP. Human review required. All automation disabled.

The kill switch is persistent — once triggered, it stays active until
the cooldown period expires or a human manually resets it.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from utils.logger import get_logger

log = get_logger("kill_switch")

STATE_PATH = Path("memory/kill_switch_state.json")


class KillSwitch:
    """3-level emergency brake with persistent state."""

    # Default thresholds
    DAILY_LOSS_LIMIT = 0.03       # default — overridden by config below
    # Day 81+ hotfix: load from config (default 20.0% = 0.20).
    # Was hard-coded 0.03 (3%) — user wants 20%.
    try:
        from config import DAILY_LOSS_LIMIT_PCT as _CFG_DLL
        DAILY_LOSS_LIMIT = float(_CFG_DLL) / 100.0  # percent → fraction
    except Exception as e:
        # P0 fix: fail-closed on safety-critical config (mirrors circuit_breaker.py:54).
        # Previously silently fell back to 0.20 (20% daily loss) — too permissive.
        import logging as _logging
        _log = _logging.getLogger("kill_switch")
        _log.critical(f"[KillSwitch] DAILY_LOSS_LIMIT_PCT config load failed: {e}; "
                      f"crashing to prevent silent 20% daily-loss fallback.")
        raise RuntimeError(f"Safety-critical config missing: DAILY_LOSS_LIMIT_PCT — {e}") from e
    WEEKLY_LOSS_LIMIT = 0.08      # 8% weekly loss
    MAX_DRAWDOWN_LIMIT = 0.15     # 15% total drawdown
    DAILY_COOLDOWN_HOURS = 16     # rest of trading day + overnight
    WEEKLY_COOLDOWN_DAYS = 7      # 1 week pause
    DRAWDOWN_COOLDOWN_DAYS = 30   # manual reset required

    def __init__(self):
        self._lock = threading.RLock()
        self._state = self._load()

    def _load(self) -> Dict[str, Any]:
        """
        Day 102+ CRITICAL hotfix: FAIL CLOSED on corruption, not open.

        Previously, any read error (corrupt JSON, missing file, permission
        error) returned a fresh all-False state — silently disabling the
        kill switch and allowing trades. A kill switch that silently
        disables itself on corruption is worse than no kill switch.

        Now: on corruption, return a state with level_3_active=True
        (FULL STOP) so trading is blocked until a human investigates.
        This mirrors the fail-closed behavior already used by
        CircuitBreaker and RiskEngine.
        """
        if STATE_PATH.exists():
            try:
                state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
                # Validate required keys — if any are missing, treat as corrupt
                required_keys = (
                    "level_1_active", "level_2_active", "level_3_active",
                    "daily_loss_pct", "weekly_loss_pct", "current_drawdown_pct",
                )
                if not all(k in state for k in required_keys):
                    raise ValueError(f"missing required keys: {required_keys}")
                return state
            except Exception as e:
                log.critical(
                    f"[KillSwitch] STATE FILE CORRUPT ({e}) — "
                    f"FAILING CLOSED (Level 3 FULL STOP). "
                    f"Manual reset required after investigation."
                )
                # Fail closed: trigger Level 3 (max drawdown = full stop)
                return {
                    "level_1_active": False,
                    "level_1_until": None,
                    "level_2_active": False,
                    "level_2_until": None,
                    "level_3_active": True,  # ← BLOCKS ALL TRADING
                    "level_3_until": (datetime.now(timezone.utc) + timedelta(days=30)).isoformat(),
                    "daily_loss_pct": 999999.0,
                    "weekly_loss_pct": 999999.0,
                    "current_drawdown_pct": 999999.0,
                    "peak_balance": 0.0,
                    "last_reset": datetime.now(timezone.utc).isoformat(),
                    "pause_reason": "STATE FILE CORRUPT — manual intervention required",
                    "_corrupt": True,
                }
        return {
            "level_1_active": False,
            "level_1_until": None,
            "level_2_active": False,
            "level_2_until": None,
            "level_3_active": False,
            "level_3_until": None,
            "daily_loss_pct": 0.0,
            "weekly_loss_pct": 0.0,
            "current_drawdown_pct": 0.0,
            "peak_balance": 0.0,
            "last_reset": datetime.now(timezone.utc).isoformat(),
        }

    def _save(self) -> None:
        """
        Day 102+ CRITICAL hotfix: atomic write (temp file + os.replace).

        Previously used direct STATE_PATH.write_text() — if the process
        crashed mid-write (or the disk filled, or another writer raced),
        the state file would be left truncated/corrupt. Combined with
        the old fail-open _load(), this would silently disable the kill
        switch on the next read. Atomic write guarantees the file is
        either fully-written or untouched.
        """
        import tempfile
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=str(STATE_PATH.parent),
                suffix=".tmp",
                prefix="kill_switch_",
                delete=False,
            ) as tmp_f:
                json.dump(self._state, tmp_f, indent=2)
                tmp_path = tmp_f.name
            # os.replace is atomic on the same filesystem
            import os
            os.replace(tmp_path, str(STATE_PATH))
        except Exception as e:
            log.warning(f"[KillSwitch] save failed: {e}")
            # Cleanup temp file if rename failed
            if tmp_path is not None:
                try:
                    import os
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def check(
        self,
        balance: float,
        initial_balance: float,
        daily_pnl: float = 0.0,
        weekly_pnl: float = 0.0,
    ) -> Dict[str, Any]:
        """Check all kill switch levels.

        Returns:
            {
                "trading_allowed": bool,
                "level": 0/1/2/3,
                "reason": str,
                "cooldown_until": str or None,
            }
        """
        with self._lock:
            now = datetime.now(timezone.utc)

            # Check if any cooldown has expired
            for level in ("level_1", "level_2", "level_3"):
                until_str = self._state.get(f"{level}_until")
                if until_str and self._state.get(f"{level}_active"):
                    try:
                        until = datetime.fromisoformat(until_str)
                        if now > until:
                            self._state[f"{level}_active"] = False
                            self._state[f"{level}_until"] = None
                            log.info(f"[KillSwitch] {level} cooldown expired — cleared")
                    except Exception as e:
                        log.warning(f"Suppressed exception at line 116: {e}")
                        pass

            # Level 3 (highest priority)
            if self._state.get("level_3_active"):
                # Bug #33 fix: save state before early return so drawdown
                # tracking isn't lost if the process crashes.
                self._save()
                return self._block(3, "MAXIMUM DRAWDOWN — FULL STOP. Human reset required.")

            # Calculate drawdown from PEAK balance (not initial).
            # Bug #8 fix: previously used initial_balance, so once the account
            # became profitable, drawdown was negative (e.g. initial=$10k,
            # peak=$20k, balance=$14k → old: (10k-14k)/10k = -0.4 → never
            # triggered). Now uses peak_balance like DrawdownMonitor does.
            peak = self._state.get("peak_balance", 0)
            if peak <= 0:
                peak = initial_balance
            if balance > peak:
                peak = balance
            self._state["peak_balance"] = peak

            if peak > 0:
                drawdown = (peak - balance) / peak
                self._state["current_drawdown_pct"] = drawdown

                if drawdown >= self.MAX_DRAWDOWN_LIMIT:
                    self._trigger_level3()
                    return self._block(3, f"Max drawdown {drawdown:.1%} ≥ {self.MAX_DRAWDOWN_LIMIT:.0%}")

            # Level 2
            if self._state.get("level_2_active"):
                return self._block(2, "WEEKLY LOSS LIMIT — 7 day pause active.")

            if initial_balance > 0 and weekly_pnl < 0:
                weekly_loss_pct = abs(weekly_pnl) / initial_balance
                self._state["weekly_loss_pct"] = weekly_loss_pct
                if weekly_loss_pct >= self.WEEKLY_LOSS_LIMIT:
                    self._trigger_level2()
                    return self._block(2, f"Weekly loss {weekly_loss_pct:.1%} ≥ {self.WEEKLY_LOSS_LIMIT:.0%}")

            # Level 1
            if self._state.get("level_1_active"):
                return self._block(1, "DAILY LOSS LIMIT — trading paused for today.")

            if initial_balance > 0 and daily_pnl < 0:
                daily_loss_pct = abs(daily_pnl) / initial_balance
                self._state["daily_loss_pct"] = daily_loss_pct
                if daily_loss_pct >= self.DAILY_LOSS_LIMIT:
                    self._trigger_level1()
                    return self._block(1, f"Daily loss {daily_loss_pct:.1%} ≥ {self.DAILY_LOSS_LIMIT:.0%}")

            self._save()
            return {"trading_allowed": True, "level": 0, "reason": "All clear", "cooldown_until": None}

    def _trigger_level1(self) -> None:
        self._state["level_1_active"] = True
        self._state["level_1_until"] = (datetime.now(timezone.utc) + timedelta(hours=self.DAILY_COOLDOWN_HOURS)).isoformat()
        log.warning(f"[KillSwitch] LEVEL 1 TRIGGERED — daily loss limit. Paused for {self.DAILY_COOLDOWN_HOURS}h")
        self._save()

    def _trigger_level2(self) -> None:
        self._state["level_2_active"] = True
        self._state["level_2_until"] = (datetime.now(timezone.utc) + timedelta(days=self.WEEKLY_COOLDOWN_DAYS)).isoformat()
        log.warning(f"[KillSwitch] LEVEL 2 TRIGGERED — weekly loss limit. Paused for {self.WEEKLY_COOLDOWN_DAYS} days")
        self._save()

    def _trigger_level3(self) -> None:
        self._state["level_3_active"] = True
        self._state["level_3_until"] = (datetime.now(timezone.utc) + timedelta(days=self.DRAWDOWN_COOLDOWN_DAYS)).isoformat()
        log.critical(f"[KillSwitch] LEVEL 3 TRIGGERED — maximum drawdown. FULL STOP.")
        self._save()

    def _block(self, level: int, reason: str) -> Dict[str, Any]:
        return {
            "trading_allowed": False,
            "level": level,
            "reason": reason,
            "cooldown_until": self._state.get(f"level_{level}_until"),
        }

    def manual_reset(self, level: int = 3) -> bool:
        """Manually reset a kill switch level (human override)."""
        with self._lock:
            key = f"level_{level}_active"
            if self._state.get(key):
                self._state[key] = False
                self._state[f"level_{level}_until"] = None
                self._save()
                log.info(f"[KillSwitch] Level {level} manually reset")
                return True
            return False

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "level_1_active": self._state.get("level_1_active", False),
                "level_2_active": self._state.get("level_2_active", False),
                "level_3_active": self._state.get("level_3_active", False),
                "daily_loss_pct": self._state.get("daily_loss_pct", 0),
                "weekly_loss_pct": self._state.get("weekly_loss_pct", 0),
                "current_drawdown_pct": self._state.get("current_drawdown_pct", 0),
                "thresholds": {
                    "daily": self.DAILY_LOSS_LIMIT,
                    "weekly": self.WEEKLY_LOSS_LIMIT,
                    "drawdown": self.MAX_DRAWDOWN_LIMIT,
                },
            }


# ── Singleton ───────────────────────────────────────────────────────

_SWITCH: Optional[KillSwitch] = None


def get_kill_switch() -> KillSwitch:
    global _SWITCH
    if _SWITCH is None:
        _SWITCH = KillSwitch()
    return _SWITCH
