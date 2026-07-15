# orchestrator/human_override.py — Day 60 | Human Override System
# ============================================================
# AI autonomous হলেও emergency control থাকা উচিত:
#   STOP ALL    — Close all positions, halt trading
#   CLOSE ALL   — Close positions, keep monitoring
#   PAUSE SYSTEM — Pause trading, keep monitoring
#   RESUME      — Resume from pause/stop
#
# Implemented via:
#   1. Command file (memory/human_override.json)
#   2. Telegram bot commands
#   3. Programmatic API
# ============================================================

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

log = get_logger("human_override")

from orchestrator.communication_bus import AgentMessageBus, AgentMessage
from orchestrator.system_state import SystemStateManager

from core.constants import MEMORY_DIR
OVERRIDE_CMD_PATH = MEMORY_DIR / "human_override_cmd.json"


class HumanOverrideSystem:
    """
    Emergency human control system for the AI Trader.
    
    Supports three override mechanisms:
        1. File-based: Write JSON to memory/human_override_cmd.json
        2. Telegram: Send /stop, /close, /pause, /resume commands
        3. API: Call stop_all(), close_all(), pause(), resume()
    """

    VALID_COMMANDS = ("STOP_ALL", "CLOSE_ALL", "PAUSE", "RESUME", "STATUS")

    def __init__(self, state_manager: SystemStateManager, bus: AgentMessageBus):
        self.state_mgr = state_manager
        self.bus = bus
        self._override_history: list[dict] = []
        self._last_check_time = 0
        self._poll_interval = 5  # Check command file every 5 seconds

    def start(self):
        """Start monitoring for human override commands."""
        log.info("[HumanOverride] System active — monitoring for override commands")
        log.info("[HumanOverride] Commands: STOP_ALL, CLOSE_ALL, PAUSE, RESUME, STATUS")

    def check_command_file(self) -> Optional[str]:
        """
        Check the command file for pending override commands.
        Returns the command if found, None otherwise.

        Fail-closed: if the command file exists but cannot be parsed, treat as STOP_ALL
        because the operator clearly intended to issue a command.
        """
        try:
            if OVERRIDE_CMD_PATH.exists():
                with open(OVERRIDE_CMD_PATH, "r") as f:
                    cmd_data = json.load(f)

                command = cmd_data.get("command", "").upper()
                reason = cmd_data.get("reason", "Command file")

                if command in self.VALID_COMMANDS:
                    # Execute and clear the command file
                    OVERRIDE_CMD_PATH.unlink(missing_ok=True)
                    self._execute_command(command, reason, source="file")
                    return command

        except json.JSONDecodeError as e:
            log.critical(f"[HumanOverride] command file exists but unreadable (corrupt JSON): {e}; "
                         f"treating as STOP_ALL (fail-closed)")
            # Fail-closed: corrupt command file means operator tried to issue a command.
            # Trigger STOP_ALL rather than silently continuing.
            try:
                OVERRIDE_CMD_PATH.unlink(missing_ok=True)
            except Exception:
                pass
            self._execute_command("STOP_ALL", f"corrupt command file: {e}", source="file")
            return "STOP_ALL"
        except Exception as e:
            log.critical(f"[HumanOverride] command file check failed: {e}; treating as STOP_ALL (fail-closed)")
            try:
                OVERRIDE_CMD_PATH.unlink(missing_ok=True)
            except Exception:
                pass
            self._execute_command("STOP_ALL", f"command file error: {e}", source="file")
            return "STOP_ALL"

        return None

    def stop_all(self, reason: str = "Human command — STOP ALL") -> dict:
        """
        Emergency STOP ALL:
        - Close all open positions
        - Halt all trading
        - Enter EMERGENCY risk mode
        """
        log.error(f"[HumanOverride] STOP ALL: {reason}")

        # Update state
        self.state_mgr.update(
            human_override="STOPPED",
            risk_mode="EMERGENCY",
            current_task="EMERGENCY_STOP",
        )

        # Publish on bus
        self.bus.publish(AgentMessage(
            source="human_override",
            msg_type="system_event",
            data={
                "event": "STOP_ALL",
                "reason": reason,
                "source": "human",
            },
            priority="critical",
        ))

        result = {
            "action": "STOP_ALL",
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self._record_override(result)
        return result

    def close_all(self, reason: str = "Human command — CLOSE ALL") -> dict:
        """
        Close all positions but keep system running and monitoring.
        """
        log.warning(f"[HumanOverride] CLOSE ALL: {reason}")

        self.state_mgr.update(
            current_task="CLOSING_ALL_POSITIONS",
        )

        self.bus.publish(AgentMessage(
            source="human_override",
            msg_type="system_event",
            data={
                "event": "CLOSE_ALL",
                "reason": reason,
                "source": "human",
            },
            priority="critical",
        ))

        result = {
            "action": "CLOSE_ALL",
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self._record_override(result)
        return result

    def pause(self, reason: str = "Human command — PAUSE") -> dict:
        """Pause trading, keep monitoring."""
        log.warning(f"[HumanOverride] PAUSE: {reason}")

        self.state_mgr.update(
            human_override="PAUSED",
            current_task="PAUSED",
        )

        self.bus.publish(AgentMessage(
            source="human_override",
            msg_type="system_event",
            data={"event": "PAUSE", "reason": reason, "source": "human"},
            priority="high",
        ))

        result = {
            "action": "PAUSE",
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self._record_override(result)
        return result

    def resume(self, reason: str = "Human command — RESUME") -> dict:
        """Resume from pause/stop.

        NOTE: Only clears human_override state. Does NOT clear risk_mode set by
        other safety modules (KillSwitch Level 3, CircuitBreaker pause, etc.).
        Those must be cleared by their respective manual_reset/manual_resume methods.
        """
        log.info(f"[HumanOverride] RESUME: {reason}")

        self.state_mgr.update(
            human_override=None,
            # Do NOT clobber risk_mode — let KillSwitch/CircuitBreaker clear their own state.
            current_task="Scanning Market",
        )

        self.bus.publish(AgentMessage(
            source="human_override",
            msg_type="system_event",
            data={"event": "RESUME", "reason": reason, "source": "human"},
            priority="high",
        ))

        result = {
            "action": "RESUME",
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self._record_override(result)
        return result

    def get_status(self) -> dict:
        """Get current override status."""
        state = self.state_mgr.state
        return {
            "override_active": state.human_override is not None,
            "override_state": state.human_override,
            "command_file_path": str(OVERRIDE_CMD_PATH),
            "history_count": len(self._override_history),
        }

    def _execute_command(self, command: str, reason: str, source: str = "file"):
        """Execute a command from any source."""
        commands = {
            "STOP_ALL": self.stop_all,
            "CLOSE_ALL": self.close_all,
            "PAUSE": self.pause,
            "RESUME": self.resume,
        }
        if command in commands:
            commands[command](f"{reason} (via {source})")
        elif command == "STATUS":
            self._print_status()

    def _record_override(self, result: dict):
        """Record override in history."""
        self._override_history.append(result)

    def _print_status(self):
        """Print override system status."""
        status = self.get_status()
        log.info(f"[HumanOverride] Override: {'ACTIVE' if status['override_active'] else 'NONE'}")
        if status["override_active"]:
            log.info(f"[HumanOverride] State: {status['override_state']}")

    def print_usage(self):
        """Print usage instructions."""
        bar = "=" * 50
        log.info(bar)
        log.info("  HUMAN OVERRIDE SYSTEM")
        log.info(bar)
        log.info("  Commands:")
        log.info("    STOP_ALL   — Close positions, halt trading")
        log.info("    CLOSE_ALL  — Close positions, keep monitoring")
        log.info("    PAUSE      — Pause trading, keep monitoring")
        log.info("    RESUME     — Resume from pause/stop")
        log.info("    STATUS     — Show current override status")
        log.info("")
        log.info("  Via command file:")
        log.info(f'    echo \'{{"command":"STOP_ALL"}}\' > {OVERRIDE_CMD_PATH}')
        log.info("")
        log.info("  Via Telegram bot:")
        log.info("    /stop    /close    /pause    /resume")
        log.info(bar)
