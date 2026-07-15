# core/approval_mode.py  —  Week 3 | Human Approval Mode
# ============================================================
# Mode 1: Analysis Only    — AI দেখায়, কিছু করে না
# Mode 2: Human Approval   — AI suggest করে, human approve করে
# Mode 3: Fully Autonomous — AI নিজেই সিদ্ধান্ত নেয়
# ============================================================

import json
import os
from datetime import datetime
from utils.logger import get_logger

log = get_logger("approval_mode")

MODE_STATE_PATH = "memory/approval_mode.json"

# Mode constants
MODE_ANALYSIS   = 1   # দেখো, শিখো — trade নেই
MODE_SUPERVISED = 2   # AI suggest → human confirm
MODE_AUTONOMOUS = 3   # পুরো AI-controlled


class ApprovalMode:
    """
    AI trader-এর supervision layer।

    তুমি যখন AI-কে trust করতে শুরু করবে তখন
    ধীরে ধীরে Mode 1 → 2 → 3 করবে।

    Usage:
        approval = ApprovalMode(mode=2)

        result = approval.process(signal_result)

        if result["proceed"]:
            # trade নাও
        else:
            # wait for human or skip
    """

    MODE_NAMES = {
        1: "ANALYSIS ONLY",
        2: "SUPERVISED",
        3: "AUTONOMOUS",
    }

    MODE_ICONS = {
        1: "👁️",
        2: "🤝",
        3: "🤖",
    }

    def __init__(self, mode: int = 2):
        # Round-11 audit fix: defensive clamp. Older .env.example files
        # documented APPROVAL_MODE=0 ("manual approval"), but this class
        # only supports modes 1/2/3. A raw KeyError crashed boot. Now:
        # any out-of-range value (0, negative, >3, non-int) is clamped
        # to the SAFE default (MODE_SUPERVISED=2 — manual approval per
        # trade, the documented intent of the old "0" sentinel) and a
        # warning is logged so the operator can fix their .env.
        loaded = self._load_mode()
        candidate = loaded if loaded else mode
        if candidate not in (MODE_ANALYSIS, MODE_SUPERVISED, MODE_AUTONOMOUS):
            log.warning(
                f"[ApprovalMode] Invalid mode={candidate!r} "
                f"(loaded={loaded!r}, default={mode!r}). "
                f"Valid modes are 1/2/3. Falling back to MODE_SUPERVISED (2)."
            )
            candidate = MODE_SUPERVISED
        self._mode = candidate
        # BUGFIX: previously self._pending always started as [] and was
        # never reloaded from memory/pending_approvals.json on startup.
        # In Mode 2 (Supervised) this meant any trade still awaiting human
        # approval at the moment the process restarted (crash, deploy,
        # VPS reboot) became silently invisible: get_pending() returned
        # an empty list, the operator had no way to approve/reject it, and
        # a fresh pending_id sequence starting at 1 could collide with IDs
        # already referenced in an old Telegram message. Reloading here
        # preserves pending approvals (and the id sequence) across restarts.
        self._pending   = self._load_pending()
        log.info(
            f"[ApprovalMode] Active: Mode {self._mode} — "
            f"{self.MODE_NAMES[self._mode]}"
        )

    # ── Main Process ───────────────────────────────────────────

    def process(self, signal_result: dict, telegram_bot=None) -> dict:
        """
        Signal result নিয়ে mode অনুযায়ী কী করতে হবে বলো।

        Returns:
            {
                "proceed":      True/False,
                "mode":         int,
                "mode_name":    str,
                "action":       str,   # "EXECUTE" | "WAIT_APPROVAL" | "ANALYSIS_ONLY"
                "message":      str,   # human-readable
                "signal":       dict,  # original signal
            }
        """
        final_action = signal_result.get("final_action", "NO TRADE")
        symbol       = signal_result.get("symbol", "")
        confidence   = signal_result.get("confidence", 0)
        is_trade     = final_action in ("BUY", "SELL")

        # ── Mode 1: Analysis Only ──────────────────────────────
        if self._mode == MODE_ANALYSIS:
            log.info(
                f"[Mode 1 — Analysis] {symbol}: {final_action} "
                f"({confidence}%) — NOT executed (analysis only)"
            )
            return {
                "proceed":   False,
                "mode":      1,
                "mode_name": "ANALYSIS ONLY",
                "action":    "ANALYSIS_ONLY",
                "message":   (
                    f"👁️ Analysis: {symbol} → {final_action} ({confidence}%)\n"
                    f"Mode 1 active — no trades executed.\n"
                    f"Switch to Mode 2 or 3 to trade."
                ),
                "signal": signal_result,
            }

        # ── Mode 2: Supervised ─────────────────────────────────
        if self._mode == MODE_SUPERVISED:
            if not is_trade:
                return {
                    "proceed":   False,
                    "mode":      2,
                    "mode_name": "SUPERVISED",
                    "action":    "NO_TRADE_SIGNAL",
                    "message":   f"🤝 No trade signal — {final_action}",
                    "signal":    signal_result,
                }

            # Save pending approval
            pending_id = self._add_pending(signal_result)

            summary = self._format_approval_request(signal_result, pending_id)
            log.info(f"[Mode 2 — Supervised] Waiting for approval #{pending_id}")
            log.info(summary)

            # Send to Telegram if bot is available (handle async safely)
            if telegram_bot:
                try:
                    import asyncio
                    if asyncio.iscoroutinefunction(telegram_bot.send_message):
                        from utils.async_utils import run_coro_sync
                        run_coro_sync(telegram_bot.send_message(summary))
                    else:
                        telegram_bot.send_message(summary)
                except Exception as e:
                    log.warning(f"Telegram send failed: {e}")

            return {
                "proceed":    False,
                "mode":       2,
                "mode_name":  "SUPERVISED",
                "action":     "WAIT_APPROVAL",
                "pending_id": pending_id,
                "message":    summary,
                "signal":     signal_result,
            }

        # ── Mode 3: Autonomous ─────────────────────────────────
        if self._mode == MODE_AUTONOMOUS:
            if not is_trade:
                return {
                    "proceed":   False,
                    "mode":      3,
                    "mode_name": "AUTONOMOUS",
                    "action":    "NO_TRADE_SIGNAL",
                    "message":   f"🤖 Autonomous: No trade — {final_action}",
                    "signal":    signal_result,
                }

            log.info(
                f"[Mode 3 — Autonomous] Executing: {symbol} "
                f"{final_action} ({confidence}%)"
            )
            return {
                "proceed":   True,
                "mode":      3,
                "mode_name": "AUTONOMOUS",
                "action":    "EXECUTE",
                "message":   (
                    f"🤖 Auto-executing: {symbol} {final_action} "
                    f"| Conf: {confidence}% | Entry: {signal_result.get('entry')}"
                ),
                "signal": signal_result,
            }

        return {"proceed": False, "mode": self._mode, "action": "UNKNOWN"}

    # ── Approval Flow (Mode 2) ─────────────────────────────────

    def approve(self, pending_id: int) -> dict:
        """
        Human Mode 2-এ trade approve করলে।
        Telegram callback বা console input থেকে call করা যাবে।
        """
        for p in self._pending:
            if p["id"] == pending_id:
                p["status"]    = "APPROVED"
                p["approved_at"] = datetime.utcnow().isoformat()
                self._save_pending()
                log.info(f"[Mode 2] Trade #{pending_id} APPROVED by human")
                return {
                    "proceed": True,
                    "signal":  p["signal"],
                    "message": f"✅ Trade #{pending_id} approved — executing",
                }
        return {"proceed": False, "message": f"Pending #{pending_id} not found"}

    def reject(self, pending_id: int, reason: str = "") -> dict:
        """Human Mode 2-এ trade reject করলে।"""
        for p in self._pending:
            if p["id"] == pending_id:
                p["status"]      = "REJECTED"
                p["rejected_at"] = datetime.utcnow().isoformat()
                p["reason"]      = reason
                self._save_pending()
                log.info(f"[Mode 2] Trade #{pending_id} REJECTED: {reason}")
                return {
                    "proceed": False,
                    "message": f"❌ Trade #{pending_id} rejected — {reason}",
                }
        return {"proceed": False, "message": f"Pending #{pending_id} not found"}

    def get_pending(self) -> list:
        """Mode 2-এ pending approval list।"""
        return [p for p in self._pending if p["status"] == "PENDING"]

    # ── Mode Management ────────────────────────────────────────

    def set_mode(self, mode: int) -> dict:
        """Mode পরিবর্তন করো।"""
        if mode not in (1, 2, 3):
            return {"success": False, "message": "Invalid mode (1/2/3)"}

        old_mode    = self._mode
        self._mode  = mode
        self._save_mode()

        msg = (
            f"Mode changed: {self.MODE_NAMES[old_mode]} → "
            f"{self.MODE_NAMES[mode]}"
        )
        log.info(f"[ApprovalMode] {msg}")
        return {
            "success":  True,
            "old_mode": old_mode,
            "new_mode": mode,
            "message":  msg,
        }

    @property
    def mode(self) -> int:
        return self._mode

    @property
    def mode_name(self) -> str:
        return self.MODE_NAMES.get(self._mode, "UNKNOWN")

    def print_status(self):
        icon = self.MODE_ICONS.get(self._mode, "⚪")
        bar  = "═" * 46
        print(f"\n{bar}")
        print(f"  {icon}  APPROVAL MODE")
        print(bar)
        print(f"  Current Mode  : {self._mode} — {self.mode_name}")
        print(f"  Description   : {self._mode_description()}")
        pending = self.get_pending()
        if pending:
            print(f"  Pending       : {len(pending)} awaiting approval")
        print(bar + "\n")

    # ── Helpers ────────────────────────────────────────────────

    def _mode_description(self) -> str:
        return {
            1: "AI analyzes only — no trades executed",
            2: "AI suggests — you approve before trade",
            3: "AI trades autonomously — no human needed",
        }.get(self._mode, "")

    def _format_approval_request(self, result: dict, pending_id: int) -> str:
        return (
            f"\n{'═'*46}\n"
            f"  🤝  APPROVAL REQUIRED  (#{pending_id})\n"
            f"{'═'*46}\n"
            f"  Symbol     : {result.get('symbol')}\n"
            f"  Action     : {result.get('final_action')}\n"
            f"  Entry      : {result.get('entry')}\n"
            f"  SL         : {result.get('sl')}\n"
            f"  TP         : {result.get('tp')}\n"
            f"  Confidence : {result.get('confidence')}%\n"
            f"  Lot        : {result.get('lot')}\n"
            f"  R:R        : 1:{result.get('rr')}\n"
            f"  Reasoning  : {result.get('llm_analysis', '')[:100]}\n"
            f"{'═'*46}\n"
            f"  → approval.approve({pending_id})   to execute\n"
            f"  → approval.reject({pending_id})    to skip\n"
            f"{'═'*46}\n"
        )

    def _add_pending(self, signal: dict) -> int:
        # BUGFIX: len(self._pending) + 1 collided with old ids once the
        # in-memory list diverged from what was ever persisted (e.g. after
        # a reload that only restores the last 20 entries). Basing the new
        # id on the max id ever seen keeps ids monotonic across restarts.
        pid = (max((p["id"] for p in self._pending), default=0)) + 1
        self._pending.append({
            "id":          pid,
            "status":      "PENDING",
            "created_at":  datetime.utcnow().isoformat(),
            "signal":      signal,
        })
        # Trim in-memory list too (not just the on-disk copy) so a
        # long-running process doesn't accumulate pending history forever.
        if len(self._pending) > 200:
            self._pending = self._pending[-200:]
        self._save_pending()
        return pid

    def _load_pending(self) -> list:
        """Reload pending approvals from disk so Mode 2 survives a restart."""
        path = "memory/pending_approvals.json"
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception as e:
                log.warning(f"Could not load pending approvals: {e}")
        return []

    def _save_pending(self):
        os.makedirs("memory", exist_ok=True)
        # P1 fix: atomic write (temp + os.replace) to prevent crash-corruption.
        # Mirrors the pattern used by kill_switch.py:119, circuit_breaker.py:356,
        # risk_engine.py:311, drawdown_controller.py:452, autonomous_risk.py:1165.
        tmp_path = "memory/pending_approvals.json.tmp"
        with open(tmp_path, "w") as f:
            json.dump(self._pending[-20:], f, indent=2)  # শেষ ২০টা রাখো
        try:
            os.replace(tmp_path, "memory/pending_approvals.json")
        except OSError as e:
            log.error(f"[ApprovalMode] atomic save failed: {e}")
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _load_mode(self) -> int | None:
        if os.path.exists(MODE_STATE_PATH):
            try:
                with open(MODE_STATE_PATH) as f:
                    return json.load(f).get("mode")
            except Exception as e:
                log.warning(f"Suppressed exception at line 314: {e}")
                pass
        return None

    def _save_mode(self):
        os.makedirs("memory", exist_ok=True)
        with open(MODE_STATE_PATH, "w") as f:
            json.dump({"mode": self._mode}, f)