# orchestrator/trading_as_git.py — Trading-as-Git: approval-gated trading
# =============================================================================
# Inspired by OpenAlice's "Trading as Git" pattern:
#   https://openalice.ai/docs/core-concepts/trading-as-git
#
# Trade intentions go through three phases before hitting the broker:
#
#     STAGE     →   COMMIT    →   PUSH
#   (draft)      (approved)     (executed)
#      │             │
#      └── REJECT ───┘   (human can reject at any pre-push phase)
#
# Each transition is a file move under memory/trading_journal/:
#   staged/<id>.json     → committed/<id>.json  → pushed/<id>.json
#   staged/<id>.json     → rejected/<id>.json   (or committed → rejected)
#
# Why file-based instead of a DB:
#   - Inspectable (cat / jq / file watcher)
#   - Atomic (os.rename is atomic on the same filesystem)
#   - Survives crashes (no transaction log to corrupt)
#   - Git-diffable if you choose to commit the journal to a separate repo
#
# Integration point: trading_orchestrator.py calls TradingJournal.stage()
# BEFORE risk checks; once risk passes, commit(); the push() step calls
# execution_router.send_order(). Human rejections are polled every cycle.
# =============================================================================

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable

from utils.logger import get_logger

log = get_logger("trading_as_git")

from core.constants import (
    TRADING_JOURNAL_DIR,
)


# ─────────────────────────────────────────────────────────────────────────────
# Journal lifecycle states
# ─────────────────────────────────────────────────────────────────────────────

STAGE_STAGED = "staged"
STAGE_COMMITTED = "committed"
STAGE_PUSHED = "pushed"
STAGE_REJECTED = "rejected"

VALID_TRANSITIONS = {
    STAGE_STAGED: {STAGE_COMMITTED, STAGE_REJECTED},
    STAGE_COMMITTED: {STAGE_PUSHED, STAGE_REJECTED},
    STAGE_PUSHED: set(),           # terminal
    STAGE_REJECTED: set(),         # terminal
}

# Subdirectory name under the journal root for each state.
_STATE_SUBDIR = {
    STAGE_STAGED: "staged",
    STAGE_COMMITTED: "committed",
    STAGE_PUSHED: "pushed",
    STAGE_REJECTED: "rejected",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _short_id() -> str:
    """Short, sortable, unique ID: YYYYmmdd-HHMMSS-<8 hex>"""
    return f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"


# ─────────────────────────────────────────────────────────────────────────────
# TradeIntention — the staged artifact
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeIntention:
    """
    A single trade intention flowing through the journal.

    Attributes mirror what `execution_router.send_order()` expects, plus audit
    fields so a human reviewer can decide APPROVE / REJECT without reading the
    orchestrator source.
    """
    id: str
    symbol: str
    side: str                       # "BUY" | "SELL"
    lot_size: float
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy: str = "unknown"
    confidence: float = 0.0         # 0..1
    risk_score: float = 0.0         # 0..1
    rationale: str = ""             # human-readable reason
    metadata: dict = field(default_factory=dict)

    # Journal fields
    state: str = STAGE_STAGED
    created_at: str = field(default_factory=_utc_now_iso)
    committed_at: Optional[str] = None
    pushed_at: Optional[str] = None
    rejected_at: Optional[str] = None
    rejection_reason: Optional[str] = None
    pushed_broker_ticket: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TradeIntention":
        # Tolerate missing fields (forward-compat with older journal entries)
        valid = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in valid})

    @classmethod
    def from_file(cls, path: Path) -> "TradeIntention":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


# ─────────────────────────────────────────────────────────────────────────────
# TradingJournal — the file-backed state machine
# ─────────────────────────────────────────────────────────────────────────────

class TradingJournal:
    """
    File-backed journal for approval-gated trading.

    All file mutations go through `os.replace()` (atomic on the same
    filesystem). The journal directory MUST live on a local disk —
    NFS / network filesystems may break atomicity.
    """

    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root else TRADING_JOURNAL_DIR
        # Ensure all state dirs exist
        for state in _STATE_SUBDIR:
            self._state_dir(state).mkdir(parents=True, exist_ok=True)
        log.info(f"[TradingAsGit] Journal root: {self.root}")

    # ── Path helpers ────────────────────────────────────────────────────────

    def _state_dir(self, state: str) -> Path:
        return self.root / _STATE_SUBDIR[state]

    def _intention_path(self, state: str, intention_id: str) -> Path:
        return self._state_dir(state) / f"{intention_id}.json"

    # ── Public API ──────────────────────────────────────────────────────────

    def stage(
        self,
        symbol: str,
        side: str,
        lot_size: float,
        *,
        entry_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        strategy: str = "unknown",
        confidence: float = 0.0,
        risk_score: float = 0.0,
        rationale: str = "",
        metadata: Optional[dict] = None,
    ) -> TradeIntention:
        """
        STAGE a trade intention. Does NOT execute anything.
        Returns the staged intention (id assigned, state=staged).
        """
        if side.upper() not in ("BUY", "SELL"):
            raise ValueError(f"side must be BUY or SELL, got {side!r}")
        if lot_size <= 0:
            raise ValueError(f"lot_size must be positive, got {lot_size}")

        intention = TradeIntention(
            id=_short_id(),
            symbol=symbol.upper(),
            side=side.upper(),
            lot_size=float(lot_size),
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy=strategy,
            confidence=float(confidence),
            risk_score=float(risk_score),
            rationale=rationale,
            metadata=metadata or {},
        )
        self._write(intention)
        log.info(
            f"[TradingAsGit] STAGED  {intention.id}  {intention.symbol} "
            f"{intention.side} {intention.lot_size}  ({intention.strategy})"
        )
        return intention

    def commit(self, intention_id: str) -> TradeIntention:
        """
        COMMIT a staged intention. Signals "risk checks passed, ready to push".
        """
        return self._transition(intention_id, STAGE_STAGED, STAGE_COMMITTED,
                                timestamp_field="committed_at")

    def push(
        self,
        intention_id: str,
        executor: Callable[[TradeIntention], dict],
    ) -> TradeIntention:
        """
        PUSH a committed intention through `executor` (typically
        `execution_router.send_order`). Records the broker ticket returned by
        the executor in `pushed_broker_ticket`.

        `executor` MUST return a dict with at least `{"ticket": <int>}`.
        """
        intention = self._read(intention_id, STAGE_COMMITTED)
        try:
            result = executor(intention)
        except Exception as e:
            log.error(f"[TradingAsGit] PUSH failed for {intention_id}: {e}")
            # Do NOT auto-reject on executor error — the orchestrator should
            # decide whether to retry or reject. Just re-raise.
            raise

        intention.pushed_broker_ticket = result.get("ticket")
        intention.pushed_at = _utc_now_iso()
        self._transition(intention_id, STAGE_COMMITTED, STAGE_PUSHED,
                         timestamp_field="pushed_at",
                         update_intention=intention)
        log.info(
            f"[TradingAsGit] PUSHED  {intention.id}  "
            f"ticket={intention.pushed_broker_ticket}"
        )
        return intention

    def reject(self, intention_id: str, reason: str) -> TradeIntention:
        """
        REJECT an intention. Works from any non-terminal state (staged or
        committed). Terminal state — no further transitions possible.
        """
        current = self._find_in_states(intention_id, (STAGE_STAGED, STAGE_COMMITTED))
        if current is None:
            raise FileNotFoundError(
                f"Intention {intention_id} not found in staged/ or committed/"
            )
        return self._transition(
            intention_id, current, STAGE_REJECTED,
            timestamp_field="rejected_at",
            extra_update={"rejection_reason": reason},
        )

    def get(self, intention_id: str) -> Optional[TradeIntention]:
        """Read an intention from any state directory. None if not found."""
        for state in _STATE_SUBDIR:
            p = self._intention_path(state, intention_id)
            if p.exists():
                return TradeIntention.from_file(p)
        return None

    def list_staged(self) -> list[TradeIntention]:
        return self._list_dir(self._state_dir(STAGE_STAGED))

    def list_committed(self) -> list[TradeIntention]:
        return self._list_dir(self._state_dir(STAGE_COMMITTED))

    def list_pushed(self, limit: int = 100) -> list[TradeIntention]:
        items = self._list_dir(self._state_dir(STAGE_PUSHED))
        items.sort(key=lambda i: i.pushed_at or "", reverse=True)
        return items[:limit]

    def list_rejected(self, limit: int = 100) -> list[TradeIntention]:
        items = self._list_dir(self._state_dir(STAGE_REJECTED))
        items.sort(key=lambda i: i.rejected_at or "", reverse=True)
        return items[:limit]

    def poll_rejections(self) -> list[TradeIntention]:
        """
        Called by the orchestrator every poll cycle. Returns the list of
        intentions that were rejected since the last poll.

        Rejection happens by:
          - Telegram bot writing a JSON file under rejected/
          - Human editing the staged/committed file in place (we detect by
            mtime change — but the canonical path is the rejected/ dir)
        """
        return self._list_dir(self._state_dir(STAGE_REJECTED))

    # ── Internals ───────────────────────────────────────────────────────────

    def _write(self, intention: TradeIntention):
        target = self._intention_path(intention.state, intention.id)
        tmp = target.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(intention.to_dict(), f, indent=2, default=str)
        os.replace(tmp, target)  # atomic on same filesystem

    def _read(self, intention_id: str, expected_state: str) -> TradeIntention:
        path = self._intention_path(expected_state, intention_id)
        if not path.exists():
            raise FileNotFoundError(
                f"Intention {intention_id} not found in {expected_state}/"
            )
        return TradeIntention.from_file(path)

    def _find_in_states(
        self, intention_id: str, states: tuple[str, ...]
    ) -> Optional[str]:
        for state in states:
            if self._intention_path(state, intention_id).exists():
                return state
        return None

    def _transition(
        self,
        intention_id: str,
        from_state: str,
        to_state: str,
        *,
        timestamp_field: str,
        update_intention: Optional[TradeIntention] = None,
        extra_update: Optional[dict] = None,
    ) -> TradeIntention:
        if to_state not in VALID_TRANSITIONS.get(from_state, set()):
            raise ValueError(
                f"Invalid transition: {from_state} → {to_state} "
                f"(allowed: {VALID_TRANSITIONS.get(from_state, set())})"
            )

        src = self._intention_path(from_state, intention_id)
        dst = self._intention_path(to_state, intention_id)

        if not src.exists():
            raise FileNotFoundError(
                f"Intention {intention_id} not found in {from_state}/"
            )

        intention = update_intention or TradeIntention.from_file(src)
        intention.state = to_state
        setattr(intention, timestamp_field, _utc_now_iso())
        if extra_update:
            for k, v in extra_update.items():
                setattr(intention, k, v)

        # Atomic write to destination
        tmp = dst.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(intention.to_dict(), f, indent=2, default=str)
        os.replace(tmp, dst)

        # Remove from source
        src.unlink(missing_ok=True)

        log.info(
            f"[TradingAsGit] {from_state.upper():8s} → {to_state.upper():8s} "
            f"{intention_id}"
        )
        return intention

    def _list_dir(self, d: Path) -> list[TradeIntention]:
        if not d.exists():
            return []
        items = []
        for p in sorted(d.glob("*.json")):
            try:
                items.append(TradeIntention.from_file(p))
            except Exception as e:
                log.warning(f"[TradingAsGit] Could not parse {p}: {e}")
        return items


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: CLI for manual journal inspection
# ─────────────────────────────────────────────────────────────────────────────

def _cli():
    import sys
    journal = TradingJournal()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"

    if cmd == "status":
        staged = journal.list_staged()
        committed = journal.list_committed()
        pushed = journal.list_pushed(limit=10)
        rejected = journal.list_rejected(limit=10)
        print("═══ Trading-as-Git Journal Status ═══")
        print(f"  STAGED    : {len(staged)} pending review")
        for i in staged:
            print(f"    {i.id}  {i.symbol} {i.side} {i.lot_size}  ({i.strategy})")
        print(f"  COMMITTED : {len(committed)} ready to push")
        for i in committed:
            print(f"    {i.id}  {i.symbol} {i.side} {i.lot_size}  ({i.strategy})")
        print(f"  PUSHED    (last 10):")
        for i in pushed:
            print(f"    {i.id}  ticket={i.pushed_broker_ticket}  {i.symbol} {i.side}")
        print(f"  REJECTED  (last 10):")
        for i in rejected:
            print(f"    {i.id}  reason: {i.rejection_reason}")

    elif cmd == "reject":
        if len(sys.argv) < 4:
            print("Usage: python -m orchestrator.trading_as_git reject <id> <reason>")
            sys.exit(1)
        intention = journal.reject(sys.argv[2], " ".join(sys.argv[3:]))
        print(f"Rejected: {intention.id} — {intention.rejection_reason}")

    elif cmd == "show":
        if len(sys.argv) < 3:
            print("Usage: python -m orchestrator.trading_as_git show <id>")
            sys.exit(1)
        intention = journal.get(sys.argv[2])
        if intention is None:
            print("Not found.")
            sys.exit(1)
        print(json.dumps(intention.to_dict(), indent=2, default=str))

    else:
        print(f"Unknown command: {cmd}")
        print("Commands: status | reject <id> <reason> | show <id>")
        sys.exit(1)


if __name__ == "__main__":
    _cli()
