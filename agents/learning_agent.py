# agents/learning_agent.py — Day 12 | Self-Learning Agent
# Institutional hardening pass — see README_learning_agent.md for full audit notes.

from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, TypedDict

from utils.logger import get_logger
from core.constants import MEMORY_DIR

log = get_logger("learning_agent")

DEFAULT_PATH = str(MEMORY_DIR / "trade_memory.json")
PATH = DEFAULT_PATH  # kept for backward compatibility with any external `from agents.learning_agent import PATH`
MAX_HISTORY = 500


class TradeResult(str, Enum):
    """Valid closed-trade outcomes. Using an Enum instead of a raw string
    prevents silent typos (e.g. "Win" vs "WIN") from corrupting stats."""
    WIN = "WIN"
    LOSS = "LOSS"
    BE = "BE"  # breakeven


class DecisionEntry(TypedDict, total=False):
    id: int
    timestamp: str
    symbol: Optional[str]
    timeframe: Optional[str]
    decision: Optional[str]
    raw_signal: Optional[str]
    gated: bool
    confidence: Optional[float]
    entry: Optional[float]
    sl: Optional[float]
    tp: Optional[float]
    lot: Optional[float]
    rr: Optional[float]
    regime: Optional[str]
    trend: Optional[str]
    rsi: Optional[float]
    patterns: list
    rule_signal: Optional[str]
    llm_signal: Optional[str]
    reasons: list
    outcome: Optional[str]
    pnl_pips: Optional[float]
    result: Optional[str]
    closed_at: Optional[str]


class LearningAgent:
    """
    Persists every trading decision to a JSON-backed log and later backfills
    the realized outcome (WIN/LOSS/BE) once a trade closes, so pattern-level
    performance stats can feed the strategy-selection / RL layers upstream.

    This is a best-effort *learning log*, not the system of record for PnL —
    that lives in the SQLite-backed `self._memory.on_trade_closed()` path in
    core/trader.py. This file caps at `max_history` entries by design; older
    entries are discarded on rotation, not archived.

    Thread-safety: a per-instance `threading.Lock` serializes all
    read-modify-write access to the JSON file within one process. If more
    than one process writes to the same `path` (e.g. a live trader and a
    separate backtester both pointed at the same file), add cross-process
    locking (e.g. `filelock.FileLock(path + ".lock")`) — a threading.Lock
    only protects threads inside this interpreter.
    """

    def __init__(self, path: str = DEFAULT_PATH, max_history: int = MAX_HISTORY) -> None:
        self.path = path
        self.max_history = max_history
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Write path: decision opened
    # ------------------------------------------------------------------
    def save_decision(
        self,
        decision_out: dict,
        analysis_out: dict,
        market_out: dict,
    ) -> int:
        """Persist a decision entry to the trade-memory file.

        Returns the new entry's stable, monotonic id (max(existing ids) + 1,
        so ids survive history truncation on rotation — see class docstring).
        Stash this id on the trade context so the close handler can later
        call `update_outcome(decision_id, ...)` directly instead of relying
        on the `update_outcome_by_symbol` fallback.

        Args:
            decision_out: output of the decision/execution stage — expects
                keys like `decision`, `raw_signal`, `gated_by_permission`,
                `confidence`, `entry`, `sl`, `tp`, `lot`, `rr`, `reasons`.
            analysis_out: output of the analysis stage — expects
                `pat_ctx.recent_patterns`, `signal.signal`, `llm.signal`.
            market_out: output of the market/data stage — expects `symbol`,
                `timeframe`, `regime.regime`, `ind_ctx.trend`, `ind_ctx.rsi`.

        Thread-safe: holds the instance lock for the full read-modify-write.
        """
        with self._lock:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            history = self._load()

            next_id = 1
            if history:
                existing_ids = [e.get("id") for e in history if isinstance(e.get("id"), int)]
                if existing_ids:
                    next_id = max(existing_ids) + 1

            entry: DecisionEntry = {
                "id": next_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": market_out.get("symbol"),
                "timeframe": market_out.get("timeframe"),
                "decision": decision_out.get("decision"),
                "raw_signal": decision_out.get("raw_signal"),
                "gated": decision_out.get("gated_by_permission", False),
                "confidence": decision_out.get("confidence"),
                "entry": decision_out.get("entry"),
                "sl": decision_out.get("sl"),
                "tp": decision_out.get("tp"),
                "lot": decision_out.get("lot"),
                "rr": decision_out.get("rr"),
                "regime": market_out.get("regime", {}).get("regime"),
                "trend": market_out.get("ind_ctx", {}).get("trend"),
                "rsi": market_out.get("ind_ctx", {}).get("rsi"),
                "patterns": analysis_out.get("pat_ctx", {}).get("recent_patterns", []),
                "rule_signal": analysis_out.get("signal", {}).get("signal"),
                "llm_signal": analysis_out.get("llm", {}).get("signal"),
                "reasons": decision_out.get("reasons", []),
                "outcome": None,
                "pnl_pips": None,
                "result": None,
            }

            history.append(entry)
            self._save(history)
            log.info(f"[LearningAgent] Decision #{entry['id']} saved — {entry['decision']}")
            return entry["id"]

    # ------------------------------------------------------------------
    # Read path: aggregate stats
    # ------------------------------------------------------------------
    def get_performance_stats(self) -> dict:
        """Aggregate win rate, average PnL, and per-pattern win/loss counts
        over all closed trades currently retained in the log.

        `win_rate` is computed over decisive trades only (WIN + LOSS);
        breakeven trades are counted in `closed_trades`/`avg_pnl_pips` but
        excluded from the win-rate denominator so BE-heavy periods don't
        silently dilute the reported edge.
        """
        with self._lock:
            history = self._load()

        closed = [t for t in history if t.get("result")]
        if not closed:
            return {"total_decisions": len(history), "closed_trades": 0}

        wins = [t for t in closed if t["result"] == TradeResult.WIN.value]
        losses = [t for t in closed if t["result"] == TradeResult.LOSS.value]
        breakeven = [t for t in closed if t["result"] == TradeResult.BE.value]

        decisive = len(wins) + len(losses)
        win_rate = round(len(wins) / decisive * 100, 1) if decisive else 0.0
        avg_pnl = round(sum(t.get("pnl_pips") or 0.0 for t in closed) / len(closed), 1)

        pat_stats: dict[str, dict[str, int]] = {}
        for t in closed:
            result = t.get("result")
            if result not in (TradeResult.WIN.value, TradeResult.LOSS.value):
                continue  # BE trades don't count as either a pattern win or loss
            for p in (t.get("patterns") or []):
                bucket = pat_stats.setdefault(p, {"win": 0, "loss": 0})
                bucket["win" if result == TradeResult.WIN.value else "loss"] += 1

        return {
            "total_decisions": len(history),
            "closed_trades": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "breakeven": len(breakeven),
            "win_rate": win_rate,
            "avg_pnl_pips": avg_pnl,
            "pattern_stats": pat_stats,
        }

    # ------------------------------------------------------------------
    # Write path: outcome backfill
    # ------------------------------------------------------------------
    def update_outcome(self, decision_id: int, result: str, pnl_pips: float = 0.0) -> bool:
        """Mark a previously-saved decision as WIN/LOSS/BE, by id.

        Raises:
            ValueError: if `result` is not one of TradeResult's values.

        Returns:
            True if a matching entry was found and updated, False otherwise.
        """
        validated_result = self._validate_result(result)
        with self._lock:
            history = self._load()
            updated = False
            for entry in history:
                if entry.get("id") == decision_id:
                    entry["result"] = validated_result
                    entry["pnl_pips"] = pnl_pips
                    entry["outcome"] = validated_result  # legacy alias
                    entry["closed_at"] = datetime.now(timezone.utc).isoformat()
                    updated = True
                    break
            if updated:
                self._save(history)

        if updated:
            log.info(f"[LearningAgent] Decision #{decision_id} updated: {validated_result} | {pnl_pips} pips")
        else:
            log.warning(f"[LearningAgent] Decision #{decision_id} not found — outcome not saved")
        return updated

    def update_outcome_by_symbol(self, symbol: str, result: str, pnl_pips: float = 0.0) -> Optional[int]:
        """Fallback: mark the most recent OPEN decision for `symbol`.

        Used when the close handler doesn't have the original decision id
        (e.g. trade was opened by a previous process run, or the id was lost
        in transit). Returns the decision id that was updated, or None if no
        open decision was found for that symbol.

        Note: with multiple concurrent open positions on the same symbol,
        this will match the *most recently opened* one — it cannot
        disambiguate between several simultaneously-open trades on the same
        pair. Prefer `update_outcome(decision_id, ...)` whenever the id is
        available.
        """
        validated_result = self._validate_result(result)
        with self._lock:
            history = self._load()
            target_idx = None
            for i in range(len(history) - 1, -1, -1):
                entry = history[i]
                if entry.get("symbol") == symbol and not entry.get("result"):
                    target_idx = i
                    break

            if target_idx is None:
                log.warning(f"[LearningAgent] No open decision found for {symbol} — outcome not saved")
                return None

            entry = history[target_idx]
            entry["result"] = validated_result
            entry["pnl_pips"] = pnl_pips
            entry["outcome"] = validated_result
            entry["closed_at"] = datetime.now(timezone.utc).isoformat()
            self._save(history)
            decision_id = entry["id"]

        log.info(
            f"[LearningAgent] Decision #{decision_id} ({symbol}) updated via fallback: "
            f"{validated_result} | {pnl_pips} pips"
        )
        return decision_id

    @staticmethod
    def _validate_result(result: str) -> str:
        try:
            return TradeResult(result).value
        except ValueError as exc:
            raise ValueError(
                f"Invalid trade result {result!r} — must be one of "
                f"{[r.value for r in TradeResult]}"
            ) from exc

    # ------------------------------------------------------------------
    # Storage internals — callers must hold self._lock
    # ------------------------------------------------------------------
    def _load(self) -> list:
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            # A silent `except Exception: return []` here would quietly
            # discard the whole history on any corruption without a trace.
            # Instead: back the corrupted file up for forensics/manual
            # recovery, log loudly, and start fresh so the process doesn't
            # crash-loop on a bad file.
            corrupt_backup = f"{self.path}.corrupt.{int(datetime.now(timezone.utc).timestamp())}"
            try:
                os.replace(self.path, corrupt_backup)
                log.error(
                    f"[LearningAgent] {self.path} is corrupted ({exc}); "
                    f"backed up to {corrupt_backup} and starting with empty history."
                )
            except OSError as backup_exc:
                log.error(
                    f"[LearningAgent] {self.path} is corrupted ({exc}) and could not be "
                    f"backed up ({backup_exc}). Starting with empty history."
                )
            return []

    def _save(self, data: list) -> None:
        """Atomic write: temp file + os.replace. Truncates to the last
        `max_history` entries (oldest entries are dropped, not archived —
        see class docstring)."""
        data_to_save = data[-self.max_history:]
        dir_name = os.path.dirname(self.path) or "."
        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=dir_name, suffix=".tmp",
                prefix="learning_", delete=False, encoding="utf-8",
            ) as tmp_f:
                json.dump(data_to_save, tmp_f, indent=2)
                tmp_path = tmp_f.name
            os.replace(tmp_path, self.path)
        except Exception:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            raise