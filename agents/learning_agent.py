# agents/learning_agent.py  —  Day 12 | Self-Learning Agent

import json
import os
from datetime import datetime
from utils.logger import get_logger

log  = get_logger("learning_agent")
PATH = "memory/trade_memory.json"


class LearningAgent:
    """
    প্রতিটা decision save করে।
    ভবিষ্যতে outcome জানলে শিখবে।
    Pattern performance track করবে।
    """

    def save_decision(
        self,
        decision_out:  dict,
        analysis_out:  dict,
        market_out:    dict,
    ) -> int:
        """Persist a decision entry to memory/trade_memory.json.

        Day 102+ hotfix: returns the new entry's id (was None) so the
        caller can stash it on the trade context dict. When the trade
        eventually closes, the close handler can call
        update_outcome(decision_id, ...) instead of falling back to the
        symbol-based search.

        Day 102+ hotfix #2: STABLE MONOTONIC IDS. Previously id was
        computed as `len(history) + 1`, which breaks when _save()
        truncates history to last 500 entries. After truncation, the
        next save_decision would compute id=501 (not 601), colliding
        with the id of an older entry that was just truncated. Any
        external reference to the truncated id (e.g., stashed on a
        trade context dict waiting for the close event) would now
        point to a different trade. Fix: id = max(existing ids) + 1,
        so ids only ever increase, even across truncation events.
        """
        os.makedirs("memory", exist_ok=True)
        history = self._load()

        # Stable monotonic id — survives truncation
        next_id = 1
        if history:
            existing_ids = [e.get("id", 0) for e in history if isinstance(e.get("id"), int)]
            if existing_ids:
                next_id = max(existing_ids) + 1

        entry = {
            "id":          next_id,
            "timestamp":   datetime.utcnow().isoformat(),
            "symbol":      market_out.get("symbol"),
            "timeframe":   market_out.get("timeframe"),
            "decision":    decision_out.get("decision"),
            "raw_signal":  decision_out.get("raw_signal"),
            "gated":       decision_out.get("gated_by_permission", False),
            "confidence":  decision_out.get("confidence"),
            "entry":       decision_out.get("entry"),
            "sl":          decision_out.get("sl"),
            "tp":          decision_out.get("tp"),
            "lot":         decision_out.get("lot"),
            "rr":          decision_out.get("rr"),
            "regime":      market_out.get("regime", {}).get("regime"),
            "trend":       market_out.get("ind_ctx", {}).get("trend"),
            "rsi":         market_out.get("ind_ctx", {}).get("rsi"),
            "patterns":    analysis_out.get("pat_ctx", {}).get("recent_patterns", []),
            "rule_signal": analysis_out.get("signal", {}).get("signal"),
            "llm_signal":  analysis_out.get("llm", {}).get("signal"),
            "reasons":     decision_out.get("reasons", []),
            # outcome পরে update হবে (backtester/live)
            "outcome":     None,
            "pnl_pips":    None,
            "result":      None,   # WIN / LOSS / BE
        }

        history.append(entry)
        self._save(history)
        log.info(f"[LearningAgent] Decision #{entry['id']} saved — {entry['decision']}")
        return entry["id"]

    def get_performance_stats(self) -> dict:
        history = self._load()
        closed  = [t for t in history if t.get("result")]

        if not closed:
            return {"total_decisions": len(history), "closed_trades": 0}

        wins    = [t for t in closed if t["result"] == "WIN"]
        losses  = [t for t in closed if t["result"] == "LOSS"]
        win_rate = round(len(wins) / len(closed) * 100, 1)
        avg_pnl  = round(
            sum(t.get("pnl_pips", 0) for t in closed) / len(closed), 1
        )

        # Pattern performance
        pat_stats = {}
        for t in closed:
            for p in (t.get("patterns") or []):
                if p not in pat_stats:
                    pat_stats[p] = {"win": 0, "loss": 0}
                if t["result"] == "WIN":
                    pat_stats[p]["win"] += 1
                else:
                    pat_stats[p]["loss"] += 1

        return {
            "total_decisions": len(history),
            "closed_trades":   len(closed),
            "wins":            len(wins),
            "losses":          len(losses),
            "win_rate":        win_rate,
            "avg_pnl_pips":    avg_pnl,
            "pattern_stats":   pat_stats,
        }

    # ── Day 102+ hotfix: outcome backfill ───────────────────────
    # Previously, the JSON-only LearningAgent had no way to mark a
    # decision as WIN/LOSS once the trade closed. The close handler
    # in core/trader.py called self._memory.on_trade_closed() (which
    # updates SQLite) but never touched this JSON file. As a result,
    # every entry stayed {"result": null} forever — and
    # get_performance_stats() returned "0 closed" / "WR: N/A".
    #
    # These two methods close that gap:
    #   - update_outcome(decision_id, result, pnl_pips)  → by id
    #   - update_outcome_by_symbol(symbol, result, ...)  → by pair
    #     (used as fallback when the close event lost the id)

    def update_outcome(self, decision_id: int, result: str, pnl_pips: float = 0.0) -> bool:
        """Mark a previously-saved decision as WIN/LOSS/BE.

        Returns True if an entry was found and updated, False otherwise.
        """
        history = self._load()
        updated = False
        for entry in history:
            if entry.get("id") == decision_id:
                entry["result"]   = result
                entry["pnl_pips"] = pnl_pips
                entry["outcome"]  = result  # legacy alias
                entry["closed_at"] = datetime.utcnow().isoformat()
                updated = True
                break
        if updated:
            self._save(history)
            log.info(f"[LearningAgent] Decision #{decision_id} updated: {result} | {pnl_pips} pips")
        else:
            log.warning(f"[LearningAgent] Decision #{decision_id} not found — outcome not saved")
        return updated

    def update_outcome_by_symbol(self, symbol: str, result: str, pnl_pips: float = 0.0) -> int | None:
        """Fallback: mark the most recent OPEN decision for `symbol`.

        Used when the close handler doesn't have the original decision id
        (e.g. trade was opened by a previous process run, or memory_trade_id
        was lost in transit). Returns the decision id that was updated,
        or None if no open decision was found for that symbol.
        """
        history = self._load()
        # Walk backwards to find the most recent entry for this symbol
        # that doesn't yet have a result.
        target_idx = None
        for i in range(len(history) - 1, -1, -1):
            entry = history[i]
            if entry.get("symbol") == symbol and not entry.get("result"):
                target_idx = i
                break

        if target_idx is None:
            return None

        entry = history[target_idx]
        entry["result"]   = result
        entry["pnl_pips"] = pnl_pips
        entry["outcome"]  = result
        entry["closed_at"] = datetime.utcnow().isoformat()
        self._save(history)
        log.info(
            f"[LearningAgent] Decision #{entry['id']} ({symbol}) updated via fallback: "
            f"{result} | {pnl_pips} pips"
        )
        return entry["id"]

    def _load(self) -> list:
        if not os.path.exists(PATH):
            return []
        try:
            with open(PATH) as f:
                return json.load(f)
        except Exception:
            return []

    def _save(self, data: list) -> None:
        """CRITICAL FIX: atomic write using temp file + rename.
        Previous code wrote directly to PATH — if process crashes mid-write,
        the file is corrupted and all learning history is lost.
        """
        import tempfile
        data_to_save = data[-500:]  # শেষ 500টা রাখো
        # Write to temp file first, then atomic rename
        dir_name = os.path.dirname(PATH) or "."
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=dir_name, suffix=".tmp",
                prefix="learning_", delete=False
            ) as tmp_f:
                json.dump(data_to_save, tmp_f, indent=2)
                tmp_path = tmp_f.name
            # Atomic rename (on same filesystem, rename is atomic)
            os.replace(tmp_path, PATH)
        except Exception:
            # Cleanup temp file if rename failed
            try:
                os.unlink(tmp_path)
            except (OSError, UnboundLocalError):
                pass
            raise