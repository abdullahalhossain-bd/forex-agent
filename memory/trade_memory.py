"""
memory/trade_memory.py — JSON-backed trade signal & outcome memory.

Reconstructed from production caller contracts. The original implementation
existed on the production machine but was never committed to git.

Behavioral contract (verified from caller sites):
  - __init__(seed_rules: bool=True): load memory/trade_memory.json; if
    seed_rules=True and the file is empty, seed it with default rule
    entries (matches log message: "[TradeMemory] Trading rules seeding
    completed"). Always logs a warning when vector memory (sentence-
    transformers) is unavailable — matches historical log: "[TradeMemory]
    Running without vector memory (embedding model unavailable)".
  - get_context_for_ai(symbol) -> dict: returns a small context dict
    summarizing recent trades for the symbol. Used by AITrader to enrich
    LLM prompts. Never raises.
  - get_pattern_context(symbol, regime, pattern) -> dict: returns
    win/loss stats for past occurrences of (symbol, regime, pattern).
    Never raises.
  - on_signal_generated(result, market_out, analysis_out) -> int|None:
    persists a new entry, returns its 1-indexed id (or None on failure).
  - on_trade_closed(trade_id, result, pnl) -> None: updates the entry's
    outcome/result/pnl_pips fields.
  - print_stats() -> None: prints a one-line summary to logger.
  - .db: a database.db.TraderDB instance (lazily attached by AITrader
    during its __init__ — see core/trader.py:337). The original TradeMemory
    held a reference to TraderDB for orphan-trade reconciliation. We
    expose `.db` as None by default and let AITrader set it.

Persistence:
  - File: memory/trade_memory.json (path: core.constants.TRADE_MEMORY_PATH)
  - Format: list of dicts (see backups/*/trade_memory.json for schema).
  - Atomic writes via tempfile + os.replace (matches KillSwitch pattern).

Vector memory:
  - The original implementation used sentence-transformers to embed
    trade summaries for semantic search. We DO NOT require this — the
    production logs show it ran WITHOUT vector memory ("embedding model
    unavailable"). We attempt to import sentence_transformers and degrade
    gracefully.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.constants import TRADE_MEMORY_PATH
from utils.logger import get_logger

log = get_logger("trade_memory")


class TradeMemory:
    """JSON-backed signal/trade memory with optional vector search."""

    def __init__(self, seed_rules: bool = True):
        self._lock = threading.RLock()
        self._path: Path = TRADE_MEMORY_PATH
        self._records: List[Dict[str, Any]] = []
        self._model = None  # sentence-transformers model (optional)
        self._embeddings: List[Any] = []  # cached embeddings (parallel to _lessons)
        self._lessons: List[Dict[str, Any]] = []  # parallel to _embeddings
        self.db = None  # AITrader attaches a TraderDB after construction
        # BUGFIX (2026-08-11 audit): learning/mistake_analyzer.py calls
        # `self.memory.pattern.add_losing_pattern(...)` /
        # `add_winning_pattern(...)` on every closed trade — this attribute
        # didn't exist on the original reconstruction (missed caller
        # contract), so both calls raised AttributeError every time,
        # silently swallowed by mistake_analyzer's own try/except. The
        # pattern-learning half of the self-learning loop has never
        # actually persisted anything as a result. Wiring up a real
        # PatternMemory fixes that; see class below.
        self.pattern = PatternMemory(self._path.parent / "pattern_memory.json")

        # Load existing state
        self._load()
        self._load_vector_lessons()

        # Optional vector memory — historical logs show this was ALWAYS
        # unavailable in production ("embedding model unavailable"), so
        # the warning is the expected path.
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            log.info("[TradeMemory] Vector memory initialized (sentence-transformers)")
        except Exception:
            log.info(
                "[TradeMemory] Running without vector memory — sentence-transformers not installed; "
                "semantic search disabled. Install with pip install -U sentence-transformers to enable."
            )

        # Seed default rules if requested and memory is empty
        if seed_rules and not self._records:
            self._seed_default_rules()
            log.info("[TradeMemory] Trading rules seeding completed")

    # ── Persistence ──────────────────────────────────────────────

    def _load(self) -> None:
        """Load records from disk. On corruption, start empty (fail-safe)."""
        with self._lock:
            if not self._path.exists():
                self._records = []
                return
            try:
                with open(self._path, "r") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self._records = data
                else:
                    log.warning("[TradeMemory] State file is not a list — starting empty")
                    self._records = []
            except Exception as e:
                log.warning("[TradeMemory] Failed to load state: %s — starting empty", e)
                self._records = []

    def _save(self) -> None:
        """Atomic write to disk (tempfile + os.replace)."""
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", dir=str(self._path.parent), suffix=".tmp",
                    prefix="trade_memory_", delete=False
                ) as tmp_f:
                    json.dump(self._records, tmp_f, indent=2, default=str)
                    tmp_path = tmp_f.name
                os.replace(tmp_path, self._path)
            except Exception:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                raise

    def _seed_default_rules(self) -> None:
        """Seed minimal default rules so the file isn't empty.

        Historical logs show "Trading rules seeding completed" — we don't
        know the exact rules, but the schema is fixed (see backups). We
        add a single sentinel entry so the file is non-empty.
        """
        sentinel = {
            "id": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": "SYSTEM",
            "timeframe": "—",
            "decision": "WAIT",
            "raw_signal": "NO TRADE",
            "gated": True,
            "confidence": 0,
            "entry": None,
            "sl": None,
            "tp": None,
            "lot": 0,
            "rr": 0,
            "regime": "—",
            "trend": "—",
            "rsi": 0,
            "patterns": [],
            "rule_signal": "WAIT",
            "llm_signal": "WAIT",
            "reasons": ["Sentinel entry — seeded by TradeMemory"],
            "outcome": None,
            "pnl_pips": None,
            "result": None,
        }
        self._records = [sentinel]
        try:
            self._save()
        except Exception as e:
            log.warning("[TradeMemory] Failed to seed rules: %s", e)

    def _load_vector_lessons(self) -> None:
        """Load persisted lesson texts and re-embed them if the model is
        available. We persist text/metadata, not raw embedding vectors —
        smaller JSON, and re-embedding a few hundred short strings at boot
        is cheap compared to shipping numpy arrays through JSON.
        """
        vpath = self._path.parent / "vector_lessons.json"
        try:
            if vpath.exists():
                with open(vpath, "r") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self._lessons = data
        except Exception as e:
            log.warning("[TradeMemory] Failed to load vector lessons: %s", e)
            self._lessons = []

        if self._model is not None and self._lessons:
            try:
                texts = [entry.get("text", "") for entry in self._lessons]
                self._embeddings = list(self._model.encode(texts))
            except Exception as e:
                log.warning("[TradeMemory] Failed to re-embed lessons: %s", e)
                self._embeddings = []

    def _save_vector_lessons(self) -> None:
        vpath = self._path.parent / "vector_lessons.json"
        try:
            vpath.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = None
            with tempfile.NamedTemporaryFile(
                mode="w", dir=str(vpath.parent), suffix=".tmp",
                prefix="vector_lessons_", delete=False
            ) as tmp_f:
                json.dump(self._lessons, tmp_f, indent=2, default=str)
                tmp_path = tmp_f.name
            os.replace(tmp_path, vpath)
        except Exception as e:
            log.warning("[TradeMemory] Failed to save vector lessons: %s", e)

    def add_vector_lesson(self, text: str, pair: str = "") -> None:
        """Store a free-text lesson (e.g. an LLM-generated loss/win
        analysis) for later semantic recall via find_similar(). No-op if
        the embedding model isn't available — the lesson text itself is
        still persisted so it isn't lost, just not semantically searchable
        until sentence-transformers is installed.
        """
        with self._lock:
            entry = {
                "text": text,
                "pair": pair,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._lessons.append(entry)
            if self._model is not None:
                try:
                    self._embeddings.append(self._model.encode(text))
                except Exception as e:
                    log.warning("[TradeMemory] add_vector_lesson embed failed: %s", e)
            self._save_vector_lessons()

    def find_similar(self, query: str, limit: int = 2) -> List[Dict[str, Any]]:
        """Return up to `limit` past lessons most semantically similar to
        `query`, most-similar first. Returns [] if the embedding model
        isn't available or no lessons have been stored yet — callers
        already treat an empty result as "no similar memory found", so
        this degrades gracefully rather than raising.
        """
        if self._model is None or not self._embeddings:
            return []
        try:
            import numpy as np
            q = np.asarray(self._model.encode(query), dtype=float)
            q_norm = np.linalg.norm(q) or 1.0
            scored = []
            for entry, emb in zip(self._lessons, self._embeddings):
                v = np.asarray(emb, dtype=float)
                v_norm = np.linalg.norm(v) or 1.0
                sim = float(np.dot(q, v) / (q_norm * v_norm))
                scored.append((sim, entry))
            scored.sort(key=lambda t: t[0], reverse=True)
            return [
                {"memory": entry["text"], "pair": entry.get("pair", ""), "score": sim}
                for sim, entry in scored[:limit]
            ]
        except Exception as e:
            log.warning("[TradeMemory] find_similar failed: %s", e)
            return []

    # ── Public API ───────────────────────────────────────────────

    def get_context_for_ai(self, symbol: str) -> Dict[str, Any]:
        """Return a small context dict summarizing recent trades for `symbol`.

        Used to enrich LLM prompts AND by AITrader.evaluate_decision_core
        (core/trader.py:907-931) which expects the keys:
          - total_trades: int  (total decisions for this symbol)
          - overall_win_rate: float  (0-100, percentage)
        Never raises — returns a safe dict on any error.
        """
        try:
            with self._lock:
                symbol_records = [r for r in self._records if r.get("symbol") == symbol]
                recent = symbol_records[-20:]
                wins = sum(1 for r in symbol_records if r.get("result") == "WIN")
                losses = sum(1 for r in symbol_records if r.get("result") == "LOSS")
                breakeven = sum(1 for r in symbol_records if r.get("result") == "BREAKEVEN")
                closed = wins + losses + breakeven
                total = len(symbol_records)
                wr = (wins / closed * 100.0) if closed else 0.0
                return {
                    "symbol": symbol,
                    "total_trades": total,
                    "overall_win_rate": wr,
                    "wins": wins,
                    "losses": losses,
                    "breakeven": breakeven,
                    "recent_count": len(recent),
                    "win_rate": wr / 100.0 if closed else 0.0,
                    "last_decision": recent[-1].get("decision") if recent else None,
                    "last_result": recent[-1].get("result") if recent else None,
                }
        except Exception:
            # AITrader accesses memory_ctx["total_trades"] — must always
            # return a dict with that key, even on failure.
            return {"total_trades": 0, "overall_win_rate": 0.0}

    def get_pattern_context(self, *args, **kwargs) -> Dict[str, Any]:
        """Return win/loss stats for past occurrences of (symbol, regime, pattern).

        Supports TWO call signatures (both used in core/trader.py):
          1. get_pattern_context(symbol, regime, pattern) — positional
          2. get_pattern_context(pair=, trend=, rsi=, pattern=, regime=) — kwargs

        Returns a dict with keys expected by AITrader:
          - similar_wins: int
          - similar_losses: int
          - warning: bool (True if losses > wins)
          - occurrences: int
          - win_rate: float (0-1)
        Never raises.
        """
        try:
            # Normalize args
            if kwargs:
                symbol = kwargs.get("pair") or kwargs.get("symbol") or ""
                regime = kwargs.get("regime") or ""
                pattern = kwargs.get("pattern") or ""
            elif len(args) >= 3:
                symbol, regime, pattern = args[0], args[1], args[2]
            else:
                return {"similar_wins": 0, "similar_losses": 0, "warning": False}

            with self._lock:
                matches = [
                    r for r in self._records
                    if r.get("symbol") == symbol
                    and (not regime or r.get("regime") == regime)
                    and pattern in (r.get("patterns") or [])
                    and r.get("result") is not None
                ]
                wins = sum(1 for r in matches if r.get("result") == "WIN")
                losses = sum(1 for r in matches if r.get("result") == "LOSS")
                return {
                    "symbol": symbol,
                    "regime": regime,
                    "pattern": pattern,
                    "occurrences": len(matches),
                    "similar_wins": wins,
                    "similar_losses": losses,
                    "wins": wins,
                    "losses": losses,
                    "win_rate": wins / len(matches) if matches else 0.0,
                    "warning": losses > wins and len(matches) >= 3,
                }
        except Exception:
            return {"similar_wins": 0, "similar_losses": 0, "warning": False}

    def on_signal_generated(self, result: Dict[str, Any],
                             market_out: Dict[str, Any],
                             analysis_out: Dict[str, Any]) -> Optional[int]:
        """Persist a new signal entry. Returns its 1-indexed id, or None."""
        try:
            with self._lock:
                next_id = (max((r.get("id", 0) for r in self._records), default=0) + 1)
                entry = {
                    "id": next_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "symbol": result.get("symbol") or market_out.get("symbol", "—"),
                    "timeframe": result.get("timeframe") or market_out.get("timeframe", "—"),
                    "decision": result.get("decision", "WAIT"),
                    "raw_signal": result.get("raw_signal", "NO TRADE"),
                    "gated": bool(result.get("gated", False)),
                    "confidence": float(result.get("confidence", 0) or 0),
                    "entry": result.get("entry"),
                    "sl": result.get("sl"),
                    "tp": result.get("tp"),
                    "lot": float(result.get("lot", 0) or 0),
                    "rr": float(result.get("rr", 0) or 0),
                    "regime": (analysis_out or {}).get("regime", "—"),
                    "trend": (analysis_out or {}).get("trend", "—"),
                    "rsi": float((analysis_out or {}).get("rsi", 0) or 0),
                    "patterns": list((analysis_out or {}).get("patterns", []) or []),
                    "rule_signal": result.get("rule_signal", "WAIT"),
                    "llm_signal": result.get("llm_signal", "WAIT"),
                    "reasons": list(result.get("reasons", []) or []),
                    "outcome": None,
                    "pnl_pips": None,
                    "result": None,
                }
                self._records.append(entry)
                self._save()
                return next_id
        except Exception as e:
            log.warning("[TradeMemory] on_signal_generated failed: %s", e)
            return None

    def on_trade_closed(self, trade_id: int, result: str, pnl: float) -> None:
        """Update an entry's outcome. Idempotent — missing id is a warning."""
        try:
            with self._lock:
                for r in self._records:
                    if r.get("id") == trade_id:
                        r["result"] = result
                        r["pnl_pips"] = float(pnl or 0)
                        r["outcome"] = "closed"
                        self._save()
                        return
                log.warning("[TradeMemory] on_trade_closed: id %s not found", trade_id)
        except Exception as e:
            log.warning("[TradeMemory] on_trade_closed failed: %s", e)

    def print_stats(self) -> None:
        """Print a one-line summary to logger."""
        try:
            with self._lock:
                total = len(self._records)
                wins = sum(1 for r in self._records if r.get("result") == "WIN")
                losses = sum(1 for r in self._records if r.get("result") == "LOSS")
                open_trades = sum(1 for r in self._records
                                  if r.get("outcome") is None and r.get("decision") in ("BUY", "SELL"))
                wr = (wins / (wins + losses) * 100) if (wins + losses) else 0.0
                log.info(
                    "[TradeMemory] %d decisions | %d open | WR: %.1f%% (%dW/%dL)",
                    total, open_trades, wr, wins, losses
                )
        except Exception as e:
            log.warning("[TradeMemory] print_stats failed: %s", e)


class PatternMemory:
    """JSON-backed pattern win/loss memory.

    Reconstructed 2026-08-12 to fix the NameError that prevented
    TradingEngine from initializing (memory/trade_memory.py line 78
    referenced PatternMemory but no class was defined).

    Behavioral contract (from learning/mistake_analyzer.py caller):
      - __init__(path: Path): load pattern_memory.json (empty list if missing)
      - add_losing_pattern(symbol, regime, pattern, pnl_pips) -> None
      - add_winning_pattern(symbol, regime, pattern, pnl_pips) -> None
      - get_stats(symbol, regime, pattern) -> dict with wins/losses/total/win_rate

    Persistence:
      - File: memory/pattern_memory.json
      - Format: list of dicts with keys: symbol, regime, pattern,
        result (WIN/LOSS), pnl_pips, timestamp
      - Atomic writes via tempfile + os.replace
    """

    def __init__(self, path: Path):
        self._lock = threading.RLock()
        self._path: Path = Path(path)
        self._records: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        """Load pattern memory from JSON file."""
        try:
            if self._path.exists():
                with open(self._path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self._records = data
                    else:
                        log.warning(
                            "[PatternMemory] Unexpected JSON structure in %s, starting fresh",
                            self._path,
                        )
        except Exception as e:
            log.warning("[PatternMemory] Failed to load %s: %s", self._path, e)
            self._records = []

    def _save(self) -> None:
        """Persist pattern memory to JSON file atomically."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=str(self._path.parent),
                suffix=".tmp",
                delete=False,
                encoding="utf-8",
            ) as tmp:
                json.dump(self._records, tmp, indent=2, default=str)
                tmp_path = tmp.name
            os.replace(tmp_path, str(self._path))
        except Exception as e:
            log.warning("[PatternMemory] Failed to save %s: %s", self._path, e)

    def add_losing_pattern(
        self, symbol: str, regime: str, pattern: str, pnl_pips: float = 0.0
    ) -> None:
        """Record a losing pattern occurrence."""
        try:
            with self._lock:
                self._records.append(
                    {
                        "symbol": symbol,
                        "regime": regime,
                        "pattern": pattern,
                        "result": "LOSS",
                        "pnl_pips": float(pnl_pips),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
                self._save()
        except Exception as e:
            log.warning("[PatternMemory] add_losing_pattern failed: %s", e)

    def add_winning_pattern(
        self, symbol: str, regime: str, pattern: str, pnl_pips: float = 0.0
    ) -> None:
        """Record a winning pattern occurrence."""
        try:
            with self._lock:
                self._records.append(
                    {
                        "symbol": symbol,
                        "regime": regime,
                        "pattern": pattern,
                        "result": "WIN",
                        "pnl_pips": float(pnl_pips),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
                self._save()
        except Exception as e:
            log.warning("[PatternMemory] add_winning_pattern failed: %s", e)

    def get_stats(self, symbol: str, regime: str, pattern: str) -> Dict[str, Any]:
        """Return win/loss stats for a (symbol, regime, pattern) tuple."""
        try:
            with self._lock:
                matching = [
                    r
                    for r in self._records
                    if r.get("symbol") == symbol
                    and r.get("regime") == regime
                    and r.get("pattern") == pattern
                ]
                wins = sum(1 for r in matching if r.get("result") == "WIN")
                losses = sum(1 for r in matching if r.get("result") == "LOSS")
                total = len(matching)
                win_rate = (wins / total * 100) if total else 0.0
                avg_pnl = (
                    sum(r.get("pnl_pips", 0) for r in matching) / total
                    if total
                    else 0.0
                )
                return {
                    "total": total,
                    "wins": wins,
                    "losses": losses,
                    "win_rate": round(win_rate, 2),
                    "avg_pnl_pips": round(avg_pnl, 2),
                }
        except Exception as e:
            log.warning("[PatternMemory] get_stats failed: %s", e)
            return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "avg_pnl_pips": 0.0}

    def print_stats(self) -> None:
        """Print a one-line summary to logger."""
        try:
            with self._lock:
                total = len(self._records)
                wins = sum(1 for r in self._records if r.get("result") == "WIN")
                losses = sum(1 for r in self._records if r.get("result") == "LOSS")
                wr = (wins / (wins + losses) * 100) if (wins + losses) else 0.0
                log.info(
                    "[PatternMemory] %d patterns | WR: %.1f%% (%dW/%dL)",
                    total, wr, wins, losses,
                )
        except Exception as e:
            log.warning("[PatternMemory] print_stats failed: %s", e)