"""
backtest/persistence.py — Atomic persistence primitives for the Phase 3 runner.

Provides:
  - atomic_write_json(path, obj): write JSON via temp file + atomic rename
  - append_jsonl(path, obj): append a single JSON record as a line
  - read_jsonl(path): read all records from a JSONL file
  - read_jsonl_ids(path): read just the trade_id / record_id fields (for dedup)
  - serialize_trade(trade): SimulatedTrade → JSON-safe dict
  - deserialize_trade(d): JSON-safe dict → SimulatedTrade
  - RunDir: encapsulates the backtest_runs/<run_id>/ directory layout

These primitives are used by the persistent_runner to checkpoint state,
persist trades, and resume after interrupt.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from backtest.broker_sim import SimulatedTrade


# ── Atomic write ──────────────────────────────────────────────────────────

def atomic_write_json(path: Path | str, obj: Any, indent: int = 2) -> None:
    """Write obj as JSON to path via temp file + atomic rename.

    Guarantees that path is NEVER in a corrupted state — either the old
    version exists, or the new version exists, but never a half-written file.

    Uses os.replace() which is atomic on POSIX and Windows for same-filesystem
    renames.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file in the SAME directory (so os.replace is atomic)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=indent, default=str, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        # Cleanup temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def read_json(path: Path | str) -> Any:
    """Read JSON from path. Returns None if file doesn't exist or is corrupted."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# ── JSONL append/read ─────────────────────────────────────────────────────

def append_jsonl(path: Path | str, obj: Any) -> None:
    """Append a single JSON record as one line to a JSONL file.

    Each call opens, writes one line, flushes, fsyncs, closes — so the
    record is durable immediately after the call returns. This is slower
    than buffering but guarantees no data loss on crash.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, default=str, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def append_jsonl_batch(path: Path | str, objs: Iterable[Any]) -> int:
    """Append multiple records to a JSONL file in one open/write/fsync/close.

    More efficient than append_jsonl when many records need to be written
    at once (e.g. flush every K bars). Returns the number of records written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "a", encoding="utf-8") as f:
        for obj in objs:
            line = json.dumps(obj, default=str, ensure_ascii=False)
            f.write(line + "\n")
            count += 1
        f.flush()
        os.fsync(f.fileno())
    return count


def read_jsonl(path: Path | str) -> Iterator[dict]:
    """Yield each JSON record from a JSONL file. Skips malformed lines."""
    path = Path(path)
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def read_jsonl_count(path: Path | str) -> int:
    """Count lines in a JSONL file (fast — doesn't parse JSON)."""
    path = Path(path)
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def read_jsonl_ids(path: Path | str, id_field: str = "trade_id") -> set:
    """Read just the id_field values from a JSONL file (for dedup checks).

    Faster than read_jsonl + extract since we still parse JSON, but only
    keep the id field in memory.
    """
    ids = set()
    path = Path(path)
    if not path.exists():
        return ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if id_field in rec:
                    ids.add(rec[id_field])
            except json.JSONDecodeError:
                continue
    return ids


# ── Trade serialization ──────────────────────────────────────────────────

def serialize_trade(trade: SimulatedTrade) -> dict:
    """Convert a SimulatedTrade to a JSON-safe dict.

    Includes a `_type: "SimulatedTrade"` marker so deserialize_trade
    can reconstruct it later.
    """
    d = trade.to_dict()
    d["_type"] = "SimulatedTrade"
    # Ensure all values are JSON-safe (timestamps → strings, etc.)
    for k, v in list(d.items()):
        if isinstance(v, datetime):
            d[k] = v.isoformat()
        elif hasattr(v, "isoformat"):  # pd.Timestamp etc.
            d[k] = str(v)
    return d


def deserialize_trade(d: dict) -> SimulatedTrade:
    """Reconstruct a SimulatedTrade from a serialized dict.

    Strips the _type marker and passes the rest to the SimulatedTrade
    constructor. Tolerant of missing/extra fields.
    """
    d = {k: v for k, v in d.items() if k != "_type"}
    # SimulatedTrade is a dataclass — only pass fields it actually has
    valid_fields = {f.name for f in SimulatedTrade.__dataclass_fields__.values()}
    filtered = {k: v for k, v in d.items() if k in valid_fields}
    return SimulatedTrade(**filtered)


# ── Run directory layout ─────────────────────────────────────────────────

class RunDir:
    """Encapsulates the backtest_runs/<run_id>/ directory layout.

    Layout:
        backtest_runs/<run_id>/
            config.json          # immutable run config
            checkpoint.json      # latest checkpoint (atomic writes)
            summary.json         # latest summary stats (atomic writes, updated every K bars)
            trades/
                <symbol>.jsonl   # one line per closed trade, per symbol
            losses/
                <symbol>.jsonl   # subset of trades where pnl_usd < 0
            llm_analysis/
                <symbol>.jsonl   # LLM analysis output (async, may lag)
                queue.jsonl      # losses awaiting LLM analysis
            logs/
                <symbol>.log     # per-symbol verbose log (if verbose)
            metrics.json         # final metrics (only at end of run)
    """

    def __init__(self, run_id: str, root: Path | str | None = None):
        self.run_id = run_id
        if root is None:
            try:
                from config import PROJECT_ROOT
                root = Path(PROJECT_ROOT) / "backtest_runs"
            except Exception:
                root = Path(__file__).resolve().parents[1] / "backtest_runs"
        self.root = Path(root) / run_id
        self.trades_dir = self.root / "trades"
        self.losses_dir = self.root / "losses"
        self.llm_dir = self.root / "llm_analysis"
        self.logs_dir = self.root / "logs"

    def mkdirs(self) -> None:
        """Create all subdirectories."""
        for d in (self.root, self.trades_dir, self.losses_dir,
                  self.llm_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ── File paths ──────────────────────────────────────────────────────

    @property
    def config_path(self) -> Path:
        return self.root / "config.json"

    @property
    def checkpoint_path(self) -> Path:
        return self.root / "checkpoint.json"

    @property
    def summary_path(self) -> Path:
        return self.root / "summary.json"

    @property
    def metrics_path(self) -> Path:
        return self.root / "metrics.json"

    @property
    def llm_queue_path(self) -> Path:
        return self.llm_dir / "queue.jsonl"

    @property
    def llm_summary_path(self) -> Path:
        return self.llm_dir / "loss_summary.json"

    def trades_path(self, symbol: str) -> Path:
        return self.trades_dir / f"{symbol}.jsonl"

    def losses_path(self, symbol: str) -> Path:
        return self.losses_dir / f"{symbol}.jsonl"

    def llm_analysis_path(self, symbol: str) -> Path:
        return self.llm_dir / f"{symbol}.jsonl"

    def log_path(self, symbol: str) -> Path:
        return self.logs_dir / f"{symbol}.log"

    # ── Config ──────────────────────────────────────────────────────────

    def write_config(self, config: dict) -> None:
        atomic_write_json(self.config_path, config)

    def read_config(self) -> dict | None:
        return read_json(self.config_path)

    # ── Checkpoint ──────────────────────────────────────────────────────

    def write_checkpoint(self, checkpoint: dict) -> None:
        """Atomically write the latest checkpoint."""
        checkpoint["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        atomic_write_json(self.checkpoint_path, checkpoint)

    def read_checkpoint(self) -> dict | None:
        """Read the latest checkpoint. Returns None if not found or corrupted."""
        return read_json(self.checkpoint_path)

    # ── Summary ─────────────────────────────────────────────────────────

    def write_summary(self, summary: dict) -> None:
        """Atomically write the latest summary stats."""
        summary["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        atomic_write_json(self.summary_path, summary)

    def read_summary(self) -> dict | None:
        return read_json(self.summary_path)

    # ── Trades ──────────────────────────────────────────────────────────

    def append_trade(self, symbol: str, trade_dict: dict) -> None:
        """Append a closed trade to the per-symbol JSONL file."""
        append_jsonl(self.trades_path(symbol), trade_dict)

    def append_loss(self, symbol: str, trade_dict: dict) -> None:
        """Append a loss trade to the per-symbol losses JSONL file."""
        append_jsonl(self.losses_path(symbol), trade_dict)

    def read_trades(self, symbol: str) -> Iterator[dict]:
        """Yield all closed trades for a symbol."""
        yield from read_jsonl(self.trades_path(symbol))

    def count_trades(self, symbol: str) -> int:
        """Count closed trades for a symbol (fast line count)."""
        return read_jsonl_count(self.trades_path(symbol))

    def existing_trade_ids(self, symbol: str) -> set:
        """Return set of trade_ids already persisted for a symbol."""
        return read_jsonl_ids(self.trades_path(symbol), "trade_id")

    # ── LLM queue ───────────────────────────────────────────────────────

    def append_llm_queue(self, loss_record: dict) -> None:
        """Append a loss record to the LLM analysis queue."""
        append_jsonl(self.llm_queue_path, loss_record)

    def read_llm_queue(self) -> Iterator[dict]:
        yield from read_jsonl(self.llm_queue_path)

    def append_llm_analysis(self, symbol: str, analysis: dict) -> None:
        append_jsonl(self.llm_analysis_path(symbol), analysis)

    def read_llm_analyses(self, symbol: str) -> Iterator[dict]:
        yield from read_jsonl(self.llm_analysis_path(symbol))
