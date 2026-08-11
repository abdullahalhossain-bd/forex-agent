"""
memory/knowledge_store.py — simple key/value JSON-backed knowledge store.

Reconstructed from production caller contracts. The original implementation
existed on the production machine but was never committed to git.

Behavioral contract (verified from caller sites):
  - __init__(): no arguments required.
  - add_memory(text: str, metadata: dict|None=None) -> None: appends a
    text entry with metadata to the knowledge base. Used by
    backtest/engine.py:444 and research/research_agent.py:608.

Both callers wrap the call in try/except — failures are non-fatal.

Persistence:
  - File: memory/knowledge_store.json
  - Format: list of {text, metadata, timestamp}
  - Atomic writes via tempfile + os.replace.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.constants import MEMORY_DIR
from utils.logger import get_logger

log = get_logger("knowledge_store")

_STORE_PATH: Path = MEMORY_DIR / "knowledge_store.json"


class KnowledgeStore:
    """Simple JSON-backed knowledge store."""

    def __init__(self):
        self._path = _STORE_PATH
        self._lock = threading.Lock()

    def _load(self) -> List[Dict[str, Any]]:
        if not self._path.exists():
            return []
        try:
            with open(self._path, "r") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def add_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Append a text entry with metadata. Non-fatal."""
        try:
            with self._lock:
                records = self._load()
                entry = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "text": text,
                    "metadata": metadata or {},
                }
                records.append(entry)
                self._save(records)
        except Exception as e:
            log.warning("[KnowledgeStore] add_memory failed: %s", e)

    def _save(self, records: List[Dict[str, Any]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=str(self._path.parent), suffix=".tmp",
                prefix="knowledge_store_", delete=False
            ) as tmp_f:
                json.dump(records, tmp_f, indent=2, default=str)
                tmp_path = tmp_f.name
            os.replace(tmp_path, self._path)
        except Exception:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            raise
