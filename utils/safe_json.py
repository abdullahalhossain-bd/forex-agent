"""Lightweight JSON I/O helpers with atomic writes and file locking.

These helpers are intentionally small and dependency-light so the trading
stack can use them even in minimal environments.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def safe_read_json(path: str | os.PathLike[str] | Path) -> Any:
    """Read JSON from disk, returning ``None`` for missing files."""
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_write_json(path: str | os.PathLike[str] | Path, payload: Any) -> None:
    """Atomically write JSON data to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent)) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
        temp_name = handle.name
    os.replace(temp_name, path)


__all__ = ["safe_read_json", "safe_write_json"]
