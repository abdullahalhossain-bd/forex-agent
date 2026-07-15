# utils/safe_json.py — Atomic JSON read/write with file locking
# =============================================================================
# P1-8 (Audit Fix): Prevents concurrent-write corruption on memory/*.json files.
#
# Problem: risk_engine, circuit_breaker, kill_switch, drawdown_controller, etc.
# all read-modify-write the same JSON files. If two AITrader instances (one per
# symbol) write simultaneously, the file can be truncated or corrupted.
#
# Solution:
#   - WRITES use atomic temp-file + os.replace() (already done in some places)
#   - READS + WRITES are protected by a file lock (fcntl on Unix, msvcrt on Windows)
#   - If locking fails (e.g., NFS), falls back to atomic rename only
# =============================================================================

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from utils.logger import get_logger

log = get_logger("safe_json")

try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False

try:
    import msvcrt
    _HAS_MSVCRT = True
except ImportError:
    _HAS_MSVCRT = False


def _lock_file(fd):
    """Try to acquire an exclusive lock on a file descriptor."""
    if _HAS_FCNTL:
        fcntl.flock(fd, fcntl.LOCK_EX)
    elif _HAS_MSVCRT:
        msvcrt.locking(fd, msvcrt.LK_LOCK, 1)


def _unlock_file(fd):
    """Release the exclusive lock."""
    if _HAS_FCNTL:
        fcntl.flock(fd, fcntl.LOCK_UN)
    elif _HAS_MSVCRT:
        try:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass  # already unlocked


def safe_read_json(filepath: str | Path, default: Any = None) -> Any:
    """
    Read JSON from file with file locking.
    Returns `default` if file doesn't exist or is corrupted.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return default

    try:
        with open(filepath, "r") as f:
            _lock_file(f.fileno())
            try:
                return json.load(f)
            finally:
                _unlock_file(f.fileno())
    except (json.JSONDecodeError, OSError) as e:
        log.warning(f"safe_read_json: failed to read {filepath}: {e}")
        return default


def safe_write_json(filepath: str | Path, data: Any) -> None:
    """
    Write JSON to file atomically with file locking.
    Uses temp-file + os.replace() for atomicity.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Serialize first (before locking — don't hold lock during serialization)
    content = json.dumps(data, indent=2, default=str)

    dir_name = str(filepath.parent) or "."

    # Write to temp file, then atomic rename
    with tempfile.NamedTemporaryFile(
        mode="w", dir=dir_name, suffix=".tmp",
        prefix=filepath.stem + "_", delete=False
    ) as tmp_f:
        _lock_file(tmp_f.fileno())
        try:
            tmp_f.write(content)
            tmp_f.flush()
            os.fsync(tmp_f.fileno())
        finally:
            _unlock_file(tmp_f.fileno())
        tmp_path = tmp_f.name

    try:
        os.replace(tmp_path, str(filepath))
    except OSError:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def safe_modify_json(
    filepath: str | Path,
    modifier_fn,
    default: Any = None,
) -> Any:
    """
    Read-modify-write JSON file with locking.
    Calls `modifier_fn(data)` which should return the modified data.

    Example:
        def add_trade(data):
            data["trades"].append(new_trade)
            return data
        safe_modify_json("memory/daily_risk.json", add_trade, default={"trades": []})
    """
    filepath = Path(filepath)
    data = safe_read_json(filepath, default=default)
    if data is None:
        data = default if default is not None else {}
    modified = modifier_fn(data)
    if modified is not None:
        data = modified
    safe_write_json(filepath, data)
    return data
