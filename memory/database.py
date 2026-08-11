"""
memory/database.py — backward-compat shim for `from memory.database import Database`.

Reconstructed from production caller contracts. The original implementation
existed on the production machine but was never committed to git.

Behavioral contract (verified from caller sites):
  - broker/journal_bridge.py:84: `self._learning_db = Database()` — used
    to log MT5 trade closures to a learning database.
  - scripts/diagnostics/diagnose_layers.py:178: `db = Database()` then
    accesses `db._conn.cursor()` to run `SELECT 1`.

The simplest faithful reconstruction is to re-export `database.db.TraderDB`
as `Database`. TraderDB has a `_conn` attribute (SQLite connection) and
provides the trade-journaling methods that journal_bridge.py expects.
"""
from __future__ import annotations

# Re-export TraderDB as Database — preserves the historical import path
# `from memory.database import Database` without duplicating logic.
from database.db import TraderDB as Database

__all__ = ["Database"]
