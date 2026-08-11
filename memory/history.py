"""
memory/history.py — per-(symbol, timeframe) bias history with outcome tracking.

Reconstructed from production caller contracts. The original implementation
existed on the production machine but was never committed to git.

Behavioral contract (verified from caller sites):
  - __init__(): no arguments required.
  - save(symbol, timeframe, bias_ctx, ind_ctx) -> None: appends a new
    analysis entry to memory/analysis_history.json. Used by AITrader at
    core/trader.py:3323 (wrapped in try/except — non-fatal).
  - update_result(index, result, pnl) -> None: back-fills the outcome of
    a previously-saved entry by 0-indexed position. Used by AITrader at
    core/trader.py:2977 (wrapped in try/except — non-fatal).

The save() call is invoked AFTER the decision is made but BEFORE the
trade is closed. update_result() is invoked when the trade closes, using
the decision_id (1-indexed) stashed at open time, minus 1 (to convert
to 0-indexed position). See core/trader.py:2971-2981 for the comment.

Persistence:
  - File: memory/analysis_history.json (path: core.constants.ANALYSIS_HISTORY_PATH)
  - Format: list of dicts {symbol, timeframe, timestamp, bias, ind_ctx, outcome, pnl}
  - Atomic writes via tempfile + os.replace.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.constants import ANALYSIS_HISTORY_PATH
from utils.logger import get_logger

log = get_logger("analysis_history")


class AnalysisHistory:
    """Per-(symbol, timeframe) bias history with outcome tracking."""

    # Singleton — every callsite uses `AnalysisHistory()` (no assignment)
    # so each call creates a fresh instance that re-reads the JSON file.
    # We use a module-level lock to serialize writes.
    _write_lock = threading.Lock()

    def __init__(self):
        self._path = ANALYSIS_HISTORY_PATH

    def _load(self) -> List[Dict[str, Any]]:
        if not self._path.exists():
            return []
        try:
            with open(self._path, "r") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception as e:
            log.warning("[AnalysisHistory] load failed: %s — starting empty", e)
            return []

    def _save(self, records: List[Dict[str, Any]]) -> None:
        with AnalysisHistory._write_lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", dir=str(self._path.parent), suffix=".tmp",
                    prefix="analysis_history_", delete=False
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

    def save(self, symbol: str, timeframe: str,
             bias_ctx: Dict[str, Any], ind_ctx: Dict[str, Any]) -> None:
        """Append a new analysis entry. Non-fatal — caller wraps in try/except."""
        try:
            records = self._load()
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "timeframe": timeframe,
                "bias": bias_ctx.get("bias") if bias_ctx else None,
                "confidence_pct": bias_ctx.get("confidence_pct") if bias_ctx else None,
                "recommendation": bias_ctx.get("recommendation") if bias_ctx else None,
                "has_conflict": bias_ctx.get("has_conflict") if bias_ctx else None,
                "ind_ctx": ind_ctx,
                "outcome": None,
                "pnl": None,
            }
            records.append(entry)
            self._save(records)
        except Exception as e:
            log.warning("[AnalysisHistory] save failed: %s", e)

    def update_result(self, index: int, result: str, pnl: float) -> None:
        """Back-fill outcome for entry at 0-indexed `index`. Non-fatal."""
        try:
            records = self._load()
            if 0 <= index < len(records):
                records[index]["outcome"] = result
                records[index]["pnl"] = float(pnl or 0)
                self._save(records)
            else:
                log.warning(
                    "[AnalysisHistory] update_result: index %d out of range (len=%d)",
                    index, len(records)
                )
        except Exception as e:
            log.warning("[AnalysisHistory] update_result failed: %s", e)
