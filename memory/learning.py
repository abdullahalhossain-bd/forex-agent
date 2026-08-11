"""
memory/learning.py — aggregated performance report generator.

Reconstructed from production caller contracts. The original implementation
existed on the production machine but was never committed to git.

Behavioral contract (verified from caller sites):
  - __init__(): no arguments required. May optionally accept a path.
  - print_report() -> None: prints a multi-line performance summary to
    logger. Used by AITrader.get_learning_report() (core/trader.py:2495).

Note: AITrader ALSO instantiates `agents.learning_agent.LearningAgent`
(stored as `self._learn`), which is a SEPARATE class with its own
JSON-backed decision log. This `LearningEngine` provides aggregate
reporting across multiple sources (TradeMemory + LearningAgent +
CircuitBreaker). The reconstruction keeps it simple — it pulls stats
from TradeMemory if available, otherwise prints an empty report.
"""
from __future__ import annotations

from typing import Optional
from utils.logger import get_logger

log = get_logger("learning_engine")


class LearningEngine:
    """Aggregated learning report generator."""

    def __init__(self, path: Optional[str] = None):
        # `path` accepted for backward compat with `core/runtime.py:204`:
        #   registry.register("learning_engine", lambda r: LearningEngine())
        # The original implementation may have used a JSON path for its
        # own state; we don't persist anything new — we read from
        # TradeMemory on demand.
        self._path = path

    def print_report(self) -> None:
        """Print a multi-line performance report to logger.

        Pulls stats from TradeMemory if available. Never raises.
        """
        try:
            # Try to pull from TradeMemory via the registry. We use a
            # lazy import to avoid circular dependencies.
            from memory.trade_memory import TradeMemory
            tm = TradeMemory(seed_rules=False)
            # Reuse TradeMemory's stats summary
            import json
            with tm._lock:
                records = tm._records
                total = len(records)
                wins = sum(1 for r in records if r.get("result") == "WIN")
                losses = sum(1 for r in records if r.get("result") == "LOSS")
                breakeven = sum(1 for r in records if r.get("result") == "BREAKEVEN")
                open_trades = sum(1 for r in records
                                  if r.get("outcome") is None
                                  and r.get("decision") in ("BUY", "SELL"))
                total_pnl = sum((r.get("pnl_pips") or 0) for r in records)
                wr = (wins / (wins + losses) * 100) if (wins + losses) else 0.0

            log.info("=" * 60)
            log.info("[LearningEngine] Performance Report")
            log.info("=" * 60)
            log.info("  Total decisions : %d", total)
            log.info("  Wins            : %d", wins)
            log.info("  Losses          : %d", losses)
            log.info("  Breakeven       : %d", breakeven)
            log.info("  Open trades     : %d", open_trades)
            log.info("  Win rate        : %.2f%%", wr)
            log.info("  Total PnL (pips): %.2f", total_pnl)
            log.info("=" * 60)
        except Exception as e:
            log.warning("[LearningEngine] print_report failed: %s", e)
