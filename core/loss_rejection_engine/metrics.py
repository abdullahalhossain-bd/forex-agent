
"""LRE Performance Metrics Tracker.
Tracks Loss Rejection Rate and Winner Preservation Rate."""
from __future__ import annotations
import json, time, logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

METRICS_DIR = Path(__file__).parent.parent.parent / "memory" / "lre_models"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class TradeRecord:
    timestamp: float
    symbol: str
    direction: str
    pnl: float
    was_blocked: bool
    was_shadow_blocked: bool
    l1_verdict: str = ""
    l2_verdict: str = ""
    l3_verdict: str = ""
    l1_composite: float = 0.0
    l2_loss_prob: float = 0.0
    l3_distance: float = 0.0

@dataclass
class LREMetrics:
    loss_rejection_rate: float = 0.0
    winner_preservation_rate: float = 0.0
    total_trades: int = 0
    total_winners: int = 0
    total_losers: int = 0
    blocked_losses: int = 0
    blocked_winners: int = 0
    kept_winners: int = 0
    kept_losers: int = 0
    shadow_correct_blocks: int = 0
    shadow_wrong_blocks: int = 0


class LREMetricsTracker:
    """Tracks LRE performance metrics over time.

    Key Metrics:
      - Loss Rejection Rate (LRR) = blocked_losses / total_losses
      - Winner Preservation Rate (WPR) = kept_winners / total_winners
    """

    def __init__(self, max_records: int = 500):
        self._records: deque = deque(maxlen=max_records)
        self._path = METRICS_DIR / "lre_metrics.jsonl"
        self._load()

    def record(self, r: TradeRecord):
        self._records.append(r)
        self._append_to_disk(r)

    def get_metrics(self) -> LREMetrics:
        if not self._records:
            return LREMetrics()
        m = LREMetrics()
        m.total_trades = len(self._records)
        for r in self._records:
            is_winner = r.pnl > 0
            if is_winner:
                m.total_winners += 1
            else:
                m.total_losers += 1
            if r.was_blocked or r.was_shadow_blocked:
                if not is_winner:
                    m.blocked_losses += 1
                else:
                    m.blocked_winners += 1
            else:
                if is_winner:
                    m.kept_winners += 1
                else:
                    m.kept_losers += 1
            # Shadow accuracy
            if r.was_shadow_blocked:
                if not is_winner:
                    m.shadow_correct_blocks += 1
                else:
                    m.shadow_wrong_blocks += 1
        m.loss_rejection_rate = m.blocked_losses / m.total_losers if m.total_losers > 0 else 0
        m.winner_preservation_rate = m.kept_winners / m.total_winners if m.total_winners > 0 else 0
        return m

    def get_summary(self) -> str:
        m = self.get_metrics()
        lines = [
            f"LRE Metrics ({m.total_trades} trades)",
            f"  LRR (Loss Rejection Rate): {m.loss_rejection_rate:.1%}",
            f"  WPR (Winner Preservation):  {m.winner_preservation_rate:.1%}",
            f"  Winners: {m.total_winners} | Losers: {m.total_losers}",
            f"  Blocked losses: {m.blocked_losses} | Blocked winners: {m.blocked_winners}",
            f"  Kept winners: {m.kept_winners} | Kept losers: {m.kept_losers}",
        ]
        if m.blocked_losses + m.blocked_winners > 0:
            lines.append(f"  Shadow accuracy: {m.shadow_correct_blocks}/{m.blocked_losses + m.blocked_winners}")
        return "\n".join(lines)

    def _append_to_disk(self, r: TradeRecord):
        try:
            with open(self._path, "a") as f:
                f.write(json.dumps({
                    "ts": r.timestamp, "symbol": r.symbol, "dir": r.direction,
                    "pnl": r.pnl, "blocked": r.was_blocked,
                    "shadow": r.was_shadow_blocked,
                    "l1": r.l1_verdict, "l2": r.l2_verdict, "l3": r.l3_verdict,
                }) + "\n")
        except Exception as e:
            log.debug(f"[LRE-Metrics] Write error: {e}")

    def _load(self):
        if not self._path.exists(): return
        try:
            with open(self._path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    d = json.loads(line)
                    self._records.append(TradeRecord(
                        timestamp=d.get("ts", 0), symbol=d.get("symbol", ""),
                        direction=d.get("dir", ""), pnl=d.get("pnl", 0),
                        was_blocked=d.get("blocked", False),
                        was_shadow_blocked=d.get("shadow", False),
                        l1_verdict=d.get("l1", ""), l2_verdict=d.get("l2", ""),
                        l3_verdict=d.get("l3", ""),
                    ))
        except Exception as e:
            log.warning(f"[LRE-Metrics] Load error: {e}")
