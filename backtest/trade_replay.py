"""
backtest/trade_replay.py — Per-Trade Replay Logger

Records detailed information for every trade taken or rejected during backtest.
Used for Trade Replay section in the final report and for post-hoc analysis.
"""

from __future__ import annotations

import json
import csv
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any
from datetime import datetime
from pathlib import Path

log = logging.getLogger("trade_replay")


@dataclass
class TradeReplayEntry:
    """Detailed record for a single trade decision."""
    trade_id: int = 0
    bar_index: int = 0
    chart_time: str = ""
    symbol: str = ""
    timeframe: str = ""

    # Entry/Exit
    direction: str = ""  # BUY, SELL, WAIT
    entry_price: float = 0.0
    exit_price: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    lot_size: float = 0.0

    # Outcome
    pnl_pips: float = 0.0
    pnl_usd: float = 0.0
    exit_reason: str = ""  # TP, SL, timeout, end_of_backtest
    hold_bars: int = 0
    hold_time_str: str = ""
    commission_usd: float = 0.0
    slippage_pips: float = 0.0

    # Decision detail
    confidence: float = 0.0
    strategy: str = ""
    rejection_reason: str = ""  # if rejected: WAIT, risk_rejected, permission_blocked, etc.

    # Agent votes (from DecisionAgent)
    agent_votes: Dict[str, str] = field(default_factory=dict)
    # e.g. {"master_analyst": "BUY", "rule_engine": "SELL", "llm_analyst": "BUY"}

    # Analysis context snapshot
    regime: str = ""
    session: str = ""
    trend: str = ""
    rsi: float = 0.0
    adx: float = 0.0
    atr: float = 0.0

    # Strategy breakdown contributions
    strategy_contributions: Dict[str, float] = field(default_factory=dict)
    # e.g. {"ICT AMD": 0.3, "SR Zones": 0.2, "SMC": 0.1}

    # Market structure at entry
    market_structure: str = ""
    structure_bias: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class TradeReplayLogger:
    """Collects and persists per-trade replay data."""

    def __init__(self, output_dir: str = "backtest/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.entries: List[TradeReplayEntry] = []
        self._counter = 0

    def log_trade(self, entry: TradeReplayEntry) -> None:
        self._counter += 1
        if entry.trade_id == 0:
            entry.trade_id = self._counter
        self.entries.append(entry)

    def log_rejection(self, bar_index: int, chart_time: str, symbol: str,
                      timeframe: str, reason: str, confidence: float = 0.0,
                      direction: str = "WAIT", agent_votes: Dict = None,
                      regime: str = "", session: str = "") -> None:
        entry = TradeReplayEntry(
            trade_id=0, bar_index=bar_index, chart_time=chart_time,
            symbol=symbol, timeframe=timeframe, direction=direction,
            rejection_reason=reason, confidence=confidence,
            agent_votes=agent_votes or {}, regime=regime, session=session,
        )
        self.log_trade(entry)

    def get_trades(self) -> List[TradeReplayEntry]:
        """Get only executed trades (not rejections)."""
        return [e for e in self.entries if e.direction in ("BUY", "SELL")]

    def get_rejections(self) -> List[TradeReplayEntry]:
        """Get only rejected/wait entries."""
        return [e for e in self.entries if e.direction not in ("BUY", "SELL")]

    def save_csv(self, filename: str = "trade_replay.csv") -> str:
        """Save trade replay to CSV file."""
        if not self.entries:
            return ""
        path = self.output_dir / filename
        trades_only = self.get_trades()
        if not trades_only:
            return ""
        fieldnames = list(TradeReplayEntry.__dataclass_fields__.keys())
        # Remove agent_votes and strategy_contributions (dicts) for CSV
        simple_fields = [f for f in fieldnames if f not in ("agent_votes", "strategy_contributions")]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=simple_fields, extrasaction="ignore")
            writer.writeheader()
            for entry in trades_only:
                writer.writerow(asdict(entry))
        log.info(f"[TradeReplay] Saved {len(trades_only)} trades to {path}")
        return str(path)

    def save_json(self, filename: str = "trade_replay.json") -> str:
        """Save full trade replay (including rejections) to JSON."""
        if not self.entries:
            return ""
        path = self.output_dir / filename
        data = [e.to_dict() for e in self.entries]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        log.info(f"[TradeReplay] Saved {len(self.entries)} entries to {path}")
        return str(path)

    def summary(self) -> Dict:
        trades = self.get_trades()
        rejections = self.get_rejections()
        return {
            "total_entries": len(self.entries),
            "total_trades": len(trades),
            "total_rejections": len(rejections),
        }
