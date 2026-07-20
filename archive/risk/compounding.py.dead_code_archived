"""
risk/compounding.py — Compounding Growth Engine (Day 81+)
==========================================================

Masterclass concept: Grow a small balance into a large one through
systematic profit reinvestment. Instead of withdrawing profits, the
bot adds them to the trading capital, so each subsequent trade uses
a slightly larger lot size.

Compounding formula:
    new_balance = old_balance + realized_profit
    new_lot = base_lot * (new_balance / initial_balance)

This module tracks the balance history and computes the compounding
multiplier for the position sizer.

Usage:
    from risk.compounding import get_compounding_engine

    engine = get_compounding_engine(initial_balance=10000)
    engine.record_profit(trade_pnl=50.0)
    multiplier = engine.get_lot_multiplier()  # e.g. 1.05 if balance grew 5%
    current_balance = engine.current_balance  # 10050
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from utils.logger import get_logger

log = get_logger("compounding")

STATE_FILE = "memory/compounding_state.json"


@dataclass
class TradeRecord:
    """One closed trade's PnL record."""
    timestamp: str
    pair: str
    pnl_usd: float
    balance_after: float


class CompoundingEngine:
    """
    Tracks realized PnL and computes a lot-size multiplier that grows
    the trading size as the balance grows.

    Rules:
      - Initial balance is set once (first time the engine is created)
      - Each closed trade's PnL is added to the current balance
      - Lot multiplier = current_balance / initial_balance
      - If balance drops below 50% of initial, multiplier is capped at 0.5
        (don't over-trade when losing)
      - If balance grows >2x initial, multiplier is capped at 2.0
        (don't over-leverage when winning big)
    """

    MAX_MULTIPLIER = 2.0   # cap at 2x base lot even if balance doubled
    MIN_MULTIPLIER = 0.5   # floor at 0.5x base lot if balance halved

    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.trade_history: List[TradeRecord] = []
        self._load_state()

    def _load_state(self) -> None:
        """Load compounding state from disk (persists across restarts)."""
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, "r") as f:
                    data = json.load(f)
                self.initial_balance = data.get("initial_balance", self.initial_balance)
                self.current_balance = data.get("current_balance", self.current_balance)
                self.trade_history = [
                    TradeRecord(**t) for t in data.get("trade_history", [])
                ]
                log.info(
                    f"[Compounding] State loaded | "
                    f"initial=${self.initial_balance} | "
                    f"current=${self.current_balance:.2f} | "
                    f"trades={len(self.trade_history)}"
                )
        except Exception as e:
            log.debug(f"Compounding state load failed: {e}")

    def _save_state(self) -> None:
        """Save compounding state to disk."""
        try:
            os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
            with open(STATE_FILE, "w") as f:
                json.dump({
                    "initial_balance": self.initial_balance,
                    "current_balance": self.current_balance,
                    "trade_history": [
                        {
                            "timestamp": t.timestamp,
                            "pair": t.pair,
                            "pnl_usd": t.pnl_usd,
                            "balance_after": t.balance_after,
                        }
                        for t in self.trade_history[-200:]  # keep last 200
                    ],
                }, f, indent=2)
        except Exception as e:
            log.debug(f"Compounding state save failed: {e}")

    def record_profit(self, trade_pnl: float, pair: str = "") -> float:
        """Record a closed trade's PnL and update balance.

        Args:
            trade_pnl: Profit/loss in USD (positive = win, negative = loss)
            pair:      Trading pair (for record-keeping)

        Returns:
            New balance after applying PnL
        """
        self.current_balance += trade_pnl
        self.trade_history.append(TradeRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            pair=pair,
            pnl_usd=trade_pnl,
            balance_after=self.current_balance,
        ))
        self._save_state()

        growth_pct = ((self.current_balance - self.initial_balance)
                      / self.initial_balance * 100)
        log.info(
            f"[Compounding] Trade PnL: ${trade_pnl:+.2f} | "
            f"Balance: ${self.current_balance:.2f} | "
            f"Growth: {growth_pct:+.1f}% | "
            f"Multiplier: {self.get_lot_multiplier():.3f}"
        )
        return self.current_balance

    def get_lot_multiplier(self) -> float:
        """Compute the lot-size multiplier based on current balance.

        Returns:
            Multiplier between MIN_MULTIPLIER (0.5) and MAX_MULTIPLIER (2.0)
        """
        if self.initial_balance <= 0:
            return 1.0
        raw = self.current_balance / self.initial_balance
        return max(self.MIN_MULTIPLIER, min(self.MAX_MULTIPLIER, raw))

    def get_stats(self) -> dict:
        """Return compounding statistics for dashboard."""
        total_trades = len(self.trade_history)
        if total_trades == 0:
            return {
                "initial_balance": self.initial_balance,
                "current_balance": self.current_balance,
                "growth_pct":      0.0,
                "lot_multiplier":  1.0,
                "total_trades":    0,
                "wins":            0,
                "losses":          0,
                "win_rate":        0.0,
            }

        wins = [t for t in self.trade_history if t.pnl_usd > 0]
        losses = [t for t in self.trade_history if t.pnl_usd < 0]
        growth = ((self.current_balance - self.initial_balance)
                  / self.initial_balance * 100)

        return {
            "initial_balance":  self.initial_balance,
            "current_balance":  self.current_balance,
            "growth_pct":       round(growth, 2),
            "lot_multiplier":   round(self.get_lot_multiplier(), 3),
            "total_trades":     total_trades,
            "wins":             len(wins),
            "losses":           len(losses),
            "win_rate":         round(len(wins) / total_trades * 100, 1),
            "total_profit":     round(sum(t.pnl_usd for t in wins), 2),
            "total_loss":       round(sum(t.pnl_usd for t in losses), 2),
            "net_pnl":          round(self.current_balance - self.initial_balance, 2),
        }

    def reset(self, new_initial: float = None) -> None:
        """Reset compounding (start fresh). Use with caution!"""
        self.initial_balance = new_initial or self.current_balance
        self.current_balance = self.initial_balance
        self.trade_history = []
        self._save_state()
        log.info(f"[Compounding] Reset | initial=${self.initial_balance}")


# ── Singleton ─────────────────────────────────────────────────

_ENGINE: Optional[CompoundingEngine] = None


def get_compounding_engine(initial_balance: float = 10000.0) -> CompoundingEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = CompoundingEngine(initial_balance=initial_balance)
    return _ENGINE
