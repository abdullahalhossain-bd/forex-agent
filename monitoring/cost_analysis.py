"""
monitoring/cost_analysis.py — Day 97+ Cost Analysis Tracker
============================================================
Tracks ALL trading costs to calculate true net profitability:

  - Spread cost (entry + exit)
  - Commission (per lot, per trade)
  - Swap (overnight holding fees)
  - Slippage cost (expected vs actual fill)

Without this, you CANNOT know if your strategy is actually profitable.

Usage:
    from monitoring.cost_analysis import CostTracker
    ct = CostTracker()
    ct.record_trade(
        symbol="EURUSD", lot=0.10, direction="BUY",
        entry_spread_pips=1.2, exit_spread_pips=1.5,
        commission_usd=3.5, swap_usd=-1.2,
        gross_pnl=50.0, slippage_cost_pips=0.3
    )
    stats = ct.get_cost_summary()
    # → {"gross_pnl": 500, "total_costs": 87, "net_pnl": 413, "cost_pct": 17.4%}
"""

import json
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import deque

from utils.logger import get_logger
from core.constants import get_pip_size, get_pip_value_usd

log = get_logger("cost_analysis")


class CostTracker:
    """Tracks all trading costs and calculates true net profitability.

    The "hidden killer" in forex trading is costs. A strategy that looks
    profitable in backtest can lose money live because:
      - Spread: 0.5-2 pips per trade (entry + exit = 1-4 pips)
      - Commission: $3.5-7 per lot per trade
      - Swap: -$1 to -$5 per lot per night held
      - Slippage: 0.5-3 pips on average

    This tracker makes ALL costs visible so you can see if your edge
    survives after costs.
    """

    DB_PATH = "database/cost_analysis.db"
    ROLLING_WINDOW = 100

    def __init__(self):
        Path("database").mkdir(exist_ok=True)
        self._init_db()
        self._recent: deque = deque(maxlen=self.ROLLING_WINDOW)
        self._load_recent()

    def _init_db(self):
        with sqlite3.connect(self.DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cost_log (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp           TEXT NOT NULL,
                    symbol              TEXT NOT NULL,
                    direction           TEXT NOT NULL,
                    lot                 REAL,
                    entry_spread_pips   REAL,
                    exit_spread_pips    REAL,
                    commission_usd      REAL,
                    swap_usd            REAL,
                    slippage_cost_pips  REAL,
                    slippage_cost_usd   REAL,
                    gross_pnl_usd       REAL,
                    total_cost_usd      REAL,
                    net_pnl_usd         REAL,
                    cost_pct            REAL,
                    holding_hours       REAL,
                    ticket              INTEGER
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cost_symbol ON cost_log(symbol)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cost_timestamp ON cost_log(timestamp)
            """)

    def _load_recent(self):
        try:
            with sqlite3.connect(self.DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    f"SELECT * FROM cost_log ORDER BY id DESC LIMIT {self.ROLLING_WINDOW}"
                ).fetchall()
                for row in reversed(rows):
                    self._recent.append(dict(row))
        except Exception as e:
            log.warning(f"[CostTracker] Failed to load recent: {e}")

    def record_trade(
        self,
        symbol: str,
        direction: str,
        lot: float,
        entry_spread_pips: float = 0.0,
        exit_spread_pips: float = 0.0,
        commission_usd: float = 0.0,
        swap_usd: float = 0.0,
        slippage_cost_pips: float = 0.0,
        gross_pnl_usd: float = 0.0,
        holding_hours: float = 0.0,
        ticket: Optional[int] = None,
    ) -> dict:
        """Record one completed trade with all costs.

        Args:
            entry_spread_pips: spread at entry time
            exit_spread_pips: spread at exit time
            commission_usd: total commission for this trade (both sides)
            swap_usd: total swap (negative = paid, positive = received)
            slippage_cost_pips: total slippage in pips (entry + exit)
            gross_pnl_usd: PnL before costs (price move × lot × pip_value)
            holding_hours: how long the trade was open
        """
        # Calculate slippage cost in USD
        pip_value = get_pip_value_usd(symbol)
        slippage_cost_usd = slippage_cost_pips * pip_value * lot

        # Total spread cost in USD
        total_spread_pips = entry_spread_pips + exit_spread_pips
        spread_cost_usd = total_spread_pips * pip_value * lot

        # Total costs
        total_cost_usd = (
            spread_cost_usd
            + commission_usd
            + abs(swap_usd)  # swap is negative when you pay
            + slippage_cost_usd
        )

        # Net PnL
        net_pnl_usd = gross_pnl_usd - total_cost_usd

        # Cost as % of gross
        cost_pct = round((total_cost_usd / abs(gross_pnl_usd) * 100), 1) if gross_pnl_usd else 0

        now = datetime.now(timezone.utc).isoformat()
        record = {
            "timestamp": now,
            "symbol": symbol,
            "direction": direction,
            "lot": lot,
            "entry_spread_pips": entry_spread_pips,
            "exit_spread_pips": exit_spread_pips,
            "commission_usd": commission_usd,
            "swap_usd": swap_usd,
            "slippage_cost_pips": slippage_cost_pips,
            "slippage_cost_usd": round(slippage_cost_usd, 2),
            "gross_pnl_usd": round(gross_pnl_usd, 2),
            "total_cost_usd": round(total_cost_usd, 2),
            "net_pnl_usd": round(net_pnl_usd, 2),
            "cost_pct": cost_pct,
            "holding_hours": holding_hours,
            "ticket": ticket,
        }

        try:
            with sqlite3.connect(self.DB_PATH) as conn:
                conn.execute("""
                    INSERT INTO cost_log
                    (timestamp, symbol, direction, lot, entry_spread_pips,
                     exit_spread_pips, commission_usd, swap_usd,
                     slippage_cost_pips, slippage_cost_usd, gross_pnl_usd,
                     total_cost_usd, net_pnl_usd, cost_pct, holding_hours, ticket)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, tuple(record.values()))
        except Exception as e:
            log.warning(f"[CostTracker] DB insert failed: {e}")

        self._recent.append(record)

        # Warn if costs eating too much
        if cost_pct > 30:
            log.warning(
                f"[CostTracker] ⚠️ Costs eating {cost_pct}% of gross PnL on {symbol} "
                f"(gross=${gross_pnl_usd:.2f}, costs=${total_cost_usd:.2f})"
            )

        return record

    def get_cost_summary(self, symbol: Optional[str] = None) -> dict:
        """Get aggregated cost statistics."""
        records = list(self._recent)
        if symbol:
            records = [r for r in records if r.get("symbol") == symbol]

        if not records:
            return {"total_trades": 0}

        total_gross = sum(r.get("gross_pnl_usd", 0) for r in records)
        total_cost = sum(r.get("total_cost_usd", 0) for r in records)
        total_net = sum(r.get("net_pnl_usd", 0) for r in records)

        total_spread = sum(
            (r.get("entry_spread_pips", 0) + r.get("exit_spread_pips", 0))
            for r in records
        )
        total_commission = sum(r.get("commission_usd", 0) for r in records)
        total_swap = sum(abs(r.get("swap_usd", 0)) for r in records)
        total_slippage = sum(r.get("slippage_cost_usd", 0) for r in records)

        wins = sum(1 for r in records if r.get("net_pnl_usd", 0) > 0)
        losses = sum(1 for r in records if r.get("net_pnl_usd", 0) < 0)

        return {
            "total_trades": len(records),
            "gross_pnl_usd": round(total_gross, 2),
            "total_costs_usd": round(total_cost, 2),
            "net_pnl_usd": round(total_net, 2),
            "cost_pct_of_gross": round(total_cost / abs(total_gross) * 100, 1) if total_gross else 0,
            "spread_cost_usd": round(total_spread, 2),
            "commission_usd": round(total_commission, 2),
            "swap_cost_usd": round(total_swap, 2),
            "slippage_cost_usd": round(total_slippage, 2),
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / len(records) * 100, 1) if records else 0,
            "avg_cost_per_trade": round(total_cost / len(records), 2),
            "avg_holding_hours": round(
                sum(r.get("holding_hours", 0) for r in records) / len(records), 1
            ),
        }

    def get_per_symbol_summary(self) -> Dict[str, dict]:
        symbols = set(r.get("symbol") for r in self._recent if r.get("symbol"))
        return {s: self.get_cost_summary(symbol=s) for s in symbols}

    def print_report(self):
        """Print a formatted cost analysis report."""
        summary = self.get_cost_summary()

        bar = "═" * 52
        log.info(bar)
        log.info("  💰  COST ANALYSIS REPORT")
        log.info(bar)

        if summary["total_trades"] == 0:
            log.info("  No trades recorded yet.")
            log.info(bar)
            return

        log.info(f"  Total Trades   : {summary['total_trades']}")
        log.info(f"  Win Rate       : {summary['win_rate']}% ({summary['wins']}W / {summary['losses']}L)")
        log.info(f"  ── PnL ──")
        log.info(f"  Gross PnL      : ${summary['gross_pnl_usd']:.2f}")
        log.info(f"  Total Costs    : ${summary['total_costs_usd']:.2f} ({summary['cost_pct_of_gross']}% of gross)")
        log.info(f"  Net PnL        : ${summary['net_pnl_usd']:.2f}")
        log.info(f"  ── Cost Breakdown ──")
        log.info(f"  Spread         : ${summary['spread_cost_usd']:.2f}")
        log.info(f"  Commission     : ${summary['commission_usd']:.2f}")
        log.info(f"  Swap           : ${summary['swap_cost_usd']:.2f}")
        log.info(f"  Slippage       : ${summary['slippage_cost_usd']:.2f}")
        log.info(f"  ── Averages ──")
        log.info(f"  Cost/Trade     : ${summary['avg_cost_per_trade']:.2f}")
        log.info(f"  Holding Time   : {summary['avg_holding_hours']:.1f}h")

        per_symbol = self.get_per_symbol_summary()
        if len(per_symbol) > 1:
            log.info(f"  ── Per Symbol ──")
            for sym, s in sorted(per_symbol.items()):
                net = s["net_pnl_usd"]
                icon = "🟢" if net > 0 else "🔴" if net < 0 else "⚪"
                log.info(f"    {icon} {sym:<8} net=${net:.2f}  cost={s['cost_pct_of_gross']}%  WR={s['win_rate']}%")

        log.info(bar)


# ── Singleton ─────────────────────────────────────────────────────

_CT: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    global _CT
    if _CT is None:
        _CT = CostTracker()
    return _CT
