# backtest/trade_analyzer.py — Comprehensive trade statistics analyzer
# =============================================================================
# Ported from: https://github.com/mementum/backtrader/blob/master/backtrader/analyzers/tradeanalyzer.py
# Original author: Daniel Rodriguez (mementum) — GPL v3
#
# Comprehensive trade statistics from a list of closed trades. Computes:
#   - Total open/closed trades
#   - Win/loss streaks (current + longest)
#   - PNL: total, average, won total/avg/max, lost total/avg/max
#   - Long/short breakdown
#   - Holding length (bars in market): total/avg/max/min, by won/lost, by long/short
#
# This is a standalone Python implementation — no backtrader dependency.
# Pass in a list of trade dicts and get a full statistics report.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from utils.logger import get_logger

log = get_logger("trade_analyzer")


@dataclass
class TradeRecord:
    """A single closed trade record."""
    pnl: float                    # profit/loss in account currency
    pnl_pct: float = 0.0          # profit/loss as percentage
    direction: str = "long"       # "long" or "short"
    entry_price: float = 0.0
    exit_price: float = 0.0
    entry_time: Optional[str] = None
    exit_time: Optional[str] = None
    holding_bars: int = 0         # bars in market
    size: float = 0.0             # position size
    symbol: str = ""


class TradeAnalyzer:
    """
    Analyze a list of closed trades and compute comprehensive statistics.

    Usage
    -----
    >>> analyzer = TradeAnalyzer()
    >>> for trade in trades:
    ...     analyzer.add_trade(trade)
    >>> stats = analyzer.get_analysis()
    >>> print(f"Win rate: {stats['won']['count']}/{stats['total']['closed']}")

    Or from a list:
    >>> analyzer = TradeAnalyzer.from_list(trade_dicts)
    >>> stats = analyzer.get_analysis()
    """

    def __init__(self):
        self.trades: list[TradeRecord] = []
        self._open_count: int = 0

    def add_trade(self, trade: TradeRecord) -> None:
        """Add a closed trade to the analyzer."""
        self.trades.append(trade)

    def add_open(self) -> None:
        """Record that a trade was opened (for open count tracking)."""
        self._open_count += 1

    @classmethod
    def from_list(cls, trades: list[dict]) -> "TradeAnalyzer":
        """Create analyzer from a list of trade dicts."""
        analyzer = cls()
        for t in trades:
            trade = TradeRecord(
                pnl=float(t.get("pnl", 0)),
                pnl_pct=float(t.get("pnl_pct", 0)),
                direction=t.get("direction", "long"),
                entry_price=float(t.get("entry_price", 0)),
                exit_price=float(t.get("exit_price", 0)),
                entry_time=t.get("entry_time"),
                exit_time=t.get("exit_time"),
                holding_bars=int(t.get("holding_bars", 0)),
                size=float(t.get("size", 0)),
                symbol=t.get("symbol", ""),
            )
            analyzer.add_trade(trade)
        return analyzer

    def get_analysis(self) -> dict:
        """
        Compute and return full trade analysis.

        Returns a dict with keys:
            total: {total, open, closed}
            streak: {won.current, won.longest, lost.current, lost.longest}
            pnl: {net.total, net.average}
            won: {count, total, average, max}
            lost: {count, total, average, max}
            long: {count, total, average, won, lost, won_total, lost_total}
            short: {count, total, average, won, lost, won_total, lost_total}
            length: {total, average, max, min, won, lost}
        """
        closed = self.trades
        n_closed = len(closed)

        if n_closed == 0:
            return {
                "total": {"total": 0, "open": self._open_count, "closed": 0},
                "streak": {"won": {"current": 0, "longest": 0},
                           "lost": {"current": 0, "longest": 0}},
                "pnl": {"net": {"total": 0.0, "average": 0.0}},
                "won": {"count": 0, "total": 0.0, "average": 0.0, "max": 0.0},
                "lost": {"count": 0, "total": 0.0, "average": 0.0, "max": 0.0},
                "long": {"count": 0, "total": 0.0, "average": 0.0},
                "short": {"count": 0, "total": 0.0, "average": 0.0},
                "length": {"total": 0, "average": 0.0, "max": 0, "min": 0},
            }

        pnls = np.array([t.pnl for t in closed])
        won_mask = pnls > 0
        lost_mask = pnls < 0

        won_pnls = pnls[won_mask]
        lost_pnls = pnls[lost_mask]

        # Streaks
        won_streak_cur, won_streak_max = self._compute_streaks(pnls > 0)
        lost_streak_cur, lost_streak_max = self._compute_streaks(pnls < 0)

        # Long/Short breakdown
        long_trades = [t for t in closed if t.direction == "long"]
        short_trades = [t for t in closed if t.direction == "short"]

        long_pnls = np.array([t.pnl for t in long_trades]) if long_trades else np.array([])
        short_pnls = np.array([t.pnl for t in short_trades]) if short_trades else np.array([])

        # Holding length
        lengths = np.array([t.holding_bars for t in closed])
        won_lengths = lengths[won_mask] if won_mask.any() else np.array([])
        lost_lengths = lengths[lost_mask] if lost_mask.any() else np.array([])

        return {
            "total": {
                "total": n_closed + self._open_count,
                "open": self._open_count,
                "closed": n_closed,
            },
            "streak": {
                "won": {"current": int(won_streak_cur), "longest": int(won_streak_max)},
                "lost": {"current": int(lost_streak_cur), "longest": int(lost_streak_max)},
            },
            "pnl": {
                "net": {
                    "total": float(pnls.sum()),
                    "average": float(pnls.mean()),
                },
            },
            "won": {
                "count": int(won_mask.sum()),
                "total": float(won_pnls.sum()) if len(won_pnls) > 0 else 0.0,
                "average": float(won_pnls.mean()) if len(won_pnls) > 0 else 0.0,
                "max": float(won_pnls.max()) if len(won_pnls) > 0 else 0.0,
            },
            "lost": {
                "count": int(lost_mask.sum()),
                "total": float(lost_pnls.sum()) if len(lost_pnls) > 0 else 0.0,
                "average": float(lost_pnls.mean()) if len(lost_pnls) > 0 else 0.0,
                "max": float(lost_pnls.min()) if len(lost_pnls) > 0 else 0.0,
            },
            "long": {
                "count": len(long_trades),
                "total": float(long_pnls.sum()) if len(long_pnls) > 0 else 0.0,
                "average": float(long_pnls.mean()) if len(long_pnls) > 0 else 0.0,
                "won": int((long_pnls > 0).sum()) if len(long_pnls) > 0 else 0,
                "lost": int((long_pnls < 0).sum()) if len(long_pnls) > 0 else 0,
            },
            "short": {
                "count": len(short_trades),
                "total": float(short_pnls.sum()) if len(short_pnls) > 0 else 0.0,
                "average": float(short_pnls.mean()) if len(short_pnls) > 0 else 0.0,
                "won": int((short_pnls > 0).sum()) if len(short_pnls) > 0 else 0,
                "lost": int((short_pnls < 0).sum()) if len(short_pnls) > 0 else 0,
            },
            "length": {
                "total": int(lengths.sum()),
                "average": float(lengths.mean()),
                "max": int(lengths.max()),
                "min": int(lengths.min()),
                "won": {
                    "average": float(won_lengths.mean()) if len(won_lengths) > 0 else 0.0,
                    "max": int(won_lengths.max()) if len(won_lengths) > 0 else 0,
                },
                "lost": {
                    "average": float(lost_lengths.mean()) if len(lost_lengths) > 0 else 0.0,
                    "max": int(lost_lengths.max()) if len(lost_lengths) > 0 else 0,
                },
            },
        }

    @staticmethod
    def _compute_streaks(condition: np.ndarray) -> tuple[int, int]:
        """Compute current and longest streak of True values."""
        if len(condition) == 0:
            return (0, 0)

        current = 0
        longest = 0
        for val in condition:
            if val:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        return (current, longest)

    def summary(self) -> str:
        """Return a formatted summary string."""
        s = self.get_analysis()
        lines = [
            f"{'='*50}",
            f"TRADE ANALYSIS",
            f"{'='*50}",
            f"Total Trades:     {s['total']['closed']}",
            f"Open Trades:      {s['total']['open']}",
            f"",
            f"Wins:             {s['won']['count']}",
            f"Losses:           {s['lost']['count']}",
            f"Win Rate:         {s['won']['count']/max(1,s['total']['closed'])*100:.1f}%",
            f"",
            f"Net PNL:          ${s['pnl']['net']['total']:.2f}",
            f"Avg PNL:          ${s['pnl']['net']['average']:.2f}",
            f"",
            f"Won Total:        ${s['won']['total']:.2f}",
            f"Won Avg:          ${s['won']['average']:.2f}",
            f"Won Max:          ${s['won']['max']:.2f}",
            f"",
            f"Lost Total:       ${s['lost']['total']:.2f}",
            f"Lost Avg:         ${s['lost']['average']:.2f}",
            f"Lost Max:         ${s['lost']['max']:.2f}",
            f"",
            f"Win Streak:       {s['streak']['won']['current']} (longest: {s['streak']['won']['longest']})",
            f"Loss Streak:      {s['streak']['lost']['current']} (longest: {s['streak']['lost']['longest']})",
            f"",
            f"Long Trades:      {s['long']['count']} (won: {s['long']['won']}, lost: {s['long']['lost']})",
            f"Short Trades:     {s['short']['count']} (won: {s['short']['won']}, lost: {s['short']['lost']})",
            f"",
            f"Avg Hold (bars):  {s['length']['average']:.1f}",
            f"Max Hold:         {s['length']['max']}",
            f"Min Hold:         {s['length']['min']}",
            f"{'='*50}",
        ]
        return "\n".join(lines)


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Create synthetic trades
    trades = [
        TradeRecord(pnl=50, direction="long", holding_bars=5, entry_price=1.085, exit_price=1.090),
        TradeRecord(pnl=-30, direction="short", holding_bars=3, entry_price=1.090, exit_price=1.087),
        TradeRecord(pnl=80, direction="long", holding_bars=8, entry_price=1.085, exit_price=1.093),
        TradeRecord(pnl=20, direction="long", holding_bars=2, entry_price=1.093, exit_price=1.095),
        TradeRecord(pnl=-15, direction="short", holding_bars=4, entry_price=1.095, exit_price=1.097),
        TradeRecord(pnl=60, direction="long", holding_bars=6, entry_price=1.085, exit_price=1.091),
    ]

    analyzer = TradeAnalyzer()
    for t in trades:
        analyzer.add_trade(t)

    stats = analyzer.get_analysis()
    print(analyzer.summary())

    # Assertions
    assert stats["total"]["closed"] == 6
    assert stats["won"]["count"] == 4
    assert stats["lost"]["count"] == 2
    assert abs(stats["pnl"]["net"]["total"] - 165.0) < 0.01
    assert stats["streak"]["won"]["longest"] >= 2  # at least 2 consecutive wins
    assert stats["long"]["count"] == 4
    assert stats["short"]["count"] == 2
    assert stats["length"]["max"] == 8
    assert stats["length"]["min"] == 2

    # Empty
    empty = TradeAnalyzer()
    empty_stats = empty.get_analysis()
    assert empty_stats["total"]["closed"] == 0

    print("\nTrade analyzer smoke test passed.")
