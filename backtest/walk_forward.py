# backtest/walk_forward.py
# ============================================================
# Walk-Forward Optimization Engine
# ============================================================
# Splits historical data into rolling windows:
#   In-Sample (IS) → optimize parameters
#   Out-of-Sample (OOS) → validate (no parameter changes)
#
# Measures Walk-Forward Efficiency (WFE) = OOS / IS
# WFE > 50% = strategy generalizes (not overfitted)
#
# ⚠️ Round-20 AUDIT FIX — LIMITATION DOCUMENTATION:
# ─────────────────────────────────────────────────
# The operator's institutional audit confirmed that this module does
# NOT perform true walk-forward OPTIMIZATION. It:
#   1. Runs the strategy ONCE with fixed parameters on the full dataset
#   2. Splits the resulting trades into IS/OOS windows AFTER the fact
#   3. Computes WFE = OOS_performance / IS_performance
#
# True walk-forward optimization would:
#   1. Fit strategy parameters on the IS window
#   2. Apply those fitted parameters to the OOS window
#   3. Roll forward and repeat (re-fit on each IS window)
#
# The current approach only checks "did performance degrade in the OOS
# period?" — it does NOT test whether the strategy was overfit to the
# IS period. A strategy that's equally bad in both IS and OOS would
# get WFE ≈ 100% (PASS), even though it's worthless.
#
# "WFE PASS" therefore means "performance didn't degrade over time",
# NOT "strategy is not overfitted". Interpret results accordingly.
#
# To implement true walk-forward optimization, the run() method would
# need to accept a param_optimizer callable that re-fits parameters
# on each IS window. This is a future enhancement.
# ============================================================

import logging
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """Single walk-forward window result."""
    window_num: int
    is_start: str
    is_end: str
    oos_start: str
    oos_end: str
    is_trades: int
    oos_trades: int
    is_pnl: float
    oos_pnl: float
    is_win_rate: float
    oos_win_rate: float
    wfe: float  # OOS_PnL / IS_PnL


@dataclass
class WalkForwardResult:
    """Complete walk-forward analysis result."""
    total_windows: int = 0
    total_is_pnl: float = 0.0
    total_oos_pnl: float = 0.0
    overall_wfe: float = 0.0
    pass_min_wfe: bool = False
    windows: List[WalkForwardWindow] = field(default_factory=list)
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "total_windows": self.total_windows,
            "total_is_pnl": self.total_is_pnl,
            "total_oos_pnl": self.total_oos_pnl,
            "overall_wfe": self.overall_wfe,
            "pass": self.pass_min_wfe,
            "windows": [
                {
                    "window": w.window_num,
                    "is_pnl": w.is_pnl,
                    "oos_pnl": w.oos_pnl,
                    "is_wr": w.is_win_rate,
                    "oos_wr": w.oos_win_rate,
                    "wfe": w.wfe,
                }
                for w in self.windows
            ],
            "detail": self.detail,
        }


def run_walk_forward(
    trades: List,
    train_pct: float = 0.7,
    n_windows: int = 5,
    min_wfe: float = 0.50,
) -> WalkForwardResult:
    """
    Run walk-forward analysis on closed trades.

    Args:
        trades: list of SimulatedTrade objects (chronological order)
        train_pct: fraction of each window for in-sample training
        n_windows: number of rolling windows
        min_wfe: minimum walk-forward efficiency to pass

    Returns:
        WalkForwardResult with per-window IS/OOS breakdown
    """
    if len(trades) < 30:
        return WalkForwardResult(
            detail=f"Insufficient trades ({len(trades)} < 30)"
        )

    # Sort by entry time (chronological)
    sorted_trades = sorted(trades, key=lambda t: t.entry_time)

    n = len(sorted_trades)
    window_size = n // n_windows
    train_size = int(window_size * train_pct)

    result = WalkForwardResult(total_windows=n_windows)
    total_is = 0.0
    total_oos = 0.0

    for w in range(n_windows):
        start = w * window_size
        end = min(start + window_size, n)
        window_trades = sorted_trades[start:end]

        is_trades = window_trades[:train_size]
        oos_trades = window_trades[train_size:]

        is_pnl = sum(t.pnl_usd for t in is_trades)
        oos_pnl = sum(t.pnl_usd for t in oos_trades)

        is_wins = sum(1 for t in is_trades if t.pnl_usd > 0)
        oos_wins = sum(1 for t in oos_trades if t.pnl_usd > 0)

        is_wr = is_wins / len(is_trades) * 100 if is_trades else 0
        oos_wr = oos_wins / len(oos_trades) * 100 if oos_trades else 0

        wfe = oos_pnl / is_pnl if is_pnl > 0 else 0

        wf_window = WalkForwardWindow(
            window_num=w + 1,
            is_start=is_trades[0].entry_time if is_trades else "",
            is_end=is_trades[-1].entry_time if is_trades else "",
            oos_start=oos_trades[0].entry_time if oos_trades else "",
            oos_end=oos_trades[-1].entry_time if oos_trades else "",
            is_trades=len(is_trades),
            oos_trades=len(oos_trades),
            is_pnl=round(is_pnl, 2),
            oos_pnl=round(oos_pnl, 2),
            is_win_rate=round(is_wr, 1),
            oos_win_rate=round(oos_wr, 1),
            wfe=round(wfe, 3),
        )
        result.windows.append(wf_window)
        total_is += is_pnl
        total_oos += oos_pnl

    result.total_is_pnl = round(total_is, 2)
    result.total_oos_pnl = round(total_oos, 2)
    result.overall_wfe = round(total_oos / total_is, 3) if total_is > 0 else 0
    result.pass_min_wfe = result.overall_wfe >= min_wfe and total_oos > 0

    result.detail = (
        f"IS P&L=${total_is:.2f}, OOS P&L=${total_oos:.2f}, "
        f"WFE={result.overall_wfe:.1%} "
        f"({'PASS' if result.pass_min_wfe else 'FAIL — overfitted or unprofitable OOS'})"
    )

    log.info(f"[WalkForward] {result.detail}")
    return result


def print_walk_forward_table(result: WalkForwardResult):
    """Print walk-forward results as a table."""
    print("\n" + "=" * 70)
    print("  WALK-FORWARD OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"  {'Window':<8} {'IS Trades':<12} {'IS P&L':<12} {'IS WR':<10} "
          f"{'OOS Trades':<12} {'OOS P&L':<12} {'OOS WR':<10} {'WFE':<8}")
    print("-" * 70)

    for w in result.windows:
        print(f"  {w.window_num:<8} {w.is_trades:<12} ${w.is_pnl:<10.2f} {w.is_win_rate:<10.1f} "
              f"{w.oos_trades:<12} ${w.oos_pnl:<10.2f} {w.oos_win_rate:<10.1f} {w.wfe:<8.1%}")

    print("-" * 70)
    print(f"  {'TOTAL':<8} {'':<12} ${result.total_is_pnl:<10.2f} {'':<10} "
          f"{'':<12} ${result.total_oos_pnl:<10.2f} {'':<10} {result.overall_wfe:<8.1%}")
    print(f"\n  Result: {'✅ PASS' if result.pass_min_wfe else '❌ FAIL'}")
    print(f"  Detail: {result.detail}")
    print("=" * 70)


# ============================================================
# CLI entry
# ============================================================

if __name__ == "__main__":
    from backtest.broker_sim import SimulatedTrade
    from datetime import datetime, timezone, timedelta

    # Generate fake trades
    np.random.seed(42)
    trades = []
    base_time = datetime(2023, 1, 1, tzinfo=timezone.utc)

    for i in range(100):
        is_win = np.random.random() < 0.6
        pnl = np.random.normal(40, 15) if is_win else np.random.normal(-25, 10)
        t = SimulatedTrade(
            trade_id=i, symbol="EURUSD", direction="BUY",
            entry_time=(base_time + timedelta(hours=i)).isoformat(),
            entry_price=1.0850, requested_entry=1.0850,
            stop_loss=1.0820, take_profit=1.0910,
            lot_size=0.1, pnl_pips=pnl, pnl_usd=pnl * 10,
        )
        trades.append(t)

    result = run_walk_forward(trades, n_windows=5)
    print_walk_forward_table(result)
