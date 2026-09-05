"""Compatibility entry point for the research live-mirroring replay.

The canonical research path is:
    historical data -> AITrader decision core -> canonical execution adapter
    -> historical position monitor -> deterministic P/L

This facade keeps existing callers working without restoring the legacy
BrokerSimulator lifecycle or introducing an independent strategy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class CompatibilityMetrics:
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_pnl_usd: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    avg_win_usd: float = 0.0
    avg_loss_usd: float = 0.0


@dataclass
class UnifiedBacktestResult:
    symbol: str
    timeframe: str
    bars: int
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)
    rejection_stats: dict = field(default_factory=dict)
    metrics: CompatibilityMetrics = field(default_factory=CompatibilityMetrics)
    error: Optional[str] = None
    forensics_path: Optional[str] = None


def _metrics_from_lifecycles(lifecycles: list[dict], starting_balance: float) -> CompatibilityMetrics:
    """Compute descriptive metrics only; never optimize the strategy."""
    closed = [x for x in lifecycles if x.get("status") == "CLOSED" and x.get("pnl_usd") is not None]
    if not closed:
        return CompatibilityMetrics()
    pnls = [float(x["pnl_usd"]) for x in closed]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    equity = float(starting_balance)
    peak = equity
    max_dd = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak * 100.0)
    mean = sum(pnls) / len(pnls)
    variance = sum((p - mean) ** 2 for p in pnls) / len(pnls)
    std = variance ** 0.5
    downside = [p for p in pnls if p < 0]
    downside_std = (sum(p * p for p in downside) / len(downside)) ** 0.5 if downside else 0.0
    return CompatibilityMetrics(
        total_trades=len(closed),
        win_rate=100.0 * len(wins) / len(closed),
        profit_factor=(gross_win / gross_loss) if gross_loss else (float("inf") if gross_win else 0.0),
        total_pnl_usd=sum(pnls),
        max_drawdown_pct=max_dd,
        sharpe_ratio=(mean / std) * (len(pnls) ** 0.5) if std else 0.0,
        sortino_ratio=(mean / downside_std) * (len(pnls) ** 0.5) if downside_std else 0.0,
        avg_win_usd=(gross_win / len(wins)) if wins else 0.0,
        avg_loss_usd=(sum(losses) / len(losses)) if losses else 0.0,
    )


def run_unified_backtest(
    symbol: str,
    df: pd.DataFrame,
    timeframe: str = "H1",
    starting_balance: float = 10000.0,
    warmup_bars: int = 300,
    max_open_trades: int | None = None,
    max_hold_bars: int = 100,
    spread_pips: Optional[float] = None,
    commission_per_lot: float | None = None,
    slippage_pips: float | None = None,
    db_path: str = "backtest/backtest_run.db",
    verbose: bool = False,
    save_forensics: bool = True,
    forensics_path: Optional[str] = None,
    bypass_checks: set[str] | list[str] | None = None,
    clock=None,
    **_ignored,
) -> UnifiedBacktestResult:
    """Compatibility wrapper around the canonical live-mirroring runner."""
    from backtest.live_mirroring_runner import run_live_mirroring_replay

    result = run_live_mirroring_replay(
        symbol=symbol,
        timeframe=timeframe,
        df=df,
        starting_balance=starting_balance,
        warmup_bars=warmup_bars,
        max_open_trades=max_open_trades,
        max_hold_bars=max_hold_bars,
        spread_pips=float(spread_pips or 0.0),
        slippage_pips=float(slippage_pips or 0.0),
        commission_per_lot=float(commission_per_lot or 0.0),
        db_path=db_path,
        clock=clock,
        bypass_checks=bypass_checks,
    )
    lifecycle_dicts = [record.lifecycle for record in result.trades]
    closed = [x for x in lifecycle_dicts if x.get("status") == "CLOSED"]
    return UnifiedBacktestResult(
        symbol=result.symbol,
        timeframe=result.timeframe,
        bars=result.bars_seen,
        trades=lifecycle_dicts,
        equity_curve=[],
        rejection_stats={
            **result.rejection_stats,
            "closed_trades": len(closed),
            "open_trades": len(result.open_trade_ids),
            "legacy_broker_simulator": "DISABLED",
            "canonical_execution": "ENABLED",
        },
        metrics=_metrics_from_lifecycles(lifecycle_dicts, starting_balance),
        error=None,
        forensics_path=forensics_path,
    )
