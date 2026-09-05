"""backtest.quick_backtest — strict convenience facade.

The previous version called `run_unified_backtest()` with arguments from an
older API (`strategy`, `start`, `end`, `initial_capital`) that no longer exist.
It then silently fell back to the legacy BacktestEngine, which meant a caller
could believe they had run a live-mirroring backtest while actually running a
different strategy engine.

This facade now has one policy: use the strict live-mirror engine only. There
is no silent legacy fallback.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

import pandas as pd

from utils.logger import get_logger

log = get_logger("quick_backtest")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return str(value)
    return value


def _load_input_df(
    *,
    df: Optional[pd.DataFrame],
    csv_path: Optional[str],
    start: Optional[str],
    end: Optional[str],
) -> pd.DataFrame:
    if df is None and csv_path is None:
        raise ValueError(
            "quick_backtest requires `df=` or `csv_path=`. "
            "The old implicit legacy loader has been removed because it "
            "could silently run a non-live-parity engine."
        )

    if df is None:
        from backtest.data_loader import HistoricalDataLoader
        df = HistoricalDataLoader().load_csv(
            csv_path, pair="UNKNOWN", timeframe="15m", enrich=False
        )
    else:
        df = df.copy(deep=True)

    if start is not None:
        start_ts = pd.Timestamp(start)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")
        df = df.loc[df.index >= start_ts]
    if end is not None:
        end_ts = pd.Timestamp(end)
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        else:
            end_ts = end_ts.tz_convert("UTC")
        df = df.loc[df.index <= end_ts]
    return df


def quick_backtest(
    strategy: Any = None,
    symbol: str = "EURUSD",
    timeframe: str = "15m",
    start: Optional[str] = None,
    end: Optional[str] = None,
    initial_capital: float = 10_000.0,
    spread_pips: Optional[float] = None,
    commission_per_lot: Optional[float] = None,
    df: Optional[pd.DataFrame] = None,
    csv_path: Optional[str] = None,
    **extra_kwargs: Any,
) -> Dict[str, Any]:
    """Run the strict live-trading-mirror backtest.

    `strategy` is retained only for source compatibility and is deliberately
    ignored: a live-mirroring run must use the production AITrader decision
    kernel, not a caller-supplied alternate strategy.

    Supply either `df=` or `csv_path=`. `start`/`end` are applied before the
    strict replay validator. `extra_kwargs` are forwarded only to the strict
    engine, so unsupported parameters fail loudly instead of selecting a
    different engine.
    """
    if strategy is not None:
        log.warning(
            "[QuickBT] `strategy` is ignored in live-mirror mode; "
            "AITrader.evaluate_decision_core is the only decision kernel."
        )

    data = _load_input_df(df=df, csv_path=csv_path, start=start, end=end)

    from backtest.live_mirror import run_live_mirror_backtest

    result = run_live_mirror_backtest(
        symbol=symbol,
        df=data,
        timeframe=timeframe,
        starting_balance=float(initial_capital),
        spread_pips=spread_pips,
        commission_per_lot=commission_per_lot,
        **extra_kwargs,
    )

    if hasattr(result, "metrics"):
        metrics = _jsonable(result.metrics)
        trades = _jsonable(result.trades)
        equity_curve = _jsonable(result.equity_curve)
        return {
            "metrics": metrics,
            "trades": trades,
            "equity_curve": equity_curve,
            "rejection_stats": _jsonable(result.rejection_stats),
            "engine_used": "live_mirror",
            "forensics_path": result.forensics_path,
            "error": result.error,
        }

    return {"engine_used": "live_mirror", "result": _jsonable(result)}


def quick_metrics(
    trades: list,
    equity_curve: list,
    initial_capital: float,
) -> Dict[str, float]:
    """Compute basic metrics from a trade list + equity curve.

    Kept as a pure utility for callers that already have serialized trades.
    """
    if not trades:
        return {
            "total_return_pct": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe": 0.0,
            "total_trades": 0,
        }

    def _pnl(t: Any) -> float:
        if isinstance(t, dict):
            return float(t.get("pnl", t.get("pnl_usd", 0.0)) or 0.0)
        return float(getattr(t, "pnl", getattr(t, "pnl_usd", 0.0)) or 0.0)

    pnls = [_pnl(t) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))

    final_equity = (
        equity_curve[-1][1]
        if equity_curve and isinstance(equity_curve[-1], (tuple, list))
        else (equity_curve[-1] if equity_curve else initial_capital)
    )
    total_return_pct = (final_equity - initial_capital) / initial_capital * 100

    peak = float(initial_capital)
    max_dd = 0.0
    for item in equity_curve:
        eq = float(item[1] if isinstance(item, (tuple, list)) else item)
        peak = max(peak, eq)
        if peak > 0:
            max_dd = max(max_dd, (peak - eq) / peak * 100)

    try:
        import numpy as np
        eqs = [float(item[1] if isinstance(item, (tuple, list)) else item) for item in equity_curve]
        rets = np.diff(eqs) / np.array(eqs[:-1]) if len(eqs) > 1 else np.array([])
        sharpe = float(np.mean(rets) / (np.std(rets) + 1e-9) * (252 ** 0.5)) if len(rets) else 0.0
    except Exception:
        sharpe = 0.0

    return {
        "total_return_pct": round(total_return_pct, 2),
        "win_rate": round(len(wins) / len(pnls) * 100, 2),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss else float("inf"),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "total_trades": len(pnls),
    }
