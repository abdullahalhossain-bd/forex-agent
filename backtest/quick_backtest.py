# backtest/quick_backtest.py — high-level convenience wrapper
# ============================================================
# One-call backtest API for the common case: load candles, run a
# strategy, get a metrics dict. Wraps the existing backtest.unified_engine
# (preferred) and falls back to backtest.engine if the unified engine
# is unavailable.
#
# Usage:
#     from backtest.quick_backtest import quick_backtest
#     result = quick_backtest(
#         strategy=my_strategy_fn,
#         symbol="EURUSD",
#         timeframe="15m",
#         start="2024-01-01",
#         end="2024-06-30",
#         initial_capital=10_000,
#     )
#     print(result.metrics)
#
# This module is a thin facade — the real backtesting logic lives in
# backtest/unified_engine.py and backtest/engine.py. It exists because
# those modules require 20+ lines of setup; for ad-hoc research you
# usually just want the metrics.
# ============================================================

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from utils.logger import get_logger

log = get_logger("quick_backtest")


def quick_backtest(
    strategy: Callable[[Any], Dict[str, Any]],
    symbol: str = "EURUSD",
    timeframe: str = "15m",
    start: Optional[str] = None,
    end: Optional[str] = None,
    initial_capital: float = 10_000.0,
    spread_pips: float = 0.5,
    commission_per_lot: float = 7.0,
    **extra_kwargs: Any,
) -> Dict[str, Any]:
    """
    Run a backtest and return a metrics dict.

    Args:
        strategy: a callable that takes a candle DataFrame and returns
            a signal dict {"direction": "BUY"|""SELL"|"HOLD",
                            "sl": float, "tp": float, "lot": float}
            per row, OR a strategy object exposing .on_bar(df) -> signal.
        symbol: e.g. "EURUSD".
        timeframe: "15m", "1h", "4h", "1d".
        start/end: ISO date strings; if None, use whatever the data
            loader returns.
        initial_capital: starting account balance in account currency.
        spread_pips: bid/ask spread to model.
        commission_per_lot: round-turn commission per standard lot.
        extra_kwargs: forwarded to the underlying engine.

    Returns:
        {
            "metrics": {
                "total_return_pct": ...,
                "win_rate": ...,
                "profit_factor": ...,
                "max_drawdown_pct": ...,
                "sharpe": ...,
                "total_trades": ...,
            },
            "trades": [...],         # list of trade dicts
            "equity_curve": [...],   # list of (timestamp, equity) tuples
            "engine_used": "unified" | "legacy",
        }
    """
    # Try the unified engine first — it shares AITrader's decision core
    # and is the only engine that predicts live behavior accurately.
    try:
        from backtest.unified_engine import run_unified_backtest
        log.info(f"[QuickBT] Using unified engine for {symbol} {timeframe}")
        result = run_unified_backtest(
            strategy=strategy,
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            initial_capital=initial_capital,
            spread_pips=spread_pips,
            commission_per_lot=commission_per_lot,
            **extra_kwargs,
        )
        result.setdefault("engine_used", "unified")
        return result
    except ImportError:
        log.info("[QuickBT] unified_engine unavailable, falling back to legacy engine")
    except Exception as e:
        log.warning(f"[QuickBT] unified_engine failed: {e} — falling back to legacy engine")

    # Legacy fallback — does NOT share live decision core, results may
    # diverge from live behavior. Useful only for pure strategy-shape
    # research, not for predicting live P&L.
    try:
        from backtest.engine import BacktestEngine
        from backtest.data_loader import HistoricalDataLoader
        log.info(f"[QuickBT] Using legacy engine for {symbol} {timeframe}")
        loader = HistoricalDataLoader()
        df = loader.load(symbol, timeframe, start=start, end=end)
        engine = BacktestEngine(initial_capital=initial_capital,
                                spread_pips=spread_pips,
                                commission_per_lot=commission_per_lot)
        result = engine.run(df, strategy=strategy, **extra_kwargs)
        result.setdefault("engine_used", "legacy")
        return result
    except Exception as e:
        log.error(f"[QuickBT] Both engines failed: {e}")
        return {
            "metrics": {},
            "trades": [],
            "equity_curve": [],
            "engine_used": "none",
            "error": str(e),
        }


def quick_metrics(trades: list, equity_curve: list, initial_capital: float) -> Dict[str, float]:
    """
    Compute standard backtest metrics from a trade list + equity curve.

    Pure function — no I/O. Useful for custom backtest loops that don't
    fit the quick_backtest() shape.
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

    wins = [t for t in trades if t.get("pnl", 0) > 0]
    losses = [t for t in trades if t.get("pnl", 0) < 0]
    gross_win = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))

    final_equity = equity_curve[-1][1] if equity_curve else initial_capital
    total_return_pct = (final_equity - initial_capital) / initial_capital * 100

    # Max drawdown from equity curve
    peak = initial_capital
    max_dd = 0.0
    for _, eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    # Sharpe (simplified — assumes 0% risk-free, per-bar returns)
    try:
        import numpy as np
        eqs = [e for _, e in equity_curve]
        if len(eqs) > 1:
            rets = np.diff(eqs) / np.array(eqs[:-1])
            sharpe = float(np.mean(rets) / (np.std(rets) + 1e-9) * (252 ** 0.5))
        else:
            sharpe = 0.0
    except Exception:
        sharpe = 0.0

    return {
        "total_return_pct": round(total_return_pct, 2),
        "win_rate": round(len(wins) / len(trades) * 100, 2) if trades else 0.0,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf"),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "total_trades": len(trades),
    }
