"""
ml/pipeline/phase11_backtest.py — Backtesting Engine (Phase 11)
==============================================================
Runs realistic backtests with institutional-grade metrics.
"""
from __future__ import annotations

import json, logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ml.pipeline.utils import MODEL_OUTPUT_DIR, PipelineConfig, PipelineTimer, get_pipeline_logger

log = get_pipeline_logger("phase11_backtest")


def run_backtests(
    datasets: Dict,
    config: Optional[PipelineConfig] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Run backtest for each symbol using trained models on test data."""
    config = config or PipelineConfig()
    results = {}
    
    with PipelineTimer("Phase 11: Backtesting", log):
        for symbol, split in datasets.items():
            results[symbol] = {}
            test_df = split.test
            features = split.feature_columns
            
            X_test = np.nan_to_num(test_df[features].values, nan=0.0)
            
            for model_type in config.supervised_models:
                model_dir = MODEL_OUTPUT_DIR / symbol / model_type
                model_path = model_dir / "model.pkl"
                if not model_path.exists():
                    continue
                
                try:
                    # Round-19 audit fix: use safe_pickle instead of raw
                    # pickle.load. The rest of the codebase uses safe_pickle
                    # for security (arbitrary code execution protection).
                    # This was the one outlier that bypassed the hardening.
                    from utils.safe_pickle import safe_load
                    with model_path.open("rb") as f:
                        model = safe_load(f)
                    
                    predictions = model.predict(X_test)
                    metrics = _calculate_backtest_metrics(
                        test_df, predictions, symbol, config
                    )
                    results[symbol][model_type] = metrics
                    
                    # Save report
                    report_dir = MODEL_OUTPUT_DIR / symbol / model_type / "backtest"
                    report_dir.mkdir(parents=True, exist_ok=True)
                    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
                    
                    log.info(f"  {symbol} {model_type}: PnL=${metrics['net_profit']:.2f} "
                             f"Sharpe={metrics['sharpe_ratio']:.2f} WR={metrics['win_rate']:.1f}%")
                except Exception as e:
                    import traceback
                    log.warning(f"  {symbol} {model_type}: backtest failed -- {e}")
                    log.debug(traceback.format_exc())
    
    return results


def _calculate_backtest_metrics(df: pd.DataFrame, predictions, symbol: str, config: PipelineConfig) -> Dict[str, Any]:
    """Calculate institutional-grade backtest metrics."""
    # Ensure predictions is a flat 1D numpy array
    predictions = np.atleast_1d(np.asarray(predictions, dtype=int)).ravel()
    
    close = df["close"].values
    n = min(len(df) - 1, len(predictions))
    
    if n < 10:
        return {"error": "insufficient_test_data", "total_trades": 0, "net_profit": 0,
                "sharpe_ratio": 0, "max_drawdown_pct": 0, "win_rate": 0,
                "profit_factor": 0, "sortino_ratio": 0, "avg_rr_ratio": 0,
                "expectancy": 0, "recovery_factor": 0, "avg_holding_candles": 0,
                "final_balance": config.initial_balance}
    
    # Simulate trading based on predictions
    pip_mult = 10000 if not symbol.endswith("JPY") and symbol != "XAUUSD" else 100
    
    balance = config.initial_balance
    peak = balance
    equity_curve = [balance]
    trades = []
    position = None
    
    for i in range(n):
        pred = int(predictions[i])
        price = float(close[i])
        next_price = float(close[i + 1])
        
        # Close existing position
        if position is not None:
            pnl_pips = (next_price - position["entry"]) * position["dir"] * pip_mult
            pnl_usd = pnl_pips * position["lot"] * 10
            balance += pnl_usd
            trades.append({"pnl": pnl_usd, "pips": pnl_pips, "holding": i - position["step"]})
            position = None
        
        # Open new position on BUY/SELL signal
        if pred in (1, 2):
            direction = 1 if pred == 1 else -1
            lot = 0.1
            position = {"entry": price, "dir": direction, "lot": lot, "step": i}
        
        # Calculate equity
        if position is not None:
            unrealized = (next_price - position["entry"]) * position["dir"] * pip_mult * position["lot"] * 10
            equity = balance + unrealized
        else:
            equity = balance
        equity_curve.append(equity)
        peak = max(peak, equity)
    
    # Calculate metrics
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    
    net_profit = sum(pnls)
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    win_rate = (len(wins) / len(trades) * 100) if trades else 0
    
    # FIX (2026-08-11): Compute Sharpe/Sortino from per-trade fractional
    # returns (PnL / equity-before-trade), not from absolute USD PnL.
    # Using absolute USD PnL makes the ratio scale linearly with lot size
    # and gives a t-statistic, not a Sharpe ratio. Also use correct
    # annualization: divide by sqrt(tpy), not multiply.
    BARS_PER_DAY = {"M1": 960, "M5": 288, "M15": 96, "H1": 24, "H4": 6, "D1": 1}
    timeframe = getattr(config, "timeframe", "M15") or "M15"
    tpy = 252 * BARS_PER_DAY.get(timeframe, 96)
    if len(pnls) > 1:
        eq_before = np.concatenate([[config.initial_balance], equity_curve[:-1]])
        eq_before_safe = np.where(np.asarray(eq_before) > 0, np.asarray(eq_before), 1.0)
        rets_arr = np.asarray(pnls) / eq_before_safe
        ar = float(np.mean(rets_arr))
        sr = float(np.std(rets_arr, ddof=1))
        sharpe = (ar * tpy) / (sr * np.sqrt(tpy)) if sr > 0 else 0
    else:
        sharpe = 0
        rets_arr = np.asarray(pnls) / max(config.initial_balance, 1.0)

    # Sortino
    downside = rets_arr[rets_arr < 0]
    dstd = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
    ar = float(np.mean(rets_arr)) if len(rets_arr) > 0 else 0.0
    sortino = (ar * tpy) / (dstd * np.sqrt(tpy)) if dstd > 0 else 0
    
    # Max drawdown
    eq = np.array(equity_curve)
    peak_arr = np.maximum.accumulate(eq)
    dd = (peak_arr - eq) / peak_arr
    max_dd = float(dd.max()) * 100 if len(dd) > 0 else 0
    
    avg_rr = np.mean([t["pips"] for t in wins]) / abs(np.mean([t["pips"] for t in losses])) if wins and losses else 0
    expectancy = net_profit / len(trades) if trades else 0
    recovery_factor = net_profit / (max_dd * config.initial_balance / 100) if max_dd > 0 else 0
    
    avg_holding = np.mean([t["holding"] for t in trades]) if trades else 0
    
    return {
        "net_profit": round(net_profit, 2),
        "profit_factor": round(profit_factor, 2),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "win_rate": round(win_rate, 1),
        "total_trades": len(trades),
        "avg_rr_ratio": round(avg_rr, 2),
        "expectancy": round(expectancy, 2),
        "recovery_factor": round(recovery_factor, 2),
        "avg_holding_candles": round(avg_holding, 1),
        "final_balance": round(balance, 2),
    }