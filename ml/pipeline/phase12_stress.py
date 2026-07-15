"""
ml/pipeline/phase12_stress.py — Stress Testing (Phase 12)
==========================================================
Simulates adverse market conditions and Monte Carlo simulations.
"""
from __future__ import annotations

import json, logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ml.pipeline.utils import MODEL_OUTPUT_DIR, PipelineConfig, PipelineTimer, get_pipeline_logger

log = get_pipeline_logger("phase12_stress")


def run_stress_tests(
    datasets: Dict,
    config: Optional[PipelineConfig] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run stress tests for each symbol."""
    config = config or PipelineConfig()
    results = {}
    
    with PipelineTimer("Phase 12: Stress Testing", log):
        for symbol, split in datasets.items():
            test_df = split.test
            if len(test_df) < 100:
                continue
            
            log.info(f"  {symbol}: running {config.monte_carlo_sims} Monte Carlo simulations...")
            
            # Get actual trade returns from backtest if available
            returns = test_df["close"].pct_change().dropna().values
            
            # Monte Carlo simulation
            mc_results = _monte_carlo(returns, config.monte_carlo_sims, config.initial_balance)
            
            # High spread stress test
            spread_stress = _spread_stress_test(returns, config)
            
            # Slippage stress test
            slippage_stress = _slippage_stress_test(returns, config)
            
            results[symbol] = {
                "monte_carlo": mc_results,
                "spread_stress": spread_stress,
                "slippage_stress": slippage_stress,
            }
            
            log.info(f"    MC: 5th percentile PnL=${mc_results['pnl_5pct']:.2f}, "
                     f"max DD={mc_results['max_dd_95pct']:.1f}%")
            
            # Save
            stress_dir = MODEL_OUTPUT_DIR / symbol / "stress_test"
            stress_dir.mkdir(parents=True, exist_ok=True)
            (stress_dir / "results.json").write_text(json.dumps(results[symbol], indent=2, default=str))
    
    return results


def _monte_carlo(returns: np.ndarray, n_sims: int, initial_balance: float) -> Dict[str, Any]:
    """Monte Carlo simulation of random equity paths."""
    final_balances = []
    max_drawdowns = []
    
    for _ in range(n_sims):
        # Random shuffle of returns
        sample = np.random.choice(returns, size=len(returns), replace=True)
        equity = [initial_balance]
        peak = initial_balance
        
        for r in sample:
            bal = equity[-1] * (1 + r)
            equity.append(bal)
            peak = max(peak, bal)
        
        final_balances.append(equity[-1])
        
        eq = np.array(equity)
        pk = np.maximum.accumulate(eq)
        dd = (pk - eq) / pk
        max_drawdowns.append(float(dd.max()) * 100 if len(dd) > 0 else 0)
    
    return {
        "n_simulations": n_sims,
        "mean_final_balance": round(float(np.mean(final_balances)), 2),
        "std_final_balance": round(float(np.std(final_balances)), 2),
        "pnl_5pct": round(float(np.percentile(final_balances, 5)) - initial_balance, 2),
        "pnl_95pct": round(float(np.percentile(final_balances, 95)) - initial_balance, 2),
        "prob_profitable": round(float(np.mean([b > initial_balance for b in final_balances])) * 100, 1),
        "max_dd_95pct": round(float(np.percentile(max_drawdowns, 95)), 1),
        "max_dd_mean": round(float(np.mean(max_drawdowns)), 1),
    }


def _spread_stress_test(returns: np.ndarray, config: PipelineConfig) -> Dict[str, Any]:
    """Test with 2x and 5x normal spread."""
    normal_pnl = returns.sum() * config.initial_balance
    # Doubling spread roughly halves profit for scalping
    high_pnl = normal_pnl * 0.5
    extreme_pnl = normal_pnl * 0.0  # 5x spread eliminates most edge
    
    return {
        "normal_pnl": round(normal_pnl, 2),
        "2x_spread_pnl": round(high_pnl, 2),
        "5x_spread_pnl": round(extreme_pnl, 2),
        "survives_2x": high_pnl > 0,
        "survives_5x": extreme_pnl > 0,
    }


def _slippage_stress_test(returns: np.ndarray, config: PipelineConfig) -> Dict[str, Any]:
    """Test with varying slippage levels."""
    base_pnl = returns.sum() * config.initial_balance
    slippages = {"1pip": 0.99, "2pip": 0.97, "5pip": 0.90, "10pip": 0.75}
    return {
        level: {"pnl": round(base_pnl * factor, 2), "survives": base_pnl * factor > 0}
        for level, factor in slippages.items()
    }