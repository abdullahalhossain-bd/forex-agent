"""
ml/pipeline/phase13_model_selection.py — Model Selection (Phase 13)
====================================================================
Selects the best model using composite scoring:
  Profit, Drawdown, Sharpe, Consistency, Risk, Generalization
"""
from __future__ import annotations

import json, logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ml.pipeline.utils import MODEL_OUTPUT_DIR, PipelineConfig, PipelineTimer, get_pipeline_logger

log = get_pipeline_logger("phase13_selection")


def select_best_models(
    backtest_results: Dict[str, Dict[str, Dict]],
    walkforward_results: Dict[str, List[Dict]],
    config: Optional[PipelineConfig] = None,
) -> Dict[str, Dict[str, Any]]:
    """Select best model per symbol using composite score."""
    config = config or PipelineConfig()
    best = {}
    
    with PipelineTimer("Phase 13: Model Selection", log):
        for symbol, models in backtest_results.items():
            scores = {}
            for model_type, metrics in models.items():
                score = _composite_score(metrics, walkforward_results.get(symbol, []))
                scores[model_type] = {"score": score, "metrics": metrics}
                log.info(f"  {symbol} {model_type}: composite_score={score:.3f}")
            
            if scores:
                winner = max(scores, key=lambda k: scores[k]["score"])
                best[symbol] = {
                    "best_model": winner,
                    "score": scores[winner]["score"],
                    "all_scores": {k: v["score"] for k, v in scores.items()},
                    "best_metrics": scores[winner]["metrics"],
                }
                
                # Save best_model.json
                out_dir = MODEL_OUTPUT_DIR / symbol
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "best_model.json").write_text(json.dumps(best[symbol], indent=2, default=str))
                log.info(f"  >>> {symbol} BEST: {winner} (score={best[symbol]['score']:.3f})")
    
    return best


def _composite_score(metrics: Dict, wf_results: List[Dict], weights=None) -> float:
    """Calculate composite model score (0-100)."""
    if weights is None:
        weights = {
            "profit": 0.25, "drawdown": 0.20, "sharpe": 0.15,
            "consistency": 0.15, "win_rate": 0.10, "generalization": 0.15,
        }
    
    # Normalize each component to 0-100
    profit_score = min(100, max(0, metrics.get("net_profit", 0) / 100 * 10))
    
    dd = metrics.get("max_drawdown_pct", 100)
    dd_score = max(0, 100 - dd * 2)  # 0% DD = 100, 50% DD = 0
    
    sharpe = metrics.get("sharpe_ratio", 0)
    sharpe_score = min(100, max(0, sharpe * 10 + 50))
    
    wr = metrics.get("win_rate", 0)
    wr_score = min(100, max(0, wr * 1.5))
    
    # Consistency: recovery factor
    rf = metrics.get("recovery_factor", 0)
    consistency_score = min(100, max(0, rf * 10 + 50))
    
    # Generalization: walk-forward consistency
    if wf_results:
        accs = [f.get("accuracy", 0) for f in wf_results]
        gen_score = min(100, max(0, np.mean(accs) * 100))
    else:
        gen_score = 50  # Neutral if no WF data
    
    score = (
        weights["profit"] * profit_score +
        weights["drawdown"] * dd_score +
        weights["sharpe"] * sharpe_score +
        weights["consistency"] * consistency_score +
        weights["win_rate"] * wr_score +
        weights["generalization"] * gen_score
    )
    
    return round(score, 2)