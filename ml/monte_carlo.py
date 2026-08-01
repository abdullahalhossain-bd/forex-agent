"""
ml/monte_carlo.py — Monte Carlo Trade Simulation (Day 72)
===========================================================

Tests whether a strategy's backtested results are due to skill or luck.

Process:
  1. Take the original trade sequence (win/loss list)
  2. Randomly shuffle the order 1000 times
  3. For each shuffle, compute: total return, max drawdown, survival
  4. Compare original results to the distribution of shuffled results

If the original strategy's profit factor is in the top 5% of shuffled
simulations → it's skill, not luck. If it's average → it's luck.

Output:
    {
        "simulations": 1000,
        "original_profit_factor": 1.8,
        "average_profit_factor": 1.2,
        "percentile": 92,           # original is better than 92% of random
        "worst_drawdown": 18%,
        "survival_rate": 94,        # 94% of sims ended profitable
        "probability_of_ruin": 2,   # 2% of sims blew the account
        "score": 85,
        "passed": True,
    }
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np

from utils.logger import get_logger

log = get_logger("monte_carlo")


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation results."""
    simulations: int = 0
    original_profit_factor: float = 0.0
    average_profit_factor: float = 0.0
    median_profit_factor: float = 0.0
    percentile: float = 0.0          # original's percentile among sims
    worst_drawdown_pct: float = 0.0  # worst case across all sims
    average_drawdown_pct: float = 0.0
    survival_rate: float = 0.0       # % of sims that ended profitable
    probability_of_ruin: float = 0.0 # % of sims that blew account
    score: float = 0.0
    passed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in asdict(self).items()}


class MonteCarloSimulator:
    """Monte Carlo trade-sequence simulation."""

    def __init__(self, n_simulations: int = 1000, ruin_threshold: float = -0.50):
        """Args:
            n_simulations: Number of shuffled simulations.
            ruin_threshold: Account loss % that counts as "ruin" (default -50%).
        """
        self.n_simulations = n_simulations
        self.ruin_threshold = ruin_threshold

    def simulate(
        self,
        trade_results: List[float],
        initial_balance: float = 10000.0,
    ) -> MonteCarloResult:
        """Run Monte Carlo simulation on a list of trade PnL values.

        Args:
            trade_results: List of dollar PnL per trade (e.g. [50, -30, 80, -20, ...]).
            initial_balance: Starting account balance.

        Returns:
            MonteCarloResult with simulation statistics.
        """
        result = MonteCarloResult()
        if not trade_results or len(trade_results) < 10:
            log.warning("[MonteCarlo] need ≥10 trades, got %d", len(trade_results))
            return result

        trades = np.array(trade_results, dtype=float)
        n_trades = len(trades)

        # Original metrics
        original_pf = self._profit_factor(trades)
        original_dd = self._max_drawdown_pct(trades, initial_balance)

        # Run simulations
        sim_pfs: List[float] = []
        sim_dds: List[float] = []
        sim_final_balances: List[float] = []
        ruin_count = 0
        survive_count = 0

        rng = np.random.RandomState(42)
        for i in range(self.n_simulations):
            shuffled = rng.permutation(trades)
            sim_pf = self._profit_factor(shuffled)
            sim_dd = self._max_drawdown_pct(shuffled, initial_balance)
            final_balance = initial_balance + np.sum(shuffled)

            sim_pfs.append(sim_pf)
            sim_dds.append(sim_dd)
            sim_final_balances.append(final_balance)

            if final_balance < initial_balance * (1 + self.ruin_threshold):
                ruin_count += 1
            if final_balance > initial_balance:
                survive_count += 1

        # Aggregate
        result.simulations = self.n_simulations
        result.original_profit_factor = round(original_pf, 3)
        result.average_profit_factor = round(float(np.mean(sim_pfs)), 3)
        result.median_profit_factor = round(float(np.median(sim_pfs)), 3)
        result.worst_drawdown_pct = round(float(np.max(sim_dds)) * 100, 1)
        result.average_drawdown_pct = round(float(np.mean(sim_dds)) * 100, 1)
        result.survival_rate = round((survive_count / self.n_simulations) * 100, 1)
        result.probability_of_ruin = round((ruin_count / self.n_simulations) * 100, 1)

        # Percentile: compare original's max drawdown to shuffled sims.
        # If original drawdown is LOWER than most shuffles → good trade ordering.
        # PF is order-independent so we use drawdown instead.
        sim_dds_arr = np.array(sim_dds)
        result.percentile = round(float(np.mean(sim_dds_arr >= original_dd)) * 100, 1)
        # percentile=90 means original DD is lower (better) than 90% of shuffles

        # Score: higher percentile + higher survival + lower ruin = better
        percentile_score = result.percentile  # 0-100
        survival_score = result.survival_rate  # 0-100
        ruin_penalty = result.probability_of_ruin  # 0-100
        result.score = max(0, min(100,
            percentile_score * 0.4 + survival_score * 0.4 - ruin_penalty * 0.2
        ))
        result.passed = (
            result.score >= 60
            and result.percentile >= 70
            and result.probability_of_ruin <= 5
        )

        log.info(
            f"[MonteCarlo] {self.n_simulations} sims | "
            f"orig PF={original_pf:.2f} (percentile={result.percentile}) | "
            f"survival={result.survival_rate}% ruin={result.probability_of_ruin}% | "
            f"score={result.score:.1f} passed={result.passed}"
        )
        return result

    def _profit_factor(self, trades: np.ndarray) -> float:
        """Gross profit / gross loss."""
        gains = trades[trades > 0].sum()
        losses = abs(trades[trades < 0].sum())
        if losses == 0:
            return float("inf") if gains > 0 else 0.0
        return float(gains / losses)

    def _max_drawdown_pct(self, trades: np.ndarray, initial: float) -> float:
        """Max drawdown as a fraction (0-1)."""
        equity = initial
        peak = initial
        max_dd = 0.0
        for t in trades:
            equity += t
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        return max_dd


# ── Singleton ───────────────────────────────────────────────────────

_SIM: Optional[MonteCarloSimulator] = None


def get_monte_carlo_simulator() -> MonteCarloSimulator:
    global _SIM
    if _SIM is None:
        _SIM = MonteCarloSimulator()
    return _SIM
