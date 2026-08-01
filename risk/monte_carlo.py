# risk/monte_carlo.py — Day 58 | Monte Carlo Simulation Engine
# ============================================================
# Simulates thousands of possible trading outcomes to understand
# risk of ruin, expected returns, and worst-case scenarios.
#
# Uses:
#   - Risk of Ruin calculation
#   - Equity curve simulation
#   - Percentile analysis (5th, 50th, 95th)
#   - Survival rate estimation
#
# This helps the AI make informed decisions about:
#   - How much to risk per trade
#   - Whether the current strategy is safe
#   - What the expected drawdown could be
# ============================================================

import random
import math
from typing import Optional
from utils.logger import get_logger

log = get_logger("monte_carlo")


class MonteCarloEngine:
    """
    Monte Carlo Simulation Engine for Trading Risk Analysis.

    AI-এর Risk Simulation Brain.
    Thousands of possible trading outcomes simulate করে
    future-এর risk বুঝতে সাহায্য করে।

    How it works:
      1. Take current strategy statistics (win rate, avg win, avg loss)
      2. Simulate N trading paths (e.g., 10,000 paths)
      3. Each path has M trades (e.g., 100 trades)
      4. Every trade: random WIN/LOSS based on win rate
      5. Track equity curve for each path
      6. Analyze distribution of final equities

    Outputs:
      - Worst case / Average / Best case returns
      - Risk of Ruin (probability of losing 50% of capital)
      - Survival Rate (probability of keeping >50% capital)
      - Percentile analysis
      - Maximum drawdown distribution

    Usage:
        mc = MonteCarloEngine()
        result = mc.run(
            win_rate=0.55,
            avg_win_pct=1.5,
            avg_loss_pct=1.0,
            n_simulations=10000,
            n_trades=100,
            initial_balance=10000,
            risk_per_trade=0.01,
        )
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Args:
            seed: Random seed for reproducibility (None = random).
        """
        self.rng = random.Random(seed)
        log.info("[MonteCarlo] Engine initialized")

    def run(
        self,
        win_rate: float = 0.55,
        avg_win_pct: float = 1.5,
        avg_loss_pct: float = 1.0,
        n_simulations: int = 10000,
        n_trades: int = 100,
        initial_balance: float = 10000.0,
        risk_per_trade: float = 0.01,
        ruin_threshold: float = 0.5,
    ) -> dict:
        """
        Run Monte Carlo simulation.

        Args:
            win_rate: Probability of winning (0.0 - 1.0)
            avg_win_pct: Average winning trade return as % of balance
            avg_loss_pct: Average losing trade return as % of balance
            n_simulations: Number of simulation paths (e.g., 10000)
            n_trades: Number of trades per path (e.g., 100)
            initial_balance: Starting balance
            risk_per_trade: Risk per trade as fraction (e.g., 0.01 = 1%)
            ruin_threshold: Ruin defined as balance < this fraction of initial

        Returns:
            Complete simulation results dict.
        """
        if win_rate <= 0 or win_rate >= 1:
            return self._empty_result()

        final_equities: list[float] = []
        max_drawdowns: list[float] = []
        ruin_count = 0
        worst_path: list[float] = []
        best_path: list[float] = []

        worst_final = float("inf")
        best_final = float("-inf")

        for sim_idx in range(n_simulations):
            balance = initial_balance
            peak = initial_balance
            path_max_dd = 0.0
            path = [balance]

            for _ in range(n_trades):
                if self.rng.random() < win_rate:
                    # WIN
                    gain = balance * risk_per_trade * avg_win_pct
                    balance += gain
                else:
                    # LOSS
                    loss = balance * risk_per_trade * avg_loss_pct
                    balance -= loss

                balance = max(0, balance)
                path.append(balance)

                # Track peak and drawdown
                if balance > peak:
                    peak = balance
                dd = (peak - balance) / peak * 100 if peak > 0 else 0
                if dd > path_max_dd:
                    path_max_dd = dd

            final_equities.append(balance)
            max_drawdowns.append(path_max_dd)

            # Check ruin
            if balance < initial_balance * ruin_threshold:
                ruin_count += 1

            # Track best/worst paths
            if balance < worst_final:
                worst_final = balance
                worst_path = path
            if balance > best_final:
                best_final = balance
                best_path = path

        # Calculate statistics
        final_equities.sort()
        max_drawdowns.sort()

        n = len(final_equities)
        median_idx = n // 2
        p5_idx = max(0, int(n * 0.05) - 1)
        p95_idx = min(n - 1, int(n * 0.95))

        median_final = final_equities[median_idx]
        p5_final = final_equities[p5_idx]
        p95_final = final_equities[p95_idx]

        avg_final = sum(final_equities) / n
        avg_dd = sum(max_drawdowns) / n
        worst_dd = max_drawdowns[-1]  # worst drawdown across all sims
        p95_dd = max_drawdowns[min(n-1, int(n * 0.95))]

        # Calculate return percentages
        avg_return_pct = (avg_final - initial_balance) / initial_balance * 100
        median_return_pct = (median_final - initial_balance) / initial_balance * 100
        worst_return_pct = (final_equities[0] - initial_balance) / initial_balance * 100
        best_return_pct = (final_equities[-1] - initial_balance) / initial_balance * 100

        survival_rate = 1.0 - (ruin_count / n_simulations)
        risk_of_ruin = ruin_count / n_simulations

        # Expected value per trade
        ev_per_trade = (
            win_rate * avg_win_pct * risk_per_trade
            - (1 - win_rate) * avg_loss_pct * risk_per_trade
        )

        result = {
            "n_simulations": n_simulations,
            "n_trades": n_trades,
            "win_rate": win_rate,
            "avg_win_pct": avg_win_pct,
            "avg_loss_pct": avg_loss_pct,
            "risk_per_trade": risk_per_trade,
            "initial_balance": initial_balance,
            # Return statistics
            "average_pct": round(avg_return_pct, 1),
            "average_usd": round(avg_final, 2),
            "median_pct": round(median_return_pct, 1),
            "median_usd": round(median_final, 2),
            "worst_case_pct": round(worst_return_pct, 1),
            "worst_case_usd": round(final_equities[0], 2),
            "best_case_pct": round(best_return_pct, 1),
            "best_case_usd": round(final_equities[-1], 2),
            "percentile_5_pct": round(
                (p5_final - initial_balance) / initial_balance * 100, 1
            ),
            "percentile_95_pct": round(
                (p95_final - initial_balance) / initial_balance * 100, 1
            ),
            # Risk metrics
            "survival_rate": round(survival_rate, 4),
            "risk_of_ruin": round(risk_of_ruin, 4),
            "avg_max_drawdown_pct": round(avg_dd, 1),
            "worst_max_drawdown_pct": round(worst_dd, 1),
            "percentile_95_drawdown_pct": round(p95_dd, 1),
            "ev_per_trade_pct": round(ev_per_trade * 100, 3),
            # Path samples
            "worst_path_sample": [round(v, 2) for v in worst_path[::max(1, n_trades//10)]],
            "best_path_sample": [round(v, 2) for v in best_path[::max(1, n_trades//10)]],
        }

        log.info(
            f"[MonteCarlo] {n_simulations} simulations x {n_trades} trades | "
            f"Avg: {avg_return_pct:+.1f}% | "
            f"Worst: {worst_return_pct:+.1f}% | "
            f"Survival: {survival_rate*100:.1f}% | "
            f"Risk of Ruin: {risk_of_ruin*100:.1f}%"
        )

        return result

    def calculate_risk_of_ruin(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        risk_per_trade: float,
        max_trades: int = 500,
    ) -> float:
        """
        Quick risk of ruin calculation using simulation.

        Risk of Ruin = probability that balance falls below 50% of initial.
        """
        result = self.run(
            win_rate=win_rate,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            n_simulations=5000,  # faster, fewer sims
            n_trades=max_trades,
            initial_balance=10000,
            risk_per_trade=risk_per_trade,
            ruin_threshold=0.5,
        )
        return result["risk_of_ruin"]

    def find_optimal_risk(
        self,
        win_rate: float,
        avg_win_pct: float,
        avg_loss_pct: float,
        n_simulations: int = 5000,
        n_trades: int = 100,
        initial_balance: float = 10000,
        min_survival: float = 0.95,
    ) -> dict:
        """
        Find optimal risk per trade that maximizes returns
        while maintaining minimum survival rate.

        Tests risk levels from 0.1% to 5% in 0.1% increments.

        Returns:
            {"optimal_risk_pct": 1.2, "expected_return": 18.5, ...}
        """
        best_risk = 0.001
        best_return = float("-inf")
        results = []

        for risk_pct_10 in range(1, 51):  # 0.1% to 5.0%
            risk = risk_pct_10 / 1000
            result = self.run(
                win_rate=win_rate,
                avg_win_pct=avg_win_pct,
                avg_loss_pct=avg_loss_pct,
                n_simulations=n_simulations,
                n_trades=n_trades,
                initial_balance=initial_balance,
                risk_per_trade=risk,
            )

            results.append({
                "risk_pct": risk * 100,
                "expected_return": result["average_pct"],
                "survival_rate": result["survival_rate"],
                "risk_of_ruin": result["risk_of_ruin"],
                "worst_case": result["worst_case_pct"],
            })

            if result["survival_rate"] >= min_survival:
                if result["average_pct"] > best_return:
                    best_return = result["average_pct"]
                    best_risk = risk

        # Sort by risk for reference
        results.sort(key=lambda x: x["risk_pct"])

        log.info(
            f"[MonteCarlo] Optimal risk: {best_risk*100:.1f}% "
            f"(expected return: {best_return:+.1f}%, "
            f"min survival: {min_survival*100:.0f}%)"
        )

        return {
            "optimal_risk_pct": round(best_risk * 100, 1),
            "optimal_risk_fraction": round(best_risk, 4),
            "expected_return": round(best_return, 1),
            "all_results": results[:10],  # top 10 results
        }

    def _empty_result(self) -> dict:
        """Return empty result for invalid inputs."""
        return {
            "n_simulations": 0, "n_trades": 0,
            "average_pct": 0, "worst_case_pct": 0,
            "best_case_pct": 0, "survival_rate": 0,
            "risk_of_ruin": 1.0,
        }

    def print_simulation_result(self, result: dict) -> None:
        """Print formatted simulation results."""
        bar = "=" * 52
        print(f"\n{bar}")
        print("  MONTE CARLO SIMULATION RESULTS")
        print(bar)
        print(f"  Simulations       : {result['n_simulations']:,}")
        print(f"  Trades per path   : {result['n_trades']}")
        print(f"  Win Rate          : {result['win_rate']*100:.1f}%")
        print(f"  Risk per trade    : {result['risk_per_trade']*100:.2f}%")
        print()
        print(f"  Average Return    : {result['average_pct']:+.1f}%")
        print(f"  Median Return     : {result['median_pct']:+.1f}%")
        print(f"  5th Percentile    : {result['percentile_5_pct']:+.1f}%")
        print(f"  95th Percentile   : {result['percentile_95_pct']:+.1f}%")
        print(f"  Worst Case        : {result['worst_case_pct']:+.1f}%")
        print(f"  Best Case         : {result['best_case_pct']:+.1f}%")
        print()
        print(f"  Survival Rate      : {result['survival_rate']*100:.1f}%")
        print(f"  Risk of Ruin      : {result['risk_of_ruin']*100:.1f}%")
        print(f"  Avg Max Drawdown  : {result['avg_max_drawdown_pct']:.1f}%")
        print(f"  Worst Max DD      : {result['worst_max_drawdown_pct']:.1f}%")
        print(f"  EV per Trade      : {result['ev_per_trade_pct']:+.3f}%")
        print(bar + "\n")
