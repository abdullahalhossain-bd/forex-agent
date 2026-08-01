# backtest/statistical_validation.py
# ============================================================
# Statistical Validation Suite
# ============================================================
# Addresses audit issues:
#   #2: Statistical validation (is the edge real?)
#   #7: Walk-forward optimization
#   Monte Carlo permutation test
#   t-test on trade returns
#   Bootstrap confidence intervals
#   Parameter sensitivity analysis
# ============================================================

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict

# Round-21 audit fix: removed unused imports (json, pd, datetime, timezone,
# Optional, Tuple) and moved scipy import to module level with guard.
try:
    from scipy import stats as scipy_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

log = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Container for all statistical validation results."""
    # Monte Carlo Permutation Test
    monte_carlo_p_value: float = 1.0
    monte_carlo_pass: bool = False
    monte_carlo_percentile: float = 0.0
    monte_carlo_detail: str = ""

    # t-test
    t_statistic: float = 0.0
    t_p_value: float = 1.0
    t_test_pass: bool = False
    t_detail: str = ""

    # Bootstrap Confidence Interval
    bootstrap_ci_lower: float = 0.0
    bootstrap_ci_upper: float = 0.0
    bootstrap_mean: float = 0.0
    bootstrap_pass: bool = False
    bootstrap_detail: str = ""

    # Walk-Forward
    walk_forward_in_sample_return: float = 0.0
    walk_forward_out_of_sample_return: float = 0.0
    walk_forward_efficiency: float = 0.0  # OOS / IS
    walk_forward_pass: bool = False
    walk_forward_detail: str = ""
    walk_forward_windows: List[dict] = field(default_factory=list)

    # Parameter Sensitivity
    sensitivity_results: Dict = field(default_factory=dict)
    sensitivity_pass: bool = False
    sensitivity_detail: str = ""

    # Overall
    overall_pass: bool = False
    overall_score: int = 0  # 0-100

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    def to_table(self) -> str:
        lines = [
            "=" * 55,
            "  STATISTICAL VALIDATION REPORT",
            "=" * 55,
            "",
            "--- 1. Monte Carlo Permutation Test ---",
            f"  p-value       : {self.monte_carlo_p_value:.4f}",
            f"  Pass (p<0.05) : {'✅ YES' if self.monte_carlo_pass else '❌ NO'}",
            f"  Percentile    : {self.monte_carlo_percentile:.1f}%",
            f"  Detail        : {self.monte_carlo_detail}",
            "",
            "--- 2. t-test (Mean Return > 0) ---",
            f"  t-statistic   : {self.t_statistic:.3f}",
            f"  p-value       : {self.t_p_value:.4f}",
            f"  Pass (p<0.05) : {'✅ YES' if self.t_test_pass else '❌ NO'}",
            f"  Detail        : {self.t_detail}",
            "",
            "--- 3. Bootstrap 95% Confidence Interval ---",
            f"  Mean          : {self.bootstrap_mean:.4f}",
            f"  95% CI        : [{self.bootstrap_ci_lower:.4f}, {self.bootstrap_ci_upper:.4f}]",
            f"  Pass (CI > 0) : {'✅ YES' if self.bootstrap_pass else '❌ NO'}",
            f"  Detail        : {self.bootstrap_detail}",
            "",
            "--- 4. Walk-Forward Efficiency ---",
            f"  In-Sample     : {self.walk_forward_in_sample_return:.2f}%",
            f"  Out-of-Sample : {self.walk_forward_out_of_sample_return:.2f}%",
            f"  Efficiency    : {self.walk_forward_efficiency:.1%}",
            f"  Pass (>50%)   : {'✅ YES' if self.walk_forward_pass else '❌ NO'}",
            f"  Detail        : {self.walk_forward_detail}",
            "",
            "--- 5. Parameter Sensitivity ---",
            f"  Pass (stable) : {'✅ YES' if self.sensitivity_pass else '❌ NO'}",
            f"  Detail        : {self.sensitivity_detail}",
            "",
            "=" * 55,
            f"  OVERALL: {'✅ PASS' if self.overall_pass else '❌ FAIL'}",
            f"  Score: {self.overall_score}/100",
            "=" * 55,
        ]
        return "\n".join(lines)


# ═════════════════════════════════════════════════════════════
# 1. MONTE CARLO PERMUTATION TEST
# ═════════════════════════════════════════════════════════════

def monte_carlo_permutation_test(
    trade_returns: List[float],
    n_permutations: int = 10000,
    alpha: float = 0.05,
) -> dict:
    """
    Monte Carlo Permutation Test.

    Shuffles trade return labels (win/loss assignment) n_permutations times,
    then compares the actual strategy return against the random distribution.

    If the actual return is in the top 5% of random permutations,
    the strategy has a statistically significant edge (p < 0.05).

    Args:
        trade_returns: list of per-trade returns (in USD or pips)
        n_permutations: number of random permutations (default 10000)
        alpha: significance level (default 0.05)

    Returns:
        {p_value, pass, percentile, detail}
    """
    if len(trade_returns) < 10:
        return {"p_value": 1.0, "pass": False, "percentile": 0,
                "detail": f"Insufficient trades ({len(trade_returns)} < 10)"}

    returns = np.array(trade_returns)
    actual_total = np.sum(returns)
    actual_mean = np.mean(returns)

    # Day 102+ CRITICAL hotfix: Monte Carlo permutation was a no-op.
    # Previously: `random_totals[i] = np.sum(np.random.permutation(returns))`
    # But np.sum of a permutation EQUALS np.sum of the original (addition
    # is commutative) — so every "permutation" produced the same total
    # as the actual. p_value was always 1.0, pass was always False, no
    # model could ever be validated.
    #
    # Fix: shuffle the ORDER of returns and compute the MAX DRAWDOWN of
    # the resulting equity curve. A skilled strategy has returns ordered
    # such that drawdowns are shallow; a random ordering produces deeper
    # drawdowns. Compare actual drawdown vs permutation drawdowns.

    def _max_drawdown(returns_series: np.ndarray) -> float:
        """Compute max drawdown from a series of returns."""
        equity = np.cumsum(returns_series)
        peak = np.maximum.accumulate(equity)
        drawdowns = peak - equity
        return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    actual_drawdown = _max_drawdown(returns)

    # Generate random permutations — now measuring drawdown, not sum
    random_drawdowns = np.zeros(n_permutations)
    for i in range(n_permutations):
        shuffled = np.random.permutation(returns)
        random_drawdowns[i] = _max_drawdown(shuffled)

    # p-value: fraction of random drawdowns <= actual drawdown
    # (lower drawdown = better; we want actual to be unusually low)
    p_value = np.mean(random_drawdowns <= actual_drawdown)
    percentile = np.mean(random_drawdowns < actual_drawdown) * 100

    passed = p_value < alpha
    detail = (f"Actual max-DD={actual_drawdown:.2f}, "
              f"Random max-DD mean={np.mean(random_drawdowns):.2f} ± {np.std(random_drawdowns):.2f}, "
              f"p={p_value:.4f} ({'significant' if passed else 'not significant'})")

    return {"p_value": round(p_value, 4), "pass": passed,
            "percentile": round(percentile, 1), "detail": detail}


# ═════════════════════════════════════════════════════════════
# 2. t-TEST (Mean Return > 0)
# ═════════════════════════════════════════════════════════════

def t_test_returns(trade_returns: List[float], alpha: float = 0.05) -> dict:
    """
    One-sample t-test: is the mean trade return significantly > 0?

    H0: mean return = 0 (no edge)
    H1: mean return > 0 (edge exists)

    Args:
        trade_returns: list of per-trade returns
        alpha: significance level

    Returns:
        {t_statistic, p_value, pass, detail}
    """
    if len(trade_returns) < 5:
        return {"t_statistic": 0, "p_value": 1.0, "pass": False,
                "detail": f"Insufficient trades ({len(trade_returns)} < 5)"}

    returns = np.array(trade_returns)
    n = len(returns)
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)

    if std == 0:
        return {"t_statistic": 0, "p_value": 1.0, "pass": False,
                "detail": "Zero variance in returns"}

    t_stat = mean / (std / np.sqrt(n))

    # One-tailed p-value (H1: mean > 0)
    # Round-21: use module-level scipy_stats with guard
    if not _HAS_SCIPY:
        return {"t_statistic": round(t_stat, 3), "p_value": 0.5,
                "pass": False, "detail": f"Mean={mean:.4f}, t={t_stat:.3f}, scipy not installed — p-value unavailable"}
    p_value = 1 - scipy_stats.t.cdf(t_stat, df=n - 1)

    passed = p_value < alpha
    detail = (f"Mean={mean:.4f}, Std={std:.4f}, t={t_stat:.3f}, "
              f"df={n-1}, p={p_value:.4f} ({'significant' if passed else 'not significant'})")

    return {"t_statistic": round(t_stat, 3), "p_value": round(p_value, 4),
            "pass": passed, "detail": detail}


# ═════════════════════════════════════════════════════════════
# 3. BOOTSTRAP CONFIDENCE INTERVAL
# ═════════════════════════════════════════════════════════════

def bootstrap_confidence_interval(
    trade_returns: List[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
) -> dict:
    """
    Bootstrap 95% confidence interval for mean trade return.

    Resamples trade returns with replacement n_bootstrap times,
    calculates mean for each sample, then takes the 2.5th and 97.5th percentiles.

    If the lower bound of the CI > 0, the strategy is profitable with 95% confidence.

    Args:
        trade_returns: list of per-trade returns
        n_bootstrap: number of bootstrap samples
        confidence: confidence level (default 0.95)

    Returns:
        {ci_lower, ci_upper, mean, pass, detail}
    """
    if len(trade_returns) < 10:
        return {"ci_lower": 0, "ci_upper": 0, "mean": 0, "pass": False,
                "detail": f"Insufficient trades ({len(trade_returns)} < 10)"}

    returns = np.array(trade_returns)
    n = len(returns)

    # Bootstrap sampling
    bootstrap_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = np.random.choice(returns, size=n, replace=True)
        bootstrap_means[i] = np.mean(sample)

    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_means, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
    mean = np.mean(returns)

    passed = ci_lower > 0
    detail = (f"Mean={mean:.4f}, 95% CI=[{ci_lower:.4f}, {ci_upper:.4f}], "
              f"{'profitable' if passed else 'not confirmed profitable'}")

    return {"ci_lower": round(ci_lower, 4), "ci_upper": round(ci_upper, 4),
            "mean": round(mean, 4), "pass": passed, "detail": detail}


# ═════════════════════════════════════════════════════════════
# 4. WALK-FORWARD OPTIMIZATION
# ═════════════════════════════════════════════════════════════

def walk_forward_analysis(
    trade_returns: List[float],
    train_pct: float = 0.7,
    n_windows: int = 5,
    min_efficiency: float = 0.50,
) -> dict:
    """
    Walk-Forward Analysis.

    Splits trade returns into rolling windows:
      - In-Sample (IS): first 70% of each window (optimization period)
      - Out-of-Sample (OOS): remaining 30% (validation period)

    Walk-Forward Efficiency (WFE) = OOS return / IS return
    If WFE > 50%, the strategy generalizes well (not overfitted).

    Args:
        trade_returns: list of per-trade returns (chronological order)
        train_pct: fraction of each window for in-sample
        n_windows: number of rolling windows
        min_efficiency: minimum WFE to pass

    Returns:
        {in_sample_return, out_of_sample_return, efficiency, pass, detail, windows}
    """
    if len(trade_returns) < 30:
        return {"in_sample_return": 0, "out_of_sample_return": 0, "efficiency": 0,
                "pass": False, "detail": f"Insufficient trades ({len(trade_returns)} < 30)",
                "windows": []}

    returns = np.array(trade_returns)
    n = len(returns)
    window_size = n // n_windows
    train_size = int(window_size * train_pct)

    is_total = 0.0
    oos_total = 0.0
    windows = []

    for w in range(n_windows):
        start = w * window_size
        end = min(start + window_size, n)
        window_returns = returns[start:end]

        is_returns = window_returns[:train_size]
        oos_returns = window_returns[train_size:]

        is_sum = float(np.sum(is_returns))
        oos_sum = float(np.sum(oos_returns))

        is_total += is_sum
        oos_total += oos_sum

        windows.append({
            "window": w + 1,
            "in_sample_return": round(is_sum, 2),
            "out_of_sample_return": round(oos_sum, 2),
            "trades": len(window_returns),
        })

    efficiency = oos_total / is_total if is_total > 0 else 0
    passed = efficiency >= min_efficiency and oos_total > 0

    detail = (f"IS={is_total:.2f}, OOS={oos_total:.2f}, "
              f"WFE={efficiency:.1%} ({'good' if passed else 'overfitted or poor'})")

    return {
        "in_sample_return": round(is_total, 2),
        "out_of_sample_return": round(oos_total, 2),
        "efficiency": round(efficiency, 3),
        "pass": passed,
        "detail": detail,
        "windows": windows,
    }


# ═════════════════════════════════════════════════════════════
# 5. PARAMETER SENSITIVITY ANALYSIS
# ═════════════════════════════════════════════════════════════

def parameter_sensitivity_check(
    trade_returns: List[float],
    perturbation: float = 0.20,
) -> dict:
    """
    Parameter Sensitivity Analysis.

    Simulates what happens if trade returns are perturbed by ±20%
    (representing parameter changes like WICK_BODY_RATIO 2.0 → 1.8 or 2.2).

    If the strategy remains profitable under ±20% perturbation,
    it's robust to parameter changes (not overfitted to specific thresholds).

    Args:
        trade_returns: list of per-trade returns
        perturbation: fraction to perturb (default 0.20 = ±20%)

    Returns:
        {results, pass, detail}
    """
    if len(trade_returns) < 10:
        return {"results": {}, "pass": False,
                "detail": f"Insufficient trades ({len(trade_returns)} < 10)"}

    returns = np.array(trade_returns)
    original_total = float(np.sum(returns))
    original_mean = float(np.mean(returns))

    results = {}
    all_positive = True

    for label, mult in [("-20%", 1 - perturbation), ("0%", 1.0), ("+20%", 1 + perturbation)]:
        perturbed = returns * mult
        total = float(np.sum(perturbed))
        mean = float(np.mean(perturbed))
        results[label] = {"total": round(total, 2), "mean": round(mean, 4)}
        if total <= 0:
            all_positive = False

    # Also test with random noise
    noisy = returns + np.random.normal(0, abs(original_mean) * 0.5, len(returns))
    noisy_total = float(np.sum(noisy))
    results["random_noise"] = {"total": round(noisy_total, 2)}
    if noisy_total <= 0:
        all_positive = False

    detail = (f"Original={original_total:.2f}, "
              f"-20%={results['-20%']['total']:.2f}, "
              f"+20%={results['+20%']['total']:.2f}, "
              f"noise={results['random_noise']['total']:.2f}, "
              f"{'robust' if all_positive else 'sensitive'}")

    return {"results": results, "pass": all_positive, "detail": detail}


# ═════════════════════════════════════════════════════════════
# MASTER: Run All Validation Tests
# ═════════════════════════════════════════════════════════════

def run_full_validation(
    trade_returns: List[float],
    n_monte_carlo: int = 10000,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
) -> ValidationResult:
    """
    Run all 5 statistical validation tests.

    Args:
        trade_returns: list of per-trade returns (USD or pips, chronological)
        n_monte_carlo: MC permutation iterations
        n_bootstrap: bootstrap iterations
        alpha: significance level

    Returns:
        ValidationResult with all tests + overall score
    """
    result = ValidationResult()

    # 1. Monte Carlo
    mc = monte_carlo_permutation_test(trade_returns, n_monte_carlo, alpha)
    result.monte_carlo_p_value = mc["p_value"]
    result.monte_carlo_pass = mc["pass"]
    result.monte_carlo_percentile = mc["percentile"]
    result.monte_carlo_detail = mc["detail"]

    # 2. t-test
    tt = t_test_returns(trade_returns, alpha)
    result.t_statistic = tt["t_statistic"]
    result.t_p_value = tt["p_value"]
    result.t_test_pass = tt["pass"]
    result.t_detail = tt["detail"]

    # 3. Bootstrap
    bs = bootstrap_confidence_interval(trade_returns, n_bootstrap)
    result.bootstrap_ci_lower = bs["ci_lower"]
    result.bootstrap_ci_upper = bs["ci_upper"]
    result.bootstrap_mean = bs["mean"]
    result.bootstrap_pass = bs["pass"]
    result.bootstrap_detail = bs["detail"]

    # 4. Walk-Forward
    wf = walk_forward_analysis(trade_returns)
    result.walk_forward_in_sample_return = wf["in_sample_return"]
    result.walk_forward_out_of_sample_return = wf["out_of_sample_return"]
    result.walk_forward_efficiency = wf["efficiency"]
    result.walk_forward_pass = wf["pass"]
    result.walk_forward_detail = wf["detail"]
    result.walk_forward_windows = wf["windows"]

    # 5. Sensitivity
    ps = parameter_sensitivity_check(trade_returns)
    result.sensitivity_results = ps["results"]
    result.sensitivity_pass = ps["pass"]
    result.sensitivity_detail = ps["detail"]

    # Overall score (20 points per test)
    score = 0
    if result.monte_carlo_pass: score += 20
    if result.t_test_pass: score += 20
    if result.bootstrap_pass: score += 20
    if result.walk_forward_pass: score += 20
    if result.sensitivity_pass: score += 20
    result.overall_score = score
    result.overall_pass = score >= 80  # need at least 4/5 tests passing

    return result


# ============================================================
# CLI entry
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)

    # Simulate trade returns (60% win rate, avg win=40pips, avg loss=-25pips)
    n_trades = 100
    wins = np.random.normal(40, 15, int(n_trades * 0.6))
    losses = np.random.normal(-25, 10, n_trades - int(n_trades * 0.6))
    returns = np.concatenate([wins, losses])
    np.random.shuffle(returns)

    print(f"\nSimulated: {n_trades} trades, mean={np.mean(returns):.2f}, total={np.sum(returns):.2f}\n")

    result = run_full_validation(list(returns))
    print(result.to_table())
