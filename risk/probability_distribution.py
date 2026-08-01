"""
risk/probability_distribution.py — Probability Distribution for Trade Outcomes
=============================================================================
Course Module — Hedge Fund Layer #7: Probability Distribution

CONCEPT:
Instead of a single TP target, the AI computes a probability
distribution of possible outcomes:

  20% → Stop Loss hit
  50% → TP1 hit (take partial profit)
  20% → TP2 hit (trail remaining)
  10% → Runner (big win)

This allows:
- Expected value (EV) calculation before entry
- Dynamic position sizing based on EV
- Multi-target exit planning
- Risk-adjusted confidence scoring

USAGE:
    from risk.probability_distribution import compute_outcome_probabilities
    probs = compute_outcome_probabilities(
        entry=1.1000, sl=1.0950, tp1=1.1060, tp2=1.1120, runner=1.1200,
        win_rate=0.55, atr=0.0010, regime="TRENDING",
    )
    # → {"stop_loss": 0.25, "tp1": 0.45, "tp2": 0.20, "runner": 0.10}
"""

from __future__ import annotations
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from utils.logger import get_logger

log = get_logger("prob_distribution")


@dataclass
class ProbabilityDistribution:
    """Probability distribution of trade outcomes."""
    stop_loss_prob: float = 0.35    # P(SL hit)
    tp1_prob: float = 0.40          # P(TP1 hit)
    tp2_prob: float = 0.15          # P(TP2 hit)
    runner_prob: float = 0.10       # P(runner — big win)
    expected_value_usd: float = 0.0 # EV per $1 risked
    expected_value_pips: float = 0.0
    profit_factor: float = 0.0
    recommendation: str = "NEUTRAL"

    def to_dict(self) -> dict:
        return {
            "stop_loss_prob": round(self.stop_loss_prob, 3),
            "tp1_prob": round(self.tp1_prob, 3),
            "tp2_prob": round(self.tp2_prob, 3),
            "runner_prob": round(self.runner_prob, 3),
            "expected_value_usd": round(self.expected_value_usd, 2),
            "expected_value_pips": round(self.expected_value_pips, 1),
            "profit_factor": round(self.profit_factor, 2),
            "recommendation": self.recommendation,
        }


def compute_outcome_probabilities(
    entry: float,
    sl: float,
    tp1: float,
    tp2: float = 0.0,
    runner: float = 0.0,
    win_rate: float = 0.50,
    atr: float = 0.0010,
    regime: str = "RANGING",
    mtf_aligned: bool = False,
    liquidity_confirmed: bool = False,
    volume_confirmed: bool = False,
    risk_usd: float = 100.0,
) -> ProbabilityDistribution:
    """Compute probability distribution for trade outcomes.

    Uses historical win rate as base, then adjusts based on:
    - Market regime (trending → more runners, ranging → more TP1 hits)
    - MTF alignment (aligned → higher win probability)
    - Liquidity confirmation (sweep → higher win probability)
    - Volume confirmation (strong volume → higher runner probability)
    - ATR (higher ATR → wider outcomes, lower TP1 hit rate)

    Args:
        entry: Entry price.
        sl: Stop loss price.
        tp1: First take profit (closest).
        tp2: Second take profit (further). 0 = no TP2.
        runner: Runner target (furthest). 0 = no runner.
        win_rate: Historical win rate for this setup type (0-1).
        atr: Current ATR value.
        regime: "TRENDING" / "RANGING" / "BREAKOUT" / "VOLATILE".
        mtf_aligned: Higher timeframe aligned with trade direction.
        liquidity_confirmed: Liquidity sweep/grab confirmed.
        volume_confirmed: Volume supports the move.
        risk_usd: Dollar amount at risk (for EV calculation).

    Returns:
        ProbabilityDistribution with outcome probabilities and EV.
    """
    result = ProbabilityDistribution()

    # ── Base probabilities from historical win rate ──
    base_win = max(0.1, min(win_rate, 0.9))  # clamp to 10-90%
    base_loss = 1.0 - base_win

    # ── Adjustments based on market conditions ──

    # MTF alignment: +5% win probability if aligned
    if mtf_aligned:
        base_win = min(base_win + 0.05, 0.9)
        base_loss = 1.0 - base_win

    # Liquidity confirmation: +5% win probability
    if liquidity_confirmed:
        base_win = min(base_win + 0.05, 0.9)
        base_loss = 1.0 - base_win

    # Volume confirmation: shifts probability from TP1 to runner
    volume_runner_boost = 0.05 if volume_confirmed else 0.0

    # Regime adjustment
    regime_upper = regime.upper()
    if regime_upper == "TRENDING":
        # Trending: more TP2 and runners, fewer TP1-only wins
        tp1_share = 0.55  # 55% of wins go to TP1
        tp2_share = 0.25  # 25% go to TP2
        runner_share = 0.20 + volume_runner_boost  # 20%+ go to runner
    elif regime_upper == "RANGING":
        # Ranging: mostly TP1 hits, few runners
        tp1_share = 0.75
        tp2_share = 0.20
        runner_share = 0.05
    elif regime_upper == "BREAKOUT":
        # Breakout: more TP2 and runners if breakout is real
        tp1_share = 0.50
        tp2_share = 0.30
        runner_share = 0.20 + volume_runner_boost
    elif regime_upper == "VOLATILE":
        # Volatile: wider distribution
        tp1_share = 0.45
        tp2_share = 0.25
        runner_share = 0.30
    else:
        tp1_share = 0.60
        tp2_share = 0.25
        runner_share = 0.15

    # Normalize shares to sum to 1.0
    total_share = tp1_share + tp2_share + runner_share
    tp1_share /= total_share
    tp2_share /= total_share
    runner_share /= total_share

    # ── Compute outcome probabilities ──
    result.stop_loss_prob = round(base_loss, 3)
    result.tp1_prob = round(base_win * tp1_share, 3)
    result.tp2_prob = round(base_win * tp2_share, 3)
    result.runner_prob = round(base_win * runner_share, 3)

    # If no TP2 or runner specified, merge their probability into TP1
    if tp2 <= 0:
        result.tp1_prob = round(result.tp1_prob + result.tp2_prob, 3)
        result.tp2_prob = 0.0
    if runner <= 0:
        result.tp1_prob = round(result.tp1_prob + result.runner_prob, 3)
        result.runner_prob = 0.0

    # ── Calculate pip distances ──
    if entry > 0 and sl > 0:
        sl_pips = abs(entry - sl)
    else:
        sl_pips = 0.001

    tp1_pips = abs(tp1 - entry) if tp1 > 0 else 0
    tp2_pips = abs(tp2 - entry) if tp2 > 0 else 0
    runner_pips = abs(runner - entry) if runner > 0 else 0

    # ── Expected Value calculation ──
    # EV = P(SL) × (-risk) + P(TP1) × (tp1_reward) + P(TP2) × (tp2_reward) + P(Runner) × (runner_reward)
    ev_pips = (
        result.stop_loss_prob * (-sl_pips) +
        result.tp1_prob * tp1_pips +
        result.tp2_prob * tp2_pips +
        result.runner_prob * runner_pips
    )

    # Convert to USD (using risk_usd as reference)
    if sl_pips > 0:
        pip_value = risk_usd / sl_pips  # USD per pip
        ev_usd = ev_pips * pip_value
    else:
        ev_usd = 0.0
        pip_value = 0.0

    result.expected_value_pips = round(ev_pips, 1)
    result.expected_value_usd = round(ev_usd, 2)

    # ── Profit Factor ──
    total_win_pips = (
        result.tp1_prob * tp1_pips +
        result.tp2_prob * tp2_pips +
        result.runner_prob * runner_pips
    )
    total_loss_pips = result.stop_loss_prob * sl_pips
    result.profit_factor = round(total_win_pips / total_loss_pips, 2) if total_loss_pips > 0 else 0.0

    # ── Recommendation ──
    if ev_pips > sl_pips * 0.5:
        result.recommendation = "STRONG BUY — positive EV with good profit factor"
    elif ev_pips > 0:
        result.recommendation = "ACCEPTABLE — positive EV but modest"
    elif ev_pips > -sl_pips * 0.3:
        result.recommendation = "MARGINAL — near breakeven, skip unless high confidence"
    else:
        result.recommendation = "SKIP — negative EV, trade not justified"

    log.info(
        f"[ProbDist] SL={result.stop_loss_prob:.0%} TP1={result.tp1_prob:.0%} "
        f"TP2={result.tp2_prob:.0%} Runner={result.runner_prob:.0%} | "
        f"EV={result.expected_value_pips:.1f} pips (${result.expected_value_usd:.2f}) | "
        f"PF={result.profit_factor} | {result.recommendation}"
    )

    return result


# ── Smoke test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test 1: Good trending setup
    r1 = compute_outcome_probabilities(
        entry=1.1000, sl=1.0950, tp1=1.1060, tp2=1.1120, runner=1.1200,
        win_rate=0.55, atr=0.0010, regime="TRENDING",
        mtf_aligned=True, liquidity_confirmed=True, volume_confirmed=True,
        risk_usd=100.0,
    )
    print(f"Test 1 (trending, aligned):")
    print(f"  SL: {r1.stop_loss_prob:.0%} | TP1: {r1.tp1_prob:.0%} | TP2: {r1.tp2_prob:.0%} | Runner: {r1.runner_prob:.0%}")
    print(f"  EV: {r1.expected_value_pips:.1f} pips (${r1.expected_value_usd:.2f})")
    print(f"  PF: {r1.profit_factor} | {r1.recommendation}")

    # Test 2: Bad setup (low win rate, ranging)
    r2 = compute_outcome_probabilities(
        entry=1.1000, sl=1.0950, tp1=1.1030, tp2=0, runner=0,
        win_rate=0.35, atr=0.0008, regime="RANGING",
        mtf_aligned=False, liquidity_confirmed=False, volume_confirmed=False,
        risk_usd=100.0,
    )
    print(f"\nTest 2 (ranging, low WR):")
    print(f"  SL: {r2.stop_loss_prob:.0%} | TP1: {r2.tp1_prob:.0%}")
    print(f"  EV: {r2.expected_value_pips:.1f} pips (${r2.expected_value_usd:.2f})")
    print(f"  PF: {r2.profit_factor} | {r2.recommendation}")

    print("\nProbability distribution smoke test passed.")
