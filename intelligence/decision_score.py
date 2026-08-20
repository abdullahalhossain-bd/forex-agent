"""
intelligence/decision_score.py — Weighted factor scoring system
=================================================================

Day 67 — Professional multi-factor decision scoring.

Each analysis factor (SMC, Liquidity, Session, Currency Strength,
Intermarket, News, Technical) gets:
  * a **weight** (sum = 100%)
  * a **direction** (BUY / SELL / NEUTRAL)
  * a **strength** (0-100)
  * a **confidence** (0-100)

The weighted score produces a single 0-100 BUY score and 0-100 SELL score.
The difference determines the final direction.

Weight allocation (calibrated for institutional-style trading):
  SMC (Market Structure)     : 25%  ← institutional footprint
  Liquidity                  : 20%  ← where stops/resting orders are
  Currency Strength          : 15%  ← relative strength model
  Intermarket                : 15%  ← DXY/Gold/VIX confirmation
  News Intelligence          : 10%  ← fundamental bias
  Technical (RSI/MACD/EMA)   : 10%  ← momentum confirmation
  Session                    : 5%   ← time-of-day quality

Contradiction rule: if 2+ top-weight factors disagree strongly,
the confluence score is penalized (handled in signal_validator.py).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

log = get_logger("decision_score")


# ── Factor weights (must sum to 100) ────────────────────────────────
FACTOR_WEIGHTS = {
    "smc":               25,   # Market structure (BOS/CHoCH/OB/FVG)
    "liquidity":         20,   # Liquidity sweeps / equal highs/lows
    "currency_strength": 15,   # Currency relative strength
    "intermarket":       15,   # DXY/Gold/VIX/US10Y/SP500
    "news":              10,   # News Intelligence bias
    "technical":         10,   # RSI/MACD/EMA/Pattern
    "session":            5,   # Session quality / time-of-day
}

# Sanity check — weights must sum to 100
assert sum(FACTOR_WEIGHTS.values()) == 100, f"Weights must sum to 100, got {sum(FACTOR_WEIGHTS.values())}"


@dataclass
class FactorScore:
    """One analysis factor's contribution to the confluence score."""
    name: str                       # smc / liquidity / ...
    direction: str                  # BUY / SELL / NEUTRAL
    strength: float                 # 0-100 (how strong is this signal?)
    confidence: float               # 0-100 (how confident in this signal?)
    weight: float                   # weight % (0-100)
    weighted_score: float = 0.0     # filled by calculate()
    reasoning: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def aligned_direction(self) -> str:
        return self.direction

    @property
    def is_meaningful(self) -> bool:
        """A factor is meaningful if it has a clear direction + strength.

        Threshold lowered 30 → 20 (2026-08-19): many real modules (technical
        at 40% rule conf, currency bias at 25-35) were silently excluded from
        aligned_factors, so live logs showed "1 factors (≥2)" even when MTF
        and bias agreed. 20 still filters pure noise (<20) without erasing
        moderate but valid directional votes.
        """
        return self.direction in ("BUY", "SELL") and self.strength >= 20

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConfluenceScore:
    """The final aggregated confluence score."""
    buy_score: float = 0.0         # 0-100 weighted BUY strength
    sell_score: float = 0.0        # 0-100 weighted SELL strength
    net_score: float = 0.0         # buy_score - sell_score (-100 to +100)
    final_direction: str = "NEUTRAL"  # BUY / SELL / NEUTRAL
    aligned_factors: int = 0       # count of factors aligned with final_direction
    total_factors: int = 0
    factors: List[FactorScore] = field(default_factory=list)
    setup_quality: str = "AVOID"   # A+ / A / B / AVOID
    confidence: float = 0.0        # 0-100 final calibrated confidence
    has_contradiction: bool = False
    contradiction_reason: str = ""
    experiment_alignment: str = ""  # EXPERIMENT Iteration 9; empty on live

    def to_dict(self) -> Dict[str, Any]:
        return {
            "buy_score": round(self.buy_score, 2),
            "sell_score": round(self.sell_score, 2),
            "net_score": round(self.net_score, 2),
            "final_direction": self.final_direction,
            "aligned_factors": self.aligned_factors,
            "total_factors": self.total_factors,
            "factors": [f.to_dict() for f in self.factors],
            "setup_quality": self.setup_quality,
            "confidence": round(self.confidence, 2),
            "has_contradiction": self.has_contradiction,
            "contradiction_reason": self.contradiction_reason,
        }



# EXPERIMENT (Iteration 9, owner-approved) — NOT a production "fix"
# Offline CSV backtests cannot populate session (always NEUTRAL by design),
# news, or intermarket (live-only). Counting them toward min_aligned made ≥3 untestable.
# When flag AND is_backtest_mode(): aligned/total use only {smc, technical, liquidity, currency_strength}
# Live path unchanged. Re-review when live feeds make those modules directional.
BACKTEST_EFFECTIVE_ALIGNMENT_ONLY = True
BACKTEST_COUNTED_ALIGNMENT_MODULES = frozenset({
    "smc", "technical", "liquidity", "currency_strength",
})

class DecisionScorer:
    """Computes weighted confluence scores from individual factor inputs."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or FACTOR_WEIGHTS.copy()

    def score(self, factors: List[FactorScore]) -> ConfluenceScore:
        """Compute final confluence score from a list of factor scores."""
        result = ConfluenceScore(total_factors=len(factors))

        buy_weighted = 0.0
        sell_weighted = 0.0
        total_weight_used = 0.0

        for f in factors:
            # Apply weight
            f.weight = self.weights.get(f.name, 0)
            # Weighted contribution: strength × confidence × weight / 10000
            contribution = (f.strength * f.confidence * f.weight) / 10000.0
            f.weighted_score = round(contribution, 2)

            # BUGFIX (audit 2026-08-11, evidence: penalty_attribution.csv /
            # blocked_trade_analysis.csv — 218 "Min confidence" blocks, 72.5%
            # naturally-low pre-penalty confidence): a factor with
            # direction == "NEUTRAL" mathematically can NEVER contribute to
            # buy_weighted/sell_weighted below, yet its full weight was still
            # being added to total_weight_used (the normalization
            # denominator). This silently wasted weight on every single
            # decision cycle:
            #   - "session" is *always* NEUTRAL by design (see
            #     _session_factor: "Session is direction-neutral — it boosts
            #     the OTHER factors' weight") — but instead of boosting
            #     anything, its 5% weight was permanently unusable dead
            #     weight in the denominator, uniformly deflating net_score
            #     (and therefore confidence) on every decision regardless of
            #     how good the session actually was.
            #   - "intermarket" (15% weight) and "news" (10% weight) fall
            #     back to NEUTRAL/0 whenever their upstream data is missing
            #     or blocked — the exact same dilution effect, but for a
            #     missing-data reason rather than a by-design one.
            # In both cases this is a missing/wasted-weight bug, not a
            # legitimate low-confidence signal. Fix: only count a factor's
            # weight toward the normalization denominator when it actually
            # participated in the BUY/SELL tally (i.e. is directional).
            # NEUTRAL factors still appear in result.factors / total_factors
            # for transparency, and still influence has_contradiction logic
            # downstream — this only stops them from mechanically capping
            # confidence via unusable denominator weight.
            if f.direction in ("BUY", "SELL"):
                total_weight_used += f.weight

            if f.direction == "BUY":
                buy_weighted += contribution
            elif f.direction == "SELL":
                sell_weighted += contribution

            result.factors.append(f)

        # Normalize to 0-100 scale (max possible = sum of weight actually
        # used by directional factors — NEUTRAL factors contribute 0 and so
        # are excluded from the denominator, see BUGFIX above).
        max_possible = max(total_weight_used, 1.0)
        result.buy_score = round((buy_weighted / max_possible) * 100, 2)
        result.sell_score = round((sell_weighted / max_possible) * 100, 2)
        result.net_score = round(result.buy_score - result.sell_score, 2)

        # Final direction — lowered threshold so more coherent setups remain directional
        if abs(result.net_score) < 3:
            result.final_direction = "NEUTRAL"
        elif result.net_score > 0:
            result.final_direction = "BUY"
        else:
            result.final_direction = "SELL"

        # Count aligned factors
        # EXPERIMENT (Iteration 9): backtest-only effective denominator. Live unchanged.
        _count_from = result.factors
        try:
            from core.constants import is_backtest_mode
            if BACKTEST_EFFECTIVE_ALIGNMENT_ONLY and is_backtest_mode():
                _count_from = [
                    f for f in result.factors
                    if getattr(f, "name", None) in BACKTEST_COUNTED_ALIGNMENT_MODULES
                ]
                result.total_factors = len(_count_from)
                result.experiment_alignment = "BACKTEST_EFFECTIVE_ALIGNMENT_ONLY"
        except Exception:
            _count_from = result.factors
        result.aligned_factors = sum(
            1 for f in _count_from
            if f.direction == result.final_direction and f.is_meaningful
        )

        # Confidence-pipeline simplification: continuous confidence formula.
        # Previously: two independent hard thresholds (aligned_pct AND abs_net)
        # that BOTH had to clear, else setup_quality=AVOID and confidence=0.
        # Now: a single continuous confidence from net_score (no cliff),
        # and quality bands are purely informational labels — they no longer
        # force confidence to zero.  Weak setups get low-but-nonlinear
        # confidence instead of being killed.
        if result.final_direction in ("BUY", "SELL"):
            aligned_pct = (result.aligned_factors / max(result.total_factors, 1)) * 100
            abs_net = abs(result.net_score)
            if aligned_pct >= 50 and abs_net >= 25:
                result.setup_quality = "A+"
            elif aligned_pct >= 35 and abs_net >= 15:
                result.setup_quality = "A"
            elif aligned_pct >= 20 and abs_net >= 8:
                result.setup_quality = "B"
            else:
                result.setup_quality = "C"  # was "AVOID" — now just a weak label
            # Continuous confidence: linear ramp from ~10 (near-zero edge)
            # to ~95 (very strong edge).  No cliff — every net_score > 3
            # produces a usable, nonzero confidence.
            result.confidence = min(95.0, max(10.0, 10.0 + abs_net * 1.2))
        else:
            result.setup_quality = "NEUTRAL"
            result.confidence = 0.0

        from utils.confidence_trace import confidence_trace
        confidence_trace.record(
            module="decision_score",
            before=abs(result.net_score),
            after=result.confidence,
            reason=f"direction={result.final_direction}, quality={result.setup_quality}, "
                   f"net={result.net_score}, aligned={result.aligned_factors}/{result.total_factors}",
        )

        return result


# ── singleton ───────────────────────────────────────────────────────
_SCORER: Optional[DecisionScorer] = None


def get_scorer() -> DecisionScorer:
    global _SCORER
    if _SCORER is None:
        _SCORER = DecisionScorer()
    return _SCORER
