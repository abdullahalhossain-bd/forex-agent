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
        """A factor is meaningful if it has a clear direction + strength."""
        return self.direction in ("BUY", "SELL") and self.strength >= 30

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
            total_weight_used += f.weight

            if f.direction == "BUY":
                buy_weighted += contribution
            elif f.direction == "SELL":
                sell_weighted += contribution

            result.factors.append(f)

        # Normalize to 0-100 scale (max possible = sum of weights × 100 × 100 / 10000 = sum of weights)
        max_possible = max(total_weight_used, 1.0)
        result.buy_score = round((buy_weighted / max_possible) * 100, 2)
        result.sell_score = round((sell_weighted / max_possible) * 100, 2)
        result.net_score = round(result.buy_score - result.sell_score, 2)

        # Final direction — lowered threshold from 10 to 5
        # Original 10 was too high, causing many setups to be NEUTRAL
        if abs(result.net_score) < 5:
            result.final_direction = "NEUTRAL"
        elif result.net_score > 0:
            result.final_direction = "BUY"
        else:
            result.final_direction = "SELL"

        # Count aligned factors
        result.aligned_factors = sum(
            1 for f in result.factors
            if f.direction == result.final_direction and f.is_meaningful
        )

        # Setup quality rating (only if direction is BUY/SELL)
        # Lowered thresholds — original was too strict, blocking most trades.
        if result.final_direction in ("BUY", "SELL"):
            aligned_pct = (result.aligned_factors / max(result.total_factors, 1)) * 100
            abs_net = abs(result.net_score)
            if aligned_pct >= 60 and abs_net >= 30:
                result.setup_quality = "A+"
            elif aligned_pct >= 40 and abs_net >= 20:
                result.setup_quality = "A"
            elif aligned_pct >= 25 and abs_net >= 10:
                result.setup_quality = "B"
            else:
                result.setup_quality = "AVOID"
            # Base confidence from net score — lowered starting point from 40 to 30
            result.confidence = min(95.0, 30.0 + abs_net * 0.9)
        else:
            result.setup_quality = "AVOID"
            result.confidence = 0.0

        return result


# ── singleton ───────────────────────────────────────────────────────
_SCORER: Optional[DecisionScorer] = None


def get_scorer() -> DecisionScorer:
    global _SCORER
    if _SCORER is None:
        _SCORER = DecisionScorer()
    return _SCORER
