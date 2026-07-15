"""
risk/confidence_scaler.py — Confidence-Based Position Scaling (Day 76)
=======================================================================

Scales position size based on the Master Decision Engine's confidence.

| Confidence | Multiplier | Description          |
|---:|---:|---|
| < 40%  | 0.0× | Reject — below minimum (was 55%, lowered for new systems) |
| 40-54% | 0.3× | Low — very small size (new tier for early learning phase) |
| 55-64% | 0.5× | Marginal — reduced size                                   |
| 65-74% | 0.8× | Acceptable — slightly reduced                             |
| 75-84% | 1.0× | Normal — full base size                                   |
| 85-89% | 1.3× | High — boosted size                                       |
| 90-94% | 1.5× | Very high — strong boost                                  |
| 95%+   | 1.8× | Exceptional — max boost                                   |

Hard cap: never exceed 2.0× regardless of confidence.

FIXED (new system early-learning phase):
  - Rejection threshold: 55% → 40%
  - New LOW tier: 40-54% → 0.3× (small lot, not rejected)
  - Reason: fresh bot with 0 trade history gets confidence ~44%
    due to Bayesian penalty. Old threshold of 55% caused permanent
    NO TRADE loop — bot never traded, never built history, stuck forever.
    0.3× lot is conservative but allows the first trades to happen.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from utils.logger import get_logger

log = get_logger("confidence_scaler")

MAX_CONFIDENCE_MULT = 2.0  # absolute hard cap


@dataclass
class ConfidenceResult:
    """Output of confidence scaling."""
    factor: float       # multiplier (0.0 to 2.0)
    level: str          # REJECT / LOW / MARGINAL / ACCEPTABLE / NORMAL / HIGH / VERY_HIGH / EXCEPTIONAL
    confidence: float   # original confidence
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConfidenceScaler:
    """Confidence-based position size scaling."""

    # Confidence thresholds and multipliers.
    # FIXED: added LOW tier (40-54% → 0.3×) so new systems with
    # Bayesian-penalised confidence (~44%) still get small trades
    # instead of being permanently rejected.
    THRESHOLDS = [
        (95, 1.8, "EXCEPTIONAL"),
        (90, 1.5, "VERY_HIGH"),
        (85, 1.3, "HIGH"),
        (75, 1.0, "NORMAL"),
        (65, 0.8, "ACCEPTABLE"),
        (55, 0.5, "MARGINAL"),
        (40, 0.3, "LOW"),       # NEW: early learning phase tier
    ]

    def scale(self, confidence: float) -> ConfidenceResult:
        """Calculate confidence-based position multiplier.

        Args:
            confidence: Master Decision confidence (0-100).

        Returns:
            ConfidenceResult with multiplier + level.
        """
        confidence = max(0.0, min(100.0, confidence))

        for threshold, mult, level in self.THRESHOLDS:
            if confidence >= threshold:
                return ConfidenceResult(
                    factor=mult,
                    level=level,
                    confidence=confidence,
                    reason=f"Confidence {confidence:.0f}% → {level} → ×{mult}",
                )

        # Below 40% — reject
        return ConfidenceResult(
            factor=0.0,
            level="REJECT",
            confidence=confidence,
            reason=f"Confidence {confidence:.0f}% < 40% minimum → reject",
        )


# ── Singleton ───────────────────────────────────────────────────────

_SCALER: Optional[ConfidenceScaler] = None


def get_confidence_scaler() -> ConfidenceScaler:
    global _SCALER
    if _SCALER is None:
        _SCALER = ConfidenceScaler()
    return _SCALER