"""
intelligence/currency_impact.py — Currency impact mapping engine
=================================================================

Converts a SentimentResult (tone + currency) into a directional bias for
each affected forex pair. This is the bridge between fundamental news
and tradeable pairs.

Mapping logic:
  Fed HAWKISH   → USD↑  → EURUSD↓ GBPUSD↓ USDJPY↑ (USD strengthens)
  Fed DOVISH    → USD↓  → EURUSD↑ GBPUSD↑ USDJPY↓ (USD weakens)
  ECB HAWKISH   → EUR↑  → EURUSD↑ EURJPY↑ EURGBP↑
  ECB DOVISH    → EUR↓  → EURUSD↓ EURJPY↓ EURGBP↓
  BoE HAWKISH   → GBP↑  → GBPUSD↑ GBPJPY↑ EURGBP↓
  BoE DOVISH    → GBP↓  → GBPUSD↓ GBPJPY↓ EURGBP↑
  BoJ HAWKISH   → JPY↑  → USDJPY↓ EURJPY↓ GBPJPY↓ (JPY strengthens)
  BoJ DOVISH    → JPY↓  → USDJPY↑ EURJPY↑ GBPJPY↑

Output:
    {
        "currency": "USD",
        "bias": "BULLISH",
        "duration_hours": 4,
        "confidence": 85,
        "pair_biases": {
            "EURUSD": "BEARISH",
            "GBPUSD": "BEARISH",
            "USDJPY": "BULLISH",
            ...
        }
    }
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

log = get_logger("currency_impact")


# ── Currency strength direction per tone ────────────────────────────
# When a central bank is HAWKISH, that currency STRENGTHENS.
# When DOVISH, that currency WEAKENS.
TONE_TO_CURRENCY_BIAS = {
    "HAWKISH": "BULLISH",   # currency strengthens
    "DOVISH":  "BEARISH",   # currency weakens
    "NEUTRAL": "NEUTRAL",
}


# ── Pair impact duration by impact_score ────────────────────────────
DURATION_BY_IMPACT = [
    (0.8, 8),   # impact_score >= 0.8 → 8 hours
    (0.6, 4),   # impact_score >= 0.6 → 4 hours
    (0.4, 2),   # impact_score >= 0.4 → 2 hours
    (0.2, 1),   # impact_score >= 0.2 → 1 hour
    (0.0, 0),   # no impact
]


@dataclass
class CurrencyImpact:
    """Currency-level bias from a news event."""
    currency: str                       # USD / EUR / GBP / JPY
    bias: str                           # BULLISH / BEARISH / NEUTRAL
    duration_hours: int
    confidence: float                   # 0-100
    source_event: str
    tone: str                           # HAWKISH / DOVISH / NEUTRAL
    impact_score: float
    pair_biases: Dict[str, str] = field(default_factory=dict)  # pair → BULLISH/BEARISH/NEUTRAL

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CurrencyImpactEngine:
    """Maps sentiment results to per-pair directional biases."""

    def __init__(self, all_pairs: Optional[List[str]] = None):
        self.all_pairs = [p.upper() for p in (all_pairs or [])]

    def calculate(
        self,
        currency: str,
        tone: str,
        impact_score: float,
        confidence: float = 50.0,
        source_event: str = "",
    ) -> CurrencyImpact:
        """Calculate directional bias for all pairs containing this currency."""
        currency = currency.upper()
        tone = tone.upper()

        # Currency-level bias
        currency_bias = TONE_TO_CURRENCY_BIAS.get(tone, "NEUTRAL")

        # Duration
        duration_hours = 0
        for threshold, dur in DURATION_BY_IMPACT:
            if impact_score >= threshold:
                duration_hours = dur
                break

        # Per-pair bias
        pair_biases: Dict[str, str] = {}
        for pair in self.all_pairs:
            if len(pair) != 6:
                continue
            base, quote = pair[:3], pair[3:]
            if currency not in (base, quote):
                continue

            # If currency is the BASE: currency BULLISH → pair BULLISH
            # If currency is the QUOTE: currency BULLISH → pair BEARISH
            if currency_bias == "NEUTRAL":
                pair_biases[pair] = "NEUTRAL"
            elif base == currency:
                pair_biases[pair] = currency_bias
            else:  # quote == currency
                pair_biases[pair] = "BEARISH" if currency_bias == "BULLISH" else "BULLISH"

        return CurrencyImpact(
            currency=currency,
            bias=currency_bias,
            duration_hours=duration_hours,
            confidence=confidence,
            source_event=source_event,
            tone=tone,
            impact_score=impact_score,
            pair_biases=pair_biases,
        )

    def merge_impacts(self, impacts: List[CurrencyImpact]) -> Dict[str, str]:
        """Merge multiple currency impacts into a single per-pair bias map.

        Rules:
          - If only one currency affects a pair → use that bias.
          - If both base AND quote currencies have signals → check alignment:
            * Both bullish/bearish → NEUTRAL (offsetting)
            * Base bullish + quote bearish → strong BULLISH (or vice versa)
          - Conflicts → NEUTRAL with reduced confidence.
        """
        per_pair: Dict[str, List[str]] = {}
        for impact in impacts:
            for pair, bias in impact.pair_biases.items():
                per_pair.setdefault(pair, []).append(bias)

        merged: Dict[str, str] = {}
        for pair, biases in per_pair.items():
            if not biases:
                continue
            bullish_count = sum(1 for b in biases if b == "BULLISH")
            bearish_count = sum(1 for b in biases if b == "BEARISH")
            if bullish_count > bearish_count:
                merged[pair] = "BULLISH"
            elif bearish_count > bullish_count:
                merged[pair] = "BEARISH"
            else:
                merged[pair] = "NEUTRAL"
        return merged


# ── singleton ───────────────────────────────────────────────────────
_ENGINE: Optional[CurrencyImpactEngine] = None


def get_currency_impact_engine(all_pairs: Optional[List[str]] = None) -> CurrencyImpactEngine:
    global _ENGINE
    if _ENGINE is None or (all_pairs and _ENGINE.all_pairs != [p.upper() for p in all_pairs]):
        _ENGINE = CurrencyImpactEngine(all_pairs or [])
    return _ENGINE
