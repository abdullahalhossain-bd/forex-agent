"""
intelligence/event_classifier.py — Special event detection & risk rules
=========================================================================

Classifies news events into special categories and applies pre-defined
trading rules for each:

  * **FOMC**        — VERY HIGH impact. Block trades 60min before, 30min after.
  * **FOMC Press**  — Same as FOMC.
  * **NFP**         — EXTREME impact. Block 60min before. Wait for volatility
                      normalization 15min after.
  * **CPI**         — HIGH impact. Block 30min before, 15min after.
  * **Rate Decision** — VERY HIGH. Block 30min before/after.
  * **Central Bank Speech** — HIGH. Block 15min before, during, 15min after.
  * **GDP**         — MEDIUM. Reduce confidence 10%, no block.
  * **PMI**         — MEDIUM. Reduce confidence 5%, no block.

Output for each event:
    {
        "category": "FOMC" | "NFP" | "CPI" | "RATE_DECISION" | "CB_SPEECH" | "GDP" | "PMI" | "OTHER",
        "risk_level": "EXTREME" | "VERY_HIGH" | "HIGH" | "MEDIUM" | "LOW",
        "block_before_min": 60,
        "block_after_min": 30,
        "confidence_penalty": 0,    # 0 = block entirely, >0 = reduce confidence
        "action": "BLOCK" | "REDUCE_CONFIDENCE" | "MONITOR",
        "volatility_multiplier": 2.5,  # expected ATR multiplier during event
    }
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

log = get_logger("event_classifier")


# ── Event classification rules ──────────────────────────────────────
# Each rule: regex pattern → category + risk params
EVENT_RULES = [
    # ── EXTREME ────────────────────────────────────────────────────
    {
        "pattern": r"\b(non.?farm|nfp|employment change|payrolls)\b",
        "category": "NFP",
        "risk_level": "EXTREME",
        "block_before_min": 60,
        "block_after_min": 15,
        "confidence_penalty": 0,
        "action": "BLOCK",
        "volatility_multiplier": 3.0,
    },
    # ── VERY HIGH ──────────────────────────────────────────────────
    {
        "pattern": r"\b(fomc|federal open market|rate decision|interest rate decision)\b",
        "category": "FOMC",
        "risk_level": "VERY_HIGH",
        "block_before_min": 60,
        "block_after_min": 30,
        "confidence_penalty": 0,
        "action": "BLOCK",
        "volatility_multiplier": 2.5,
    },
    {
        "pattern": r"\b(press conference|presser)\b.*\b(fed|powell|fomc)\b|\b(fed|powell|fomc)\b.*\b(press conference|presser)\b",
        "category": "FOMC_PRESS",
        "risk_level": "VERY_HIGH",
        "block_before_min": 30,
        "block_after_min": 30,
        "confidence_penalty": 0,
        "action": "BLOCK",
        "volatility_multiplier": 2.5,
    },
    # ── HIGH ───────────────────────────────────────────────────────
    {
        "pattern": r"\bcpi\b|consumer price index|inflation rate|\binflation\b",
        "category": "CPI",
        "risk_level": "HIGH",
        "block_before_min": 30,
        "block_after_min": 15,
        "confidence_penalty": 0,
        "action": "BLOCK",
        "volatility_multiplier": 2.0,
    },
    {
        "pattern": r"\bppi\b|producer price index",
        "category": "PPI",
        "risk_level": "HIGH",
        "block_before_min": 30,
        "block_after_min": 15,
        "confidence_penalty": 0,
        "action": "BLOCK",
        "volatility_multiplier": 1.8,
    },
    {
        "pattern": r"\b(ecb|boe|boj|fed)\b.*\b(speech|statement|testimony)\b|\b(speech|statement|testimony)\b.*\b(ecb|boe|boj|fed|powell|lagarde|bailey|ueda)\b",
        "category": "CB_SPEECH",
        "risk_level": "HIGH",
        "block_before_min": 15,
        "block_after_min": 15,
        "confidence_penalty": 0,
        "action": "BLOCK",
        "volatility_multiplier": 1.8,
    },
    {
        "pattern": r"\b(monetary policy|policy statement)\b.*\b(ecb|boe|boj|fed)\b",
        "category": "RATE_DECISION",
        "risk_level": "VERY_HIGH",
        "block_before_min": 45,
        "block_after_min": 30,
        "confidence_penalty": 0,
        "action": "BLOCK",
        "volatility_multiplier": 2.5,
    },
    # ── MEDIUM ─────────────────────────────────────────────────────
    {
        "pattern": r"\bgdp\b|gross domestic product",
        "category": "GDP",
        "risk_level": "MEDIUM",
        "block_before_min": 0,
        "block_after_min": 0,
        "confidence_penalty": 10,
        "action": "REDUCE_CONFIDENCE",
        "volatility_multiplier": 1.4,
    },
    {
        "pattern": r"\bpmi\b|purchasing managers",
        "category": "PMI",
        "risk_level": "MEDIUM",
        "block_before_min": 0,
        "block_after_min": 0,
        "confidence_penalty": 5,
        "action": "REDUCE_CONFIDENCE",
        "volatility_multiplier": 1.3,
    },
    {
        "pattern": r"\b(retail sales|unemployment|jobless claims|trade balance)\b",
        "category": "ECON_DATA",
        "risk_level": "MEDIUM",
        "block_before_min": 15,
        "block_after_min": 10,
        "confidence_penalty": 0,
        "action": "BLOCK",
        "volatility_multiplier": 1.5,
    },
]


@dataclass
class EventClassification:
    """Classification result for a single news event."""
    category: str               # FOMC / NFP / CPI / CB_SPEECH / etc.
    risk_level: str             # EXTREME / VERY_HIGH / HIGH / MEDIUM / LOW
    block_before_min: int
    block_after_min: int
    confidence_penalty: float
    action: str                 # BLOCK / REDUCE_CONFIDENCE / MONITOR
    volatility_multiplier: float
    is_high_impact: bool = False  # HIGH or above

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EventClassifier:
    """Classifies news events and returns trading rules per category."""

    def classify(self, event_name: str, impact: str = "MEDIUM") -> EventClassification:
        """Classify an event by its name. Returns the matched rule or a default."""
        name_lower = event_name.lower()
        for rule in EVENT_RULES:
            if re.search(rule["pattern"], name_lower, re.IGNORECASE):
                is_high = rule["risk_level"] in ("EXTREME", "VERY_HIGH", "HIGH")
                return EventClassification(
                    category=rule["category"],
                    risk_level=rule["risk_level"],
                    block_before_min=rule["block_before_min"],
                    block_after_min=rule["block_after_min"],
                    confidence_penalty=rule["confidence_penalty"],
                    action=rule["action"],
                    volatility_multiplier=rule["volatility_multiplier"],
                    is_high_impact=is_high,
                )

        # No match — default based on impact level
        impact_upper = (impact or "MEDIUM").upper()
        if impact_upper == "HIGH":
            return EventClassification(
                category="OTHER", risk_level="HIGH",
                block_before_min=15, block_after_min=10,
                confidence_penalty=0, action="BLOCK",
                volatility_multiplier=1.5, is_high_impact=True,
            )
        if impact_upper == "MEDIUM":
            return EventClassification(
                category="OTHER", risk_level="MEDIUM",
                block_before_min=0, block_after_min=0,
                confidence_penalty=5, action="REDUCE_CONFIDENCE",
                volatility_multiplier=1.2, is_high_impact=False,
            )
        return EventClassification(
            category="OTHER", risk_level="LOW",
            block_before_min=0, block_after_min=0,
            confidence_penalty=0, action="MONITOR",
            volatility_multiplier=1.0, is_high_impact=False,
        )

    def is_in_block_window(
        self,
        classification: EventClassification,
        event_time_iso: str,
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Check if `now` falls within the event's block window.

        Returns:
            {
                "in_block_window": bool,
                "minutes_until_event": int (negative = passed),
                "minutes_since_event": int (negative = not yet),
                "block_reason": str,
            }
        """
        if now is None:
            now = datetime.now(timezone.utc)
        try:
            event_time = datetime.fromisoformat(event_time_iso.replace("Z", "+00:00"))
            if event_time.tzinfo is None:
                event_time = event_time.replace(tzinfo=timezone.utc)
        except Exception:
            return {"in_block_window": False, "minutes_until_event": 0,
                    "minutes_since_event": 0, "block_reason": "invalid time"}

        delta_before = (event_time - now).total_seconds() / 60.0  # positive = future
        delta_after = (now - event_time).total_seconds() / 60.0   # positive = past

        # In block window before event?
        if 0 < delta_before <= classification.block_before_min:
            return {
                "in_block_window": True,
                "minutes_until_event": int(delta_before),
                "minutes_since_event": -int(delta_before),
                "block_reason": f"{classification.category} in {int(delta_before)}min — block window active",
            }
        # In block window after event?
        if 0 < delta_after <= classification.block_after_min:
            return {
                "in_block_window": True,
                "minutes_until_event": -int(delta_after),
                "minutes_since_event": int(delta_after),
                "block_reason": f"{classification.category} {int(delta_after)}min ago — post-event block window",
            }
        # During the event itself (within ±1 min)?
        if abs(delta_before) <= 1:
            return {
                "in_block_window": True,
                "minutes_until_event": 0,
                "minutes_since_event": 0,
                "block_reason": f"{classification.category} happening NOW",
            }
        return {
            "in_block_window": False,
            "minutes_until_event": int(delta_before) if delta_before > 0 else -int(abs(delta_before)),
            "minutes_since_event": int(delta_after) if delta_after > 0 else -int(abs(delta_after)),
            "block_reason": "",
        }


# ── singleton ───────────────────────────────────────────────────────
_CLASSIFIER: Optional[EventClassifier] = None


def get_event_classifier() -> EventClassifier:
    global _CLASSIFIER
    if _CLASSIFIER is None:
        _CLASSIFIER = EventClassifier()
    return _CLASSIFIER
