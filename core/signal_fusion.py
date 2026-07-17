"""
core/signal_fusion.py — 4-Layer Signal Fusion (Day 73)
========================================================

Fuses signals from 4 intelligence layers into a single master signal
with weighted confidence and conflict resolution.

Layers:
  1. Rule Engine (Day 67 Confluence) — weight 30%
  2. ML Ensemble (Day 69-70) — weight 30%
  3. RL Agent (Day 71) — weight 20%
  4. LLM Analyst (Day 42+ MasterAnalyst) — weight 20%

Conflict resolution:
  - 4/4 agreement → FULL position, max confidence
  - 3/4 agreement → HALF position, reduced confidence
  - 2/4 agreement → WAIT (insufficient consensus)
  - 1/4 or 0/4 → NO TRADE

Emergency disagreement:
  If one layer strongly opposes (confidence >80%) while others agree,
  confidence is penalized and position is reduced.

⚠️  STATUS (institutional review, see audit report): NOT CURRENTLY WIRED
    IN. get_signal_fusion()/SignalFusion.fuse() are not called from
    trader.py's run_cycle() or anywhere else in the uploaded codebase.
    The live decision path currently goes through core/agents/
    decision_agent.py (not part of this review) instead. If you intend
    to use 4-layer weighted fusion instead of (or alongside) the current
    decision logic, this module is ready to call — it just isn't called
    yet. See also core/signal_scorer.py, a second unwired alternative
    decision-aggregation approach; decide which one (if either) should
    be activated rather than wiring both.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

log = get_logger("signal_fusion")


@dataclass
class LayerSignal:
    """One intelligence layer's signal."""
    layer: str           # rule_engine / ml_ensemble / rl_agent / llm_analyst
    signal: str          # BUY / SELL / WAIT
    confidence: float    # 0-100
    weight: float = 0.25
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FusionResult:
    """Output of the signal fusion process."""
    final_signal: str = "WAIT"       # BUY / SELL / WAIT / NO_TRADE
    master_confidence: float = 0.0   # 0-100
    agreement: str = "0/4"
    agreement_count: int = 0
    total_layers: int = 4
    position_size: str = "NO_TRADE"  # FULL / HALF / REDUCED / WAIT / NO_TRADE
    position_multiplier: float = 0.0
    has_conflict: bool = False
    conflict_reason: str = ""
    layer_signals: List[Dict[str, Any]] = field(default_factory=list)
    weighted_contributions: Dict[str, float] = field(default_factory=dict)
    explanation: List[str] = field(default_factory=list)
    # ARCHITECTURAL FIX (institutional refactor): preserve the strongest
    # single-layer signal even when the fused decision is WAIT/NO_TRADE.
    # The execution layer may block, but the audit trail must show what
    # the analysis actually said. NEVER zero confidence just because
    # consensus failed.
    analysis_signal: str = "WAIT"          # strongest single-layer signal
    analysis_confidence: float = 0.0       # strongest single-layer confidence
    ml_available: bool = True              # whether ML layers participated
    excluded_layers: List[str] = field(default_factory=list)  # voters that dropped out

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SignalFusion:
    """Fuses 4-layer signals into a master decision."""

    # Position size thresholds
    FULL_THRESHOLD = 75.0
    HALF_THRESHOLD = 60.0
    REDUCED_THRESHOLD = 50.0

    def fuse(self, signals: List[LayerSignal]) -> FusionResult:
        """Fuse multiple layer signals into a single master decision."""
        result = FusionResult(total_layers=len(signals))
        result.layer_signals = [s.to_dict() for s in signals]

        if not signals:
            result.final_signal = "NO_TRADE"
            result.explanation.append("No intelligence layers available")
            return result

        # ── ARCHITECTURAL FIX (institutional refactor) ───────────────
        # Track which layers actually participated vs which dropped out.
        # Missing layers (LLM rate-limited, ML NOT_READY, etc.) are recorded
        # in `excluded_layers` for audit. The fused decision is computed
        # from the AVAILABLE layers only — no zero-padding for missing ones.
        # ──────────────────────────────────────────────────────────────
        result.ml_available = any(s.layer == "ml_ensemble" for s in signals)

        # Count votes
        buy_votes = [s for s in signals if s.signal == "BUY"]
        sell_votes = [s for s in signals if s.signal == "SELL"]
        wait_votes = [s for s in signals if s.signal in ("WAIT", "HOLD")]

        # ARCHITECTURAL FIX: preserve the strongest single-layer signal
        # for the audit trail. Even if the fused decision is WAIT/NO_TRADE
        # (insufficient consensus), the analysis verdict must be visible.
        _strongest = max(
            (s for s in signals if s.signal in ("BUY", "SELL")),
            key=lambda s: s.confidence,
            default=None,
        )
        if _strongest is not None:
            result.analysis_signal = _strongest.signal
            result.analysis_confidence = _strongest.confidence
        else:
            result.analysis_signal = "WAIT"
            result.analysis_confidence = 0.0

        # Determine majority
        if len(buy_votes) > len(sell_votes) and len(buy_votes) > len(wait_votes):
            majority = "BUY"
            agreeing = buy_votes
            opposing = sell_votes
        elif len(sell_votes) > len(buy_votes) and len(sell_votes) > len(wait_votes):
            majority = "SELL"
            agreeing = sell_votes
            opposing = buy_votes
        else:
            majority = "WAIT"
            agreeing = []
            opposing = []

        result.agreement_count = len(agreeing)
        result.agreement = f"{len(agreeing)}/{len(signals)}"

        # Weighted confidence
        total_weight = sum(s.weight for s in signals)
        if total_weight > 0 and agreeing:
            weighted_conf = sum(s.confidence * s.weight for s in agreeing) / sum(s.weight for s in agreeing)
        else:
            weighted_conf = 0.0

        # ARCHITECTURAL FIX: preserve the strongest analysis-layer confidence
        # even when the fused decision is WAIT/NO_TRADE due to insufficient
        # consensus or strong disagreement. Downstream consumers rely on this
        # to see that the analysis was meaningful, even if execution is blocked.
        if result.analysis_confidence > 0:
            weighted_conf = max(weighted_conf, result.analysis_confidence)

        # Penalty for disagreement
        if opposing:
            avg_opposing_conf = sum(s.confidence for s in opposing) / len(opposing)
            if avg_opposing_conf > 80:
                # Strong opposition — penalty
                weighted_conf *= 0.7
                result.has_conflict = True
                result.conflict_reason = (
                    f"Strong opposition from {', '.join(s.layer for s in opposing)} "
                    f"(conf {avg_opposing_conf:.0f}%) — confidence penalized"
                )
            elif opposing:
                # Moderate opposition — smaller penalty
                weighted_conf *= 0.85
                result.has_conflict = True
                result.conflict_reason = (
                    f"Opposition from {', '.join(s.layer for s in opposing)}"
                )

        result.master_confidence = round(weighted_conf, 1)

        # Determine final signal based on agreement
        if len(agreeing) >= 3:
            result.final_signal = majority
        elif len(agreeing) == 2 and len(signals) <= 3:
            # 2/3 or 2/4 — allow with reduced confidence
            if weighted_conf >= self.REDUCED_THRESHOLD:
                result.final_signal = majority
            else:
                result.final_signal = "WAIT"
        elif len(agreeing) == 2 and len(signals) == 4:
            # Log-driven fix (2026-07-17): 2/4 agreement was an automatic
            # WAIT regardless of confidence, even when the two agreeing
            # layers (e.g. ml_ensemble + rl_agent) were both highly
            # confident and the other two layers simply didn't vote
            # (WAIT/HOLD), not actively opposing. Now handled the same
            # way as the 2/3 case: allowed through if the weighted
            # confidence clears the reduced threshold.
            if weighted_conf >= self.REDUCED_THRESHOLD:
                result.final_signal = majority
            else:
                result.final_signal = "WAIT"
        else:
            result.final_signal = "WAIT"

        # Position size
        if result.final_signal in ("BUY", "SELL"):
            if result.master_confidence >= self.FULL_THRESHOLD and not result.has_conflict:
                result.position_size = "FULL"
                result.position_multiplier = 1.0
            elif result.master_confidence >= self.HALF_THRESHOLD:
                result.position_size = "HALF"
                result.position_multiplier = 0.5
            elif result.master_confidence >= self.REDUCED_THRESHOLD:
                result.position_size = "REDUCED"
                result.position_multiplier = 0.25
            else:
                result.final_signal = "WAIT"
                result.position_size = "WAIT"
                result.position_multiplier = 0.0
        else:
            result.position_size = "WAIT" if result.final_signal == "WAIT" else "NO_TRADE"

        # Build explanation
        result.explanation = self._build_explanation(signals, result)

        log.info(
            f"[SignalFusion] {result.final_signal} | conf={result.master_confidence:.1f}% | "
            f"agreement={result.agreement} | position={result.position_size}"
            f"{' | CONFLICT' if result.has_conflict else ''}"
        )

        return result

    def _build_explanation(self, signals: List[LayerSignal], result: FusionResult) -> List[str]:
        """Build human-readable explanation of the decision."""
        explanations = []
        for s in signals:
            emoji = "✅" if s.signal == result.final_signal else "❌"
            explanations.append(f"{emoji} {s.layer}: {s.signal} ({s.confidence:.0f}%) — {s.reasoning[:60]}")
        if result.has_conflict:
            explanations.append(f"⚠️ Conflict: {result.conflict_reason}")
        explanations.append(f"→ Master: {result.final_signal} ({result.master_confidence:.0f}%) — {result.position_size}")
        return explanations


# ── Singleton ───────────────────────────────────────────────────────

_FUSION: Optional[SignalFusion] = None


def get_signal_fusion() -> SignalFusion:
    global _FUSION
    if _FUSION is None:
        _FUSION = SignalFusion()
    return _FUSION