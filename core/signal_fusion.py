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

Conflict resolution (2026-08-13 update):
  - Directional votes only (BUY/SELL). WAIT/HOLD = abstain, not opposition.
  - Missing / NOT_READY layers have weight zeroed and redistributed.
  - 3+ directional agreement → FULL / HIGH confidence
  - 2 directional agreement → allowed if weighted_conf >= REDUCED_THRESHOLD
  - 1 directional → allowed only at high confidence (REDUCED size)
  - Active opposition (BUY vs SELL) still penalizes confidence.

⚠️  STATUS: WIRED IN via MasterDecisionEngine.decide() → self.fusion.fuse()
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

log = get_logger("signal_fusion")


@dataclass
class LayerSignal:
    """One intelligence layer's signal."""
    layer: str           # rule_engine / ml_ensemble / rl_agent / llm_analyst
    signal: str          # BUY / SELL / WAIT / HOLD / NOT_READY
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
    agreement: str = "0/0"
    agreement_count: int = 0
    total_layers: int = 4
    position_size: str = "NO_TRADE"  # FULL / HALF / REDUCED / WAIT / NO_TRADE
    position_multiplier: float = 0.0
    has_conflict: bool = False
    conflict_reason: str = ""
    layer_signals: List[Dict[str, Any]] = field(default_factory=list)
    weighted_contributions: Dict[str, float] = field(default_factory=dict)
    explanation: List[str] = field(default_factory=list)
    # Strongest single-layer directional signal (audit trail)
    analysis_signal: str = "WAIT"
    analysis_confidence: float = 0.0
    ml_available: bool = True
    excluded_layers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SignalFusion:
    """Fuses multi-layer signals into a master decision."""

    # Position size thresholds
    FULL_THRESHOLD = 70.0
    HALF_THRESHOLD = 55.0
    # 2026-08-13: lowered from 45 → 40 so two agreeing directional
    # layers with moderate confidence can still produce a tradeable
    # signal instead of permanent WAIT.
    REDUCED_THRESHOLD = 40.0

    # Layers that are considered "not participating" (abstain / missing)
    _ABSTAIN_SIGNALS = {"WAIT", "HOLD", "NOT_READY", "NO_TRADE", ""}

    def fuse(self, signals: List[LayerSignal]) -> FusionResult:
        """Fuse multiple layer signals into a single master decision."""
        result = FusionResult(total_layers=len(signals))
        result.layer_signals = [s.to_dict() for s in signals]

        if not signals:
            result.final_signal = "NO_TRADE"
            result.explanation.append("No intelligence layers available")
            return result

        # ── 1. Mark non-participating layers & redistribute weight ──
        working = self._normalize_weights(signals, result)

        # ── 2. Split directional vs abstaining ──
        directional = [s for s in working if s.signal in ("BUY", "SELL")]
        buy_votes = [s for s in directional if s.signal == "BUY"]
        sell_votes = [s for s in directional if s.signal == "SELL"]

        result.ml_available = any(
            s.layer == "ml_ensemble" and s.signal not in self._ABSTAIN_SIGNALS
            for s in signals
        )

        # Strongest single directional vote (audit + thin-agreement fallback)
        _strongest = max(
            directional,
            key=lambda s: s.confidence,
            default=None,
        )
        if _strongest is not None:
            result.analysis_signal = _strongest.signal
            result.analysis_confidence = _strongest.confidence
        else:
            result.analysis_signal = "WAIT"
            result.analysis_confidence = 0.0

        # ── 3. Majority among directional votes only ──
        # WAIT no longer vetoes a real BUY/SELL the way an opposing
        # directional vote does. Abstentions simply reduce the voter pool.
        if len(buy_votes) > len(sell_votes):
            majority = "BUY"
            agreeing = buy_votes
            opposing = sell_votes
        elif len(sell_votes) > len(buy_votes):
            majority = "SELL"
            agreeing = sell_votes
            opposing = buy_votes
        else:
            # Tie or no directional votes
            majority = "WAIT"
            agreeing = []
            opposing = buy_votes + sell_votes  # both sides if pure tie

        result.agreement_count = len(agreeing)
        n_directional = len(directional)
        result.agreement = f"{len(agreeing)}/{n_directional}" if n_directional else "0/0"
        result.total_layers = n_directional  # report directional pool size

        # ── 4. Weighted confidence from agreeing directional layers ──
        if agreeing:
            w_sum = sum(s.weight for s in agreeing)
            if w_sum > 0:
                weighted_conf = sum(s.confidence * s.weight for s in agreeing) / w_sum
            else:
                weighted_conf = sum(s.confidence for s in agreeing) / len(agreeing)
        else:
            weighted_conf = 0.0

        # Thin agreement (<2 directional): surface strongest raw confidence
        # for audit only — final_signal will still be WAIT below unless
        # single high-confidence path triggers.
        if result.analysis_confidence > 0 and len(agreeing) < 2:
            weighted_conf = max(weighted_conf, result.analysis_confidence)

        # ── 5. Opposition penalty (only real BUY vs SELL conflict) ──
        if opposing and agreeing:
            avg_opp = sum(s.confidence for s in opposing) / len(opposing)
            if avg_opp > 80:
                weighted_conf *= 0.70
                result.has_conflict = True
                result.conflict_reason = (
                    f"Strong opposition from "
                    f"{', '.join(s.layer for s in opposing)} "
                    f"(conf {avg_opp:.0f}%) — confidence penalized"
                )
            else:
                weighted_conf *= 0.85
                result.has_conflict = True
                result.conflict_reason = (
                    f"Opposition from {', '.join(s.layer for s in opposing)}"
                )

        # Confidence calibration (never allow 100%)
        try:
            from core.entry_safety_filters import EntrySafetyFilters
            weighted_conf = EntrySafetyFilters.calibrate_confidence(weighted_conf)
        except Exception as e:
            log.debug(f"[SignalFusion] Confidence calibration unavailable: {e}")

        result.master_confidence = round(min(99.0, max(0.0, weighted_conf)), 1)

        # ── 6. Final signal from directional agreement ──
        if len(agreeing) >= 3:
            result.final_signal = majority
        elif len(agreeing) == 2:
            if result.master_confidence >= self.REDUCED_THRESHOLD:
                result.final_signal = majority
            else:
                result.final_signal = "WAIT"
        elif len(agreeing) == 1 and result.master_confidence >= self.FULL_THRESHOLD:
            # Single high-confidence directional layer (e.g. only Rule
            # Engine is live and ML/RL/LLM abstained). Allow at REDUCED
            # size so the system is not completely silent when 3/4
            # layers are offline.
            result.final_signal = majority
            result.has_conflict = True
            result.conflict_reason = (
                result.conflict_reason
                or "Single-layer directional signal — reduced size"
            )
        else:
            result.final_signal = "WAIT"

        # ── 7. Position size ──
        if result.final_signal in ("BUY", "SELL"):
            if (
                result.master_confidence >= self.FULL_THRESHOLD
                and not result.has_conflict
                and len(agreeing) >= 3
            ):
                result.position_size = "FULL"
                result.position_multiplier = 1.0
            elif result.master_confidence >= self.HALF_THRESHOLD and len(agreeing) >= 2:
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
            result.position_size = (
                "WAIT" if result.final_signal == "WAIT" else "NO_TRADE"
            )
            result.position_multiplier = 0.0

        # Weighted contributions (for audit)
        for s in working:
            result.weighted_contributions[s.layer] = round(
                s.confidence * s.weight, 2
            )

        result.explanation = self._build_explanation(working, result)

        log.info(
            f"[SignalFusion] {result.final_signal} | "
            f"conf={result.master_confidence:.1f}% | "
            f"agreement={result.agreement} | "
            f"position={result.position_size}"
            f"{' | CONFLICT' if result.has_conflict else ''}"
            f"{' | excluded=' + ','.join(result.excluded_layers) if result.excluded_layers else ''}"
        )
        return result

    def _normalize_weights(
        self, signals: List[LayerSignal], result: FusionResult
    ) -> List[LayerSignal]:
        """Zero weight on abstaining / missing layers and redistribute.

        Returns a *new* list of LayerSignal with adjusted weights so the
        original caller objects are not mutated.
        """
        working: List[LayerSignal] = []
        active_weight = 0.0

        for s in signals:
            sig = (s.signal or "").upper().strip()
            # Treat NOT_READY / empty / very-low-confidence WAIT as non-voters
            is_abstain = (
                sig in self._ABSTAIN_SIGNALS
                or (sig in ("WAIT", "HOLD") and s.confidence < 30)
            )
            # ML layer that is present but not producing a real signal
            if s.layer == "ml_ensemble" and is_abstain:
                result.ml_available = False

            if is_abstain:
                result.excluded_layers.append(s.layer)
                working.append(
                    LayerSignal(
                        layer=s.layer,
                        signal=sig or "WAIT",
                        confidence=s.confidence,
                        weight=0.0,
                        reasoning=s.reasoning or "abstain/not ready",
                    )
                )
            else:
                working.append(
                    LayerSignal(
                        layer=s.layer,
                        signal=sig,
                        confidence=s.confidence,
                        weight=s.weight,
                        reasoning=s.reasoning,
                    )
                )
                active_weight += s.weight

        # Redistribute: active layers share the full 1.0 mass
        if active_weight > 0:
            for s in working:
                if s.weight > 0:
                    s.weight = s.weight / active_weight
        # else: all abstained — weights stay 0, final will be WAIT/NO_TRADE

        if result.excluded_layers:
            log.debug(
                f"[SignalFusion] excluded (weight→0): {result.excluded_layers}"
            )
        return working

    def _build_explanation(
        self, signals: List[LayerSignal], result: FusionResult
    ) -> List[str]:
        explanations = []
        for s in signals:
            if s.weight == 0:
                emoji = "⚪"
            elif s.signal == result.final_signal:
                emoji = "✅"
            else:
                emoji = "❌"
            explanations.append(
                f"{emoji} {s.layer}: {s.signal} ({s.confidence:.0f}%, "
                f"w={s.weight:.2f}) — {(s.reasoning or '')[:60]}"
            )
        if result.has_conflict:
            explanations.append(f"⚠️ Conflict: {result.conflict_reason}")
        explanations.append(
            f"→ Master: {result.final_signal} "
            f"({result.master_confidence:.0f}%) — {result.position_size}"
        )
        return explanations


# ── Singleton ───────────────────────────────────────────────────────

_FUSION: Optional[SignalFusion] = None


def get_signal_fusion() -> SignalFusion:
    global _FUSION
    if _FUSION is None:
        _FUSION = SignalFusion()
    return _FUSION
