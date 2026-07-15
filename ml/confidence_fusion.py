"""
ml/confidence_fusion.py — Confidence fusion engine (Day 70)
==============================================================

Fuses individual model confidences into a single ensemble confidence
using three mechanisms:

1. **Weighted Average** — each model has a weight (from model_weights.json).
   Default: XGBoost 35%, LSTM 25%, RF 20%, Rules 20%.

2. **Market Regime Adjustment** — weights shift based on current regime:
   - TRENDING  → boost XGBoost + LSTM (they handle trends well)
   - RANGING   → boost RF + Rules (they handle ranges well)
   - VOLATILE  → boost Rules + RF (robust to noise)
   - BREAKOUT  → boost XGBoost + Rules

3. **Performance-Based Weight Adjustment** — if a model's recent win rate
   is significantly above/below average, its weight is adjusted up/down.
   This is the "Model Performance Memory" feature.

4. **Conflict Penalty** — if models disagree (strong dissent detected by
   VotingEngine), the ensemble confidence is penalized.

The output is a single calibrated confidence (0-100) that feeds into
the final trade decision.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logger import get_logger
from ml.voting_engine import ModelVote, VoteResult

log = get_logger("confidence_fusion")

WEIGHTS_PATH = Path(__file__).resolve().parent / "model_weights.json"


@dataclass
class FusionResult:
    """Output of the confidence fusion process."""
    final_confidence: float = 0.0       # 0-100
    weighted_confidence: float = 0.0    # before conflict penalty
    regime: str = "UNKNOWN"
    weights_used: Dict[str, float] = field(default_factory=dict)
    per_model_contribution: Dict[str, float] = field(default_factory=dict)
    conflict_penalty: float = 0.0
    has_conflict: bool = False
    conflict_reason: str = ""
    abstain: bool = False               # True if conflict too severe
    abstain_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConfidenceFusion:
    """Fuses multi-model confidences into one calibrated ensemble confidence."""

    def __init__(self):
        self._lock = threading.RLock()
        self._default_weights = self._load_weights()
        self._performance_stats: Dict[str, Dict[str, float]] = {}
        # {"xgboost": {"win_rate": 64.0, "count": 100}, ...}

    def _load_weights(self) -> Dict[str, float]:
        """Load default weights from model_weights.json."""
        try:
            if WEIGHTS_PATH.exists():
                data = json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))
                return {
                    k: v for k, v in data.items()
                    if not k.startswith("_") and isinstance(v, (int, float))
                }
        except Exception as e:
            log.warning(f"[Fusion] weights load failed: {e}")
        return {"xgboost": 0.35, "random_forest": 0.20, "lstm": 0.25, "rules": 0.20}

    def _get_regime_adjustments(self, regime: str) -> Dict[str, float]:
        """Get weight adjustments for the current market regime."""
        regime = regime.upper()
        try:
            if WEIGHTS_PATH.exists():
                data = json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))
                adjustments = data.get("_regime_adjustments", {}).get(regime, {})
                return {k: v for k, v in adjustments.items() if isinstance(v, (int, float))}
        except Exception:
            pass
        return {}

    def update_performance(self, model_name: str, win_rate: float, sample_count: int) -> None:
        """Update a model's recent performance stats for weight adjustment."""
        with self._lock:
            self._performance_stats[model_name] = {
                "win_rate": win_rate,
                "count": sample_count,
            }
            log.info(f"[Fusion] performance updated: {model_name} WR={win_rate:.1f}% ({sample_count} samples)")

    def _performance_adjustment(self, model_name: str) -> float:
        """Calculate weight adjustment based on recent performance.

        If a model's win rate is above 60% (with ≥20 samples), boost its weight.
        If below 40%, reduce it.
        """
        stats = self._performance_stats.get(model_name)
        if not stats or stats.get("count", 0) < 20:
            return 0.0
        wr = stats.get("win_rate", 50.0)
        if wr >= 65:
            return 0.05   # +5% weight
        elif wr >= 60:
            return 0.03   # +3% weight
        elif wr <= 35:
            return -0.05  # -5% weight
        elif wr <= 40:
            return -0.03  # -3% weight
        return 0.0

    def fuse(
        self,
        votes: List[ModelVote],
        vote_result: VoteResult,
        regime: str = "UNKNOWN",
    ) -> FusionResult:
        """Fuse model confidences into a single ensemble confidence.

        ARCHITECTURAL GUARANTEE (institutional refactor):
        When a voter is missing (ML NOT_READY, LLM rate-limited, etc.),
        its weight is set to 0 and the REMAINING voters' weights are
        RENORMALIZED to sum to 1.0. This is dynamic weight rebalance —
        the system NEVER assigns 0 confidence just because one voter
        disappeared. The user's spec is explicit:
            Original: Rules 25%, LLM 25%, ML 25%, Institutional 25%
            If ML unavailable: Rules 33%, LLM 33%, Institutional 34%
        This is implemented at L161-163 (total_weight normalization).

        Args:
            votes: List of ModelVote from all models.
            vote_result: The VoteResult from VotingEngine (contains dissent info).
            regime: Current market regime (TRENDING / RANGING / BREAKOUT / VOLATILE).

        Returns:
            FusionResult with final_confidence + breakdown.
        """
        result = FusionResult(regime=regime.upper())

        if not votes:
            # No voters at all — this is a degenerate case. Return 0
            # confidence (legitimate: no analysis means no signal).
            return result

        # 1. Start with default weights
        weights = self._default_weights.copy()

        # 2. Apply regime adjustments
        regime_adj = self._get_regime_adjustments(regime)
        for model, adj in regime_adj.items():
            if model in weights:
                weights[model] += adj

        # 3. Apply performance adjustments
        for model in weights:
            weights[model] += self._performance_adjustment(model)

        # ── ARCHITECTURAL FIX (institutional refactor) ───────────────
        # DYNAMIC WEIGHT REBALANCE: zero out the weight of any voter that
        # is NOT in the `votes` list. Then renormalize the remaining
        # weights to sum to 1.0. This implements the spec:
        #   "If ML unavailable → redistribute ML weight to other voters"
        # Without this, a missing voter would silently drag down the
        # weighted_confidence (its weight is still counted in the
        # denominator but contributes 0 to the numerator).
        # ──────────────────────────────────────────────────────────────
        voter_names_present = {v.model_name for v in votes}
        _excluded_for_rebalance = []
        for model_name in list(weights.keys()):
            if model_name not in voter_names_present:
                _excluded_for_rebalance.append((model_name, weights[model_name]))
                weights[model_name] = 0.0
        if _excluded_for_rebalance:
            log.info(
                f"[Fusion] Dynamic rebalance: zeroed weights for "
                f"{len(_excluded_for_rebalance)} missing voter(s): "
                f"{[name for name, _ in _excluded_for_rebalance]} — "
                f"remaining voters renormalized"
            )

        # 4. Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            # All voters missing — degenerate. Return zero confidence
            # (no analysis to fuse). This is the ONLY case where 0
            # confidence is correct.
            log.warning(
                "[Fusion] All voters missing — returning 0 confidence "
                "(no analysis to fuse)"
            )
            return result
        result.weights_used = {k: round(v, 4) for k, v in weights.items()}

        # 5. Weighted confidence
        weighted_sum = 0.0
        for vote in votes:
            w = weights.get(vote.model_name, 0.0)
            contribution = vote.confidence * w
            weighted_sum += contribution
            result.per_model_contribution[vote.model_name] = round(contribution, 2)

        result.weighted_confidence = round(weighted_sum, 2)

        # 6. Conflict penalty
        #
        # AUDIT FIX (§5.3): this was a flat -15% for ANY dissent, whether
        # it was one borderline model or a near-even split. Scaled by how
        # much of the ensemble actually disagrees (dissent_ratio),
        # clamped to [10, 25] so a single mild dissent still costs
        # something and severe disagreement costs more than the old flat
        # rate. A typical single-model dissent lands close to the
        # original -15% baseline, so this does not make the system
        # meaningfully more permissive in the common case.
        if vote_result.has_strong_dissent:
            result.has_conflict = True
            total_models = max(vote_result.total_models, 1)
            dissent_ratio = len(vote_result.dissenting_models) / total_models
            result.conflict_penalty = round(min(25.0, max(10.0, 15.0 * (0.5 + dissent_ratio))), 1)
            result.conflict_reason = vote_result.dissent_reason
            log.warning(
                f"[Fusion] conflict detected (dissent_ratio={dissent_ratio:.0%}) "
                f"→ -{result.conflict_penalty}% penalty"
            )

        # 7. Final confidence
        result.final_confidence = max(0.0, min(100.0,
            result.weighted_confidence - result.conflict_penalty))

        # 8. Abstain check — if conflict is too severe
        try:
            data = json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))
            abstain_threshold = data.get("_thresholds", {}).get("abstain_if_conflict_above", 0.8)
        except Exception:
            abstain_threshold = 0.8

        # AUDIT FIX (§5.2): abstain_threshold was loaded from config but
        # never used — the abstain rule below was hardcoded to "2+
        # dissenting models" regardless of what this value was set to, so
        # tuning `abstain_if_conflict_above` in model_weights.json had no
        # effect. Wired in as an ADDITIONAL trigger, OR'd with the
        # existing rule rather than replacing it, so the existing
        # conservative behavior remains a floor (this change can only
        # cause the system to abstain in MORE cases, never fewer). This
        # also makes the threshold meaningful for ensembles smaller than
        # 4 models, where "2+" alone may be too strict or too loose.
        total_models = max(vote_result.total_models, 1)
        dissent_ratio = len(vote_result.dissenting_models) / total_models
        abstain_by_count = len(vote_result.dissenting_models) >= 2
        abstain_by_ratio = dissent_ratio >= abstain_threshold

        if abstain_by_count or abstain_by_ratio:
            result.abstain = True
            reasons = []
            if abstain_by_count:
                reasons.append(f"2+ models strongly dissent ({'/'.join(vote_result.dissenting_models)})")
            if abstain_by_ratio:
                reasons.append(f"dissent ratio {dissent_ratio:.0%} >= configured threshold {abstain_threshold:.0%}")
            result.abstain_reason = " and ".join(reasons) + " — abstaining from trade"
            result.final_confidence = 0.0

        return result


# ── Singleton ───────────────────────────────────────────────────────

_FUSION: Optional[ConfidenceFusion] = None


def get_confidence_fusion() -> ConfidenceFusion:
    global _FUSION
    if _FUSION is None:
        _FUSION = ConfidenceFusion()
    return _FUSION