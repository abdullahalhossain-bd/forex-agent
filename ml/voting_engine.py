"""
ml/voting_engine.py — Voting & agreement system (Day 70)
==========================================================

Collects predictions from all AI brains (XGBoost, RF, LSTM, Rules) and
applies professional voting rules:

  Agreement | Action      | Position Size
  ----------|-------------|-------------
  4/4       | FULL        | 100% of calculated lot
  3/4       | HALF        | 50% of calculated lot
  2/4       | WAIT        | 0% (no trade)
  1/4       | NO_TRADE    | 0%
  0/4       | NO_TRADE    | 0%

The voting engine also detects "strong dissent" — when one model strongly
opposes the majority. In that case, even a 3/4 agreement is downgraded
to "reduced position" and a conflict warning is raised.

Output:
    {
        "decision": "BUY" | "SELL" | "WAIT" | "NO_TRADE",
        "agreement": "3/4",
        "agreement_count": 3,
        "total_models": 4,
        "position_size": "HALF",
        "position_multiplier": 0.5,
        "has_strong_dissent": bool,
        "dissenting_model": "lstm",
        "dissent_reason": "LSTM strongly opposes (SELL 80% vs majority BUY)",
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

log = get_logger("voting_engine")


# ── Position size mapping ───────────────────────────────────────────
# Lowered requirements: 2/4 agreement now gets REDUCED position instead of WAIT.
# Original was too strict — 2/4 = WAIT blocked too many good trades.
def _agreement_position(agreement_count: int, total_models: int) -> tuple:
    """Bug #9 fix: dynamic position sizing for any number of voters.

    Previously AGREEMENT_TO_POSITION only had keys {0..4}, so 5/5
    agreement (possible when LLM/RL voters are added) defaulted to
    NO_TRADE. Now handles any voter count dynamically.
    """
    if total_models <= 0:
        return ("NO_TRADE", 0.0)
    ratio = agreement_count / total_models
    if ratio >= 1.0:
        return ("FULL", 1.0)
    elif ratio >= 0.75:
        return ("HALF", 0.5)
    elif ratio >= 0.5:
        return ("REDUCED", 0.25)
    else:
        return ("NO_TRADE", 0.0)


# Legacy mapping kept for backward compat (4-voter systems)
AGREEMENT_TO_POSITION = {
    4: ("FULL", 1.0),
    3: ("HALF", 0.5),
    2: ("REDUCED", 0.25),
    1: ("NO_TRADE", 0.0),
    0: ("NO_TRADE", 0.0),
}

# Strong dissent threshold: if a model's confidence in the OPPOSITE direction
# is above this, it counts as "strong dissent" even if majority agrees.
STRONG_DISSENT_THRESHOLD = 70.0


@dataclass
class ModelVote:
    """One model's vote."""
    model_name: str          # xgboost / random_forest / lstm / rules
    signal: str              # BUY / SELL / WAIT
    confidence: float        # 0-100
    probability: float = 0.5  # BUY probability (0-1)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VoteResult:
    """Result of the voting process."""
    decision: str = "NO_TRADE"       # BUY / SELL / WAIT / NO_TRADE
    agreement: str = "0/0"           # "3/4"
    agreement_count: int = 0
    total_models: int = 0
    position_size: str = "NO_TRADE"  # FULL / HALF / WAIT / NO_TRADE
    position_multiplier: float = 0.0
    majority_confidence: float = 0.0  # Round-22: avg confidence of winning side
    buy_votes: int = 0
    sell_votes: int = 0
    wait_votes: int = 0
    has_strong_dissent: bool = False
    dissenting_models: List[str] = field(default_factory=list)
    dissent_reason: str = ""
    votes: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VotingEngine:
    """Collects model votes and produces a decision with position sizing."""

    def vote(self, model_votes: List[ModelVote]) -> VoteResult:
        """Run the voting process on a list of model votes."""
        result = VoteResult(total_models=len(model_votes))
        result.votes = [v.to_dict() for v in model_votes]

        if not model_votes:
            return result

        # Count votes
        buy_votes = [v for v in model_votes if v.signal == "BUY"]
        sell_votes = [v for v in model_votes if v.signal == "SELL"]
        wait_votes = [v for v in model_votes if v.signal == "WAIT"]

        result.buy_votes = len(buy_votes)
        result.sell_votes = len(sell_votes)
        result.wait_votes = len(wait_votes)

        # Determine majority direction
        if result.buy_votes > result.sell_votes and result.buy_votes > result.wait_votes:
            decision = "BUY"
            agreement_count = result.buy_votes
            majority_confidence = sum(v.confidence for v in buy_votes) / len(buy_votes) if buy_votes else 0
        elif result.sell_votes > result.buy_votes and result.sell_votes > result.wait_votes:
            decision = "SELL"
            agreement_count = result.sell_votes
            majority_confidence = sum(v.confidence for v in sell_votes) / len(sell_votes) if sell_votes else 0
        else:
            # Tie or WAIT majority
            decision = "WAIT"
            agreement_count = max(result.buy_votes, result.sell_votes)
            majority_confidence = 0

        result.decision = decision
        result.agreement_count = agreement_count
        result.agreement = f"{agreement_count}/{len(model_votes)}"
        result.majority_confidence = round(majority_confidence, 1)  # Round-22: wire it in

        # Position size from agreement (Bug #9: dynamic for any voter count)
        pos_label, pos_mult = _agreement_position(agreement_count, len(model_votes))
        result.position_size = pos_label
        result.position_multiplier = pos_mult

        # If WAIT or NO_TRADE, skip dissent check
        if decision in ("WAIT", "NO_TRADE") or pos_mult == 0:
            return result

        # ── Strong dissent detection ───────────────────────────────
        # Check if any model STRONGLY opposes the majority direction
        opposing_direction = "SELL" if decision == "BUY" else "BUY"
        dissenters: List[str] = []
        for v in model_votes:
            if v.signal == opposing_direction and v.confidence >= STRONG_DISSENT_THRESHOLD:
                dissenters.append(v.model_name)
                result.has_strong_dissent = True

        if dissenters:
            result.dissenting_models = dissenters
            result.dissent_reason = (
                f"{'/'.join(dissenters)} strongly opposes "
                f"({opposing_direction} ≥{STRONG_DISSENT_THRESHOLD:.0f}% vs majority {decision})"
            )
            # Downgrade position: FULL → HALF, HALF → reduced
            if result.position_size == "FULL":
                result.position_size = "HALF"
                result.position_multiplier = 0.5
                log.warning(
                    f"[Voting] Strong dissent from {dissenters} — "
                    f"downgraded FULL → HALF"
                )
            elif result.position_size == "HALF":
                result.position_size = "REDUCED"
                result.position_multiplier = 0.25
                log.warning(
                    f"[Voting] Strong dissent from {dissenters} — "
                    f"downgraded HALF → REDUCED (25%)"
                )

        return result


# ── Singleton ───────────────────────────────────────────────────────

_ENGINE: Optional[VotingEngine] = None


def get_voting_engine() -> VotingEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = VotingEngine()
    return _ENGINE
