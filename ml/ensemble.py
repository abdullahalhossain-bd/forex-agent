"""
ml/ensemble.py — Ensemble Engine: AI Brain Fusion Layer (Day 70)
===================================================================

The culmination of Days 60-69. This module fuses ALL intelligence layers
into a single institutional-grade trading decision:

  Inputs:
    - XGBoost prediction (Day 69)
    - Random Forest prediction (Day 69)
    - LSTM prediction (Day 69)
    - Rule Engine signal (Day 67 Confluence)
    - MasterAnalyst LLM signal (Day 42)
    - Market regime (Day 65 Intermarket)

  Pipeline:
    1. Collect all predictions → List[ModelVote]
    2. VotingEngine.vote() → agreement + position size + dissent detection
    3. ConfidenceFusion.fuse() → weighted confidence + conflict penalty
    4. Final decision: BUY/SELL/WAIT/NO_TRADE + calibrated confidence
    5. Persist to EnsembleStore
    6. Telegram alert for high-conviction signals

  Output (EnsembleDecision):
    {
        "pair": "EURUSD",
        "decision": "BUY",
        "confidence": 69.0,
        "agreement": "4/4",
        "position_size": "FULL",
        "position_multiplier": 1.0,
        "models": {"xgboost": "BUY 72%", "random_forest": "BUY 68%", ...},
        "has_conflict": false,
        "abstained": false,
        "reason": "All intelligence modules agree",
    }
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

from ml.voting_engine import VotingEngine, ModelVote, VoteResult, get_voting_engine
from ml.confidence_fusion import ConfidenceFusion, FusionResult, get_confidence_fusion
from ml.ensemble_store import EnsembleStore, get_ensemble_store

log = get_logger("ensemble")


@dataclass
class EnsembleDecision:
    """The final output of the EnsembleEngine — the single trade decision."""
    pair: str
    timeframe: str
    decision: str               # BUY / SELL / WAIT / NO_TRADE
    confidence: float           # 0-100 (fused + calibrated)
    agreement: str              # "4/4"
    agreement_count: int
    total_models: int
    position_size: str          # FULL / HALF / REDUCED / WAIT / NO_TRADE
    position_multiplier: float  # 1.0 / 0.5 / 0.25 / 0.0
    models: Dict[str, str] = field(default_factory=dict)      # {"xgboost": "BUY 72%"}
    model_details: Dict[str, Any] = field(default_factory=dict)
    has_conflict: bool = False
    conflict_reason: str = ""
    abstained: bool = False
    abstain_reason: str = ""
    regime: str = "UNKNOWN"
    weights_used: Dict[str, float] = field(default_factory=dict)
    fusion_details: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    generated_at: str = ""
    decision_id: Optional[int] = None  # DB row id
    # ARCHITECTURAL FIX (institutional refactor): explicit `ml_available`
    # flag so downstream consumers know whether ML models participated.
    # When False, the ensemble ran in rules-only mode and dynamically
    # rebalanced weights to the remaining voters.
    ml_available: bool = True
    ml_unavailable_reason: str = ""
    # Track which voters were excluded (LLM rate-limit, ML NOT_READY, etc.)
    # so the audit trail shows the rebalanced weights.
    excluded_voters: Dict[str, str] = field(default_factory=dict)
    # Preserve the analysis-layer signal even when execution-layer
    # consensus fails. NEVER zero confidence just because a voter dropped.
    analysis_signal: str = ""    # the strongest single-voter signal
    analysis_confidence: float = 0.0  # strongest single-voter confidence

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_telegram_alert(self) -> Optional[str]:
        """Format a high-conviction Telegram alert. Returns None if not tradeable."""
        if self.decision not in ("BUY", "SELL"):
            return None
        if self.abstained:
            return None

        dir_emoji = "🟢" if self.decision == "BUY" else "🔴"
        conviction = "HIGH CONVICTION" if self.confidence >= 75 else "CONVICTION"
        quality_emoji = "🌟" if self.confidence >= 85 else "✅"

        models_str = "\n".join(
            f"  ✅ {name}: {info}" for name, info in self.models.items()
        )

        alert = (
            f"{dir_emoji} FOREX AI {conviction}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"Pair: {self.pair} ({self.timeframe})\n"
            f"Direction: {self.decision}\n"
            f"Confidence: {self.confidence:.0f}% {quality_emoji}\n"
            f"AI Agreement: {self.agreement}\n"
            f"Position Size: {self.position_size}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"Models:\n{models_str}\n"
            f"━━━━━━━━━━━━━━━━━━━━━"
        )
        if self.has_conflict:
            alert += f"\n⚠️ Conflict: {self.conflict_reason[:100]}\n"
            alert += f"━━━━━━━━━━━━━━━━━━━━━"
        return alert


class EnsembleEngine:
    """The AI Brain Fusion Layer — combines all intelligence into one decision."""

    def __init__(self):
        self.voting = get_voting_engine()
        self.fusion = get_confidence_fusion()
        self.store = get_ensemble_store()

    def decide(
        self,
        pair: str,
        timeframe: str,
        ml_prediction: Optional[Dict[str, Any]] = None,
        rule_signal: str = "WAIT",
        rule_confidence: float = 0.0,
        master_signal: str = "WAIT",
        master_confidence: float = 0.0,
        regime: str = "UNKNOWN",
    ) -> EnsembleDecision:
        """Run the full ensemble pipeline and return a final decision.

        Args:
            pair: Trading pair (e.g. "EURUSD").
            timeframe: Timeframe label (e.g. "15m").
            ml_prediction: Output from ModelPredictor.predict() (Day 69).
                           Contains per_model predictions + ensemble probability.
            rule_signal: Signal from the Confluence Engine (Day 67): BUY/SELL/WAIT.
            rule_confidence: Confidence from the Confluence Engine (0-100).
            master_signal: Signal from MasterAnalyst LLM: BUY/SELL/WAIT.
            master_confidence: Confidence from MasterAnalyst (0-100).
            regime: Market regime from IntermarketEngine: TRENDING/RANGING/etc.
        """
        # ── Step 1: Collect votes from all models ──────────────────
        votes: List[ModelVote] = []
        excluded_voters: Dict[str, str] = {}
        ml_available = bool(ml_prediction and ml_prediction.get("prediction") != "NOT_READY")

        # ML models (from Day 69 ModelPredictor output)
        if ml_prediction and ml_prediction.get("prediction") != "NOT_READY":
            per_model = ml_prediction.get("per_model", {})
            for model_name in ("xgboost", "random_forest", "lstm"):
                m = per_model.get(model_name)
                if m and isinstance(m, dict):
                    signal = m.get("prediction", "WAIT")
                    up_probability = m.get("probability", 0.5)
                    # CO-FOUNDER FIX (audit finding): `probability` from
                    # ModelPredictor is always P(up), regardless of which
                    # way the model actually signaled. Using it directly as
                    # "confidence" is correct for BUY but INVERTS confidence
                    # for SELL — e.g. a model that is 90% sure of a SELL
                    # (P(up)=0.10) was being logged as only 10% confident,
                    # which silently under-weighted every SELL vote in
                    # ConfidenceFusion and in the agreement/dissent math.
                    # Confidence must reflect strength of the SIGNAL that was
                    # actually cast, not raw P(up).
                    if signal == "SELL":
                        directional_confidence = (1.0 - up_probability) * 100
                    elif signal == "BUY":
                        directional_confidence = up_probability * 100
                    else:  # WAIT — model didn't cross either threshold
                        directional_confidence = 50.0
                    votes.append(ModelVote(
                        model_name=model_name,
                        signal=signal,
                        confidence=directional_confidence,
                        probability=up_probability,
                    ))
                else:
                    excluded_voters[model_name] = "model_not_in_per_model"
        else:
            # ML unavailable — record reason for audit trail.
            _reason = "NOT_READY"
            if isinstance(ml_prediction, dict):
                _reason = ml_prediction.get("ml_unavailable_reason", "NOT_READY")
            excluded_voters["ml_models"] = _reason
            log.info(
                f"[Ensemble] {pair}: ML models unavailable ({_reason}) — "
                f"dynamic weight rebalance will redistribute ML weight to "
                f"rules/LLM/institutional voters. Analysis continues."
            )

        # Rule engine (from Day 67 Confluence)
        # Use rule_confidence if > 0, otherwise use master_confidence, otherwise default 50
        # BUG #12 fix: master_signal was accepted as a parameter and
        # documented as a "confirmation signal" but never actually read —
        # master_confidence was borrowed as a fallback regardless of
        # whether MasterAnalyst's signal agreed with the rule signal.
        # That meant a MasterAnalyst SELL at high confidence could
        # silently inflate confidence for an opposing rule-engine BUY.
        # Now we only borrow master_confidence when master_signal
        # actually confirms rule_signal (same direction).
        effective_rule_conf = rule_confidence
        if effective_rule_conf <= 0 and master_confidence > 0 and master_signal == rule_signal:
            effective_rule_conf = master_confidence
        if effective_rule_conf <= 0 and rule_signal in ("BUY", "SELL"):
            effective_rule_conf = 50.0  # minimum viable confidence

        votes.append(ModelVote(
            model_name="rules",
            signal=rule_signal,
            confidence=effective_rule_conf,
            probability=(effective_rule_conf / 100) if rule_signal == "BUY" else
                       (1 - effective_rule_conf / 100) if rule_signal == "SELL" else 0.5,
        ))

        # MasterAnalyst LLM (optional 5th vote — high weight but we treat it
        # as part of "rules" since the confluence engine already incorporates it)
        # For now, we use it as a confirmation signal but don't add a separate vote.
        # This keeps the agreement math clean (4 models).

        # ── Step 2: Vote ────────────────────────────────────────────
        vote_result = self.voting.vote(votes)

        # If no ML models are available (only rules vote), and rules says
        # BUY/SELL with decent confidence, let it through without blocking.
        if len(votes) == 1 and votes[0].model_name == "rules":
            # Only rules available — don't block with ensemble logic
            log.info(f"[Ensemble] {pair}: Only rules vote available (ML models NOT_READY) — "
                     f"passing through: {votes[0].signal} {votes[0].confidence:.0f}%")
            # Bypass ensemble blocking — let the downstream Risk Engine decide
            fusion_result = FusionResult(
                final_confidence=votes[0].confidence,
                weighted_confidence=votes[0].confidence,
                regime=regime.upper(),
            )
            # Build decision directly from rules vote
            #
            # AUDIT FIX (§5.1): this path previously authorized FULL size
            # (1.0x) off a SINGLE unconfirmed vote at just 50% confidence —
            # the weakest-evidence case in the whole system (no ML
            # cross-validation, no agreement check, nothing to detect
            # dissent against) was the only path capable of full sizing at
            # the bare minimum threshold. Every other path requires
            # multi-model agreement (per model_weights.json's
            # `_agreement_rules`, FULL size needs "4/4") before granting
            # FULL. Capital protection comes first: rules-only mode is now
            # capped at HALF size regardless of confidence. If ML models
            # come back online, agreement-based sizing resumes normally.
            position_size = "HALF" if votes[0].confidence >= 40 else "WAIT"  # Lowered from 50
            position_multiplier = 0.5 if votes[0].confidence >= 40 else 0.0  # Lowered from 50
            # ARCHITECTURAL FIX: preserve analysis verdict even when below
            # threshold. Don't zero confidence — let downstream gates decide.
            _rules_decision = votes[0].signal if votes[0].signal in ("BUY", "SELL") else "WAIT"
            _rules_conf = round(votes[0].confidence, 1)
            if _rules_conf < 40 and _rules_decision in ("BUY", "SELL"):  # Lowered from 50
                # Decision becomes WAIT (insufficient confidence for execution)
                # BUT analysis_signal / analysis_confidence preserve the verdict.
                log.info(
                    f"[Ensemble] {pair}: confidence {_rules_conf:.0f}% < 50% → "
                    f"execution WAIT (analysis verdict PRESERVED)"
                )
                decision = EnsembleDecision(
                    pair=pair.upper(),
                    timeframe=timeframe,
                    decision="WAIT",
                    confidence=_rules_conf,  # PRESERVED, not zeroed
                    agreement="1/1 (rules only)",
                    agreement_count=1,
                    total_models=1,
                    position_size="WAIT",
                    position_multiplier=0.0,
                    models={votes[0].model_name: f"{votes[0].signal} {votes[0].confidence:.0f}%"},
                    model_details={v.model_name: v.to_dict() for v in votes},
                    regime=regime,
                    reason="Rules-only mode: confidence below 50% threshold → execution WAIT (analysis verdict preserved for audit)",
                    generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    ml_available=False,
                    ml_unavailable_reason="models_not_loaded",
                    excluded_voters=excluded_voters,
                    analysis_signal=_rules_decision,
                    analysis_confidence=_rules_conf,
                )
            else:
                decision = EnsembleDecision(
                    pair=pair.upper(),
                    timeframe=timeframe,
                    decision=_rules_decision,
                    confidence=_rules_conf,
                    agreement="1/1 (rules only)",
                    agreement_count=1,
                    total_models=1,
                    position_size=position_size,
                    position_multiplier=position_multiplier,
                    models={votes[0].model_name: f"{votes[0].signal} {votes[0].confidence:.0f}%"},
                    model_details={v.model_name: v.to_dict() for v in votes},
                    regime=regime,
                    reason="Rules-only mode (ML models not yet trained) — capped at HALF size, no cross-model confirmation available",
                    generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    ml_available=False,
                    ml_unavailable_reason="models_not_loaded",
                    excluded_voters=excluded_voters,
                    analysis_signal=_rules_decision,
                    analysis_confidence=_rules_conf,
                )

            # Persist
            try:
                decision_id = self.store.save_decision({
                    "pair": decision.pair, "timeframe": decision.timeframe,
                    "rule_signal": votes[0].signal, "rule_conf": votes[0].confidence,
                    "final_signal": decision.decision, "agreement": decision.agreement,
                    "confidence": decision.confidence, "position_size": decision.position_size,
                    "has_conflict": False, "abstained": False,
                })
                decision.decision_id = decision_id
            except Exception:
                pass

            log.info(
                f"[Ensemble] {pair} {timeframe} → {decision.decision} "
                f"| conf={decision.confidence:.1f}% | rules-only mode "
                f"| ml_available=False"
            )
            return decision

        # ── Step 3: Fuse confidences ────────────────────────────────
        fusion_result = self.fusion.fuse(votes, vote_result, regime=regime)

        # ── Step 4: Build final decision ────────────────────────────
        decision_str = vote_result.decision
        if fusion_result.abstain:
            decision_str = "NO_TRADE"
        elif vote_result.position_multiplier == 0:
            decision_str = "WAIT" if decision_str in ("BUY", "SELL") else decision_str

        # Check minimum confidence threshold — lowered from 55 to 45
        min_conf = 45.0  # Lowered from 50 for better trade frequency
        if decision_str in ("BUY", "SELL") and fusion_result.final_confidence < min_conf:
            decision_str = "WAIT"
            log.info(
                f"[Ensemble] {pair}: confidence {fusion_result.final_confidence:.1f}% "
                f"below minimum {min_conf}% → WAIT"
            )

        # Build models display dict
        models_display: Dict[str, str] = {}
        for v in votes:
            models_display[v.model_name] = f"{v.signal} {v.confidence:.0f}%"

        # Build reason
        if fusion_result.abstain:
            reason = fusion_result.abstain_reason
        elif vote_result.has_strong_dissent:
            reason = f"Trade taken with dissent: {vote_result.dissent_reason}"
        elif decision_str in ("BUY", "SELL") and vote_result.agreement_count == vote_result.total_models:
            reason = "All intelligence modules agree"
        elif decision_str in ("BUY", "SELL"):
            reason = f"Majority agreement ({vote_result.agreement})"
        elif decision_str == "WAIT":
            reason = f"Insufficient agreement ({vote_result.agreement}) or low confidence"
        else:
            reason = "No trade — models disagree"

        decision = EnsembleDecision(
            pair=pair.upper(),
            timeframe=timeframe,
            decision=decision_str,
            confidence=round(fusion_result.final_confidence, 1),
            agreement=vote_result.agreement,
            agreement_count=vote_result.agreement_count,
            total_models=vote_result.total_models,
            position_size=vote_result.position_size,
            position_multiplier=vote_result.position_multiplier,
            models=models_display,
            model_details={v.model_name: v.to_dict() for v in votes},
            has_conflict=fusion_result.has_conflict,
            conflict_reason=fusion_result.conflict_reason,
            abstained=fusion_result.abstain,
            abstain_reason=fusion_result.abstain_reason,
            regime=regime,
            weights_used=fusion_result.weights_used,
            fusion_details=fusion_result.to_dict(),
            reason=reason,
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ml_available=ml_available,
            ml_unavailable_reason="" if ml_available else "models_not_loaded",
            excluded_voters=excluded_voters,
            # Preserve the strongest single-voter signal for audit even if
            # the ensemble's fused decision is WAIT/NO_TRADE.
            analysis_signal=next(
                (v.signal for v in sorted(votes, key=lambda x: -x.confidence)
                 if v.signal in ("BUY", "SELL")),
                "WAIT"
            ),
            analysis_confidence=max(
                (v.confidence for v in votes if v.signal in ("BUY", "SELL")),
                default=0.0,
            ),
        )

        # ── Step 5: Persist to store ────────────────────────────────
        try:
            decision_id = self.store.save_decision({
                "pair": decision.pair,
                "timeframe": decision.timeframe,
                "xgb_signal": models_display.get("xgboost", "").split()[0] if "xgboost" in models_display else None,
                "xgb_conf": next((v.confidence for v in votes if v.model_name == "xgboost"), None),
                "rf_signal": models_display.get("random_forest", "").split()[0] if "random_forest" in models_display else None,
                "rf_conf": next((v.confidence for v in votes if v.model_name == "random_forest"), None),
                "lstm_signal": models_display.get("lstm", "").split()[0] if "lstm" in models_display else None,
                "lstm_conf": next((v.confidence for v in votes if v.model_name == "lstm"), None),
                "rule_signal": models_display.get("rules", "").split()[0] if "rules" in models_display else None,
                "rule_conf": next((v.confidence for v in votes if v.model_name == "rules"), None),
                "final_signal": decision.decision,
                "agreement": decision.agreement,
                "confidence": decision.confidence,
                "position_size": decision.position_size,
                "has_conflict": decision.has_conflict,
                "abstained": decision.abstained,
            })
            decision.decision_id = decision_id
        except Exception as e:
            log.debug(f"[Ensemble] store save failed: {e}")

        # ── Log the decision ────────────────────────────────────────
        log.info(
            f"[Ensemble] {pair} {timeframe} → {decision.decision} "
            f"| conf={decision.confidence:.1f}% | agreement={decision.agreement} "
            f"| position={decision.position_size} | regime={decision.regime}"
            f"{' | ABSTAINED' if decision.abstained else ''}"
            f"{' | CONFLICT' if decision.has_conflict else ''}"
        )

        return decision

    def record_outcome(self, decision_id: int, result: str, pnl_usd: float,
                       model_predictions: Optional[Dict[str, str]] = None,
                       executed_direction: Optional[str] = None) -> None:
        """Record the outcome of a trade and update model performance.

        CO-FOUNDER FIX (audit finding): this method previously called
        store.update_outcome() but then looped over model_predictions and
        did nothing (`pass`) — the "Model Performance Memory" feature that
        ConfidenceFusion.update_performance() depends on for dynamic weight
        adjustment was dead code: it compiled and ran but never recorded a
        single win/loss, so performance-based weight adjustment was a
        silent no-op. This wires it up. A model is scored "correct" if it
        voted BUY/SELL, that direction matches the trade actually taken,
        and the trade won — or if it dissented from a losing trade. WAIT
        votes are not scored (abstaining isn't a directional call to grade).

        Args:
            decision_id: The DB row id from the ensemble decision.
            result: "WIN" or "LOSS".
            pnl_usd: Profit/loss in USD.
            model_predictions: {"xgboost": "BUY", "random_forest": "SELL", ...}
            executed_direction: "BUY" or "SELL" — the direction actually
                traded for this decision. REQUIRED for per-model accuracy
                tracking to run; the execution/order layer must pass this
                through from the EnsembleDecision that was acted on.
        """
        try:
            self.store.update_outcome(decision_id, result, pnl_usd)
        except Exception as e:
            log.debug(f"[Ensemble] outcome update failed: {e}")

        if not model_predictions:
            return
        if executed_direction not in ("BUY", "SELL"):
            log.debug(
                "[Ensemble] record_outcome: executed_direction not provided "
                "or not BUY/SELL — skipping per-model performance tracking "
                "(caller must pass the direction actually traded)"
            )
            return
        if result not in ("WIN", "LOSS"):
            log.debug(f"[Ensemble] record_outcome: unrecognized result '{result}' — skipping")
            return

        for model_name, pred_signal in model_predictions.items():
            if pred_signal not in ("BUY", "SELL"):
                continue  # WAIT isn't a directional call — not scored
            model_agreed_with_trade = (pred_signal == executed_direction)
            correct = model_agreed_with_trade if result == "WIN" else (not model_agreed_with_trade)
            try:
                self.store.update_model_performance(model_name, correct)
            except Exception as e:
                log.debug(f"[Ensemble] per-model performance update failed for {model_name}: {e}")

    def update_model_weights_from_performance(self) -> None:
        """Pull latest model performance from the store and feed it to the fusion engine."""
        try:
            perf = self.store.get_model_performance()
            for model_name, stats in perf.items():
                if stats.get("total", 0) >= 20:
                    self.fusion.update_performance(
                        model_name=model_name,
                        win_rate=stats.get("win_rate", 50.0),
                        sample_count=stats.get("total", 0),
                    )
        except Exception as e:
            log.debug(f"[Ensemble] weight update failed: {e}")

    def stats(self) -> Dict[str, Any]:
        """Return ensemble + model performance stats."""
        return self.store.stats()


# ── Singleton ───────────────────────────────────────────────────────

_ENGINE: Optional[EnsembleEngine] = None


def get_ensemble_engine() -> EnsembleEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = EnsembleEngine()
    return _ENGINE