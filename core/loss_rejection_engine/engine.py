"""3-Layer Loss Rejection Engine - Main Orchestrator.
Coordinates L1 (structural), L2 (meta-labeler), L3 (OOD detector).""" 
from __future__ import annotations
import logging, os, time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

LRE_ENABLED = os.getenv("LRE_ENABLED", "1") == "1"
LRE_SHADOW_MODE = os.getenv("LRE_SHADOW_MODE", "1") == "1"

from core.loss_rejection_engine.layer1_structural_filters import (
    StructuralFilterLayer, Layer1Output,
)
from core.loss_rejection_engine.layer2_meta_labeler import (
    MetaLabeler, MetaLabelerOutput,
)
from core.loss_rejection_engine.layer3_ood_detector import (
    OODDetector, OODOutput,
)

@dataclass
class LREResult:
    blocked: bool = False
    shadow_blocked: bool = False
    l1: Optional[Layer1Output] = None
    l2: Optional[MetaLabelerOutput] = None
    l3: Optional[OODOutput] = None
    composite_verdict: str = "PASS"
    confidence_penalty: float = 0.0
    reason: str = ""
    processing_time_ms: float = 0.0


class LossRejectionEngine:
    """3-Layer Loss Rejection Engine.

    Layer 1: Rule-based structural filters (10 filters, composite score)
    Layer 2: ML Meta Labeler (binary accept/reject)
    Layer 3: OOD Detector (distribution shift detection)

    SHADOW MODE (default ON): logs rejections but does NOT block trades.
    When shadow mode is OFF: REJECT verdicts actually block trades.
    WARN verdicts: downgrade confidence but do not block.
    Any error defaults to PASS (fail-safe).
    """

    def __init__(self):
        self.layer1 = StructuralFilterLayer()
        self.layer2 = MetaLabeler()
        self.layer3 = OODDetector()
        self._total_evaluated = 0
        self._total_blocked = 0
        self._total_shadow_blocked = 0
        self._start_time = time.time()
        log.info("[LRE] 3-Layer Loss Rejection Engine initialized")
        log.info("[LRE] Shadow mode: %s | Enabled: %s", LRE_SHADOW_MODE, LRE_ENABLED)

    def evaluate(self, dec_out: Dict[str, Any], analysis_out: Dict[str, Any],
                 market_out: Dict[str, Any], *, symbol: str = "") -> LREResult:
        t0 = time.time()
        self._total_evaluated += 1
        direction = (dec_out.get("decision") or "WAIT").upper()
        result = LREResult()

        if not LRE_ENABLED:
            return result
        if direction not in ("BUY", "SELL"):
            return result

        # Layer 1: Structural Filters
        try:
            l1_out = self.layer1.evaluate(dec_out, analysis_out, market_out, symbol=symbol)
            result.l1 = l1_out
            dec_out["_lre_l1_score"] = l1_out.composite_score
            dec_out["_lre_l1_verdict"] = l1_out.verdict
            if not l1_out.pass_through:
                result.blocked = True
                result.shadow_blocked = True
                result.composite_verdict = "REJECT"
                result.reason = f"L1 REJECT: {l1_out.primary_reason}"
        except Exception as e:
            log.warning("[LRE] Layer 1 error (non-fatal): %s", e)

        # Layer 2: Meta Labeler
        try:
            l2_out = self.layer2.evaluate(dec_out, analysis_out, market_out, symbol=symbol)
            result.l2 = l2_out
            dec_out["_lre_l2_verdict"] = l2_out.verdict
            dec_out["_lre_l2_loss_prob"] = l2_out.loss_probability
            if not l2_out.pass_through:
                result.blocked = True
                result.shadow_blocked = True
                result.composite_verdict = "REJECT"
                result.reason = f"L2 REJECT: {l2_out.reason}"
            elif l2_out.verdict == "WARN":
                result.confidence_penalty += 5.0
        except Exception as e:
            log.warning("[LRE] Layer 2 error (non-fatal): %s", e)

        # Layer 3: OOD Detector
        try:
            l3_out = self.layer3.evaluate(dec_out, analysis_out, market_out, symbol=symbol)
            result.l3 = l3_out
            dec_out["_lre_l3_verdict"] = l3_out.verdict
            dec_out["_lre_l3_distance"] = l3_out.distance
            if not l3_out.pass_through:
                result.blocked = True
                result.shadow_blocked = True
                result.composite_verdict = "REJECT"
                result.reason = f"L3 REJECT: {l3_out.reason}"
            elif l3_out.verdict == "WARN":
                result.confidence_penalty += 5.0
        except Exception as e:
            log.warning("[LRE] Layer 3 error (non-fatal): %s", e)

        # Record features for L2/L3 learning
        meta_features = dec_out.get("_lre_meta_features")
        if meta_features:
            try: self.layer2._feature_buffer.append(meta_features)
            except: pass
        ood_features = dec_out.get("_lre_ood_features")
        if ood_features:
            try: self.layer3.record_features(ood_features)
            except: pass

        # Composite verdict for WARN
        if not result.blocked:
            warns = []
            if result.l1 and result.l1.verdict == "WARN": warns.append("L1")
            if result.l2 and result.l2.verdict == "WARN": warns.append("L2")
            if result.l3 and result.l3.verdict == "WARN": warns.append("L3")
            if warns:
                result.composite_verdict = "WARN"
                result.reason = f"WARN from: {', '.join(warns)}"
                result.confidence_penalty += 3.0 * len(warns)

        # Apply confidence penalty
        if result.confidence_penalty > 0 and "confidence" in dec_out:
            orig = dec_out["confidence"]
            dec_out["confidence"] = max(0, orig - result.confidence_penalty)
            dec_out["_lre_confidence_penalty"] = result.confidence_penalty
            log.debug("[LRE] Confidence: %.0f -> %.0f", orig, dec_out["confidence"])

        # Shadow mode: log but do not block
        if LRE_SHADOW_MODE and result.blocked:
            result.blocked = False
            result.shadow_blocked = True
            self._total_shadow_blocked += 1
            log.info("[LRE] SHADOW BLOCK | %s %s | %s | L1=%s L2=%s L3=%s",
                    direction, symbol, result.reason,
                    result.l1.verdict if result.l1 else "N/A",
                    result.l2.verdict if result.l2 else "N/A",
                    result.l3.verdict if result.l3 else "N/A")
        elif result.blocked:
            self._total_blocked += 1
            log.info("[LRE] BLOCKED | %s %s | %s", direction, symbol, result.reason)

        result.processing_time_ms = (time.time() - t0) * 1000
        return result

    def record_trade_outcome(self, symbol, direction, pnl, **context):
        price_zone = context.get("price_zone", "mid")
        regime = context.get("regime", "unknown")
        try: self.layer1.record_trade_outcome(symbol, direction, price_zone, regime, pnl)
        except: pass

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_evaluated": self._total_evaluated,
            "total_blocked": self._total_blocked,
            "total_shadow_blocked": self._total_shadow_blocked,
            "shadow_mode": LRE_SHADOW_MODE,
            "enabled": LRE_ENABLED,
            "uptime_seconds": time.time() - self._start_time,
        }
