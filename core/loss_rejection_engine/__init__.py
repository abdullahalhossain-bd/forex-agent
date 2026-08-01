"""
core/loss_rejection_engine/ - 3-Layer Loss Rejection Engine (Meta Labeling)

Architecture:
  Layer 1: Rule-based structural filters (10 proprietary filters)
  Layer 2: ML-based Meta Labeler (binary accept/reject classifier)
  Layer 3: Out-of-Distribution (OOD) detector (unseen market conditions)

Key Metrics:
  - Loss Rejection Rate (LRR) = blocked_losses / total_losses
  - Winner Preservation Rate (WPR) = kept_winners / total_winners (target >=95%)

Env Vars: LRE_ENABLED, LRE_SHADOW_MODE, LRE_L1_REJECT, LRE_L1_WARN,
          LRE_META_REJECT, LRE_META_WARN, LRE_OOD_REJECT, LRE_OOD_WARN
"""
from core.loss_rejection_engine.engine import LossRejectionEngine, LREResult
from core.loss_rejection_engine.layer1_structural_filters import (
    StructuralFilterLayer, Layer1Output, FilterResult,
)
from core.loss_rejection_engine.layer2_meta_labeler import MetaLabeler, MetaLabelerOutput
from core.loss_rejection_engine.layer3_ood_detector import OODDetector, OODOutput
from core.loss_rejection_engine.metrics import LREMetricsTracker, LREMetrics

__all__ = [
    "LossRejectionEngine", "LREResult",
    "StructuralFilterLayer", "Layer1Output", "FilterResult",
    "MetaLabeler", "MetaLabelerOutput",
    "OODDetector", "OODOutput",
    "LREMetricsTracker", "LREMetrics",
]
