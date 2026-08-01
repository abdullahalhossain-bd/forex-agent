"""Layer 3: Out-of-Distribution Detector.
Detects when current market conditions are outside training distribution.""" 
from __future__ import annotations
import logging, os, pickle, json, datetime
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import numpy as np
except ImportError:
    np = None

log = logging.getLogger(__name__)
OOD_DIR = Path(__file__).parent.parent.parent / "memory" / "lre_models"
OOD_DIR.mkdir(parents=True, exist_ok=True)
OOD_REJECT_THRESHOLD = float(os.getenv("LRE_OOD_REJECT", "4.0"))
OOD_WARN_THRESHOLD = float(os.getenv("LRE_OOD_WARN", "2.5"))
OOD_MIN_SAMPLES = int(os.getenv("LRE_OOD_MIN_SAMPLES", "100"))
OOD_FEATURE_WINDOW = int(os.getenv("LRE_OOD_WINDOW", "500"))

@dataclass
class OODOutput:
    distance: float = 0.0
    verdict: str = "PASS"
    pass_through: bool = True
    n_reference_samples: int = 0
    ood_features: Dict[str, float] = field(default_factory=dict)
    reason: str = ""

class OODDetector:
    """OOD detector using Mahalanobis-like distance."""
    OOD_FEATURES = [
        "confidence", "rr", "sl_pips", "tp_pips",
        "atr", "rsi", "regime_confidence", "trend_strength",
        "smc_score", "session_quality", "l1_composite",
        "hour_sin", "hour_cos",
    ]

    def __init__(self):
        self._ref_features: deque = deque(maxlen=OOD_FEATURE_WINDOW)
        self._means = None
        self._stds = None
        self._feature_names: List[str] = []
        self._n_seen: int = 0
        self._path = OOD_DIR / "ood_reference.pkl"
        self._load_reference()

    def _extract_ood_features(self, dec_out, analysis_out, market_out) -> Dict[str, float]:
        features = {}
        ind = market_out.get("ind_ctx", {}) or {}
        features["confidence"] = float(dec_out.get("confidence", 60))
        features["rr"] = float(dec_out.get("rr", 0) or 2.0)
        features["sl_pips"] = float(dec_out.get("sl_pips", 0) or 20.0)
        features["tp_pips"] = float(dec_out.get("tp_pips", 0) or 40.0)
        features["atr"] = 0.0
        if isinstance(ind, dict):
            atr = ind.get("atr")
            if isinstance(atr, dict): atr = atr.get("value")
            if atr: features["atr"] = float(atr)
        features["rsi"] = 50.0
        if isinstance(ind, dict):
            rsi = ind.get("rsi")
            if isinstance(rsi, dict): rsi = rsi.get("value")
            if rsi: features["rsi"] = float(rsi)
        regime = market_out.get("regime") or {}
        if isinstance(regime, dict):
            features["regime_confidence"] = float(regime.get("confidence", 0.5))
            features["trend_strength"] = float(regime.get("trend_strength", 0.5))
        else:
            features["regime_confidence"] = 0.5
            features["trend_strength"] = 0.5
        smc = analysis_out.get("smc") or analysis_out.get("smc_ctx") or {}
        features["smc_score"] = float(smc.get("score", smc.get("total_score", 0)))
        session = analysis_out.get("session") or analysis_out.get("session_ctx") or {}
        sq = str(session.get("quality", session.get("session_quality", "MEDIUM")))
        features["session_quality"] = 1.0 if "HIGH" in sq.upper() else (0.5 if "MEDIUM" in sq.upper() else 0.0)
        l1 = dec_out.get("_lre_l1_score")
        features["l1_composite"] = float(l1) if l1 else 0.0
        hour = datetime.datetime.now().hour
        features["hour_sin"] = np.sin(2 * np.pi * hour / 24) if np else 0.0
        features["hour_cos"] = np.cos(2 * np.pi * hour / 24) if np else 0.0
        return {k: features.get(k, 0.0) for k in self.OOD_FEATURES}

    def _load_reference(self):
        if not self._path.exists(): return
        try:
            with open(self._path, "rb") as f:
                saved = pickle.load(f)
            self._ref_features = deque(saved.get("features", []), maxlen=OOD_FEATURE_WINDOW)
            self._n_seen = saved.get("n_seen", 0)
            self._compute_stats()
            log.info(f"[LRE-L3] Reference loaded: {len(self._ref_features)} samples")
        except Exception as e:
            log.warning(f"[LRE-L3] Load failed: {e}")

    def _save_reference(self):
        try:
            with open(self._path, "wb") as f:
                pickle.dump({"features": list(self._ref_features), "n_seen": self._n_seen}, f)
        except Exception as e:
            log.warning(f"[LRE-L3] Save failed: {e}")

    def _compute_stats(self):
        if not self._ref_features or np is None: return
        try:
            arr = np.array(self._ref_features)
            self._means = np.mean(arr, axis=0)
            self._stds = np.std(arr, axis=0).clip(min=1e-8)
        except: pass

    def record_features(self, features: Dict[str, float]):
        vec = [features.get(k, 0.0) for k in self.OOD_FEATURES]
        self._ref_features.append(vec)
        self._n_seen += 1
        if len(self._ref_features) % 50 == 0:
            self._compute_stats()
            self._save_reference()

    def evaluate(self, dec_out, analysis_out, market_out, **kwargs) -> OODOutput:
        direction = (dec_out.get("decision") or "WAIT").upper()
        if direction not in ("BUY", "SELL"):
            return OODOutput(verdict="PASS", reason="No trade signal")
        features = self._extract_ood_features(dec_out, analysis_out, market_out)
        dec_out["_lre_ood_features"] = features
        n_ref = len(self._ref_features)
        if n_ref < OOD_MIN_SAMPLES:
            return OODOutput(verdict="PASS", n_reference_samples=n_ref,
                           reason=f"Building ref ({n_ref}/{OOD_MIN_SAMPLES})", ood_features=features)
        if self._means is None: self._compute_stats()
        if self._means is None: return OODOutput(verdict="PASS", reason="No stats")
        try:
            vec = np.array([features.get(k, 0.0) for k in self.OOD_FEATURES])
            z_scores = np.abs(vec - self._means) / self._stds
            median_z = np.median(z_scores)
            max_z = np.max(z_scores)
            distance = 0.6 * median_z + 0.4 * max_z
            if distance >= OOD_REJECT_THRESHOLD:
                verdict, pt = "REJECT", False
            elif distance >= OOD_WARN_THRESHOLD:
                verdict, pt = "WARN", True
            else:
                verdict, pt = "PASS", True
            reason = f"OOD dist={distance:.2f}"
            if verdict != "PASS":
                unusual = []
                for i, name in enumerate(self.OOD_FEATURES):
                    if z_scores[i] > 2.5: unusual.append(f"{name}={z_scores[i]:.1f}z")
                reason += " | unusual: " + ", ".join(unusual[:3])
                log.info(f"[LRE-L3] {verdict} {reason}")
            return OODOutput(distance=round(distance, 3), verdict=verdict, pass_through=pt,
                           n_reference_samples=n_ref, ood_features=features, reason=reason)
        except Exception as e:
            log.warning(f"[LRE-L3] Error: {e}")
            return OODOutput(verdict="PASS", reason=f"Error: {e}")
