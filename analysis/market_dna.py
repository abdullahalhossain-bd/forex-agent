# analysis/market_dna.py
# ============================================================
# Market DNA — unsupervised market-regime clustering used as a
# CONTEXT / VALIDATION layer, never as a signal generator.
#
# Design constraints (from architecture review, do not relax
# without re-reading the review notes):
#
#   1. FROZEN MODEL: a fitted detector is used for a fixed window
#      (see DNAConfig.model_freeze_days). It is never refit on data
#      that includes bars used for live prediction during that
#      window. This sidesteps HDBSCAN's cluster-boundary drift
#      problem entirely, rather than trying to "migrate" old
#      cluster IDs into a new model (which is leaky/undefined).
#
#   2. NO NATIVE .predict(): HDBSCAN has no incremental predict.
#      Live assignment MUST use hdbscan.approximate_predict(),
#      which also gives a soft membership score used as
#      cluster-confidence.
#
#   3. UNKNOWN IS NOT REJECTED: label == -1 (noise / never-seen
#      regime) is surfaced as its own state, downstream code
#      decides risk reduction — this module never says BUY/SELL/
#      APPROVE/REJECT.
#
#   4. Feature set is NOT reinvented here — it reuses
#      features.indicators_v5.get_feature_columns(), which is
#      already the vetted, closed-bar-only feature pipeline used
#      elsewhere in this repo. Duplicating feature engineering here
#      would double the leakage-audit surface for no benefit.
# ============================================================

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

try:
    import hdbscan
    _HAS_HDBSCAN = True
except ImportError:
    _HAS_HDBSCAN = False

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import MODEL_DIR
from features.indicators_v5 import get_feature_columns
from utils.logger import get_logger

log = get_logger(__name__)

DNA_MODEL_DIR = MODEL_DIR / "market_dna"
DNA_MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ── Configuration ────────────────────────────────────────────
@dataclass
class DNAConfig:
    # HDBSCAN — default lowered to 5 after testing on EURUSD H1 showed:
    #   min_cluster_size=30 → 0 clusters (100% noise)
    #   min_cluster_size=10 → 0-2 clusters (varies by data)
    #   min_cluster_size=5  → 2-4 clusters (reliable)
    #   min_cluster_size=3  → 3-6 clusters (more granular)
    # Forex data is mostly "normal" bars — only extremes form clusters.
    # Use --min-cluster-size flag in setup_and_train_market_dna.py to override.
    min_cluster_size: int = 5
    min_samples: Optional[int] = 3  # explicit small value (was None = min_cluster_size)

    # PCA — mandatory once feature count is non-trivial (curse of
    # dimensionality flagged in review). retain 95% variance.
    pca_variance: float = 0.95
    pca_min_features_to_trigger: int = 12

    # Live assignment
    unknown_confidence_floor: float = 0.50  # below this -> treat as UNKNOWN
    unknown_position_multiplier: float = 0.25

    # Model lifecycle — see FROZEN MODEL note above.
    model_freeze_days: int = 90

    feature_cols: list = field(default_factory=get_feature_columns)


# ── Detector ─────────────────────────────────────────────────
class MarketDNADetector:
    """
    Fit on a closed historical window, freeze, serve live
    approximate-predict for `DNAConfig.model_freeze_days`, then get
    replaced wholesale by a new fit (see refit_if_stale()).

    A detector instance is IMMUTABLE once fitted — there is no
    partial_fit / online-update path. This is intentional: online
    updates to a density-based clusterer are exactly the kind of
    silent-leak vector flagged in the ai-ml-standards checklist.
    """

    def __init__(self, config: Optional[DNAConfig] = None):
        if not _HAS_HDBSCAN:
            raise ImportError(
                "hdbscan not installed. Add `hdbscan>=0.8.33` to requirements.txt "
                "and `pip install hdbscan`."
            )
        self.cfg = config or DNAConfig()
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        self.clusterer: Optional["hdbscan.HDBSCAN"] = None

        # Provenance / audit trail — every model on disk must be able
        # to answer "exactly what data was I trained on?"
        self.model_id: Optional[str] = None
        self.trained_at: Optional[str] = None
        self.train_window_start: Optional[str] = None
        self.train_window_end: Optional[str] = None
        self.n_train_rows: int = 0
        self.n_clusters_: Optional[int] = None

    # ── Fit (historical, offline only) ──────────────────────
    def fit(self, train_df: pd.DataFrame, *, time_col: str = "time") -> "MarketDNADetector":
        """
        Fit scaler + (optional) PCA + HDBSCAN on a CLOSED historical
        window only.

        CRITICAL: `train_df` must not contain any bar that will later
        be used as a "live" prediction input while this model is
        active — that would be look-ahead leakage into the cluster
        geometry itself, not just into a downstream statistic.
        Callers (see dna_walkforward.py) are responsible for
        enforcing the train/eval time split; this method trusts its
        input.
        """
        missing = [c for c in self.cfg.feature_cols if c not in train_df.columns]
        if missing:
            raise ValueError(f"train_df missing required feature columns: {missing}")

        X_raw = train_df[self.cfg.feature_cols].to_numpy(dtype=float)
        if not np.isfinite(X_raw).all():
            raise ValueError(
                "train_df contains NaN/inf in feature columns — run the "
                "indicators_v5 pipeline (drop_nan=True) before fitting."
            )

        self.scaler = StandardScaler().fit(X_raw)
        X = self.scaler.transform(X_raw)

        if len(self.cfg.feature_cols) >= self.cfg.pca_min_features_to_trigger:
            self.pca = PCA(n_components=self.cfg.pca_variance, random_state=42).fit(X)
            X = self.pca.transform(X)
            log.info(
                f"[MarketDNA] PCA reduced {len(self.cfg.feature_cols)} features "
                f"-> {X.shape[1]} components ({self.cfg.pca_variance*100:.0f}% variance)."
            )
        else:
            self.pca = None

        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.cfg.min_cluster_size,
            min_samples=self.cfg.min_samples,
            prediction_data=True,  # required for approximate_predict later
        )
        self.clusterer.fit(X)

        labels = self.clusterer.labels_
        self.n_clusters_ = int(len(set(labels)) - (1 if -1 in labels else 0))
        self.n_train_rows = len(train_df)
        self.trained_at = datetime.now(timezone.utc).isoformat()

        ts = pd.to_datetime(train_df[time_col]) if time_col in train_df.columns else None
        self.train_window_start = str(ts.min()) if ts is not None else None
        self.train_window_end = str(ts.max()) if ts is not None else None
        self.model_id = f"dna_{self.trained_at.replace(':', '').replace('-', '')[:15]}"

        noise_pct = 100.0 * (labels == -1).sum() / len(labels)
        log.info(
            f"[MarketDNA] Fit complete — model_id={self.model_id} "
            f"clusters={self.n_clusters_} noise={noise_pct:.1f}% "
            f"rows={self.n_train_rows} window=[{self.train_window_start} .. {self.train_window_end}]"
        )
        return self

    # ── Live inference ───────────────────────────────────────
    def predict_live(self, live_row: pd.DataFrame) -> dict:
        """
        Soft-assign ONE closed-bar feature row to a cluster.

        Returns a context dict — never a trade decision. Downstream
        (dna_journal / risk layer) decides what APPROVE/REJECT/
        REDUCE means for this cluster.
        """
        self._require_fitted()
        missing = [c for c in self.cfg.feature_cols if c not in live_row.columns]
        if missing:
            raise ValueError(f"live_row missing required feature columns: {missing}")

        X_raw = live_row[self.cfg.feature_cols].to_numpy(dtype=float)
        if not np.isfinite(X_raw).all():
            log.warning("[MarketDNA] Non-finite features in live row -> UNKNOWN")
            return self._unknown_result(confidence=0.0)

        X = self.scaler.transform(X_raw)
        if self.pca is not None:
            X = self.pca.transform(X)

        labels, strengths = hdbscan.approximate_predict(self.clusterer, X)
        cluster_id = int(labels[0])
        confidence = float(strengths[0])

        if cluster_id == -1 or confidence < self.cfg.unknown_confidence_floor:
            return self._unknown_result(confidence=confidence, raw_label=cluster_id)

        return {
            "state": "KNOWN",
            "cluster_id": cluster_id,
            "confidence": confidence,
            "model_id": self.model_id,
        }

    def _unknown_result(self, *, confidence: float, raw_label: int = -1) -> dict:
        return {
            "state": "UNKNOWN",
            "cluster_id": None,
            "raw_label": raw_label,
            "confidence": confidence,
            "model_id": self.model_id,
            "suggested_position_multiplier": self.cfg.unknown_position_multiplier,
            "reason": (
                "never-seen regime" if raw_label == -1 else
                f"low membership confidence ({confidence:.2f} < {self.cfg.unknown_confidence_floor})"
            ),
        }

    # ── Lifecycle helpers ─────────────────────────────────────
    def is_stale(self, as_of: Optional[datetime] = None) -> bool:
        """True once this frozen model has exceeded its freeze window."""
        self._require_fitted()
        as_of = as_of or datetime.now(timezone.utc)
        trained = datetime.fromisoformat(self.trained_at)
        age_days = (as_of - trained).total_seconds() / 86400.0
        return age_days >= self.cfg.model_freeze_days

    def cluster_signature(self, train_df: pd.DataFrame, cluster_id: int, top_n: int = 5) -> dict:
        """
        Human-readable centroid summary — for DASHBOARD/LOGGING only.
        Not used as a cluster-matching mechanism (that would reintroduce
        the rule-based bucketing this module is meant to avoid).
        """
        self._require_fitted()
        labels = self.clusterer.labels_
        mask = labels == cluster_id
        if mask.sum() == 0:
            return {"cluster_id": cluster_id, "error": "empty cluster in training data"}

        sub = train_df.loc[mask, self.cfg.feature_cols]
        overall = train_df[self.cfg.feature_cols]
        z = (sub.mean() - overall.mean()) / (overall.std() + 1e-10)
        top = z.abs().sort_values(ascending=False).head(top_n)
        return {
            "cluster_id": cluster_id,
            "n_members": int(mask.sum()),
            "distinctive_features": {k: round(float(z[k]), 2) for k in top.index},
        }

    # ── Persistence ────────────────────────────────────────────
    def save(self, path: Optional[Path] = None) -> Path:
        self._require_fitted()
        path = path or (DNA_MODEL_DIR / f"{self.model_id}.joblib")
        joblib.dump(self, path)
        meta_path = path.with_suffix(".json")
        meta_path.write_text(json.dumps(self.metadata(), indent=2))
        log.info(f"[MarketDNA] Saved model {self.model_id} -> {path}")
        return path

    @staticmethod
    def load(path: Path) -> "MarketDNADetector":
        obj = joblib.load(path)
        if not isinstance(obj, MarketDNADetector):
            raise TypeError(f"{path} does not contain a MarketDNADetector")
        return obj

    def metadata(self) -> dict:
        return {
            "model_id": self.model_id,
            "trained_at": self.trained_at,
            "train_window_start": self.train_window_start,
            "train_window_end": self.train_window_end,
            "n_train_rows": self.n_train_rows,
            "n_clusters": self.n_clusters_,
            "min_cluster_size": self.cfg.min_cluster_size,
            "pca_components": None if self.pca is None else int(self.pca.n_components_),
            "feature_cols": self.cfg.feature_cols,
        }

    def _require_fitted(self):
        if self.clusterer is None:
            raise RuntimeError("MarketDNADetector.fit() has not been called yet.")
