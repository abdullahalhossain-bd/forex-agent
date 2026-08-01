# analysis/dna_drift.py
# ============================================================
# Market DNA — cluster-distribution drift monitoring via PSI.
#
# Graduated thresholds (from review round 3) instead of a single
# hard cutoff — a single 0.35-only trigger reacts only after a
# regime shift is already large; this gives an early warning band.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)

PSI_BANDS = (
    (0.00, 0.10, "NORMAL"),
    (0.10, 0.25, "MONITOR"),
    (0.25, 0.35, "ALERT"),
    (0.35, float("inf"), "MANDATORY_REFIT"),
)


@dataclass
class DriftResult:
    psi: float
    status: str
    per_cluster_psi: dict


def _bucket_proportions(labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """Proportion of observations in each cluster id, 0..n_clusters-1.
    UNKNOWN (-1) is treated as its own bucket, not dropped — a rising
    UNKNOWN share IS the drift signal in many regime-change scenarios."""
    counts = np.zeros(n_clusters + 1)  # index n_clusters == UNKNOWN bucket
    for lbl in labels:
        idx = n_clusters if lbl == -1 else int(lbl)
        if 0 <= idx <= n_clusters:
            counts[idx] += 1
    total = counts.sum()
    return counts / total if total > 0 else counts


def population_stability_index(
    reference_labels: np.ndarray,
    current_labels: np.ndarray,
    n_clusters: int,
    epsilon: float = 1e-4,
) -> DriftResult:
    """
    PSI = sum( (cur% - ref%) * ln(cur% / ref%) ) over cluster buckets.

    `reference_labels` should come from the model's own TRAINING
    assignments (labels_ at fit time). `current_labels` should be a
    recent live window's approximate_predict() outputs. Both must be
    labeled by the SAME frozen model — comparing labels across two
    different HDBSCAN fits is meaningless (see cluster-instability
    note in market_dna.py).
    """
    ref = _bucket_proportions(reference_labels, n_clusters)
    cur = _bucket_proportions(current_labels, n_clusters)

    ref = np.clip(ref, epsilon, None)
    cur = np.clip(cur, epsilon, None)

    per_bucket = (cur - ref) * np.log(cur / ref)
    psi = float(per_bucket.sum())

    status = "NORMAL"
    for lo, hi, label in PSI_BANDS:
        if lo <= psi < hi:
            status = label
            break

    per_cluster = {
        (i if i < n_clusters else "UNKNOWN"): round(float(per_bucket[i]), 5)
        for i in range(len(per_bucket))
    }

    if status in ("ALERT", "MANDATORY_REFIT"):
        log.warning(f"[dna_drift] PSI={psi:.4f} status={status} — {per_cluster}")
    else:
        log.info(f"[dna_drift] PSI={psi:.4f} status={status}")

    return DriftResult(psi=round(psi, 5), status=status, per_cluster_psi=per_cluster)
