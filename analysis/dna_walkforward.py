# analysis/dna_walkforward.py
# ============================================================
# Market DNA — walk-forward validation harness with a THREE-WAY
# time split, specifically to avoid nested leakage between:
#
#   Fold A (clusterer fit)  — HDBSCAN + scaler + PCA trained here.
#   Fold B (meta/calibration fit) — confidence-blending weights /
#           calibration curve trained here, using Fold A's frozen
#           detector to LABEL Fold B data (no re-fitting).
#   Fold C (evaluation) — final walk-forward metrics computed here,
#           using both Fold A's detector and Fold B's calibration,
#           untouched by either fitting step.
#
# Two model-fit stages sharing one evaluation fold is the exact
# "ensemble weights fit on the same data being evaluated" leak
# flagged in ai-ml-standards.md — this harness makes that
# structurally impossible by construction, not by convention.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from analysis.market_dna import DNAConfig, MarketDNADetector
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class ThreeWaySplit:
    fold_a: pd.DataFrame   # clusterer training
    fold_b: pd.DataFrame   # meta-model / calibration training
    fold_c: pd.DataFrame   # held-out evaluation


def make_three_way_split(
    df: pd.DataFrame,
    *,
    time_col: str = "time",
    fold_a_frac: float = 0.5,
    fold_b_frac: float = 0.25,
) -> ThreeWaySplit:
    """
    Chronological (never shuffled) split. fold_c gets whatever
    remains after fold_a + fold_b — must be > 0.
    """
    if not (0 < fold_a_frac < 1 and 0 < fold_b_frac < 1 and fold_a_frac + fold_b_frac < 1):
        raise ValueError("fold_a_frac + fold_b_frac must be < 1, leaving room for fold_c")

    d = df.sort_values(time_col).reset_index(drop=True)
    n = len(d)
    a_end = int(n * fold_a_frac)
    b_end = a_end + int(n * fold_b_frac)

    split = ThreeWaySplit(
        fold_a=d.iloc[:a_end].reset_index(drop=True),
        fold_b=d.iloc[a_end:b_end].reset_index(drop=True),
        fold_c=d.iloc[b_end:].reset_index(drop=True),
    )
    log.info(
        f"[dna_walkforward] Split n={n} -> "
        f"A(clusterer)={len(split.fold_a)} "
        f"B(meta)={len(split.fold_b)} "
        f"C(eval)={len(split.fold_c)}"
    )
    return split


def fit_frozen_detector(fold_a: pd.DataFrame, config: Optional[DNAConfig] = None) -> MarketDNADetector:
    """Fold A only. Never touches fold_b or fold_c."""
    return MarketDNADetector(config).fit(fold_a)


def label_fold(
    detector: MarketDNADetector,
    fold: pd.DataFrame,
    *,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    progress_every: int = 500,
) -> pd.DataFrame:
    """
    Row-by-row approximate_predict over a fold, using an ALREADY
    FROZEN detector. This is how fold_b and fold_c get cluster
    labels — never by fitting on them.

    progress_callback(done, total), if given, is called every
    `progress_every` rows (and once at completion) so long-running
    labeling can report progress without this module depending on
    any particular UI/printing method.
    """
    out = fold.copy()
    total = len(fold)
    cluster_ids, confidences, states = [], [], []
    for i, (_, row) in enumerate(fold.iterrows(), start=1):
        result = detector.predict_live(pd.DataFrame([row]))
        cluster_ids.append(result.get("cluster_id"))
        confidences.append(result.get("confidence"))
        states.append(result.get("state"))
        if progress_callback is not None and (i % progress_every == 0 or i == total):
            progress_callback(i, total)
    out["dna_cluster_id"] = cluster_ids
    out["dna_confidence"] = confidences
    out["dna_state"] = states
    return out


def fit_meta_calibration(
    fold_b_labeled: pd.DataFrame,
    calibrate_fn: Callable[[pd.DataFrame], object],
) -> object:
    """
    Fold B only (already labeled by the Fold-A-frozen detector).
    `calibrate_fn` is caller-supplied (e.g. Platt scaling / isotonic
    regression / logistic stacking over [rule_conf, ml_conf, rl_conf,
    dna_conf] -> realized outcome) — this harness only enforces WHICH
    data it's allowed to see, not the calibration method itself.
    """
    return calibrate_fn(fold_b_labeled)


def evaluate(
    fold_c_labeled: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], dict],
) -> dict:
    """
    Fold C only — never seen by either the detector fit or the
    calibration fit. This is the only fold whose numbers are allowed
    to be reported as "expected live performance".
    """
    result = metric_fn(fold_c_labeled)
    log.info(f"[dna_walkforward] Fold C evaluation: {result}")
    return result


def run_full_validation(
    df: pd.DataFrame,
    *,
    calibrate_fn: Callable[[pd.DataFrame], object],
    metric_fn: Callable[[pd.DataFrame], dict],
    config: Optional[DNAConfig] = None,
    time_col: str = "time",
) -> dict:
    """Convenience wrapper chaining split -> fit -> label -> calibrate -> evaluate."""
    split = make_three_way_split(df, time_col=time_col)

    detector = fit_frozen_detector(split.fold_a, config)

    fold_b_labeled = label_fold(detector, split.fold_b)
    calibrator = fit_meta_calibration(fold_b_labeled, calibrate_fn)

    fold_c_labeled = label_fold(detector, split.fold_c)
    metrics = evaluate(fold_c_labeled, metric_fn)

    return {
        "detector_metadata": detector.metadata(),
        "fold_b_size": len(fold_b_labeled),
        "fold_c_size": len(fold_c_labeled),
        "fold_c_metrics": metrics,
        "calibrator": calibrator,
        "detector": detector,
    }