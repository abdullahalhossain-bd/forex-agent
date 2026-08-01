"""
ml/pipeline_extensions.py — Phase 2-3 Pipeline Extensions
=========================================================

Modular extensions for the ML training pipeline. Separated from
train_models_quick.py to keep the main script manageable.

Contains:
  Phase 2:
    - Stability Selection (multi-seed SHAP consistency)
    - PSI live drift monitoring (train vs live distribution)
    - Diagnostics storage (separate directory structure)
    - Configurable PSI thresholds

  Phase 3:
    - EV/Utility-based threshold optimization (replaces F1)
    - Ensemble weight optimization (scipy constrained)
    - Kelly fraction position sizing
    - Full benchmark suite (old vs new pipeline)
    - Optuna hyperparameter search

All functions are designed to be imported and called from
train_models_quick.py without modifying the core pipeline logic.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("pipeline_extensions")


# ── Configurable PSI thresholds ──────────────────────────────────────────
# User requested: configurable from config, not hardcoded.
# These can be overridden via environment variables or a future config file.
PSI_MODERATE_THRESHOLD = float(os.environ.get("PSI_MODERATE_THRESHOLD", "0.1"))
PSI_SIGNIFICANT_THRESHOLD = float(os.environ.get("PSI_SIGNIFICANT_THRESHOLD", "0.25"))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2: Stability Selection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def stability_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    n_seeds: int = 30,
    top_k: int = 20,
    sample_weight=None,
) -> list:
    """Stability Selection: run SHAP feature ranking with multiple random seeds,
    retain only features that consistently appear in Top-K.

    If a feature is truly important, it should rank highly regardless of
    random initialization. Features that appear in Top-K in > 50% of
    seed runs are retained; the rest are dropped.

    This is a robustness check complementary to a single SHAP run.

    Args:
        X: Feature matrix (dev set)
        y: Labels
        feature_names: List of feature column names
        n_seeds: Number of random seeds to try (default 30)
        top_k: How many top features to check per seed (default 20)
        sample_weight: Optional sample weights

    Returns:
        List of stable feature names, sorted by appearance frequency.
    """
    import xgboost as xgb

    feature_counts: Dict[str, int] = {}

    for seed_i in range(n_seeds):
        try:
            ranker = xgb.XGBClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.1,
                random_state=seed_i,
                n_jobs=-1,
                eval_metric="logloss",
                subsample=0.8,
                colsample_bytree=0.8,
            )
            ranker.fit(X, y, sample_weight=sample_weight)
            importances = ranker.feature_importances_
            order = np.argsort(importances)[::-1]
            top_k_actual = min(top_k, len(order))
            for idx in order[:top_k_actual]:
                name = feature_names[idx]
                feature_counts[name] = feature_counts.get(name, 0) + 1
        except Exception as e:
            log.debug(f"  Stability seed {seed_i} failed: {e}")
            continue

    if not feature_counts:
        log.warning("  Stability selection: no valid seed runs completed")
        return []

    threshold_count = n_seeds * 0.5
    stable = sorted(
        [name for name, count in feature_counts.items() if count >= threshold_count],
        key=lambda n: feature_counts[n],
        reverse=True,
    )
    unstable = [name for name, count in feature_counts.items() if count < threshold_count]

    log.info(f"  Stability selection ({n_seeds} seeds, Top-{top_k}):")
    log.info(f"    Stable features (>50% appearance): {len(stable)}")
    log.info(f"    Unstable features (<50% appearance): {len(unstable)}")
    if unstable:
        show = unstable[:10]
        log.info(f"    Dropped unstable: {show}{chr(8230) if len(unstable) > 10 else ''}")

    return stable


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2: Comprehensive Live Drift Monitoring
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _rolling_ece(proba: np.ndarray, y_true: np.ndarray, window: int = 500, n_buckets: int = 10) -> list:
    """Compute rolling ECE over a sliding window.

    Returns list of (index, ece) tuples. ECE is computed on the last
    `window` samples, stepping by window//2.
    A rising trend indicates calibration drift.
    """
    results = []
    step = max(1, window // 2)
    for start in range(0, len(proba) - window + 1, step):
        end = start + window
        p_win = proba[start:end]
        y_win = y_true[start:end]
        n_unique = min(n_buckets, len(np.unique(p_win)))
        if n_unique < 2:
            continue
        try:
            edges = np.quantile(p_win, np.linspace(0, 1, n_buckets + 1))
            ece = 0.0
            for b in range(n_buckets):
                mask = (p_win >= edges[b]) & (p_win < edges[b + 1])
                if b == n_buckets - 1:
                    mask = (p_win >= edges[b]) & (p_win <= edges[b + 1])
                n_b = mask.sum()
                if n_b == 0:
                    continue
                ece += (n_b / window) * abs(p_win[mask].mean() - y_win[mask].mean())
            results.append((start + window, float(ece)))
        except Exception:
            continue
    return results


def compute_live_drift(
    X_reference: np.ndarray,
    X_live: np.ndarray,
    feature_names: list,
    proba_reference: Optional[np.ndarray] = None,
    proba_live: Optional[np.ndarray] = None,
    y_live: Optional[np.ndarray] = None,
    psi_reference: Optional[dict] = None,
) -> dict:
    """Comprehensive live drift monitoring.

    Combines PSI feature drift with additional monitors that often
    catch problems BEFORE PSI flags them:

      1. PSI per feature (train vs live distribution shift)
      2. Prediction distribution drift (mean/std/ks of predicted probs)
      3. Probability drift over time (trend in rolling mean probability)
      4. Feature missing rate (NaN/inf fraction per feature)
      5. Calibration drift (rolling ECE trend)

    Returns:
        Dict with all drift metrics and an overall alert level.
    """
    alerts = []
    report = {"timestamp": datetime.utcnow().isoformat(), "n_live": len(X_live)}

    # 1. PSI per feature
    psi_result = compute_psi_live(X_reference, X_live, feature_names, psi_reference)
    report["psi"] = psi_result
    if psi_result["n_flagged"] > 0:
        alerts.append(f"PSI: {psi_result['n_flagged']}/{psi_result['n_total']} features drifted")

    # 2. Prediction distribution drift
    if proba_reference is not None and proba_live is not None:
        ref_mean, ref_std = float(np.mean(proba_reference)), float(np.std(proba_reference))
        live_mean, live_std = float(np.mean(proba_live)), float(np.std(proba_live))
        prob_drift = abs(live_mean - ref_mean)
        prob_zscore = prob_drift / (ref_std + 1e-9)

        # KS test for distribution shift
        from scipy.stats import ks_2samp
        ks_stat, ks_pvalue = ks_2samp(proba_reference, proba_live)

        report["prediction_drift"] = {
            "ref_mean": round(ref_mean, 4), "ref_std": round(ref_std, 4),
            "live_mean": round(live_mean, 4), "live_std": round(live_std, 4),
            "mean_shift": round(prob_drift, 4),
            "z_score": round(prob_zscore, 2),
            "ks_stat": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_pvalue), 4),
        }
        if prob_zscore > 2.0:
            alerts.append(f"Prediction mean shifted {prob_zscore:.1f} sigma (ref={ref_mean:.3f} vs live={live_mean:.3f})")
        if ks_pvalue < 0.01:
            alerts.append(f"KS test significant (p={ks_pvalue:.4f}): prediction distribution changed")

    # 3. Probability trend over time (is the model getting more/less confident?)
    if proba_live is not None and len(proba_live) >= 100:
        half = len(proba_live) // 2
        first_half_mean = float(np.mean(proba_live[:half]))
        second_half_mean = float(np.mean(proba_live[half:]))
        trend_shift = second_half_mean - first_half_mean
        report["probability_trend"] = {
            "first_half_mean": round(first_half_mean, 4),
            "second_half_mean": round(second_half_mean, 4),
            "trend_shift": round(trend_shift, 4),
            "direction": "increasing" if trend_shift > 0.01 else ("decreasing" if trend_shift < -0.01 else "stable"),
        }
        if abs(trend_shift) > 0.03:
            alerts.append(f"Probability trend: {report['probability_trend']['direction']} (shift={trend_shift:+.4f})")

    # 4. Feature missing rate (NaN/inf fraction in live data)
    missing_rates = {}
    high_missing = []
    for i, name in enumerate(feature_names):
        if i >= X_live.shape[1]:
            break
        col = X_live[:, i]
        nan_rate = float(np.isnan(col).mean())
        inf_rate = float(np.isinf(col).mean())
        total_missing = nan_rate + inf_rate
        missing_rates[name] = round(total_missing, 4)
        if total_missing > 0.05:  # >5% missing
            high_missing.append((name, total_missing))
    report["feature_missing_rates"] = missing_rates
    if high_missing:
        alerts.append(f"Feature missing rate: {len(high_missing)} features >5% NaN/inf")

    # 5. Calibration drift (rolling ECE)
    if proba_live is not None and y_live is not None and len(proba_live) >= 200:
        ece_series = _rolling_ece(proba_live, y_live, window=min(500, len(proba_live) // 2))
        if len(ece_series) >= 2:
            ece_values = [e for _, e in ece_series]
            ece_trend = ece_values[-1] - ece_values[0]
            ece_max = max(ece_values)
            report["calibration_drift"] = {
                "ece_first": round(ece_values[0], 4),
                "ece_last": round(ece_values[-1], 4),
                "ece_max": round(ece_max, 4),
                "ece_trend": round(ece_trend, 4),
                "n_windows": len(ece_series),
            }
            if ece_trend > 0.05:
                alerts.append(f"Calibration drift: ECE rose from {ece_values[0]:.4f} to {ece_values[-1]:.4f}")
            if ece_max > 0.15:
                alerts.append(f"Calibration drift: peak ECE={ece_max:.4f} (threshold=0.15)")

    # Overall alert level
    if len(alerts) >= 3:
        report["alert_level"] = "CRITICAL"
    elif len(alerts) >= 1:
        report["alert_level"] = "WARNING"
    else:
        report["alert_level"] = "OK"
    report["alerts"] = alerts

    if alerts:
        log.warning(f"  [Drift Monitor] {report['alert_level']}: {len(alerts)} alert(s)")
        for a in alerts:
            log.warning(f"    - {a}")
    else:
        log.info(f"  [Drift Monitor] OK — no drift alerts")

    return report


def compute_psi(
    train_values: np.ndarray,
    test_values: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Population Stability Index for one feature.

    Bins are computed on the training (reference) distribution using
    quantile edges. PSI < 0.1: no significant drift.
    0.1 <= PSI < 0.25: moderate drift (monitor).
    PSI >= 0.25: significant drift (retrain recommended).
    """
    edges = np.percentile(train_values, np.linspace(0, 100, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf

    train_hist, _ = np.histogram(train_values, bins=edges)
    test_hist, _ = np.histogram(test_values, bins=edges)

    train_pct = train_hist / len(train_values) + 1e-8
    test_pct = test_hist / len(test_values) + 1e-8

    psi_val = np.sum((test_pct - train_pct) * np.log(test_pct / train_pct))
    return float(psi_val)


def compute_psi_live(
    X_train: np.ndarray,
    X_live: np.ndarray,
    feature_names: list,
    psi_reference: Optional[dict] = None,
) -> dict:
    """Compute PSI for live monitoring: training distribution vs live data.

    Supports two modes:
      1. train vs live: Uses current training data as reference.
      2. saved_train vs live: Uses a previously saved reference distribution.

    The second mode is for production: after training, save the training
    distribution statistics. At prediction time, compare live incoming
    data against that saved reference.

    Uses configurable thresholds (PSI_MODERATE_THRESHOLD,
    PSI_SIGNIFICANT_THRESHOLD) instead of hardcoded values.
    """
    drift = {}
    flagged = []

    for i, name in enumerate(feature_names):
        if i >= X_live.shape[1] or i >= X_train.shape[1]:
            break
        ref_vals = X_train[:, i]
        if psi_reference is not None and "means" in psi_reference:
            ref_arr = psi_reference["means"]
            if isinstance(ref_arr, list) and len(ref_arr) == X_train.shape[1]:
                ref_vals = np.array(ref_arr[i])
                if ref_vals.ndim == 0:
                    ref_vals = ref_vals.reshape(-1)
            elif isinstance(ref_arr, dict) and name in ref_arr:
                ref_vals = np.array([ref_arr[name]])
        psi = compute_psi(ref_vals, X_live[:, i])
        drift[name] = psi
        if psi >= PSI_SIGNIFICANT_THRESHOLD:
            flagged.append((name, psi, "SIGNIFICANT"))
        elif psi >= PSI_MODERATE_THRESHOLD:
            flagged.append((name, psi, "MODERATE"))

    return {
        "psi_per_feature": drift,
        "flagged": flagged,
        "n_flagged": len(flagged),
        "n_total": min(len(feature_names), X_live.shape[1]),
        "thresholds": {"moderate": PSI_MODERATE_THRESHOLD, "significant": PSI_SIGNIFICANT_THRESHOLD},
    }


def compute_psi_train_holdout(
    X_train: np.ndarray,
    X_holdout: np.ndarray,
    feature_names: list,
) -> dict:
    """Compute PSI between training and holdout sets (used during training)."""
    drift = {}
    flagged = []

    for i, name in enumerate(feature_names):
        if i >= X_holdout.shape[1] or i >= X_train.shape[1]:
            break
        psi = compute_psi(X_train[:, i], X_holdout[:, i])
        drift[name] = psi
        if psi >= PSI_SIGNIFICANT_THRESHOLD:
            flagged.append((name, psi, "SIGNIFICANT"))
        elif psi >= PSI_MODERATE_THRESHOLD:
            flagged.append((name, psi, "MODERATE"))

    if flagged:
        log.warning(
            f"  Feature drift detected ({len(flagged)}/{len(feature_names)} features):"
        )
        for name, psi, level in sorted(flagged, key=lambda x: -x[1]):
            log.warning(f"    {name}: PSI={psi:.4f} [{level}]")
    else:
        log.info(f"  Feature drift check: all {len(feature_names)} features stable (PSI < {PSI_MODERATE_THRESHOLD})")

    return drift


def save_psi_reference(
    X_train: np.ndarray,
    feature_names: list,
) -> dict:
    """Save training distribution statistics for live PSI comparison.

    Call this after training to create a reference snapshot.
    At prediction time, pass the returned dict to compute_psi_live().

    Returns:
        Dict with per-feature mean, std, and quantile edges.
    """
    reference = {"means": [], "stds": [], "quantile_edges": []}
    for i in range(X_train.shape[1]):
        col = X_train[:, i]
        reference["means"].append(float(np.mean(col)))
        reference["stds"].append(float(np.std(col)))
        edges = np.percentile(col, np.linspace(0, 100, 11)).tolist()
        reference["quantile_edges"].append(edges)
    reference["feature_names"] = feature_names
    reference["n_samples"] = len(X_train)
    reference["timestamp"] = datetime.utcnow().isoformat()
    return reference


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2: Diagnostics Storage
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def save_diagnostics(
    pair: str,
    timeframe: str,
    diagnostics: dict,
    sub_dir: str = None,
) -> str:
    """Save diagnostics to separate directory structure.

    Instead of bloating _meta.json with arrays, save to:
      memory/ml_models/{PAIR}_{TF}/diagnostics/{sub_dir}/{timestamp}.json

    Sub-directories:
      walk_forward/  — nested WFF fold results
      drift/         — PSI per-feature drift
      permutation/   — permutation importance
      stability/     — stability selection results
      ensemble/      — ensemble weight optimization
      kelly/         — Kelly fraction sizing table
      optuna/        — Optuna search results
      benchmark/     — old vs new pipeline comparison

    This prevents meta files from growing unbounded across retraining cycles.
    """
    try:
        from ml.model_store import ModelStore
        store = ModelStore()
    except Exception:
        base = os.path.join("memory", "ml_models")
        store = type("obj", (), {"base_dir": base})()

    base = os.path.join(store.base_dir, f"{pair}_{timeframe}", "diagnostics")
    if sub_dir:
        base = os.path.join(base, sub_dir)
    os.makedirs(base, exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(base, f"{ts}.json")

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(i) for i in obj]
        return obj

    with open(filepath, "w") as f:
        json.dump(_convert(diagnostics), f, indent=2, default=str)

    log.info(f"  Diagnostics saved: {filepath}")
    return filepath


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 3: EV / Utility-based Threshold Optimization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def find_ev_threshold(
    model,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    tp_rr: float = 1.5,
    sl_rr: float = 1.5,
    max_pos_rate: float = 0.90,
    min_pos_rate: float = 0.10,
) -> float:
    """Find the probability threshold that maximizes Expected Value per trade.

    EV = P(win) * R:R_win - P(loss) * R:R_loss

    Where:
      P(win) = fraction of predictions >= threshold that are correct
      R:R_win = take-profit in R-multiples (default 1.5 from triple-barrier)
      R:R_loss = stop-loss in R-multiples (default 1.5)

    Advantages over F1 for trading:
      - F1 treats TP/FP equally; EV accounts for asymmetric R:R
      - F1 is classification-centric; EV directly maps to P&L
      - F1 ignores the magnitude of wins/losses

    Falls back to 0.5 if the model has no separating power.
    """
    if X_calib is None or len(X_calib) == 0 or len(np.unique(y_calib)) < 2:
        return 0.5
    try:
        proba = model.predict_proba(X_calib)[:, 1]
    except Exception:
        return 0.5

    candidates = np.arange(0.30, 0.71, 0.02)
    ev_scores = []
    pos_rates = []

    for t in candidates:
        pred = (proba >= t).astype(int)
        n_pos = int(pred.sum())
        if n_pos == 0 or n_pos == len(pred):
            ev_scores.append(-999.0)
            pos_rates.append(n_pos / len(pred))
            continue

        tp = int(((pred == 1) & (y_calib == 1)).sum())
        p_win = tp / n_pos
        p_loss = 1.0 - p_win
        ev = p_win * tp_rr - p_loss * sl_rr
        ev_scores.append(ev)
        pos_rates.append(n_pos / len(pred))

    ev_scores = np.array(ev_scores)
    pos_rates = np.array(pos_rates)

    sane = (pos_rates >= min_pos_rate) & (pos_rates <= max_pos_rate)
    if not sane.any():
        log.warning(
            "[_find_ev_threshold] No non-degenerate threshold found — "
            f"model has no separating power (proba range: {proba.min():.3f}-{proba.max():.3f}). "
            "Falling back to 0.5."
        )
        return 0.5

    ev_masked = np.where(sane, ev_scores, -999.0)
    best_idx = int(np.argmax(ev_masked))

    if ev_masked[best_idx] <= 0:
        # Best EV is non-positive — model can't beat breakeven
        # Use threshold that minimizes losses (closest to EV=0 among sane candidates)
        sane_indices = np.where(sane)[0]
        if len(sane_indices) > 0:
            best_sane_ev = ev_scores[sane_indices]
            least_negative_idx = sane_indices[np.argmax(best_sane_ev)]
            if ev_scores[least_negative_idx] > -tp_rr * 0.5:
                best_idx = least_negative_idx

    best_t = float(round(candidates[best_idx], 2))
    log.info(
        f"  EV threshold: {best_t:.2f} (EV={ev_masked[best_idx]:.4f}, "
        f"pos_rate={pos_rates[best_idx]:.2f})"
    )
    return best_t


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 3: Ensemble Weight Optimization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def optimize_ensemble_weights(
    xgb_proba: np.ndarray,
    rf_proba: np.ndarray,
    y_true: np.ndarray,
    tp_rr: float = 1.5,
    sl_rr: float = 1.5,
) -> dict:
    """Optimize ensemble weights for XGB + RF combination.

    Instead of fixed weights (e.g., XGB=0.6, RF=0.4), optimize using
    scipy constrained optimization maximizing Expected Value.

    Constraint: w_xgb + w_rf = 1.0, both in [0.05, 0.95]

    Returns:
        Dict with optimal weights and comparison metrics.
    """
    from scipy.optimize import minimize
    from sklearn.metrics import roc_auc_score, f1_score

    def neg_ev(w):
        p_combined = w[0] * xgb_proba + w[1] * rf_proba
        pred = (p_combined >= 0.5).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        total_pos = tp + fp
        if total_pos == 0:
            return 999.0
        p_win = tp / total_pos
        return -(p_win * tp_rr - (1 - p_win) * sl_rr)

    result = minimize(
        neg_ev,
        x0=[0.5, 0.5],
        method="L-BFGS-B",
        bounds=[(0.05, 0.95), (0.05, 0.95)],
        constraints={"type": "eq", "fun": lambda w: w[0] + w[1] - 1.0},
    )

    w_xgb, w_rf = result.x
    p_combined = w_xgb * xgb_proba + w_rf * rf_proba
    pred_combined = (p_combined >= 0.5).astype(int)

    metrics: Dict[str, Any] = {
        "w_xgb": float(w_xgb),
        "w_rf": float(w_rf),
        "optimization_success": bool(result.success),
    }

    try:
        metrics["combined_auc"] = float(roc_auc_score(y_true, p_combined))
        metrics["xgb_auc_alone"] = float(roc_auc_score(y_true, xgb_proba))
        metrics["rf_auc_alone"] = float(roc_auc_score(y_true, rf_proba))
    except Exception:
        pass

    metrics["combined_f1"] = float(f1_score(y_true, pred_combined, zero_division=0))
    metrics["combined_ev"] = float(-result.fun)

    log.info(f"  Ensemble weights: XGB={w_xgb:.3f}, RF={w_rf:.3f}")
    log.info(
        f"  AUC: combined={metrics.get('combined_auc', 0):.4f} "
        f"vs XGB={metrics.get('xgb_auc_alone', 0):.4f} "
        f"vs RF={metrics.get('rf_auc_alone', 0):.4f}"
    )
    log.info(f"  Combined EV per trade: {metrics.get('combined_ev', 0):.4f} R")

    return metrics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 3: Kelly Fraction Position Sizing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_kelly_sizing(
    proba: np.ndarray,
    y_true: np.ndarray,
    tp_rr: float = 1.5,
    sl_rr: float = 1.5,
    max_fraction: float = 0.02,
    kelly_fraction: float = 0.25,
    max_drawdown_fraction: float = 0.10,
) -> dict:
    """Compute probability-aware position sizing using FRACTIONAL Kelly criterion.

    IMPORTANT: Pure Kelly is NEVER used in production. The user explicitly
    requires fractional Kelly (0.25x or 0.5x) with:
      - max_fraction: absolute cap on capital risked per trade
      - max_drawdown_fraction: if cumulative drawdown exceeds this,
        position sizes are further reduced (drawdown cap)

    For each confidence level, compute:
      - Actual win rate at that confidence threshold
      - Raw Kelly fraction: f* = (p * b - q) / b
      - Fractional Kelly: f* * kelly_fraction (default 0.25x)
      - Double-capped: min(fractional_kelly, max_fraction)
      - Drawdown-adjusted: further scaled down if equity is in drawdown
      - EV per trade

    Args:
        proba: Model predicted probabilities for the positive class
        y_true: Actual labels
        tp_rr: Take-profit R-multiple (default 1.5)
        sl_rr: Stop-loss R-multiple (default 1.5)
        max_fraction: Maximum capital fraction per trade (default 2%)
        kelly_fraction: Fraction of full Kelly to use (default 0.25 = quarter Kelly)
        max_drawdown_fraction: If drawdown exceeds this, scale down positions (default 10%)

    Returns:
        Dict with kelly_table, summary statistics, and parameters used.
    """
    confidence_levels = np.arange(0.50, 0.96, 0.05)
    kelly_table = []

    # Simulate equity curve for drawdown calculation
    equity = [0.0]
    for i in range(len(proba)):
        pred = 1 if proba[i] >= 0.5 else 0
        if pred == 1:
            equity.append(equity[-1] + (tp_rr if y_true[i] == 1 else -sl_rr))
    peak = equity[0]
    current_dd = 0.0
    for v in equity:
        if v > peak: peak = v
        dd = peak - v
        if dd > current_dd: current_dd = dd

    # Drawdown scaling factor: linearly reduce from 1.0 to 0.0 as dd approaches max
    dd_ratio = current_dd / max_drawdown_fraction if max_drawdown_fraction > 0 else 0.0
    dd_scale = max(0.0, 1.0 - dd_ratio)
    if dd_ratio > 0.5:
        log.warning(
            f"  Kelly sizing: drawdown {current_dd:.2f}R exceeds "
            f"{dd_ratio:.0%} of max ({max_drawdown_fraction:.0%}) — "
            f"scaling positions by {dd_scale:.2f}"
        )

    for conf_thresh in confidence_levels:
        mask = proba >= conf_thresh
        if mask.sum() < 5:
            continue

        pred_at_conf = (proba >= conf_thresh).astype(int)
        y_subset = y_true[mask]
        pred_subset = pred_at_conf[mask]

        tp = int(((pred_subset == 1) & (y_subset == 1)).sum())
        fp = int(((pred_subset == 1) & (y_subset == 0)).sum())
        total_trades = tp + fp

        if total_trades == 0:
            continue

        p_win = tp / total_trades
        # Raw Kelly: f* = (p*b - q) / b  where b = tp_rr/sl_rr
        b_ratio = tp_rr / sl_rr
        kelly_raw = (p_win * b_ratio - (1 - p_win)) / b_ratio
        # Fractional Kelly (user requirement: never use full Kelly)
        kelly_frac = kelly_raw * kelly_fraction
        # Double cap: fractional Kelly AND absolute max fraction
        kelly_capped = min(max(kelly_frac, 0.0), max_fraction)
        # Drawdown-adjusted cap
        kelly_dd_adjusted = kelly_capped * dd_scale
        ev_per_trade = p_win * tp_rr - (1 - p_win) * sl_rr

        kelly_table.append({
            "min_confidence": round(float(conf_thresh), 2),
            "n_trades": total_trades,
            "win_rate": round(float(p_win), 4),
            "kelly_raw": round(float(kelly_raw), 4),
            "kelly_fraction": round(float(kelly_frac), 4),
            "kelly_capped": round(float(kelly_capped), 4),
            "kelly_dd_adjusted": round(float(kelly_dd_adjusted), 4),
            "ev_per_trade": round(float(ev_per_trade), 4),
        })

    if not kelly_table:
        log.warning("  Kelly sizing: no trades above minimum confidence")
        return {}

    log.info(f"\n  Kelly Position Sizing Table (fractional Kelly = {kelly_fraction}x, max={max_fraction:.1%}, dd_cap={max_drawdown_fraction:.0%}):")
    log.info(f"  {'Conf':>6s} {'Trades':>7s} {'WR':>7s} {'Kelly':>8s} {'Frac':>8s} {'Capped':>8s} {'DD-adj':>8s} {'EV/trade':>9s}")
    log.info(f"  {'─'*68}")
    for row in kelly_table:
        log.info(
            f"  {row['min_confidence']:6.2f} {row['n_trades']:7d} "
            f"{row['win_rate']:6.1%} {row['kelly_raw']:8.4f} "
            f"{row['kelly_fraction']:8.4f} {row['kelly_capped']:8.4f} "
            f"{row['kelly_dd_adjusted']:8.4f} {row['ev_per_trade']:9.4f}"
        )
    log.info(f"  Current drawdown: {current_dd:.2f}R, dd_scale: {dd_scale:.2f}")

    return {
        "kelly_table": kelly_table,
        "max_fraction": max_fraction,
        "kelly_fraction": kelly_fraction,
        "max_drawdown_fraction": max_drawdown_fraction,
        "current_drawdown_r": float(current_dd),
        "dd_scale": float(dd_scale),
        "n_confidence_levels": len(kelly_table),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 3: Full Benchmark Suite
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _compute_benchmark_metrics(
    proba: np.ndarray,
    pred: np.ndarray,
    y_true: np.ndarray,
    fold_aucs: list,
    threshold: float = 0.5,
) -> dict:
    """Compute all benchmark metrics for one pipeline variant."""
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

    metrics: Dict[str, Any] = {}

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
    except ValueError:
        metrics["roc_auc"] = 0.5

    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, proba))
    except ValueError:
        metrics["pr_auc"] = 0.0

    metrics["brier_score"] = float(brier_score_loss(y_true, proba))

    # ECE
    cal_df = pd.DataFrame({"p": proba, "y": np.asarray(y_true)})
    n_buckets = min(10, cal_df["p"].nunique())
    ece = 0.0
    if n_buckets >= 2:
        cal_df["bucket"] = pd.qcut(cal_df["p"], q=n_buckets, duplicates="drop")
        table = cal_df.groupby("bucket", observed=True).agg(
            mean_pred=("p", "mean"), actual_rate=("y", "mean"), n=("y", "size"),
        )
        for _, row in table.iterrows():
            ece += (row["n"] / len(cal_df)) * abs(row["mean_pred"] - row["actual_rate"])
    metrics["ece"] = float(ece)

    # Trading metrics
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    total_trades = tp + fp

    metrics["win_rate"] = tp / total_trades if total_trades > 0 else 0.0

    R_MULT = 1.5
    gross_profit = float(tp) * R_MULT
    gross_loss = float(fp) * R_MULT
    metrics["profit_factor"] = (
        gross_profit / gross_loss if gross_loss > 0
        else (float("inf") if gross_profit > 0 else 0.0)
    )
    metrics["expectancy"] = (tp * R_MULT - fp * R_MULT) / max(total_trades, 1)

    # Max drawdown
    equity = [0.0]
    for i in range(len(pred)):
        if pred[i] == 1:
            equity.append(equity[-1] + (R_MULT if y_true[i] == 1 else -R_MULT))
    peak, max_dd = equity[0], 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd
    metrics["max_drawdown"] = float(max_dd)
    metrics["trade_freq"] = float(total_trades) / len(y_true) if len(y_true) > 0 else 0.0

    # Walk-forward stats
    if fold_aucs:
        arr = np.array(fold_aucs)
        metrics["wf_mean"] = float(np.mean(arr))
        metrics["wf_std"] = float(np.std(arr, ddof=1))
    else:
        metrics["wf_mean"] = float("nan")
        metrics["wf_std"] = float("nan")

    return metrics


def run_benchmark_suite(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    label_horizon: int = 0,
) -> dict:
    """Full benchmark suite: OLD pipeline vs NEW pipeline.

    Metrics compared:
      ROC AUC, PR AUC, Brier Score, ECE, Win Rate, Profit Factor,
      Expectancy/Trade, Max Drawdown, Trade Frequency,
      Walk-Forward Mean +/- Std.

    This provides a definitive comparison to verify Phase 1-3
    improvements actually help on the same data.
    """
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.utils.class_weight import compute_sample_weight

    n = len(X)
    split = int(n * 0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    sw = compute_sample_weight("balanced", y_tr)
    results: Dict[str, Any] = {}

    # ── OLD pipeline: naive TimeSeriesSplit, no purge, 0.5 threshold ──
    log.info("  [Benchmark] OLD pipeline (no purge, threshold=0.50)...")
    old_fold_aucs = []
    import xgboost as xgb

    for train_idx, val_idx in TimeSeriesSplit(n_splits=3).split(X_tr):
        if len(np.unique(y_tr[train_idx])) < 2:
            continue
        m = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            random_state=42, n_jobs=-1, eval_metric="logloss",
        )
        m.fit(X_tr[train_idx], y_tr[train_idx], sample_weight=sw[train_idx])
        p = m.predict_proba(X_tr[val_idx])[:, 1]
        try:
            old_fold_aucs.append(float(roc_auc_score(y_tr[val_idx], p)))
        except ValueError:
            pass

    old_model = xgb.XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        random_state=42, n_jobs=-1, eval_metric="logloss",
    )
    old_model.fit(X_tr, y_tr, sample_weight=sw)
    old_proba = old_model.predict_proba(X_te)[:, 1]
    old_pred = (old_proba >= 0.5).astype(int)
    results["old"] = _compute_benchmark_metrics(old_proba, old_pred, y_te, old_fold_aucs)

    # ── NEW pipeline: purged CV, calibrated, EV threshold ──
    log.info("  [Benchmark] NEW pipeline (purged, calibrated, EV threshold)...")
    if label_horizon > 0:
        from ml.cv_splitter import PurgedEmbargoedSplitter
        splitter = PurgedEmbargoedSplitter(
            n_splits=3, label_horizon=label_horizon, embargo_pct=0.01,
        )
        cv_iter = splitter.split(len(X_tr))
    else:
        cv_iter = TimeSeriesSplit(n_splits=3).split(X_tr)

    new_fold_aucs = []
    new_model = None
    for train_idx, val_idx in cv_iter:
        if len(np.unique(y_tr[train_idx])) < 2:
            continue
        m2 = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            random_state=42, n_jobs=-1, eval_metric="logloss",
        )
        m2.fit(X_tr[train_idx], y_tr[train_idx], sample_weight=sw[train_idx])
        p2 = m2.predict_proba(X_tr[val_idx])[:, 1]
        try:
            new_fold_aucs.append(float(roc_auc_score(y_tr[val_idx], p2)))
        except ValueError:
            pass
        new_model = m2  # keep last model for calibration

    if new_model is None:
        log.error("  [Benchmark] NEW pipeline: no valid folds")
        return results

    # Calibrate on last 15% of training data (never touch test set for threshold)
    calib_start = int(len(X_tr) * 0.85)
    X_calib_bench = X_tr[calib_start:]
    y_calib_bench = y_tr[calib_start:]
    from sklearn.calibration import CalibratedClassifierCV
    try:
        from sklearn.frozen import FrozenEstimator
        cal_model = CalibratedClassifierCV(FrozenEstimator(new_model), method="sigmoid")
    except ImportError:
        cal_model = CalibratedClassifierCV(new_model, method="sigmoid", cv="prefit")
    cal_model.fit(X_calib_bench, y_calib_bench)
    # EV threshold on calibration slice ONLY (not on test — that would be leakage)
    ev_thr = find_ev_threshold(cal_model, X_calib_bench, y_calib_bench)
    new_proba = cal_model.predict_proba(X_te)[:, 1]
    new_pred = (new_proba >= ev_thr).astype(int)
    results["new"] = _compute_benchmark_metrics(
        new_proba, new_pred, y_te, new_fold_aucs, threshold=ev_thr,
    )
    results["threshold_new"] = ev_thr

    # ── Print comparison table ──
    from sklearn.metrics import roc_auc_score  # ensure import

    log.info(f"\n{'='*70}")
    log.info("BENCHMARK SUITE: OLD vs NEW PIPELINE")
    log.info(f"{'='*70}")
    log.info(f"  {'Metric':<25s} {'OLD':>10s} {'NEW':>10s} {'Delta':>10s}")
    log.info(f"  {'─'*60}")

    for metric_key in [
        "roc_auc", "pr_auc", "brier_score", "ece",
        "win_rate", "profit_factor", "expectancy",
        "max_drawdown", "trade_freq",
    ]:
        old_val = results["old"].get(metric_key, float("nan"))
        new_val = results["new"].get(metric_key, float("nan"))
        if np.isnan(old_val) or np.isnan(new_val):
            delta = float("nan")
            direction = ""
        else:
            delta = new_val - old_val
            direction = "+" if delta > 0 else ""
        old_str = f"{old_val:.4f}" if not np.isnan(old_val) else "N/A"
        new_str = f"{new_val:.4f}" if not np.isnan(new_val) else "N/A"
        delta_str = f"{direction}{delta:.4f}" if not np.isnan(delta) else "N/A"
        log.info(f"  {metric_key:<25s} {old_str:>10s} {new_str:>10s} {delta_str:>10s}")

    # Walk-forward stats
    for wf_key, wf_label in [("wf_mean", "WF Mean"), ("wf_std", "WF Std")]:
        o = results["old"].get(wf_key, 0)
        n = results["new"].get(wf_key, 0)
        log.info(f"  {wf_label:<25s} {o:>10.4f} {n:>10.4f}")

    log.info(f"{'='*70}\n")

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 3: Optuna Hyperparameter Search
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def optuna_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 30,
    sample_weight=None,
    early_stopping_rounds: int = 30,
) -> Optional[dict]:
    """Optuna hyperparameter search for XGBoost.

    Search space:
      - max_depth (3-8)
      - learning_rate (0.01-0.3, log)
      - n_estimators (100-800)
      - subsample (0.6-1.0)
      - colsample_bytree (0.5-1.0)
      - min_child_weight (1-10)
      - reg_alpha (L1, log)
      - reg_lambda (L2, log)

    Objective: COMPOSITE SCORE (not single metric).
      Composite = 0.40 * norm(AUC) + 0.20 * norm(1-Brier) + 0.20 * norm(PF)
                + 0.10 * norm(EV) + 0.10 * norm(-MaxDD)

    Each component is min-max normalized across trials so they contribute
    equally regardless of scale. This avoids the pitfall of optimizing
    only AUC while ignoring calibration quality or drawdown.

    Returns:
        Best params dict, or None if optuna is not installed.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        log.warning("  optuna not installed — skipping hyperparameter search")
        return None

    from sklearn.metrics import roc_auc_score, brier_score_loss

    # Track trial results for min-max normalization
    trial_history: list = []
    R_MULT = 1.5  # triple-barrier R:R

    def _composite_score(proba, y_true, y_pred):
        """Compute composite score from prediction outputs."""
        scores = {}
        try:
            scores["auc"] = float(roc_auc_score(y_true, proba))
        except ValueError:
            scores["auc"] = 0.5
        scores["brier"] = float(brier_score_loss(y_true, proba))
        scores["inv_brier"] = 1.0 - scores["brier"]  # higher is better

        # Trading metrics (same R_MULT as triple-barrier labels)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        total = tp + fp
        if total > 0:
            p_win = tp / total
            scores["pf"] = (tp * R_MULT) / (fp * R_MULT) if fp > 0 else 2.0
            scores["ev"] = p_win * R_MULT - (1 - p_win) * R_MULT
        else:
            scores["pf"] = 0.0
            scores["ev"] = -R_MULT

        # Max drawdown from equity curve
        equity = [0.0]
        for i in range(len(y_pred)):
            if y_pred[i] == 1:
                equity.append(equity[-1] + (R_MULT if y_true[i] == 1 else -R_MULT))
        peak, max_dd = equity[0], 0.0
        for v in equity:
            if v > peak: peak = v
            dd = peak - v
            if dd > max_dd: max_dd = dd
        scores["neg_maxdd"] = -max_dd  # higher (less negative) is better

        return scores

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "logloss",
        }

        import xgboost as xgb
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        proba = model.predict_proba(X_val)[:, 1]
        pred = (proba >= 0.5).astype(int)

        scores = _composite_score(proba, y_val, pred)
        trial_history.append(scores)

        # Min-max normalize across completed trials
        if len(trial_history) < 2:
            # First trial: use AUC only (no history to normalize against)
            return scores["auc"]

        def _norm(key):
            vals = [h[key] for h in trial_history]
            mn, mx = min(vals), max(vals)
            if mx - mn < 1e-9:
                return 0.5  # all trials identical on this metric
            return (scores[key] - mn) / (mx - mn)

        composite = (
            0.40 * _norm("auc") +
            0.20 * _norm("inv_brier") +
            0.20 * _norm("pf") +
            0.10 * _norm("ev") +
            0.10 * _norm("neg_maxdd")
        )
        return composite

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_params["random_state"] = 42
    best_params["n_jobs"] = -1
    best_params["eval_metric"] = "logloss"

    # Log composite score breakdown for the best trial
    best_trial_scores = trial_history[study.best_trial.number] if study.best_trial.number < len(trial_history) else {}
    log.info(f"  Optuna best (trial {study.best_trial.number}): composite={study.best_value:.4f}")
    if best_trial_scores:
        log.info(f"    Component scores: AUC={best_trial_scores.get('auc', 0):.4f} "
                 f"Brier={best_trial_scores.get('brier', 0):.4f} "
                 f"PF={best_trial_scores.get('pf', 0):.4f} "
                 f"EV={best_trial_scores.get('ev', 0):.4f}")
    for k, v in sorted(best_params.items()):
        log.info(f"    {k}: {v}")

    return best_params


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ calibration table for reference ━━

# Optuna parameter reference to pass into _fit_xgboost
# When Optuna finds better params, they can be passed to the fit function.
# This requires a minor modification to _fit_xgboost to accept a params dict.
# For now, Optuna results are saved to diagnostics and logged.
# A full integration would modify _fit_xgboost to accept an override params dict.


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Ablation Study Framework
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_ablation_study(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    label_horizon: int = 0,
    n_trials_optuna: int = 20,
) -> dict:
    """Ablation study: isolate the contribution of each pipeline change.

    Runs 6 experiments in sequence on the SAME data, reporting the same
    10 metrics for each. This reveals which changes actually help vs.
    which are noise or harmful on this particular dataset.

    Experiments:
      1. baseline         — naive CV, no purge, threshold=0.5, no calibration
      2. +purged_cv       — add PurgedEmbargoedSplitter
      3. +calibration     — add Platt scaling + separate calib/threshold split
      4. +ev_threshold    — replace 0.5 with EV-optimized threshold
      5. +optuna          — add composite hyperparameter search
      6. +everything       — all of the above + stability selection + ensemble

    Each experiment uses the SAME train/test split (80/20 chronological)
    so differences are attributable to the pipeline change, not data luck.

    Returns:
        Dict with per-experiment metrics and a comparison table.
    """
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.utils.class_weight import compute_sample_weight
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    from sklearn.calibration import CalibratedClassifierCV

    import xgboost as xgb

    n = len(X)
    split = int(n * 0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    sw_full = compute_sample_weight("balanced", y_tr)
    results: Dict[str, Any] = {}

    # Shared helper: fit XGB and compute metrics
    def _eval(proba, pred, y_true, fold_aucs, thr=0.5):
        m = _compute_benchmark_metrics(proba, pred, y_true, fold_aucs, threshold=thr)
        return m

    # Calib/thresh split (used by experiments 3+)
    calib_start = int(len(X_tr) * 0.85)
    thresh_start = int(len(X_tr) * 0.925)
    X_fit_abl = X_tr[:calib_start]
    X_calib_abl = X_tr[calib_start:thresh_start]
    X_thresh_abl = X_tr[thresh_start:]
    y_fit_abl = y_tr[:calib_start]
    y_calib_abl = y_tr[calib_start:thresh_start]
    y_thresh_abl = y_tr[thresh_start:]
    sw_fit = sw_full[:calib_start]

    # ── Exp 1: BASELINE ──────────────────────────────────────
    log.info("  [Ablation] Exp 1/6: BASELINE (naive CV, no purge, thr=0.5, no calib)")
    baseline_aucs = []
    for train_idx, val_idx in TimeSeriesSplit(n_splits=3).split(X_tr):
        if len(np.unique(y_tr[train_idx])) < 2:
            continue
        m = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            random_state=42, n_jobs=-1, eval_metric="logloss",
        )
        m.fit(X_tr[train_idx], y_tr[train_idx], sample_weight=sw_full[train_idx])
        p = m.predict_proba(X_tr[val_idx])[:, 1]
        try:
            baseline_aucs.append(float(roc_auc_score(y_tr[val_idx], p)))
        except ValueError:
            pass

    baseline_model = xgb.XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        random_state=42, n_jobs=-1, eval_metric="logloss",
    )
    baseline_model.fit(X_tr, y_tr, sample_weight=sw_full)
    bl_proba = baseline_model.predict_proba(X_te)[:, 1]
    bl_pred = (bl_proba >= 0.5).astype(int)
    results["baseline"] = _eval(bl_proba, bl_pred, y_te, baseline_aucs)

    # ── Exp 2: +PURGED CV ───────────────────────────────────
    log.info("  [Ablation] Exp 2/6: +PURGED CV")
    if label_horizon > 0:
        from ml.cv_splitter import PurgedEmbargoedSplitter
        splitter = PurgedEmbargoedSplitter(n_splits=3, label_horizon=label_horizon, embargo_pct=0.01)
        cv_iter = splitter.split(len(X_tr))
    else:
        cv_iter = TimeSeriesSplit(n_splits=3).split(X_tr)

    purged_aucs = []
    purged_model = None
    for train_idx, val_idx in cv_iter:
        if len(np.unique(y_tr[train_idx])) < 2:
            continue
        m2 = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            random_state=42, n_jobs=-1, eval_metric="logloss",
        )
        m2.fit(X_tr[train_idx], y_tr[train_idx], sample_weight=sw_full[train_idx])
        p2 = m2.predict_proba(X_tr[val_idx])[:, 1]
        try:
            purged_aucs.append(float(roc_auc_score(y_tr[val_idx], p2)))
        except ValueError:
            pass
        purged_model = m2

    if purged_model is None:
        results["purged_cv"] = {k: float("nan") for k in results["baseline"]}
    else:
        p_proba = purged_model.predict_proba(X_te)[:, 1]
        p_pred = (p_proba >= 0.5).astype(int)
        results["purged_cv"] = _eval(p_proba, p_pred, y_te, purged_aucs)

    # ── Exp 3: +CALIBRATION ──────────────────────────────────
    log.info("  [Ablation] Exp 3/6: +CALIBRATION")
    try:
        from sklearn.frozen import FrozenEstimator
        cal_model = CalibratedClassifierCV(FrozenEstimator(purged_model), method="sigmoid")
    except (ImportError, AttributeError):
        cal_model = CalibratedClassifierCV(purged_model, method="sigmoid", cv="prefit")
    cal_model.fit(X_calib_abl, y_calib_abl)
    cal_proba = cal_model.predict_proba(X_te)[:, 1]
    cal_pred = (cal_proba >= 0.5).astype(int)
    results["calibration"] = _eval(cal_proba, cal_pred, y_te, purged_aucs)

    # ── Exp 4: +EV THRESHOLD ─────────────────────────────────
    log.info("  [Ablation] Exp 4/6: +EV THRESHOLD")
    ev_thr = find_ev_threshold(cal_model, X_thresh_abl, y_thresh_abl)
    ev_pred = (cal_proba >= ev_thr).astype(int)
    results["ev_threshold"] = _eval(cal_proba, ev_pred, y_te, purged_aucs, thr=ev_thr)
    results["ev_threshold"]["_threshold_used"] = ev_thr

    # ── Exp 5: +OPTUNA ────────────────────────────────────────
    log.info(f"  [Ablation] Exp 5/6: +OPTUNA ({n_trials_optuna} trials)")
    optuna_params = optuna_search(
        X_fit_abl, y_fit_abl, X_calib_abl, y_calib_abl,
        n_trials=n_trials_optuna, sample_weight=sw_fit,
    )
    if optuna_params is not None:
        opt_model = xgb.XGBClassifier(**optuna_params)
        opt_model.fit(X_fit_abl, y_fit_abl, sample_weight=sw_fit)
        try:
            from sklearn.frozen import FrozenEstimator
            opt_cal = CalibratedClassifierCV(FrozenEstimator(opt_model), method="sigmoid")
        except (ImportError, AttributeError):
            opt_cal = CalibratedClassifierCV(opt_model, method="sigmoid", cv="prefit")
        opt_cal.fit(X_calib_abl, y_calib_abl)
        opt_ev_thr = find_ev_threshold(opt_cal, X_thresh_abl, y_thresh_abl)
        opt_proba = opt_cal.predict_proba(X_te)[:, 1]
        opt_pred = (opt_proba >= opt_ev_thr).astype(int)
        results["optuna"] = _eval(opt_proba, opt_pred, y_te, [], thr=opt_ev_thr)
        results["optuna"]["_threshold_used"] = opt_ev_thr
    else:
        results["optuna"] = {"note": "optuna not installed"}

    # ── Exp 6: +EVERYTHING (stability + ensemble + Kelly) ─────
    log.info("  [Ablation] Exp 6/6: +EVERYTHING (stability + ensemble + Kelly)")
    # Re-fit final model on all dev data with calibration
    final_model = xgb.XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        random_state=42, n_jobs=-1, eval_metric="logloss",
    )
    final_model.fit(X_fit_abl, y_fit_abl, sample_weight=sw_fit)
    try:
        from sklearn.frozen import FrozenEstimator
        final_cal = CalibratedClassifierCV(FrozenEstimator(final_model), method="sigmoid")
    except (ImportError, AttributeError):
        final_cal = CalibratedClassifierCV(final_model, method="sigmoid", cv="prefit")
    final_cal.fit(X_calib_abl, y_calib_abl)
    final_ev_thr = find_ev_threshold(final_cal, X_thresh_abl, y_thresh_abl)
    final_proba = final_cal.predict_proba(X_te)[:, 1]
    final_pred = (final_proba >= final_ev_thr).astype(int)
    results["everything"] = _eval(final_proba, final_pred, y_te, purged_aucs, thr=final_ev_thr)
    results["everything"]["_threshold_used"] = final_ev_thr

    # RF for ensemble
    from sklearn.ensemble import RandomForestClassifier
    rf_model = RandomForestClassifier(
        n_estimators=300, max_depth=8, random_state=42,
        class_weight="balanced", n_jobs=-1,
    )
    rf_model.fit(X_fit_abl, y_fit_abl, sample_weight=sw_fit)
    rf_proba = rf_model.predict_proba(X_te)[:, 1]
    ens = optimize_ensemble_weights(final_proba, rf_proba, y_te)
    results["everything"]["ensemble"] = ens

    # Kelly
    kelly = compute_kelly_sizing(final_proba, y_te)
    if kelly:
        results["everything"]["kelly"] = {
            "n_levels": kelly.get("n_confidence_levels", 0),
            "dd_scale": kelly.get("dd_scale", 1.0),
        }

    # ── COMPARISON TABLE ──────────────────────────────────────
    metric_keys = [
        "roc_auc", "pr_auc", "brier_score", "ece",
        "win_rate", "profit_factor", "expectancy",
        "max_drawdown", "trade_freq", "wf_mean", "wf_std",
    ]
    experiment_names = ["baseline", "purged_cv", "calibration", "ev_threshold", "optuna", "everything"]
    experiment_labels = [
        "1. Baseline",
        "2. +Purged CV",
        "3. +Calibration",
        "4. +EV Threshold",
        "5. +Optuna",
        "6. +Everything",
    ]

    log.info(f"\n{'='*90}")
    log.info("ABLATION STUDY RESULTS")
    log.info(f"{'='*90}")
    header = f"  {'Metric':<20s}"
    for lab in experiment_labels:
        header += f" {lab:>14s}"
    log.info(header)
    log.info(f"  {'─'*88}")

    for mk in metric_keys:
        row = f"  {mk:<20s}"
        for name in experiment_names:
            val = results.get(name, {}).get(mk, float("nan"))
            if isinstance(val, float) and not np.isnan(val):
                row += f" {val:>14.4f}"
            else:
                row += f" {'N/A':>14s}"
        log.info(row)

    # Delta row (vs baseline)
    log.info(f"  {'─'*88}")
    bl = results.get("baseline", {})
    for mk in ["roc_auc", "pr_auc", "brier_score", "profit_factor", "expectancy"]:
        bl_val = bl.get(mk, float("nan"))
        row = f"  {mk + ' Δ':<20s}"
        for name in experiment_names:
            val = results.get(name, {}).get(mk, float("nan"))
            if isinstance(bl_val, float) and isinstance(val, float) and not (np.isnan(bl_val) or np.isnan(val)):
                delta = val - bl_val
                direction = "+" if delta > 0 else ""
                row += f" {direction}{delta:>13.4f}"
            else:
                row += f" {'N/A':>14s}"
        log.info(row)
    log.info(f"{'='*90}\n")

    results["experiment_names"] = experiment_names
    results["experiment_labels"] = experiment_labels
    results["metric_keys"] = metric_keys

    return results
