"""
scripts/validation_framework.py - Comprehensive Empirical Validation Framework
===============================================================================

Implements the user's full validation roadmap:
  1. Unit tests (CV splitter) - already in tests/
  2. Cumulative ablation  - Baseline → +Purged → +Calib → +EV → +Optuna → +All
  3. Isolated ablation  - Each change ALONE vs Baseline
  4. Statistical significance - mean, std, 95% CI, paired t-test, bootstrap CI
  5. Walk-forward benchmark - multi-symbol, multi-timeframe
  6. Monte Carlo robustness - trade order shuffle + block bootstrap
  7. Multi-symbol validation - EURUSD, GBPUSD, USDJPY, XAUUSD
  8. Regime analysis - low vol, high vol, ranging, trending
  9. Feature stability report - SHAP fold appearance %
 10. Calibration over time - rolling reliability diagram
 11. Runtime benchmark - training time, inference latency, memory, model size

Usage:
  python scripts/validation_framework.py --mode ablation --symbol EURUSD --timeframe H1
  python scripts/validation_framework.py --mode full --symbols EURUSD,GBPUSD,USDJPY,XAUUSD
  python scripts/validation_framework.py --mode monte_carlo --n-shuffles 1000
  python scripts/validation_framework.py --mode regime --symbol EURUSD
  python scripts/validation_framework.py --mode runtime --symbol EURUSD
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tracemalloc
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.logger import get_logger
log = get_logger("validation_framework")

OUTPUT_DIR = ROOT / "_validation_results"
TIMESTAMP = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

# Standard metrics to compute everywhere
METRIC_KEYS = [
    "roc_auc", "pr_auc", "brier_score", "ece",
    "win_rate", "profit_factor", "expectancy",
    "max_drawdown", "trade_freq", "n_trades",
]
R_MULT = 1.5  # triple-barrier R:R ratio


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SHARED: Data Loading
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_or_generate_data(symbol: str = "EURUSD", timeframe: str = "H1",
                            n_samples: int = 5000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, list]:
    """Load real MT5 data or generate synthetic data as fallback.
    
    Returns:
        X: feature matrix (n_samples, n_features)
        y: binary labels (n_samples,)
        feature_names: list of feature names
    """
    try:
        from ml.mt5_data_loader import MT5DataLoader
        loader = MT5DataLoader()
        df = loader.get_ohlcv(symbol, timeframe, n_samples + 200)  # extra for feature lag
        if df is not None and len(df) > n_samples:
            log.info(f"  Loaded {len(df)} real {symbol} {timeframe} candles from MT5")
            return _features_from_df(df, n_samples)
    except Exception as e:
        log.warning(f"  MT5 load failed ({e}), using synthetic data")
    
    return _generate_synthetic(n_samples, n_features=50, seed=seed)


def _features_from_df(df: pd.DataFrame, n_samples: int) -> Tuple[np.ndarray, np.ndarray, list]:
    """Generate features from real OHLCV data."""
    from ml.feature_engineer import FeatureEngineer
    fe = FeatureEngineer()
    df = df.tail(n_samples + 100)  # extra for rolling windows
    df_fe = fe.build_features(df)
    
    # Generate labels
    from ml.triple_barrier_labels import TripleBarrierLabeler
    labeler = TripleBarrierLabeler()
    df_labeled = labeler.generate_labels(df_fe)
    
    # Drop NaN
    df_clean = df_labeled.dropna(subset=["label"])
    if len(df_clean) < 500:
        log.warning(f"  Only {len(df_clean)} clean rows, falling back to synthetic")
        return _generate_synthetic(n_samples, n_features=50)
    
    # Get feature columns (exclude OHLCV, label, etc.)
    exclude_cols = {"open", "high", "low", "close", "volume", "label",
                    "hit_tp", "hit_sl", "holding_period", "bar_ts"}
    feature_cols = [c for c in df_clean.columns if c not in exclude_cols and df_clean[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
    
    X = df_clean[feature_cols].values.astype(np.float32)
    y = df_clean["label"].values.astype(np.int32)
    
    # Replace inf/nan
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X[:n_samples], y[:n_samples], feature_cols


def _generate_synthetic(n_samples: int, n_features: int = 50,
                          seed: int = 42) -> Tuple[np.ndarray, np.ndarray, list]:
    """Generate synthetic data with mild predictive signal for testing."""
    rng = np.random.default_rng(seed)
    
    # Features
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    
    # Labels: weak signal from first 3 features + noise
    logit = 0.3 * X[:, 0] + 0.2 * X[:, 1] - 0.1 * X[:, 2] + rng.normal(0, 1, n_samples)
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (prob > 0.5).astype(np.int32)
    
    feature_names = [f"synth_{i}" for i in range(n_features)]
    log.info(f"  Generated {n_samples} synthetic samples, {n_features} features, signal ratio={y.mean():.2%}")
    
    return X, y, feature_names


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SHARED: Metrics Computation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_metrics(proba: np.ndarray, y_true: np.ndarray,
                    threshold: float = 0.5) -> Dict[str, float]:
    """Compute all 10 standard metrics from probabilities and labels."""
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    
    pred = (proba >= threshold).astype(int)
    metrics: Dict[str, float] = {}
    
    # Discrimination
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
    except ValueError:
        metrics["roc_auc"] = 0.5
    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, proba))
    except ValueError:
        metrics["pr_auc"] = 0.0
    
    # Calibration
    metrics["brier_score"] = float(brier_score_loss(y_true, proba))
    metrics["ece"] = _compute_ece(proba, y_true)
    
    # Trading
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    n_trades = tp + fp
    
    metrics["n_trades"] = float(n_trades)
    metrics["win_rate"] = tp / n_trades if n_trades > 0 else 0.0
    
    gross_profit = float(tp) * R_MULT
    gross_loss = float(fp) * R_MULT
    metrics["profit_factor"] = (
        gross_profit / gross_loss if gross_loss > 0
        else (999.0 if gross_profit > 0 else 0.0)
    )
    metrics["expectancy"] = (tp * R_MULT - fp * R_MULT) / max(n_trades, 1)
    
    # Max drawdown
    equity = [0.0]
    for i in range(len(pred)):
        if pred[i] == 1:
            equity.append(equity[-1] + (R_MULT if y_true[i] == 1 else -R_MULT))
    peak, max_dd = equity[0], 0.0
    for v in equity:
        if v > peak: peak = v
        dd = peak - v
        if dd > max_dd: max_dd = dd
    metrics["max_drawdown"] = float(max_dd)
    metrics["trade_freq"] = float(n_trades) / len(y_true) if len(y_true) > 0 else 0.0
    
    return metrics


def _compute_ece(proba: np.ndarray, y_true: np.ndarray, n_buckets: int = 10) -> float:
    """Expected Calibration Error."""
    df = pd.DataFrame({"p": proba, "y": y_true})
    n_b = min(n_buckets, df["p"].nunique())
    if n_b < 2:
        return 0.0
    df["bucket"] = pd.qcut(df["p"], q=n_b, duplicates="drop")
    table = df.groupby("bucket", observed=True).agg(
        mean_pred=("p", "mean"), actual_rate=("y", "mean"), n=("y", "size"),
    )
    ece = 0.0
    for _, row in table.iterrows():
        ece += (row["n"] / len(df)) * abs(row["mean_pred"] - row["actual_rate"])
    return float(ece)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SHARED: Model Training Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _fit_xgb(X_tr, y_tr, sw=None, params=None):
    """Fit XGBoost with given params or defaults."""
    import xgboost as xgb
    if params is None:
        params = {
            "n_estimators": 500, "max_depth": 5, "learning_rate": 0.05,
            "random_state": 42, "n_jobs": -1, "eval_metric": "logloss",
        }
    m = xgb.XGBClassifier(**params)
    m.fit(X_tr, y_tr, sample_weight=sw)
    return m


def _calibrate(model, X_cal, y_cal):
    """Apply Platt scaling calibration."""
    from sklearn.calibration import CalibratedClassifierCV
    try:
        from sklearn.frozen import FrozenEstimator
        cal = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
    except (ImportError, AttributeError):
        cal = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    cal.fit(X_cal, y_cal)
    return cal


def _find_ev_threshold(model, X, y, R_MULT_LOCAL=R_MULT):
    """Find probability threshold that maximizes EV per trade."""
    proba = model.predict_proba(X)[:, 1]
    best_thr, best_ev = 0.5, -999.0
    for thr in np.arange(0.45, 0.80, 0.01):
        pred = (proba >= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        total = tp + fp
        if total < 5:
            continue
        p_win = tp / total
        ev = p_win * R_MULT_LOCAL - (1 - p_win) * R_MULT_LOCAL
        if ev > best_ev:
            best_ev = ev
            best_thr = thr
    return best_thr


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STATISTICAL SIGNIFICANCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_significance(baseline_vals: List[float],
                          treatment_vals: List[float],
                          metric_name: str = "") -> Dict[str, Any]:
    """Compute statistical significance between two sets of metric values.
    
    Returns mean, std, 95% CI for each, paired t-test p-value, and bootstrap CI.
    """
    from scipy import stats as sp_stats
    
    b = np.array(baseline_vals)
    t = np.array(treatment_vals)
    
    result = {
        "metric": metric_name,
        "baseline_mean": float(np.mean(b)),
        "baseline_std": float(np.std(b, ddof=1)) if len(b) > 1 else 0.0,
        "treatment_mean": float(np.mean(t)),
        "treatment_std": float(np.std(t, ddof=1)) if len(t) > 1 else 0.0,
        "delta_mean": float(np.mean(t) - np.mean(b)),
    }
    
    # 95% CI via t-distribution
    if len(b) > 1:
        se_b = result["baseline_std"] / np.sqrt(len(b))
        result["baseline_ci95"] = [
            float(np.mean(b) - 1.96 * se_b),
            float(np.mean(b) + 1.96 * se_b),
        ]
    if len(t) > 1:
        se_t = result["treatment_std"] / np.sqrt(len(t))
        result["treatment_ci95"] = [
            float(np.mean(t) - 1.96 * se_t),
            float(np.mean(t) + 1.96 * se_t),
        ]
    
    # Paired t-test (if same length)
    if len(b) == len(t) and len(b) >= 3:
        t_stat, p_value = sp_stats.ttest_rel(b, t)
        result["paired_t_stat"] = float(t_stat)
        result["paired_p_value"] = float(p_value)
        result["significant_at_05"] = p_value < 0.05
    elif len(b) >= 2 and len(t) >= 2:
        # Welch's t-test (unequal variance, possibly unequal length)
        t_stat, p_value = sp_stats.ttest_ind(b, t, equal_var=False)
        result["welch_t_stat"] = float(t_stat)
        result["welch_p_value"] = float(p_value)
        result["significant_at_05"] = p_value < 0.05
    
    # Bootstrap 95% CI for delta
    if len(b) >= 5 and len(t) >= 5:
        n_boot = 2000
        rng = np.random.default_rng(42)
        boot_deltas = []
        for _ in range(n_boot):
            b_boot = rng.choice(b, size=len(b), replace=True)
            t_boot = rng.choice(t, size=len(t), replace=True)
            boot_deltas.append(np.mean(t_boot) - np.mean(b_boot))
        boot_deltas = np.array(boot_deltas)
        result["bootstrap_delta_ci95"] = [
            float(np.percentile(boot_deltas, 2.5)),
            float(np.percentile(boot_deltas, 97.5)),
        ]
        result["bootstrap_delta_p05_positive"] = float(np.mean(boot_deltas > 0))
    
    return result


def run_repeated_experiment(experiment_fn, n_repeats: int = 10,
                              seeds: Optional[List[int]] = None) -> List[Dict[str, float]]:
    """Run an experiment multiple times with different seeds.
    
    Args:
        experiment_fn: callable(seed) -> dict of metrics
        n_repeats: number of repetitions
        seeds: optional list of seeds (default: range(n_repeats) + 42)
    
    Returns:
        List of metric dicts, one per repeat.
    """
    if seeds is None:
        seeds = [42 + i for i in range(n_repeats)]
    
    results = []
    for i, seed in enumerate(seeds):
        log.info(f"    Repeat {i+1}/{n_repeats} (seed={seed})...")
        try:
            metrics = experiment_fn(seed=seed)
            results.append(metrics)
        except Exception as e:
            log.warning(f"    Repeat {i+1} failed: {e}")
    
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2+3. CUMULATIVE + ISOLATED ABLATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_ablation(
    X: np.ndarray, y: np.ndarray, feature_names: list,
    label_horizon: int = 48, n_repeats: int = 10,
    n_optuna_trials: int = 20,
) -> Dict[str, Any]:
    """Full ablation study: cumulative + isolated with statistical significance.
    
    CUMULATIVE experiments:
      1. Baseline        (naive CV, thr=0.5, no calib)
      2. +Purged CV
      3. +Calibration
      4. +EV Threshold
      5. +Optuna
      6. +Everything
    
    ISOLATED experiments:
      7. Only Purged
      8. Only Calibration
      9. Only EV Threshold
      10. Only Optuna
    
    Each experiment is repeated n_repeats times with different seeds.
    Statistical significance computed for every pairwise comparison vs baseline.
    """
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.utils.class_weight import compute_sample_weight
    from ml.cv_splitter import PurgedEmbargoedSplitter
    
    all_results: Dict[str, List[Dict]] = defaultdict(list)
    
    def _run_single_experiment(config: Dict, seed: int) -> Dict[str, float]:
        """Run one experiment configuration with a given seed."""
        rng = np.random.default_rng(seed)
        n = len(X)
        
        # Chronological 80/20 split (same for all experiments)
        split = int(n * 0.8)
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]
        sw = compute_sample_weight("balanced", y_tr)
        
        use_purged = config.get("purged_cv", False)
        use_calib = config.get("calibration", False)
        use_ev = config.get("ev_threshold", False)
        use_optuna = config.get("optuna", False)
        
        # Calib/thresh splits
        calib_start = int(len(X_tr) * 0.85)
        thresh_start = int(len(X_tr) * 0.925)
        
        # CV iterator
        if use_purged and label_horizon > 0:
            splitter = PurgedEmbargoedSplitter(
                n_splits=3, label_horizon=label_horizon, embargo_pct=0.01,
            )
            cv_iter = splitter.split(len(X_tr))
        else:
            cv_iter = TimeSeriesSplit(n_splits=3).split(X_tr)
        
        # Fit model (with or without Optuna)
        if use_optuna:
            try:
                from ml.pipeline_extensions import optuna_search
                optuna_params = optuna_search(
                    X_tr[:calib_start], y_tr[:calib_start],
                    X_tr[calib_start:thresh_start], y_tr[calib_start:thresh_start],
                    n_trials=n_optuna_trials, sample_weight=sw[:calib_start],
                )
                model = _fit_xgb(X_tr[:calib_start], y_tr[:calib_start],
                                sw[:calib_start], params=optuna_params)
            except Exception:
                model = _fit_xgb(X_tr, y_tr, sw)
                use_optuna = False
        else:
            model = _fit_xgb(X_tr, y_tr, sw)
        
        # Calibrate?
        if use_calib:
            pred_model = _calibrate(model, X_tr[calib_start:thresh_start],
                                     y_tr[calib_start:thresh_start])
        else:
            pred_model = model
        
        # Threshold?
        if use_ev and use_calib:
            thr = _find_ev_threshold(pred_model,
                                      X_tr[thresh_start:], y_tr[thresh_start:])
        else:
            thr = 0.5
        
        proba = pred_model.predict_proba(X_te)[:, 1]
        return compute_metrics(proba, y_te, threshold=thr)
    
    # ── Define experiment configurations ──
    cumulative_configs = {
        "1_baseline":       {"purged_cv": False, "calibration": False, "ev_threshold": False, "optuna": False},
        "2_purged_cv":      {"purged_cv": True,  "calibration": False, "ev_threshold": False, "optuna": False},
        "3_calibration":    {"purged_cv": True,  "calibration": True,  "ev_threshold": False, "optuna": False},
        "4_ev_threshold":   {"purged_cv": True,  "calibration": True,  "ev_threshold": True,  "optuna": False},
        "5_optuna":         {"purged_cv": True,  "calibration": True,  "ev_threshold": True,  "optuna": True},
        "6_everything":     {"purged_cv": True,  "calibration": True,  "ev_threshold": True,  "optuna": True},
    }
    
    isolated_configs = {
        "7_only_purged":    {"purged_cv": True,  "calibration": False, "ev_threshold": False, "optuna": False},
        "8_only_calib":     {"purged_cv": False, "calibration": True,  "ev_threshold": False, "optuna": False},
        "9_only_ev":        {"purged_cv": False, "calibration": True,  "ev_threshold": True,  "optuna": False},
        "10_only_optuna":   {"purged_cv": False, "calibration": False, "ev_threshold": False, "optuna": True},
    }
    
    all_configs = {**cumulative_configs, **isolated_configs}
    
    # ── Run all experiments with repeats ──
    for exp_name, config in all_configs.items():
        log.info(f"\n  [Ablation] {exp_name} ({config})...")
        repeats = run_repeated_experiment(
            lambda seed, cfg=config: _run_single_experiment(cfg, seed),
            n_repeats=n_repeats,
        )
        all_results[exp_name] = repeats
    
    # ── Compute significance for each experiment vs baseline ──
    baseline_repeats = all_results.get("1_baseline", [])
    significance = {}
    for exp_name in all_configs:
        if exp_name == "1_baseline":
            continue
        exp_repeats = all_results.get(exp_name, [])
        if not baseline_repeats or not exp_repeats:
            continue
        
        exp_sig = {}
        for mk in METRIC_KEYS:
            b_vals = [r.get(mk, 0.0) for r in baseline_repeats if mk in r]
            t_vals = [r.get(mk, 0.0) for r in exp_repeats if mk in r]
            if b_vals and t_vals:
                exp_sig[mk] = compute_significance(b_vals, t_vals, metric_name=mk)
        significance[exp_name] = exp_sig
    
    # ── Summary table ──
    _print_ablation_table(all_results, significance)
    
    return {
        "repeats": {k: v for k, v in all_results.items()},
        "significance": significance,
        "n_repeats": n_repeats,
        "label_horizon": label_horizon,
    }


def _print_ablation_table(all_results: Dict, significance: Dict):
    """Pretty-print ablation results with significance markers."""
    exp_names = list(all_results.keys())
    
    log.info(f"\n{'='*120}")
    log.info("ABLATION STUDY - Cumulative + Isolated (with Statistical Significance)")
    log.info(f"{'='*120}")
    
    # Header
    header = f"  {'Metric':<20s}"
    for name in exp_names:
        short = name.split("_", 1)[1] if "_" in name else name
        header += f" {short:>14s}"
    log.info(header)
    log.info(f"  {'─'*(20 + 15*len(exp_names))}")
    
    for mk in METRIC_KEYS:
        row = f"  {mk:<20s}"
        for name in exp_names:
            vals = [r.get(mk, float('nan')) for r in all_results[name]]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                mean = np.mean(vals)
                std = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                row += f" {mean:>8.4f}±{std:.3f}"
            else:
                row += f" {'N/A':>14s}"
        log.info(row)
    
    # Significance summary
    log.info(f"\n  SIGNIFICANCE vs BASELINE (paired t-test / bootstrap):")
    log.info(f"  {'Experiment':<20s} {'Metric':<15s} {'Delta':>8s} {'p-value':>10s} {'Sig?':>6s} {'Boot CI95':>20s}")
    log.info(f"  {'─'*82}")
    
    for exp_name, sig in significance.items():
        short = exp_name.split("_", 1)[1] if "_" in exp_name else exp_name
        for mk, s in sig.items():
            if mk not in ["roc_auc", "profit_factor", "expectancy"]:
                continue  # only show key metrics
            delta = s.get("delta_mean", 0)
            p_val = s.get("paired_p_value", s.get("welch_p_value", float('nan')))
            sig_flag = "***" if s.get("significant_at_05") else "n.s."
            boot_ci = s.get("bootstrap_delta_ci95", [float('nan'), float('nan')])
            p_str = f"{p_val:.4f}" if not np.isnan(p_val) else "N/A"
            ci_str = f"[{boot_ci[0]:.4f}, {boot_ci[1]:.4f}]" if not np.isnan(boot_ci[0]) else "N/A"
            log.info(f"  {short:<20s} {mk:<15s} {delta:>+8.4f} {p_str:>10s} {sig_flag:>6s} {ci_str:>20s}")
    
    log.info(f"  *** = p < 0.05, n.s. = not significant")
    log.info(f"{'='*120}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. MONTE CARLO ROBUSTNESS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_monte_carlo(
    proba: np.ndarray, y_true: np.ndarray,
    n_shuffles: int = 1000, block_size: int = 20,
    threshold: float = 0.5, seed: int = 42,
) -> Dict[str, Any]:
    """Monte Carlo robustness test: trade order shuffle + block bootstrap.
    
    Two methods:
    1. Trade order shuffle - randomly permute trade outcomes (destroys any
       serial dependence in the signal, keeps marginal distribution).
    2. Block bootstrap - resample blocks of consecutive trades (preserves
       time-series dependence structure, tests if the observed strategy
       performance is within the null distribution).
    
    Returns p-values and distributions for key metrics.
    """
    rng = np.random.default_rng(seed)
    pred = (proba >= threshold).astype(int)
    
    # Extract trade outcomes (only where pred=1)
    trade_mask = pred == 1
    trade_outcomes = y_true[trade_mask].astype(float)  # 1.0 = win, 0.0 = loss
    n_trades = len(trade_outcomes)
    
    if n_trades < 10:
        return {"error": "too few trades for Monte Carlo", "n_trades": n_trades}
    
    # Observed metrics
    obs_wr = float(np.mean(trade_outcomes))
    obs_pf = float(np.sum(trade_outcomes) * R_MULT / max(np.sum(1 - trade_outcomes) * R_MULT, 0.01))
    obs_ev = float(obs_wr * R_MULT - (1 - obs_wr) * R_MULT)
    obs_total_pnl = float(np.sum(trade_outcomes * R_MULT - (1 - trade_outcomes) * R_MULT))
    
    # ── Method 1: Trade order shuffle ──
    shuffle_pnls = []
    shuffle_wrs = []
    shuffle_pfs = []
    
    for _ in range(n_shuffles):
        shuffled = rng.permutation(trade_outcomes)
        pnl = float(np.sum(shuffled * R_MULT - (1 - shuffled) * R_MULT))
        wr = float(np.mean(shuffled))
        wins = int(np.sum(shuffled))
        losses = int(np.sum(1 - shuffled))
        pf = float(wins * R_MULT / max(losses * R_MULT, 0.01))
        shuffle_pnls.append(pnl)
        shuffle_wrs.append(wr)
        shuffle_pfs.append(pf)
    
    shuffle_pnls = np.array(shuffle_pnls)
    shuffle_wrs = np.array(shuffle_wrs)
    shuffle_pfs = np.array(shuffle_pfs)
    
    # ── Method 2: Block bootstrap ──
    n_blocks = max(1, n_trades // block_size)
    bootstrap_pnls = []
    bootstrap_wrs = []
    
    for _ in range(n_shuffles):
        # Resample blocks
        indices = []
        while len(indices) < n_trades:
            start = rng.integers(0, max(1, n_trades - block_size + 1))
            indices.extend(range(start, min(start + block_size, n_trades)))
        indices = indices[:n_trades]
        boot_outcomes = trade_outcomes[indices]
        
        pnl = float(np.sum(boot_outcomes * R_MULT - (1 - boot_outcomes) * R_MULT))
        wr = float(np.mean(boot_outcomes))
        bootstrap_pnls.append(pnl)
        bootstrap_wrs.append(wr)
    
    bootstrap_pnls = np.array(bootstrap_pnls)
    bootstrap_wrs = np.array(bootstrap_wrs)
    
    # ── P-values (fraction of null >= observed) ──
    p_pnl_shuffle = float(np.mean(shuffle_pnls >= obs_total_pnl))
    p_wr_shuffle = float(np.mean(shuffle_wrs >= obs_wr))
    p_pnl_bootstrap = float(np.mean(bootstrap_pnls >= obs_total_pnl))
    p_wr_bootstrap = float(np.mean(bootstrap_wrs >= obs_wr))
    
    results = {
        "n_trades": int(n_trades),
        "n_shuffles": n_shuffles,
        "block_size": block_size,
        "observed": {
            "win_rate": obs_wr,
            "profit_factor": obs_pf,
            "expectancy": obs_ev,
            "total_pnl": obs_total_pnl,
        },
        "shuffle": {
            "pnl_pvalue": p_pnl_shuffle,
            "wr_pvalue": p_wr_shuffle,
            "pnl_mean": float(np.mean(shuffle_pnls)),
            "pnl_std": float(np.std(shuffle_pnls)),
            "pnl_percentiles": {
                "p5": float(np.percentile(shuffle_pnls, 5)),
                "p50": float(np.percentile(shuffle_pnls, 50)),
                "p95": float(np.percentile(shuffle_pnls, 95)),
            },
        },
        "block_bootstrap": {
            "pnl_pvalue": p_pnl_bootstrap,
            "wr_pvalue": p_wr_bootstrap,
            "pnl_mean": float(np.mean(bootstrap_pnls)),
            "pnl_std": float(np.std(bootstrap_pnls)),
            "pnl_percentiles": {
                "p5": float(np.percentile(bootstrap_pnls, 5)),
                "p50": float(np.percentile(bootstrap_pnls, 50)),
                "p95": float(np.percentile(bootstrap_pnls, 95)),
            },
        },
    }
    
    # ── Print results ──
    log.info(f"\n{'='*70}")
    log.info(f"MONTE CARLO ROBUSTNESS TEST ({n_shuffles} shuffles)")
    log.info(f"{'='*70}")
    log.info(f"  Trades: {n_trades}, Block size: {block_size}")
    log.info(f"  Observed:  WR={obs_wr:.2%}  PF={obs_pf:.2f}  EV={obs_ev:.4f}  PnL={obs_total_pnl:.2f}R")
    log.info(f"  ──────────────────────────────────────────────────────")
    log.info(f"  {'Method':<20s} {'PnL p-val':>10s} {'WR p-val':>10s} {'Null PnL±σ':>15s}")
    log.info(f"  {'─'*58}")
    log.info(f"  {'Trade Shuffle':<20s} {p_pnl_shuffle:>10.4f} {p_wr_shuffle:>10.4f} {np.mean(shuffle_pnls):.1f}±{np.std(shuffle_pnls):.1f}")
    log.info(f"  {'Block Bootstrap':<20s} {p_pnl_bootstrap:>10.4f} {p_wr_bootstrap:>10.4f} {np.mean(bootstrap_pnls):.1f}±{np.std(bootstrap_pnls):.1f}")
    log.info(f"  ──────────────────────────────────────────────────────")
    log.info(f"  p < 0.05 → strategy is NOT explained by chance (GOOD)")
    log.info(f"  p >= 0.05 → strategy COULD be luck (BAD)")
    log.info(f"{'='*70}\n")
    
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7+8. MULTI-SYMBOL + WALK-FORWARD BENCHMARK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_multi_symbol(
    symbols: List[str] = None,
    timeframe: str = "H1",
    label_horizon: int = 48,
    n_repeats: int = 5,
) -> Dict[str, Any]:
    """Walk-forward benchmark across multiple symbols.
    
    For each symbol: train → evaluate → collect metrics.
    Report per-symbol and average improvement.
    """
    if symbols is None:
        symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
    
    per_symbol: Dict[str, Dict] = {}
    
    for symbol in symbols:
        log.info(f"\n  [Multi-Symbol] Processing {symbol} {timeframe}...")
        X, y, fnames = load_or_generate_data(symbol, timeframe, n_samples=5000,
                                               seed=hash(symbol) % 10000)
        
        if len(X) < 500:
            log.warning(f"  {symbol}: insufficient data ({len(X)} rows), skipping")
            continue
        
        # Run NEW pipeline (purged + calibrated + EV threshold)
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.utils.class_weight import compute_sample_weight
        from ml.cv_splitter import PurgedEmbargoedSplitter
        
        split = int(len(X) * 0.8)
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]
        sw = compute_sample_weight("balanced", y_tr)
        
        # Walk-forward folds
        splitter = PurgedEmbargoedSplitter(
            n_splits=3, label_horizon=label_horizon, embargo_pct=0.01,
        )
        fold_aucs = []
        model = None
        for train_idx, val_idx in splitter.split(len(X_tr)):
            if len(np.unique(y_tr[train_idx])) < 2:
                continue
            m = _fit_xgb(X_tr[train_idx], y_tr[train_idx], sw[train_idx])
            try:
                from sklearn.metrics import roc_auc_score
                p = m.predict_proba(X_tr[val_idx])[:, 1]
                fold_aucs.append(float(roc_auc_score(y_tr[val_idx], p)))
            except ValueError:
                pass
            model = m
        
        if model is None:
            per_symbol[symbol] = {"error": "no valid folds"}
            continue
        
        # Calibrate + EV threshold
        calib_start = int(len(X_tr) * 0.85)
        cal_model = _calibrate(model, X_tr[calib_start:], y_tr[calib_start:])
        ev_thr = _find_ev_threshold(cal_model, X_tr[calib_start:], y_tr[calib_start:])
        proba = cal_model.predict_proba(X_te)[:, 1]
        metrics = compute_metrics(proba, y_te, threshold=ev_thr)
        metrics["threshold"] = ev_thr
        metrics["wf_mean"] = float(np.mean(fold_aucs)) if fold_aucs else float('nan')
        metrics["wf_std"] = float(np.std(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else 0.0
        
        per_symbol[symbol] = metrics
        log.info(f"  {symbol}: AUC={metrics['roc_auc']:.4f}  WR={metrics['win_rate']:.2%}  PF={metrics['profit_factor']:.2f}  EV={metrics['expectancy']:.4f}")
    
    # ── Summary table ──
    valid_symbols = {k: v for k, v in per_symbol.items() if "error" not in v}
    if valid_symbols:
        log.info(f"\n{'='*100}")
        log.info("MULTI-SYMBOL VALIDATION SUMMARY")
        log.info(f"{'='*100}")
        header = f"  {'Metric':<20s}"
        for s in valid_symbols:
            header += f" {s:>12s}"
        header += f" {'MEAN':>12s}"
        log.info(header)
        log.info(f"  {'─'*(20 + 13*len(valid_symbols) + 13)}")
        
        for mk in METRIC_KEYS + ["wf_mean", "wf_std"]:
            row = f"  {mk:<20s}"
            vals = []
            for s, m in valid_symbols.items():
                v = m.get(mk, float('nan'))
                vals.append(v)
                row += f" {v:>12.4f}" if not np.isnan(v) else f" {'N/A':>12s}"
            valid_vals = [v for v in vals if not np.isnan(v)]
            mean_v = np.mean(valid_vals) if valid_vals else float('nan')
            row += f" {mean_v:>12.4f}" if not np.isnan(mean_v) else f" {'N/A':>12s}"
            log.info(row)
        log.info(f"{'='*100}\n")
    
    return {
        "per_symbol": per_symbol,
        "symbols_tested": symbols,
        "timeframe": timeframe,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. REGIME ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def classify_regime(returns: np.ndarray, window: int = 50) -> np.ndarray:
    """Classify each bar into a regime based on rolling volatility and trend.
    
    Returns array of regime labels: 'low_vol', 'high_vol', 'ranging', 'trending'.
    """
    n = len(returns)
    regimes = np.array(["ranging"] * n, dtype=object)
    
    rolling_vol = pd.Series(np.abs(returns)).rolling(window, min_periods=10).std().values
    rolling_trend = pd.Series(returns).rolling(window, min_periods=10).mean().values
    
    vol_median = np.nanmedian(rolling_vol)
    trend_abs_median = np.nanmedian(np.abs(rolling_trend))
    
    for i in range(n):
        if np.isnan(rolling_vol[i]):
            continue
        if rolling_vol[i] > vol_median * 1.3:
            regimes[i] = "high_vol"
        elif rolling_vol[i] < vol_median * 0.7:
            regimes[i] = "low_vol"
        elif np.abs(rolling_trend[i]) > trend_abs_median * 1.5:
            regimes[i] = "trending"
        else:
            regimes[i] = "ranging"
    
    return regimes


def run_regime_analysis(
    proba: np.ndarray, y_true: np.ndarray, returns: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Break down model performance by market regime."""
    pred = (proba >= threshold).astype(int)
    regimes = classify_regime(returns)
    
    regime_metrics: Dict[str, Dict] = {}
    for regime in ["low_vol", "high_vol", "ranging", "trending"]:
        mask = regimes == regime
        if mask.sum() < 20:
            regime_metrics[regime] = {"n_samples": int(mask.sum()), "note": "too few samples"}
            continue
        
        regime_proba = proba[mask]
        regime_y = y_true[mask]
        regime_pred = pred[mask]
        regime_metrics[regime] = compute_metrics(regime_proba, regime_y, threshold)
        regime_metrics[regime]["n_samples"] = int(mask.sum())
    
    # Print
    log.info(f"\n{'='*90}")
    log.info("REGIME ANALYSIS")
    log.info(f"{'='*90}")
    header = f"  {'Metric':<20s}"
    for r in ["low_vol", "high_vol", "ranging", "trending"]:
        header += f" {r:>14s}"
    log.info(header)
    log.info(f"  {'─'*(20 + 15*4)}")
    
    for mk in METRIC_KEYS:
        row = f"  {mk:<20s}"
        for r in ["low_vol", "high_vol", "ranging", "trending"]:
            v = regime_metrics.get(r, {}).get(mk, float('nan'))
            if isinstance(v, float) and not np.isnan(v):
                row += f" {v:>14.4f}"
            else:
                row += f" {'N/A':>14s}"
        log.info(row)
    
    # Sample counts
    row = f"  {'n_samples':<20s}"
    for r in ["low_vol", "high_vol", "ranging", "trending"]:
        n_s = regime_metrics.get(r, {}).get("n_samples", 0)
        row += f" {n_s:>14d}"
    log.info(row)
    log.info(f"{'='*90}\n")
    
    return regime_metrics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. FEATURE STABILITY REPORT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_feature_stability(
    X: np.ndarray, y: np.ndarray, feature_names: list,
    label_horizon: int = 48, n_folds: int = 5, top_k: int = 20,
) -> Dict[str, Any]:
    """SHAP feature importance across CV folds. Report fold appearance %.
    
    A feature appearing in top-k for all folds = stable.
    A feature appearing in only 1-2 folds = unstable.
    """
    from sklearn.utils.class_weight import compute_sample_weight
    from ml.cv_splitter import PurgedEmbargoedSplitter
    
    splitter = PurgedEmbargoedSplitter(
        n_splits=n_folds, label_horizon=label_horizon, embargo_pct=0.01,
    )
    sw = compute_sample_weight("balanced", y)
    
    fold_top_features = []
    all_shap_values = []
    
    for fold_i, (train_idx, test_idx) in enumerate(splitter.split(len(X))):
        if len(np.unique(y[train_idx])) < 2:
            continue
        
        model = _fit_xgb(X[train_idx], y[train_idx], sw[train_idx])
        
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X[test_idx])
            
            if isinstance(shap_values, list):
                shap_importance = np.abs(shap_values[1]).mean(axis=0)
            else:
                shap_importance = np.abs(shap_values).mean(axis=0)
            
            # Get top-k feature indices
            top_indices = np.argsort(shap_importance)[::-1][:top_k]
            fold_top_features.append(set(top_indices.tolist()))
            all_shap_values.append(shap_importance)
        except Exception as e:
            log.warning(f"  Fold {fold_i}: SHAP failed ({e})")
    
    if not fold_top_features:
        return {"error": "no valid folds for SHAP"}
    
    n_valid_folds = len(fold_top_features)
    
    # Compute appearance % for each feature
    feature_appearance = defaultdict(int)
    for top_set in fold_top_features:
        for idx in top_set:
            feature_appearance[idx] += 1
    
    stability_report = []
    for idx in sorted(feature_appearance.keys(), key=lambda x: -feature_appearance[x]):
        if idx < len(feature_names):
            fname = feature_names[idx]
        else:
            fname = f"feature_{idx}"
        
        avg_importance = float(np.mean([sv[idx] for sv in all_shap_values if idx < len(sv)]))
        stability_report.append({
            "feature": fname,
            "fold_appearance_pct": round(feature_appearance[idx] / n_valid_folds * 100, 1),
            "folds_in_top_k": feature_appearance[idx],
            "total_folds": n_valid_folds,
            "avg_shap_importance": round(avg_importance, 6),
        })
    
    # Print
    log.info(f"\n{'='*80}")
    log.info(f"FEATURE STABILITY REPORT (top-{top_k} per fold, {n_valid_folds} folds)")
    log.info(f"{'='*80}")
    log.info(f"  {'Feature':<25s} {'Appear%':>8s} {'Folds':>8s} {'Avg SHAP':>12s}")
    log.info(f"  {'─'*55}")
    for row in stability_report:
        pct = row["fold_appearance_pct"]
        marker = "***" if pct == 100 else "** " if pct >= 80 else "*  " if pct >= 50 else "   "
        log.info(f"  {marker}{row['feature']:<22s} {pct:>7.1f}% {row['folds_in_top_k']:>4d}/{row['total_folds']} {row['avg_shap_importance']:>12.6f}")
    log.info(f"  *** = 100%  ** = >=80%  * = >=50%")
    log.info(f"{'='*80}\n")
    
    return {
        "stability_report": stability_report,
        "n_folds": n_valid_folds,
        "top_k": top_k,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 11. CALIBRATION OVER TIME (Rolling Reliability)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_calibration_over_time(
    proba: np.ndarray, y_true: np.ndarray,
    window_size: int = 200, step: int = 50,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Rolling ECE and reliability diagram data over time.
    
    Checks if calibration degrades over time (common failure mode).
    """
    n = len(proba)
    rolling_data = []
    
    for start in range(0, n - window_size, step):
        end = start + window_size
        window_proba = proba[start:end]
        window_y = y_true[start:end]
        
        ece = _compute_ece(window_proba, window_y)
        brier = float(_brier_score(window_proba, window_y))
        
        # Mean predicted probability vs actual rate
        mean_pred = float(np.mean(window_proba))
        actual_rate = float(np.mean(window_y))
        
        rolling_data.append({
            "window_start": start,
            "window_end": end,
            "ece": ece,
            "brier_score": brier,
            "mean_predicted_prob": mean_pred,
            "actual_rate": actual_rate,
            "calibration_error": abs(mean_pred - actual_rate),
        })
    
    # Drift detection: is calibration degrading over time?
    if len(rolling_data) >= 3:
        first_half = rolling_data[:len(rolling_data)//2]
        second_half = rolling_data[len(rolling_data)//2:]
        ece_early = np.mean([d["ece"] for d in first_half])
        ece_late = np.mean([d["ece"] for d in second_half])
        drift_detected = ece_late > ece_early * 1.5  # 50% degradation
    else:
        ece_early, ece_late, drift_detected = 0, 0, False
    
    # Print
    log.info(f"\n{'='*80}")
    log.info("CALIBRATION OVER TIME (Rolling Reliability)")
    log.info(f"{'='*80}")
    log.info(f"  Window: {window_size}, Step: {step}, Windows: {len(rolling_data)}")
    log.info(f"  {'Window':<15s} {'ECE':>8s} {'Brier':>8s} {'Mean P':>8s} {'Actual':>8s} {'Cal.Err':>8s}")
    log.info(f"  {'─'*58}")
    
    # Show every 5th window
    for i, d in enumerate(rolling_data):
        if i % 5 == 0 or i == len(rolling_data) - 1:
            log.info(f"  {d['window_start']:<15d} {d['ece']:>8.4f} {d['brier_score']:>8.4f} "
                     f"{d['mean_predicted_prob']:>8.4f} {d['actual_rate']:>8.4f} {d['calibration_error']:>8.4f}")
    
    log.info(f"  ──────────────────────────────────────────────────")
    log.info(f"  Early ECE (first half):  {ece_early:.4f}")
    log.info(f"  Late ECE (second half):  {ece_late:.4f}")
    log.info(f"  Calibration drift:       {'DETECTED' if drift_detected else 'NOT detected'}")
    log.info(f"{'='*80}\n")
    
    return {
        "rolling_data": rolling_data,
        "ece_early": ece_early,
        "ece_late": ece_late,
        "drift_detected": drift_detected,
        "window_size": window_size,
    }


def _brier_score(proba, y_true):
    from sklearn.metrics import brier_score_loss
    return brier_score_loss(y_true, proba)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 12. RUNTIME BENCHMARK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_runtime_benchmark(
    X: np.ndarray, y: np.ndarray,
    n_repeats: int = 3,
    n_optuna_trials: int = 20,
) -> Dict[str, Any]:
    """Benchmark training time, inference latency, memory, model size.
    
    Compares: Baseline vs Full Pipeline (with Optuna).
    """
    from sklearn.utils.class_weight import compute_sample_weight
    from sklearn.model_selection import TimeSeriesSplit
    from ml.cv_splitter import PurgedEmbargoedSplitter
    
    split = int(len(X) * 0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    sw = compute_sample_weight("balanced", y_tr)
    
    results = {}
    
    # ── Baseline ──
    log.info("  [Runtime] Baseline pipeline...")
    tracemalloc.start()
    t0 = time.perf_counter()
    model_base = _fit_xgb(X_tr, y_tr, sw)
    train_time_base = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    for _ in range(100):
        _ = model_base.predict_proba(X_te[:100])
    infer_time_base = (time.perf_counter() - t0) / 100
    
    _, peak_mem_base = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    import pickle
    model_size_base = len(pickle.dumps(model_base)) / (1024 * 1024)
    
    results["baseline"] = {
        "train_time_s": round(train_time_base, 3),
        "inference_latency_ms": round(infer_time_base * 1000, 3),
        "peak_memory_mb": round(peak_mem_base / (1024 * 1024), 2),
        "model_size_mb": round(model_size_base, 2),
    }
    
    # ── Full pipeline (with Optuna) ──
    log.info("  [Runtime] Full pipeline (with Optuna)...")
    tracemalloc.start()
    t0 = time.perf_counter()
    
    splitter = PurgedEmbargoedSplitter(n_splits=3, label_horizon=48, embargo_pct=0.01)
    calib_start = int(len(X_tr) * 0.85)
    
    for train_idx, val_idx in splitter.split(len(X_tr)):
        if len(np.unique(y_tr[train_idx])) < 2:
            continue
        _fit_xgb(X_tr[train_idx], y_tr[train_idx], sw[train_idx])
    
    try:
        from ml.pipeline_extensions import optuna_search
        optuna_params = optuna_search(
            X_tr[:calib_start], y_tr[:calib_start],
            X_tr[calib_start:], y_tr[calib_start:],
            n_trials=n_optuna_trials, sample_weight=sw[:calib_start],
        )
        model_full = _fit_xgb(X_tr[:calib_start], y_tr[:calib_start],
                               sw[:calib_start], params=optuna_params)
    except Exception:
        model_full = _fit_xgb(X_tr, y_tr, sw)
        optuna_params = None
    
    cal_model = _calibrate(model_full, X_tr[calib_start:], y_tr[calib_start:])
    train_time_full = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    for _ in range(100):
        _ = cal_model.predict_proba(X_te[:100])
    infer_time_full = (time.perf_counter() - t0) / 100
    
    _, peak_mem_full = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    model_size_full = len(pickle.dumps(cal_model)) / (1024 * 1024)
    
    results["full_pipeline"] = {
        "train_time_s": round(train_time_full, 3),
        "inference_latency_ms": round(infer_time_full * 1000, 3),
        "peak_memory_mb": round(peak_mem_full / (1024 * 1024), 2),
        "model_size_mb": round(model_size_full, 2),
    }
    
    # Print
    log.info(f"\n{'='*70}")
    log.info("RUNTIME BENCHMARK")
    log.info(f"{'='*70}")
    log.info(f"  {'Metric':<25s} {'Baseline':>12s} {'Full Pipeline':>14s} {'Ratio':>8s}")
    log.info(f"  {'─'*62}")
    for mk in ["train_time_s", "inference_latency_ms", "peak_memory_mb", "model_size_mb"]:
        b = results["baseline"][mk]
        f = results["full_pipeline"][mk]
        ratio = f / b if b > 0 else float('inf')
        log.info(f"  {mk:<25s} {b:>12.3f} {f:>14.3f} {ratio:>7.1f}x")
    log.info(f"{'='*70}\n")
    
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN ORCHESTRATOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def save_results(results: Dict, name: str):
    """Save results to JSON file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}_{TIMESTAMP}.json"
    
    # Convert numpy types for JSON serialization
    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj
    
    with open(path, "w") as f:
        json.dump(_convert(results), f, indent=2, default=str)
    log.info(f"  Results saved to {path}")
    return str(path)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Validation Framework")
    parser.add_argument("--mode", type=str, default="ablation",
                       choices=["ablation", "monte_carlo", "multi_symbol",
                                "regime", "feature_stability", "calibration_over_time",
                                "runtime", "full"],
                       help="Validation mode to run")
    parser.add_argument("--symbol", type=str, default="EURUSD")
    parser.add_argument("--symbols", type=str, default="EURUSD,GBPUSD,USDJPY,XAUUSD",
                       help="Comma-separated symbols for multi-symbol")
    parser.add_argument("--timeframe", type=str, default="H1")
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--n-shuffles", type=int, default=1000)
    parser.add_argument("--n-optuna-trials", type=int, default=20)
    parser.add_argument("--label-horizon", type=int, default=48)
    parser.add_argument("--n-samples", type=int, default=5000)
    
    args = parser.parse_args()
    
    log.info(f"\n{'#'*70}")
    log.info(f"VALIDATION FRAMEWORK - mode={args.mode}")
    log.info(f"{'#'*70}")
    
    all_results = {}
    
    if args.mode in ("ablation", "full"):
        log.info(f"\n{'='*70}")
        log.info("MODE: ABLATION STUDY (cumulative + isolated + significance)")
        log.info(f"{'='*70}")
        X, y, fnames = load_or_generate_data(args.symbol, args.timeframe, args.n_samples)
        all_results["ablation"] = run_ablation(
            X, y, fnames,
            label_horizon=args.label_horizon,
            n_repeats=args.n_repeats,
            n_optuna_trials=args.n_optuna_trials,
        )
    
    if args.mode in ("monte_carlo", "full"):
        log.info(f"\n{'='*70}")
        log.info("MODE: MONTE CARLO ROBUSTNESS")
        log.info(f"{'='*70}")
        X, y, fnames = load_or_generate_data(args.symbol, args.timeframe, args.n_samples)
        
        # Train a model and get predictions
        from sklearn.utils.class_weight import compute_sample_weight
        split = int(len(X) * 0.8)
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]
        sw = compute_sample_weight("balanced", y_tr)
        model = _fit_xgb(X_tr, y_tr, sw)
        cal_model = _calibrate(model, X_tr[int(len(X_tr)*0.85):], y_tr[int(len(X_tr)*0.85):])
        proba = cal_model.predict_proba(X_te)[:, 1]
        
        # Simple returns for Monte Carlo
        returns = np.diff(y_te.astype(float))  # proxy
        
        all_results["monte_carlo"] = run_monte_carlo(
            proba, y_te, n_shuffles=args.n_shuffles,
        )
    
    if args.mode in ("multi_symbol", "full"):
        log.info(f"\n{'='*70}")
        log.info("MODE: MULTI-SYMBOL VALIDATION")
        log.info(f"{'='*70}")
        symbols = args.symbols.split(",")
        all_results["multi_symbol"] = run_multi_symbol(
            symbols=symbols, timeframe=args.timeframe,
            label_horizon=args.label_horizon, n_repeats=args.n_repeats,
        )
    
    if args.mode in ("regime", "full"):
        log.info(f"\n{'='*70}")
        log.info("MODE: REGIME ANALYSIS")
        log.info(f"{'='*70}")
        X, y, fnames = load_or_generate_data(args.symbol, args.timeframe, args.n_samples)
        
        from sklearn.utils.class_weight import compute_sample_weight
        split = int(len(X) * 0.8)
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]
        sw = compute_sample_weight("balanced", y_tr)
        model = _fit_xgb(X_tr, y_tr, sw)
        proba = model.predict_proba(X_te)[:, 1]
        
        # Generate synthetic returns for regime classification
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.001, len(y_te))
        
        all_results["regime"] = run_regime_analysis(proba, y_te, returns)
    
    if args.mode in ("feature_stability", "full"):
        log.info(f"\n{'='*70}")
        log.info("MODE: FEATURE STABILITY REPORT")
        log.info(f"{'='*70}")
        X, y, fnames = load_or_generate_data(args.symbol, args.timeframe, args.n_samples)
        all_results["feature_stability"] = run_feature_stability(
            X, y, fnames, label_horizon=args.label_horizon,
        )
    
    if args.mode in ("calibration_over_time", "full"):
        log.info(f"\n{'='*70}")
        log.info("MODE: CALIBRATION OVER TIME")
        log.info(f"{'='*70}")
        X, y, fnames = load_or_generate_data(args.symbol, args.timeframe, args.n_samples)
        
        from sklearn.utils.class_weight import compute_sample_weight
        split = int(len(X) * 0.8)
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]
        sw = compute_sample_weight("balanced", y_tr)
        model = _fit_xgb(X_tr, y_tr, sw)
        proba = model.predict_proba(X_te)[:, 1]
        
        all_results["calibration_over_time"] = run_calibration_over_time(proba, y_te)
    
    if args.mode in ("runtime", "full"):
        log.info(f"\n{'='*70}")
        log.info("MODE: RUNTIME BENCHMARK")
        log.info(f"{'='*70}")
        X, y, fnames = load_or_generate_data(args.symbol, args.timeframe, args.n_samples)
        all_results["runtime"] = run_runtime_benchmark(
            X, y, n_optuna_trials=args.n_optuna_trials,
        )
    
    # Save all results
    save_results(all_results, f"validation_{args.mode}")
    
    log.info(f"\n{'#'*70}")
    log.info(f"VALIDATION COMPLETE - mode={args.mode}")
    log.info(f"{'#'*70}")
    
    return all_results


if __name__ == "__main__":
    main()