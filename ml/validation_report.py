"""
ml/validation_report.py — Validation Report Generator (Day 72)
=================================================================

Generates human-readable validation reports + persists them for the
dashboard. Also provides a convenience function to validate all models
in the ModelStore at once.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger
from ml.validation import ValidationReport, get_validation_engine

log = get_logger("validation_report")


def generate_text_report(report: ValidationReport) -> str:
    """Generate a human-readable text validation report."""
    lines = [
        "═══════════════════════════════════════════════════════",
        f"  🛡 MODEL VALIDATION REPORT",
        "═══════════════════════════════════════════════════════",
        f"  Model:    {report.model_name} {report.version}",
        f"  Date:     {report.validated_at}",
        "───────────────────────────────────────────────────────",
        f"  1. Walk-Forward Test:    {'✅ PASS' if report.walk_forward.get('passed') else '❌ FAIL'}",
        f"     Score: {report.walk_forward.get('score', 0):.0f}/100",
        f"     Avg PF: {report.walk_forward.get('avg_profit_factor', 0):.2f}",
        f"     Avg WR: {report.walk_forward.get('avg_win_rate', 0):.1%}",
        f"     Folds: {len(report.walk_forward.get('folds', []))}",
        "───────────────────────────────────────────────────────",
        f"  2. Monte Carlo Sim:      {'✅ PASS' if report.monte_carlo.get('passed') else '❌ FAIL'}",
        f"     Score: {report.monte_carlo.get('score', 0):.0f}/100",
        f"     Simulations: {report.monte_carlo.get('simulations', 0)}",
        f"     Percentile: {report.monte_carlo.get('percentile', 0)}%",
        f"     Survival: {report.monte_carlo.get('survival_rate', 0)}%",
        f"     Prob of Ruin: {report.monte_carlo.get('probability_of_ruin', 0)}%",
        "───────────────────────────────────────────────────────",
        f"  3. Regime Robustness:    {'✅ PASS' if report.regime.get('passed') else '❌ FAIL'}",
        f"     Score: {report.regime.get('score', 0):.0f}/100",
        f"     Regimes passed: {report.regime.get('regimes_passed', 0)}/{report.regime.get('total_regimes', 0)}",
        f"     Avg PF: {report.regime.get('avg_profit_factor', 0):.2f}",
        f"     Min PF: {report.regime.get('min_profit_factor', 0):.2f}",
        "───────────────────────────────────────────────────────",
        f"  4. Sensitivity Test:     {'✅ PASS' if report.sensitivity.get('passed') else '❌ FAIL'}",
        f"     Score: {report.sensitivity.get('score', 0):.0f}/100",
        f"     Status: {report.sensitivity.get('status', 'UNKNOWN')}",
        f"     Accuracy drop: {report.sensitivity.get('accuracy_drop', 0):.1%}",
        "───────────────────────────────────────────────────────",
        f"  5. Data Leakage:         {report.leakage.get('status', 'UNKNOWN')}",
        f"     Suspicious cols: {report.leakage.get('suspicious_columns', [])}",
        "───────────────────────────────────────────────────────",
        f"  6. Benchmark:            {'✅' if report.benchmark.get('passed') else '❌'}",
        f"     Model: {report.benchmark.get('model_accuracy', 0):.1%}",
        f"     Random: {report.benchmark.get('random_accuracy', 0):.1%}",
        f"     Buy&Hold: {report.benchmark.get('buyhold_accuracy', 0):.1%}",
        "═══════════════════════════════════════════════════════",
        f"  FINAL SCORE: {report.final_score:.0f}/100",
        f"  STATUS: {'✅ APPROVED FOR TRADING' if report.approved else '❌ REJECTED'}",
        f"  REASON: {report.reason}",
        "═══════════════════════════════════════════════════════",
    ]
    return "\n".join(lines)


def validate_all_models(pair: str = "EURUSD", timeframe: str = "15m") -> List[Dict[str, Any]]:
    """Validate all models in the ModelStore for a given pair.

    Returns a list of validation report dicts.
    """
    from ml.model_store import get_model_store
    from ml.feature_store import get_feature_store
    from ml.dataset_builder import get_dataset_builder

    store = get_model_store()
    builder = get_dataset_builder()
    engine = get_validation_engine()
    reports: List[Dict[str, Any]] = []

    # Load dataset
    dataset = builder.build_from_store(pair=pair, timeframe=timeframe, min_samples=None)
    if dataset is None:
        log.warning(f"[ValidationReport] no dataset for {pair} {timeframe}")
        return reports

    # Round-22 audit fix: use ONLY the test set for validation.
    # Previously: pd.concat([X_train, X_val, X_test]) — this recombined
    # training data into the validation set, so Monte Carlo simulation
    # and accuracy metrics were computed on data the model had ALREADY
    # SEEN during training. This made overfit models look good and
    # could APPROVE models that would fail in live trading.
    #
    # Now: use only dataset.X_test / dataset.y_test — data the model
    # has NEVER seen. This is the only honest way to validate.
    X = dataset.X_test
    y = dataset.y_test

    if len(X) == 0:
        log.warning(f"[ValidationReport] empty test set for {pair} {timeframe} — cannot validate")
        return reports

    # Validate each model type
    for model_type in ("xgboost", "random_forest", "lstm"):
        model = store.load_model(pair, timeframe, model_type)
        if model is None:
            continue

        log.info(f"[ValidationReport] validating {model_type} for {pair} {timeframe} on {len(X)} test samples")
        report = engine.validate(
            model=model, X=X, y=y,
            model_name=model_type, version="latest",
        )
        reports.append(report.to_dict())

    return reports


def get_validation_status() -> Dict[str, Any]:
    """Get overall validation status for dashboard."""
    engine = get_validation_engine()
    stats = engine.stats()
    champion = engine.get_champion()
    return {
        "stats": stats,
        "champion": champion,
        "approval_threshold": 75,
        "last_updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
