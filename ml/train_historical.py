"""
ml/train_historical.py — Institutional Historical Training Pipeline
===================================================================
Fully automated 18-phase training pipeline.

Usage:
    python -m ml.train_historical
    python -m ml.train_historical --symbols EURUSD,GBPUSD --timeframes M15,H1
    python -m ml.train_historical --skip-rl --no-optuna
    python -m ml.train_historical --resume 8  # Resume from phase 8
    python -m ml.train_historical --fresh      # Force restart (ignore cache)
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Optional

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.pipeline.utils import (
    DATA_HISTORY_DIR, PIPELINE_CACHE_DIR, MODEL_OUTPUT_DIR,
    PipelineConfig, PipelineTimer, get_pipeline_logger,
)

log = get_pipeline_logger("train_historical")

# Phase imports
from ml.pipeline.phase1_data_collection import collect_data
from ml.pipeline.phase2_validation import validate_data
from ml.pipeline.phase3_features import compute_features
from ml.pipeline.phase4_labels import generate_labels
from ml.pipeline.phase5_regime import detect_regimes
from ml.pipeline.phase6_dataset import create_datasets
from ml.pipeline.phase8_train import train_all_models
from ml.pipeline.phase9_optuna import optimize_hyperparams
from ml.pipeline.phase10_walkforward import walk_forward_validation
from ml.pipeline.phase11_backtest import run_backtests
from ml.pipeline.phase12_stress import run_stress_tests
from ml.pipeline.phase13_selection import select_best_models
from ml.pipeline.phase14_save import save_production_artifacts
from ml.pipeline.phase15_retrain import check_and_retrain
from ml.pipeline.phase16_logging import start_run_log, log_phase, get_system_stats, save_run_log
from ml.pipeline.phase17_dashboard import generate_dashboard


# ── Checkpoint helpers ──────────────────────────────────────────

def _phase_checkpoint_path(phase_num: int) -> Path:
    """Each phase writes a tiny checkpoint file when it completes."""
    return PIPELINE_CACHE_DIR / f"_phase_{phase_num}_done.marker"


def _is_phase_done(phase_num: int) -> bool:
    return _phase_checkpoint_path(phase_num).exists()


def _mark_phase_done(phase_num: int) -> None:
    PIPELINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _phase_checkpoint_path(phase_num).write_text(
        f"completed at {time.time()}\n"
    )


def _has_raw_data(config: PipelineConfig) -> bool:
    """Check if Phase 1 data exists on disk (for resume validation)."""
    for symbol in config.symbols:
        tf = config.primary_timeframe
        p = DATA_HISTORY_DIR / symbol / f"{symbol}_{tf}.parquet"
        if p.exists():
            try:
                import pandas as pd
                df = pd.read_parquet(p)
                if len(df) >= 500:
                    return True
            except Exception:
                pass
    return False


def _has_featured_data(config: PipelineConfig) -> bool:
    """Check if Phase 3 feature cache exists on disk."""
    for symbol in config.symbols:
        tf = config.primary_timeframe
        p = PIPELINE_CACHE_DIR / f"{symbol}_{tf}_features.parquet"
        if p.exists():
            try:
                import pandas as pd
                df = pd.read_parquet(p)
                if len(df) >= 500:
                    return True
            except Exception:
                pass
    return False


def _clear_checkpoints(phase_num: Optional[int] = None) -> None:
    """Remove checkpoint markers (for --fresh or selective resume)."""
    if phase_num is not None:
        p = _phase_checkpoint_path(phase_num)
        if p.exists():
            p.unlink()
    else:
        for p in PIPELINE_CACHE_DIR.glob("_phase_*_done.marker"):
            p.unlink()


def run_pipeline(config: Optional[PipelineConfig] = None, args=None) -> Dict:
    """Execute the full 18-phase pipeline."""
    config = config or PipelineConfig()
    
    if args:
        if args.symbols:
            config.symbols = [s.strip() for s in args.symbols.split(",")]
        if args.timeframes:
            config.timeframes = [t.strip() for t in args.timeframes.split(",")]
        if args.resume:
            config.resume_from_phase = args.resume
        if args.timesteps:
            config.rl_timesteps = args.timesteps
        if args.trials:
            config.optuna_trials = args.trials
        if args.no_optuna:
            config.optuna_trials = 0
        if args.skip_rl:
            config.rl_algorithms = []
        if args.fresh:
            _clear_checkpoints()
            # Also clear feature cache to force recompute
            for p in PIPELINE_CACHE_DIR.glob("*_features.parquet"):
                p.unlink()
            log.info("Fresh start: cleared all checkpoints and feature cache")
    
    total_start = time.time()
    log.info("=" * 60)
    log.info("  INSTITUTIONAL HISTORICAL TRAINING PIPELINE")
    log.info(f"  Symbols: {config.symbols}")
    log.info(f"  Timeframes: {config.timeframes}")
    log.info(f"  Primary TF: {config.primary_timeframe}")
    log.info(f"  Supervised models: {config.supervised_models}")
    log.info(f"  RL algorithms: {config.rl_algorithms}")
    log.info(f"  Resume from phase: {config.resume_from_phase}")
    log.info("=" * 60)
    
    start_run_log(config)
    
    # Initialize results containers for cross-phase data flow
    data_result: Dict = {}
    validation_reports: Dict = {}
    featured_data: Dict = {}
    datasets: Dict = {}
    training_results: Dict = {}
    wf_results: Dict = {}
    backtest_results: Dict = {}
    best_models: Dict = {}
    pipeline_failed = False
    
    def _should_run(phase_num: int) -> bool:
        """Determine if a phase should execute (considering resume + checkpoints)."""
        if config.resume_from_phase > phase_num:
            # User said skip — but VERIFY data exists
            if phase_num == 1 and not _has_raw_data(config):
                log.warning(f"Phase {phase_num}: resume requested but NO raw data found on disk -- forcing re-run")
                config.resume_from_phase = phase_num  # Reset resume to this phase
                return True
            if phase_num == 3 and not _has_featured_data(config):
                log.warning(f"Phase {phase_num}: resume requested but NO featured data found -- forcing re-run")
                config.resume_from_phase = phase_num
                return True
            return False
        # Phase should run — check if already done via checkpoint
        if _is_phase_done(phase_num) and config.cache_datasets:
            return False
        return True
    
    # ── Phase 1: Data Collection ───────────────────────────────
    if _should_run(1):
        try:
            data_result = collect_data(config)
            if not data_result:
                log.error("Phase 1: collected ZERO symbols — CANNOT continue pipeline")
                log_phase("phase1_data_collection", "FAILED", error="No data collected")
                pipeline_failed = True
            else:
                log_phase("phase1_data_collection", "OK", {"symbols_collected": len(data_result)})
                _mark_phase_done(1)
        except Exception as e:
            log.error(f"Phase 1 FATAL: {e}")
            log_phase("phase1_data_collection", "FAILED", error=str(e))
            pipeline_failed = True
    else:
        log.info("[SKIP] Phase 1: Data Collection (resume or checkpoint)")
    
    # ── Phase 2: Data Validation ──────────────────────────────
    if not pipeline_failed and _should_run(2):
        try:
            validation_reports = validate_data(config)
            log_phase("phase2_validation", "OK", {"symbols_validated": len(validation_reports)})
            _mark_phase_done(2)
        except Exception as e:
            log.error(f"Phase 2 failed: {e}")
            log_phase("phase2_validation", "FAILED", error=str(e))
    elif not pipeline_failed:
        log.info("[SKIP] Phase 2: Data Validation (resume or checkpoint)")
    
    # ── Phase 3: Feature Engineering ──────────────────────────
    if not pipeline_failed and _should_run(3):
        try:
            featured_data = compute_features(config)
            if not featured_data:
                log.warning("Phase 3: no symbols produced features — data may be insufficient")
                log_phase("phase3_features", "WARNING", error="No features generated")
            else:
                log_phase("phase3_features", "OK", {"symbols_processed": len(featured_data)})
                _mark_phase_done(3)
        except Exception as e:
            log.error(f"Phase 3 failed: {e}")
            log_phase("phase3_features", "FAILED", error=str(e))
    
    # ── Phase 4: Label Generation ─────────────────────────────
    if not pipeline_failed and featured_data and _should_run(4):
        try:
            featured_data = generate_labels(featured_data, config)
            log_phase("phase4_labels", "OK")
            _mark_phase_done(4)
        except Exception as e:
            log.error(f"Phase 4 failed: {e}")
            log_phase("phase4_labels", "FAILED", error=str(e))
    
    # ── Phase 5: Regime Detection ─────────────────────────────
    if not pipeline_failed and featured_data and _should_run(5):
        try:
            featured_data = detect_regimes(featured_data, config)
            log_phase("phase5_regime", "OK")
            _mark_phase_done(5)
        except Exception as e:
            log.error(f"Phase 5 failed: {e}")
            log_phase("phase5_regime", "FAILED", error=str(e))
    
    # ── Phase 6: Dataset Creation ─────────────────────────────
    if not pipeline_failed and featured_data and _should_run(6):
        try:
            datasets = create_datasets(featured_data, config)
            if not datasets:
                log.warning("Phase 6: no datasets created (data may be too short)")
                log_phase("phase6_dataset", "WARNING", error="No datasets")
            else:
                log_phase("phase6_dataset", "OK", {"symbols_split": len(datasets)})
                _mark_phase_done(6)
        except Exception as e:
            log.error(f"Phase 6 failed: {e}")
            log_phase("phase6_dataset", "FAILED", error=str(e))
    elif not featured_data and config.resume_from_phase <= 6 and not pipeline_failed:
        log.warning("Phase 6: SKIPPED (no featured data from Phase 3)")
    
    # ── Phase 7: RL Environment ───────────────────────────────
    log_phase("phase7_rl_env", "OK")  # Env is used inside phase 8
    _mark_phase_done(7)
    
    # ── Phase 8: Model Training ───────────────────────────────
    if not pipeline_failed and datasets and _should_run(8):
        try:
            training_results = train_all_models(datasets, config)
            if training_results:
                log_phase("phase8_train", "OK", {"symbols_trained": len(training_results)})
                _mark_phase_done(8)
            else:
                log.error("Phase 8: no models trained")
                log_phase("phase8_train", "FAILED", error="No models trained")
        except Exception as e:
            log.error(f"Phase 8 failed: {e}")
            log_phase("phase8_train", "FAILED", error=str(e))
    elif not datasets and config.resume_from_phase <= 8 and not pipeline_failed:
        log.warning("Phase 8: SKIPPED (no datasets from Phase 6)")
    
    # ── Phase 9: Hyperparameter Optimization ──────────────────
    if not pipeline_failed and datasets and config.optuna_trials > 0 and _should_run(9):
        try:
            optuna_results = optimize_hyperparams(datasets, config)
            log_phase("phase9_optuna", "OK", {"symbols_optimized": len(optuna_results)})
            _mark_phase_done(9)
        except Exception as e:
            log.warning(f"Phase 9 skipped: {e}")
            log_phase("phase9_optuna", "SKIPPED", error=str(e))
    else:
        if config.optuna_trials > 0:
            log.info("[SKIP] Phase 9: Hyperparameter Optimization (--no-optuna or no data)")
    
    # ── Phase 10: Walk-Forward Validation ─────────────────────
    if not pipeline_failed and datasets and _should_run(10):
        try:
            wf_results = walk_forward_validation(datasets, config)
            log_phase("phase10_walkforward", "OK", {"symbols_validated": len(wf_results)})
            _mark_phase_done(10)
        except Exception as e:
            log.error(f"Phase 10 failed: {e}")
            log_phase("phase10_walkforward", "FAILED", error=str(e))
    
    # ── Phase 11: Backtesting ─────────────────────────────────
    if not pipeline_failed and datasets and _should_run(11):
        try:
            backtest_results = run_backtests(datasets, config)
            log_phase("phase11_backtest", "OK", {"symbols_backtested": len(backtest_results)})
            _mark_phase_done(11)
        except Exception as e:
            log.error(f"Phase 11 failed: {e}")
            log_phase("phase11_backtest", "FAILED", error=str(e))
    
    # ── Phase 12: Stress Testing ──────────────────────────────
    if not pipeline_failed and datasets and _should_run(12):
        try:
            stress_results = run_stress_tests(datasets, config)
            log_phase("phase12_stress", "OK")
            _mark_phase_done(12)
        except Exception as e:
            log.warning(f"Phase 12 skipped: {e}")
            log_phase("phase12_stress", "SKIPPED", error=str(e))
    
    # ── Phase 13: Model Selection ─────────────────────────────
    if not pipeline_failed and _should_run(13):
        try:
            best_models = select_best_models(backtest_results, wf_results, config)
            if best_models:
                log_phase("phase13_selection", "OK", {"best_models": list(best_models.keys())})
                _mark_phase_done(13)
            else:
                log.warning("Phase 13: no best models selected (no backtest results)")
                log_phase("phase13_selection", "WARNING", error="No models to select from")
        except Exception as e:
            log.error(f"Phase 13 failed: {e}")
            log_phase("phase13_selection", "FAILED", error=str(e))
    
    # ── Phase 14: Save Production Artifacts ───────────────────
    if not pipeline_failed and datasets and best_models and _should_run(14):
        try:
            artifacts = save_production_artifacts(datasets, best_models, config)
            log_phase("phase14_save", "OK")
            _mark_phase_done(14)
        except Exception as e:
            log.error(f"Phase 14 failed: {e}")
            log_phase("phase14_save", "FAILED", error=str(e))
    
    # ── Phase 15: Auto Retrain Check ──────────────────────────
    if not pipeline_failed and datasets and best_models and _should_run(15):
        try:
            retrain_results = check_and_retrain(datasets, best_models, config)
            log_phase("phase15_retrain", "OK")
            _mark_phase_done(15)
        except Exception as e:
            log_phase("phase15_retrain", "SKIPPED", error=str(e))
    
    # ── Phase 16: Logging ────────────────────────────────────
    log_phase("phase16_logging", "OK", get_system_stats())
    
    # ── Phase 17: Dashboard ───────────────────────────────────
    if _should_run(17):
        try:
            dashboard_path = generate_dashboard(datasets, best_models, backtest_results, config)
            log_phase("phase17_dashboard", "OK", {"path": dashboard_path})
        except Exception as e:
            log.warning(f"Phase 17 skipped: {e}")
    
    # ── Phase 18: Complete ────────────────────────────────────
    total_elapsed = time.time() - total_start
    mins, secs = divmod(total_elapsed, 60)
    
    log.info("=" * 60)
    if pipeline_failed:
        log.error(f"  PIPELINE FAILED at Phase 1 (no data) — {int(mins)}m {secs:.0f}s")
        log.error("  Fix MT5 connection, then re-run: python -m ml.train_historical")
    else:
        log.info(f"  PIPELINE COMPLETE — {int(mins)}m {secs:.0f}s")
        if best_models:
            for symbol, info in best_models.items():
                log.info(f"  {symbol}: {info['best_model']} (score={info['score']:.1f})")
        elif datasets:
            log.warning("  Pipeline finished but no best models selected — check phase 8/11 logs")
        else:
            log.warning("  Pipeline finished but produced no outputs — check phase 1-3 logs")
    log.info("=" * 60)
    
    # Save run log
    log_path = save_run_log()
    
    return {
        "best_models": best_models,
        "pipeline_failed": pipeline_failed,
        "total_seconds": round(total_elapsed, 1),
        "log_path": str(log_path),
    }


# ── CLI ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Institutional Forex AI Training Pipeline")
    parser.add_argument("--symbols", help="Comma-separated symbols (e.g. EURUSD,GBPUSD)")
    parser.add_argument("--timeframes", help="Comma-separated timeframes (e.g. M15,H1,D1)")
    parser.add_argument("--resume", type=int, default=1,
                        help="Resume from phase N (1-17, default=1 for full run)")
    parser.add_argument("--fresh", action="store_true",
                        help="Force restart: clear all checkpoints and caches")
    parser.add_argument("--timesteps", type=int, help="RL training timesteps (default: 500000)")
    parser.add_argument("--trials", type=int, help="Optuna optimization trials (default: 50)")
    parser.add_argument("--skip-rl", action="store_true", help="Skip RL training (PPO, A2C)")
    parser.add_argument("--no-optuna", action="store_true", help="Skip hyperparameter optimization")
    
    args = parser.parse_args()
    run_pipeline(args=args)