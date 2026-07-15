"""
ml/pipeline/utils.py — Shared pipeline utilities
"""
from __future__ import annotations
import json, hashlib, logging, os, time, sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Paths ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_HISTORY_DIR = PROJECT_ROOT / "data" / "history"
PIPELINE_CACHE_DIR = PROJECT_ROOT / "data" / "pipeline_cache"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "data" / "trained_models"
LOG_DIR = PROJECT_ROOT / "logs" / "pipeline"

for d in (DATA_HISTORY_DIR, PIPELINE_CACHE_DIR, MODEL_OUTPUT_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────
@dataclass
class PipelineConfig:
    """Central configuration for the entire pipeline."""
    # Symbols
    symbols: List[str] = field(default_factory=lambda: ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "XAUUSD"])
    
    # Timeframes for data collection
    timeframes: List[str] = field(default_factory=lambda: ["M1", "M5", "M15", "M30", "H1", "H4", "D1"])
    
    # Primary timeframe for feature engineering
    primary_timeframe: str = "M15"
    
    # Data collection
    history_years: int = 5
    data_format: str = "parquet"  # parquet or csv
    
    # Train/val/test split (chronological)
    train_pct: float = 0.70
    val_pct: float = 0.15
    # test_pct = 1.0 - train_pct - val_pct = 0.15
    
    # Feature engineering
    feature_sets: List[str] = field(default_factory=lambda: [
        "trend", "momentum", "volume", "volatility",
        "market_structure", "session", "time"
    ])
    
    # Label horizons (in candles)
    label_horizons: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    
    # Training
    supervised_models: List[str] = field(default_factory=lambda: [
        "xgboost", "lightgbm", "catboost", "random_forest"
    ])
    rl_algorithms: List[str] = field(default_factory=lambda: ["ppo", "a2c"])
    
    # Training params
    optuna_trials: int = 50
    optuna_early_stopping: int = 10
    walk_forward_folds: int = 5
    walk_forward_train_pct: float = 0.7
    walk_forward_min_test_size: int = 50
    
    # RL params
    rl_timesteps: int = 500_000
    rl_eval_episodes: int = 20
    
    # Backtesting
    initial_balance: float = 10000.0
    risk_per_trade: float = 0.01
    max_spread_pips: float = 5.0
    slippage_pips: float = 0.5
    
    # Stress testing
    monte_carlo_sims: int = 1000
    
    # Production
    num_workers: int = 4
    use_gpu: bool = True
    cache_datasets: bool = True
    resume_from_phase: int = 1
    
    # Logging
    log_level: str = "INFO"
    tensorboard_dir: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
    
    @classmethod
    def load(cls, path: Path) -> "PipelineConfig":
        return cls(**json.loads(path.read_text()))


class PipelineTimer:
    """Context manager for timing pipeline phases."""
    def __init__(self, phase_name: str, logger: logging.Logger):
        self.phase_name = phase_name
        self.logger = logger
        self.start: float = 0
        self.elapsed: float = 0
    
    def __enter__(self):
        self.start = time.perf_counter()
        self.logger.info(f"[{self.phase_name}] Starting...")
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        mins, secs = divmod(self.elapsed, 60)
        self.logger.info(f"[{self.phase_name}] Completed in {int(mins)}m {secs:.1f}s")


def get_pipeline_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Get a pipeline logger with both console and file output."""
    logger = logging.getLogger(f"pipeline.{name}")
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        # Console
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(ch)
        # File
        if log_file is None:
            log_file = str(LOG_DIR / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(fh)
    return logger


def dataset_hash(df) -> str:
    """Create a hash of a DataFrame for change detection."""
    import pandas as pd
    buf = str(df.shape).encode()
    buf += str(df.columns.tolist()).encode()
    if len(df) > 0:
        buf += str(df.iloc[0].values.tobytes()[:1000]).encode()
        buf += str(df.iloc[-1].values.tobytes()[:1000]).encode()
    return hashlib.md5(buf).hexdigest()[:12]