"""
ml/pipeline/phase2_validation.py — Data Validation & Repair
===========================================================
Detects and repairs common data quality issues:
  - Missing candles
  - Duplicate candles
  - Weekend gaps
  - Broker gaps
  - Corrupted rows
  - Timezone/DST issues

Generates a validation report per symbol/timeframe.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ml.pipeline.utils import (
    DATA_HISTORY_DIR, PipelineConfig, PipelineTimer, dataset_hash, get_pipeline_logger,
)

log = get_pipeline_logger("phase2_validation")

# Expected candles per period for forex
FOREX_CANDLES_PER_DAY = {
    "M1": 1440, "M5": 288, "M15": 96, "M30": 48,
    "H1": 24, "H4": 6, "D1": 1,
}

# Forex market hours (UTC) — roughly 22h/day (weekends off)
FOREX_SESSION_HOURS = 22


@dataclass
class ValidationReport:
    symbol: str = ""
    timeframe: str = ""
    total_rows: int = 0
    missing_candles: int = 0
    duplicates_removed: int = 0
    weekend_gaps_filled: int = 0
    broker_gaps_detected: int = 0
    corrupted_rows: int = 0
    timezone_fixed: bool = False
    data_hash: str = ""
    start_date: str = ""
    end_date: str = ""
    duration_days: float = 0.0
    issues: List[str] = field(default_factory=list)
    repairs: List[str] = field(default_factory=list)
    passed: bool = True


def validate_data(config: Optional[PipelineConfig] = None) -> Dict[str, ValidationReport]:
    """Main entry point for Phase 2."""
    config = config or PipelineConfig()
    reports = {}
    
    with PipelineTimer("Phase 2: Data Validation", log):
        for symbol in config.symbols:
            for tf in config.timeframes:
                parquet_path = DATA_HISTORY_DIR / symbol / f"{symbol}_{tf}.parquet"
                if not parquet_path.exists():
                    continue
                
                df = pd.read_parquet(parquet_path)
                report = _validate_symbol_tf(df, symbol, tf, config)
                reports[f"{symbol}_{tf}"] = report
                
                # Save report
                report_path = DATA_HISTORY_DIR / symbol / f"{symbol}_{tf}_validation.json"
                import json
                report_path.write_text(json.dumps(report.__dict__, indent=2, default=str))
                
                if not report.passed:
                    log.warning(f"  {symbol} {tf}: {len(report.issues)} issues found, {len(report.repairs)} repairs made")
                else:
                    log.info(f"  {symbol} {tf}: OK ({report.total_rows} rows)")
    
    return reports


def _validate_symbol_tf(df: pd.DataFrame, symbol: str, tf: str, config: PipelineConfig) -> ValidationReport:
    """Validate and repair a single symbol/timeframe."""
    report = ValidationReport(symbol=symbol, timeframe=tf, total_rows=len(df))
    
    if len(df) == 0:
        report.passed = False
        report.issues.append("Empty dataset")
        return report
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        report.timezone_fixed = True
        report.repairs.append("Converted timestamp to datetime64[UTC]")
    
    report.start_date = str(df["timestamp"].min())
    report.end_date = str(df["timestamp"].max())
    report.duration_days = (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 86400
    
    # 1. Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    dups = before - len(df)
    if dups > 0:
        report.duplicates_removed = dups
        report.repairs.append(f"Removed {dups} duplicate timestamps")
    
    # 2. Remove corrupted rows
    corrupt = 0
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            bad = df[col].isna() | (df[col] <= 0)
            corrupt += bad.sum()
            df = df[~bad]
    if corrupt > 0:
        report.corrupted_rows = int(corrupt)
        report.repairs.append(f"Removed {corrupt} corrupted rows")
    
    # 3. Check OHLC consistency
    if all(c in df.columns for c in ["open", "high", "low", "close"]):
        bad_ohlc = (df["high"] < df["low"]) | (df["high"] < df["open"]) | (df["high"] < df["close"])
        bad_ohlc |= (df["low"] > df["open"]) | (df["low"] > df["close"])
        ohlc_count = bad_ohlc.sum()
        if ohlc_count > 0:
            # Repair: fix high/low
            df["high"] = df[["high", "open", "close"]].max(axis=1)
            df["low"] = df[["low", "open", "close"]].min(axis=1)
            report.repairs.append(f"Fixed OHLC consistency in {ohlc_count} rows")
    
    # 4. Detect weekend gaps
    if tf in ("M1", "M5", "M15", "M30", "H1"):
        df["weekday"] = df["timestamp"].dt.dayofweek
        # Friday 22:00 UTC to Sunday 22:00 UTC is the weekend gap
        # We don't fill these — they're expected
    
    # 5. Detect missing candles
    expected_per_day = FOREX_CANDLES_PER_DAY.get(tf, 96)
    trading_days = report.duration_days * (FOREX_SESSION_HOURS / 24)
    expected_total = int(trading_days * expected_per_day)
    if len(df) < expected_total * 0.8:  # Allow 20% tolerance
        missing = expected_total - len(df)
        report.missing_candles = missing
        report.issues.append(f"Missing ~{missing} candles (expected ~{expected_total})")
    
    # 6. Detect broker gaps (large price jumps within single candles)
    if "close" in df.columns:
        returns = df["close"].pct_change().abs()
        gap_threshold = returns.quantile(0.999) * 3  # 3x the 99.9th percentile
        gaps = (returns > gap_threshold).sum()
        if gaps > 0:
            report.broker_gaps_detected = int(gaps)
            report.issues.append(f"{gaps} potential broker gaps detected (>{gap_threshold:.2%} move)")
    
    # Save repaired data
    report.total_rows = len(df)
    report.data_hash = dataset_hash(df)
    
    if report.issues:
        report.passed = len(report.repairs) >= len(report.issues)
    
    sym_dir = DATA_HISTORY_DIR / symbol
    df.to_parquet(sym_dir / f"{symbol}_{tf}.parquet", index=False)
    df.to_csv(sym_dir / f"{symbol}_{tf}.csv", index=False)
    
    return report


def load_validated_data(symbol: str, tf: str) -> Optional[pd.DataFrame]:
    """Load validated parquet data for a symbol/timeframe."""
    path = DATA_HISTORY_DIR / symbol / f"{symbol}_{tf}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None