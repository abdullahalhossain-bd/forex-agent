"""
ml/pipeline/phase1_data_collection.py — MT5 Historical Data Collection
=====================================================================
Auto-connects to MetaTrader 5, downloads multi-timeframe historical
candles for all configured symbols, and saves as parquet + csv.

 Handles:
  - Auto initialize / login / reconnect
  - Multi-timeframe download (M1, M5, M15, M30, H1, H4, D1)
  - 5-10 years of history
  - Pagination for large datasets (>100k candles)
  - Incremental updates (only download new candles)
  - Retry with backoff on transient MT5 errors
  - Graceful fallback for non-Windows (skip with warning)
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml.pipeline.utils import (
    DATA_HISTORY_DIR, PipelineConfig, PipelineTimer, get_pipeline_logger,
)

log = get_pipeline_logger("phase1_data")

# ── MT5 Timeframe Constants ────────────────────────────────────
# These are the EXACT integer values that mt5.copy_rates_from_pos()
# expects. Using strings (e.g. "1M") causes a C-level exception:
#   "returned a result with an exception set"
#
# Source: https://www.mql5.com/en/docs/python_mt5/mt5copyratesfrompos
MT5_TF_CONSTANTS = {
    "M1":  1,       # PERIOD_M1
    "M5":  5,       # PERIOD_M5
    "M15": 15,      # PERIOD_M15
    "M30": 30,      # PERIOD_M30
    "H1":  16385,   # PERIOD_H1
    "H4":  16388,   # PERIOD_H4
    "D1":  16408,   # PERIOD_D1
}

# Candles per day for each timeframe (forex ~22h/day, 5 days/week)
CANDLES_PER_DAY = {
    "M1": 1320,     # 22h * 60
    "M5": 264,      # 22h * 12
    "M15": 88,      # 22h * 4
    "M30": 44,      # 22h * 2
    "H1": 22,       # 22h
    "H4": 6,        # ~22h / 4
    "D1": 1,
}

# MT5 hard limit: copy_rates_from_pos returns max ~100k rows per call
MT5_MAX_REQUEST = 100_000


def collect_data(config: Optional[PipelineConfig] = None) -> Dict[str, Path]:
    """Main entry point for Phase 1. Returns dict of symbol -> data directory.
    
    Raises RuntimeError if MT5 init fails or no data collected at all.
    """
    config = config or PipelineConfig()
    
    with PipelineTimer("Phase 1: Data Collection", log):
        mt5_available = _check_mt5()
        
        if not mt5_available:
            log.warning("MT5 not available (non-Windows?) — skipping data collection")
            log.info("Place historical data manually in: %s", DATA_HISTORY_DIR)
            return {}
        
        if not _init_mt5():
            raise RuntimeError("MT5 initialization failed — cannot collect data")
        
        results = {}
        total_candles = 0
        errors = []
        
        for symbol in config.symbols:
            sym_dir = DATA_HISTORY_DIR / symbol
            sym_dir.mkdir(parents=True, exist_ok=True)
            symbol_ok = False
            
            for tf in config.timeframes:
                try:
                    df = _download_symbol_tf(symbol, tf, config)
                    if df is not None and len(df) > 0:
                        _save_candles(df, symbol, tf, config)
                        total_candles += len(df)
                        results[symbol] = sym_dir
                        symbol_ok = True
                except Exception as e:
                    err_msg = f"{symbol} {tf}: {e}"
                    log.error(f"  ERROR: {err_msg}")
                    errors.append(err_msg)
            
            if not symbol_ok:
                log.warning(f"  {symbol}: FAILED to download ANY timeframe")
        
        # Shut down our MT5 session (pipeline doesn't need persistent connection)
        try:
            import MetaTrader5 as mt5
            mt5.shutdown()
            log.info("MT5 connection closed")
        except Exception:
            pass
        
        # Summary
        n_symbols = len(results)
        log.info(f"  Collected data for {n_symbols}/{len(config.symbols)} symbols, "
                 f"{total_candles:,} total candles across {len(config.timeframes)} timeframes")
        
        if n_symbols == 0:
            if errors:
                raise RuntimeError(
                    f"Phase 1 collected ZERO symbols. Errors:\n  - " +
                    "\n  - ".join(errors) +
                    "\nCheck: 1) MT5 terminal is open, 2) Symbol names match your broker, "
                    "3) Account has market data access"
                )
            raise RuntimeError(
                "Phase 1 collected ZERO data. No symbols configured or all skipped."
            )
        
        return results


def _check_mt5() -> bool:
    try:
        import MetaTrader5 as mt5
        # Verify the module actually loaded (not just imported)
        _ = mt5.version()
        return True
    except (ImportError, Exception):
        return False


def _init_mt5() -> bool:
    """Initialize MT5 with auto-reconnect (3 attempts)."""
    import MetaTrader5 as mt5
    
    for attempt in range(3):
        if mt5.initialize():
            log.info("MT5 initialized successfully")
            account = mt5.account_info()
            if account:
                log.info("Account: %s | Balance: %.2f | Server: %s",
                         account.login, account.balance, account.server)
            return True
        
        err_code = mt5.last_error()
        log.warning("MT5 init attempt %d/3 failed (error=%s), retrying in 2s...",
                     attempt + 1, err_code)
        time.sleep(2)
    
    log.error("MT5 initialization failed after 3 attempts")
    return False


def _download_symbol_tf(symbol: str, tf: str, config: PipelineConfig) -> Optional[pd.DataFrame]:
    """Download historical candles for a symbol+timeframe with pagination and retry."""
    import MetaTrader5 as mt5
    
    # ── Resolve MT5 timeframe constant ──
    if tf not in MT5_TF_CONSTANTS:
        log.error(f"  Unknown timeframe '{tf}'. Valid: {list(MT5_TF_CONSTANTS.keys())}")
        return None
    
    mt5_tf = MT5_TF_CONSTANTS[tf]
    
    # ── Calculate total candles needed ──
    cpd = CANDLES_PER_DAY.get(tf, 96)
    trading_days = int(config.history_years * 252)  # 252 trading days/year
    total_wanted = trading_days * cpd
    
    log.info(f"  Downloading {symbol} {tf} (~{total_wanted:,} candles, {config.history_years}y)...")
    
    # ── Check for existing data (incremental update) ──
    existing_path = DATA_HISTORY_DIR / symbol / f"{symbol}_{tf}.parquet"
    existing_df = None
    existing_count = 0
    
    if existing_path.exists() and config.cache_datasets:
        try:
            existing_df = pd.read_parquet(existing_path)
            existing_count = len(existing_df)
            if existing_count > 0:
                log.info(f"  Existing data: {existing_count:,} rows, will append new candles")
        except Exception as e:
            log.warning(f"  Could not read existing data: {e}")
            existing_df = None
    
    # ── Strategy: try copy_rates_from_pos with adaptive chunk size ──
    # Some brokers/servers reject large requests for certain timeframes.
    # We start with the max (100k) and halve on "Invalid params" error.
    all_frames = []
    start_pos = 0
    max_empty = 3
    empty_count = 0
    max_retries = 3
    chunk_sizes_to_try = [100000, 50000, 20000, 10000]  # Adaptive fallback
    current_chunk_cap = chunk_sizes_to_try[0]
    
    while True:
        chunk_size = min(current_chunk_cap, total_wanted - start_pos)
        if chunk_size <= 0:
            break
        
        rates = None
        hit_invalid_params = False
        
        for retry in range(max_retries):
            rates = mt5.copy_rates_from_pos(symbol, mt5_tf, start_pos, chunk_size)
            
            if rates is not None and len(rates) > 0:
                break
            
            err = mt5.last_error()
            err_code = err[0] if isinstance(err, tuple) else getattr(err, 'code', None)
            
            # "Invalid params" (-2) often means chunk too large for this TF/server
            if err_code == -2:
                hit_invalid_params = True
                break  # Don't retry — try smaller chunk instead
            
            if retry < max_retries - 1:
                log.debug(f"  Retry {retry+1}/{max_retries} for {symbol} {tf} pos={start_pos} "
                          f"(MT5 error: {err})")
                time.sleep(0.5 * (retry + 1))
            else:
                log.warning(f"  {symbol} {tf}: failed at pos={start_pos} "
                            f"(MT5 error: {err}) after {max_retries} retries")
        
        # If we got "Invalid params", try a smaller chunk size
        if hit_invalid_params:
            next_cap = None
            for cs in chunk_sizes_to_try:
                if cs < current_chunk_cap:
                    next_cap = cs
                    break
            if next_cap is None:
                log.warning(f"  {symbol} {tf}: all chunk sizes failed (invalid params)")
                break
            log.info(f"  {symbol} {tf}: reducing chunk {current_chunk_cap} -> {next_cap} (server limit)")
            current_chunk_cap = next_cap
            time.sleep(0.3)
            continue  # Retry with smaller chunk at same start_pos
        
        if rates is None or len(rates) == 0:
            empty_count += 1
            if empty_count >= max_empty:
                break
            time.sleep(0.1)
            continue
        
        empty_count = 0
        df_chunk = pd.DataFrame(rates)
        all_frames.append(df_chunk)
        
        downloaded = sum(len(f) for f in all_frames)
        log.info(f"    Chunk: +{len(rates):,} candles (total: {downloaded:,})")
        
        if len(rates) < chunk_size:
            break
        
        start_pos += len(rates)
    
    if not all_frames:
        log.warning(f"  No data returned for {symbol} {tf}")
        return existing_df
    
    # ── Combine all chunks ──
    df = pd.concat(all_frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={"time": "timestamp", "tick_volume": "tick_vol", "real_volume": "real_vol"})
    
    # Drop columns that may not exist on all brokers
    for col in ["spread"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    
    # ── Merge with existing ──
    if existing_df is not None and len(existing_df) > 0:
        before = len(df)
        df = pd.concat([existing_df, df], ignore_index=True)
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
        added = len(df) - existing_count
        log.info(f"  Merged: {existing_count:,} existing + {added:,} new = {len(df):,} total")
    else:
        log.info(f"  {symbol} {tf}: {len(df):,} candles "
                 f"({df['timestamp'].min()} -> {df['timestamp'].max()})")
    
    return df


def _save_candles(df: pd.DataFrame, symbol: str, tf: str, config: PipelineConfig) -> None:
    """Save candles as parquet (primary) and csv."""
    sym_dir = DATA_HISTORY_DIR / symbol
    sym_dir.mkdir(parents=True, exist_ok=True)
    
    # Parquet (primary)
    parquet_path = sym_dir / f"{symbol}_{tf}.parquet"
    df.to_parquet(parquet_path, index=False, engine="pyarrow")
    size_kb = parquet_path.stat().st_size / 1024
    log.info(f"  Saved {parquet_path.name} ({size_kb:.1f} KB, {len(df):,} rows)")
    
    # CSV (secondary) — skip for M1 (too large)
    if tf not in ("M1",):
        csv_path = sym_dir / f"{symbol}_{tf}.csv"
        df.to_csv(csv_path, index=False)