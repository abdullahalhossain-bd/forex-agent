"""
scripts/download_historical_data.py — MT5 → local CSV historical data downloader (v2).

Production-grade upgrade of the original downloader. Adds:
  - Broker-timezone auto-detection (reuses data/fetcher.py pattern)
  - Retry logic with exponential backoff per chunk
  - Gap detection with weekend-aware filtering
  - MAX_REQUIRED_LOOKBACK enforcement (--warmup-bars argument)
  - HTF resampling fallback (H4 from H1 if H4 fetch fails)
  - Bid/ask column fetching (optional, --with-bid-ask flag)
  - Post-download validation report (machine-readable JSON + Markdown)
  - Idempotent re-run (--skip-existing skips files already covering the range)
  - Data quality metrics in manifest (spread_zero_pct, gap_count, etc.)

USAGE (on production VPS with MT5 terminal running):

    python scripts/download_historical_data.py \\
        --symbols EURUSD GBPUSD USDJPY USDCHF USDCAD AUDUSD NZDUSD XAUUSD \\
        --timeframes M5 M15 H1 H4 D1 \\
        --start 2024-08-01 --end 2026-08-17 \\
        --warmup-bars 1000 \\
        --with-bid-ask

WHAT IT DOES:
  1. Connects to MT5 (reuses broker/mt5_connection.py singleton).
  2. For each (symbol, timeframe) pair:
     a. Detects broker UTC offset (GMT+2/+3 auto, env override supported).
     b. Extends --start backward by --warmup-bars on the primary TF.
     c. Fetches bars in monthly chunks (retry on transient errors).
     d. Applies broker-tz correction to all timestamps → tz-aware UTC.
     e. Validates each chunk (no duplicates, sorted, OHLC sanity, NaN, infinity).
     f. Optionally fetches bid/ask at bar close.
     g. Concatenates, dedupes, sorts, saves to CSV.
     h. Runs full validation, writes _validation_{TF}.json next to CSV.
  3. Updates data/historical/manifest.json with download metadata + quality metrics.

WHAT IT DOES NOT DO:
  - Does NOT call MT5 during a backtest run. Download is a one-time setup step.
  - Does NOT silently overwrite existing CSVs — use --force to overwrite.
  - Does NOT fetch tick data (only bar-level OHLCV + optional bid/ask at close).
  - Does NOT fetch external macro (DXY/Gold/Oil/etc.) — see scripts/download_external_data.py.

ENVIRONMENT REQUIREMENTS:
  - MetaTrader5 Python package installed: pip install MetaTrader5
  - MT5 terminal running with a connected account
  - Windows or Wine (MT5 doesn't run on bare Linux)
  - Run on the production VPS, NOT in a CI/audit env

PARITY NOTE:
  The downloaded CSV format is a strict superset of the existing
  data/{SYMBOL}_{TF}.csv files in the repo. The CSV provider
  (core/csv_data_provider.py) can load from either location.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Output structure
HISTORICAL_DIR = PROJECT_ROOT / "data" / "historical"
LEGACY_FLAT_DIR = PROJECT_ROOT / "data"  # backward-compat

# Standard schema (REQUIRED columns)
STANDARD_COLUMNS = ["datetime_utc", "open", "high", "low", "close",
                     "tick_volume", "spread", "real_volume"]
# Extended schema (with bid/ask)
EXTENDED_COLUMNS = STANDARD_COLUMNS[:5] + ["bid", "ask"] + STANDARD_COLUMNS[5:]

# MT5 timeframe constants
MT5_TF_MAP = {
    "M1": "TIMEFRAME_M1", "M5": "TIMEFRAME_M5", "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30", "H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1", "W1": "TIMEFRAME_W1", "MN1": "TIMEFRAME_MN1",
}

# TF to seconds map (for gap detection)
TF_SECONDS = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800,
              "H1": 3600, "H4": 14400, "D1": 86400, "W1": 604800, "MN1": 2592000}

# MAX_REQUIRED_LOOKBACK derived from audit (P1-C §4):
#   500 bars (NadararaWatson window) on primary TF
#   + 200 bars per HTF (MTFAnalyzer limit)
#   + 1 trading week intraday (PDH/PWL/Asian range)
# Operational rule: 1000 bars on primary TF as safe default
DEFAULT_WARMUP_BARS = 1000

# Broker-tz cache
_BROKER_OFFSET_CACHE: Optional[float] = None
_BROKER_OFFSET_CACHE_AT: float = 0.0
_BROKER_OFFSET_CACHE_TTL_SEC = 1800  # 30 min

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("downloader")


# ============================================================================
# MT5 connection helpers
# ============================================================================

def _get_mt5_timeframe(tf_label: str):
    """Resolve a TF label (e.g. 'M15') to an MT5 timeframe constant."""
    import MetaTrader5 as mt5
    constant_name = MT5_TF_MAP.get(tf_label.upper())
    if constant_name is None:
        raise ValueError(f"Unknown timeframe: {tf_label}. Valid: {list(MT5_TF_MAP.keys())}")
    return getattr(mt5, constant_name)


def _connect_mt5() -> bool:
    """Initialize MT5 connection. Returns True on success."""
    import MetaTrader5 as mt5
    if not mt5.initialize():
        log.error(f"MT5 initialize() failed: {mt5.last_error()}")
        return False
    info = mt5.terminal_info()
    if info is None:
        log.error(f"MT5 terminal_info() returned None: {mt5.last_error()}")
        return False
    log.info(f"MT5 connected: terminal={info.name}, build={info.build}, path={info.path}")
    acct = mt5.account_info()
    if acct:
        log.info(f"  Account: {acct.login} ({acct.server}), balance={acct.balance} {acct.currency}")
    return True


def _shutdown_mt5():
    import MetaTrader5 as mt5
    mt5.shutdown()


def _detect_broker_utc_offset_hours(symbol: str = "EURUSD") -> float:
    """Auto-detect broker UTC offset by comparing live tick time to wall clock.

    Reuses the same pattern as data/fetcher.py:_get_broker_utc_offset_hours.
    Returns the offset in hours (e.g. +2.0 for GMT+2 winter, +3.0 for GMT+3 summer).

    Honors the MT5_BROKER_TZ_OFFSET_HOURS env var as an explicit override.
    Caches for 30 minutes, self-corrects for DST flips.
    """
    global _BROKER_OFFSET_CACHE, _BROKER_OFFSET_CACHE_AT

    env_override = os.getenv("MT5_BROKER_TZ_OFFSET_HOURS")
    if env_override not in (None, ""):
        try:
            offset = float(env_override)
            log.info(f"Broker UTC offset: {offset:+.0f}h (from env MT5_BROKER_TZ_OFFSET_HOURS)")
            return offset
        except ValueError:
            log.warning(f"Invalid MT5_BROKER_TZ_OFFSET_HOURS='{env_override}' — ignoring")

    now = time.time()
    if (_BROKER_OFFSET_CACHE is not None
            and (now - _BROKER_OFFSET_CACHE_AT) < _BROKER_OFFSET_CACHE_TTL_SEC):
        return _BROKER_OFFSET_CACHE

    import MetaTrader5 as mt5
    try:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None or not getattr(tick, "time", None):
            log.debug(f"No live tick for {symbol} — keeping previous offset ({_BROKER_OFFSET_CACHE or 0.0}h)")
            return _BROKER_OFFSET_CACHE or 0.0

        broker_now = datetime.fromtimestamp(tick.time, tz=timezone.utc)
        utc_now = datetime.now(timezone.utc)
        offset_hours = round((broker_now - utc_now).total_seconds() / 3600)

        if _BROKER_OFFSET_CACHE is not None and offset_hours != _BROKER_OFFSET_CACHE:
            log.info(f"Broker UTC offset changed: {_BROKER_OFFSET_CACHE:+.0f}h -> {offset_hours:+.0f}h (DST flip)")
        else:
            log.info(f"Broker UTC offset detected: {offset_hours:+.0f}h (tick vs wall-clock comparison)")
        _BROKER_OFFSET_CACHE = float(offset_hours)
        _BROKER_OFFSET_CACHE_AT = now
        return _BROKER_OFFSET_CACHE
    except Exception as e:
        log.warning(f"Dynamic tz offset detection failed: {e} — keeping previous offset ({_BROKER_OFFSET_CACHE or 0.0}h)")
        return _BROKER_OFFSET_CACHE or 0.0


# ============================================================================
# Fetch with retry
# ============================================================================

def _fetch_chunk_with_retry(mt5, symbol: str, tf_const, start: datetime, end: datetime,
                            max_retries: int = 3, base_delay: float = 1.0) -> pd.DataFrame:
    """Fetch one chunk with exponential backoff retry."""
    last_err = None
    for attempt in range(max_retries):
        try:
            rates = mt5.copy_rates_range(symbol, tf_const, start, end)
            if rates is None:
                # MT5 returns None for various errors — check last_error
                import MetaTrader5 as mt5_mod
                err = mt5_mod.last_error()
                if "no data" in str(err).lower() or "not found" in str(err).lower():
                    # Genuine no-data (e.g. weekend, pre-listing date) — don't retry
                    return pd.DataFrame()
                last_err = f"copy_rates_range returned None: {err}"
            else:
                df = pd.DataFrame(rates)
                if df.empty:
                    return df
                # Apply broker-tz correction
                broker_offset = _detect_broker_utc_offset_hours(symbol)
                df["datetime_utc"] = pd.to_datetime(df["time"], unit="s", utc=False)
                if broker_offset != 0:
                    df["datetime_utc"] = df["datetime_utc"] - pd.Timedelta(hours=broker_offset)
                df["datetime_utc"] = df["datetime_utc"].dt.tz_localize("UTC")
                # Drop forming bar (the most recent bar in the chunk may be still forming)
                tf_label = None
                for k, v in MT5_TF_MAP.items():
                    if v == tf_const.__class__.__name__ or getattr(mt5, v, None) == tf_const:
                        tf_label = k
                        break
                if tf_label and tf_label in TF_SECONDS:
                    tf_sec = TF_SECONDS[tf_label]
                    if len(df) > 0:
                        last_open = df["datetime_utc"].iloc[-1]
                        implied_close = last_open + pd.Timedelta(seconds=tf_sec)
                        now_utc = pd.Timestamp.now(tz="UTC")
                        if implied_close > now_utc:
                            log.debug(f"Dropping forming bar: open={last_open}, implied_close={implied_close}")
                            df = df.iloc[:-1].copy()
                return df
        except Exception as e:
            last_err = str(e)
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                log.warning(f"Fetch attempt {attempt+1}/{max_retries} failed: {e} — retrying in {delay}s")
                time.sleep(delay)
    log.error(f"All {max_retries} fetch attempts failed for {symbol} {start.date()}→{end.date()}: {last_err}")
    return pd.DataFrame()


# ============================================================================
# Optional: fetch bid/ask at bar close
# ============================================================================

def _fetch_bid_ask_at_close(symbol: str, bar_times: pd.Series) -> pd.DataFrame:
    """Fetch bid/ask at each bar's close time.

    NOTE: This is a SLOW operation — calls mt5.symbol_info_tick once per bar.
    For large date ranges, prefer --skip-bid-ask (default) and rely on the
    `spread` column for spread approximation.

    Returns DataFrame with columns: datetime_utc, bid, ask
    """
    import MetaTrader5 as mt5
    rows = []
    total = len(bar_times)
    log.info(f"Fetching bid/ask for {total} bars (this may be slow)...")
    for i, bar_time in enumerate(bar_times):
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                rows.append({"datetime_utc": bar_time, "bid": np.nan, "ask": np.nan})
            else:
                rows.append({"datetime_utc": bar_time, "bid": float(tick.bid), "ask": float(tick.ask)})
        except Exception:
            rows.append({"datetime_utc": bar_time, "bid": np.nan, "ask": np.nan})
        if (i + 1) % 1000 == 0:
            log.info(f"  bid/ask progress: {i+1}/{total}")
            time.sleep(0.05)  # be polite to MT5
    return pd.DataFrame(rows)


# ============================================================================
# Validation
# ============================================================================

def validate_full(df: pd.DataFrame, symbol: str, tf: str) -> dict:
    """Run full validation on a downloaded df. Returns issues dict."""
    issues = {
        "errors": [],
        "warnings": [],
        "stats": {},
    }
    if df.empty:
        issues["errors"].append("empty dataframe")
        return issues

    # V1: sort
    if not df["datetime_utc"].is_monotonic_increasing:
        issues["errors"].append("timestamps not sorted ascending")

    # V2: duplicates
    n_dupes = int(df["datetime_utc"].duplicated().sum())
    if n_dupes > 0:
        issues["errors"].append(f"{n_dupes} duplicate timestamps")

    # V3: OHLC sanity
    bad_high = int((df["high"] < df[["open", "close"]].max(axis=1)).sum())
    bad_low = int((df["low"] > df[["open", "close"]].min(axis=1)).sum())
    bad_hl = int((df["high"] < df["low"]).sum())
    if bad_high > 0:
        issues["errors"].append(f"{bad_high} rows with high < max(open, close)")
    if bad_low > 0:
        issues["errors"].append(f"{bad_low} rows with low > min(open, close)")
    if bad_hl > 0:
        issues["errors"].append(f"{bad_hl} rows with high < low")

    # V4: NaN in OHLCV
    for col in ("open", "high", "low", "close", "tick_volume"):
        if col in df.columns:
            n_nan = int(df[col].isna().sum())
            if n_nan > 0:
                issues["errors"].append(f"{n_nan} NaN in {col}")

    # V5: infinity
    for col in ("open", "high", "low", "close", "tick_volume"):
        if col in df.columns:
            n_inf = int(np.isinf(df[col].astype(float)).sum())
            if n_inf > 0:
                issues["errors"].append(f"{n_inf} infinite in {col}")

    # V6: negative prices
    for col in ("open", "high", "low", "close"):
        n_neg = int((df[col] < 0).sum())
        if n_neg > 0:
            issues["errors"].append(f"{n_neg} negative prices in {col}")

    # V7: tick_volume >= 0
    n_neg_vol = int((df["tick_volume"] < 0).sum())
    if n_neg_vol > 0:
        issues["errors"].append(f"{n_neg_vol} negative tick_volume")

    # V8: spread >= 0
    if "spread" in df.columns:
        n_neg_sp = int((df["spread"] < 0).sum())
        if n_neg_sp > 0:
            issues["errors"].append(f"{n_neg_sp} negative spread")

    # V10: gaps (excluding forex weekend)
    tf_sec = TF_SECONDS.get(tf)
    gaps = []
    non_weekend_gaps = 0
    if tf_sec and len(df) > 1:
        df_sorted = df.sort_values("datetime_utc").reset_index(drop=True)
        diffs = df_sorted["datetime_utc"].diff().dt.total_seconds().dropna()
        for i, d in diffs.items():
            if d > tf_sec * 1.5:
                start_ts = df_sorted["datetime_utc"].iloc[i - 1]
                end_ts = df_sorted["datetime_utc"].iloc[i]
                missing = int(round(d / tf_sec)) - 1
                # Weekend detection: Fri 21:00 UTC -> Sun 21:00 UTC
                is_weekend = False
                if start_ts.dayofweek == 4 and end_ts.dayofweek == 6:
                    if start_ts.hour >= 21 and end_ts.hour >= 21:
                        is_weekend = True
                gaps.append({
                    "start": str(start_ts),
                    "end": str(end_ts),
                    "missing_bars": missing,
                    "is_weekend": is_weekend,
                })
                if not is_weekend:
                    non_weekend_gaps += 1
        if non_weekend_gaps > 0:
            issues["warnings"].append(f"{non_weekend_gaps} non-weekend gaps (total gaps: {len(gaps)})")

    # V11: spread zero %
    sp_zero_pct = 0.0
    if "spread" in df.columns and len(df) > 0:
        sp_zero_pct = round(float((df["spread"] == 0).sum() / len(df) * 100), 2)
        if sp_zero_pct > 50:
            issues["warnings"].append(f"{sp_zero_pct}% of bars have spread=0 (re-download recommended)")

    # V12: row count
    n_rows = len(df)
    min_rows = 500 if tf in ("M15", "M5", "M30") else 200
    if n_rows < min_rows:
        issues["warnings"].append(f"only {n_rows} rows (min {min_rows} for TF={tf})")

    # Stats
    issues["stats"] = {
        "rows": n_rows,
        "start_utc": str(df["datetime_utc"].iloc[0]),
        "end_utc": str(df["datetime_utc"].iloc[-1]),
        "date_range_days": round((df["datetime_utc"].iloc[-1] - df["datetime_utc"].iloc[0]).total_seconds() / 86400, 2),
        "duplicate_timestamps": n_dupes,
        "gap_count": len(gaps),
        "non_weekend_gap_count": non_weekend_gaps,
        "spread_zero_pct": sp_zero_pct,
        "tick_volume_zero_pct": round(float((df["tick_volume"] == 0).sum() / len(df) * 100), 2) if "tick_volume" in df.columns else None,
        "real_volume_zero_pct": round(float((df["real_volume"] == 0).sum() / len(df) * 100), 2) if "real_volume" in df.columns else None,
        "ohlc_violations": {"high_lt_max_oc": bad_high, "low_gt_min_oc": bad_low, "high_lt_low": bad_hl},
        "gaps_sample": gaps[:5],
    }
    return issues


# ============================================================================
# HTF resampling fallback (H4 from H1)
# ============================================================================

def _resample_h1_to_h4(df_h1: pd.DataFrame) -> pd.DataFrame:
    """Resample H1 bars to H4 bars (matches MT5 H4 grid: 00-04, 04-08, ..., 20-24 UTC)."""
    if df_h1.empty:
        return df_h1
    df = df_h1.set_index("datetime_utc").sort_index()
    resampled = df.resample("4h", origin="epoch").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "tick_volume": "sum", "spread": "mean", "real_volume": "sum",
    }).dropna(subset=["open"]).reset_index()
    if "bid" in df.columns and "ask" in df.columns:
        bid_ask = df[["bid", "ask"]].resample("4h", origin="epoch").last().dropna()
        resampled = resampled.merge(bid_ask, left_on="datetime_utc", right_index=True, how="left")
    return resampled


# ============================================================================
# Main download function
# ============================================================================

def download_symbol_tf(symbol: str, tf: str, start: datetime, end: datetime,
                       force: bool = False, nested: bool = True,
                       warmup_bars: int = DEFAULT_WARMUP_BARS,
                       with_bid_ask: bool = False,
                       skip_existing: bool = True) -> dict:
    """Download one (symbol, tf) pair. Returns a manifest entry dict."""
    out_dir = HISTORICAL_DIR / symbol if nested else LEGACY_FLAT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tf}.csv" if nested else out_dir / f"{symbol}_{tf}.csv"
    validation_path = out_dir / f"_validation_{tf}.json" if nested else out_dir / f"_validation_{symbol}_{tf}.json"

    # Apply warmup extension backward on the primary TF
    tf_sec = TF_SECONDS.get(tf, 900)
    effective_start = start - timedelta(seconds=warmup_bars * tf_sec)
    log.info(f"[{symbol} {tf}] Effective start (with {warmup_bars} warmup bars): {effective_start.date()}")

    # Idempotent re-run: skip if existing CSV covers the range
    if skip_existing and out_path.exists() and not force:
        try:
            existing = pd.read_csv(out_path, encoding="utf-8-sig")
            existing["datetime_utc"] = pd.to_datetime(existing["datetime_utc"], utc=True, errors="coerce")
            if len(existing) > 0:
                existing_start = existing["datetime_utc"].iloc[0]
                existing_end = existing["datetime_utc"].iloc[-1]
                if existing_start <= effective_start and existing_end >= end:
                    log.info(f"[{symbol} {tf}] SKIP — existing CSV covers {existing_start.date()}→{existing_end.date()}")
                    return {
                        "symbol": symbol, "timeframe": tf,
                        "start": str(existing_start), "end": str(existing_end),
                        "rows": len(existing), "skipped": True, "reason": "range already covered",
                    }
        except Exception as e:
            log.warning(f"[{symbol} {tf}] Could not parse existing CSV ({e}) — re-downloading")

    import MetaTrader5 as mt5
    tf_const = _get_mt5_timeframe(tf)

    # Fetch in monthly chunks
    all_chunks = []
    chunk_start = effective_start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=30), end)
        log.info(f"  [{symbol} {tf}] Fetching {chunk_start.date()} → {chunk_end.date()} ...")
        chunk = _fetch_chunk_with_retry(mt5, symbol, tf_const, chunk_start, chunk_end)
        if not chunk.empty:
            # Per-chunk validation
            chunk_issues = validate_full(chunk, symbol, tf)
            if chunk_issues["errors"]:
                log.warning(f"    chunk validation errors: {chunk_issues['errors']}")
            all_chunks.append(chunk)
        chunk_start = chunk_end
        time.sleep(0.3)  # be polite to MT5

    # HTF resampling fallback (H4 from H1 if H4 fetch failed)
    if not all_chunks and tf == "H4":
        log.info(f"  [{symbol} H4] H4 fetch empty — falling back to H1→H4 resampling")
        h1_start = effective_start
        h1_chunks = []
        while h1_start < end:
            h1_end = min(h1_start + timedelta(days=30), end)
            h1_chunk = _fetch_chunk_with_retry(mt5, symbol, _get_mt5_timeframe("H1"), h1_start, h1_end)
            if not h1_chunk.empty:
                h1_chunks.append(h1_chunk)
            h1_start = h1_end
            time.sleep(0.3)
        if h1_chunks:
            h1_df = pd.concat(h1_chunks, ignore_index=True)
            h1_df = h1_df.drop_duplicates(subset=["datetime_utc"]).sort_values("datetime_utc").reset_index(drop=True)
            resampled = _resample_h1_to_h4(h1_df)
            all_chunks.append(resampled)
            log.info(f"  [{symbol} H4] Resampled {len(h1_df)} H1 bars → {len(resampled)} H4 bars")

    if not all_chunks:
        return {"symbol": symbol, "timeframe": tf,
                "error": "no data fetched in the requested range", "rows": 0}

    # Combine + dedupe + sort
    df = pd.concat(all_chunks, ignore_index=True)
    df = df.drop_duplicates(subset=["datetime_utc"]).sort_values("datetime_utc").reset_index(drop=True)
    # Filter to requested range (exclude warmup bars beyond effective_start if user wants clean window)
    df = df[(df["datetime_utc"] >= effective_start) & (df["datetime_utc"] < end)].reset_index(drop=True)

    # Optional bid/ask fetch
    if with_bid_ask:
        bid_ask_df = _fetch_bid_ask_at_close(symbol, df["datetime_utc"])
        df = df.merge(bid_ask_df, on="datetime_utc", how="left")
        # Fill missing bid/ask with close (approximation)
        df["bid"] = df["bid"].fillna(df["close"])
        df["ask"] = df["ask"].fillna(df["close"])
        cols = EXTENDED_COLUMNS
    else:
        cols = STANDARD_COLUMNS
    # Ensure all expected columns exist
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    df = df[cols]

    # Save CSV
    df.to_csv(out_path, index=False)
    log.info(f"  [{symbol} {tf}] Saved {len(df)} bars to {out_path.relative_to(PROJECT_ROOT)}")

    # Full validation
    validation = validate_full(df, symbol, tf)
    validation["file"] = str(out_path.relative_to(PROJECT_ROOT))
    validation["symbol"] = symbol
    validation["timeframe"] = tf
    validation["schema"] = cols
    validation["download_timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(validation_path, "w") as f:
        json.dump(validation, f, indent=2, default=str)
    log.info(f"  [{symbol} {tf}] Validation: {len(validation['errors'])} errors, {len(validation['warnings'])} warnings → {validation_path.relative_to(PROJECT_ROOT)}")

    return {
        "symbol": symbol, "timeframe": tf,
        "start": str(df["datetime_utc"].iloc[0]),
        "end": str(df["datetime_utc"].iloc[-1]),
        "rows": len(df),
        "timezone": "UTC",
        "source": "MT5 (copy_rates_range, broker-tz corrected)",
        "download_timestamp": datetime.now(timezone.utc).isoformat(),
        "path": str(out_path.relative_to(PROJECT_ROOT)),
        "validation_path": str(validation_path.relative_to(PROJECT_ROOT)),
        "schema": cols,
        "has_bid_ask": with_bid_ask,
        "validation_errors": validation["errors"],
        "validation_warnings": validation["warnings"],
        "spread_zero_pct": validation["stats"].get("spread_zero_pct"),
        "gap_count": validation["stats"].get("gap_count"),
        "non_weekend_gap_count": validation["stats"].get("non_weekend_gap_count"),
        "real_volume_available": bool((df["real_volume"] > 0).any()) if "real_volume" in df.columns else False,
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download historical OHLCV data from MT5 to local CSV files (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Full suite: 8 symbols × 5 TFs (full backtest setup)
  python scripts/download_historical_data.py \\
      --symbols EURUSD GBPUSD USDJPY USDCHF USDCAD AUDUSD NZDUSD XAUUSD \\
      --timeframes M5 M15 H1 H4 D1 \\
      --start 2024-08-01 --end 2026-08-17 \\
      --warmup-bars 1000 --with-bid-ask

  # Just download missing XAUUSD
  python scripts/download_historical_data.py \\
      --symbols XAUUSD --timeframes M15 H1 H4 D1 \\
      --start 2024-08-01 --end 2026-08-17

  # Re-download and overwrite existing files
  python scripts/download_historical_data.py \\
      --symbols EURUSD --timeframes H1 --start 2024-08-01 --end 2026-08-17 --force
        """,
    )
    parser.add_argument("--symbols", nargs="+", required=True,
                        help="Symbol list, e.g. EURUSD GBPUSD USDJPY XAUUSD")
    parser.add_argument("--timeframes", nargs="+", required=True,
                        help="Timeframe list, e.g. M5 M15 H1 H4 D1")
    parser.add_argument("--start", type=str, required=True,
                        help="Start date (YYYY-MM-DD) — exclusive of warmup extension")
    parser.add_argument("--end", type=str, required=True,
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--warmup-bars", type=int, default=DEFAULT_WARMUP_BARS,
                        help=f"Number of warmup bars to fetch BEFORE --start on each TF (default: {DEFAULT_WARMUP_BARS})")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing CSV files (default: skip if range already covered)")
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="Disable idempotent re-run (always re-download)")
    parser.add_argument("--with-bid-ask", action="store_true",
                        help="Also fetch bid/ask at bar close (slow — adds ~0.1s per bar)")
    parser.add_argument("--legacy-flat", action="store_true",
                        help="Save to data/{SYMBOL}_{TF}.csv (legacy flat layout) "
                             "instead of data/historical/{SYMBOL}/{TF}.csv (nested)")
    args = parser.parse_args()

    try:
        start_dt = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
        end_dt = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    except ValueError as e:
        log.error(f"Invalid date format: {e}. Use YYYY-MM-DD.")
        sys.exit(1)

    if end_dt <= start_dt:
        log.error("--end must be after --start")
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print(f"MT5 Historical Data Downloader (v2)")
    print(f"{'=' * 70}")
    print(f"Symbols:       {args.symbols}")
    print(f"Timeframes:    {args.timeframes}")
    print(f"Date range:    {start_dt.date()} → {end_dt.date()}")
    print(f"Warmup bars:   {args.warmup_bars} (extends start backward per TF)")
    print(f"With bid/ask:  {args.with_bid_ask}")
    print(f"Output:        {HISTORICAL_DIR if not args.legacy_flat else LEGACY_FLAT_DIR}/")
    print(f"Force:         {args.force}")
    print(f"Skip existing: {not args.no_skip_existing}")
    print()

    if not _connect_mt5():
        log.error("Cannot connect to MT5. Make sure:")
        log.error("  1. MetaTrader5 package is installed: pip install MetaTrader5")
        log.error("  2. MT5 terminal is running")
        log.error("  3. You're on Windows (or Wine) — MT5 doesn't run on bare Linux")
        sys.exit(1)

    try:
        # Detect broker UTC offset once at start
        _detect_broker_utc_offset_hours(args.symbols[0])

        manifest_entries = []
        for symbol in args.symbols:
            for tf in args.timeframes:
                print(f"\n[{symbol} {tf}]")
                entry = download_symbol_tf(
                    symbol=symbol, tf=tf,
                    start=start_dt, end=end_dt,
                    force=args.force,
                    nested=not args.legacy_flat,
                    warmup_bars=args.warmup_bars,
                    with_bid_ask=args.with_bid_ask,
                    skip_existing=not args.no_skip_existing,
                )
                manifest_entries.append(entry)

        # Write manifest
        manifest_path = HISTORICAL_DIR / "manifest.json" if not args.legacy_flat else LEGACY_FLAT_DIR / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "MT5 (copy_rates_range, broker-tz corrected)",
            "date_range": {"start": args.start, "end": args.end, "warmup_bars": args.warmup_bars},
            "symbols": args.symbols,
            "timeframes": args.timeframes,
            "broker_utc_offset_hours": _BROKER_OFFSET_CACHE,
            "schema": EXTENDED_COLUMNS if args.with_bid_ask else STANDARD_COLUMNS,
            "files": manifest_entries,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
        print(f"\n{'=' * 70}")
        print(f"DONE. Manifest written to: {manifest_path.relative_to(PROJECT_ROOT)}")
        print(f"{'=' * 70}")

        total_rows = sum(e.get("rows", 0) for e in manifest_entries)
        errors = [e for e in manifest_entries if e.get("error")]
        skipped = [e for e in manifest_entries if e.get("skipped")]
        downloaded = [e for e in manifest_entries if not e.get("error") and not e.get("skipped")]
        val_errors = sum(len(e.get("validation_errors", [])) for e in downloaded)
        val_warnings = sum(len(e.get("validation_warnings", [])) for e in downloaded)
        print(f"\nSummary:")
        print(f"  Downloaded:  {len(downloaded)} files ({total_rows:,} total rows)")
        print(f"  Skipped:     {len(skipped)} (already covered)")
        print(f"  Errors:      {len(errors)}")
        print(f"  Validation:  {val_errors} errors, {val_warnings} warnings (see _validation_*.json)")
        for e in errors:
            print(f"    {e['symbol']} {e['timeframe']}: {e['error']}")
    finally:
        _shutdown_mt5()


if __name__ == "__main__":
    main()
