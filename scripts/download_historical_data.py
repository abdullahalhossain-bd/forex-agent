"""
scripts/download_historical_data.py — MT5 → local CSV historical data downloader.

Downloads OHLCV + spread + real_volume bars from MT5 for the requested
symbols/timeframes/date range, saves to local CSV files under
data/historical/{SYMBOL}/{TF}.csv (or data/{SYMBOL}_{TF}.csv for
backward compatibility — both paths are supported by the loader).

USAGE (on production VPS with MT5 terminal running):

    python scripts/download_historical_data.py \\
        --symbols EURUSD GBPUSD USDJPY USDCHF USDCAD AUDUSD NZDUSD XAUUSD \\
        --timeframes M15 H1 H4 D1 \\
        --start 2025-07-01 \\
        --end 2026-08-08

WHAT IT DOES:
  1. Connects to MT5 (uses existing broker/mt5_connection.py singleton).
  2. For each (symbol, timeframe) pair, fetches bars in monthly chunks
     (to avoid MT5's per-call rate limits).
  3. Validates each chunk (no duplicates, sorted, OHLC sanity).
  4. Saves to data/historical/{SYMBOL}/{TF}.csv with columns:
       datetime_utc, open, high, low, close, tick_volume, spread, real_volume
  5. Updates data/historical/manifest.json with download metadata.

WHAT IT DOES NOT DO:
  - Does NOT call MT5 during a backtest run. Download is a one-time setup step.
  - Does NOT silently overwrite existing CSVs — use --force to overwrite.
  - Does NOT fetch tick data (only bar-level OHLCV). Tick-level modules
    (LiveFeed, MicrostructureEngine) remain skipped in backtest.

ENVIRONMENT REQUIREMENTS:
  - MetaTrader5 Python package installed: pip install MetaTrader5
  - MT5 terminal running with a connected account
  - Windows or Wine (MT5 doesn't run on bare Linux)
  - Run on the production VPS, NOT in a CI/audit env

PARITY NOTE:
  The downloaded CSV format matches the existing data/{SYMBOL}_{TF}.csv
  files in the repo (same column names, same UTC timezone). The CSV
  provider can load from either location.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Output structure (user-requested nested form)
HISTORICAL_DIR = PROJECT_ROOT / "data" / "historical"
LEGACY_FLAT_DIR = PROJECT_ROOT / "data"  # backward-compat: data/{SYMBOL}_{TF}.csv

# MT5 timeframe constants (imported lazily so this script can be syntax-checked
# without MetaTrader5 installed)
MT5_TF_MAP = {
    "M1": "TIMEFRAME_M1",
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1",
    "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
    "W1": "TIMEFRAME_W1",
    "MN1": "TIMEFRAME_MN1",
}


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
        print(f"[ERROR] MT5 initialize() failed: {mt5.last_error()}")
        return False
    info = mt5.terminal_info()
    if info is None:
        print(f"[ERROR] MT5 terminal_info() returned None: {mt5.last_error()}")
        return False
    print(f"[OK] MT5 connected: terminal={info.name}, build={info.build}, "
          f"path={info.path}")
    acct = mt5.account_info()
    if acct:
        print(f"     Account: {acct.login} ({acct.server}), balance={acct.balance} {acct.currency}")
    return True


def _shutdown_mt5():
    import MetaTrader5 as mt5
    mt5.shutdown()


def _fetch_chunk(mt5, symbol: str, tf_const, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch one chunk of bars from MT5 in [start, end) range.

    Returns DataFrame with columns:
      datetime_utc, open, high, low, close, tick_volume, spread, real_volume
    """
    rates = mt5.copy_rates_range(symbol, tf_const, start, end)
    if rates is None:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    if df.empty:
        return df
    # MT5 returns time as epoch seconds. Convert to UTC datetime.
    df["datetime_utc"] = pd.to_datetime(df["time"], unit="s", utc=True)
    # Rename MT5 columns to match existing CSV convention
    df = df.rename(columns={
        "tick_volume": "tick_volume",  # already correct
        "real_volume": "real_volume",  # already correct
    })
    # Keep only the columns we want, in the right order
    cols = ["datetime_utc", "open", "high", "low", "close",
            "tick_volume", "spread", "real_volume"]
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    df = df[cols]
    # Drop duplicates (MT5 sometimes returns the last bar of the previous chunk)
    df = df.drop_duplicates(subset=["datetime_utc"]).sort_values("datetime_utc").reset_index(drop=True)
    return df


def _validate_chunk(df: pd.DataFrame, symbol: str, tf: str) -> list[str]:
    """Validate a chunk. Returns list of issues (empty list = OK)."""
    issues = []
    if df.empty:
        return ["empty"]
    # Sort check
    if not df["datetime_utc"].is_monotonic_increasing:
        issues.append("not sorted ascending")
    # Duplicate check
    n_dupes = df["datetime_utc"].duplicated().sum()
    if n_dupes > 0:
        issues.append(f"{n_dupes} duplicate timestamps")
    # OHLC sanity
    bad_high = (df["high"] < df[["open", "close"]].max(axis=1)).sum()
    bad_low = (df["low"] > df[["open", "close"]].min(axis=1)).sum()
    neg = (df[["open", "high", "low", "close"]] < 0).any(axis=1).sum()
    if bad_high > 0:
        issues.append(f"{bad_high} rows with high < max(open, close)")
    if bad_low > 0:
        issues.append(f"{bad_low} rows with low > min(open, close)")
    if neg > 0:
        issues.append(f"{neg} rows with negative prices")
    return issues


def download_symbol_tf(symbol: str, tf: str, start: datetime, end: datetime,
                        force: bool = False, nested: bool = True) -> dict:
    """Download one (symbol, tf) pair. Returns a manifest entry dict."""
    out_dir = HISTORICAL_DIR / symbol if nested else LEGACY_FLAT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tf}.csv" if nested else out_dir / f"{symbol}_{tf}.csv"

    if out_path.exists() and not force:
        print(f"[SKIP] {symbol} {tf} — already exists at {out_path} (use --force to overwrite)")
        existing = pd.read_csv(out_path)
        return {
            "symbol": symbol, "timeframe": tf,
            "start": str(existing["datetime_utc"].iloc[0]) if len(existing) > 0 else None,
            "end": str(existing["datetime_utc"].iloc[-1]) if len(existing) > 0 else None,
            "rows": len(existing),
            "skipped": True, "reason": "already exists (use --force to overwrite)",
        }

    import MetaTrader5 as mt5
    tf_const = _get_mt5_timeframe(tf)

    # Fetch in monthly chunks to avoid MT5 per-call limits
    all_chunks = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=30), end)
        print(f"  Fetching {symbol} {tf}: {chunk_start.date()} → {chunk_end.date()} ...", end=" ", flush=True)
        try:
            chunk = _fetch_chunk(mt5, symbol, tf_const, chunk_start, chunk_end)
        except Exception as e:
            print(f"FAIL ({e})")
            return {
                "symbol": symbol, "timeframe": tf,
                "error": f"fetch failed at {chunk_start}: {e}",
                "rows": 0,
            }
        print(f"{len(chunk)} bars")
        if not chunk.empty:
            issues = _validate_chunk(chunk, symbol, tf)
            if issues:
                print(f"    ⚠️  validation issues: {issues}")
            all_chunks.append(chunk)
        chunk_start = chunk_end
        # Be polite to MT5 — small delay between calls
        time.sleep(0.3)

    if not all_chunks:
        return {
            "symbol": symbol, "timeframe": tf,
            "error": "no data fetched in the requested range",
            "rows": 0,
        }

    # Combine + dedupe + sort
    df = pd.concat(all_chunks, ignore_index=True)
    df = df.drop_duplicates(subset=["datetime_utc"]).sort_values("datetime_utc").reset_index(drop=True)

    # Save
    df.to_csv(out_path, index=False)
    print(f"  ✅ Saved {len(df)} bars to {out_path.relative_to(PROJECT_ROOT)}")

    return {
        "symbol": symbol,
        "timeframe": tf,
        "start": str(df["datetime_utc"].iloc[0]),
        "end": str(df["datetime_utc"].iloc[-1]),
        "rows": len(df),
        "timezone": "UTC",
        "source": "MT5 (copy_rates_range)",
        "download_timestamp": datetime.now(timezone.utc).isoformat(),
        "path": str(out_path.relative_to(PROJECT_ROOT)),
        "spread_available": bool((df["spread"] > 0).any()) if "spread" in df.columns else False,
        "real_volume_available": bool((df["real_volume"] > 0).any()) if "real_volume" in df.columns else False,
        "missing_ranges": [],  # populated by validator script later
    }


def main():
    parser = argparse.ArgumentParser(
        description="Download historical OHLCV data from MT5 to local CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Download 8 symbols × 4 timeframes (full backtest suite)
  python scripts/download_historical_data.py \\
      --symbols EURUSD GBPUSD USDJPY USDCHF USDCAD AUDUSD NZDUSD XAUUSD \\
      --timeframes M15 H1 H4 D1 \\
      --start 2025-07-01 --end 2026-08-08

  # Just download missing XAUUSD (other symbols already exist)
  python scripts/download_historical_data.py \\
      --symbols XAUUSD --timeframes M15 H1 H4 D1 \\
      --start 2025-07-01 --end 2026-08-08

  # Re-download and overwrite existing files
  python scripts/download_historical_data.py \\
      --symbols EURUSD --timeframes H1 --start 2025-07-01 --end 2026-08-08 --force
        """,
    )
    parser.add_argument("--symbols", nargs="+", required=True,
                        help="Symbol list, e.g. EURUSD GBPUSD USDJPY XAUUSD")
    parser.add_argument("--timeframes", nargs="+", required=True,
                        help="Timeframe list, e.g. M15 H1 H4 D1")
    parser.add_argument("--start", type=str, required=True,
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True,
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing CSV files")
    parser.add_argument("--legacy-flat", action="store_true",
                        help="Save to data/{SYMBOL}_{TF}.csv (legacy flat layout) "
                             "instead of data/historical/{SYMBOL}/{TF}.csv (nested)")
    args = parser.parse_args()

    try:
        start_dt = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
        end_dt = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    except ValueError as e:
        print(f"[ERROR] Invalid date format: {e}. Use YYYY-MM-DD.")
        sys.exit(1)

    if end_dt <= start_dt:
        print("[ERROR] --end must be after --start")
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print(f"MT5 Historical Data Downloader")
    print(f"{'=' * 70}")
    print(f"Symbols:     {args.symbols}")
    print(f"Timeframes:  {args.timeframes}")
    print(f"Date range:  {start_dt.date()} → {end_dt.date()}")
    print(f"Output:      {HISTORICAL_DIR if not args.legacy_flat else LEGACY_FLAT_DIR}/")
    print(f"Force:       {args.force}")
    print()

    # Connect to MT5
    if not _connect_mt5():
        print("\n[ERROR] Cannot connect to MT5. Make sure:")
        print("  1. MetaTrader5 package is installed: pip install MetaTrader5")
        print("  2. MT5 terminal is running")
        print("  3. You're on Windows (or Wine) — MT5 doesn't run on bare Linux")
        sys.exit(1)

    try:
        # Download each (symbol, tf) pair
        manifest_entries = []
        for symbol in args.symbols:
            for tf in args.timeframes:
                print(f"\n[{symbol} {tf}]")
                entry = download_symbol_tf(
                    symbol=symbol, tf=tf,
                    start=start_dt, end=end_dt,
                    force=args.force,
                    nested=not args.legacy_flat,
                )
                manifest_entries.append(entry)

        # Write manifest
        manifest_path = HISTORICAL_DIR / "manifest.json" if not args.legacy_flat else LEGACY_FLAT_DIR / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "MT5 (copy_rates_range)",
            "date_range": {"start": args.start, "end": args.end},
            "symbols": args.symbols,
            "timeframes": args.timeframes,
            "files": manifest_entries,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
        print(f"\n{'=' * 70}")
        print(f"DONE. Manifest written to: {manifest_path.relative_to(PROJECT_ROOT)}")
        print(f"{'=' * 70}")

        # Summary
        total_rows = sum(e.get("rows", 0) for e in manifest_entries)
        errors = [e for e in manifest_entries if e.get("error")]
        skipped = [e for e in manifest_entries if e.get("skipped")]
        downloaded = [e for e in manifest_entries if not e.get("error") and not e.get("skipped")]
        print(f"\nSummary:")
        print(f"  Downloaded: {len(downloaded)} files ({total_rows:,} total rows)")
        print(f"  Skipped:    {len(skipped)} (already existed)")
        print(f"  Errors:     {len(errors)}")
        for e in errors:
            print(f"    {e['symbol']} {e['timeframe']}: {e['error']}")
    finally:
        _shutdown_mt5()


if __name__ == "__main__":
    main()
