#!/usr/bin/env python3
"""
scripts/download_external_data.py — Download external macro / intermarket data.

Downloads historical OHLCV for DXY, Gold, Oil, US10Y, SP500, VIX using yfinance.
Saves to data/external/{ASSET}_D1.csv with the same schema as the MT5 downloader:
    datetime_utc, open, high, low, close, tick_volume, spread, real_volume
(spread and real_volume will be 0 — yfinance doesn't provide them)

USAGE:
    python scripts/download_external_data.py --start 2024-08-01 --end 2026-08-17

REQUIREMENTS:
    pip install yfinance
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

# yfinance tickers
EXTERNAL_ASSETS = {
    "DXY": "DX-Y.NYB",      # US Dollar Index
    "GOLD": "GC=F",          # Gold futures
    "OIL": "CL=F",           # WTI Crude Oil futures
    "US10Y": "^TNX",         # 10-Year Treasury Yield
    "SP500": "^GSPC",        # S&P 500
    "VIX": "^VIX",           # CBOE Volatility Index
}


def download_asset(name: str, yf_ticker: str, start: str, end: str) -> dict:
    """Download one external asset via yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    print(f"  [{name}] Downloading {yf_ticker} {start} → {end} ...")
    try:
        df = yf.download(yf_ticker, start=start, end=end, auto_adjust=False, progress=False)
    except Exception as e:
        return {"asset": name, "ticker": yf_ticker, "error": str(e), "rows": 0}

    if df is None or df.empty:
        return {"asset": name, "ticker": yf_ticker, "error": "no data returned", "rows": 0}

    # yfinance returns MultiIndex columns when downloading single ticker in newer versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Rename to standard schema
    df = df.reset_index()
    rename_map = {
        "Date": "datetime_utc",
        "Datetime": "datetime_utc",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "tick_volume",
    }
    df = df.rename(columns=rename_map)

    # Convert to UTC
    if "datetime_utc" in df.columns:
        df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
        df = df.dropna(subset=["datetime_utc"])

    # Add required columns (spread and real_volume not available from yfinance)
    if "tick_volume" not in df.columns:
        df["tick_volume"] = 0
    df["spread"] = 0
    df["real_volume"] = df.get("tick_volume", 0)  # yfinance Volume IS the real volume

    # Keep only standard columns
    cols = ["datetime_utc", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    df = df[cols]

    # Save
    out_path = EXTERNAL_DIR / f"{name}_D1.csv"
    df.to_csv(out_path, index=False)
    print(f"  [{name}] Saved {len(df)} rows to {out_path.relative_to(PROJECT_ROOT)}")

    return {
        "asset": name, "ticker": yf_ticker,
        "start": str(df["datetime_utc"].iloc[0]),
        "end": str(df["datetime_utc"].iloc[-1]),
        "rows": len(df), "path": str(out_path.relative_to(PROJECT_ROOT)),
    }


def main():
    parser = argparse.ArgumentParser(description="Download external macro data via yfinance")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--assets", nargs="+", default=list(EXTERNAL_ASSETS.keys()),
                        help=f"Asset list (default: {list(EXTERNAL_ASSETS.keys())})")
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"External Macro Data Downloader (yfinance)")
    print(f"{'=' * 60}")
    print(f"Assets:  {args.assets}")
    print(f"Range:   {args.start} → {args.end}")
    print(f"Output:  {EXTERNAL_DIR}/")
    print()

    results = []
    for asset in args.assets:
        if asset not in EXTERNAL_ASSETS:
            print(f"  ⚠️  Unknown asset: {asset}")
            continue
        results.append(download_asset(asset, EXTERNAL_ASSETS[asset], args.start, args.end))

    # Write manifest
    manifest_path = EXTERNAL_DIR / "manifest.json"
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "yfinance",
        "date_range": {"start": args.start, "end": args.end},
        "assets": args.assets,
        "files": results,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\nManifest: {manifest_path.relative_to(PROJECT_ROOT)}")
    print(f"\nSummary: {sum(1 for r in results if not r.get('error'))}/{len(results)} assets downloaded")


if __name__ == "__main__":
    main()
