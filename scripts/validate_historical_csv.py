#!/usr/bin/env python3
"""
scripts/validate_historical_csv.py — Standalone CSV validation CLI.

Runs the same validation logic as the downloader on existing CSV files.
Useful for:
  - Validating CSVs downloaded by older versions of the downloader
  - Spot-checking data quality before backtest runs
  - Generating machine-readable validation reports

USAGE:
    python scripts/validate_historical_csv.py
    python scripts/validate_historical_csv.py --symbol EURUSD --tf M15
    python scripts/validate_historical_csv.py --data-dir data/historical/EURUSD
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Import validation logic from the downloader
from scripts.download_historical_data import validate_full, TF_SECONDS, STANDARD_COLUMNS, EXTENDED_COLUMNS

DATA_DIR = PROJECT_ROOT / "data"
HISTORICAL_DIR = DATA_DIR / "historical"


def find_csvs(symbol: str = None, tf: str = None, data_dir: Path = None) -> list[Path]:
    """Find CSV files matching the criteria."""
    if data_dir:
        return sorted(data_dir.glob("*.csv"))
    csvs = []
    # Nested layout
    if HISTORICAL_DIR.exists():
        for sym_dir in HISTORICAL_DIR.iterdir():
            if not sym_dir.is_dir(): continue
            if symbol and sym_dir.name != symbol: continue
            for csv in sym_dir.glob("*.csv"):
                if csv.name.startswith("_"): continue  # skip validation files
                if tf and csv.stem != tf: continue
                csvs.append(csv)
    # Flat layout
    for csv in DATA_DIR.glob("*.csv"):
        parts = csv.stem.split("_")
        if len(parts) < 2: continue
        sym, file_tf = parts[0], parts[-1]
        if symbol and sym != symbol: continue
        if tf and file_tf != tf: continue
        csvs.append(csv)
    return sorted(set(csvs))


def main():
    parser = argparse.ArgumentParser(description="Validate historical CSV files")
    parser.add_argument("--symbol", type=str, help="Filter by symbol (e.g. EURUSD)")
    parser.add_argument("--tf", type=str, help="Filter by timeframe (e.g. M15)")
    parser.add_argument("--data-dir", type=str, help="Custom data directory")
    parser.add_argument("--json", action="store_true", help="Output as JSON instead of markdown")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else None
    csvs = find_csvs(symbol=args.symbol, tf=args.tf, data_dir=data_dir)
    if not csvs:
        print("No CSV files found matching criteria")
        sys.exit(1)

    print(f"Found {len(csvs)} CSV files to validate")
    all_results = []
    for csv in csvs:
        print(f"\nValidating {csv.name}...")
        df = pd.read_csv(csv, encoding="utf-8-sig")
        # Parse symbol/TF from filename
        parts = csv.stem.split("_")
        sym = parts[0] if parts else "UNKNOWN"
        tf = parts[-1] if len(parts) >= 2 else "UNKNOWN"
        # If nested layout (csv is just "M15.csv"), get symbol from parent dir
        if csv.parent != DATA_DIR and csv.parent != HISTORICAL_DIR:
            sym = csv.parent.name
        # Parse timestamps
        ts_col = None
        for c in ("datetime_utc", "datetime", "time", "timestamp", "date"):
            if c in df.columns:
                ts_col = c
                break
        if ts_col is None:
            print(f"  ERROR: no timestamp column")
            all_results.append({"file": str(csv), "symbol": sym, "timeframe": tf, "errors": ["no timestamp column"]})
            continue
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df = df.dropna(subset=[ts_col]).rename(columns={ts_col: "datetime_utc"})
        df = df.sort_values("datetime_utc").reset_index(drop=True)
        result = validate_full(df, sym, tf)
        result["file"] = str(csv.relative_to(PROJECT_ROOT))
        result["symbol"] = sym
        result["timeframe"] = tf
        all_results.append(result)
        if result["errors"]:
            print(f"  ❌ {len(result['errors'])} errors:")
            for e in result["errors"]:
                print(f"     - {e}")
        if result["warnings"]:
            print(f"  ⚠️  {len(result['warnings'])} warnings:")
            for w in result["warnings"]:
                print(f"     - {w}")
        stats = result.get("stats", {})
        if stats:
            print(f"  Stats: {stats.get('rows',0)} rows, {stats.get('date_range_days',0)} days, "
                  f"{stats.get('spread_zero_pct',0)}% zero spread, {stats.get('non_weekend_gap_count',0)} non-weekend gaps")

    # Output summary
    if args.json:
        print(json.dumps(all_results, indent=2, default=str))
    else:
        out_path = PROJECT_ROOT / "docs" / "audit" / "evidence" / "P-validation-cli-report.md"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write("# CSV Validation CLI Report\n\n")
            f.write(f"**Generated:** {datetime.now(timezone.utc).isoformat()}\n")
            f.write(f"**Files validated:** {len(all_results)}\n\n")
            f.write("## Summary Table\n\n")
            f.write("| File | Symbol | TF | Rows | Days | Spread 0% | Gaps | Non-WKND Gaps | Errors | Warnings |\n")
            f.write("|------|--------|----|-----|------|-----------|------|---------------|--------|----------|\n")
            for r in all_results:
                stats = r.get("stats", {})
                f.write(f"| {r['file']} | {r.get('symbol','-')} | {r.get('timeframe','-')} | "
                        f"{stats.get('rows',0)} | {stats.get('date_range_days',0)} | "
                        f"{stats.get('spread_zero_pct',0)}% | {stats.get('gap_count',0)} | "
                        f"{stats.get('non_weekend_gap_count',0)} | "
                        f"{len(r.get('errors',[]))} | {len(r.get('warnings',[]))} |\n")
            f.write("\n## Per-File Detail\n\n")
            for r in all_results:
                f.write(f"### {r['file']}\n\n")
                if r.get("errors"):
                    f.write("**Errors:**\n")
                    for e in r["errors"]: f.write(f"- {e}\n")
                if r.get("warnings"):
                    f.write("**Warnings:**\n")
                    for w in r["warnings"]: f.write(f"- {w}\n")
                stats = r.get("stats", {})
                f.write(f"\n**Stats:** {json.dumps(stats, indent=2, default=str)}\n\n")
        print(f"\n📄 Report written to: {out_path}")


if __name__ == "__main__":
    main()
