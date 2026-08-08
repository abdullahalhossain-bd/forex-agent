"""
scripts/generate_manifest.py — Generate / update data/historical/manifest.json.

Scans data/historical/ (and data/ as fallback) for CSV files and produces
a manifest describing each dataset's:
  - symbol, timeframe
  - date range (start, end)
  - row count
  - timezone
  - source (CSV file path)
  - download timestamp (file mtime)
  - missing_ranges (from validation report if available)
  - spread_available
  - real_volume_available
  - tick_volume_available

Output: data/historical/manifest.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

HISTORICAL_DIR = PROJECT_ROOT / "data" / "historical"
LEGACY_DIR = PROJECT_ROOT / "data"


def _detect_symbol_timeframe(filepath: Path) -> tuple[str, str]:
    """Detect (symbol, timeframe) from path."""
    if "historical" in filepath.parts:
        # data/historical/{SYMBOL}/{TF}.csv
        idx = filepath.parts.index("historical")
        if idx + 2 < len(filepath.parts):
            symbol = filepath.parts[idx + 1]
            timeframe = filepath.stem.upper()
            return symbol, timeframe
    # Flat: data/{SYMBOL}_{TF}.csv
    parts = filepath.stem.split("_")
    if len(parts) >= 2:
        return parts[0], parts[-1].upper()
    return filepath.stem, "UNKNOWN"


def scan_csvs() -> list[dict]:
    """Scan both layouts and return a list of manifest entries."""
    entries = []

    # Nested: data/historical/{SYMBOL}/{TF}.csv
    if HISTORICAL_DIR.exists():
        for sym_dir in sorted(HISTORICAL_DIR.iterdir()):
            if not sym_dir.is_dir():
                continue
            for csv in sorted(sym_dir.glob("*.csv")):
                entry = _build_entry(csv)
                entries.append(entry)

    # Flat: data/{SYMBOL}_{TF}.csv (only add if not already in nested)
    nested_keys = {(e["symbol"], e["timeframe"]) for e in entries}
    for csv in sorted(LEGACY_DIR.glob("*_*.csv")):
        if "historical" in csv.parts:
            continue
        symbol, tf = _detect_symbol_timeframe(csv)
        if (symbol, tf) not in nested_keys:
            entry = _build_entry(csv)
            entries.append(entry)

    return entries


def _build_entry(filepath: Path) -> dict:
    """Build a manifest entry for a single CSV."""
    symbol, timeframe = _detect_symbol_timeframe(filepath)
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return {
            "symbol": symbol, "timeframe": timeframe,
            "path": str(filepath.relative_to(PROJECT_ROOT)),
            "error": f"read failed: {e}",
        }

    # Find timestamp column
    ts_col = None
    for c in ("datetime_utc", "datetime", "time", "timestamp", "date"):
        if c in df.columns:
            ts_col = c
            break
    if ts_col is None:
        return {
            "symbol": symbol, "timeframe": timeframe,
            "path": str(filepath.relative_to(PROJECT_ROOT)),
            "rows": len(df), "error": "no timestamp column",
        }

    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df["_ts"] = ts
    df = df.dropna(subset=["_ts"]).sort_values("_ts")
    df = df.drop_duplicates(subset=["_ts"])

    # File mtime as download_timestamp
    stat = filepath.stat()
    download_ts = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()

    # Spread availability
    spread_avail = False
    spread_nonzero_pct = 0.0
    if "spread" in df.columns:
        nonzero = (df["spread"] > 0).sum()
        spread_nonzero_pct = round(100 * nonzero / max(len(df), 1), 2)
        spread_avail = nonzero > 0

    # Volume availability
    tick_vol_avail = False
    if "tick_volume" in df.columns:
        tick_vol_avail = (df["tick_volume"] > 0).any()
    elif "volume" in df.columns:
        tick_vol_avail = (df["volume"] > 0).any()

    real_vol_avail = False
    if "real_volume" in df.columns:
        real_vol_avail = (df["real_volume"] > 0).any()

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "start": str(df["_ts"].iloc[0]) if len(df) > 0 else None,
        "end": str(df["_ts"].iloc[-1]) if len(df) > 0 else None,
        "rows": int(len(df)),
        "timezone": "UTC",
        "source": "local_csv",
        "path": str(filepath.relative_to(PROJECT_ROOT)),
        "download_timestamp": download_ts,
        "missing_ranges": [],  # populated by validate_historical_data.py
        "spread_available": spread_avail,
        "spread_nonzero_pct": spread_nonzero_pct,
        "tick_volume_available": bool(tick_vol_avail),
        "real_volume_available": bool(real_vol_avail),
    }


def main():
    print("Generating dataset manifest...\n")
    entries = scan_csvs()

    # Group by symbol
    by_symbol: dict[str, list[dict]] = {}
    for e in entries:
        by_symbol.setdefault(e["symbol"], []).append(e)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "local_csv",
        "symbols": sorted(by_symbol.keys()),
        "timeframes": sorted({e["timeframe"] for e in entries}),
        "total_files": len(entries),
        "total_rows": sum(e.get("rows", 0) for e in entries),
        "files": entries,
    }

    # Write
    out_path = HISTORICAL_DIR / "manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, default=str))

    # Print summary
    print(f"{'Symbol':<10} {'TF':<5} {'Rows':>7} {'Start':<27} {'End':<27} {'Spread':<10} {'Path'}")
    print("-" * 110)
    for e in entries:
        spread_str = f"yes({e.get('spread_nonzero_pct', 0):.0f}%)" if e.get("spread_available") else "NO"
        print(f"{e['symbol']:<10} {e['timeframe']:<5} {e.get('rows',0):>7} "
              f"{e.get('start','?')[:27]:<27} {e.get('end','?')[:27]:<27} "
              f"{spread_str:<10} {e.get('path','?')}")

    print(f"\nManifest written to: {out_path.relative_to(PROJECT_ROOT)}")
    print(f"Total: {manifest['total_files']} files, {manifest['total_rows']:,} rows, "
          f"{len(manifest['symbols'])} symbols, {len(manifest['timeframes'])} timeframes")


if __name__ == "__main__":
    main()