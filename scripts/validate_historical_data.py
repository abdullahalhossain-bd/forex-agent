"""
scripts/validate_historical_data.py — Validate historical CSV files.

For each CSV in data/historical/{SYMBOL}/{TF}.csv (or data/{SYMBOL}_{TF}.csv),
validates:

STRUCTURE:
  - timestamp column exists (datetime_utc / datetime / time)
  - timestamps sorted ascending
  - no duplicate timestamps
  - OHLC numeric (no NaN, no strings)
  - high >= max(open, close)
  - low <= min(open, close)
  - no negative prices

MISSING DATA:
  - Detects gaps (expected interval vs actual)
  - Distinguishes weekend gaps (expected for FX) from real data gaps
  - Reports gap count + sample gaps per file

SPREAD:
  - Reports whether `spread` column is present
  - Reports % of rows with non-zero spread
  - If spread is unavailable, explicitly flags it

TIMEZONE:
  - Verifies timestamps are UTC-aware
  - Reports timezone offset (should be +00:00)

Outputs:
  - data/historical/validation_report.json (machine-readable)
  - stdout summary table
  - Updates data/historical/manifest.json with validation results

Usage:
    python scripts/validate_historical_data.py
    python scripts/validate_historical_data.py --dir data/historical
    python scripts/validate_historical_data.py --file data/eurusd_h1.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# Timeframe → expected interval in minutes
TF_INTERVALS = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 60, "H4": 240, "D1": 1440, "W1": 10080,
}


def _detect_timeframe(filepath: Path) -> str | None:
    """Detect timeframe from filename. Returns 'M15', 'H1', etc. or None."""
    name = filepath.stem
    parts = name.split("_")
    if len(parts) >= 2:
        return parts[-1].upper()
    # If nested form: data/historical/EURUSD/H1.csv
    return filepath.stem.upper()


def _detect_symbol(filepath: Path) -> str | None:
    """Detect symbol from path."""
    # Nested: data/historical/EURUSD/H1.csv → parent dir is symbol
    if "historical" in filepath.parts:
        idx = filepath.parts.index("historical")
        if idx + 1 < len(filepath.parts) - 1:
            return filepath.parts[idx + 1]
    # Flat: data/EURUSD_H1.csv → first part of stem
    name = filepath.stem
    parts = name.split("_")
    if len(parts) >= 2:
        return parts[0]
    return None


def _is_weekend_gap(prev_ts: pd.Timestamp, curr_ts: pd.Timestamp) -> bool:
    """Check if a gap between two timestamps spans a weekend (FX market closed).

    FX market closes Friday ~22:00 UTC and opens Sunday ~22:00 UTC.
    Returns True if the gap is fully within this closed period.
    """
    # Friday after 22:00 UTC → Sunday 22:00 UTC is closed
    if prev_ts.weekday() == 4:  # Friday
        if prev_ts.hour >= 22 or (prev_ts.hour >= 21 and prev_ts.minute >= 45):
            # Friday late → next open is Sunday 22:00 / Monday 00:00
            if curr_ts.weekday() in (5, 6):  # Sat or Sun
                return True
            if curr_ts.weekday() == 0 and curr_ts.hour < 2:  # Monday early
                return True
    if prev_ts.weekday() == 5:  # Saturday
        if curr_ts.weekday() in (5, 6):
            return True
        if curr_ts.weekday() == 0 and curr_ts.hour < 2:
            return True
    if prev_ts.weekday() == 6:  # Sunday
        if curr_ts.weekday() == 6 and curr_ts.hour < 22:
            return True
    return False


def validate_csv(filepath: Path) -> dict:
    """Validate a single CSV. Returns findings dict."""
    result = {
        "file": str(filepath.relative_to(PROJECT_ROOT)),
        "symbol": _detect_symbol(filepath),
        "timeframe": _detect_timeframe(filepath),
        "exists": filepath.exists(),
    }
    if not filepath.exists():
        result["error"] = "file does not exist"
        return result

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        result["error"] = f"read failed: {e}"
        return result

    result["rows"] = len(df)
    result["columns"] = list(df.columns)

    # Find timestamp column
    ts_col = None
    for candidate in ("datetime_utc", "datetime", "time", "timestamp", "date"):
        if candidate in df.columns:
            ts_col = candidate
            break
    if ts_col is None:
        result["error"] = "no timestamp column found"
        return result
    result["timestamp_column"] = ts_col

    # Parse timestamps (UTC)
    try:
        ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    except Exception as e:
        result["error"] = f"timestamp parse failed: {e}"
        return result

    n_unparseable = int(ts.isna().sum())
    result["unparseable_timestamps"] = n_unparseable

    df["_ts"] = ts
    df = df.sort_values("_ts").dropna(subset=["_ts"])

    # Sorted check (before sort)
    was_sorted = df["_ts"].is_monotonic_increasing if len(df) > 0 else True
    result["was_sorted_ascending"] = bool(was_sorted)

    # Duplicates
    n_dupes = int(df["_ts"].duplicated().sum())
    result["duplicate_timestamps"] = n_dupes
    df = df.drop_duplicates(subset=["_ts"])

    result["start"] = str(df["_ts"].iloc[0]) if len(df) > 0 else None
    result["end"] = str(df["_ts"].iloc[-1]) if len(df) > 0 else None

    # Timezone check
    if hasattr(df["_ts"].iloc[0], "tzinfo") and df["_ts"].iloc[0].tzinfo is not None:
        result["timezone"] = str(df["_ts"].iloc[0].tzinfo)
    else:
        result["timezone"] = "naive (no tzinfo)"
    result["timezone_utc"] = result["timezone"] == "UTC"

    # Detect gaps
    tf = result.get("timeframe")
    expected_minutes = TF_INTERVALS.get(tf, 60) if tf else 60
    expected_delta = pd.Timedelta(minutes=expected_minutes)

    real_gaps = []
    weekend_gaps = []
    if len(df) > 1:
        ts_series = df["_ts"].reset_index(drop=True)
        for i in range(1, len(ts_series)):
            delta = ts_series.iloc[i] - ts_series.iloc[i - 1]
            if delta > expected_delta * 1.5:
                if _is_weekend_gap(ts_series.iloc[i - 1], ts_series.iloc[i]):
                    weekend_gaps.append({
                        "after": str(ts_series.iloc[i - 1]),
                        "before": str(ts_series.iloc[i]),
                        "gap_hours": round(delta.total_seconds() / 3600, 1),
                    })
                else:
                    real_gaps.append({
                        "after": str(ts_series.iloc[i - 1]),
                        "before": str(ts_series.iloc[i]),
                        "gap_hours": round(delta.total_seconds() / 3600, 1),
                        "missing_bars_estimate": int(delta / expected_delta) - 1,
                    })
    result["real_gaps_count"] = len(real_gaps)
    result["weekend_gaps_count"] = len(weekend_gaps)
    result["real_gaps_sample"] = real_gaps[:5]
    result["missing_ranges"] = real_gaps  # full list for manifest

    # Spread availability
    if "spread" in df.columns:
        result["spread_available"] = True
        result["spread_nonzero_pct"] = round(
            100 * (df["spread"] > 0).sum() / max(len(df), 1), 2
        )
        if (df["spread"] > 0).any():
            result["spread_min"] = float(df.loc[df["spread"] > 0, "spread"].min())
            result["spread_max"] = float(df["spread"].max())
            result["spread_mean"] = round(float(df.loc[df["spread"] > 0, "spread"].mean()), 2)
        else:
            result["spread_note"] = "Column present but all values are 0 — spread NOT usable"
    else:
        result["spread_available"] = False
        result["spread_note"] = "Historical spread unavailable — backtest will use DEFAULT_SPREAD_PIPS table"

    # Real volume
    if "real_volume" in df.columns:
        result["real_volume_available"] = True
        result["real_volume_nonzero_pct"] = round(
            100 * (df["real_volume"] > 0).sum() / max(len(df), 1), 2
        )
    else:
        result["real_volume_available"] = False

    # Tick volume
    if "tick_volume" in df.columns:
        result["tick_volume_available"] = True
        result["tick_volume_nonzero_pct"] = round(
            100 * (df["tick_volume"] > 0).sum() / max(len(df), 1), 2
        )
    elif "volume" in df.columns:
        result["tick_volume_available"] = True
        result["tick_volume_nonzero_pct"] = round(
            100 * (df["volume"] > 0).sum() / max(len(df), 1), 2
        )
    else:
        result["tick_volume_available"] = False
        result["tick_volume_note"] = "Volume column missing — volume-based indicators (OBV, VWAP, CMF, MFI) will neutralize"

    # OHLC sanity
    ohlc_issues = []
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            ohlc_issues.append(f"missing column: {col}")
    if not ohlc_issues:
        for col in ("open", "high", "low", "close"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        bad_high = int((df["high"] < df[["open", "close"]].max(axis=1)).sum())
        bad_low = int((df["low"] > df[["open", "close"]].min(axis=1)).sum())
        neg = int((df[["open", "high", "low", "close"]] < 0).any(axis=1).sum())
        nan = int(df[["open", "high", "low", "close"]].isna().any(axis=1).sum())
        if bad_high > 0:
            ohlc_issues.append(f"{bad_high} rows where high < max(open, close)")
        if bad_low > 0:
            ohlc_issues.append(f"{bad_low} rows where low > min(open, close)")
        if neg > 0:
            ohlc_issues.append(f"{neg} rows with negative prices")
        if nan > 0:
            ohlc_issues.append(f"{nan} rows with NaN OHLC")
    result["ohlc_issues"] = ohlc_issues

    # Pass/fail
    errors = []
    if n_unparseable > 0:
        errors.append(f"{n_unparseable} unparseable timestamps")
    if n_dupes > 0:
        errors.append(f"{n_dupes} duplicate timestamps")
    if ohlc_issues:
        errors.extend(ohlc_issues)
    if not result.get("timezone_utc"):
        errors.append(f"timezone is {result.get('timezone')} (expected UTC)")
    result["valid"] = len(errors) == 0
    result["errors"] = errors

    return result


def find_csvs(data_dir: Path) -> list[Path]:
    """Find all CSV files in either nested or flat layout."""
    csvs = []
    # Nested: data/historical/{SYMBOL}/{TF}.csv
    nested_root = data_dir / "historical"
    if nested_root.exists():
        for sym_dir in sorted(nested_root.iterdir()):
            if sym_dir.is_dir():
                for csv in sorted(sym_dir.glob("*.csv")):
                    csvs.append(csv)
    # Flat: data/{SYMBOL}_{TF}.csv
    for csv in sorted(data_dir.glob("*_*.csv")):
        if "historical" not in csv.parts:
            csvs.append(csv)
    return csvs


def main():
    parser = argparse.ArgumentParser(description="Validate historical CSV files")
    parser.add_argument("--dir", type=str, default=str(PROJECT_ROOT / "data"),
                        help="Data directory to scan (default: data/)")
    parser.add_argument("--file", type=str, default=None,
                        help="Validate a single file (default: scan directory)")
    parser.add_argument("--out", type=str, default=str(PROJECT_ROOT / "data" / "historical" / "validation_report.json"),
                        help="Output JSON report path")
    args = parser.parse_args()

    if args.file:
        files = [Path(args.file)]
    else:
        files = find_csvs(Path(args.dir))

    print(f"\nValidating {len(files)} CSV file(s)\n")
    print(f"{'File':<45} {'Symbol':<8} {'TF':<5} {'Rows':>7} {'Spread':<10} {'TZ':<6} {'Status'}")
    print("-" * 100)

    all_results = []
    for csv in files:
        r = validate_csv(csv)
        all_results.append(r)
        if "error" in r and "exists" in r["error"]:
            print(f"{r['file']:<45} ERROR: {r['error']}")
            continue
        spread_str = f"yes({r.get('spread_nonzero_pct', 0):.0f}%)" if r.get("spread_available") else "NO"
        tz_str = "UTC" if r.get("timezone_utc") else r.get("timezone", "?")[:6]
        status = "✅ VALID" if r.get("valid") else f"❌ {len(r.get('errors', []))} errors"
        print(f"{r['file']:<45} {r.get('symbol','?'):<8} {r.get('timeframe','?'):<5} "
              f"{r.get('rows',0):>7} {spread_str:<10} {tz_str:<6} {status}")

    # Save JSON report
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nDetailed report: {out_path.relative_to(PROJECT_ROOT)}")

    # Summary
    valid_count = sum(1 for r in all_results if r.get("valid"))
    error_count = sum(1 for r in all_results if not r.get("valid") and "error" not in r)
    missing_count = sum(1 for r in all_results if "error" in r)
    total_rows = sum(r.get("rows", 0) for r in all_results)
    print(f"\nSummary:")
    print(f"  Valid:    {valid_count} files")
    print(f"  Errors:   {error_count} files")
    print(f"  Missing:  {missing_count} files")
    print(f"  Total rows: {total_rows:,}")

    if error_count > 0:
        print("\nFiles with errors:")
        for r in all_results:
            if not r.get("valid") and "error" not in r:
                print(f"  {r['file']}: {r.get('errors')}")

    # Update manifest if it exists
    manifest_path = PROJECT_ROOT / "data" / "historical" / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            # Augment manifest entries with validation results
            for entry in manifest.get("files", []):
                sym = entry.get("symbol")
                tf = entry.get("timeframe")
                for r in all_results:
                    if r.get("symbol") == sym and r.get("timeframe") == tf:
                        entry["validation"] = {
                            "valid": r.get("valid", False),
                            "real_gaps_count": r.get("real_gaps_count", 0),
                            "weekend_gaps_count": r.get("weekend_gaps_count", 0),
                            "spread_available": r.get("spread_available", False),
                            "spread_nonzero_pct": r.get("spread_nonzero_pct", 0),
                            "tick_volume_available": r.get("tick_volume_available", False),
                            "real_volume_available": r.get("real_volume_available", False),
                            "timezone_utc": r.get("timezone_utc", False),
                        }
                        entry["missing_ranges"] = r.get("missing_ranges", [])
                        break
            manifest["validation_run_at"] = datetime.now().isoformat()
            manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
            print(f"  Updated manifest: {manifest_path.relative_to(PROJECT_ROOT)}")
        except Exception as e:
            print(f"  Warning: could not update manifest: {e}")


if __name__ == "__main__":
    main()
