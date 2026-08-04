#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "data"
DEST_DIR = PROJECT_ROOT / "data" / "history"
FILE_RE = re.compile(r"^([A-Z0-9]+)_(M1|M5|M15|M30|H1|H4|D1)\.csv$")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default=None)
    args = parser.parse_args()
    wanted = set(args.symbols.upper().split(",")) if args.symbols else None

    imported = 0
    for csv_path in sorted(SRC_DIR.glob("*.csv")):
        m = FILE_RE.match(csv_path.name)
        if not m:
            continue
        symbol, tf = m.group(1), m.group(2)
        if wanted and symbol not in wanted:
            continue

        df = pd.read_csv(csv_path)
        if "datetime_utc" in df.columns:
            df = df.rename(columns={"datetime_utc": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        if "tick_volume" in df.columns and "volume" not in df.columns:
            df["volume"] = df["tick_volume"]

        keep = [c for c in ["timestamp", "open", "high", "low", "close", "volume",
                             "tick_volume", "spread", "real_volume"] if c in df.columns]
        df = df[keep].sort_values("timestamp").drop_duplicates(subset=["timestamp"])

        out_dir = DEST_DIR / symbol
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{symbol}_{tf}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  {symbol} {tf}: {len(df):,} rows -> {out_path}")
        imported += 1

    print(f"\nImported {imported} symbol/timeframe file(s) into {DEST_DIR}")
    print("Now run:  python -m ml.train_historical --skip-rl")

if __name__ == "__main__":
    main()