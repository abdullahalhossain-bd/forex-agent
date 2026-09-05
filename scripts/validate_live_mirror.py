"""Validation harness for the strict live-mirror backtest.

Examples:
  python scripts/validate_live_mirror.py --csv data/EURUSD_M15.csv --symbol EURUSD --timeframe M15 --bars 500

The harness intentionally fails closed. It reports deterministic parity, input
leakage, and replay-engine errors rather than changing thresholds to improve
metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from backtest.live_mirror import run_live_mirror_backtest, validate_historical_ohlcv


def _fingerprint(result) -> str:
    payload = {
        "trades": result.trades,
        "equity_curve": result.equity_curve,
        "rejection_stats": result.rejection_stats,
        "metrics": getattr(result.metrics, "__dict__", result.metrics),
        "error": result.error,
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--timeframe", default="M15")
    ap.add_argument("--bars", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=300)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    time_col = next((c for c in ("time", "timestamp", "datetime", "date") if c in df.columns), None)
    if time_col is None:
        raise SystemExit("CSV needs time/timestamp/datetime/date column")
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.set_index(time_col).sort_index()
    df.columns = [str(c).lower() for c in df.columns]
    if args.bars:
        df = df.tail(args.bars)
    validate_historical_ohlcv(df)

    first = run_live_mirror_backtest(symbol=args.symbol, df=df, timeframe=args.timeframe,
                                     warmup_bars=args.warmup,
                                     db_path="backtest/results/validation_run_1.db")
    second = run_live_mirror_backtest(symbol=args.symbol, df=df, timeframe=args.timeframe,
                                      warmup_bars=args.warmup,
                                      db_path="backtest/results/validation_run_2.db")

    f1, f2 = _fingerprint(first), _fingerprint(second)
    deterministic = f1 == f2
    report = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "rows": len(df),
        "start": str(df.index[0]),
        "end": str(df.index[-1]),
        "deterministic": deterministic,
        "fingerprint_1": f1,
        "fingerprint_2": f2,
        "run_1_error": first.error,
        "run_2_error": second.error,
        "run_1_metrics": getattr(first.metrics, "__dict__", first.metrics),
        "run_2_metrics": getattr(second.metrics, "__dict__", second.metrics),
        "acceptance": "PASS" if deterministic and not first.error and not second.error else "FAIL",
    }
    out = Path("backtest/results/live_mirror_validation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["acceptance"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
