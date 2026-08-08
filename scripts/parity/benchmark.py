"""
scripts/parity/benchmark.py — Baseline performance benchmark for backtest.

Runs the unified_engine backtest on a synthetic dataset and measures:
  - Total bars processed
  - Total wall-clock time
  - Bars per second

This is the BEFORE baseline. After Phase 3 optimization, re-run this
script and compare to verify speed improvement.

Usage:
    python scripts/parity/benchmark.py --bars 200
    python scripts/parity/benchmark.py --bars 500 --symbol EURUSD

Output:
    scripts/parity/benchmark_baseline.json (machine-readable)
    stdout (human-readable summary)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def make_synthetic_bars(symbol: str, bars: int, start_price: float, seed: int = 42) -> pd.DataFrame:
    """Deterministic synthetic OHLCV data."""
    rng = np.random.default_rng(seed)
    if symbol == "XAUUSD":
        vol = 1.5; digits = 2
    elif symbol.endswith("JPY"):
        vol = 0.10; digits = 3
    else:
        vol = 0.0008; digits = 5
    returns = rng.normal(0, 1, bars) * vol
    closes = np.round(start_price + np.cumsum(returns), digits)
    opens = np.roll(closes, 1); opens[0] = start_price
    intrabar = np.abs(rng.normal(0, vol * 0.3, bars))
    highs = np.round(np.maximum(opens, closes) + intrabar, digits)
    lows = np.round(np.minimum(opens, closes) - intrabar, digits)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    times = [start + timedelta(hours=i) for i in range(bars)]
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": rng.integers(100, 5000, bars),
    }, index=pd.DatetimeIndex(times, name="time"))


def run_benchmark(symbol: str, bars: int, warmup: int = 50) -> dict:
    """Run the unified_engine backtest and return timing info.

    Note: uses warmup=50 (not 300) for the baseline because pandas-ta
    isn't installed in this env, so indicators degrade gracefully and
    we don't need the full 300-bar warmup to start producing decisions.
    The post-Phase-3 benchmark should use the same warmup for fair
    comparison.
    """
    df = make_synthetic_bars(symbol, bars + warmup, 1.0850 if symbol != "XAUUSD" else 2010.0)

    # Try to import the unified engine — if it fails due to missing deps,
    # fall back to a minimal "simulate the per-bar loop" benchmark.
    try:
        from backtest.unified_engine import run_unified_backtest
        from core.constants import set_backtest_mode
        set_backtest_mode(True)

        t0 = time.perf_counter()
        try:
            result = run_unified_backtest(
                symbol=symbol, df=df, timeframe="H1",
                starting_balance=10000.0,
                warmup_bars=warmup,
                max_open_trades=10,
                max_hold_bars=100,
                save_forensics=False,
                verbose=False,
                db_path=":memory:",
            )
        except Exception as e:
            return {
                "symbol": symbol, "bars": bars, "warmup": warmup,
                "error": f"run_unified_backtest failed: {e}",
                "elapsed_sec": time.perf_counter() - t0,
            }
        elapsed = time.perf_counter() - t0

        return {
            "symbol": symbol,
            "bars_requested": bars,
            "bars_processed": result.bars if hasattr(result, "bars") else bars,
            "warmup": warmup,
            "elapsed_sec": round(elapsed, 3),
            "bars_per_sec": round(bars / elapsed, 2) if elapsed > 0 else 0,
            "trades": len(result.trades) if hasattr(result, "trades") else 0,
            "rejection_stats": result.rejection_stats if hasattr(result, "rejection_stats") else {},
            "error": result.error if hasattr(result, "error") and result.error else None,
        }
    except ImportError as e:
        # Fallback: simulate the per-bar cost using just HistoricalMT5Provider
        # (the dominant cost is indicator recompute per bar)
        return _fallback_benchmark(symbol, df, warmup)


def _fallback_benchmark(symbol: str, df: pd.DataFrame, warmup: int) -> dict:
    """Minimal benchmark: just run HistoricalMT5Provider.get_market_out
    per bar to measure the indicator-recompute cost."""
    from core.data_provider import HistoricalMT5Provider
    from core.constants import set_backtest_mode
    set_backtest_mode(True)

    provider = HistoricalMT5Provider(df, symbol, "H1")
    n_bars = len(df) - warmup

    t0 = time.perf_counter()
    for i in range(warmup, len(df)):
        provider.advance_to(i)
        try:
            provider.get_market_out(symbol, "H1")
        except Exception:
            pass
    elapsed = time.perf_counter() - t0

    return {
        "symbol": symbol,
        "bars_requested": n_bars,
        "bars_processed": n_bars,
        "warmup": warmup,
        "elapsed_sec": round(elapsed, 3),
        "bars_per_sec": round(n_bars / elapsed, 2) if elapsed > 0 else 0,
        "mode": "fallback_provider_only",
        "note": "pandas-ta / unified_engine deps not installed — ran HistoricalMT5Provider only",
    }


def main():
    parser = argparse.ArgumentParser(description="Backtest performance baseline")
    parser.add_argument("--bars", type=int, default=200,
                        help="Number of bars to backtest (excludes warmup)")
    parser.add_argument("--warmup", type=int, default=50,
                        help="Warmup bars (lower than 300 for env without pandas-ta)")
    parser.add_argument("--symbol", type=str, default="EURUSD")
    parser.add_argument("--out", type=str, default="scripts/parity/benchmark_baseline.json")
    args = parser.parse_args()

    print(f"Running baseline benchmark: {args.symbol} {args.bars} bars (warmup={args.warmup})")
    result = run_benchmark(args.symbol, args.bars, warmup=args.warmup)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))

    print()
    print("=" * 60)
    print("BASELINE BENCHMARK RESULT")
    print("=" * 60)
    print(f"Symbol:         {result.get('symbol', '?')}")
    print(f"Bars:           {result.get('bars_processed', '?')}")
    print(f"Warmup:         {result.get('warmup', '?')}")
    print(f"Elapsed (sec):  {result.get('elapsed_sec', '?')}")
    print(f"Bars/sec:       {result.get('bars_per_sec', '?')}")
    print(f"Trades:         {result.get('trades', 'n/a')}")
    if result.get("mode"):
        print(f"Mode:           {result['mode']}")
    if result.get("error"):
        print(f"Error:          {result['error']}")
    print("=" * 60)
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
