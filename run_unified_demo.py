#!/usr/bin/env python3
"""
run_unified_demo.py — End-to-End Demonstration of the Unified Signal Engine
============================================================================

This script demonstrates the complete 5-engine unified signal system on
either real MT5 data or synthetic OHLC data. It shows:
  1. S/R Zone detection
  2. Stop Hunt reversal signals
  3. ICT/SMC AMD+FVG+MSS pipeline (1:6 R:R)
  4. Multi-Strategy PA (8-step + session filter + MTF)
  5. High-Reliability Candlestick Patterns (20-pattern library)
  6. Consensus voting across all engines

Usage:
    # Synthetic data demo (default)
    python run_unified_demo.py

    # Specific pair + timeframe
    python run_unified_demo.py --pair EURUSD --timeframe 4H

    # Real MT5 data (requires MT5 terminal running on Windows)
    python run_unified_demo.py --source mt5 --pair USDJPY --timeframe 1H

    # Output JSON only (no pretty text)
    python run_unified_demo.py --json
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


def make_synthetic_ohlc(
    n: int = 200,
    base: float = 1.0850,
    seed: int = 42,
    vol: float = 0.0008,
    freq: str = "4h",
    start: str = "2024-06-03 06:00",
) -> pd.DataFrame:
    """Generate synthetic OHLC data for demonstration."""
    np.random.seed(seed)
    dates = pd.date_range(start, periods=n, freq=freq)
    close = base + np.cumsum(np.random.randn(n) * vol)
    return pd.DataFrame({
        "open":  close + np.random.randn(n) * vol * 0.3,
        "high":  close + abs(np.random.randn(n)) * vol * 1.5,
        "low":   close - abs(np.random.randn(n)) * vol * 1.5,
        "close": close,
    }, index=dates)


def fetch_mt5_data(symbol: str, timeframe: str, bars: int = 200):
    """Fetch real OHLC data from MT5 (Windows only)."""
    try:
        import MetaTrader5 as mt5
        from data.fetcher import DataFetcher
    except ImportError:
        print("ERROR: MetaTrader5 package not installed. Use --source synthetic.")
        sys.exit(1)

    tf_map = {
        "1H": mt5.TIMEFRAME_H1,
        "4H": mt5.TIMEFRAME_H4,
        "1D": mt5.TIMEFRAME_D1,
        "M30": mt5.TIMEFRAME_M30,
    }
    mt5_tf = tf_map.get(timeframe.upper(), mt5.TIMEFRAME_H4)

    if not mt5.initialize():
        print(f"ERROR: MT5 initialize failed: {mt5.last_error()}")
        sys.exit(1)

    rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, bars)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        print(f"ERROR: No data returned from MT5 for {symbol}")
        sys.exit(1)

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df.rename(columns={
        "open": "open", "high": "high", "low": "low",
        "close": "close", "tick_volume": "volume",
    }, inplace=True)
    return df[["open", "high", "low", "close", "volume"]]


def fetch_lower_tf(symbol: str, primary_tf: str, bars: int = 400):
    """Fetch lower timeframe data for MTF confirmation."""
    tf_to_lower = {"4H": "2H", "1H": "M30"}
    lower_tf = tf_to_lower.get(primary_tf.upper())
    if not lower_tf:
        return None
    try:
        return fetch_mt5_data(symbol, lower_tf, bars)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Unified Signal Engine Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--pair", default="EURUSD",
                        help="Trading pair (default: EURUSD)")
    parser.add_argument("--timeframe", default="4H",
                        choices=["1H", "4H", "1D", "M30"],
                        help="Primary timeframe (default: 4H)")
    parser.add_argument("--source", default="synthetic",
                        choices=["synthetic", "mt5"],
                        help="Data source (default: synthetic)")
    parser.add_argument("--bars", type=int, default=200,
                        help="Number of bars to fetch (default: 200)")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON only (no pretty text)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for synthetic data (default: 42)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  UNIFIED SIGNAL ENGINE — DEMO")
    print(f"  Pair: {args.pair} | Timeframe: {args.timeframe} | Source: {args.source}")
    print(f"{'='*60}\n")

    # ── Fetch data ──
    if args.source == "mt5":
        print("Fetching MT5 data...")
        df = fetch_mt5_data(args.pair, args.timeframe, args.bars)
        lower_df = fetch_lower_tf(args.pair, args.timeframe)
    else:
        freq_map = {"1H": "1h", "4H": "4h", "1D": "D", "M30": "30min"}
        freq = freq_map.get(args.timeframe, "4h")
        df = make_synthetic_ohlc(
            n=args.bars, seed=args.seed, vol=0.0008, freq=freq,
        )
        # Build a lower TF for MTF confirmation
        if args.timeframe == "4H":
            lower_df = make_synthetic_ohlc(
                n=args.bars * 2, seed=args.seed + 1, vol=0.0006, freq="2h",
            )
        elif args.timeframe == "1H":
            lower_df = make_synthetic_ohlc(
                n=args.bars * 2, seed=args.seed + 1, vol=0.0004, freq="30min",
            )
        else:
            lower_df = None

    print(f"Data loaded: {len(df)} candles")
    print(f"Date range: {df.index[0]} → {df.index[-1]}")
    print(f"Price range: {df['low'].min():.5f} – {df['high'].max():.5f}")
    print(f"Current close: {df['close'].iloc[-1]:.5f}")
    print()

    # ── Run Unified Signal Engine ──
    from analysis.unified_signal_engine import UnifiedSignalEngine

    engine = UnifiedSignalEngine(timeframe=args.timeframe)
    result = engine.analyze(df, symbol=args.pair, lower_tf_df=lower_df)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(engine.to_prompt_text(result))

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    consensus = result.get("consensus", {})
    print(f"  Consensus Action    : {consensus.get('action', 'N/A')}")
    print(f"  Confidence          : {consensus.get('confidence', 'N/A')}")
    print(f"  Buy Score           : {consensus.get('buy_score', 0):.1f}")
    print(f"  Sell Score          : {consensus.get('sell_score', 0):.1f}")
    print(f"  Patterns Detected   : {len(result.get('detected_patterns', []))}")
    print(f"  S/R Zones           : {len(result.get('zones', {}).get('support_resistance', []))}")

    # Per-engine summary
    print(f"\n  Per-Engine Signals:")
    engines = [
        ("StopHunt",        result.get("stop_hunt", {}).get("signal", {})),
        ("ICT/AMD",         result.get("ict_amd", {}).get("signal", {})),
        ("Multi-Strategy PA", result.get("multi_strategy_pa", {}).get("signal", {})),
    ]
    for name, sig in engines:
        action = sig.get("action", "N/A")
        conf = sig.get("confidence", "N/A")
        print(f"    {name:20s}: {action:8s} (conf={conf})")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
