#!/usr/bin/env python3
"""
Run the REAL unified backtest engine on CSV data.

This uses backtest.unified_engine.run_unified_backtest() which is the
SAME pipeline Demo/Real uses (AITrader.evaluate_decision_core). It is
NOT a separate strategy-object backtester.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path("/home/z/my-project/repos/forex-agent")
sys.path.insert(0, str(PROJECT_ROOT))

# Set backtest mode env BEFORE any imports that read it
os.environ["BACKTEST_MODE"] = "1"
os.environ["SIMULATION_MODE"] = "true"
os.environ["TEST_MODE"] = "false"
os.environ["ECONCAL_OUTAGE_ALLOWS_TRADES"] = "true"
os.environ["ML_MODEL_CONSISTENCY_ACTION"] = "warn"

# Minimal logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
)
log = logging.getLogger("real_backtest")

# Suppress noisy loggers
for name in ["urllib3", "httpx", "groq", "google_genai", "chromadb",
             "sentence_transformers", "matplotlib", "PIL", "asyncio"]:
    logging.getLogger(name).setLevel(logging.ERROR)

warnings.filterwarnings("ignore")


def load_csv(pair: str, timeframe: str) -> pd.DataFrame:
    """Load CSV from data/{PAIR}_{TF}.csv."""
    path = PROJECT_ROOT / "data" / f"{pair}_{timeframe}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    # Normalize: first column is timestamp
    cols = {c.lower(): c for c in df.columns}
    if "datetime_utc" in cols:
        df = df.rename(columns={cols["datetime_utc"]: "time"})
    elif "datetime" in cols:
        df = df.rename(columns={cols["datetime"]: "time"})
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    df = df.dropna(subset=["time"]).sort_values("time").set_index("time")
    for c in ("open", "high", "low", "close", "tick_volume", "spread"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "tick_volume" not in df.columns:
        df["tick_volume"] = 1000
    if "volume" not in df.columns:
        df["volume"] = df["tick_volume"]
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="EURUSD,GBPUSD,USDJPY",
                        help="comma-separated pairs")
    parser.add_argument("--timeframe", default="H1")
    parser.add_argument("--bars", type=int, default=500,
                        help="max bars to replay (after warmup)")
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--balance", type=float, default=10000.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    pairs = [p.strip().upper() for p in args.pairs.split(",")]

    # Import here so env vars apply first
    from backtest.unified_engine import run_unified_backtest

    all_results = []

    for pair in pairs:
        log.warning(f"\n{'='*70}")
        log.warning(f"  BACKTEST: {pair} {args.timeframe}")
        log.warning(f"{'='*70}")

        df = load_csv(pair, args.timeframe)
        if df is None or len(df) < args.warmup + 100:
            log.error(f"  {pair} {args.timeframe}: no CSV or < {args.warmup+100} rows")
            continue
        log.warning(f"  Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

        # Limit to last N bars if requested
        if args.bars and len(df) > args.warmup + args.bars:
            df = df.iloc[-(args.warmup + args.bars):]
            log.warning(f"  Truncated to last {len(df)} bars")
        else:
            log.warning(f"  Using {len(df)} bars (warmup={args.warmup} + replay={len(df)-args.warmup})")

        t0 = time.time()
        try:
            result = run_unified_backtest(
                symbol=pair,
                df=df,
                timeframe=args.timeframe,
                starting_balance=args.balance,
                warmup_bars=args.warmup,
                max_open_trades=3,
                max_hold_bars=100,
                verbose=args.verbose,
                save_forensics=False,
                db_path=f"/tmp/bt_{pair}_{args.timeframe}.db",
            )
            elapsed = time.time() - t0

            n_trades = len(result.trades)
            wins = sum(1 for t in result.trades if t.pnl_usd > 0)
            losses = sum(1 for t in result.trades if t.pnl_usd <= 0)
            wr = (wins / n_trades * 100) if n_trades > 0 else 0.0
            total_pnl = sum(t.pnl_usd for t in result.trades)
            final_balance = result.equity_curve[-1] if result.equity_curve else args.balance

            # Average RR (need to inspect trade objects)
            avg_rr = 0.0
            if n_trades > 0:
                rrs = []
                for t in result.trades:
                    # try various attribute names
                    rr = getattr(t, "rr_ratio", None) or getattr(t, "rr", None)
                    if rr is None:
                        # compute from pnl_pips vs stop_pips if available
                        pnl_pips = getattr(t, "pnl_pips", 0) or 0
                        stop_pips = getattr(t, "stop_pips", 0) or 0
                        if stop_pips and stop_pips > 0:
                            rr = abs(pnl_pips) / stop_pips
                    if rr:
                        rrs.append(rr)
                avg_rr = sum(rrs) / len(rrs) if rrs else 0.0

            log.warning(f"\n  RESULT: {pair} {args.timeframe}")
            log.warning(f"  Bars replayed : {result.bars}")
            log.warning(f"  Trades closed : {n_trades}")
            log.warning(f"  Wins/Losses   : {wins}/{losses}")
            log.warning(f"  Winrate       : {wr:.1f}%")
            log.warning(f"  Avg R:R       : {avg_rr:.2f}")
            log.warning(f"  Net P&L       : ${total_pnl:+.2f}")
            log.warning(f"  Final balance : ${final_balance:.2f}")
            log.warning(f"  Rejection stats: {result.rejection_stats}")
            log.warning(f"  Elapsed       : {elapsed:.1f}s")

            if result.error:
                log.error(f"  ERROR: {result.error}")

            all_results.append({
                "pair": pair,
                "timeframe": args.timeframe,
                "bars": result.bars,
                "trades": n_trades,
                "wins": wins,
                "losses": losses,
                "winrate": wr,
                "avg_rr": avg_rr,
                "net_pnl": total_pnl,
                "final_balance": final_balance,
                "rejection_stats": result.rejection_stats,
                "error": result.error,
                "elapsed_sec": round(elapsed, 1),
            })

        except Exception as e:
            import traceback
            log.error(f"  {pair} {args.timeframe} crashed: {e}")
            log.error(traceback.format_exc())

    # Summary
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print(f"  {'Pair':8s} {'TF':4s} {'Bars':6s} {'Trades':7s} {'WR':7s} {'RR':6s} {'PnL':10s} {'Bal':10s}")
    for r in all_results:
        print(f"  {r['pair']:8s} {r['timeframe']:4s} {r['bars']:6d} "
              f"{r['trades']:7d} {r['winrate']:6.1f}% {r['avg_rr']:5.2f} "
              f"${r['net_pnl']:+9.2f} ${r['final_balance']:9.2f}")
    print("=" * 80)

    # Save to CSV
    out_path = "/home/z/my-project/download/real_backtest_results.csv"
    pd.DataFrame(all_results).to_csv(out_path, index=False)
    print(f"  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
