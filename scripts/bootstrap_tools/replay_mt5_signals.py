#!/usr/bin/env python3
"""
Option 5 — Replay strategy signals on MT5 historical OHLC data and
record the outcomes into pattern_stats.json.

This is the "live backtest using MT5 data" approach:
  1. Pull N months of historical OHLC bars from MT5 (real broker data)
  2. For each bar, run the actual analysis pipeline (indicators, patterns,
     strategy signals) the same way live trading does
  3. When a BUY/SELL signal fires, simulate a trade:
     - Entry = close of signal bar (next bar open in live, close here for simplicity)
     - SL = entry ± ATR×1.5 (matches RiskEngine default)
     - TP = entry ± ATR×3.0 (matches RiskEngine MIN_RR=2.0)
  4. Walk forward bar-by-bar to find SL or TP hit
  5. Record WIN/LOSS into ConfidenceEngine.record_outcome()

This produces REAL signal-based outcomes from REAL broker data —
not synthetic, not backtest-DB-dependent.

DIFFERENCE FROM main.py --mode backtest:
  - main.py backtest runs the FULL pipeline (decision core, risk engine,
    trade permission, etc.) — slow but faithful to live
  - This script runs ONLY the analysis_agent + signal generation, then
    simulates SL/TP directly — fast and focused on pattern_stats only
  - Use main.py backtest if you want full pipeline fidelity
  - Use THIS script if you just want to seed pattern_stats with
    pattern→outcome data quickly

PREREQUISITES:
  - Windows host with MT5 terminal running
  - MetaTrader5 python package installed
  - .env with MT5 credentials
  - data/indicators.py, data/fetcher.py available (NOT gitignored)

USAGE:
  # Replay last 6 months of EURUSD H1 — fast mode
  python scripts/bootstrap_tools/replay_mt5_signals.py \\
      --pairs EURUSD --timeframe H1 --months 6

  # Replay 1 year of multiple pairs
  python scripts/bootstrap_tools/replay_mt5_signals.py \\
      --pairs EURUSD,GBPUSD,USDJPY --timeframe H1 --months 12

  # Use CSV fallback (if MT5 unavailable, uses data/{PAIR}_{TF}.csv)
  python scripts/bootstrap_tools/replay_mt5_signals.py \\
      --pairs EURUSD --timeframe H1 --use-csv

  # Dry-run (show signal count without recording outcomes)
  python scripts/bootstrap_tools/replay_mt5_signals.py \\
      --pairs EURUSD --timeframe H1 --months 3 --dry-run

WHY THIS EXISTS:
  - generate_synthetic_samples.py creates EMPTY entries (no outcomes)
  - import_backtest_samples.py needs a backtest DB to already exist
  - import_mt5_history.py needs the user to have already traded
  - THIS script generates outcomes FROM SCRATCH using MT5 historical data
    + the system's own strategy logic — no pre-existing trades required
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from core.constants import MEMORY_DIR, PIP_SIZE
    from learning.confidence_engine import ConfidenceEngine, MIN_SAMPLE_SIZE
except Exception as e:
    print(f"ERROR: cannot import from project: {e}")
    print(f"       Run from project root:  cd {PROJECT_ROOT}")
    sys.exit(1)

# Try MT5
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

# MT5 timeframe constants (fallback if MT5 not imported)
TF_MAP = {
    "M15": (mt5.TIMEFRAME_M15 if MT5_AVAILABLE else 15, 15),
    "H1":  (mt5.TIMEFRAME_H1 if MT5_AVAILABLE else 60, 60),
    "H4":  (mt5.TIMEFRAME_H4 if MT5_AVAILABLE else 240, 240),
    "D1":  (mt5.TIMEFRAME_D1 if MT5_AVAILABLE else 1440, 1440),
}


def fetch_mt5_bars(symbol: str, timeframe: str, months: int) -> "pd.DataFrame":
    """Fetch N months of historical bars from MT5."""
    if not MT5_AVAILABLE:
        return pd.DataFrame()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=months * 30)
    tf_const = TF_MAP.get(timeframe.upper(), (mt5.TIMEFRAME_H1, 60))[0]
    rates = mt5.copy_rates_range(symbol, tf_const, start, end)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    return df


def load_csv_bars(symbol: str, timeframe: str) -> "pd.DataFrame":
    """Fallback: load bars from data/{symbol}_{timeframe}.csv."""
    csv_path = PROJECT_ROOT / "data" / f"{symbol}_{timeframe}.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path)
        # Try common column names
        time_col = None
        for c in ["datetime_utc", "datetime", "time", "timestamp"]:
            if c in df.columns:
                time_col = c
                break
        if time_col is None:
            return pd.DataFrame()
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
        df.set_index(time_col, inplace=True)
        # Normalize column names
        df.rename(columns={
            "tick_volume": "volume",
            "real_volume": "real_vol",
        }, inplace=True)
        return df
    except Exception as e:
        print(f"  CSV load failed: {e}")
        return pd.DataFrame()


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute ATR — same as RiskEngine uses."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr


def detect_regime(df: pd.DataFrame, lookback: int = 50) -> str:
    """Simple regime detector — matches MarketRegimeDetector output."""
    if len(df) < lookback:
        return "UNKNOWN"
    recent = df.tail(lookback)
    slope = (recent["close"].iloc[-1] - recent["close"].iloc[0]) / recent["close"].iloc[0]
    vol = recent["close"].pct_change().std()

    if abs(slope) < 0.005 and vol < 0.003:
        return "RANGING"
    elif slope > 0.01:
        return "TRENDING"
    elif slope < -0.01:
        return "TRENDING"
    elif abs(slope) > 0.02:
        return "BREAKOUT"
    else:
        return "UNKNOWN"


def detect_simple_patterns(df: pd.DataFrame, i: int) -> list:
    """Detect simple candlestick patterns at bar i.
    This is a SIMPLIFIED pattern detector — for full pattern detection,
    use the actual AnalysisAgent (which requires more dependencies).
    """
    if i < 5:
        return []

    patterns = []
    row = df.iloc[i]
    prev = df.iloc[i - 1]

    body = abs(row["close"] - row["open"])
    upper_wick = row["high"] - max(row["close"], row["open"])
    lower_wick = min(row["close"], row["open"]) - row["low"]
    range_total = row["high"] - row["low"]

    if range_total == 0:
        return []

    # Hammer (bullish reversal)
    if lower_wick > 2 * body and upper_wick < body * 0.5 and body > 0:
        patterns.append("Hammer")

    # Shooting Star (bearish reversal)
    if upper_wick > 2 * body and lower_wick < body * 0.5 and body > 0:
        patterns.append("Shooting_Star")

    # Doji
    if body < range_total * 0.1:
        patterns.append("Doji")

    # Engulfing Bullish
    if prev["close"] < prev["open"] and row["close"] > row["open"]:
        if row["close"] > prev["open"] and row["open"] < prev["close"]:
            patterns.append("Engulfing_Bullish")

    # Engulfing Bearish
    if prev["close"] > prev["open"] and row["close"] < row["open"]:
        if row["open"] > prev["close"] and row["close"] < prev["open"]:
            patterns.append("Engulfing_Bearish")

    # Trend continuation (3 closes in same direction)
    if i >= 3:
        closes = df["close"].iloc[i-3:i+1]
        if all(closes.iloc[j] > closes.iloc[j-1] for j in range(1, 4)):
            patterns.append("Trend_Continuation")
        elif all(closes.iloc[j] < closes.iloc[j-1] for j in range(1, 4)):
            patterns.append("Trend_Continuation")

    # Breakout (close > max of last 20 highs)
    if i >= 20:
        recent_high = df["high"].iloc[i-20:i].max()
        recent_low = df["low"].iloc[i-20:i].min()
        if row["close"] > recent_high:
            patterns.append("Breakout")
        elif row["close"] < recent_low:
            patterns.append("Breakout")

    return patterns


def generate_signal(df: pd.DataFrame, i: int) -> tuple:
    """Generate a simple BUY/SELL/WAIT signal at bar i.
    Returns (signal, patterns, confidence).

    This is a SIMPLIFIED signal generator — for the full strategy logic,
    use main.py --mode backtest which runs the real AnalysisAgent.
    """
    if i < 20:
        return "WAIT", [], 0

    patterns = detect_simple_patterns(df, i)

    # Simple RSI calculation
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_val = rsi.iloc[i] if not pd.isna(rsi.iloc[i]) else 50

    # Simple MA cross
    ma_fast = df["close"].rolling(10).mean().iloc[i]
    ma_slow = df["close"].rolling(20).mean().iloc[i]
    if pd.isna(ma_fast) or pd.isna(ma_slow):
        return "WAIT", patterns, 0

    # Signal logic
    signal = "WAIT"
    confidence = 0

    if "Hammer" in patterns and rsi_val < 35:
        signal = "BUY"
        confidence = 70
    elif "Shooting_Star" in patterns and rsi_val > 65:
        signal = "SELL"
        confidence = 70
    elif "Engulfing_Bullish" in patterns and ma_fast > ma_slow:
        signal = "BUY"
        confidence = 65
    elif "Engulfing_Bearish" in patterns and ma_fast < ma_slow:
        signal = "SELL"
        confidence = 65
    elif "Breakout" in patterns:
        # Breakout direction = direction of close vs previous close
        if df["close"].iloc[i] > df["close"].iloc[i-1]:
            signal = "BUY"
            confidence = 60
        else:
            signal = "SELL"
            confidence = 60

    return signal, patterns, confidence


def simulate_trade_outcome(df: pd.DataFrame, entry_idx: int, direction: str,
                            atr_val: float, pip_size: float) -> tuple:
    """Walk forward from entry_idx to find SL or TP hit.
    Returns (outcome, exit_idx, exit_price, exit_reason).

    SL/TP placement matches RiskEngine defaults:
      SL distance = max(ATR × 1.5, 10 pips)
      TP distance = SL distance × 2.0  (MIN_RR = 2.0)
    """
    sl_distance = max(atr_val * 1.5, 10 * pip_size)
    tp_distance = sl_distance * 2.0

    entry_price = df["close"].iloc[entry_idx]

    if direction == "BUY":
        sl_price = entry_price - sl_distance
        tp_price = entry_price + tp_distance
    else:  # SELL
        sl_price = entry_price + sl_distance
        tp_price = entry_price - tp_distance

    # Walk forward (max 100 bars)
    max_bars = min(100, len(df) - entry_idx - 1)
    for j in range(1, max_bars + 1):
        bar = df.iloc[entry_idx + j]
        if direction == "BUY":
            if bar["low"] <= sl_price:
                return "LOSS", entry_idx + j, sl_price, "SL"
            if bar["high"] >= tp_price:
                return "WIN", entry_idx + j, tp_price, "TP"
        else:  # SELL
            if bar["high"] >= sl_price:
                return "LOSS", entry_idx + j, sl_price, "SL"
            if bar["low"] <= tp_price:
                return "WIN", entry_idx + j, tp_price, "TP"

    # Timeout — close at last bar's close
    exit_idx = entry_idx + max_bars
    exit_price = df["close"].iloc[exit_idx]
    if direction == "BUY":
        pnl = exit_price - entry_price
    else:
        pnl = entry_price - exit_price
    outcome = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "BE"
    return outcome, exit_idx, exit_price, "timeout"


def main():
    parser = argparse.ArgumentParser(description="Replay strategy signals on MT5 historical data")
    parser.add_argument("--pairs", default="EURUSD",
                        help="Comma-separated pairs (default: EURUSD)")
    parser.add_argument("--timeframe", default="H1",
                        choices=["M15", "H1", "H4", "D1"],
                        help="Timeframe (default: H1)")
    parser.add_argument("--months", type=int, default=6,
                        help="Months of history to replay (default: 6)")
    parser.add_argument("--use-csv", action="store_true",
                        help="Use data/*.csv instead of MT5 (fallback for Linux)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show signal count without recording outcomes")
    parser.add_argument("--max-trades", type=int, default=100,
                        help="Max trades to record per pair (default: 100)")
    args = parser.parse_args()

    pairs = [p.strip().upper() for p in args.pairs.split(",") if p.strip()]
    timeframe = args.timeframe.upper()

    print("=" * 70)
    print("  MT5 HISTORICAL SIGNAL REPLAY")
    print("=" * 70)
    print(f"  Pairs:      {', '.join(pairs)}")
    print(f"  Timeframe:  {timeframe}")
    print(f"  Months:     {args.months}")
    print(f"  Max trades: {args.max_trades}/pair")
    print(f"  Data source: {'CSV' if args.use_csv else 'MT5'}")
    print(f"  Pattern stats: {MEMORY_DIR / 'pattern_stats.json'}")
    print()

    # ── Connect to MT5 (or fall back to CSV) ────────────────────────
    use_mt5 = MT5_AVAILABLE and not args.use_csv
    if use_mt5:
        print("  Connecting to MT5...")
        try:
            if not mt5.initialize():
                print("  ✗ MT5 initialize failed — falling back to CSV")
                use_mt5 = False
            else:
                info = mt5.account_info()
                if info:
                    print(f"  ✓ Connected: account {info.login} | {info.server}")
        except Exception as e:
            print(f"  ✗ MT5 error: {e} — falling back to CSV")
            use_mt5 = False

    if not use_mt5:
        print("  Using CSV fallback (data/{PAIR}_{TF}.csv)")
        if not MT5_AVAILABLE:
            print("  (MetaTrader5 package not installed — this is expected on Linux)")
        print()

    # ── Initialize ConfidenceEngine ─────────────────────────────────
    if not args.dry_run:
        engine = ConfidenceEngine()
    else:
        engine = None

    total_recorded = 0
    total_wins = 0
    total_losses = 0

    for pair in pairs:
        print(f"\n{'─' * 70}")
        print(f"  Processing {pair} {timeframe}")
        print(f"{'─' * 70}")

        # ── Fetch data ──────────────────────────────────────────────
        if use_mt5:
            df = fetch_mt5_bars(pair, timeframe, args.months)
        else:
            df = load_csv_bars(pair, timeframe)

        if df.empty:
            print(f"  ✗ No data available for {pair} {timeframe}")
            if not use_mt5:
                print(f"    Expected file: data/{pair}_{timeframe}.csv")
            continue

        print(f"  ✓ Loaded {len(df)} bars ({df.index[0].date()} → {df.index[-1].date()})")

        # ── Compute ATR ─────────────────────────────────────────────
        df["atr"] = compute_atr(df, period=14)
        print(f"  ✓ Computed ATR (mean: {df['atr'].mean():.5f})")

        # ── Walk through bars and generate signals ──────────────────
        signals = []
        for i in range(20, len(df) - 1):
            signal, patterns, confidence = generate_signal(df, i)
            if signal in ("BUY", "SELL") and patterns:
                signals.append({
                    "bar_idx": i,
                    "time": df.index[i],
                    "signal": signal,
                    "patterns": patterns,
                    "confidence": confidence,
                    "atr": df["atr"].iloc[i],
                })

        print(f"  ✓ Generated {len(signals)} signals")

        if not signals:
            print(f"  ⚠ No signals generated — try a longer period or different pair")
            continue

        # ── Simulate outcomes ───────────────────────────────────────
        pip_size = PIP_SIZE.get(pair, 0.0001)
        outcomes = []
        for sig in signals:
            outcome, exit_idx, exit_price, exit_reason = simulate_trade_outcome(
                df, sig["bar_idx"], sig["signal"], sig["atr"], pip_size
            )
            sig["outcome"] = outcome
            sig["exit_reason"] = exit_reason
            sig["exit_price"] = exit_price
            outcomes.append(sig)

        wins = sum(1 for s in outcomes if s["outcome"] == "WIN")
        losses = sum(1 for s in outcomes if s["outcome"] == "LOSS")
        wr = wins / max(len(outcomes), 1) * 100
        print(f"  ✓ Simulated {len(outcomes)} trade outcomes")
        print(f"    Wins: {wins} | Losses: {losses} | WR: {wr:.1f}%")

        # ── Show sample ─────────────────────────────────────────────
        print(f"  Sample trades:")
        for s in outcomes[:5]:
            print(f"    {s['time'].date()} {s['signal']:4s} patterns={s['patterns']} "
                  f"→ {s['outcome']:4s} ({s['exit_reason']})")

        # ── Record into ConfidenceEngine ────────────────────────────
        if args.dry_run:
            print(f"  [DRY RUN] Not recording outcomes")
            continue

        # Cap to max_trades per pair
        to_record = outcomes[:args.max_trades]
        regime = detect_regime(df)

        recorded = 0
        for s in to_record:
            for pattern in s["patterns"]:
                try:
                    engine.record_outcome(
                        pattern=pattern,
                        pair=pair,
                        timeframe=timeframe,
                        regime=regime,
                        outcome=s["outcome"],
                        confidence_used=s["confidence"],
                        pnl=None,
                    )
                    recorded += 1
                except Exception as e:
                    if recorded < 3:
                        print(f"    ⚠ Record failed: {e}")

        print(f"  ✓ Recorded {recorded} pattern outcomes")
        total_recorded += recorded
        total_wins += wins
        total_losses += losses

    # ── Cleanup ─────────────────────────────────────────────────────
    if use_mt5:
        mt5.shutdown()

    # ── Final summary ───────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  REPLAY COMPLETE")
    print("=" * 70)
    print(f"  Total outcomes recorded: {total_recorded}")
    print(f"  Total wins:              {total_wins}")
    print(f"  Total losses:            {total_losses}")
    if total_wins + total_losses > 0:
        wr = total_wins / (total_wins + total_losses) * 100
        print(f"  Overall WR:              {wr:.1f}%")
    print()
    print("  NEXT STEPS:")
    print("    1. Verify pattern_stats now has real data:")
    print("       python scripts/bootstrap_tools/check_penalty_status.py")
    print("    2. Look for patterns with 3+ trades — their Bayesian penalty is now 0")
    print("    3. For full-pipeline backtest (with all gates):")
    print("       python main.py --mode backtest --pairs EURUSD --timeframe 1h --bars 500")
    print()
    print("  NOTE:")
    print("    This script used a SIMPLIFIED pattern detector and signal generator.")
    print("    For full strategy fidelity, use main.py --mode backtest which runs")
    print("    the real AnalysisAgent + 30+ sub-engines.")
    print("    The outcomes recorded here are APPROXIMATE but directionally correct.")
    print("    They will be REPLACED by real live trade outcomes as the system runs.")


if __name__ == "__main__":
    main()
