#!/usr/bin/env python3
"""
Option 4 — Import REAL MT5 account trade history into pattern_stats.json.

THE MOST LEGITIMATE BOOTSTRAP:
  Instead of generating fake empty entries (Option 1) or running a backtest
  (Option 2), this script pulls the user's ACTUAL closed trades from MT5
  account history (mt5.history_deals_get) and feeds them to the
  ConfidenceEngine as real training data.

  This is the user's own trading history — every WIN, every LOSS is real,
  recorded by their broker. The engine learns from REAL outcomes.

WHAT THIS SCRIPT DOES:
  1. Connects to MT5 (Windows host with terminal running)
  2. Calls mt5.history_deals_get(start, end) to get all closed deals
  3. For each closed position:
     - Identifies entry direction (BUY/SELL)
     - Identifies exit (SL hit / TP hit / manual close)
     - Records WIN if profit > 0, LOSS if profit < 0, BE if profit == 0
     - Looks up the pattern that fired at entry time (from trade_memory.json)
       — if no pattern found, uses "manual_close" as pattern name
  4. Calls ConfidenceEngine.record_outcome() for each
  5. Reports summary

PREREQUISITES:
  - Windows host with MT5 terminal running
  - MetaTrader5 python package installed (pip install MetaTrader5)
  - .env with MT5 credentials (or interactive login)
  - Some closed trades in account history (otherwise nothing to import)

USAGE:
  # Import last 30 days of trade history
  python scripts/bootstrap_tools/import_mt5_history.py --days 30

  # Import last 90 days (3 months — recommended for good sample size)
  python scripts/bootstrap_tools/import_mt5_history.py --days 90

  # Import specific date range
  python scripts/bootstrap_tools/import_mt5_history.py \\
      --start 2026-01-01 --end 2026-06-30

  # Dry-run (show what would be imported)
  python scripts/bootstrap_tools/import_mt5_history.py --days 30 --dry-run

  # Override default pattern name when trade_memory has no record
  python scripts/bootstrap_tools/import_mt5_history.py --days 90 \\
      --default-pattern "mt5_real_trade"

WHAT IF MT5 IS NOT AVAILABLE (Linux/CI):
  The script prints a clear error and suggests alternatives:
  - Run on Windows host with MT5 terminal running
  - Use import_backtest_samples.py instead (CSV-based fallback)
  - Use generate_synthetic_samples.py (empty entries, fastest)

WHY THIS IS BETTER THAN BACKTEST IMPORT:
  - Backtest outcomes depend on the strategy logic being correct
  - MT5 history is what ACTUALLY happened to your account
  - Includes manual trades, partial closes, news-event trades — real-world
    outcomes that backtests cannot reproduce
  - These are YOUR trades, on YOUR account, with YOUR broker's spread/slippage
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from core.constants import MEMORY_DIR
    from learning.confidence_engine import ConfidenceEngine, MIN_SAMPLE_SIZE
except Exception as e:
    print(f"ERROR: cannot import from project: {e}")
    print(f"       Run from project root:  cd {PROJECT_ROOT}")
    sys.exit(1)

# Try to import MT5
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False


def connect_mt5() -> bool:
    """Initialize MT5 connection."""
    if not MT5_AVAILABLE:
        return False
    try:
        if not mt5.initialize():
            # Try with .env credentials if available
            login = os.getenv("MT5_LOGIN") or os.getenv("MT5_ACCOUNT")
            password = os.getenv("MT5_PASSWORD") or os.getenv("MT5_PASS")
            server = os.getenv("MT5_SERVER") or os.getenv("MT5_BROKER")
            if login and password and server:
                if not mt5.initialize(
                    login=int(login),
                    password=password,
                    server=server,
                ):
                    return False
            else:
                return False
        return mt5.terminal_info() is not None
    except Exception as e:
        print(f"  MT5 initialize failed: {e}")
        return False


def fetch_closed_deals(start: datetime, end: datetime) -> list:
    """Fetch closed deals from MT5 history.
    Returns list of deal objects with: symbol, type, entry, price, profit, time, position_id, comment.
    """
    deals = mt5.history_deals_get(start, end)
    if deals is None:
        return []
    return list(deals)


def pair_entry_exit_by_position(deals: list) -> list:
    """Group deals by position_id and pair entry+exit deals.

    MT5 records 2 deals per closed position:
      - DEAL_ENTRY_IN  (entry deal: type 0=BUY, 1=SELL)
      - DEAL_ENTRY_OUT (exit deal: has profit field with realized P&L)

    Returns list of dicts with: symbol, direction, entry_time, exit_time,
    entry_price, exit_price, profit, comment, volume.
    """
    # Group by position_id
    by_pos = {}
    for d in deals:
        pos_id = getattr(d, "position_id", None)
        if pos_id is None or pos_id == 0:
            continue
        by_pos.setdefault(pos_id, []).append(d)

    trades = []
    for pos_id, pos_deals in by_pos.items():
        entry_deal = None
        exit_deal = None
        for d in pos_deals:
            entry = getattr(d, "entry", None)
            if entry == getattr(mt5, "DEAL_ENTRY_IN", 0):
                entry_deal = d
            elif entry in (getattr(mt5, "DEAL_ENTRY_OUT", 1),
                           getattr(mt5, "DEAL_ENTRY_OUT_BY", 3)):
                exit_deal = d

        if not entry_deal or not exit_deal:
            continue  # position still open, or only one side recorded

        # Determine direction from entry deal type
        # mt5.DEAL_TYPE_BUY = 0, mt5.DEAL_TYPE_SELL = 1
        deal_type = getattr(entry_deal, "type", None)
        if deal_type == getattr(mt5, "DEAL_TYPE_BUY", 0):
            direction = "BUY"
        elif deal_type == getattr(mt5, "DEAL_TYPE_SELL", 1):
            direction = "SELL"
        else:
            continue

        profit = float(getattr(exit_deal, "profit", 0) or 0)
        # Commission + swap adjust the realized P&L
        commission = float(getattr(exit_deal, "commission", 0) or 0)
        swap = float(getattr(exit_deal, "swap", 0) or 0)
        net_profit = profit + commission + swap

        # Outcome classification
        if net_profit > 0:
            outcome = "WIN"
        elif net_profit < 0:
            outcome = "LOSS"
        else:
            outcome = "BE"

        # Exit reason (best-effort from comment)
        comment = str(getattr(exit_deal, "comment", "") or "").lower()
        if "sl" in comment or "stop loss" in comment:
            exit_reason = "SL"
        elif "tp" in comment or "take profit" in comment:
            exit_reason = "TP"
        elif comment:
            exit_reason = comment[:40]
        else:
            exit_reason = "manual" if abs(net_profit) < 0.01 else ("TP" if net_profit > 0 else "SL")

        trades.append({
            "position_id": pos_id,
            "symbol": str(getattr(entry_deal, "symbol", "")),
            "direction": direction,
            "entry_time": datetime.fromtimestamp(getattr(entry_deal, "time", 0),
                                                  tz=timezone.utc),
            "exit_time": datetime.fromtimestamp(getattr(exit_deal, "time", 0),
                                                 tz=timezone.utc),
            "entry_price": float(getattr(entry_deal, "price", 0) or 0),
            "exit_price": float(getattr(exit_deal, "price", 0) or 0),
            "volume": float(getattr(entry_deal, "volume", 0) or 0),
            "profit": net_profit,
            "gross_profit": profit,
            "commission": commission,
            "swap": swap,
            "comment": str(getattr(exit_deal, "comment", "") or ""),
            "exit_reason": exit_reason,
            "outcome": outcome,
        })

    return trades


def load_trade_memory_lookup() -> dict:
    """Load trade_memory.json and build a lookup index by entry time.
    Allows us to find what pattern/strategy fired when the MT5 trade was opened.
    """
    tm_path = MEMORY_DIR / "trade_memory.json"
    if not tm_path.exists():
        return {}
    try:
        with open(tm_path, encoding="utf-8") as f:
            entries = json.load(f)
    except Exception:
        return {}

    # Build lookup: (symbol, entry_time_minute_bucket) → patterns
    lookup = {}
    for e in entries:
        sym = e.get("symbol", "").upper()
        ts = e.get("timestamp") or e.get("entry_time") or ""
        if not sym or not ts:
            continue
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except Exception:
            continue
        # Bucket by symbol + minute (entry time may differ by seconds)
        key = (sym, dt.strftime("%Y%m%d%H%M"))
        patterns = e.get("patterns", [])
        regime = e.get("regime", "UNKNOWN")
        tf = e.get("timeframe", "H1")
        lookup.setdefault(key, []).append({
            "patterns": patterns,
            "regime": regime,
            "timeframe": tf,
            "decision": e.get("decision"),
        })
    return lookup


def find_pattern_for_trade(trade: dict, lookup: dict, default_pattern: str) -> tuple:
    """Find the pattern that fired when this MT5 trade was opened.
    Returns (pattern, regime, timeframe). Falls back to default_pattern
    if no trade_memory entry is found near the entry time.
    """
    sym = trade["symbol"].upper()
    entry_dt = trade["entry_time"]

    # Try exact minute bucket first, then +/- 1 minute
    for delta_min in [0, -1, 1, -2, 2, -5, 5]:
        bucket_dt = entry_dt + timedelta(minutes=delta_min)
        key = (sym, bucket_dt.strftime("%Y%m%d%H%M"))
        matches = lookup.get(key, [])
        if matches:
            m = matches[0]
            patterns = m.get("patterns", [])
            pattern = patterns[0] if patterns else default_pattern
            regime = m.get("regime", "UNKNOWN")
            tf = m.get("timeframe", "H1")
            return pattern, regime, tf

    # No trade_memory match — this is likely a manual trade
    return default_pattern, "UNKNOWN", "H1"


def main():
    parser = argparse.ArgumentParser(description="Import REAL MT5 account history into pattern_stats.json")
    parser.add_argument("--days", type=int, default=90,
                        help="Import last N days of trade history (default: 90)")
    parser.add_argument("--start", type=str, default="",
                        help="Start date YYYY-MM-DD (overrides --days)")
    parser.add_argument("--end", type=str, default="",
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--default-pattern", default="mt5_real_trade",
                        help="Pattern name when trade_memory has no record (default: 'mt5_real_trade')")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be imported without writing")
    parser.add_argument("--symbols", default="",
                        help="Comma-separated symbol filter (default: all)")
    args = parser.parse_args()

    print("=" * 70)
    print("  MT5 REAL HISTORY IMPORTER")
    print("=" * 70)
    print(f"  Pattern stats target: {MEMORY_DIR / 'pattern_stats.json'}")
    print(f"  Default pattern:      {args.default_pattern}")
    print()

    # ── MT5 availability check ──────────────────────────────────────
    if not MT5_AVAILABLE:
        print("  ✗ MetaTrader5 package not installed.")
        print()
        print("  This script requires MT5 — it only runs on Windows with the")
        print("  terminal running. On Linux/CI, use these alternatives:")
        print()
        print("    1. python scripts/bootstrap_tools/generate_synthetic_samples.py")
        print("       (creates empty entries — breaks chicken-and-egg loop)")
        print()
        print("    2. python scripts/bootstrap_tools/import_backtest_samples.py --db <path>")
        print("       (imports backtest outcomes — needs a backtest DB)")
        print()
        print("    3. Run on your Windows host:")
        print("       pip install MetaTrader5")
        print("       python scripts/bootstrap_tools/import_mt5_history.py --days 90")
        return

    # ── Date range ──────────────────────────────────────────────────
    end_dt = datetime.now(timezone.utc)
    if args.end:
        try:
            end_dt = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
        except Exception:
            print(f"  ✗ Invalid --end date: {args.end} (use YYYY-MM-DD)")
            return

    if args.start:
        try:
            start_dt = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
        except Exception:
            print(f"  ✗ Invalid --start date: {args.start} (use YYYY-MM-DD)")
            return
    else:
        start_dt = end_dt - timedelta(days=args.days)

    print(f"  Date range:           {start_dt.date()} → {end_dt.date()} ({args.days} days)")
    print()

    # ── Connect to MT5 ──────────────────────────────────────────────
    print("  Connecting to MT5...")
    if not connect_mt5():
        print("  ✗ MT5 connection failed.")
        print()
        print("  Troubleshooting:")
        print("    1. Is MT5 terminal running on this machine?")
        print("    2. Are MT5_LOGIN / MT5_PASSWORD / MT5_SERVER set in .env?")
        print("    3. Does the account have any closed trades in the date range?")
        return

    # Show account info for verification
    info = mt5.account_info()
    if info:
        print(f"  ✓ Connected: account {info.login} | balance=${info.balance:.2f} | {info.server}")
    print()

    # ── Fetch closed deals ──────────────────────────────────────────
    print(f"  Fetching closed deals from {start_dt.date()} → {end_dt.date()}...")
    deals = fetch_closed_deals(start_dt, end_dt)
    print(f"  ✓ Fetched {len(deals)} deal records (includes both entry and exit deals)")
    print()

    if not deals:
        print("  ✗ No deals found in the specified date range.")
        print("    Check: did you trade during this period on this account?")
        mt5.shutdown()
        return

    # ── Pair entry+exit deals ───────────────────────────────────────
    print("  Pairing entry + exit deals by position_id...")
    trades = pair_entry_exit_by_position(deals)
    print(f"  ✓ {len(trades)} closed positions reconstructed")
    print()

    if not trades:
        print("  ✗ No closed positions found (all positions still open?)")
        mt5.shutdown()
        return

    # ── Filter by symbol if requested ───────────────────────────────
    if args.symbols:
        wanted = [s.strip().upper() for s in args.symbols.split(",")]
        trades = [t for t in trades if t["symbol"].upper() in wanted]
        print(f"  Filtered to symbols {wanted}: {len(trades)} trades remain")
        print()

    # ── Load trade_memory lookup ────────────────────────────────────
    print("  Loading trade_memory.json lookup (to find pattern for each trade)...")
    lookup = load_trade_memory_lookup()
    print(f"  ✓ {len(lookup)} minute-bucket entries indexed")
    print()

    # ── Show summary before importing ───────────────────────────────
    wins = sum(1 for t in trades if t["outcome"] == "WIN")
    losses = sum(1 for t in trades if t["outcome"] == "LOSS")
    bes = sum(1 for t in trades if t["outcome"] == "BE")
    total_pnl = sum(t["profit"] for t in trades)

    print("  TRADE SUMMARY:")
    print(f"    Total closed:        {len(trades)}")
    print(f"    Wins:                {wins}")
    print(f"    Losses:              {losses}")
    print(f"    Break-even:          {bes}")
    print(f"    Win rate:            {wins/max(len(trades),1)*100:.1f}%")
    print(f"    Net P&L:             ${total_pnl:.2f}")
    print(f"    Avg P&L/trade:       ${total_pnl/max(len(trades),1):.2f}")
    print()

    # Show sample trades
    if trades:
        print("  SAMPLE TRADES (first 5):")
        for t in trades[:5]:
            print(f"    {t['entry_time'].date()} {t['symbol']:6s} {t['direction']:4s} "
                  f"vol={t['volume']:.2f} | {t['outcome']:4s} ${t['profit']:+.2f} "
                  f"| exit={t['exit_reason']}")
        print()

    # ── Dry-run exit ────────────────────────────────────────────────
    if args.dry_run:
        print("  [DRY RUN] No file written. Remove --dry-run to apply.")
        mt5.shutdown()
        return

    # ── Import into ConfidenceEngine ────────────────────────────────
    print("  Importing into ConfidenceEngine...")
    engine = ConfidenceEngine()
    imported = 0
    skipped = 0
    pattern_counts = {}

    for t in trades:
        pattern, regime, tf = find_pattern_for_trade(t, lookup, args.default_pattern)
        try:
            engine.record_outcome(
                pattern=pattern,
                pair=t["symbol"],
                timeframe=tf,
                regime=regime,
                outcome=t["outcome"],
                confidence_used=None,  # not tracked in MT5 history
                pnl=t["profit"],
            )
            imported += 1
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"    ⚠ Skipped: {e}")

    mt5.shutdown()
    print()
    print(f"  ✓ Imported: {imported}")
    print(f"    Skipped:  {skipped}")
    print()
    print("  PATTERN DISTRIBUTION (top 10):")
    for pat, count in sorted(pattern_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {count:3d} trades  {pat}")
    print()

    # ── Final report ────────────────────────────────────────────────
    print("=" * 70)
    print("  IMPORT COMPLETE")
    print("=" * 70)
    print(f"  Pattern stats: {MEMORY_DIR / 'pattern_stats.json'}")
    print(f"  Trades added:  {imported}")
    print(f"  Win rate used: {wins/max(len(trades),1)*100:.1f}%")
    print()
    print("  NEXT STEPS:")
    print("    1. Verify penalty status:")
    print("       python scripts/bootstrap_tools/check_penalty_status.py")
    print("    2. The Bayesian penalty should now be 0 for any pattern with")
    print(f"       3+ imported trades (MIN_SAMPLE_SIZE = {MIN_SAMPLE_SIZE}).")
    print("    3. As live trades close, they will MIX with this MT5 history.")
    print("       After ~10 new live trades per pattern, live data dominates.")
    print("    4. To re-import (e.g. after more trades), re-run this script.")
    print("       It will OVERWRITE the previous imports — ConfidenceEngine")
    print("       uses record_outcome() which increments counters, so re-running")
    print("       the same date range WILL double-count. Use a fresh date range")


if __name__ == "__main__":
    main()
