"""
blocked_trade_outcome_audit.py — Were the blocked trades actually correct?
============================================================================

The existing blocked_audit.py / blocked_first_failure.py / blocked_module_audit.py
scripts answer "WHY was a trade blocked?" (which gate failed). None of them
answer the more important question: "WAS BLOCKING IT THE RIGHT CALL?"

This script answers that by replaying real price history. For every blocked
signal that had a genuine entry/SL/TP (i.e. it got as far as being priced by
RiskEngine, then got vetoed later — by TradePermission, LiveRiskManager,
Devil's Advocate, etc.), it fetches the actual candles that followed and
checks which level price touched first:

  - SL hit first  -> the block was CORRECT (it would have been a loss).
  - TP hit first  -> the block was WRONG (it would have been a win) — this
                     is money the filters cost you.
  - neither hit   -> inconclusive within the lookahead window.

Requires a live MetaTrader5 terminal + package (Windows), logged into the
SAME account/server the bot traded on, since prices must match exactly.
Run this on the machine where the bot runs, not in a sandboxed environment.

Usage:
    python blocked_trade_outcome_audit.py
    python blocked_trade_outcome_audit.py --log logs/execution.log --days 3
    python blocked_trade_outcome_audit.py --symbol EURUSD --timeframe M15
    python blocked_trade_outcome_audit.py --csv logs/blocked_outcome_report.csv

Notes / limitations:
  - Only evaluates signals that reached RiskEngine with real entry/sl/tp.
    Pure "Signal is WAIT — no trade" blocks are excluded (there was never
    a concrete trade to evaluate — WAIT is not a directional call).
  - Signals from before the 2026-08-20 fix that added `entry` to the
    risk.evaluated log line are skipped with a warning (no entry price to
    replay from). Re-run this after the bot has logged some trades post-fix.
  - Ambiguous same-candle SL+TP touches are conservatively scored as the
    SL hitting first (standard backtest convention — never gives the
    system credit it can't prove it earned).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────
# MT5 import guard (mirrors data/fetcher.py's pattern) — this
# script can be imported/parsed on non-Windows machines even
# though it can only actually RUN the price-replay part on the
# Windows box with a live MT5 terminal.
# ─────────────────────────────────────────────────────────────
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False

TIMEFRAME_MAP = {
    "M1": None, "M5": None, "M15": None, "M30": None,
    "H1": None, "H4": None, "D1": None,
}


def _mt5_timeframe(name: str):
    if not MT5_AVAILABLE:
        return None
    mapping = {
        "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    return mapping.get(name.upper())


@dataclass
class BlockedTrade:
    evaluation_id: str
    symbol: str
    direction: str          # "BUY" or "SELL"
    entry: float
    sl: float
    tp: float
    ts: datetime
    failed_checks: list = field(default_factory=list)
    blocked_by: str = ""    # short label: which stage actually vetoed it


@dataclass
class Outcome:
    trade: BlockedTrade
    result: str              # "would_have_lost" | "would_have_won" | "inconclusive"
    hit_price: Optional[float] = None
    hit_ts: Optional[datetime] = None
    pips: Optional[float] = None   # signed: positive = what you missed/avoided, in price units


# ─────────────────────────────────────────────────────────────
# Step 1 — parse execution.log into per-evaluation_id records
# ─────────────────────────────────────────────────────────────
def _norm_event_name(obj: dict) -> Optional[str]:
    """Return the event name whether or not it's nested (defensive: the
    known 2026-08-20 core/trader.py double-nesting bug — now fixed — could
    still be present in OLD log lines written before the fix)."""
    ev = obj.get("event")
    if isinstance(ev, dict):
        return ev.get("event")
    return ev


def _flatten(obj: dict) -> dict:
    """If this line has the old nested {"event": {...}} shape, merge the
    inner dict up so downstream code can treat every line uniformly."""
    ev = obj.get("event")
    if isinstance(ev, dict):
        merged = dict(obj)
        merged.pop("event", None)
        merged.update(ev)
        return merged
    return obj


def load_candidates(log_path: Path, symbol_filter: Optional[str]) -> list[BlockedTrade]:
    by_eval: dict[str, dict] = defaultdict(dict)

    skipped_no_entry = 0
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            name = _norm_event_name(obj)
            if name not in ("risk.evaluated", "risk.finalized", "permission.checked"):
                continue
            obj = _flatten(obj)
            eid = obj.get("evaluation_id")
            if not eid:
                continue
            by_eval[eid][name] = obj
            by_eval[eid].setdefault("symbol", obj.get("symbol"))
            by_eval[eid].setdefault("ts", obj.get("ts"))

    candidates: list[BlockedTrade] = []
    for eid, rec in by_eval.items():
        symbol = rec.get("symbol")
        if symbol_filter and symbol != symbol_filter:
            continue

        risk_eval = rec.get("risk.evaluated") or {}
        risk_fin = rec.get("risk.finalized") or {}
        perm = rec.get("permission.checked") or {}

        # Must have had a real, priced setup at some point.
        if not risk_eval.get("approved"):
            continue
        sl = risk_eval.get("sl")
        tp = risk_eval.get("tp")
        entry = risk_eval.get("entry")
        if sl is None or tp is None:
            continue

        # Must have ultimately been BLOCKED (either finalized False, or
        # never got through permission).
        finally_allowed = perm.get("allowed")
        if finally_allowed is True:
            continue  # it was actually taken — not a "blocked" case
        if risk_fin and risk_fin.get("approved") is True and finally_allowed is not False:
            continue  # ambiguous / actually went through

        if entry is None:
            skipped_no_entry += 1
            continue

        direction = (risk_eval.get("signal") or perm.get("raw_signal") or "").upper()
        if direction not in ("BUY", "SELL"):
            dec = str(perm.get("decision") or "").upper()
            if "BUY" in dec:
                direction = "BUY"
            elif "SELL" in dec:
                direction = "SELL"
        if direction not in ("BUY", "SELL"):
            continue

        ts_raw = rec.get("ts")
        try:
            ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
        except Exception:
            continue

        blocked_by = "risk_engine" if risk_fin.get("approved") is False else "permission"

        candidates.append(BlockedTrade(
            evaluation_id=eid, symbol=symbol, direction=direction,
            entry=float(entry), sl=float(sl), tp=float(tp), ts=ts,
            failed_checks=perm.get("failed_checks") or [],
            blocked_by=blocked_by,
        ))

    if skipped_no_entry:
        print(
            f"[note] {skipped_no_entry} blocked signal(s) had no logged "
            f"entry price (written before the 2026-08-20 log fix) and were "
            f"skipped — re-run after the bot logs new trades.",
            file=sys.stderr,
        )
    return candidates


# ─────────────────────────────────────────────────────────────
# Step 2 — replay real price action after each blocked signal
# ─────────────────────────────────────────────────────────────
def _connect_mt5() -> bool:
    if not MT5_AVAILABLE:
        return False
    if not mt5.initialize():
        return False
    return True


def replay_outcome(trade: BlockedTrade, timeframe: str, lookahead_days: int) -> Outcome:
    tf = _mt5_timeframe(timeframe)
    if tf is None:
        return Outcome(trade=trade, result="inconclusive")

    date_from = trade.ts
    date_to = trade.ts + timedelta(days=lookahead_days)
    rates = mt5.copy_rates_range(trade.symbol, tf, date_from, date_to)
    if rates is None or len(rates) == 0:
        return Outcome(trade=trade, result="inconclusive")

    for bar in rates:
        bar_time = datetime.fromtimestamp(int(bar["time"]), tz=timezone.utc)
        if bar_time < trade.ts:
            continue
        high, low = float(bar["high"]), float(bar["low"])

        if trade.direction == "BUY":
            sl_hit = low <= trade.sl
            tp_hit = high >= trade.tp
        else:  # SELL
            sl_hit = high >= trade.sl
            tp_hit = low <= trade.tp

        if sl_hit and tp_hit:
            # Ambiguous same-candle touch — conservative: SL first.
            pips = -(abs(trade.entry - trade.sl))
            return Outcome(trade=trade, result="would_have_lost",
                            hit_price=trade.sl, hit_ts=bar_time, pips=pips)
        if sl_hit:
            pips = -(abs(trade.entry - trade.sl))
            return Outcome(trade=trade, result="would_have_lost",
                            hit_price=trade.sl, hit_ts=bar_time, pips=pips)
        if tp_hit:
            pips = abs(trade.tp - trade.entry)
            return Outcome(trade=trade, result="would_have_won",
                            hit_price=trade.tp, hit_ts=bar_time, pips=pips)

    return Outcome(trade=trade, result="inconclusive")


# ─────────────────────────────────────────────────────────────
# Step 3 — report
# ─────────────────────────────────────────────────────────────
def print_report(outcomes: list[Outcome]) -> None:
    total = len(outcomes)
    if total == 0:
        print("No blocked trades with a complete entry/SL/TP were found to evaluate.")
        return

    won = [o for o in outcomes if o.result == "would_have_won"]
    lost = [o for o in outcomes if o.result == "would_have_lost"]
    inconclusive = [o for o in outcomes if o.result == "inconclusive"]

    print("=" * 70)
    print("  BLOCKED TRADE OUTCOME AUDIT")
    print("=" * 70)
    print(f"  Evaluated               : {total}")
    print(f"  Block was CORRECT (avoided a loss) : {len(lost)}  "
          f"({len(lost)/total*100:.1f}%)")
    print(f"  Block was WRONG (missed a win)     : {len(won)}  "
          f"({len(won)/total*100:.1f}%)")
    print(f"  Inconclusive (no SL/TP hit in window): {len(inconclusive)}  "
          f"({len(inconclusive)/total*100:.1f}%)")
    decided = len(won) + len(lost)
    if decided:
        print(f"  Block accuracy (of decided cases)  : {len(lost)/decided*100:.1f}%")
    missed_pips = sum(o.pips for o in won if o.pips is not None)
    saved_pips = sum(-o.pips for o in lost if o.pips is not None)
    print(f"  Total price-distance missed by blocking wins : {missed_pips:.5f}")
    print(f"  Total price-distance avoided by blocking losses: {saved_pips:.5f}")
    print()

    print("-- By symbol --")
    by_symbol = defaultdict(lambda: {"won": 0, "lost": 0, "inc": 0})
    for o in outcomes:
        key = "won" if o.result == "would_have_won" else (
            "lost" if o.result == "would_have_lost" else "inc")
        by_symbol[o.trade.symbol][key] += 1
    for sym, d in sorted(by_symbol.items(), key=lambda x: -(x[1]["won"] + x[1]["lost"])):
        n = d["won"] + d["lost"] + d["inc"]
        dec = d["won"] + d["lost"]
        acc = f"{d['lost']/dec*100:.0f}%" if dec else "n/a"
        print(f"  {sym:10s} n={n:3d}  correct_blocks={d['lost']:3d}  "
              f"missed_wins={d['won']:3d}  inconclusive={d['inc']:3d}  block_accuracy={acc}")
    print()

    print("-- By failed check (which gate blocked it) --")
    by_check = defaultdict(lambda: {"won": 0, "lost": 0})
    for o in outcomes:
        if o.result not in ("would_have_won", "would_have_lost"):
            continue
        key = "won" if o.result == "would_have_won" else "lost"
        checks = o.trade.failed_checks or ["<unknown>"]
        for c in checks:
            by_check[c][key] += 1
    for check, d in sorted(by_check.items(), key=lambda x: -(x[1]["won"] + x[1]["lost"])):
        n = d["won"] + d["lost"]
        acc = f"{d['lost']/n*100:.0f}%" if n else "n/a"
        verdict = "net helpful" if d["lost"] > d["won"] else (
            "net harmful" if d["won"] > d["lost"] else "even")
        print(f"  {check:35s} n={n:3d}  correct={d['lost']:3d}  "
              f"wrong={d['won']:3d}  accuracy={acc:>5s}  ({verdict})")


def write_csv(outcomes: list[Outcome], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["evaluation_id", "symbol", "direction", "ts", "entry", "sl", "tp",
                    "result", "hit_price", "hit_ts", "pips", "blocked_by", "failed_checks"])
        for o in outcomes:
            t = o.trade
            w.writerow([t.evaluation_id, t.symbol, t.direction, t.ts.isoformat(),
                        t.entry, t.sl, t.tp, o.result, o.hit_price,
                        o.hit_ts.isoformat() if o.hit_ts else "", o.pips,
                        t.blocked_by, ";".join(t.failed_checks)])
    print(f"\nDetailed CSV written to: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", default="logs/execution.log",
                         help="Path to execution.log (default: logs/execution.log)")
    parser.add_argument("--symbol", default=None, help="Only audit this symbol")
    parser.add_argument("--timeframe", default="M15",
                         choices=["M1", "M5", "M15", "M30", "H1", "H4", "D1"],
                         help="Candle timeframe to replay with (default: M15)")
    parser.add_argument("--days", type=int, default=3,
                         help="Max lookahead window in days per trade (default: 3)")
    parser.add_argument("--csv", default=None, help="Write a detailed CSV report to this path")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"ERROR: log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    candidates = load_candidates(log_path, args.symbol)
    if not candidates:
        print("No blocked trades with a real entry/SL/TP found in this log. "
              "(WAIT-only blocks are excluded — there's nothing to replay.)")
        return

    if not MT5_AVAILABLE:
        print(
            "ERROR: MetaTrader5 package is not installed / not importable.\n"
            "This script must run on the machine where the bot trades, with a\n"
            "live MT5 terminal logged into the SAME account/server, so replayed\n"
            "prices match exactly. Install with: pip install MetaTrader5",
            file=sys.stderr,
        )
        sys.exit(1)

    if not _connect_mt5():
        print(f"ERROR: mt5.initialize() failed: {mt5.last_error()}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(candidates)} blocked trade(s) with a real entry/SL/TP — replaying price...")
    outcomes = []
    for t in candidates:
        outcomes.append(replay_outcome(t, args.timeframe, args.days))

    mt5.shutdown()

    print_report(outcomes)
    if args.csv:
        write_csv(outcomes, Path(args.csv))


if __name__ == "__main__":
    main()