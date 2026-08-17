"""
Rule-engine backtest driver.

Uses the REAL production rule engine (analysis/unified_signal_engine.py ->
UnifiedSignalEngine, which internally orchestrates StopHuntSignalEngine,
ICTAMDSignalEngine, MultiStrategyPAEngine, LiquidityPoolAnalyzer,
CCIStateMachine, HighReliabilityPatternDetector, SupportResistance) — no
reimplementation of the signal logic. Only two harmless import-time stubs
were added (utils/logger.py, risk/rr_policy.py) because those two small
modules were not part of the uploaded core/agents/analysis zips; rr_policy
reproduces the exact constants documented in core/constants.py
(MIN_RR_PROD=2.0, stop_hunt override=1.4).

Methodology (see README to the user for the full caveats):
  - M15 CSVs resampled to H1 (multi_strategy_pa_engine.py hard-restricts
    ALLOWED_TIMEFRAMES to {1H,4H,D1}; running at M15 would silently zero
    out the PA layer for every bar).
  - Rolling window of WINDOW closed H1 bars only (no look-ahead: the bar
    the decision is made on is always fully closed).
  - Entry fill = next bar's OPEN (decision made at bar close -> filled at
    the next bar, not the same bar it was decided on).
  - Spread cost applied at entry using the CSV's own spread column.
  - SL/TP taken from the specific sub-engine that won the consensus vote
    (each engine already computes its own structure-based SL/TP; we do
    not invent an arbitrary ATR multiple).
  - One open trade at a time per symbol (no pyramiding/averaging).
  - Trade outcome walked forward bar-by-bar on H1 highs/lows; if SL and
    TP are both inside the same bar's range, SL is assumed hit first
    (conservative, avoids inflating win rate).
  - Max holding period -> forced close at TIMEOUT_BARS if neither level
    is hit (marks TIMEOUT, priced at that bar's close).
"""
import sys
import json
import time
import pickle
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

from analysis.unified_signal_engine import UnifiedSignalEngine

WINDOW = 150
TIMEOUT_BARS = 200
FILL_TIMEOUT_BARS = 20  # pending order must be touched within this many H1 bars or it expires
POINT = 1e-5  # 5-digit FX quote convention used by all 4 uploaded pairs


def load_h1(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime_utc"])
    df = df.set_index("datetime")
    h1 = (
        df.resample("1h")
        .agg({
            "open": "first", "high": "max", "low": "min", "close": "last",
            "tick_volume": "sum", "spread": "mean",
        })
        .dropna()
        .reset_index()
    )
    return h1


REAL_ENGINE_KEYS = {
    "StopHunt": ("stop_hunt", "signal"),
    "ICT/AMD": ("ict_amd", "signal"),
    "PA": ("multi_strategy_pa", "signal"),
    "Liquidity": ("liquidity", "signal"),
    "CCI": ("cci_state", "signal"),
}


def extract_trade_levels(res: dict, side: str):
    """Pick SL/TP from the highest-weight REAL (non-pattern) engine that
    voted for the winning side and actually carries price levels."""
    voting = sorted(res["consensus"].get("voting_engines", []), key=lambda v: -v["weight"])
    for v in voting:
        name = v["engine"]
        if name not in REAL_ENGINE_KEYS or v["action"] != side:
            continue
        block_key, sig_key = REAL_ENGINE_KEYS[name]
        sig = res.get(block_key, {}).get(sig_key, {})
        entry = sig.get("entry_price")
        sl = sig.get("stop_loss")
        tp = sig.get("take_profit") or sig.get("take_profit_suggested")
        if entry is not None and sl is not None and tp is not None:
            return name, entry, sl, tp
    return None, None, None, None


def run_symbol(symbol: str, csv_path: str, out_path: str, limit: int = None,
                time_budget_sec: float = 240.0, ckpt_path: str = None,
                engine_kwargs: dict = None):
    h1 = load_h1(csv_path)
    if limit:
        h1 = h1.iloc[:limit].reset_index(drop=True)
    n = len(h1)
    eng = UnifiedSignalEngine(timeframe="H1", **(engine_kwargs or {}))

    ckpt_path = ckpt_path or (out_path + ".ckpt.pkl")
    if Path(ckpt_path).exists():
        with open(ckpt_path, "rb") as f:
            state = pickle.load(f)
        i = state["i"]
        trades = state["trades"]
        unfilled = state["unfilled"]
        open_trade = state["open_trade"]
        pending = state["pending"]
        print(f"[{symbol}] RESUMED at bar {i - WINDOW}/{n - WINDOW}, "
              f"{len(trades)} trades so far, {unfilled} unfilled", flush=True)
    else:
        i = WINDOW
        trades = []
        unfilled = 0
        open_trade = None
        pending = None

    t_start = time.time()
    log_every = 500
    finished = False

    while i < n - 1:
        if time.time() - t_start > time_budget_sec:
            break
        bar = h1.iloc[i]

        # ---- manage an already-FILLED open trade ----
        if open_trade is not None:
            side = open_trade["side"]
            sl, tp = open_trade["sl"], open_trade["tp"]
            hit_sl = (bar["low"] <= sl) if side == "BUY" else (bar["high"] >= sl)
            hit_tp = (bar["high"] >= tp) if side == "BUY" else (bar["low"] <= tp)
            bars_held = i - open_trade["entry_idx"]
            outcome, exit_price = None, None
            if hit_sl:  # conservative: SL checked first if both hit same bar
                outcome, exit_price = "LOSS", sl
            elif hit_tp:
                outcome, exit_price = "WIN", tp
            elif bars_held >= TIMEOUT_BARS:
                outcome, exit_price = "TIMEOUT", bar["close"]

            if outcome is not None:
                r_risk = abs(open_trade["entry"] - sl)
                pnl_price = (exit_price - open_trade["entry"]) if side == "BUY" else (open_trade["entry"] - exit_price)
                r_mult = pnl_price / r_risk if r_risk > 0 else 0.0
                trades.append({
                    "symbol": symbol, "entry_time": str(open_trade["entry_time"]),
                    "exit_time": str(bar["datetime"]), "side": side,
                    "engine": open_trade["engine"], "confidence": open_trade["confidence"],
                    "entry": open_trade["entry"], "sl": sl, "tp": tp,
                    "exit_price": exit_price, "outcome": outcome,
                    "r_multiple": round(r_mult, 3), "bars_held": bars_held,
                })
                open_trade = None
            i += 1
            continue

        # ---- manage a PENDING (not-yet-filled) limit order ----
        if pending is not None:
            side = pending["side"]
            entry_lvl = pending["entry_raw"]
            touched = (bar["low"] <= entry_lvl <= bar["high"])
            bars_pending = i - pending["signal_idx"]
            if touched:
                spread_price = float(bar["spread"]) * POINT if not pd.isna(bar["spread"]) else 0.0
                entry = entry_lvl + spread_price if side == "BUY" else entry_lvl - spread_price
                open_trade = {
                    "side": side, "entry": entry, "sl": pending["sl"], "tp": pending["tp"],
                    "entry_idx": i, "entry_time": bar["datetime"],
                    "engine": pending["engine"], "confidence": pending["confidence"],
                }
                pending = None
                i += 1
                continue
            elif bars_pending >= FILL_TIMEOUT_BARS:
                unfilled += 1
                pending = None
                # fall through to check for a fresh signal this same bar
            else:
                i += 1
                continue

        # ---- no open/pending trade: ask the rule engine for a signal on the just-closed bar ----
        sub = h1.iloc[i - WINDOW:i].reset_index(drop=True)
        try:
            res = eng.analyze(sub, symbol=symbol)
        except Exception:
            i += 1
            continue

        action = res["consensus"]["action"]
        if action in ("BUY", "SELL"):
            engine_name, entry_lvl, sl, tp = extract_trade_levels(res, action)
            if engine_name is not None:
                # Sanity: the sub-engine's own SL/TP must bracket its own
                # entry price -- guards against any future engine change
                # producing degenerate geometry.
                valid = (action == "BUY" and sl < entry_lvl < tp) or (action == "SELL" and tp < entry_lvl < sl)
                if valid:
                    pending = {
                        "side": action, "entry_raw": entry_lvl, "sl": sl, "tp": tp,
                        "signal_idx": i, "engine": engine_name,
                        "confidence": res["consensus"]["confidence"],
                    }
        i += 1

        if (i - WINDOW) % log_every == 0:
            elapsed = time.time() - t_start
            done = i - WINDOW
            total = n - WINDOW
            print(f"[{symbol}] {done}/{total} bars | {len(trades)} trades closed | "
                  f"{unfilled} expired unfilled | elapsed={elapsed:.0f}s", flush=True)

    if i >= n - 1:
        finished = True

    if finished:
        out_df = pd.DataFrame(trades)
        out_df.to_csv(out_path, index=False)
        if Path(ckpt_path).exists():
            Path(ckpt_path).unlink()
        print(f"[{symbol}] DONE. {len(trades)} trades ({unfilled} unfilled) -> {out_path}", flush=True)
        return out_df
    else:
        state = {"i": i, "trades": trades, "unfilled": unfilled,
                  "open_trade": open_trade, "pending": pending}
        with open(ckpt_path, "wb") as f:
            pickle.dump(state, f)
        print(f"[{symbol}] PAUSED at bar {i - WINDOW}/{n - WINDOW} "
              f"({100*(i-WINDOW)/(n-WINDOW):.1f}%) | {len(trades)} trades so far "
              f"| {unfilled} unfilled | checkpoint -> {ckpt_path}", flush=True)
        return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--time_budget", type=float, default=240.0)
    ap.add_argument("--disable_ict", action="store_true")
    ap.add_argument("--disable_pa", action="store_true")
    ap.add_argument("--disable_liquidity", action="store_true")
    ap.add_argument("--cci_min_rr", type=float, default=2.0)
    ap.add_argument("--ckpt", type=str, default=None)
    args = ap.parse_args()
    engine_kwargs = dict(
        enable_ict_amd=not args.disable_ict,
        enable_pa=not args.disable_pa,
        enable_liquidity=not args.disable_liquidity,
        cci_min_rr=args.cci_min_rr,
    )
    run_symbol(args.symbol, args.csv, args.out, args.limit, args.time_budget,
               ckpt_path=args.ckpt, engine_kwargs=engine_kwargs)
