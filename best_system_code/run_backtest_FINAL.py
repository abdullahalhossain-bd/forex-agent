"""
Combined system backtest -- mirrors the REAL production precedence found in
AnalysisAgent.run(): primary strategy/signal_engine.py::SignalEngine.generate()
runs first every bar; ONLY when it returns WAIT does the fallback
analysis/unified_signal_engine.py::UnifiedSignalEngine get consulted
(matching the "OPTIONAL -- failure does not break main pipeline" / "only
fill in when nothing upstream already found a trade" comments in the real
code).

Harmful-layer removals applied, each backed by the backtest evidence from
this conversation:
  - Fallback: ICT/AMD sub-engine disabled (enable_ict_amd=False) -- was
    191/354 trades at 11.5% WR, -37R, the majority of the fallback layer's
    loss.
  - Primary: NOT run on GBPSEK -- 278/1087 trades at 21.2% WR, -130.5R,
    clean (non-directional) failure, ~2/3 of the primary layer's total
    loss. GBPSEK is instead traded through the fallback engine only, which
    empirically had its BEST win rate (43.4%) on this exact pair.

Everything else (indicator approximations, pending-order fill model for
the fallback engine's structural entries, direct next-bar-open fill for
the primary engine's trend/pullback entries, spread cost, one-trade-at-a-
time, conservative same-bar SL/TP resolution) is unchanged from the two
prior standalone backtests -- see run_backtest_primary.py and
run_backtest.py for the detailed rationale on each.
"""
import sys
import time
import pickle
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

from strategy.signal_engine import SignalEngine
from analysis.market_regime import MarketRegimeDetector
from analysis.support_resistance import SupportResistance
from analysis.advanced_patterns import AdvancedPatternDetector
from risk.atr_risk_manager import get_stop_loss, get_take_profit
from analysis.unified_signal_engine import UnifiedSignalEngine

from run_backtest_primary import (
    load_h1, compute_indicators, build_ind_ctx, build_mtf_bias,
    MIN_CONFIDENCE, SL_ATR_MULT, TP_ATR_MULT,
)
from run_backtest import extract_trade_levels, WINDOW as FALLBACK_WINDOW

PRIMARY_WINDOW = 250
TIMEOUT_BARS = 200
FILL_TIMEOUT_BARS = 20
POINT = 1e-5


def run_symbol(symbol, csv_path, out_path, time_budget_sec=220.0, ckpt_path=None,
                limit=None, use_primary=True):
    h1_raw = load_h1(csv_path)
    if limit:
        h1_raw = h1_raw.iloc[:limit].reset_index(drop=True)
    h1 = compute_indicators(h1_raw)
    n = len(h1)

    sig_engine = SignalEngine()
    regime_det = MarketRegimeDetector()
    sr_engine = SupportResistance()
    adv_engine = AdvancedPatternDetector()
    fb_engine = UnifiedSignalEngine(timeframe="H1", enable_ict_amd=False)

    ckpt_path = ckpt_path or (out_path + ".ckpt.pkl")
    if Path(ckpt_path).exists():
        with open(ckpt_path, "rb") as f:
            state = pickle.load(f)
        i = state["i"]; trades = state["trades"]; unfilled = state["unfilled"]
        open_trade = state["open_trade"]; pending = state["pending"]
        print(f"[{symbol}] RESUMED at {i-PRIMARY_WINDOW}/{n-PRIMARY_WINDOW}, {len(trades)} trades", flush=True)
    else:
        i = PRIMARY_WINDOW
        trades = []
        unfilled = 0
        open_trade = None
        pending = None

    t_start = time.time()
    log_every = 500

    while i < n - 1:
        if time.time() - t_start > time_budget_sec:
            break
        bar = h1.iloc[i]

        # ---- manage filled open trade ----
        if open_trade is not None:
            side = open_trade["side"]; sl, tp = open_trade["sl"], open_trade["tp"]
            hit_sl = (bar["low"] <= sl) if side == "BUY" else (bar["high"] >= sl)
            hit_tp = (bar["high"] >= tp) if side == "BUY" else (bar["low"] <= tp)
            bars_held = i - open_trade["entry_idx"]
            outcome, exit_price = None, None
            if hit_sl:
                outcome, exit_price = "LOSS", sl
            elif hit_tp:
                outcome, exit_price = "WIN", tp
            elif bars_held >= TIMEOUT_BARS:
                outcome, exit_price = "TIMEOUT", bar["close"]
            if outcome is not None:
                r_risk = abs(open_trade["entry"] - sl)
                pnl = (exit_price - open_trade["entry"]) if side == "BUY" else (open_trade["entry"] - exit_price)
                r_mult = pnl / r_risk if r_risk > 0 else 0.0
                trades.append({
                    "symbol": symbol, "entry_time": str(open_trade["entry_time"]),
                    "exit_time": str(bar["datetime"]), "side": side,
                    "source": open_trade["source"], "confidence": open_trade["confidence"],
                    "signal_type": open_trade["signal_type"],
                    "entry": open_trade["entry"], "sl": sl, "tp": tp,
                    "exit_price": exit_price, "outcome": outcome,
                    "r_multiple": round(r_mult, 3), "bars_held": bars_held,
                })
                open_trade = None
            i += 1
            continue

        # ---- manage pending fallback limit order ----
        if pending is not None:
            side = pending["side"]; entry_lvl = pending["entry_raw"]
            touched = (bar["low"] <= entry_lvl <= bar["high"])
            bars_pending = i - pending["signal_idx"]
            if touched:
                spread_price = float(bar["spread"]) * POINT if not pd.isna(bar["spread"]) else 0.0
                entry = entry_lvl + spread_price if side == "BUY" else entry_lvl - spread_price
                open_trade = {
                    "side": side, "entry": entry, "sl": pending["sl"], "tp": pending["tp"],
                    "entry_idx": i, "entry_time": bar["datetime"],
                    "source": pending["source"], "confidence": pending["confidence"],
                    "signal_type": pending["signal_type"],
                }
                pending = None
                i += 1
                continue
            elif bars_pending >= FILL_TIMEOUT_BARS:
                unfilled += 1
                pending = None
            else:
                i += 1
                continue

        sub_p = h1.iloc[i - PRIMARY_WINDOW:i].reset_index(drop=True)
        sub_fb = h1.iloc[i - FALLBACK_WINDOW:i].reset_index(drop=True)
        if sub_p["atr"].iloc[-1] != sub_p["atr"].iloc[-1] or sub_p["ema_200"].iloc[-1] != sub_p["ema_200"].iloc[-1]:
            i += 1
            continue

        opened = False

        # ---- 1) PRIMARY engine first (skipped entirely for GBPSEK -- confirmed harmful there) ----
        if use_primary:
            try:
                ind_ctx = build_ind_ctx(sub_p)
                regime_result = regime_det.detect(sub_p)
                sr_result = sr_engine.analyze(sub_p, symbol=symbol)
                sr_ctx = sr_engine.get_ai_context(sr_result)
                advanced_pat_ctx = adv_engine.get_ai_context(sub_p, ind_ctx=ind_ctx, sr_ctx=sr_ctx, regime_ctx=regime_result)
                mtf_bias = build_mtf_bias(h1_raw, i)
                p_result = sig_engine.generate(
                    ind_ctx=ind_ctx, pat_ctx={}, sr_ctx=sr_ctx, regime=regime_result,
                    mtf_bias=mtf_bias, advanced_pat_ctx=advanced_pat_ctx, fib_ctx={}, extended_ctx=None,
                )
            except Exception:
                p_result = {"signal": "WAIT", "confidence": 0}

            p_signal = p_result.get("signal", "WAIT")
            p_conf = p_result.get("confidence", 0)
            # ---- Candidate D: adaptive confirmation (NOT a hard block) ----
            # Evidence (see full_loss_mechanism_report.md): NY + London-NY
            # overlap sessions accounted for -74.5R of -75.2R total PRIMARY
            # loss. Rather than blocking those sessions outright (Candidate
            # F: worked, but blocked ~50% of ALL trades and had a weak OOS
            # month), only the highest confidence tier (>=85) is allowed to
            # trade during those two sessions; every other session keeps
            # the normal MIN_CONFIDENCE floor unchanged.
            hour = pd.Timestamp(bar["datetime"]).hour
            bad_session = (12 <= hour < 21)  # LONDON_NY_OVERLAP (12-16) + NY (16-21)
            required_conf = 85 if bad_session else MIN_CONFIDENCE
            if p_signal in ("BUY", "STRONG_BUY", "SELL", "STRONG_SELL") and p_conf >= required_conf:
                side = "BUY" if "BUY" in p_signal else "SELL"
                fill_bar = h1.iloc[i + 1]
                spread_price = float(fill_bar["spread"]) * POINT if not pd.isna(fill_bar["spread"]) else 0.0
                raw_entry = fill_bar["open"]
                entry = raw_entry + spread_price if side == "BUY" else raw_entry - spread_price
                atr_val = float(bar["atr"])
                sl = get_stop_loss(side, entry, atr_val, SL_ATR_MULT)
                tp = get_take_profit(side, entry, atr_val, TP_ATR_MULT)
                valid = (side == "BUY" and sl < entry < tp) or (side == "SELL" and tp < entry < sl)
                if valid:
                    open_trade = {
                        "side": side, "entry": entry, "sl": sl, "tp": tp,
                        "entry_idx": i + 1, "entry_time": fill_bar["datetime"],
                        "source": "PRIMARY", "confidence": p_conf, "signal_type": p_signal,
                    }
                    opened = True

        # ---- 2) FALLBACK engine only if primary didn't open a trade this bar ----
        if not opened:
            try:
                fb_result = fb_engine.analyze(sub_fb, symbol=symbol)
            except Exception:
                fb_result = None
            if fb_result is not None:
                action = fb_result["consensus"]["action"]
                if action in ("BUY", "SELL"):
                    engine_name, entry_lvl, sl, tp = extract_trade_levels(fb_result, action)
                    if engine_name is not None:
                        valid = (action == "BUY" and sl < entry_lvl < tp) or (action == "SELL" and tp < entry_lvl < sl)
                        if valid:
                            pending = {
                                "side": action, "entry_raw": entry_lvl, "sl": sl, "tp": tp,
                                "signal_idx": i, "source": f"FALLBACK:{engine_name}",
                                "confidence": fb_result["consensus"]["confidence"],
                                "signal_type": action,
                            }
        i += 1

        if (i - PRIMARY_WINDOW) % log_every == 0:
            elapsed = time.time() - t_start
            print(f"[{symbol}] {i-PRIMARY_WINDOW}/{n-PRIMARY_WINDOW} bars | {len(trades)} trades | "
                  f"{unfilled} fb-unfilled | elapsed={elapsed:.0f}s", flush=True)

    if i >= n - 1:
        out_df = pd.DataFrame(trades)
        out_df.to_csv(out_path, index=False)
        if Path(ckpt_path).exists():
            Path(ckpt_path).unlink()
        print(f"[{symbol}] DONE. {len(trades)} trades -> {out_path}", flush=True)
        return out_df
    else:
        with open(ckpt_path, "wb") as f:
            pickle.dump({"i": i, "trades": trades, "unfilled": unfilled,
                         "open_trade": open_trade, "pending": pending}, f)
        print(f"[{symbol}] PAUSED at {i-PRIMARY_WINDOW}/{n-PRIMARY_WINDOW} "
              f"({100*(i-PRIMARY_WINDOW)/(n-PRIMARY_WINDOW):.1f}%) | {len(trades)} trades", flush=True)
        return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--time_budget", type=float, default=220.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no_primary", action="store_true")
    args = ap.parse_args()
    run_symbol(args.symbol, args.csv, args.out, args.time_budget,
               limit=args.limit, use_primary=not args.no_primary)
