"""
backtest/ml_ensemble_backtest.py — Out-of-sample backtest for the ML ensemble
==============================================================================

Same idea as backtest/rl_policy_backtest.py, but for the SUPERVISED ensemble
(ml/model_trainer.py + ml/model_predictor.py) instead of the RL policy —
so the two can be compared apples-to-apples on the SAME held-out candles
and the SAME cost-accurate execution engine (ForexTradingEnv: spread,
SL/TP, lot sizing).

The ensemble outputs a per-bar BUY/SELL/WAIT signal (not RL actions), so
this script translates that into env actions:
    BUY  -> open long   (if flat)
    SELL -> open short  (if flat)
    WAIT -> hold
No RL-style CLOSE decision — exits are left to the env's own SL/TP, same
as how the ensemble is actually used in this codebase (RL/heuristic layer
decides exits; the ensemble only proposes entries).

Usage:
    python -m backtest.ml_ensemble_backtest --pair EURUSD --timeframe 15m
"""
from __future__ import annotations

import argparse
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("ml_ensemble_backtest")


def run_backtest(pair: str = "EURUSD", timeframe: str = "15m",
                  train_window: int = 5000, test_candles: int = 1500,
                  initial_balance: float = 10000.0,
                  min_confidence: float = 0.55) -> Dict[str, Any]:
    from data.fetcher import DataFetcher
    from ml.feature_engineer import get_feature_engineer
    from ml.model_predictor import ModelPredictor
    from ml.rl_environment import ForexTradingEnv

    # 1. Same held-out slice as the RL backtest, for apples-to-apples comparison
    fetch_limit = train_window + test_candles
    fetcher = DataFetcher()
    mt5_tf = {"15m": "M15", "1h": "H1", "4h": "H4", "1d": "D1"}.get(timeframe, "M15")
    df_all = fetcher.fetch_ohlcv(symbol=pair, timeframe=mt5_tf, limit=fetch_limit)
    if df_all is None or len(df_all) < train_window + 200:
        raise RuntimeError(
            f"Not enough history returned ({0 if df_all is None else len(df_all)} candles) "
            f"to carve out a held-out test slice. Try a smaller --test-candles."
        )
    df_all = df_all.reset_index(drop=True)
    n_test = min(test_candles, len(df_all) - train_window)
    test_df = df_all.iloc[:n_test].reset_index(drop=True)
    log.info(f"[MLBacktest] {pair} {timeframe}: testing on {len(test_df)} candles "
             f"NOT included in the last-{train_window}-candle training window")

    predictor = ModelPredictor()
    if not predictor.is_ready(pair=pair):
        raise RuntimeError(
            f"No trained ML models found for {pair}/{timeframe}. "
            f"Train first with `python -m ml.model_trainer --pair {pair}`."
        )
    engineer = get_feature_engineer()

    # 2. Precompute a BUY/SELL/WAIT signal for every bar (expanding window
    #    only — sub_df = test_df.iloc[:i+1] — no look-ahead).
    signals: List[str] = [None] * len(test_df)
    for i in range(len(test_df)):
        sub_df = test_df.iloc[:i + 1]
        if len(sub_df) < 5:
            continue
        try:
            feats = engineer.build_feature_vector(df=sub_df, analysis_out={}, pair=pair, timeframe=timeframe)
            pred = predictor.predict(features=feats, pair=pair, timeframe=timeframe)
            prob = pred.get("probability", 0.5)
            label = pred.get("prediction", "WAIT")
            # Require the model to actually be confident, not just "not WAIT"
            if label == "BUY" and prob >= min_confidence:
                signals[i] = "BUY"
            elif label == "SELL" and prob >= min_confidence:
                signals[i] = "SELL"
        except Exception as e:
            log.debug(f"[MLBacktest] prediction failed at bar {i}: {e}")

    n_signals = sum(1 for s in signals if s in ("BUY", "SELL"))
    log.info(f"[MLBacktest] ensemble proposed {n_signals} entry signals "
             f"(>= {min_confidence:.0%} confidence) across {len(test_df)} bars")

    # 3. Drive the SAME cost-accurate environment used for the RL backtest.
    #    No features_df needed here (state isn't consumed by an RL policy);
    #    we drive actions directly from `signals`.
    env = ForexTradingEnv(df=test_df, features_df=None,
                           initial_balance=initial_balance, pair=pair)
    obs, info = env.reset()
    done = False
    step_idx = 0
    while not done:
        sig = signals[step_idx] if step_idx < len(signals) else None
        if sig == "BUY" and env.position.direction == "NONE":
            action = 1
        elif sig == "SELL" and env.position.direction == "NONE":
            action = 2
        else:
            action = 0  # HOLD — let SL/TP manage any open position
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_idx += 1

    trades: List[Dict] = getattr(env, "closed_trades", [])
    return _summarize(trades, initial_balance, env.balance, step_idx, pair, timeframe, n_signals)


def _summarize(trades: List[Dict], starting_balance: float, ending_balance: float,
               steps: int, pair: str, timeframe: str, n_signals: int) -> Dict[str, Any]:
    if not trades:
        return {
            "pair": pair, "timeframe": timeframe, "steps": steps,
            "signals_proposed": n_signals, "total_trades": 0,
            "verdict": "NO TRADES TAKEN despite signals — check min_confidence, "
                       "or ensemble genuinely found nothing tradeable in this window.",
        }

    pnls = np.array([t["pnl_usd"] for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    win_rate = len(wins) / len(pnls) * 100
    gross_profit = wins.sum() if len(wins) else 0.0
    gross_loss = abs(losses.sum()) if len(losses) else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    equity = np.concatenate([[starting_balance], starting_balance + np.cumsum(pnls)])
    running_max = np.maximum.accumulate(equity)
    drawdown_pct = (running_max - equity) / running_max * 100
    max_drawdown_pct = float(np.max(drawdown_pct))
    total_return_pct = (ending_balance - starting_balance) / starting_balance * 100

    result = {
        "pair": pair, "timeframe": timeframe, "steps": steps,
        "signals_proposed": n_signals,
        "total_trades": len(trades),
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf (no losing trades)",
        "gross_profit_usd": round(gross_profit, 2),
        "gross_loss_usd": round(gross_loss, 2),
        "total_pnl_usd": round(float(pnls.sum()), 2),
        "max_drawdown_pct": round(max_drawdown_pct, 1),
        "starting_balance": round(starting_balance, 2),
        "ending_balance": round(ending_balance, 2),
        "total_return_pct": round(total_return_pct, 2),
    }

    checks_passed = (
        result["total_trades"] >= 20 and
        win_rate >= 40 and
        (profit_factor >= 1.2 if profit_factor != float("inf") else True) and
        max_drawdown_pct < 25
    )
    result["verdict"] = (
        "Looks reasonable enough for PAPER trading (still not live money)."
        if checks_passed else
        "NOT ready — do not paper/live trade this yet. See metrics above."
    )
    return result


def _print_report(result: Dict[str, Any]) -> None:
    print("=" * 55)
    print("  ML ENSEMBLE — OUT-OF-SAMPLE BACKTEST")
    print("=" * 55)
    for k, v in result.items():
        if k == "verdict":
            continue
        print(f"  {k:22s}: {v}")
    print("-" * 55)
    print(f"  VERDICT: {result.get('verdict')}")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Out-of-sample backtest for the ML ensemble")
    parser.add_argument("--pair", default="EURUSD")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--train-window", type=int, default=5000,
                         help="How many recent candles training used")
    parser.add_argument("--test-candles", type=int, default=1500,
                         help="How many held-out candles to test on")
    parser.add_argument("--balance", type=float, default=10000.0)
    parser.add_argument("--min-confidence", type=float, default=0.55,
                         help="Minimum ensemble probability required to take a signal")
    args = parser.parse_args()

    result = run_backtest(
        pair=args.pair, timeframe=args.timeframe,
        train_window=args.train_window, test_candles=args.test_candles,
        initial_balance=args.balance, min_confidence=args.min_confidence,
    )
    _print_report(result)