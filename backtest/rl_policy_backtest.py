"""
backtest/rl_policy_backtest.py — Out-of-sample backtest for the trained RL policy
==================================================================================

Loads the saved PPO policy (ml/rl_policy/ppo_forex_latest.zip) and drives it
through candles it was NEVER trained on, using the same ForexTradingEnv used
for training (so costs/PnL/SL-TP logic are identical between train and test —
no "the backtest scores it differently than training" mismatch).

IMPORTANT — what "out-of-sample" means here:
  ml/train_rl.py fetches the *latest* N candles and trains on all of them.
  There usually isn't fresh "future" data sitting around right after training
  finishes. So this script fetches a LONGER history and tests on the OLDER
  slice that sits before the window train_rl.py used — the model never saw
  these candles during training, even though they aren't literally "the
  future". Once real time passes and new candles accumulate past the
  training cutoff, re-run this against those for a truer forward test.

Usage:
    python -m backtest.rl_policy_backtest --pair EURUSD --timeframe 15m
"""
from __future__ import annotations

import argparse
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("rl_policy_backtest")


def run_backtest(pair: str = "EURUSD", timeframe: str = "15m",
                  train_window: int = 5000, test_candles: int = 1500,
                  initial_balance: float = 10000.0) -> Dict[str, Any]:
    from data.fetcher import DataFetcher
    from ml.train_rl import build_features_df
    from ml.rl_environment import ForexTradingEnv
    from ml.rl_agent import get_rl_agent

    # 1. Fetch a longer history than training used, so we have an
    #    older, never-trained-on slice to test on.
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
    log.info(f"[Backtest] {pair} {timeframe}: testing on {len(test_df)} candles "
             f"NOT included in the last-{train_window}-candle training window")

    # 2. Build features the SAME way training did (already fixed to be 1:1
    #    row-aligned with df — see build_features_df in ml/train_rl.py).
    features_df = build_features_df(test_df, pair)
    if features_df is None or features_df.empty or len(features_df) != len(test_df):
        log.warning("[Backtest] feature build failed or misaligned — env will fall back "
                    "to raw OHLCV state (less informative, but not wrong).")
        features_df = None

    # 3. Load the trained policy
    agent = get_rl_agent()
    if not agent._model_loaded:
        raise RuntimeError(
            "No trained PPO model is loaded (ml/rl_policy/ppo_forex_latest.zip missing "
            "or failed the quality gate). Train first with `python -m ml.train_rl`."
        )

    # 4. Drive the (already-fixed) environment with the policy, deterministic
    env = ForexTradingEnv(df=test_df, features_df=features_df,
                           initial_balance=initial_balance, pair=pair)
    obs, info = env.reset()
    done = False
    steps = 0
    while not done:
        action_result = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action_result.action)
        done = terminated or truncated
        steps += 1

    trades: List[Dict] = getattr(env, "closed_trades", [])
    return _summarize(trades, initial_balance, env.balance, steps, pair, timeframe)


def _summarize(trades: List[Dict], starting_balance: float, ending_balance: float,
               steps: int, pair: str, timeframe: str) -> Dict[str, Any]:
    if not trades:
        return {
            "pair": pair, "timeframe": timeframe, "steps": steps,
            "total_trades": 0, "verdict": "NO TRADES TAKEN — nothing to evaluate. "
            "Policy may be too conservative, or the held-out window is too short.",
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

    # Simple, conservative gate — same spirit as backtest/honest_backtest_engine.py's
    # deploy checks, just inline here so this script is self-contained.
    checks_passed = (
        result["total_trades"] >= 20 and
        win_rate >= 40 and
        (profit_factor >= 1.2 if profit_factor != float("inf") else True) and
        max_drawdown_pct < 25
    )
    result["verdict"] = (
        "Looks reasonable enough for PAPER trading (still not live money)."
        if checks_passed else
        "NOT ready — do not paper/live trade this policy yet. See metrics above."
    )
    return result


def _print_report(result: Dict[str, Any]) -> None:
    print("=" * 55)
    print("  RL POLICY — OUT-OF-SAMPLE BACKTEST")
    print("=" * 55)
    for k, v in result.items():
        if k == "verdict":
            continue
        print(f"  {k:22s}: {v}")
    print("-" * 55)
    print(f"  VERDICT: {result.get('verdict')}")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Out-of-sample backtest for the trained RL policy")
    parser.add_argument("--pair", default="EURUSD")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--train-window", type=int, default=5000,
                         help="How many recent candles training used (must match your ml.train_rl run)")
    parser.add_argument("--test-candles", type=int, default=1500,
                         help="How many held-out candles to test on")
    parser.add_argument("--balance", type=float, default=10000.0)
    args = parser.parse_args()

    result = run_backtest(
        pair=args.pair, timeframe=args.timeframe,
        train_window=args.train_window, test_candles=args.test_candles,
        initial_balance=args.balance,
    )
    _print_report(result)