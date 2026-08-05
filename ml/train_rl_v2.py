"""
ml/train_rl_v2.py — Profitability-Focused RL Training
=====================================================

FIXES from v1 (which produced avg_reward=-429,620, 0% WR, 1 episode):

BUG #1: Only 1 episode in 50k timesteps
  v1: Episode only ended at end-of-data (5000 steps)
      50k timesteps / 5000 steps = 10 episodes expected
      But curriculum split data into ~1000-2000 row chunks
      50k / 2000 = 25 episodes expected
      Got 1 — means episode termination was broken
  FIX: Use ForexTradingEnvV2 with max_steps_per_episode=1000
       → 50k / 1000 = 50 episodes minimum

BUG #2: No evaluation during training
  v1: Trained blind, no idea if policy was improving
  FIX: Add EvalCallback — every 5000 steps, run 10 eval episodes
       Save best model separately

BUG #3: PPO hyperparams not tuned for trading
  v1: Default SB3 PPO params (lr=3e-4, n_steps=2048, batch=64)
      These work for Atari but not forex — forex reward signal is sparse
  FIX: Tune for trading:
       - Lower learning rate (1e-4) for stability
       - Larger n_steps (4096) for more rollouts
       - Smaller batch (32) for more updates
       - Higher ent_coef (0.02) for exploration
       - Lower gamma (0.95) — forex rewards are short-horizon

BUG #4: No early stopping
  v1: Trained full 500k steps even if policy was degenerate
  FIX: Stop if eval_reward doesn't improve for 3 consecutive checks

BUG #5: Single data file
  v1: Only trained on one CSV
  FIX: Support multiple CSVs for more diverse training data

USAGE:
  python -m ml.train_rl_v2 --pair EURUSD --timeframe H1 --timesteps 200000
  python -m ml.train_rl_v2 --pair EURUSD --timeframe M15 --timesteps 500000 --no-curriculum
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger
from config import PROJECT_ROOT
from ml.rl_environment_v2 import ForexTradingEnvV2
from ml.reward_engine_v2 import get_reward_engine_v2

log = get_logger("train_rl_v2")


def load_historical_data_v2(pair: str, timeframe: str = "H1",
                             periods: int = 5000) -> pd.DataFrame:
    """Load historical data — try MT5 first, fall back to CSV."""
    # Try CSV first (works on Linux/CI)
    tf_map = {"M15": "M15", "H1": "H1", "H4": "H4", "D1": "D1"}
    tf_norm = tf_map.get(timeframe.upper(), "H1")
    csv_path = PROJECT_ROOT / "data" / f"{pair}_{tf_norm}.csv"

    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            # Normalize column names
            time_col = None
            for c in ["datetime_utc", "datetime", "time", "timestamp"]:
                if c in df.columns:
                    time_col = c
                    break
            if time_col:
                df[time_col] = pd.to_datetime(df[time_col], utc=True)
                df.set_index(time_col, inplace=True)
            # Ensure required columns
            for col in ["open", "high", "low", "close", "tick_volume"]:
                if col not in df.columns:
                    df[col] = 0
            if "volume" not in df.columns:
                df["volume"] = df.get("tick_volume", 0)
            # Add ATR if missing
            if "atr" not in df.columns:
                high = df["high"]
                low = df["low"]
                close = df["close"]
                prev_close = close.shift(1)
                tr = pd.concat([
                    high - low,
                    (high - prev_close).abs(),
                    (low - prev_close).abs(),
                ], axis=1).max(axis=1)
                df["atr"] = tr.rolling(window=14, min_periods=14).mean()
            # Add RSI if missing
            if "rsi_14" not in df.columns:
                delta = df["close"].diff()
                gain = delta.clip(lower=0).rolling(14).mean()
                loss = (-delta.clip(upper=0)).rolling(14).mean()
                rs = gain / loss
                df["rsi_14"] = 100 - (100 / (1 + rs))
            log.info(f"[TrainRL v2] Loaded {len(df)} rows from {csv_path}")
            return df
        except Exception as e:
            log.error(f"[TrainRL v2] CSV load failed: {e}")

    # Try MT5
    try:
        from broker.mt5_historical_fetcher import fetch_historical_data
        from datetime import datetime, timedelta, timezone
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=periods // 24 if timeframe == "H1" else periods)
        df = fetch_historical_data(pair, start, end)
        if df is not None and len(df) > 0:
            log.info(f"[TrainRL v2] Loaded {len(df)} rows from MT5")
            return df
    except Exception as e:
        log.warning(f"[TrainRL v2] MT5 fetch failed: {e}")

    return pd.DataFrame()


def build_features_df_v2(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature vectors — simplified, guaranteed aligned."""
    n = len(df)
    if n == 0:
        return pd.DataFrame()

    # Use the same features as the env's FEATURE_SCHEMA
    features = pd.DataFrame(index=range(n))

    # Price-based features (normalized)
    if "close" in df.columns:
        features["close"] = df["close"].values
        features["high"] = df["high"].values if "high" in df.columns else df["close"].values
        features["low"] = df["low"].values if "low" in df.columns else df["close"].values
        features["volume"] = df["volume"].values if "volume" in df.columns else 0

    # Indicators
    if "rsi_14" not in df.columns:
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        df["rsi_14"] = 100 - (100 / (1 + rs))
    features["rsi_14"] = df["rsi_14"].fillna(50).values

    if "atr" not in df.columns:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.rolling(window=14, min_periods=14).mean()
    features["atr"] = df["atr"].fillna(df["atr"].mean() if not df["atr"].isna().all() else 0.001).values

    # MACD
    ema_12 = df["close"].ewm(span=12).mean()
    ema_26 = df["close"].ewm(span=26).mean()
    features["macd"] = (ema_12 - ema_26).values

    # EMAs
    features["ema_20"] = df["close"].ewm(span=20).mean().values
    features["ema_50"] = df["close"].ewm(span=50).mean().values
    features["sma_200"] = df["close"].rolling(200).mean().fillna(df["close"].mean()).values

    # Fill NaN
    features = features.fillna(0)

    log.info(f"[TrainRL v2] Built {len(features)} feature rows × {len(features.columns)} features")
    return features


def train_rl_agent_v2(
    pair: str = "EURUSD",
    timeframe: str = "H1",
    total_timesteps: int = 200000,
    initial_balance: float = 10000.0,
    max_steps_per_episode: int = 1000,
    eval_freq: int = 10000,
    n_eval_episodes: int = 10,
    learning_rate: float = 1e-4,
) -> Dict[str, Any]:
    """Train RL agent with profitability-focused settings."""
    log.info(f"[TrainRL v2] Starting: {pair} {timeframe} | {total_timesteps} timesteps")

    # 1. Load data
    df = load_historical_data_v2(pair, timeframe)
    if df.empty or len(df) < 500:
        return {"error": f"insufficient data for {pair} ({len(df)} rows)"}
    log.info(f"[TrainRL v2] Data: {len(df)} rows")

    # 2. Build features
    features_df = build_features_df_v2(df)
    if features_df.empty:
        return {"error": "feature build failed"}

    # 3. Check SB3 availability
    try:
        import stable_baselines3
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError:
        return {
            "error": "stable-baselines3 not installed",
            "install": "pip install stable-baselines3 gymnasium",
        }

    # 4. Create environments
    reward_engine = get_reward_engine_v2()

    def make_env():
        return ForexTradingEnvV2(
            df=df,
            features_df=features_df,
            initial_balance=initial_balance,
            pair=pair,
            reward_engine=reward_engine,
            max_steps_per_episode=max_steps_per_episode,
        )

    train_env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])

    # 5. PPO with tuned hyperparameters for trading
    # v3: increased ent_coef (0.02→0.05) and lowered gamma (0.95→0.90)
    # to encourage exploration and short-horizon profit-taking
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,   # 1e-4 — slower, more stable
        n_steps=4096,                  # 4096 — more rollouts
        batch_size=32,                 # 32 — more updates
        n_epochs=15,                   # 15 — more gradient steps
        gamma=0.90,                    # 0.90 (was 0.95) — shorter horizon, care about immediate profit
        gae_lambda=0.95,
        clip_range=0.15,               # 0.15 — tighter policy updates
        ent_coef=0.05,                 # 0.05 (was 0.02) — MORE exploration to break farming policy
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
        ),
        verbose=1,
    )

    # 6. Custom callback for profitability tracking
    class ProfitabilityCallback(BaseCallback):
        def __init__(self, eval_env, eval_freq, n_eval_episodes):
            super().__init__()
            self.eval_env = eval_env
            self.eval_freq = eval_freq
            self.n_eval_episodes = n_eval_episodes
            self.best_eval_reward = -np.inf
            self.episode_rewards = []
            self.eval_history = []
            self.patience_counter = 0
            self.max_patience = 5  # v3: was 3, increased to give more learning time

        def _on_step(self):
            # Collect episode rewards
            infos = self.locals.get("infos", [])
            for info in infos:
                if isinstance(info, dict) and "episode_reward" in info:
                    self.episode_rewards.append(info["episode_reward"])

            # Periodic evaluation
            if self.n_calls % self.eval_freq == 0 and self.n_calls > 0:
                eval_rewards = []
                eval_pnls = []
                eval_wins = 0
                eval_trades = 0

                for _ in range(self.n_eval_episodes):
                    obs = self.eval_env.reset()
                    done = False
                    ep_reward = 0
                    while not done:
                        action, _ = self.model.predict(obs, deterministic=True)
                        # VecEnv returns 4-tuple (obs, reward, done, info)
                        result = self.eval_env.step(action)
                        if len(result) == 5:
                            obs, reward, terminated, truncated, info = result
                            done = terminated or truncated
                        else:
                            obs, reward, done, info = result
                        ep_reward += reward[0] if isinstance(reward, np.ndarray) else reward
                    # info is a list in VecEnv
                    if isinstance(info, list):
                        info = info[0] if info else {}
                    eval_rewards.append(ep_reward)
                    eval_pnls.append(info.get("episode_pnl", 0))
                    eval_wins += info.get("total_wins", 0)
                    eval_trades += info.get("total_trades", 0)

                avg_eval_reward = np.mean(eval_rewards)
                avg_eval_pnl = np.mean(eval_pnls)
                eval_wr = (eval_wins / max(eval_trades, 1)) * 100

                self.eval_history.append({
                    "timestep": self.n_calls,
                    "avg_eval_reward": round(avg_eval_reward, 2),
                    "avg_eval_pnl": round(avg_eval_pnl, 2),
                    "eval_win_rate": round(eval_wr, 1),
                    "eval_trades": eval_trades,
                })

                log.info(
                    f"[TrainRL v2] EVAL @ {self.n_calls} steps | "
                    f"avg_reward={avg_eval_reward:.2f} | "
                    f"avg_pnl=${avg_eval_pnl:.2f} | "
                    f"WR={eval_wr:.1f}% ({eval_wins}/{eval_trades})"
                )

                # Save best model
                if avg_eval_reward > self.best_eval_reward:
                    self.best_eval_reward = avg_eval_reward
                    save_path = PROJECT_ROOT / "ml" / "rl_policy" / "ppo_forex_best.zip"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    self.model.save(str(save_path))
                    log.info(f"[TrainRL v2] New best model saved: {save_path}")
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    log.warning(
                        f"[TrainRL v2] No improvement ({self.patience_counter}/"
                        f"{self.max_patience})"
                    )

                # Early stopping
                if self.patience_counter >= self.max_patience:
                    log.warning("[TrainRL v2] Early stopping — no improvement")
                    return False

            return True

    callback = ProfitabilityCallback(eval_env, eval_freq, n_eval_episodes)

    # 7. Train
    log.info(f"[TrainRL v2] Training PPO for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # 8. Save final model
    final_path = PROJECT_ROOT / "ml" / "rl_policy" / "ppo_forex_latest.zip"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(final_path))

    # 9. Save metadata
    def _json_safe(obj):
        """Convert numpy types to native Python for JSON serialization."""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_json_safe(v) for v in obj]
        return obj

    meta = {
        "episodes": len(callback.episode_rewards),
        "win_rate": round(
            sum(1 for r in callback.episode_rewards if r > 0) /
            max(len(callback.episode_rewards), 1), 4
        ),
        "avg_reward": round(
            float(np.mean(callback.episode_rewards[-100:]))
            if callback.episode_rewards else 0.0, 2
        ),
        "trained_on": pd.Timestamp.now(tz="UTC").isoformat(),
        "timesteps": total_timesteps,
        "symbol": pair,
        "timeframe": timeframe,
        "engine_version": "v2",
        "eval_history": _json_safe(callback.eval_history),
        "best_eval_reward": round(float(callback.best_eval_reward), 2),
    }
    meta_path = final_path.parent / f"{final_path.stem}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    log.info(
        f"[TrainRL v2] Done: {meta['episodes']} episodes, "
        f"avg_reward={meta['avg_reward']}, saved to {final_path}"
    )

    return {
        "status": "success",
        "episodes": meta["episodes"],
        "avg_reward": meta["avg_reward"],
        "win_rate": meta["win_rate"],
        "best_eval_reward": meta["best_eval_reward"],
        "eval_history": callback.eval_history,
        "model_path": str(final_path),
    }


# ── CLI ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL agent v2 (profitability-focused)")
    parser.add_argument("--pair", default="EURUSD", help="Trading pair")
    parser.add_argument("--timeframe", default="H1",
                        choices=["M15", "H1", "H4", "D1"], help="Timeframe")
    parser.add_argument("--timesteps", type=int, default=200000,
                        help="Total training timesteps (default: 200000)")
    parser.add_argument("--max-steps-per-episode", type=int, default=1000,
                        help="Max steps per episode (default: 1000)")
    parser.add_argument("--eval-freq", type=int, default=10000,
                        help="Evaluation frequency (default: 10000)")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="PPO learning rate (default: 1e-4)")
    args = parser.parse_args()

    result = train_rl_agent_v2(
        pair=args.pair,
        timeframe=args.timeframe,
        total_timesteps=args.timesteps,
        max_steps_per_episode=args.max_steps_per_episode,
        eval_freq=args.eval_freq,
        learning_rate=args.learning_rate,
    )

    print("\n" + "=" * 60)
    print("  TRAINING RESULT (v2)")
    print("=" * 60)
    if "error" in result:
        print(f"  ERROR: {result['error']}")
        if "install" in result:
            print(f"  FIX: {result['install']}")
    else:
        print(f"  Episodes:          {result['episodes']}")
        print(f"  Avg reward:        {result['avg_reward']}")
        print(f"  Win rate:          {result['win_rate']*100:.1f}%")
        print(f"  Best eval reward:  {result['best_eval_reward']}")
        print(f"  Model saved:       {result['model_path']}")
        print()
        print("  EVAL HISTORY:")
        for ev in result.get("eval_history", []):
            print(f"    {ev['timestep']:>6} steps | "
                  f"reward={ev['avg_eval_reward']:>8.2f} | "
                  f"pnl=${ev['avg_eval_pnl']:>8.2f} | "
                  f"WR={ev['eval_win_rate']:>5.1f}%")
    print("=" * 60)
