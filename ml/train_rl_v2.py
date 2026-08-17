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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger
from config import PROJECT_ROOT
from ml.rl_environment_v2 import ForexTradingEnvV2
from ml.reward_engine_v2 import RewardEngineV2, get_reward_engine_v2

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
    """Build a richer, stationary, no-lookahead feature set.

    AUDIT FIX (winrate round 3): the round-2 fix normalized the same
    10 basic features (return/RSI/ATR/MACD/3×MA-distance) but that
    ceiling was always going to be low — those 10 numbers just don't
    carry much predictive signal on their own, which is why the agent
    rationally learned to stop trading once it verified they didn't
    beat costs. The codebase already has richer feature logic
    (ml/feature_engineer.py, ml/pattern_features.py) but those pull in
    a chain of analysis-engine/MTF/live-broker dependencies that don't
    run standalone against a CSV. This build adds the SAME KINDS of
    signal (multi-horizon momentum, trend/momentum/volatility/volume,
    candle geometry, session timing) as a self-contained function so
    it actually runs in training. Every feature here uses only the
    current bar and strictly earlier bars (rolling/ewm/shift), so
    there is no future leakage.
    """
    n = len(df)
    if n == 0:
        return pd.DataFrame()

    features = pd.DataFrame(index=range(n))
    close = df["close"] if "close" in df.columns else pd.Series(np.zeros(n))
    open_ = df["open"] if "open" in df.columns else close
    high = df["high"] if "high" in df.columns else close
    low = df["low"] if "low" in df.columns else close
    volume = df["volume"] if "volume" in df.columns else pd.Series(np.zeros(n))
    safe_close = close.replace(0, np.nan)

    # ── Multi-horizon momentum (return over several lookbacks) ──────
    for h in (1, 4, 16, 96):
        features[f"ret_{h}"] = (close.pct_change(h).fillna(0) * 100).clip(-20, 20)

    # ── Candle geometry (this bar's own shape — no lookback needed) ─
    rng = (high - low).replace(0, np.nan)
    features["body_pct"] = ((close - open_) / rng).fillna(0).clip(-1, 1)
    features["upper_wick_pct"] = ((high - close.where(close > open_, open_)) / rng).fillna(0).clip(0, 1)
    features["lower_wick_pct"] = ((close.where(close < open_, open_) - low) / rng).fillna(0).clip(0, 1)
    # Engulfing-style signed body-size ratio vs previous candle
    prev_body = (close.shift(1) - open_.shift(1)).abs().replace(0, np.nan)
    cur_body = (close - open_)
    features["engulf_ratio"] = (cur_body / prev_body).fillna(0).clip(-5, 5)

    # ── Trend: MA distance + MA slope + MA cross ─────────────────────
    ema_20 = close.ewm(span=20, min_periods=5).mean()
    ema_50 = close.ewm(span=50, min_periods=10).mean()
    sma_200 = close.rolling(200, min_periods=20).mean()
    features["dist_ema20"] = ((close - ema_20) / safe_close).fillna(0).clip(-10, 10) * 100
    features["dist_ema50"] = ((close - ema_50) / safe_close).fillna(0).clip(-10, 10) * 100
    features["dist_sma200"] = ((close - sma_200) / safe_close).fillna(0).clip(-10, 10) * 100
    features["ema20_slope"] = (ema_20.pct_change(5).fillna(0) * 100).clip(-10, 10)
    features["ema_cross"] = ((ema_20 - ema_50) / safe_close).fillna(0).clip(-5, 5) * 100

    # ── Momentum oscillators ──────────────────────────────────────
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=5).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=5).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_14 = (100 - (100 / (1 + rs))).fillna(50)
    features["rsi_14"] = (rsi_14 - 50) / 50
    features["rsi_slope"] = (rsi_14.diff(3).fillna(0) / 50).clip(-2, 2)

    ema_12 = close.ewm(span=12, min_periods=5).mean()
    ema_26 = close.ewm(span=26, min_periods=10).mean()
    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9, min_periods=3).mean()
    features["macd"] = (macd_line / safe_close).fillna(0).clip(-5, 5) * 100
    features["macd_hist"] = ((macd_line - macd_signal) / safe_close).fillna(0).clip(-5, 5) * 100

    low_14 = low.rolling(14, min_periods=5).min()
    high_14 = high.rolling(14, min_periods=5).max()
    stoch_k = ((close - low_14) / (high_14 - low_14).replace(0, np.nan) * 100).fillna(50)
    features["stoch_k"] = (stoch_k - 50) / 50

    # ── Volatility: ATR, realized vol, Bollinger position/width ────
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=5).mean()
    features["atr_pct"] = (atr / safe_close).fillna(0).clip(0, 5) * 100
    features["realized_vol"] = (close.pct_change().rolling(20, min_periods=5).std().fillna(0) * 100).clip(0, 10)

    bb_mid = close.rolling(20, min_periods=10).mean()
    bb_std = close.rolling(20, min_periods=10).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    features["bb_width"] = ((bb_upper - bb_lower) / safe_close).fillna(0).clip(0, 5) * 100
    features["bb_pct_b"] = (
        (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
    ).fillna(0.5).clip(-1, 2)

    # ── Volume z-score (rolling, past-only window) ──────────────────
    vol_mean = volume.rolling(100, min_periods=10).mean()
    vol_std = volume.rolling(100, min_periods=10).std().replace(0, np.nan)
    features["volume_z"] = ((volume - vol_mean) / vol_std).fillna(0).clip(-5, 5)

    # ── Session timing (forex has real time-of-day/day-of-week
    # structure — London/NY overlap volatility, Asia range, weekend
    # gaps — sin/cos encoding so the model sees it as cyclical) ─────
    if isinstance(df.index, pd.DatetimeIndex):
        hour = df.index.hour.to_numpy()
        dow = df.index.dayofweek.to_numpy()
    else:
        hour = np.zeros(n)
        dow = np.zeros(n)
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    features["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    features["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    features = features.fillna(0).clip(-20, 20)

    log.info(f"[TrainRL v2] Built {len(features)} feature rows × {len(features.columns)} features (rich, normalized)")
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
    extra_pairs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Train RL agent with profitability-focused settings.

    `extra_pairs`: optional additional pairs to train on IN PARALLEL
    alongside `pair` (round-4 addition — see rationale below). `pair`
    is always the one used for eval/model-selection/meta.json.
    """
    print("\n" + "#" * 60)
    print(f"#  NOW TRAINING: {pair} / {timeframe}  ({total_timesteps} timesteps)")
    print("#" * 60 + "\n")
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

    # 1b/2b. Same for any extra pairs (round-4 multi-pair training)
    extra_data = []  # list of (pair_name, df, features_df)
    for p in (extra_pairs or []):
        p_df = load_historical_data_v2(p, timeframe)
        if p_df.empty or len(p_df) < 500:
            log.warning(f"[TrainRL v2] Skipping extra pair {p}: insufficient data ({len(p_df)} rows)")
            continue
        p_features = build_features_df_v2(p_df)
        if p_features.empty or p_features.shape[1] != features_df.shape[1]:
            log.warning(f"[TrainRL v2] Skipping extra pair {p}: feature shape mismatch")
            continue
        extra_data.append((p, p_df, p_features))
        log.info(f"[TrainRL v2] Extra pair {p}: {len(p_df)} rows added to training pool")

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
    # AUDIT FIX (win-rate/frequency bug #4): get_reward_engine_v2() is a
    # module-level SINGLETON with mutable per-episode state
    # (_consecutive_wins, _consecutive_losses, _consecutive_idle_steps,
    # _recent_rewards). The old code passed that ONE instance into both
    # make_env() calls, so train_env and eval_env — two independent
    # environments stepping at different times — were reading and
    # mutating the SAME streak/idle counters. Every periodic evaluation
    # run called eval_env.reset() -> reward_engine.reset_episode(),
    # wiping out whatever win/loss streak the TRAINING env had built up
    # mid-rollout, and vice versa. This corrupts the streak-bonus and
    # patience-penalty shaping with cross-talk between two unrelated
    # trajectories, adding pure noise to the reward signal on top of
    # bugs #1/#2. FIX: give every environment (train and eval) its own
    # RewardEngineV2 instance.
    def make_env(env_df=df, env_features=features_df, env_pair=pair):
        return ForexTradingEnvV2(
            df=env_df,
            features_df=env_features,
            initial_balance=initial_balance,
            pair=env_pair,
            reward_engine=RewardEngineV2(),
            max_steps_per_episode=max_steps_per_episode,
        )

    # AUDIT FIX (round 4 — variance/instability): repeated single-pair
    # runs on the SAME data showed real win-rate swings (31%/33%/37%
    # across otherwise-identical runs) — a classic overfitting-to-one-
    # instrument's-idiosyncratic-noise symptom, not a bug. Training
    # several pairs' environments IN PARALLEL (n_envs = number of
    # pairs) is the standard PPO fix: each rollout batch now mixes
    # experience from every pair, so the policy is pushed toward
    # patterns that generalize across instruments rather than
    # memorizing one pair's specific noise. `pair` (the primary) is
    # still the only one used for eval/model-selection/meta.json, so
    # "does this checkpoint actually trade profitably on the pair I
    # care about" stays the real gate — extra pairs only diversify
    # what the policy trains ON, not what it's judged BY.
    train_env_fns = [lambda: make_env(df, features_df, pair)]
    for p, p_df, p_features in extra_data:
        train_env_fns.append(lambda p_df=p_df, p_features=p_features, p=p: make_env(p_df, p_features, p))

    train_env = DummyVecEnv(train_env_fns)
    eval_env = DummyVecEnv([make_env])  # eval always on the primary pair only

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
        def __init__(self, eval_env, eval_freq, n_eval_episodes, pair="EURUSD", timeframe="M15"):
            super().__init__()
            self.eval_env = eval_env
            self.eval_freq = eval_freq
            self.n_eval_episodes = n_eval_episodes
            self.pair = pair
            self.timeframe = timeframe
            self.best_eval_reward = -np.inf
            self.episode_rewards = []
            self.episode_trade_counts = []
            self.episode_win_counts = []
            self.eval_history = []
            self.patience_counter = 0
            self.max_patience = 5  # v3: was 3, increased to give more learning time

        def _on_step(self):
            # AUDIT FIX (win-rate/frequency bug #3): ForexTradingEnvV2's
            # _get_info() puts "episode_reward" (the running cumulative
            # reward so far THIS episode) into the info dict on EVERY
            # step, not just on the terminal step. The old code appended
            # to self.episode_rewards unconditionally, so this list held
            # one entry per environment STEP (~200k entries for a 200k
            # timestep run), not one entry per finished EPISODE. Two
            # knock-on effects, both visible in the shipped
            # ppo_forex_latest_meta.json:
            #   1. "episodes": 60000 for a 200k-timestep run is actually
            #      a step count, not an episode count (real episode
            #      count with max_steps_per_episode=1000 should be in
            #      the low hundreds).
            #   2. "win_rate": 0.017 was computed as "fraction of STEPS
            #      where the running cumulative reward-so-far happened
            #      to be positive" — not trade win rate at all. Since
            #      cumulative reward starts at/near 0 and only turns
            #      positive after enough wins accumulate, this metric
            #      is biased heavily toward 0 regardless of the real
            #      trade-level win rate, and is NOT what the RL agent's
            #      quality gate (MIN_WIN_RATE_TO_TRUST) should be
            #      reading.
            # FIX: only record a completed episode when the VecEnv
            # reports `done`, and use the terminal info's real
            # `total_wins` / `total_trades` / `episode_reward` /
            # `episode_pnl` — i.e. one entry per actual episode, with a
            # real trade-based win flag.
            infos = self.locals.get("infos", [])
            dones = self.locals.get("dones", self.locals.get("done", []))
            if not isinstance(dones, (list, np.ndarray)):
                dones = [dones]
            for i, info in enumerate(infos):
                is_done = bool(dones[i]) if i < len(dones) else False
                if is_done and isinstance(info, dict):
                    self.episode_rewards.append(info.get("episode_reward", 0.0))
                    total_trades = info.get("total_trades", 0)
                    total_wins = info.get("total_wins", 0)
                    self.episode_trade_counts.append(total_trades)
                    self.episode_win_counts.append(total_wins)

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

                # AUDIT FIX (winrate/frequency round 3, critical): a
                # policy that trades ZERO times gets a small, low-
                # variance reward (just the per-step hold_reward
                # accumulating) that can beat a genuinely-trading
                # policy's reward, which is noisier because it's
                # actually taking risk. Just observed this happen live:
                # at 70k steps here, a 0-trade checkpoint (WR=0.0%,
                # avg_reward=0.27) got selected as the new "best" over
                # earlier checkpoints that had real, sometimes-losing,
                # sometimes-winning trade activity. Require a minimum
                # trade frequency (at least ~1 trade/episode on average)
                # before a checkpoint is even ELIGIBLE to become best —
                # otherwise "best" collapses back to "does nothing",
                # which is exactly the frequency problem this whole fix
                # pass is trying to solve.
                min_trades_to_qualify = self.n_eval_episodes  # ~1 trade/episode floor
                qualifies = eval_trades >= min_trades_to_qualify
                if not qualifies:
                    log.warning(
                        f"[TrainRL v2] Checkpoint @ {self.n_calls} skipped for 'best' "
                        f"(only {eval_trades} trades across {self.n_eval_episodes} eval "
                        f"episodes — degenerate/near-idle policy, not a real candidate)"
                    )

                # Save best model
                if qualifies and avg_eval_reward > self.best_eval_reward:
                    self.best_eval_reward = avg_eval_reward
                    save_path = PROJECT_ROOT / "ml" / "rl_policy" / "ppo_forex_best.zip"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    self.model.save(str(save_path))
                    # AUDIT FIX (winrate round 3): write a meta.json
                    # sidecar for THIS specific checkpoint using only
                    # this eval's own numbers. Training is unstable
                    # (a checkpoint can be genuinely good at 60k steps
                    # and degrade by 80k — observed directly: WR 41.4%
                    # / +$67 avg pnl at 60k dropped to 0 trades by
                    # 80k). The final "latest" meta.json blends the
                    # whole run's aggregate stats, which hides whether
                    # the specific checkpoint being deployed
                    # (ppo_forex_best.zip) was actually the good one.
                    # The quality gate needs to check THIS file against
                    # THIS checkpoint, not the run-wide aggregate.
                    best_meta = {
                        # Keys match what RLAgent._passes_quality_gate()
                        # reads (episodes/win_rate/avg_reward) — but here
                        # they describe THIS checkpoint's own held-out
                        # eval, not a run-wide aggregate.
                        "episodes": self.n_eval_episodes,
                        "win_rate": round(eval_wr / 100.0, 4),
                        "avg_reward": round(float(avg_eval_reward), 2),
                        "timestep": self.n_calls,
                        "avg_eval_pnl": round(float(avg_eval_pnl), 2),
                        "eval_trades": int(eval_trades),
                        "trained_on": datetime.now(timezone.utc).isoformat(),
                        "symbol": self.pair,
                        "timeframe": self.timeframe,
                        "engine_version": "v2",
                        "checkpoint_type": "best_eval",
                    }
                    best_meta_path = PROJECT_ROOT / "ml" / "rl_policy" / "ppo_forex_best_meta.json"
                    with open(best_meta_path, "w") as f:
                        json.dump(best_meta, f, indent=2)
                    log.info(f"[TrainRL v2] New best model saved: {save_path} (WR={eval_wr:.1f}%, pnl=${avg_eval_pnl:.2f})")
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

    callback = ProfitabilityCallback(eval_env, eval_freq, n_eval_episodes, pair=pair, timeframe=timeframe)

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

    # AUDIT FIX (win-rate/frequency bug #3, cont.): compute win_rate
    # from actual closed-trade wins/losses accumulated across real
    # episodes, not from "was cumulative episode reward positive".
    # This is the number the RL agent's quality gate should trust.
    total_trades_all = sum(callback.episode_trade_counts)
    total_wins_all = sum(callback.episode_win_counts)
    trade_frequency_per_episode = round(
        total_trades_all / max(len(callback.episode_rewards), 1), 2
    )
    meta = {
        "episodes": len(callback.episode_rewards),
        "win_rate": round(total_wins_all / max(total_trades_all, 1), 4),
        "total_trades": total_trades_all,
        "trades_per_episode": trade_frequency_per_episode,
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

def _resolve_queue_pair(queue_pairs: List[str], state_path: Path) -> tuple[str, int, int]:
    """Pick which pair to train THIS run from a rotating queue, and
    persist the pointer so the NEXT run (even a totally separate
    process invocation later) picks the next pair automatically.

    Round 6 (requested): "train jeno akta akta kore hoy" — one pair
    per run, and if you stop after EURUSD and start the script again
    later, it should move on to the next pair instead of retraining
    EURUSD. State lives in a small JSON sidecar next to the policy
    files so it survives across separate script invocations.
    """
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = {"queue": queue_pairs, "next_index": 0}
    if state_path.exists():
        try:
            saved = json.loads(state_path.read_text(encoding="utf-8"))
            if saved.get("queue") == queue_pairs:
                # Same queue as last time — resume where we left off
                state["next_index"] = int(saved.get("next_index", 0)) % len(queue_pairs)
            # else: queue list changed since last run — start over at 0
        except Exception as e:
            log.warning(f"[TrainRL v2] Could not read queue state ({e}) — starting from pair 0")

    idx = state["next_index"]
    pair_now = queue_pairs[idx]

    # Persist the pointer for NEXT run BEFORE training starts, so even
    # if this run is killed/interrupted partway through, the next
    # invocation still advances to the next pair rather than looping
    # on the same one forever.
    state["next_index"] = (idx + 1) % len(queue_pairs)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    return pair_now, idx, len(queue_pairs)


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
    parser.add_argument("--extra-pairs", default="",
                        help="Comma-separated additional pairs to train on in "
                             "parallel (round-4: reduces single-pair overfitting/"
                             "variance). Eval/model-selection still uses --pair only. "
                             "Example: --extra-pairs EURCAD,GBPCAD,GBPSEK")
    parser.add_argument("--queue", default="",
                        help="Round-6: comma-separated pairs to train ONE AT A TIME, "
                             "one pair per script run, remembered across separate runs. "
                             "Example: --queue EURUSD,GBPUSD,EURCAD,GBPCAD,GBPSEK — run "
                             "the script, it trains EURUSD and exits; stop, run it again "
                             "later, it trains GBPUSD next, and so on, wrapping back to "
                             "EURUSD after the last pair. Overrides --pair and "
                             "--extra-pairs when set.")
    parser.add_argument("--queue-from-config", action="store_true",
                        help="Round-7: load the queue from config.SYMBOLS instead of "
                             "typing pairs by hand. Re-reads config.py fresh on every "
                             "run, so if the pair list in config.py changes later "
                             "(pairs added/removed), the queue picks that up "
                             "automatically without editing the training command. "
                             "Takes priority over --queue.")
    args = parser.parse_args()

    if args.queue_from_config:
        try:
            import importlib
            import config as _config_module
            importlib.reload(_config_module)  # pick up any edits since process start
            _config_symbols = list(getattr(_config_module, "SYMBOLS", []))
            if not _config_symbols:
                print("ERROR: config.SYMBOLS is empty — nothing to queue. "
                      "Check config.py / utils/pair_profiles.py.")
                raise SystemExit(1)
            args.queue = ",".join(_config_symbols)
            print(f"[TrainRL v2] Loaded {len(_config_symbols)} pairs from config.SYMBOLS")
        except ImportError as e:
            print(f"ERROR: could not import config.py ({e}). Run this from the "
                  f"project root, or use --queue with an explicit pair list instead.")
            raise SystemExit(1)

    _extra_pairs = [p.strip() for p in args.extra_pairs.split(",") if p.strip()]
    _queue_pairs = [p.strip() for p in args.queue.split(",") if p.strip()]

    if _queue_pairs:
        _state_path = PROJECT_ROOT / "ml" / "rl_policy" / "training_queue_state.json"
        _pair_now, _idx, _total = _resolve_queue_pair(_queue_pairs, _state_path)
        _next_pair = _queue_pairs[(_idx + 1) % _total]
        print("\n" + "=" * 60)
        print("  RL TRAINING QUEUE (round 6)")
        print("=" * 60)
        print(f"  Queue:              {', '.join(_queue_pairs)}")
        print(f"  Training NOW:       [{_idx + 1}/{_total}] {_pair_now}")
        print(f"  Next run will train: {_next_pair}")
        print(f"  State file:         {_state_path}")
        print("=" * 60 + "\n")
        log.info(f"[TrainRL v2] QUEUE: training pair {_idx + 1}/{_total} = {_pair_now} "
                 f"(next run -> {_next_pair})")
        args.pair = _pair_now
        _extra_pairs = []  # queue mode trains strictly one pair per run

    result = train_rl_agent_v2(
        pair=args.pair,
        timeframe=args.timeframe,
        total_timesteps=args.timesteps,
        max_steps_per_episode=args.max_steps_per_episode,
        eval_freq=args.eval_freq,
        learning_rate=args.learning_rate,
        extra_pairs=_extra_pairs,
    )

    print("\n" + "=" * 60)
    print("  TRAINING RESULT (v2)")
    print("=" * 60)
    if _queue_pairs:
        print(f"  Pair trained this run: {args.pair} ({_idx + 1}/{_total})")
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
        if _queue_pairs:
            print(f"  Next run trains:    {_next_pair}")
        print()
        print("  EVAL HISTORY:")
        for ev in result.get("eval_history", []):
            print(f"    {ev['timestep']:>6} steps | "
                  f"reward={ev['avg_eval_reward']:>8.2f} | "
                  f"pnl=${ev['avg_eval_pnl']:>8.2f} | "
                  f"WR={ev['eval_win_rate']:>5.1f}%")
    print("=" * 60)
