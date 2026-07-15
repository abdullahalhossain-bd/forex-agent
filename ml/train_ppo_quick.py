"""
ml/train_ppo_quick.py — Quick PPO bootstrap with synthetic data (Day 102)
==========================================================================
Generates synthetic OHLCV data and trains a PPO policy so the system
has a ppo_forex_latest.zip to load.  NOT production-grade — just lets
the full RL code path execute instead of falling back to heuristic.

Usage:
    python -m ml.train_ppo_quick
    python -m ml.train_ppo_quick --timesteps 100000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PROJECT_ROOT as _PROOT, DEFAULT_TIMEFRAME
from utils.logger import get_logger

log = get_logger("train_ppo_quick")

# Where to save the model
POLICY_PATH = _PROOT / "ml" / "rl_policy" / "ppo_forex_latest.zip"


def generate_synthetic_ohlcv(n_rows: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic forex OHLCV data.

    Uses a random walk with mean-reversion and volatility clustering
    to produce data that resembles real 15m EURUSD candles.
    """
    np.random.seed(seed)

    # Price simulation with mean-reversion + volatility clustering
    price = 1.1000  # Starting EURUSD price
    prices = [price]
    volatility = 0.0003  # Base 15m volatility

    for i in range(n_rows - 1):
        # GARCH-like volatility clustering
        volatility = 0.7 * volatility + 0.3 * abs(np.random.randn()) * 0.0005
        volatility = max(0.00005, min(volatility, 0.003))

        # Mean-reverting random walk
        drift = (1.1000 - price) * 0.001  # Gentle pull toward 1.10
        change = drift + volatility * np.random.randn()
        price = max(price + change, 0.5)
        prices.append(price)

    closes = np.array(prices)

    # Generate OHLC from closes
    noise = np.random.uniform(0.2, 1.0, n_rows)
    highs = closes + noise * abs(np.random.randn(n_rows)) * 0.0005
    lows = closes - noise * abs(np.random.randn(n_rows)) * 0.0005
    opens = closes + np.random.randn(n_rows) * 0.0001

    # Ensure OHLC consistency
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))

    # Volume with intraday pattern
    hour_of_day = np.random.randint(0, 24, n_rows)
    base_volume = 1000
    volume = base_volume * (1 + 0.5 * np.sin(hour_of_day / 24 * 2 * np.pi))
    volume *= (1 + 0.3 * np.random.randn(n_rows))
    volume = np.abs(volume).astype(int)

    # Build datetime index (15m candles starting from 2024-01-01)
    dates = pd.date_range("2024-01-01", periods=n_rows, freq="15min")

    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volume,
    }, index=dates)

    # Add basic indicators for the environment
    df["rsi"] = _compute_rsi(closes, 14)
    df["sma_20"] = pd.Series(closes).rolling(20).mean().values
    df["sma_50"] = pd.Series(closes).rolling(50).mean().values
    df["atr"] = _compute_atr(highs, lows, closes, 14)
    df["adx"] = _compute_adx(highs, lows, closes, 14)

    return df.dropna().reset_index(drop=True)


def _compute_rsi(close, period=14):
    """Simple RSI calculation."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.convolve(gain, np.ones(period)/period, mode='same')
    avg_loss = np.convolve(loss, np.ones(period)/period, mode='same')
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _compute_atr(high, low, close, period=14):
    """Simple ATR calculation."""
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1)),
        ),
    )
    atr = pd.Series(tr).rolling(period).mean().values
    return atr


def _compute_adx(high, low, close, period=14):
    """Simple ADX approximation."""
    plus_dm = np.where(
        (high - np.roll(high, 1)) > (np.roll(low, 1) - low),
        np.maximum(high - np.roll(high, 1), 0),
        0.0,
    )
    minus_dm = np.where(
        (np.roll(low, 1) - low) > (high - np.roll(high, 1)),
        np.maximum(np.roll(low, 1) - low, 0),
        0.0,
    )
    atr = _compute_atr(high, low, close, period)
    atr = np.where(atr > 0, atr, 1e-10)
    plus_di = 100 * pd.Series(plus_dm).rolling(period).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(period).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / np.where(
        (plus_di + minus_di) > 0, plus_di + minus_di, 1e-10
    )
    adx = pd.Series(dx).rolling(period).mean().values
    return adx


def train_quick(timesteps: int = 50000) -> dict:
    """Train PPO on synthetic data and save model."""
    # Check SB3 availability
    try:
        from stable_baselines3 import PPO
        from gymnasium import spaces
    except ImportError as e:
        return {"error": f"stable-baselines3 or gymnasium not installed: {e}"}

    log.info(f"[PPO Quick] Generating synthetic data...")
    df = generate_synthetic_ohlcv(n_rows=3000)
    log.info(f"[PPO Quick] Data ready: {len(df)} rows, {len(df.columns)} columns")

    # Import project environment — now natively Gymnasium-compatible (Day 102)
    from ml.rl_environment import ForexTradingEnv

    env = ForexTradingEnv(
        df=df,
        initial_balance=10000.0,
        pair="EURUSD",
        pip_size=0.0001,
        spread_pips=1.5,
    )

    # Quick sanity check (optional — catches API mismatches early)
    try:
        from gymnasium.utils.env_checker import check_env
        check_env(env, skip_render_check=True)
        log.info("[PPO Quick] Gymnasium env check passed")
    except Exception as e:
        log.warning(f"[PPO Quick] env check warning (non-fatal): {e}")

    log.info(f"[PPO Quick] Training PPO for {timesteps} timesteps...")
    log.info(f"[PPO Quick] Observation space: {env.observation_space.shape if env.observation_space else 'None'}")
    log.info(f"[PPO Quick] Action space: {env.action_space}")

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
    )

    model.learn(total_timesteps=timesteps)

    # Save
    POLICY_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(POLICY_PATH))
    log.info(f"[PPO Quick] Model saved to {POLICY_PATH}")

    return {
        "status": "success",
        "timesteps": timesteps,
        "model_path": str(POLICY_PATH),
        "data_rows": len(df),
        "observation_dim": env.observation_space.shape if env.observation_space else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick PPO bootstrap with synthetic data")
    parser.add_argument("--timesteps", type=int, default=50000, help="Training timesteps")
    args = parser.parse_args()

    result = train_quick(timesteps=args.timesteps)
    if "error" in result:
        print(f"[ERROR] {result['error']}")
        sys.exit(1)
    else:
        print(f"\n[PPO Quick] Training complete!")
        for k, v in result.items():
            print(f"  {k}: {v}")