"""
ml/train_ppo_quick.py — Quick PPO bootstrap with REAL MetaTrader5 data (Day 102+)
==================================================================================
Trains a PPO policy using real MT5 historical data so the system has a
ppo_forex_latest.zip to load. Production-grade by default, with optional
--debug-synthetic fallback for debugging only.

Usage:
    python -m ml.train_ppo_quick
    python -m ml.train_ppo_quick --symbol EURUSD --timeframe M15 --bars 100000
    python -m ml.train_ppo_quick --timesteps 100000
    python -m ml.train_ppo_quick --debug-synthetic  # DEBUG ONLY
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone

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

    ⚠️ DEBUG ONLY: This function should ONLY be used for debugging/testing.
    Production training MUST use real MT5 data.

    Uses a random walk with mean-reversion and volatility clustering
    to produce data that resembles real 15m EURUSD candles.
    """
    log.warning("⚠️ GENERATING SYNTHETIC DATA - DEBUG MODE ONLY ⚠️")
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

    return df.dropna().reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators as features (same pipeline as supervised ML models).

    This reuses the same feature engineering pipeline used by scripts/train_models_quick.py
    to ensure consistency between RL and supervised learning approaches.

    Features added:
      - Returns: ret_1, ret_3, ret_5, ret_10
      - Volatility: vol_5, vol_10
      - RSI: rsi_14
      - SMAs: sma_10, sma_20, sma_50
      - ATR: atr_14
      - MACD: macd, macd_signal

    CRITICAL: No look-ahead bias. All features use only past/current data.
    """
    # Returns
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_10"] = df["close"].pct_change(10)

    # Volatility
    df["vol_5"] = df["ret_1"].rolling(5).std()
    df["vol_10"] = df["ret_1"].rolling(10).std()

    # RSI (simplified)
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # SMAs
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()

    # ATR
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()

    # MACD
    ema_12 = df["close"].ewm(span=12).mean()
    ema_26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()

    return df


def train_quick(
    symbol: str = "EURUSD",
    timeframe: str = "M15",
    bars: int = 100000,
    timesteps: int = 50000,
    use_synthetic: bool = False,
) -> dict:
    """Train PPO on real MT5 data (or synthetic for debug) and save model.

    Args:
        symbol: Trading symbol (default: EURUSD)
        timeframe: Timeframe string (default: M15)
        bars: Number of bars to fetch (default: 100000)
        timesteps: Training timesteps (default: 50000)
        use_synthetic: If True, use synthetic data (DEBUG ONLY)

    Returns:
        Dict with training results and metadata
    """
    # Check SB3 availability
    try:
        from stable_baselines3 import PPO
        from gymnasium import spaces
    except ImportError as e:
        return {"error": f"stable-baselines3 or gymnasium not installed: {e}"}

    # Import project environment
    from ml.rl_environment import ForexTradingEnv

    # ── 1. Fetch data ────────────────────────────────────────────────
    if use_synthetic:
        log.warning("⚠️ Using SYNTHETIC data (DEBUG MODE)")
        df = generate_synthetic_ohlcv(n_rows=min(bars, 3000))
        log.info(f"  Generated {len(df)} bars of synthetic data")
        start_date = df.index[0] if len(df) > 0 else None
        end_date = df.index[-1] if len(df) > 0 else None
        rows_downloaded = len(df)
    else:
        # Use the MT5 data loader (same as scripts/train_models_quick.py)
        from ml.mt5_data_loader import MT5DataLoader

        log.info("Fetching REAL MetaTrader5 historical data...")
        log.info(f"  Symbol: {symbol} | Timeframe: {timeframe} | Bars: {bars}")

        loader = MT5DataLoader()

        result = loader.fetch(symbol=symbol, timeframe=timeframe, bars=bars)
        loader.shutdown()

        if result.dataframe is None:
            log.error(f"Failed to fetch MT5 data for {symbol} {timeframe}")
            if result.errors:
                log.error(f"Errors: {result.errors}")
            return {"error": f"Failed to fetch MT5 data: {result.errors}"}

        df = result.dataframe
        rows_downloaded = result.rows_downloaded
        start_date = result.start_date
        end_date = result.end_date

        log.info(f"  ✅ Downloaded {rows_downloaded} candles from MT5")
        log.info(f"  ✅ After cleaning: {result.rows_after_cleaning} rows")
        log.info(f"  ✅ Date range: {start_date} → {end_date}")

    # ── 2. Add features (same pipeline as supervised ML models) ─────
    log.info("Adding technical indicators (feature engineering)...")
    df = add_features(df)
    df = df.dropna()
    log.info(f"  ✅ After feature computation: {len(df)} usable rows")

    if len(df) < 50:
        log.error(f"  Not enough data ({len(df)} rows) — need at least 50")
        return {"error": f"Insufficient data: {len(df)} rows"}

    # ── 3. Build RL environment ─────────────────────────────────────
    log.info("Building PPO environment from OHLCV + engineered features...")

    env = ForexTradingEnv(
        df=df,
        initial_balance=10000.0,
        pair=symbol,
        pip_size=0.0001,
        spread_pips=1.5,
    )

    # Quick sanity check (optional — catches API mismatches early)
    try:
        from gymnasium.utils.env_checker import check_env
        check_env(env, skip_render_check=True)
        log.info("  ✅ Gymnasium env check passed")
    except Exception as e:
        log.warning(f"  ⚠️ env check warning (non-fatal): {e}")

    # Log observation dimensions
    obs_shape = env.observation_space.shape if env.observation_space else None
    log.info(f"  Observation space shape: {obs_shape}")
    log.info(f"  Action space: {env.action_space}")

    # ── 4. Train PPO ────────────────────────────────────────────────
    log.info(f"Training PPO for {timesteps} timesteps...")

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

    # Track episode statistics during training
    episode_rewards = []
    episode_lengths = []

    class EpisodeStatsCallback:
        def __init__(self):
            self.episode_rewards = []
            self.episode_lengths = []

        def __call__(self, locals_, globals_):
            # Extract info from the training loop
            if "infos" in locals_:
                for info in locals_["infos"]:
                    if "episode" in info:
                        self.episode_rewards.append(info["episode"]["r"])
                        self.episode_lengths.append(info["episode"]["l"])
            return True

    callback = EpisodeStatsCallback()

    model.learn(total_timesteps=timesteps, callback=callback)

    # Collect episode statistics
    if callback.episode_rewards:
        avg_reward = np.mean(callback.episode_rewards)
        std_reward = np.std(callback.episode_rewards)
        avg_length = np.mean(callback.episode_lengths)
        log.info(f"\n=== Episode Statistics ===")
        log.info(f"  Episodes completed: {len(callback.episode_rewards)}")
        log.info(f"  Average reward: {avg_reward:.4f} ± {std_reward:.4f}")
        log.info(f"  Average episode length: {avg_length:.1f} steps")

    # ── 5. Save model ───────────────────────────────────────────────
    POLICY_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(POLICY_PATH))
    log.info(f"\n✅ Model saved to {POLICY_PATH}")

    return {
        "status": "success",
        "symbol": symbol,
        "timeframe": timeframe,
        "bars_requested": bars,
        "bars_used": len(df),
        "rows_downloaded": rows_downloaded,
        "date_range": f"{start_date} → {end_date}",
        "observation_dim": obs_shape,
        "timesteps": timesteps,
        "model_path": str(POLICY_PATH),
        "episodes": len(callback.episode_rewards) if callback.episode_rewards else 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quick PPO bootstrap with REAL MT5 historical data"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="EURUSD",
        help="Trading symbol (default: EURUSD)"
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="M15",
        help="Timeframe: M1, M5, M15, M30, H1, H4, D1 (default: M15)"
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=100000,
        help="Number of bars to fetch from MT5 (default: 100000)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50000,
        help="Training timesteps (default: 50000)"
    )
    parser.add_argument(
        "--debug-synthetic",
        action="store_true",
        help="Use synthetic data instead of MT5 (DEBUG ONLY - not for production)"
    )
    args = parser.parse_args()

    if args.debug_synthetic:
        log.warning("⚠️ DEBUG MODE: Using SYNTHETIC data ⚠️")
        log.warning("⚠️ Production models MUST use real MT5 data ⚠️")
    else:
        log.info("Using REAL MetaTrader5 historical data for training")

    log.info(f"Symbol: {args.symbol} | Timeframe: {args.timeframe} | Bars: {args.bars}")
    log.info(f"Training timesteps: {args.timesteps}")

    result = train_quick(
        symbol=args.symbol,
        timeframe=args.timeframe,
        bars=args.bars,
        timesteps=args.timesteps,
        use_synthetic=args.debug_synthetic,
    )

    if "error" in result:
        print(f"\n[ERROR] {result['error']}")
        sys.exit(1)
    else:
        print(f"\n{'='*60}")
        print("[PPO Quick] Training complete!")
        print(f"{'='*60}")
        for k, v in result.items():
            if k != "status":
                print(f"  {k}: {v}")
        print(f"{'='*60}")