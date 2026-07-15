"""
ml/train_rl.py — RL Training Script with Curriculum Learning (Day 71)
======================================================================

Trains the PPO agent on historical data with a curriculum:
  Stage 1: Simple trend market (easy patterns)
  Stage 2: Range market (harder — more patience needed)
  Stage 3: High volatility (risk management focus)
  Stage 4: Full mixed market (real-world simulation)

Each stage trains for a portion of total timesteps, then the model
moves to the next stage. This produces a more robust policy than
training on mixed data from the start.

Usage:
    from ml.train_rl import train_rl_agent
    result = train_rl_agent(pair="EURUSD", timeframe="15m", total_timesteps=500000)

Or from command line:
    python -m ml.train_rl --pair EURUSD --timesteps 500000
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("train_rl")


@dataclass
class TrainingStage:
    """One stage of curriculum learning."""
    name: str
    description: str
    timesteps: int
    filter_fn: Optional[str] = None  # column filter function name


CURRICULUM = [
    TrainingStage(
        name="trend_market",
        description="Stage 1: Simple trend market — easy patterns",
        timesteps=0.25,  # 25% of total
    ),
    TrainingStage(
        name="range_market",
        description="Stage 2: Range market — patience training",
        timesteps=0.25,
    ),
    TrainingStage(
        name="volatile_market",
        description="Stage 3: High volatility — risk management",
        timesteps=0.20,
    ),
    TrainingStage(
        name="full_mixed",
        description="Stage 4: Full mixed market — real-world simulation",
        timesteps=0.30,
    ),
]


def load_historical_data(pair: str, timeframe: str = "15m", periods: int = 5000) -> pd.DataFrame:
    """Load historical OHLCV data for training."""
    try:
        from data.fetcher import DataFetcher
        fetcher = DataFetcher()
        df = fetcher.fetch_ohlcv(pair, timeframe, periods=periods)
        if df is not None and len(df) > 0:
            # Add indicators
            try:
                from data.indicators import Indicators
                df = Indicators().add_all(df)
            except Exception:
                pass
            log.info(f"[TrainRL] loaded {len(df)} rows for {pair} {timeframe}")
            return df
    except Exception as e:
        log.error(f"[TrainRL] data load failed: {e}")
    return pd.DataFrame()


def build_features_df(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """Build feature vectors for each row using FeatureEngineer."""
    try:
        from ml.feature_engineer import get_feature_engineer
        engineer = get_feature_engineer()
        features_list = []
        for i in range(len(df)):
            sub_df = df.iloc[:i+1]
            if len(sub_df) < 5:
                continue
            feats = engineer.build_feature_vector(
                df=sub_df, analysis_out={}, pair=pair, timeframe="15m",
            )
            features_list.append(feats)
        if features_list:
            return pd.DataFrame(features_list)
    except Exception as e:
        log.warning(f"[TrainRL] feature build failed: {e}")
    return pd.DataFrame()


def split_by_regime(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Split dataframe by market regime for curriculum learning."""
    splits = {}
    if "regime" not in df.columns and len(df) == 0:
        return {stage.name: df for stage in CURRICULUM}

    try:
        # Use ADX or trend column if available
        if "adx" in df.columns:
            splits["trend_market"] = df[df["adx"] > 25].copy()
            splits["range_market"] = df[df["adx"] <= 20].copy()
            splits["volatile_market"] = df[df.get("atr", pd.Series(0)) > df["atr"].median() * 1.5].copy() if "atr" in df.columns else df.copy()
            splits["full_mixed"] = df.copy()
        else:
            # No regime info — use all data for all stages
            for stage in CURRICULUM:
                splits[stage.name] = df.copy()
    except Exception:
        for stage in CURRICULUM:
            splits[stage.name] = df.copy()

    return splits


def train_rl_agent(
    pair: str = "EURUSD",
    timeframe: str = "15m",
    total_timesteps: int = 500000,
    use_curriculum: bool = True,
    initial_balance: float = 10000.0,
) -> Dict[str, Any]:
    """Train the RL agent with optional curriculum learning.

    Args:
        pair: Trading pair.
        timeframe: Timeframe.
        total_timesteps: Total training timesteps.
        use_curriculum: Whether to use curriculum learning.
        initial_balance: Starting balance for simulation.

    Returns:
        Dict with training results.
    """
    from ml.rl_agent import get_rl_agent
    from ml.rl_environment import ForexTradingEnv
    from ml.rl_policy_store import get_rl_policy_store

    log.info(f"[TrainRL] Starting RL training: {pair} {timeframe} | {total_timesteps} timesteps | curriculum={use_curriculum}")

    # 1. Load data
    df = load_historical_data(pair, timeframe, periods=5000)
    if df.empty or len(df) < 200:
        return {"error": f"insufficient data for {pair} ({len(df)} rows)"}

    # 2. Build features
    features_df = build_features_df(df, pair)
    if features_df.empty:
        features_df = None
        log.warning("[TrainRL] no features — using raw OHLCV")

    # 3. Split by regime for curriculum
    if use_curriculum:
        splits = split_by_regime(df)
    else:
        splits = {"full_mixed": df}

    # 4. Train each stage
    agent = get_rl_agent()
    results = {"stages": [], "total_timesteps": 0}
    remaining_timesteps = total_timesteps

    for stage in CURRICULUM:
        if not use_curriculum and stage.name != "full_mixed":
            continue
        stage_df = splits.get(stage.name, df)
        if stage_df.empty or len(stage_df) < 100:
            log.info(f"[TrainRL] skipping {stage.name} — not enough data ({len(stage_df)} rows)")
            continue

        stage_timesteps = int(total_timesteps * stage.timesteps)
        if stage.name == "full_mixed" and not use_curriculum:
            stage_timesteps = total_timesteps

        log.info(f"[TrainRL] === Stage: {stage.name} === ({stage_timesteps} timesteps)")
        log.info(f"[TrainRL] {stage.description}")

        # Build environment for this stage
        stage_features = features_df.iloc[:len(stage_df)].copy() if features_df is not None else None
        env = ForexTradingEnv(
            df=stage_df,
            features_df=stage_features,
            initial_balance=initial_balance,
            pair=pair,
        )

        # Train
        stage_result = agent.train(env, total_timesteps=stage_timesteps)
        results["stages"].append({
            "stage": stage.name,
            "description": stage.description,
            "timesteps": stage_timesteps,
            "rows": len(stage_df),
            "result": stage_result,
        })
        results["total_timesteps"] += stage_timesteps

        if "error" in stage_result:
            log.error(f"[TrainRL] stage {stage.name} failed: {stage_result['error']}")
            break

    # 5. Save final policy with versioning
    try:
        policy_store = get_rl_policy_store()
        latest_model = Path("ml/rl_policy/ppo_forex_latest.zip")
        if latest_model.exists():
            version = policy_store.save_policy(
                model_path=latest_model,
                episodes=agent._training_episodes,
                avg_reward=agent._avg_reward,
                win_rate=0.0,  # would need backtesting to compute
                notes=f"Curriculum trained on {pair} {timeframe}",
            )
            results["policy_version"] = version
    except Exception as e:
        log.warning(f"[TrainRL] policy save failed: {e}")

    results["agent_status"] = agent.status()
    log.info(f"[TrainRL] Training complete: {results['total_timesteps']} total timesteps")
    return results


# ── CLI entry point ─────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL agent for forex trading")
    parser.add_argument("--pair", default="EURUSD", help="Trading pair")
    parser.add_argument("--timeframe", default="15m", help="Timeframe")
    parser.add_argument("--timesteps", type=int, default=500000, help="Total training timesteps")
    parser.add_argument("--no-curriculum", action="store_true", help="Disable curriculum learning")
    args = parser.parse_args()

    result = train_rl_agent(
        pair=args.pair,
        timeframe=args.timeframe,
        total_timesteps=args.timesteps,
        use_curriculum=not args.no_curriculum,
    )
    print("\n=== Training Result ===")
    for stage in result.get("stages", []):
        print(f"  {stage['stage']}: {stage['timesteps']} steps — {stage['result'].get('status', 'unknown')}")
    print(f"\n  Total timesteps: {result.get('total_timesteps', 0)}")
    print(f"  Policy version: {result.get('policy_version', 'N/A')}")
    print(f"  Agent status: {result.get('agent_status', {})}")
