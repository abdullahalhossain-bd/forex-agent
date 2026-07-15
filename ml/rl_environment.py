"""
ml/rl_environment.py — Forex Trading RL Environment (Day 71 → Day 102)
====================================================================

Gymnasium-compatible trading environment for reinforcement learning.

The agent observes market state (features from Day 68 FeatureEngineer),
chooses an action (HOLD/BUY/SELL/CLOSE), and receives a reward from
the RewardEngine.

State space:  ~16 features (price, indicators, position status, account status)
Action space: 4 discrete actions
              0 = HOLD  (do nothing — wait for better setup)
              1 = BUY   (open long position)
              2 = SELL  (open short position)
              3 = CLOSE (close current position)

The environment simulates trading on historical data, tracking:
  - Account balance + equity
  - Open position (entry, SL, TP, direction)
  - Trades per day (overtrading prevention)
  - Peak balance (drawdown tracking)

Day 102: Migrated from legacy Gym API to Gymnasium API.
  - reset() now returns (obs, info) tuple
  - step() now returns (obs, reward, terminated, truncated, info) 5-tuple
  - Inherits from gymnasium.Env for SB3 2.9+ compatibility
  - No adapter wrapper needed in training scripts
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, SupportsFloat

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from utils.logger import get_logger
from ml.reward_engine import RewardEngine, get_reward_engine

log = get_logger("rl_environment")


# Action constants
ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL = 2
ACTION_CLOSE = 3
ACTIONS = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "CLOSE"}


@dataclass
class Position:
    """Open position state."""
    direction: str = "NONE"    # LONG / SHORT / NONE
    entry: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    lot: float = 0.0
    opened_at_step: int = 0


class ForexTradingEnv(gym.Env):
    """Gymnasium-compatible forex trading environment.

    Works with stable-baselines3 2.9+ (PPO, A2C, etc.) natively.
    No adapter wrapper needed.

    Day 102: Migrated to Gymnasium API.
      - Inherits from gymnasium.Env (isinstance check passes in SB3)
      - reset() returns (obs, info) — Gymnasium standard
      - step() returns (obs, reward, terminated, truncated, info) — 5-tuple
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        features_df: Optional[pd.DataFrame] = None,
        initial_balance: float = 10000.0,
        risk_per_trade: float = 0.01,
        pip_size: float = 0.0001,
        spread_pips: float = 1.5,
        pair: str = "EURUSD",
        reward_engine: Optional[RewardEngine] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        # ── Gymnasium render mode ─────────────────────────────────
        self.render_mode = render_mode

        # ── Trading parameters ─────────────────────────────────────
        self.df = df.reset_index(drop=True)
        self.features_df = features_df.reset_index(drop=True) if features_df is not None else None
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.pip_size = pip_size
        self.spread_pips = spread_pips
        self.pair = pair
        self.reward_engine = reward_engine or get_reward_engine()

        # ── State space ─────────────────────────────────────────────
        if self.features_df is not None:
            self.n_features = len(self.features_df.columns) + 6  # +6 for position/account state
        else:
            self.n_features = min(20, len(df.columns)) + 6

        # ── Gymnasium spaces (set once, never change) ──────────────
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)
        self.action_space_size = 4

        # ── Episode state (reset on reset()) ───────────────────────
        self.current_step = 0
        self.balance = initial_balance
        self.peak_balance = initial_balance
        self.position = Position()
        self.trades_today = 0
        self.current_day = None
        self.total_trades = 0
        self.total_wins = 0
        self.total_losses = 0
        self.episode_reward = 0.0
        self.episode_pnl = 0.0

    # ── Gymnasium API ───────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment for a new episode.

        Returns:
            (observation, info) tuple — Gymnasium standard.
        """
        super().reset(seed=seed, options=options)

        self.current_step = 0
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.position = Position()
        self.trades_today = 0
        self.current_day = None
        self.total_trades = 0
        self.total_wins = 0
        self.total_losses = 0
        self.episode_reward = 0.0
        self.episode_pnl = 0.0

        return self._get_state(), self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step.

        Returns:
            (observation, reward, terminated, truncated, info) — Gymnasium 5-tuple.
        """
        if self.current_step >= len(self.df) - 1:
            # Close any open position at end of data
            if self.position.direction != "NONE":
                self._close_position(reason="end_of_data")
            return self._get_state(), 0.0, True, False, self._get_info()

        action_name = ACTIONS.get(action, "HOLD")
        pnl_this_step = 0.0
        reward = 0.0

        # Track day change for overtrading counter
        try:
            current_dt = self.df.iloc[self.current_step].name if hasattr(self.df.iloc[self.current_step].name, 'date') else None
            day = current_dt.date() if current_dt else self.current_step // 96  # fallback: 96 M15 candles per day
        except Exception:
            day = self.current_step // 96
        if day != self.current_day:
            self.current_day = day
            self.trades_today = 0

        # ── Execute action ─────────────────────────────────────────
        close_price = float(self.df.iloc[self.current_step].get("close", 0))
        high_price = float(self.df.iloc[self.current_step].get("high", close_price))
        low_price = float(self.df.iloc[self.current_step].get("low", close_price))

        if action_name == "BUY" and self.position.direction == "NONE":
            self._open_position("LONG", close_price, high_price, low_price)
            self.trades_today += 1
            self.total_trades += 1

        elif action_name == "SELL" and self.position.direction == "NONE":
            self._open_position("SHORT", close_price, high_price, low_price)
            self.trades_today += 1
            self.total_trades += 1

        elif action_name == "CLOSE" and self.position.direction != "NONE":
            pnl = self._close_position(reason="manual_close")
            pnl_this_step = pnl

        elif action_name == "HOLD":
            # Check if open position hit SL/TP
            if self.position.direction != "NONE":
                pnl = self._check_sl_tp(high_price, low_price)
                if pnl != 0:
                    pnl_this_step = pnl

        # ── Mark-to-market for open positions ─────────────────────
        if self.position.direction != "NONE" and pnl_this_step == 0:
            # Unrealized PnL (not realized, but track for equity)
            pass

        # ── Calculate reward ──────────────────────────────────────
        rr_ratio = 0.0
        if self.position.direction != "NONE" and self.position.sl > 0:
            risk = abs(self.position.entry - self.position.sl)
            reward_dist = abs(self.position.tp - self.position.entry) if self.position.tp > 0 else 0
            rr_ratio = reward_dist / risk if risk > 0 else 0

        drawdown = (self.peak_balance - self.balance) / self.peak_balance if self.peak_balance > 0 else 0

        reward_rb = self.reward_engine.calculate(
            action=action_name,
            pnl_usd=pnl_this_step,
            balance=self.balance,
            initial_balance=self.initial_balance,
            risk_pct=self.risk_per_trade,
            rr_ratio=rr_ratio,
            trades_today=self.trades_today,
            peak_balance=self.peak_balance,
            position_open=(self.position.direction != "NONE"),
        )
        reward = reward_rb.total
        self.episode_reward += reward
        self.episode_pnl += pnl_this_step

        # ── Advance ────────────────────────────────────────────────
        self.current_step += 1

        # Update peak balance
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance

        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        return self._get_state(), float(reward), terminated, truncated, self._get_info()

    # ── Position management ────────────────────────────────────────

    def _open_position(self, direction: str, entry: float, high: float, low: float) -> None:
        """Open a new position with SL/TP based on ATR."""
        try:
            atr = float(self.df.iloc[self.current_step].get("atr", 0.001))
        except Exception:
            atr = 0.001

        sl_distance = atr * 1.5
        tp_distance = atr * 3.0  # 1:2 R:R

        # Apply spread cost
        spread_cost = self.spread_pips * self.pip_size
        if direction == "LONG":
            entry_price = entry + spread_cost / 2  # buy at ask
            sl = entry_price - sl_distance
            tp = entry_price + tp_distance
        else:
            entry_price = entry - spread_cost / 2  # sell at bid
            sl = entry_price + sl_distance
            tp = entry_price - tp_distance

        # Lot size from risk %
        risk_usd = self.balance * self.risk_per_trade
        lot = risk_usd / (sl_distance / self.pip_size * 10) if sl_distance > 0 else 0.01  # simplified pip value
        lot = max(0.01, min(round(lot, 2), 10.0))

        self.position = Position(
            direction=direction,
            entry=entry_price,
            sl=sl,
            tp=tp,
            lot=lot,
            opened_at_step=self.current_step,
        )
        log.debug(f"[RL Env] OPEN {direction} @ {entry_price:.5f} SL={sl:.5f} TP={tp:.5f} lot={lot}")

    def _close_position(self, reason: str = "manual") -> float:
        """Close the current position. Returns realized PnL."""
        if self.position.direction == "NONE":
            return 0.0

        close_price = float(self.df.iloc[self.current_step].get("close", 0))
        if self.position.direction == "LONG":
            pnl = (close_price - self.position.entry) / self.pip_size * 10 * self.position.lot
        else:
            pnl = (self.position.entry - close_price) / self.pip_size * 10 * self.position.lot

        self.balance += pnl
        if pnl > 0:
            self.total_wins += 1
        else:
            self.total_losses += 1

        log.debug(f"[RL Env] CLOSE {self.position.direction} @ {close_price:.5f} PnL=${pnl:.2f} ({reason})")
        self.position = Position()
        return pnl

    def _check_sl_tp(self, high: float, low: float) -> float:
        """Check if SL or TP was hit. Returns realized PnL if closed, 0 otherwise."""
        if self.position.direction == "NONE":
            return 0.0

        if self.position.direction == "LONG":
            if low <= self.position.sl:
                return self._close_at_price(self.position.sl, "SL hit")
            if high >= self.position.tp:
                return self._close_at_price(self.position.tp, "TP hit")
        else:  # SHORT
            if high >= self.position.sl:
                return self._close_at_price(self.position.sl, "SL hit")
            if low <= self.position.tp:
                return self._close_at_price(self.position.tp, "TP hit")
        return 0.0

    def _close_at_price(self, price: float, reason: str) -> float:
        """Close position at a specific price (SL/TP)."""
        if self.position.direction == "NONE":
            return 0.0
        if self.position.direction == "LONG":
            pnl = (price - self.position.entry) / self.pip_size * 10 * self.position.lot
        else:
            pnl = (self.position.entry - price) / self.pip_size * 10 * self.position.lot

        self.balance += pnl
        if pnl > 0:
            self.total_wins += 1
        else:
            self.total_losses += 1
        log.debug(f"[RL Env] CLOSE {self.position.direction} @ {price:.5f} PnL=${pnl:.2f} ({reason})")
        self.position = Position()
        return pnl

    # ── State representation ───────────────────────────────────────

    def _get_state(self) -> np.ndarray:
        """Get the current state as a normalized feature vector.

        CRITICAL FIX: Uses FEATURE_SCHEMA for consistent feature ordering.
        Previously, features_df.values used DataFrame column order, which
        could change if features are added/removed — causing the model to
        receive garbage input with permuted features.
        """
        if self.current_step >= len(self.df):
            return np.zeros(self.n_features, dtype=np.float32)

        # FIXED: Feature schema — always extract in this exact order
        FEATURE_SCHEMA = [
            "close", "high", "low", "volume",
            "rsi_14", "atr", "macd", "ema_20", "ema_50", "sma_200",
        ]

        # Market features (always in FEATURE_SCHEMA order)
        if self.features_df is not None and self.current_step < len(self.features_df):
            row = self.features_df.iloc[self.current_step]
        else:
            row = self.df.iloc[self.current_step]

        market_features = np.array([
            float(row.get(f, 0) if row.get(f) is not None else 0)
            for f in FEATURE_SCHEMA
        ], dtype=np.float32)

        # Position + account state (6 features)
        position_state = np.array([
            1.0 if self.position.direction == "LONG" else 0.0,
            1.0 if self.position.direction == "SHORT" else 0.0,
            self.position.entry / 10000.0 if self.position.entry > 0 else 0.0,
            self.balance / self.initial_balance,
            self.trades_today / 20.0,
            (self.peak_balance - self.balance) / self.peak_balance if self.peak_balance > 0 else 0.0,
        ], dtype=np.float32)

        state = np.concatenate([market_features, position_state])

        # Ensure consistent size
        if len(state) < self.n_features:
            state = np.pad(state, (0, self.n_features - len(state)))
        elif len(state) > self.n_features:
            state = state[:self.n_features]

        # Replace NaN/inf
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        return state.astype(np.float32)

    def _get_info(self) -> Dict[str, Any]:
        """Get info dict for the current step."""
        return {
            "step": self.current_step,
            "balance": round(self.balance, 2),
            "equity": round(self.balance, 2),  # simplified
            "position": self.position.direction,
            "trades_today": self.trades_today,
            "total_trades": self.total_trades,
            "total_wins": self.total_wins,
            "total_losses": self.total_losses,
            "win_rate": (self.total_wins / self.total_trades * 100) if self.total_trades > 0 else 0.0,
            "episode_reward": round(self.episode_reward, 2),
            "episode_pnl": round(self.episode_pnl, 2),
            "drawdown_pct": round(((self.peak_balance - self.balance) / self.peak_balance * 100) if self.peak_balance > 0 else 0, 2),
        }

    def render(self) -> None:
        """Print current state (Gymnasium standard — no mode param)."""
        info = self._get_info()
        print(f"Step {info['step']} | Balance ${info['balance']:.2f} | "
              f"Pos: {info['position']} | Trades: {info['total_trades']} | "
              f"WR: {info['win_rate']:.1f}% | DD: {info['drawdown_pct']:.1f}%")

    def close(self) -> None:
        """Clean up environment resources."""
        pass
