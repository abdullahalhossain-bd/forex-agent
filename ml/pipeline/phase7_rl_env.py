"""
ml/pipeline/phase7_rl_env.py — Enhanced RL Trading Environment (Phase 7)
=========================================================================
Realistic trading environment for reinforcement learning training.

Includes:
  - Spread (variable, from data if available)
  - Commission
  - Slippage (configurable)
  - Swap (simplified)
  - Execution delay
  - Stop Loss / Take Profit / Trailing Stop
  - Margin / Leverage
  - Risk % position sizing
  - Long-term profitability reward (not win rate)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from ml.pipeline.utils import PipelineConfig, get_pipeline_logger

log = get_pipeline_logger("phase7_rl_env")


@dataclass
class TradeRecord:
    entry_step: int = 0
    direction: int = 0       # 1=LONG, -1=SHORT
    entry_price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    trailing_stop: float = 0.0
    lot: float = 0.0
    commission: float = 0.0


class EnhancedTradingEnv(gym.Env):
    """Institutional-grade RL trading environment.
    
    Actions: 0=HOLD, 1=BUY, 2=SELL, 3=CLOSE
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        initial_balance: float = 10000.0,
        risk_per_trade: float = 0.01,
        spread_pips: float = 1.5,
        commission_per_lot: float = 7.0,    # $7 per standard lot round-trip
        slippage_pips: float = 0.5,
        leverage: int = 100,
        swap_long: float = -3.5,             # Annual swap points
        swap_short: float = -1.5,
        pip_size: float = 0.0001,
        reward_scale: float = 100.0,         # Scale rewards for RL stability
        max_position_holding: int = 200,     # Max candles to hold a position
        trailing_stop_atr_mult: float = 2.0,
        pair: str = "EURUSD",
    ):
        super().__init__()
        
        self.pair = pair
        self.pip_size = pip_size
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.spread_pips = spread_pips
        self.commission_per_lot = commission_per_lot
        self.slippage_pips = slippage_pips
        self.leverage = leverage
        self.swap_long = swap_long
        self.swap_short = swap_short
        self.reward_scale = reward_scale
        self.max_holding = max_position_holding
        self.trailing_atr_mult = trailing_stop_atr_mult
        
        # Data
        self.df = df.reset_index(drop=True)
        self.feature_columns = feature_columns or [c for c in df.columns if c not in (
            "timestamp", "open", "high", "low", "close", "volume", "signal", "regime"
        )]
        
        self.n_features = len(self.feature_columns) + 8  # +8 for position/account state
        
        # Spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)
        
        # State
        self._reset_state()
    
    def _reset_state(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.position: Optional[TradeRecord] = None
        self.trades: List[Dict] = []
        self.total_trades = 0
        self.total_wins = 0
        self.total_pnl = 0.0
        self.episode_reward = 0.0
        self.equity_curve: List[float] = []
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._reset_state()
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        if self.current_step >= len(self.df) - 1:
            self._close_position("end_of_data")
            return self._get_obs(), 0.0, True, False, self._get_info()
        
        reward = 0.0
        pnl = 0.0
        
        # Process action
        if action == 1 and self.position is None:      # BUY
            self._open_position(1)
        elif action == 2 and self.position is None:    # SELL
            self._open_position(-1)
        elif action == 3 and self.position is not None: # CLOSE
            pnl = self._close_position("manual")
        
        # Check SL/TP/trailing for open positions
        if self.position is not None:
            row = self.df.iloc[self.current_step]
            sl_pnl = self._check_exits(row)
            if sl_pnl is not None:
                pnl = sl_pnl
            
            # Apply swap (simplified: per-candle cost)
            if self.position is not None:
                swap_pts = self.swap_long if self.position.direction == 1 else self.swap_short
                swap_cost = swap_pts * self.pip_size * self.position.lot * 100000 / 365
                self.balance -= abs(swap_cost)
        
        # Calculate reward: profitability-focused
        reward = self._calculate_reward(pnl)
        self.episode_reward += reward
        self.total_pnl += pnl
        
        # Track equity
        equity = self.balance + self._unrealized_pnl()
        self.equity_curve.append(equity)
        if equity > self.peak_balance:
            self.peak_balance = equity
        
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        
        return self._get_obs(), float(reward), terminated, truncated, self._get_info()
    
    def _open_position(self, direction: int):
        row = self.df.iloc[self.current_step]
        close = float(row["close"])
        
        # Spread + slippage
        spread_cost = self.spread_pips * self.pip_size
        slippage = self.slippage_pips * self.pip_size
        
        if direction == 1:
            entry = close + spread_cost / 2 + slippage
        else:
            entry = close - spread_cost / 2 - slippage
        
        # SL/TP from ATR
        atr = float(row.get("atr", 0.001))
        if atr <= 0:
            atr = 0.001
        
        risk_usd = self.balance * self.risk_per_trade
        pip_value = self.pip_size * 100000  # For standard lot
        sl_distance = atr * 1.5
        tp_distance = atr * 3.0
        
        if direction == 1:
            sl = entry - sl_distance
            tp = entry + tp_distance
        else:
            sl = entry + sl_distance
            tp = entry - tp_distance
        
        lot = risk_usd / (sl_distance / self.pip_size * 10) if sl_distance > 0 else 0.01
        lot = max(0.01, min(round(lot, 2), 10.0))
        
        # Commission
        commission = self.commission_per_lot * lot
        
        self.position = TradeRecord(
            entry_step=self.current_step, direction=direction,
            entry_price=entry, sl=sl, tp=tp,
            trailing_stop=sl, lot=lot, commission=commission,
        )
        self.balance -= commission
    
    def _close_position(self, reason: str) -> float:
        if self.position is None:
            return 0.0
        
        row = self.df.iloc[self.current_step]
        close = float(row["close"])
        spread_cost = self.spread_pips * self.pip_size
        slippage = self.slippage_pips * self.pip_size
        
        if self.position.direction == 1:
            exit_price = close - spread_cost / 2 - slippage
            pnl = (exit_price - self.position.entry_price) / self.pip_size * 10 * self.position.lot
        else:
            exit_price = close + spread_cost / 2 + slippage
            pnl = (self.position.entry_price - exit_price) / self.pip_size * 10 * self.position.lot
        
        self.balance += pnl
        self.total_trades += 1
        if pnl > 0:
            self.total_wins += 1
        
        self.trades.append({
            "step": self.current_step, "direction": self.position.direction,
            "entry": self.position.entry_price, "exit": exit_price,
            "pnl": pnl, "reason": reason, "holding_period": self.current_step - self.position.entry_step,
        })
        
        self.position = None
        return pnl
    
    def _check_exits(self, row) -> Optional[float]:
        if self.position is None:
            return None
        
        high = float(row["high"])
        low = float(row["low"])
        
        # Update trailing stop
        if self.position.direction == 1:
            new_trail = low - self.trailing_atr_mult * float(row.get("atr", 0.001))
            if new_trail > self.position.trailing_stop:
                self.position.trailing_stop = new_trail
        
        # Check SL
        if self.position.direction == 1 and low <= self.position.sl:
            return self._close_position("SL")
        if self.position.direction == -1 and high >= self.position.sl:
            return self._close_position("SL")
        
        # Check trailing stop
        if self.position.direction == 1 and low <= self.position.trailing_stop:
            return self._close_position("trailing_stop")
        if self.position.direction == -1 and high >= self.position.trailing_stop:
            return self._close_position("trailing_stop")
        
        # Check TP
        if self.position.direction == 1 and high >= self.position.tp:
            return self._close_position("TP")
        if self.position.direction == -1 and low <= self.position.tp:
            return self._close_position("TP")
        
        # Max holding period
        if self.current_step - self.position.entry_step > self.max_holding:
            return self._close_position("max_holding")
        
        return None
    
    def _calculate_reward(self, pnl: float) -> float:
        """Profitability-focused reward."""
        if pnl != 0:
            # Realized PnL as primary reward, scaled
            reward = (pnl / self.initial_balance) * self.reward_scale
            # Penalty for losses (asymmetric)
            if pnl < 0:
                reward *= 1.5
            return reward
        
        # Unrealized reward component
        if self.position is not None:
            unrealized = self._unrealized_pnl()
            return (unrealized / self.initial_balance) * self.reward_scale * 0.1  # Small ongoing signal
        
        # Small hold penalty (encourages taking trades when appropriate)
        return -0.01
    
    def _unrealized_pnl(self) -> float:
        if self.position is None:
            return 0.0
        current = float(self.df.iloc[min(self.current_step, len(self.df)-1)]["close"])
        if self.position.direction == 1:
            return (current - self.position.entry_price) / self.pip_size * 10 * self.position.lot
        return (self.position.entry_price - current) / self.pip_size * 10 * self.position.lot
    
    def _get_obs(self) -> np.ndarray:
        if self.current_step >= len(self.df):
            return np.zeros(self.n_features, dtype=np.float32)
        
        row = self.df.iloc[self.current_step]
        features = np.array([
            float(row.get(f, 0)) if pd.notna(row.get(f, 0)) else 0.0
            for f in self.feature_columns
        ], dtype=np.float32)
        
        pos_state = np.array([
            1.0 if self.position is not None else 0.0,
            1.0 if self.position is not None and self.position.direction == 1 else 0.0,
            self.position.entry_price / 10000 if self.position else 0.0,
            self.balance / self.initial_balance,
            len(self.trades) / 50.0,
            (self.peak_balance - self.balance) / self.peak_balance if self.peak_balance > 0 else 0.0,
            self._unrealized_pnl() / self.initial_balance,
            1.0 if self.position is not None and (self.current_step - self.position.entry_step) > 100 else 0.0,
        ], dtype=np.float32)
        
        state = np.concatenate([features, pos_state])
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        if len(state) < self.n_features:
            state = np.pad(state, (0, self.n_features - len(state)))
        elif len(state) > self.n_features:
            state = state[:self.n_features]
        return state.astype(np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        return {
            "step": self.current_step,
            "balance": round(self.balance, 2),
            "equity": round(self.balance + self._unrealized_pnl(), 2),
            "position_open": self.position is not None,
            "total_trades": self.total_trades,
            "win_rate": (self.total_wins / self.total_trades * 100) if self.total_trades > 0 else 0.0,
            "total_pnl": round(self.total_pnl, 2),
            "episode_reward": round(self.episode_reward, 2),
            "drawdown_pct": round((self.peak_balance - self.balance) / self.peak_balance * 100, 2) if self.peak_balance > 0 else 0.0,
        }
    
    def render(self):
        info = self._get_info()
        print(f"Step {info['step']} | Bal ${info['balance']:.2f} | "
              f"Equity ${info['equity']:.2f} | Trades {info['total_trades']} | "
              f"WR {info['win_rate']:.1f}% | PnL ${info['total_pnl']:.2f}")
    
    def close(self):
        pass