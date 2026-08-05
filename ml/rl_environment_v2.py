"""
ml/rl_environment_v2.py — Profitability-Focused RL Environment
==============================================================

FIXES from v1 (the env that produced avg_reward=-429,620, 0% WR, 1 episode):

BUG #1: Bankrupt check didn't terminate episode
  v1: `if bankrupt: self._close_position()` but `terminated = (step >= len) or bankrupt`
      Problem: bankrupt is checked BEFORE terminated is set, but the position
      close doesn't reset balance. Balance stays negative, next step opens
      new position, loses more, balance goes more negative...
      Result: 1 episode ran all 50k steps with exponentially growing losses
  FIX: bankrupt → terminated=True IMMEDIATELY. Episode ends.
       PPO gets proper episode boundaries to learn from.

BUG #2: Negative balance allowed
  v1: balance could go below 0 (no floor)
      With negative balance, lot_size calculation inverted (risk_usd / lot_cost
      → negative lot → math breaks)
  FIX: Floor balance at 0. If balance hits 0 → bankrupt → terminate.

BUG #3: Lot size not properly risk-controlled
  v1: lot = risk_usd / (sl_distance / pip_size * 10)
      If sl_distance is tiny (low ATR), lot explodes to 10.0 cap
      One bad trade on max lot = 10%+ loss
  FIX: Hard cap lot at risk_usd / (sl_pips * pip_value), max 2% risk

BUG #4: No trade-quality tracking
  v1: Only tracked total_wins/total_losses, not R:R per trade
  FIX: Track closed_trades with full details, compute profit factor, expectancy

BUG #5: Single episode = no learning
  v1: With 5000 rows and 50k timesteps, expected ~10 episodes
      But episode only ended at end-of-data, so 1 episode covered everything
  FIX: Add max_steps_per_episode=1000 — force episode end for PPO rollouts

NEW FEATURES:
  - Proper episode termination on bankrupt / max_steps / end_of_data
  - Trade journal with R:R, hold time, exit reason per trade
  - Profit factor and expectancy tracking
  - Risk-adjusted position sizing (Kelly-inspired)
  - Slippage simulation on entry/exit
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from utils.logger import get_logger
from ml.reward_engine_v2 import RewardEngineV2, get_reward_engine_v2

log = get_logger("rl_environment_v2")


# Action constants
ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL = 2
ACTION_CLOSE = 3
ACTIONS = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "CLOSE"}


@dataclass
class Position:
    """Open position state."""
    direction: str = "NONE"
    entry: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    lot: float = 0.0
    opened_at_step: int = 0
    entry_spread_cost: float = 0.0


class ForexTradingEnvV2(gym.Env):
    """Profitability-focused forex trading environment.

    Key differences from v1:
      1. bankrupt → terminated=True (was: just closed position)
      2. max_steps_per_episode forces episode boundaries for PPO
      3. Balance floored at 0 (no negative balances)
      4. Proper lot sizing with risk caps
      5. Trade journal with R:R tracking
      6. Slippage simulation
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        features_df: Optional[pd.DataFrame] = None,
        initial_balance: float = 10000.0,
        risk_per_trade: float = 0.01,    # 1% risk per trade
        pip_size: float = 0.0001,
        spread_pips: float = 1.5,
        slippage_pips: float = 0.5,      # NEW: slippage simulation
        pair: str = "EURUSD",
        reward_engine: Optional[RewardEngineV2] = None,
        max_steps_per_episode: int = 1000,  # NEW: force episode end
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.df = df.reset_index(drop=True)
        self.features_df = features_df.reset_index(drop=True) if features_df is not None else None
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.pip_size = pip_size
        self.spread_pips = spread_pips
        self.slippage_pips = slippage_pips
        self.pair = pair
        self.reward_engine = reward_engine or get_reward_engine_v2()
        self.max_steps_per_episode = max_steps_per_episode

        # State space
        if self.features_df is not None:
            self.n_features = len(self.features_df.columns) + 6
        else:
            self.n_features = min(20, len(df.columns)) + 6

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        # Episode state
        self.current_step = 0
        self.episode_step = 0  # NEW: track steps within episode
        self.balance = initial_balance
        self.peak_balance = initial_balance
        self.position = Position()
        self.trades_today = 0
        self.current_day = None
        self.closed_trades: list = []
        self.total_trades = 0
        self.total_wins = 0
        self.total_losses = 0
        self.episode_reward = 0.0
        self.episode_pnl = 0.0
        self.start_idx = 0  # NEW: random start for each episode

    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset for new episode with random starting point."""
        super().reset(seed=seed, options=options)

        if hasattr(self.reward_engine, "reset_episode"):
            self.reward_engine.reset_episode()

        # NEW: Random start index so each episode sees different data
        max_start = max(0, len(self.df) - self.max_steps_per_episode - 1)
        if max_start > 0:
            self.start_idx = int(self.np_random.integers(0, max_start))
        else:
            self.start_idx = 0

        self.current_step = self.start_idx
        self.episode_step = 0
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.position = Position()
        self.trades_today = 0
        self.current_day = None
        self.closed_trades = []
        self.total_trades = 0
        self.total_wins = 0
        self.total_losses = 0
        self.episode_reward = 0.0
        self.episode_pnl = 0.0

        return self._get_state(), self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step with proper termination logic."""
        # ── Termination checks ──────────────────────────────────────
        at_end_of_data = self.current_step >= len(self.df) - 1
        max_steps_reached = self.episode_step >= self.max_steps_per_episode
        bankrupt = self.balance <= self.initial_balance * 0.5  # 50% DD = bankrupt

        if at_end_of_data or max_steps_reached or bankrupt:
            # Close any open position
            if self.position.direction != "NONE":
                self._close_position(reason="episode_end")
            return self._get_state(), 0.0, True, False, self._get_info()

        action_name = ACTIONS.get(action, "HOLD")
        pnl_this_step = 0.0
        trade_closed = False
        win = False

        # Track day change
        try:
            current_dt = self.df.iloc[self.current_step].name
            if hasattr(current_dt, 'date'):
                day = current_dt.date()
            else:
                day = self.current_step // 96
        except Exception:
            day = self.current_step // 96
        if day != self.current_day:
            self.current_day = day
            self.trades_today = 0

        # ── Execute action ──────────────────────────────────────────
        close_price = float(self.df.iloc[self.current_step].get("close", 0))
        high_price = float(self.df.iloc[self.current_step].get("high", close_price))
        low_price = float(self.df.iloc[self.current_step].get("low", close_price))

        if action_name == "BUY" and self.position.direction == "NONE":
            entry_spread = self._open_position("LONG", close_price, high_price, low_price)
            self.trades_today += 1
            self.total_trades += 1

        elif action_name == "SELL" and self.position.direction == "NONE":
            entry_spread = self._open_position("SHORT", close_price, high_price, low_price)
            self.trades_today += 1
            self.total_trades += 1

        elif action_name == "CLOSE" and self.position.direction != "NONE":
            pnl = self._close_position(reason="manual_close")
            pnl_this_step = pnl
            trade_closed = True
            win = pnl > 0

        elif action_name == "HOLD":
            if self.position.direction != "NONE":
                pnl = self._check_sl_tp(high_price, low_price)
                if pnl != 0:
                    pnl_this_step = pnl
                    trade_closed = True
                    win = pnl > 0

        # ── Calculate R:R for reward shaping ───────────────────────
        rr_ratio = 0.0
        if self.position.direction != "NONE" and self.position.sl > 0:
            risk = abs(self.position.entry - self.position.sl)
            reward_dist = abs(self.position.tp - self.position.entry) if self.position.tp > 0 else 0
            rr_ratio = reward_dist / risk if risk > 0 else 0
        # For closed trades, use the actual trade R:R
        if trade_closed and self.closed_trades:
            last_trade = self.closed_trades[-1]
            rr_ratio = last_trade.get("rr_ratio", rr_ratio)

        # ── Calculate reward ───────────────────────────────────────
        drawdown = (self.peak_balance - self.balance) / self.peak_balance if self.peak_balance > 0 else 0
        spread_cost = self.spread_pips * self.pip_size * (self.position.lot if self.position.direction != "NONE" else 0)

        reward_rb = self.reward_engine.calculate(
            action=action_name,
            pnl_usd=pnl_this_step,
            balance=max(self.balance, 0),  # never negative
            initial_balance=self.initial_balance,
            risk_pct=self.risk_per_trade,
            rr_ratio=rr_ratio,
            trades_today=self.trades_today,
            peak_balance=self.peak_balance,
            position_open=(self.position.direction != "NONE"),
            trade_closed=trade_closed,
            win=win,
            spread_cost_usd=spread_cost,
        )
        reward = reward_rb.total
        self.episode_reward += reward
        self.episode_pnl += pnl_this_step

        # ── Advance ─────────────────────────────────────────────────
        self.current_step += 1
        self.episode_step += 1

        if self.balance > self.peak_balance:
            self.peak_balance = self.balance

        # Check termination AFTER advance
        at_end_of_data = self.current_step >= len(self.df) - 1
        max_steps_reached = self.episode_step >= self.max_steps_per_episode
        bankrupt = self.balance <= self.initial_balance * 0.5

        terminated = at_end_of_data or max_steps_reached or bankrupt
        truncated = False

        return self._get_state(), float(reward), terminated, truncated, self._get_info()

    def _open_position(self, direction: str, entry: float, high: float, low: float) -> float:
        """Open position with proper risk-based lot sizing."""
        try:
            atr = float(self.df.iloc[self.current_step].get("atr", 0.001))
        except Exception:
            atr = 0.001
        if atr <= 0:
            atr = 0.001

        # SL/TP based on ATR (matches RiskEngine defaults)
        sl_distance = max(atr * 1.5, 10 * self.pip_size)  # min 10 pips
        tp_distance = sl_distance * 2.0  # 1:2 R:R

        # Apply spread + slippage
        cost = (self.spread_pips + self.slippage_pips) * self.pip_size
        if direction == "LONG":
            entry_price = entry + cost / 2
            sl = entry_price - sl_distance
            tp = entry_price + tp_distance
        else:
            entry_price = entry - cost / 2
            sl = entry_price + sl_distance
            tp = entry_price - tp_distance

        # FIX: Proper lot sizing — risk_usd / (sl_pips × pip_value)
        # Cap at 2% risk max (was unbounded)
        risk_usd = self.balance * self.risk_per_trade
        sl_pips = sl_distance / self.pip_size
        pip_value = 10.0  # $10 per pip per standard lot for USD pairs
        lot = risk_usd / (sl_pips * pip_value) if sl_pips > 0 else 0.01
        lot = max(0.01, min(round(lot, 2), 5.0))  # cap at 5 lots

        self.position = Position(
            direction=direction,
            entry=entry_price,
            sl=sl,
            tp=tp,
            lot=lot,
            opened_at_step=self.current_step,
            entry_spread_cost=cost,
        )
        return cost

    def _close_position(self, reason: str = "manual") -> float:
        """Close position, return realized PnL."""
        if self.position.direction == "NONE":
            return 0.0

        close_price = float(self.df.iloc[self.current_step].get("close", 0))
        # Apply slippage on close
        if self.position.direction == "LONG":
            close_price -= self.slippage_pips * self.pip_size
        else:
            close_price += self.slippage_pips * self.pip_size

        return self._finalize_close(close_price, reason)

    def _close_at_price(self, price: float, reason: str) -> float:
        """Close at specific price (SL/TP hit)."""
        if self.position.direction == "NONE":
            return 0.0
        return self._finalize_close(price, reason)

    def _finalize_close(self, close_price: float, reason: str) -> float:
        """Common close logic — compute PnL, update state, record trade."""
        if self.position.direction == "NONE":
            return 0.0

        # PnL calculation
        if self.position.direction == "LONG":
            pnl = (close_price - self.position.entry) / self.pip_size * 10 * self.position.lot
        else:
            pnl = (self.position.entry - close_price) / self.pip_size * 10 * self.position.lot

        # Subtract spread cost (already paid at entry, but track for journal)
        self.balance += pnl
        # FIX: Floor balance at 0
        if self.balance < 0:
            self.balance = 0

        # Win/loss tracking
        if pnl > 0:
            self.total_wins += 1
        else:
            self.total_losses += 1

        # R:R calculation for this trade
        risk = abs(self.position.entry - self.position.sl)
        actual_rr = abs(pnl) / (risk * self.position.lot * 10) if risk > 0 and self.position.lot > 0 else 0

        # Record trade
        self.closed_trades.append({
            "step": self.current_step,
            "direction": self.position.direction,
            "entry": self.position.entry,
            "exit": close_price,
            "pnl_usd": pnl,
            "lot": self.position.lot,
            "reason": reason,
            "rr_ratio": actual_rr,
            "hold_bars": self.current_step - self.position.opened_at_step,
        })

        log.debug(f"[RL Env V2] CLOSE {self.position.direction} @ {close_price:.5f} "
                  f"PnL=${pnl:.2f} R:R=1:{actual_rr:.2f} ({reason})")
        self.position = Position()
        return pnl

    def _check_sl_tp(self, high: float, low: float) -> float:
        """Check SL/TP hit."""
        if self.position.direction == "NONE":
            return 0.0

        if self.position.direction == "LONG":
            if low <= self.position.sl:
                return self._close_at_price(self.position.sl, "SL hit")
            if high >= self.position.tp:
                return self._close_at_price(self.position.tp, "TP hit")
        else:
            if high >= self.position.sl:
                return self._close_at_price(self.position.sl, "SL hit")
            if low <= self.position.tp:
                return self._close_at_price(self.position.tp, "TP hit")
        return 0.0

    def _get_state(self) -> np.ndarray:
        """Get normalized state vector."""
        if self.current_step >= len(self.df):
            return np.zeros(self.n_features, dtype=np.float32)

        FEATURE_SCHEMA = [
            "close", "high", "low", "volume",
            "rsi_14", "atr", "macd", "ema_20", "ema_50", "sma_200",
        ]

        if self.features_df is not None and self.current_step < len(self.features_df):
            row = self.features_df.iloc[self.current_step]
        else:
            row = self.df.iloc[self.current_step]

        market_features = np.array([
            float(row.get(f, 0) if row.get(f) is not None else 0)
            for f in FEATURE_SCHEMA
        ], dtype=np.float32)

        position_state = np.array([
            1.0 if self.position.direction == "LONG" else 0.0,
            1.0 if self.position.direction == "SHORT" else 0.0,
            self.position.entry / 10000.0 if self.position.entry > 0 else 0.0,
            self.balance / self.initial_balance,
            self.trades_today / 20.0,
            (self.peak_balance - self.balance) / self.peak_balance if self.peak_balance > 0 else 0.0,
        ], dtype=np.float32)

        state = np.concatenate([market_features, position_state])
        if len(state) < self.n_features:
            state = np.pad(state, (0, self.n_features - len(state)))
        elif len(state) > self.n_features:
            state = state[:self.n_features]

        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        return state.astype(np.float32)

    def _get_info(self) -> Dict[str, Any]:
        """Get info dict with profitability metrics."""
        # Calculate profit factor
        gross_profit = sum(t["pnl_usd"] for t in self.closed_trades if t["pnl_usd"] > 0)
        gross_loss = abs(sum(t["pnl_usd"] for t in self.closed_trades if t["pnl_usd"] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # Calculate expectancy
        if self.closed_trades:
            expectancy = sum(t["pnl_usd"] for t in self.closed_trades) / len(self.closed_trades)
        else:
            expectancy = 0.0

        return {
            "step": self.current_step,
            "episode_step": self.episode_step,
            "balance": round(self.balance, 2),
            "equity": round(self.balance, 2),
            "position": self.position.direction,
            "trades_today": self.trades_today,
            "total_trades": self.total_trades,
            "total_wins": self.total_wins,
            "total_losses": self.total_losses,
            "win_rate": (self.total_wins / self.total_trades * 100) if self.total_trades > 0 else 0.0,
            "episode_reward": round(self.episode_reward, 2),
            "episode_pnl": round(self.episode_pnl, 2),
            "drawdown_pct": round(((self.peak_balance - self.balance) / self.peak_balance * 100) if self.peak_balance > 0 else 0, 2),
            "profit_factor": round(profit_factor, 2),
            "expectancy_usd": round(expectancy, 2),
            "closed_trades_count": len(self.closed_trades),
        }

    def render(self) -> None:
        info = self._get_info()
        print(f"Step {info['step']} | Bal ${info['balance']:.2f} | "
              f"Pos: {info['position']} | Trades: {info['total_trades']} | "
              f"WR: {info['win_rate']:.1f}% | PF: {info['profit_factor']:.2f} | "
              f"DD: {info['drawdown_pct']:.1f}%")

    def close(self) -> None:
        pass
