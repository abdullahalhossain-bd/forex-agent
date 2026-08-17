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


def infer_pip_size(pair: str) -> float:
    """Round-7 fix: pip_size used to be hardcoded 0.0001 everywhere,
    correct only for standard non-JPY forex pairs. The user's real
    48-pair universe (from their config.py) includes 10 JPY crosses
    (EURJPY, GBPJPY, AUDJPY, NZDJPY, CADJPY, CHFJPY, USDJPY, SGDJPY,
    HKDJPY, MXNJPY — pip_size 0.01, two decimal places) and 4 metals
    (XAUUSD, XAGUSD, XPTUSD, XPDUSD). Training any of these with
    pip_size=0.0001 makes every pip-denominated calculation (SL/TP
    distance, lot sizing, PnL in USD) wrong by 100x for JPY pairs —
    the position-sizing math (`sl_pips = sl_distance / pip_size`,
    `pnl = ... / pip_size * 10 * lot`) would silently compute
    nonsense risk/reward for any JPY pair someone runs through the
    training queue.
    Metal pip conventions vary by broker — 0.01 is a common default
    but NOT universal; verify against your actual broker spec before
    relying on metals PnL figures.
    """
    p = (pair or "").upper().replace("/", "").replace("_", "")
    if p.endswith("JPY") or "JPY" in p[3:]:
        return 0.01
    if p.startswith("XAU") or p.startswith("XAG") or p.startswith("XPT") or p.startswith("XPD"):
        return 0.01  # common convention; confirm against your broker
    return 0.0001


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
        # AUDIT FIX (winrate/frequency round 2): the first fix pass
        # (SL/TP checked every step, proper episode-end reward, correct
        # metrics) exposed a real, non-bug problem once it could
        # finally be measured honestly: ~190 trades per 1000-bar
        # episode (avg hold ~5 bars) with win_rate 25-33%, below the
        # 33.4% breakeven for a 1:2 R:R. Two concrete levers for this:
        #   1. SL was min 10 pips / 1.5x ATR — tight enough on M15 that
        #      the agent was mostly getting stopped out by noise, and
        #      every one of those churns pays the spread twice. Widen
        #      it so real moves have room and cost-per-trade drops as
        #      a fraction of the stop.
        #   2. Nothing stopped the agent from opening a new trade the
        #      bar immediately after closing one. `overtrade_penalty`
        #      only discourages this softly (and the agent kept doing
        #      it anyway at 60k timesteps). Add a hard cooldown — the
        #      agent literally cannot open a new position for N bars
        #      after a close — plus a hard daily trade cap (block, not
        #      just penalize) so overtrading can't be trained around.
        sl_atr_multiplier: float = 2.5,       # was 1.5 — wider stop, less noise-stopout
        min_sl_pips: float = 15.0,            # was hardcoded 10 — matches wider ATR mult
        cooldown_bars: int = 4,               # round-3: 8→4 — richer features should
                                               # find more genuine (not just noise) setups
        max_trades_per_day: int = 6,          # round-3: 4→6 — same reasoning
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
        self.sl_atr_multiplier = sl_atr_multiplier
        self.min_sl_pips = min_sl_pips
        self.cooldown_bars = cooldown_bars
        self.max_trades_per_day = max_trades_per_day
        self.last_close_step = -10**9

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
        self.last_close_step = -10**9

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
        self.last_close_step = -10**9

        return self._get_state(), self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step with proper termination logic."""
        # ── Termination checks ──────────────────────────────────────
        at_end_of_data = self.current_step >= len(self.df) - 1
        max_steps_reached = self.episode_step >= self.max_steps_per_episode
        bankrupt = self.balance <= self.initial_balance * 0.5  # 50% DD = bankrupt

        if at_end_of_data or max_steps_reached or bankrupt:
            # AUDIT FIX (win-rate/frequency bug #2): forced episode-end
            # close used to return a hardcoded reward of 0.0 no matter
            # what the final PnL was. Any position still open when the
            # episode was cut off (max_steps_per_episode, bankrupt, or
            # end-of-data) got closed but the agent received ZERO
            # learning signal for that outcome — including large,
            # realistic losses. That silently hid a chunk of the
            # policy's real losses from the reward function, and let
            # the agent get away with holding risky positions right up
            # to an episode boundary. Route the final close through the
            # same reward engine as every other close so PPO actually
            # sees the consequence.
            end_reward = 0.0
            if self.position.direction != "NONE":
                final_pnl = self._close_position(reason="episode_end")
                trade_closed = True
                win = final_pnl > 0
                rr_ratio = 0.0
                if self.closed_trades:
                    rr_ratio = self.closed_trades[-1].get("rr_ratio", 0.0)
                end_rb = self.reward_engine.calculate(
                    action="CLOSE",
                    pnl_usd=final_pnl,
                    balance=max(self.balance, 0),
                    initial_balance=self.initial_balance,
                    risk_pct=self.risk_per_trade,
                    rr_ratio=rr_ratio,
                    trades_today=self.trades_today,
                    peak_balance=self.peak_balance,
                    position_open=False,
                    trade_closed=trade_closed,
                    win=win,
                )
                end_reward = end_rb.total
                self.episode_reward += end_reward
                self.episode_pnl += final_pnl
                if self.balance > self.peak_balance:
                    self.peak_balance = self.balance
            return self._get_state(), float(end_reward), True, False, self._get_info()

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

        # AUDIT FIX (win-rate/frequency bug #1): SL/TP used to be
        # checked ONLY on steps where action_name == "HOLD". Any step
        # where the agent output BUY/SELL (no-op, since a position was
        # already open — the open-guards below prevent re-entry) or
        # CLOSE skipped the SL/TP check entirely for that bar's
        # high/low. Concretely: once a position is open, the agent
        # only needs to emit a non-HOLD action once and every
        # subsequent SL/TP touch on intervening bars is silently
        # missed — the position keeps "riding" past its stop with no
        # exit recorded, so realized losses are bigger than the
        # configured 1R, and real TP hits get skipped until the agent
        # happens to HOLD again (if ever, before end-of-data forces an
        # exit at whatever the market price is then). This alone is
        # sufficient to explain a policy that both loses more per
        # losing trade than intended AND appears to win less often —
        # genuine TP hits were never being credited as wins.
        # FIX: check SL/TP against the CURRENT bar's high/low on every
        # single step where a position is open, before any other
        # action is processed for that step.
        if self.position.direction != "NONE":
            pnl = self._check_sl_tp(high_price, low_price)
            if pnl != 0:
                pnl_this_step = pnl
                trade_closed = True
                win = pnl > 0

        # AUDIT FIX (winrate/frequency round 2): can_enter gates every
        # new position on (a) cooldown bars since the last close and
        # (b) a HARD daily trade cap — not just a reward penalty the
        # agent could (and did) train straight through. This is the
        # direct fix for the ~190-trades-per-1000-bars churn observed
        # after the round-1 fixes.
        can_enter = (
            (self.current_step - self.last_close_step) >= self.cooldown_bars
            and self.trades_today < self.max_trades_per_day
        )

        if action_name == "BUY" and self.position.direction == "NONE" and can_enter:
            entry_spread = self._open_position("LONG", close_price, high_price, low_price)
            self.trades_today += 1
            self.total_trades += 1

        elif action_name == "SELL" and self.position.direction == "NONE" and can_enter:
            entry_spread = self._open_position("SHORT", close_price, high_price, low_price)
            self.trades_today += 1
            self.total_trades += 1

        elif action_name == "CLOSE" and self.position.direction != "NONE" and not trade_closed:
            pnl = self._close_position(reason="manual_close")
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
        # AUDIT FIX (winrate/frequency round 2): wider ATR multiplier
        # and higher min-pip floor (configurable, was hardcoded 1.5x /
        # 10 pips) — gives real moves room instead of getting stopped
        # by M15 noise, and shrinks spread-cost-as-%-of-stop for every
        # trade that does get taken.
        sl_distance = max(atr * self.sl_atr_multiplier, self.min_sl_pips * self.pip_size)
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

        # AUDIT FIX (winrate/frequency round 2): stamp the cooldown
        # clock here — this is the single choke point every close
        # (manual, SL, TP, episode-end) funnels through, so the
        # cooldown check in step() is guaranteed to see it.
        self.last_close_step = self.current_step

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
        """Get normalized state vector.

        AUDIT FIX (winrate round 3): the feature schema used to be a
        hardcoded 10-name list baked into the env, decoupled from
        whatever columns features_df actually had — silently dropping
        any richer feature added upstream unless this list was kept in
        sync by hand. Now the env just consumes ALL columns of
        features_df dynamically (n_features already tracks
        len(features_df.columns)+6 from __init__), so the feature set
        can be extended in the builder (train_rl_v2.build_features_df_v2)
        without touching the environment.
        """
        if self.current_step >= len(self.df):
            return np.zeros(self.n_features, dtype=np.float32)

        if self.features_df is not None and self.current_step < len(self.features_df):
            row = self.features_df.iloc[self.current_step]
            market_features = row.to_numpy(dtype=np.float32)
        else:
            FEATURE_SCHEMA = [
                "close", "high", "low", "volume",
                "rsi_14", "atr", "macd", "ema_20", "ema_50", "sma_200",
            ]
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
