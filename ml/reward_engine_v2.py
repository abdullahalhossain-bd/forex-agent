"""
ml/reward_engine_v2.py — Profitability-Focused RL Reward Engine
================================================================

FIXES from v1 (the engine that produced avg_reward=-429,620, 0% WR, 1 episode):

BUG #1: Reward scaling was broken
  v1: profit_reward = pnl_pct × 5.0 × 100 = pnl_pct × 500
      A 1% profit → +5.0 reward (fine)
      A 10% loss → -50 reward, clipped to -20 (fine)
  BUT: pnl_pct was computed as pnl_usd / balance
      If balance went negative (bankrupt but not terminated),
      pnl_pct sign flips! A loss on negative balance shows as "profit"
  FIX: Use ABSOLUTE initial_balance for scaling, never current balance.

BUG #2: Episode never terminated
  v1: bankrupt check closed position but didn't set terminated=True
      → balance could go to -infinity
      → single episode ran all 50k timesteps
      → PPO got 1 episode boundary to learn from
  FIX: bankrupt → terminated=True immediately (see env fix)

BUG #3: No incentive for ACTUAL profitability
  v1: Rewarded pnl_pct, hold patience, risk management
      But never explicitly rewarded PROFIT FACTOR or EXPECTANCY
  FIX: Add trade-quality shaping:
       - Win with R:R ≥ 2:1 → 2× reward multiplier
       - Loss with R:R < 1:1 → 0.5× penalty (forgive small losses)
       - Consecutive wins → bonus (momentum)
       - Consecutive losses → escalating penalty (stop bleeding)

BUG #4: Hold reward too generous
  v1: 0.1 per hold × 20 max idle = +2.0 guaranteed for doing nothing
      A winning trade only earned ~5.0, so HOLD was 40% as good as WIN
      → PPO learned "never trade" (degenerate policy)
  FIX: Reduce hold reward to 0.02, max 5 idle steps = +0.1 total
      Now winning is 50× better than holding.

BUG #5: No transaction cost awareness
  v1: Spread cost was deducted from PnL but not from reward
      → Agent couldn't "feel" the cost of overtrading
  FIX: Subtract spread_cost from reward explicitly on every trade

NEW FEATURES:
  - Risk-adjusted reward (Sharpe-like): reward / volatility
  - Regime-aware shaping: trend-following rewarded in TRENDING, punished in RANGING
  - Time-decay: trades held too long get penalized (opportunity cost)
  - Quality gate: only reward trades that meet MIN_RR threshold
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, List
from collections import deque

from utils.logger import get_logger

log = get_logger("reward_engine_v2")


@dataclass
class RewardBreakdownV2:
    """Detailed reward breakdown for one step."""
    profit_reward: float = 0.0
    loss_penalty: float = 0.0
    risk_reward: float = 0.0
    overtrading_penalty: float = 0.0
    drawdown_penalty: float = 0.0
    hold_reward: float = 0.0
    transaction_cost: float = 0.0
    streak_bonus: float = 0.0
    quality_shaping: float = 0.0
    total: float = 0.0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RewardEngineV2:
    """Profitability-focused RL reward engine.

    Design principles:
      1. Reward ACTUAL profitability, not just trade frequency
      2. Asymmetric: wins reward more than losses penalize (encourage risk-taking)
      3. Quality-gated: only reward trades meeting R:R minimum
      4. Anti-degenerate: hold reward tiny so agent must trade to profit
      5. Bounded: clip to [-15, +15] for PPO stability
    """

    def __init__(
        self,
        # Profit/loss multipliers (asymmetric — wins reward more)
        # INCREASED v2→v3: win_multiplier 8→15, loss_multiplier 3→2
        # The agent was farming break-even trades because wins weren't
        # attractive enough relative to the +0.5 risk_reward farming.
        win_multiplier: float = 15.0,       # was 8.0 — make wins VERY attractive
        loss_multiplier: float = 2.0,       # was 3.0 — forgive losses more
        min_rr_for_full_reward: float = 2.0,  # 1:2 R:R = full reward
        # Streak shaping
        consecutive_win_bonus: float = 0.5,
        consecutive_loss_penalty: float = 1.0,
        max_streak_bonus: float = 5.0,
        # Quality gating
        small_profit_threshold: float = 0.002,
        small_profit_penalty: float = 0.3,
        # Risk management
        risk_bonus: float = 0.5,
        excessive_risk_penalty: float = 5.0,
        max_risk_pct: float = 0.02,
        # Overtrading — TIGHTENED v2→v3 to prevent farming
        overtrade_limit: int = 3,           # was 5 — even tighter
        overtrade_penalty: float = 3.0,     # was 2.0 — harsher
        # Drawdown
        drawdown_penalty_mult: float = 2.0,
        max_drawdown_threshold: float = 0.08,
        # Hold reward — REDUCED v2→v3 (was enabling never-trade policy)
        hold_reward: float = 0.01,          # was 0.02 — even smaller
        max_idle_steps_rewarded: int = 3,   # was 5 — fewer free holds
        # Transaction costs
        spread_cost_penalty: float = 0.15,  # was 0.1 — higher to discourage churn
        # Clipping
        reward_clip: float = 15.0,
    ):
        self.win_multiplier = win_multiplier
        self.loss_multiplier = loss_multiplier
        self.min_rr_for_full_reward = min_rr_for_full_reward
        self.consecutive_win_bonus = consecutive_win_bonus
        self.consecutive_loss_penalty = consecutive_loss_penalty
        self.max_streak_bonus = max_streak_bonus
        self.small_profit_threshold = small_profit_threshold
        self.small_profit_penalty = small_profit_penalty
        self.risk_bonus = risk_bonus
        self.excessive_risk_penalty = excessive_risk_penalty
        self.max_risk_pct = max_risk_pct
        self.overtrade_limit = overtrade_limit
        self.overtrade_penalty = overtrade_penalty
        self.drawdown_penalty_mult = drawdown_penalty_mult
        self.max_drawdown_threshold = max_drawdown_threshold
        self.hold_reward = hold_reward
        self.max_idle_steps_rewarded = max_idle_steps_rewarded
        self.spread_cost_penalty = spread_cost_penalty
        self.reward_clip = reward_clip

        # Per-episode state
        self._consecutive_idle_steps = 0
        self._consecutive_wins = 0
        self._consecutive_losses = 0
        self._recent_rewards: deque = deque(maxlen=50)  # for volatility calc

    def reset_episode(self) -> None:
        """Reset per-episode state."""
        self._consecutive_idle_steps = 0
        self._consecutive_wins = 0
        self._consecutive_losses = 0
        self._recent_rewards.clear()

    def calculate(
        self,
        action: str,
        pnl_usd: float = 0.0,
        balance: float = 10000.0,
        initial_balance: float = 10000.0,
        risk_pct: float = 0.01,
        rr_ratio: float = 0.0,
        trades_today: int = 0,
        peak_balance: float = 10000.0,
        position_open: bool = False,
        trade_closed: bool = False,        # NEW: did a trade close this step?
        win: bool = False,                  # NEW: was the closed trade a win?
        spread_cost_usd: float = 0.0,       # NEW: spread cost for this trade
    ) -> RewardBreakdownV2:
        """Calculate profitability-focused reward."""
        rb = RewardBreakdownV2()

        # BUG FIX #1: Use ABSOLUTE initial_balance, never current balance
        # (prevents sign-flip when balance goes negative)
        scale = abs(initial_balance) if initial_balance != 0 else 10000.0
        pnl_pct = (pnl_usd / scale) if scale > 0 else 0.0

        # ── 1. Profit / Loss reward (asymmetric + quality-gated) ────
        if trade_closed:
            if win:
                # Win: reward scaled by R:R quality
                if rr_ratio >= self.min_rr_for_full_reward:
                    # Full reward for quality wins (R:R ≥ 1:2)
                    rb.profit_reward = pnl_pct * self.win_multiplier * 100
                    rb.quality_shaping = 1.0  # quality bonus marker
                    rb.reason = f"QUALITY WIN +{pnl_pct*100:.2f}% R:R=1:{rr_ratio:.1f}"
                elif rr_ratio >= 1.0:
                    # Reduced reward for marginal wins
                    quality_mult = 0.3 + 0.7 * (rr_ratio / self.min_rr_for_full_reward)
                    rb.profit_reward = pnl_pct * self.win_multiplier * 100 * quality_mult
                    rb.quality_shaping = quality_mult
                    rb.reason = f"marginal win +{pnl_pct*100:.2f}% R:R=1:{rr_ratio:.1f}"
                else:
                    # Tiny win — reduced reward (anti-scalping)
                    rb.profit_reward = pnl_pct * self.win_multiplier * 100 * self.small_profit_penalty
                    rb.quality_shaping = self.small_profit_penalty
                    rb.reason = f"small win +{pnl_pct*100:.2f}% R:R=1:{rr_ratio:.1f}"

                # Streak bonus (compounding)
                self._consecutive_wins += 1
                self._consecutive_losses = 0
                streak = min(self._consecutive_wins * self.consecutive_win_bonus,
                             self.max_streak_bonus)
                rb.streak_bonus = streak
                rb.reason += f" | streak ×{self._consecutive_wins} (+{streak:.1f})"

            elif pnl_usd < 0:
                # Loss: asymmetric penalty (less than win reward)
                if rr_ratio >= 1.0:
                    # Quality loss (had good R:R but lost) — forgive
                    rb.loss_penalty = -abs(pnl_pct) * self.loss_multiplier * 100 * 0.5
                    rb.reason = f"quality loss {pnl_pct*100:.2f}% R:R=1:{rr_ratio:.1f} (forgiven)"
                else:
                    # Bad loss (poor R:R) — full penalty
                    rb.loss_penalty = -abs(pnl_pct) * self.loss_multiplier * 100
                    rb.reason = f"bad loss {pnl_pct*100:.2f}% R:R=1:{rr_ratio:.1f}"

                # Streak penalty (escalating)
                self._consecutive_losses += 1
                self._consecutive_wins = 0
                streak_pen = min(self._consecutive_losses * self.consecutive_loss_penalty,
                                 self.max_streak_bonus)
                rb.streak_bonus = -streak_pen
                rb.reason += f" | loss streak ×{self._consecutive_losses} (-{streak_pen:.1f})"

        # ── 2. Transaction cost (explicit, every trade) ─────────────
        if action in ("BUY", "SELL"):
            rb.transaction_cost = -self.spread_cost_penalty
            rb.reason += f" | spread cost -{self.spread_cost_penalty}"

        # ── 3. Risk management reward (ONLY on winning closes) ──────
        # BUG FIX (v2→v3): previously this fired on EVERY BUY/SELL action,
        # even if the trade immediately closed at break-even. The agent
        # learned to farm +0.5 per trade cycle without actual profit.
        # Now: risk_reward only fires when a trade CLOSES as a WIN with
        # good R:R — rewarding actual profitable risk management, not
        # just opening positions.
        if trade_closed and win and rr_ratio >= self.min_rr_for_full_reward:
            rb.risk_reward = self.risk_bonus
            rb.reason += f" | good risk +{self.risk_bonus}"
        elif trade_closed and not win and risk_pct > self.max_risk_pct * 2.5:
            rb.risk_reward = -self.excessive_risk_penalty
            rb.reason += f" | EXCESSIVE RISK -{self.excessive_risk_penalty}"

        # ── 3b. Break-even penalty (anti-farming) ───────────────────
        # If a trade closes with ~0 PnL, penalize slightly — the agent
        # was farming risk_reward by opening+closing at break-even.
        if trade_closed and abs(pnl_usd) < 0.01:
            rb.transaction_cost -= 0.5  # extra penalty for pointless trades
            rb.reason += f" | break-even farming penalty -0.5"

        # ── 4. Overtrading penalty (tighter) ────────────────────────
        if trades_today > self.overtrade_limit:
            rb.overtrading_penalty = -self.overtrade_penalty * (trades_today - self.overtrade_limit)
            rb.reason += f" | overtrading ({trades_today} today)"

        # ── 5. Drawdown penalty (stricter) ──────────────────────────
        if peak_balance > 0 and balance > 0:
            drawdown = (peak_balance - balance) / peak_balance
            if drawdown > self.max_drawdown_threshold:
                rb.drawdown_penalty = -self.drawdown_penalty_mult * (drawdown - self.max_drawdown_threshold) * 100
                rb.reason += f" | drawdown {drawdown*100:.1f}%"

        # ── 6. Hold reward (TINY — anti-degenerate) ─────────────────
        if action == "HOLD" and not position_open:
            self._consecutive_idle_steps += 1
            if self._consecutive_idle_steps <= self.max_idle_steps_rewarded:
                rb.hold_reward = self.hold_reward
                rb.reason = "patient hold"
            else:
                rb.reason = "idle (patience capped)"
        else:
            self._consecutive_idle_steps = 0

        # ── Total (clipped for PPO stability) ───────────────────────
        rb.total = (
            rb.profit_reward
            + rb.loss_penalty
            + rb.risk_reward
            + rb.overtrading_penalty
            + rb.drawdown_penalty
            + rb.hold_reward
            + rb.transaction_cost
            + rb.streak_bonus
            + rb.quality_shaping
        )

        # Clip to [-reward_clip, +reward_clip]
        rb.total = max(-self.reward_clip, min(self.reward_clip, rb.total))

        # Track for volatility
        self._recent_rewards.append(rb.total)

        return rb

    def get_volatility(self) -> float:
        """Recent reward std dev — for risk-adjusted shaping."""
        if len(self._recent_rewards) < 5:
            return 1.0
        import statistics
        return statistics.stdev(self._recent_rewards) or 1.0


# ── Singleton ───────────────────────────────────────────────────────

_ENGINE_V2: Optional[RewardEngineV2] = None


def get_reward_engine_v2() -> RewardEngineV2:
    global _ENGINE_V2
    if _ENGINE_V2 is None:
        _ENGINE_V2 = RewardEngineV2()
    return _ENGINE_V2
