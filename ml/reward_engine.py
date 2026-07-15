"""
ml/reward_engine.py — RL Reward System (Day 71)
==================================================

The most critical component of RL training. Bad reward design = bad policy.

Reward components:
  1. **Profit reward**      — +profit_pct × multiplier (winning trades)
  2. **Loss penalty**       — -loss_pct × multiplier (losing trades)
  3. **Risk management**    — +bonus for 1% risk + good R:R, -penalty for high risk
  4. **Overtrading penalty** — -penalty if trades_per_day > limit
  5. **Drawdown penalty**   — -penalty proportional to account drawdown
  6. **Hold reward**        — small + for correctly waiting (avoiding bad trades)
  7. **Reward hacking protection** — diminishing returns on small profits

All rewards are normalized to roughly [-20, +20] range per step to keep
PPO training stable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

from utils.logger import get_logger

log = get_logger("reward_engine")


@dataclass
class RewardBreakdown:
    """Detailed reward breakdown for one step."""
    profit_reward: float = 0.0
    loss_penalty: float = 0.0
    risk_reward: float = 0.0
    overtrading_penalty: float = 0.0
    drawdown_penalty: float = 0.0
    hold_reward: float = 0.0
    hacking_penalty: float = 0.0
    total: float = 0.0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RewardEngine:
    """Calculates RL rewards with anti-hacking protections."""

    def __init__(
        self,
        profit_multiplier: float = 5.0,
        loss_multiplier: float = 5.0,
        risk_bonus: float = 1.0,
        overtrade_limit: int = 10,
        overtrade_penalty: float = 3.0,
        drawdown_penalty_mult: float = 1.5,
        max_drawdown_threshold: float = 0.10,
        hold_reward: float = 0.1,
        small_profit_threshold: float = 0.001,  # 0.1% — below this, hacking penalty
    ):
        self.profit_multiplier = profit_multiplier
        self.loss_multiplier = loss_multiplier
        self.risk_bonus = risk_bonus
        self.overtrade_limit = overtrade_limit
        self.overtrade_penalty = overtrade_penalty
        self.drawdown_penalty_mult = drawdown_penalty_mult
        self.max_drawdown_threshold = max_drawdown_threshold
        self.hold_reward = hold_reward
        self.small_profit_threshold = small_profit_threshold

    def calculate(
        self,
        action: str,                    # BUY / SELL / HOLD / CLOSE
        pnl_usd: float = 0.0,           # profit/loss for this step
        balance: float = 10000.0,       # current account balance
        initial_balance: float = 10000.0,
        risk_pct: float = 0.01,         # risk % used for this trade
        rr_ratio: float = 0.0,          # reward:risk ratio
        trades_today: int = 0,          # trades taken today
        peak_balance: float = 10000.0,  # highest balance ever
        position_open: bool = False,    # is a position currently open?
    ) -> RewardBreakdown:
        """Calculate the reward for one step."""
        rb = RewardBreakdown()

        pnl_pct = (pnl_usd / balance) if balance > 0 else 0.0

        # ── 1. Profit / Loss reward ────────────────────────────────
        if pnl_usd > 0:
            # Profit reward with diminishing returns (anti-hacking)
            if pnl_pct < self.small_profit_threshold:
                # Very small profit — reduced reward (prevent scalping exploitation)
                rb.profit_reward = pnl_pct * self.profit_multiplier * 0.3
                rb.hacking_penalty = -0.5  # small penalty for tiny trades
                rb.reason = "small profit — reduced reward (anti-hacking)"
            else:
                rb.profit_reward = pnl_pct * self.profit_multiplier * 100  # scale to ~1-10 range
                rb.reason = f"profit +{pnl_pct*100:.2f}%"
        elif pnl_usd < 0:
            rb.loss_penalty = -abs(pnl_pct) * self.loss_multiplier * 100
            rb.reason = f"loss {pnl_pct*100:.2f}%"

        # ── 2. Risk management reward ──────────────────────────────
        if action in ("BUY", "SELL"):
            if risk_pct <= 0.02 and rr_ratio >= 1.5:
                rb.risk_reward = self.risk_bonus  # +1 for good risk management
            elif risk_pct > 0.05:
                rb.risk_reward = -3.0  # -3 for excessive risk
                rb.reason += " | excessive risk penalty"

        # ── 3. Overtrading penalty ─────────────────────────────────
        if trades_today > self.overtrade_limit:
            rb.overtrading_penalty = -self.overtrade_penalty * (trades_today - self.overtrade_limit)
            rb.reason += f" | overtrading ({trades_today} today)"

        # ── 4. Drawdown penalty ────────────────────────────────────
        if peak_balance > 0:
            drawdown = (peak_balance - balance) / peak_balance
            if drawdown > self.max_drawdown_threshold:
                rb.drawdown_penalty = -self.drawdown_penalty_mult * (drawdown - self.max_drawdown_threshold) * 100
                rb.reason += f" | drawdown {drawdown*100:.1f}%"

        # ── 5. Hold reward (correct patience) ──────────────────────
        if action == "HOLD" and not position_open:
            rb.hold_reward = self.hold_reward  # small reward for patience
            rb.reason = "patient hold"

        # ── Total ──────────────────────────────────────────────────
        rb.total = (
            rb.profit_reward
            + rb.loss_penalty
            + rb.risk_reward
            + rb.overtrading_penalty
            + rb.drawdown_penalty
            + rb.hold_reward
            + rb.hacking_penalty
        )

        # Clip to [-20, 20] for training stability
        rb.total = max(-20.0, min(20.0, rb.total))

        return rb


# ── Singleton ───────────────────────────────────────────────────────

_ENGINE: Optional[RewardEngine] = None


def get_reward_engine() -> RewardEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = RewardEngine()
    return _ENGINE
