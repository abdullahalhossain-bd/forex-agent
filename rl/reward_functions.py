# rl/reward_functions.py — Trading reward functions for RL
# =============================================================================
# Ported from: https://github.com/zeroxt32/Forex-Expert-Advisor-Python/blob/master/T32_v5.py
# Original: ForexCustomEnv.softplus_reward() and get_reward()
# Original author: zeroxt32 — MIT license
#
# Reward functions for reinforcement learning trading agents. The key
# innovation is the SOFTPLUS REWARD which:
#
#   1. ASYMMETRIC SCALING — positive profit is scaled down (reward/10),
#      negative profit is scaled up (reward*10). Losses hurt more than wins help.
#   2. TIME DECAY — log(1 + exp(-time_step/40)) decreases as trade stays open.
#   3. THRESHOLD ADAPTATION — compares current profit against best-so-far.
#   4. EARLY TRADE BONUS — young profitable trades get boosted.
# =============================================================================

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from utils.logger import get_logger

log = get_logger("reward_functions")


def softplus_reward(
    profit: float,
    time_step: int,
    max_profit: float = 0.1,
    position: str = "close",
    threshold: float = 0.8,
) -> float:
    """Softplus reward (ported from T32_v5.py ForexCustomEnv.softplus_reward)."""
    loss = -min(profit, 0)
    threshold = 0.75 * max_profit if profit > 0 else 0.8

    if time_step <= 3 and (0 <= profit <= 10 and position in ["sell", "buy"]):
        time_step = 3
        profit = 4

    reward = (
        np.log(1 + np.exp((profit - loss - threshold) / 2))
        - np.log(2)
        + np.log(1 + np.exp(-time_step / 40))
    )
    return reward / 10 if profit > 0 else reward * 10


def simple_pnl_reward(profit: float, **kwargs) -> float:
    """Simplest reward: raw profit/loss."""
    return float(profit)


def asymmetric_reward(
    profit: float,
    win_multiplier: float = 1.0,
    loss_multiplier: float = 2.0,
    **kwargs,
) -> float:
    """Wins × win_mult, losses × loss_mult (risk-averse by default)."""
    if profit > 0:
        return profit * win_multiplier
    return profit * loss_multiplier


def sharpe_reward(profits: list[float], risk_free_rate: float = 0.0, **kwargs) -> float:
    """Sharpe-ratio-style reward from a list of recent profits."""
    if len(profits) < 2:
        return 0.0
    arr = np.array(profits)
    std = arr.std()
    if std == 0:
        return 0.0
    return float((arr.mean() - risk_free_rate) / std)


@dataclass
class TradeState:
    profit: float
    time_step: int
    max_profit: float = 0.0
    position: str = "close"
    entry_price: float = 0.0
    current_price: float = 0.0
    balance: float = 1000.0
    episode_profit: float = 0.0


def get_reward(trade: TradeState, func=softplus_reward, **kwargs) -> float:
    """Compute reward for a trade state (mirrors original get_reward)."""
    return func(
        profit=trade.profit, time_step=trade.time_step,
        max_profit=trade.max_profit, position=trade.position, **kwargs,
    )


if __name__ == "__main__":
    r_win = softplus_reward(profit=5.0, time_step=2, max_profit=5.0, position="buy")
    r_loss = softplus_reward(profit=-5.0, time_step=2, max_profit=0.0, position="buy")
    r_hold = softplus_reward(profit=0.0, time_step=10, max_profit=0.0, position="close")
    print(f"Win (+5):   {r_win:.4f}")
    print(f"Loss (-5):  {r_loss:.4f}")
    print(f"Hold (0):   {r_hold:.4f}")
    assert r_win > 0 and r_loss < 0
    assert abs(r_loss) > r_win, "loss should hurt more than win helps"

    r_short = softplus_reward(profit=5.0, time_step=2, max_profit=5.0, position="buy")
    r_long = softplus_reward(profit=5.0, time_step=50, max_profit=5.0, position="buy")
    print(f"Short hold: {r_short:.4f}, Long hold: {r_long:.4f}")
    assert r_short > r_long

    assert simple_pnl_reward(5.0) == 5.0
    assert asymmetric_reward(-5.0) == -10.0
    assert sharpe_reward([1, 2, 3]) > 0
    assert sharpe_reward([1]) == 0.0

    trade = TradeState(profit=3.0, time_step=5, max_profit=3.0, position="buy")
    assert get_reward(trade) > 0

    print("\nReward functions smoke test passed.")
