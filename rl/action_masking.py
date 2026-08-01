# rl/action_masking.py — Action masking for RL trading environments
# =============================================================================
# Ported from: https://github.com/Seanman519/RLBOT/blob/master/rlbot/gym_env/mask.py
# Original author: Seanman519 — MIT license
#
# Action masking prevents an RL agent from taking invalid actions. In a trading
# environment, this means:
#   - Can't open a new position if one is already open (or at max size)
#   - Can't close a position that doesn't exist
#   - Can't exceed max position size
#   - Can force "hold" during certain conditions (news, session close, etc.)
#
# The action map translates discrete RL actions (integers) to trading operations:
#   Action 0: HOLD (always available)
#   Action 1: Open SHORT lot 1
#   Action 2: Close SHORT lot 1
#   Action 3: Open LONG lot 1
#   Action 4: Close LONG lot 1
#   ... (more for multi-lot)
#
# The mask is a binary array: 1 = action allowed, 0 = action blocked.
# =============================================================================

from __future__ import annotations

import numpy as np
from typing import Optional

from utils.logger import get_logger

log = get_logger("action_masking")


def build_action_map(
    max_short: int = 1,
    max_long: int = 1,
    num_symbols: int = 1,
) -> np.ndarray:
    """
    Build the action map: translates discrete RL actions to trading operations.

    Each row = one action. Columns:
        [0] portfolio_index (which symbol)
        [1] direction: -1 (short), 1 (long), 0 (hold)
        [2] operation: 1 (open), -1 (close), 0 (hold)
        [3] lot_size: 1, 2, ... (how many lots)

    Action 0 is always HOLD.

    Parameters
    ----------
    max_short : max short lots per symbol.
    max_long : max long lots per symbol.
    num_symbols : number of tradeable symbols.

    Returns
    -------
    np.ndarray of shape (num_actions, 4).
    """
    action_map = [[-1, 0, 0, 0]]  # HOLD (always index 0)

    for sym_idx in range(num_symbols):
        # Short actions
        for lot in range(max_short):
            action_map.append([sym_idx, -1, 1, lot + 1])  # open short
            action_map.append([sym_idx, -1, -1, lot + 1])  # close short
        # Long actions
        for lot in range(max_long):
            action_map.append([sym_idx, 1, 1, lot + 1])   # open long
            action_map.append([sym_idx, 1, -1, lot + 1])   # close long

    return np.array(action_map, dtype=int)


def build_portfolio(num_symbols: int = 1, max_short: int = 1, max_long: int = 1) -> np.ndarray:
    """
    Build the portfolio array (gym representation of open positions).

    Columns:
        [0] portfolio_index
        [1] symbol_index
        [2] pip_value (placeholder)
        [3] max_short
        [4] max_long
        [5] pos_size (current, 0 = flat)
        [6] pos_dir (0=flat, -1=short, 1=long)
    """
    portfolio = np.zeros((num_symbols, 7), dtype=float)
    for i in range(num_symbols):
        portfolio[i, 0] = i        # portfolio index
        portfolio[i, 1] = i        # symbol index
        portfolio[i, 3] = max_short
        portfolio[i, 4] = max_long
    return portfolio


def make_action_mask(
    action_map: np.ndarray,
    portfolio: np.ndarray,
    *,
    must_hold: bool = False,
) -> np.ndarray:
    """
    Compute the action mask: which actions are currently allowed?

    Parameters
    ----------
    action_map : (num_actions, 4) array from build_action_map().
    portfolio : (num_symbols, 7) array from build_portfolio() (updated with current positions).
    must_hold : if True, only HOLD (action 0) is allowed. Used during:
        - News blackout periods
        - Session close
        - Weekends
        - First minute after opening a position (min hold time)

    Returns
    -------
    Binary mask: 1 = allowed, 0 = blocked. Same length as action_map.
    """
    num_actions = len(action_map)
    mask = np.zeros(num_actions, dtype=np.float32)

    # HOLD is always available
    mask[0] = 1.0

    if must_hold:
        return mask

    for port_idx in range(len(portfolio)):
        pos_size = int(portfolio[port_idx, 5])
        pos_dir = int(portfolio[port_idx, 6])
        max_short = int(portfolio[port_idx, 3])
        max_long = int(portfolio[port_idx, 4])

        # Actions for this symbol
        sym_actions = action_map[:, 0] == port_idx

        # OPEN actions
        open_mask = sym_actions & (action_map[:, 2] == 1)
        if pos_size == 0:
            # Can open in either direction
            # Short
            short_open = open_mask & (action_map[:, 1] == -1)
            valid_short = short_open & (action_map[:, 3] <= max_short)
            mask[valid_short] = 1.0
            # Long
            long_open = open_mask & (action_map[:, 1] == 1)
            valid_long = long_open & (action_map[:, 3] <= max_long)
            mask[valid_long] = 1.0
        # If position already open, don't allow opening more (simplified)

        # CLOSE actions
        close_mask = sym_actions & (action_map[:, 2] == -1)
        if pos_size > 0:
            # Can only close in the direction of the open position
            close_valid = close_mask & (action_map[:, 1] == pos_dir)
            close_valid = close_valid & (action_map[:, 3] <= pos_size)
            mask[close_valid] = 1.0

    return mask


def apply_mask(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply mask to RL agent logits: set blocked actions to -inf before softmax.

    Parameters
    ----------
    logits : raw action logits from the RL agent (shape: num_actions).
    mask : binary mask (1=allowed, 0=blocked).

    Returns
    -------
    Masked logits (blocked actions → -inf).
    """
    masked = logits.copy()
    masked[mask == 0] = -np.inf
    return masked


def get_valid_actions(action_map: np.ndarray, mask: np.ndarray) -> list[str]:
    """
    Return human-readable list of currently valid actions.
    """
    labels = []
    for i, (valid, action) in enumerate(zip(mask, action_map)):
        if valid:
            port_idx, direction, operation, lot = action
            if i == 0:
                labels.append("HOLD")
            else:
                dir_str = "LONG" if direction == 1 else "SHORT"
                op_str = "OPEN" if operation == 1 else "CLOSE"
                labels.append(f"{op_str} {dir_str} lot={lot} sym={port_idx}")
    return labels


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Build action map for 1 symbol, max 2 short + 2 long
    amap = build_action_map(max_short=2, max_long=2, num_symbols=1)
    print(f"Action map ({len(amap)} actions):")
    for i, a in enumerate(amap):
        port, d, op, lot = a
        if i == 0:
            print(f"  [{i}] HOLD")
        else:
            dir_s = "LONG" if d == 1 else "SHORT"
            op_s = "OPEN" if op == 1 else "CLOSE"
            print(f"  [{i}] {op_s} {dir_s} lot={lot}")
    assert len(amap) == 9  # 1 hold + 2*2 short + 2*2 long

    # Empty portfolio → can open, can't close
    port = build_portfolio(num_symbols=1, max_short=2, max_long=2)
    mask = make_action_mask(amap, port)
    valid = get_valid_actions(amap, mask)
    print(f"\nEmpty portfolio — valid actions: {valid}")
    assert "HOLD" in valid
    assert any("OPEN" in v for v in valid)
    assert not any("CLOSE" in v for v in valid)

    # Open a SHORT position → can close, can't open
    port[0, 5] = 1  # pos_size = 1
    port[0, 6] = -1  # pos_dir = short
    mask = make_action_mask(amap, port)
    valid = get_valid_actions(amap, mask)
    print(f"\nShort position open — valid actions: {valid}")
    assert any("CLOSE SHORT" in v for v in valid)
    assert not any("OPEN" in v for v in valid)

    # must_hold → only HOLD
    mask = make_action_mask(amap, port, must_hold=True)
    valid = get_valid_actions(amap, mask)
    print(f"\nMust hold — valid actions: {valid}")
    assert valid == ["HOLD"]

    # Apply mask to logits
    logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    masked = apply_mask(logits, mask)
    print(f"\nMasked logits: {masked}")
    assert masked[0] == 1.0  # HOLD allowed
    assert np.isinf(masked[1])  # everything else blocked

    print("\nAction masking smoke test passed.")
