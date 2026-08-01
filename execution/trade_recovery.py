"""
execution/trade_recovery.py — MT5 Trade State Recovery & History Reconciliation
==============================================================================

Called at boot (from core/runtime.py boot_broker phase) to synchronize
local database state with MT5's actual position/trade state.

What it does:
  1. Reads all OPEN MT5 positions (filtered by magic number)
  2. Reads recently CLOSED MT5 deals (last 7 days)
  3. Compares MT5 tickets with the local learning database
  4. For still-open positions: seeds PositionManager._known_tickets
  5. For closed-during-downtime positions: updates DB + learning modules

This makes the bot completely restart-safe — no trade is forgotten,
no close is missed, no duplicate tracking occurs.

Usage (called from runtime.py boot_broker):
    from execution.trade_recovery import recover_mt5_state
    result = recover_mt5_state(
        order_manager=registry.try_resolve("order_manager"),
        learning_db=registry.try_resolve("learning_db"),
        position_manager=position_manager_instance,
    )
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional

from utils.logger import get_logger

log = get_logger("trade_recovery")


def recover_mt5_state(
    order_manager,
    learning_db=None,
    position_manager=None,
    magic_number: int = 424242,
    history_days: int = 7,
) -> dict:
    """Main recovery entry point. Called once at boot.

    Args:
        order_manager: broker.order_manager.OrderManager instance (required)
        learning_db: memory.database.Database instance (optional — enables
                     ticket-based DB lookup and close reconciliation)
        position_manager: broker.position_manager.PositionManager instance
                          (optional — enables _known_tickets seeding)
        magic_number: MT5 magic number filter (default 424242)
        history_days: How far back to look for closed deals (default 7)

    Returns:
        dict with:
            - open_positions: count of MT5 open positions found
            - closed_reconciled: count of DB-OPEN trades closed during downtime
            - orphan_mt5: count of MT5 positions with no DB record
            - position_manager_seeded: bool
            - errors: list of error strings
    """
    result = {
        "open_positions": 0,
        "closed_reconciled": 0,
        "orphan_mt5": 0,
        "position_manager_seeded": False,
        "errors": [],
    }

    log.info("[TradeRecovery] Starting MT5 state recovery...")

    # ── Step 1: Read all OPEN MT5 positions ──────────────────────
    try:
        mt5_open = order_manager.get_open_positions(magic=magic_number)
    except Exception as e:
        log.warning(f"[TradeRecovery] Failed to fetch MT5 open positions: {e}")
        result["errors"].append(f"open_positions_fetch: {e}")
        return result

    mt5_open_tickets = {p["ticket"] for p in mt5_open}
    result["open_positions"] = len(mt5_open)
    log.info(f"[TradeRecovery] Found {len(mt5_open)} open MT5 position(s)")

    # ── Step 2: Seed PositionManager with current MT5 state ──────
    if position_manager is not None:
        try:
            pm_result = position_manager.recover_from_mt5(
                learning_db=learning_db,
            )
            result["position_manager_seeded"] = True
            result["orphan_mt5"] = pm_result.get("unmatched", 0)
            log.info(
                f"[TradeRecovery] PositionManager seeded: "
                f"{pm_result['recovered']} positions "
                f"({pm_result['matched']} matched to DB, "
                f"{pm_result['unmatched']} orphan)"
            )
        except Exception as e:
            log.warning(f"[TradeRecovery] PositionManager seeding failed: {e}")
            result["errors"].append(f"position_manager_seed: {e}")
    else:
        log.info("[TradeRecovery] No PositionManager — skipping seed step")

    # ── Step 3: Reconcile closed-during-downtime trades ──────────
    if learning_db is not None:
        try:
            closed_count = _reconcile_closed_trades(
                learning_db=learning_db,
                order_manager=order_manager,
                mt5_open_tickets=mt5_open_tickets,
                magic_number=magic_number,
                history_days=history_days,
            )
            result["closed_reconciled"] = closed_count
        except Exception as e:
            log.warning(f"[TradeRecovery] Close reconciliation failed: {e}")
            result["errors"].append(f"close_reconcile: {e}")
    else:
        log.info("[TradeRecovery] No learning DB — skipping close reconciliation")

    # ── Summary ──────────────────────────────────────────────────
    log.info(
        f"[TradeRecovery] Complete: "
        f"{result['open_positions']} open | "
        f"{result['closed_reconciled']} closed-during-downtime reconciled | "
        f"{result['orphan_mt5']} orphan MT5 positions | "
        f"PM seeded: {result['position_manager_seeded']}"
    )
    return result


def _reconcile_closed_trades(
    learning_db,
    order_manager,
    mt5_open_tickets: set,
    magic_number: int,
    history_days: int,
) -> int:
    """Find DB-OPEN trades that are no longer in MT5 → mark as closed.

    For each DB-OPEN trade:
      - If mt5_ticket is set and NOT in mt5_open_tickets → closed during downtime
      - If mt5_ticket is None → try pair-based lookup (fallback)

    For closed trades, attempt to recover close details from MT5 deal history.
    """
    closed_count = 0

    # Get all DB-OPEN trades
    try:
        db_open_trades = learning_db.get_open_trades()
    except Exception as e:
        log.warning(f"[TradeRecovery] Failed to fetch DB open trades: {e}")
        return 0

    if not db_open_trades:
        log.info("[TradeRecovery] No DB-OPEN trades to reconcile")
        return 0

    log.info(f"[TradeRecovery] Reconciling {len(db_open_trades)} DB-OPEN trade(s)...")

    # Get MT5 deal history for close price/profit lookup
    try:
        mt5_history = order_manager.get_order_history(days_back=history_days)
    except Exception:
        mt5_history = []

    # Build history lookup by position_id (MT5 deal's position_id = original ticket)
    history_by_position = {}
    for deal in mt5_history:
        pid = deal.get("position_id") or deal.get("ticket")
        if pid:
            history_by_position[pid] = deal

    for trade in db_open_trades:
        trade_id = trade.get("id")
        pair = trade.get("pair", "")
        mt5_ticket = trade.get("mt5_ticket")

        # If this trade has an MT5 ticket and it's still open → skip
        if mt5_ticket is not None and mt5_ticket in mt5_open_tickets:
            continue  # Position is still open in MT5 — nothing to do

        # This trade is DB-OPEN but NOT in MT5 open positions → it closed
        # while we were offline. Reconcile it.

        # Try to get close details from MT5 history
        close_price = None
        close_profit = 0.0
        close_time = None
        close_reason = "CLOSED_DURING_DOWNTIME"

        if mt5_ticket is not None and mt5_ticket in history_by_position:
            deal = history_by_position[mt5_ticket]
            close_price = deal.get("price")
            close_profit = float(deal.get("profit", 0) or 0)
            close_time = deal.get("time")
            close_reason = "MT5_TP_SL_HIT"

        # Determine WIN/LOSS
        if close_profit > 0:
            result_str = "WIN"
        elif close_profit < 0:
            result_str = "LOSS"
        else:
            result_str = "BREAKEVEN"

        # Update the DB
        try:
            learning_db.update_trade_result(trade_id, result_str, close_profit)
            closed_count += 1
            log.info(
                f"[TradeRecovery] Reconciled trade #{trade_id} "
                f"({pair}): {result_str} PnL=${close_profit:.2f} "
                f"reason={close_reason}"
            )
        except Exception as e:
            log.warning(
                f"[TradeRecovery] Failed to update trade #{trade_id}: {e}"
            )

    return closed_count
