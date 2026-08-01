from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logger import get_logger
from core.constants import MEMORY_DIR, DATABASE_DIR

log = get_logger("orphan_cleanup")


def _mt5_positions_get(retries: int = 2, delay: float = 0.3, **kwargs):
    """Call mt5.positions_get() with retry logic.

    MT5 can return None intermittently. This helper retries
    a few times before giving up, reducing false negatives.
    """
    import MetaTrader5 as mt5
    import time
    for attempt in range(retries + 1):
        try:
            result = mt5.positions_get(**kwargs) if kwargs else mt5.positions_get()
            if result is not None:
                return result
        except Exception:
            pass
        if attempt < retries:
            time.sleep(delay)
    return None


def reconcile_open_positions(
    db,
    mt5_conn=None,
    paper_trader=None,
) -> Dict[str, Any]:
    """Reconcile DB trades with MT5 live positions.

    Args:
        db: TraderDB instance (or None to use default)
        mt5_conn: MT5Connection instance (or None to skip MT5 check)
        paper_trader: PaperTrader instance (or None to skip paper check)

    Returns:
        {'closed': N, 'kept': M, 'errors': E, 'mt5_tickets': [...]}
    """
    result = {"closed": 0, "kept": 0, "errors": 0, "mt5_tickets": []}
    log.info("[OrphanCleanup] Starting reconciliation...")

    # ── Step 1: get live MT5 open positions (tickets) ──
    mt5_tickets: set[int] = set()
    live_pairs_from_mt5: set[str] = set()
    
    if mt5_conn is not None:
        try:
            import MetaTrader5 as mt5
            positions = _mt5_positions_get()
            if positions:
                for p in positions:
                    ticket_id = int(p.ticket)
                    mt5_tickets.add(ticket_id)
                    result["mt5_tickets"].append(ticket_id)
                    if hasattr(p, 'symbol') and p.symbol:
                        live_pairs_from_mt5.add(str(p.symbol))
            log.info(f"[OrphanCleanup] MT5 live positions: {len(mt5_tickets)}")
        except Exception as e:
            log.warning(f"[OrphanCleanup] MT5 positions_get failed: {e}")
    else:
        log.info("[OrphanCleanup] No MT5 connection — will reconcile against paper trader only")

    # ── Step 2: get DB open trades ──
    db_path = DATABASE_DIR / "trader.db"
    if not db_path.exists():
        log.info("[OrphanCleanup] No DB file — nothing to reconcile")
        return result

    try:
        conn = sqlite3.connect(str(db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("SELECT id, pair, type, lot, entry, open_time, mt5_ticket FROM trades WHERE status = 'OPEN'")
        open_rows = cur.fetchall()
        log.info(f"[OrphanCleanup] DB open trades: {len(open_rows)}")

        if not open_rows:
            conn.close()
            # If no open trades in DB, pass current live MT5 pairs to avoid complete wiping if trades exist in MT5
            _clear_stale_open_pairs(live_pairs=list(live_pairs_from_mt5) if mt5_conn is not None else None)
            return result

        # Remaining active pairs after cleanup to update daily_risk.json
        currently_active_pairs = set()

        # ── Step 3: for each DB-OPEN trade, check if it's really open ──
        for row in open_rows:
            trade_id = row["id"]
            pair = row["pair"]
            trade_type = row["type"]

            is_orphan = False
            if mt5_conn is not None:
                if not mt5_tickets:
                    # MT5 has zero open positions → all DB-OPEN trades are orphans
                    is_orphan = True
                else:
                    # BUGFIX: sqlite3.Row does not have .keys() method. Use safe dynamic lookup.
                    has_ticket_col = "mt5_ticket" in row.keys() if hasattr(row, "keys") else True
                    db_ticket = row["mt5_ticket"] if has_ticket_col else None
                    
                    if db_ticket and int(db_ticket) in mt5_tickets:
                        # Exact ticket match — definitely not an orphan
                        is_orphan = False
                    else:
                        # Fall back to pair+type matching (less reliable)
                        try:
                            matching = _mt5_positions_get(symbol=pair)
                            if not matching:
                                is_orphan = True
                                log.info(
                                    f"[OrphanCleanup] Trade #{trade_id} {pair} {trade_type}: "
                                    f"no matching MT5 position → orphan"
                                )
                            elif len(matching) > 1:
                                log.warning(
                                    f"[OrphanCleanup] Trade #{trade_id} {pair} {trade_type}: "
                                    f"{len(matching)} MT5 positions found — "
                                    f"pair-based matching ambiguous, skipping "
                                    f"(add mt5_ticket column for reliable matching)"
                                )
                        except Exception as e:
                            log.warning(f"[OrphanCleanup] positions_get({pair}) failed: {e}")
            else:
                # Paper trader fallback
                if paper_trader is not None:
                    try:
                        has_open = paper_trader.has_open_position(pair, trade_type)
                        if not has_open:
                            is_orphan = True
                            log.info(
                                f"[OrphanCleanup] Trade #{trade_id} {pair} {trade_type}: "
                                f"paper trader has no open position → orphan"
                            )
                    except Exception as e:
                        log.warning(f"[OrphanCleanup] paper_trader check failed: {e}")

            if is_orphan:
                try:
                    context = {"close_reason": "auto_orphan_cleanup"}
                    cur.execute(
                        "UPDATE trades SET status = 'CLOSED', "
                        "close_time = ?, result = 'AUTO_CLOSED', "
                        "context_json = ? "
                        "WHERE id = ?",
                        (datetime.now(timezone.utc).isoformat(),
                         json.dumps(context), trade_id),
                    )
                    result["closed"] += 1
                    log.warning(f"[OrphanCleanup] Closed orphan trade #{trade_id} {pair} {trade_type}")
                except Exception as e:
                    result["errors"] += 1
                    log.error(f"[OrphanCleanup] Failed to close trade #{trade_id}: {e}")
            else:
                result["kept"] += 1
                currently_active_pairs.add(pair)
                log.info(f"[OrphanCleanup] ✓ Kept trade #{trade_id} {pair} (still open)")

        conn.commit()
        conn.close()
        
        # ── Step 4: Sync authoritative remaining pairs to daily_risk.json ──
        # BUGFIX: Instead of wiping out everything, we pass the verified active open pairs.
        _clear_stale_open_pairs(live_pairs=list(currently_active_pairs))

    except Exception as e:
        result["errors"] += 1
        log.error(f"[OrphanCleanup] DB error: {e}", exc_info=True)

    log.info(
        f"[OrphanCleanup] Done — closed={result['closed']}, "
        f"kept={result['kept']}, errors={result['errors']}"
    )
    return result


def _clear_stale_open_pairs(live_pairs: Optional[List[str]] = None) -> None:
    """Safely updates or clears the open_pairs list in daily_risk.json.
    BUGFIX: Wiping out this file completely blocks RiskEngine correlation checks. 
    Now it dynamically retains verified active pairs or resets gracefully if explicitly empty.
    """
    dr_path = MEMORY_DIR / "daily_risk.json"
    if not dr_path.exists():
        return
    try:
        data = json.loads(dr_path.read_text())
        
        # If valid live pairs list is provided, synchronize it cleanly
        target_pairs = sorted(list(set(live_pairs))) if live_pairs is not None else []
        
        data["open_pairs"] = target_pairs
        data["open_trades"] = len(target_pairs)
        
        dr_path.write_text(json.dumps(data, indent=2))
        log.info(f"[OrphanCleanup] Synced daily_risk.json state: open_pairs={target_pairs}")
        
    except Exception as e:
        log.warning(f"[OrphanCleanup] daily_risk.json sync/clear failed: {e}")


def quick_close_all_db_open() -> int:
    """One-shot utility: mark ALL DB-OPEN trades as CLOSED."""
    db_path = DATABASE_DIR / "trader.db"
    if not db_path.exists():
        return 0
    try:
        conn = sqlite3.connect(str(db_path), timeout=5.0)
        cur = conn.cursor()
        context = json.dumps({"close_reason": "manual_cleanup"})
        cur.execute(
            "UPDATE trades SET status = 'CLOSED', "
            "close_time = ?, result = 'AUTO_CLOSED', "
            "context_json = ? "
            "WHERE status = 'OPEN'",
            (datetime.now(timezone.utc).isoformat(), context),
        )
        closed = cur.rowcount
        conn.commit()
        conn.close()
        log.info(f"[OrphanCleanup] quick_close_all_db_open: {closed} trades closed")
        _clear_stale_open_pairs(live_pairs=[])
        return closed
    except Exception as e:
        log.error(f"[OrphanCleanup] quick_close_all_db_open failed: {e}")
        return 0