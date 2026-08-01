"""execution/emergency_exit.py — Emergency exit manager

Round-22 audit fix: NOW WIRED into core/trader.py as a panic button.
Previously this module was completely dead (0 importers) and only
supported paper trades — MT5 positions couldn't be closed in an
emergency. Now supports both paper and MT5 execution modes.

Usage:
    from execution.emergency_exit import get_emergency_exit_manager
    mgr = get_emergency_exit_manager()
    result = mgr.close_all_positions(
        reason="News shock detected",
        paper_trader=self._paper,
        mt5_connection=self._mt5_conn,
        notifier=self.notifier,
    )
"""
from __future__ import annotations
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional
from utils.logger import get_logger
log = get_logger("emergency_exit")

@dataclass
class EmergencyExitResult:
    triggered: bool = False; reason: str = ""; positions_closed: int = 0
    positions_failed: int = 0; total_pnl: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    def to_dict(self): return {"triggered":self.triggered,"reason":self.reason,"positions_closed":self.positions_closed,"positions_failed":self.positions_failed,"total_pnl":round(self.total_pnl,2),"timestamp":self.timestamp}

class EmergencyExitManager:
    def __init__(self): self._last_trigger = None

    def close_all_positions(self, reason="Manual", paper_trader=None, mt5_connection=None, notifier=None, magic_number=424242):
        """Close ALL open positions immediately — panic button.

        Round-22 audit fix: added MT5 live position close support.
        Previously: mt5_connection parameter was accepted but NEVER
        USED — only paper_trader positions were closed. In a real
        emergency (news shock, connection loss, system fault), live
        MT5 positions would have been left open with no way to close
        them from this code path.

        Now: if mt5_connection is provided and connected, fetches all
        open MT5 positions and closes each one via order_send().
        Falls through to paper_trader if MT5 is unavailable.
        """
        log.critical(f"[EmergencyExit] TRIGGERED — {reason}")
        result = EmergencyExitResult(triggered=True, reason=reason)

        # ── MT5 live positions (Round-22 fix) ──────────────────
        if mt5_connection is not None:
            try:
                import MetaTrader5 as mt5
                positions = mt5.positions_get(magic=magic_number) or []
                log.critical(f"[EmergencyExit] Found {len(positions)} MT5 positions to close")
                for pos in positions:
                    try:
                        # Determine close direction
                        symbol = pos.symbol
                        pos_type = pos.type  # 0=BUY, 1=SELL
                        volume = pos.volume
                        tick = mt5.symbol_info_tick(symbol)
                        if tick is None:
                            log.warning(f"[EmergencyExit] No tick for {symbol} — skipping")
                            result.positions_failed += 1
                            continue
                        # Close BUY → sell at bid, Close SELL → buy at ask
                        close_price = tick.bid if pos_type == 0 else tick.ask
                        trade_type = mt5.ORDER_TYPE_SELL if pos_type == 0 else mt5.ORDER_TYPE_BUY
                        request = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": symbol,
                            "volume": volume,
                            "type": trade_type,
                            "position": pos.ticket,
                            "price": close_price,
                            "deviation": 50,  # allow 5 pips slippage in emergency
                            "magic": magic_number,
                            "comment": f"EMERGENCY_EXIT: {reason[:50]}",
                            "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": mt5.ORDER_FILLING_IOC,
                        }
                        result_send = mt5.order_send(request)
                        if result_send is not None and result_send.retcode == 10009:
                            result.positions_closed += 1
                            log.info(f"[EmergencyExit] Closed MT5 #{pos.ticket} {symbol} {volume} lots")
                        else:
                            result.positions_failed += 1
                            err_code = result_send.retcode if result_send else "None"
                            log.error(f"[EmergencyExit] Failed to close MT5 #{pos.ticket} — retcode={err_code}")
                    except Exception as e:
                        result.positions_failed += 1
                        log.error(f"[EmergencyExit] MT5 close error for ticket {pos.ticket}: {e}")
            except ImportError:
                log.warning("[EmergencyExit] MetaTrader5 not available — MT5 positions not closed")
            except Exception as e:
                log.error(f"[EmergencyExit] MT5 position fetch failed: {e}")

        # ── Paper positions ─────────────────────────────────────
        if paper_trader is not None:
            try:
                for trade in list(paper_trader.open_positions):
                    try:
                        closed = paper_trader.close_trade(trade, "EMERGENCY_EXIT", float(trade.get("current_price", trade.get("entry",0))))
                        if closed: result.positions_closed += 1; result.total_pnl += float(closed.get("pnl",0) or 0)
                    except Exception as e:
                        log.error(f"[EmergencyExit] paper close failed for trade={trade.get('symbol','?')}: {e}", exc_info=True)
                        result.positions_failed += 1
            except Exception as e:
                log.error(f"[EmergencyExit] paper iteration failed: {e}", exc_info=True)

        # ── Notify ──────────────────────────────────────────────
        if notifier is not None:
            try:
                import asyncio
                msg = (f"🚨 EMERGENCY EXIT TRIGGERED\n"
                       f"Reason: {reason}\n"
                       f"Closed: {result.positions_closed}\n"
                       f"Failed: {result.positions_failed}\n"
                       f"PnL: ${result.total_pnl:.2f}")
                try:
                    # Bug #5 fix: deprecated asyncio.get_event_loop() replaced.
                    # Use a dedicated thread to avoid conflicts with any running event loop.
                    import concurrent.futures
                    import threading as _threading

                    def _send_in_new_loop():
                        _loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(_loop)
                        try:
                            _loop.run_until_complete(notifier.send_message(msg, priority=True))
                        finally:
                            _loop.close()

                    t = _threading.Thread(target=_send_in_new_loop, daemon=True)
                    t.start()
                    t.join(timeout=10)
                except Exception as e:
                    log.warning(f"[EmergencyExit] Telegram send failed: {e}")
            except Exception as e:
                log.warning(f"[EmergencyExit] notifier block failed: {e}")

        self._last_trigger = result
        log.critical(f"[EmergencyExit] Complete — closed {result.positions_closed}, failed {result.positions_failed}")
        return result

    def check_news_shock(self, current, previous, threshold_pct=5.0):
        if previous <= 0: return False
        return abs(current-previous)/previous*100 >= threshold_pct

_emergency_manager = None
def get_emergency_exit_manager():
    global _emergency_manager
    if _emergency_manager is None: _emergency_manager = EmergencyExitManager()
    return _emergency_manager
