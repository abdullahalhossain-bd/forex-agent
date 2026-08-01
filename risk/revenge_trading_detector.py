"""risk/revenge_trading_detector.py — Revenge trading detection"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from utils.logger import get_logger
log = get_logger("revenge_detector")
COOLDOWN_AFTER_LOSS_MINUTES = 15; MAX_TRADES_PER_HOUR = 3; MAX_LOSSES_PER_HOUR = 2
REVENGE_LOT_INCREASE_MULT = 1.5

@dataclass
class RevengeTradingResult:
    is_revenge: bool = False; severity: str = "NONE"; reasons: list = field(default_factory=list)
    recommended_cooldown_minutes: int = 0; recommended_max_lot: float = 0.0

def check_revenge_trading(recent_trades, proposed_trade, now=None):
    result = RevengeTradingResult()
    now = now or datetime.now(timezone.utc)
    if not recent_trades: return result
    reasons = []; score = 0
    recent_losses = [t for t in recent_trades if t.get("result")=="LOSS" and _parse_time(t.get("close_time"))]
    if recent_losses:
        mrl = max(recent_losses, key=lambda t: _parse_time(t.get("close_time")) or datetime.min.replace(tzinfo=timezone.utc))
        lt = _parse_time(mrl.get("close_time"))
        if lt:
            mins = (now-lt).total_seconds()/60
            if mins < COOLDOWN_AFTER_LOSS_MINUTES: reasons.append(f"{mins:.0f} min since loss"); score += 3
    one_hour_ago = now - timedelta(hours=1)
    trades_last_hour = [t for t in recent_trades if _parse_time(t.get("close_time")) and _parse_time(t.get("close_time")) > one_hour_ago]
    if len(trades_last_hour) >= MAX_TRADES_PER_HOUR: reasons.append(f"{len(trades_last_hour)} trades/hour"); score += 2
    losses_last_hour = [t for t in trades_last_hour if t.get("result")=="LOSS"]
    if len(losses_last_hour) >= MAX_LOSSES_PER_HOUR: reasons.append(f"{len(losses_last_hour)} losses/hour"); score += 4
    if recent_losses:
        mrl = max(recent_losses, key=lambda t: _parse_time(t.get("close_time")) or datetime.min.replace(tzinfo=timezone.utc))
        ll = float(mrl.get("lot",0) or 0); pl = float(proposed_trade.get("lot",0) or 0)
        if ll > 0 and pl > ll * REVENGE_LOT_INCREASE_MULT: reasons.append(f"Lot {pl/ll:.1f}x after loss"); score += 4
    if score >= 7: result.severity="HIGH"; result.is_revenge=True; result.recommended_cooldown_minutes=60
    elif score >= 4: result.severity="MEDIUM"; result.is_revenge=True; result.recommended_cooldown_minutes=30
    elif score >= 2: result.severity="LOW"; result.is_revenge=True; result.recommended_cooldown_minutes=15
    result.reasons = reasons; result.recommended_max_lot = float(proposed_trade.get("lot",0) or 0)
    if result.is_revenge: log.warning(f"[RevengeDetector] {result.severity}: {reasons}")
    return result

def _parse_time(tv):
    if tv is None: return None
    if isinstance(tv, datetime): return tv if tv.tzinfo else tv.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(tv)); return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError): return None
