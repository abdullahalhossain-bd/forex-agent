# backtest/broker_sim.py — Realistic Broker Simulation
import logging, random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List, Dict
import numpy as np
log = logging.getLogger(__name__)
DEFAULT_SPREAD_PIPS = {"EURUSD": 1.5, "GBPUSD": 2.0, "USDJPY": 1.5, "AUDUSD": 1.8, "USDCAD": 2.0, "USDCHF": 2.0, "NZDUSD": 2.0, "XAUUSD": 25.0}
DEFAULT_COMMISSION_PER_LOT = 7.0; DEFAULT_SLIPPAGE_PIPS = 2.0; DEFAULT_SLIPPAGE_STDEV = 1.0
DEFAULT_PARTIAL_FILL_PROB = 0.05; DEFAULT_MARGIN_REQ_PCT = 0.01
def _pip_value(s):
    s = s.upper()
    if s.endswith("JPY"): return 0.01
    if s == "XAUUSD": return 0.1
    return 0.0001
def _pip_to_price(p, s): return p * _pip_value(s)
@dataclass
class SimulatedTrade:
    trade_id: int; symbol: str; direction: str; entry_time: str; entry_price: float
    requested_entry: float; stop_loss: float; take_profit: float; lot_size: float
    confidence: int = 0; strategy: str = ""
    exit_time: str = ""; exit_price: float = 0.0; exit_reason: str = ""
    pnl_pips: float = 0.0; pnl_usd: float = 0.0; commission_usd: float = 0.0
    slippage_pips: float = 0.0; hold_bars: int = 0
    confluence_factors: int = 0; quality_grade: str = "F"
    def to_dict(self): return {k: v for k, v in self.__dict__.items()}
class BrokerSimulator:
    def __init__(self, spread_pips=None, commission_per_lot=DEFAULT_COMMISSION_PER_LOT, slippage_pips=DEFAULT_SLIPPAGE_PIPS, slippage_stdev=DEFAULT_SLIPPAGE_STDEV, partial_fill_prob=DEFAULT_PARTIAL_FILL_PROB, margin_req_pct=DEFAULT_MARGIN_REQ_PCT, starting_balance=10000.0):
        self.spread_pips = spread_pips or DEFAULT_SPREAD_PIPS; self.commission_per_lot = commission_per_lot
        self.slippage_pips = slippage_pips; self.slippage_stdev = slippage_stdev
        self.partial_fill_prob = partial_fill_prob; self.margin_req_pct = margin_req_pct
        self.balance = starting_balance; self._tc = 0
    def open_trade(self, symbol, direction, entry_price, sl, tp, lot, bar_time, confidence=0, strategy="", confluence_factors=0, quality_grade="F"):
        self._tc += 1; pip = _pip_value(symbol)
        slip_p = max(0, np.random.normal(self.slippage_pips, self.slippage_stdev)); slip = _pip_to_price(slip_p, symbol)
        fp = entry_price + slip if direction.upper() == "BUY" else entry_price - slip
        comm = self.commission_per_lot * lot; al = lot
        if random.random() < self.partial_fill_prob: al = round(lot * random.uniform(0.5, 0.95), 2)
        return SimulatedTrade(trade_id=self._tc, symbol=symbol, direction=direction.upper(),
            entry_time=bar_time.isoformat() if isinstance(bar_time, datetime) else str(bar_time),
            entry_price=round(fp, 5), requested_entry=round(entry_price, 5), stop_loss=round(sl, 5),
            take_profit=round(tp, 5), lot_size=al, confidence=confidence, strategy=strategy,
            commission_usd=round(comm, 2), slippage_pips=round(slip_p, 1), confluence_factors=confluence_factors, quality_grade=quality_grade)
    def check_exit(self, trade, bar_high, bar_low, bar_close, bar_time):
        pip = _pip_value(trade.symbol)
        slip_p = max(0, np.random.normal(self.slippage_pips * 0.5, self.slippage_stdev * 0.5)); slip = _pip_to_price(slip_p, trade.symbol)
        if trade.direction == "BUY":
            sl_hit = bar_low <= trade.stop_loss; tp_hit = bar_high >= trade.take_profit
            if sl_hit and tp_hit: ep, er = trade.stop_loss - slip, "SL"
            elif sl_hit: ep, er = trade.stop_loss - slip, "SL"
            elif tp_hit: ep, er = trade.take_profit - slip, "TP"
            else: return None
            pnl_p = (ep - trade.entry_price) / pip
        else:
            sl_hit = bar_high >= trade.stop_loss; tp_hit = bar_low <= trade.take_profit
            if sl_hit and tp_hit: ep, er = trade.stop_loss + slip, "SL"
            elif sl_hit: ep, er = trade.stop_loss + slip, "SL"
            elif tp_hit: ep, er = trade.take_profit + slip, "TP"
            else: return None
            pnl_p = (trade.entry_price - ep) / pip
        cs = 100000 if trade.symbol != "XAUUSD" else 100; pvu = pip * cs
        pnl_u = pnl_p * pvu * trade.lot_size - trade.commission_usd
        trade.exit_time = bar_time.isoformat() if isinstance(bar_time, datetime) else str(bar_time)
        trade.exit_price = round(ep, 5); trade.exit_reason = er
        trade.pnl_pips = round(pnl_p, 1); trade.pnl_usd = round(pnl_u, 2)
        self.balance += pnl_u; return trade
    def close_trade(self, trade, close_price, bar_time, reason="manual"):
        pip = _pip_value(trade.symbol)
        slip_p = max(0, np.random.normal(self.slippage_pips * 0.5, self.slippage_stdev * 0.5)); slip = _pip_to_price(slip_p, trade.symbol)
        if trade.direction == "BUY": ep = close_price - slip; pnl_p = (ep - trade.entry_price) / pip
        else: ep = close_price + slip; pnl_p = (trade.entry_price - ep) / pip
        cs = 100000 if trade.symbol != "XAUUSD" else 100; pvu = pip * cs
        pnl_u = pnl_p * pvu * trade.lot_size - trade.commission_usd
        trade.exit_time = bar_time.isoformat() if isinstance(bar_time, datetime) else str(bar_time)
        trade.exit_price = round(ep, 5); trade.exit_reason = reason
        trade.pnl_pips = round(pnl_p, 1); trade.pnl_usd = round(pnl_u, 2)
        self.balance += pnl_u; return trade
    def get_balance(self): return round(self.balance, 2)
