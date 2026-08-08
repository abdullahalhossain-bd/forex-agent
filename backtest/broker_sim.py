# backtest/broker_sim.py — Realistic Broker Simulation
"""
PARITY NOTE
-----------
This module is the ONLY place where backtest simulates broker fills.
It must mirror live `broker/order_manager.py` behavior for:
  - pip size per symbol (now sourced from backtest.symbol_specs → core.constants.PIP_SIZE)
  - digits/precision for rounding SL/TP/entry (now from symbol_specs.DIGITS, mirrors MT5)
  - contract size for USD P&L (now from symbol_specs.CONTRACT_SIZE)
  - spread limit enforcement (now uses broker.spread_monitor.MAX_SPREAD_PIPS — same table as live)

What is intentionally different from live (per project parity rule):
  - Fills are simulated via bar OHLC SL/TP touch detection, not real MT5 order_send.
  - Slippage is drawn from np.random.normal (seeded by unified_engine for reproducibility).
  - Partial fills modeled with 5% probability (rare on retail MT5 demo, but realistic for live).

No trading logic / threshold / filter changes vs live — only execution simulation.
"""
import logging, random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List, Dict
import numpy as np

from backtest.symbol_specs import (
    get_pip_size, get_digits, get_contract_size,
    round_to_digits, pip_to_price, pip_value_usd_per_lot,
)

log = logging.getLogger(__name__)

# ── Spread assumptions ────────────────────────────────────────────
# DEFAULT_SPREAD_PIPS is used ONLY when the caller doesn't pass an
# explicit spread_pips value. These are typical average spreads; they
# do NOT replace live spread-limit enforcement (see open_trade()).
DEFAULT_SPREAD_PIPS = {
    "EURUSD": 1.5, "GBPUSD": 2.0, "USDJPY": 1.5, "AUDUSD": 1.8,
    "USDCAD": 2.0, "USDCHF": 2.0, "NZDUSD": 2.0, "XAUUSD": 25.0,
}

# Commission/slippage defaults — overridden by unified_engine to use
# core.constants.COMMISSION_USD_PER_LOT and BROKER_SLIPPAGE_PIPS.
DEFAULT_COMMISSION_PER_LOT = 7.0
DEFAULT_SLIPPAGE_PIPS = 2.0
DEFAULT_SLIPPAGE_STDEV = 1.0
DEFAULT_PARTIAL_FILL_PROB = 0.05
DEFAULT_MARGIN_REQ_PCT = 0.01


def _pip_value(s):
    """Pip size for a symbol — sourced from core.constants.PIP_SIZE
    via backtest.symbol_specs. Replaces the old hardcoded
    {JPY:0.01, XAUUSD:0.1, else:0.0001} table that was 10× wrong for XAUUSD.
    """
    return get_pip_size(s)


def _pip_to_price(p, s):
    return pip_to_price(p, s)


def _digits(s):
    """MT5-style digits for the symbol — used for rounding SL/TP/entry
    to the same precision live OrderManager._normalize_order_params
    would round to before sending to MT5.
    """
    return get_digits(s)


def _contract_size(s):
    """MT5-style trade_contract_size — replaces the old hardcoded
    `100000 if not XAUUSD else 100` table.
    """
    return get_contract_size(s)


def _round_price(price, symbol):
    """Round to symbol's digits — equivalent to
    `round(float(price), digits)` in OrderManager._normalize_order_params.
    """
    return round_to_digits(price, symbol)


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
    def __init__(self, spread_pips=None, commission_per_lot=DEFAULT_COMMISSION_PER_LOT,
                 slippage_pips=DEFAULT_SLIPPAGE_PIPS, slippage_stdev=DEFAULT_SLIPPAGE_STDEV,
                 partial_fill_prob=DEFAULT_PARTIAL_FILL_PROB,
                 margin_req_pct=DEFAULT_MARGIN_REQ_PCT,
                 starting_balance=10000.0,
                 enforce_spread_limit=True):
        """
        Args:
            spread_pips: either a float (same for all symbols) or a dict
                {symbol: spread_pips}. If None, uses DEFAULT_SPREAD_PIPS.
            enforce_spread_limit: if True, reject open_trade() calls when
                the assumed spread exceeds broker.spread_monitor.MAX_SPREAD_PIPS
                for the symbol. This mirrors live OrderManager behavior
                (which rejects with "Spread too wide" retcode). Set to
                False only for unit tests.
        """
        if spread_pips is None:
            self.spread_pips = dict(DEFAULT_SPREAD_PIPS)
        elif isinstance(spread_pips, dict):
            self.spread_pips = spread_pips
        else:
            # Single float — apply to all symbols (used when caller passes
            # a known spread from historical data)
            self._single_spread = float(spread_pips)
            self.spread_pips = None
        self.commission_per_lot = commission_per_lot
        self.slippage_pips = slippage_pips
        self.slippage_stdev = slippage_stdev
        self.partial_fill_prob = partial_fill_prob
        self.margin_req_pct = margin_req_pct
        self.enforce_spread_limit = enforce_spread_limit
        self.balance = starting_balance
        self._tc = 0

    def _get_spread_pips(self, symbol: str) -> float:
        """Return the assumed spread for a symbol. If a single float
        was passed at construction, use that; otherwise use the dict.
        Falls back to DEFAULT_SPREAD_PIPS then 2.0.
        """
        if hasattr(self, "_single_spread"):
            return self._single_spread
        if self.spread_pips is None:
            return DEFAULT_SPREAD_PIPS.get(symbol, 2.0)
        return self.spread_pips.get(symbol, DEFAULT_SPREAD_PIPS.get(symbol, 2.0))

    def _check_spread_limit(self, symbol: str) -> tuple[bool, str]:
        """Mirror live OrderManager spread rejection. Returns (allowed, reason).
        Live source: broker/spread_monitor.MAX_SPREAD_PIPS table.
        """
        if not self.enforce_spread_limit:
            return True, ""
        try:
            from broker.spread_monitor import MAX_SPREAD_PIPS, get_spread_limit
            spread = self._get_spread_pips(symbol)
            limit = get_spread_limit(symbol) if hasattr(globals().get("spread_monitor"), "get_spread_limit") else MAX_SPREAD_PIPS.get(symbol, 3.0)
            if spread > limit:
                return False, f"Spread too wide ({spread:.2f} pips > {limit:.2f} pips limit)"
        except ImportError:
            # spread_monitor unavailable — fail open (don't block backtest
            # on import errors, but log once)
            if not getattr(self, "_warned_spread_monitor", False):
                log.warning("[BrokerSimulator] broker.spread_monitor unavailable — spread limit NOT enforced")
                self._warned_spread_monitor = True
        return True, ""

    def open_trade(self, symbol, direction, entry_price, sl, tp, lot, bar_time,
                   confidence=0, strategy="", confluence_factors=0, quality_grade="F"):
        # ── Spread limit enforcement (parity with live OrderManager) ──
        allowed, reason = self._check_spread_limit(symbol)
        if not allowed:
            log.info(f"[BrokerSimulator] {symbol} {direction} REJECTED — {reason}")
            return None

        self._tc += 1
        pip = _pip_value(symbol)
        slip_p = max(0, np.random.normal(self.slippage_pips, self.slippage_stdev))
        slip = _pip_to_price(slip_p, symbol)
        fp = entry_price + slip if direction.upper() == "BUY" else entry_price - slip
        comm = self.commission_per_lot * lot
        al = lot
        if random.random() < self.partial_fill_prob:
            al = round(lot * random.uniform(0.5, 0.95), 2)
        return SimulatedTrade(
            trade_id=self._tc, symbol=symbol, direction=direction.upper(),
            entry_time=bar_time.isoformat() if isinstance(bar_time, datetime) else str(bar_time),
            entry_price=_round_price(fp, symbol),
            requested_entry=_round_price(entry_price, symbol),
            stop_loss=_round_price(sl, symbol),
            take_profit=_round_price(tp, symbol),
            lot_size=al, confidence=confidence, strategy=strategy,
            commission_usd=round(comm, 2),
            slippage_pips=round(slip_p, 1),
            confluence_factors=confluence_factors, quality_grade=quality_grade,
        )

    def check_exit(self, trade, bar_high, bar_low, bar_close, bar_time):
        pip = _pip_value(trade.symbol)
        slip_p = max(0, np.random.normal(self.slippage_pips * 0.5, self.slippage_stdev * 0.5))
        slip = _pip_to_price(slip_p, trade.symbol)
        if trade.direction == "BUY":
            sl_hit = bar_low <= trade.stop_loss; tp_hit = bar_high >= trade.take_profit
            if sl_hit and tp_hit:
                # Heuristic: assume the level closer to entry was hit first
                if abs(trade.entry_price - trade.take_profit) < abs(trade.entry_price - trade.stop_loss):
                    ep, er = trade.take_profit - slip, "TP"
                else:
                    ep, er = trade.stop_loss - slip, "SL"
            elif sl_hit: ep, er = trade.stop_loss - slip, "SL"
            elif tp_hit: ep, er = trade.take_profit - slip, "TP"
            else: return None
            pnl_p = (ep - trade.entry_price) / pip
        else:
            sl_hit = bar_high >= trade.stop_loss; tp_hit = bar_low <= trade.take_profit
            if sl_hit and tp_hit:
                # Heuristic: assume the level closer to entry was hit first
                if abs(trade.entry_price - trade.take_profit) < abs(trade.entry_price - trade.stop_loss):
                    ep, er = trade.take_profit + slip, "TP"
                else:
                    ep, er = trade.stop_loss + slip, "SL"
            elif sl_hit: ep, er = trade.stop_loss + slip, "SL"
            elif tp_hit: ep, er = trade.take_profit + slip, "TP"
            else: return None
            pnl_p = (trade.entry_price - ep) / pip
        cs = _contract_size(trade.symbol)
        pvu = pip_value_usd_per_lot(trade.symbol)
        pnl_u = pnl_p * pvu * trade.lot_size - trade.commission_usd
        trade.exit_time = bar_time.isoformat() if isinstance(bar_time, datetime) else str(bar_time)
        trade.exit_price = _round_price(ep, trade.symbol)
        trade.exit_reason = er
        trade.pnl_pips = round(pnl_p, 1)
        trade.pnl_usd = round(pnl_u, 2)
        self.balance += pnl_u
        return trade

    def close_trade(self, trade, close_price, bar_time, reason="manual"):
        pip = _pip_value(trade.symbol)
        slip_p = max(0, np.random.normal(self.slippage_pips * 0.5, self.slippage_stdev * 0.5))
        slip = _pip_to_price(slip_p, trade.symbol)
        if trade.direction == "BUY":
            ep = close_price - slip
            pnl_p = (ep - trade.entry_price) / pip
        else:
            ep = close_price + slip
            pnl_p = (trade.entry_price - ep) / pip
        pvu = pip_value_usd_per_lot(trade.symbol)
        pnl_u = pnl_p * pvu * trade.lot_size - trade.commission_usd
        trade.exit_time = bar_time.isoformat() if isinstance(bar_time, datetime) else str(bar_time)
        trade.exit_price = _round_price(ep, trade.symbol)
        trade.exit_reason = reason
        trade.pnl_pips = round(pnl_p, 1)
        trade.pnl_usd = round(pnl_u, 2)
        self.balance += pnl_u
        return trade

    def get_balance(self):
        return round(self.balance, 2)