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
    spread_pips: float = 0.0; spread_cost_usd: float = 0.0
    slippage_pips: float = 0.0; hold_bars: int = 0
    confluence_factors: int = 0; quality_grade: str = "F"
    def to_dict(self): return {k: v for k, v in self.__dict__.items()}


def _spread_cost_usd(symbol: str, lot: float, spread_pips: float) -> float:
    pip_value = pip_value_usd_per_lot(symbol)
    return round(spread_pips * pip_value * lot, 2)


def _half_spread_price(spread_pips: float, symbol: str) -> float:
    return _pip_to_price(spread_pips / 2.0, symbol)


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

    def _check_spread_limit(self, symbol: str, spread_pips: float | None = None) -> tuple[bool, str]:
        """Mirror live OrderManager spread rejection. Returns (allowed, reason).
        Live source: broker/spread_monitor.MAX_SPREAD_PIPS table.

        FIX (2026-08-19 audit Bug 1): previously imported `get_spread_limit`
        which does NOT exist in broker.spread_monitor — the ImportError was
        silently swallowed and the function returned True for every call,
        making the spread limit a no-op. Now imports only MAX_SPREAD_PIPS
        (which DOES exist) so the limit table is actually consulted.
        """
        if not self.enforce_spread_limit:
            return True, ""
        try:
            from broker.spread_monitor import MAX_SPREAD_PIPS
            spread = spread_pips if spread_pips is not None else self._get_spread_pips(symbol)
            limit = MAX_SPREAD_PIPS.get(symbol, 3.0)
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
                   spread_pips: float | None = None,
                   confidence=0, strategy="", confluence_factors=0, quality_grade="F"):
        # ── Spread limit enforcement (parity with live OrderManager) ──
        actual_spread_pips = spread_pips if spread_pips is not None else self._get_spread_pips(symbol)
        allowed, reason = self._check_spread_limit(symbol, actual_spread_pips)
        if not allowed:
            log.info(f"[BrokerSimulator] {symbol} {direction} REJECTED — {reason}")
            return None

        self._tc += 1
        pip = _pip_value(symbol)
        slew_p = max(0, np.random.normal(self.slippage_pips, self.slippage_stdev))
        slip = _pip_to_price(slew_p, symbol)
        half_spread = _half_spread_price(actual_spread_pips, symbol)
        # FIX (2026-08-19 audit Bug 2): entry_price from unified_engine is
        # BID-based (MT5 historical OHLC is BID). For BUY we want ASK fill
        # (entry + slip + half_spread) — was correct. For SELL we want BID
        # fill (entry - slip) — was wrongly subtracting half_spread again,
        # understating SELL entry by half_spread on every trade.
        if direction.upper() == "BUY":
            fp = entry_price + slip + half_spread
        else:  # SELL — BID fill, no spread adjustment
            fp = entry_price - slip
        # FIX (2026-08-19 audit Bug 8): commission must be on FILLED lot,
        # not requested lot. Move partial-fill decision above commission.
        al = lot
        if random.random() < self.partial_fill_prob:
            al = round(lot * random.uniform(0.5, 0.95), 2)
        comm = self.commission_per_lot * al
        return SimulatedTrade(
            trade_id=self._tc, symbol=symbol, direction=direction.upper(),
            entry_time=bar_time.isoformat() if isinstance(bar_time, datetime) else str(bar_time),
            entry_price=_round_price(fp, symbol),
            requested_entry=_round_price(entry_price, symbol),
            stop_loss=_round_price(sl, symbol),
            take_profit=_round_price(tp, symbol),
            lot_size=al, confidence=confidence, strategy=strategy,
            commission_usd=round(comm, 2),
            spread_pips=round(actual_spread_pips, 1),
            spread_cost_usd=_spread_cost_usd(symbol, al, actual_spread_pips),
            slippage_pips=round(slew_p, 1),
            confluence_factors=confluence_factors, quality_grade=quality_grade,
        )

    def check_exit(self, trade, bar_high, bar_low, bar_close, bar_time):
        pip = _pip_value(trade.symbol)
        spread_pips = getattr(trade, "spread_pips", 0.0) or self._get_spread_pips(trade.symbol)
        half_spread = _half_spread_price(spread_pips, trade.symbol)
        slip_p = max(0, np.random.normal(self.slippage_pips * 0.5, self.slippage_stdev * 0.5))
        slip = _pip_to_price(slip_p, trade.symbol)
        if trade.direction == "BUY":
            # FIX (2026-08-19 audit Bug 3): SL/TP for BUY are stored as BID
            # levels (MT5 closes BUY at BID touching SL or TP). The trigger
            # detection (bar_low <= SL) is BID-vs-BID — correct. The exit
            # fill must also be at BID (no spread subtraction); was wrongly
            # subtracting half_spread, understating every BUY exit by
            # half_spread — eroded profit factor systematically.
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
            # SELL: SL/TP are ASK levels. Exit fills at ASK (entry + half_spread).
            sl_hit = bar_high >= trade.stop_loss; tp_hit = bar_low <= trade.take_profit
            if sl_hit and tp_hit:
                # Heuristic: assume the level closer to entry was hit first
                if abs(trade.entry_price - trade.take_profit) < abs(trade.entry_price - trade.stop_loss):
                    ep, er = trade.take_profit + half_spread + slip, "TP"
                else:
                    ep, er = trade.stop_loss + half_spread + slip, "SL"
            elif sl_hit: ep, er = trade.stop_loss + half_spread + slip, "SL"
            elif tp_hit: ep, er = trade.take_profit + half_spread + slip, "TP"
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
        spread_pips = getattr(trade, "spread_pips", 0.0) or self._get_spread_pips(trade.symbol)
        half_spread = _half_spread_price(spread_pips, trade.symbol)
        slip_p = max(0, np.random.normal(self.slippage_pips * 0.5, self.slippage_stdev * 0.5))
        slip = _pip_to_price(slip_p, trade.symbol)
        if trade.direction == "BUY":
            # FIX (2026-08-19 audit Bug 3): close_price is bar close = BID.
            # BUY closes at BID — no spread subtraction.
            ep = close_price - slip
            pnl_p = (ep - trade.entry_price) / pip
        else:
            ep = close_price + half_spread + slip
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