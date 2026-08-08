"""
backtest/symbol_specs.py — Single source of truth for backtest symbol specs.

PARITY NOTE
-----------
Live trading reads `digits`, `volume_step`, `volume_min`, `volume_max`,
`trade_contract_size`, and `point` directly from MT5's `symbol_info()`
(see broker/order_manager.py:_normalize_order_params). Backtest has no MT5
connection, so we need an offline mirror of those specs.

This module is the SINGLE place in backtest where symbol→digits,
symbol→contract_size, and symbol→pip_size mappings live. It imports
`PIP_SIZE` from core.constants (the same source live order_manager.py
reads indirectly via MT5's `point * 10`), and adds digits/contract_size
tables that mirror MT5's typical published specs.

If a future audit finds these tables drifting from what live MT5 returns
on the production VPS, fix the table here — do NOT add a parallel table
elsewhere.
"""
from __future__ import annotations

from core.constants import PIP_SIZE


# ── Digits ─────────────────────────────────────────────────────────
# Mirrors MT5 `symbol_info().digits` for the symbols most brokers publish.
# Live reads this from MT5 directly; backtest must hardcode the same values
# so SL/TP rounding in BrokerSimulator matches the rounding OrderManager
# would have applied.
#
# Standard MT5 conventions:
#   5-digit FX majors (EURUSD, GBPUSD, ...) → 5 digits (point = 0.00001, pip = 0.0001)
#   3-digit JPY crosses (USDJPY, GBPJPY, …)  → 3 digits (point = 0.001,   pip = 0.01)
#   XAUUSD                                   → 2 digits (point = 0.01,    pip = 0.01)
#   XAGUSD                                   → 3 digits (point = 0.001,   pip = 0.001)
#   US30                                     → 1 digit  (point = 1.0,     pip = 1.0)
#   NAS100                                   → 1 digit  (point = 1.0,     pip = 0.01 — index pip varies by broker)
DIGITS: dict[str, int] = {
    # 5-digit FX majors
    "EURUSD": 5, "GBPUSD": 5, "AUDUSD": 5, "NZDUSD": 5, "USDCAD": 5, "USDCHF": 5,
    # 3-digit JPY crosses
    "USDJPY": 3, "GBPJPY": 3, "EURJPY": 3, "AUDJPY": 3, "NZDJPY": 3, "CADJPY": 3, "CHFJPY": 3,
    # Minor crosses — 5-digit
    "EURGBP": 5, "EURAUD": 5, "EURNZD": 5, "EURCAD": 5, "EURCHF": 5,
    "GBPAUD": 5, "GBPNZD": 5, "GBPCAD": 5, "GBPCHF": 5,
    "AUDCAD": 5, "AUDCHF": 5, "AUDNZD": 5,
    "NZDCAD": 5, "NZDCHF": 5,
    "CADCHF": 5,
    # Commodities
    "XAUUSD": 2, "XAGUSD": 3,
    # Indices — vary by broker; using common IC Markets / FTMO values
    "US30":   1, "NAS100": 1,
    # Default fallback (matches MT5 default for unknown FX symbols)
    "DEFAULT": 5,
}


# ── Contract size ─────────────────────────────────────────────────
# Mirrors MT5 `symbol_info().trade_contract_size`.
# Live reads this from MT5 directly; backtest uses these typical values.
CONTRACT_SIZE: dict[str, float] = {
    # FX standard lot = 100,000 units of base currency
    "EURUSD": 100_000, "GBPUSD": 100_000, "AUDUSD": 100_000, "NZDUSD": 100_000,
    "USDCAD": 100_000, "USDCHF": 100_000,
    "USDJPY": 100_000, "GBPJPY": 100_000, "EURJPY": 100_000, "AUDJPY": 100_000,
    "NZDJPY": 100_000, "CADJPY": 100_000, "CHFJPY": 100_000,
    "EURGBP": 100_000, "EURAUD": 100_000, "EURNZD": 100_000, "EURCAD": 100_000, "EURCHF": 100_000,
    "GBPAUD": 100_000, "GBPNZD": 100_000, "GBPCAD": 100_000, "GBPCHF": 100_000,
    "AUDCAD": 100_000, "AUDCHF": 100_000, "AUDNZD": 100_000,
    "NZDCAD": 100_000, "NZDCHF": 100_000, "CADCHF": 100_000,
    # Metals — XAU = 100 oz/lot, XAG = 5000 oz/lot (industry standard)
    "XAUUSD": 100, "XAGUSD": 5000,
    # Indices — cash index CFD, 1 unit per lot
    "US30":   1, "NAS100": 1,
    # Default
    "DEFAULT": 100_000,
}


def get_pip_size(symbol: str) -> float:
    """Return pip size (price increment per pip) for a symbol.

    Imports from `core.constants.PIP_SIZE` — the SAME table live
    `broker/order_manager.py` indirectly uses (via MT5's
    `point * 10 if digits in (3,5) else point`, which evaluates to
    exactly these values for standard brokers).
    """
    return PIP_SIZE.get(symbol.upper(), PIP_SIZE["DEFAULT"])


def get_digits(symbol: str) -> int:
    """Return MT5-style digit count for a symbol (for price rounding).

    Live reads this from `mt5.symbol_info().digits`; backtest uses this
    table so SL/TP/entry rounding matches what OrderManager would have
    applied before sending to MT5.
    """
    return DIGITS.get(symbol.upper(), DIGITS["DEFAULT"])


def get_contract_size(symbol: str) -> float:
    """Return MT5-style trade_contract_size for a symbol.

    Live reads this from `mt5.symbol_info().trade_contract_size`;
    backtest uses this table so USD P&L per pip per lot matches.
    """
    return CONTRACT_SIZE.get(symbol.upper(), CONTRACT_SIZE["DEFAULT"])


def round_to_digits(price: float, symbol: str) -> float:
    """Round a price to the symbol's MT5-declared digits.

    Equivalent to live `OrderManager._normalize_order_params` line:
        sl_n = round(float(sl), digits)
    """
    if not price:
        return 0.0
    return round(float(price), get_digits(symbol))


def pip_to_price(pips: float, symbol: str) -> float:
    """Convert a pip count to a price delta for the given symbol."""
    return pips * get_pip_size(symbol)


def price_to_pips(price_delta: float, symbol: str) -> float:
    """Convert a price delta to pip count for the given symbol."""
    return price_delta / get_pip_size(symbol)


def pip_value_usd_per_lot(symbol: str) -> float:
    """USD value of 1 pip move, per 1.0 lot.

    Formula: pip_size × contract_size (gives USD per pip per lot for
    USD-quoted symbols; for non-USD-quoted symbols this is an
    approximation that matches what broker_sim.py previously computed
    inline).

    Live uses `core.constants.PIP_VALUE_USD` table or
    `get_live_pip_value_per_lot(symbol, mt5_conn)` when MT5 is
    connected. Backtest uses this computed value, which equals the
    PIP_VALUE_USD table for FX majors and XAUUSD.
    """
    return get_pip_size(symbol) * get_contract_size(symbol)
