# risk/usd_tp_sl_calculator.py — Cross-currency USD-denominated TP/SL calculator
# =============================================================================
# Ported from: https://github.com/tanvird3/TradingRobot/blob/master/TakeProfitStopLoss.mqh
# Original author: khantanvir (Tanvir) — license not specified in source
#
# Converts a USD-denominated profit/loss target into a price level for any
# forex pair — USD-quoted, USD-base, or cross. This is non-trivial because:
#
#   - EURUSD: 1 pip = $0.0001 × 100,000 units = $10 per lot. Direct: TP = price + $/lot/units.
#   - USDJPY: 1 pip = 0.01 JPY. To get $1 profit, the JPY price must move by
#             (1 / units) × (1 / USDJPY_rate) — because JPY is the quote currency
#             and we need to convert JPY profit to USD.
#   - EURGBP: neither currency is USD. Need to convert via GBPUSD (or its inverse).
#
# The original MQL4 handles all three cases. This port does the same.
#
# Public API:
#   tp_long(symbol, lot_size, usd_profit, ask, usd_quote_fn) -> float
#   tp_short(symbol, lot_size, usd_profit, bid, usd_quote_fn) -> float
#   sl_long(symbol, lot_size, usd_loss, ask, usd_quote_fn) -> float
#   sl_short(symbol, lot_size, usd_loss, bid, usd_quote_fn) -> float
#
# `usd_quote_fn` is a callable that returns the USD price for a given currency:
#   - For USD itself: returns 1.0
#   - For "EUR": returns 1/USDJPY? no, returns EURUSD rate
#   - For "JPY": returns 1/USDJPY rate (how many USD per JPY)
# In live MT5, you'd plug in `lambda cur: mt5.symbol_info("USD"+cur).ask`
# (with handling for when "USD"+cur doesn't exist — try cur+"USD" and invert).
#
# The default `usd_quote_fn` is None → assumes the symbol itself is USD-quoted
# (the simplest case, which covers EURUSD, GBPUSD, AUDUSD, etc.).
# =============================================================================

from __future__ import annotations

from typing import Callable, Optional

from utils.logger import get_logger

log = get_logger("usd_tp_sl_calc")

UNITS_PER_LOT = 100_000  # standard lot for forex


def _parse_currency(symbol: str) -> tuple[str, str]:
    """
    Parse a 6-character forex symbol into (base, quote).
    'EURUSD' → ('EUR', 'USD')
    'USDJPY' → ('USD', 'JPY')
    'EURGBP' → ('EUR', 'GBP')
    """
    s = symbol.upper().replace("/", "").replace("_", "").strip()
    if len(s) != 6:
        raise ValueError(f"Expected 6-char forex symbol, got {symbol!r}")
    return s[:3], s[3:]


def _default_usd_quote_fn(currency: str) -> float:
    """
    Default fallback: assume USD = 1.0, everything else = 1.0 (i.e., assume
    the symbol is USD-quoted). This makes the calculator behave correctly
    for EURUSD/GBPUSD/AUDUSD/etc. without needing a market data feed.
    For cross pairs or USD-base pairs, supply a real usd_quote_fn.
    """
    if currency.upper() == "USD":
        return 1.0
    log.warning(
        f"_default_usd_quote_fn({currency!r}) returning 1.0 — supply a real "
        f"usd_quote_fn for non-USD-quoted pairs."
    )
    return 1.0


def tp_long(
    symbol: str,
    lot_size: float,
    usd_profit: float,
    ask: float,
    usd_quote_fn: Optional[Callable[[str], float]] = None,
) -> float:
    """
    Compute the take-profit price for a LONG position that should yield
    `usd_profit` USD when closed, given current `ask` price.

    For EURUSD (USD-quoted):
        TP = ask + (usd_profit / (UNITS_PER_LOT * lot_size))
    For USDJPY (USD-base):
        TP = (1 + usd_profit / (UNITS_PER_LOT * lot_size)) * ask
    For EURGBP (cross):
        quote_usd = usd_quote_fn('GBP')  # how many USD per 1 GBP
        TP = (ask / quote_usd + usd_profit / (UNITS_PER_LOT * lot_size)) * quote_usd
    """
    fn = usd_quote_fn or _default_usd_quote_fn
    base, quote = _parse_currency(symbol)
    units = UNITS_PER_LOT * lot_size

    # Round-19 audit fix: guard against zero division.
    # lot_size=0 or a broken data feed (quote_usd=0) would otherwise
    # produce a raw ZeroDivisionError instead of a clean risk error.
    if units <= 0:
        raise ValueError(
            f"tp_long: invalid units ({units}) — lot_size={lot_size}, "
            f"UNITS_PER_LOT={UNITS_PER_LOT}. Cannot compute TP with zero lot."
        )

    if quote == "USD":
        # Direct: TP in same price units as ask
        return ask + usd_profit / units
    elif base == "USD":
        # USD-base (e.g., USDJPY): TP scales with ask
        return (1.0 + usd_profit / units) * ask
    else:
        # Cross pair (e.g., EURGBP): convert via the quote currency's USD rate
        quote_usd = fn(quote)
        if quote_usd <= 0:
            raise ValueError(
                f"tp_long: invalid quote_usd rate ({quote_usd}) for "
                f"{quote}. Cannot compute cross-pair TP with zero rate."
            )
        return (ask / quote_usd + usd_profit / units) * quote_usd


def tp_short(
    symbol: str,
    lot_size: float,
    usd_profit: float,
    bid: float,
    usd_quote_fn: Optional[Callable[[str], float]] = None,
) -> float:
    """Mirror of tp_long for SHORT positions (TP below bid)."""
    fn = usd_quote_fn or _default_usd_quote_fn
    base, quote = _parse_currency(symbol)
    units = UNITS_PER_LOT * lot_size

    # Round-19: same zero-division guard as tp_long
    if units <= 0:
        raise ValueError(
            f"tp_short: invalid units ({units}) — lot_size={lot_size}, "
            f"UNITS_PER_LOT={UNITS_PER_LOT}. Cannot compute TP with zero lot."
        )

    if quote == "USD":
        return bid - usd_profit / units
    elif base == "USD":
        return (1.0 - usd_profit / units) * bid
    else:
        quote_usd = fn(quote)
        if quote_usd <= 0:
            raise ValueError(
                f"tp_short: invalid quote_usd rate ({quote_usd}) for "
                f"{quote}. Cannot compute cross-pair TP with zero rate."
            )
        return (bid / quote_usd - usd_profit / units) * quote_usd


def sl_long(
    symbol: str,
    lot_size: float,
    usd_loss: float,
    ask: float,
    usd_quote_fn: Optional[Callable[[str], float]] = None,
) -> float:
    """
    Compute the stop-loss price for a LONG position that should lose at most
    `usd_loss` USD when hit. `usd_loss` should be a positive number (the
    magnitude of the loss); the returned SL will be below `ask`.
    """
    fn = usd_quote_fn or _default_usd_quote_fn
    base, quote = _parse_currency(symbol)
    units = UNITS_PER_LOT * lot_size

    if quote == "USD":
        return ask - usd_loss / units
    elif base == "USD":
        return (1.0 - usd_loss / units) * ask
    else:
        quote_usd = fn(quote)
        return (ask / quote_usd - usd_loss / units) * quote_usd


def sl_short(
    symbol: str,
    lot_size: float,
    usd_loss: float,
    bid: float,
    usd_quote_fn: Optional[Callable[[str], float]] = None,
) -> float:
    """Mirror of sl_long for SHORT positions (SL above bid)."""
    fn = usd_quote_fn or _default_usd_quote_fn
    base, quote = _parse_currency(symbol)
    units = UNITS_PER_LOT * lot_size

    if quote == "USD":
        return bid + usd_loss / units
    elif base == "USD":
        return (1.0 + usd_loss / units) * bid
    else:
        quote_usd = fn(quote)
        return (bid / quote_usd + usd_loss / units) * quote_usd


# ── Convenience: compute all four at once ────────────────────────────────────

def tp_sl_for_trade(
    symbol: str,
    side: str,
    lot_size: float,
    usd_target: float,
    usd_stop: float,
    entry_price: float,
    usd_quote_fn: Optional[Callable[[str], float]] = None,
) -> tuple[float, float]:
    """
    Compute (TP, SL) for a trade in one call.

    Parameters
    ----------
    symbol : forex pair, e.g. "EURUSD"
    side : "BUY" (long) or "SELL" (short)
    lot_size : position size in lots (1.0 = 100,000 units)
    usd_target : desired USD profit at TP (positive number)
    usd_stop : max USD loss at SL (positive number)
    entry_price : current ask (for BUY) or bid (for SELL)
    usd_quote_fn : callable(currency) → USD price of 1 unit of that currency.
        Required for cross pairs and USD-base pairs. Optional for USD-quoted.

    Returns
    -------
    (tp_price, sl_price)
    """
    side = side.upper()
    if side == "BUY":
        return (
            tp_long(symbol, lot_size, usd_target, entry_price, usd_quote_fn),
            sl_long(symbol, lot_size, usd_stop, entry_price, usd_quote_fn),
        )
    elif side == "SELL":
        return (
            tp_short(symbol, lot_size, usd_target, entry_price, usd_quote_fn),
            sl_short(symbol, lot_size, usd_stop, entry_price, usd_quote_fn),
        )
    else:
        raise ValueError(f"side must be BUY or SELL, got {side!r}")


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # EURUSD (USD-quoted): 1 lot, $10 profit
    # EURUSD ask = 1.1000 → TP = 1.1000 + 10/(100000*1.0) = 1.1000 + 0.0001 = 1.1001
    tp = tp_long("EURUSD", lot_size=1.0, usd_profit=10.0, ask=1.1000)
    print(f"EURUSD long TP for $10 profit at 1.1000: {tp:.5f}  (expect 1.10010)")
    assert abs(tp - 1.10010) < 1e-6

    # EURUSD short SL for $10 loss at 1.1000: SL = 1.1000 + 0.0001 = 1.10010
    sl = sl_short("EURUSD", lot_size=1.0, usd_loss=10.0, bid=1.1000)
    print(f"EURUSD short SL for $10 loss at 1.1000: {sl:.5f}  (expect 1.10010)")
    assert abs(sl - 1.10010) < 1e-6

    # USDJPY (USD-base): 1 lot, $10 profit, ask=150.00
    # TP = (1 + 10/(100000*1)) * 150 = (1 + 0.0001) * 150 = 150.015
    tp = tp_long("USDJPY", lot_size=1.0, usd_profit=10.0, ask=150.00,
                 usd_quote_fn=lambda c: 1.0 / 150.0 if c == "JPY" else 1.0)
    print(f"USDJPY long TP for $10 profit at 150.00: {tp:.5f}  (expect 150.01500)")
    assert abs(tp - 150.015) < 1e-4

    # EURGBP (cross): 1 lot, $10 profit, ask=0.8500, GBPUSD=1.2500
    # quote_usd = GBPUSD = 1.2500
    # TP = (0.8500 / 1.2500 + 10/100000) * 1.2500 = (0.68 + 0.0001) * 1.2500 = 0.850125
    tp = tp_long("EURGBP", lot_size=1.0, usd_profit=10.0, ask=0.8500,
                 usd_quote_fn=lambda c: 1.2500 if c == "GBP" else 1.0)
    print(f"EURGBP long TP for $10 profit at 0.8500 (GBPUSD=1.2500): {tp:.5f}  (expect 0.85013)")
    assert abs(tp - 0.850125) < 1e-6

    # tp_sl_for_trade convenience
    tp, sl = tp_sl_for_trade("EURUSD", "BUY", lot_size=0.5,
                             usd_target=25.0, usd_stop=50.0, entry_price=1.0850)
    print(f"EURUSD BUY 0.5 lot, target=$25, stop=$50: TP={tp:.5f}, SL={sl:.5f}")
    assert tp > 1.0850 and sl < 1.0850

    # Verify BUY/SELL symmetry for the same target/stop
    tp_b, sl_b = tp_sl_for_trade("EURUSD", "BUY", 1.0, 10.0, 10.0, 1.1000)
    tp_s, sl_s = tp_sl_for_trade("EURUSD", "SELL", 1.0, 10.0, 10.0, 1.1000)
    print(f"  BUY:  TP={tp_b:.5f}, SL={sl_b:.5f}")
    print(f"  SELL: TP={tp_s:.5f}, SL={sl_s:.5f}")
    assert abs(tp_b - sl_s) < 1e-9 and abs(sl_b - tp_s) < 1e-9, \
        "BUY TP should equal SELL SL for symmetric targets"

    print("\nUSD TP/SL calculator smoke test passed.")
