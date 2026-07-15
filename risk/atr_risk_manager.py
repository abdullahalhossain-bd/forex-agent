# risk/atr_risk_manager.py — ATR-based SL/TP + position sizing
# =============================================================================
# Ported from: https://github.com/bruh7463/forex_bot/blob/master/risk/manager.py
# Original author: bruh7463 — educational project
#
# ATR-based risk management:
#   - Stop Loss = entry ± (sl_atr_mult × ATR)
#   - Take Profit = entry ± (tp_atr_mult × ATR)
#   - Position size = (balance × risk_pct) / (sl_pips × pip_value_per_lot)
#
# The ATR (Average True Range) adapts to volatility — wide SL in volatile
# markets, tight SL in calm markets. This is superior to fixed-pip SL/TP
# which gets stopped out too easily in volatile conditions and leaves money
# on the table in calm conditions.
#
# Default risk parameters (matching the original):
#   - Risk per trade: 1% of balance
#   - SL: 2 × ATR
#   - TP: 3 × ATR (1:1.5 reward:risk)
#
# The position sizing converts the SL distance to pips, then uses the
# standard $10/lot/pip approximation for forex.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass

from utils.logger import get_logger

log = get_logger("atr_risk_manager")

# Standard pip sizes
PIP_SIZES = {
    "JPY": 0.01,   # JPY pairs: USDJPY, EURJPY, etc.
    "DEFAULT": 0.0001,  # All other pairs
}

# Approximate pip value per standard lot (100,000 units) in USD
PIP_VALUE_PER_LOT = 10.0  # $10 per pip per standard lot for most (non-JPY) pairs

# P1 fix (audit §4.5): this module previously hardcoded PIP_VALUE_PER_LOT
# (10.0) for every pair, including JPY crosses, where pip value per
# standard lot is NOT $10 at typical JPY price levels (it scales with
# price — live_risk_manager.py already uses the convention of 9.0 for
# JPY pairs when calling position_sizer.py; this mirrors that same
# convention here so atr_risk_manager.py's own standalone sizing isn't
# silently wrong for JPY pairs).
PIP_VALUE_PER_LOT_JPY = 9.0


def _get_pip_value_per_lot(pair: str) -> float:
    """Pair-conditional pip value per standard lot (JPY vs non-JPY)."""
    return PIP_VALUE_PER_LOT_JPY if "JPY" in pair.upper() else PIP_VALUE_PER_LOT


@dataclass
class TradeRiskParams:
    """Container for calculated risk parameters."""
    signal: str
    entry_price: float
    stop_loss: float
    take_profit: float
    sl_pips: float
    tp_pips: float
    position_size: float       # in lots (MT5) or units (OANDA)
    risk_amount: float         # in account currency
    risk_reward_ratio: float
    atr: float
    sl_atr_mult: float
    tp_atr_mult: float


def _get_pip_size(pair: str) -> float:
    """Get pip size for a pair (0.01 for JPY pairs, 0.0001 otherwise)."""
    return PIP_SIZES["JPY"] if "JPY" in pair.upper() else PIP_SIZES["DEFAULT"]


def get_stop_loss(
    signal: str,
    current_price: float,
    atr: float,
    sl_atr_mult: float = 2.0,
) -> float:
    """
    Calculate stop loss price based on ATR.

    BUY:  SL = price - (sl_atr_mult × ATR)
    SELL: SL = price + (sl_atr_mult × ATR)
    HOLD: SL = price (no trade)
    """
    if signal == "BUY":
        sl = current_price - (sl_atr_mult * atr)
    elif signal == "SELL":
        sl = current_price + (sl_atr_mult * atr)
    else:
        sl = current_price
    return round(sl, 5)


def get_take_profit(
    signal: str,
    current_price: float,
    atr: float,
    tp_atr_mult: float = 3.0,
) -> float:
    """
    Calculate take profit price based on ATR.

    BUY:  TP = price + (tp_atr_mult × ATR)
    SELL: TP = price - (tp_atr_mult × ATR)
    HOLD: TP = price (no trade)
    """
    if signal == "BUY":
        tp = current_price + (tp_atr_mult * atr)
    elif signal == "SELL":
        tp = current_price - (tp_atr_mult * atr)
    else:
        tp = current_price
    return round(tp, 5)


def calculate_position_size(
    account_balance: float,
    risk_pct: float,
    stop_loss_pips: float,
    pair: str = "EURUSD",
    *,
    return_units: bool = False,
) -> float:
    """
    Calculate position size based on account balance and risk.

    Parameters
    ----------
    account_balance : current account balance in account currency.
    risk_pct : risk per trade as a fraction (0.01 = 1%).
    stop_loss_pips : SL distance in pips.
    pair : currency pair (determines pip value for JPY vs non-JPY).
    return_units : if True, return units (for OANDA); if False, return lots
        (for MT5). Default False (MT5).

    Returns
    -------
    Position size in lots (MT5) or units (OANDA).
    """
    if stop_loss_pips <= 0:
        raise ValueError("stop_loss_pips must be > 0")
    if not 0 < risk_pct <= 1:
        raise ValueError("risk_pct must be between 0 and 1")

    risk_amount = account_balance * risk_pct
    pip_value_per_lot = _get_pip_value_per_lot(pair)
    position_size_lots = risk_amount / (stop_loss_pips * pip_value_per_lot)

    if return_units:
        # OANDA: 1 lot = 100,000 units
        return max(1, int(position_size_lots * 100_000))
    else:
        # MT5: lots as float, min 0.01
        return max(0.01, round(position_size_lots, 2))


def calculate_risk_reward(
    entry_price: float,
    stop_loss: float,
    take_profit: float,
) -> float:
    """Calculate reward:risk ratio (e.g., 1.5 = reward is 1.5x the risk)."""
    risk = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)
    if risk == 0:
        return 0.0
    return round(reward / risk, 2)


def compute_trade_params(
    signal: str,
    current_price: float,
    atr: float,
    account_balance: float,
    pair: str = "EURUSD",
    *,
    risk_pct: float = 0.01,
    sl_atr_mult: float = 2.0,
    tp_atr_mult: float = 3.0,
    return_units: bool = False,
) -> TradeRiskParams:
    """
    All-in-one: compute SL, TP, position size, and R:R for a trade.

    Parameters
    ----------
    signal : "BUY", "SELL", or "HOLD".
    current_price : current market price.
    atr : Average True Range value.
    account_balance : account balance in account currency.
    pair : currency pair (for pip size).
    risk_pct : risk per trade (default 1%).
    sl_atr_mult : SL distance in ATR multiples (default 2.0).
    tp_atr_mult : TP distance in ATR multiples (default 3.0).
    return_units : True for OANDA (units), False for MT5 (lots).

    Returns
    -------
    TradeRiskParams dataclass with all computed values.
    """
    sl = get_stop_loss(signal, current_price, atr, sl_atr_mult)
    tp = get_take_profit(signal, current_price, atr, tp_atr_mult)

    pip_size = _get_pip_size(pair)
    sl_pips = abs(current_price - sl) / pip_size
    tp_pips = abs(tp - current_price) / pip_size

    size = calculate_position_size(
        account_balance, risk_pct, sl_pips, pair, return_units=return_units
    )

    rr = calculate_risk_reward(current_price, sl, tp)
    risk_amount = account_balance * risk_pct

    return TradeRiskParams(
        signal=signal,
        entry_price=current_price,
        stop_loss=sl,
        take_profit=tp,
        sl_pips=round(sl_pips, 1),
        tp_pips=round(tp_pips, 1),
        position_size=size,
        risk_amount=round(risk_amount, 2),
        risk_reward_ratio=rr,
        atr=atr,
        sl_atr_mult=sl_atr_mult,
        tp_atr_mult=tp_atr_mult,
    )


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # BUY on EURUSD
    params = compute_trade_params(
        signal="BUY", current_price=1.10000, atr=0.00080,
        account_balance=10000.0, pair="EURUSD",
        risk_pct=0.01, sl_atr_mult=2.0, tp_atr_mult=3.0,
    )
    print("=== EURUSD BUY ===")
    print(f"  Entry:     {params.entry_price:.5f}")
    print(f"  SL:        {params.stop_loss:.5f} ({params.sl_pips:.1f} pips)")
    print(f"  TP:        {params.take_profit:.5f} ({params.tp_pips:.1f} pips)")
    print(f"  Size:      {params.position_size} lots")
    print(f"  Risk:      ${params.risk_amount:.2f}")
    print(f"  R:R:       1:{params.risk_reward_ratio}")

    assert params.stop_loss == 1.09840  # 1.10000 - 2*0.00080
    assert params.take_profit == 1.10240  # 1.10000 + 3*0.00080
    assert abs(params.sl_pips - 16.0) < 0.1  # 2*0.00080/0.0001 = 16 pips
    assert abs(params.tp_pips - 24.0) < 0.1
    assert params.risk_reward_ratio == 1.5  # 24/16 = 1.5

    # SELL on USDJPY (JPY pip = 0.01)
    params_jpy = compute_trade_params(
        signal="SELL", current_price=150.000, atr=0.800,
        account_balance=10000.0, pair="USDJPY",
        risk_pct=0.01, sl_atr_mult=2.0, tp_atr_mult=3.0,
    )
    print("\n=== USDJPY SELL ===")
    print(f"  Entry:     {params_jpy.entry_price:.3f}")
    print(f"  SL:        {params_jpy.stop_loss:.3f} ({params_jpy.sl_pips:.1f} pips)")
    print(f"  TP:        {params_jpy.take_profit:.3f} ({params_jpy.tp_pips:.1f} pips)")
    print(f"  Size:      {params_jpy.position_size} lots")

    assert params_jpy.stop_loss == 151.600  # 150 + 2*0.800
    assert params_jpy.take_profit == 147.600  # 150 - 3*0.800
    # sl_pips = 1.600 / 0.01 = 160 pips
    assert abs(params_jpy.sl_pips - 160.0) < 0.1

    # Position size check: risk $100, 16 pips SL, $10/lot/pip → 100/(16*10) = 0.625
    # Rounded to 2 decimals: 0.62 (banker's rounding) or 0.63 (standard) — both OK
    params_eurusd = compute_trade_params(
        "BUY", 1.10000, 0.00080, 10000.0, "EURUSD"
    )
    print(f"\nEURUSD size: {params_eurusd.position_size} (expected ~0.62-0.63)")
    assert 0.60 <= params_eurusd.position_size <= 0.65

    # OANDA units
    params_oanda = compute_trade_params(
        "BUY", 1.10000, 0.00080, 10000.0, "EURUSD", return_units=True
    )
    print(f"OANDA units: {params_oanda.position_size}")
    assert params_oanda.position_size >= 1  # at least 1 unit

    # HOLD signal — SL and TP = entry price (no trade)
    sl_hold = get_stop_loss("HOLD", 1.10000, 0.00080)
    tp_hold = get_take_profit("HOLD", 1.10000, 0.00080)
    assert sl_hold == 1.10000
    assert tp_hold == 1.10000
    print(f"\nHOLD: SL={sl_hold:.5f}, TP={tp_hold:.5f} (no trade)")

    print("\nATR risk manager smoke test passed.")