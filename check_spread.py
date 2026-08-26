import MetaTrader5 as mt5


SYMBOL = "EURSEKm"


def get_actual_spread(symbol):
    # Connect to MT5
    if not mt5.initialize():
        print("❌ MT5 initialize failed")
        print("Error:", mt5.last_error())
        return

    # Make sure symbol is available
    if not mt5.symbol_select(symbol, True):
        print(f"❌ Cannot select {symbol}")
        mt5.shutdown()
        return

    info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)

    if info is None or tick is None:
        print(f"❌ No market data for {symbol}")
        mt5.shutdown()
        return

    # Raw spread
    raw_spread = tick.ask - tick.bid

    # Spread in points
    spread_points = raw_spread / info.point

    # Pip size
    # 5-digit / 3-digit symbols: 1 pip = 10 points
    if info.digits in (3, 5):
        pip_size = info.point * 10
    else:
        pip_size = info.point

    # Spread in pips
    spread_pips = raw_spread / pip_size

    print("=" * 65)
    print(f"MT5 ACTUAL SPREAD: {symbol}")
    print("=" * 65)

    print(f"Bid              : {tick.bid}")
    print(f"Ask              : {tick.ask}")
    print(f"Raw Spread       : {raw_spread}")
    print(f"Spread (points)  : {spread_points:.2f}")
    print(f"Spread (pips)    : {spread_pips:.2f}")
    print(f"Point size       : {info.point}")
    print(f"Digits           : {info.digits}")
    print(f"Contract size    : {info.trade_contract_size}")
    print(f"Tick size        : {info.trade_tick_size}")
    print("=" * 65)

    mt5.shutdown()


if __name__ == "__main__":
    get_actual_spread(SYMBOL)