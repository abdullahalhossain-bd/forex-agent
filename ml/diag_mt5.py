# diag_mt5.py
import MetaTrader5 as mt5

if not mt5.initialize():
    print("initialize failed:", mt5.last_error())
    quit()

print("terminal:", mt5.terminal_info())

symbol = "EURUSD"
print("symbol_select:", mt5.symbol_select(symbol, True))
print("symbol_info:", mt5.symbol_info(symbol))

for n in [10, 100, 1000, 5000, 20000, 100000]:
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, n)
    if rates is None:
        print(f"bars={n} -> FAILED, error={mt5.last_error()}")
    else:
        print(f"bars={n} -> OK, got {len(rates)} rows")

mt5.shutdown()