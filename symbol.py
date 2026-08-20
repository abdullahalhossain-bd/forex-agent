import MetaTrader5 as mt5

# ============================================================
# MT5 CONFIG
# ============================================================
LOGIN = 434119023
PASSWORD = "Abdullah1@"
SERVER = "Exness-MT5Trial7"
PATH = r"C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe"


# ============================================================
# SETTINGS
# ============================================================
TIMEFRAMES = {
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
}

# Minimum bars we consider useful for testing
MIN_BARS = {
    "M15": 5000,
    "H1": 3000,
    "H4": 1000,
}


# ============================================================
# CONNECT
# ============================================================
if not mt5.initialize(
    path=PATH,
    login=LOGIN,
    password=PASSWORD,
    server=SERVER
):
    print("❌ MT5 connection failed")
    print("Error:", mt5.last_error())
    quit()

print("✅ Connected to Exness MT5")
print()


# ============================================================
# GET VISIBLE SYMBOLS
# ============================================================
all_symbols = mt5.symbols_get()

if all_symbols is None:
    print("❌ Failed to get symbols")
    print("Error:", mt5.last_error())
    mt5.shutdown()
    quit()


visible_symbols = [
    s for s in all_symbols
    if s.visible
]


# ============================================================
# FOREX FILTER
# ============================================================
CURRENCIES = {
    "USD", "EUR", "GBP", "JPY",
    "AUD", "NZD", "CAD", "CHF",
    "NOK", "SEK", "DKK", "PLN",
    "CZK", "HUF", "SGD", "HKD",
    "MXN", "ZAR", "TRY", "ILS"
}


def is_forex(symbol_name):

    name = symbol_name

    # Exness suffix
    if name.endswith("m"):
        name = name[:-1]

    if len(name) != 6:
        return False

    base = name[:3]
    quote = name[3:]

    return (
        base in CURRENCIES
        and quote in CURRENCIES
        and base != quote
    )


forex_symbols = [
    s for s in visible_symbols
    if is_forex(s.name)
]

forex_symbols.sort(key=lambda x: x.name)


# ============================================================
# HELPER
# ============================================================
def fmt(value, decimals=2):

    if value is None:
        return "N/A"

    try:
        return f"{value:.{decimals}f}"
    except:
        return str(value)


def get_clean_name(symbol_name):

    if symbol_name.endswith("m"):
        return symbol_name[:-1]

    return symbol_name


# ============================================================
# AUDIT
# ============================================================
results = []

print("=" * 150)
print("EXNESS FOREX SYMBOL AUDIT")
print("=" * 150)
print(f"Total Forex symbols: {len(forex_symbols)}")
print()


for index, symbol in enumerate(forex_symbols, 1):

    name = symbol.name
    clean_name = get_clean_name(name)

    # --------------------------------------------------------
    # Symbol info
    # --------------------------------------------------------
    info = mt5.symbol_info(name)

    if info is None:
        print(f"{index:3}. {name:<12} ❌ symbol_info unavailable")
        continue

    # --------------------------------------------------------
    # Tick
    # --------------------------------------------------------
    tick = mt5.symbol_info_tick(name)

    if tick:

        bid = tick.bid
        ask = tick.ask

        if bid and ask and info.point:
            spread_points = (ask - bid) / info.point
        else:
            spread_points = None

    else:
        bid = None
        ask = None
        spread_points = None


    # --------------------------------------------------------
    # Historical bars
    # --------------------------------------------------------
    bars = {}

    for tf_name, tf in TIMEFRAMES.items():

        rates = mt5.copy_rates_from_pos(
            name,
            tf,
            0,
            MIN_BARS[tf_name]
        )

        if rates is None:
            bars[tf_name] = 0
        else:
            bars[tf_name] = len(rates)


    # --------------------------------------------------------
    # Trade mode
    # --------------------------------------------------------
    trade_mode = info.trade_mode

    if trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
        trade_status = "FULL"

    elif trade_mode == mt5.SYMBOL_TRADE_MODE_LONGONLY:
        trade_status = "LONG_ONLY"

    elif trade_mode == mt5.SYMBOL_TRADE_MODE_SHORTONLY:
        trade_status = "SHORT_ONLY"

    elif trade_mode == mt5.SYMBOL_TRADE_MODE_CLOSEONLY:
        trade_status = "CLOSE_ONLY"

    else:
        trade_status = "DISABLED"


    # --------------------------------------------------------
    # Data status
    # --------------------------------------------------------
    m15_ok = bars["M15"] >= MIN_BARS["M15"]
    h1_ok = bars["H1"] >= MIN_BARS["H1"]
    h4_ok = bars["H4"] >= MIN_BARS["H4"]

    data_ok = m15_ok and h1_ok and h4_ok


    # --------------------------------------------------------
    # Spread status
    # --------------------------------------------------------
    if spread_points is None:

        spread_status = "UNKNOWN"

    elif spread_points <= 20:

        spread_status = "GOOD"

    elif spread_points <= 50:

        spread_status = "MEDIUM"

    else:

        spread_status = "HIGH"


    # --------------------------------------------------------
    # Overall classification
    # --------------------------------------------------------
    if (
        trade_status == "FULL"
        and data_ok
        and spread_points is not None
        and spread_points <= 50
    ):

        status = "✅ TRADEABLE"

    elif trade_status == "DISABLED":

        status = "❌ DISABLED"

    elif not data_ok:

        status = "⚠️ LOW DATA"

    elif spread_points is not None and spread_points > 50:

        status = "⚠️ HIGH SPREAD"

    else:

        status = "⚠️ REVIEW"


    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------
    results.append({
        "symbol": name,
        "clean": clean_name,
        "spread": spread_points,
        "bid": bid,
        "ask": ask,
        "digits": info.digits,
        "point": info.point,
        "min_lot": info.volume_min,
        "lot_step": info.volume_step,
        "max_lot": info.volume_max,
        "contract": info.trade_contract_size,
        "tick_value": info.trade_tick_value,
        "tick_size": info.trade_tick_size,
        "trade_mode": trade_status,
        "M15": bars["M15"],
        "H1": bars["H1"],
        "H4": bars["H4"],
        "spread_status": spread_status,
        "status": status,
    })


    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------
    print(
        f"{index:3}. "
        f"{name:<12} | "
        f"Spread: {fmt(spread_points, 1):>7} pts | "
        f"M15: {bars['M15']:>5} | "
        f"H1: {bars['H1']:>5} | "
        f"H4: {bars['H4']:>5} | "
        f"Mode: {trade_status:<10} | "
        f"{status}"
    )


# ============================================================
# SUMMARY
# ============================================================
print()
print("=" * 150)
print("SUMMARY")
print("=" * 150)

tradeable = [
    r for r in results
    if r["status"] == "✅ TRADEABLE"
]

low_data = [
    r for r in results
    if r["status"] == "⚠️ LOW DATA"
]

high_spread = [
    r for r in results
    if r["status"] == "⚠️ HIGH SPREAD"
]

disabled = [
    r for r in results
    if r["status"] == "❌ DISABLED"
]

review = [
    r for r in results
    if r["status"] == "⚠️ REVIEW"
]


print(f"Total Forex symbols : {len(results)}")
print(f"✅ Tradeable        : {len(tradeable)}")
print(f"⚠️ Low Data         : {len(low_data)}")
print(f"⚠️ High Spread      : {len(high_spread)}")
print(f"⚠️ Review            : {len(review)}")
print(f"❌ Disabled         : {len(disabled)}")


# ============================================================
# TRADEABLE SYMBOLS
# ============================================================
print()
print("=" * 80)
print("✅ TRADEABLE SYMBOLS")
print("=" * 80)

for i, r in enumerate(tradeable, 1):
    print(
        f"{i:3}. {r['symbol']:<12} "
        f"Spread={fmt(r['spread'], 1)} pts | "
        f"M15={r['M15']} | "
        f"H1={r['H1']} | "
        f"H4={r['H4']}"
    )


# ============================================================
# LOW DATA
# ============================================================
if low_data:

    print()
    print("=" * 80)
    print("⚠️ LOW DATA SYMBOLS")
    print("=" * 80)

    for r in low_data:

        print(
            f"{r['symbol']:<12} "
            f"M15={r['M15']} | "
            f"H1={r['H1']} | "
            f"H4={r['H4']}"
        )


# ============================================================
# HIGH SPREAD
# ============================================================
if high_spread:

    print()
    print("=" * 80)
    print("⚠️ HIGH SPREAD SYMBOLS")
    print("=" * 80)

    for r in high_spread:

        print(
            f"{r['symbol']:<12} "
            f"Spread={fmt(r['spread'], 1)} pts"
        )


# ============================================================
# SHUTDOWN
# ============================================================
mt5.shutdown()

print()
print("✅ MT5 disconnected")