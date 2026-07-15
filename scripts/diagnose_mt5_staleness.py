#!/usr/bin/env python3
"""
Diagnostic: Check MT5 data freshness and broker timezone offset.

Run this on the Windows machine where MT5 terminal is installed:
    python scripts/diagnose_mt5_staleness.py

This will:
1. Connect to MT5
2. Check the latest bar timestamp for each configured symbol
3. Compare against true UTC now
4. Suggest the correct MT5_BROKER_TZ_OFFSET_HOURS value
5. Verify the MT5 terminal is receiving fresh ticks

If the bot's logs show "STALE DATA" warnings, this script will tell you WHY.
"""
import os
import sys
from datetime import datetime, timezone

# Load .env
from dotenv import load_dotenv
load_dotenv()

try:
    import MetaTrader5 as mt5
except ImportError:
    print("❌ MetaTrader5 not installed. Run this script on the Windows machine")
    print("   where MT5 terminal is installed.")
    sys.exit(1)

print("=" * 70)
print("  MT5 DATA FRESHNESS DIAGNOSTIC")
print("=" * 70)
print(f"  Python UTC now : {datetime.now(timezone.utc).isoformat()}")
print(f"  Server local   : {datetime.now().isoformat()}")
print(f"  Configured offset: MT5_BROKER_TZ_OFFSET_HOURS={os.getenv('MT5_BROKER_TZ_OFFSET_HOURS', '0')}")
print()

# Initialize MT5
login = int(os.getenv("MT5_LOGIN", "0"))
password = os.getenv("MT5_PASSWORD", "")
server = os.getenv("MT5_SERVER", "")
path = os.getenv("MT5_PATH", "")

print(f"  MT5 Login  : {login}")
print(f"  MT5 Server : {server}")
print(f"  MT5 Path   : {path}")
print()

if not mt5.initialize(path=path if path else None, login=login, password=password, server=server):
    print(f"❌ MT5 initialize() failed: {mt5.last_error()}")
    sys.exit(1)

print("✅ MT5 initialized")
terminal = mt5.terminal_info()
if terminal:
    print(f"  Terminal      : {terminal.name}")
    print(f"  Connected     : {terminal.connected}")
    print(f"  Trading allowed: {terminal.trade_allowed}")
    print(f"  Community balance: {terminal.community_balance}")
print()

account = mt5.account_info()
if account:
    print(f"  Account       : {account.login} ({account.server})")
    print(f"  Balance       : {account.balance} {account.currency}")
    print(f"  Equity        : {account.equity}")
    print(f"  Margin free   : {account.margin_free}")
print()

# Check symbols from .env
symbols_str = os.getenv("SYMBOLS", "EURUSD,GBPUSD,USDJPY,USDCHF,AUDUSD,USDCAD,NZDUSD,XAUUSD")
symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
print(f"  Symbols to check: {symbols}")
print()

now_utc = datetime.now(timezone.utc)
print(f"{'Symbol':<10} {'TF':<5} {'Last Bar (raw)':<28} {'Age (raw)':<12} {'Age (offset-corrected)':<25} {'Status'}")
print("-" * 110)

offset_hours = float(os.getenv("MT5_BROKER_TZ_OFFSET_HOURS", "0") or 0)

for sym in symbols:
    for tf_name, tf_const in [("M15", mt5.TIMEFRAME_M15), ("H1", mt5.TIMEFRAME_H1)]:
        rates = mt5.copy_rates_from_pos(sym, tf_const, 0, 5)
        if rates is None or len(rates) == 0:
            print(f"{sym:<10} {tf_name:<5} NO DATA — {mt5.last_error()}")
            continue

        last_bar_raw = datetime.fromtimestamp(rates[-1]["time"], timezone.utc)
        age_raw = (now_utc - last_bar_raw).total_seconds()

        # Apply offset correction
        last_bar_corrected = last_bar_raw - __import__("datetime").timedelta(hours=offset_hours)
        age_corrected = (now_utc - last_bar_corrected).total_seconds()

        if age_corrected < 0:
            status = "FUTURE_BAR (offset too high)"
        elif age_corrected > 960 and tf_name == "M15":
            status = "⚠️  STALE (>16min for M15)"
        elif age_corrected > 3660 and tf_name == "H1":
            status = "⚠️  STALE (>61min for H1)"
        else:
            status = "✅ Fresh"

        print(f"{sym:<10} {tf_name:<5} {last_bar_raw.isoformat():<28} {age_raw/60:>8.1f}m   {age_corrected/60:>8.1f}m ({offset_hours}h offset)   {status}")

print()
print("=" * 70)
print("  INTERPRETATION")
print("=" * 70)
print("""
1. If 'Age (raw)' is NEGATIVE for all symbols:
   → MT5_BROKER_TZ_OFFSET_HOURS is too LOW. The broker is ahead of UTC.
   → Increase the offset until 'Age (offset-corrected)' is small and positive.

2. If 'Age (offset-corrected)' is LARGE (>16min for M15) for all symbols:
   → The MT5 terminal is NOT receiving fresh ticks.
   → Check: Is the MT5 terminal window open and connected?
   → Check: Is the broker server reachable? (right-click symbol → Symbol window)
   → Check: Are the symbols subscribed? (right-click → Show All, then right-click → Subscribe)

3. If 'Age (offset-corrected)' is LARGE for only SOME symbols:
   → Those specific symbols are not subscribed or have no recent trades.
   → Right-click the symbol in Market Watch → Subscribe.

4. If 'Age (raw)' ≈ 'Age (offset-corrected)' (difference ≈ offset_hours * 60):
   → The offset is being applied correctly. The data IS genuinely stale.
   → Fix the MT5 terminal connection, not the offset.

5. If status shows '✅ Fresh' but the bot still logs STALE DATA:
   → The bot may be using a cached dataframe. Restart the bot.
""")

mt5.shutdown()
print("MT5 shutdown. Diagnostic complete.")
