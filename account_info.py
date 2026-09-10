"""
account_info.py — Read-only MT5 account diagnostic.

Prints the account's currency, balance, equity, margin, and leverage
exactly as MT5 reports them, plus a symbol_info() sanity check on
EURUSDc's tick value — so we can confirm what unit `balance` is
actually in (real USD vs Cent-account units) before trusting
INITIAL_BALANCE_USD in .env.

This makes NO trades and NO changes to your account. Safe to run.

Usage:
    python account_info.py

Credentials are read from .env (MT5_LOGIN, MT5_PASSWORD, MT5_SERVER,
MT5_PATH) — NOT hardcoded, unlike symbol.py. See .env.example.
"""

import os
import MetaTrader5 as mt5
from dotenv import load_dotenv

load_dotenv()

LOGIN = os.getenv("MT5_LOGIN")
PASSWORD = os.getenv("MT5_PASSWORD")
SERVER = os.getenv("MT5_SERVER")
PATH = os.getenv("MT5_PATH") or None

if not (LOGIN and PASSWORD and SERVER):
    print("❌ Missing MT5_LOGIN / MT5_PASSWORD / MT5_SERVER in .env")
    print("   Set these in your .env file (see .env.example) before running.")
    raise SystemExit(1)

# ============================================================
# CONNECT
# ============================================================
init_kwargs = dict(login=int(LOGIN), password=PASSWORD, server=SERVER)
if PATH:
    init_kwargs["path"] = PATH

if not mt5.initialize(**init_kwargs):
    print("❌ MT5 connection failed")
    print("Error:", mt5.last_error())
    raise SystemExit(1)

print("✅ Connected to MT5")
print()

# ============================================================
# ACCOUNT INFO
# ============================================================
acct = mt5.account_info()

if acct is None:
    print("❌ Failed to get account_info()")
    print("Error:", mt5.last_error())
    mt5.shutdown()
    raise SystemExit(1)

print("=" * 70)
print("MT5 ACCOUNT INFO")
print("=" * 70)
print(f"  Login             : {acct.login}")
print(f"  Server            : {SERVER}")
print(f"  Account currency  : {acct.currency}")
print(f"  Trade mode        : {acct.trade_mode}")
print(f"  Leverage          : 1:{acct.leverage}")
print(f"  Balance           : {acct.balance}")
print(f"  Equity            : {acct.equity}")
print(f"  Margin            : {acct.margin}")
print(f"  Margin free       : {acct.margin_free}")
print(f"  Margin level      : {acct.margin_level}")
print(f"  Credit            : {acct.credit}")

is_cent_like = str(acct.currency).upper() in ("USC", "CENT", "EURC", "GBPC")
print()
print(
    f"  -> account_info().currency = {acct.currency!r} — "
    f"{'looks like a CENT-unit currency code' if is_cent_like else 'looks like a standard currency code'}"
)

# ============================================================
# EURUSDc SYMBOL SANITY CHECK (pip value / tick value)
# ============================================================
SYMBOL = "EURUSDc"
print()
print("=" * 70)
print(f"{SYMBOL} SYMBOL INFO (for pip-value cross-check)")
print("=" * 70)

if not mt5.symbol_select(SYMBOL, True):
    print(f"❌ Could not select {SYMBOL}")
else:
    info = mt5.symbol_info(SYMBOL)
    tick = mt5.symbol_info_tick(SYMBOL)
    if info is None:
        print(f"❌ symbol_info({SYMBOL}) unavailable")
    else:
        print(f"  Digits            : {info.digits}")
        print(f"  Point             : {info.point}")
        print(f"  Contract size     : {info.trade_contract_size}")
        print(f"  Tick size         : {info.trade_tick_size}")
        print(f"  Tick value        : {info.trade_tick_value}")
        print(f"  Volume min        : {info.volume_min}")
        print(f"  Volume step       : {info.volume_step}")
        print(f"  Volume max        : {info.volume_max}")
        if tick:
            print(f"  Bid / Ask         : {tick.bid} / {tick.ask}")

        pip_size = 0.0001 if info.digits in (4, 5) else info.point
        if info.trade_tick_size and info.trade_tick_size > 0:
            pip_value_per_lot = info.trade_tick_value * (pip_size / info.trade_tick_size)
            print()
            print(f"  -> Computed pip value per 1.0 lot: {pip_value_per_lot:.4f} "
                  f"(in account currency units, i.e. {acct.currency})")
            print(f"  -> For a {info.volume_min}-lot trade with a 20-pip SL, "
                  f"risk = {info.volume_min * 20 * pip_value_per_lot:.4f} {acct.currency}")

# ============================================================
# SUMMARY / RECOMMENDATION
# ============================================================
print()
print("=" * 70)
print("WHAT TO SET INITIAL_BALANCE_USD TO")
print("=" * 70)
print(f"  Set INITIAL_BALANCE_USD={acct.balance:g} in your .env")
print(f"  (this is account_info().balance, exactly as MT5 reports it — ")
print(f"   the risk engine works in this SAME unit throughout, so it")
print(f"   must never be divided/multiplied by 100 anywhere.)")

mt5.shutdown()
print()
print("✅ MT5 disconnected")