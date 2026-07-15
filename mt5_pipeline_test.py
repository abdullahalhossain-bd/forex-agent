#!/usr/bin/env python3
"""
=====================================================
  MT5 Pipeline Test — Standalone Verification Script
=====================================================
Run this on your Windows machine to verify end-to-end:
  1. MT5 terminal connects
  2. Candle data fetches
  3. Live tick data reads
  4. Account info shows
  5. Test trade places (with auto-close for safety)

Usage (from project root, on Windows):
    python mt5_pipeline_test.py             # Full test (NO trade)
    python mt5_pipeline_test.py --trade     # Also place + close a 0.01 lot test trade

Requires:
    pip install MetaTrader5 pandas
    MT5 terminal running + logged into demo account
=====================================================
"""
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path so we can use the project's modules
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(str(PROJECT_ROOT / ".env"))
except ImportError:
    pass

# ─── Step 1: Import MetaTrader5 ──────────────────────────────
try:
    import MetaTrader5 as mt5
    import pandas as pd
except ImportError as e:
    print(f"\n[ERROR] Missing dependency: {e}")
    print("Install with:  pip install MetaTrader5 pandas")
    sys.exit(1)


def banner(text: str, char="=") -> None:
    line = char * 60
    print(f"\n{line}\n  {text}\n{line}")


def step(num: int, title: str) -> None:
    print(f"\n── Step {num}: {title} {'─' * (40 - len(title))}")


# ─── Step 2: Connect to MT5 ──────────────────────────────────
def connect_mt5() -> bool:
    step(1, "MT5 Connection")
    login = int(os.getenv("MT5_LOGIN", "0"))
    password = os.getenv("MT5_PASSWORD", "")
    server = os.getenv("MT5_SERVER", "")
    path = os.getenv("MT5_PATH", "") or None

    if not login or not password or not server:
        print("  [X] MT5 credentials missing in .env")
        print(f"      MT5_LOGIN={login}, MT5_PASSWORD={'***' if password else '(empty)'}, MT5_SERVER='{server}'")
        return False

    print(f"  Login:   {login}")
    print(f"  Server:  {server}")
    print(f"  Path:    {path or '(auto)'}")

    if not mt5.initialize(path=path) if path else not mt5.initialize():
        print(f"  [X] mt5.initialize() failed: {mt5.last_error()}")
        return False

    if not mt5.login(login=login, password=password, server=server):
        print(f"  [X] mt5.login() failed: {mt5.last_error()}")
        mt5.shutdown()
        return False

    print("  [v] Connected to MT5 terminal")
    return True


# ─── Step 3: Show account info ───────────────────────────────
def show_account() -> None:
    step(2, "Account Information")
    info = mt5.account_info()
    if info is None:
        print(f"  [X] account_info() failed: {mt5.last_error()}")
        return
    print(f"  Login:         {info.login}")
    print(f"  Balance:       ${info.balance:.2f}")
    print(f"  Equity:        ${info.equity:.2f}")
    print(f"  Margin:        ${info.margin:.2f}")
    print(f"  Free Margin:   ${info.margin_free:.2f}")
    print(f"  Margin Level:  {info.margin_level:.1f}%")
    print(f"  Currency:      {info.currency}")
    print(f"  Leverage:      1:{info.leverage}")
    print(f"  Trade Allowed: {info.trade_allowed}")


# ─── Step 4: Fetch candle data (your example code) ───────────
def fetch_candles(symbol: str = "EURUSD", timeframe_str: str = "M15", count: int = 10) -> None:
    step(3, f"Candle Data — {symbol} {timeframe_str} (last {count})")

    tf_map = {
        "M1":  mt5.TIMEFRAME_M1,
        "M5":  mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1":  mt5.TIMEFRAME_H1,
        "H4":  mt5.TIMEFRAME_H4,
        "D1":  mt5.TIMEFRAME_D1,
    }
    tf = tf_map.get(timeframe_str.upper(), mt5.TIMEFRAME_M15)

    if not mt5.symbol_select(symbol, True):
        print(f"  [X] symbol_select({symbol}) failed: {mt5.last_error()}")
        return

    rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
    if rates is None or len(rates) == 0:
        print(f"  [X] copy_rates_from_pos failed: {mt5.last_error()}")
        return

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df[["time", "open", "high", "low", "close", "tick_volume"]]
    df.rename(columns={"tick_volume": "volume"}, inplace=True)

    print(f"  Fetched {len(df)} candles")
    print("  ┌─" + "─" * 78 + "─┐")
    print(f"  │ {'Time':<20} {'Open':<10} {'High':<10} {'Low':<10} {'Close':<10} {'Volume':<10} │")
    print("  ├─" + "─" * 78 + "─┤")
    for _, row in df.tail(5).iterrows():
        print(f"  │ {str(row['time']):<20} {row['open']:<10.5f} {row['high']:<10.5f} "
              f"{row['low']:<10.5f} {row['close']:<10.5f} {int(row['volume']):<10} │")
    print("  └─" + "─" * 78 + "─┘")


# ─── Step 5: Live tick data (your example code) ──────────────
def fetch_tick(symbol: str = "EURUSD") -> None:
    step(4, f"Live Tick — {symbol}")
    tick = mt5.symbol_info_tick(symbol)
    if tick is None or tick.time == 0:
        print(f"  [X] symbol_info_tick({symbol}) failed")
        return

    info = mt5.symbol_info(symbol)
    digits = info.digits if info else 5
    spread_points = tick.ask - tick.bid
    spread_pips = round(spread_points * (10 ** (digits - 1)), 1) if digits else 0

    print(f"  Symbol:  {symbol}")
    print(f"  Bid:     {tick.bid:.{digits}f}")
    print(f"  Ask:     {tick.ask:.{digits}f}")
    print(f"  Spread:  {spread_pips} pips")
    print(f"  Time:    {datetime.fromtimestamp(tick.time, tz=timezone.utc).isoformat()}")


# ─── Step 6: Symbol info (filling mode detection) ───────────
def show_filling_mode(symbol: str = "EURUSD") -> int:
    step(5, f"Symbol Filling Mode — {symbol}")
    info = mt5.symbol_info(symbol)
    if info is None:
        print(f"  [X] symbol_info({symbol}) failed")
        return 0

    mode = info.filling_mode
    print(f"  filling_mode bitmask: {mode} (binary: {bin(mode)})")
    print(f"  FOK supported:     {'yes' if mode & 1 else 'NO'}")
    print(f"  IOC supported:     {'yes' if mode & 2 else 'NO'}")
    print(f"  RETURN supported:  {'yes' if mode & 4 else 'NO'}")

    # Pick the recommended filling mode
    if mode & 2:
        recommended = "ORDER_FILLING_IOC"
    elif mode & 1:
        recommended = "ORDER_FILLING_FOK"
    elif mode & 4:
        recommended = "ORDER_FILLING_RETURN"
    else:
        recommended = "ORDER_FILLING_IOC (default — broker reports no support)"

    print(f"  Recommended:       {recommended}")
    return mode


# ─── Step 7: Test trade (optional) ───────────────────────────
def place_test_trade(symbol: str = "EURUSD", lot: float = 0.01) -> None:
    step(6, f"Test Trade — {symbol} BUY {lot} lot (auto-close)")

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"  [X] No tick data")
        return

    info = mt5.symbol_info(symbol)
    pip = info.point * 10

    price = tick.ask
    sl = price - (25 * pip)   # 25 pips SL
    tp = price + (50 * pip)   # 50 pips TP

    # Auto-detect filling mode (the bug we just fixed)
    filling_mode = mt5.ORDER_FILLING_IOC
    if info and info.filling_mode & 2:
        filling_mode = mt5.ORDER_FILLING_IOC
    elif info and info.filling_mode & 1:
        filling_mode = mt5.ORDER_FILLING_FOK

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       lot,
        "type":         mt5.ORDER_TYPE_BUY,
        "price":        price,
        "sl":           sl,
        "tp":           tp,
        "deviation":    10,
        "magic":        234000,
        "comment":      "ai_pipeline_test",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": filling_mode,
    }
    print(f"  Order: BUY {symbol} lot={lot}")
    print(f"  Price: {price:.5f} | SL: {sl:.5f} | TP: {tp:.5f}")
    print(f"  Filling mode: {filling_mode}")

    result = mt5.order_send(request)
    if result is None:
        print(f"  [X] order_send returned None: {mt5.last_error()}")
        return

    if result.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
        print(f"  [v] Trade OPENED")
        print(f"      Ticket: {result.order}")
        print(f"      Price:  {result.price}")
        print(f"      Volume: {result.volume}")

        # Immediately close it (safety — we don't want test trades lingering)
        print(f"\n  Closing ticket {result.order} in 2 seconds...")
        time.sleep(2)

        close_tick = mt5.symbol_info_tick(symbol)
        close_request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       lot,
            "type":         mt5.ORDER_TYPE_SELL,
            "position":     result.order,
            "price":        close_tick.bid,
            "deviation":    10,
            "magic":        234000,
            "comment":      "ai_pipeline_test_close",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }
        close_result = mt5.order_send(close_request)
        if close_result and close_result.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
            print(f"  [v] Trade CLOSED — profit ≈ ${close_result.profit:.2f}")
        else:
            print(f"  [X] Close failed: retcode={close_result.retcode if close_result else 'None'}")
            print(f"      (Manually close ticket {result.order} in MT5 terminal)")
    else:
        print(f"  [X] Trade FAILED")
        print(f"      retcode: {result.retcode}")
        print(f"      comment: {result.comment}")
        if result.retcode == 10030:
            print(f"      → 10030 = Unsupported filling mode. Try changing ORDER_FILLING_FOK to ORDER_FILLING_IOC.")


# ─── Main ────────────────────────────────────────────────────
def main():
    banner("MT5 PIPELINE TEST")
    do_trade = "--trade" in sys.argv

    if not connect_mt5():
        sys.exit(1)

    try:
        show_account()
        fetch_candles("EURUSD", "M15", 10)
        fetch_tick("EURUSD")
        show_filling_mode("EURUSD")

        if do_trade:
            place_test_trade("EURUSD", 0.01)
        else:
            print(f"\n── Step 6: Test Trade (skipped — use --trade to enable) {'─' * 8}")

        banner("ALL CHECKS COMPLETE", "=")
        if not do_trade:
            print("  Run again with --trade to place + close a 0.01 lot test trade")
        print("  If all steps showed [v], your MT5 pipeline is working end-to-end.")
    finally:
        mt5.shutdown()
        print("\n  MT5 connection closed.")


if __name__ == "__main__":
    main()
