"""
diag_mt5.py - MT5 Terminal Bar Limit Diagnostic Tool
=====================================================

This script helps identify the exact "Max bars in history" limit of your
MT5 terminal configuration. Run this before training to determine the
optimal CHUNK_SIZE for your setup.

Usage:
    python diag_mt5.py
"""

import MetaTrader5 as mt5
import sys


def main():
    print("=" * 70)
    print("MT5 Terminal Bar Limit Diagnostic")
    print("=" * 70)
    
    # Initialize MT5
    if not mt5.initialize():
        print(f"❌ initialize failed: {mt5.last_error()}")
        sys.exit(1)
    
    print("\n✅ MT5 initialized successfully")
    
    # Get terminal info
    terminal_info = mt5.terminal_info()
    if terminal_info:
        print(f"\nTerminal Info:")
        print(f"  Path: {terminal_info.path}")
        print(f"  Connected: {terminal_info.connected}")
        print(f"  Build: {terminal_info.build}")
        print(f"  Company: {terminal_info.company}")
    
    # Test symbol
    symbol = "EURUSD"
    print(f"\n{'=' * 70}")
    print(f"Testing Symbol: {symbol}")
    print(f"{'=' * 70}")
    
    # Select symbol
    if not mt5.symbol_select(symbol, True):
        print(f"❌ Failed to select {symbol}: {mt5.last_error()}")
        mt5.shutdown()
        sys.exit(1)
    
    print(f"✅ Symbol {symbol} selected")
    
    # Get symbol info
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info:
        print(f"\nSymbol Info:")
        print(f"  Visible: {symbol_info.visible}")
        print(f"  Spread: {symbol_info.spread}")
        print(f"  Digits: {symbol_info.digits}")
        print(f"  Trade Mode: {symbol_info.trade_mode}")
    
    # Test different bar counts
    test_sizes = [10, 100, 1000, 5000, 10000, 20000, 50000, 100000, 150000]
    timeframe = mt5.TIMEFRAME_M15
    
    print(f"\n{'=' * 70}")
    print(f"Testing copy_rates_from_pos with M15 timeframe")
    print(f"{'=' * 70}\n")
    
    results = []
    max_success = 0
    
    for n in test_sizes:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
        
        if rates is None:
            error = mt5.last_error()
            print(f"❌ bars={n:>6} -> FAILED (error={error})")
            results.append((n, False, error))
        else:
            got = len(rates)
            status = "✅" if got == n else "⚠️"
            print(f"{status} bars={n:>6} -> OK (got {got} rows)")
            results.append((n, True, got))
            if got == n:
                max_success = n
            else:
                print(f"   ⚠️  Requested {n} but only got {got} - limit reached!")
    
    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    
    if max_success == 0:
        print("\n❌ No successful fetches! Check MT5 connection and symbol availability.")
    else:
        print(f"\n✅ Maximum successful fetch: {max_success:,} bars")
        
        # Find the failure point
        failure_point = None
        for n, success, _ in results:
            if not success:
                failure_point = n
                break
        
        if failure_point:
            print(f"❌ First failure at: {failure_point:,} bars")
            print(f"\n💡 RECOMMENDATION:")
            print(f"   Set CHUNK_SIZE = {min(max_success, 5000)} in mt5_data_loader.py")
            print(f"   (Using 80% of max_success for safety margin)")
            
            # Suggest terminal setting change
            print(f"\n💡 ALTERNATIVE: Increase terminal limit")
            print(f"   Tools → Options → Charts → 'Max bars in chart'")
            print(f"   Set to 'Unlimited' or at least {max_success + 50000}")
        else:
            print(f"✅ All tests passed up to {max(test_sizes):,} bars!")
            print(f"\n💡 Your terminal can handle large fetches. CHUNK_SIZE = 5000 is safe.")
    
    print(f"\n{'=' * 70}")
    
    # Cleanup
    mt5.shutdown()
    print("MT5 shutdown complete")


if __name__ == "__main__":
    main()
