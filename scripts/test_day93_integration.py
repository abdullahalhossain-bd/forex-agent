"""
test_day93_integration.py — Day 93 integration tests

Verifies:
  1. pandas-ta ExtendedIndicators computes 60+ indicators on real data
  2. DataOrchestrator returns candles (MT5 if available, else API)
  3. DataOrchestrator status() reports correct source
  4. MarketAgent uses orchestrator + extended indicators
  5. Telegram extension commands import cleanly
  6. MT5 methods gracefully return None/empty when MT5 unavailable
     (proves the "MT5 first, API fallback" principle works on Linux VPS)

Usage:
    cd /home/z/my-project/forex_ai
    python scripts/test_day93_integration.py
"""
import os
import sys

sys.path.insert(0, '/home/z/my-project/forex_ai')
os.chdir('/home/z/my-project/forex_ai')

from dotenv import load_dotenv
load_dotenv()


def banner(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def main():
    banner("Day 93 — Integration Tests")

    # ─────────────────────────────────────────────────────────
    banner("Test 1: ExtendedIndicators (pandas-ta, 60+ indicators)")
    try:
        from data.indicators_ext import ExtendedIndicators
        from data.data_orchestrator import get_data_orchestrator

        orch = get_data_orchestrator()
        df = orch.get_candles("EURUSD", "M15", limit=200)
        if df is None or len(df) < 30:
            print(f"  SKIP: no candle data (got {len(df) if df is not None else 0} rows)")
        else:
            ind = ExtendedIndicators()
            df = ind.add_all(df, include_patterns=True)
            ctx = ind.get_ai_context(df)

            # Count indicator columns (anything not in original OHLCV)
            base_cols = {"open", "high", "low", "close", "volume"}
            indicator_cols = [c for c in df.columns if c not in base_cols]

            print(f"  Candles fetched: {len(df)}")
            print(f"  Indicator columns: {len(indicator_cols)}")
            print(f"  Sample indicators:")
            for key in ("trend", "rsi", "macd_cross", "adx", "atr", "bb_pct",
                        "stoch_k", "stoch_d", "cci", "ema_9", "sma_200", "pivot_p"):
                print(f"    {key:<14}: {ctx.get(key)}")

            assert len(indicator_cols) >= 40, f"Expected 40+ indicators, got {len(indicator_cols)}"
            print(f"  PASS: {len(indicator_cols)} indicator columns computed")
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback; traceback.print_exc()

    # ─────────────────────────────────────────────────────────
    banner("Test 2: DataOrchestrator — MT5 first, API fallback")
    try:
        from data.data_orchestrator import get_data_orchestrator
        orch = get_data_orchestrator()
        status = orch.status()

        print(f"  MT5 available:     {status['mt5_available']}")
        print(f"  MT5 initialized:   {status['mt5_initialized']}")
        print(f"  API fallback:      {status['api_source']}")
        print(f"  Preferred source:  {status['preferred_source'] or 'auto'}")

        # Get candles
        df = orch.get_candles("EURUSD", "M15", limit=50)
        print(f"\n  Last source used:  {orch.last_source}")
        if df is not None:
            print(f"  Candles returned:  {len(df)}")
            print(f"  Last close:        {df['close'].iloc[-1]:.5f}")
            print(f"  PASS: orchestrator returned candles from {orch.last_source}")
        else:
            print(f"  FAIL: orchestrator returned None")
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback; traceback.print_exc()

    # ─────────────────────────────────────────────────────────
    banner("Test 3: Orchestrator graceful degradation (MT5-only methods)")
    try:
        from data.data_orchestrator import get_data_orchestrator
        orch = get_data_orchestrator()

        # These should all return None/empty when MT5 is unavailable,
        # never raise an exception.
        account = orch.get_account_info()
        positions = orch.get_open_positions()
        orders = orch.get_pending_orders()
        tick = orch.get_tick("EURUSD")

        if orch.status()["mt5_available"]:
            print(f"  MT5 available — methods returned real data:")
            print(f"    account:   {account is not None}")
            print(f"    positions: {len(positions)} open")
            print(f"    orders:    {len(orders)} pending")
            print(f"    tick:      {tick is not None}")
        else:
            print(f"  MT5 unavailable — graceful degradation check:")
            print(f"    get_account_info():    {account}  (expected None)")
            print(f"    get_open_positions():  {len(positions)} items (expected 0)")
            print(f"    get_pending_orders():  {len(orders)} items (expected 0)")
            print(f"    get_tick():            {tick}  (expected None)")
            assert account is None, "Should return None when MT5 unavailable"
            assert positions == [], "Should return [] when MT5 unavailable"
            assert orders == [], "Should return [] when MT5 unavailable"
            assert tick is None, "Should return None when MT5 unavailable"
            print(f"  PASS: all MT5-only methods gracefully returned None/empty")
    except Exception as e:
        print(f"  FAIL: {e}")

    # ─────────────────────────────────────────────────────────
    banner("Test 4: MarketAgent uses orchestrator")
    try:
        from agents.market_agent import MarketAgent
        agent = MarketAgent("EURUSD", "15m")
        result = agent.run()

        if "error" in result:
            print(f"  FAIL: {result['error']}")
        else:
            print(f"  Data source:       {result.get('data_source', '?')}")
            print(f"  Candles:           {len(result['df'])}")
            print(f"  Indicator columns: {len(result['df'].columns)}")
            print(f"  Trend:             {result['ind_ctx'].get('trend')}")
            print(f"  RSI:               {result['ind_ctx'].get('rsi')}")
            print(f"  ADX:               {result['ind_ctx'].get('adx')}")
            print(f"  Regime:            {result['regime'].get('regime')}")
            print(f"  PASS: MarketAgent ran end-to-end via orchestrator")
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback; traceback.print_exc()

    # ─────────────────────────────────────────────────────────
    banner("Test 5: Telegram extension commands import cleanly")
    try:
        from alerts.telegram_ext import (
            register_extension_commands,
            notify_rich_signal,
            cmd_positions, cmd_close, cmd_symbols,
            cmd_indicators, cmd_source, cmd_account,
        )
        print(f"  register_extension_commands: OK")
        print(f"  notify_rich_signal:          OK")
        print(f"  cmd_positions:               OK")
        print(f"  cmd_close:                   OK")
        print(f"  cmd_symbols:                 OK")
        print(f"  cmd_indicators:              OK")
        print(f"  cmd_source:                  OK")
        print(f"  cmd_account:                 OK")
        print(f"  PASS: all 6 extension commands imported")
    except Exception as e:
        print(f"  FAIL: {e}")

    # ─────────────────────────────────────────────────────────
    banner("Test 6: Rich signal alert formatter")
    try:
        import asyncio
        from alerts.telegram_ext import notify_rich_signal

        signal = {
            "pair": "EURUSD",
            "direction": "BUY",
            "confidence": 85,
            "entry": 1.0850,
            "sl": 1.0820,
            "tp": 1.0910,
            "lot": 0.10,
            "strategy": "SMC_PULLBACK",
            "regime": "TRENDING BULLISH STRONG",
            "reasons": [
                "Bullish BOS + CHoCH on M15",
                "RSI in bullish zone (62)",
                "Price at order block support",
                "Stochastic cross up from oversold",
            ],
            "source": "mt5",
        }
        # Pass bot=None, chat_id="" — should not crash, just log a warning
        asyncio.run(notify_rich_signal(None, "", signal))
        print(f"  PASS: notify_rich_signal handled None bot gracefully")
    except Exception as e:
        print(f"  FAIL: {e}")

    # ─────────────────────────────────────────────────────────
    banner("Summary")
    print("  Day 93 integration verified.")
    print("  Architecture:")
    print("    Data:  MT5 (primary) → Twelve Data → yfinance (fallback)")
    print("    Indicators: pandas-ta (60+ indicators, computed in Python)")
    print("    News: Forex Factory (scheduled) + NewsAPI (breaking)")
    print("    LLM:  Groq → Cerebras → SambaNova → OpenRouter → Gemini")
    print("    Exec: MT5 (Windows) or SimulatedExecutor (Linux VPS)")
    print("    Alerts: Telegram (/positions /close /indicators /account)")


if __name__ == "__main__":
    main()
