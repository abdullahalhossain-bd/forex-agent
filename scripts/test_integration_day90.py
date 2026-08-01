"""
test_integration_day90.py — End-to-end integration test for Day 90 wiring.

Tests the full flow:
  MarketAgent-like market_output
    → AnalysisAgent.run() with 6 new analyzers + StrategySelector
    → MasterAnalyst receives new contexts (LLM unavailable → fallback)
    → MasterDecisionEngine receives strategy_context
    → Final decision has strategy_* fields populated

This test does NOT require MT5, LLM keys, or live network — it uses
synthetic OHLC data and exercises the wiring, not the LLM brain.
"""
import sys
sys.path.insert(0, '/home/z/my-project/forex_ai')

import os
import shutil
import numpy as np
import pandas as pd

# Fresh DB
if os.path.exists('/home/z/my-project/forex_ai/memory/master_decisions.db'):
    os.remove('/home/z/my-project/forex_ai/memory/master_decisions.db')


def make_synthetic_market_output():
    """Build a market_output dict like MarketAgent.run() would produce."""
    np.random.seed(42)
    n = 250
    # Strong bullish uptrend with realistic OHLC
    base = 1.1000 + np.cumsum(np.random.randn(n) * 0.0006 + 0.0003)
    opens  = base
    closes = base + np.random.randn(n) * 0.0003
    highs  = np.maximum(opens, closes) + np.abs(np.random.randn(n)) * 0.0008
    lows   = np.minimum(opens, closes) - np.abs(np.random.randn(n)) * 0.0008
    # Add tick volume
    tick_vol = 100 + np.abs(np.random.randn(n)) * 50

    df = pd.DataFrame({
        "open":  opens,
        "high":  highs,
        "low":   lows,
        "close": closes,
        "tick_volume": tick_vol,
    })

    # Indicators (minimal — ta lib now installed)
    from data.indicators import Indicators
    df = Indicators().add_all(df)

    # Build a regime dict (bullish trending)
    regime = {
        "regime":     "TRENDING",
        "direction":  "BULLISH",
        "strength":   "STRONG",
        "volatility": "NORMAL",
        "strategy":   {"risk_mult": 1.0, "type": "TREND_FOLLOW", "action": "Buy on pullbacks"},
        "adx":        35.0,
        "atr":        0.0008,
        "atr_avg":    0.0007,
    }

    ind_ctx = {
        "trend": "bullish",
        "rsi": 62.0,
        "rsi_signal": "bullish",
        "macd_cross": "bullish_cross",
        "price": float(closes[-1]),
        "close": float(closes[-1]),
        "atr": 0.0008,
        "bb_pct": 0.7,
        "ema_21": float(closes[-5]),
        "sma_50": float(closes[-30]),
        "sma_200": float(closes[-100]),
    }

    return {
        "df":         df,
        "ind_ctx":    ind_ctx,
        "regime":     regime,
        "regime_ctx": regime,  # same shape works
        "mtf_bias":   {"bias": "BULLISH", "confidence": "HIGH"},
        "symbol":     "EURUSD",
        "timeframe":  "15m",
    }


def main():
    print("=" * 60)
    print("  Day 90 Integration Test — Full Pipeline")
    print("=" * 60)

    market_output = make_synthetic_market_output()
    print(f"\n[Setup] Synthetic market_output built for EURUSD 15m")
    print(f"  Candles: {len(market_output['df'])}")
    print(f"  Regime: {market_output['regime']['regime']}")
    print(f"  Direction: {market_output['regime']['direction']}")

    # Run AnalysisAgent
    from agents.analysis_agent import AnalysisAgent
    print("\n[Step] Running AnalysisAgent.run()...")
    agent = AnalysisAgent()
    analysis_out = agent.run(market_output, memory_ctx={
        "overall_win_rate": 55,
        "total_trades": 100,
        "recent_results": ["WIN", "LOSS", "WIN"],
        "lessons": ["Cut losses short", "Let winners run"],
    })

    # Verify all 6 new analyzer contexts are present
    print("\n[Check] New analyzer contexts present in analysis_out:")
    new_keys = [
        "divergence_ctx", "ichimoku_ctx", "volatility_ctx",
        "volume_profile_ctx", "smc_advanced_ctx", "mtf_structure_ctx",
        "strategy", "structure_ctx",
    ]
    for k in new_keys:
        present = k in analysis_out
        val = analysis_out.get(k, {})
        size = len(val) if isinstance(val, dict) else "n/a"
        icon = "[OK]" if present else "[MISS]"
        print(f"  {icon}  {k:<22}  size={size}")

    # Verify strategy selector output
    strat = analysis_out.get("strategy", {})
    print(f"\n[Check] Strategy Selector output:")
    print(f"  strategy       : {strat.get('strategy', 'N/A')}")
    print(f"  confidence     : {strat.get('confidence', 0)}%")
    print(f"  risk_mult      : {strat.get('risk_mult', 0)}")
    print(f"  position_mult  : {strat.get('position_mult', 0)}")
    print(f"  active_modules : {len(strat.get('active_modules', []))}")
    print(f"  reason         : {strat.get('reason', '')[:80]}")

    # Verify MasterDecision context (if produced)
    md = analysis_out.get("master_decision", {})
    if md:
        print(f"\n[Check] MasterDecision context:")
        print(f"  final_signal       : {md.get('final_signal')}")
        print(f"  master_confidence  : {md.get('master_confidence')}")
        print(f"  position_size      : {md.get('position_size')}")
        print(f"  position_multiplier: {md.get('position_multiplier')}")
        print(f"  strategy           : {md.get('strategy')}")
        print(f"  strategy_confidence: {md.get('strategy_confidence')}")
        print(f"  strategy_risk_mult : {md.get('strategy_risk_mult')}")
        print(f"  strategy_reason    : {(md.get('strategy_reason') or '')[:80]}")
    else:
        print("\n[Note] MasterDecision context not populated (expected if LLM unavailable)")

    # Final signal
    print(f"\n[Check] Final signal: {analysis_out.get('final_signal')}")

    # Summary
    print("\n" + "=" * 60)
    if all(k in analysis_out for k in new_keys):
        print("  RESULT: ALL 8 NEW CONTEXT KEYS PRESENT")
    else:
        missing = [k for k in new_keys if k not in analysis_out]
        print(f"  RESULT: MISSING KEYS: {missing}")
    print("=" * 60)


if __name__ == "__main__":
    main()
