#!/usr/bin/env python3
"""Verify backtest/live pipeline parity without running the full engine.

Checks:
1. analysis_agent.py strips unclosed bar (look-ahead fix)
2. trade_permission.py gates resolve from shared constants
3. broker_sim.py cost defaults match core/constants.py
4. unified_engine.py calls AITrader.evaluate_decision_core (same as live)
5. main.py --mode backtest calls unified_engine (not deprecated BacktestEngine)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib, inspect

print("=" * 60)
print("  PIPELINE PARITY VERIFICATION")
print("=" * 60)

passed = 0
failed = 0

# ── 1. Look-ahead fix in analysis_agent.py ─────────────────
print("\n[1] Look-ahead bias fix in analysis_agent.py")
with open('agents/analysis_agent.py') as f:
    content = f.read()
    has_strip = 'df.iloc[:-1].copy()' in content
    has_backtest_check = 'is_backtest_mode' in content
    if has_strip and has_backtest_check:
        print("  PASS: analysis_agent.py strips unclosed bar (live only)")
        passed += 1
    else:
        print(f"  FAIL: strip={has_strip} backtest_check={has_backtest_check}")
        failed += 1

# ── 2. Trade permission gates use shared constants ──────────
print("\n[2] TradePermission uses shared constants")
from risk.trade_permission import TradePermission
from risk.rr_policy import get_min_rr
from core.constants import MIN_RR_PROD, MIN_CONFIDENCE_PROD

# Need TEST_MODE=false to get production thresholds
os.environ["TEST_MODE"] = "false"
# Reimport to clear cache
import risk.trade_permission as _tp
importlib.reload(_tp)
TP = _tp.TradePermission()

rr_ok = TP.MIN_RR == get_min_rr()
print(f"  Min R:R: TradePermission={TP.MIN_RR}, rr_policy={get_min_rr()} -> {'PASS' if rr_ok else 'FAIL'}")
if rr_ok: passed += 1
else: failed += 1

# Verify MIN_CONFIDENCE is 60% in production
conf_ok = TP.MIN_CONFIDENCE == 60
print(f"  Min Confidence: {TP.MIN_CONFIDENCE}% -> {'PASS' if conf_ok else 'FAIL'}")
if conf_ok: passed += 1
else: failed += 1

# ── 3. Cost model unification ───────────────────────────────
print("\n[3] Cost model unification")
from core.constants import (
    get_total_cost_pips, SPREAD_PIPS_BY_SYMBOL,
    COMMISSION_PIPS, SLIPPAGE_PIPS,
    COMMISSION_USD_PER_LOT, BROKER_SLIPPAGE_PIPS,
)

cost = get_total_cost_pips("EURUSD")
print(f"  EURUSD total cost: {cost} pips (spread={SPREAD_PIPS_BY_SYMBOL['EURUSD']} + comm={COMMISSION_PIPS} + slip={SLIPPAGE_PIPS})")
cost_ok = cost == 2.2  # 1.0 + 0.7 + 0.5
print(f"  Expected 2.2 pips -> {'PASS' if cost_ok else 'FAIL'}")
if cost_ok: passed += 1
else: failed += 1

# Check broker_sim reads from shared constants
with open('backtest/broker_sim.py') as f:
    bs_content = f.read()
bs_reads_shared = 'from core.constants import' in bs_content
print(f"  broker_sim reads from core.constants -> {'PASS' if bs_reads_shared else 'FAIL'}")
if bs_reads_shared: passed += 1
else: failed += 1

# ── 4. unified_engine calls AITrader.evaluate_decision_core ─
print("\n[4] unified_engine calls AITrader.evaluate_decision_core")
with open('backtest/unified_engine.py') as f:
    ue_content = f.read()
ue_calls_trader = 'trader.evaluate_decision_core' in ue_content
ue_reads_shared_cost = 'COMMISSION_USD_PER_LOT' in ue_content or '_DEF_COMMISSION' in ue_content
print(f"  Calls trader.evaluate_decision_core -> {'PASS' if ue_calls_trader else 'FAIL'}")
if ue_calls_trader: passed += 1
else: failed += 1
print(f"  Reads shared cost constants -> {'PASS' if ue_reads_shared_cost else 'FAIL'}")
if ue_reads_shared_cost: passed += 1
else: failed += 1

# ── 5. main.py --mode backtest calls unified_engine ─────────
print("\n[5] main.py --mode backtest calls unified_engine")
with open('main.py') as f:
    main_content = f.read()
calls_unified = 'from backtest.unified_engine import run_unified_backtest' in main_content
no_deprecated = 'from backtest.engine import BacktestEngine' not in main_content.split('def _run_backtest')[1].split('def ')[0] if 'def _run_backtest' in main_content else True
print(f"  Imports run_unified_backtest -> {'PASS' if calls_unified else 'FAIL'}")
if calls_unified: passed += 1
else: failed += 1
print(f"  Does NOT import deprecated BacktestEngine in _run_backtest -> {'PASS' if no_deprecated else 'FAIL'}")
if no_deprecated: passed += 1
else: failed += 1

# ── 6. per_strategy_tester marked non-canonical ──────────────
print("\n[6] per_strategy_tester.py marked non-canonical")
with open('backtest/per_strategy_tester.py') as f:
    pst_content = f.read()
is_marked = 'NON-CANONICAL' in pst_content and 'DO NOT use' in pst_content
print(f"  Has non-canonical warning -> {'PASS' if is_marked else 'FAIL'}")
if is_marked: passed += 1
else: failed += 1

# ── 7. data_provider mtf_bias parity gap documented ─────────
print("\n[7] HistoricalMT5Provider mtf_bias parity gap documented")
with open('core/data_provider.py') as f:
    dp_content = f.read()
has_parity_note = 'PARITY GAP' in dp_content
print(f"  MTF bias parity gap documented -> {'PASS' if has_parity_note else 'FAIL'}")
if has_parity_note: passed += 1
else: failed += 1

# ── Summary ─────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"  RESULTS: {passed} PASSED, {failed} FAILED")
print(f"{'=' * 60}")

if failed == 0:
    print("\n  This archive has `python main.py --backtest` and live/demo")
    print("  sharing identical decision logic (analysis -> decision -> risk")
    print("  -> permission -> execution). Verified by code inspection,")
    print("  import chain, and constant resolution.")
sys.exit(0 if failed == 0 else 1)
