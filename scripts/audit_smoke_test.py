"""
Smoke test: verify all audit fixes compile and have the intended semantics.
Run: python /home/z/my-project/scripts/audit_smoke_test.py
"""
import os
import sys
import importlib

# Add forex-agent to path
sys.path.insert(0, "/home/z/my-project/forex-agent")

failures = []
passes = []


def check(name, cond, detail=""):
    if cond:
        passes.append(name)
    else:
        failures.append(f"{name}: {detail}")


# 1. Verify all fixed files import cleanly
files_to_check = [
    "config",
    "risk.trade_permission",
    "risk.trade_frequency",
    "core.trader",
    "core.devils_advocate",
    "broker.order_manager",
    "broker.position_manager",
    "execution.execution_router",
    "execution.trade_recovery",
    "execution.emergency_exit",
    "fundamental.news_filter",
    "agents.analysis_agent",
]

for mod_name in files_to_check:
    try:
        importlib.import_module(mod_name)
        passes.append(f"import {mod_name}")
    except Exception as e:
        failures.append(f"import {mod_name}: {e}")

# 2. Verify critical env defaults
from config import MT5_FALLBACK_TO_SIMULATION
check(
    "MT5_FALLBACK_TO_SIMULATION defaults to False (audit fix)",
    MT5_FALLBACK_TO_SIMULATION is False,
    f"got {MT5_FALLBACK_TO_SIMULATION}",
)

# 3. Verify trade_frequency persists
from risk.trade_frequency import TradeFrequencyController
# Clear state first so the test is deterministic
state_file = "/home/z/my-project/forex-agent/memory/trade_frequency_state.json"
if os.path.exists(state_file):
    os.remove(state_file)

ctrl = TradeFrequencyController()
# Sanity: should be 0 trades today after wipe
check(
    "TradeFrequencyController starts at 0 trades after state wipe",
    ctrl.trade_count_today() == 0,
    f"got {ctrl.trade_count_today()}",
)
ctrl.record_trade("EURUSD", "BUY")
check(
    "TradeFrequencyController persists to disk",
    os.path.exists(state_file),
    f"file not found at {state_file}",
)

# Reload and verify it loaded the persisted trade
ctrl2 = TradeFrequencyController()
check(
    "TradeFrequencyController reloads persisted state",
    ctrl2.trade_count_today() == 1,
    f"got {ctrl2.trade_count_today()}",
)

# Cleanup
if os.path.exists(state_file):
    os.remove(state_file)

# 4. Verify retcode set includes 10010
from broker.order_manager import RETCODE_SUCCESS
check(
    "RETCODE_SUCCESS includes 10010 (DONE_PARTIAL)",
    10010 in RETCODE_SUCCESS,
    f"got {RETCODE_SUCCESS}",
)
check(
    "RETCODE_SUCCESS includes 10008 (PLACED)",
    10008 in RETCODE_SUCCESS,
    f"got {RETCODE_SUCCESS}",
)
check(
    "RETCODE_SUCCESS includes 10009 (DONE)",
    10009 in RETCODE_SUCCESS,
    f"got {RETCODE_SUCCESS}",
)

# 5. Verify DevilsAdvocate constants
from core.devils_advocate import DECISION_TAKE, DECISION_REJECT
check("DECISION_TAKE constant", DECISION_TAKE == "TAKE")
check("DECISION_REJECT constant", DECISION_REJECT == "REJECT")

# 6. Verify EmergencyExitResult has fatal_error field
from execution.emergency_exit import EmergencyExitResult
r = EmergencyExitResult()
check(
    "EmergencyExitResult has fatal_error field",
    hasattr(r, "fatal_error"),
    "attribute missing",
)
check(
    "EmergencyExitResult.to_dict includes fatal_error",
    "fatal_error" in r.to_dict(),
    "key missing in dict",
)

# 7. Verify position_manager imports _resolve_filling_mode from order_manager
import broker.position_manager as pm
check(
    "position_manager imports _om_resolve_filling_mode from order_manager",
    hasattr(pm, "_om_resolve_filling_mode"),
    "attribute missing",
)

# Summary
print(f"\n{'='*60}")
print(f"  AUDIT SMOKE TEST RESULTS")
print(f"{'='*60}")
print(f"  PASSED: {len(passes)}")
print(f"  FAILED: {len(failures)}")
if failures:
    print(f"\n  FAILURES:")
    for f in failures:
        print(f"    - {f}")
print(f"{'='*60}")
sys.exit(1 if failures else 0)
