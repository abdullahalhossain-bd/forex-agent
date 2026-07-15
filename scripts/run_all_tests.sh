#!/usr/bin/env bash
# Run all test suites in sequence and report results.
set -u  # do not use -e: we want to run every test even if some fail

cd /home/z/my-project/forex_ai

TESTS=(
    tests/test_core.py
    tests/test_pipeline.py
    tests/test_sr_zones.py
    tests/test_high_reliability_patterns.py
    tests/test_stop_hunt_signal_engine.py
    tests/test_ict_amd_signal_engine.py
    tests/test_multi_strategy_pa_engine.py
    tests/test_unified_signal_engine.py
    tests/test_entry_quality_guardrails.py
    tests/test_triple_top_bottom.py
    tests/test_book_pages_106_120.py
    tests/test_book_pages_136_151.py
    tests/test_odd_enhancers.py
    tests/test_flip_zones.py
    tests/test_cci_state_machine.py
    tests/test_curve_mtf.py
    tests/test_risk_management.py
    tests/test_adaptive_backtest_system.py
)

PASS=0
FAIL=0
FAILED_TESTS=()

echo "========================================================"
echo "  RUNNING ${#TESTS[@]} TEST SUITES"
echo "========================================================"

for t in "${TESTS[@]}"; do
    echo ""
    echo "── $t ──"
    if python "$t" > /tmp/test_out.log 2>&1; then
        # Extract pass count from RESULT line if present
        result=$(grep "RESULT:" /tmp/test_out.log | tail -1)
        if [ -n "$result" ]; then
            echo "  PASS  $result"
        else
            echo "  PASS (no RESULT line)"
        fi
        PASS=$((PASS + 1))
    else
        echo "  FAIL — last 20 lines of output:"
        tail -20 /tmp/test_out.log | sed 's/^/    /'
        FAIL=$((FAIL + 1))
        FAILED_TESTS+=("$t")
    fi
done

echo ""
echo "========================================================"
echo "  SUMMARY: $PASS passed, $FAIL failed (of ${#TESTS[@]})"
if [ "$FAIL" -gt 0 ]; then
    echo "  Failed suites:"
    for ft in "${FAILED_TESTS[@]}"; do
        echo "    - $ft"
    done
fi
echo "========================================================"
exit 0
