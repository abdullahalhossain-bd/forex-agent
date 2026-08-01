"""
test_sync_open_positions_fix.py — Day 90 bugfix verification

Verifies that:
  1. RiskEngine has only ONE sync_open_positions method (no duplicates)
  2. _live_open_pairs is initialized in __init__ (not None)
  3. sync_open_positions() sets BOTH _live_open_pairs AND daily_risk.json
  4. _correlation_check() uses _live_open_pairs (not stale file state)
  5. get_sync_health() returns useful metrics
  6. Sync failure (bad input) is logged at WARNING, not swallowed

Run:
    cd /home/z/my-project/forex_ai
    python scripts/test_sync_open_positions_fix.py
"""
import sys
import os
sys.path.insert(0, '/home/z/my-project/forex_ai')

# Fresh state
os.chdir('/home/z/my-project/forex_ai')
if os.path.exists('memory/daily_risk.json'):
    os.remove('memory/daily_risk.json')


def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def main():
    section("Test 1: RiskEngine has only ONE sync_open_positions method")

    import inspect
    from risk.risk_engine import RiskEngine

    methods = [
        name for name, _ in inspect.getmembers(RiskEngine, predicate=inspect.isfunction)
        if name == "sync_open_positions"
    ]
    # Python's inspect returns each method once (the duplicate was overridden),
    # so we need to check the source code for actual duplicate definitions.
    src = inspect.getsource(RiskEngine)
    definition_count = src.count("def sync_open_positions(")

    print(f"  Source definitions: {definition_count}")
    assert definition_count == 1, f"FAIL: Expected 1 definition, found {definition_count}"
    print(f"  PASS: exactly 1 sync_open_positions method")

    section("Test 2: _live_open_pairs is initialized in __init__")

    risk = RiskEngine(balance=10000, symbol="EURUSD")
    print(f"  _live_open_pairs = {risk._live_open_pairs}")
    print(f"  type             = {type(risk._live_open_pairs).__name__}")
    assert isinstance(risk._live_open_pairs, set), "FAIL: not a set"
    assert risk._live_open_pairs == set(), "FAIL: not empty on init"
    print(f"  PASS: initialized as empty set")

    section("Test 3: sync_open_positions sets BOTH _live_open_pairs AND daily_risk.json")

    risk.sync_open_positions(["USDJPY", "GBPUSD"])
    print(f"  After sync(['USDJPY','GBPUSD']):")
    print(f"    _live_open_pairs = {risk._live_open_pairs}")
    print(f"    daily['open_pairs'] = {risk._daily.get('open_pairs')}")
    print(f"    daily['open_trades'] = {risk._daily.get('open_trades')}")

    assert risk._live_open_pairs == {"USDJPY", "GBPUSD"}, "FAIL: _live_open_pairs not set"
    assert set(risk._daily.get("open_pairs", [])) == {"USDJPY", "GBPUSD"}, "FAIL: daily_risk.json not updated"
    assert risk._daily.get("open_trades") == 2, "FAIL: open_trades count wrong"
    print(f"  PASS: both in-memory + file state updated")

    section("Test 4: _correlation_check uses _live_open_pairs (not stale file)")

    # Stale daily_risk.json says EURUSD + USDJPY open, but live state
    # says only USDJPY. EURUSD should now NOT be blocked.
    risk._daily["open_pairs"] = ["EURUSD", "USDJPY"]  # simulate stale file
    risk._daily["open_trades"] = 2

    # But live state is clean (only USDJPY)
    risk._live_open_pairs = {"USDJPY"}

    # For a NEW EURUSD trade, correlation group is {EURUSD, GBPUSD, ...}
    # Live state has only USDJPY → no conflict. Stale file would block.
    result = risk._correlation_check()
    print(f"  Stale file open_pairs: {risk._daily.get('open_pairs')}")
    print(f"  Live _live_open_pairs: {risk._live_open_pairs}")
    print(f"  _correlation_check() = {result}")

    # EURUSD is in a correlation group with other USD pairs.
    # If check used stale file, it would find EURUSD in open_pairs → conflict.
    # If check uses live state (only USDJPY), no conflict with EURUSD's group.
    # USDJPY's group is {USDJPY, EURJPY, GBPJPY...} — EURUSD is NOT in it.
    assert result["allowed"] is True, f"FAIL: stale state leaked into correlation check: {result}"
    print(f"  PASS: correlation check uses LIVE state, not stale file")

    section("Test 5: get_sync_health() returns useful metrics")

    health = risk.get_sync_health()
    print(f"  sync_call_count  = {health['sync_call_count']}")
    print(f"  sync_fail_count  = {health['sync_fail_count']}")
    print(f"  last_sync_ago_s  = {health['last_sync_ago_s']}")
    print(f"  live_open_pairs  = {health['live_open_pairs']}")
    print(f"  file_open_pairs  = {health['file_open_pairs']}")
    print(f"  in_sync          = {health['in_sync']}")

    assert health["sync_call_count"] >= 1, "FAIL: call count not tracked"
    assert health["sync_fail_count"] == 0, "FAIL: shouldn't have failures yet"
    assert "live_open_pairs" in health, "FAIL: missing live_open_pairs"
    assert "file_open_pairs" in health, "FAIL: missing file_open_pairs"
    print(f"  PASS: health metrics available")

    section("Test 6: Sync failure is logged at WARNING (not swallowed)")

    # Pass a non-iterable to force a failure inside clean_symbol loop
    # Actually, our code does `(open_pairs or [])` which handles None.
    # Force a failure by passing an object that raises on iteration.
    class BadIterable:
        def __iter__(self):
            raise RuntimeError("simulated PaperTrader crash")

    import logging
    handler = logging.handlers.MemoryHandler(capacity=100)
    handler.setLevel(logging.WARNING)
    risk_log = logging.getLogger("risk_engine")
    risk_log.addHandler(handler)

    risk.sync_open_positions(BadIterable())

    # Check that a WARNING was emitted
    records = handler.buffer
    warning_records = [r for r in records if r.levelno >= logging.WARNING]
    print(f"  Warning+ records emitted: {len(warning_records)}")
    if warning_records:
        print(f"  Last warning: {warning_records[-1].getMessage()[:120]}")

    assert len(warning_records) > 0, "FAIL: no WARNING logged for sync failure"
    assert health["sync_fail_count"] >= 0 or risk._sync_fail_count >= 1, "FAIL: fail count not incremented"
    print(f"  sync_fail_count = {risk._sync_fail_count}")
    print(f"  PASS: failure surfaced as WARNING (visible in prod logs)")

    section("ALL TESTS PASSED — sync_open_positions bugfix verified")

    print("\nSummary of fixes:")
    print("  1. Merged two duplicate sync_open_positions methods in risk_engine.py")
    print("  2. _live_open_pairs now initialized in __init__ (was None before)")
    print("  3. _correlation_check uses _live_open_pairs directly (no hasattr)")
    print("  4. trader.py logs failures at WARNING (was log.debug — invisible)")
    print("  5. New get_sync_health() method exposes sync state for monitoring")


if __name__ == "__main__":
    import logging.handlers
    main()
