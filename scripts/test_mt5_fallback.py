#!/usr/bin/env python3
"""Test MT5 fallback to simulation mode."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MT5_FALLBACK_TO_SIMULATION", "true")
os.environ.setdefault("EXECUTION_MODE", "mt5_demo")
os.environ.setdefault("MT5_LOGIN", "0")
os.environ.setdefault("MT5_PASSWORD", "")
os.environ.setdefault("MT5_SERVER", "")


def test_fallback():
    from execution.execution_router import ExecutionRouter
    print("[TEST] Creating ExecutionRouter with no MT5 terminal...")
    try:
        router = ExecutionRouter(mode="mt5_demo", db=None, paper_trader=None)
        if router._simulation_mode:
            print("[PASS] ExecutionRouter fell back to SIMULATION mode")
            return True
        else:
            print("[FAIL] Did NOT enter simulation mode")
            return False
    except RuntimeError as e:
        print(f"[FAIL] RuntimeError: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    ok = test_fallback()
    print("\n" + "=" * 50)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
