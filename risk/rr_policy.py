# risk/rr_policy.py — Single-call resolver for minimum reward:risk ratio
# =============================================================================
# Why this file exists: an audit against the Nison rule-engine spec
# (global_filters.risk_reward_gate, default 2:1) found that core/constants.py
# already defines the intended single source of truth —
#
#     MIN_RR_PROD: float = _env_float("MIN_RR_PROD", 2.0)
#     MIN_RR_TEST: float = _env_float("MIN_RR_TEST", 1.0)
#
# — but risk/trade_permission.py (the LIVE trade-permission gate) had its own
# separate hardcoded class attribute `MIN_RR_PROD = 1.5`, which had silently
# drifted away from core/constants.py's 2.0. scripts/fix_execution_pipeline.py
# shows this was a deliberate one-off patch at some point ("Lower MIN_RR_PROD
# from 2.0 to 1.5") that was never reflected back into core/constants.py or
# into analysis/ict_amd_signal_engine.py, analysis/multi_strategy_pa_engine.py
# (both still 2.0), or analysis/stop_hunt_signal_engine.py (1.4) — exactly the
# kind of conflicting-copies drift the rule-spec audit was meant to catch.
#
# This module does not introduce a new number — it just gives every R:R gate
# ONE function to call so they all resolve from core/constants.py instead of
# keeping their own local copy that can drift again later.
# =============================================================================

from __future__ import annotations

# Some strategy engines intentionally trade a lower-R:R style (e.g. a
# stop-hunt/liquidity-grab scalp strategy that exits fast). Rather than let
# that live as an unexplained local constant, the exception is declared here,
# by strategy name, so it stays visible and reviewable in one place.
STRATEGY_OVERRIDES: dict[str, float] = {
    "stop_hunt": 1.4,   # analysis/stop_hunt_signal_engine.py — fast scalp exits
}


def get_min_rr(*, strategy: str | None = None, test_mode: bool = False) -> float:
    """
    Resolve the minimum reward:risk ratio to enforce, from core/constants.py
    (env-overridable via MIN_RR_PROD / MIN_RR_TEST), with an optional
    declared per-strategy exception.
    """
    from core.constants import MIN_RR_PROD, MIN_RR_TEST

    if test_mode:
        return MIN_RR_TEST
    if strategy and strategy in STRATEGY_OVERRIDES:
        return STRATEGY_OVERRIDES[strategy]
    return MIN_RR_PROD


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    os.environ.pop("MIN_RR_PROD", None)
    os.environ.pop("MIN_RR_TEST", None)

    assert get_min_rr() == 2.0
    assert get_min_rr(test_mode=True) == 1.0
    assert get_min_rr(strategy="stop_hunt") == 1.4
    print("rr_policy smoke test passed — resolves from core/constants.py.")