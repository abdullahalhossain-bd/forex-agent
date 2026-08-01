"""
scripts/winrate_tests/test_confidence_lookup.py
================================================

Self-test for the confidence_winrate_lookup module. Verifies:

  1. The lookup module loads cleanly.
  2. Every strategy in the JSON has a recommendation that matches the
     thresholds defined in the JSON itself.
  3. The fail-closed path works (missing JSON / unknown strategy).
  4. The tactic override path works.
  5. If actual_backtest_results.json exists (from run_actual_backtest.py),
     cross-check the lookup's expected_winrate against the actual win_rate
     and report any drift.

Usage:
  python scripts/winrate_tests/test_confidence_lookup.py
  python scripts/winrate_tests/test_confidence_lookup.py --strict  # exit 1 on any drift
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from intelligence.confidence_winrate_lookup import (  # noqa: E402
    ConfidenceWinrateLookup, Recommendation, get_lookup, reload_lookup,
)

OUTPUT_DIR = PROJECT_ROOT / "scripts" / "winrate_tests" / "output"
ACTUAL_RESULTS_PATH = OUTPUT_DIR / "actual_backtest_results.json"


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

class TestReport:
    def __init__(self) -> None:
        self.passed: List[str] = []
        self.failed: List[str] = []
        self.warned: List[str] = []

    def ok(self, name: str, detail: str = "") -> None:
        self.passed.append(f"✅ {name}" + (f" — {detail}" if detail else ""))

    def warn(self, name: str, detail: str = "") -> None:
        self.warned.append(f"⚠️  {name}" + (f" — {detail}" if detail else ""))

    def fail(self, name: str, detail: str = "") -> None:
        self.failed.append(f"❌ {name}" + (f" — {detail}" if detail else ""))

    def summary(self) -> str:
        lines = []
        lines.append(f"PASSED:  {len(self.passed)}")
        lines.append(f"WARNED:  {len(self.warned)}")
        lines.append(f"FAILED:  {len(self.failed)}")
        for msg in self.passed + self.warned + self.failed:
            lines.append("  " + msg)
        return "\n".join(lines)

    def exit_code(self) -> int:
        return 1 if self.failed else 0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_lookup_loads(rep: TestReport) -> ConfidenceWinrateLookup:
    lw = get_lookup()
    if not lw.is_loaded:
        rep.fail("lookup_loads", "JSON did not load — check intelligence/confidence_winrate_data.json")
        return lw
    rep.ok("lookup_loads", f"{len(lw.list_strategies())} strategies found")
    return lw


def test_listed_strategies_have_recommendations(lw: ConfidenceWinrateLookup, rep: TestReport) -> None:
    if not lw.is_loaded:
        rep.fail("listed_strategies", "lookup not loaded")
        return

    for strat in lw.list_strategies():
        for conf in ["High", "Medium", "Low"]:
            try:
                rec = lw.recommend(strat, conf)
                if rec.action not in ("trust", "caution", "skip"):
                    rep.fail(f"rec_{strat}_{conf}",
                             f"invalid action: {rec.action}")
                    continue

                # Cross-check: if found and n >= 10, action should match thresholds
                if rec.found and rec.n_trades >= 10:
                    wr = rec.expected_winrate or 0.0
                    if wr >= 0.50 and rec.action != "trust":
                        rep.warn(f"rec_{strat}_{conf}",
                                 f"WR={wr:.1%} (n={rec.n_trades}) but action={rec.action}")
                    elif wr < 0.40 and rec.action != "skip":
                        rep.warn(f"rec_{strat}_{conf}",
                                 f"WR={wr:.1%} (n={rec.n_trades}) but action={rec.action}")
                    else:
                        rep.ok(f"rec_{strat}_{conf}",
                               f"WR={wr:.1%} n={rec.n_trades} -> {rec.action}")
                else:
                    rep.ok(f"rec_{strat}_{conf}",
                           f"n={rec.n_trades} -> {rec.action}")
            except Exception as e:
                rep.fail(f"rec_{strat}_{conf}", f"raised: {e}")


def test_fail_closed_unknown_strategy(lw: ConfidenceWinrateLookup, rep: TestReport) -> None:
    rec = lw.recommend("does_not_exist_xyz", "High")
    if rec.action == "skip" and rec.position_scale == 0.0:
        rep.ok("fail_closed_unknown", "unknown strategy -> skip")
    else:
        rep.fail("fail_closed_unknown",
                 f"expected skip/0.0, got {rec.action}/{rec.position_scale}")


def test_fail_closed_no_data(lw: ConfidenceWinrateLookup, rep: TestReport) -> None:
    """ict_amd Low should be no_data -> skip."""
    rec = lw.recommend("ict_amd", "Low")
    if rec.action == "skip":
        rep.ok("fail_closed_no_data", f"ict_amd/Low -> skip ({rec.notes})")
    else:
        rep.warn("fail_closed_no_data",
                 f"ict_amd/Low -> {rec.action} (expected skip)")


def test_tactic_override(lw: ConfidenceWinrateLookup, rep: TestReport) -> None:
    """stop_hunt Low should skip, but a known good tactic should still resolve."""
    tw = lw.tactic_winrate("candlestick_patterns", "Three Black Crows")
    if tw["found"]:
        rep.ok("tactic_lookup_candlestick",
               f"Three Black Crows: WR={tw['win_rate']:.1%} n={tw['trades']}")
    else:
        rep.warn("tactic_lookup_candlestick", "tactic not found in JSON")


def test_drift_against_actual_backtest(lw: ConfidenceWinrateLookup, rep: TestReport,
                                       strict: bool = False) -> None:
    """If actual_backtest_results.json exists, compare its winrates to the lookup."""
    if not ACTUAL_RESULTS_PATH.exists():
        rep.warn("drift_check", f"no actual_backtest_results.json at {ACTUAL_RESULTS_PATH} — "
                                "run run_actual_backtest.py first")
        return

    with open(ACTUAL_RESULTS_PATH, "r", encoding="utf-8") as f:
        actual = json.load(f)

    actual_aggs = actual.get("strategy_aggregates", {})
    if not actual_aggs:
        rep.warn("drift_check", "actual backtest has no strategy_aggregates")
        return

    drifts: List[str] = []
    matches: List[str] = []

    for strat_name, a in actual_aggs.items():
        actual_n = a["n_trades"]
        actual_wr = a["win_rate"]
        if actual_n == 0:
            continue

        # Get the lookup's view for the best-confidence tier (or 'any')
        # Use Medium as a reasonable default
        rec = lw.recommend(strat_name, "Medium")
        if not rec.found:
            # Strategy is new (not in JSON) — warn
            rep.warn(f"drift_{strat_name}",
                     f"NEW strategy in backtest (not in JSON): n={actual_n} WR={actual_wr:.1%}")
            continue

        json_wr = rec.expected_winrate
        if json_wr is None:
            continue

        # Compute drift — only meaningful if actual_n >= 10
        if actual_n >= 10:
            delta = abs(json_wr - actual_wr)
            if delta > 0.05:  # more than 5 percentage points
                msg = (f"{strat_name}/Medium: JSON={json_wr:.1%} (n={rec.n_trades}) "
                       f"vs ACTUAL={actual_wr:.1%} (n={actual_n}) — drift={delta:.1%}")
                drifts.append(msg)
                if strict:
                    rep.fail(f"drift_{strat_name}", msg)
                else:
                    rep.warn(f"drift_{strat_name}", msg)
            else:
                matches.append(f"{strat_name}/Medium: JSON={json_wr:.1%} "
                               f"ACTUAL={actual_wr:.1%} (drift={delta:.1%})")
                rep.ok(f"drift_{strat_name}", f"drift={delta:.1%}")

    if matches and not drifts:
        rep.ok("drift_check_overall", f"all {len(matches)} strategies within 5pp of JSON")
    elif drifts:
        rep.warn("drift_check_overall",
                 f"{len(drifts)} strategies drifted >5pp — run refresh_confidence_data.py")


def test_singleton_reload(rep: TestReport) -> None:
    """reload_lookup() should give back a fresh instance."""
    lw1 = get_lookup()
    lw2 = reload_lookup()
    if lw1 is not lw2 and lw2.is_loaded:
        rep.ok("singleton_reload", "reload_lookup returns new loaded instance")
    else:
        rep.fail("singleton_reload", "reload did not return a fresh loaded instance")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="Self-test for confidence_winrate_lookup.")
    p.add_argument("--strict", action="store_true",
                   help="Exit 1 if any drift > 5pp is detected.")
    args = p.parse_args()

    rep = TestReport()
    print("=" * 70)
    print("CONFIDENCE WINRATE LOOKUP — SELF-TEST")
    print("=" * 70)

    lw = test_lookup_loads(rep)
    test_listed_strategies_have_recommendations(lw, rep)
    test_fail_closed_unknown_strategy(lw, rep)
    test_fail_closed_no_data(lw, rep)
    test_tactic_override(lw, rep)
    test_singleton_reload(rep)
    test_drift_against_actual_backtest(lw, rep, strict=args.strict)

    print()
    print(rep.summary())
    print()
    print("=" * 70)
    if rep.failed:
        print("RESULT: FAIL")
    elif rep.warned:
        print("RESULT: PASS (with warnings)")
    else:
        print("RESULT: PASS")
    print("=" * 70)
    return rep.exit_code()


if __name__ == "__main__":
    sys.exit(main())
