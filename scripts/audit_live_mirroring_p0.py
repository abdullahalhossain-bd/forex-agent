"""Static P0 audit for the live-trading-mirroring replay contract.

This script does not alter strategy parameters or trading behavior. It checks
for obvious look-ahead / wall-clock hazards and verifies that the unified
replay path is wired to the live AITrader decision core.

Run from repository root:
    py -3 scripts/audit_live_mirroring_p0.py
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_LOOKAHEAD = ("shift(-1", "iloc[i+1", "iloc[i + 1", "future(", "bfill(", "center=True")
WALL_CLOCK = ("datetime.now(", "datetime.utcnow(", "time.time(", ".today(")
EXTERNAL_MARKERS = (
    "EconomicCalendarAPI", "NewsAPIProvider", "FRED", "RetailSentiment",
    "IntermarketEngine", "InstitutionalFlowEngine", "NetworkMonitor",
)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def main() -> int:
    result = {
        "contract": "research/live_mirroring/PARITY_CONTRACT.md",
        "checks": [],
        "verdict": "UNKNOWN",
    }

    unified = read(ROOT / "backtest" / "unified_engine.py")
    trader = read(ROOT / "core" / "trader.py")
    clock = read(ROOT / "core" / "clock.py")
    constants = read(ROOT / "core" / "constants.py")

    def check(name, passed, detail):
        result["checks"].append({"name": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    check(
        "shared_decision_core",
        "evaluate_decision_core" in unified and "evaluate_decision_core" in trader,
        "Unified replay must call AITrader.evaluate_decision_core().",
    )
    check(
        "replay_clock_exists",
        all(x in clock for x in ("class ReplayClock", "def now", "def advance", "def current_date", "def current_session")),
        "ReplayClock API is present.",
    )
    check(
        "backtest_mode_enabled",
        "set_backtest_mode(True)" in unified and "def is_backtest_mode" in constants,
        "Replay enables the global backtest isolation flag.",
    )
    check(
        "isolated_db",
        "TraderDB(db_path=db_path)" in unified and "db_path" in unified,
        "Replay trader receives an explicit database path.",
    )
    check(
        "deterministic_random_seed",
        "_random.seed(42)" in unified and "_np.random.seed(42)" in unified,
        "Broker simulation RNG is seeded.",
    )
    check(
        "future_pattern_scan_replay_core",
        not any(p in unified for p in FORBIDDEN_LOOKAHEAD),
        "No obvious forbidden look-ahead pattern in unified replay engine.",
    )

    # AST parse is deliberately included: a source file that cannot parse is a hard P0 failure.
    for rel in ("core/clock.py", "core/data_provider.py", "backtest/unified_engine.py", "core/trader.py"):
        try:
            ast.parse(read(ROOT / rel), filename=rel)
            check(f"syntax:{rel}", True, "AST parse succeeded.")
        except SyntaxError as exc:
            check(f"syntax:{rel}", False, f"SyntaxError: {exc}")

    # Report wall-clock uses outside the dedicated clock implementation.
    wall_hits = []
    for path in ROOT.rglob("*.py"):
        if any(part in {".git", ".venv", "venv", "__pycache__"} for part in path.parts):
            continue
        if path == ROOT / "core" / "clock.py":
            continue
        text = read(path)
        for marker in WALL_CLOCK:
            if marker in text:
                wall_hits.append({"file": str(path.relative_to(ROOT)), "marker": marker})
    result["wall_clock_hits"] = wall_hits[:500]
    check(
        "wall_clock_audit",
        len(wall_hits) == 0,
        f"Found {len(wall_hits)} wall-clock call sites outside core/clock.py; manual classification required.",
    )

    external_hits = []
    analysis_path = ROOT / "agents" / "analysis_agent.py"
    analysis_text = read(analysis_path)
    for marker in EXTERNAL_MARKERS:
        if marker in analysis_text:
            external_hits.append(marker)
    result["analysis_external_dependencies"] = sorted(set(external_hits))
    check(
        "external_context_audit",
        True,
        "External dependencies are inventoried; each must be disabled or proven timestamp-safe before replay use.",
    )

    # Conservative overall rule: static audit can never certify full parity.
    hard_fail = any(c["status"] == "FAIL" for c in result["checks"] if c["name"].startswith("syntax:") or c["name"] in {"shared_decision_core", "replay_clock_exists", "isolated_db"})
    result["verdict"] = "FAILED" if hard_fail else "PARTIALLY_PROVEN"

    out = ROOT / "backtest" / "results" / "p0_live_mirroring_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nWrote: {out.relative_to(ROOT)}")
    return 1 if hard_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
