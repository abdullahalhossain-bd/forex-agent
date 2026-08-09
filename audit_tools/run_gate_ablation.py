"""
audit_tools/run_gate_ablation.py — Controlled gate-ablation orchestrator
=======================================================================

Runs the verified backtest command once per gate-bypass condition:
    baseline (no bypass)
    bypass_min_confidence
    bypass_session_quality
    bypass_confluence_quality
    bypass_risk_approved
    bypass_sr_zone_alignment
    bypass_valid_signal
    bypass_trend_alignment

For each run:
  - Sets FOREX_BYPASS_CHECKS env var to the gate name (or empty for baseline)
  - Invokes backtest.persistent_runner with the EXACT verified command:
        py -3.13 -m backtest.persistent_runner \\
            --symbols EURUSD --timeframe H1 --workers 1 --no-llm
  - Captures the run's final metrics.json
  - Produces a single result JSON per experiment in --out-dir:
        {experiment_name}.json
  with at minimum:
        trades, wins, losses, win_rate, net_pnl, profit_factor,
        max_drawdown, average_trade, blocked_trades,
        remaining_blockers (per-gate pass/fail counts AFTER the bypass
        is in place — i.e. which OTHER gates still blocked trades)

SAFETY:
  - Bypasses are passed ONLY via ephemeral env vars on the subprocess.
    Nothing is written to config files; nothing persists between runs.
  - The live trading path does NOT read FOREX_BYPASS_CHECKS — it can
    only affect this script's subprocess invocations.
  - Each run uses a fresh --run-id so checkpoints/state never collide.
  - Baseline is always run first to establish the reference.

USAGE (on the user's Windows machine, from the project root):
    py -3.13 audit_tools\\run_gate_ablation.py \\
        --out-dir download\\ablation_results

Or to reproduce the user's exact command per-experiment:
    py -3.13 audit_tools\\run_gate_ablation.py \\
        --symbols EURUSD --timeframe H1 --workers 1 --no-llm \\
        --out-dir download\\ablation_results
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# ── The 7 gates the user listed, mapped to TradePermission gate names ──
# These names match risk/trade_permission._BYPASS_CHECK_ALIASES exactly,
# so TradePermission._bypass_check() will recognize them.
GATES = [
    # (experiment_name,    gate_name passed to FOREX_BYPASS_CHECKS)
    ("baseline",                  ""),                        # no bypass
    ("bypass_min_confidence",     "Min confidence"),
    ("bypass_session_quality",    "Session quality"),
    ("bypass_confluence_quality", "Confluence quality"),
    ("bypass_risk_approved",      "Risk approved"),
    ("bypass_sr_zone_alignment",  "S/R zone alignment"),
    ("bypass_valid_signal",       "Valid signal"),
    ("bypass_trend_alignment",    "Trend alignment (regime)"),
]

# Default command mirrors the user's verified backtest command:
#   py -3.13 -m backtest.persistent_runner --symbols EURUSD --timeframe H1 --workers 1 --no-llm
DEFAULT_PY = "py"
DEFAULT_PY_FLAGS = ["-3.13"]
DEFAULT_MODULE = "backtest.persistent_runner"
DEFAULT_ARGS = ["--symbols", "EURUSD", "--timeframe", "H1",
                "--workers", "1", "--no-llm"]


def _resolve_python_launcher(py: str, py_flags: list[str]) -> tuple[str, list[str]]:
    """Resolve the Python launcher to use for subprocess invocations.

    On Windows the operator's verified command uses `py -3.13`. On Linux /
    macOS the `py` launcher is typically not installed — fall back to
    `python3.13` or `python3` so the same orchestrator works in both envs.

    Returns (launcher_path_or_name, flags).
    """
    import shutil as _sh
    candidates = [
        (py, py_flags),
        ("python3.13", []),
        ("python3", []),
        ("python", []),
    ]
    for name, flags in candidates:
        if _sh.which(name):
            return name, flags
    # Last resort: return the original (will fail with a clear error)
    return py, py_flags


def find_run_dir(project_root: Path, run_id: str) -> Path:
    """Locate the run's output directory. persistent_runner writes to
    backtest_results/<run_id>/ (see backtest/persistence.RunDir)."""
    # Try common locations
    candidates = [
        project_root / "backtest_results" / run_id,
        project_root / "backtest" / "results" / run_id,
        project_root / "results" / run_id,
    ]
    for c in candidates:
        if c.is_dir():
            return c
    # Fall back to glob
    for pattern in ("backtest_results", "backtest/results", "results"):
        base = project_root / pattern
        if base.is_dir():
            for d in base.iterdir():
                if d.is_dir() and d.name == run_id:
                    return d
    return project_root / "backtest_results" / run_id  # default even if missing


def load_metrics(run_dir: Path) -> dict:
    """Load the run's metrics.json (written by persistent_runner at end)."""
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return {}
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"_error": f"failed to read metrics.json: {e}"}


def load_summary(run_dir: Path) -> dict:
    """Load the run's summary.json (continuously updated during run)."""
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def extract_result(experiment_name: str, bypassed_gate: str,
                    run_id: str, run_dir: Path, elapsed_sec: float,
                    returncode: int, stdout_tail: str) -> dict:
    """Build the per-experiment result JSON dict.

    Pulls together: trade stats (from per-symbol stats), gate_stats
    (per-gate pass/fail counts AFTER bypass is in place), and metadata.
    """
    metrics = load_metrics(run_dir)
    summary = load_summary(run_dir)
    # per_symbol stats live in either metrics['results_per_symbol'][SYMBOL]
    # or summary['symbols'][SYMBOL] — try both.
    symbol_key = "EURUSD"  # default for the verified command
    per_sym = (
        metrics.get("results_per_symbol", {}).get(symbol_key)
        or summary.get("symbols", {}).get(symbol_key)
        or {}
    )
    gate_stats = per_sym.get("gate_stats", {})
    rejection_stats = per_sym.get("rejection_stats", {}) or metrics.get("rejection_stats", {})

    # "remaining_blockers" = gates that STILL failed even with the bypass
    # active. This is the key signal for "is this gate redundantly blocking
    # vs uniquely blocking?" — if bypassing gate X doesn't change Y's fail
    # count, the two gates are independent. If bypassing X drops Y's fail
    # count to 0, Y was a downstream consequence of X.
    remaining_blockers = {
        name: counts.get("failed", 0)
        for name, counts in gate_stats.items()
        if counts.get("failed", 0) > 0
    }

    return {
        "experiment": experiment_name,
        "bypassed_gate": bypassed_gate or None,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_sec": round(elapsed_sec, 2),
        "returncode": returncode,
        # ── Trade stats (the user's required fields) ──
        "trades": per_sym.get("trades", 0),
        "wins": per_sym.get("wins", 0),
        "losses": per_sym.get("losses", 0),
        "win_rate": per_sym.get("win_rate", 0.0),
        "net_pnl": per_sym.get("net_pnl", 0.0),
        "profit_factor": per_sym.get("profit_factor", 0.0),
        "max_drawdown": per_sym.get("max_drawdown", 0.0),
        "average_trade": per_sym.get("average_trade", 0.0),
        "blocked_trades": per_sym.get("blocked_trades", 0),
        # ── Per-gate pass/fail (full picture, not just first-blocker) ──
        "gate_stats": gate_stats,
        # ── Gates that STILL blocked trades after the bypass ──
        "remaining_blockers": remaining_blockers,
        # ── Sanity: which gate this run bypassed ──
        "bypassed_gates": per_sym.get("bypassed_gates", []),
        # ── Rejection stats (early-exit buckets) ──
        "rejection_stats": rejection_stats,
        # ── Last 4KB of stdout (for debugging if returncode != 0) ──
        "stdout_tail": stdout_tail[-4096:] if returncode != 0 else "",
    }


def run_one_experiment(
    project_root: Path,
    experiment_name: str,
    bypassed_gate: str,
    py: str,
    py_flags: list[str],
    module: str,
    extra_args: list[str],
    timeout_sec: int,
) -> dict:
    """Run one experiment (one subprocess invocation).

    Sets FOREX_BYPASS_CHECKS to `bypassed_gate` (or unsets for baseline),
    runs persistent_runner, captures the result.
    """
    run_id = f"ablation_{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    cmd = [py, *py_flags, "-m", module,
           "--run-id", run_id,
           *extra_args]

    # Build clean env: copy parent env, then set/unset FOREX_BYPASS_CHECKS.
    env = os.environ.copy()
    if bypassed_gate:
        env["FOREX_BYPASS_CHECKS"] = bypassed_gate
    else:
        env.pop("FOREX_BYPASS_CHECKS", None)

    print(f"\n{'='*70}")
    print(f"[ablation] experiment: {experiment_name}")
    print(f"[ablation]   bypass: {bypassed_gate or '(none — baseline)'}")
    print(f"[ablation]   run_id: {run_id}")
    print(f"[ablation]   cmd: {' '.join(cmd)}")
    print(f"{'='*70}", flush=True)

    # Force the child's own stdout to be unbuffered even though it's
    # connected to a pipe (not a real tty). persistent_runner's logging
    # handler already calls flush() on every record, but this belts-
    # and-braces it and covers any plain print() calls in the child too.
    child_env = dict(env)
    child_env.setdefault("PYTHONUNBUFFERED", "1")

    t0 = time.perf_counter()
    output_lines: list[str] = []
    timed_out = {"hit": False}
    returncode: int

    try:
        # NOTE: capture_output=True on subprocess.run() buffers ALL
        # child stdout/stderr in memory and only returns it after the
        # process exits — that's why nothing appeared on screen while
        # baseline/each bypass experiment was running. Popen + line-by-
        # line reading streams output live while still letting us keep
        # the full text for the JSON result / failure diagnostics.
        proc = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # merge so live errors show too
            text=True,
            bufsize=1,  # line-buffered on our end
        )

        def _kill_on_timeout():
            if proc.poll() is None:
                timed_out["hit"] = True
                proc.kill()

        watchdog = threading.Timer(timeout_sec, _kill_on_timeout)
        watchdog.daemon = True
        watchdog.start()

        try:
            for line in proc.stdout:
                print(f"[{experiment_name}] {line}", end="", flush=True)
                output_lines.append(line)
        finally:
            proc.wait()
            watchdog.cancel()

        stdout_tail = "".join(output_lines)

        if timed_out["hit"]:
            returncode = -1
            print(f"[ablation] TIMEOUT after {timeout_sec}s")
        else:
            returncode = proc.returncode
            if returncode != 0:
                print(f"[ablation] FAILED (rc={returncode})")
                print(f"[ablation] output tail:\n{stdout_tail[-2000:]}")
    except Exception as e:
        returncode = -2
        stdout_tail = "".join(output_lines) + f"\nexception: {e}"
        print(f"[ablation] EXCEPTION: {e}")

    elapsed = time.perf_counter() - t0
    run_dir = find_run_dir(project_root, run_id)
    result = extract_result(
        experiment_name=experiment_name,
        bypassed_gate=bypassed_gate,
        run_id=run_id,
        run_dir=run_dir,
        elapsed_sec=elapsed,
        returncode=returncode,
        stdout_tail=stdout_tail,
    )
    print(f"[ablation] done in {elapsed:.1f}s | trades={result['trades']} "
          f"| net_pnl={result['net_pnl']:+.2f} | blockers={result['remaining_blockers']}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Controlled gate-ablation orchestrator for the Forex backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--out-dir", default="download/ablation_results",
                        help="Directory to write per-experiment JSON results")
    parser.add_argument("--py", default=DEFAULT_PY,
                        help=f'Python launcher (default: "{DEFAULT_PY}")')
    parser.add_argument("--py-flags", default=",".join(DEFAULT_PY_FLAGS),
                        help="Comma-separated flags for the launcher (default: -3.13)")
    parser.add_argument("--module", default=DEFAULT_MODULE,
                        help=f"backtest module to run (default: {DEFAULT_MODULE})")
    parser.add_argument("--symbols", nargs="+", default=["EURUSD"])
    parser.add_argument("--timeframe", default="H1")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-llm", action="store_true", default=True)
    parser.add_argument("--timeout-sec", type=int, default=3600,
                        help="Per-experiment timeout (default: 3600s)")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Only run these experiment names (default: all 8)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build extra args for persistent_runner
    extra_args = []
    extra_args += ["--symbols", *args.symbols]
    extra_args += ["--timeframe", args.timeframe]
    extra_args += ["--workers", str(args.workers)]
    if args.no_llm:
        extra_args += ["--no-llm"]

    py_flags = [f for f in args.py_flags.split(",") if f]
    # Resolve the launcher (handles Windows `py` vs Linux `python3`)
    args.py, py_flags = _resolve_python_launcher(args.py, py_flags)
    print(f"[ablation] Using Python launcher: {args.py} {' '.join(py_flags)}".strip())

    # Select experiments
    experiments = GATES
    if args.only:
        experiments = [e for e in GATES if e[0] in args.only]
        if not experiments:
            print(f"No experiments match --only {args.only}")
            return 1

    print(f"[ablation] Running {len(experiments)} experiment(s)")
    print(f"[ablation] Project root: {project_root}")
    print(f"[ablation] Output dir: {out_dir}")
    print(f"[ablation] Extra args: {extra_args}")

    results = []
    for exp_name, gate_name in experiments:
        result = run_one_experiment(
            project_root=project_root,
            experiment_name=exp_name,
            bypassed_gate=gate_name,
            py=args.py,
            py_flags=py_flags,
            module=args.module,
            extra_args=extra_args,
            timeout_sec=args.timeout_sec,
        )
        # Write per-experiment JSON immediately (so partial results survive
        # if the user Ctrl+C's midway)
        out_path = out_dir / f"{exp_name}.json"
        out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"[ablation] wrote {out_path}")
        results.append(result)

    # Write summary.json comparing all experiments
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "verified_command": f"{args.py} {' '.join(py_flags)} -m {args.module} "
                            f"{' '.join(extra_args)}",
        "experiments": [
            {
                "name": r["experiment"],
                "bypassed_gate": r["bypassed_gate"],
                "trades": r["trades"],
                "wins": r["wins"],
                "losses": r["losses"],
                "win_rate": r["win_rate"],
                "net_pnl": r["net_pnl"],
                "profit_factor": r["profit_factor"],
                "max_drawdown": r["max_drawdown"],
                "average_trade": r["average_trade"],
                "blocked_trades": r["blocked_trades"],
                "remaining_blockers": r["remaining_blockers"],
                "returncode": r["returncode"],
                "elapsed_sec": r["elapsed_sec"],
            }
            for r in results
        ],
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\n[ablation] SUMMARY written to {summary_path}")
    print(f"[ablation] {len(results)} experiment(s) complete")

    # Print comparison table
    print("\n" + "="*100)
    print(f"{'Experiment':<30} {'Trades':>7} {'WR':>7} {'Net P&L':>12} {'PF':>6} {'MaxDD':>10} {'Blocked':>8}")
    print("-"*100)
    for r in results:
        print(f"{r['experiment']:<30} {r['trades']:>7} {r['win_rate']:>6.2f}% "
              f"${r['net_pnl']:>+11.2f} {r['profit_factor']:>5.2f} "
              f"${r['max_drawdown']:>+9.2f} {r['blocked_trades']:>8}")
    print("="*100)

    return 0


if __name__ == "__main__":
    sys.exit(main())