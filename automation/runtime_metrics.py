# automation/runtime_metrics.py  —  Day 51 | Runtime Metrics & Reporting ⭐
# ============================================================
# Doc Section 2 (4-Hour Test measurements), 5 (Speed Measurement),
# 8 (Autonomous Report Generation) — সবগুলোই "সংখ্যা গোনা ও সময় মাপা"
# এই common category, তাই একটা module-এ একসাথে।
#
# একটা context manager (`timer()`) দেয় যা দিয়ে pipeline-এর যেকোনো
# stage (data fetch, analysis, vision, execution) wrap করলেই automatic
# timing record হয়ে যায় — AutonomousRunner প্রতিটা stage-কে এটা দিয়েই
# মাপবে, আলাদা করে time.time() লেখার দরকার নেই।
# ============================================================

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone

from utils.logger import get_logger

log = get_logger("runtime_metrics")

REPORT_DIR = "logs/automation_reports"


class RuntimeMetrics:
    """
    Usage:
        metrics = RuntimeMetrics()
        metrics.start_session()

        with metrics.timer("data_fetch"):
            data = fetch_data()

        with metrics.timer("analysis"):
            result = analyze(data)

        metrics.record_cycle(symbol="EURUSD", outcome="EXECUTED")
        metrics.record_error("VISION_TIMEOUT", action="retry", result="success")
        metrics.record_reconnect()

        report = metrics.build_report()
        metrics.save_report(report)
    """

    def __init__(self):
        self._session_start = None
        # CRITICAL FIX: use deque with maxlen to prevent unbounded memory growth.
        # Previous code used list() which grows forever — RAM leak on long runs.
        from collections import deque
        self._stage_timings: dict[str, deque] = {}   # stage_name -> deque([durations_ms...])
        self._cycles: deque = deque(maxlen=10000)     # cap at 10k cycles
        self._errors: deque = deque(maxlen=5000)      # cap at 5k errors
        self._reconnects = 0
        self._signals_found = 0
        self._trades_taken = 0

    # ═══════════════════════════════════════════════════════
    # SESSION LIFECYCLE
    # ═══════════════════════════════════════════════════════

    def start_session(self) -> None:
        self._session_start = time.time()
        log.info("[RuntimeMetrics] Session started")

    def session_duration_sec(self) -> float:
        if not self._session_start:
            return 0.0
        return round(time.time() - self._session_start, 1)

    # ═══════════════════════════════════════════════════════
    # STAGE TIMING  (doc Section 5 — Speed Measurement)
    # ═══════════════════════════════════════════════════════

    @contextmanager
    def timer(self, stage_name: str):
        """
        with metrics.timer("vision"):
            ... vision call ...
        স্বয়ংক্রিয়ভাবে milliseconds-এ duration record করে — exception উঠলেও
        timing record হয় (finally), শুধু exception আবার raise হয়ে চলে যায়।
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 1)
            # FIX: use deque with maxlen for bounded memory
            from collections import deque
            if stage_name not in self._stage_timings:
                self._stage_timings[stage_name] = deque(maxlen=1000)
            self._stage_timings[stage_name].append(duration_ms)

    def get_stage_stats(self, stage_name: str) -> dict:
        durations = self._stage_timings.get(stage_name, [])
        if not durations:
            return {"count": 0, "avg_ms": None, "min_ms": None, "max_ms": None}
        return {
            "count": len(durations),
            "avg_ms": round(sum(durations) / len(durations), 1),
            "min_ms": round(min(durations), 1),
            "max_ms": round(max(durations), 1),
        }

    def get_all_stage_stats(self) -> dict:
        return {stage: self.get_stage_stats(stage) for stage in self._stage_timings}

    def average_decision_time_sec(self) -> float:
        """
        doc-এর "Final: Average decision time: 5.5 seconds" — data_fetch +
        analysis + vision + execution stage-গুলোর average sum (cycle-প্রতি)।
        """
        relevant = ["data_fetch", "analysis", "vision", "execution"]
        total_ms = sum(self.get_stage_stats(s).get("avg_ms") or 0 for s in relevant)
        return round(total_ms / 1000, 2)

    # ═══════════════════════════════════════════════════════
    # CYCLE / SIGNAL / TRADE COUNTING  (doc Section 2)
    # ═══════════════════════════════════════════════════════

    def record_cycle(self, symbol: str, outcome: str) -> None:
        """outcome: FlowController.run_cycle()-এর 'stage' ফিল্ড (EXECUTED, NO_TRADE,
        CONFLICT_BLOCKED, RISK_REJECTED, ইত্যাদি) — সরাসরি pass করো।"""
        self._cycles.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol, "outcome": outcome,
        })
        if outcome not in ("NO_TRADE", "DECISION_WAIT"):
            self._signals_found += 1
        if outcome == "EXECUTED":
            self._trades_taken += 1

    def record_error(self, error_type: str, action: str = None, result: str = None) -> None:
        self._errors.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": error_type, "action": action, "result": result,
        })

    def record_reconnect(self) -> None:
        self._reconnects += 1

    # ═══════════════════════════════════════════════════════
    # FINAL REPORT  (doc Section 8 — Autonomous Report Generation)
    # ═══════════════════════════════════════════════════════

    def build_report(self, system_status: str = None) -> dict:
        duration_sec = self.session_duration_sec()
        crashes = sum(1 for e in self._errors if e["type"] == "CRASH")

        status = system_status or self._infer_status(crashes)

        report = {
            "duration_hours": round(duration_sec / 3600, 2),
            "duration_sec": duration_sec,
            "market_scans": len(self._cycles),
            "signals_found": self._signals_found,
            "trades_taken": self._trades_taken,
            "errors": len(self._errors),
            "crashes": crashes,
            "reconnects": self._reconnects,
            "successful_cycles": sum(1 for c in self._cycles if "ERROR" not in c["outcome"]),
            "stage_timings": self.get_all_stage_stats(),
            "average_decision_time_sec": self.average_decision_time_sec(),
            "system_status": status,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        return report

    def _infer_status(self, crashes: int) -> str:
        if crashes > 0:
            return "FAIL"
        if len(self._errors) > 10:
            return "WARN"
        return "PASS"

    def save_report(self, report: dict = None) -> str:
        os.makedirs(REPORT_DIR, exist_ok=True)
        report = report or self.build_report()
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = os.path.join(REPORT_DIR, f"report_{ts}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)
            log.info(f"[RuntimeMetrics] Report saved → {path}")
        except Exception as e:
            log.error(f"[RuntimeMetrics] Could not save report: {e}")
        return path

    # ═══════════════════════════════════════════════════════
    # PRINT SUMMARY  (doc-এর exact output format অনুসরণ করে)
    # ═══════════════════════════════════════════════════════

    def print_report(self, report: dict = None) -> None:
        report = report or self.build_report()
        bar = "═" * 50
        icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(report["system_status"], "⚪")

        print(f"\n{bar}")
        print("  📊  AUTOMATION TEST REPORT  (Day 51)")
        print(bar)
        print(f"  Duration         : {report['duration_hours']} hours")
        print(f"  Market Scans     : {report['market_scans']}")
        print(f"  Signals Found    : {report['signals_found']}")
        print(f"  Trades Taken     : {report['trades_taken']}")
        print(f"  Errors           : {report['errors']}")
        print(f"  Crashes          : {report['crashes']}")
        print(f"  Reconnects       : {report['reconnects']}")
        print(f"  Successful Cycles: {report['successful_cycles']}")
        print()
        print(f"  ── Speed (avg ms) ──")
        for stage, stats in report["stage_timings"].items():
            if stats["count"]:
                print(f"  {stage:<14} {stats['avg_ms']:>8} ms  (n={stats['count']})")
        print(f"\n  Average decision time: {report['average_decision_time_sec']}s")
        print()
        print(f"  System Status    : {icon} {report['system_status']}")
        print(bar + "\n")