# hybrid/confidence_calibrator.py  —  Day 49 Bonus #2 | Confidence Calibration ⭐
# ============================================================
# Doc Bonus #2:
#     "85% confidence setup — বাস্তবে win rate কত? তারপর confidence
#      adjust করবে।"
#
# এটা একটা classic ML concept-এর lightweight version — "calibration".
# একটা ভালো-calibrated model-এর 80% confidence trade-গুলো বাস্তবেই
# প্রায় ৮০% সময় win করা উচিত। বাস্তবে practice-এ models সাধারণত
# overconfident হয় — তাই calibration historical win-rate দিয়ে
# confidence-কে "সত্যি" সংখ্যার কাছে টেনে আনে।
#
# Method: bucket-based calibration (Platt scaling-এর simpler বিকল্প,
# কোনো extra ML library লাগে না)।
#   1. প্রতিটা closed trade-কে confidence bucket-এ ভাগ করো (0-50,
#      50-60, 60-70, 70-80, 80-90, 90-100)
#   2. প্রতি bucket-এর actual win rate বের করো
#   3. নতুন prediction আসলে তার bucket-এর historical win rate দিয়ে
#      blend করো (raw confidence-কে পুরোপুরি override না করে — sample
#      size কম থাকলে raw confidence-কেই বেশি weight দেওয়া হয়)
#
# Data source: learning_agent.py-এর memory/trade_memory.json
# (LearningAgent ইতিমধ্যে confidence + result/outcome save করে —
# এই module শুধু read করে analyze করে, নতুন storage বানায়নি)।
# ============================================================

import json
import os
import threading
import time

from utils.logger import get_logger
from core.constants import MEMORY_DIR

log = get_logger("confidence_calibrator")

TRADE_MEMORY_PATH = str(MEMORY_DIR / "trade_memory.json")

BUCKETS = [(0, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 101)]
MIN_SAMPLES_FOR_TRUST = 10   # bucket-এ এর কম sample থাকলে raw confidence-কেই বেশি বিশ্বাস করো

# H-H1 fix: build_calibration_report() reads + reprocesses the ENTIRE trade
# history on every call, but calibrate() calls it once per trading decision
# (per symbol, per cycle) — a full-file reparse per signal is wasteful.
# Cache is short-lived AND invalidated whenever the file's mtime changes
# (i.e. a trade just closed), so it never serves stale data past a real
# update — it only skips redundant re-reads within the same instant.
_REPORT_CACHE_TTL_SEC = 5.0


class ConfidenceCalibrator:
    """
    Usage:
        cal = ConfidenceCalibrator()
        report = cal.build_calibration_report()
        adjusted = cal.calibrate(raw_confidence=85)
        # adjusted ~ 85-এর bucket-এ historical win rate যদি ৭০% হয়,
        # তাহলে blend করে কিছুটা কমানো confidence ফেরত দেবে
    """

    def __init__(self, memory_path: str = TRADE_MEMORY_PATH):
        self.memory_path = memory_path
        # H-C3 fix: guards both the file read and the cached-report state
        # below. ExecutionRouter/FlowController may call calibrate() from
        # multiple symbol-threads concurrently — without this, concurrent
        # reads of trade_memory.json (which LearningAgent may be writing to
        # at the same time) could interleave/corrupt, and the cache fields
        # could be read/written inconsistently across threads.
        self._lock = threading.Lock()
        self._report_cache = None
        self._report_cache_mtime = None
        self._report_cache_time = 0.0

    # ═══════════════════════════════════════════════════════
    # 1. BUCKET ANALYSIS
    # ═══════════════════════════════════════════════════════

    def build_calibration_report(self) -> dict:
        """প্রতিটা confidence bucket-এর জন্য predicted vs actual win rate বের করো।

        H-C3/H-H1 fix: reads happen under a lock (safe against concurrent
        writers/readers of trade_memory.json), and the resulting report is
        cached until either the file's mtime changes or _REPORT_CACHE_TTL_SEC
        elapses — avoids re-parsing the full trade history on every single
        calibrate() call.
        """
        with self._lock:
            current_mtime = self._file_mtime()
            now = time.monotonic()
            cache_fresh = (
                self._report_cache is not None
                and current_mtime == self._report_cache_mtime
                and (now - self._report_cache_time) < _REPORT_CACHE_TTL_SEC
            )
            if cache_fresh:
                return self._report_cache

            history = self._load_closed_trades_locked()
            if history is None:
                # H-C8 fix: corrupted/unreadable file — reuse last good
                # report if we have one, rather than collapsing calibration
                # to "no data" on a transient glitch.
                if self._report_cache is not None:
                    log.warning("[ConfidenceCalibrator] Reusing last good calibration report after read failure")
                    return self._report_cache
                history = []
            report = {}

            for lo, hi in BUCKETS:
                bucket_trades = [
                    t for t in history
                    if t.get("confidence") is not None and lo <= t["confidence"] < hi
                ]
                n = len(bucket_trades)
                if n == 0:
                    report[f"{lo}-{hi}"] = {
                        "samples": 0, "predicted_avg": None,
                        "actual_win_rate": None, "trustworthy": False,
                    }
                    continue

                wins = sum(1 for t in bucket_trades if t.get("result") == "WIN")
                predicted_avg = round(sum(t["confidence"] for t in bucket_trades) / n, 1)
                actual_win_rate = round(wins / n * 100, 1)

                report[f"{lo}-{hi}"] = {
                    "samples": n,
                    "predicted_avg": predicted_avg,
                    "actual_win_rate": actual_win_rate,
                    "trustworthy": n >= MIN_SAMPLES_FOR_TRUST,
                    "gap": round(predicted_avg - actual_win_rate, 1),
                }

            self._report_cache = report
            self._report_cache_mtime = current_mtime
            self._report_cache_time = now
            return report

    # ═══════════════════════════════════════════════════════
    # 2. CALIBRATE A NEW PREDICTION  ⭐⭐⭐⭐⭐
    # ═══════════════════════════════════════════════════════

    def calibrate(self, raw_confidence: int) -> dict:
        """
        নতুন trade-এর raw confidence নিয়ে historical bucket win-rate দিয়ে
        adjust করো।

        Blend logic:
            sample কম (< MIN_SAMPLES_FOR_TRUST)  → raw confidence-কেই বেশি weight (90%)
            sample যথেষ্ট (>= MIN_SAMPLES_FOR_TRUST) → historical win rate-কে বেশি weight (70%)

        এভাবে শুরুতে (যখন data কম) AI-এর raw judgment respect করা হয়,
        পরে যত বেশি trade হবে তত বেশি reality-grounded calibration হবে।
        """
        # H-C1/H-M1 fix: raw_confidence comes from DecisionValidator's
        # final_score, which can legitimately be a float and was never
        # range-checked. Coerce + clamp so bucket lookup and the blend math
        # below can't receive a wrong-typed or out-of-range value.
        try:
            raw_confidence = int(round(float(raw_confidence)))
        except (TypeError, ValueError):
            log.warning(f"[ConfidenceCalibrator] Invalid raw_confidence={raw_confidence!r} — defaulting to 0")
            raw_confidence = 0
        raw_confidence = max(0, min(100, raw_confidence))

        report = self.build_calibration_report()
        bucket_key = self._find_bucket(raw_confidence)
        bucket = report.get(bucket_key, {})

        if not bucket or bucket.get("samples", 0) == 0:
            return {
                "raw_confidence": raw_confidence,
                "calibrated_confidence": raw_confidence,
                "bucket": bucket_key,
                "adjustment": 0,
                "note": "No historical data for this bucket — using raw confidence",
            }

        actual_win_rate = bucket["actual_win_rate"]
        n = bucket["samples"]

        if n >= MIN_SAMPLES_FOR_TRUST:
            weight_actual = 0.70
        else:
            # sample size বাড়ার সাথে সাথে gradually actual win-rate-কে বেশি বিশ্বাস করো
            weight_actual = 0.30 * (n / MIN_SAMPLES_FOR_TRUST)

        weight_raw = 1 - weight_actual
        calibrated = round(raw_confidence * weight_raw + actual_win_rate * weight_actual)
        calibrated = max(0, min(99, calibrated))

        result = {
            "raw_confidence": raw_confidence,
            "calibrated_confidence": calibrated,
            "bucket": bucket_key,
            "bucket_samples": n,
            "bucket_actual_win_rate": actual_win_rate,
            "adjustment": calibrated - raw_confidence,
            "note": (
                f"Bucket {bucket_key}% historically wins {actual_win_rate}% of the time "
                f"(n={n}) — {'trusted' if n >= MIN_SAMPLES_FOR_TRUST else 'low-sample, partial trust'}"
            ),
        }

        if abs(result["adjustment"]) >= 10:
            log.info(
                f"[ConfidenceCalibrator] ⚠️ Large adjustment: {raw_confidence}% → "
                f"{calibrated}% (bucket {bucket_key} actual win-rate {actual_win_rate}%)"
            )
        return result

    def _find_bucket(self, confidence: int) -> str:
        for lo, hi in BUCKETS:
            if lo <= confidence < hi:
                return f"{lo}-{hi}"
        return f"{BUCKETS[-1][0]}-{BUCKETS[-1][1]}"

    # ═══════════════════════════════════════════════════════
    # 3. OVERALL CALIBRATION HEALTH  (is the AI over/under-confident?)
    # ═══════════════════════════════════════════════════════

    def get_calibration_health(self) -> dict:
        """
        Overall — AI কি systematically overconfident না underconfident?
        Trustworthy bucket-গুলোর gap (predicted_avg - actual_win_rate)
        average করে বোঝা যায়।
        """
        report = self.build_calibration_report()
        trustworthy = [b for b in report.values() if b.get("trustworthy")]

        if not trustworthy:
            return {"status": "INSUFFICIENT_DATA", "avg_gap": None}

        avg_gap = round(sum(b["gap"] for b in trustworthy) / len(trustworthy), 1)

        if avg_gap > 10:
            status = "OVERCONFIDENT"
        elif avg_gap < -10:
            status = "UNDERCONFIDENT"
        else:
            status = "WELL_CALIBRATED"

        return {"status": status, "avg_gap": avg_gap, "trustworthy_buckets": len(trustworthy)}

    # ═══════════════════════════════════════════════════════
    # INTERNAL
    # ═══════════════════════════════════════════════════════

    def _file_mtime(self):
        try:
            return os.path.getmtime(self.memory_path)
        except OSError:
            return None

    def _load_closed_trades_locked(self):
        """Caller must already hold self._lock. H-C8 fix: on a corrupted/
        unreadable/malformed file, return None (sentinel) instead of silently
        returning [] — the caller can then fall back to the last good cached
        report rather than wiping calibration to 'no data' on a transient
        read glitch."""
        if not os.path.exists(self.memory_path):
            return []
        try:
            with open(self.memory_path, encoding="utf-8") as f:
                history = json.load(f)
            if not isinstance(history, list):
                raise ValueError(f"expected a list, got {type(history).__name__}")
        except Exception as e:
            log.error(f"[ConfidenceCalibrator] Could not load trade memory: {e}")
            return None
        return [t for t in history if isinstance(t, dict) and t.get("result") in ("WIN", "LOSS")]

    def _load_closed_trades(self) -> list:
        """Thread-safe public entry point (kept for backward compatibility
        with any external caller)."""
        with self._lock:
            history = self._load_closed_trades_locked()
            return history if history is not None else []

    # ═══════════════════════════════════════════════════════
    # PRINT SUMMARY
    # ═══════════════════════════════════════════════════════

    def print_report(self) -> None:
        report = self.build_calibration_report()
        health = self.get_calibration_health()
        bar = "═" * 54
        print(f"\n{bar}")
        print("  🎯  CONFIDENCE CALIBRATION  (Day 49)")
        print(bar)
        print(f"  Overall status : {health['status']}  (avg gap: {health.get('avg_gap')})")
        print()
        for bucket, stats in report.items():
            if stats["samples"] == 0:
                print(f"  {bucket:<8}%  — no data")
                continue
            trust = "✅" if stats["trustworthy"] else "🔸"
            print(
                f"  {bucket:<8}%  {trust}  n={stats['samples']:<4} "
                f"predicted_avg={stats['predicted_avg']:<6} "
                f"actual_win={stats['actual_win_rate']:<6} "
                f"gap={stats['gap']:+}"
            )
        print(bar + "\n")