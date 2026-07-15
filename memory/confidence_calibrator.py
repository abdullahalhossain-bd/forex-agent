# memory/confidence_calibrator.py  —  Week 3 | Confidence Calibration
# ============================================================
# AI কি overconfident নাকি underconfident?
# Trade result দেখে confidence prediction quality measure করো
# ============================================================

import json
import os
from collections import defaultdict
from utils.logger import get_logger

log = get_logger("confidence_calibrator")

CALIBRATION_PATH = "memory/confidence_calibration.json"


class ConfidenceCalibrator:
    """
    AI-এর confidence score কতটা accurate সেটা track করে।

    Example:
        AI বলে: "90% confident" → Actual win: 55%
        → AI is OVERCONFIDENT

        AI বলে: "60% confident" → Actual win: 78%
        → AI is UNDERCONFIDENT

    এই feedback loop দিয়ে AI নিজের confidence adjust করতে শিখবে।

    Usage:
        calibrator = ConfidenceCalibrator()

        # Trade নেওয়ার সময়
        calibrator.record_prediction(confidence=82, trade_id=5)

        # Trade close হওয়ার পরে
        calibrator.record_outcome(trade_id=5, result="WIN")

        # AI-র জন্য adjustment factor পাও
        factor = calibrator.get_adjustment_factor(confidence=80)
        adjusted_conf = 80 * factor  # calibrated confidence
    """

    # Confidence buckets
    BUCKETS = ["50-59", "60-69", "70-79", "80-89", "90+"]

    def __init__(self):
        self._data = self._load()

    # ── Record ─────────────────────────────────────────────────

    def record_prediction(self, confidence: int, trade_id: int):
        """Trade নেওয়ার সময় AI-এর confidence record করো।"""
        pending = self._data.setdefault("pending", {})
        pending[str(trade_id)] = {
            "confidence": confidence,
            "bucket":     self._get_bucket(confidence),
        }
        self._save()
        log.info(f"[Calibrator] Recorded conf={confidence}% for trade #{trade_id}")

    def record_outcome(self, trade_id: int, result: str) -> dict | None:
        """Trade close হওয়ার পরে result record করো। result: 'WIN'|'LOSS'"""
        pending = self._data.get("pending", {})
        pred    = pending.pop(str(trade_id), None)

        if not pred:
            return None

        bucket = pred["bucket"]
        conf   = pred["confidence"]
        is_win = result == "WIN"

        buckets = self._data.setdefault("buckets", {})
        b       = buckets.setdefault(bucket, {"total": 0, "wins": 0, "confidences": []})
        b["total"]         += 1
        b["wins"]          += int(is_win)
        b["confidences"].append(conf)

        self._data["pending"] = pending
        self._save()

        win_rate = b["wins"] / b["total"] * 100
        avg_conf = sum(b["confidences"]) / len(b["confidences"])

        log.info(
            f"[Calibrator] Trade #{trade_id}: {result} | "
            f"Conf bucket {bucket}% → actual WR {win_rate:.1f}%"
        )
        return {
            "bucket":      bucket,
            "actual_wr":   round(win_rate, 1),
            "avg_conf":    round(avg_conf, 1),
            "calibration": self._diagnose(avg_conf, win_rate),
        }

    # ── Analysis ───────────────────────────────────────────────

    def get_calibration_report(self) -> dict:
        """
        সব bucket-এর calibration status।

        Returns:
            {
                "50-59": {"actual_wr": 48, "avg_conf": 55, "status": "WELL_CALIBRATED"},
                "80-89": {"actual_wr": 55, "avg_conf": 84, "status": "OVERCONFIDENT"},
                ...
                "overall": {"bias": "OVERCONFIDENT", "adjustment": 0.85}
            }
        """
        buckets = self._data.get("buckets", {})
        report  = {}

        for bucket, data in buckets.items():
            if data["total"] < 3:
                continue  # সামান্য data দিয়ে judge করবো না
            wr       = data["wins"] / data["total"] * 100
            avg_conf = sum(data["confidences"]) / len(data["confidences"])
            report[bucket] = {
                "total":       data["total"],
                "wins":        data["wins"],
                "actual_wr":   round(wr, 1),
                "avg_conf":    round(avg_conf, 1),
                "status":      self._diagnose(avg_conf, wr),
                "gap":         round(wr - avg_conf, 1),  # positive = underconfident
            }

        # Overall bias
        if report:
            all_gaps = [v["gap"] for v in report.values()]
            avg_gap  = sum(all_gaps) / len(all_gaps)
            if avg_gap < -10:
                bias, adj = "OVERCONFIDENT", 0.85
            elif avg_gap > 10:
                bias, adj = "UNDERCONFIDENT", 1.10
            else:
                bias, adj = "WELL_CALIBRATED", 1.00
            report["overall"] = {
                "bias":       bias,
                "avg_gap":    round(avg_gap, 1),
                "adjustment": adj,
            }

        return report

    def get_adjustment_factor(self, confidence: int) -> float:
        """
        AI-এর raw confidence-এ apply করার জন্য adjustment factor।

        Example:
            conf = 85
            factor = calibrator.get_adjustment_factor(85)  # 0.85 if overconfident
            adjusted = conf * factor  # 72.25 — more realistic
        """
        bucket  = self._get_bucket(confidence)
        buckets = self._data.get("buckets", {})
        data    = buckets.get(bucket)

        if not data or data["total"] < 5:
            return 1.0  # পর্যাপ্ত data নেই — adjust করবো না

        wr       = data["wins"] / data["total"] * 100
        avg_conf = sum(data["confidences"]) / len(data["confidences"])
        gap      = wr - avg_conf

        if gap < -15:   return 0.80   # খুব overconfident
        if gap < -8:    return 0.90   # overconfident
        if gap > 15:    return 1.15   # খুব underconfident
        if gap > 8:     return 1.05   # underconfident
        return 1.00                   # well calibrated

    def get_ai_context(self) -> str:
        """LLM prompt-এ inject করার জন্য calibration summary।"""
        report = self.get_calibration_report()
        if not report:
            return "Confidence calibration: insufficient data yet."

        overall = report.get("overall", {})
        bias    = overall.get("bias", "UNKNOWN")
        adj     = overall.get("adjustment", 1.0)

        lines = [f"Confidence Calibration Status: {bias}"]
        if bias == "OVERCONFIDENT":
            lines.append(
                f"Your confidence scores run ~{abs(overall.get('avg_gap', 0)):.0f}% "
                f"higher than actual results. Reduce stated confidence by {int((1-adj)*100)}%."
            )
        elif bias == "UNDERCONFIDENT":
            lines.append("You are more accurate than your confidence suggests. Trust your analysis more.")

        for bucket, data in report.items():
            if bucket == "overall" or not isinstance(data, dict):
                continue
            if data.get("total", 0) >= 3:
                lines.append(
                    f"  Conf {bucket}%: actual WR {data['actual_wr']}% "
                    f"({data['status']})"
                )

        return "\n".join(lines)

    def print_report(self):
        report = self.get_calibration_report()
        bar    = "═" * 52
        print(f"\n{bar}")
        print(f"  🎯  CONFIDENCE CALIBRATION REPORT")
        print(bar)

        if not report:
            print("  Not enough data yet (need 3+ closed trades per bucket)")
            print(bar + "\n")
            return

        overall = report.pop("overall", {})
        print(f"  Overall Bias  : {overall.get('bias', 'N/A')}")
        print(f"  Adjustment    : ×{overall.get('adjustment', 1.0)}")
        print(f"  Avg Gap       : {overall.get('avg_gap', 0):+.1f}% (positive = underconfident)")
        print(f"\n  {'Bucket':<10} {'Trades':>6} {'Avg Conf':>10} {'Actual WR':>10} {'Status'}")
        print(f"  {'─'*52}")
        for bucket, data in sorted(report.items()):
            print(
                f"  {bucket:<10} {data['total']:>6} "
                f"{data['avg_conf']:>9.1f}% "
                f"{data['actual_wr']:>9.1f}% "
                f"  {data['status']}"
            )
        print(bar + "\n")

    # ── Internal ───────────────────────────────────────────────

    def _get_bucket(self, conf: int) -> str:
        if conf < 60:  return "50-59"
        if conf < 70:  return "60-69"
        if conf < 80:  return "70-79"
        if conf < 90:  return "80-89"
        return "90+"

    def _diagnose(self, avg_conf: float, actual_wr: float) -> str:
        gap = actual_wr - avg_conf
        if gap < -15: return "VERY_OVERCONFIDENT"
        if gap < -5:  return "OVERCONFIDENT"
        if gap > 15:  return "VERY_UNDERCONFIDENT"
        if gap > 5:   return "UNDERCONFIDENT"
        return "WELL_CALIBRATED"

    def _load(self) -> dict:
        os.makedirs("memory", exist_ok=True)
        if os.path.exists(CALIBRATION_PATH):
            try:
                with open(CALIBRATION_PATH) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"buckets": {}, "pending": {}}

    def _save(self):
        with open(CALIBRATION_PATH, "w") as f:
            json.dump(self._data, f, indent=2)