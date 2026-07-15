"""
intelligence/confidence_calibrator.py — Confidence calibration
================================================================

Day 67 — Auto-calibrates AI confidence based on actual past performance.

The problem: AI says "91% confidence" but real win rate is only 60%.
This module tracks every trade decision + outcome and computes a
calibration curve. When the AI next claims X% confidence, we adjust
downward (or upward) based on historical accuracy at that confidence
bucket.

Buckets: 0-19, 20-39, 40-59, 60-69, 70-79, 80-89, 90-100

Storage: memory/confidence_calibration.json
Schema:
    {
        "buckets": {
            "70-79": {"count": 12, "wins": 8, "losses": 4, "win_rate": 66.7},
            ...
        },
        "total_recorded": 50,
        "overall_win_rate": 62.0,
        "last_updated": "2026-06-22T10:00:00+00:00"
    }

Calibration rule:
    adjustment = (actual_win_rate - predicted_confidence) × 0.5
    calibrated_confidence = max(0, min(100, predicted + adjustment))

The 0.5 factor means we move halfway toward reality — neither overreact
to small samples nor ignore clear miscalibration.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from utils.logger import get_logger

log = get_logger("confidence_calibrator")

CALIBRATION_PATH = Path("memory/confidence_calibration.json")

# ── Confidence buckets ──────────────────────────────────────────────
BUCKETS = [
    (0, 19, "0-19"),
    (20, 39, "20-39"),
    (40, 59, "40-59"),
    (60, 69, "60-69"),
    (70, 79, "70-79"),
    (80, 89, "80-89"),
    (90, 100, "90-100"),
]


def _bucket_for(confidence: float) -> str:
    for lo, hi, label in BUCKETS:
        if lo <= confidence <= hi:
            return label
    return "0-19"


def _empty_state() -> Dict[str, Any]:
    return {
        "buckets": {label: {"count": 0, "wins": 0, "losses": 0, "win_rate": 0.0} for _, _, label in BUCKETS},
        "total_recorded": 0,
        "overall_win_rate": 0.0,
        "last_updated": None,
    }


class ConfidenceCalibrator:
    """Tracks confidence-vs-actual-win-rate and adjusts future predictions."""

    def __init__(self):
        self._lock = threading.RLock()
        self._state = self._load()

    def _load(self) -> Dict[str, Any]:
        if not CALIBRATION_PATH.exists():
            return _empty_state()
        try:
            return json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning(f"[Calibrator] load failed: {e}")
            return _empty_state()

    def _save(self) -> None:
        try:
            CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
            self._state["last_updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            CALIBRATION_PATH.write_text(json.dumps(self._state, indent=2), encoding="utf-8")
        except Exception as e:
            log.warning(f"[Calibrator] save failed: {e}")

    def record_outcome(self, predicted_confidence: float, won: bool) -> None:
        """Record one trade outcome (predicted conf + actual win/loss)."""
        with self._lock:
            bucket = _bucket_for(predicted_confidence)
            b = self._state["buckets"].setdefault(bucket, {"count": 0, "wins": 0, "losses": 0, "win_rate": 0.0})
            b["count"] += 1
            if won:
                b["wins"] += 1
            else:
                b["losses"] += 1
            b["win_rate"] = round((b["wins"] / b["count"]) * 100, 1) if b["count"] else 0.0
            self._state["total_recorded"] = self._state.get("total_recorded", 0) + 1
            total_wins = sum(b.get("wins", 0) for b in self._state["buckets"].values())
            self._state["overall_win_rate"] = round(
                (total_wins / max(self._state["total_recorded"], 1)) * 100, 1
            )
            self._save()

    def calibrate(self, predicted_confidence: float) -> Dict[str, Any]:
        """Return calibrated confidence + adjustment details.

        Returns:
            {
                "original": float,
                "calibrated": float,
                "adjustment": float,
                "bucket": str,
                "bucket_win_rate": float,
                "bucket_count": int,
                "reason": str,
            }
        """
        with self._lock:
            bucket = _bucket_for(predicted_confidence)
            b = self._state["buckets"].get(bucket, {})
            bucket_count = b.get("count", 0)
            bucket_win_rate = b.get("win_rate", 0.0)

            # Need at least 5 samples in bucket before calibrating
            if bucket_count < 5:
                return {
                    "original": predicted_confidence,
                    "calibrated": predicted_confidence,
                    "adjustment": 0.0,
                    "bucket": bucket,
                    "bucket_win_rate": bucket_win_rate,
                    "bucket_count": bucket_count,
                    "reason": f"insufficient samples ({bucket_count} < 5) — no calibration",
                }

            # Adjustment = halfway toward actual win rate
            adjustment = (bucket_win_rate - predicted_confidence) * 0.5
            calibrated = max(0.0, min(100.0, predicted_confidence + adjustment))
            return {
                "original": round(predicted_confidence, 2),
                "calibrated": round(calibrated, 2),
                "adjustment": round(adjustment, 2),
                "bucket": bucket,
                "bucket_win_rate": bucket_win_rate,
                "bucket_count": bucket_count,
                "reason": f"bucket {bucket} actual WR={bucket_win_rate}% vs predicted={predicted_confidence}% → adjust {adjustment:+.1f}",
            }

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_recorded": self._state.get("total_recorded", 0),
                "overall_win_rate": self._state.get("overall_win_rate", 0.0),
                "buckets": self._state.get("buckets", {}),
                "last_updated": self._state.get("last_updated"),
            }


# ── singleton ───────────────────────────────────────────────────────
_CALIBRATOR: Optional[ConfidenceCalibrator] = None


def get_calibrator() -> ConfidenceCalibrator:
    global _CALIBRATOR
    if _CALIBRATOR is None:
        _CALIBRATOR = ConfidenceCalibrator()
    return _CALIBRATOR
