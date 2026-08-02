# analysis/failure_cascade_detector.py
# ============================================================
# Failure Cascade Detector — sequence-level meta-labeling filter
# ============================================================
# GAP THIS FILLS (repo audit, 2026-07-31):
#   analysis/market_structure.py, analysis/order_block.py and
#   analysis/fvg_detector.py each validate ONE stage of the classic ICT
#   cascade (CHOCH -> BOS -> OB -> FVG -> entry) in isolation. None of them
#   ask the SEQUENCE-level question:
#
#       "Does THIS PARTICULAR shape of cascade -- which stages actually
#        fired, in what order, how many bars apart, how stale is it right
#        now -- historically resolve into a winning trade, or does it match
#        the shape of past failures?"
#
# This module does not generate a signal and does not touch entry logic
# (direction, SL, TP are untouched). It is a pure META-LABELING filter in
# the Lopez de Prado sense: "trust this cascade, or don't" -- see
# https://en.wikipedia.org/wiki/Meta-Labeling and this repo's own
# risk/entry_score.py / risk/institutional_entry_framework.py, which apply
# the same philosophy at the single-score level.
#
# NO-LOOKAHEAD CONTRACT:
#   extract_cascade_signature() only ever reads df up to and including the
#   last CLOSED bar passed in. It reuses:
#     - MarketStructure.analyze()   (already bar-by-bar causal, see its own
#       docstring: "BOS/CHoCH is only ever evaluated against swings that
#       were ALREADY confirmed strictly before the current bar's close")
#     - OrderBlockDetector.detect(closed_bars_only=True)
#     - FVGDetector.detect()
#   No future bar is ever read. See bias-and-validation.md.
#
# CALIBRATION STATUS:
#   Ships UNCALIBRATED. Every check returns severity="WARNING" (never
#   "BLOCK") until scripts/calibrate_failure_cascade.py has produced
#   data/failure_cascade_stats.json with >= MIN_SAMPLES_FOR_TRUST samples
#   per signature bucket. This mirrors the documented reason
#   entry_score.py / institutional_entry_framework.py were kept LOG-ONLY
#   in core/obsolete.py: promoting an unvalidated filter to a hard block
#   risks rejecting winners, which violates the stated objective
#   (Winner Preservation >= 95% before increasing Loss Rejection).
# ============================================================

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from utils.logger import get_logger
from analysis.market_structure import MarketStructure
from analysis.order_block import OrderBlockDetector
from analysis.fvg_detector import FVGDetector

log = get_logger("failure_cascade_detector")

STAGE_ORDER = ["CHOCH", "BOS", "OB", "FVG"]
MIN_SAMPLES_FOR_TRUST = 30
DEFAULT_LOOKBACK_BARS = 60
DEFAULT_STATS_PATH = "data/failure_cascade_stats.json"

# analysis/market_structure.py uses UP/DOWN; order_block.py and
# fvg_detector.py use BULLISH/BEARISH. Normalize on BULLISH/BEARISH here.
_DIR_MAP = {"UP": "BULLISH", "DOWN": "BEARISH"}


@dataclass
class CascadeSignature:
    direction: str          # BULLISH | BEARISH
    stages_present: tuple   # subset of STAGE_ORDER, in actual chronological order
    gap_buckets: tuple      # "TIGHT" (<5 bars) | "NORMAL" (5-15) | "WIDE" (>15) between consecutive stages
    bars_since_last_stage: int  # staleness of the cascade as of the current bar

    def key(self) -> str:
        """Canonical (exact) bucket key used to look up empirical calibration stats."""
        stages = ">".join(self.stages_present) if self.stages_present else "NONE"
        gaps = "-".join(self.gap_buckets) if self.gap_buckets else "NONE"
        return f"{self.direction}|{stages}|{gaps}"

    def coarse_key(self) -> str:
        """
        Round-15: the exact key() space is combinatorially too large
        (4 stages x orderings x 3 gap buckets per pair) to ever collect
        n>=MIN_SAMPLES_FOR_TRUST samples for most buckets — measured:
        even after calibrating 33 real pairs of H1 history, almost every
        exact signature seen in a fresh backtest had 0-3 samples.
        This coarser key collapses to (direction, stage COUNT, dominant
        gap category) — a few dozen possible buckets instead of
        thousands — so real samples actually accumulate. Used as a
        fallback when the exact key isn't trusted yet.
        """
        n_stages = len(self.stages_present)
        if self.gap_buckets:
            from collections import Counter
            dominant_gap = Counter(self.gap_buckets).most_common(1)[0][0]
        else:
            dominant_gap = "NONE"
        return f"{self.direction}|STAGES={n_stages}|DOMINANT_GAP={dominant_gap}"


def _gap_bucket(bars: int) -> str:
    if bars < 5:
        return "TIGHT"
    if bars <= 15:
        return "NORMAL"
    return "WIDE"


def extract_cascade_signature(
    df: pd.DataFrame,
    direction: str,
    lookback_bars: int = DEFAULT_LOOKBACK_BARS,
) -> Optional[CascadeSignature]:
    """
    Build a cascade signature using ONLY bars up to and including the last
    row of `df`.

    Caller contract (same as OrderBlockDetector / FVGDetector):
    `df` must contain CLOSED bars only, and must already have an 'atr'
    column (see data/indicators.py).

    direction: "BUY"/"SELL" (or "BULLISH"/"BEARISH"/"LONG") -- the direction
    of the CANDIDATE trade being evaluated right now.

    Returns None if there isn't enough data to build a signature (fails
    closed -- caller should treat that as "check skipped", not "check
    failed").
    """
    if len(df) < 30:
        return None
    if "atr" not in df.columns:
        log.warning("[FailureCascade] 'atr' column missing on df — cannot build signature")
        return None

    wanted_dir = "BULLISH" if direction.upper() in ("BUY", "BULLISH", "LONG") else "BEARISH"
    n = len(df)
    window_start = max(0, n - lookback_bars)

    # --- Stage 1 & 2: BOS / CHoCH history (already bar-by-bar causal) ---
    ms_state = MarketStructure().analyze(df)
    struct_events = [
        e for e in ms_state["events"]
        if e["index"] >= window_start and _DIR_MAP.get(e["direction"]) == wanted_dir
    ]

    # --- Stage 3: Order Blocks ---
    obs = OrderBlockDetector().detect(df, closed_bars_only=True)
    obs = [o for o in obs if o["index"] >= window_start and o["direction"] == wanted_dir]

    # --- Stage 4: Fair Value Gaps ---
    fvgs = FVGDetector().detect(df)
    fvgs = [g for g in fvgs if g["index"] >= window_start and g["direction"] == wanted_dir]

    # --- Merge into one time-ordered event list ---
    events = []
    for e in struct_events:
        events.append((e["type"], e["index"]))   # "BOS" or "CHOCH"
    for o in obs:
        events.append(("OB", o["index"]))
    for g in fvgs:
        events.append(("FVG", g["index"]))

    if not events:
        return None

    events.sort(key=lambda t: t[1])

    # Keep only the most recent occurrence of each stage within the window —
    # a cascade is defined by its latest formation of each stage type.
    last_of_stage = {}
    for stage, idx in events:
        last_of_stage[stage] = idx

    present_sorted_by_time = sorted(last_of_stage.items(), key=lambda t: t[1])
    stages_present = tuple(s for s, _ in present_sorted_by_time)

    gap_buckets = tuple(
        _gap_bucket(present_sorted_by_time[i][1] - present_sorted_by_time[i - 1][1])
        for i in range(1, len(present_sorted_by_time))
    )

    bars_since_last_stage = (n - 1) - present_sorted_by_time[-1][1]

    return CascadeSignature(
        direction=wanted_dir,
        stages_present=stages_present,
        gap_buckets=gap_buckets,
        bars_since_last_stage=bars_since_last_stage,
    )


def _load_stats(stats_path: str) -> dict:
    if not stats_path or not os.path.exists(stats_path):
        return {}
    try:
        with open(stats_path, "r") as f:
            return json.load(f)
    except Exception as exc:
        log.warning(f"[FailureCascade] Could not load stats file {stats_path}: {exc}")
        return {}


def score_cascade(signature: CascadeSignature, stats_path: str = DEFAULT_STATS_PATH) -> dict:
    """
    Look up (or, absent calibration, heuristically flag) a cascade signature.

    Returns:
        {
            "fail_rate":      float | None,  # empirical, only if calibrated
            "n_samples":      int,
            "calibrated":     bool,
            "heuristic_flag": str | None,
        }
    """
    stats = _load_stats(stats_path)
    bucket = stats.get(signature.key())

    if bucket and bucket.get("n_samples", 0) >= MIN_SAMPLES_FOR_TRUST:
        return {
            "fail_rate": bucket["fail_rate"],
            "n_samples": bucket["n_samples"],
            "calibrated": True,
            "heuristic_flag": None,
            "bucket_level": "exact",
        }

    # Round-15: exact bucket not trusted (or absent) — try the coarser
    # (direction, stage-count, dominant-gap) bucket before falling back
    # to the heuristic. See CascadeSignature.coarse_key().
    coarse_bucket = stats.get(signature.coarse_key())
    if coarse_bucket and coarse_bucket.get("n_samples", 0) >= MIN_SAMPLES_FOR_TRUST:
        return {
            "fail_rate": coarse_bucket["fail_rate"],
            "n_samples": coarse_bucket["n_samples"],
            "calibrated": True,
            "heuristic_flag": None,
            "bucket_level": "coarse",
        }

    # --- Provisional heuristic only (NOT a validated statistic). Never used
    # to justify a BLOCK severity -- see module docstring. ---
    heuristic_flag = None
    if len(signature.stages_present) < 2:
        heuristic_flag = "incomplete cascade (<2 confirmed stages) — low evidentiary weight"
    elif signature.gap_buckets.count("TIGHT") == len(signature.gap_buckets) and len(signature.gap_buckets) >= 2:
        heuristic_flag = "all stages compressed into a few bars — looks like one impulsive spike, not a developed cascade"
    elif signature.bars_since_last_stage > 20:
        heuristic_flag = f"cascade is stale (last stage confirmed {signature.bars_since_last_stage} bars ago) — signal aging risk"

    return {
        "fail_rate": None,
        "n_samples": bucket.get("n_samples", 0) if bucket else 0,
        "calibrated": False,
        "heuristic_flag": heuristic_flag,
        "bucket_level": None,
    }


def check_failure_cascade(
    df: pd.DataFrame,
    symbol: str,
    direction: str,
    ind_ctx: Optional[dict] = None,
    lookback_bars: int = DEFAULT_LOOKBACK_BARS,
    stats_path: str = DEFAULT_STATS_PATH,
) -> dict:
    """
    Entry-quality-guardrail-style check. Returns the same dict shape as
    risk.entry_quality_guardrails.EntryQualityResult.to_dict(), so it can be
    appended to the `results` list in risk/trade_permission.py exactly the
    way the existing 13 checks are (see run_all_entry_quality_checks).

    Severity is ALWAYS "WARNING" in this version. Do not change this to
    "BLOCK" until calibration exists — see module docstring.
    """
    flag_name = "failure_cascade_signature"
    sig = extract_cascade_signature(df, direction, lookback_bars=lookback_bars)

    if sig is None:
        return {
            "flag_name": flag_name,
            "passed": True,
            "reason": "Insufficient data to build a cascade signature — check skipped, not failed.",
            "severity": "WARNING",
            "details": {},
        }

    result = score_cascade(sig, stats_path=stats_path)

    if result["calibrated"]:
        # Threshold is a placeholder — review the actual fail_rate
        # distribution across buckets after calibration before trusting 0.5.
        passed = result["fail_rate"] < 0.5
        reason = (
            f"Cascade {sig.key()} has empirical fail rate "
            f"{result['fail_rate']:.0%} over {result['n_samples']} historical samples."
        )
    else:
        passed = result["heuristic_flag"] is None
        reason = result["heuristic_flag"] or (
            f"Cascade {sig.key()} observed, not yet calibrated "
            f"({result['n_samples']} samples logged so far) — logging only, no penalty applied."
        )

    return {
        "flag_name": flag_name,
        "passed": bool(passed),
        "reason": reason,
        "severity": "WARNING",
        "details": {
            "signature": sig.key(),
            "stages_present": sig.stages_present,
            "gap_buckets": sig.gap_buckets,
            "bars_since_last_stage": sig.bars_since_last_stage,
            **result,
        },
    }
