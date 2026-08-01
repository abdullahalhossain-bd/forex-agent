"""
Calibrate Failure Cascade Detector

Walk-forward calibration of cascade-pattern fail rates.
Optimised: runs FVG / MarketStructure / OrderBlock detection ONCE on
full data, then walks forward by filtering pre-computed event lists by
index — zero re-computation per step.

Correct API usage:
  - FVGDetector().detect(df)         → list[dict]
  - MarketStructure().analyze(df)    → dict  {events: [...], swings: [...], trend}
  - OrderBlockDetector().detect(df)  → list[dict]
"""

import argparse
import json
import logging
import sys
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("calibrate")

# Needed before importing analysis.* below.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# ── Import the ACTUAL production key-building logic ─────────────────
# Do NOT reimplement key/bucket formatting here. A prior version of this
# script duplicated CascadeSignature.key()'s logic by hand (direction
# naming, gap-bucket thresholds, gap separator, stage-name casing) and it
# silently drifted out of sync with analysis/failure_cascade_detector.py —
# meaning every calibration run produced a stats file that
# check_failure_cascade() could never successfully look up. Importing the
# real symbols makes that class of bug structurally impossible: if
# production changes its key format, this script changes with it.
from analysis.failure_cascade_detector import (
    CascadeSignature,
    _gap_bucket,
    STAGE_ORDER,
    _DIR_MAP,
    MIN_SAMPLES_FOR_TRUST,
)

# ── Config ────────────────────────────────────────────────────────────
HOLDING_PERIOD_BARS = 10
TP_ATR_MULT = 1.5
SL_ATR_MULT = 1.0
ATR_PERIOD = 14
STEP = 10                        # walk-forward step (bars)
LOOKBACK = 60                    # cascade lookback window
STATS_PATH = Path(__file__).resolve().parent.parent / "data" / "failure_cascade_stats.json"

# _label_outcome() returns "WIN" / "LOSS" / "TIMEOUT" — this maps those to the
# plural keys used in pattern_stats / results dicts. Single source of truth so
# the two never drift out of sync again.
_OUTCOME_KEY_MAP = {"WIN": "wins", "LOSS": "losses", "TIMEOUT": "timeouts"}

# ── Load & Prepare ────────────────────────────────────────────────────
def _load_and_prepare(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    col_map = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("datetime_utc", "time", "datetime", "timestamp", "date"):
            col_map[c] = "time"
        elif cl in ("open", "open_price"):
            col_map[c] = "open"
        elif cl in ("high", "high_price"):
            col_map[c] = "high"
        elif cl in ("low", "low_price"):
            col_map[c] = "low"
        elif cl in ("close", "close_price"):
            col_map[c] = "close"
        elif cl in ("volume", "vol", "tick_volume"):
            col_map[c] = "volume"
    df = df.rename(columns=col_map)
    for req in ("time", "open", "high", "low", "close"):
        if req not in df.columns:
            raise ValueError(f"Missing required column: {req}")
    if "volume" not in df.columns:
        df["volume"] = 0
        logger.warning("'volume' column missing — using dummy 0")
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def _add_atr(df: pd.DataFrame) -> pd.DataFrame:
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(df)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = np.zeros(n)
    atr[:ATR_PERIOD] = np.nan
    atr[ATR_PERIOD - 1] = np.mean(tr[:ATR_PERIOD])
    alpha = 1.0 / ATR_PERIOD
    for i in range(ATR_PERIOD, n):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
    df["atr"] = atr
    return df


# ── Detect ONCE on full data ─────────────────────────────────────────
def _detect_all(df: pd.DataFrame):
    """
    Run all three detectors on the FULL dataframe ONCE.
    Returns (events, fvgs, obs) — pre-computed event lists.
    Each item has an 'index' field we can filter on during walk-forward.
    """
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))

    # 1) MarketStructure → BOS / CHoCH events
    from analysis.market_structure import MarketStructure
    ms = MarketStructure()
    state = ms.analyze(df, strength=2)
    events = state.get("events", [])
    print(f"  MarketStructure: {len(events)} BOS/CHoCH events, trend={state.get('trend')}")

    # 2) FVGs
    from analysis.fvg_detector import FVGDetector
    fvg_det = FVGDetector()
    fvgs = fvg_det.detect(df, max_results=0)
    print(f"  FVGDetector: {len(fvgs)} fair value gaps")

    # 3) Order Blocks (max_results=0 = uncapped)
    from analysis.order_block import OrderBlockDetector
    ob_det = OrderBlockDetector(require_structure_break=False, require_fvg=False)
    obs = ob_det.detect(df, max_results=0)
    print(f"  OrderBlockDetector: {len(obs)} order blocks")

    return events, fvgs, obs


# ── Extract cascade from pre-computed lists ──────────────────────────
def _extract_cascade(events: list, fvgs: list, obs: list,
                      direction: str, current_idx: int,
                      lookback: int = LOOKBACK):
    """
    Build a CascadeSignature from pre-computed event lists — mirrors
    extract_cascade_signature() in analysis/failure_cascade_detector.py
    bar-by-bar, just against pre-filtered lists instead of recomputing
    detectors every step. Only considers events with index in
    (current_idx - lookback, current_idx].

    Returns (signature, identity) where signature is a real CascadeSignature
    (so .key() is guaranteed identical to what check_failure_cascade() will
    generate at inference time) and identity is a hashable tuple identifying
    the underlying real-world event (direction + raw stage indices) — used
    by the caller to avoid resampling the same physical cascade multiple
    times as it ages through the lookback window. Returns (None, None) if
    no stage is present.
    """
    lo = current_idx - lookback
    hi = current_idx

    # direction naming must match extract_cascade_signature()'s wanted_dir
    wanted_dir = "BULLISH" if direction.upper() in ("BUY", "BULLISH", "LONG") else "BEARISH"

    choch_name, bos_name, ob_name, fvg_name = STAGE_ORDER  # ["CHOCH","BOS","OB","FVG"]

    # 1) Find most recent CHoCH and BOS in window — filtered by direction,
    #    same as extract_cascade_signature()'s struct_events filter.
    recent_choch = None
    recent_bos = None
    for e in events:
        idx = e["index"]
        if idx <= lo or idx > hi:
            continue
        if _DIR_MAP.get(e.get("direction")) != wanted_dir:
            continue
        etype = e["type"].upper()
        if etype == choch_name and (recent_choch is None or idx > recent_choch):
            recent_choch = idx
        if etype == bos_name and (recent_bos is None or idx > recent_bos):
            recent_bos = idx

    # 2) Find most recent OB in window (match direction)
    recent_ob = None
    for o in obs:
        idx = o["index"]  # ob_end_index
        if idx <= lo or idx > hi:
            continue
        if o.get("direction", "").upper() == wanted_dir:
            if recent_ob is None or idx > recent_ob:
                recent_ob = idx

    # 3) Find most recent FVG in window (match direction)
    recent_fvg = None
    for g in fvgs:
        idx = g["index"]
        if idx <= lo or idx > hi:
            continue
        if g.get("direction", "").upper() == wanted_dir:
            if recent_fvg is None or idx > recent_fvg:
                recent_fvg = idx

    # Build stage map (index → stage name), using STAGE_ORDER's own names
    stages = {}
    if recent_choch is not None:
        stages[recent_choch] = choch_name
    if recent_bos is not None:
        stages[recent_bos] = bos_name
    if recent_ob is not None:
        stages[recent_ob] = ob_name
    if recent_fvg is not None:
        stages[recent_fvg] = fvg_name

    if not stages:
        return None, None

    # Sort by index (chronological)
    sorted_stages = sorted(stages.items(), key=lambda x: x[0])
    stages_present = tuple(name for _, name in sorted_stages)
    stage_indices = tuple(idx for idx, _ in sorted_stages)

    # Gap buckets between consecutive stages — use production's own
    # bucketing function so thresholds/names can never drift.
    gap_buckets = tuple(
        _gap_bucket(stage_indices[j + 1] - stage_indices[j])
        for j in range(len(stage_indices) - 1)
    )

    # Bars since last stage
    bars_since = current_idx - stage_indices[-1]

    signature = CascadeSignature(
        direction=wanted_dir,
        stages_present=stages_present,
        gap_buckets=gap_buckets,
        bars_since_last_stage=int(bars_since),
    )
    # Identity of the underlying real-world event (not just its bucketed
    # shape) — used by the caller to dedupe repeated re-observations of the
    # SAME cascade as it ages through the lookback window. Two different
    # cascades can share a signature.key() (same shape) but must never
    # share this identity (same physical event) unless they really are the
    # same bars.
    identity = (wanted_dir, stage_indices)
    return signature, identity


# ── Label outcome ─────────────────────────────────────────────────────
def _label_outcome(df: pd.DataFrame, entry_idx: int,
                   direction: str) -> str:
    if entry_idx + HOLDING_PERIOD_BARS >= len(df):
        return "TIMEOUT"
    entry_price = df.iloc[entry_idx]["close"]
    atr = df.iloc[entry_idx].get("atr", np.nan)
    if np.isnan(atr) or atr <= 0:
        atr = df.iloc[entry_idx]["high"] - df.iloc[entry_idx]["low"]
    if direction == "LONG":
        tp = entry_price + TP_ATR_MULT * atr
        sl = entry_price - SL_ATR_MULT * atr
    else:
        tp = entry_price - TP_ATR_MULT * atr
        sl = entry_price + SL_ATR_MULT * atr
    for j in range(entry_idx + 1, min(entry_idx + HOLDING_PERIOD_BARS + 1, len(df))):
        h, l = df.iloc[j]["high"], df.iloc[j]["low"]
        if direction == "LONG":
            if h >= tp: return "WIN"
            if l <= sl: return "LOSS"
        else:
            if l <= tp: return "WIN"
            if h >= sl: return "LOSS"
    return "TIMEOUT"


# ── Main ──────────────────────────────────────────────────────────────
def calibrate(csv_path: str, symbol: str, timeframe: str,
              append: bool = False):
    print(f"[calibrate] Loading {csv_path} …")
    df = _load_and_prepare(csv_path)
    print(f"[calibrate] {len(df)} bars | {df['time'].iloc[0]} → {df['time'].iloc[-1]}")

    df = _add_atr(df)

    # ★ Detect indicators ONCE on full data
    print("[calibrate] Running detectors (one-time) …")
    events, fvgs, obs = _detect_all(df)
    print("[calibrate] Detection complete.")

    # Walk-forward
    pattern_stats = {}
    seen_cascades = set()   # (direction, raw_stage_indices) already sampled — dedup
    total = wins = losses = timeouts = 0
    start = max(LOOKBACK, 100)
    directions = ["LONG", "SHORT"]
    end = len(df) - HOLDING_PERIOD_BARS

    print(f"[calibrate] Walking forward: bar {start} → {end} (step={STEP}) …")

    for i in range(start, end, STEP):
        for direction in directions:
            signature, identity = _extract_cascade(events, fvgs, obs, direction, i, LOOKBACK)
            if signature is None:
                continue
            if identity in seen_cascades:
                # Same underlying event as an earlier step (still inside the
                # lookback window) — already sampled once at its freshest
                # observation. Counting it again would inflate n without
                # adding an independent observation.
                continue
            seen_cascades.add(identity)

            key = signature.key()  # identical format to what check_failure_cascade() looks up
            outcome = _label_outcome(df, i, direction)

            if key not in pattern_stats:
                pattern_stats[key] = {"wins": 0, "losses": 0, "timeouts": 0, "n_samples": 0}
            outcome_key = _OUTCOME_KEY_MAP.get(outcome)
            if outcome_key is None:
                # Defensive: _label_outcome contract changed underneath us —
                # fail loudly instead of silently mis-bucketing or KeyError-ing later.
                raise ValueError(
                    f"_label_outcome returned unexpected value {outcome!r} "
                    f"(expected one of {sorted(_OUTCOME_KEY_MAP)})"
                )
            pattern_stats[key][outcome_key] += 1
            pattern_stats[key]["n_samples"] += 1

            total += 1
            if outcome == "WIN":    wins += 1
            elif outcome == "LOSS":  losses += 1
            else:                    timeouts += 1

        if (i - start) % (STEP * 20) == 0:
            pct = (i - start) / max(1, end - start) * 100
            print(f"  {pct:5.1f}% | patterns={len(pattern_stats)} | "
                  f"samples={total} (W={wins} L={losses} T={timeouts})")

    # Compute fail rates
    # NOTE: field name is "n_samples", not "total" — score_cascade() in
    # analysis/failure_cascade_detector.py reads bucket.get("n_samples", 0)
    # to decide whether a bucket is calibrated. A mismatched field name here
    # means every bucket silently reads as 0 samples and is never trusted,
    # regardless of how much data was actually collected.
    results = {}
    for key, stats in pattern_stats.items():
        effective = stats["wins"] + stats["losses"]
        fail_rate = stats["losses"] / effective if effective > 0 else 0.5
        results[key] = {
            "fail_rate": round(fail_rate, 4),
            "n_samples": stats["n_samples"],
            "wins": stats["wins"],
            "losses": stats["losses"],
            "timeouts": stats["timeouts"],
            "trusted": stats["n_samples"] >= MIN_SAMPLES_FOR_TRUST,
        }

    # Sort trusted-first, then by fail_rate — an untrusted n=1 "100% fail"
    # bucket is noise, not signal, and shouldn't outrank a trusted n=45 bucket
    # just because its raw fail_rate happens to be higher.
    results = dict(sorted(
        results.items(),
        key=lambda x: (x[1]["trusted"], x[1]["fail_rate"]),
        reverse=True,
    ))

    # Merge if --append
    if append and STATS_PATH.exists():
        with open(STATS_PATH) as f:
            existing = json.load(f)
        for key, stats in results.items():
            if key in existing:
                old = existing[key]
                stats["n_samples"] += old.get("n_samples", old.get("total", 0))
                stats["wins"] += old.get("wins", 0)
                stats["losses"] += old.get("losses", 0)
                stats["timeouts"] += old.get("timeouts", 0)
                effective = stats["wins"] + stats["losses"]
                stats["fail_rate"] = round(stats["losses"] / effective, 4) if effective > 0 else 0.5
                stats["trusted"] = stats["n_samples"] >= MIN_SAMPLES_FOR_TRUST
        results = {**existing, **results}
        results = dict(sorted(
            results.items(),
            key=lambda x: (x[1]["trusted"], x[1]["fail_rate"]),
            reverse=True,
        ))

    # Save
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    total_trusted = [k for k, v in results.items() if v["trusted"]]
    print(f"\n{'='*60}")
    print(f"CALIBRATION COMPLETE")
    print(f"{'='*60}")
    print(f"This-run samples : {total}")
    print(f"This-run W/L/T   : {wins} / {losses} / {timeouts}")
    print(f"Total unique patterns (merged file): {len(results)}")
    print(f"Total trusted (≥{MIN_SAMPLES_FOR_TRUST}, merged file): {len(total_trusted)}")

    trusted_sorted = [(k, v) for k, v in results.items() if v["trusted"]]
    print(f"\nTop 10 riskiest TRUSTED patterns (n≥{MIN_SAMPLES_FOR_TRUST}):")
    if not trusted_sorted:
        print("  (none yet — keep running --append across more symbols/history)")
    for i, (key, stats) in enumerate(trusted_sorted[:10]):
        print(f"  {i+1:2d}. {stats['fail_rate']:.1%} fail | "
              f"n={stats['n_samples']:3d} (W={stats['wins']} L={stats['losses']}) | {key}")

    untrusted_count = len(results) - len(total_trusted)
    if untrusted_count:
        print(f"\n({untrusted_count} additional patterns exist with n<{MIN_SAMPLES_FOR_TRUST} "
              f"samples — excluded above as statistically unreliable; still saved to disk)")
    print(f"\nSaved to: {STATS_PATH}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate failure cascade detector")
    parser.add_argument("--csv", required=True, help="Path to OHLC CSV")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--append", action="store_true", help="Merge with existing stats")
    args = parser.parse_args()
    calibrate(args.csv, args.symbol, args.timeframe, args.append)