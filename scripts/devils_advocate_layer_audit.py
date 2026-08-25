from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ============================================================
# Configuration
# ============================================================

DEFAULT_AUDIT_FILE = Path("memory/devils_advocate_audit.jsonl")

# Minimum observations before making a strong recommendation.
MIN_N = 20

# A layer is considered a strong loss association when:
# - enough samples exist
# - win rate is materially below baseline
# - average R is negative
LOSS_WR_THRESHOLD = 0.35
GOOD_WR_THRESHOLD = 0.50

# Minimum difference from overall baseline to call something meaningful.
MIN_WR_DELTA = 0.08
MIN_R_DELTA = 0.15


# ============================================================
# Utilities
# ============================================================

def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    if not path.exists():
        raise FileNotFoundError(f"Audit file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARN] Invalid JSON at line {line_no}: {exc}")
                continue

            if isinstance(obj, dict):
                rows.append(obj)

    return rows


def outcome_from_row(row: Dict[str, Any]) -> Optional[str]:
    """
    Try several common outcome representations.

    Supported:
      future_outcome = WIN / LOSS
      outcome = WIN / LOSS
      result = WIN / LOSS
      status = WIN / LOSS

    Also accepts:
      r_multiple > 0 => WIN
      r_multiple < 0 => LOSS
    """

    for key in ("future_outcome", "outcome", "result", "status"):
        value = normalize_text(row.get(key))

        if value in {"win", "won", "profit", "profitable"}:
            return "WIN"

        if value in {"loss", "lost", "losing", "losses"}:
            return "LOSS"

    r = safe_float(row.get("r_multiple"))

    if r is not None:
        if r > 0:
            return "WIN"
        if r < 0:
            return "LOSS"

    return None


def r_value(row: Dict[str, Any]) -> Optional[float]:
    return safe_float(row.get("r_multiple"))


def decision(row: Dict[str, Any]) -> str:
    value = normalize_text(
        row.get("llm_decision") or row.get("decision")
    ).upper()

    if value in {"TAKE", "REJECT", "UNCERTAIN"}:
        return value

    return "UNKNOWN"


def get_nested(row: Dict[str, Any], *keys: str) -> Any:
    cur: Any = row

    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)

    return cur


# ============================================================
# Layer extraction
# ============================================================

def evidence_text(row: Dict[str, Any]) -> str:
    parts: List[str] = []

    for key in (
        "supporting_evidence",
        "contradicting_evidence",
        "reasons_for_rejection",
        "critical_failure",
        "thesis_claims",
    ):
        value = row.get(key)

        if isinstance(value, list):
            parts.extend(str(x) for x in value)

        elif value is not None:
            parts.append(str(value))

    return " ".join(parts).lower()


def add_layer(
    layers: Dict[str, List[str]],
    row_id: str,
    name: str,
) -> None:
    layers[name].append(row_id)


def extract_layers(row: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Extract observable conditions from the audit row.

    IMPORTANT:
    These are diagnostic labels, not trading rules.
    """

    row_id = str(row.get("trade_id") or row.get("timestamp") or id(row))

    layers: Dict[str, List[str]] = defaultdict(list)

    text = evidence_text(row)

    # --------------------------------------------------------
    # DA decision
    # --------------------------------------------------------

    da = decision(row)

    if da == "TAKE":
        add_layer(layers, row_id, "DA_TAKE")

    elif da == "REJECT":
        add_layer(layers, row_id, "DA_REJECT")

    elif da == "UNCERTAIN":
        add_layer(layers, row_id, "DA_UNCERTAIN")

    # --------------------------------------------------------
    # Direct evidence fields
    # --------------------------------------------------------

    evidence = row.get("evidence")

    if isinstance(evidence, dict):

        structure = evidence.get("structure") or {}
        htf = evidence.get("htf") or {}
        context = evidence.get("context") or {}
        momentum = evidence.get("momentum") or {}
        execution = evidence.get("execution") or {}
        location = evidence.get("location") or {}

        # BOS
        bos = normalize_text(structure.get("bos"))

        if "bullish" in bos or "bearish" in bos:
            add_layer(layers, row_id, "BOS_PRESENT")

            signal = normalize_text(row.get("signal"))

            if (
                (signal == "buy" and "bullish" in bos)
                or
                (signal == "sell" and "bearish" in bos)
            ):
                add_layer(layers, row_id, "BOS_ALIGNED")

            else:
                add_layer(layers, row_id, "BOS_OPPOSING")

        # CHoCH
        choch = normalize_text(structure.get("choch"))

        if "bullish" in choch or "bearish" in choch:
            add_layer(layers, row_id, "CHOCH_PRESENT")

            signal = normalize_text(row.get("signal"))

            if (
                (signal == "buy" and "bullish" in choch)
                or
                (signal == "sell" and "bearish" in choch)
            ):
                add_layer(layers, row_id, "CHOCH_ALIGNED")

            else:
                add_layer(layers, row_id, "CHOCH_OPPOSING")

        # Displacement
        displacement = normalize_text(structure.get("displacement"))

        if displacement in {"none", "unknown", ""}:
            add_layer(layers, row_id, "NO_DISPLACEMENT")

        elif displacement not in {"unknown", ""}:
            add_layer(layers, row_id, "DISPLACEMENT_PRESENT")

        # H4
        h4 = normalize_text(htf.get("h4_trend"))

        if h4 in {"sideways", "neutral", "ranging"}:
            add_layer(layers, row_id, "H4_SIDEWAYS")

        elif h4:
            signal = normalize_text(row.get("signal"))

            if (
                (signal == "buy" and "bear" in h4)
                or
                (signal == "sell" and "bull" in h4)
            ):
                add_layer(layers, row_id, "HTF_OPPOSING")

        # H1
        h1 = normalize_text(htf.get("h1_trend"))

        if h1:
            signal = normalize_text(row.get("signal"))

            if (
                (signal == "buy" and "bull" in h1)
                or
                (signal == "sell" and "bear" in h1)
            ):
                add_layer(layers, row_id, "H1_ALIGNED")

            elif (
                (signal == "buy" and "bear" in h1)
                or
                (signal == "sell" and "bull" in h1)
            ):
                add_layer(layers, row_id, "H1_OPPOSING")

        # Regime
        regime = normalize_text(
            context.get("market_regime") or
            momentum.get("volatility_regime")
        )

        if regime in {"ranging", "range", "sideways"}:
            add_layer(layers, row_id, "RANGING_REGIME")

        elif regime in {"trending", "trend"}:
            add_layer(layers, row_id, "TRENDING_REGIME")

        # RR
        rr = safe_float(execution.get("rr_ratio"))

        if rr is not None:
            if rr >= 2.0:
                add_layer(layers, row_id, "RR_2_PLUS")

            elif rr < 1.5:
                add_layer(layers, row_id, "LOW_RR")

        # Spread
        spread = safe_float(execution.get("spread_pips"))

        if spread is not None and spread > 0:
            add_layer(layers, row_id, "SPREAD_PRESENT")

        # Location
        zone = normalize_text(
            location.get("support_resistance_zone")
            or location.get("supply_demand_zone")
        )

        if zone and zone != "unknown":
            add_layer(layers, row_id, "MAPPED_LOCATION")

    # --------------------------------------------------------
    # Text-based entry-quality extraction
    # --------------------------------------------------------

    # These appear in your permission.checked logs / DA evidence
    # if they were passed through.
    entry_quality_terms = {
        "tp_structure_validation": "TP_STRUCTURE_VALIDATION_FAILED",
        "sl_swing_anchor": "SL_SWING_ANCHOR_FAILED",
        "indecision_candles": "INDECISION_CANDLES",
        "fresh_high_rejection": "FRESH_HIGH_REJECTION",
        "fresh_low_rejection": "FRESH_LOW_REJECTION",
        "exhaustion_filter": "EXHAUSTION_FILTER",
        "rejection_psychology": "REJECTION_PSYCHOLOGY",
    }

    for needle, layer_name in entry_quality_terms.items():
        if needle in text:
            add_layer(layers, row_id, layer_name)

    # --------------------------------------------------------
    # Combinations
    # --------------------------------------------------------

    present = set(layers.keys())

    if {
        "H4_SIDEWAYS",
        "NO_DISPLACEMENT",
    }.issubset(present):
        add_layer(layers, row_id, "H4_SIDEWAYS_NO_DISPLACEMENT")

    if {
        "RANGING_REGIME",
        "NO_DISPLACEMENT",
    }.issubset(present):
        add_layer(layers, row_id, "RANGING_NO_DISPLACEMENT")

    if {
        "H4_SIDEWAYS",
        "RANGING_REGIME",
        "NO_DISPLACEMENT",
    }.issubset(present):
        add_layer(layers, row_id, "SIDEWAYS_RANGE_NO_DISPLACEMENT")

    if {
        "TP_STRUCTURE_VALIDATION_FAILED",
        "NO_DISPLACEMENT",
    }.issubset(present):
        add_layer(layers, row_id, "BAD_TP_PLUS_NO_DISPLACEMENT")

    return layers


# ============================================================
# Statistics
# ============================================================

def calc_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    usable = []

    for row in rows:
        outcome = outcome_from_row(row)

        if outcome is None:
            continue

        usable.append((row, outcome))

    n = len(usable)

    if n == 0:
        return {
            "n": 0,
            "wins": 0,
            "losses": 0,
            "wr": None,
            "avg_r": None,
            "sum_r": None,
        }

    wins = sum(1 for _, outcome in usable if outcome == "WIN")
    losses = sum(1 for _, outcome in usable if outcome == "LOSS")

    rs = [
        r_value(row)
        for row, _ in usable
        if r_value(row) is not None
    ]

    avg_r = sum(rs) / len(rs) if rs else None
    sum_r = sum(rs) if rs else None

    return {
        "n": n,
        "wins": wins,
        "losses": losses,
        "wr": wins / n if n else None,
        "avg_r": avg_r,
        "sum_r": sum_r,
    }


# ============================================================
# Audit
# ============================================================

def build_layer_membership(
    rows: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:

    membership: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        layers = extract_layers(row)

        for layer_name in layers:
            membership[layer_name].append(row)

    return membership


def classify_layer(
    stats: Dict[str, Any],
    baseline: Dict[str, Any],
) -> str:

    n = stats["n"]

    if n < MIN_N:
        return "INSUFFICIENT_DATA"

    wr = stats["wr"]
    avg_r = stats["avg_r"]

    baseline_wr = baseline["wr"]
    baseline_r = baseline["avg_r"]

    wr_delta = (
        wr - baseline_wr
        if wr is not None and baseline_wr is not None
        else 0.0
    )

    r_delta = (
        avg_r - baseline_r
        if avg_r is not None and baseline_r is not None
        else 0.0
    )

    # Strong negative layer
    if (
        wr is not None
        and wr <= LOSS_WR_THRESHOLD
        and wr_delta <= -MIN_WR_DELTA
        and avg_r is not None
        and r_delta <= -MIN_R_DELTA
    ):
        return "STRONG_WARNING"

    # Good supporting layer
    if (
        wr is not None
        and wr >= GOOD_WR_THRESHOLD
        and wr_delta >= MIN_WR_DELTA
        and avg_r is not None
        and r_delta >= MIN_R_DELTA
    ):
        return "GOOD"

    # Mildly negative
    if (
        wr_delta <= -MIN_WR_DELTA
        or r_delta <= -MIN_R_DELTA
    ):
        return "WEAK / NEGATIVE"

    # Mildly positive
    if (
        wr_delta >= MIN_WR_DELTA
        or r_delta >= MIN_R_DELTA
    ):
        return "WEAK / POSITIVE"

    return "NO_CLEAR_EDGE"


def print_layer_report(
    membership: Dict[str, List[Dict[str, Any]]],
    baseline: Dict[str, Any],
) -> None:

    results = []

    for layer, rows in membership.items():

        stats = calc_stats(rows)

        if stats["n"] == 0:
            continue

        verdict = classify_layer(stats, baseline)

        results.append(
            (
                verdict,
                layer,
                stats,
            )
        )

    order = {
        "STRONG_WARNING": 0,
        "WEAK / NEGATIVE": 1,
        "NO_CLEAR_EDGE": 2,
        "WEAK / POSITIVE": 3,
        "GOOD": 4,
        "INSUFFICIENT_DATA": 5,
    }

    results.sort(key=lambda x: (order.get(x[0], 99), -x[2]["n"]))

    print()
    print("=" * 95)
    print("DEVIL'S ADVOCATE LAYER AUDIT")
    print("=" * 95)

    print(
        f"{'VERDICT':<20}"
        f"{'LAYER':<38}"
        f"{'N':>6}"
        f"{'WR':>9}"
        f"{'AVG R':>10}"
        f"{'SUM R':>10}"
    )

    print("-" * 95)

    for verdict, layer, stats in results:

        wr = (
            f"{stats['wr'] * 100:.1f}%"
            if stats["wr"] is not None
            else "N/A"
        )

        avg_r = (
            f"{stats['avg_r']:+.3f}"
            if stats["avg_r"] is not None
            else "N/A"
        )

        sum_r = (
            f"{stats['sum_r']:+.2f}"
            if stats["sum_r"] is not None
            else "N/A"
        )

        print(
            f"{verdict:<20}"
            f"{layer:<38}"
            f"{stats['n']:>6}"
            f"{wr:>9}"
            f"{avg_r:>10}"
            f"{sum_r:>10}"
        )

    print("=" * 95)


# ============================================================
# Decision matrix
# ============================================================

def print_decision_matrix(rows: List[Dict[str, Any]]) -> None:

    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        da = decision(row)
        outcome = outcome_from_row(row)

        if outcome is None:
            continue

        buckets[(da, outcome)].append(row)

    print()
    print("=" * 70)
    print("DA DECISION × REALIZED OUTCOME")
    print("=" * 70)

    for da in ("TAKE", "REJECT", "UNCERTAIN"):

        win_rows = buckets.get((da, "WIN"), [])
        loss_rows = buckets.get((da, "LOSS"), [])

        total = len(win_rows) + len(loss_rows)

        if total == 0:
            continue

        wr = len(win_rows) / total

        print(
            f"{da:<12}"
            f"N={total:<6}"
            f"W={len(win_rows):<6}"
            f"L={len(loss_rows):<6}"
            f"WR={wr * 100:5.1f}%"
        )

    print()
    print("Interpretation:")
    print("  TAKE → LOSS  = DA missed a losing trade")
    print("  REJECT → WIN  = DA incorrectly removed a winner")
    print("  REJECT → LOSS = useful rejection")
    print("  TAKE → WIN    = correct approval")
    print("=" * 70)


# ============================================================
# Veto candidates
# ============================================================

def print_veto_candidates(
    membership: Dict[str, List[Dict[str, Any]]],
    baseline: Dict[str, Any],
) -> None:

    candidates = []

    for layer, rows in membership.items():

        stats = calc_stats(rows)

        if stats["n"] < MIN_N:
            continue

        if stats["wr"] is None or stats["avg_r"] is None:
            continue

        baseline_wr = baseline["wr"]
        baseline_r = baseline["avg_r"]

        if baseline_wr is None or baseline_r is None:
            continue

        wr_delta = stats["wr"] - baseline_wr
        r_delta = stats["avg_r"] - baseline_r

        if (
            stats["wr"] <= LOSS_WR_THRESHOLD
            and wr_delta <= -MIN_WR_DELTA
            and r_delta <= -MIN_R_DELTA
        ):
            candidates.append(
                (
                    layer,
                    stats,
                    wr_delta,
                    r_delta,
                )
            )

    candidates.sort(key=lambda x: x[3])

    print()
    print("=" * 85)
    print("POTENTIAL VETO CANDIDATES")
    print("=" * 85)

    if not candidates:
        print("No statistically strong veto candidate found.")
        print("Do NOT add a hard veto from current data.")
        return

    for layer, stats, wr_delta, r_delta in candidates:

        print()
        print(f"[CANDIDATE] {layer}")
        print(f"  Samples       : {stats['n']}")
        print(f"  Win rate      : {stats['wr'] * 100:.1f}%")
        print(f"  Baseline WR   : {baseline['wr'] * 100:.1f}%")
        print(f"  WR delta      : {wr_delta * 100:+.1f} pp")
        print(f"  Average R     : {stats['avg_r']:+.3f}")
        print(f"  Baseline AvgR : {baseline['avg_r']:+.3f}")
        print(f"  AvgR delta    : {r_delta:+.3f}")
        print("  Recommendation: INVESTIGATE FOR CONDITIONAL/HARD VETO")

    print()
    print("=" * 85)


# ============================================================
# Combination audit
# ============================================================

def combination_report(rows: List[Dict[str, Any]]) -> None:

    combinations = [
        (
            "H4_SIDEWAYS + NO_DISPLACEMENT",
            {"H4_SIDEWAYS", "NO_DISPLACEMENT"},
        ),
        (
            "RANGING + NO_DISPLACEMENT",
            {"RANGING_REGIME", "NO_DISPLACEMENT"},
        ),
        (
            "SIDEWAYS + RANGE + NO_DISPLACEMENT",
            {
                "H4_SIDEWAYS",
                "RANGING_REGIME",
                "NO_DISPLACEMENT",
            },
        ),
        (
            "TP_STRUCTURE_FAILED + NO_DISPLACEMENT",
            {
                "TP_STRUCTURE_VALIDATION_FAILED",
                "NO_DISPLACEMENT",
            },
        ),
    ]

    print()
    print("=" * 85)
    print("DANGEROUS COMBINATION AUDIT")
    print("=" * 85)

    for name, required in combinations:

        selected = []

        for row in rows:

            layers = extract_layers(row)
            present = set(layers.keys())

            if required.issubset(present):
                selected.append(row)

        stats = calc_stats(selected)

        if stats["n"] == 0:
            print(f"{name:<55} N=0")
            continue

        wr = (
            f"{stats['wr'] * 100:.1f}%"
            if stats["wr"] is not None
            else "N/A"
        )

        avg_r = (
            f"{stats['avg_r']:+.3f}"
            if stats["avg_r"] is not None
            else "N/A"
        )

        print(
            f"{name:<55}"
            f"N={stats['n']:<5}"
            f"WR={wr:<8}"
            f"AvgR={avg_r}"
        )

    print("=" * 85)


# ============================================================
# Data quality
# ============================================================

def print_data_quality(rows: List[Dict[str, Any]]) -> None:

    total = len(rows)
    resolved = sum(
        1 for row in rows
        if outcome_from_row(row) is not None
    )

    unresolved = total - resolved

    print()
    print("=" * 70)
    print("DATA QUALITY")
    print("=" * 70)
    print(f"Total DA audit rows : {total}")
    print(f"Rows with outcome   : {resolved}")
    print(f"Rows without result : {unresolved}")

    if total:
        print(
            f"Outcome coverage    : "
            f"{resolved / total * 100:.1f}%"
        )

    if resolved == 0:
        print()
        print("WARNING:")
        print("No realized outcomes found.")
        print("Layer quality cannot be judged yet.")
        print("Populate future_outcome/r_multiple first.")

    print("=" * 70)


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description="Audit Devil's Advocate evidence layers against realized outcomes."
    )

    parser.add_argument(
        "--file",
        default=str(DEFAULT_AUDIT_FILE),
        help="Path to devils_advocate_audit.jsonl",
    )

    args = parser.parse_args()

    path = Path(args.file)

    print(f"Loading: {path}")

    rows = load_jsonl(path)

    if not rows:
        print("No audit rows found.")
        return

    print_data_quality(rows)

    usable = [
        row
        for row in rows
        if outcome_from_row(row) is not None
    ]

    if not usable:
        return

    baseline = calc_stats(usable)

    print()
    print("=" * 70)
    print("OVERALL BASELINE")
    print("=" * 70)
    print(f"Trades       : {baseline['n']}")
    print(f"Wins         : {baseline['wins']}")
    print(f"Losses       : {baseline['losses']}")
    print(f"Win rate     : {baseline['wr'] * 100:.1f}%")
    print(f"Average R    : {baseline['avg_r']:+.3f}")
    print(f"Total R      : {baseline['sum_r']:+.2f}")
    print("=" * 70)

    membership = build_layer_membership(usable)

    print_layer_report(
        membership,
        baseline,
    )

    print_decision_matrix(usable)

    print_veto_candidates(
        membership,
        baseline,
    )

    combination_report(usable)

    print()
    print("=" * 85)
    print("IMPORTANT")
    print("=" * 85)
    print(
        "A STRONG_WARNING is NOT automatically a veto. "
        "Validate it on a separate/OOS sample before changing live rules."
    )
    print(
        "A GOOD layer means it correlates with better realized outcomes; "
        "it does not prove causality."
    )
    print("=" * 85)


if __name__ == "__main__":
    main()