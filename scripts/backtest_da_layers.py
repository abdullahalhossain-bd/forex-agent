"""
Devil's Advocate Layer Backtest / Ablation Analyzer

Purpose:
    Determine which DA evidence layers actually have predictive/veto value
    in historical trades.

This does NOT retrain the DA.
This does NOT modify strategy parameters.
This does NOT use future outcome inside the decision.

It answers:

    1. Which layer catches losers?
    2. Which layer rejects winners?
    3. Which layer has real discriminatory power?
    4. Which layer is mostly generic/narrative?
    5. Which combination is strongest?
    6. What should become a HARD VETO?
    7. What should remain SOFT evidence?
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

# Change this to your actual backtest result file.
BACKTEST_FILE = Path("memory/backtest_trades.jsonl")

# Alternative CSV support.
BACKTEST_CSV = Path("memory/backtest_trades.csv")

# Minimum observations required before trusting a layer.
MIN_SAMPLES = 30

# A layer must catch at least this fraction of losers
# to be considered potentially useful.
MIN_LOSS_CAPTURE = 0.50

# False reject rate threshold.
MAX_FALSE_REJECT = 0.20


# ============================================================
# NORMALIZATION
# ============================================================

WIN_VALUES = {
    "WIN",
    "WON",
    "TP",
    "PROFIT",
    "PROFITABLE",
    "1",
    "TRUE",
}

LOSS_VALUES = {
    "LOSS",
    "LOST",
    "SL",
    "STOP",
    "LOSS_TRADE",
    "-1",
    "FALSE",
}


def normalize_outcome(x):
    if x is None:
        return None

    s = str(x).strip().upper()

    if s in WIN_VALUES:
        return "WIN"

    if s in LOSS_VALUES:
        return "LOSS"

    try:
        v = float(x)

        if v > 0:
            return "WIN"

        if v < 0:
            return "LOSS"
    except Exception:
        pass

    return None


def load_data():
    if BACKTEST_FILE.exists():
        rows = []

        with BACKTEST_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue

        return pd.DataFrame(rows)

    if BACKTEST_CSV.exists():
        return pd.read_csv(BACKTEST_CSV)

    raise FileNotFoundError(
        f"\nNo backtest file found.\n"
        f"Expected one of:\n"
        f"  {BACKTEST_FILE}\n"
        f"  {BACKTEST_CSV}\n"
    )


# ============================================================
# GENERIC FIELD ACCESS
# ============================================================

def get_nested(row, *paths):
    """
    Try multiple possible paths.

    Example:
        get_nested(row,
            ("evidence", "htf", "h4_trend"),
            ("h4_trend",)
        )
    """

    for path in paths:
        cur = row

        try:
            for key in path:
                if isinstance(cur, dict):
                    cur = cur.get(key)
                else:
                    cur = None

            if cur is not None:
                return cur

        except Exception:
            pass

    return None


def text(x):
    if x is None:
        return ""

    if isinstance(x, (list, tuple, set)):
        return " ".join(map(str, x)).lower()

    if isinstance(x, dict):
        return json.dumps(x, default=str).lower()

    return str(x).lower()


# ============================================================
# LAYER DEFINITIONS
# ============================================================

def layer_htf(row, signal):
    h4 = text(
        get_nested(
            row,
            ("evidence", "htf", "h4_trend"),
            ("htf", "h4_trend"),
            ("h4_trend",),
        )
    )

    h1 = text(
        get_nested(
            row,
            ("evidence", "htf", "h1_trend"),
            ("htf", "h1_trend"),
            ("h1_trend",),
        )
    )

    signal = str(signal).upper()

    bullish = any(x in h4 for x in ["bull", "strong_bull"])
    bearish = any(x in h4 for x in ["bear", "strong_bear"])

    bullish_h1 = any(x in h1 for x in ["bull", "strong_bull"])
    bearish_h1 = any(x in h1 for x in ["bear", "strong_bear"])

    if signal == "BUY":
        if bullish and bullish_h1:
            return "SUPPORT"

        if bearish and bearish_h1:
            return "OPPOSE"

    if signal == "SELL":
        if bearish and bearish_h1:
            return "SUPPORT"

        if bullish and bullish_h1:
            return "OPPOSE"

    if any(x in h4 + h1 for x in ["sideways", "ranging", "neutral"]):
        return "NEUTRAL"

    return "UNKNOWN"


def layer_structure(row, signal):
    bos = text(
        get_nested(
            row,
            ("evidence", "structure", "bos"),
            ("structure", "bos"),
            ("bos",),
        )
    )

    choch = text(
        get_nested(
            row,
            ("evidence", "structure", "choch"),
            ("structure", "choch"),
            ("choch",),
        )
    )

    signal = str(signal).upper()

    bullish = any(x in bos + choch for x in ["bullish"])
    bearish = any(x in bos + choch for x in ["bearish"])

    if signal == "BUY":
        if bullish:
            return "SUPPORT"
        if bearish:
            return "OPPOSE"

    if signal == "SELL":
        if bearish:
            return "SUPPORT"
        if bullish:
            return "OPPOSE"

    return "NEUTRAL"


def layer_displacement(row, signal):
    d = text(
        get_nested(
            row,
            ("evidence", "structure", "displacement"),
            ("structure", "displacement"),
            ("displacement",),
        )
    )

    if not d or d in {"unknown", "none", "null"}:
        return "MISSING"

    signal = str(signal).upper()

    bullish = "bull" in d
    bearish = "bear" in d

    if signal == "BUY":
        return "SUPPORT" if bullish else "OPPOSE" if bearish else "NEUTRAL"

    if signal == "SELL":
        return "SUPPORT" if bearish else "OPPOSE" if bullish else "NEUTRAL"

    return "UNKNOWN"


def layer_liquidity(row, signal):
    x = text(
        get_nested(
            row,
            ("evidence", "structure", "liquidity_sweep"),
            ("structure", "liquidity_sweep"),
            ("liquidity_sweep",),
        )
    )

    if not x or x in {"unknown", "none", "null"}:
        return "MISSING"

    signal = str(signal).upper()

    if signal == "BUY":
        if "bull" in x or "low" in x:
            return "SUPPORT"

        if "bear" in x or "high" in x:
            return "OPPOSE"

    if signal == "SELL":
        if "bear" in x or "high" in x:
            return "SUPPORT"

        if "bull" in x or "low" in x:
            return "OPPOSE"

    return "NEUTRAL"


def layer_location(row, signal):
    location = text(
        get_nested(
            row,
            ("evidence", "location", "support_resistance_zone"),
            ("location", "support_resistance_zone"),
            ("support_resistance_zone",),
            ("sr_zone",),
        )
    )

    if not location or location in {"unknown", "none", "null"}:
        return "MISSING"

    return "SUPPORT"


def layer_regime(row, signal):
    regime = text(
        get_nested(
            row,
            ("evidence", "context", "market_regime"),
            ("evidence", "momentum", "volatility_regime"),
            ("regime",),
            ("market_regime",),
        )
    )

    if not regime:
        return "MISSING"

    if any(x in regime for x in ["chaotic", "extreme", "high_volatility"]):
        return "OPPOSE"

    if any(x in regime for x in ["ranging", "sideways"]):
        return "NEUTRAL"

    return "SUPPORT"


def layer_rr(row, signal):
    rr = get_nested(
        row,
        ("evidence", "execution", "rr_ratio"),
        ("execution", "rr_ratio"),
        ("rr_ratio",),
    )

    try:
        rr = float(rr)
    except Exception:
        return "MISSING"

    if rr < 1.5:
        return "OPPOSE"

    if rr >= 2.0:
        return "SUPPORT"

    return "NEUTRAL"


def layer_counter_evidence(row, signal):
    value = get_nested(
        row,
        ("counter_evidence_strength",),
        ("llm", "counter_evidence_strength"),
        ("result", "counter_evidence_strength"),
    )

    try:
        value = float(value)
    except Exception:
        return "MISSING"

    thesis = get_nested(
        row,
        ("thesis_quality",),
        ("llm", "thesis_quality"),
        ("result", "thesis_quality"),
    )

    try:
        thesis = float(thesis)
    except Exception:
        thesis = 0.5

    if value >= thesis:
        return "OPPOSE"

    if value >= thesis * 0.75:
        return "WARNING"

    return "SUPPORT"


# ============================================================
# LAYER REGISTRY
# ============================================================

LAYERS = {
    "HTF alignment": layer_htf,
    "Structure BOS/CHoCH": layer_structure,
    "Displacement": layer_displacement,
    "Liquidity sweep": layer_liquidity,
    "S/R location": layer_location,
    "Market regime": layer_regime,
    "RR": layer_rr,
    "Counter-evidence": layer_counter_evidence,
}


# ============================================================
# ANALYSIS
# ============================================================

def evaluate_layer(df, layer_name, func):
    rows = []

    for _, row in df.iterrows():
        outcome = normalize_outcome(row.get("_outcome"))

        if outcome is None:
            continue

        signal = row.get("_signal", "")

        try:
            state = func(row.to_dict(), signal)
        except Exception:
            state = "ERROR"

        rows.append({
            "outcome": outcome,
            "state": state,
        })

    if not rows:
        return None

    data = pd.DataFrame(rows)

    total = len(data)
    wins = int((data.outcome == "WIN").sum())
    losses = int((data.outcome == "LOSS").sum())

    useful_states = data[
        data.state.isin(["OPPOSE", "WARNING"])
    ]

    loser_caught = int(
        ((data.outcome == "LOSS") &
         data.state.isin(["OPPOSE", "WARNING"])).sum()
    )

    winner_rejected = int(
        ((data.outcome == "WIN") &
         data.state.isin(["OPPOSE", "WARNING"])).sum()
    )

    loss_capture = (
        loser_caught / losses
        if losses
        else 0.0
    )

    false_reject = (
        winner_rejected / wins
        if wins
        else 0.0
    )

    # What happens when this layer says SUPPORT?
    support = data[data.state == "SUPPORT"]

    support_wr = (
        (support.outcome == "WIN").mean()
        if len(support)
        else float("nan")
    )

    # What happens when this layer says OPPOSE?
    oppose = data[data.state == "OPPOSE"]

    oppose_wr = (
        (oppose.outcome == "WIN").mean()
        if len(oppose)
        else float("nan")
    )

    return {
        "layer": layer_name,
        "samples": total,
        "wins": wins,
        "losses": losses,
        "loss_capture": loss_capture,
        "false_reject": false_reject,
        "support_samples": len(support),
        "support_winrate": support_wr,
        "oppose_samples": len(oppose),
        "oppose_winrate": oppose_wr,
    }


# ============================================================
# COMBINATION ANALYSIS
# ============================================================

def analyze_combinations(df):
    """
    Find combinations of hard contradictions.

    Example:
        HTF OPPOSE + Structure OPPOSE
        HTF OPPOSE + Displacement MISSING
        Structure OPPOSE + Location SUPPORT
    """

    layer_states = defaultdict(dict)

    for idx, row in df.iterrows():
        signal = row["_signal"]

        for name, fn in LAYERS.items():
            try:
                layer_states[idx][name] = fn(row.to_dict(), signal)
            except Exception:
                layer_states[idx][name] = "ERROR"

    combinations = [
        ("HTF alignment", "Structure BOS/CHoCH"),
        ("HTF alignment", "Displacement"),
        ("HTF alignment", "S/R location"),
        ("Structure BOS/CHoCH", "Displacement"),
        ("Structure BOS/CHoCH", "S/R location"),
        ("Structure BOS/CHoCH", "Liquidity sweep"),
        ("Displacement", "Liquidity sweep"),
    ]

    results = []

    for a, b in combinations:

        subset = []

        for idx, row in df.iterrows():
            if normalize_outcome(row["_outcome"]) is None:
                continue

            sa = layer_states[idx].get(a)
            sb = layer_states[idx].get(b)

            if sa == "OPPOSE" and sb == "OPPOSE":
                subset.append(row)

        if not subset:
            continue

        outcomes = [
            normalize_outcome(x["_outcome"])
            for x in subset
        ]

        losses = outcomes.count("LOSS")
        wins = outcomes.count("WIN")

        results.append({
            "combination": f"{a} + {b}",
            "samples": len(outcomes),
            "losses": losses,
            "wins": wins,
            "winrate": wins / len(outcomes),
        })

    return pd.DataFrame(results)


# ============================================================
# CLASSIFICATION
# ============================================================

def classify(row):
    n = row["samples"]

    if n < MIN_SAMPLES:
        return "INSUFFICIENT_DATA"

    if (
        row["loss_capture"] >= MIN_LOSS_CAPTURE
        and row["false_reject"] <= MAX_FALSE_REJECT
    ):
        return "STRONG_VETO_CANDIDATE"

    if row["loss_capture"] >= 0.35 and row["false_reject"] <= 0.30:
        return "USEFUL_SOFT_FILTER"

    if row["false_reject"] > 0.35:
        return "DANGEROUS_FALSE_REJECTOR"

    if row["loss_capture"] < 0.20:
        return "WEAK_FOR_VETO"

    return "WEAK/MIXED"


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nLoading backtest data...")

    df = load_data()

    print(f"Loaded rows: {len(df)}")

    # --------------------------------------------------------
    # Normalize common column names
    # --------------------------------------------------------

    outcome_candidates = [
        "outcome",
        "result",
        "trade_outcome",
        "future_outcome",
        "pnl",
        "pnl_usd",
        "r_multiple",
    ]

    signal_candidates = [
        "signal",
        "decision",
        "action",
        "direction",
    ]

    outcome_col = None

    for c in outcome_candidates:
        if c in df.columns:
            outcome_col = c
            break

    if outcome_col is None:
        raise RuntimeError(
            "\nCould not find trade outcome column.\n"
            "Need one of:\n"
            + "\n".join(f"  - {x}" for x in outcome_candidates)
        )

    signal_col = None

    for c in signal_candidates:
        if c in df.columns:
            signal_col = c
            break

    if signal_col is None:
        raise RuntimeError(
            "\nCould not find signal column.\n"
            "Need one of:\n"
            + "\n".join(f"  - {x}" for x in signal_candidates)
        )

    df["_outcome"] = df[outcome_col]
    df["_signal"] = df[signal_col].astype(str).str.upper()

    df = df[
        df["_signal"].isin(["BUY", "SELL"])
    ].copy()

    df["_outcome_norm"] = df["_outcome"].apply(normalize_outcome)

    df = df[
        df["_outcome_norm"].notna()
    ].copy()

    print(f"Valid trades: {len(df)}")
    print(
        f"Wins : {(df['_outcome_norm'] == 'WIN').sum()}"
    )
    print(
        f"Losses: {(df['_outcome_norm'] == 'LOSS').sum()}"
    )

    # IMPORTANT:
    # evaluate_layer expects _outcome to be normalized.
    df["_outcome"] = df["_outcome_norm"]

    # --------------------------------------------------------
    # Layer analysis
    # --------------------------------------------------------

    results = []

    for name, fn in LAYERS.items():

        result = evaluate_layer(
            df,
            name,
            fn,
        )

        if result:
            results.append(result)

    report = pd.DataFrame(results)

    if report.empty:
        print("\nNo analyzable layers.")
        return

    report["classification"] = report.apply(
        classify,
        axis=1,
    )

    report = report.sort_values(
        ["loss_capture", "false_reject"],
        ascending=[False, True],
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print("\n")
    print("=" * 110)
    print("DEVIL'S ADVOCATE — LAYER QUALITY BACKTEST")
    print("=" * 110)

    print(
        report[
            [
                "layer",
                "samples",
                "loss_capture",
                "false_reject",
                "support_winrate",
                "oppose_winrate",
                "classification",
            ]
        ].to_string(
            index=False,
            formatters={
                "loss_capture": "{:.1%}".format,
                "false_reject": "{:.1%}".format,
                "support_winrate": lambda x:
                    "N/A" if pd.isna(x) else f"{x:.1%}",
                "oppose_winrate": lambda x:
                    "N/A" if pd.isna(x) else f"{x:.1%}",
            },
        )
    )

    # --------------------------------------------------------
    # HARD VETO CANDIDATES
    # --------------------------------------------------------

    strong = report[
        report.classification == "STRONG_VETO_CANDIDATE"
    ]

    print("\n")
    print("=" * 110)
    print("HARD VETO CANDIDATES")
    print("=" * 110)

    if strong.empty:
        print("No layer currently qualifies as a strong hard-veto candidate.")
    else:
        for _, r in strong.iterrows():
            print(
                f"\n✓ {r['layer']}"
                f"\n  Losses caught : {r['loss_capture']:.1%}"
                f"\n  Winners rejected: {r['false_reject']:.1%}"
                f"\n  Samples       : {int(r['samples'])}"
            )

    # --------------------------------------------------------
    # DANGEROUS LAYERS
    # --------------------------------------------------------

    dangerous = report[
        report.classification == "DANGEROUS_FALSE_REJECTOR"
    ]

    print("\n")
    print("=" * 110)
    print("DANGEROUS / OVER-REJECTING LAYERS")
    print("=" * 110)

    if dangerous.empty:
        print("No obviously dangerous layer found.")
    else:
        for _, r in dangerous.iterrows():
            print(
                f"\n⚠ {r['layer']}"
                f"\n  False reject : {r['false_reject']:.1%}"
                f"\n  Loss capture : {r['loss_capture']:.1%}"
            )

    # --------------------------------------------------------
    # WEAK LAYERS
    # --------------------------------------------------------

    weak = report[
        report.classification == "WEAK_FOR_VETO"
    ]

    print("\n")
    print("=" * 110)
    print("WEAK AS VETO")
    print("=" * 110)

    if weak.empty:
        print("No clearly weak layer.")
    else:
        for _, r in weak.iterrows():
            print(
                f"  - {r['layer']}: "
                f"loss_capture={r['loss_capture']:.1%}, "
                f"false_reject={r['false_reject']:.1%}"
            )

    # --------------------------------------------------------
    # COMBINATIONS
    # --------------------------------------------------------

    combos = analyze_combinations(df)

    print("\n")
    print("=" * 110)
    print("DOUBLE-CONTRADICTION ANALYSIS")
    print("=" * 110)

    if combos.empty:
        print("No contradiction combinations found.")
    else:
        print(
            combos.sort_values(
                "winrate"
            ).to_string(
                index=False,
                formatters={
                    "winrate": "{:.1%}".format,
                },
            )
        )

    # --------------------------------------------------------
    # FINAL RECOMMENDATION
    # --------------------------------------------------------

    print("\n")
    print("=" * 110)
    print("RECOMMENDATION")
    print("=" * 110)

    for _, r in report.iterrows():

        if r["classification"] == "STRONG_VETO_CANDIDATE":
            print(
                f"  HARD VETO candidate : {r['layer']}"
            )

        elif r["classification"] == "USEFUL_SOFT_FILTER":
            print(
                f"  SOFT FILTER          : {r['layer']}"
            )

        elif r["classification"] == "WEAK_FOR_VETO":
            print(
                f"  DO NOT VETO ON THIS  : {r['layer']}"
            )

        elif r["classification"] == "DANGEROUS_FALSE_REJECTOR":
            print(
                f"  REVIEW / DISABLE     : {r['layer']}"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()