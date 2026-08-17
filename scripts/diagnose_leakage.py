#!/usr/bin/env python3
"""
diagnose_leakage.py — find why a pair's real (non-bootstrap) data still
trains to unrealistically high accuracy.

Checks, in order:
  1. Exact duplicate rows (same feature vector, ignoring meta columns) —
     and specifically, duplicates that appear in BOTH the train-range and
     test-range of a naive chronological split. If a row (or a near-copy)
     shows up in both train and test, a model can "memorize" it and test
     accuracy stops meaning anything.
  2. Per-feature correlation with the label. A single feature with |corr|
     close to 1.0 is a near-certain leak (the feature IS the label, or was
     computed from data the label also depends on).
  3. Zero-variance columns (uninformative but harmless on their own).

Usage:
    python scripts/diagnose_leakage.py --pair EURUSD --timeframe 15m
"""
import argparse
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Diagnose why real data trains to suspiciously high accuracy")
    parser.add_argument("--pair", type=str, required=True)
    parser.add_argument("--timeframe", type=str, default="15m")
    parser.add_argument("--train-pct", type=float, default=0.70)
    parser.add_argument("--val-pct", type=float, default=0.15)
    args = parser.parse_args()

    import pandas as pd
    import numpy as np
    from ml.feature_store import get_feature_store

    store = get_feature_store()
    df = store.load_training_data(
        pair=args.pair, timeframe=args.timeframe, min_samples=0, include_bootstrap=False,
    )
    if df.empty:
        print("No real (non-bootstrap) data found for this pair/timeframe.")
        sys.exit(1)

    print("=" * 60)
    print(f"  Leakage Diagnostic — {args.pair} {args.timeframe}")
    print("=" * 60)
    print(f"  Total rows loaded (source != 'bootstrap'): {len(df)}")

    if "label" not in df.columns or df["label"].isna().all():
        print("FAIL: no usable 'label' column in this data.")
        sys.exit(1)

    df = df[df["label"].notna()].copy()
    print(f"  Rows with a label: {len(df)}")

    meta_cols = [c for c in df.columns if c.startswith("_") or c in
                 ("outcome", "pnl_usd", "forward_pips", "label_ternary",
                  "label_forward_return", "label_forward_pips",
                  "label_mae_pips", "label_mfe_pips", "label_r_multiple",
                  "sample_weight")]
    feature_df = df.drop(columns=meta_cols, errors="ignore")
    label = feature_df["label"].astype(int)
    features_only = feature_df.drop(columns=["label"], errors="ignore")

    # ── 1. Duplicate rows ────────────────────────────────────────────
    print()
    print("-- Duplicate rows (exact feature-vector matches) --")
    dup_mask = features_only.duplicated(keep=False)
    dup_count = int(dup_mask.sum())
    print(f"  Rows that are exact duplicates of another row: {dup_count} / {len(features_only)} "
          f"({dup_count / len(features_only):.1%})")

    n = len(feature_df)
    train_end = int(n * args.train_pct)
    val_end = int(n * (args.train_pct + args.val_pct))
    train_idx = set(range(0, train_end))
    test_idx = set(range(val_end, n))

    train_fp = features_only.iloc[sorted(train_idx)].apply(lambda r: tuple(r), axis=1)
    test_fp = features_only.iloc[sorted(test_idx)].apply(lambda r: tuple(r), axis=1)
    train_set = set(train_fp)
    overlap = sum(1 for fp in test_fp if fp in train_set)
    print(f"  Test rows whose EXACT feature vector also appears in train: "
          f"{overlap} / {len(test_fp)} ({overlap / max(1, len(test_fp)):.1%})")
    if overlap / max(1, len(test_fp)) > 0.05:
        print("  ⚠ HIGH — this alone can explain near-100% test accuracy: the model")
        print("    can simply memorize these exact rows from train and 'predict'")
        print("    the identical label seen at train time.")

    print()
    print("-- Same-bar combination check (label vs sign(close - open) of the SAME row) --")
    print("   (This catches a leak that per-column correlation misses: if the label was")
    print("   computed from the current bar's own OHLC instead of a FUTURE bar, individual")
    print("   price columns look weakly correlated, but their sign/difference perfectly")
    print("   determines the label -- trivial for tree models, invisible to Pearson corr.)")
    price_cols_present = all(c in feature_df.columns for c in ("price_open", "price_close"))
    if price_cols_present:
        same_bar_pred = (feature_df["price_close"] > feature_df["price_open"]).astype(int)
        match_rate = float((same_bar_pred == label).mean())
        print(f"  sign(close-open) matches label: {match_rate:.1%}")
        if match_rate > 0.95 or match_rate < 0.05:
            print("  ⚠ STRONG MATCH — the label is very likely derived from this same bar's")
            print("    own close/open, not from a FUTURE bar. This is a classic same-bar")
            print("    (non-forward-looking) labeling bug: whatever code calls")
            print("    store.save_features(..., label=...) for real/live data should be")
            print("    computing the label from price N candles AFTER the feature row, not")
            print("    from the feature row's own OHLC.")
    else:
        print("  (price_open/price_close not present in this feature set — skipped)")

    # ── 2. Per-feature correlation with label ───────────────────────
    print()
    print("-- Per-feature correlation with label (top 15 by |corr|) --")
    numeric_features = features_only.select_dtypes(include=[np.number])
    corrs = {}
    for col in numeric_features.columns:
        s = numeric_features[col]
        if s.std(ddof=0) == 0:
            continue
        try:
            corrs[col] = float(np.corrcoef(s.values, label.values)[0, 1])
        except Exception:
            continue
    ranked = sorted(corrs.items(), key=lambda kv: abs(kv[1]), reverse=True)
    for col, c in ranked[:15]:
        flag = "  <-- SUSPECT (near-perfect)" if abs(c) > 0.9 else ("  <-- high" if abs(c) > 0.5 else "")
        print(f"  {col:30s} corr={c:+.3f}{flag}")

    # ── 3. Zero-variance columns ─────────────────────────────────────
    print()
    print("-- Zero-variance columns (uninformative, not leakage by themselves) --")
    zero_var = [c for c in features_only.columns if features_only[c].std(ddof=0) == 0]
    print(f"  {zero_var if zero_var else '(none)'}")

    print()
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    top_corr = ranked[0] if ranked else None
    if top_corr and abs(top_corr[1]) > 0.9:
        print(f"  Likely cause: feature '{top_corr[0]}' is almost perfectly correlated "
              f"with the label (corr={top_corr[1]:+.3f}). Check where this feature is "
              f"computed and whether it's derived from the same data the label uses.")
    elif overlap / max(1, len(test_fp)) > 0.05:
        print("  Likely cause: train/test row duplication (see above). The pipeline that "
              "writes rows into ml_features.db is probably re-saving the same "
              "feature+label combination multiple times (e.g. every time the bot "
              "reprocesses the same historical candle), so identical rows end up on "
              "both sides of the chronological split.")
    else:
        print("  No single dominant cause found by this script. Next step: share the code "
              "that calls store.save_features(..., label=...) for REAL data (likely in "
              "core/ or strategies/, not ml/) so the feature-computation and labeling "
              "logic around that call can be checked directly for look-ahead bias.")


if __name__ == "__main__":
    main()