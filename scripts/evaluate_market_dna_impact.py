#!/usr/bin/env python3
# scripts/evaluate_market_dna_impact.py
# ============================================================
# Market DNA — impact evaluation.
#
# Answers ONE question: "does filtering/sizing trades through the
# Market DNA cluster context measurably help, on data the detector
# never saw during fitting?" It does NOT answer "is this a good
# trading strategy" — the entry logic used here is a fixed, simple
# reference signal (EMA-cross + ATR stop/target), used ONLY as a
# consistent case-control set to test the FILTER, not the strategy.
#
# Methodology (three-way split, same as analysis/dna_walkforward.py):
#   Fold A -> fit the HDBSCAN detector (frozen).
#   Fold B -> label with the frozen detector, run the reference
#             strategy, build the cluster journal (Wilson CI, tiers)
#             from Fold B's trades only.
#   Fold C -> label with the frozen detector, run the reference
#             strategy again, then compare:
#               BASELINE   = every Fold C trade, unfiltered
#               DNA-FILTER = Fold C trades restricted/sized by the
#                            Fold-B-derived journal's recommendation
#             Fold C trades were used in NEITHER the detector fit
#             nor the journal build — this is the only fold whose
#             numbers this script is allowed to call "out-of-sample".
#
# Usage:
#   python -m scripts.evaluate_market_dna_impact
#   python -m scripts.evaluate_market_dna_impact --symbol EURUSD --timeframe H1
# ============================================================

import argparse
import sqlite3
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_DIR
from database.db import DB_PATH
from features.indicators_v5 import add_indicators
from analysis.market_dna import DNAConfig
from analysis.dna_walkforward import make_three_way_split, fit_frozen_detector, label_fold
from analysis.dna_journal import build_cluster_journal, lookup, decision_context, trade_count_tier
from analysis.dna_drift import population_stability_index
from utils.logger import get_logger

log = get_logger(__name__)

MIN_TRADES_FOR_VERDICT = 100
N_BOOTSTRAP = 2000


# ── Data loading ─────────────────────────────────────────────

# Common header spellings seen across brokers/exporters (MT4/MT5,
# Dukascopy, HistData, TradingView, etc.) — mapped onto our canonical
# OHLCV(+time) schema. Matching is case-insensitive and ignores
# surrounding whitespace/underscores.
_COLUMN_ALIASES = {
    "time": {"time", "date", "datetime", "timestamp", "localtime", "local time",
              "gmttime", "gmt time", "opentime", "open time", "date time"},
    "open": {"open", "o"},
    "high": {"high", "h"},
    "low": {"low", "l"},
    "close": {"close", "c", "adjclose", "adj close"},
    "volume": {"volume", "vol", "tickvol", "tick volume", "v"},
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename whatever columns are present to canonical time/OHLCV
    names, matched case/whitespace-insensitively. Leaves unrecognized
    columns untouched."""
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower().replace("_", " ")
        for canonical, aliases in _COLUMN_ALIASES.items():
            if key in aliases and canonical not in rename.values():
                rename[col] = canonical
                break
    return df.rename(columns=rename)


def _read_candle_file(path: Path) -> pd.DataFrame:
    """Reads a parquet/csv candle file and normalizes it to canonical
    time/open/high/low/close/volume columns, tolerating whatever
    header spelling the exporter used. Raises ValueError if a
    required column (time/open/high/low/close) can't be identified."""
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    df = _normalize_columns(df)

    required = {"time", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path.name}: couldn't identify column(s) {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )

    if "volume" not in df.columns:
        df["volume"] = 0

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    if df["time"].isna().any():
        n_bad = int(df["time"].isna().sum())
        log.warning(f"{path.name}: {n_bad} rows had unparseable time values and were dropped")
        df = df.dropna(subset=["time"])

    return df[["time", "open", "high", "low", "close", "volume"]]


# Timeframe spellings that should all be treated as equivalent, keyed
# by a canonical token. Filenames are matched against every alias.
_TIMEFRAME_ALIASES = {
    "1m": {"1m", "m1", "min1", "1min"},
    "5m": {"5m", "m5", "5min"},
    "15m": {"15m", "m15", "15min"},
    "30m": {"30m", "m30", "30min"},
    "1h": {"1h", "h1", "60m"},
    "4h": {"4h", "h4", "240m"},
    "1d": {"1d", "d1", "daily"},
    "1w": {"1w", "w1", "weekly"},
}


def _timeframe_tokens(timeframe: str) -> set[str]:
    key = timeframe.strip().lower()
    for canonical, aliases in _TIMEFRAME_ALIASES.items():
        if key in aliases:
            return aliases
    return {key}


def _filename_matches(path: Path, symbol: str, timeframe: str) -> bool:
    name = path.stem.lower().replace(" ", "").replace("-", "_")
    tokens = set(name.replace(".", "_").split("_"))
    symbol_ok = symbol.lower() in name  # symbol usually appears unsplit, e.g. "EURUSD"
    tf_ok = bool(tokens & _timeframe_tokens(timeframe)) or any(
        alias in name for alias in _timeframe_tokens(timeframe)
    )
    return symbol_ok and tf_ok


def load_candles(symbol: str, timeframe: str) -> tuple[pd.DataFrame, str]:
    """Returns (df, source_label). Tries DB first, then a candle file
    under DATA_DIR/history whose filename matches the requested
    symbol+timeframe. Raises if nothing matching is found — it will
    NOT silently substitute a different symbol/timeframe's data."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql(
                "SELECT time, open, high, low, close, volume FROM candles "
                "WHERE symbol = ? AND timeframe = ? ORDER BY time",
                conn, params=[symbol, timeframe],
            )
        if len(df) >= 2000:
            return df, f"db:{symbol}/{timeframe}"
    except sqlite3.OperationalError:
        pass

    hist_dir = DATA_DIR / "history"
    if hist_dir.exists():
        all_files = list(hist_dir.rglob("*.parquet")) + list(hist_dir.rglob("*.csv"))
        matching = sorted(
            [p for p in all_files if _filename_matches(p, symbol, timeframe)],
            key=lambda p: -p.stat().st_size,
        )
        other = sorted(
            [p for p in all_files if p not in matching],
            key=lambda p: -p.stat().st_size,
        )

        skipped = []
        for path in matching:
            try:
                df = _read_candle_file(path)
            except Exception as e:
                skipped.append(f"  - {path.name}: {e}")
                continue
            if len(df) >= 2000:
                label = "SYNTHETIC" if "synthetic" in path.name.lower() else "real"
                return df, f"{path.suffix[1:]}:{path.name} ({label})"

        if skipped:
            log.warning("Files matching {symbol}/{timeframe} were found but skipped:\n"
                        + "\n".join(skipped))

        if other:
            names = ", ".join(p.name for p in other[:10])
            raise SystemExit(
                f"No candle file found matching symbol={symbol} timeframe={timeframe} "
                f"in {hist_dir}. Found other file(s) that DON'T match "
                f"(refusing to silently substitute them): {names}\n"
                f"Either point --symbol/--timeframe at one of these, or add a file "
                f"named like '{symbol}_{timeframe}.csv'."
            )

    raise SystemExit(
        "No usable candle data found (need >= 2000 bars with recognizable "
        "time/OHLC columns). Run `python -m scripts.setup_market_dna` first, "
        "or check the column headers in your data/history CSV files."
    )


# ── Reference strategy (proxy — NOT this repo's real strategy) ─
def generate_reference_trades(
    df: pd.DataFrame,
    *,
    atr_col: str = "atr_14",
    close_col: str = "close",
    sl_mult: float = 1.5,
    tp_mult: float = 2.5,
    max_hold: int = 48,
) -> pd.DataFrame:
    """
    EMA-cross entries (uses the already-computed `ema_fast_mid`
    column), ATR stop/target, resolved bar-by-bar. One position at a
    time (no pyramiding/overlap) so results are simple to interpret.

    This is a FIXED, deliberately simple reference signal — it exists
    only to generate a consistent set of trade outcomes to test
    whether DNA context improves them. It is not a recommendation and
    is not the repo's actual multi-agent strategy.
    """
    sign = np.sign(df["ema_fast_mid"].to_numpy())
    cross_up = (sign[:-1] <= 0) & (sign[1:] > 0)
    cross_down = (sign[:-1] >= 0) & (sign[1:] < 0)

    trades = []
    i = 1
    n = len(df)
    in_position = False
    while i < n - 1:
        if not in_position and (cross_up[i - 1] or cross_down[i - 1]):
            direction = 1 if cross_up[i - 1] else -1
            entry_idx = i
            entry_price = df[close_col].iat[entry_idx]
            atr = df[atr_col].iat[entry_idx]
            if not np.isfinite(atr) or atr <= 0:
                i += 1
                continue
            sl = entry_price - direction * sl_mult * atr
            tp = entry_price + direction * tp_mult * atr

            exit_idx, exit_price, result = None, None, None
            for j in range(entry_idx + 1, min(entry_idx + 1 + max_hold, n)):
                hi, lo = df["high"].iat[j], df["low"].iat[j]
                hit_sl = (lo <= sl) if direction == 1 else (hi >= sl)
                hit_tp = (hi >= tp) if direction == 1 else (lo <= tp)
                if hit_sl:  # conservative: SL wins ties (same-bar hit both)
                    exit_idx, exit_price, result = j, sl, "LOSS"
                    break
                if hit_tp:
                    exit_idx, exit_price, result = j, tp, "WIN"
                    break
            if exit_idx is None:
                exit_idx = min(entry_idx + max_hold, n - 1)
                exit_price = df[close_col].iat[exit_idx]
                result = "WIN" if direction * (exit_price - entry_price) > 0 else "LOSS"

            pnl_r = direction * (exit_price - entry_price) / (sl_mult * atr)
            trades.append({
                "entry_idx": entry_idx, "exit_idx": exit_idx, "direction": direction,
                "result": result, "pnl": pnl_r,
            })
            i = exit_idx + 1
            continue
        i += 1

    return pd.DataFrame(trades)


# ── Metrics ──────────────────────────────────────────────────
def summarize(trades: pd.DataFrame, label: str) -> dict:
    if len(trades) == 0:
        return {"label": label, "n": 0}
    wins = (trades["result"] == "WIN").sum()
    n = len(trades)
    win_rate = wins / n
    gp = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gl = -trades.loc[trades["pnl"] < 0, "pnl"].sum()
    pf = float(gp / gl) if gl > 0 else float("inf") if gp > 0 else 0.0
    expectancy = float(trades["pnl"].mean())
    return {"label": label, "n": n, "win_rate": round(win_rate, 4),
            "profit_factor": round(pf, 3) if np.isfinite(pf) else None,
            "expectancy_r": round(expectancy, 4)}


def bootstrap_expectancy_diff(a: pd.Series, b: pd.Series, n_boot: int = N_BOOTSTRAP) -> dict:
    """Bootstrap CI on (mean(b) - mean(a)) — is the DNA-filtered set's
    expectancy meaningfully different from baseline, or within noise?"""
    if len(a) < 10 or len(b) < 10:
        return {"diff": None, "ci_low": None, "ci_high": None, "significant": False}
    rng = np.random.default_rng(7)
    a_arr, b_arr = a.to_numpy(), b.to_numpy()
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        diffs[k] = rng.choice(b_arr, size=len(b_arr), replace=True).mean() - \
                   rng.choice(a_arr, size=len(a_arr), replace=True).mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "diff": round(float(diffs.mean()), 4),
        "ci_low": round(float(lo), 4),
        "ci_high": round(float(hi), 4),
        "significant": bool(lo > 0 or hi < 0),  # CI excludes zero
    }


def main():
    parser = argparse.ArgumentParser(description="Market DNA impact evaluation")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--timeframe", default="H1")
    parser.add_argument("--min-cluster-size", type=int, default=30)
    parser.add_argument(
        "--min-samples", type=int, default=None,
        help="HDBSCAN min_samples (density strictness). Defaults to "
             "min-cluster-size if not set. Lower values (e.g. 5-15) pull more "
             "borderline points OUT of the noise/UNKNOWN bucket at the cost "
             "of looser cluster boundaries.",
    )
    parser.add_argument(
        "--scan", action="store_true",
        help="Instead of running the full (slow) evaluation, fit the detector "
             "on Fold A for a GRID of min-cluster-size/min-samples combos and "
             "just report cluster count + noise%% for each. Skips the slow "
             "row-by-row labeling step entirely. Use this to find a promising "
             "combo before running the full evaluation.",
    )
    parser.add_argument(
        "--scan-cluster-sizes", default="15,20,30,50,75,100",
        help="Comma-separated min-cluster-size values to try with --scan.",
    )
    parser.add_argument(
        "--scan-samples", default="5,10,15",
        help="Comma-separated min-samples values to try with --scan "
             "(each combined with every cluster-size).",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("MARKET DNA — IMPACT EVALUATION (reference-strategy proxy test)")
    print("=" * 70)

    raw, source = load_candles(args.symbol, args.timeframe)
    is_synthetic = "SYNTHETIC" in source
    print(f"\nData source: {source}  ({len(raw)} bars)")
    if is_synthetic:
        print("NOTE: synthetic data — this run validates the PIPELINE mechanics "
              "(does the plumbing work, does filtering do anything at all), "
              "NOT real-market performance. Re-run against real history before "
              "trusting the verdict for live use.")

    df = add_indicators(raw)
    print(f"After indicators (warmup dropped): {len(df)} bars")

    split = make_three_way_split(df, fold_a_frac=0.5, fold_b_frac=0.25)

    if args.scan:
        cluster_sizes = [int(x) for x in args.scan_cluster_sizes.split(",") if x.strip()]
        sample_values = [int(x) for x in args.scan_samples.split(",") if x.strip()]
        print(f"\nSCAN MODE: fitting {len(cluster_sizes) * len(sample_values)} combos on "
              f"Fold A only (no labeling/evaluation). This just tells you which "
              f"combos are even worth a full run.\n")
        print(f"  {'min_cluster_size':>17}  {'min_samples':>11}  {'clusters':>8}  {'noise_%':>8}")
        results = []
        with warnings.catch_warnings():
            warnings.simplefilter("once", UserWarning)
            for mcs in cluster_sizes:
                for ms in sample_values:
                    if ms > mcs:
                        continue  # min_samples > min_cluster_size is rarely useful and just wastes a fit
                    try:
                        cfg = DNAConfig(min_cluster_size=mcs, min_samples=ms)
                        det = fit_frozen_detector(split.fold_a, cfg)
                        labels = det.clusterer.labels_
                        noise_pct = 100.0 * (labels == -1).sum() / len(labels)
                        results.append((mcs, ms, det.n_clusters_, noise_pct))
                        print(f"  {mcs:>17}  {ms:>11}  {det.n_clusters_:>8}  {noise_pct:>7.1f}%")
                    except Exception as e:
                        print(f"  {mcs:>17}  {ms:>11}  FAILED: {e}")
        print("\nPick a (min_cluster_size, min_samples) combo above with >=2 clusters "
              "and reasonable noise%, then re-run WITHOUT --scan using those values "
              "for the full evaluation.")
        return

    cfg = DNAConfig(min_cluster_size=args.min_cluster_size, min_samples=args.min_samples)
    effective_min_samples = args.min_samples if args.min_samples is not None else args.min_cluster_size
    print(f"\nHDBSCAN params: min_cluster_size={args.min_cluster_size}, "
          f"min_samples={effective_min_samples}"
          + (" (defaulted from min-cluster-size)" if args.min_samples is None else ""))

    detector = fit_frozen_detector(split.fold_a, cfg)
    print(f"\nDetector fit on Fold A: {detector.n_clusters_} clusters, "
          f"{detector.n_train_rows} rows [{detector.train_window_start} .. {detector.train_window_end}]")

    if detector.n_clusters_ == 0:
        print("\n" + "=" * 70)
        print("VERDICT: NO CLUSTERS FOUND")
        print("=" * 70)
        print(
            f"HDBSCAN found 0 clusters with min_cluster_size={args.min_cluster_size}, "
            f"min_samples={effective_min_samples} — every row would be labeled noise/UNKNOWN. "
            "Labeling Fold B/C would be pointless (and would spam thousands of "
            "'Clusterer does not have any defined clusters' warnings).\n"
            "Try a SMALLER --min-cluster-size and/or --min-samples (e.g. halve both) "
            "and re-run."
        )
        return

    def _make_progress_printer(label: str, start_time: float):
        def _cb(done: int, total: int) -> None:
            pct = 100.0 * done / total
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0
            eta_sec = (total - done) / rate if rate > 0 else 0
            eta_min = int(eta_sec // 60)
            eta_s = int(eta_sec % 60)
            end = "\n" if done == total else ""
            print(f"\r  [{label}] {done}/{total} ({pct:5.1f}%)  ETA {eta_min}m{eta_s:02d}s   ",
                  end=end, flush=True)
        return _cb

    print("Labeling Fold B/C (this loops row-by-row over approximate_predict, may take a moment)...")
    with warnings.catch_warnings():
        # hdbscan's approximate_predict() re-emits the SAME UserWarning
        # on every single call when relevant (e.g. near-empty clusters);
        # since label_fold calls it once per row, that can mean
        # thousands of duplicate warning lines. Show it once, not once
        # per row.
        warnings.simplefilter("once", UserWarning)
        fold_b_labeled = label_fold(
            detector, split.fold_b,
            progress_callback=_make_progress_printer("Fold B", time.time()),
        )
        fold_c_labeled = label_fold(
            detector, split.fold_c,
            progress_callback=_make_progress_printer("Fold C", time.time()),
        )

    unknown_rate_b = (fold_b_labeled["dna_state"] == "UNKNOWN").mean()
    unknown_rate_c = (fold_c_labeled["dna_state"] == "UNKNOWN").mean()
    print(f"UNKNOWN rate — Fold B: {unknown_rate_b:.1%}   Fold C: {unknown_rate_c:.1%}")

    # Reference-strategy trades on each fold, then attach cluster labels
    trades_b = generate_reference_trades(fold_b_labeled)
    trades_c = generate_reference_trades(fold_c_labeled)
    trades_b["cluster_id"] = fold_b_labeled["dna_cluster_id"].reindex(trades_b["entry_idx"]).to_numpy()
    trades_c["cluster_id"] = fold_c_labeled["dna_cluster_id"].reindex(trades_c["entry_idx"]).to_numpy()
    print(f"\nReference-strategy trades — Fold B (journal source): {len(trades_b)}   "
          f"Fold C (out-of-sample eval): {len(trades_c)}")

    if len(trades_b) < MIN_TRADES_FOR_VERDICT or len(trades_c) < MIN_TRADES_FOR_VERDICT:
        print("\n" + "=" * 70)
        print(f"VERDICT: INSUFFICIENT DATA")
        print("=" * 70)
        print(f"Need >= {MIN_TRADES_FOR_VERDICT} trades in both Fold B and Fold C to say "
              f"anything statistically meaningful. Got B={len(trades_b)}, C={len(trades_c)}.")
        print("Use more history (longer date range) and re-run.")
        return

    journal = build_cluster_journal(trades_b, model_id=detector.model_id)
    tier_counts = pd.Series([s.tier for s in journal]).value_counts().to_dict()
    print(f"\nFold-B journal: {len(journal)} clusters. Tier distribution: {tier_counts}")

    print("\nPer-cluster journal (from Fold B, applied to Fold C):")
    print(f"  {'cluster':>7}  {'trades':>6}  {'win_rate':>8}  {'ci_high':>7}  "
          f"{'expectancy_R':>12}  {'PF':>6}  {'tier':<16}  recommendation")
    for s in sorted(journal, key=lambda s: s.cluster_id):
        dec = decision_context(s)
        exp_str = f"{s.expectancy_r:+.4f}" if s.expectancy_r is not None else "n/a"
        pf_str = f"{s.profit_factor:.2f}" if s.profit_factor is not None else "n/a"
        print(f"  {s.cluster_id:>7}  {s.trades:>6}  {s.win_rate:>8.3f}  {s.ci_high:>7.3f}  "
              f"{exp_str:>12}  {pf_str:>6}  {s.tier:<16}  {dec['recommendation']}")

    # Apply Fold-B-derived journal to Fold-C trades
    decisions = trades_c["cluster_id"].apply(lambda cid: decision_context(lookup(journal, cid)) if pd.notna(cid) else decision_context(None))
    trades_c = trades_c.assign(
        recommendation=[d["recommendation"] for d in decisions],
        position_multiplier=[d["position_multiplier"] for d in decisions],
    )

    baseline = trades_c
    approved_only = trades_c[trades_c["recommendation"] == "APPROVE"]
    sized = trades_c.assign(sized_pnl=trades_c["pnl"] * trades_c["position_multiplier"])

    print("\n--- Fold C results (out-of-sample) ---")
    for label, subset, col in [
        ("BASELINE (no filter)", baseline, "pnl"),
        ("DNA-FILTERED (APPROVE only)", approved_only, "pnl"),
    ]:
        s = summarize(subset, label)
        print(f"  {label:32s} n={s.get('n',0):4d}  win_rate={s.get('win_rate')}  "
              f"PF={s.get('profit_factor')}  expectancy_R={s.get('expectancy_r')}")

    baseline_expectancy = float(sized["pnl"].mean())
    sized_expectancy = float(sized["sized_pnl"].mean())
    print(f"\n  Position-sizing effect (all Fold C trades, size-adjusted): "
          f"unsized_expectancy_R={baseline_expectancy:.4f}  sized_expectancy_R={sized_expectancy:.4f}")

    boot = bootstrap_expectancy_diff(baseline["pnl"], approved_only["pnl"]) if len(approved_only) >= 10 else None

    # Drift: Fold A training distribution vs Fold C live-period distribution
    fold_a_labels = detector.clusterer.labels_
    fold_c_labels = fold_c_labeled["dna_cluster_id"].astype("float64").fillna(-1).astype(int).to_numpy()
    drift = population_stability_index(fold_a_labels, fold_c_labels, n_clusters=detector.n_clusters_)
    print(f"\nDrift (Fold A training dist vs Fold C live dist): PSI={drift.psi}  status={drift.status}")

    # ── Verdict ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    reasons = []
    approve_coverage = len(approved_only) / len(trades_c) if len(trades_c) else 0
    reasons.append(f"- APPROVE coverage: {approve_coverage:.1%} of Fold C trades kept by the filter")
    reasons.append(f"- UNKNOWN rate (Fold C): {unknown_rate_c:.1%}")
    if boot:
        reasons.append(f"- Bootstrap expectancy lift (APPROVE vs baseline): "
                        f"{boot['diff']} [{boot['ci_low']}, {boot['ci_high']}] "
                        f"{'(excludes zero — likely real)' if boot['significant'] else '(includes zero — could be noise)'}")
    reasons.append(f"- Drift status: {drift.status}")

    if len(approved_only) < 30:
        verdict = "NOT ENOUGH APPROVED TRADES TO JUDGE"
        reasons.append("- Fewer than 30 APPROVE-tier trades in Fold C — can't evaluate the filtered subset reliably.")
    elif boot and boot["significant"] and boot["diff"] > 0:
        verdict = "IMPACTFUL — recommend using as a position-sizing/context filter"
        reasons.append("- Filtered subset shows a statistically distinguishable expectancy improvement out-of-sample.")
    elif boot and boot["significant"] and boot["diff"] < 0:
        verdict = "DO NOT USE — filter is currently HURTING out-of-sample expectancy"
        reasons.append("- Filtered subset is statistically WORSE than baseline. Do not deploy this config as-is.")
    else:
        verdict = "INCONCLUSIVE — collect more data before deciding"
        reasons.append("- Expectancy difference is within bootstrap noise; can't distinguish filter effect from chance yet.")

    print(f"\n  {verdict}\n")
    for r in reasons:
        print(f"  {r}")

    if is_synthetic:
        print("\n  Caveat: this verdict is based on SYNTHETIC data (pipeline smoke test).")
        print("  Re-run with real multi-year history before this verdict applies to live trading.")
    print("\n  Caveat: entries came from a fixed reference strategy (EMA-cross), not this")
    print("  repo's actual multi-agent signal logic. A real deployment decision should")
    print("  re-run this same fold-A/B/C methodology using real historical trade outcomes")
    print("  once enough of them exist (see analysis/README_MARKET_DNA.md).")


if __name__ == "__main__":
    main()