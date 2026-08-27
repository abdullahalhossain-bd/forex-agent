#!/usr/bin/env python3
"""
scripts/train_champions.py — Terminal champion retraining (Task 7)
===================================================================

Retrains ML champions from REAL historical OHLC CSVs so the live pipeline
(memory/ml_models/_registry.json) gets healthy models again — no MT5, no
FeatureStore, runs fully offline in a terminal:

    python3 scripts/train_champions.py                          # all CSV pairs, M15
    python3 scripts/train_champions.py --pairs EURUSD GBPUSD    # subset
    python3 scripts/train_champions.py --timeframes 15m 1h      # multiple TFs
    python3 scripts/train_champions.py --stride 1               # max samples
    python3 scripts/train_champions.py --list                   # show inventory

How it works
------------
1.  Loads data/{PAIR}_{TF}.csv (author-supplied real candles for the 7 majors).
2.  Adds causal indicator columns exactly as production market data does
    (sma_9/20/50/200, ema_*, rsi_14 (+rsi), macd/macd_signal,
    bb_high/bb_low, atr_14 (+atr), volume).
3.  For every bar (optionally strided), builds the EXACT live feature vector
    via ml.feature_engineer.FeatureEngineer.build_feature_vector(window, {})
    with a per-bar session_ctx from analysis/session_analyzer.py — same code
    path used at inference time.
4.  Attaches raw open/high/low/close as labeler helpers, hands the frame to
    ModelTrainer.train_all(dataset_df=..., labeling_method="triple_barrier",
    use_purged_split=True). dataset_builder drops helper OHLC after labeling
    (schema-parity fix 2026-08-27), degenerate-model guards refuse to save
    junk champions, ModelStore writes portable relative registry entries.
5.  Prints a per-pair summary + writes logs/ml/champion_train_report_*.json.

NOTE on coverage: only the 7 majors have CSVs in this repo clone. Other
pairs stay cold until data is supplied or the Windows/MT5 box re-runs this
script there (same script works — it only needs data/*.csv present).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd

# ── Canonical timeframe mapping (registry keys use "15m"/"1h"/"4h") ──
TF_CSV_SUFFIX = {"15m": "M15", "m15": "M15", "M15": "M15",
                 "1h": "H1", "H1": "H1",
                 "4h": "H4", "H4": "H4"}
WARMUP_BARS = 260          # ≥ largest lookback (sma200/ema200) + change_20 headroom
DEFAULT_HORIZON = 16       # triple-barrier holding period == DEFAULT_PURGE_WINDOW


def discover_inventory() -> list[tuple[str, str, int]]:
    """Return [(pair, tf_key, rows)] for every usable CSV."""
    out = []
    for tf_key, suffix in [("15m", "M15"), ("1h", "H1"), ("4h", "H4")]:
        for p in sorted(glob.glob(str(PROJECT_ROOT / "data" / f"*_{suffix}.csv"))):
            pair = Path(p).name.split("_")[0].upper()
            try:
                rows = sum(1 for _ in open(p, "rb")) - 1
            except OSError:
                continue
            if rows > WARMUP_BARS + 400:
                out.append((pair, tf_key, rows))
    return out


def load_csv(pair: str, tf_key: str) -> pd.DataFrame:
    suffix = TF_CSV_SUFFIX[tf_key]
    path = PROJECT_ROOT / "data" / f"{pair}_{suffix}.csv"
    df = pd.read_csv(path, parse_dates=["datetime_utc"])
    df = df.rename(columns={"tick_volume": "volume"})
    df["volume"] = df["volume"].fillna(0.0)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df = df.set_index("datetime_utc").sort_index()
    keep = ["open", "high", "low", "close", "volume"]
    return df[keep].astype(float)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Causal (rolling) indicators — computed once over the FULL series so
    sliced windows carry correct values without recomputation cost."""
    c, h, l = df["close"], df["high"], df["low"]
    for n in (9, 20, 50, 200):
        df[f"sma_{n}"] = c.rolling(n).mean()
        df[f"ema_{n}"] = c.ewm(span=n, adjust=False).mean()
        # warm-up EMA values from partial windows are slightly biased; blank them
        df.loc[df.index[: n - 1], f"ema_{n}"] = np.nan

    # RSI-14 Wilder
    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    df["rsi_14"] = rsi
    df["rsi"] = rsi

    # MACD 12/26/9
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # Bollinger 20/2
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std(ddof=0)
    df["bb_high"] = mid + 2.0 * sd
    df["bb_low"] = mid - 2.0 * sd

    # ATR-14 Wilder
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    df["atr"] = atr
    df["atr_14"] = atr
    return df


def build_feature_rows(df_full: pd.DataFrame, pair: str, tf_key: str,
                       window: int, stride: int) -> tuple[pd.DataFrame, dict]:
    """Iterate bars → live-schema feature rows (+ labeler helper OHLC)."""
    from ml.feature_engineer import get_feature_engineer

    try:
        from analysis.session_analyzer import SessionAnalyzer
        sa = SessionAnalyzer()
        def sess_ctx(ts):
            try:
                name = sa.get_current_session(ts.to_pydatetime())["primary_session"]
                return {"current_session": name}
            except Exception:
                return {}
    except Exception:
        def sess_ctx(ts):
            return {}

    fe = get_feature_engineer()
    idx = df_full.index
    n = len(df_full)
    rows = []
    t0 = time.time()
    skipped = 0

    for i in range(WARMUP_BARS, n, stride):
        lo = i - window
        win = df_full.iloc[lo if lo >= 0 else 0: i + 1]
        ts = idx[i]
        ao = {"session_ctx": sess_ctx(ts)}
        try:
            feats = fe.build_feature_vector(win, ao, pair=pair, timeframe=tf_key)
        except Exception:
            skipped += 1
            continue
        last = df_full.iloc[i]
        row = {k: float(v) for k, v in feats.items()}
        # labeler helper columns (dropped again inside dataset_builder post-labeling)
        row["open"] = float(last["open"])
        row["high"] = float(last["high"])
        row["low"] = float(last["low"])
        row["close"] = float(last["close"])
        rows.append(row)

    stats = {
        "bars_total": int(n),
        "rows_built": len(rows),
        "rows_skipped_engine_error": skipped,
        "feature_build_sec": round(time.time() - t0, 1),
        "window": window,
        "stride": stride,
    }
    logf(f"    rows={len(rows)} ({stats['feature_build_sec']}s, engine_skipped={skipped})")
    return pd.DataFrame(rows), stats


_LOG_TS = datetime.now(timezone.utc)


def logf(msg: str) -> None:
    print(f"[train_champions {_LOG_TS:%H:%M:%S}] {msg}", flush=True)


def train_pair_trainer(pair: str, tf_key: str, feats_df: pd.DataFrame,
                       horizon: int, min_samples: int) -> dict:
    from ml.model_trainer import get_model_trainer

    trainer = get_model_trainer()
    result = trainer.train_all(
        pair=pair.upper(), timeframe=tf_key,
        min_samples=min_samples,
        labeling_method="triple_barrier",
        use_purged_split=True,
        label_horizon=horizon,
        include_bootstrap=False,
        dataset_df=feats_df,
    )
    return result.to_dict()


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline champion retraining from CSVs")
    ap.add_argument("--pairs", nargs="*", default=None,
                    help="e.g. EURUSD GBPUSD (default: all CSV pairs)")
    ap.add_argument("--timeframes", nargs="*", default=["15m"],
                    help="subset of 15m 1h 4h (default: 15m)")
    ap.add_argument("--window", type=int, default=320,
                    help="feature window bars (default 320)")
    ap.add_argument("--stride", type=int, default=2,
                    help="sample every Nth bar (default 2)")
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON,
                    help=f"triple-barrier holding period (default {DEFAULT_HORIZON})")
    ap.add_argument("--min-samples", type=int, default=300)
    ap.add_argument("--list", action="store_true", help="show inventory and exit")
    args = ap.parse_args()

    inventory = discover_inventory()
    if args.list:
        for p, t, r in inventory:
            print(f"{p:>8} {t:>4} {r:>7} rows")
        print(f"total: {len(inventory)} datasets")
        return 0
    if not inventory:
        logf("ERROR: no data/*_{M15,H1,H4}.csv found under data/")
        return 2

    _suffix_to_key = {"M15": "15m", "H1": "1h", "H4": "4h"}
    wanted_tf = {_suffix_to_key.get(TF_CSV_SUFFIX.get(t, t), t) for t in args.timeframes}
    jobs = [(p, t, r) for p, t, r in inventory
            if (args.pairs is None or p.upper() in {x.upper() for x in args.pairs})
            and t in wanted_tf]
    if not jobs:
        logf("ERROR: requested pairs/timeframes have no usable CSV. Use --list.")
        return 2

    logf(f"jobs={[f'{p}/{t}' for p, t, _ in jobs]} stride={args.stride} "
         f"window={args.window} horizon={args.horizon}")

    report = {"started": _LOG_TS.isoformat(), "results": []}
    overall_ok = True

    for pair, tf_key, csv_rows in jobs:
        logf(f"▶ {pair} {tf_key}: loading CSV ({csv_rows} rows)")
        try:
            df = add_indicators(load_csv(pair, tf_key))
            feats_df, feat_stats = build_feature_rows(df, pair, tf_key,
                                                      args.window, args.stride)
            if len(feats_df) < args.min_samples:
                logf(f"  ✗ {pair}/{tf_key}: only {len(feats_df)} feature rows "
                     f"(need ≥{args.min_samples}) — skip")
                report["results"].append({"pair": pair, "timeframe": tf_key,
                                          "status": "insufficient_rows",
                                          "rows": int(len(feats_df)),
                                          **feat_stats})
                overall_ok = False
                continue

            logf(f"  training triple_barrier(h={args.horizon}) purged split …")
            res = train_pair_trainer(pair, tf_key, feats_df,
                                     args.horizon, args.min_samples)
            entry = {"pair": pair, "timeframe": tf_key, "status":
                     "ok" if res["models_trained"] else "no_model_saved",
                     **feat_stats,
                     "models_trained": res["models_trained"],
                     "best_model": res["best_model"],
                     "metrics": res["metrics"],
                     "errors": res["errors"]}
            flag = "✓" if res["models_trained"] else "✗"
            logf(f"  {flag} {pair}/{tf_key}: trained={res['models_trained']} "
                 f"best={res['best_model']}")
            for m, met in res["metrics"].items():
                logf(f"     · {m}: acc={met.get('accuracy')} auc={met.get('roc_auc')} "
                     f"prec={met.get('precision')} rec={met.get('recall')}")
            report["results"].append(entry)
            if not res["models_trained"]:
                overall_ok = False
        except Exception as exc:
            import traceback
            traceback.print_exc()
            logf(f"  ✗ {pair}/{tf_key}: EXCEPTION {exc}")
            report["results"].append({"pair": pair, "timeframe": tf_key,
                                      "status": "exception", "error": str(exc)})
            overall_ok = False

    report["finished"] = datetime.now(timezone.utc).isoformat()
    report["overall_ok"] = overall_ok
    out_dir = PROJECT_ROOT / "logs" / "ml"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"champion_train_report_{_LOG_TS:%Y%m%d_%H%M%S}.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    logf(f"report → {out_path}")
    logf("DONE " + ("OK" if overall_ok else "WITH FAILURES"))
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
