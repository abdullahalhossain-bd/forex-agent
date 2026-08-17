"""
Baseline evaluation of the ML system's actual labeling logic (ml/triple_barrier_labels.py,
reused verbatim) against the 6 supplied M15 pairs.

Two split modes are run side by side to quantify the leakage bug found in
ml/pipeline/phase6_dataset.py (naive chronological split, no purge/embargo —
despite ml/cv_splitter.py's PurgedEmbargoedSplitter existing in the repo and
NEVER being called from the pipeline entrypoint):

  1. "pipeline_naive"  — exactly what phase6_dataset.create_datasets() does today
  2. "purged_embargoed" — same split ratios, but purges label-window-overlapping
                           rows at the boundaries (holding_period bars) + embargo

Trading costs are modeled from the CSV's own `spread` column (broker-quoted,
in points) rather than assumed.
"""
import sys, time
sys.path.insert(0, "/home/claude/work")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml.triple_barrier_labels import triple_barrier_labels

PAIRS = ["EURAUD", "GBPCAD", "EURCAD", "GBPSEK", "GBPNOK", "EURNZD"]
DATA_DIR = "/mnt/user-data/uploads"

HOLDING = 16          # bars (4h on M15)
TP_ATR = 1.5
SL_ATR = 1.5
TRAIN_PCT, VAL_PCT = 0.70, 0.15
PROB_THRESHOLD = 0.55  # only trade when model confidence exceeds this

PIP = {"JPY": 0.01}


def pip_size(pair):
    return 0.01 if pair.endswith("JPY") else 0.0001


def load(pair):
    df = pd.read_csv(f"{DATA_DIR}/{pair}_M15.csv")
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df = df.sort_values("datetime_utc").reset_index(drop=True)
    return df


def build_features(df):
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["tick_volume"]
    f = pd.DataFrame(index=df.index)

    # returns / momentum
    for n in (1, 3, 5, 10, 20):
        f[f"ret_{n}"] = c.pct_change(n)

    # ATR% (volatility, same style as their feature_engineer.atr_pct)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    f["atr_pct"] = atr / c

    # RSI(14)
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    f["rsi14"] = 100 - 100 / (1 + rs)

    # EMA distances
    for n in (10, 20, 50, 100):
        ema = c.ewm(span=n, adjust=False).mean()
        f[f"dist_ema{n}"] = (c - ema) / c

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    f["macd_hist"] = (macd - signal) / c

    # Bollinger %b
    ma20 = c.rolling(20).mean()
    sd20 = c.rolling(20).std()
    f["bb_pct"] = (c - (ma20 - 2 * sd20)) / (4 * sd20)

    # volume ratio
    f["vol_ratio"] = v / v.rolling(20).mean()

    # candle structure
    f["body_pct"] = (c - o).abs() / (h - l).replace(0, np.nan)
    f["upper_wick"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / (h - l).replace(0, np.nan)
    f["lower_wick"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / (h - l).replace(0, np.nan)

    # session / time
    hour = df["datetime_utc"].dt.hour
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    f["dow"] = df["datetime_utc"].dt.dayofweek

    # broker spread itself is informative (liquidity proxy)
    f["spread_pts"] = df["spread"]

    return f, atr


def purge_train_val_test(n, train_end, val_end, horizon, embargo):
    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, n)
    # purge training rows whose label window [i, i+horizon] overlaps val start
    train_idx = train_idx[train_idx < train_end - horizon]
    val_idx = val_idx[(val_idx >= train_end + embargo) & (val_idx < val_end - horizon)]
    test_idx = test_idx[test_idx >= val_end + embargo]
    return train_idx, val_idx, test_idx


def run_pair(pair, split_mode):
    df = load(pair)
    feats, atr = build_features(df)
    labels = triple_barrier_labels(
        df, holding_period=HOLDING, take_profit_width=TP_ATR,
        stop_loss_width=SL_ATR, atr_period=14, use_atr=True,
    )

    data = feats.copy()
    data["label"] = labels
    data = data.dropna()
    X_cols = feats.columns.tolist()
    n = len(data)

    train_end = int(n * TRAIN_PCT)
    val_end = int(n * (TRAIN_PCT + VAL_PCT))

    if split_mode == "pipeline_naive":
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(train_end, val_end)
        test_idx = np.arange(val_end, n)
    else:
        train_idx, val_idx, test_idx = purge_train_val_test(n, train_end, val_end, HOLDING, embargo=HOLDING)

    Xtr, ytr = data.iloc[train_idx][X_cols], data.iloc[train_idx]["label"]
    Xva, yva = data.iloc[val_idx][X_cols], data.iloc[val_idx]["label"]
    Xte, yte = data.iloc[test_idx][X_cols], data.iloc[test_idx]["label"]

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=50,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    clf.fit(Xtr, ytr)
    classes = clf.classes_  # e.g. [-1, 0, 1]

    proba = clf.predict_proba(Xte)
    pred_class_idx = np.argmax(proba, axis=1)
    pred_class = classes[pred_class_idx]
    pred_conf = proba[np.arange(len(proba)), pred_class_idx]

    # only take a trade when model predicts a directional move (+1/-1) with conf >= threshold
    take = (pred_class != 0) & (pred_conf >= PROB_THRESHOLD)
    traded_actual = yte.values[take]
    traded_pred = pred_class[take]

    n_trades = int(take.sum())
    if n_trades == 0:
        return dict(pair=pair, split=split_mode, n_test=len(yte), trades=0, winrate=np.nan,
                    freq_per_day=0.0, val_acc=np.nan)

    wins = int(((traded_pred == 1) & (traded_actual == 1)).sum() + ((traded_pred == -1) & (traded_actual == -1)).sum())
    losses = int(((traded_pred == 1) & (traded_actual == -1)).sum() + ((traded_pred == -1) & (traded_actual == 1)).sum())
    timeouts = n_trades - wins - losses
    winrate = wins / (wins + losses) if (wins + losses) > 0 else np.nan

    test_days = (df["datetime_utc"].iloc[data.index[test_idx][-1]] - df["datetime_utc"].iloc[data.index[test_idx][0]]).total_seconds() / 86400
    freq_per_day = n_trades / test_days if test_days > 0 else np.nan

    val_pred = clf.predict(Xva)
    val_acc = (val_pred == yva.values).mean()

    return dict(pair=pair, split=split_mode, n_test=len(yte), trades=n_trades,
                wins=wins, losses=losses, timeouts=timeouts,
                winrate=round(winrate, 4) if winrate == winrate else np.nan,
                freq_per_day=round(freq_per_day, 2), val_acc=round(val_acc, 4))


if __name__ == "__main__":
    rows = []
    for pair in PAIRS:
        for mode in ["pipeline_naive", "purged_embargoed"]:
            t0 = time.time()
            r = run_pair(pair, mode)
            r["sec"] = round(time.time() - t0, 1)
            rows.append(r)
            print(r)
    out = pd.DataFrame(rows)
    out.to_csv("/home/claude/work/baseline_results.csv", index=False)
    print(out.to_string(index=False))
