import sys, time
sys.path.insert(0, "/home/claude/work")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from backtest_baseline import PAIRS, HOLDING, TP_ATR, SL_ATR, PROB_THRESHOLD, load, build_features
from ml.triple_barrier_labels import triple_barrier_labels

N_FOLDS = 5
EMBARGO = HOLDING


def walk_forward_pair(pair):
    df = load(pair)
    feats, atr = build_features(df)
    labels = triple_barrier_labels(df, holding_period=HOLDING, take_profit_width=TP_ATR,
                                    stop_loss_width=SL_ATR, atr_period=14, use_atr=True)
    data = feats.copy()
    data["label"] = labels
    data["dt"] = df["datetime_utc"]
    data = data.dropna()
    X_cols = feats.columns.tolist()
    n = len(data)

    # 5 expanding-window folds: each fold's test block is a distinct ~1/6 chronological slice
    fold_edges = np.linspace(int(n * 0.4), n, N_FOLDS + 1).astype(int)
    results = []
    for k in range(N_FOLDS):
        test_start, test_end = fold_edges[k], fold_edges[k + 1]
        train_end = test_start - EMBARGO
        if train_end < 500:
            continue
        Xtr, ytr = data.iloc[:train_end][X_cols], data.iloc[:train_end]["label"]
        Xte, yte = data.iloc[test_start:test_end][X_cols], data.iloc[test_start:test_end]["label"]

        clf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=50,
                                      class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(Xtr, ytr)
        classes = clf.classes_
        proba = clf.predict_proba(Xte)
        idx = np.argmax(proba, axis=1)
        pred = classes[idx]
        conf = proba[np.arange(len(proba)), idx]
        take = (pred != 0) & (conf >= PROB_THRESHOLD)
        tp, ta = pred[take], yte.values[take]
        n_trades = int(take.sum())
        wins = int(((tp == 1) & (ta == 1)).sum() + ((tp == -1) & (ta == -1)).sum())
        losses = int(((tp == 1) & (ta == -1)).sum() + ((tp == -1) & (ta == 1)).sum())
        winrate = wins / (wins + losses) if (wins + losses) else np.nan
        d0, d1 = data["dt"].iloc[test_start], data["dt"].iloc[test_end - 1]
        results.append(dict(pair=pair, fold=k, period=f"{d0.date()}..{d1.date()}",
                             n_trades=n_trades, wins=wins, losses=losses,
                             winrate=round(winrate, 3) if winrate == winrate else np.nan))
    return results


if __name__ == "__main__":
    all_rows = []
    for pair in PAIRS:
        t0 = time.time()
        rows = walk_forward_pair(pair)
        all_rows.extend(rows)
        print(f"{pair} done in {time.time()-t0:.1f}s")
    out = pd.DataFrame(all_rows)
    out.to_csv("/home/claude/work/walkforward_results.csv", index=False)
    print(out.to_string(index=False))
    print("\n--- per-pair winrate std dev across folds (robustness check) ---")
    print(out.groupby("pair")["winrate"].agg(["mean", "std", "min", "max", "count"]))
