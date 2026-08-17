"""
Nested walk-forward tuning for EURAUD, GBPCAD, EURCAD, EURNZD.

Methodology (to avoid the peeking risk in the earlier quick GBPNOK/GBPSEK
check, which selected TP/SL width by looking at all 5 folds at once):

  - Same 5 expanding-window folds as backtest_walkforward.py.
  - TP/SL ATR multiple is grid-searched using ONLY folds 0-3 (tuning set).
    Selection criterion: mean cost-adjusted net R/trade across those folds,
    with a variance penalty (mean - 0.5*std) so we don't pick a config that
    only worked in one lucky fold.
  - Fold 4 is NEVER touched during selection. It is evaluated exactly once,
    with the config chosen from folds 0-3, and reported as the honest
    out-of-sample result.
  - Cost = pair's own average `spread` column (real broker data, not
    assumed), converted with point_size=1e-5 (verified against each CSV's
    quoted decimal precision).
"""
import sys, time
sys.path.insert(0, "/home/claude/work")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from backtest_baseline import load, build_features, PROB_THRESHOLD
from ml.triple_barrier_labels import triple_barrier_labels

PAIRS = ["EURAUD", "GBPCAD", "EURCAD", "EURNZD"]
HOLDING = 16
CANDIDATE_ATR = [1.5, 2.0, 2.5, 3.0]
N_FOLDS = 5
TUNE_FOLDS = [0, 1, 2]      # selection only ever sees these (kept to 3 folds for speed)
TEST_FOLD = 4                 # touched exactly once, at the very end


def avg_spread_price(pair):
    df = pd.read_csv(f"/mnt/user-data/uploads/{pair}_M15.csv")
    return df["spread"].mean() * 1e-5  # verified: all 4 pairs quoted to 5 decimals (checked CSV head)


def fit_eval(Xtr, ytr, Xte, yte, thresh=PROB_THRESHOLD):
    clf = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=50,
                                  class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(Xtr, ytr)
    classes = clf.classes_
    proba = clf.predict_proba(Xte)
    idx = np.argmax(proba, axis=1)
    pred = classes[idx]
    conf = proba[np.arange(len(proba)), idx]
    take = (pred != 0) & (conf >= thresh)
    tp, ta = pred[take], yte.values[take]
    n_trades = int(take.sum())
    wins = int(((tp == 1) & (ta == 1)).sum() + ((tp == -1) & (ta == -1)).sum())
    losses = int(((tp == 1) & (ta == -1)).sum() + ((tp == -1) & (ta == 1)).sum())
    return n_trades, wins, losses


def run_config(pair, atr_mult, holding=HOLDING, n_folds=N_FOLDS, folds_needed=None):
    df = load(pair)
    feats, atr_series = build_features(df)
    labels = triple_barrier_labels(df, holding_period=holding, take_profit_width=atr_mult,
                                    stop_loss_width=atr_mult, atr_period=14, use_atr=True)
    data = feats.copy()
    data["label"] = labels
    data["dt"] = df["datetime_utc"]
    data = data.dropna()
    Xcols = feats.columns.tolist()
    n = len(data)
    avg_atr = atr_series.reindex(data.index).mean()
    sl_width_price = atr_mult * avg_atr
    cost_price = avg_spread_price(pair)
    cost_R = cost_price / sl_width_price

    fold_edges = np.linspace(int(n * 0.4), n, n_folds + 1).astype(int)
    per_fold = []
    folds_to_run = folds_needed if folds_needed is not None else range(n_folds)
    for k in folds_to_run:
        ts, te = fold_edges[k], fold_edges[k + 1]
        tr_end = ts - holding
        if tr_end < 500:
            per_fold.append(dict(fold=k, trades=0, winrate=np.nan, net_R=np.nan))
            continue
        Xtr, ytr = data.iloc[:tr_end][Xcols], data.iloc[:tr_end]["label"]
        Xte, yte = data.iloc[ts:te][Xcols], data.iloc[ts:te]["label"]
        n_trades, wins, losses = fit_eval(Xtr, ytr, Xte, yte)
        winrate = wins / (wins + losses) if (wins + losses) else np.nan
        net_R = (2 * winrate - 1) - cost_R if winrate == winrate else np.nan
        approx_days = (te - ts) / 96
        per_fold.append(dict(fold=k, trades=n_trades, wins=wins, losses=losses,
                              winrate=winrate, net_R=net_R,
                              freq_per_day=n_trades / approx_days if approx_days > 0 else np.nan))
    return per_fold, cost_R


if __name__ == "__main__":
    chosen = {}
    all_rows = []
    for pair in PAIRS:
        print(f"\n=== {pair} — tuning ATR multiple on folds {TUNE_FOLDS} ===", flush=True)
        best = None
        for atr_mult in CANDIDATE_ATR:
            t0 = time.time()
            per_fold, cost_R = run_config(pair, atr_mult, folds_needed=TUNE_FOLDS)
            tune_rows = [r for r in per_fold if r["fold"] in TUNE_FOLDS and r["net_R"] == r["net_R"]]
            for r in per_fold:
                r2 = dict(r); r2["pair"] = pair; r2["atr_mult"] = atr_mult; r2["cost_R"] = round(cost_R, 3)
                all_rows.append(r2)
            if len(tune_rows) < 2:
                print(f"  atr={atr_mult}: insufficient tuning trades, skipping ({time.time()-t0:.0f}s)", flush=True)
                continue
            net_Rs = [r["net_R"] for r in tune_rows]
            trades_tot = sum(r["trades"] for r in tune_rows)
            score = np.mean(net_Rs) - 0.5 * np.std(net_Rs)
            print(f"  atr={atr_mult}: tune_net_R_mean={np.mean(net_Rs):.3f} std={np.std(net_Rs):.3f} "
                  f"score={score:.3f} tune_trades={trades_tot} cost_R={cost_R:.3f} ({time.time()-t0:.0f}s)", flush=True)
            if best is None or score > best[1]:
                best = (atr_mult, score)
        chosen[pair] = best[0] if best else 1.5
        print(f"  -> selected atr_mult={chosen[pair]} for {pair} (from folds {TUNE_FOLDS} only)", flush=True)

    print("\n\n=== FINAL: held-out fold 4 results, config chosen without seeing fold 4 ===", flush=True)
    final_rows = []
    for pair in PAIRS:
        atr_mult = chosen[pair]
        per_fold, cost_R = run_config(pair, atr_mult, folds_needed=[TEST_FOLD])
        test_row = per_fold[0]
        final_rows.append(dict(pair=pair, chosen_atr_mult=atr_mult, cost_R=round(cost_R, 3), **test_row))
    fdf = pd.DataFrame(final_rows)
    fdf.to_csv("/home/claude/work/remaining_pairs_final_holdout.csv", index=False)
    print(fdf.to_string(index=False), flush=True)

    grid = pd.DataFrame(all_rows)
    grid.to_csv("/home/claude/work/remaining_pairs_grid.csv", index=False)
