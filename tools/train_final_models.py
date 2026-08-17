import sys, joblib, json
sys.path.insert(0, "/home/claude/work")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from backtest_baseline import load, build_features, PROB_THRESHOLD
from ml.triple_barrier_labels import triple_barrier_labels

FINAL_CONFIG = {
    "EURAUD": 2.5, "GBPCAD": 2.5, "EURCAD": 1.5,
    "EURNZD": 3.0, "GBPNOK": 4.0, "GBPSEK": 4.0,
}
HOLDING = 16

report = []
for pair, atr_mult in FINAL_CONFIG.items():
    df = load(pair)
    feats, atr_series = build_features(df)
    labels = triple_barrier_labels(df, holding_period=HOLDING, take_profit_width=atr_mult,
                                    stop_loss_width=atr_mult, atr_period=14, use_atr=True)
    data = feats.copy()
    data["label"] = labels
    data = data.dropna()
    Xcols = feats.columns.tolist()
    n = len(data)

    # label balance sanity check
    dist = data["label"].value_counts(normalize=True).sort_index()

    # overfit check: hold out the LAST 15% (with embargo), train on rest,
    # compare train accuracy to this never-before-touched slice
    split = int(n * 0.85)
    tr_end = split - HOLDING
    Xtr, ytr = data.iloc[:tr_end][Xcols], data.iloc[:tr_end]["label"]
    Xho, yho = data.iloc[split:][Xcols], data.iloc[split:]["label"]

    clf_check = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=50,
                                        class_weight="balanced", random_state=42, n_jobs=-1)
    clf_check.fit(Xtr, ytr)
    train_acc = (clf_check.predict(Xtr) == ytr.values).mean()
    holdout_acc = (clf_check.predict(Xho) == yho.values).mean()

    # final PRODUCTION model: trained on ALL available data (standard practice
    # once walk-forward has validated the methodology/config — see report)
    clf_prod = RandomForestClassifier(n_estimators=400, max_depth=6, min_samples_leaf=50,
                                       class_weight="balanced", random_state=42, n_jobs=-1)
    clf_prod.fit(data[Xcols], data["label"])
    joblib.dump({"model": clf_prod, "feature_columns": Xcols, "atr_mult": atr_mult,
                 "holding_period": HOLDING, "prob_threshold": PROB_THRESHOLD},
                f"/home/claude/work/model_{pair}.joblib")

    importances = pd.Series(clf_prod.feature_importances_, index=Xcols).sort_values(ascending=False)

    report.append(dict(
        pair=pair, atr_mult=atr_mult, n_rows=n,
        label_dist=dict(dist.round(3)),
        train_acc=round(train_acc, 4), holdout_acc=round(holdout_acc, 4),
        overfit_gap=round(train_acc - holdout_acc, 4),
        top5_features=importances.head(5).round(4).to_dict(),
    ))
    print(f"{pair}: n={n} label_dist={dict(dist.round(3))} train_acc={train_acc:.3f} "
          f"holdout_acc={holdout_acc:.3f} gap={train_acc-holdout_acc:.3f}")
    print(f"   top5 features: {importances.head(5).round(3).to_dict()}")

with open("/home/claude/work/model_diagnostics.json", "w") as f:
    json.dump(report, f, indent=2, default=str)
print("\nSaved models + diagnostics.")
