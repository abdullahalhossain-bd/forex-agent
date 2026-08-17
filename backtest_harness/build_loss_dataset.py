import sys
sys.path.insert(0, ".")
import pandas as pd
import numpy as np
from run_backtest_primary import load_h1, compute_indicators

CSV_MAP = {
    "EURAUD": "/home/claude/work/EURAUD_M15.csv",
    "EURCAD": "/home/claude/work/EURCAD_M15.csv",
    "GBPCAD": "/home/claude/work/GBPCAD_M15.csv",
}

df = pd.read_csv("/home/claude/work/results/lossaudit_ALL_trades.csv")
prim = df[df.source == "PRIMARY"].copy()
prim["entry_time"] = pd.to_datetime(prim["entry_time"])
prim["exit_time"] = pd.to_datetime(prim["exit_time"])

# hour / day-of-week -- trivial, no engine needed
prim["hour"] = prim["entry_time"].dt.hour
prim["day_of_week"] = prim["entry_time"].dt.day_name()

mfe_list, mae_list, atr_pctl_list, range_pctl_list = [], [], [], []

for sym in CSV_MAP:
    h1 = compute_indicators(load_h1(CSV_MAP[sym]))
    h1 = h1.set_index("datetime")
    h1["range"] = h1["high"] - h1["low"]
    atr_series = h1["atr"].dropna()
    range_series = h1["range"].dropna()

    sub = prim[prim.symbol == sym]
    for idx, row in sub.iterrows():
        side = row["side"]
        entry, sl, tp = row["entry"], row["sl"], row["tp"]
        r_risk = abs(entry - sl)
        t0, t1 = row["entry_time"], row["exit_time"]
        try:
            window = h1.loc[t0:t1]
        except Exception:
            window = pd.DataFrame()
        if len(window) == 0 or r_risk == 0:
            mfe_list.append(np.nan); mae_list.append(np.nan)
        else:
            if side == "BUY":
                mfe = (window["high"].max() - entry) / r_risk
                mae = (entry - window["low"].min()) / r_risk
            else:
                mfe = (entry - window["low"].min()) / r_risk
                mae = (window["high"].max() - entry) / r_risk
            mfe_list.append(round(float(mfe), 3))
            mae_list.append(round(float(mae), 3))

        # ATR / range percentile at entry, vs trailing 500-bar history
        try:
            atr_at_entry = h1.loc[:t0, "atr"].iloc[-1]
            hist = h1.loc[:t0, "atr"].dropna().iloc[-500:]
            atr_pctl = (hist < atr_at_entry).mean() * 100 if len(hist) > 20 else np.nan
        except Exception:
            atr_pctl = np.nan
        try:
            range_at_entry = h1.loc[:t0, "range"].iloc[-1]
            hist_r = h1.loc[:t0, "range"].dropna().iloc[-500:]
            range_pctl = (hist_r < range_at_entry).mean() * 100 if len(hist_r) > 20 else np.nan
        except Exception:
            range_pctl = np.nan
        atr_pctl_list.append(round(float(atr_pctl), 1) if pd.notna(atr_pctl) else np.nan)
        range_pctl_list.append(round(float(range_pctl), 1) if pd.notna(range_pctl) else np.nan)

prim["mfe_r"] = mfe_list
prim["mae_r"] = mae_list
prim["atr_percentile"] = atr_pctl_list
prim["range_percentile"] = range_pctl_list

# chronological IS/VAL/OOS split (same as before)
t0, t1 = prim.entry_time.min(), prim.entry_time.max()
span = t1 - t0
is_end = t0 + span * 0.6
val_end = t0 + span * 0.8
prim["split"] = pd.cut(
    prim.entry_time,
    bins=[t0 - pd.Timedelta(days=1), is_end, val_end, t1 + pd.Timedelta(days=1)],
    labels=["IS", "VAL", "OOS"],
)

prim.to_csv("/home/claude/work/results/loss_mechanism_dataset.csv", index=False)
print("saved", len(prim), "rows")
print(prim[["mfe_r", "mae_r", "atr_percentile", "range_percentile"]].describe())
