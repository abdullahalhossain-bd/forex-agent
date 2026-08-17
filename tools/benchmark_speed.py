"""benchmark_speed.py — proves the fast labeler matches the original and measures speedup."""
import sys, time
sys.path.insert(0, "/home/claude/work")
import pandas as pd
from ml.triple_barrier_labels import triple_barrier_labels as orig_labels
from fast_triple_barrier import fast_triple_barrier_labels as fast_labels

PAIRS = ["EURAUD", "GBPCAD", "EURCAD", "GBPSEK", "GBPNOK", "EURNZD"]

print(f"{'pair':8s} {'orig_sec':>9s} {'fast_sec':>9s} {'speedup':>8s} {'agree%':>8s} {'mismatches':>11s}")
for pair in PAIRS:
    df = pd.read_csv(f"/mnt/user-data/uploads/{pair}_M15.csv")
    t0 = time.time()
    orig = orig_labels(df, holding_period=16, take_profit_width=2.0, stop_loss_width=2.0, atr_period=14, use_atr=True)
    t_orig = time.time() - t0

    t0 = time.time()
    fast = fast_labels(df, holding_period=16, take_profit_width=2.0, stop_loss_width=2.0, atr_period=14)
    t_fast = time.time() - t0

    cmp = pd.DataFrame({"o": orig, "f": fast}).dropna()
    agree = (cmp["o"] == cmp["f"]).mean()
    mism = int((cmp["o"] != cmp["f"]).sum())

    print(f"{pair:8s} {t_orig:9.3f} {t_fast:9.4f} {t_orig/t_fast:7.1f}x {agree*100:7.3f}% {mism:11d}")
