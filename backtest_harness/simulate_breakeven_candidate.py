"""
Candidate E (discovered solution): breakeven-stop exit mechanics.

Evidence: 74/454 (16.3%) of all PRIMARY losses reached >=1.0R of favorable
excursion (as far as their own stop distance, in the right direction)
before fully reversing and hitting the original SL. Every LOSS in this
backtest exits at exactly -1.0R (SL-first same-bar convention), so
converting these specific trades to breakeven would recover +74R against a
-75.2R baseline primary total -- moving net R from -75.2 to ~-1.2.

This tests that claim for real via full bar-by-bar re-simulation (not just
counting), with a sensitivity sweep across trigger thresholds and the same
IS/VAL/OOS split used throughout this review.
"""
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
TIMEOUT_BARS = 200


def simulate_breakeven(df_trades: pd.DataFrame, h1_by_symbol: dict, trigger_r: float) -> pd.DataFrame:
    rows = []
    for _, row in df_trades.iterrows():
        sym = row["symbol"]
        h1 = h1_by_symbol[sym]
        side = row["side"]
        entry, orig_sl, tp = row["entry"], row["sl"], row["tp"]
        r_risk = abs(entry - orig_sl)
        t0 = pd.Timestamp(row["entry_time"])
        try:
            start_idx = h1.index.get_indexer([t0], method="nearest")[0]
        except Exception:
            rows.append(row.to_dict() | {"r_multiple_be": row["r_multiple"], "be_triggered": False})
            continue

        cur_sl = orig_sl
        triggered = False
        outcome, exit_r = row["outcome"], row["r_multiple"]  # fallback to original
        for j in range(start_idx, min(start_idx + TIMEOUT_BARS, len(h1))):
            bar = h1.iloc[j]
            hit_sl = (bar["low"] <= cur_sl) if side == "BUY" else (bar["high"] >= cur_sl)
            if hit_sl:
                pnl = (cur_sl - entry) if side == "BUY" else (entry - cur_sl)
                exit_r = pnl / r_risk if r_risk > 0 else 0.0
                outcome = "BE_SCRATCH" if triggered else "LOSS"
                break
            if not triggered:
                mfe = ((bar["high"] - entry) / r_risk) if side == "BUY" else ((entry - bar["low"]) / r_risk)
                if mfe >= trigger_r:
                    cur_sl = entry
                    triggered = True
            hit_tp = (bar["high"] >= tp) if side == "BUY" else (bar["low"] <= tp)
            if hit_tp:
                exit_r = (tp - entry) / r_risk if side == "BUY" else (entry - tp) / r_risk
                outcome = "WIN"
                break
        rows.append(row.to_dict() | {"r_multiple_be": round(float(exit_r), 3),
                                       "outcome_be": outcome, "be_triggered": triggered})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.read_csv("/home/claude/work/results/loss_mechanism_dataset.csv")
    h1_cache = {}
    for sym, path in CSV_MAP.items():
        h1 = compute_indicators(load_h1(path)).set_index("datetime")
        h1_cache[sym] = h1

    def stats(g, col="r_multiple"):
        w = (g[col] > 0).sum(); l = (g[col] < 0).sum(); c = w + l
        wr = w / c * 100 if c else float("nan")
        gw = g.loc[g[col] > 0, col].sum(); gl = abs(g.loc[g[col] < 0, col].sum())
        pf = gw / gl if gl > 0 else float("nan")
        return pd.Series({"n": len(g), "WR": round(wr, 2), "PF": round(pf, 3),
                           "ExpR": round(g[col].mean(), 3), "NetR": round(g[col].sum(), 2)})

    print("=== Baseline (no breakeven stop) ===")
    for split in ["IS", "VAL", "OOS"]:
        d = df[df.split == split]
        print(f"{split}:", stats(d).to_dict())

    print()
    print("=== Sensitivity sweep: breakeven trigger threshold ===")
    for trig in [0.5, 0.7, 0.8, 1.0, 1.2, 1.5]:
        sim = simulate_breakeven(df, h1_cache, trig)
        print(f"--- trigger={trig}R ---")
        for split in ["IS", "VAL", "OOS"]:
            d = sim[sim.split == split]
            print(f"  {split}:", stats(d, "r_multiple_be").to_dict(),
                  f"| triggered={d.be_triggered.sum()}/{len(d)}")
        sim.to_csv(f"/home/claude/work/results/breakeven_sim_trigger_{trig}.csv", index=False)
