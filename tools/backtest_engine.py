"""
backtest_engine.py — full-system backtest (fast).

Simulates: feature build -> trained model -> confidence filter -> triple-
barrier trade outcome -> fixed-fractional position sizing -> equity curve
-> performance report. Uses fast_triple_barrier_labels() for the P&L
simulation (~200x faster than the original loop-based labeler — see
benchmark_speed.py), so re-running this across all 6 pairs takes seconds,
not minutes, letting you re-check after every data refresh instead of
once a week.

Two data sources, same downstream code:
  --source mt5   pulls fresh bars from a running MT5 terminal
                 (requires the `MetaTrader5` package + Windows/MT5 install
                 — this only runs on the machine with your MT5 terminal,
                 NOT in this sandbox, which is Linux-only. Not executed
                 or testable here for that reason.)
  --source csv   reads the same M15 CSVs you originally uploaded
                 (this is what's actually been run and verified below)

Usage:
    python3 backtest_engine.py --source csv --data-dir /path/to/csvs
    python3 backtest_engine.py --source mt5 --mt5-days 400
"""
from __future__ import annotations
import argparse, sys, time
sys.path.insert(0, "/home/claude/work")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from fast_triple_barrier import fast_triple_barrier_labels
from backtest_baseline import build_features  # same feature set the models were trained on

BARRIER_CONFIG = {
    "EURAUD": {"atr_multiplier": 2.5, "holding_period": 16, "prob_threshold": 0.55},
    "GBPCAD": {"atr_multiplier": 2.5, "holding_period": 16, "prob_threshold": 0.55},
    "EURCAD": {"atr_multiplier": 1.5, "holding_period": 16, "prob_threshold": 0.55},
    "EURNZD": {"atr_multiplier": 3.0, "holding_period": 16, "prob_threshold": 0.55},
    "GBPNOK": {"atr_multiplier": 4.0, "holding_period": 16, "prob_threshold": 0.55},
    "GBPSEK": {"atr_multiplier": 4.0, "holding_period": 16, "prob_threshold": 0.55},
}

RISK_PCT_PER_TRADE = 0.005  # 0.5% of equity per trade — matches your real config.py's
                             # RISK_PER_TRADE=0.005 (confirmed by reading the actual file;
                             # earlier runs of this engine used an assumed 1%, which
                             # overstated drawdown and total_return_pct by roughly 2x)
POINT_SIZE = 1e-5           # verified against all 6 CSVs' quoted decimal precision
N_FOLDS = 5                 # walk-forward folds, same methodology as prior reports
MAX_COST_R = 0.5            # skip a trade if spread alone would eat >50% of its risk
                             # budget (see engine notes: per-bar cost_R can spike to
                             # 1.0-2.2x during low-ATR periods even for pairs whose
                             # AVERAGE cost_R looks fine — this filter catches those
                             # bars specifically instead of relying on an average)


# ── data sources ──────────────────────────────────────────────────────

def load_csv(pair: str, data_dir: str) -> pd.DataFrame:
    df = pd.read_csv(f"{data_dir}/{pair}_M15.csv")
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    return df.sort_values("datetime_utc").reset_index(drop=True)


def load_mt5(pair: str, days: int = 400, timeframe: str = "M15") -> pd.DataFrame:
    """
    Pulls fresh bars from a running MT5 terminal. Requires:
        pip install MetaTrader5
    and a logged-in MT5 terminal on the SAME machine (Windows, or Wine).
    This function is NOT run or tested in this sandbox (Linux, no MT5
    terminal available here) — verify it against your own terminal before
    trusting its output; the CSV path above is what's actually been run.
    """
    import MetaTrader5 as mt5
    from datetime import datetime, timedelta, timezone

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")

    tf_map = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
              "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1}
    tf = tf_map[timeframe]

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    rates = mt5.copy_rates_range(pair, tf, start, end)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise RuntimeError(f"MT5 returned no data for {pair} — check symbol name/visibility in Market Watch")

    df = pd.DataFrame(rates)
    df["datetime_utc"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={"tick_volume": "tick_volume", "spread": "spread", "real_volume": "real_volume"})
    return df[["datetime_utc", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]]


# ── backtest core ─────────────────────────────────────────────────────

def run_pair(pair: str, df: pd.DataFrame, n_folds: int = N_FOLDS) -> dict:
    cfg = BARRIER_CONFIG[pair]
    feats, atr_series = build_features(df)
    labels = fast_triple_barrier_labels(
        df, holding_period=cfg["holding_period"],
        take_profit_width=cfg["atr_multiplier"], stop_loss_width=cfg["atr_multiplier"],
        atr_period=14,
    )
    data = feats.copy()
    data["label"] = labels
    data["spread_price"] = df["spread"].reindex(data.index) * POINT_SIZE
    data["atr"] = atr_series.reindex(data.index)
    data["dt"] = df["datetime_utc"]
    data = data.dropna()
    Xcols = feats.columns.tolist()
    n = len(data)

    fold_edges = np.linspace(int(n * 0.4), n, n_folds + 1).astype(int)
    equity = 1.0
    equity_curve = []
    trade_log = []

    for k in range(n_folds):
        ts, te = fold_edges[k], fold_edges[k + 1]
        tr_end = ts - cfg["holding_period"]
        if tr_end < 500:
            continue
        Xtr, ytr = data.iloc[:tr_end][Xcols], data.iloc[:tr_end]["label"]
        fold = data.iloc[ts:te]

        clf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=50,
                                      class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(Xtr, ytr)
        classes = clf.classes_
        proba = clf.predict_proba(fold[Xcols])
        idx = np.argmax(proba, axis=1)
        pred = classes[idx]
        conf = proba[np.arange(len(proba)), idx]
        pred_full = pred
        sl_width_price_full = cfg["atr_multiplier"] * fold["atr"].to_numpy()
        cost_R_full = fold["spread_price"].to_numpy() / np.maximum(sl_width_price_full, 1e-9)

        take = (pred != 0) & (conf >= cfg["prob_threshold"]) & (cost_R_full <= MAX_COST_R)

        sub = fold[take]
        pred_sub = pred[take]
        cost_R = cost_R_full[take]

        actual = sub["label"].to_numpy()
        category = np.select(
            [actual == pred_sub, actual == -pred_sub, actual == 0],
            ["win", "loss", "timeout"], default="timeout",
        )
        # win: actual barrier hit matches predicted direction -> +1R
        # loss: actual barrier hit is the OPPOSITE direction -> -1R
        # timeout: actual==0 (neither barrier hit in the holding window) ->
        # ~breakeven, NOT a loss. (BUG CAUGHT IN TESTING: an earlier version
        # of this script scored every timeout as -1R via `pred != label`,
        # which is wrong — a timed-out trade closes near entry, not at the
        # stop — and silently wiped out GBPNOK/GBPSEK's equity curve to
        # -100% in testing purely from this bug, not a real result. Fixed
        # and re-verified before this script was shown.)
        outcome_R = np.select(
            [category == "win", category == "loss", category == "timeout"],
            [1.0, -1.0, 0.0],
            default=0.0,
        ) - cost_R

        for r, dt, cat in zip(outcome_R, sub["dt"], category):
            risk_usd = equity * RISK_PCT_PER_TRADE
            pnl = risk_usd * r
            equity += pnl
            equity_curve.append((dt, equity))
            trade_log.append((dt, r, equity, cat))

    if not equity_curve:
        return dict(pair=pair, trades=0)

    ec = pd.DataFrame(equity_curve, columns=["dt", "equity"]).set_index("dt")
    rs = np.array([t[1] for t in trade_log])
    cats = np.array([t[3] for t in trade_log])
    wins = int((cats == "win").sum())
    losses = int((cats == "loss").sum())
    timeouts = int((cats == "timeout").sum())
    total_trades = len(rs)
    winrate = wins / (wins + losses) if (wins + losses) else np.nan
    gross_win = rs[rs > 0].sum()
    gross_loss = -rs[rs < 0].sum()
    profit_factor = gross_win / gross_loss if gross_loss > 0 else np.inf

    running_max = ec["equity"].cummax()
    drawdown = (ec["equity"] - running_max) / running_max
    max_dd = drawdown.min()

    days = (ec.index[-1] - ec.index[0]).total_seconds() / 86400
    trades_per_day = total_trades / days if days > 0 else np.nan
    total_return = ec["equity"].iloc[-1] - 1.0
    total_R = rs.sum()  # compounding-independent: sum of R-multiples, unaffected by position-sizing assumptions

    r_mean, r_std = rs.mean(), rs.std()
    sharpe_like = (r_mean / r_std * np.sqrt(252 * trades_per_day)) if r_std > 0 and trades_per_day > 0 else np.nan

    return dict(pair=pair, trades=total_trades, wins=wins, losses=losses, timeouts=timeouts,
                winrate=round(winrate, 4), total_R=round(total_R, 1),
                profit_factor=round(profit_factor, 3), max_drawdown_pct=round(max_dd * 100, 2),
                total_return_pct=round(total_return * 100, 2), trades_per_day=round(trades_per_day, 2),
                sharpe_like=round(sharpe_like, 2) if sharpe_like == sharpe_like else np.nan,
                days_tested=round(days, 1)), ec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["csv", "mt5"], default="csv")
    ap.add_argument("--data-dir", default="/mnt/user-data/uploads")
    ap.add_argument("--mt5-days", type=int, default=400)
    args = ap.parse_args()

    results = []
    curves = {}
    t0 = time.time()
    for pair in BARRIER_CONFIG:
        if args.source == "csv":
            df = load_csv(pair, args.data_dir)
        else:
            df = load_mt5(pair, days=args.mt5_days)
        out = run_pair(pair, df)
        if isinstance(out, tuple):
            r, ec = out
            results.append(r)
            curves[pair] = ec
        else:
            results.append(out)
        print(results[-1])
    print(f"\nTotal wall time for all 6 pairs: {time.time()-t0:.1f}s")

    rdf = pd.DataFrame(results)
    rdf.to_csv("/home/claude/work/backtest_engine_results.csv", index=False)
    print(rdf.to_string(index=False))
    return rdf, curves


if __name__ == "__main__":
    main()
