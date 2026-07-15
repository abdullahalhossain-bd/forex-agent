#!/usr/bin/env python3
"""Parameter sweep runner for synthetic backtests.
Generates synthetic data and calls run_backtest.run_backtest over a parameter grid.
Saves summary results to `backtest/parameter_sweep_results.csv` and individual trade CSVs.
"""
import csv
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from run_backtest import generate_synthetic_data, run_backtest

OUT_DIR = PROJECT_ROOT / "backtest"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    pair = "EURUSD"
    tf = "H1"
    bars = 500
    seeds = [42, 123]
    risks = [0.01, 0.02, 0.03]
    max_trades_list = [1, 2, 3]
    max_hold_list = [50, 100, 200]

    summary_path = OUT_DIR / "parameter_sweep_results.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp","seed","risk","max_trades","max_hold","total_trades","win_rate","profit_factor","total_pnl_usd","max_drawdown_pct"])

        for seed in seeds:
            df = generate_synthetic_data(pair, bars=bars, seed=seed)
            for risk in risks:
                for max_trades in max_trades_list:
                    for max_hold in max_hold_list:
                        ts = datetime.utcnow().isoformat()
                        print(f"Running seed={seed} risk={risk} max_trades={max_trades} max_hold={max_hold}")
                        res = run_backtest(symbol=pair, df=df, timeframe=tf, starting_balance=10000.0, risk_pct=risk, max_open_trades=max_trades, max_hold_bars=max_hold, verbose=False)
                        metrics = res.get("metrics")
                        total_trades = metrics.total_trades if metrics else 0
                        win_rate = getattr(metrics, "win_rate", None)
                        pf = getattr(metrics, "profit_factor", None)
                        pnl = getattr(metrics, "total_pnl_usd", None)
                        dd = getattr(metrics, "max_drawdown_pct", None)

                        writer.writerow([ts, seed, risk, max_trades, max_hold, total_trades, win_rate, pf, pnl, dd])
                        fh.flush()

                        # Save trades for this run if any
                        trades = res.get("trades", [])
                        if trades:
                            csv_path = OUT_DIR / f"sweep_trades_seed{seed}_r{int(risk*100)}_mt{max_trades}_mh{max_hold}.csv"
                            import pandas as pd
                            trades_df = pd.DataFrame([t.to_dict() for t in trades])
                            trades_df.to_csv(csv_path, index=False)

    print(f"Parameter sweep complete. Summary: {summary_path}")

if __name__ == "__main__":
    main()
