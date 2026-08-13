#!/usr/bin/env python3
"""
Per-pair config sweep — find the best config for EACH pair individually.

For each pair, try multiple configs (conservative, moderate, aggressive,
very aggressive) and report which one gives the best profit factor.
This produces the data needed to build pair_profiles.py.
"""
from __future__ import annotations

import os
import sys
import json
import time
import warnings
from pathlib import Path
from itertools import product

import pandas as pd

PROJECT_ROOT = Path("/home/z/my-project/repos/forex-agent")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, "/home/z/my-project/scripts")

os.environ["BACKTEST_MODE"] = "1"
os.environ["SIMULATION_MODE"] = "true"

import logging
logging.basicConfig(level=logging.WARNING)
for n in ["urllib3","httpx","groq","google_genai","chromadb",
          "sentence_transformers","matplotlib","PIL","asyncio",
          "indicators","indicators_ext","data.fetcher","data_orchestrator",
          "data.validator","data.indicator_registry","data.live_feed"]:
    logging.getLogger(n).setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from fast_backtest import load_csv, compute_indicators, backtest_pair

# Configs to sweep — each pair will try ALL of these
CONFIGS = {
    "ultra_strict":   {"min_conf": 90, "min_factors": 6, "min_rr": 2.0, "session": 1, "stop_atr": 1.8, "target_atr": 3.5},
    "strict":         {"min_conf": 85, "min_factors": 5, "min_rr": 2.0, "session": 1, "stop_atr": 1.8, "target_atr": 3.5},
    "moderate_strict":{"min_conf": 80, "min_factors": 4, "min_rr": 1.5, "session": 1, "stop_atr": 1.5, "target_atr": 3.0},
    "moderate":       {"min_conf": 70, "min_factors": 4, "min_rr": 1.5, "session": 1, "stop_atr": 1.5, "target_atr": 3.0},
    "moderate_loose": {"min_conf": 70, "min_factors": 3, "min_rr": 1.5, "session": 1, "stop_atr": 1.5, "target_atr": 2.5},
    "loose":          {"min_conf": 60, "min_factors": 3, "min_rr": 1.5, "session": 1, "stop_atr": 1.5, "target_atr": 2.5},
    "no_session":     {"min_conf": 85, "min_factors": 5, "min_rr": 2.0, "session": 0, "stop_atr": 1.8, "target_atr": 3.5},
    "tight_tp":       {"min_conf": 85, "min_factors": 5, "min_rr": 1.5, "session": 1, "stop_atr": 1.5, "target_atr": 2.5},
    "wide_sl":        {"min_conf": 85, "min_factors": 5, "min_rr": 2.0, "session": 1, "stop_atr": 2.5, "target_atr": 4.5},
    "asian_only":     {"min_conf": 70, "min_factors": 3, "min_rr": 1.5, "session": 0, "stop_atr": 1.5, "target_atr": 3.0},
}

PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"]

def main():
    # Cache indicator-computed dfs
    pair_dfs = {}
    for pair in PAIRS:
        df = load_csv(pair, "H1")
        if df is None:
            continue
        df = compute_indicators(df)
        pair_dfs[pair] = df

    results = []
    for pair, df in pair_dfs.items():
        print(f"\n{'='*70}")
        print(f"  {pair} — sweeping {len(CONFIGS)} configs")
        print(f"{'='*70}")
        pair_results = []
        for config_name, cfg in CONFIGS.items():
            # Set env vars for this config
            os.environ["BT_MIN_CONFIDENCE"] = str(cfg["min_conf"])
            os.environ["BT_MIN_FACTORS"] = str(cfg["min_factors"])
            os.environ["BT_MIN_RR"] = str(cfg["min_rr"])
            os.environ["BT_SESSION_FILTER"] = str(cfg["session"])
            os.environ["BT_STOP_ATR"] = str(cfg["stop_atr"])
            os.environ["BT_TARGET_ATR"] = str(cfg["target_atr"])
            os.environ["BT_SIMULATE_LLM"] = "0"

            t0 = time.time()
            try:
                res = backtest_pair(pair, "H1", df, warmup=250)
                res["config"] = config_name
                res["config_detail"] = cfg
                res["duration_sec"] = round(time.time() - t0, 1)
                pair_results.append(res)
                wr = res["winrate"]
                pf = res["profit_factor"]
                n = res["trades"]
                pnl = res["net_pnl"]
                tag = "✓" if pf >= 1.0 else "✗"
                print(f"  {tag} {config_name:18s} | WR={wr:5.1f}% | PF={pf:5.2f} | N={n:4d} | PnL=${pnl:+8.2f}")
            except Exception as e:
                print(f"  ✗ {config_name:18s} | CRASHED: {e}")

        # Sort by PF descending, take best
        pair_results.sort(key=lambda r: r["profit_factor"], reverse=True)
        best = pair_results[0] if pair_results else None
        if best:
            print(f"\n  BEST for {pair}: {best['config']} → WR={best['winrate']:.1f}% PF={best['profit_factor']:.2f} PnL=${best['net_pnl']:+.2f}")
            results.append({
                "pair": pair,
                "best_config": best["config"],
                "best_config_detail": best["config_detail"],
                "best_winrate": best["winrate"],
                "best_pf": best["profit_factor"],
                "best_pnl": best["net_pnl"],
                "best_trades": best["trades"],
                "all_results": [
                    {
                        "config": r["config"],
                        "winrate": r["winrate"],
                        "pf": r["profit_factor"],
                        "trades": r["trades"],
                        "pnl": r["net_pnl"],
                    }
                    for r in pair_results
                ],
            })

    # Save
    out_path = "/home/z/my-project/download/per_pair_config_sweep.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Summary table
    print(f"\n\n{'='*90}")
    print(f"  PER-PAIR BEST CONFIG SUMMARY")
    print(f"{'='*90}")
    print(f"  {'Pair':8s} {'Best Config':18s} {'WR':7s} {'PF':6s} {'N':5s} {'PnL':10s}")
    for r in results:
        print(f"  {r['pair']:8s} {r['best_config']:18s} "
              f"{r['best_winrate']:5.1f}% {r['best_pf']:5.2f} "
              f"{r['best_trades']:4d} ${r['best_pnl']:+9.2f}")
    print(f"\n  Saved: {out_path}")

if __name__ == "__main__":
    main()
