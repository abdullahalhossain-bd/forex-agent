# 📦 Backtest Validation Suite

Institutional-grade backtesting infrastructure for the forex-agent project.

## Quick Start

```bash
# 1. Run baseline (all modules, original config) — produces baseline report
python scripts/backtest_validation/institutional_backtest.py \
  --pairs EURUSD,GBPUSD,USDJPY --timeframe H1 \
  --max-candles 1500 --confidence-threshold 0.45

# 2. Run improved (evidence-based fixes applied) — produces improved report
python scripts/backtest_validation/improved_backtest.py

# 3. Inspect outputs
ls _backtest_validation/
ls _backtest_validation/improved/
```

## What it does

### `institutional_backtest.py` (Baseline)
Replays historical candles through the REAL production analysis pipeline:
- Loads CSV from `data/`
- For each candle: only `df.iloc[0:i+1]` is visible (NO look-ahead)
- Calls production modules: market_structure, support_resistance, liquidity_zones,
  fvg_detector, order_block, smc_engine, market_regime, adx_trend_filter,
  atr_sl_finder, session_analyzer
- Risk filters: session filter, kill switch, drawdown guard, max consec loss
- Executes at NEXT bar OPEN (models real latency)
- Applies spread (1.5p) + slippage (1.5p) + commission ($7/lot) on every fill
- Stop-loss can be skipped on gaps (gap risk modeled)
- Maximum holding period: 50 bars

### `improved_backtest.py` (Improved)
Same engine, but applies 7 evidence-based fixes from the baseline ablation:
1. DISABLES `adx_trend_filter` (ablation proved it costs 26pp of WR)
2. Fixes confidence calc (entropy-weighted, was always 0.95)
3. Skips New York session (0% WR in baseline)
4. Skips TRENDING regime (9.5% WR in baseline)
5. Skips USDJPY pair (0% WR in baseline)
6. Tightens max consec losses to 3 (was 5)
7. Tightens drawdown kill to 15% (was 20%)

## Outputs

Both scripts write to `_backtest_validation/` (and `_backtest_validation/improved/` for the improved run):

- **csv/** — trades.csv (30+ fields per trade), metrics, ablation, calibration, breakdowns
- **json/** — full_report.json, ablation.json, dataset_registry.json, improvement_recommendations.json
- **charts/** — equity curve, drawdown, monthly returns, pair ranking, session, calibration, ablation impact
- **reports/** — INSTITUTIONAL_VALIDATION_REPORT.md, EXECUTIVE_SUMMARY.md, IMPROVED_VALIDATION_REPORT.md

## Methodology — No Look-Ahead, Realistic Costs

- ✅ Each candle only sees `df.iloc[0:i+1]` (no future data leakage)
- ✅ Entry at NEXT bar OPEN (models real latency)
- ✅ Spread (1.5 pips) + Slippage (1.5 pips) + Commission ($7/lot) on EVERY trade
- ✅ Stop-loss can be skipped on gaps (gap risk modeled)
- ✅ Maximum holding period: 50 bars
- ✅ Risk per trade: 1% of account balance
- ✅ Position sizing: ATR-based, capped at 2.0 lots
- ✅ Production analysis modules called per candle (no shortcuts)

## Reproducibility

All results are deterministic:
- Random seeds use `hashlib.md5(pair + bar_idx)` (not Python's randomized hash)
- Same input → same output, every run, every machine
