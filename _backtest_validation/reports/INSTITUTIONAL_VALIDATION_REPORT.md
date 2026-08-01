# 🏛️ Institutional-Grade Backtest Validation Report

**Generated:** 2026-07-27T19:42:33.654422+00:00  
**Pairs:** EURUSD,GBPUSD,USDJPY  
**Timeframe:** H1  
**Max candles per pair:** 1500  
**Confidence threshold:** 0.45  
**Starting balance:** $10,000.00

---

## ⚠️ Methodology — No Look-Ahead, Realistic Costs

- ✅ Each candle only sees `df.iloc[0:i+1]` (no future data)
- ✅ Entry at NEXT bar OPEN (models real latency)
- ✅ Spread (1.5 pips) + Slippage (1.5 pips) + Commission ($7/lot) applied to EVERY trade
- ✅ Stop-loss can be skipped on gaps (gap risk modeled)
- ✅ Maximum holding period: 50 bars
- ✅ Risk per trade: 1% of account balance
- ✅ Position sizing: ATR-based, capped at 2.0 lots
- ✅ Production analysis modules called per candle (no shortcuts)

---

## 📊 Headline Performance

| Metric | Value |
|---|---|
| Starting Balance | $10,000.00 |
| Ending Balance | $8,386.90 |
| Net Profit | $-1,613.10 (-16.13%) |
| Gross Profit | $469.30 |
| Gross Loss | $2,082.40 |
| Total Trades | 22 |
| Winning Trades | 3 |
| Losing Trades | 19 |
| Win Rate | 13.64% |
| Loss Rate | 86.36% |
| Profit Factor | 0.225 |
| Expectancy (USD) | $-73.32 |
| Expectancy (R) | -0.71R |
| Average Win | $156.43 (39.2 pips) |
| Average Loss | $-109.60 (-34.8 pips) |
| Largest Win | $157.80 |
| Largest Loss | $-117.44 |
| Average R:R | 1:1.43 |
| Max Drawdown | 16.13% ($1,613.10) |
| Max Consecutive Wins | 2 |
| Max Consecutive Losses | 10 |
| Avg Trade Duration | 3000 min |
| Sharpe Ratio | -368837.179 |
| Sortino Ratio | -7432127.181 |
| Calmar Ratio | -1.0 |
| Ulcer Index | 8.622 |
| Recovery Factor | -1.0 |

---

## 📅 Monthly Returns

| Month | Return % |
|---|---|
| 2026-05 | -16.13% |

---

## 📆 Yearly Returns

| Year | Return % |
|---|---|
| 2026 | -16.13% |

---

## 💱 Pair Performance

| Pair | Trades | WR% | PnL USD | PF | Avg R |
|---|---|---|---|---|---|
| GBPUSD | 9 | 22.22 | $-458.23 | 0.41 | -0.517 |
| USDJPY | 5 | 0.0 | $-519.39 | 0.0 | -1.068 |
| EURUSD | 8 | 12.5 | $-635.48 | 0.2 | -0.806 |

---

## 🌍 Session Performance

| Session | Trades | WR% | PnL USD | PF |
|---|---|---|---|---|
| London_NY_Overlap | 3 | 33.33 | $-59.05 | 0.73 |
| London | 10 | 20.0 | $-558.89 | 0.36 |
| NewYork | 9 | 0.0 | $-995.16 | 0.0 |

---

## 🔄 Direction Performance

| Direction | Trades | WR% | PnL USD | PF |
|---|---|---|---|---|
| long | 5 | 20.0 | $-286.47 | 0.35 |
| short | 17 | 11.76 | $-1326.63 | 0.19 |

---

## 🧪 Regime Performance

| Regime | Trades | WR% | PnL USD | PF |
|---|---|---|---|---|
| TRENDING | 21 | 9.52 | $-1769.92 | 0.15 |
| BREAKOUT | 1 | 100.0 | $156.82 | 0 |

---

## 📈 Confidence Calibration

| Bin | Trades | Avg Confidence | Actual WR | Calibration Error |
|---|---|---|---|---|
| 0.90-1.01 | 22 | 95.0% | 13.6% | 81.4% |

---

## 🔬 Module Ablation Study

Each row = backtest result with that ONE module disabled. Drop in performance = module's contribution.

| Disabled Module | Trades | WR% | PF | Net Profit | Δ WR | Δ PF | Δ PnL |
|---|---|---|---|---|---|---|---|
| market_structure | 22 | 13.64 | 0.225 | $-1613.10 | +0.0 | +0.00 | +0 |
| support_resistance | 19 | 10.53 | 0.169 | $-1531.29 | -3.1 | -0.06 | +82 |
| liquidity_zones | 22 | 13.64 | 0.225 | $-1613.10 | +0.0 | +0.00 | +0 |
| fvg_detector | 22 | 13.64 | 0.225 | $-1613.10 | +0.0 | +0.00 | +0 |
| order_block | 22 | 13.64 | 0.225 | $-1613.10 | +0.0 | +0.00 | +0 |
| smc_engine | 22 | 13.64 | 0.225 | $-1613.10 | +0.0 | +0.00 | +0 |
| adx_trend_filter | 30 | 40.0 | 1.011 | $21.82 | +26.4 | +0.79 | +1635 |
| session_analyzer | 22 | 13.64 | 0.225 | $-1613.10 | +0.0 | +0.00 | +0 |

---

## 🚦 Deployment Verdict

⚠️ **NOT YET READY** — Issues found:
- Profit Factor too low (0.225 < 1.3)
- Win Rate too low (13.64% < 50%)
- Sharpe too low (-368837.179 < 1.0)

**Recommendation:** Use this report to identify weak modules via ablation, re-tune SL/TP ratios, and re-run. See `improvement_recommendations.json` for next steps.

---

## 📁 Output Files

- `csv/trades.csv` — every trade with 30+ fields
- `csv/metrics_summary.csv` — full metrics table
- `csv/pair_ranking.csv` — pair performance breakdown
- `csv/session_breakdown.csv` — session performance
- `csv/ablation.csv` — module ablation results
- `csv/confidence_calibration.csv` — probability calibration
- `json/full_report.json` — complete machine-readable report
- `json/dataset_registry.json` — data quality registry
- `charts/equity_curve.png` — equity + drawdown
- `charts/monthly_returns.png` — monthly P&L bar chart
- `charts/pair_ranking.png` — pair ranking chart
- `charts/confidence_calibration.png` — calibration plot
- `charts/session_breakdown.png` — session WR + count
- `charts/ablation_impact.png` — module contribution chart
