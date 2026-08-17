# Deliverable 4 — Historical Data Specification

**Project:** Forex AI Autonomous Trading System
**Audit date:** 2026-08-17

This document specifies exactly what historical data must be downloaded to reproduce the live trading system in backtest / forward simulation mode.

---

## 1. Scope

### 1.1 Symbols

| Symbol class | Symbols | Required? | Notes |
|--------------|---------|-----------|-------|
| FX majors (existing) | AUDUSD, EURUSD, GBPUSD, NZDUSD, USDCAD, USDCHF, USDJPY | ✅ REQUIRED | Currently in CSVs |
| FX crosses (for CorrelationEngine) | EURGBP, EURJPY, EURCHF, EURAUD, EURCAD, EURNZD, GBPAUD, GBPCAD, GBPCHF, GBPJPY, GBPNZD, AUDCAD, AUDCHF, AUDJPY, AUDNZD, NZDCAD, NZDCHF, NZDJPY, CADCHF, CADJPY, CHFJPY | ⚠ RECOMMENDED | 21 of the 28 cross pairs the engine fetches live; downloading these enables full correlation reproduction |
| Metals | XAUUSD (gold), XAGUSD (silver) | ⚠ RECOMMENDED if traded | XAUUSD commonly traded; XAGUSD optional |
| Macro externals | DXY (US Dollar Index), Gold (GC=F), Oil (CL=F), US10Y (^TNX), S&P500 (^GSPC), VIX (^VIX) | ⚠ RECOMMENDED | For IntermarketEngine historical reproduction |
| CFTC COT | COT weekly reports per asset | ⚠ OPTIONAL | Public data from CFTC.gov; backtest uses synthetic proxy if missing |

### 1.2 Timeframes

| TF | Required? | Live usage | Notes |
|----|-----------|------------|-------|
| M1 | Optional | Only for ultra-fine entry timing (not in current config) | Skip unless ML/RL training needs it |
| M5 | ⚠ RECOMMENDED | In `MTFAnalyzer.analyze()` MTF chain (`mtf_analyzer.py:108` uses `["4h","1h","15m","5m"]`) | Currently MISSING from existing CSVs |
| M15 | ✅ REQUIRED | Primary signal-generation TF (config.DEFAULT_TIMEFRAME) | Currently in CSVs |
| M30 | Optional | Not in current MTF chain | Skip |
| H1 | ✅ REQUIRED | In MTF chain, also for H4 fallback resampling | Currently in CSVs |
| H4 | ✅ REQUIRED | HTF bias, SMC engine self-fetches | Currently in CSVs |
| D1 | ⚠ RECOMMENDED | In `MarketAgent` MTF chain (`["1d","4h","1h","15m"]`) | Currently MISSING from existing CSVs |
| W1 | Optional | Not in active use | Skip |
| MN1 | Optional | Not in active use | Skip |

### 1.3 Date range

**Minimum:** 2024-01-01 → 2026-08-17 (≈2.5 years)
- Provides 2 years of warmup + 6 months of recent history
- Allows for ML/RL training with sufficient samples

**Recommended:** 2022-01-01 → 2026-08-17 (≈4.5 years)
- Covers multiple market regimes (bull/bear/ranging)
- Sufficient for ML walk-forward validation

**Note on warmup:** The downloader MUST extend `--start` backward by at least `MAX_REQUIRED_LOOKBACK` bars on the primary TF to ensure indicators are non-NaN at the start of the target backtest window.

For M15 primary TF:
- 1000 bars warmup = ~10 calendar days (excluding weekends)
- 500 bars NadarayaWatson window = ~5 days
- 1 week of intraday bars for PDH/PWL/Asian range = ~5 days

So if target backtest starts `2026-01-01`, downloader should fetch from at least `2025-12-15` (15 days warmup). For safety: `2025-11-01` (60 days warmup) for the first run.

---

## 2. CSV Schema Specification

### 2.1 Required columns (strict order)

```
datetime_utc, open, high, low, close, tick_volume, spread, real_volume
```

| Column | Type | Format | Units | Notes |
|--------|------|--------|-------|-------|
| `datetime_utc` | str | ISO 8601 with `+00:00` offset (e.g. `2025-07-25 07:00:00+00:00`) | UTC bar open time | MUST be tz-aware UTC; loader uses `pd.to_datetime(col, utc=True)` |
| `open` | float | decimal (e.g. `1.17473`) | Price | 5-digit FX, 3-digit JPY |
| `high` | float | decimal | Price | Must be ≥ `max(open, close)` |
| `low` | float | decimal | Price | Must be ≤ `min(open, close)` |
| `close` | float | decimal | Price | |
| `tick_volume` | int | integer (e.g. `3210`) | Tick count in bar | NOT real consolidated volume — forex has none |
| `spread` | int | integer (e.g. `8`) | Points (not pips!) | Conversion: pips = points / 10 for 5-digit FX, / 10 for 3-digit JPY, / 1 for XAU/indices |
| `real_volume` | int | integer | Lots | Always 0 for FX (kept for schema completeness) |

### 2.2 Optional columns (recommended)

```
bid, ask
```

| Column | Type | Format | Units | Notes |
|--------|------|--------|-------|-------|
| `bid` | float | decimal | Price | Bid at bar close — for realistic fill price (SELL) |
| `ask` | float | decimal | Price | Ask at bar close — for realistic fill price (BUY) |

If `bid` and `ask` are present, `BrokerSimulator` should be updated to fill BUY at `ask` and SELL at `bid` (instead of at `close`).

### 2.3 File layout

**Preferred (nested):**
```
data/historical/
├── manifest.json
├── EURUSD/
│   ├── M5.csv
│   ├── M15.csv
│   ├── H1.csv
│   ├── H4.csv
│   └── D1.csv
├── GBPUSD/
│   └── ...
└── external/
    ├── DXY_D1.csv
    ├── GOLD_D1.csv
    ├── OIL_D1.csv
    ├── US10Y_D1.csv
    ├── SP500_D1.csv
    ├── VIX_D1.csv
    └── cot/
        ├── EURUSD_cot.csv
        └── ...
```

**Legacy flat (backward-compatible):**
```
data/
├── EURUSD_M15.csv
├── EURUSD_H1.csv
├── EURUSD_H4.csv
└── ...
```

The `HistoricalCSVProvider._find_csv()` (csv_data_provider.py:79) checks nested first, then flat. Both layouts work.

### 2.4 Validation rules

A downloaded CSV is considered valid if:

| Rule | Check | Severity |
|------|-------|----------|
| V1 | Timestamps sorted ascending | ERROR (reject) |
| V2 | No duplicate timestamps | ERROR (reject) |
| V3 | OHLC sanity: `high >= max(open, close)` AND `low <= min(open, close)` AND `high >= low` | ERROR (reject) |
| V4 | No NaN in OHLCV columns | ERROR (reject) |
| V5 | No infinite values | ERROR (reject) |
| V6 | No negative prices | ERROR (reject) |
| V7 | `tick_volume >= 0` | ERROR (reject) |
| V8 | `spread >= 0` | WARN (accept but flag) |
| V9 | Timestamps tz-aware UTC | ERROR (reject) |
| V10 | No gaps > 1.5 × TF interval (excluding forex weekend) | WARN (accept but flag) |
| V11 | Zero-spread percentage < 50% | WARN (accept but flag) |
| V12 | At least 500 rows on primary TF (warmup) | WARN (accept but flag) |
| V13 | At least 200 rows on each HTF | WARN (accept but flag) |

Validation report saved to `data/historical/{SYMBOL}/_validation_{TF}.json` per file.

---

## 3. Lookback Requirements (Derived MAX_REQUIRED_LOOKBACK)

Based on P1-C §4 audit, the effective minimum lookback per TF:

| TF | Min bars (live parity floor) | Calendar equivalent | Notes |
|----|-------------------------------|----------------------|-------|
| M1 | 500 | ~5 days | NadarayaWatson window |
| M5 | 200 | ~7 days | MTFAnalyzer limit |
| M15 | 1000 | ~14 days | MarketAgent fetch + NadarayaWatson + 1 week PDH/PWL |
| M30 | 200 | ~4 days | MTFAnalyzer equivalent |
| H1 | 300 | ~13 days | MarketAgent + 200 MTF + Ichimoku |
| H4 | 200 | ~33 days | MTFAnalyzer + SMCEngine + 300 OB decay |
| D1 | 200 | ~200 days | CurveMTF swing-style + 400 OB decay |
| W1 | 50 | ~50 weeks | CurveMTF position style |

**Plus:** at least **1 full trading week** of intraday bars on the primary TF for PDH/PWL/Asian range computation.

### Recommended download window per TF

| TF | Recommended bars per symbol | Recommended calendar window |
|----|----------------------------|-----------------------------|
| M15 | 50,000 (≈1.5 years) | 2025-01-01 → 2026-08-17 |
| H1 | 15,000 (≈2 years) | 2024-08-01 → 2026-08-17 |
| H4 | 4,000 (≈2 years) | 2024-08-01 → 2026-08-17 |
| D1 | 500 (≈2 years) | 2024-08-01 → 2026-08-17 |
| M5 | 100,000 (≈1 year) | 2025-08-01 → 2026-08-17 |

---

## 4. Final Data Specification Table

| # | Symbol | TF | Date Range | Bars (target) | Fields | Source | Status |
|---|--------|-----|------------|----------------|--------|--------|--------|
| 1 | EURUSD | M15 | 2025-07-25 → 2026-07-24 | 24,780 | OHLC + tick_volume + spread | MT5 | ✅ EXISTS (re-download for non-zero spreads) |
| 2 | EURUSD | H1 | 2025-07-25 → 2026-07-24 | 6,197 | OHLC + tick_volume + spread | MT5 | ✅ EXISTS (re-download) |
| 3 | EURUSD | H4 | 2025-07-25 → 2026-07-24 | 1,551 | OHLC + tick_volume + spread | MT5 | ✅ EXISTS (70% spread=0 — re-download) |
| 4 | EURUSD | M5 | 2025-07-25 → 2026-07-24 | ~74,000 | OHLC + tick_volume + spread | MT5 | ❌ MISSING — DOWNLOAD |
| 5 | EURUSD | D1 | 2024-08-01 → 2026-08-17 | ~500 | OHLC + tick_volume + spread | MT5 | ❌ MISSING — DOWNLOAD |
| 6-10 | GBPUSD | M15, M5, H1, H4, D1 | same | similar | same | MT5 | 3 exist (re-download), 2 missing |
| 11-15 | USDJPY | same | same | similar | same | MT5 | 3 exist (re-download), 2 missing |
| 16-20 | USDCHF | same | same | similar | same | MT5 | 3 exist (re-download), 2 missing |
| 21-25 | USDCAD | same | same | similar | same | MT5 | 3 exist (re-download), 2 missing |
| 26-30 | AUDUSD | same | same | similar | same | MT5 | 3 exist (re-download), 2 missing |
| 31-35 | NZDUSD | same | same | similar | same | MT5 | 3 exist (re-download), 2 missing |
| 36-40 | XAUUSD | M15, M5, H1, H4, D1 | same | similar | same | MT5 | ❌ ALL MISSING — DOWNLOAD |
| 41-44 | DXY, GOLD, OIL, US10Y | D1 | 2024-08-01 → 2026-08-17 | ~500 each | OHLCV | yfinance | ❌ MISSING — DOWNLOAD |
| 45-46 | SP500, VIX | D1 | same | ~500 each | OHLCV | yfinance | ❌ MISSING — DOWNLOAD |
| 47-67 | 21 cross pairs (EURGBP, EURJPY, ...) | M15, H1, H4 | 2025-07-25 → 2026-07-24 | ~32,000 each | OHLC + tick_volume | MT5 | ❌ MISSING — RECOMMENDED |
| 68 | COT history (EURUSD, GBPUSD, USDJPY weekly) | W1 | 2024-01-01 → 2026-08-17 | ~130 each | net_position | CFTC.gov | ❌ MISSING — OPTIONAL |

---

## 5. FINAL DATA REQUIREMENTS TABLE

| Data | Required? | Why | Timeframe | Source | CSV Field | Status |
|------|-----------|-----|-----------|--------|-----------|--------|
| `datetime_utc` (tz-aware UTC) | ✅ REQUIRED | Session detection, Asian range, PDH/PWL, VWAP, kill zones, MTF alignment | All TFs | MT5 `r["time"]` (epoch sec, broker-tz corrected) | `datetime_utc` | ✅ Present in all 21 CSVs |
| `open` | ✅ REQUIRED | OHLC sanity, candle patterns, candle geometry | All TFs | MT5 `r["open"]` | `open` | ✅ Present |
| `high` | ✅ REQUIRED | ATR, ADX, Stochastic, CCI, Bollinger, Ichimoku, Donchian, market structure, S/R, liquidity sweeps | All TFs | MT5 `r["high"]` | `high` | ✅ Present |
| `low` | ✅ REQUIRED | Same as `high` | All TFs | MT5 `r["low"]` | `low` | ✅ Present |
| `close` | ✅ REQUIRED | EMA, SMA, RSI, MACD, Bollinger, ATR (close-to-close), SupportResistance, Fibonacci, Ichimoku | All TFs | MT5 `r["close"]` | `close` | ✅ Present |
| `tick_volume` | ✅ REQUIRED | VWAP, OBV, MFI, CMF, A/D Line, VWMA, Volume RSI, VolumeProfile, VolumeConfirmation, VW-MACD, CandlestickEngine volume_z | All TFs | MT5 `r["tick_volume"]` | `tick_volume` | ✅ Present (0% zeros) |
| `spread` (per-bar, points) | ✅ REQUIRED | SignalEngine spread filter, cost-aware EV gate, RiskEngine correlation adjustment | All TFs | MT5 `r["spread"]` (int points) | `spread` | ⚠ Present but 15-70% zeros — RE-DOWNLOAD |
| `bid` | ⚠ RECOMMENDED | Realistic fill price for BUY orders (currently fills at `close`) | All TFs | MT5 `tick.bid` at bar close | `bid` | ❌ MISSING — ADD COLUMN |
| `ask` | ⚠ RECOMMENDED | Realistic fill price for SELL orders | All TFs | MT5 `tick.ask` at bar close | `ask` | ❌ MISSING — ADD COLUMN |
| `real_volume` | ❌ OPTIONAL | Always 0 for FX; pipeline drops it at `data/fetcher.py:966` | All TFs | MT5 `r["real_volume"]` (always 0 for FX) | `real_volume` | ⚠ Present but always 0; unused |
| `time_msc` (ms precision) | ❌ OPTIONAL | Not used in pipeline | n/a | MT5 `tick.time_msc` | n/a | ❌ Not stored (acceptable) |
| Tick stream (last 60s) | ❌ OPTIONAL | MicrostructureEngine (gated OFF in backtest) | tick | MT5 `copy_ticks_range` | n/a | ❌ Live-only (acceptable) |
| Market depth (L2 order book) | ❌ OPTIONAL | Not used in pipeline | n/a | MT5 `market_book_add` (not called) | n/a | ❌ Live-only (acceptable) |
| DXY history | ⚠ RECOMMENDED | IntermarketEngine risk-on/off + macro bias | D1 | yfinance `DX-Y.NYB` | `data/external/DXY_D1.csv` | ❌ MISSING — DOWNLOAD |
| Gold history | ⚠ RECOMMENDED | IntermarketEngine | D1 | yfinance `GC=F` | `data/external/GOLD_D1.csv` | ❌ MISSING — DOWNLOAD |
| Oil history | ⚠ RECOMMENDED | IntermarketEngine | D1 | yfinance `CL=F` | `data/external/OIL_D1.csv` | ❌ MISSING — DOWNLOAD |
| US10Y history | ⚠ RECOMMENDED | IntermarketEngine | D1 | yfinance `^TNX` | `data/external/US10Y_D1.csv` | ❌ MISSING — DOWNLOAD |
| SP500 history | ⚠ RECOMMENDED | IntermarketEngine | D1 | yfinance `^GSPC` | `data/external/SP500_D1.csv` | ❌ MISSING — DOWNLOAD |
| VIX history | ⚠ RECOMMENDED | IntermarketEngine | D1 | yfinance `^VIX` | `data/external/VIX_D1.csv` | ❌ MISSING — DOWNLOAD |
| 21 cross-pair history | ⚠ RECOMMENDED | CorrelationEngine full 28-pair matrix | M15, H1, H4 | MT5 | `data/historical/{CROSS}/{TF}.csv` | ❌ MISSING — DOWNLOAD |
| CFTC COT history | ❌ OPTIONAL | InstitutionalFlowEngine (gated OFF in backtest, uses synthetic proxy) | W1 | CFTC.gov | `data/external/cot/{PAIR}_cot.csv` | ❌ MISSING — OPTIONAL |
| News history | ⚠ RECOMMENDED | NewsIntelligence context for LLM | event | Forex Factory archive | `data/economic_calendar_history.json` | ❌ MISSING — DOWNLOAD |
| XAUUSD CSVs | ⚠ RECOMMENDED (if traded) | Live config typically includes gold | M15, M5, H1, H4, D1 | MT5 | `data/historical/XAUUSD/{TF}.csv` | ❌ MISSING — DOWNLOAD |

---

## 6. Impossible / Broker-Dependent Data

These cannot be reliably reproduced from historical sources:

| Data | Why impossible | Mitigation |
|------|----------------|------------|
| Live tick stream (bid/ask/last/flags per tick, last 60s) | Historical tick databases are broker-specific; the live MT5 terminal only returns recent ticks via `copy_ticks_range` | Accept that `MicrostructureEngine` is gated OFF in backtest |
| Live market depth (Level-2 order book) | `mt5.market_book_add` not used; no historical L2 stored | Accept — no consumer in pipeline |
| Live news events (real-time) | Real-time event data; historical news databases are paid (Bloomberg, Refinitiv, Dow Jones) | Use Forex Factory archive for backtest window |
| Live retail sentiment (Myfxbook) | Live-only API; historical snapshot service is paid | Accept — `SentimentEngine` uses cached/synthetic in backtest |
| Broker-specific SL/TP execution priority | Some brokers honor SL before TP on a bar that hits both; others honor TP first | `BrokerSimulator` defaults to SL priority (conservative) — document this assumption |
| Broker-specific swap rates (overnight financing) | Not modeled in `BrokerSimulator` at all; live MT5 applies swap automatically | Add swap model to `BrokerSimulator` (optional) |
| Broker-specific commission per lot | `BrokerSimulator` uses a configurable `_commission_per_lot` parameter; live broker may have different tiers | Match broker's actual commission tier in `BrokerSimulator` config |
| Requotes / partial fills / order rejections | Live MT5 can reject orders; `BrokerSimulator` always fills | Accept — backtest is optimistic |
| Latency / slippage on order placement | Live order placement has 100-500ms latency between `mt5.order_send` and fill; backtest assumes instant fill at next-bar open | Add configurable latency simulation in `BrokerSimulator` (optional) |
| Bid/Ask spread at fill time | Even with `bid`/`ask` columns at bar close, the spread at the moment of order placement (between bar closes) may differ | Accept approximation — bar-close spread is the best available proxy |

---

## 7. Approximation Flags

Per Phase 14 spec, where live execution-specific information is unavailable, an explicit approximation flag must be set.

| Approximation | Flag | Default | Affects |
|---------------|------|---------|--------|
| Fill price (BUY) at `close` instead of `ask` | `HISTORICAL_FILL_AT_CLOSE=True` | True until bid/ask columns added | `BrokerSimulator.open_trade` |
| Fill price (SELL) at `close` instead of `bid` | `HISTORICAL_FILL_AT_CLOSE=True` | True until bid/ask columns added | `BrokerSimulator.open_trade` |
| Spread from CSV column (not real-time tick spread) | `HISTORICAL_SPREAD_FROM_CSV=True` | True | `HistoricalCSVProvider._get_spread_pips` |
| Spread fallback to `DEFAULT_SPREAD_PIPS` table when CSV value is 0 | `HISTORICAL_SPREAD_FALLBACK=True` | True | `HistoricalCSVProvider._get_spread_pips` |
| No live tick stream (MicrostructureEngine OFF) | `MICROSTRUCTURE_DISABLED=True` | True in backtest | `analysis/microstructure.py:is_backtest_mode()` |
| No live news fetch (uses snapshot) | `NEWS_SNAPSHOT_MODE=True` | True in backtest | `intelligence/news_sources.py:181` |
| No live COT (synthetic proxy) | `COT_SYNTHETIC=True` | True in backtest | `analysis/institutional_flow.py:127` |
| No live correlation matrix (returns neutral) | `CORRELATION_NEUTRAL=True` | True in backtest | `analysis/correlation_engine.py` (when fetch fails) |
| No live intermarket macro (returns neutral) | `INTERMARKET_NEUTRAL=True` | True in backtest | `analysis/intermarket.py` (when fetch fails) |
| No DevilsAdvocateGate (LLM veto skipped) | `LLM_VETO_DISABLED=True` | True in backtest | `backtest/unified_engine.py:745` (skipped) |
| No ApprovalMode (operator workflow skipped) | `APPROVAL_MODE_DISABLED=True` | True in backtest | `backtest/unified_engine.py:745` (skipped) |
| No swap (overnight financing) | `SWAP_NOT_MODELED=True` | True in `BrokerSimulator` | `execution/broker_simulator.py` |
| No latency (instant fill) | `LATENCY_NOT_MODELED=True` | True in `BrokerSimulator` | `execution/broker_simulator.py` |
| No requotes / rejections | `ALWAYS_FILLS=True` | True in `BrokerSimulator` | `execution/broker_simulator.py` |

These flags should be set automatically when `is_backtest_mode()` returns True, and logged at the start of the backtest run so the user is aware of the approximations.
