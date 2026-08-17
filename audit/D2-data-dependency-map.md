# Deliverable 2 — Data Dependency Map

**Project:** Forex AI Autonomous Trading System
**Audit date:** 2026-08-17
**Source:** Compiled from P1-A, P1-B, P1-C, P1-D evidence reports

This document is a complete runtime data dependency map: for every module/feature in the live trading cycle, what raw market data it consumes, from which source, in what timeframe, with what lookback, and whether the historical CSV must supply it.

---

## 1. Master Data Dependency Table

| Module | Data Needed | Source | Timeframe | Lookback | Historical CSV Needed? |
|--------|-------------|--------|-----------|----------|----------------------|
| **MarketAgent.run** (live) | OHLC + tick_volume + spread | `mt5.copy_rates_from_pos(symbol, tf, 0, 300)` | primary TF (M15 default) | 300 bars | ✅ YES (CSV must have OHLC + tick_volume + spread) |
| **MultiTimeframeAnalyzer** | OHLC for 4 TFs | Per-TF `fetch_ohlcv(..., limit=200)` | D1, H4, H1, M15 (M5 if used) | 200 bars/TF | ✅ YES — H4, H1, M15, M5 CSVs required (D1 optional) |
| **DataValidator** | OHLC + DatetimeIndex | df from fetcher | primary TF | n/a (validates) | implicit |
| **add_canonical_indicators** | OHLC + volume (optional) | df (extended adds 139 columns via pandas-ta) | primary TF | 200 (sma_200) | ✅ YES |
| **MarketRegimeDetector** | H, L, C (+ optional ema_21/sma_50/sma_200 cols) | df (self-computes if cols absent) | primary TF | 200 | ✅ YES |
| **SessionAnalyzer** | DatetimeIndex (UTC) | df.index | any | 1 bar (current) | ✅ YES — UTC tz-aware index required |
| **PatternDetector** | O, H, L, C | df | primary TF | 1–3 bars | ✅ YES |
| **SupportResistance** | O, H, L, C, ATR | df | primary TF | swing_window per TF (M15=4, H1=4, H4=5, D1=5) | ✅ YES |
| **LiquidityEngine** | H, L, C, ATR + DatetimeIndex | df | primary TF | 20 bars + ≥1 week (PDH/PWL) | ✅ YES — UTC DatetimeIndex required |
| **AdvancedPatternDetector** | O, H, L, C | df | primary TF | 100 bars | ✅ YES |
| **IntermarketEngine** | DXY, Gold, Oil, US10Y, SP500, VIX | external `MacroDataProvider` | multi | varies | ❌ Live-only (backtest-mode fallback exists) |
| **CurrencyStrengthEngine** | close of 28 cross pairs | MT5/API fetch | M15 | 100 bars/pair | ❌ DISABLED in live code (winrate audit 2026-08-13) |
| **SentimentEngine** | news + retail sentiment | external (NewsAPI + Myfxbook + RSS) | multi | real-time | ❌ Live-only (backtest uses snapshot from `data/economic_calendar.json`) |
| **SMCEngine** | H, L, C, ATR (per TF) | self-fetches H4 + M15 via `fetcher.fetch_ohlcv(symbol, "4h"/"15m", limit=150)` | H4 + M15 | 150 bars/TF | ✅ YES — H4 + M15 CSVs required |
| **MarketStructureEngine** | H, L, C | df | primary TF | 5 bars (strength=2) | ✅ YES |
| **MTFStructureEngine** | H, L, C (per TF) | caller fetches H4 + M15 | H4 + M15 | swing_window 8 + 3 | ✅ YES — H4 + M15 CSVs required |
| **DivergenceEngine** | H, L, C (+ RSI/MACD precomputed) | df | primary TF | 14 (RSI) + 26 (MACD) | ✅ YES |
| **VolatilityEngine** | H, L, C | df | primary TF | 120 bars (BB width percentile + ATR percentile) | ✅ YES |
| **CorrelationEngine** | close of 28 cross pairs + ATR | MT5/API fetch | primary TF | 50 bars/pair | ❌ Live-only — backtest skips correlation; RiskEngine correlation_adjustment defaults to 1.0 |
| **InstitutionalFlowEngine** | CFTC weekly COT (or OHLC for synthetic proxy) | external HTTP (CFTC.gov) | weekly | 50 weeks | ⚠ GATED OFF in backtest_mode — uses synthetic large-candle proxy from OHLC |
| **MicrostructureEngine** | MT5 tick stream: bid, ask, last, volume_real | `mt5.copy_ticks_range(symbol, ..., COPY_TICKS_ALL)` (last 60s) | tick | 60 seconds | ❌ Live-only — gated OFF in backtest_mode |
| **VolumeProfileEngine** | H, L, C, tick_volume | df | primary TF | ≥30 bars (uses full df) | ⚠ DISABLED 2026-07-30 (constructed but not run) |
| **VolumeConfirmation** | close + volume | df | primary TF | 20 + 10 = 30 bars | ✅ YES (when re-enabled) |
| **CandlestickEngine** | O, H, L, C (volume optional) | df + `_pattern_context` | primary TF | 100 bars | ✅ YES |
| **IchimokuEngine** | H, L, C | df | primary TF | 52 + 26 + 5 = 83 bars | ⚠ DISABLED 2026-07-30 |
| **FibonacciEngine** | H, L, C, ATR (optional H4) | df + optional H4 | primary TF | 250 bars | ⚠ DISABLED 2026-07-31 |
| **StopHuntDetector** | O, H, L, C, ATR + liquidity_levels | df | primary TF | ~10 bars post-level | ✅ YES (via `analysis/liquidity_engine.py`) |
| **StopHuntDirectLane** | O, H, L, C | df | primary TF | ~10 bars | ✅ YES — overrides blend when blend=WAIT |
| **SignalEngine** (Rule layer) | close + precomputed ema_50/ema_200/adx/atr + spread + spread_avg_20 + timestamp | df + ind_ctx | primary TF | 200 (EMA200) + 20 (spread avg) | ✅ YES — **spread column required** (else spread filter silently no-ops) |
| **MasterDecisionEngine** | output of 4 layers | in-memory | primary | n/a | implicit |
| **ML Predictor** | 110-feature dict from `FeatureEngineer.build_feature_vector(df, analysis_out, pair, tf)` | in-memory | primary TF | last 200+ bars OHLCV | ❌ DISABLED in live code |
| **RLAgent** | 29-dim state vector (23 market + 6 account) | in-memory | primary TF | last 200 bars | ✅ YES (when active) |
| **LLM Analyst** | 22 context blocks (close/ema9/sma20/rsi/macd/macd_signal/atr/bb_position + pattern + SR + regime + rule-signal + MTF + adv_pattern + session + intermarket + sentiment + news + SMC + fib + vision + bias + memory) | in-memory | primary | n/a | implicit (downstream of OHLCV) |
| **RiskEngine.evaluate** | signal + entry + atr + regime + correlation_ctx | downstream of market_out | primary | n/a | implicit |
| **TradePermission.check** | dec_out + risk_out + news_ctx + session_ctx + execution_filters | downstream | primary | n/a | implicit |
| **PaperTrader / PositionManager** | tick.bid, tick.ask, mt5.positions_get | MT5 live | tick | real-time | ❌ Live-only |
| **OrderManager.place_market_order** | tick.bid, tick.ask, symbol_info.digits/point/spread, account_info.free_margin | MT5 live | tick | real-time | ❌ Live-only |
| **`mt5.order_send`** | request dict | MT5 live | n/a | n/a | ❌ Live-only |

---

## 2. Raw Data Categories Required

### 2.1 OHLCV (Required, All TFs in MTF_CHAIN)

| Field | Live MT5 source | Historical CSV column | Status |
|-------|-----------------|----------------------|--------|
| `open` | `r["open"]` from `copy_rates_from_pos` | `open` | ✅ Present in all 21 existing CSVs |
| `high` | `r["high"]` | `high` | ✅ Present |
| `low` | `r["low"]` | `low` | ✅ Present |
| `close` | `r["close"]` | `close` | ✅ Present |
| `tick_volume` | `r["tick_volume"]` (renamed to `volume`) | `tick_volume` | ✅ Present in all 21 CSVs (renamed to `volume` by loader) |
| `real_volume` | `r["real_volume"]` (NEVER READ by any consumer — dropped at `data/fetcher.py:966` and `data/backtest_ohlcv_cache.py:96`) | `real_volume` | ⚠ Present in CSVs but always 0; dropped silently in pipeline |

### 2.2 Bid / Ask / Spread

| Field | Live MT5 source | Historical CSV column | Status |
|-------|-----------------|----------------------|--------|
| `bid` | `tick.bid` from `mt5.symbol_info_tick` | **MISSING** | ❌ Not in CSV — historical fills at `close`, not at ask (for BUY) or bid (for SELL) |
| `ask` | `tick.ask` | **MISSING** | ❌ Not in CSV |
| `spread` (per-bar, points) | `r["spread"]` from `copy_rates_from_pos` (int points) | `spread` (int points) | ✅ Present in all 21 CSVs but **15-70% of bars are 0** depending on symbol/TF |
| `spread_pips` (derived) | `(ask-bid) × 10^(digits-1)` (live_feed, mt5_data) | `spread_points / 10` for FX, `/1` for XAU/indices (`csv_data_provider.py:448-456`) | ✅ Derived from CSV `spread` column |

### 2.3 Tick Data / Market Microstructure

| Field | Live MT5 source | Historical CSV column | Status |
|-------|-----------------|----------------------|--------|
| `tick.time` | `mt5.symbol_info_tick` / `mt5.copy_ticks_range` | **MISSING** | ❌ Live-only |
| `tick.time_msc` | `tick.time_msc` (NEVER READ — ms precision lost) | **MISSING** | ❌ Live-only |
| `tick.last` | `tick.last` | **MISSING** | ❌ Live-only |
| `tick.flags` | `tick.flags` | **MISSING** | ❌ Live-only |
| Tick stream (60s window) | `mt5.copy_ticks_range` | **MISSING** | ❌ Live-only — `MicrostructureEngine` gated OFF in backtest |

### 2.4 Time / Timezone

| Field | Live MT5 source | Historical CSV column | Status |
|-------|-----------------|----------------------|--------|
| Bar timestamp | `r["time"]` (epoch sec) → broker-tz auto-corrected by `data/fetcher.py:_get_broker_utc_offset_hours` → `tz_localize("UTC")` | `datetime_utc` (ISO 8601 UTC) | ✅ Both UTC-aware after fetcher fix |
| Bar timestamp precision | 1 second (`time_msc` discarded) | 1 second | ✅ Match |
| `current_time()` | `LiveMT5Provider.current_time()` → ⚠ **`datetime.utcnow()` (NAIVE)** | `HistoricalCSVProvider.current_time()` → ✅ tz-aware UTC | ⚠ PARITY VIOLATION (P1-A R4) — fix in Deliverable 5 |
| Sessions (Asia/London/NY) | `analysis/session_analyzer.py` uses `datetime.now(timezone.utc)` when `dt=None`; `analysis_agent.py:259-273` passes `dt=_bar_dt` from `df.index[-1]` | (same — caller passes bar timestamp) | ✅ Fixed via call-site pattern |
| Weekend close | Fri 21:00 UTC → Sun 21:00 UTC (broker/data_validator.py:180) | (implicit in CSV gaps) | ✅ Match |

### 2.5 External / Non-OHLCV Data

| Field | Live source | Historical reproduction | Status |
|-------|-------------|-------------------------|--------|
| News (Forex Factory + RSS + NewsAPI) | `intelligence/news_sources.py` live fetch | `data/economic_calendar.json` snapshot | ⚠ Live-only — backtest uses stale snapshot |
| Retail sentiment (Myfxbook) | `analysis/myfxbook_sentiment.py` | NOT reproducible | ❌ Live-only |
| DXY, Gold, Oil, US10Y, SP500, VIX | `MacroDataProvider` live | NOT reproducible | ❌ Live-only — `IntermarketEngine` has backtest fallback (uses cached or returns neutral) |
| CFTC weekly COT | `analysis/institutional_flow.py:_fetch_cot_from_cftc()` | NOT reproducible | ⚠ Gated OFF in backtest_mode — uses synthetic large-candle proxy from OHLC |
| 28 cross-pair close | Live MT5/API fetch | NOT reproducible | ❌ Live-only — `CorrelationEngine` runs in backtest but returns neutral if fetch fails |
| 28 cross-pair strength | Live fetch (DISABLED) | NOT reproducible | ❌ DISABLED in agent (`if False:` block) |

---

## 3. Timeframe × Module Matrix

Cells indicate lookback bars required on that TF for the module to be non-NaN at the current bar.

| Module | M1 | M5 | M15 | M30 | H1 | H4 | D1 | W1 |
|--------|----|----|-----|-----|----|----|----|----|
| `MarketAgent` (primary fetch) | — | — | 300 | — | 300 | — | — | — |
| `MultiTimeframeAnalyzer` | — | 200 | 200 | 200 | 200 | 200 | 200 | — |
| `SMCEngine` (self-fetch) | — | — | 150 | — | — | 150 | — | — |
| `MTFStructureEngine` (caller fetches) | — | — | swing(3) | — | — | swing(8) | — | — |
| `MTFAnalyzer` (self-fetch) | — | 200 | 200 | 200 | 200 | 200 | — | — |
| `SmartMoneyEngine` (full MTF) | — | — | 200 | — | 200 | 200 | 200 | — |
| `LiquidityEngine` (PDH/PWL/Asian range) | — | — | ≥1 week intraday | — | ≥1 week | — | — | — |
| `IchimokuEngine` (DISABLED) | — | — | 83 | — | 83 | 83 | — | — |
| `NadarayaWatson` (REPAINTING) | — | — | 500 | — | 500 | 500 | — | — |
| `VolumeProfileEngine` (DISABLED) | — | — | 30+ (full df) | — | 30+ | 30+ | — | — |
| `VolatilityEngine` | — | — | 120 | — | 120 | 120 | 120 | — |
| `MarketRegimeDetector` | — | — | 200 | — | 200 | 200 | 200 | — |
| `SignalEngine` (Rule layer) | — | — | 200 | — | 200 | 200 | — | — |
| `CorrelationEngine` | — | — | 50 × 28 pairs | — | 50 × 28 | 50 × 28 | — | — |
| ML Feature Engineer | — | — | 200 | — | 200 | 200 | — | — |
| ML Institutional Adapter | — | — | 220 | — | 220 | 220 | — | — |
| RL State (V2) | — | — | 200 | — | 200 | 200 | — | — |

---

## 4. MAX_REQUIRED_LOOKBACK Derivation

The codebase has **NO central `MAX_REQUIRED_LOOKBACK` constant**. The effective floor is the maximum across all consumers:

| Rank | Lookback | Source | TF |
|------|----------|--------|----|
| 1 | **500 bars** | `NadarayaWatson.window_size` (`nadaraya_watson_envelope.py:58`) | per-call (typically H1/H4) |
| 2 | **300 bars** | `MarketAgent` primary fetch (`agents/market_agent.py:186` `limit=300`) | primary TF (M15 default) |
| 3 | **300 bars** | `HistoricalCSVProvider.lookback_bars` (`core/csv_data_provider.py:219`) | primary TF (parity with live) |
| 4 | **220 bars** | `MIN_WARMUP_BARS` in `institutional_feature_adapter.py` | primary TF |
| 5 | **200 bars** | `MTFAnalyzer` per-TF fetch (`mtf_analyzer.py:427` `limit=200`) | H4 + H1 + M15 + M5 |
| 6 | **200 bars** | `MultiTimeframeAnalyzer.analyze(["1d","4h","1h","15m"])` per-TF fetch (`agents/market_agent.py:171`) | all MTF chain TFs |
| 7 | **200 bars** | `sma_200` / `ema_200` in `data/indicators.py:25,30` and `data/indicators_ext.py:289` | per-call |
| 8 | **150 bars** | `SMCEngine` H4 + M15 fetch (`smc_engine.py:109-110` `limit=150`) | H4 + M15 |
| 9 | **150 bars** | `OrderBlockDetector.decay_half_life` for M15 (`order_block.py:87`) | M15 (H4=300, D1=400) |
| 10 | **120 bars** | `VolatilityEngine.squeeze_lookback` (`volatility.py:58`) | per-call |
| 11 | **100 bars** | `EXTERNAL_LOOKBACK` in `liquidity_structure.py:51` | per-call |
| 12 | **100 bars** | `volatility_lookback` / `AdvancedPatternDetector(lookback=100)` / `CurrencyStrengthEngine(candle_limit=100)` | per-call |
| 13 | **1 trading week** | PDH/PDL/PWH/PWL + Asian range (`analysis/liquidity_zones.py:180-192`) | intraday TF |

### Practical floor (derived)

For live parity:

```
MAX_REQUIRED_LOOKBACK = 500 bars primary TF (NadararaWatson)
                      + 200 bars per HTF (H4, H1, M5) for MTFAnalyzer
                      + 1 trading week of intraday bars (PDH/PWL/Asian range)
                      + 50 bars × 28 cross pairs (CorrelationEngine) — live only
```

In M15-equivalent bars (1 H1 = 4 M15; 1 H4 = 16 M15):

- 500 × 1 (M15) = 500
- 1 week ≈ 5 trading days × 24h × 4 (M15/h) − weekend = ~460 M15 bars
- Plus 200 H4 bars × 16 = 3,200 M15-equivalent (if H4 needs 200 closed bars before signal window)

**Operational rule for the downloader:** download at minimum 1,000 bars on the primary TF + 200 bars on each HTF in `MTF_CHAIN = [H4, H1, M15, M5]` + at least 2 weeks of additional history beyond the target backtest window for warmup. For NadarayaWatson consumers, download at least 500 bars beyond the backtest start date.

---

## 5. Indicator → Raw Data Trace (Selected Examples)

Per Phase 3 spec. Each indicator traced from name → implementation → required raw columns → required TF → required lookback → required history.

### 5.1 ATR(14)

```
ATR(14)
  ↓
data/indicators.py:86-90  /  data/indicators_ext.py:427-430
  ↓
ta.atr(High, Low, Close, length=14) — Wilder smoothing
  ↓
Required raw: High, Low, Close
  ↓
Required TF: per-call (typically primary TF — M15 default)
  ↓
Required lookback: 14 bars (Wilder smoothing → effective ~28 bars for non-NaN at current bar)
  ↓
Required history: 14 bars on primary TF
  ↓
CSV must contain OHLC for primary TF
```

### 5.2 RSI(14)

```
RSI(14)
  ↓
data/indicators.py:61-64  /  data/indicators_ext.py:329-332
  ↓
ta.rsi(Close, length=14) — Wilder smoothing
  ↓
Required raw: Close
  ↓
Required TF: per-call (primary TF)
  ↓
Required lookback: 14 bars (~28 effective)
  ↓
Required history: 14 bars on primary TF
  ↓
CSV must contain Close for primary TF
```

### 5.3 EMA(200)

```
EMA(200)
  ↓
data/indicators_ext.py:289  (length=200)
  ↓
ta.ema(Close, length=200) — pandas ewm with adjust=False
  ↓
Required raw: Close
  ↓
Required TF: per-call
  ↓
Required lookback: 200 bars (effective warmup for stable EMA)
  ↓
Required history: 200 bars on primary TF
  ↓
CSV must contain Close for primary TF
```

### 5.4 ADX(14)

```
ADX(14)
  ↓
data/indicators.py:33-45  /  data/indicators_ext.py:568-576
  ↓
ta.adx(High, Low, Close, length=14) + DI+/DI-
  ↓
Required raw: High, Low, Close
  ↓
Required TF: per-call
  ↓
Required lookback: 14 (Wilder) → ~28 effective
  ↓
Required history: 14 bars on primary TF
  ↓
CSV must contain H, L, C for primary TF
```

### 5.5 Bollinger Bands(20, 2)

```
BB(20, 2)
  ↓
data/indicators_ext.py:433-451
  ↓
ta.bbands(Close, length=20, std=2) → bb_upper, bb_mid, bb_lower, bb_width, bb_pct_b
  ↓
Required raw: Close
  ↓
Required TF: per-call
  ↓
Required lookback: 20 bars
  ↓
Required history: 20 bars on primary TF
  ↓
CSV must contain Close for primary TF
```

### 5.6 MACD(12, 26, 9)

```
MACD(12, 26, 9)
  ↓
data/indicators_ext.py:553-566
  ↓
ta.macd(Close, fast=12, slow=26, signal=9) → macd, macd_signal, macd_hist
  ↓
Required raw: Close
  ↓
Required TF: per-call
  ↓
Required lookback: 26 + 9 = 35 bars
  ↓
Required history: 35 bars on primary TF
  ↓
CSV must contain Close for primary TF
```

### 5.7 VWAP

```
VWAP
  ↓
data/indicators_ext.py:503-528
  ↓
ta.vwap(High, Low, Close, volume) — anchored intraday; falls back to cumulative TP×vol
  ↓
Required raw: High, Low, Close, Volume + DatetimeIndex
  ↓
Required TF: per-call (typically primary intraday TF)
  ↓
Required lookback: 1 bar (cumulative)
  ↓
Required history: full intraday session
  ↓
CSV must contain H, L, C, tick_volume + UTC DatetimeIndex
⚠ SILENTLY SKIPPED when volume column absent
```

### 5.8 Stochastic(14, 3)

```
Stochastic(14, 3)
  ↓
data/indicators_ext.py:339-345
  ↓
ta.stoch(High, Low, Close, k=14, d=3) → stoch_k, stoch_d
  ↓
Required raw: High, Low, Close
  ↓
Required TF: per-call
  ↓
Required lookback: 14 + 3 = 17 bars
  ↓
Required history: 17 bars on primary TF
  ↓
CSV must contain H, L, C for primary TF
```

### 5.9 Supertrend(10, 3.0)

```
Supertrend(10, 3.0)
  ↓
analysis/supertrend.py:38
  ↓
ATR(10) + HL2 mid, ratcheted bands
  ↓
Required raw: High, Low, Close
  ↓
Required TF: per-call
  ↓
Required lookback: 10 bars (ATR)
  ↓
Required history: 10 bars on primary TF
  ↓
CSV must contain H, L, C for primary TF
```

### 5.10 Market Regime (composite)

```
MarketRegimeDetector
  ↓
analysis/market_regime.py:67
  ↓
ADX(14) + ATR + EMA21/SMA50/SMA200 alignment
  ↓
Required raw: High, Low, Close (uses precomputed ema_21/sma_50/sma_200 if present, else self-computes)
  ↓
Required TF: per-call
  ↓
Required lookback: 14 (ADX) + 200 (SMA200, optional) → 200 bars
  ↓
Required history: 200 bars on primary TF
  ↓
CSV must contain H, L, C for primary TF
```

### 5.11 Support/Resistance zones

```
SupportResistance
  ↓
analysis/support_resistance.py:651
  ↓
Swing highs/lows → cluster into zones, ATR-adaptive threshold
  ↓
Required raw: Open, High, Low, Close
  ↓
Required TF: per-call (TF-aware swing_window: M5=3, M15=4, H1=4, H4=5, D1=5)
  ↓
Required lookback: swing_window + rejection count window
  ↓
Required history: ~50-100 bars on primary TF
  ↓
CSV must contain OHLC for primary TF
  ⚠ Caller contract (file:18-23): "walk-forward callers must slice df.iloc[:i+1] before calling analyze()"
```

### 5.12 Liquidity zones (PDH/PDL/PWH/PWL + Asian range)

```
LiquidityZoneMapper
  ↓
analysis/liquidity_zones.py:52
  ↓
Equal highs/lows + PDH/PDL/PWH/PWL + Asian session range
  ↓
Required raw: High, Low, Close + ATR + DatetimeIndex (UTC required)
  ↓
Required TF: intraday (M15 typically)
  ↓
Required lookback: SWING_WINDOW=5; ≥1 day for PDH/PDL; ≥1 week for PWH/PWL; Asian range = df.tail(200)
  ↓
Required history: ≥1 week of intraday bars with UTC DatetimeIndex
  ↓
CSV must contain H, L, C, ATR (or HLC for ATR self-compute) + UTC DatetimeIndex
```

### 5.13 SMCEngine (H4 + M15 self-fetch)

```
SMCEngine
  ↓
analysis/smc_engine.py:108
  ↓
H4 OB+FVG+BOS+CHoCH+sweep + M15 entry timing + pattern confirm
  ↓
Required raw: H, L, C, ATR (per TF)
  ↓
Required TF: H4 + M15 (self-fetches via _fetch_with_atr)
  ↓
Required lookback: limit=150 per TF
  ↓
Required history: 150 bars H4 + 150 bars M15
  ↓
CSV must contain H, L, C, ATR for both H4 and M15
```

### 5.14 NadarayaWatson Envelope (REPAINTING — special case)

```
NadarayaWatson Envelope
  ↓
analysis/nadaraya_watson_envelope.py:53
  ↓
Gaussian-kernel regression + MAD envelope
  ↓
Required raw: Close
  ↓
Required TF: per-call (typically H1 or H4)
  ↓
Required lookback: window_size=500 (centered window — NON-CAUSAL)
  ↓
Required history: 500 bars
  ↓
⚠ SELF-DOCUMENTED AS REPAINTING — module sets nwe_stable=False for last 500 bars
  ↓
For backtest parity, consumers MUST only read nwe_mid/upper/lower where nwe_stable=True
  ↓
CSV must contain Close for the TF NadarayaWatson runs on
```

### 5.15 CorrelationEngine (28 cross-pairs — live-only)

```
CorrelationEngine
  ↓
analysis/correlation_engine.py:66
  ↓
28-pair correlation matrix + ATR-volatility regime
  ↓
Required raw: Close (per pair) + ATR (per pair)
  ↓
Required TF: per-call
  ↓
Required lookback: lookback_periods=50 per pair × 28 pairs = 1400 close prices
  ↓
Required history: 50 bars × 28 cross pairs
  ↓
❌ NOT in historical CSV — fetched live from MT5/API
  ↓
Backtest behavior: returns neutral correlation_adjustment=1.0 if fetch fails
```

### 5.16 ML features (110-dim)

```
FeatureEngineer.build_feature_vector
  ↓
ml/feature_engineer.py:126-655
  ↓
Last bar OHLCV + 21 indicator columns (rsi/atr/macd/ema_9-50/sma_200/bb_*) + 8 currency strengths + intermarket trends + session/news/SMC/confluence contexts
  ↓
Required raw: OHLCV (last 200+ bars for sma_200) + analysis_out contexts (downstream of OHLCV)
  ↓
Required TF: primary
  ↓
Required lookback: 200 bars (sma_200)
  ↓
Required history: 200 bars on primary TF
  ↓
CSV must contain OHLCV (with tick_volume) for primary TF
  ↓
⚠ DISABLED in live code (if False: at analysis_agent.py:2001)
```

### 5.17 RL state (29-dim)

```
ForexTradingEnvV2 state
  ↓
ml/rl_environment_v2.py + ml/train_rl_v2.py:build_features_df_v2
  ↓
Multi-horizon momentum (1, 4, 16, 96 bars) + candle geometry + trend (ema20/50, sma200) + momentum oscillators (rsi, macd, stoch) + volatility (atr, bb) + volume_z (100 bars) + session (hour/dow cyclical) + 6 account state floats
  ↓
Required raw: OHLCV + DatetimeIndex
  ↓
Required TF: primary
  ↓
Required lookback: 200 bars (sma_200) + 96 bars (ret_96)
  ↓
Required history: 200 bars on primary TF with UTC DatetimeIndex
  ↓
CSV must contain OHLCV + UTC DatetimeIndex
```

### 5.18 LLM Analyst context (22 blocks)

```
MasterAnalyst LLM context
  ↓
agents/master_analyst.py
  ↓
22 context blocks: indicator ctx (close/ema9/sma20/rsi/macd/macd_signal/atr/bb_position) + pattern ctx + SR ctx + regime ctx + rule-engine signal ctx + MTF bias + advanced pattern ctx + session + intermarket + sentiment + news + SMC + fib + vision + bias + memory + divergence + ichimoku + vol_profile + smc_advanced + mtf_structure + strategy + news_api + econ_calendar + fred + retail_sentiment
  ↓
Required raw: derived from OHLCV + external feeds (news, macro, sentiment)
  ↓
Required TF: primary + H4 (for MTF context)
  ↓
Required lookback: 200 bars (downstream of indicator chain)
  ↓
Required history: 200 bars OHLCV + UTC DatetimeIndex
  ↓
CSV must contain OHLCV + spread (for cost-aware LLM context)
```

---

## 6. Summary of Required CSV Fields

Based on the full dependency audit, the historical CSV must contain:

### Required (without these, backtest silently degrades)

| Field | Why required |
|-------|--------------|
| `datetime_utc` (tz-aware UTC ISO 8601) | Session detection, Asian range, PDH/PWL, VWAP, kill zones, MTF alignment |
| `open` | OHLC sanity, candle patterns, candle geometry |
| `high` | ATR, ADX, Stochastic, CCI, Bollinger, Ichimoku, Donchian, market structure, S/R, liquidity sweeps |
| `low` | Same as `high` |
| `close` | EMA, SMA, RSI, MACD, Bollinger, BB, ATR (close-to-close), SupportResistance, Fibonacci, Ichimoku |
| `tick_volume` | VWAP, OBV, MFI, CMF, A/D Line, VWMA, Volume RSI, VolumeProfile, VolumeConfirmation, VW-MACD, CandlestickEngine volume_z |

### Required (with current gaps — 15-70% of bars are 0)

| Field | Why required |
|-------|--------------|
| `spread` (int points) | SignalEngine spread filter (`signal_engine.py:114-120`), `ind_ctx["spread_pips"]` for cost-aware EV gate, RiskEngine correlation_adjustment |

### Recommended (for higher-fidelity reproduction)

| Field | Why recommended |
|-------|-----------------|
| `bid` | Realistic fill price for BUY orders (historical currently fills at `close`) |
| `ask` | Realistic fill price for SELL orders |

### Optional (currently always 0, dropped by pipeline)

| Field | Why optional |
|-------|---------------|
| `real_volume` | Forex has no real consolidated volume; tick_volume is the convention. Keep column for schema completeness but pipeline does not consume it. |

### Impossible / Broker-Dependent (live-only, no historical reproduction)

| Field | Why impossible |
|-------|-----------------|
| Live tick stream (bid/ask/last/flags per tick) | MT5 `copy_ticks_range` returns only recent ticks; historical tick databases are broker-specific and multi-GB per day |
| `time_msc` (millisecond bar open precision) | MT5 returns second-precision in bar data; sub-second is tick-only |
| Market depth / Level-2 (order book) | `mt5.market_book_add` not used; not stored anywhere |
| Live news (Forex Factory + RSS + NewsAPI) | Real-time event data; historical news databases are paid services (Bloomberg/Refinitiv) |
| Retail sentiment (Myfxbook) | Live-only API; historical snapshot service is paid |
| DXY/Gold/Oil/US10Y/SP500/VIX | Can be downloaded separately from yfinance/AlphaVantage; not currently in CSV |
| CFTC weekly COT | Public data, downloadable from CFTC.gov — backtest currently uses synthetic proxy |
| 28 cross-pair close history | Can be downloaded — requires 28 extra symbols in CSV downloader |

---

## 7. Cross-References

- Detailed per-indicator audit: `docs/audit/evidence/P1-C-analysis-indicators-audit.md`
- ML/RL/LLM feature list: `docs/audit/evidence/P1-D-ml-rl-llm-audit.md`
- Live MT5 vs CSV field parity: `docs/audit/evidence/P1-A-data-provider-audit.md` §13
- Existing CSV column analysis: `docs/audit/evidence/P6-csv-audit.md`
