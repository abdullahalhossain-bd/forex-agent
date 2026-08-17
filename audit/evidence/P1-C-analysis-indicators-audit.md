# P1-C — Analysis & Indicators Audit

**Task ID:** P1-C
**Scope:** `data/indicators*.py`, `data/indicator_registry.py`, all `analysis/*.py` indicator / feature-engineering / pattern / SMC / liquidity / session / MTF / signal modules listed in the task brief.
**Goal:** For every feature produced at runtime, determine what raw market columns, what timeframes, and what lookback it actually needs — so the historical CSV pipeline can guarantee parity with the live pipeline.

---

## 1. Master Indicator Table

All lookback values are the **minimum bars required for the indicator to be non-NaN at the current bar** (warmup). "Caller" refers to the file:line that instantiates / runs the engine in the production cycle (`agents/analysis_agent.py`, `agents/market_agent.py`).

### 1a. `data/indicators.py` — minimal `ta`-lib wrapper (legacy)

| # | Indicator | File:Line | Formula | Raw columns needed | TF(s) | Lookback | MTF? | Called from |
|---|-----------|-----------|---------|---------------------|--------|-----------|------|-------------|
| 1 | SMA 20 / 50 / 200 | `data/indicators.py:23-25` | `ta.trend.sma_indicator(close, window)` | close | per-call | 20 / 50 / 200 | no | `agents/market_agent.py` (via `add_all`) |
| 2 | EMA 9 / 21 / 50 / 200 | `data/indicators.py:26-30` | `ta.trend.ema_indicator(close, window)` | close | per-call | 9 / 21 / 50 / 200 | no | market_agent, signal_engine |
| 3 | ADX(14) | `data/indicators.py:33-45` | `ta.trend.ADXIndicator(high, low, close, window=14)` | H, L, C | per-call | 14 (Wilder) → ~28 effective | no | market_agent, signal_engine (ADX gate) |
| 4 | Stochastic(14,3) | `data/indicators.py:47-59` | `StochasticOscillator(H, L, C, window=14, smooth_window=3)` | H, L, C | per-call | 14 + 3 = 17 | no | market_agent |
| 5 | RSI(14) | `data/indicators.py:61-64` | `RSIIndicator(close, window=14)` | close | per-call | 14 | no | market_agent, signal_engine |
| 6 | MACD(12,26,9) | `data/indicators.py:66-75` | `MACD(close)` default 12/26/9 | close | per-call | 26 + 9 = 35 | no | market_agent, signal_engine |
| 7 | Bollinger Bands(20,2) | `data/indicators.py:77-84` | `BollingerBands(close, window=20)` | close | per-call | 20 | no | market_agent |
| 8 | ATR(14) | `data/indicators.py:86-90` | `AverageTrueRange(H, L, C, window=14)` | H, L, C | per-call | 14 | no | market_agent, liquidity, SMC, fibonacci, OB |
| 9 | Trend composite | `data/indicators.py:92-104` | Reads SMA20/50/200 alignment | close | per-call | 200 (via SMA200) | no | market_agent |

### 1b. `data/indicators_ext.py` — `pandas_ta` wrapper (canonical, 139+ columns)

`add_all()` is called via `data/indicator_registry.add_canonical_indicators()` from `agents/market_agent.py` (canonical path). Requires `df` with **open, high, low, close, volume** (volume optional — padded to 0). Needs a sorted DatetimeIndex for VWAP / Ichimoku / pivots.

| # | Indicator | File:Line | Formula | Raw columns needed | Lookback | MTF? | Notes |
|---|-----------|-----------|---------|---------------------|-----------|------|-------|
| 10 | SMA 10/20/50/100/200 | `indicators_ext.py:283-292` | `ta.sma(close, length)` | close | 200 (longest) | no | |
| 11 | EMA 5/9/13/21/34/55/89 | `indicators_ext.py:294-298` | `ta.ema(close, length)` | close | 89 (longest) | no | |
| 12 | WMA 20 | `indicators_ext.py:301-302` | `ta.wma(close, length=20)` | close | 20 | no | |
| 13 | HMA 20 | `indicators_ext.py:304-306` | `ta.hma(close, length=20)` | close | 16 (≈ 20/2 + lag) | no | |
| 14 | VWMA 20 | `indicators_ext.py:308-313` | `ta.vwma(close, volume, length=20)` | close, **volume** | 20 | no | Needs volume |
| 15 | RSI 7/14/21 | `indicators_ext.py:329-332` | `ta.rsi(close, length)` | close | 21 (longest) | no | |
| 16 | Stochastic(14,3) | `indicators_ext.py:339-345` | `ta.stoch(H, L, C, k=14, d=3)` | H, L, C | 14 + 3 = 17 | no | |
| 17 | Williams %R(14) | `indicators_ext.py:348-349` | `ta.willr(H, L, C, length=14)` | H, L, C | 14 | no | |
| 18 | ROC 10/20 | `indicators_ext.py:353-356` | `ta.roc(close, length)` | close | 20 | no | |
| 19 | MFI(14) | `indicators_ext.py:360-365` | `ta.mfi(H, L, C, volume, length=14)` | H, L, C, **volume** | 14 | no | Needs volume |
| 20 | TSI(13,25) | `indicators_ext.py:367-377` | `ta.tsi(close, fast=13, slow=25)` | close | 25 + 13 = 38 | no | |
| 21 | Ultimate Oscillator | `indicators_ext.py:379-384` | `ta.uo(H, L, C)` default 7/14/28 | H, L, C | 28 | no | |
| 22 | ATR 7/14/21 | `indicators_ext.py:427-430` | `ta.atr(H, L, C, length)` | H, L, C | 21 (longest) | no | |
| 23 | Bollinger(20,2) | `indicators_ext.py:433-451` | `ta.bbands(close, length=20, std=2)` | close | 20 | no | |
| 24 | Keltner(20,2) | `indicators_ext.py:453-461` | `ta.kc(H, L, C, length=20, scalar=2)` | H, L, C | 20 + ATR(14) | no | |
| 25 | Donchian(20) | `indicators_ext.py:463-470` | `ta.donchian(H, L, lower_length=20, upper_length=20)` | H, L | 20 | no | |
| 26 | StdDev(20) | `indicators_ext.py:473-475` | `ta.stdev(close, length=20)` | close | 20 | no | |
| 27 | OBV | `indicators_ext.py:495` | `ta.obv(close, volume)` | close, **volume** | 1 | no | Needs volume |
| 28 | VWAP | `indicators_ext.py:503-528` | `ta.vwap(H, L, C, volume)` — anchored intraday; falls back to cumulative TP×vol | H, L, C, **volume**, **DatetimeIndex** | 1 (cumulative) | no | Volume required; DatetimeIndex required |
| 29 | Accumulation/Distribution | `indicators_ext.py:531` | `ta.ad(H, L, C, volume)` | H, L, C, **volume** | 1 | no | Needs volume |
| 30 | CMF(20) | `indicators_ext.py:534-537` | `ta.cmf(H, L, C, volume, length=20)` | H, L, C, **volume** | 20 | no | Needs volume |
| 31 | MACD(12,26,9) | `indicators_ext.py:553-566` | `ta.macd(close, fast=12, slow=26, signal=9)` | close | 26 + 9 = 35 | no | |
| 32 | ADX(14) + DI+/DI- | `indicators_ext.py:568-576` | `ta.adx(H, L, C, length=14)` | H, L, C | 14 (Wilder) → ~28 | no | |
| 33 | Aroon(25) | `indicators_ext.py:578-586` | `ta.aroon(H, L, length=25)` | H, L | 25 | no | |
| 34 | CCI(20) | `indicators_ext.py:588-592` | `ta.cci(H, L, C, length=20)` | H, L, C | 20 | no | |
| 35 | Volume RSI(14) | `indicators_ext.py:600-644` | Up-vol vs down-vol RSI | close, **volume** | 14 | no | Needs volume |
| 36 | Ichimoku(9,26,52,26) | `indicators_ext.py:650-664` | Tenkan/Kijun/Senkou A/B/Chikou | H, L, C | 52 + 26 displacement = **78** | no | |
| 37 | Pivot Points (classic/fib/camarilla) | `indicators_ext.py:670-717` | From previous completed candle | H, L, C | 2 (uses `iloc[-2]`) | no | |
| 38 | Candlestick patterns (30+) | `indicators_ext.py:723-747` | `ta.cdl_pattern(O, H, L, C, name="all")` | **O**, H, L, C | 1-3 | no | |
| 39 | Fractal S/R(5) | `indicators_ext.py:753-777` | Swing high/low = local max/min over ±5 bars | H, L | 5 + 5 = 11 | no | |
| 40 | Trend signal composite | `indicators_ext.py:783-818` | EMA9/EMA21/SMA50 + ADX25 | close + (H,L,C for ADX) | 50 (via SMA50) + 14 (ADX) | no | |

### 1c. `analysis/` — engine layer (each `analyze(df)` returns a dict context)

| # | Engine | File:Line (entry) | Formula | Raw columns needed | Lookback | MTF? | Called from |
|---|--------|-------------------|---------|---------------------|-----------|------|-------------|
| 41 | MarketRegimeDetector | `market_regime.py:67` | ADX(14) self-computed + ATR + EMA21/SMA50/SMA200 alignment | H, L, C (uses precomputed `ema_21/sma_50/sma_200` if present, else self-computes) | 14 (ADX) + 200 (SMA200, optional) | no | `agents/market_agent.py` |
| 42 | MarketStructure (fractal BOS/CHoCH) | `market_structure.py:39` | strength-window swing detection, close-beyond-swing = break | H, L, C | `strength=2` → 5 bars min | no (caller passes df) | `analysis/order_block.py`, `analysis/smc_engine.py` |
| 43 | MTFStructureEngine (external H4 + internal M15) | `structure_mtf.py:83` | Wraps `MarketStructureEngine` for HTF + LTF | H, L, C | swing_window 8 + 3 | **YES — caller passes df_external + df_internal** | `agents/analysis_agent.py:155` (`analyze(df_external=df_h4, df_internal=df_m15)`) |
| 44 | SupportResistance zones | `support_resistance.py:651` | Swing highs/lows → cluster into zones, ATR-adaptive threshold | H, L, C, O | swing_window per TF (M5=3, M15=4, H1=4, H4=5, D1=5) | no | `agents/analysis_agent.py:25` |
| 45 | LiquidityPoolAnalyzer (Day 61) | `liquidity.py:61` | Equal highs/lows + sweep detection + premium/discount zone | H, L, C (+ `atr` if present, else self-computes ATR14) | swing_window=5, min `5*4+10 = 30 bars` | no | `analysis/liquidity_engine.py`-indirectly via smart_money.py; original `LiquidityPoolAnalyzer` directly consumed by unified_signal_engine + entry_safety_filters |
| 46 | LiquidityEngine (Day 62 orchestrator) | `liquidity_engine.py:63` | Combines liquidity_zones + session_analysis + stop_hunt + structure + fvg | H, L, C + `atr` column (required) + DatetimeIndex | 20 bars min, plus weekly PDH/PWL → needs ≥1 week of intraday bars | **YES — Asian range / PDH / PWH needs DatetimeIndex across ≥1 week** | `agents/analysis_agent.py:459` |
| 47 | LiquidityStructureAnalyzer | `liquidity_structure.py:74` | Internal/external scope, failure-swing, trendline fit, inducement | H, L, C + `atr` | `EXTERNAL_LOOKBACK=100`, `TRENDLINE_LOOKBACK=80`, `SWING_WINDOW=5` | no (caller passes df) | `analysis/liquidity_engine.py:56` |
| 48 | LiquidityZoneMapper | `liquidity_zones.py:52` | Equal highs/lows + PDH/PDL/PWH/PWL + Asian session range | H, L, C + `atr` + **DatetimeIndex required** | SWING_WINDOW=5, ≥1 day for PDH/PDL, ≥1 week for PWH/PWL, Asian range `df.tail(200)` | **YES — PDH/PDL/PWH/PWL + Asian range need DatetimeIndex spanning ≥1 week** | `analysis/liquidity_engine.py:52` |
| 49 | VolatilityEngine (squeeze + ATR regime) | `volatility.py:74` | BB width percentile + ATR percentile | H, L, C | `bb_window=20`, `squeeze_lookback=120`, `atr_lookback=100` → **~120 bars** | no | `agents/analysis_agent.py:38` |
| 50 | VolumeProfileEngine | `volume_profile.py:77` | Price-binned tick-volume distribution + Value Area + POC | H, L, C, **volume/tick_volume** | ≥30 bars (uses full df) | no | `agents/analysis_agent.py:39` |
| 51 | VolumeConfirmation | `volume_confirmation.py:52` | Current vol vs 20-bar avg + price/volume trend divergence | C, **volume** | 20 + lookback 10 = 30 | no | `agents/analysis_agent.py:648` |
| 52 | InstitutionalFlowEngine (COT weekly) | `institutional_flow.py:101` | Live CFTC COT weekly net positioning, or synthetic large-candle proxy | (no OHLCV directly — uses `pair` name) or df for synthetic (C, O) | 50 candles for synthetic | **YES — pulls live CFTC weekly data** (`is_backtest_mode` gates this OFF in backtest) | `agents/analysis_agent.py:53` |
| 53 | MicrostructureEngine | `microstructure.py:41` | MT5 tick stream: tick speed, spread, volume burst, acceleration | **MT5 tick stream — bid, ask, last, volume_real** | 60 seconds of ticks | **YES — bypasses OHLC entirely, calls MT5.copy_ticks_range** | `agents/analysis_agent.py:1174`; gated OFF in backtest_mode |
| 54 | CandlestickEngine (3 sources merged) | `candlestick_engine.py:293` | br + mw + ml pattern detectors, confidence-scored | H, L, C, O; volume optional (used for `volume_z` context) | atr_period=14, volatility_lookback=100, location_lookback=20, volume_lookback=20 → **~100 bars** | no | `agents/analysis_agent.py` (via `_pattern_context.build_context`) |
| 55 | IchimokuEngine | `ichimoku.py:71` | Standard 9/26/52/26 | H, L, C | `52 + 26 + 5 = 83` | no | `agents/analysis_agent.py:37` |
| 56 | SuperTrend(10,3.0) | `supertrend.py:38` | ATR(10) + HL2 mid, ratcheted bands | H, L, C | period=10 | no | consumed via `extended_modules_adapter` |
| 57 | FibonacciEngine | `fibonacci.py:1` (2354 lines) | Swing HH/LL detection + retracement/extension + sweep trigger + HTF alignment | H, L, C + `atr`; optional sr_ctx, ema/vwap confluence | swing detection over 250-bar decision window per Day 43 audit note | **YES — optional HTF alignment uses H4 df** | `agents/analysis_agent.py:28` (DISABLED since 2026-07-31, fib_ctx kept empty) |
| 58 | OrderBlockDetector | `order_block.py:1` | Impulse + structure break + FVG confluence + sweep conditioning; timeframe-aware | H, L, C + `atr` (required) + optional `tick_volume` | TF params: M15 sweep_lookback=15, max_run_lookback=6, decay_half_life=150 → **~150 bars** for full decay | no (caller passes df) | `analysis/smc_engine.py`, `analysis/smart_money.py`, `analysis/mtf_analyzer.py` |
| 59 | SmartMoneyEngine | `smart_money.py:78` | Composes MarketStructure + LiquidityEngine + OB + FVG + KillZone | H, L, C + `atr` + DatetimeIndex (for session kill-zones) | ≥30 bars + KillZone UTC hour check | **YES — KILL_ZONES dict uses UTC hour; `analyze()` calls _fetch_with_atr for H4 + M15** | `agents/analysis_agent.py` (invoked indirectly via smc_engine path) |
| 60 | SMCEngine (Day 44) | `smc_engine.py:108` | H4 OB+FVG+BOS+CHoCH+sweep + M15 entry timing + pattern confirm | H, L, C + `atr` for H4 and M15 | `limit=150` for both H4 and M15 | **YES — self-fetches H4 + M15** via `_fetch_with_atr` | `agents/analysis_agent.py:30` |
| 61 | SMCAdvancedEngine (Mitigation + Inducement) | `smc_advanced.py:79` | Broken-OB retest + small-swing inducement | H, L, C + `atr` | impulse_atr_mult=1.5, ob_lookback=3, inducement_window=5 → ~30 bars | no | `agents/analysis_agent.py:41` |
| 62 | SupplyDemandZones | `supply_demand_zones.py:76` | Base of strong rallies/drops, ERC-validated, Book P35 disqualification checks | H, L, C, O | MIN_RALLY_CANDLES=3, MAX_ZONES=5, looks at base of move → ~50-100 bars | no | `agents/analysis_agent.py:635` |
| 63 | StopHuntDetector (Day 62, deprecated) | `stop_hunt_detector.py:84` | Sweep + rejection wick + close-back | H, L, C, O + `atr` + liquidity_levels list | REJECTION_LOOKBACK=3, MIN_PENETRATION_ATR=0.05 → ~10 bars after the level | no | `analysis/liquidity_engine.py:55` |
| 64 | MTFAnalyzer (Day 38) | `mtf_analyzer.py:108` | H4+H1+M15+M5 indicator agreement + BOS/CHoCH/sweep + OB/FVG/curve gate | H, L, C + ind_ctx for each TF | `limit=200` per TF | **YES — self-fetches H4/H1/M15/M5** via internal `fetcher.fetch_ohlcv(..., limit=200)` | `analysis/smc_engine.py:101` (reuses helper `_detect_bos/_detect_choch/_detect_liquidity_sweep`) |
| 65 | CurveMTF (Book 5 P126-135) | `curve_mtf.py:1` | Top-down S/D zone curve; HTF bias dominates | Consumes `nearest_demand`/`nearest_supply` dicts (precomputed by SupplyDemandZones) | none (pure value-type over precomputed zones) | **YES — caller passes HTF + LTF zones** | `analysis/mtf_analyzer.py:54` |
| 66 | SessionAnalyzer (Day 63) | `session_analyzer.py:63` | UTC hour + DST-aware NY/London/Tokyo/Sydney windows + DeadZone + London manipulation | (no OHLCV needed; uses `datetime.now(timezone.utc)` unless caller passes `dt`) | n/a | **YES — uses LIVE UTC wall-clock when `dt` param omitted** | `agents/analysis_agent.py:32` (caller passes `dt=_bar_dt` for backtest parity) |
| 67 | LondonManipulationDetector | `session_analysis.py:116` | Asian-range sweep → London fake-breakout reversal | H, L, C + `atr` + **DatetimeIndex required** | needs Asian range (00:00–08:00 UTC) + London session (07/08–10:00 UTC) same day → **≥1 trading day of intraday bars** | **YES — DST-aware hour filtering per candle** | `analysis/liquidity_engine.py:54` |
| 68 | CorrelationEngine | `correlation_engine.py:66` | 28-pair correlation matrix + ATR-volatility regime | close, `atr` | `lookback_periods=50` | **YES — fetches 28 cross pairs from MT5/API** | `agents/analysis_agent.py:52` |
| 69 | IntermarketEngine (Day 65) | `intermarket.py:93` | DXY+Gold+Oil+US10Y+SP500+VIX → risk-on/off + macro bias | n/a (pulls external macro data via `MacroDataProvider`) | n/a | **YES — fetches 6 external assets (DXY, Gold, Oil, US10Y, SP500, VIX)** | `agents/analysis_agent.py:33` |
| 70 | CurrencyStrengthEngine (Day 64) | `currency_strength.py:111` | 28-pair strength normalized to 0-100 + momentum (5-bar history) | close for each of 28 pairs | `candle_limit=100` per pair, `MOMENTUM_LOOKBACK=5` | **YES — fetches 28 cross pairs; DISABLED in live agent (winrate audit 2026-08-13)** | `agents/analysis_agent.py:34` (DISABLED — `if False:` block) |
| 71 | CurrencyRanker | `currency_ranker.py:41` | Pure post-process of strength dict | (none — consumes strength dict) | n/a | no | `analysis/currency_strength.py:103` |
| 72 | FollowThroughEngine (shadow only) | `follow_through_engine.py:1` | Al-Brooks bar-by-bar confirmation of a breakout | H, L, C, O + optional `atr`, `volume` + per-bar timestamp for session weight | eval window up to ~10 bars after breakout | **YES — uses LIVE `datetime.now(timezone.utc)` if no bar timestamp** | NOT wired into live pipeline (per module docstring — shadow/logger mode only) |
| 73 | SignalEngine (production) | `strategy/signal_engine.py:29` | Trend+RSI+MACD+Candle+SR+MTF+pattern+extended votes + HTF EMA50/200 gate + ADX gate + spread filter | close, `ema_50/ema_200/adx/atr`, `spread`/`spread_avg_20` | 200 (EMA200) + 20 (spread avg) | **YES — uses LIVE UTC `timestamp` param for session-aware confidence floor** | `core/runtime.py` via `agents/decision_agent.py` |
| 74 | UnifiedSignalEngine | `unified_signal_engine.py:1` | 7-engine voting fusion (SR + patterns + stop-hunt + ICT AMD + PA + liquidity + CCI) | H, L, C, O, `atr`, optional volume | ~150 bars (each sub-engine's needs) | no (passes df to sub-engines) | `core/runtime.py` |
| 75 | OptimalTradingTime | `optimal_trading_time.py:43` | For each UTC hour, fraction of bars where \|Δclose\| > spread | **bid, ask** (or single mid_col + synthetic spread=0) + **tz-aware DatetimeIndex** | ≥1 full day | **YES — needs bid/ask columns** | NOT wired into live pipeline (standalone tool) |
| 76 | VW-MACD | `vw_macd.py:57` | VWMA(12) − VWMA(26), signal=EMA(9) | close, **volume** (`real_volume` → `tick_volume` → `volume` candidate order) | 26 + 9 = 35 | no | consumed via `extended_modules_adapter` |
| 77 | NadarayaWatson Envelope | `nadaraya_watson_envelope.py:53` | Gaussian-kernel regression + MAD envelope | close | **`window_size=500`** (forward-looking in MQL5 port) — STATED AS REPAINTING by the module itself | no (but the centered window makes it non-causal) | consumed via `extended_modules_adapter` |
| 78 | FVGDetector (Day 46 v2) | `fvg_detector.py:102` | 3-candle imbalance; composite gap_score + decay + regime-aware | H, L, C + `atr` (required) + optional `tick_volume` | TF params: M15 decay_half_life=150 → 150 bars; min 3-candle pattern | no | `analysis/liquidity_engine.py:57`, `analysis/order_block.py:68`, `analysis/smart_money.py:72` |
| 79 | MarketContext (candlestick shared ctx) | `_pattern_context.py:128` | ATR + ATR% + location-in-range + volatility regime + 3-model trend vote + volume_z | H, L, C, optional volume | atr_period=14, volatility_lookback=100, location_lookback=20, volume_lookback=20, SMA50/EMA20/EMA50 | no | `analysis/candlestick_engine.py:47` |

---

## 2. Multi-Timeframe Usage Summary

### 2a. Indicators that THEMSELVES fetch HTF data (caller-independent)

| Engine | HTF timeframes fetched internally | Live fetch | Backtest behavior |
|--------|----------------------------------|------------|-------------------|
| `SMCEngine.analyze()` (`smc_engine.py:109-110`) | H4 + M15, `limit=150` each | Yes — `self.fetcher.fetch_ohlcv(symbol, "4h"/"15m", limit=150)` | Pulls whatever the fetcher returns (CSV/MT5/API) — **must** have H4 + M15 history |
| `MTFAnalyzer.analyze()` (`mtf_analyzer.py:108`) | H4 + H1 + M15 + M5, `limit=200` each | Yes — per-TF `fetch_ohlcv` calls | Same — needs H4/H1/M15/M5 history |
| `SmartMoneyEngine.analyze()` (`smart_money.py:78`) | D1 + H4 + H1 + M15 (full MTF pipeline) | Yes | Same |
| `CurrencyStrengthEngine` (`currency_strength.py:111`) | 28 cross pairs, `candle_limit=100` | Yes | **DISABLED** in agent (winrate audit) |
| `CorrelationEngine` (`correlation_engine.py:66`) | 28 cross pairs, `lookback_periods=50` | Yes | Same |
| `IntermarketEngine` (`intermarket.py:93`) | DXY, Gold, Oil, US10Y, SP500, VIX | Yes — external `MacroDataProvider` | Live only; backtest-mode fallback exists |
| `InstitutionalFlowEngine` (`institutional_flow.py:101`) | CFTC weekly COT (external HTTP) | Yes — `_fetch_cot_from_cftc()` | **Gated OFF in backtest_mode** — uses synthetic large-candle proxy instead |
| `MicrostructureEngine` (`microstructure.py:41`) | Live MT5 tick stream (60-second window) | Yes — `mt5.copy_ticks_range()` | **Gated OFF in backtest_mode** |
| `SessionAnalyzer.get_current_session()` (`session_analyzer.py:63`) | None — uses LIVE `datetime.now(timezone.utc)` unless `dt` passed | Yes (wall-clock) | **Caller (`analysis_agent.py:267-273`) explicitly passes `dt=_bar_dt`** (last bar timestamp) for parity |
| `SignalEngine.generate()` (`signal_engine.py:29`) | None — uses optional `timestamp` param | Yes (wall-clock when `timestamp` param passed) | Caller must pass bar timestamp |

### 2b. Indicators that REQUIRE the caller to pass HTF data

| Engine | What the caller must supply | Where it's wired |
|--------|-----------------------------|------------------|
| `MTFStructureEngine.analyze(df_external, df_internal)` | H4 df + M15 df | `agents/analysis_agent.py:949-1004` (fetches H4 via `self._h4_fetcher.fetch_ohlcv(symbol, "H4", limit=150)` or backtest CSV cache) |
| `CurveMTF.from_zones(nearest_demand, nearest_supply, current_price, atr, regime_ctx)` | Precomputed zone dicts | `analysis/mtf_analyzer.py` REVIEW-A integration |
| `LiquidityEngine.analyze(df, smc_ctx)` | Single OHLCV df with DatetimeIndex spanning ≥1 week (for PDH/PWL/Asian range) | `agents/analysis_agent.py:459` |
| `FibonacciEngine` (optional HTF trend alignment) | Optional H4 df | DISABLED |
| `OrderBlockDetector.detect(df)` | Single df per TF — TF-aware via `timeframe=` param | `analysis/mtf_analyzer.py:120` (one detector per MTF_CHAIN TF) |

### 2c. Single-timeframe indicators (caller passes one df, no HTF)

All remaining indicators in the master table — i.e. `MarketRegimeDetector`, `VolatilityEngine`, `VolumeProfileEngine`, `VolumeConfirmation`, `IchimokuEngine`, `SuperTrend`, `SupplyDemandZones`, `SMCAdvancedEngine`, `StopHuntDetector`, `SupportResistance`, `MarketStructure`, `LiquidityPoolAnalyzer`, `LiquidityStructureAnalyzer`, `LiquidityZoneMapper`, `FVGDetector`, `CandlestickEngine`, `MarketContext`, `VW-MACD`, `NadarayaWatson`, `SupplyDemandZones`, `OptimalTradingTime`.

---

## 3. Session Detection Logic (with timezone)

### 3a. Source of truth

| Module | Source TZ | DST handling | Notes |
|--------|-----------|--------------|-------|
| `analysis/session_rules.py:10-36` | **GMT / UTC** (hardcoded integer hour boundaries) | None at this layer — `SESSION_WINDOWS` dict has fixed `start`/`end` UTC hours | Raw definition; DST adjustment happens in `SessionAnalyzer` |
| `analysis/session_analyzer.py:63` (`get_current_session`) | **UTC** (`datetime.now(timezone.utc)` if `dt=None`) | **YES** — `_get_dst_flags(dt)` uses `zoneinfo.ZoneInfo("America/New_York")` and `ZoneInfo("Europe/London")` for true DST-aware NY/London session shift | When US DST active: NY 12:00-21:00 UTC; otherwise 13:00-22:00 UTC. When EU DST active: London 07:00-16:00 UTC; otherwise 08:00-17:00 UTC. Tokyo always 00:00-09:00 UTC, Sydney 22:00-07:00 UTC (crosses midnight) |
| `analysis/session_analysis.py:46-80` (`_is_eu_dst_for_timestamp`) | **UTC** (per-candle, from `df.index.hour`) | **YES** — `zoneinfo.ZoneInfo("Europe/London")` per timestamp, fallback to fixed-date heuristic | London open window: 07:00-10:00 UTC in summer (BST), 08:00-10:00 UTC in winter (GMT) |
| `analysis/liquidity_zones.py:250-288` (`asian_session_range`) | **UTC** (uses `df.index.hour`) | None — Asian range defined as `0 <= hour < 8` UTC | Hardcoded `start_hour=0, end_hour=8` UTC |
| `analysis/smart_money.py:42-47` (`KILL_ZONES`) | **UTC** | None — hardcoded hour tuples | `LONDON_OPEN=(7,10)`, `NEW_YORK_OPEN=(12,15)`, `LONDON_CLOSE=(15,17)` — **note: London open at 07 UTC is summer-only; in winter London opens 08 UTC and these labels are off by 1 hour** |
| `analysis/follow_through_engine.py:81-100` (`_infer_session`) | **UTC** | None — fixed hours | Asian 00:00-07:00, London 07:00-16:00, NY 12:00-21:00, Overlap 12:00-16:00 |
| `analysis/dead_zones` (`session_rules.py:55-58`) | **UTC** | None | 22:00-24:00 UTC (Sydney open) + 00:00-02:00 UTC (early Tokyo); `DEAD_ZONES_ENABLED=True` |

### 3b. Key parity risk

`session_analyzer.SessionAnalyzer.get_current_session(dt=None)` defaults to **`datetime.now(timezone.utc)`** — REAL wall-clock time, NOT the historical bar's timestamp. In backtest mode this silently stamps every historical bar with whatever session is live RIGHT NOW on the operator's clock. The `agents/analysis_agent.py:259-273` block fixes this by computing `_bar_dt = df.index[-1].to_pydatetime()` from the OHLC frame and passing it explicitly — but ONLY at the analysis_agent call site. Any other direct caller of `SessionAnalyzer.analyze(pair, ...)` without `dt=` will get wall-clock behavior (parity bug).

`signal_engine.SignalEngine.generate()` has the same pattern: an optional `timestamp` param. If a caller omits it, the session-aware confidence floor uses wall-clock UTC.

---

## 4. Top 10 Longest Lookbacks

Lookback = minimum bars the indicator needs **on the lowest timeframe it runs on** to be non-NaN at the most recent bar.

| Rank | Lookback (bars) | Indicator / module | File:Line | TF used at runtime | Notes |
|------|------------------|---------------------|-----------|---------------------|-------|
| 1 | **500** | NadarayaWatson Envelope `window_size` | `nadaraya_watson_envelope.py:58` | per-call (typically H1/H4) | Module self-documents as REPAINTING (forward-looking centered window in MQL5 port); `nwe_stable=False` for last `window_size` bars |
| 2 | **300** | MarketAgent primary fetch | `agents/market_agent.py:186` (`get_candles(..., limit=300)`) | primary TF (M15 or H1) | Sets the floor for any indicator that reads from this df |
| 3 | **220** | Ichimoku Senkou B (52) + displacement (26) + Chikou compare | `ichimoku.py:60` + `ichimoku.py:214-238` | per-call | Effective minimum: 52 + 26 + 5 ≈ 83 |
| 4 | **200** | SMA(200), EMA(200), trend composite | `data/indicators.py:25,30`; `data/indicators_ext.py:289`; `market_regime.py:199` | per-call | Required for `trend` label + SignalEngine HTF gate (`ema_50` vs `ema_200`) |
| 5 | **200** | `MTFAnalyzer` per-TF fetch (`limit=200`) | `analysis/mtf_analyzer.py:427`; `analysis/timeframe.py:72` | H4/H1/M15/M5 | 200 bars × 4 TFs = 800 bars fetched per cycle |
| 6 | **150** | `SMCEngine` per-TF fetch (`limit=150`) | `analysis/smc_engine.py:109-110` | H4 + M15 | 150 bars × 2 TFs = 300 bars |
| 7 | **150** | `MTFStructureEngine` H4 fetch + `OrderBlockDetector` decay | `agents/analysis_agent.py:997`; `order_block.py:87`; `fvg_detector.py:81` | H4 + M15 | OrderBlock M15 `decay_half_life=150`; H4=300; D1=400 |
| 8 | **120** | `VolatilityEngine.squeeze_lookback` | `volatility.py:58` | per-call | BB-width percentile computed over 120 bars |
| 9 | **100** | `EXTERNAL_LOOKBACK` (liquidity_structure) | `liquidity_structure.py:51` | per-call | Bars searched for "the" external high/low |
| 10 | **100** | `volatility_lookback` (candlestick MarketContext) + `AdvancedPatternDetector(lookback=100)` + `CurrencyStrengthEngine(candle_limit=100)` | `_pattern_context.py:159`; `analysis_agent.py:478`; `currency_strength.py:97` | per-call | Three independent 100-bar windows |

### Effective MAX_REQUIRED_LOOKBACK derivation

The codebase has **NO central `MAX_REQUIRED_LOOKBACK` constant** — it is implicitly the maximum of:

1. The 300-bar `MarketAgent` fetch (the floor for the primary df)
2. The 200-bar MTF chain per TF (H4/H1/M15/M5 → needs 200 H4 bars for `mtf_analyzer`, which on M15 granularity ≈ 200 × 16 = 3200 M15 bars equivalent)
3. The 500-bar NadarayaWatson `window_size` (when used)
4. PDH/PWL history implicit requirement: 1 week of intraday bars on the lowest TF
5. Asian range: ≥1 trading day of intraday bars (00:00-08:00 UTC same day)
6. FVG/OrderBlock decay: 150-400 bars depending on TF

**Practical floor for live parity**: at least **500 bars** on the primary timeframe + **200 bars** on each HTF in `MTF_CHAIN` (H4, H1, M15, M5). For `NadarayaWatson` consumers, **500 bars** minimum.

---

## 5. Features That Require Bid / Ask / Spread / Tick Volume / Real Volume (NOT reproducible from OHLC alone)

| Feature | Module | Required non-OHLC field | Notes |
|---------|--------|-------------------------|-------|
| **Spread state (WIDE/EXTREME)** | `analysis/microstructure.py:216-237` | **bid, ask** (per-tick, from MT5 `copy_ticks_range`) | Live MT5 tick stream — 60-second window; gated OFF in backtest |
| **Tick speed (BURST/DEAD)** | `analysis/microstructure.py:192-214` | **tick timestamps** (MT5 native) | Same — bypasses OHLC entirely |
| **Volume burst (BURST/NORMAL)** | `analysis/microstructure.py:239-255` | **tick volume** (MT5 `volume` / `volume_real`) | Same |
| **Price acceleration pips/sec** | `analysis/microstructure.py:257-277` | **bid, ask, tick time** | Same |
| **Spread filter in SignalEngine** | `strategy/signal_engine.py:114-120`; `data/indicator_registry.py:226-263` | **`spread` column** (pips, per-bar) + 20-bar rolling avg | Computed from bid/ask if present, else 0; **when 0, spread filter silently no-ops** (parity risk — historical CSVs without spread will silently disable this gate) |
| **OptimalTradingTime coverage** | `analysis/optimal_trading_time.py:43-100` | **bid, ask** columns (or `mid_col` + synthetic spread=0) | Not wired into live pipeline — standalone tool |
| **VWAP** | `data/indicators_ext.py:503-528` | **volume** + DatetimeIndex | Falls back to cumulative TP×vol if no DatetimeIndex; if no volume, **silently skipped** (column absent) |
| **VWMA, MFI, OBV, A/D Line, CMF, Volume RSI** | `data/indicators_ext.py:308-644` | **volume** (tick_volume or real_volume) | All silently skipped when `volume` column absent |
| **VolumeProfileEngine** | `analysis/volume_profile.py:77-152` | **volume** or **tick_volume** | Falls back to uniform volume=1.0 (POC still computed but less accurate) |
| **VolumeConfirmation** | `analysis/volume_confirmation.py:52-173` | **volume** | Returns NO_VOLUME_PENALTY=-5 adjustment when volume missing |
| **VW-MACD** | `analysis/vw_macd.py:57` | **volume** (real_volume preferred, then tick_volume, then volume) | Cannot compute without volume |
| **CandlestickEngine volume_z** | `analysis/_pattern_context.py:167-173` | **volume** or **tick_volume** | Volume z-score — optional, degrades gracefully to `volume_z=None` |
| **FollowThroughEngine volume nudge** | `analysis/follow_through_engine.py:48-52` | optional **volume** (MT5 tick_volume convention) | Shadow mode only |
| **InstitutionalFlowEngine COT data** | `analysis/institutional_flow.py:101-203` | **External CFTC weekly CoT report** | Live HTTP fetch; **gated OFF in backtest_mode** — falls back to synthetic large-candle proxy from OHLC |
| **CurrencyStrengthEngine** | `analysis/currency_strength.py:111` | close of **28 cross pairs** (live fetch) | DISABLED in agent — `if False:` block |
| **CorrelationEngine** | `analysis/correlation_engine.py:66` | close of **28 cross pairs** + `atr` | Live fetch required |
| **IntermarketEngine** | `analysis/intermarket.py:93` | **DXY, Gold, Oil, US10Y, SP500, VIX** external feeds | Live `MacroDataProvider` fetch |
| **NewsApiProvider** | `analysis/news_api_provider.py` | External news API | Live only; backtest skips |

### Practical parity verdict

To reproduce every live signal exactly in backtest you need at minimum:
1. **OHLCV** on every TF in `MTF_CHAIN` (H4/H1/M15/M5) for ≥500 bars on the lowest TF, ≥200 on each HTF.
2. **tick_volume** column (MT5 convention) on every TF — without it: VWAP, OBV, MFI, CMF, A/D, VWMA, Volume RSI, VolumeProfile, VolumeConfirmation, VW-MACD all silently no-op or degrade.
3. **bid/ask spread** column on the primary TF — without it: SignalEngine spread filter silently no-ops, MicrostructureEngine falls back to "PROCEED" recommendation, OptimalTradingTime cannot run.
4. **DatetimeIndex in UTC** on every TF — without it: Asian range, PDH/PDL/PWH/PWL, session analysis, London manipulation, kill-zone tagging all skip or fall back to wall-clock.
5. **External feeds**: DXY, Gold, Oil, US10Y, SP500, VIX history for `IntermarketEngine`; 28 cross-pair history for `CurrencyStrengthEngine`/`CorrelationEngine`; CFTC weekly COT for `InstitutionalFlowEngine`.

---

## 6. Datetime / Look-Ahead / Leakage Risks

### 6a. Direct wall-clock usage (`datetime.now(timezone.utc)` or `datetime.utcnow()`)

| File:Line | Use | Risk | Mitigation already in place |
|-----------|-----|------|------------------------------|
| `analysis/session_analyzer.py:83` | `dt = datetime.now(timezone.utc)` when `dt=None` | Backtest stamps every historical bar with current live UTC session | `agents/analysis_agent.py:259-273` passes `dt=_bar_dt` from `df.index[-1]` |
| `analysis/smart_money.py:390` | `now = datetime.now(timezone.utc)` for kill-zone check | Same | Caller must pass historical `df` — KillZone check uses `now.hour` against `df.index` indirectly; **NO direct bar-timestamp pass** — POTENTIAL LEAK |
| `analysis/microstructure.py:88,154,297` | `datetime.now(timezone.utc)` for tick fetch window + timestamps | Live-only by design | `is_backtest_mode()` guard returns fallback |
| `analysis/institutional_flow.py:273,396,459,487` | COT fetch timestamp + data_age_days = now - report_date | Live COT data applied to historical bars | `is_backtest_mode()` guard at line 127 skips COT fetch, uses synthetic proxy |
| `analysis/intermarket.py:124` | `datetime.now(timezone.utc)` timestamp | Pure log timestamp | No parity impact |
| `analysis/currency_strength.py:179,436,484` | `datetime.utcnow().isoformat()` (deprecated API) | Pure log timestamp; `utcnow()` returns naive datetime | No parity impact; should migrate to `datetime.now(timezone.utc)` for tz-aware consistency |
| `analysis/market_dna.py:173,243` | `datetime.now(timezone.utc)` for `trained_at` / `as_of` | Training-time metadata | No parity impact unless DNA model feature extraction depends on it |
| `analysis/retail_sentiment.py:202,343` | `datetime.now(timezone.utc).isoformat()` | Pure log timestamp | No parity impact |
| `analysis/final_frontier.py:141,361,404,470` | `datetime.now(timezone.utc)` for timestamps | Pure log/metadata | No parity impact |
| `analysis/myfxbook_sentiment.py:335,346,864,892,1011` | Cache timestamp + `fetched_at` | Live-only external sentiment data | No parity impact; backtest should skip |
| `analysis/news_api_provider.py:103,125,207,324,380,396,463` | `datetime.now(timezone.utc)` for daily count reset + `from_date` window + `fetched_at` | Live news fetch | Backtest should skip / freeze the clock |
| `analysis/market_state_memory.py:107,130` | `datetime.now(timezone.utc)` for state-update timestamps | In-memory state only | No parity impact |
| `analysis/shadow_follow_through_logger.py:147,180` | `datetime.now(timezone.utc)` for log timestamps | Shadow-mode logging only | No parity impact |
| `data/automated_updater.py:73`, `data/fetcher.py:710,1222,1684,1783`, `data/main.py:357,445`, `data/verify_data_coverage.py:124,184,185,476` | Various fetch-time timestamps | Data fetch layer (not in analysis path) | Out of scope for analysis parity |

### 6b. Specific look-ahead / leakage risks identified

1. **`SignalEngine.generate(timestamp=None)`** (`strategy/signal_engine.py:43`) — the new session-aware confidence floor uses wall-clock UTC if `timestamp` is omitted. Backtest callers MUST pass the historical bar's UTC timestamp for parity. **Verify every caller.**

2. **`SmartMoneyEngine.analyze()`** (`analysis/smart_money.py:390`) — uses `datetime.now(timezone.utc)` for kill-zone detection without a `df.index[-1]` fallback. Live mode is fine; backtest mode stamps every historical bar with the current live kill-zone, not the bar's actual UTC hour. **POTENTIAL LEAK — needs the same `_bar_dt` pattern as `SessionAnalyzer`.**

3. **`NadarayaWatson Envelope`** (`analysis/nadaraya_watson_envelope.py:25-29`) — the module self-documents as **REPAINTING**: the MQL5 port uses a forward-looking centered window (`reg[i]` uses bars `[i, i+1, ..., i+W-1]` — bars AFTER i). The `nwe_stable` flag marks the last `window_size=500` bars as unstable. Any consumer that reads `nwe_mid/nwe_upper/nwe_lower` on the most recent bar will see revised values as new bars arrive. **For backtest parity, consumers MUST either: (a) only read values where `nwe_stable=True`, or (b) recompute the envelope on an expanding window per bar.**

4. **`MarketRegimeDetector._add_adx`** uses `ewm(alpha=1/period, adjust=False)` — that is Wilder's smoothing, which is causal. No leakage here.

5. **`VolumeProfileEngine._build_profile`** (`analysis/volume_profile.py:158-199`) — uses `df[["open", "high", "low", "close"]].min().min()` / `.max().max()` over the **whole df**. Caller must pass an expanding slice `df.iloc[:i+1]` per bar for walk-forward parity; calling once on a full historical df is a look-ahead (the bin range is set by future highs/lows).

6. **`LiquidityZoneMapper.calculate_previous_levels`** (`analysis/liquidity_zones.py:180-192`) — uses `unique_days[-2]` / `unique_weeks[-2]` (the second-to-last completed period). This is **causally correct** as long as the df's last row is a fully closed candle. If the caller passes a df whose last row is the in-progress (forming) bar, PDH/PDL silently use today's high/low rather than yesterday's. **Caller must ensure `df.iloc[-1]` is a closed candle.**

7. **`MarketStructure._detect_fractals`** (`analysis/market_structure.py:108-120`) — swing confirmation requires `strength` bars AFTER the swing. The module's "Causality Contract" (file:9-16) explicitly states this. No leakage **if** callers respect `confirmed_at = i + strength`.

8. **`SupportResistance.analyze`** (`analysis/support_resistance.py:651-770`) — "Rejection counts and strength scores are computed on the *full* supplied DataFrame (current live strength). For pure historical walk-forward, slice the DataFrame up to the decision bar before calling analyze()" — explicit caller contract in the file header (lines 18-23).

9. **`OrderBlockDetector._scan_fill`** / **`FVGDetector._scan_fill`** — both simulate fill by walking forward through subsequent bars. No leakage as long as caller passes `df.iloc[:i+1]` (expanding window). The module docstrings (e.g. `order_block.py:50-57`) explicitly defer the multi-TF orchestration upward to `mtf_analyzer.py` to avoid a second source of truth.

10. **`SessionAnalyzer.get_current_session` DST computation** — `_get_dst_flags(dt)` uses `zoneinfo.ZoneInfo("America/New_York")` and `ZoneInfo("Europe/London")` for true DST-aware shifts. **However**, the DST check is per-call — if the caller passes a historical `dt`, DST is correctly resolved for that historical moment. ✅ No leakage.

### 6c. Hardcoded timezone offsets

No hardcoded numeric TZ offsets (e.g. `+05:30`, `-05:00`) were found anywhere in the analysis layer. All timezone logic uses either `datetime.now(timezone.utc)`, `pd.DatetimeIndex.tz_localize("UTC")`, or `zoneinfo.ZoneInfo("America/New_York" / "Europe/London")`. ✅ Clean.

---

## 7. Summary of Required Lookbacks per Timeframe

| Timeframe | Min bars required (live parity floor) | Notes |
|-----------|----------------------------------------|-------|
| M1 (when used) | 500 | NadarayaWatson window; OB decay_half_life=80 |
| M5 (entry timing) | 200 | MTFAnalyzer limit=200 |
| M15 (setup detection) | 300 | MarketAgent primary fetch + OrderBlock decay_half_life=150 + Ichimoku 52+26 |
| M30 (when used) | 200 | MTFAnalyzer equivalent |
| H1 (zone confirmation) | 300 | MarketAgent primary + 200-bar MTF + Ichimoku |
| H4 (trend direction) | 200 | MTFAnalyzer + SMCEngine + 300 decay_half_life on OB |
| D1 (swing trade bias) | 200 | CurveMTF swing-style + 400 decay_half_life on OB |
| W1 (position bias) | 50 | CurveMTF position style |

**Plus**: at least **1 full trading week** of intraday bars on the primary TF (for PDH/PDL/PWH/PWL + Asian range computation in `LiquidityZoneMapper`).

---

## 8. Actionable Items for Historical CSV Pipeline (Phase 2+)

1. **Persist `tick_volume` column** on every TF CSV — required by 7 indicators in `indicators_ext.py` + VolumeProfile + VolumeConfirmation + VW-MACD. Without it these silently skip.
2. **Persist `spread` column** (in pips) on the primary TF CSV — required by `SignalEngine` spread filter (lines 114-120). Without it the filter silently no-ops.
3. **Persist UTC `DatetimeIndex`** on every TF CSV — required by Asian range, PDH/PDL/PWH/PWL, session analysis, London manipulation, VWAP, kill zones.
4. **Persist at least 500 bars** on the primary TF, **200 bars** on each HTF in `MTF_CHAIN` (H4/H1/M15/M5).
5. **Persist DXY/Gold/Oil/US10Y/SP500/VIX history** for `IntermarketEngine` — currently fetched live only.
6. **Persist 28 cross-pair close history** for `CurrencyStrengthEngine` / `CorrelationEngine` (currently fetched live; CurrencyStrength is DISABLED but Correlation is live).
7. **Migrate every `datetime.now(timezone.utc)` / `datetime.utcnow()` call in the analysis path to accept a `bar_timestamp` parameter** (the pattern already used at `analysis_agent.py:259-273`). Audit list: `smart_money.py:390`, `signal_engine.py:43` (already optional), `microstructure.py` (already gated), `institutional_flow.py` (already gated), `news_api_provider.py`, `myfxbook_sentiment.py`.
8. **Document / enforce the NadarayaWatson repaint contract** — either remove from live consumers, mark the most recent `window_size=500` bars as STABLE=False, or recompute on an expanding window in backtest.
9. **Document / enforce the `SupportResistance.analyze` caller contract** (lines 18-23 of `support_resistance.py`) — walk-forward callers must slice `df.iloc[:i+1]` before calling.
10. **Document / enforce the `VolumeProfileEngine` caller contract** — same expanding-window requirement (the bin range uses `df.min().min()` / `df.max().max()` over the whole df).
11. **Add a centralized `MAX_REQUIRED_LOOKBACK` constant** derived from this audit (currently 500 bars = NadarayaWatson window_size; or 300 bars = MarketAgent fetch floor — whichever is larger per TF). The fetcher should always fetch `max(MAX_REQUIRED_LOOKBACK, caller-specified limit)` to prevent warmup-NaN parity bugs.
