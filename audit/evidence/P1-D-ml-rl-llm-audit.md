# P1-D — ML / RL / LLM Audit Report

**Task ID:** P1-D
**Auditor:** Explore sub-agent
**Scope:** `ml/`, `rl/`, `ai/`, `core/llm_*`, `core/ollama_validator.py`, `intelligence/`, `agents/*_agent.py`
**Repo root:** `/home/z/my-project/download/forex-agent`
**Goal:** enumerate every data dependency the ML/RL/LLM engines require so the historical CSV pipeline can reproduce them offline / live with full parity.

---

## 0. Executive Summary

| Subsystem | Status | Critical live data dependencies |
|---|---|---|
| **ML feature engineering** (`ml/feature_engineer.py`) | ~110 flat features from OHLCV + 11 analysis-out contexts | last 200+ bars OHLCV; latest indicator columns (rsi/atr/macd/ema_9/20/50/sma_200/bb_high/bb_low); 8 currency strengths; intermarket trends (DXY/Gold/VIX/SP500/US10Y); session; news_intelligence; SMC ctx; confluence ctx |
| **ML training pipeline** (`ml/dataset_builder.py`, `ml/model_trainer.py`) | SQLite feature_store (`memory/ml_features.db`) + chronological 70/15/15 split | Persisted feature rows; `label` column produced by `LabelGenerator` (ATR-scaled fixed-horizon) or `TripleBarrierLabeler` (path-dependent). Purged split (`ml/cv_splitter.py`) optional but recommended |
| **ML inference path** (`ml/model_predictor.py`, called from `agents/analysis_agent.py`) | **DISABLED in live code (`if False:` at `analysis_agent.py:2001`)** | When re-enabled: needs `full_feature_vector` dict (110 keys) + loaded `memory/ml_models/{PAIR}_{TF}/xgboost_v*.pkl` + scaler.pkl + per-model calibrated threshold |
| **Label generation** (`ml/label_generator.py`, `ml/triple_barrier_labels.py`, `ml/fast_triple_barrier.py`) | ATR-scaled forward horizon (4/8/16 bars) AND triple-barrier (TP/SL/timeout) | Future OHLC over horizon window. ATR is strictly backward-looking (no leak). Reverse-rolling MAE/MFE used (corrected leak) |
| **RL state representation** (`ml/rl_environment.py` v1, `ml/rl_environment_v2.py`) | Box(n_features) where n_features = `len(features_df.columns) + 6` | Per-step: row of `features_df` (rich feature set built from OHLCV — see §5) + 6 position/account floats. V2 fallback schema if no features_df: `[close, high, low, volume, rsi_14, atr, macd, ema_20, ema_50, sma_200] + 6` |
| **RL action space** (`rl/action_masking.py`) | Discrete(4) HOLD/BUY/SELL/CLOSE in env; portfolio action-map supports up to N×(2×max_short + 2×max_long)+1 discrete ops | None — purely agent-side |
| **RL reward** (`rl/reward_functions.py` softplus, `ml/reward_engine.py`, `ml/reward_engine_v2.py`) | Profit/loss USD + ATR-based SL/TP + spread cost + slippage | `pnl_usd = (close - entry)/pip_size * 10 * lot` (live close price from `df.iloc[current_step]["close"]`); spread_pips × pip_size deducted on entry; slippage_pips on close (V2 only) |
| **LLM cascade** (`ai/ai_analyst.py`, `agents/master_analyst.py`) | Primary Groq → Gemini → OpenRouter; Ollama REMOVED from cascade (kept as opt-in veto gate in `core/ollama_validator.py`) | Indicator context (close/ema9/sma20/rsi/macd/macd_signal/atr/bb_position); pattern ctx; SR ctx; regime ctx; rule-engine signal ctx; MTF bias; advanced pattern ctx; session; intermarket; sentiment; news; SMC; fib; vision; bias; memory; (MasterAnalyst adds: divergence/ichimoku/vol_profile/smc_advanced/mtf_structure/strategy/news_api/econ_calendar/fred/retail_sentiment) |
| **LLM cache** (`core/llm_cache.py`) | In-memory `OrderedDict`, TTL=300 s (5 min), max=200 entries, provider-agnostic key (sha256 of "any" + prompt) | None |
| **News/sentiment** (`intelligence/news_sources.py`, `intelligence/news_ai.py`, `intelligence/sentiment_model.py`) | Forex Factory JSON + central-bank schedule + 4 RSS feeds (DailyFX/ForexLive/Investing/MarketWatch) + local `data/economic_calendar.json`; sentiment via Groq/Gemini LLM on each headline | All times normalized to UTC ISO 8601; cached 5 min in-memory + persisted to `memory/news_analysis_memory.jsonl` |

**Key parity risks for the historical CSV pipeline:**
1. `ml/institutional_feature_adapter.py` requires `>= 220` warmup bars; live `MarketAgent` fetches only `limit=300` — OK but tight.
2. `ml/model_predictor.py` is **commented out at runtime** (`analysis_agent.py:2001` `if False:`), so ML is currently **not contributing to live decisions**. Any historical CSV used to retrain must reproduce *all* the same upstream contexts (session/intermarket/news/SMC/confluence) so re-enabled inference matches what training saw.
3. Feature engineer consumes BOTH raw OHLCV columns AND pre-computed indicator columns (rsi_14, atr, macd, macd_signal, ema_9/20/50, sma_200, bb_high, bb_low). If historical CSV lacks any of these, `_indicator_features` falls back to on-the-fly computation, but the on-the-fly path **silently returned 0.0 in earlier versions** — see the "AUDIT FIX" notes in `feature_engineer.py:191-313`. CSV provider MUST include all canonical indicator columns.
4. News fetch is **skipped entirely in backtest mode** (`news_sources.py:181` `if is_backtest_mode(): return local_calendar`), so historical CSVs cannot reproduce the same `news_intelligence` context that live trading saw — the historical pipeline must supply `data/economic_calendar.json` covering the historical window.
5. Timezone: every "now" in the feature engineer uses `pd.Timestamp.now(tz="UTC")` (`feature_engineer.py:400`). Historical replay must override this — see §11.

---

## 1. ML Feature Registry (`ml/feature_engineer.py`)

`FeatureEngineer.build_feature_vector(df, analysis_out, pair, timeframe)` returns a flat dict of ~110 numeric features for the **last candle** of `df`. Inputs are a pandas OHLCV DataFrame (`df`, with optional pre-computed indicator columns) and a dict of analysis contexts (`analysis_out`).

### 1.1 Price features (`_price_features`, lines 126–183)

| Feature | Inputs | Lookback | Source TF |
|---|---|---|---|
| `price_open`, `price_high`, `price_low`, `price_close`, `price_volume` | OHLCV of last candle | 1 bar | current TF |
| `candle_body`, `candle_range`, `candle_body_ratio`, `candle_upper_wick`, `candle_lower_wick` | last candle OHLC | 1 bar | current TF |
| `change_1`, `change_3`, `change_5`, `change_10`, `change_20` | `close` vs `close.iloc[-n-1]` | n+1 bars | current TF |
| `distance_ma9/20/50/200` | `(close - ma_n) / ma_n`, reads `sma_n` or `ema_n` column if present | up to 200 bars | current TF |
| `high_5/20/50`, `low_5/20/50`, `range_5/20/50_pips` | `df.tail(n).high.max()` etc. | n bars | current TF |

### 1.2 Indicator features (`_indicator_features`, lines 187–315)

| Feature | Inputs | Lookback | Notes |
|---|---|---|---|
| `rsi_7/14/21` | reads `rsi_n` column OR on-the-fly `delta.rolling(n).mean()` | 21 bars max | Audit-fix: rsi_14 used to silently default to 0.0 if no precomputed column |
| `rsi_overbought`, `rsi_oversold` | derived from `rsi_14` | 14 bars | thresholds 70/30 |
| `macd`, `macd_signal`, `macd_diff`, `macd_hist` | reads `macd`/`macd_signal` columns OR on-the-fly `ema_12 - ema_26` then `ewm(span=9)` | 26+9 bars | Audit-fix: was constant 0.0 on raw OHLCV |
| `macd_cross_up`, `macd_cross_down` | compares `macd.iloc[-1]` vs `macd_signal.iloc[-1]` and previous bar | 2 bars | |
| `bb_position` | reads `bb_high` + `bb_low` columns | 20 bars (BB std window) | defaults to 0.5 if columns absent — **silent degradation** |
| `atr`, `atr_percentage`, `atr_pips` | reads `atr`/`atr_14` column OR on-the-fly `tr.rolling(14).mean()` | 14 bars | Audit-fix: was constant 0.0 on raw OHLCV |
| `volume_ratio` | `last.volume / df.volume.tail(20).mean()` | 20 bars | defaults to 1.0 if no volume |
| `ema_9/20/50_distance`, `ema_bullish_alignment`, `ema_bearish_alignment` | reads `ema_n` column OR on-the-fly `close.ewm(span=n).mean()` | up to 50 bars | Audit-fix: was constant 0.0 on raw OHLCV |

### 1.3 Pattern features (`_pattern_features`, lines 319–379)

| Feature | Inputs | Lookback |
|---|---|---|
| `pat_doji`, `pat_hammer`, `pat_shooting_star`, `pat_pin_bar`, `pat_bullish_engulfing`, `pat_bearish_engulfing`, `pat_morning_star`, `pat_evening_star` | last candle + df column with same name (string) | 1–3 bars |
| `adv_head_and_shoulders`, `adv_inverse_head_and_shoulders`, `adv_double_top`, `adv_double_bottom`, `adv_ascending_triangle`, `adv_descending_triangle`, `adv_symmetrical_triangle`, `adv_bull_flag`, `adv_bear_flag`, `adv_rising_wedge`, `adv_falling_wedge`, `adv_cup_and_handle` | `analysis_out["advanced_pat_ctx"]["recent_patterns"]` list | historical patterns list |
| `fib_236/382/500/618/786_distance_pips`, `in_fib_zone`, `fib_zone` | `analysis_out["fib_ctx"]["retracements"]` + last close | requires Fib ctx |

### 1.4 Context features (`_context_features`, lines 383–503)

| Feature | Inputs | Lookback |
|---|---|---|
| `session_london/new_york/tokyo/sydney/asian/between_sessions/dead_zone` (7 one-hot), `session_overlap`, `session_trade_quality` | `analysis_out["session_ctx"]` | real-time UTC clock |
| `hour_utc`, `day_of_week`, `is_weekend`, `is_monday_open`, `is_friday_close` | `pd.Timestamp.now(tz="UTC")` | live wall clock — **breaks historical replay** |
| `news_blocked`, `news_confidence_change`, `hours_to_news`, `high_impact_nearby`, `news_bullish/bearish/neutral` | `analysis_out["news_intelligence"]` (next_high_impact_event.minutes_until, news_bias) | news feed |
| `currency_strength_base/quote/gap`, `eur_strength/usd_strength/gbp_strength/jpy_strength/aud_strength/cad_strength/chf_strength/nzd_strength` (8 currencies) | `analysis_out["intermarket_ctx"]["macro_pair_bias"]` or per-currency keys | derived from macro model |
| `dxy_trend`, `gold_trend`, `vix_level`, `vix_fear_elevated`, `sp500_trend`, `us10y_trend`, `macro_score`, `macro_regime_risk_on/off`, `cross_asset_confirmed` | `analysis_out["intermarket_ctx"]` | external market data |
| `sr_location`, `near_support`, `near_resistance` | `analysis_out["sr_ctx"]` | latest SR scan |

### 1.5 Multi-timeframe features (`_mtf_features`, lines 507–529)

| Feature | Inputs | Lookback |
|---|---|---|
| `mtf_bullish/bearish/neutral` | `analysis_out["mtf_bias"]` (string) | HTF trend alignment |
| `smc_trend_aligned`, `smc_grade_a_plus`, `smc_grade_a` | `analysis_out["smc_ctx"]["trend_aligned"]`, `["grade"]` | SMC HTF analysis |

### 1.6 SMC + Liquidity features (`_smc_liquidity_features`, lines 533–655)

When `analysis_out["smc_ctx"]` is present (live path):
| Feature | Inputs |
|---|---|
| `smc_buy/sell/neutral`, `smc_confluence_score`, `bos_detected`, `choch_detected`, `order_block_tap`, `fvg_detected`, `liquidity_sweep`, `liquidity_sweep_bullish/bearish` | `smc_ctx` dict |

When `smc_ctx` is **absent** (historical training fallback, `_smc_from_price_action`):
- BOS: `close > prior_high` of last `swing_lookback=20` candles (excluding current)
- FVG: 3-candle imbalance between `df.iloc[-3]` and `df.iloc[-1]`
- Order block: last opposite-colored candle before impulsive move (range ≥ 1.5× avg of last 20)
- Liquidity sweep: current bar wicks beyond prior swing extreme but closes back inside
- All lookups use `df.tail(swing_lookback + 1)` — strictly backward-looking (no leak)

### 1.7 Confluence + sentiment features (`_confluence_features`, lines 659–731)

| Feature | Inputs |
|---|---|
| `sentiment_final_score`, `sentiment_bullish/bearish` | `analysis_out["sentiment_ctx"]["final_score"]`, `["bias"]` |
| `confluence_buy_score`, `confluence_sell_score`, `confluence_net_score`, `confluence_aligned_factors`, `confluence_total_factors`, `confluence_confidence`, `quality_a_plus/a/b/avoid` | `analysis_out["confluence"]` |
| `master_buy/sell/wait`, `master_confidence` | `analysis_out["master_ctx"]["master_signal"]`, `["master_confidence"]` |
| `llm_buy/sell`, `llm_confidence` | `analysis_out["llm"]` (AIAnalyst output) |
| `rule_buy/sell`, `rule_confidence` | `analysis_out["signal"]` (rule engine output) |

### 1.8 Institutional feature adapter (`ml/institutional_feature_adapter.py`)

Separate schema used by `ml/train_historical.py` and `data/trained_models/`. Built via `ml/pipeline/phase3_features._add_*_features`:
- Trend: `ema_8/21/50/200`, `sma_20/50/200`, `vwap`, `supertrend`, `supertrend_dir`, `adx`, `atr`
- Momentum: `rsi_14/7`, `macd`, `macd_signal`, `macd_hist`, `stoch_k/d`, `cci_20`, `roc_10/20`, `williams_r`
- Volume: `obv`, `obv_ema`, `cmf_20`, `mfi_14`, `vol_sma_20`, `vol_ratio`
- Volatility: `bb_upper/mid/lower/width/pct`, `donchian_upper/lower/mid`, `keltner_upper/lower/mid`, `realized_vol_10/20/50`
- Market structure: `swing_high_10/20`, `swing_low_10/20`, `liquidity_zone_above/below`, `fvg_bullish/bearish`, `bos_bullish/bearish`, `choch_bullish/bearish`, `ob_bullish/bearish`, `nearest_support`, `nearest_resistance`, `sr_distance`
- Session: `session_london/newyork/tokyo/overlap`
- Time: `hour_sin/cos`, `day_of_week`, `day_sin/cos`, `week_of_year`, `month_sin/cos`

`MIN_WARMUP_BARS = 220` (longest lookback is sma_200 + 20-bar buffer).

---

## 2. ML Training Pipeline (`ml/dataset_builder.py`, `ml/model_trainer.py`, `ml/data_preprocessor.py`)

### 2.1 Training data path (diagram)

```
                  ┌────────────────────────────────────────────────┐
                  │  ml/mt5_data_loader.MT5DataLoader.fetch()       │  ← 100k bars from MT5
                  │  (chunked copy_rates_from_pos, validates OHLCV, │     (or manual CSV in data/history/)
                  │   removes incomplete last candle, tz=UTC)      │
                  └─────────────────────┬──────────────────────────┘
                                        ▼
                  ┌────────────────────────────────────────────────┐
                  │  ml/pipeline/phase3_features.compute_features() │
                  │  → adds ~80 indicator columns (trend/momentum/  │
                  │    volume/volatility/structure/session/time)    │
                  └─────────────────────┬──────────────────────────┘
                                        ▼
                  ┌────────────────────────────────────────────────┐
                  │  ml/pipeline/phase4_labels.generate_labels()    │  forward close.shift(-horizon)
                  │  OR ml/label_generator.LabelGenerator           │  + ATR-scaled threshold
                  │  OR ml/triple_barrier_labels.TripleBarrierLabeler│  + path-dependent TP/SL/timeout
                  └─────────────────────┬──────────────────────────┘
                                        ▼
                  ┌────────────────────────────────────────────────┐
                  │  ml/feature_store.FeatureStore (SQLite)         │  memory/ml_features.db
                  │  save_features(features=dict, label, source=   │  table: features + labels + importance
                  │    'live'|'bootstrap')                          │
                  └─────────────────────┬──────────────────────────┘
                                        ▼
                  ┌────────────────────────────────────────────────┐
                  │  ml/dataset_builder.DatasetBuilder.build_from_  │
                  │  store() → load_training_data() (excludes      │
                  │  source='bootstrap' by default)                │
                  │  → chronological 70/15/15 split                 │
                  │  → optional PurgedEmbargoedSplitter             │
                  │     (label_horizon = 4/8/16 bars)               │
                  └─────────────────────┬──────────────────────────┘
                                        ▼
                  ┌────────────────────────────────────────────────┐
                  │  ml/data_preprocessor.DataPreprocessor          │
                  │  → drop_invalid (NaN/inf)                      │
                  │  → chronological_split (NO shuffle)            │
                  │  → fit_clip_bounds + fit_scaler on TRAIN ONLY  │
                  │  → transform val/test with train-fitted stats  │
                  └─────────────────────┬──────────────────────────┘
                                        ▼
                  ┌────────────────────────────────────────────────┐
                  │  ml/model_trainer.ModelTrainer                 │
                  │   ├─ XGBClassifier(n=600, max_depth=4,         │
                  │   │    lr=0.03, scale_pos_weight=neg/pos,     │
                  │   │    early_stopping=30 on val)               │
                  │   ├─ RandomForestClassifier(n=400, depth=8,    │
                  │   │    class_weight="balanced_subsample")      │
                  │   └─ LSTM (Sequential, 64→32→16→1)             │
                  │  → _find_optimal_threshold (F1-max on X_val)    │
                  │  → ml.model_evaluator.evaluate() on X_test      │
                  │  → save_model() with cv_tags {labeling_method, │
                  │    cv_method} persisted to meta.json            │
                  └─────────────────────┬──────────────────────────┘
                                        ▼
                  memory/ml_models/{PAIR}_{TF}/{xgboost|random_forest|lstm}_v*.pkl
                  + _registry.json (portable relative paths)
```

### 2.2 Required columns for training

| Stage | Required columns |
|---|---|
| `mt5_data_loader._process_rates` | MT5 raw: `time`, `open`, `high`, `low`, `close`, `tick_volume` → renamed `tick_volume → volume` |
| `phase3_features.compute_features` | `open`, `high`, `low`, `close`, `volume` (plus a tz-aware UTC `DatetimeIndex` for session/time features) |
| `LabelGenerator.label_dataframe` | `close` (for forward return), `high` + `low` (for forward MAE/MFE) |
| `TripleBarrierLabeler.label_dataframe` | `high`, `low`, `close` |
| `DatasetBuilder.build_from_dataframe` (triple_barrier mode) | `high`, `low`, `close`, plus a `label` column produced by the labeler |
| `DatasetBuilder.build_from_dataframe` (fixed_horizon mode) | only `label` column (already on the loaded df) |
| `ModelTrainer._train_*` | all feature columns (non-`_` prefix, non-meta), already z-scored by preprocessor |

### 2.3 Train/val/test split — leakage controls

- **Chronological 70/15/15**, no shuffle (`DatasetBuilder.build_from_dataframe` lines 326–377).
- `use_purged_split=True` + `label_horizon>0` activates `PurgedEmbargoedSplitter.purge_train_val_test` (`ml/cv_splitter.py:69+`) which:
  - drops training rows whose `[i, i+h]` label window overlaps the val/test boundary;
  - applies an embargo gap after the test window;
  - refuses to silently continue if purging removes >50 % of the fold (`MAX_PURGE_RATIO_WARNING`).
- **Threshold calibration** done on `X_val` only, never `X_test` — `_find_optimal_threshold` (lines 186–222) — explicit leakage guard.
- `DataPreprocessor.fit_clip_bounds` and `fit_scaler` both **fit on TRAIN only** then transform val/test (audit-fix; previously mean/std were computed on full dataset → test-set leak, lines 86–104).
- Bootstrap rows tagged `source='bootstrap'` are excluded by default (`load_training_data` line 223 `WHERE source != 'bootstrap'`).

### 2.4 Leakage risks found in training path

See §11.

---

## 3. ML Inference Path (`ml/model_predictor.py`)

### 3.1 Runtime data flow (diagram)

```
MarketAgent.run()                                ← live cycle, every 60 s
  │  self._orchestrator.get_candles(symbol, tf, limit=300)   ← MT5 or API fallback
  │  → df (300 OHLCV bars, tz-aware UTC)
  │  → add_canonical_indicators(df)                ← adds rsi_14, macd, atr, ema_*, bb_*, etc.
  ▼
AnalysisAgent.run(df=df, ...)
  │  engineers analysis_out dict with 12 contexts (session, smc, sr, regime,
  │    intermarket, sentiment, news_intelligence, mtf_bias, confluence, ...)
  │
  │  engineer.build_feature_vector(df, unified_for_features, symbol, timeframe)
  │  → ~110-feature dict for the LAST candle only
  │
  │  store.save_features(features=full_feature_vector, label=...)   ← persisted to SQLite
  │
  │  ★★★ predictor.predict(features=full_feature_vector, pair, tf)
  │       — DISABLED: `if False:` at analysis_agent.py:2001 ★★★
  │       (memory/ml_models/_registry.json is empty as of 2026-08-13 audit)
  │
  ▼
DecisionAgent / EnsembleEngine → final BUY/SELL/WAIT
```

### 3.2 What ML predictor needs "right now" (when enabled)

| Need | Source | Lookback |
|---|---|---|
| `features: Dict[str, float]` (110 keys) | `FeatureEngineer.build_feature_vector(df, analysis_out, pair, tf)` | **last 200+ bars** of OHLCV on `df` (sma_200 / ema_50 etc.) |
| `memory/ml_models/{PAIR}_{TF}/{xgboost\|random_forest\|lstm}_v*.pkl` | `ModelStore.load_model()` (latest version) | model file on disk |
| `memory/ml_processed/scaler.pkl` | `DataPreprocessor.load_scaler()` (means+stds dict) | scaler file on disk |
| Per-model calibrated threshold | `ModelStore.get_latest_metrics(pair, tf, model_type)["threshold"]` | meta.json |
| Expected feature-name list | `ModelStore.get_feature_names(pair, tf, model_type)` | meta.json |
| Optional `df_recent` (≥ 220 bars) | If no registered model, falls through to `_predict_institutional()` which calls `institutional_feature_adapter.build_institutional_features(df_recent)` — uses the separate phase3 schema | 220 bars |

### 3.3 Lookback window summary

- **Minimum live lookback** = 300 bars (hard-coded `limit=300` in `MarketAgent` line 186). This satisfies sma_200 + 100 buffer.
- **Institutional fallback minimum** = 220 bars (`MIN_WARMUP_BARS` in `institutional_feature_adapter.py`).
- **`_smc_from_price_action` swing lookback** = 20 bars (default `swing_lookback=20`).
- For triple_barrier labels at training time, the last `holding_period` rows of `df` get NaN labels (no forward window) and are dropped.

---

## 4. Label Generation (`ml/label_generator.py`, `ml/triple_barrier_labels.py`, `ml/fast_triple_barrier.py`, `ml/dual_binary_model.py`)

### 4.1 Label types

| Module | Label type | Horizon | Future data used? | Leakage-safe? |
|---|---|---|---|---|
| `ml/label_generator.LabelGenerator.label_dataframe` (default `horizon=4`) | Ternary {BUY=+1, WAIT=0, SELL=-1} via ATR-scaled forward-pip threshold | 4 / 8 / 16 bars | `close.shift(-horizon)` for forward return; reverse-rolling `.shift(-1).[::-1].rolling(horizon).max()` for forward MAE/MFE | **YES** — ATR is Wilder-smoothed using only `df.iloc[:row_idx+1]` (line 188). MAE/MFE corrected for off-by-one in audit-fix (lines 290–302). |
| `ml/triple_barrier_labels.TripleBarrierLabeler.label_dataframe` | Path-dependent {+1 TP, -1 SL, 0 timeout} | `holding_period=10` default | Loops `i in range(n-holding_period)`, scans `high.iloc[i+1..i+h]` and `low.iloc[i+1..i+h]` | **YES** — barrier scan is purely forward; ATR computed via `_atr()` (Wilder, backward only). |
| `ml/fast_triple_barrier.fast_triple_barrier_labels` | Same semantics, vectorized | `holding_period=16` default | Uses `np.lib.stride_tricks.sliding_window_view(high, h)[1:n-h+1]` | **YES** — same first-touch logic; same-bar double-touch resolves to SL (conservative), unlike the loop version's optimistic TP tie-break. |
| `ml/dual_binary_model.create_target` | Two binary classifiers (BUY target + SELL target) | `forward_bars=5`, `threshold_pct=0.001` | `out[close_col].shift(-forward_bars)` | YES — explicitly documented as having NaN tail. |
| `ml/triple_barrier_labels.fixed_horizon_labels` | Ternary based on `close.shift(-horizon) / close - 1 > threshold` | `horizon=10` | forward close only | YES |
| `ml/triple_barrier_labels.meta_labels` | Binary "was primary signal correct" | `horizon=10` | `returns.shift(-horizon)` | YES |
| `ml/triple_barrier_labels.mfe_mae` | MFE / MAE ratios | `holding_period=10` | `high[i+1..i+1+H]` and `low[i+1..i+1+H]` | YES |

### 4.2 Future-peeking in label code

**All label computations use `shift(-N)` — this is correct for LABELS, not for features.** The label generator's docstring (lines 39–49) explicitly states: "forward_return / forward_pips / mae_pips / mfe_pips / signal class use ONLY future candles relative to the feature row (row_idx+1 .. row_idx+horizon). This is the only place future data is allowed — and only for creating training labels, never for inference features."

The `cv_splitter.PurgedEmbargoedSplitter` exists specifically to prevent these forward windows from crossing train/test boundaries.

---

## 5. RL State Representation

### 5.1 RL feature schema (V2, `ml/train_rl_v2.build_features_df_v2`)

`ForexTradingEnvV2.__init__` sets `self.n_features = len(features_df.columns) + 6`. The features_df columns produced by `build_features_df_v2(df)`:

| Category | Features | Lookback |
|---|---|---|
| Multi-horizon momentum | `ret_1`, `ret_4`, `ret_16`, `ret_96` | up to 96 bars |
| Candle geometry | `body_pct`, `upper_wick_pct`, `lower_wick_pct`, `engulf_ratio` | 1–2 bars |
| Trend (MA distance + slope + cross) | `dist_ema20`, `dist_ema50`, `dist_sma200`, `ema20_slope`, `ema_cross` | up to 200 bars |
| Momentum oscillators | `rsi_14`, `rsi_slope`, `macd`, `macd_hist`, `stoch_k` | 14–26 bars |
| Volatility | `atr_pct`, `realized_vol`, `bb_width`, `bb_pct_b` | 14–20 bars |
| Volume | `volume_z` (rolling z-score) | 100 bars |
| Session timing | `hour_sin/cos`, `dow_sin/cos` | — (DatetimeIndex required) |

Total market features: **23**. Plus 6 position/account state features:
1. `is_long` (1/0)
2. `is_short` (1/0)
3. `entry / 10000` (normalized)
4. `balance / initial_balance`
5. `trades_today / 20`
6. `(peak_balance - balance) / peak_balance` (drawdown)

**Total RL state vector dimension (V2): 23 + 6 = 29.** (`Box(low=-inf, high=inf, shape=(29,), dtype=float32)`)

### 5.2 RL feature schema (V1 fallback, `ml/rl_environment._get_state`)

When `features_df is None`, env uses hard-coded `FEATURE_SCHEMA = ["close", "high", "low", "volume", "rsi_14", "atr", "macd", "ema_20", "ema_50", "sma_200"]` → 10 market features + 6 account = **16-dim state**.

### 5.3 Raw market data needed for RL state construction

- **OHLCV** for `build_features_df_v2`: `open`, `high`, `low`, `close`, `volume` (V1 needs the same plus pre-computed `rsi_14`, `atr`, `macd`, `ema_20`, `ema_50`, `sma_200` columns OR they are added by `load_historical_data_v2` lines 91–108).
- **DatetimeIndex** (tz-aware UTC) for session/time features (`hour_sin/cos`, `dow_sin/cos`).
- Minimum history: ≥ 500 rows (`train_rl_v2.py:270` `if df.empty or len(df) < 500: return error`).

### 5.4 RL state/action/reward table

| Component | V1 (`ml/rl_environment.py`) | V2 (`ml/rl_environment_v2.py`) |
|---|---|---|
| **State** | 16-dim Box (10 market + 6 account) — fixed `FEATURE_SCHEMA` | 29-dim Box (23 market + 6 account) — dynamic from `features_df` |
| **Action space** | `Discrete(4)`: 0=HOLD, 1=BUY, 2=SELL, 3=CLOSE | `Discrete(4)` same |
| **Action masking** (`rl/action_masking.py`) | Separate `build_action_map` / `make_action_mask` / `apply_mask` — used by external RBOT-style envs, NOT by `ForexTradingEnv(V2)` (which has its own internal guards `can_enter`, `cooldown_bars`, `max_trades_per_day`) | Same |
| **Reward engine** | `RewardEngine.calculate()` — profit_reward × 5, loss_penalty × 5, risk_bonus, overtrade_penalty, drawdown_penalty, hold_reward (capped at 20 idle steps), hacking_penalty | `RewardEngineV2.calculate()` — adds transaction_cost, streak_bonus, quality_shaping; uses `initial_balance` for scaling (never current balance — bugfix #1) |
| **Reward function inputs** | `pnl_usd` (realized, USD), `balance`, `initial_balance`, `risk_pct`, `rr_ratio`, `trades_today`, `peak_balance`, `position_open` | All V1 inputs + `trade_closed`, `win`, `spread_cost_usd` |
| **Reward function raw data needed** | Per-step `close` price (to compute PnL on close), `high`/`low` (for SL/TP intrabar hit check), `atr` (for SL/TP sizing) | Same + `pip_size` (per-pair: 0.01 JPY/metals, 0.0001 standard), `spread_pips` (default 1.5), `slippage_pips` (default 0.5, V2 only) |
| **Reward function spread-awareness** | Spread cost deducted from PnL: `entry ± (spread_pips × pip_size / 2)` (lines 284–292) | Spread + slippage: `entry ± (spread_pips + slippage_pips) × pip_size / 2` (lines 429–437); spread cost also subtracted from reward as `transaction_cost` |
| **Episode termination** | End-of-data OR bankrupt (`balance ≤ 20% × initial`) | End-of-data OR `max_steps_per_episode=1000` OR bankrupt (`balance ≤ 50% × initial`) |
| **Per-pair pip_size** | Hardcoded `0.0001` in env constructor (bug for JPY/metals) | `infer_pip_size(pair)` — JPY/metals → 0.01, standard → 0.0001 |

---

## 6. LLM Modules

### 6.1 Provider cascade

| Module | Primary | Fallback 1 | Fallback 2 | Optional | Disabled |
|---|---|---|---|---|---|
| `ai/ai_analyst.py` (`AIAnalyst.analyze`) | **Groq** (`llama-3.1-8b-instant` via env `GROQ_MODEL`) | **Gemini** (`gemini-flash-lite-latest` via env `GEMINI_MODEL`) | **OpenRouter** (env `OPENROUTER_MODEL`, fallbacks 1+2) | Cerebras (`OC_INCLUDE_CEREBRAS=1`), SambaNova (`OC_INCLUDE_SAMBANOVA=1`) | **Ollama** (`OLLAMA_ENABLED = False` hardcoded; `_call_ollama` returns None — line 580+) |
| `agents/master_analyst.py` (`MasterAnalyst._call_llm`) | Groq → Gemini → OpenRouter (same cascade as AIAnalyst) | | | | Ollama removed (2026-07-25 per user request "Ollama akebare sese") |
| `intelligence/sentiment_model.py` (`SentimentModel.analyze`) | Groq → Gemini → rule-based fallback (cap: 5 LLM calls/cycle) | | | | |
| `core/ollama_validator.py` (`OllamaValidator.validate`) | Ollama (`qwen3:4b`, `OLLAMA_VALIDATOR_ENABLED=false` default off) | — | — | — | When enabled, runs as final veto gate AFTER DecisionValidator (fail-open) |
| `core/llm_key_manager.py` | Multi-key rotation per provider; classifies 429s as TPD (daily) vs RPM/TPM (transient); auto-disables decommissioned models | | | | |
| `core/llm_cache.py` | In-memory LRU `OrderedDict` keyed by `sha256(provider="any" + model + prompt)[:16]`, TTL=300s, max=200 entries | | | | |

### 6.2 LLM context fields fed in (raw market data YES/NO?)

#### 6.2.1 `AIAnalyst._build_context` (lines 484–537)

| Field group | Fields | Raw market data? |
|---|---|---|
| Header | `SYMBOL`, `TIMEFRAME` | No |
| Price & Trend | `close`, `trend`, `ema9`, `sma20` | **YES** — last close + MA values from `ind_ctx` |
| Momentum | `rsi (14)`, `macd_signal`, `macd_value` | **YES** — from `ind_ctx` |
| Volatility | `atr`, `bb_position` | **YES** — from `ind_ctx` |
| Patterns | `recent_patterns`, `advanced_pat_ctx`, `pattern_signal` | Derived from OHLC |
| Support/Resistance | `location`, `nearest_support`, `nearest_resistance`, `pivot_pp` | Derived from OHLC |
| Market Regime | `regime`, `direction`, `strength`, `volatility`, `adx` | Derived |
| Rule Engine Signal | `signal`, `confidence`, `entry`, `blocked_by`, `reasons` | Derived |
| Multi-Timeframe | `mtf_bias` | Derived |

#### 6.2.2 `MasterAnalyst._build_context` (lines 435–700+)

In addition to AIAnalyst's fields, MasterAnalyst feeds in:
- `sentiment_ctx` (final_score, bias, retail_long_pct, fg_label, dxy_trend, sentiment_reasons)
- `news_ctx` (trade_allowed, upcoming_events, risk_level)
- `memory_ctx` (overall_win_rate, total_trades, recent_results, lessons)
- `smc_ctx` (signal, direction, score, grade, h4_ob_zone, h4_fvg_zone, h4_bos, h4_choch)
- `fib_ctx` (fib_zone, fib_in_golden, fib_signal)
- `vision_ctx` (vision_trend, vision_confidence)
- `session_ctx` (current_session, session_volatility, session_strategy, session_trade_allowed, session_min_confidence, session_risk_mult) — Day 63
- `intermarket_ctx` (DXY/Gold/VIX/SP500/US10Y trends, macro_pair_bias, cross_asset_confirmed) — Day 65
- `classic_llm_ctx` (Round-13: AIAnalyst's verdict passed in for agreement/disagreement check)
- `divergence_ctx`, `ichimoku_ctx`, `volatility_ctx`, `volume_profile_ctx`, `smc_advanced_ctx`, `mtf_structure_ctx`, `strategy_ctx`
- `news_api_ctx` (Day 92: NewsAPI real-time sentiment)
- `econ_calendar_ctx`, `fred_ctx`, `retail_sentiment_ctx`

**Raw market data reaches the LLM via**: `close_price`, `atr`, `bb_pct`, `rsi`, `macd_cross`, `nearest_support`, `nearest_resistance`, `pivot`, `smc_h4_ob_zone`, `smc_h4_fvg_zone` — all derived from OHLC but **price levels are explicitly cited as concrete numbers** in the prompt (e.g. "EXACT pip distance to nearest resistance"). The prompt forbids treating structural levels as confluence (audit-fix noted in `_build_context` line 705+).

#### 6.2.3 `SentimentModel.analyze` (lines 218+)

Feeds ONLY the news headline/snippet text. **No raw market data** — purely NLP on the headline text.

#### 6.2.4 `OllamaValidator.validate` (lines 280+)

Receives the full `market_data: Dict[str, Any]` from `DecisionValidator` (caller passes "symbol, timeframe, trend, ml_confidence, spread, session, market_structure, support_resistance, order_blocks, etc."). Plus proposed signal/confidence/entry/sl/tp/rr. **Raw market data YES** — the validator gets the same context the DecisionValidator sees, including concrete entry/SL/TP prices.

### 6.3 LLM caching

`core/llm_cache.py`:
- **Cached entity**: raw LLM response string (the JSON payload before parsing).
- **Cache key**: `sha256(provider + "|" + model + "|" + prompt)[:16]`. AIAnalyst uses `provider="ai_analyst"`, `model="any"` (provider-agnostic, audit-fix at line 320).
- **TTL**: 300 seconds (5 minutes), hard-coded in `LLMCache(ttl_sec=300, max_entries=200)` line 155.
- **Max entries**: 200. Eviction: LRU FIFO popitem at capacity, after proactive expired-key cleanup.
- **Persistence**: in-memory only (`OrderedDict`). No disk persistence — cache is lost on restart.
- **Stats tracked**: hits, misses, hit_rate, tokens_saved_est, ttl_sec, current_entries.

---

## 7. News / Sentiment Data Flow

### 7.1 Sources (`intelligence/news_sources.py`)

| Source | URL / Path | Mechanism | Timezone | Cached? |
|---|---|---|---|---|
| Forex Factory economic calendar | `https://nfs.faireconomy.media/ff_calendar_thisweek.json` | HTTP GET (10 s timeout), filters HIGH/MEDIUM impact | FF returns ISO 8601 with `+00:00` → normalized to UTC | 300 s in `_cache["calendar"]` |
| Central bank schedule (hard-coded) | `CENTRAL_BANK_EVENTS` list in `news_sources.py` lines 76–99 | Computed from `datetime.now(timezone.utc)` (3rd Wed for FOMC, 2nd Thu for ECB, etc.) | UTC (defaults to 18:00 UTC) | 300 s in `_cache["central_bank"]` |
| RSS feeds | DailyFX `https://www.dailyfx.com/feeds/all`; ForexLive `https://www.forexlive.com/feed/`; Investing `https://www.investing.com/rss/news_25.rss`; MarketWatch `https://feeds.content.dowjones.io/public/rss/mw_topstories` | `requests.get`, parsed with BeautifulSoup (`lxml` preferred, `html.parser` fallback), up to 15 items per feed | RFC-822 `%a, %d %b %Y %H:%M:%S %Z` or ISO 8601 → normalized to UTC; **bugfix**: unknown pubDate → `time_iso=None` (was: `datetime.now(utc)` which silently stamped stale headlines as fresh) | 300 s in `_cache["rss"]` |
| Local economic calendar | `data/economic_calendar.json` | File read (maintained by `broker/economic_calendar.py`) | Times assumed UTC; `datetime.fromisoformat` then `replace(tzinfo=timezone.utc)` if naive | Always available — **only source used in backtest mode** (`is_backtest_mode()` short-circuits the FF/RSS fetch at line 181) |

### 7.2 Storage

- **In-memory**: `_cache` dict in `NewsSources` (300 s TTL per source).
- **Disk**: `memory/news_analysis_memory.jsonl` — append-only JSONL with one entry per news analysis cycle. Each entry has `"ts": datetime.now(timezone.utc).isoformat(timespec="seconds")` and the full `NewsBiasReport.to_dict()`. Outcome records also written here (`"type": "outcome"`) for accuracy tracking.

### 7.3 Sentiment analysis (`intelligence/news_ai.py`, `intelligence/sentiment_model.py`)

```
NewsSources.fetch_all_flat(hours_ahead=24)
   → List[NewsItem]
      ▼
EventClassifier.classify(item) → EventClassification
      ▼
SentimentModel.analyze(item.headline) → SentimentResult
   (Groq/Gemini LLM call, 5 calls/cycle cap, 2 s throttle, fallback to rule-based)
      ▼
CurrencyImpactEngine.apply(tone, currency) → per-pair bias
      ▼
NewsBiasReport {
   next_high_impact_event,
   pair_biases,           # pair → BULLISH/BEARISH/NEUTRAL
   pair_confidence_adjustments,  # pair → ±delta
   blocked_pairs,         # pair → block reason
   sentiment_summary,
   total_events_analyzed, high_impact_count, sources_used, details
}
```

---

## 8. ML / RL Leakage Risk Audit

### 8.1 `shift(-N)` usage (forward-looking) — FEATURE vs LABEL

| File:line | Code | Verdict |
|---|---|---|
| `ml/label_generator.py:283` | `future_close = df["close"].shift(-horizon)` | **LABEL** — correct, this is the only place forward data is allowed |
| `ml/label_generator.py:297-300` | `future_high = df["high"].shift(-1); future_high[::-1].rolling(horizon).max()[::-1]` | **LABEL** — MAE/MFE forward window. Audit-fixed to start at row+1 (was previously including past candles) |
| `ml/triple_barrier_labels.py:141` | `forward_return = close.shift(-horizon) / close - 1` | **LABEL** — `fixed_horizon_labels()` standalone function |
| `ml/triple_barrier_labels.py:163` | `forward_return = returns.shift(-horizon)` | **LABEL** — `meta_labels()` standalone function |
| `ml/triple_barrier_labels_OPTIONAL_spread_aware.py:167,189` | Same pattern | **LABEL** — optional variant, not in default path |
| `ml/dual_binary_model.py:100` | `future_close = out[close_col].shift(-forward_bars)` | **LABEL** — `create_target()` for dual-binary classifiers |
| `ml/pipeline/phase4_labels.py:65,73,77` | `future_close = close.shift(-horizon)`, `future_low = low.shift(-1).rolling(horizon).min()`, `future_high = high.shift(-1).rolling(horizon).max()` | **LABEL** — institutional pipeline label phase |
| `tools/ml/triple_barrier_labels.py:141,163` | Same as ml/triple_barrier_labels.py | **LABEL** — duplicate module under `tools/` |
| `scripts/train_models_quick.py:244` | `df["target"] = (df["close"].shift(-horizon) > df["close"]).astype(int)` | **LABEL** — script-level training |
| `scripts/train_missing_pairs_fast.py:139` | `df["target"] = (df["close"].shift(-5) > df["close"]).astype(int)` | **LABEL** — script-level training |
| `scripts/ml_pipeline_audit.py:150,170,195,203` | `shift(-horizon)` for target; explicitly logged as "INTENTIONAL for LABELS" | **LABEL** — diagnostic script, self-documented |
| `ai/automated_retraining.py:368` | `target = all_data[pair_col].pct_change().shift(-1)` | **LABEL** — automated retraining target (next-day return) |

**Verdict:** Every `shift(-N)` instance in the codebase is for **label generation**, not feature generation. All are correctly used as targets and dropped (or NaN-handled) before training via `DatasetBuilder.build_from_dataframe`'s `idx_label_present = labels_full.notna()` filter (line 282).

### 8.2 Future-peeking in feature code — NONE FOUND in active paths

- `ml/feature_engineer.py`: every rolling/ewm/shift is backward-looking. The forward-looking `future_window` (line 199) is inside `label_for_row`, not `build_feature_vector`. **`_smc_from_price_action`** explicitly uses `prior = window.iloc[:-1]` (line 605, "excludes current bar → no look-ahead").
- `ml/pipeline/phase3_features.py`: audit-fix at line 289 removed `center=True` from swing-high/low rolling windows ("center=True means bar i's value depends on p/2 bars of FUTURE data. This silently leaked future info"). Now uses `center=False` (default).
- `ml/train_rl_v2.build_features_df_v2`: all features use `pct_change`, `ewm`, `rolling`, `shift(1)` — strictly backward-looking. Comment at line 144: "Every feature here uses only the current bar and strictly earlier bars (rolling/ewm/shift), so there is no future leakage."

### 8.3 HTF → LTF resample-merge leakage risk

- `FeatureEngineer._mtf_features` reads `analysis_out["mtf_bias"]` as a single string ("BULLISH"/"BEARISH"/"NEUTRAL") from upstream `MultiTimeframeAnalyzer`. No HTF→LTF row merging inside the feature engineer.
- `MultiTimeframeAnalyzer.analyze(["1d", "4h", "1h", "15m"])` is called separately by `MarketAgent` (line 170). It computes HTF trend independently and returns a single bias string — no merge of HTF bars into the LTF dataframe. **No leak path identified.**
- `FeatureEngineer.aggregate_multi_timeframe` in `ml/feature_selector.py` exists as a utility but is NOT called by the live feature build path.

### 8.4 Train/test split leakage risks (active)

| Risk | File:line | Status |
|---|---|---|
| Naive chronological 70/15/15 (no purge) — label window can cross boundary | `ml/dataset_builder.py:372-377` | **MITIGATED** — `use_purged_split=True` (default in `ensemble_train.py`) activates `PurgedEmbargoedSplitter` with `label_horizon=4` |
| Clip bounds fit on full dataset (pre-split) | `ml/data_preprocessor.py:116-131` (`clean_features` deprecated path) | **MITIGATED** — `process()` now uses `fit_clip_bounds(X_train)` then `apply_clip(X_test)` (lines 207-210) |
| Scaler fit on full dataset | same file | **MITIGATED** — `fit_scaler(X_train)` only, `transform()` applied to both (lines 212-214) |
| Threshold calibrated on test set | `ml/model_trainer.py:186-222` | **MITIGATED** — `_find_optimal_threshold` uses `X_val` only, never `X_test` |
| Bootstrap rows mixed into training | `ml/feature_store.py:223` | **MITIGATED** — `WHERE source != 'bootstrap'` by default; only `data_bootstrap.bootstrap_feature_store_if_needed` writes `source='bootstrap'` rows |
| `analysis_agent.py:1942-1951` (audit-fix log) | Training on features built from the same decision that produced the label | **MITIGATED** — `unified_for_features` dict explicitly excludes `master_ctx`, `confluence`, `signal`, `llm` (kept only genuinely pre-decision market-state context) |
| `reward_engine.py` balance scaling with current balance (could go negative → sign flips) | `ml/reward_engine.py:111` | **MITIGATED in V2** — `RewardEngineV2` uses `initial_balance` for scaling (V2 bugfix #1); V1 still uses `balance` but is no longer the production path |
| `ForexTradingEnv` (V1) SL/TP only checked on HOLD action | `ml/rl_environment.py:210-216` | **MITIGATED in V2** — `rl_environment_v2.py:326-331` checks SL/TP on every step where a position is open, before action processing |
| Reward engine singleton shared between train and eval envs (cross-talk on streak counters) | `ml/train_rl_v2.py:306-319` | **MITIGATED** — each env gets its own `RewardEngineV2()` instance |

### 8.5 Residual leakage risks (worth monitoring)

1. **`ml/data_preprocessor.clean_features` (deprecated, lines 116-131)** still exists and could be called by external code. It fits clip bounds on whatever `df` is passed in. If any external caller passes train+test combined, it leaks. Consider removing or marking as `@deprecated`.
2. **`scripts/train_models_quick.py:244`** uses `df["close"].shift(-horizon) > df["close"]` as the target, with `horizon=4`. The script does NOT use `PurgedEmbargoedSplitter` (it uses sklearn `train_test_split` with `shuffle=False`). If this script is the production training path (rather than `ModelTrainer.train_all`), label-window leakage at the train/test boundary is possible. **Verify which script the live models were trained with.**
3. **`ai/automated_retraining.py:368`** uses `pct_change().shift(-1)` (next-day return) as the regression target. The TimeSeriesSplit is correct (`sklearn.model_selection.TimeSeriesSplit`), but there's no purge gap between folds — fold-i training labels can overlap fold-i+1's first training rows. Low-severity because of regression target + LSTM sequence windowing, but worth noting.
4. **Feature engineer `pd.Timestamp.now(tz="UTC")` for `hour_utc` / `day_of_week` / `is_weekend` / `is_monday_open` / `is_friday_close`** (lines 400-405). These are wall-clock features computed against the live clock, NOT against the timestamp of the last bar in `df`. In historical replay/backtest, this produces **wrong** time features (the live clock is now, not the bar's time). This is the **single biggest parity gap** between live and historical feature vectors. The historical CSV pipeline MUST supply a `df.index` that is a tz-aware UTC `DatetimeIndex`, and the feature engineer must be patched to use `df.index[-1]` instead of `Timestamp.now()` when replaying.

---

## 9. Required Historical Fields for ML/RL/LLM Reproduction

### 9.1 ML training reproduction (per symbol/timeframe)

| Field | Required by | Notes |
|---|---|---|
| `time` (tz-aware UTC) | session/time features, chronological split | MT5 returns Unix seconds → `pd.to_datetime(unit='s', utc=True)` |
| `open`, `high`, `low`, `close` | All ML features, all labels, all RL state | Standard OHLC |
| `tick_volume` (renamed → `volume`) | `volume_ratio`, RL `volume_z` | MT5 tick volume; volume_z uses 100-bar rolling mean/std |
| (optional) `real_volume` | Not used by current feature engineer | MT5 real volume is rarely populated for FX |
| (optional) `spread` | NOT persisted in MT5 fetch — `rl_environment` uses a hardcoded `spread_pips=1.5` constructor arg | For per-pair spread parity, the historical CSV should include `spread` column; otherwise V2 env uses the default 1.5 pips |
| (optional) `bid`/`ask` | NOT used by ML/RL/LLM | Only used by execution layer (`core/execution_adapter.py`) |
| Pre-computed indicator columns: `rsi_7`, `rsi_14`, `rsi_21`, `macd`, `macd_signal`, `atr`, `atr_14`, `ema_9`, `ema_20`, `ema_50`, `sma_200`, `bb_high`, `bb_low` | `FeatureEngineer._indicator_features` (preferred path; on-the-fly fallback exists but adds compute cost) | If absent, feature engineer computes on the fly — but **must ensure the on-the-fly path is exercised in training AND in live** to avoid schema mismatch (audit-fix notes at `feature_engineer.py:191-313`) |
| Intermarket data: DXY, Gold, VIX, SP500, US10Y trends (UP/DOWN/FLAT strings + numeric vix_value) | `_context_features` lines 463-487 | Sourced from `analysis_out["intermarket_ctx"]` — historical pipeline must reconstruct this from separate DXY/XAU/VIX/SPX/US10Y CSVs |
| Currency strength: per-currency `eur_strength`, `usd_strength`, `gbp_strength`, `jpy_strength`, `aud_strength`, `cad_strength`, `chf_strength`, `nzd_strength` (0-100 floats) | `_context_features` lines 453-461 | Sourced from `analysis_out["intermarket_ctx"]` — historical pipeline must reconstruct from `analysis/currency_strength.py` which reads separate currency-pair feeds |
| News events: `next_high_impact_event.minutes_until`, `news_bias`, `confidence_change`, `blocked` | `_context_features` lines 407-432 | **In backtest mode, `NewsSources` skips live fetch and uses only `data/economic_calendar.json`. The historical pipeline MUST supply `economic_calendar.json` covering the historical window.** |
| `session_ctx.current_session`, `trade_quality` | `_context_features` lines 387-394 | Derived from UTC hour-of-week; reproducer can compute deterministically |
| `smc_ctx.signal/grade/bos/choch/order_block/fvg/liquidity_sweep/confluence_score/trend_aligned` | `_smc_liquidity_features` | Sourced from `analysis/smc_engine.py` (live) OR `_smc_from_price_action` (training fallback, computed from OHLC) |
| `fib_ctx.retracements` (23.6/38.2/50.0/61.8/78.6 levels) | `_pattern_features` lines 356-376 | Sourced from `analysis/fibonacci.py` |
| `sentiment_ctx.final_score`, `bias` | `_confluence_features` | Sourced from `intelligence/news_ai.py` |
| `confluence.buy_score/sell_score/net_score/aligned_factors/total_factors/confidence/setup_quality` | `_confluence_features` | Sourced from `intelligence/confluence_engine.py` |
| `master_ctx.master_signal`, `master_confidence` | `_confluence_features` | Sourced from `agents/master_analyst.py` (LLM output) |
| `llm.signal`, `llm.confidence` | `_confluence_features` | Sourced from `ai/ai_analyst.py` (LLM output) |
| `signal.signal`, `signal.confidence` (rule engine) | `_confluence_features` | Sourced from rule-based signal engine |

### 9.2 RL training reproduction (per symbol/timeframe)

| Field | Required by | Notes |
|---|---|---|
| `open`, `high`, `low`, `close`, `volume` (or `tick_volume`) | `build_features_df_v2` | OHLCV |
| `atr` (or pre-computed) | `ForexTradingEnvV2._open_position` line 413 | If absent, `load_historical_data_v2` computes it via 14-period rolling mean of true range |
| `rsi_14` (or pre-computed) | V1 fallback `FEATURE_SCHEMA` | If absent, `load_historical_data_v2` computes it |
| `macd`, `ema_20`, `ema_50`, `sma_200` (or pre-computed) | V1 fallback `FEATURE_SCHEMA` | Computed on-the-fly by `build_features_df_v2` for V2 |
| Tz-aware UTC `DatetimeIndex` | `build_features_df_v2` lines 229-238 (`hour_sin/cos`, `dow_sin/cos`) | **Critical**: if `df.index` is not `DatetimeIndex`, session features become `np.zeros(n)` — RL agent loses time-of-day signal |
| Per-pair `pip_size` | `ForexTradingEnvV2` constructor | JPY pairs: 0.01; metals (XAU/XAG/XPT/XPD): 0.01; standard: 0.0001. Set via `infer_pip_size(pair)`. |
| `spread_pips` (constructor arg, default 1.5) | `_open_position` line 429 | Should match broker spread for the pair |
| `slippage_pips` (constructor arg, default 0.5, V2 only) | `_close_position` lines 466-468 | Simulated slippage |
| `sl_atr_multiplier` (default 2.5) | `_open_position` line 425 | SL distance = `max(atr × 2.5, 15 pips × pip_size)` |
| `min_sl_pips` (default 15.0) | same | Floor on SL distance |
| `cooldown_bars` (default 4) | `step` line 340 | Bars after a close before a new entry is allowed |
| `max_trades_per_day` (default 6) | `step` line 341 | Hard daily trade cap |

### 9.3 LLM reproduction

LLM is a **confluence layer**, not a core signal generator. In backtest mode it's bypassed entirely (`ai/ai_analyst.py:300` `if is_backtest_mode(): return self._fallback_result(...)`). For live parity, the historical pipeline does NOT need to replay LLM calls — only the upstream contexts (indicators, SMC, SR, regime, patterns, MTF, session, intermarket, sentiment, news) need to be reconstructable, and the LLM cache (5-min TTL, in-memory only) does not need persistence.

---

## 10. Action Items for Historical CSV Pipeline Parity

1. **Historical CSV schema** must include at minimum: `time` (tz-aware UTC), `open`, `high`, `low`, `close`, `tick_volume` (renamed to `volume`). Plus pre-computed indicator columns OR the pipeline must run `phase3_features.compute_features` on the raw CSV before training.
2. **Economic calendar JSON** (`data/economic_calendar.json`) must cover the historical window — otherwise news-block / news-confidence features are zero in backtest and non-zero live.
3. **Intermarket data** (DXY, XAUUSD, VIX, SPX, US10Y) historical CSVs must be supplied so `intermarket_ctx` can be reconstructed.
4. **Currency strength** historical CSVs (or a way to reconstruct `currency_strength_ctx` from the symbol universe's OHLC) must be supplied.
5. **Feature engineer patch**: replace `pd.Timestamp.now(tz="UTC")` (line 400) with `df.index[-1]` when replaying historical bars, so `hour_utc` / `day_of_week` / `is_weekend` / `is_monday_open` / `is_friday_close` reflect the bar's time, not the live clock.
6. **`scripts/train_models_quick.py`** should adopt `PurgedEmbargoedSplitter` (or be retired in favor of `ModelTrainer.train_all` which already does) — otherwise label-window leakage at the train/test boundary remains.
7. **ML model re-enablement**: `analysis_agent.py:2001` (`if False:`) currently disables live ML prediction. If ML is to contribute to live trading, this gate must be flipped back to `if True:` AND the model registry must be populated (run `python -m ml.ensemble_train`).
8. **`ml/data_preprocessor.clean_features`** (deprecated, lines 116-131) should be removed or guarded — it's a latent leakage vector if any caller invokes it on combined train+test data.

---

## 11. File Inventory Inspected

### `ml/` directory (84 .py files; key ones read in full)

| File | Purpose | Read? |
|---|---|---|
| `ml/feature_engineer.py` (744 lines) | ~110-feature flat dict builder | ✅ full |
| `ml/feature_store.py` (308 lines) | SQLite persistence for feature rows + labels | ✅ full |
| `ml/feature_selector.py` (266 lines) | LightGBM/RF/variance importance + PSI drift + MTF aggregator | ✅ full |
| `ml/dataset_builder.py` (423 lines) | 70/15/15 chronological split + purged-embargoed splitter integration | ✅ full |
| `ml/data_bootstrap.py` (147 lines) | Synthetic noise rows for first-run (source='bootstrap') | ✅ full |
| `ml/data_preprocessor.py` (241 lines) | NaN-drop, ±3σ clip, StandardScaler (train-only fit) | ✅ full |
| `ml/label_generator.py` (432 lines) | ATR-scaled fixed-horizon labels (BUY/SELL/WAIT) | ✅ full |
| `ml/triple_barrier_labels.py` (433 lines) | Path-dependent TP/SL/timeout labels + meta-labels + MFE/MAE | ✅ full |
| `ml/triple_barrier_labels_OPTIONAL_spread_aware.py` | Optional spread-aware variant | ✅ header |
| `ml/fast_triple_barrier.py` (99 lines) | Vectorized triple-barrier (numpy sliding_window_view) | ✅ full |
| `ml/dual_binary_model.py` (325 lines) | Two binary classifiers (BUY model + SELL model) | ✅ partial |
| `ml/model_trainer.py` (568 lines) | XGBoost + RandomForest + LSTM training + threshold calibration | ✅ full |
| `ml/model_predictor.py` (755 lines) | Live ensemble prediction with per-model calibrated thresholds | ✅ partial |
| `ml/model_store.py` (734 lines) | Semantic versioning + portable relative paths + cross-process lock | ✅ header |
| `ml/model_evaluator.py` (316 lines) | Standard ML + trading metrics + walk-forward | ✅ header |
| `ml/ensemble.py` (578 lines) | EnsembleEngine fusing ML+rules+LLM+regime | ✅ header |
| `ml/ensemble_train.py` (121 lines) | CLI wrapper to retrain base models + report ensemble weights | ✅ full |
| `ml/ensemble_store.py` (278 lines) | SQLite for ensemble decisions + model performance tracking | ✅ header |
| `ml/confidence_fusion.py` (288 lines) | Weighted avg + regime-adjusted weights + conflict penalty | ✅ partial |
| `ml/forecast_engine.py` (261 lines) | Conservative short-term forecast (extra 10% weight vote) | ✅ header |
| `ml/pattern_features.py` (260 lines) | Per-candlestick-pattern feature calculators (Hammer/Engulfing/etc.) | ✅ header |
| `ml/mt5_data_loader.py` (759 lines) | MT5 historical fetcher, 100k-bar chunks, validation, tz=UTC | ✅ partial |
| `ml/institutional_feature_adapter.py` (86 lines) | Live adapter for phase3 schema (MIN_WARMUP_BARS=220) | ✅ full |
| `ml/optimal_tp_predictor.py` (353 lines) | RandomForest regressor for per-trade TP distance | ✅ header |
| `ml/diag_mt5.py` (21 lines) | MT5 connectivity diagnostic script | ✅ full |
| `ml/cv_splitter.py` (284 lines) | PurgedEmbargoedSplitter — López de Prado ch.7 | ✅ header |
| `ml/reward_engine.py` (191 lines) | V1 reward (profit × 5, drawdown penalty, hold reward) | ✅ partial |
| `ml/reward_engine_v2.py` (318 lines) | V2 reward (initial_balance scaling, transaction cost, streak bonus) | ✅ partial |
| `ml/rl_environment.py` (462 lines) | V1 Gymnasium env (16-dim state, Discrete(4) actions) | ✅ full |
| `ml/rl_environment_v2.py` (635 lines) | V2 Gymnasium env (29-dim state, slippage, cooldown, max_trades_per_day) | ✅ partial |
| `ml/train_rl_v2.py` (772 lines) | PPO trainer with multi-pair parallel envs | ✅ partial |
| `ml/pipeline/phase3_features.py` (460 lines) | Institutional feature set (~80 columns) | ✅ partial |
| `ml/pipeline/phase4_labels.py` | Label generation for institutional pipeline | ✅ grep only |

### `rl/` directory

| File | Purpose | Read? |
|---|---|---|
| `rl/action_masking.py` (245 lines) | RBOT-style action map + binary mask + logit -inf | ✅ full |
| `rl/reward_functions.py` (127 lines) | Softplus reward (ported from T32_v5) + simple_pnl + asymmetric + sharpe | ✅ full |
| `rl/__init__.py` | empty | ✅ noted |

### `ai/` directory

| File | Purpose | Read? |
|---|---|---|
| `ai/ai_analyst.py` (1108 lines) | Groq→Gemini→OpenRouter cascade, 5-min cache, timeout budget | ✅ partial |
| `ai/automated_retraining.py` (947 lines) | Scheduled retraining, LSTM walk-forward, TimeSeriesSplit | ✅ partial |
| `ai/model_versioning.py` (461 lines) | Atomic JSON writes, TF + MLflow optional | ✅ header |

### `core/` LLM modules

| File | Purpose | Read? |
|---|---|---|
| `core/llm_cache.py` (156 lines) | In-memory LRU OrderedDict, TTL=300s, max=200 | ✅ full |
| `core/llm_key_manager.py` (1477 lines) | Multi-key rotation per provider, TPD vs RPM classification | ✅ partial |
| `core/ollama_validator.py` (495 lines) | Qwen3:4B final veto gate (disabled by default) | ✅ partial |

### `intelligence/` directory

| File | Purpose | Read? |
|---|---|---|
| `intelligence/news_sources.py` (443 lines) | FF + central-bank + 4 RSS + local calendar | ✅ partial |
| `intelligence/news_ai.py` (486 lines) | NewsIntelligence orchestrator, 5-min cache, JSONL persistence | ✅ partial |
| `intelligence/sentiment_model.py` (441 lines) | Groq/Gemini sentiment, 5-call/cycle cap, rule-based fallback | ✅ partial |
| `intelligence/confluence_engine.py` (590 lines) | Multi-factor weighted confluence scorer | ✅ header |
| `intelligence/signal_validator.py` (300 lines) | Pre-trade validation gates (confluence/factor/contradiction/risk/news/correlation) | ✅ header |
| `intelligence/decision_score.py` (264 lines) | 7-factor weighted scoring (SMC 25% / Liq 20% / Currency 15% / Intermarket 15% / News 10% / Technical 10% / Session 5%) | ✅ partial |

### `agents/` directory

| File | Purpose | Read? |
|---|---|---|
| `agents/analysis_agent.py` | 12-step analysis pipeline; calls `FeatureEngineer.build_feature_vector`; **ML prediction gated by `if False:` at line 2001** | ✅ partial |
| `agents/decision_agent.py` | 4-layer consensus (rule=1, llm=2, master=3, ml=2 weights); LLM-failure exclusion logic | ✅ grep only |
| `agents/master_analyst.py` (1713 lines) | MasterAnalyst LLM brain; 22 context blocks fed in prompt | ✅ partial |

### `llm/` directory

**Does not exist.** LLM functionality lives in `ai/ai_analyst.py`, `agents/master_analyst.py`, `intelligence/sentiment_model.py`, `core/llm_cache.py`, `core/llm_key_manager.py`, `core/ollama_validator.py`.

---

**End of report.**
