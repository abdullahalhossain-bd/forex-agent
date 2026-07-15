# Forex AI Trading System

> Autonomous forex trading system: multi-agent analysis → intelligence fusion →
> risk gate → approval-gated execution → learning feedback loop. Runs on
> MetaTrader 5 (live / demo) or a built-in paper simulator.

**Total files:** 383 Python files  
**Total lines:** 125,921  
**Generated:** 2026-07-04

---

## Quick Start

```bash
# 1) Create venv
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Configure
cp .env.example .env                                   # then fill in values

# 4) Run (paper mode by default)
python main.py --mode paper

# 5) Run the Streamlit dashboard
streamlit run dashboard/app.py

# 6) Inspect the Trading-as-Git journal
python -m orchestrator.trading_as_git status
```

For MT5 (live or demo), set `EXECUTION_MODE=mt5_demo` and confirm the
MetaTrader 5 terminal is running on the same Windows host. MT5 does NOT
work on Linux for live trading — use `paper` mode on Linux.

---

## Documentation

| Document | Purpose |
| --- | --- |
| [**AGENTS.md**](AGENTS.md) | Guide for AI coding agents working on this repo (module map, load-bearing wiring, common pitfalls). **Read this first if you're modifying code.** |
| [**docs/architecture.md**](docs/architecture.md) | System architecture overview, data flow diagram, failure mode table. |
| [**docs/agent-workflow.md**](docs/agent-workflow.md) | How the multi-agent layer and intelligence layer collaborate. |
| [**docs/trading-as-git.md**](docs/trading-as-git.md) | Approval-gated trading: STAGE → COMMIT → PUSH with human rejection. |
| [**docs/ported_indicators.md**](docs/ported_indicators.md) | 7 indicators ported from MQL5 (geraked/metatrader5) — SuperTrend, UTBot, AndeanOscillator, NadarayaWatsonEnvelope, DailyHighLow, ChandelierExit, AtrSlFinder. |
| [**docs/strategy_catalog.md**](docs/strategy_catalog.md) | 11 EA strategies from the upstream MQL5 repo, with confluence-weight recipes for recreating each in our system. |
| [**docs/supermao_ports.md**](docs/supermao_ports.md) | 2 strategies + 2 utilities ported from MQL4 (tanvird3/TradingRobot) — multi-band Bollinger+MACD, Ichimoku cloud, cross-currency USD TP/SL, magic-number hashing. |
| [**docs/candlestick_patterns_mw.md**](docs/candlestick_patterns_mw.md) | 33-pattern candlestick scanner ported from MotiveWave (RauchenwaldC) — Hammer, Engulfing, Morning/Evening Star, Three White Soldiers, etc. with trend-aware filtering. |
| [**docs/ml_optimized_tp.md**](docs/ml_optimized_tp.md) | ML-optimized take-profit workflow ported from MaxwellMendenhall/ml_backtest — train a RandomForest to predict per-trade TP distance from pattern geometry. |
| [**docs/candlestick_patterns_br.md**](docs/candlestick_patterns_br.md) | 11 Brazilian-book candlestick patterns (JimmyAreaFiscal/MercadoFinanceiro) with triple filtering: trend + volume + next-bar confirmation. Plus crossover helpers and MT5 data fetcher. |
| [**docs/smartedge_inspired.md**](docs/smartedge_inspired.md) | 5 architecture modules inspired by SmartEdge-EA (closed-source) — VW-MACD, SymbolLock, controlled grid scaling, basket exit, multi-strategy sets. |
| [**docs/forex_bot_ports.md**](docs/forex_bot_ports.md) | 5 modules from bruh7463/forex_bot — 28-feature indicators with cyclical time, dual-binary XGBoost, ADX filter, ATR risk manager, broker factory. |
| [**docs/zeroxt_visual_rl.md**](docs/zeroxt_visual_rl.md) | 3 modules from zeroxt32 — softplus reward function (asymmetric RL reward shaping), TimeSformer transformer block, ChartDecorator (visual trade overlay on chart images). |
| [**docs/compressed_storage.md**](docs/compressed_storage.md) | Compressed binary quote storage ported from NewYaroslav/xquotes_history (C++). Day-partitioned format with zstd compression, O(1) timestamp lookup, 12× smaller than CSV. |
| [**docs/vectorized_backtesting.md**](docs/vectorized_backtesting.md) | Vectorized backtesting framework (SMA, Bollinger, Momentum, Contrarian, ML) — fast parameter optimization. |
| [**docs/security.md**](docs/security.md) | Threat model, secrets handling, trading safety mechanisms. |
| [**docs/runbook.md**](docs/runbook.md) | Daily / weekly / monthly operational checks and incident response. |
| [**CONTRIBUTING.md**](CONTRIBUTING.md) | How to contribute: pre-commit checklist, commit conventions, review criteria. |
| [**CHANGELOG.md**](CHANGELOG.md) | Notable changes per release. |
| [**SECURITY.md**](SECURITY.md) | Vulnerability disclosure policy. |
| [**CODE_OF_CONDUCT.md**](CODE_OF_CONDUCT.md) | Community code of conduct. |

The remainder of this README is auto-generated per-module documentation
kept for reference. For day-to-day work prefer the curated docs above.

---

## Table of Contents


- [Root Files](#root-files)
- [analysis/](#analysis)
- [backtest/](#backtest)
- [risk/](#risk)
- [core/](#core)
- [agents/](#agents)
- [broker/](#broker)
- [data/](#data)
- [ml/](#ml)
- [ai/](#ai)
- [intelligence/](#intelligence)
- [learning/](#learning)
- [memory/](#memory)
- [orchestrator/](#orchestrator)
- [scanner/](#scanner)
- [strategies/](#strategies)
- [strategy/](#strategy)
- [execution/](#execution)
- [fundamental/](#fundamental)
- [alerts/](#alerts)
- [analytics/](#analytics)
- [automation/](#automation)
- [hybrid/](#hybrid)
- [dashboard/](#dashboard)
- [monitoring/](#monitoring)
- [server/](#server)
- [system/](#system)
- [database/](#database)
- [utils/](#utils)
- [visualization/](#visualization)
- [research/](#research)
- [scripts/](#scripts)
- [tests/](#tests)

---

## Root Files

| File | Lines | Description | Key Classes/Functions |
|------|-------|-------------|----------------------|
| `check_network.py` | 247 | check_network.py — Network connectivity diagnostic for forex_ai | _c, _check_dns, _check_tcp |
| `config.py` | 454 | — | Config |
| `debug_llm_failure.py` | 14 | — | — |
| `debug_project.py` | 853 | debug_project.py — Whole-project error scanner for forex_ai | CheckResult |
| `debug_silent_failure.py` | 497 | debug_silent_failure.py — Diagnose silent trade failures. | Report |
| `diagnose_trade.py` | 26 | — | — |
| `execution_diagnostics.py` | 552 | execution_diagnostics.py — End-to-end MT5 execution diagnostic. | Report |
| `fix_trades.py` | 264 | — | _backup, _read, _write |
| `main.py` | 785 | ===================================================== | ForexAISystem, SystemStatus |
| `mt5_pipeline_test.py` | 307 | ===================================================== | banner, step, connect_mt5 |
| `run_backtest.py` | 270 | run_backtest.py — Event-Driven Backtest Runner | generate_synthetic_data, run_backtest, main |
| `run_unified_demo.py` | 211 | run_unified_demo.py — End-to-End Demonstration of the Unified Signal Engine | make_synthetic_ohlc, fetch_mt5_data, fetch_lower_tf |
| `test_barrier_fixes.py` | 140 | test_barrier_fixes.py — Verify the 6 trade-blocking barriers are fixed. | check |
| `test_env.py` | 694 | test_env.py — Full Environment Check | ok, fail, warn |
| `trader.py` | 1973 | — | AITrader, AutonomousTraderSystem, _NoOp |

## analysis/

**Purpose:** Technical analysis engines — S/R zones, S/D zones, patterns, indicators, signal generation

**Files:** 60

| File | Lines | Description | Key Classes | Key Functions |
|------|-------|-------------|-------------|---------------|
| `__init__.py` | 1 | — | — | — |
| `_engine_utils.py` | 155 | — | — | atr_series, atr_value, pip_value (+3) |
| `adaptive_decision_engine.py` | 560 | analysis/adaptive_decision_engine.py — Adaptive Decision Engine | StrategySignal, Decision, StrategyStats (+1) | — |
| `advanced_patterns.py` | 1566 | — | AdvancedPatternDetector | — |
| `book_rules_index.py` | 705 | — | BookRule | get_all_rules, get_rules_by_chapter, get_rules_by_category (+7) |
| `breaker_block.py` | 224 | analysis/breaker_block.py — Breaker Block Detector (Day 81+) | BreakerBlockDetector | — |
| `cci_state_machine.py` | 360 | analysis/cci_state_machine.py — Book 5 (Frank Miller S&D) Chapter 11 C | CCISignal, CCIStateMachine | — |
| `correlation_engine.py` | 385 | analysis/correlation_engine.py — Day 96 Correlation & Volatility Risk  | CorrelationEngine | — |
| `currency_ranker.py` | 236 | — | CurrencyRanker | — |
| `currency_strength.py` | 494 | — | CurrencyStrengthEngine | — |
| `curve_mtf.py` | 465 | analysis/curve_mtf.py — Book 5 (Frank Miller S&D) Chapter 12 Multi-Fra | TradingStyle, CurvePosition, DirectionalBias (+2) | get_timeframe_triplet |
| `dat_framework.py` | 289 | analysis/dat_framework.py — DAT Framework (Direction-Area-Trigger) | DATResult, DATFramework | — |
| `database/__init__.py` | 1 | — | — | — |
| `decision_bridge.py` | 419 | analysis/decision_bridge.py — Bridge: UnifiedSignalEngine → AdaptiveDe | UnifiedToAdaptiveBridge | make_adaptive_decision |
| `divergence.py` | 515 | — | DivergenceEngine | — |
| `engulfing_bar_strategy.py` | 141 | — | EngulfingSetup, EngulfingBarStrategy | — |
| `fibonacci.py` | 915 | — | FibonacciEngine | — |
| `flip_zones.py` | 378 | analysis/flip_zones.py — Book 5 (Frank Miller S&D) Chapter 8 Flip Zone | FlipZoneEvent, ZoneState, FlipZoneDetector | — |
| `fvg_detector.py` | 142 | — | FVGDetector | — |
| `high_reliability_patterns.py` | 876 | — | DetectedPattern, HighReliabilityPatternDetector | _candle_metrics, detect_high_reliability_patterns |
| `ichimoku.py` | 408 | — | IchimokuEngine | — |
| `ict_amd_signal_engine.py` | 982 | — | ZoneInfo, AccumulationResult, ManipulationResult (+2) | _filter_by_session, _strength_to_confidence, detect_ict_amd_signal |
| `institutional_flow.py` | 278 | analysis/institutional_flow.py — Day 96 Institutional Flow + COT Intel | InstitutionalFlowEngine | — |
| `intermarket.py` | 564 | — | IntermarketEngine | — |
| `macro_data.py` | 157 | — | MacroDataProvider | — |
| `market_bias.py` | 205 | — | MarketBiasEngine | — |
| `market_regime.py` | 325 | — | MarketRegimeDetector | — |
| `microstructure.py` | 317 | analysis/microstructure.py — Day 97 Tick Microstructure Engine | MicrostructureEngine | get_microstructure_engine |
| `mtf_analyzer.py` | 873 | — | MTFAnalyzer | — |
| `multi_strategy_pa_engine.py` | 1233 | — | SRZone, SDZone, TrendInfo (+2) | _is_in_session, _is_momentum_candle, _is_baby_candle (+2) |
| `myfxbook_sentiment.py` | 431 | analysis/myfxbook_sentiment.py — Day 95 Myfxbook Community Outlook (OA | MyfxbookSentiment | get_myfxbook_sentiment |
| `news_api_provider.py` | 342 | analysis/news_api_provider.py — Day 92 NewsAPI.org integration | NewsAPIProvider | get_news_api_provider |
| `odd_enhancers.py` | 1013 | analysis/odd_enhancers.py — Book 5 (Frank Miller S&D) Chapter 6 Scorin | EnhancerScore, ZoneScoreResult, OddEnhancerScorer (+2) | — |
| `order_block.py` | 126 | — | OrderBlockDetector | — |
| `oscillator_regime_gate.py` | 233 | analysis/oscillator_regime_gate.py — Day 97+ Oscillator Regime Gating | OscillatorRegimeGate | get_oscillator_gate |
| `pair_session_map.py` | 182 | — | — | get_pair_priority, get_preferred_pairs, get_pair_session_recommendation |
| `patterns.py` | 986 | — | PatternDetector | — |
| `pin_bar_strategy.py` | 472 | — | PinBarSetup, PinBarStrategy | — |
| `retail_sentiment.py` | 372 | analysis/retail_sentiment.py — Day 94/95 Retail Sentiment + Order Book | RetailSentimentAPI | get_retail_sentiment_api |
| `risk_management.py` | 601 | analysis/risk_management.py — Book 5 (Frank Miller S&D) Chapter 14 Ris | RiskState, PositionSizeResult, RiskManager (+3) | — |
| `risk_sentiment.py` | 169 | — | RiskSentimentEngine | — |
| `sentiment.py` | 575 | — | SentimentEngine | — |
| `sentiment_data.py` | 379 | — | SentimentDataProvider | — |
| `session_analyzer.py` | 722 | — | SessionAnalyzer | — |
| `session_rules.py` | 169 | — | — | — |
| `smc_advanced.py` | 527 | — | SMCAdvancedEngine | — |
| `smc_engine.py` | 325 | — | SMCEngine | — |
| `stop_hunt_signal_engine.py` | 757 | — | StopHuntEvent, StopHuntSignalEngine | _strength_to_confidence, detect_stop_hunt_signal |
| `strength_calculator.py` | 212 | — | StrengthCalculator | — |
| `structure.py` | 568 | — | MarketStructureEngine | — |
| `structure_mtf.py` | 372 | — | MTFStructureEngine | — |
| `supply_demand_zones.py` | 825 | analysis/supply_demand_zones.py — Day 97+ Supply/Demand Zones | SupplyDemandZones | get_supply_demand_zones |
| `support_resistance.py` | 924 | — | SupportResistance | _classify_strength, _strength_emoji, _atr_pct (+1) |
| `timeframe.py` | 119 | — | MultiTimeframeAnalyzer | — |
| `trend_level_signal.py` | 296 | — | TrendLevelSignalFramework | — |
| `trendline_engine.py` | 311 | analysis/trendline_engine.py — Day 97+ Trendline Detection & Trading | TrendlineEngine | get_trendline_engine |
| `unified_signal_engine.py` | 681 | — | UnifiedSignalEngine | _zones_to_unified, detect_unified_signal |
| `volatility.py` | 386 | — | VolatilityEngine | — |
| `volume_confirmation.py` | 285 | analysis/volume_confirmation.py — Day 97+ Volume Confirmation for Brea | VolumeConfirmation | get_volume_confirmation |
| `volume_profile.py` | 441 | — | VolumeProfileEngine | — |

<details>
<summary>Detailed file descriptions (55 files)</summary>

### `adaptive_decision_engine.py`

**Description:** analysis/adaptive_decision_engine.py — Adaptive Decision Engine

**Classes (4):**

- `StrategySignal`
  - A signal from one strategy.
- `Decision`
  - Final trading decision.
- `StrategyStats`
  - Statistics for one strategy (loaded from backtest).
- `AdaptiveDecisionEngine`
  - Adaptive decision engine that learns from backtest results.
  - Methods: `__init__`, `load_backtest_results`, `load_from_file`, `decide`, `_decide_single`, `_decide_confluence`, `_decide_strict`, `_get_weight` (+2)

### `advanced_patterns.py`

**Classes (1):**

- `AdvancedPatternDetector`
  - Advanced chart pattern detection engine।
  - Methods: `__init__`, `detect_all`, `detect_head_and_shoulders`, `detect_double_top_bottom`, `detect_triple_top_bottom`, `detect_triangle`, `detect_flag`, `detect_wedge` (+2)

### `book_rules_index.py`

**Classes (1):**

- `BookRule`
  - Single rule extracted from the book.

**Functions (10):**

- `get_all_rules()`
  - Return all book rules.
- `get_rules_by_chapter()`
  - Return all rules from a specific chapter.
- `get_rules_by_category()`
  - Return all rules of a specific category.
- `get_no_trade_conditions()`
  - Return all rules that define NO_TRADE states.
- `get_deterministic_rules()`
  - Return all deterministic (directly codeable) rules.
- `get_rules_needing_confirmation()`
  - Return rules that need additional confirmation logic.
- `get_design_principles()`
  - Return architectural design principles.
- `find_rule()`
  - Find a specific rule by ID.
- ... and 2 more

### `breaker_block.py`

**Description:** analysis/breaker_block.py — Breaker Block Detector (Day 81+)

**Classes (1):**

- `BreakerBlockDetector`
  - Detects Breaker Blocks from failed Order Blocks.
  - Methods: `detect`, `_check_breaker`, `get_ai_context`

### `cci_state_machine.py`

**Description:** analysis/cci_state_machine.py — Book 5 (Frank Miller S&D) Chapter 11 CCI Module

**Classes (2):**

- `CCISignal`
  - Single CCI evaluation result.
- `CCIStateMachine`
  - Book 5 Chapter 11 — CCI entry/add/exit state machine.
  - Methods: `__init__`, `evaluate`, `_score_confluence`, `_check_exit_long`, `_check_exit_short`, `diagnose_zone_failure`

### `correlation_engine.py`

**Description:** analysis/correlation_engine.py — Day 96 Correlation & Volatility Risk Engine

**Classes (1):**

- `CorrelationEngine`
  - Currency correlation + volatility risk engine.
  - Methods: `__init__`, `analyze`, `build_matrix`, `sync_open`, `_analyze_volatility`, `_compute_atr`, `_default_volatility`, `_analyze_correlation` (+2)

### `currency_ranker.py`

**Classes (1):**

- `CurrencyRanker`
  - Usage:
  - Methods: `rank`, `find_best_pairs`, `_grade_difference`, `detect_correlation_risk`, `build_heatmap`, `detect_cycle`, `print_heatmap`

### `currency_strength.py`

**Classes (1):**

- `CurrencyStrengthEngine`
  - Day 64 — Global Currency Intelligence Engine।
  - Methods: `__init__`, `calculate_strength`, `calculate_momentum`, `rank_currencies`, `find_best_pairs`, `evaluate_setup`, `multi_timeframe_strength`, `analyze` (+2)

### `curve_mtf.py`

**Description:** analysis/curve_mtf.py — Book 5 (Frank Miller S&D) Chapter 12 Multi-Frame "Curve"

**Classes (5):**

- `TradingStyle`
  - Book P127-128: four trading styles by holding period.
- `CurvePosition`
  - Where current price sits within the curve (Book P131).
- `DirectionalBias`
  - Book P133: directional bias based on curve position.
- `Curve`
  - Book P130-131: the price range between nearest demand and supply
  - Methods: `curve_low`, `curve_high`, `position_of`, `bias_for`, `describe`
- `CurveMTF`
  - Book 5 Chapter 12 — Multi-Frame "Curve" methodology.
  - Methods: `from_zones`, `get_bias`, `check_alignment`, `resolve_conflict`, `fib_levels_for_curve`

**Functions (1):**

- `get_timeframe_triplet()`
  - Return the (long, medium, short) timeframe triplet for a style.

### `dat_framework.py`

**Description:** analysis/dat_framework.py — DAT Framework (Direction-Area-Trigger)

**Classes (2):**

- `DATResult`
  - Result of the DAT evaluation.
  - Methods: `to_dict`
- `DATFramework`
  - Direction-Area-Trigger evaluation pipeline.
  - Methods: `__init__`, `evaluate`, `_evaluate_direction`, `_evaluate_area`, `_evaluate_trigger`

### `decision_bridge.py`

**Description:** analysis/decision_bridge.py — Bridge: UnifiedSignalEngine → AdaptiveDecisionEngine

**Classes (1):**

- `UnifiedToAdaptiveBridge`
  - Converts UnifiedSignalEngine output → StrategySignal list →
  - Methods: `__init__`, `extract_signals`, `decide`

**Functions (1):**

- `make_adaptive_decision()`
  - Convenience function: make an adaptive decision from a unified result.

### `divergence.py`

**Classes (1):**

- `DivergenceEngine`
  - Price vs Indicator (RSI/MACD) divergence detector।
  - Methods: `__init__`, `detect`, `_find_pivots`, `_detect_divergence_pair`, `_score_divergence`, `_reversal_risk`, `_trend_continuation`, `get_ai_context` (+2)

### `engulfing_bar_strategy.py`

**Classes (2):**

- `EngulfingSetup`
  - Methods: `to_dict`
- `EngulfingBarStrategy`
  - Methods: `__init__`, `detect`, `_detect_engulfing`, `_check_ma_confluence`, `_check_fib_confluence`, `_check_sr_confluence`, `_calculate_entries`, `_score_quality` (+1)

### `fibonacci.py`

**Classes (1):**

- `FibonacciEngine`
  - AI-powered Fibonacci analysis engine।
  - Methods: `__init__`, `analyze`, `find_swing_points`, `calculate_retracement`, `calculate_extension`, `find_confluence`, `_fib_base_strength`, `_zone_type` (+2)

### `flip_zones.py`

**Description:** analysis/flip_zones.py — Book 5 (Frank Miller S&D) Chapter 8 Flip Zones

**Classes (3):**

- `FlipZoneEvent`
  - Emitted when a zone's type flips due to a confirmed break.
- `ZoneState`
  - Tracks a zone's lifecycle state.
  - Methods: `__post_init__`
- `FlipZoneDetector`
  - Book 5 Chapter 8 — Flip Zone detector.
  - Methods: `__init__`, `register_zone`, `update`, `get_active_zones`, `get_flipped_zones`, `get_events`, `clear`, `_infer_zone_type`

### `fvg_detector.py`

**Classes (1):**

- `FVGDetector`
  - Usage:
  - Methods: `detect`, `_build`, `nearest_active`, `print_summary`

### `high_reliability_patterns.py`

**Classes (2):**

- `DetectedPattern`
  - Single detected pattern matching spec output schema.
  - Methods: `to_spec_dict`
- `HighReliabilityPatternDetector`
  - Strict 20-pattern library — spec-compliant.
  - Methods: `__init__`, `detect`, `_check_zone_confluence`, `_make_pattern`, `_detect_hammer`, `_detect_shooting_star`, `_detect_inverted_hammer`, `_detect_hanging_man` (+2)

**Functions (2):**

- `_candle_metrics()`
  - Extract body/wick metrics from a single OHLC candle.
- `detect_high_reliability_patterns()`
  - One-shot helper — returns spec-compliant list of detected pattern dicts.

### `ichimoku.py`

**Classes (1):**

- `IchimokuEngine`
  - Standard parameters: 9 / 26 / 52 / 26
  - Methods: `__init__`, `analyze`, `_donchian_mid`, `_cloud_position`, `_tk_cross`, `_chikou_clear`, `_assess_trend`, `_signal` (+2)

### `ict_amd_signal_engine.py`

**Classes (5):**

- `ZoneInfo`
  - Single zone dict matching spec output schema.
  - Methods: `to_spec_dict`
- `AccumulationResult`
  - Methods: `to_spec_dict`
- `ManipulationResult`
  - Methods: `to_spec_dict`
- `FVGResult`
  - Methods: `to_spec_dict`
- `ICTAMDSignalEngine`
  - ICT/SMC AMD + FVG + MSS Signal Engine — spec-compliant.
  - Methods: `__init__`, `analyze`, `_step1_zones`, `_step1_accumulation`, `_step2_manipulation`, `_check_sweep_at_target`, `_find_reversal`, `_step3_fvg` (+2)

**Functions (3):**

- `_filter_by_session()`
  - Filter DataFrame to candles whose hour is in [start_hr, end_hr).
- `_strength_to_confidence()`
  - Map confluence factors → confidence Low/Medium/High.
- `detect_ict_amd_signal()`
  - One-shot helper — pass OHLC df, get spec-compliant JSON back.

### `institutional_flow.py`

**Description:** analysis/institutional_flow.py — Day 96 Institutional Flow + COT Intelligence

**Classes (1):**

- `InstitutionalFlowEngine`
  - Institutional flow tracker via COT data + synthetic fallback.
  - Methods: `__init__`, `analyze`, `_fetch_cot_data`, `_build_cot_result`, `_build_synthetic_result`, `_fallback_result`, `get_ai_context`, `print_summary`

### `intermarket.py`

**Classes (1):**

- `IntermarketEngine`
  - Usage:
  - Methods: `__init__`, `analyze`, `fetch_global_data`, `calculate_correlations`, `detect_market_regime`, `generate_macro_bias`, `_single_currency_bias`, `_resolve_pair_bias` (+2)

### `macro_data.py`

**Classes (1):**

- `MacroDataProvider`
  - Usage:
  - Methods: `__init__`, `get_all`, `_compute_asset`, `_classify_trend`, `_fallback_asset`, `_fallback`, `print_summary`

### `market_bias.py`

**Classes (1):**

- `MarketBiasEngine`
  - সব indicator + pattern + S/R + MTF একসাথে দেখে:
  - Methods: `analyze`, `_recommendation`, `print_summary`, `get_ai_context`

### `market_regime.py`

**Classes (1):**

- `MarketRegimeDetector`
  - Market Regime detect করে — 4টা dimension:
  - Methods: `__init__`, `detect`, `_add_adx`, `_detect_regime`, `_detect_direction`, `_detect_strength`, `_detect_volatility`, `_suggest_strategy` (+2)

### `microstructure.py`

**Description:** analysis/microstructure.py — Day 97 Tick Microstructure Engine

**Classes (1):**

- `MicrostructureEngine`
  - Tick-level market microstructure analyzer (MT5 native).
  - Methods: `__init__`, `analyze`, `_extract_tick_field`, `_fetch_ticks`, `_analyze_tick_speed`, `_analyze_spread`, `_analyze_volume`, `_analyze_acceleration` (+2)

**Functions (1):**

- `get_microstructure_engine()`

### `mtf_analyzer.py`

**Classes (1):**

- `MTFAnalyzer`
  - Multi-Timeframe Analysis Engine।
  - Methods: `__init__`, `analyze`, `_fetch_all_timeframes`, `_build_tf_contexts`, `_h4_structure_note`, `_h1_zone_note`, `_m15_setup_note`, `_m5_entry_note` (+2)

### `multi_strategy_pa_engine.py`

**Classes (5):**

- `SRZone`
  - Methods: `to_spec_dict`
- `SDZone`
  - Methods: `to_spec_dict`
- `TrendInfo`
- `ConfluenceZone`
- `MultiStrategyPAEngine`
  - Multi-Strategy Price Action Signal Engine — spec-compliant.
  - Methods: `__init__`, `analyze`, `_step1_sr_zones_and_bias`, `_step2_trend_structure`, `_step3_shooting_star`, `_step5_mtf_confirmation`, `_step6_supply_demand`, `_step7_confluence` (+2)

**Functions (5):**

- `_is_in_session()`
  - Check if latest candle timestamp is within 12:30-14:30 BD Time.
- `_is_momentum_candle()`
  - Momentum candle: body ≥ 70% of total range.
- `_is_baby_candle()`
  - Baby candle: small body OR large wick (weak momentum).
- `_is_shooting_star()`
  - Shooting star: small body (lower part), long upper wick (≥2× body), small lower wick.
- `detect_multi_strategy_pa_signal()`
  - One-shot helper — returns spec-compliant JSON.

### `myfxbook_sentiment.py`

**Description:** analysis/myfxbook_sentiment.py — Day 95 Myfxbook Community Outlook (OANDA alternative)

**Classes (1):**

- `MyfxbookSentiment`
  - Myfxbook Community Outlook scraper — free, no API key needed.
  - Methods: `__init__`, `available`, `get_sentiment`, `_fetch_outlook_page`, `_parse_outlook_html`, `_find_pair`, `_compute_confidence`, `_fallback_result` (+2)

**Functions (1):**

- `get_myfxbook_sentiment()`

### `news_api_provider.py`

**Description:** analysis/news_api_provider.py — Day 92 NewsAPI.org integration

**Classes (1):**

- `NewsAPIProvider`
  - NewsAPI.org client with currency-aware sentiment scoring.
  - Methods: `__init__`, `available`, `fetch_headlines_for_pair`, `_score_headline`, `_extract_currencies`, `_neutral_result`, `_fallback_result`, `get_ai_context` (+1)

**Functions (1):**

- `get_news_api_provider()`

### `odd_enhancers.py`

**Description:** analysis/odd_enhancers.py — Book 5 (Frank Miller S&D) Chapter 6 Scoring System

**Classes (5):**

- `EnhancerScore`
  - Single enhancer evaluation result.
  - Methods: `passed`
- `ZoneScoreResult`
  - Final scoring result for a zone.
  - Methods: `tradeable`
- `OddEnhancerScorer`
  - Book 5 Chapter 6 — Odd Enhancers scoring system.
  - Methods: `score_zone`, `_score_strength_of_move`, `_score_time_at_zone`, `_score_freshness`, `_score_risk_reward`, `_check_original_zone`, `_check_overlapping`, `_check_pa_confluence` (+2)
- `ConfirmationEntrySignal`
  - Result of a Tier-B confirmation-order state-machine check.
- `TierBEntryStateMachine`
  - Book Page 74-75 — Tier-B (score 7-9) entry tactics.
  - Methods: `__init__`, `check_market_order_entry`, `check_confirmation_entry`, `_is_supply_zone`

### `order_block.py`

**Classes (1):**

- `OrderBlockDetector`
  - Methods: `detect`, `nearest_active`, `print_summary`

### `oscillator_regime_gate.py`

**Description:** analysis/oscillator_regime_gate.py — Day 97+ Oscillator Regime Gating

**Classes (1):**

- `OscillatorRegimeGate`
  - Gates oscillator signals based on market regime.
  - Methods: `adjust_signal`, `get_rsi_signal`

**Functions (1):**

- `get_oscillator_gate()`

### `patterns.py`

**Classes (1):**

- `PatternDetector`
  - Candlestick pattern detector — TA-Lib ছাড়া।
  - Methods: `detect_all`, `_detect_row`, `is_doji`, `is_hammer`, `is_shooting_star`, `is_pin_bar`, `is_bullish_engulfing_row`, `is_bearish_engulfing_row` (+2)

### `pin_bar_strategy.py`

**Classes (2):**

- `PinBarSetup`
  - A detected pin bar setup with all 3 filter criteria evaluated.
  - Methods: `to_dict`
- `PinBarStrategy`
  - Book Pages 81-95 — Pin Bar Trading Strategy.
  - Methods: `__init__`, `detect`, `_detect_pin_bar`, `_check_timeframe`, `_check_trend_alignment`, `_check_level_confluence`, `_calculate_entries`, `_score_quality` (+1)

### `retail_sentiment.py`

**Description:** analysis/retail_sentiment.py — Day 94/95 Retail Sentiment + Order Book

**Classes (1):**

- `RetailSentimentAPI`
  - OANDA v20 retail sentiment + order book API.
  - Methods: `__init__`, `available`, `_base_url`, `_headers`, `get_sentiment`, `_get_sentiment_oanda`, `_fetch_position_book`, `_fetch_order_book` (+2)

**Functions (1):**

- `get_retail_sentiment_api()`

### `risk_management.py`

**Description:** analysis/risk_management.py — Book 5 (Frank Miller S&D) Chapter 14 Risk Management

**Classes (6):**

- `RiskState`
  - Snapshot of the risk-management state.
  - Methods: `risk_amount`
- `PositionSizeResult`
  - Result of a position-sizing calculation.
- `RiskManager`
  - Book 5 Chapter 14 — risk-management state machine.
  - Methods: `__init__`, `base_risk_pct`, `drawdown_pct`, `in_drawdown`, `current_risk_pct`, `update`, `get_state`, `get_risk_amount` (+2)
- `PositionSizer`
  - Book 5 Chapter 14 — position-sizing formulas.
  - Methods: `__init__`, `size_for_stock`, `size_for_forex`
- `MarginCallDetector`
  - Book 5 Chapter 14 (Page 154) — margin call detection.
  - Methods: `is_margin_call`, `max_loss_before_margin_call`, `exposure`
- `DrawdownSimulator`
  - Book 5 Chapter 14 (Page 155) — compounding drawdown simulator.
  - Methods: `simulate_losing_streak`, `compare_risk_levels`

### `risk_sentiment.py`

**Classes (1):**

- `RiskSentimentEngine`
  - Usage:
  - Methods: `analyze`, `_classify_environment`, `_classify_fear`, `_preferred_assets`, `get_ai_context`, `print_summary`

### `sentiment.py`

**Classes (1):**

- `SentimentEngine`
  - Market psychology reading engine।
  - Methods: `__init__`, `retail_positioning`, `fear_greed`, `currency_strength`, `dxy_analysis`, `final_sentiment_score`, `detect_conflict`, `get_ai_context` (+1)

### `sentiment_data.py`

**Classes (1):**

- `SentimentDataProvider`
  - Sentiment Engine-এর জন্য সব data এক জায়গা থেকে দেয়।
  - Methods: `__init__`, `get_all`, `get_retail_positioning`, `_fallback_retail`, `get_fear_greed_index`, `get_currency_strengths`, `get_dxy_data`, `_fallback_dxy` (+2)

### `session_analyzer.py`

**Classes (1):**

- `SessionAnalyzer`
  - Day 63 — Session-Based Market Intelligence।
  - Methods: `__init__`, `get_current_session`, `analyze_session_behavior`, `get_strategy_mode`, `get_pair_preference`, `detect_session_transition`, `calculate_session_confidence`, `session_smc_fusion` (+2)

### `smc_advanced.py`

**Classes (1):**

- `SMCAdvancedEngine`
  - Mitigation Block + Inducement detector।
  - Methods: `__init__`, `analyze`, `_find_order_blocks`, `_find_mitigation_blocks`, `_mb_note`, `_find_inducements`, `_bias_and_signal`, `get_ai_context` (+2)

### `smc_engine.py`

**Classes (1):**

- `SMCEngine`
  - Usage:
  - Methods: `__init__`, `analyze`, `_fetch_with_atr`, `_score_confluence`, `_rank_zone`, `_build_explanation`, `_empty_result`, `get_ai_context` (+1)

### `stop_hunt_signal_engine.py`

**Classes (2):**

- `StopHuntEvent`
  - A single confirmed stop-hunt event on a zone.
  - Methods: `to_dict`
- `StopHuntSignalEngine`
  - Combines S/R Zone detection + Stop Hunt detection + Trade signal
  - Methods: `__init__`, `analyze`, `_detect_stop_hunt`, `_check_zone_for_stop_hunt`, `_find_reversal_confirmation`, `_check_equal_highs_lows`, `_generate_signal`, `_compute_tp` (+2)

**Functions (2):**

- `_strength_to_confidence()`
  - Map strength + confluence factors → confidence Low/Medium/High.
- `detect_stop_hunt_signal()`
  - One-shot helper — pass OHLC df, get spec-compliant JSON back.

### `strength_calculator.py`

**Classes (1):**

- `StrengthCalculator`
  - Usage:
  - Methods: `compute_pair_score`, `_price_change_score`, `_trend_score`, `_momentum_score`, `_volatility_adjustment`, `_avg_atr_pct`, `normalize_scores`

### `structure.py`

**Classes (1):**

- `MarketStructureEngine`
  - Usage:
  - Methods: `__init__`, `analyze`, `_find_swing_points`, `_label_swings`, `_determine_structure`, `_detect_bos`, `_bos_confidence`, `_detect_choch` (+2)

### `structure_mtf.py`

**Classes (1):**

- `MTFStructureEngine`
  - Internal vs External structure analyzer।
  - Methods: `__init__`, `analyze`, `_detect_conflict`, `_alignment`, `_combined_bias`, `_trading_permission`, `_note`, `get_ai_context` (+2)

### `supply_demand_zones.py`

**Description:** analysis/supply_demand_zones.py — Day 97+ Supply/Demand Zones

**Classes (1):**

- `SupplyDemandZones`
  - Detects institutional supply/demand zones.
  - Methods: `detect`, `_deduplicate`, `is_erc`, `count_ercs`, `has_valid_impulse`, `draw_zone_medium_risk`, `draw_zone_high_risk`, `draw_zone_low_risk` (+2)

**Functions (1):**

- `get_supply_demand_zones()`
  - Thread-safe singleton accessor for the SupplyDemandZones instance.

### `support_resistance.py`

**Classes (1):**

- `SupportResistance`
  - AI Trader-এর S/R Zone detection engine (v2 — Zone-Based).
  - Methods: `__init__`, `find_swing_highs`, `find_swing_lows`, `_is_valid_rejection`, `_count_valid_rejections`, `_get_cluster_threshold`, `cluster_into_zones`, `_build_zone` (+2)

**Functions (4):**

- `_classify_strength()`
  - 2=Weak, 3=Medium, 4+=Strong
- `_strength_emoji()`
- `_atr_pct()`
  - ATR as % of price — used for adaptive cluster threshold.
- `detect_zones_for_llm()`
  - One-shot helper for LLM Agent integration.

### `timeframe.py`

**Classes (1):**

- `MultiTimeframeAnalyzer`
  - Professional trader-এর মতো top-down analysis।
  - Methods: `__init__`, `analyze`, `get_bias`, `print_summary`

### `trend_level_signal.py`

**Classes (1):**

- `TrendLevelSignalFramework`
  - Book Pages 79-80 — "Trend, Level, Signal" unified framework.
  - Methods: `__init__`, `analyze`, `_analyze_trend`, `_trend_answer`, `_analyze_levels`, `_analyze_signals`, `_decide_action`

### `trendline_engine.py`

**Description:** analysis/trendline_engine.py — Day 97+ Trendline Detection & Trading

**Classes (1):**

- `TrendlineEngine`
  - Detects and trades trendlines (Book Pages 63-66).
  - Methods: `analyze`, `_find_swings`, `_fit_trendline`, `_detect_channel`, `_generate_signals`, `_empty_result`

**Functions (1):**

- `get_trendline_engine()`

### `unified_signal_engine.py`

**Classes (1):**

- `UnifiedSignalEngine`
  - Master orchestrator — connects all 5 strategy engines into one system.
  - Methods: `__init__`, `analyze`, `_compute_consensus`, `_fallback_stop_hunt`, `_fallback_ict`, `_fallback_pa`, `_insufficient_data_result`, `_build_unified_result` (+2)

**Functions (2):**

- `_zones_to_unified()`
  - Merge zones from multiple engines into a unified list with consistent schema.
- `detect_unified_signal()`
  - One-shot helper — returns unified JSON.

### `volatility.py`

**Classes (1):**

- `VolatilityEngine`
  - Bollinger Squeeze + ATR regime + breakout release detector।
  - Methods: `__init__`, `analyze`, `_add_bollinger`, `_add_atr`, `_percentile`, `_squeeze_strength`, `_atr_regime`, `_detect_release` (+2)

### `volume_confirmation.py`

**Description:** analysis/volume_confirmation.py — Day 97+ Volume Confirmation for Breakouts

**Classes (1):**

- `VolumeConfirmation`
  - Validates breakouts with volume analysis.
  - Methods: `check_breakout`, `get_volume_context`, `check_trend_confirmation`

**Functions (1):**

- `get_volume_confirmation()`

### `volume_profile.py`

**Classes (1):**

- `VolumeProfileEngine`
  - Price-binned volume distribution analyzer।
  - Methods: `__init__`, `analyze`, `_build_profile`, `_value_area`, `_find_zones`, `_format_zone`, `_price_position`, `_bias` (+2)

</details>

## backtest/

**Purpose:** Backtesting framework — honest backtest engine, per-strategy tester, MT5 data fetcher, walk-forward validation

**Files:** 16

| File | Lines | Description | Key Classes | Key Functions |
|------|-------|-------------|-------------|---------------|
| `__init__.py` | 5 | — | — | — |
| `benchmark.py` | 157 | backtest/benchmark.py — Main Benchmark Runner (Day 74) | BenchmarkResult | run_benchmark, get_benchmark_history |
| `broker_sim.py` | 79 | — | SimulatedTrade, BrokerSimulator | _pip_value, _pip_to_price |
| `comparison_engine.py` | 156 | backtest/comparison_engine.py — System Comparison Engine (Day 74) | ComparisonResult, ComparisonEngine, BacktestMetrics | get_comparison_engine |
| `data_loader.py` | 165 | — | HistoricalDataLoader | load_data |
| `engine.py` | 337 | — | BacktestEngine | — |
| `honest_backtest_engine.py` | 1040 | backtest/honest_backtest_engine.py — Look-Ahead-Free Backtest Engine | HonestTrade, HonestResult, HonestBacktester (+4) | — |
| `metrics.py` | 126 | — | BacktestMetrics | calculate_metrics |
| `ml_backtest.py` | 409 | backtest/ml_backtest.py — ML Backtest Engine (Day 74) | TradeRecord, BacktestMetrics, MLBacktest | — |
| `mt5_bulk_fetcher.py` | 572 | backtest/mt5_bulk_fetcher.py — MT5 Auto-Pair Discovery + Bulk Data Fet | PairInfo, FetchResult, MT5BulkFetcher | _resolve_tf_map |
| `per_strategy_tester.py` | 1018 | backtest/per_strategy_tester.py — Per-Strategy Independent Backtester | Trade, StrategyResult, TradeSimulator (+1) | — |
| `performance_report.py` | 133 | backtest/performance_report.py — Performance Report Generator (Day 74) | — | generate_text_report, generate_telegram_report, generate_strategy_contribution |
| `report.py` | 100 | — | BacktestReport | — |
| `simulator.py` | 185 | — | TradePosition, ForexSimulator | — |
| `statistical_validation.py` | 491 | — | ValidationResult | monte_carlo_permutation_test, t_test_returns, bootstrap_confidence_interval (+3) |
| `walk_forward.py` | 206 | — | WalkForwardWindow, WalkForwardResult | run_walk_forward, print_walk_forward_table |

<details>
<summary>Detailed file descriptions (15 files)</summary>

### `benchmark.py`

**Description:** backtest/benchmark.py — Main Benchmark Runner (Day 74)

**Classes (1):**

- `BenchmarkResult`
  - Complete benchmark result.
  - Methods: `to_dict`

**Functions (2):**

- `run_benchmark()`
  - Run a full performance benchmark.
- `get_benchmark_history()`
  - Get recent benchmark results from DB.

### `broker_sim.py`

**Classes (2):**

- `SimulatedTrade`
  - Methods: `to_dict`
- `BrokerSimulator`
  - Methods: `__init__`, `open_trade`, `check_exit`, `close_trade`, `get_balance`

**Functions (2):**

- `_pip_value()`
- `_pip_to_price()`

### `comparison_engine.py`

**Description:** backtest/comparison_engine.py — System Comparison Engine (Day 74)

**Classes (3):**

- `ComparisonResult`
  - Result of comparing 3 system versions.
  - Methods: `to_dict`
- `ComparisonEngine`
  - Compares system versions and detects ML improvement.
  - Methods: `compare`, `_diagnose`
- `BacktestMetrics`
  - Methods: `__init__`, `to_dict`

**Functions (1):**

- `get_comparison_engine()`

### `data_loader.py`

**Classes (1):**

- `HistoricalDataLoader`
  - Methods: `__init__`, `load_csv`, `_normalize_columns`, `_validate_columns`, `_fill_missing_candles`, `_enrich`, `_row_regime`, `_row_direction` (+2)

**Functions (1):**

- `load_data()`

### `engine.py`

**Classes (1):**

- `BacktestEngine`
  - Methods: `__init__`, `load_dataset`, `run_strategy`, `run_walk_forward`, `compare_strategies`, `optimize_strategy`, `save_backtest_memory`, `_walk_forward_split` (+2)

### `honest_backtest_engine.py`

**Description:** backtest/honest_backtest_engine.py — Look-Ahead-Free Backtest Engine

**Classes (7):**

- `HonestTrade`
  - Trade record from honest backtest (with realistic costs).
- `HonestResult`
  - Result of an honest backtest.
- `HonestBacktester`
  - Backtester that eliminates look-ahead bias and models realistic costs.
  - Methods: `__init__`, `_pair_costs`, `_pip_size`, `simulate_trade`, `test_strategy`, `_compute_stats`
- `IncrementalZoneDetector`
  - Detects S/R zones incrementally — at each bar, only uses PAST data.
  - Methods: `__init__`, `zones_at_bar`, `_cluster_zones`, `_make_zone`
- `MonteCarloValidator`
  - Monte Carlo simulation to test if a strategy's edge is real.
  - Methods: `__init__`, `validate`
- `WalkForwardValidator`
  - Walk-forward validation — required to claim any edge is real.
  - Methods: `__init__`, `split`, `validate`
- `DeploymentGate`
  - HARD GATE: blocks live deployment until ALL criteria pass.
  - Methods: `evaluate`

### `metrics.py`

**Classes (1):**

- `BacktestMetrics`
  - Methods: `to_dict`, `to_table`

**Functions (1):**

- `calculate_metrics()`

### `ml_backtest.py`

**Description:** backtest/ml_backtest.py — ML Backtest Engine (Day 74)

**Classes (3):**

- `TradeRecord`
  - One simulated trade.
- `BacktestMetrics`
  - Performance metrics for one system version.
  - Methods: `calculate`, `to_dict`, `summary_line`
- `MLBacktest`
  - Backtest engine comparing 3 system versions.
  - Methods: `__init__`, `_init_db`, `run_backtest`, `_load_data`, `_simulate_system`, `_generate_signal`, `_open_position`, `_check_exit` (+2)

### `mt5_bulk_fetcher.py`

**Description:** backtest/mt5_bulk_fetcher.py — MT5 Auto-Pair Discovery + Bulk Data Fetcher

**Classes (3):**

- `PairInfo`
  - Metadata about a discovered trading pair.
- `FetchResult`
  - Result of fetching data for one (pair, timeframe).
- `MT5BulkFetcher`
  - Auto-discovers all tradable pairs from MT5 and fetches their
  - Methods: `__init__`, `_init_mt5`, `discover_pairs`, `_categorize_symbol`, `_default_pairs`, `filter_pairs`, `fetch`, `_fetch_mt5` (+2)

**Functions (1):**

- `_resolve_tf_map()`

### `per_strategy_tester.py`

**Description:** backtest/per_strategy_tester.py — Per-Strategy Independent Backtester

**Classes (4):**

- `Trade`
  - Single trade record.
- `StrategyResult`
  - Aggregated result for one strategy on one (pair, timeframe).
- `TradeSimulator`
  - Simulates a single trade: enters at entry_price, exits at SL/TP/timeout.
  - Methods: `__init__`, `simulate`
- `PerStrategyTester`
  - Tests each strategy INDEPENDENTLY on a given OHLCV DataFrame.
  - Methods: `__init__`, `run_all`, `_test_pin_bar`, `_test_candlestick_patterns`, `_test_sd_zones_scored`, `_test_sr_zones`, `_test_stop_hunt`, `_test_ict_amd` (+2)

### `performance_report.py`

**Description:** backtest/performance_report.py — Performance Report Generator (Day 74)

**Functions (3):**

- `generate_text_report()`
  - Generate a human-readable performance comparison report.
- `generate_telegram_report()`
  - Generate a concise Telegram alert with backtest results.
- `generate_strategy_contribution()`
  - Analyze which intelligence layers contribute to profit.

### `report.py`

**Classes (1):**

- `BacktestReport`
  - Methods: `to_text`, `save`, `_slug`

### `simulator.py`

**Classes (2):**

- `TradePosition`
- `ForexSimulator`
  - Methods: `__init__`, `open_position`, `evaluate_exit`, `force_close`, `_close_trade`, `_deterministic_slippage`, `_clean_pair`

### `statistical_validation.py`

**Classes (1):**

- `ValidationResult`
  - Container for all statistical validation results.
  - Methods: `to_dict`, `to_table`

**Functions (6):**

- `monte_carlo_permutation_test()`
  - Monte Carlo Permutation Test.
- `t_test_returns()`
  - One-sample t-test: is the mean trade return significantly > 0?
- `bootstrap_confidence_interval()`
  - Bootstrap 95% confidence interval for mean trade return.
- `walk_forward_analysis()`
  - Walk-Forward Analysis.
- `parameter_sensitivity_check()`
  - Parameter Sensitivity Analysis.
- `run_full_validation()`
  - Run all 5 statistical validation tests.

### `walk_forward.py`

**Classes (2):**

- `WalkForwardWindow`
  - Single walk-forward window result.
- `WalkForwardResult`
  - Complete walk-forward analysis result.
  - Methods: `to_dict`

**Functions (2):**

- `run_walk_forward()`
  - Run walk-forward analysis on closed trades.
- `print_walk_forward_table()`
  - Print walk-forward results as a table.

</details>

## risk/

**Purpose:** Risk management — strict risk manager, adversarial defenses, cognitive bias defenses, position sizing, circuit breakers

**Files:** 29

| File | Lines | Description | Key Classes | Key Functions |
|------|-------|-------------|-------------|---------------|
| `__init__.py` | 1 | — | — | — |
| `adversarial_defenses.py` | 1156 | risk/adversarial_defenses.py — Defenses against red team attacks | OrderAttempt, BrokerExecutionGuard, NewsEvent (+6) | — |
| `autonomous_risk.py` | 1102 | — | AutonomousRiskManager | — |
| `book_guardrails.py` | 539 | — | GuardrailResult | check_correlation_exposure, _default_fx_correlation_matrix, _lookup_correlation (+3) |
| `capital_manager.py` | 417 | — | CapitalManager | — |
| `circuit_breaker.py` | 302 | — | CircuitBreaker | — |
| `cognitive_bias_defenses.py` | 504 | risk/cognitive_bias_defenses.py — Defenses against your own mind | PreRegistration, PreRegistrationFramework, GraveyardEntry (+3) | — |
| `compounding.py` | 207 | risk/compounding.py — Compounding Growth Engine (Day 81+) | TradeRecord, CompoundingEngine | get_compounding_engine |
| `confidence_scaler.py` | 109 | risk/confidence_scaler.py — Confidence-Based Position Scaling (Day 76) | ConfidenceResult, ConfidenceScaler | get_confidence_scaler |
| `correlation_manager.py` | 161 | risk/correlation_manager.py — Correlation-Adjusted Position Sizing (Da | CorrelationResult, CorrelationManager | get_correlation_manager |
| `drawdown_controller.py` | 444 | — | DrawdownController | — |
| `drawdown_monitor.py` | 165 | risk/drawdown_monitor.py — Drawdown Monitoring (Day 75) | DrawdownStatus, DrawdownMonitor | get_drawdown_monitor |
| `entry_quality_guardrails.py` | 1715 | — | EntryQualityResult | _pip_value, _atr, _find_swing_lows (+12) |
| `expectancy.py` | 566 | — | ExpectancyCalculator | patch_analytics_expectancy |
| `exposure_manager.py` | 179 | risk/exposure_manager.py — Exposure & Correlation Manager (Day 75) | ExposureCheck, ExposureManager | get_exposure_manager |
| `kelly_calculator.py` | 137 | risk/kelly_calculator.py — Kelly Criterion Calculator (Day 76) | KellyResult, KellyCalculator | get_kelly_calculator |
| `kill_switch.py` | 223 | risk/kill_switch.py — Emergency Kill Switch (Day 75) | KillSwitch | get_kill_switch |
| `live_risk_manager.py` | 287 | risk/live_risk_manager.py — Live Risk Manager (Day 75) | CapitalTier, TradePermission, LiveRiskManager | get_live_risk_manager |
| `monte_carlo.py` | 350 | — | MonteCarloEngine | — |
| `portfolio_manager.py` | 497 | Portfolio Management System | PortfolioManager | — |
| `position_allocator.py` | 320 | — | PositionAllocator | — |
| `position_sizer.py` | 345 | risk/position_sizer.py — Advanced Position Sizing Engine (Day 76) | AdvancedPositionResult, PositionSizer | get_position_sizer |
| `risk_engine.py` | 390 | — | RiskEngine | — |
| `risk_reporter.py` | 175 | risk/risk_reporter.py — Risk Event Reporter (Day 75) | RiskReporter | get_risk_reporter |
| `risk_simulator.py` | 449 | — | RiskScenarioSimulator | — |
| `strict_risk_manager.py` | 525 | risk/strict_risk_manager.py — Strict Risk Manager (Fix for Fatal Flaw  | OpenPosition, TradeRecord, RiskCheckResult (+1) | — |
| `trade_frequency.py` | 187 | risk/trade_frequency.py — Trade Frequency Controller (Day 84+) | TradeRecord, TradeFrequencyController | _env_int, get_trade_frequency_controller |
| `trade_permission.py` | 194 | — | TradePermission | _test_mode |
| `volatility_adjuster.py` | 139 | risk/volatility_adjuster.py — Volatility-Based Position Adjustment (Da | VolatilityResult, VolatilityAdjuster | get_volatility_adjuster |

<details>
<summary>Detailed file descriptions (28 files)</summary>

### `adversarial_defenses.py`

**Description:** risk/adversarial_defenses.py — Defenses against red team attacks

**Classes (9):**

- `OrderAttempt`
  - Record of an order submission attempt.
- `BrokerExecutionGuard`
  - Defense against broker last-look, requotes, and execution failures.
  - Methods: `__init__`, `can_submit`, `record_attempt`, `should_retry_as_limit`, `get_stats`, `_prune_old_attempts`
- `NewsEvent`
  - Scheduled economic news event.
- `NewsEventBlackout`
  - Defense against spread widening + slippage during news.
  - Methods: `__init__`, `_load_calendar`, `_add_recurring_events`, `can_trade`, `should_close_position`, `add_event`
- `CrashRecoveryManager`
  - Defense against crash-induced orphaned positions.
  - Methods: `__init__`, `log_order_intent`, `confirm_order_filled`, `confirm_order_rejected`, `reconcile_on_startup`, `save_system_state`, `load_system_state`, `_read_wal` (+1)
- `StrategyDegradationMonitor`
  - Defense against strategy decay (edge erodes over time).
  - Methods: `__init__`, `record_trade`, `is_strategy_enabled`, `_check_degradation`, `_disable`, `re_enable`, `get_status`
- `VolatilityScaledSizer`
  - Defense against volatility clustering.
  - Methods: `__init__`, `calculate_risk_pct`, `_compute_atr`
- `OrderReconciler`
  - Defense against orphaned positions and state desync.
  - Methods: `__init__`, `set_broker_query_function`, `register_local_position`, `remove_local_position`, `start`, `stop`, `_poll_loop`, `_reconcile` (+1)
- `DataQualityValidator`
  - Defense against bad ticks, missing candles, and suspicious spikes.
  - Methods: `__init__`, `validate_bar`, `validate_dataframe`, `_compute_atr`, `_record_rejection`

### `autonomous_risk.py`

**Classes (1):**

- `AutonomousRiskManager`
  - Autonomous Risk Manager — AI Trader-এর Fund Manager Brain.
  - Methods: `__init__`, `evaluate_trade_signal`, `_update_risk_mode`, `set_mode`, `_calculate_adaptive_sl`, `record_trade_result`, `_update_performance_metrics`, `_get_strategy_capital_ranking` (+2)

### `book_guardrails.py`

**Classes (1):**

- `GuardrailResult`
  - Result of a single guardrail check.
  - Methods: `to_dict`

**Functions (6):**

- `check_correlation_exposure()`
  - Book Page 136 — Diversification rule.
- `_default_fx_correlation_matrix()`
  - Default approximate correlation matrix for major FX pairs.
- `_lookup_correlation()`
  - Lookup correlation with fallback to 0.
- `check_anti_revenge_trading()`
  - Book Pages 138-139 — "Don't chase losses" guardrail.
- `check_cost_aware_ev()`
  - Book Page 138 — "Don't ignore fees/commissions" guardrail.
- `run_all_guardrails()`
  - Run all 3 book guardrails + return aggregate result.

### `capital_manager.py`

**Classes (1):**

- `CapitalManager`
  - Portfolio Capital Allocation Manager.
  - Methods: `__init__`, `allocate`, `deallocate`, `compute_optimal_allocation`, `_get_best_strategy_for_pair`, `get_allocations`, `get_total_allocated`, `get_reserve` (+2)

### `circuit_breaker.py`

**Classes (1):**

- `CircuitBreaker`
  - AI-এর automatic protection system।
  - Methods: `__init__`, `allow_trade`, `record_result`, `manual_resume`, `force_learning_mode`, `reset_daily`, `get_status`, `print_status` (+2)

### `cognitive_bias_defenses.py`

**Description:** risk/cognitive_bias_defenses.py — Defenses against your own mind

**Classes (6):**

- `PreRegistration`
  - A pre-registered hypothesis — written BEFORE looking at data.
- `PreRegistrationFramework`
  - Defense against confirmation bias.
  - Methods: `__init__`, `register`, `resolve`, `get_confirmation_rate`, `_save`, `_load`
- `GraveyardEntry`
  - A failed strategy — recorded so it's not forgotten.
- `StrategyGraveyard`
  - Defense against survivorship bias.
  - Methods: `__init__`, `bury`, `is_already_failed`, `get_summary`, `_save`, `_load`
- `CalibrationTracker`
  - Defense against overconfidence bias.
  - Methods: `__init__`, `record_prediction`, `get_calibration`
- `SelectionAuditLog`
  - Defense against selection bias.
  - Methods: `__init__`, `log_selection`, `get_audit_summary`, `_save`, `_load`

### `compounding.py`

**Description:** risk/compounding.py — Compounding Growth Engine (Day 81+)

**Classes (2):**

- `TradeRecord`
  - One closed trade's PnL record.
- `CompoundingEngine`
  - Tracks realized PnL and computes a lot-size multiplier that grows
  - Methods: `__init__`, `_load_state`, `_save_state`, `record_profit`, `get_lot_multiplier`, `get_stats`, `reset`

**Functions (1):**

- `get_compounding_engine()`

### `confidence_scaler.py`

**Description:** risk/confidence_scaler.py — Confidence-Based Position Scaling (Day 76)

**Classes (2):**

- `ConfidenceResult`
  - Output of confidence scaling.
  - Methods: `to_dict`
- `ConfidenceScaler`
  - Confidence-based position size scaling.
  - Methods: `scale`

**Functions (1):**

- `get_confidence_scaler()`

### `correlation_manager.py`

**Description:** risk/correlation_manager.py — Correlation-Adjusted Position Sizing (Day 76)

**Classes (2):**

- `CorrelationResult`
  - Output of correlation adjustment.
  - Methods: `to_dict`
- `CorrelationManager`
  - Correlation-aware position sizing + portfolio heat tracking.
  - Methods: `adjust`, `_find_group`, `_calc_heat`

**Functions (1):**

- `get_correlation_manager()`

### `drawdown_controller.py`

**Classes (1):**

- `DrawdownController`
  - Account Protection System — Drawdown Controller.
  - Methods: `__init__`, `current_drawdown_pct`, `update_peak`, `get_protection_level`, `get_risk_scale`, `get_action`, `check_emergency`, `record_trade` (+2)

### `drawdown_monitor.py`

**Description:** risk/drawdown_monitor.py — Drawdown Monitoring (Day 75)

**Classes (2):**

- `DrawdownStatus`
  - Current drawdown status.
  - Methods: `to_dict`
- `DrawdownMonitor`
  - Monitors drawdown and activates protection modes.
  - Methods: `__init__`, `update`, `reset`, `status`

**Functions (1):**

- `get_drawdown_monitor()`

### `entry_quality_guardrails.py`

**Classes (1):**

- `EntryQualityResult`
  - Result of a single entry-quality check.
  - Methods: `to_dict`

**Functions (18):**

- `_pip_value()`
- `_atr()`
- `_find_swing_lows()`
  - Find swing lows in the last `lookback` bars (window=3 = 3 bars each side).
- `_find_swing_highs()`
  - Find swing highs in the last `lookback` bars.
- `_is_round_number()`
  - Check if price is near a round number. Returns (is_round, nearest_round).
- `check_chasing_filter()`
  - Red Flag 1: "Never enter after X-pip move in Y minutes without a pullback filter."
- `check_sl_swing_anchor()`
  - Red Flag 2: "A stop-loss chosen without reference to the last N swing lows is a guess."
- `check_tp_structure_validation()`
  - Red Flag 3: "TP should never be placed beyond the last visible price action."
- ... and 10 more

### `expectancy.py`

**Classes (1):**

- `ExpectancyCalculator`
  - Proper expectancy + system health evaluator।
  - Methods: `calculate`, `calculate_from_pnls`, `_calculate`, `_confidence_interval`, `_system_quality`, `_health_score`, `_recommendation`, `compare` (+2)

**Functions (1):**

- `patch_analytics_expectancy()`
  - আপনার analytics/analytics.py line 47 এ ভুল formula আছে:

### `exposure_manager.py`

**Description:** risk/exposure_manager.py — Exposure & Correlation Manager (Day 75)

**Classes (2):**

- `ExposureCheck`
  - Result of exposure check.
  - Methods: `to_dict`
- `ExposureManager`
  - Manages portfolio exposure and correlation risk.
  - Methods: `__init__`, `update_positions`, `check`, `_find_group`, `status`

**Functions (1):**

- `get_exposure_manager()`

### `kelly_calculator.py`

**Description:** risk/kelly_calculator.py — Kelly Criterion Calculator (Day 76)

**Classes (2):**

- `KellyResult`
  - Output of Kelly calculation.
  - Methods: `to_dict`
- `KellyCalculator`
  - Kelly Criterion position sizing with safety caps.
  - Methods: `calculate`

**Functions (1):**

- `get_kelly_calculator()`

### `kill_switch.py`

**Description:** risk/kill_switch.py — Emergency Kill Switch (Day 75)

**Classes (1):**

- `KillSwitch`
  - 3-level emergency brake with persistent state.
  - Methods: `__init__`, `_load`, `_save`, `check`, `_trigger_level1`, `_trigger_level2`, `_trigger_level3`, `_block` (+2)

**Functions (1):**

- `get_kill_switch()`

### `live_risk_manager.py`

**Description:** risk/live_risk_manager.py — Live Risk Manager (Day 75)

**Classes (3):**

- `CapitalTier`
  - One tier of the capital progression system.
- `TradePermission`
  - Result of trade permission check.
  - Methods: `to_dict`
- `LiveRiskManager`
  - Central risk controller — every trade passes through here.
  - Methods: `__init__`, `set_tier`, `record_trade_result`, `reset_daily`, `check_trade_permission`, `status`

**Functions (1):**

- `get_live_risk_manager()`

### `monte_carlo.py`

**Classes (1):**

- `MonteCarloEngine`
  - Monte Carlo Simulation Engine for Trading Risk Analysis.
  - Methods: `__init__`, `run`, `calculate_risk_of_ruin`, `find_optimal_risk`, `_empty_result`, `print_simulation_result`

### `portfolio_manager.py`

**Description:** Portfolio Management System

**Classes (1):**

- `PortfolioManager`
  - Manages portfolio-level risk and position sizing
  - Methods: `__init__`, `calculate_position_size`, `_apply_volatility_filter`, `_apply_correlation_filter`, `update_portfolio`, `calculate_portfolio_risk`, `_calculate_sharpe_ratio`, `_calculate_sortino_ratio` (+2)

### `position_allocator.py`

**Classes (1):**

- `PositionAllocator`
  - Position Sizing Engine with Kelly Criterion.
  - Methods: `__init__`, `calculate_kelly_risk`, `calculate_lot_size`, `get_minimum_rr`, `adjust_for_confidence`, `analyze_kelly`, `print_kelly_analysis`

### `position_sizer.py`

**Description:** risk/position_sizer.py — Advanced Position Sizing Engine (Day 76)

**Classes (2):**

- `AdvancedPositionResult`
  - Complete output of the advanced position sizing engine.
  - Methods: `to_dict`, `reason`
- `PositionSizer`
  - Advanced position sizing with 5-factor adjustment.
  - Methods: `__init__`, `calculate`, `_drawdown_mult`, `_streak_mult`

**Functions (1):**

- `get_position_sizer()`

### `risk_engine.py`

**Classes (1):**

- `RiskEngine`
  - Methods: `__init__`, `evaluate`, `_correlation_check`, `sync_open_positions`, `_load_daily`, `_fresh_day`, `_save_daily`, `record_trade_open` (+2)

### `risk_reporter.py`

**Description:** risk/risk_reporter.py — Risk Event Reporter (Day 75)

**Classes (1):**

- `RiskReporter`
  - Records risk events + sends Telegram alerts.
  - Methods: `__init__`, `_init_db`, `record_event`, `_send_telegram`, `get_recent_events`, `stats`

**Functions (1):**

- `get_risk_reporter()`

### `risk_simulator.py`

**Classes (1):**

- `RiskScenarioSimulator`
  - Risk Scenario Simulator — "What If" Analysis Engine.
  - Methods: `__init__`, `consecutive_losses`, `consecutive_wins`, `worst_day`, `best_day`, `worst_week`, `black_swan`, `strategy_failure` (+2)

### `strict_risk_manager.py`

**Description:** risk/strict_risk_manager.py — Strict Risk Manager (Fix for Fatal Flaw #8)

**Classes (4):**

- `OpenPosition`
  - Currently open position.
- `TradeRecord`
  - Historical trade record.
- `RiskCheckResult`
  - Result of a risk check.
- `StrictRiskManager`
  - Strict risk manager that prevents account blow-up.
  - Methods: `__init__`, `can_open_trade`, `_can_open_trade_unlocked`, `position_size`, `_position_size_unlocked`, `register_trade`, `close_trade`, `_clusters_for_pair` (+2)

### `trade_frequency.py`

**Description:** risk/trade_frequency.py — Trade Frequency Controller (Day 84+)

**Classes (2):**

- `TradeRecord`
- `TradeFrequencyController`
  - Tracks trades placed today and enforces min/max bounds.
  - Methods: `__init__`, `record_trade`, `trades_today`, `trade_count_today`, `can_trade_now`, `status`, `daily_summary`, `threshold_adjustment_hint`

**Functions (2):**

- `_env_int()`
- `get_trade_frequency_controller()`

### `trade_permission.py`

**Classes (1):**

- `TradePermission`
  - সব check পার হলে ALLOW, না হলে DENY।
  - Methods: `MIN_CONFIDENCE`, `MIN_ALIGNED_FACTORS`, `MIN_RR`, `check`, `print_summary`

**Functions (1):**

- `_test_mode()`
  - Lazy check — avoids importing config at module load (which would

### `volatility_adjuster.py`

**Description:** risk/volatility_adjuster.py — Volatility-Based Position Adjustment (Day 76)

**Classes (2):**

- `VolatilityResult`
  - Output of volatility adjustment.
  - Methods: `to_dict`
- `VolatilityAdjuster`
  - ATR-based volatility position adjustment.
  - Methods: `adjust`

**Functions (1):**

- `get_volatility_adjuster()`

</details>

## core/

**Purpose:** Core infrastructure — production trading system, graceful shutdown, runtime, signal fusion, monitoring

**Files:** 31

| File | Lines | Description | Key Classes | Key Functions |
|------|-------|-------------|-------------|---------------|
| `__init__.py` | 2 | — | — | — |
| `approval_mode.py` | 321 | — | ApprovalMode | — |
| `confidence_manager.py` | 189 | core/confidence_manager.py — Dynamic Weight Adjustment (Day 73) | ConfidenceManager | get_confidence_manager |
| `constants.py` | 139 | — | — | get_pip_size, get_pip_value_usd, clean_symbol (+2) |
| `decision_validator.py` | 164 | core/decision_validator.py — Final Decision Validation (Day 73) | ValidationResult, DecisionValidator | get_decision_validator |
| `event_bus.py` | 172 | core/event_bus.py — Unified cross-module event bus | Event, EventBus | get_bus, publish, subscribe |
| `exceptions.py` | 75 | — | TraderError, DataFetchError, DataValidationError (+8) | safe_execute |
| `execution_logger.py` | 151 | core/execution_logger.py — Structured execution logger for the trade p | — | _ensure_log_dir, log_event, log_signal_generated (+10) |
| `graceful_shutdown.py` | 262 | core/graceful_shutdown.py — Graceful Shutdown Manager | ShutdownState, GracefulShutdownManager | — |
| `health_monitor.py` | 364 | — | HealthStatus, HealthCheck, HealthSnapshot (+1) | get_health_monitor, register_mt5_health |
| `lifecycle.py` | 224 | core/lifecycle.py — Lifecycle manager | Phase, PhaseResult, LifecycleManager | get_lifecycle |
| `llm_cache.py` | 136 | core/llm_cache.py — Day 90 LLM Response Cache | CacheEntry, LLMCache | get_llm_cache |
| `llm_key_manager.py` | 942 | core/llm_key_manager.py — Multi-Key LLM Rotation Manager (Day 72+) | KeyHealth, _OpenAICompatClient, _OpenAICompatResponse (+5) | classify_llm_error, log_llm_call_failure, parse_groq_retry_after (+1) |
| `master_decision.py` | 460 | core/master_decision.py — Master Decision Engine (Day 73) | MasterDecision, MasterDecisionEngine | get_master_decision_engine |
| `monitoring_system.py` | 630 | Comprehensive Monitoring System | MonitoringSystem, HealthHandler | — |
| `obsolete.py` | 423 | core/obsolete.py — Explicit registry of obsolete / orphan modules | ObsoleteCategory, ObsoleteEntry | obsolete_index, obsolete_summary, is_obsolete |
| `orphan_cleanup.py` | 237 | core/orphan_cleanup.py — Auto-reconcile DB trades with MT5 live positi | — | reconcile_open_positions, _clear_stale_open_pairs, quick_close_all_db_open |
| `production_hardening.py` | 159 | — | PositionReconciler, HeartbeatMonitor, DynamicCorrelationMatrix (+1) | check_partial_fill, should_close_for_weekend, check_data_staleness (+3) |
| `production_trading_system.py` | 643 | core/production_trading_system.py — Unified Production Entry Point | ProductionTradingSystem | — |
| `professional_tools.py` | 393 | core/professional_tools.py — Professional trading enhancements | SessionAwarePairSelector, DynamicPositionSizer, JournalEntry (+1) | get_pair_selector, get_position_sizer, get_trade_journal |
| `regime_suppression.py` | 182 | core/regime_suppression.py — Day 97+ False-Signal Regime Suppression | RegimeSuppressor | get_regime_suppressor |
| `runtime.py` | 1305 | core/runtime.py — Central runtime wiring (composition root) | Runtime | get_runtime, boot_bootstrap, boot_persistence (+12) |
| `runtime_metrics.py` | 200 | core/runtime_metrics.py — Runtime metrics collector | StageStat, RuntimeMetrics | get_metrics, metric |
| `service_registry.py` | 302 | core/service_registry.py — Central service registry & dependency injec | ServiceStatus, ServiceRecord, ServiceNotFoundError (+2) | get_registry, reset_registry |
| `signal_fusion.py` | 203 | core/signal_fusion.py — 4-Layer Signal Fusion (Day 73) | LayerSignal, FusionResult, SignalFusion | get_signal_fusion |
| `signal_persistence.py` | 153 | core/signal_persistence.py — Day 97+ Signal Persistence Filter | SignalPersistenceFilter | get_signal_persistence_filter |
| `signal_scorer.py` | 279 | core/signal_scorer.py — Cumulative Score-Based Decision (Day 81+) | ScoreComponent, AdaptiveThreshold, SignalScorer | — |
| `trade_decision_log.py` | 156 | core/trade_decision_log.py — Records every trade decision (taken or no | — | log_decision, log_cycle_error, get_recent_decisions (+1) |
| `trader.py` | 2063 | — | AITrader, AutonomousTraderSystem, _NoOp | — |
| `trading_engine.py` | 84 | — | TradingEngine | — |
| `unified_signal.py` | 414 | core/unified_signal.py — Unified Signal Object (Day 81+) | UnifiedSignal | merge_signals |

<details>
<summary>Detailed file descriptions (29 files)</summary>

### `approval_mode.py`

**Classes (1):**

- `ApprovalMode`
  - AI trader-এর supervision layer।
  - Methods: `__init__`, `process`, `approve`, `reject`, `get_pending`, `set_mode`, `mode`, `mode_name` (+2)

### `confidence_manager.py`

**Description:** core/confidence_manager.py — Dynamic Weight Adjustment (Day 73)

**Classes (1):**

- `ConfidenceManager`
  - Manages dynamic weight adjustment based on historical accuracy.
  - Methods: `__init__`, `_init_db`, `record_outcome`, `_recalculate_weights`, `get_weights`, `get_layer_stats`, `status`

**Functions (1):**

- `get_confidence_manager()`

### `decision_validator.py`

**Description:** core/decision_validator.py — Final Decision Validation (Day 73)

**Classes (2):**

- `ValidationResult`
  - Final validation result.
  - Methods: `to_dict`
- `DecisionValidator`
  - Final validation gate for master decisions.
  - Methods: `validate`

**Functions (1):**

- `get_decision_validator()`

### `event_bus.py`

**Description:** core/event_bus.py — Unified cross-module event bus

**Classes (2):**

- `Event`
  - A single event on the bus.
  - Methods: `to_dict`
- `EventBus`
  - Thread-safe in-process pub/sub bus with bounded history.
  - Methods: `__init__`, `subscribe`, `publish`, `history`, `clear_history`, `subscriber_count`, `stats`

**Functions (3):**

- `get_bus()`
- `publish()`
- `subscribe()`

### `exceptions.py`

**Classes (11):**

- `TraderError`
  - Base exception for all trading system errors.
- `DataFetchError`
  - Failed to fetch market data.
- `DataValidationError`
  - Market data failed quality checks.
- `AnalysisError`
  - Analysis pipeline failed.
- `RiskError`
  - Risk engine rejected the trade.
- `ExecutionError`
  - Trade execution failed.
- `BrokerConnectionError`
  - MT5 broker connection failed.
- `LLMError`
  - AI/LLM analysis failed.
- `CircuitBreakerError`
  - Trading halted by circuit breaker.
- `ConfigurationError`
  - Invalid configuration detected.
- `TraderMemoryError`
  - Memory/database operation failed.

**Functions (1):**

- `safe_execute()`
  - Execute a function safely, catching and logging any exceptions.

### `execution_logger.py`

**Description:** core/execution_logger.py — Structured execution logger for the trade path.

**Functions (13):**

- `_ensure_log_dir()`
- `log_event()`
  - Write one JSONL line to logs/execution.log.
- `log_signal_generated()`
- `log_decision_resolved()`
- `log_risk_evaluated()`
- `log_permission_checked()`
- `log_approval_processed()`
- `log_router_start()`
- ... and 5 more

### `graceful_shutdown.py`

**Description:** core/graceful_shutdown.py — Graceful Shutdown Manager

**Classes (2):**

- `ShutdownState`
  - Tracks the shutdown process.
- `GracefulShutdownManager`
  - Manages graceful shutdown of the trading system.
  - Methods: `__init__`, `register_cleanup_callback`, `register_thread`, `is_shutting_down`, `request_shutdown`, `_handle_signal`, `run_cleanup`, `wait_for_completion` (+1)

### `health_monitor.py`

**Classes (4):**

- `HealthStatus`
- `HealthCheck`
- `HealthSnapshot`
  - Methods: `to_dict`
- `HealthMonitor`
  - Methods: `__init__`, `register_check`, `start`, `stop`, `_loop`, `run_once`, `_compute_overall`, `_collect_system_metrics` (+2)

**Functions (2):**

- `get_health_monitor()`
- `register_mt5_health()`

### `lifecycle.py`

**Description:** core/lifecycle.py — Lifecycle manager

**Classes (3):**

- `Phase`
- `PhaseResult`
- `LifecycleManager`
  - Drives the boot and shutdown of every runtime phase.
  - Methods: `__init__`, `register_phase`, `on_phase_complete`, `boot`, `shutdown`, `report`, `is_phase_complete`, `last_result`

**Functions (1):**

- `get_lifecycle()`

### `llm_cache.py`

**Description:** core/llm_cache.py — Day 90 LLM Response Cache

**Classes (2):**

- `CacheEntry`
- `LLMCache`
  - Methods: `__init__`, `make_key`, `get`, `set`, `clear`, `stats`

**Functions (1):**

- `get_llm_cache()`

### `llm_key_manager.py`

**Description:** core/llm_key_manager.py — Multi-Key LLM Rotation Manager (Day 72+)

**Classes (8):**

- `KeyHealth`
  - Tracks health of one API key.
  - Methods: `is_available`, `mark_success`, `mark_failure`, `to_dict`
- `_OpenAICompatClient`
  - Lightweight OpenAI-compatible REST client for Cerebras / SambaNova / OpenRouter.
  - Methods: `__init__`, `_do_create`
- `_OpenAICompatResponse`
  - Mimics openai.ChatCompletion response object.
  - Methods: `__init__`
- `_OpenAICompatChoice`
  - Methods: `__init__`
- `_OpenAICompatMessage`
  - Methods: `__init__`
- `LLMKeyManager`
  - Multi-key rotation manager for Groq + Gemini.
  - Methods: `__init__`, `_load_keys`, `get_groq_client`, `get_groq_key_info`, `mark_groq_success`, `mark_groq_failure`, `get_gemini_client`, `get_gemini_key_info` (+2)
- `_ChatNamespace`
  - Methods: `__init__`
- `_CompletionsNamespace`
  - Methods: `__init__`, `create`

**Functions (4):**

- `classify_llm_error()`
  - Classify LLM API failures without false positives (e.g. 'rate' in 'generate').
- `log_llm_call_failure()`
  - Log full LLM failure details for diagnosis.
- `parse_groq_retry_after()`
  - Parse 'Please try again in Xm Y.Ys' from a Groq 429 response.
- `get_llm_key_manager()`

### `master_decision.py`

**Description:** core/master_decision.py — Master Decision Engine (Day 73)

**Classes (2):**

- `MasterDecision`
  - The final output of the Master Decision Engine.
  - Methods: `to_dict`, `to_telegram_alert`
- `MasterDecisionEngine`
  - Central brain — collects, fuses, validates, and outputs the final decision.
  - Methods: `__init__`, `_init_db`, `decide`, `record_outcome`, `stats`

**Functions (1):**

- `get_master_decision_engine()`

### `monitoring_system.py`

**Description:** Comprehensive Monitoring System

**Classes (2):**

- `MonitoringSystem`
  - Comprehensive system monitoring and alerting
  - Methods: `__init__`, `_initialize_monitoring`, `start_monitoring`, `_monitoring_loop`, `_collect_system_metrics`, `_collect_application_metrics`, `_calculate_system_health`, `_is_trading_active` (+2)
- `HealthHandler`
  - Methods: `__init__`, `do_GET`, `log_message`

### `obsolete.py`

**Description:** core/obsolete.py — Explicit registry of obsolete / orphan modules

**Classes (2):**

- `ObsoleteCategory`
- `ObsoleteEntry`

**Functions (3):**

- `obsolete_index()`
  - Return a {path: entry} map for quick lookup.
- `obsolete_summary()`
  - Counts per category — useful for the final report.
- `is_obsolete()`

### `orphan_cleanup.py`

**Description:** core/orphan_cleanup.py — Auto-reconcile DB trades with MT5 live positions.

**Functions (3):**

- `reconcile_open_positions()`
  - Reconcile DB trades with MT5 live positions.
- `_clear_stale_open_pairs()`
  - Clear the open_pairs list in daily_risk.json so the RiskEngine's
- `quick_close_all_db_open()`
  - One-shot utility: mark ALL DB-OPEN trades as CLOSED.

### `production_hardening.py`

**Classes (4):**

- `PositionReconciler`
  - Methods: `__init__`, `register_position`, `unregister_position`, `set_mismatch_callback`, `start`, `stop`, `_run`, `_reconcile` (+1)
- `HeartbeatMonitor`
  - Methods: `__init__`, `update_state`, `start`, `stop`, `_run`, `_write`, `check_alive`
- `DynamicCorrelationMatrix`
  - Methods: `__init__`, `update`, `get_correlation`, `check_exposure`
- `R`

**Functions (6):**

- `check_partial_fill()`
- `should_close_for_weekend()`
- `check_data_staleness()`
- `is_candle_closed()`
- `validate_llm_output()`
- `should_use_llm_for_trading()`

### `production_trading_system.py`

**Description:** core/production_trading_system.py — Unified Production Entry Point

**Classes (1):**

- `ProductionTradingSystem`
  - Unified production trading system with all defenses wired.
  - Methods: `__init__`, `_init_layers`, `is_running`, `get_pairs`, `fetch_data`, `evaluate`, `execute_trade`, `close_trade` (+2)

### `professional_tools.py`

**Description:** core/professional_tools.py — Professional trading enhancements

**Classes (4):**

- `SessionAwarePairSelector`
  - Picks the most relevant pairs for the current trading session.
  - Methods: `__init__`, `select`, `select_with_session`
- `DynamicPositionSizer`
  - Adjusts lot size based on confidence, recent performance, and volatility.
  - Methods: `__init__`, `calculate`
- `JournalEntry`
  - One trade decision in the journal.
  - Methods: `to_csv_row`, `to_dict`
- `TradeJournal`
  - Append-only trade journal with CSV + JSONL persistence.
  - Methods: `__init__`, `next_cycle`, `log_decision`, `log_close`, `recent_decisions`, `stats`

**Functions (3):**

- `get_pair_selector()`
- `get_position_sizer()`
- `get_trade_journal()`

### `regime_suppression.py`

**Description:** core/regime_suppression.py — Day 97+ False-Signal Regime Suppression

**Classes (1):**

- `RegimeSuppressor`
  - Suppresses entry signals in known false-signal market conditions.
  - Methods: `should_suppress`, `get_regime_quality_score`

**Functions (1):**

- `get_regime_suppressor()`

### `runtime.py`

**Description:** core/runtime.py — Central runtime wiring (composition root)

**Classes (1):**

- `Runtime`
  - Facade that bundles every runtime infrastructure service.
  - Methods: `__init__`, `boot`, `_start_metrics_publisher`, `shutdown`, `is_booted`, `status`

**Functions (27):**

- `get_runtime()`
- `boot_bootstrap()`
  - Phase 1 — paths, config, logging, event bus, metrics, registry.
- `boot_persistence()`
  - Phase 2 — TraderDB + memory stores + KnowledgeStore.
- `boot_data()`
  - Phase 3 — DataFetcher, DataValidator, Indicators, AutomatedUpdater.
- `boot_market()`
  - Phase 4 — MarketScanner + CorrelationFilter + OpportunityRanker + MT5Connection.
- `boot_research()`
  - Phase 5 — ResearchAgent + HypothesisEngine + ExperimentRunner + Reports.
- `boot_fundamental()`
  - Phase 6 — NewsFilter + Day 66 NewsIntelligence engine.
- `boot_analysis()`
  - Phase 7 — Analysis engines + ML pipeline (Day 67-72).
- ... and 19 more

### `runtime_metrics.py`

**Description:** core/runtime_metrics.py — Runtime metrics collector

**Classes (2):**

- `StageStat`
  - Methods: `record`, `to_dict`
- `RuntimeMetrics`
  - Thread-safe collector of runtime metrics with rolling history.
  - Methods: `__init__`, `inc`, `set_gauge`, `get_counter`, `get_gauge`, `record_cycle`, `record_error`, `record_reconnect` (+2)

**Functions (2):**

- `get_metrics()`
- `metric()`

### `service_registry.py`

**Description:** core/service_registry.py — Central service registry & dependency injection

**Classes (5):**

- `ServiceStatus`
  - Lifecycle states a registered service can be in.
- `ServiceRecord`
  - Internal bookkeeping for a single registered service.
- `ServiceNotFoundError`
  - Raised when a service is requested that was never registered.
- `ServiceRegistrationError`
  - Raised when a service is registered twice with different factories.
- `ServiceRegistry`
  - Thread-safe service registry & DI container.
  - Methods: `__init__`, `register`, `register_instance`, `register_type`, `_index_type`, `resolve`, `try_resolve`, `resolve_type` (+2)

**Functions (2):**

- `get_registry()`
  - Get (or create) the global ServiceRegistry singleton.
- `reset_registry()`
  - Discard the global singleton — for tests only.

### `signal_fusion.py`

**Description:** core/signal_fusion.py — 4-Layer Signal Fusion (Day 73)

**Classes (3):**

- `LayerSignal`
  - One intelligence layer's signal.
  - Methods: `to_dict`
- `FusionResult`
  - Output of the signal fusion process.
  - Methods: `to_dict`
- `SignalFusion`
  - Fuses 4-layer signals into a master decision.
  - Methods: `fuse`, `_build_explanation`

**Functions (1):**

- `get_signal_fusion()`

### `signal_persistence.py`

**Description:** core/signal_persistence.py — Day 97+ Signal Persistence Filter

**Classes (1):**

- `SignalPersistenceFilter`
  - Filters out flip-flopping (unstable) signals.
  - Methods: `__init__`, `record`, `is_stable`, `get_flip_count`, `_direction`

**Functions (1):**

- `get_signal_persistence_filter()`

### `signal_scorer.py`

**Description:** core/signal_scorer.py — Cumulative Score-Based Decision (Day 81+)

**Classes (3):**

- `ScoreComponent`
- `AdaptiveThreshold`
  - Tracks recent trade timestamps and adjusts the threshold up/down
  - Methods: `__init__`, `record_trade`, `current`
- `SignalScorer`
  - Accumulates scores from all pipeline layers and decides whether
  - Methods: `__init__`, `_init_threshold`, `add`, `reset`, `total_score`, `max_possible`, `current_threshold`, `decide` (+2)

### `trade_decision_log.py`

**Description:** core/trade_decision_log.py — Records every trade decision (taken or not).

**Functions (4):**

- `log_decision()`
  - Write one decision record to memory/trade_decisions.jsonl.
- `log_cycle_error()`
  - Log a non-fatal error that occurred during a symbol cycle.
- `get_recent_decisions()`
  - Read the most recent N decision records (for dashboard/debugging).
- `get_summary()`
  - Get summary stats of all decisions in the log.

### `trader.py`

**Classes (3):**

- `AITrader`
  - Methods: `__init__`, `_publish`, `_stage`, `_record_error`, `_apply_advanced_sizing`, `get_signal`, `run_cycle`, `monitor_open_trades` (+2)
- `AutonomousTraderSystem`
  - Methods: `__init__`, `_on_webhook_command`, `_build_trader`, `run`, `stop`, `_select_cycle_symbols`, `_spawn_trader`, `backup_state` (+2)
- `_NoOp`
  - Methods: `__enter__`, `__exit__`

### `trading_engine.py`

**Classes (1):**

- `TradingEngine`
  - Thin composition root on top of AutonomousTraderSystem — adds the
  - Methods: `__init__`, `run`, `_print_banner`, `pending_approvals`, `approve`, `reject`, `circuit_breaker_status`, `resume_trading` (+1)

### `unified_signal.py`

**Description:** core/unified_signal.py — Unified Signal Object (Day 81+)

**Classes (1):**

- `UnifiedSignal`
  - Canonical signal object that flows through the entire pipeline:
  - Methods: `__post_init__`, `is_tradeable`, `is_wait`, `is_block`, `direction`, `rr_ratio`, `consensus_level`, `to_dict` (+2)

**Functions (1):**

- `merge_signals()`
  - Merge multiple agent-emitted UnifiedSignals into one canonical signal.

</details>

## agents/

**Purpose:** AI agents — analysis, decision, risk, learning, market, chart agents

**Files:** 8

| File | Lines | Description | Key Classes | Key Functions |
|------|-------|-------------|-------------|---------------|
| `__init__.py` | 1 | — | — | — |
| `analysis_agent.py` | 1702 | — | AnalysisAgent | — |
| `chart_agent.py` | 221 | — | ChartAgent | — |
| `decision_agent.py` | 412 | — | DecisionAgent | — |
| `learning_agent.py` | 103 | — | LearningAgent | — |
| `market_agent.py` | 96 | — | MarketAgent | — |
| `master_analyst.py` | 1277 | — | MasterAnalyst | — |
| `risk_agent.py` | 174 | — | RiskAgent | — |

<details>
<summary>Detailed file descriptions (7 files)</summary>

### `analysis_agent.py`

**Classes (1):**

- `AnalysisAgent`
  - Day 65 Unified Pipeline:
  - Methods: `__init__`, `run`

### `chart_agent.py`

**Classes (1):**

- `ChartAgent`
  - Methods: `__init__`, `start`, `calculate_sr_levels`, `open_tradingview`, `change_timeframe`, `add_indicator`, `_get_chart_price_range`, `_price_to_y` (+2)

### `decision_agent.py`

**Classes (1):**

- `DecisionAgent`
  - Day 42: MasterAnalyst output-কে primary signal source হিসেবে ব্যবহার করে।
  - Methods: `__init__`, `decide`, `_extract_pattern`, `_result`, `print_summary`, `get_ai_context`

### `learning_agent.py`

**Classes (1):**

- `LearningAgent`
  - প্রতিটা decision save করে।
  - Methods: `save_decision`, `get_performance_stats`, `_load`, `_save`

### `market_agent.py`

**Classes (1):**

- `MarketAgent`
  - Market data collect, validate, indicator calculate, regime detect।
  - Methods: `__init__`, `run`

### `master_analyst.py`

**Classes (1):**

- `MasterAnalyst`
  - Day 42 + Day 44 + Day 47 + Day 63 + Day 65 — Professional Forex Trader Brain।
  - Methods: `analyze`, `_build_context`, `_call_llm`, `_parse_response`, `_calculate_final_confidence`, `_fallback_result`, `get_ai_context`, `print_summary`

### `risk_agent.py`

**Classes (1):**

- `RiskAgent`
  - Risk Management Agent — calculates lot size, SL, TP from signal + ATR.
  - Methods: `__init__`, `calculate`, `_no_trade`, `print_summary`, `get_ai_context`

</details>

## broker/

**Purpose:** MT5 broker integration — connection, order management, position management, data validation

**Files:** 14

| File | Lines | Description | Key Classes | Key Functions |
|------|-------|-------------|-------------|---------------|
| `__init__.py` | 20 | — | — | — |
| `account_manager.py` | 316 | — | AccountManager | _spread_limit, _test_mode |
| `data_validator.py` | 228 | — | DataValidator | — |
| `economic_calendar.py` | 155 | — | EconomicCalendar | — |
| `health_monitor.py` | 187 | — | HealthMonitor | — |
| `journal_bridge.py` | 244 | — | JournalBridge | — |
| `market_data_manager.py` | 115 | — | MarketDataManager | — |
| `mt5_connection.py` | 332 | — | MT5Connection | — |
| `mt5_data.py` | 289 | — | MT5DataFeed | _mt5_timeframe, normalize_timeframe |
| `order_manager.py` | 612 | — | OrderManager | _resolve_filling_mode |
| `position_manager.py` | 723 | — | PositionManager | _pip, _pips |
| `safety_guard.py` | 95 | — | SafetyGuard | — |
| `spread_monitor.py` | 62 | — | SpreadMonitor | — |
| `symbol_manager.py` | 463 | — | SymbolManager | — |

<details>
<summary>Detailed file descriptions (13 files)</summary>

### `account_manager.py`

**Classes (1):**

- `AccountManager`
  - MT5 account state বোঝে এবং trade নেওয়ার আগে safety check করে।
  - Methods: `__init__`, `get_account_snapshot`, `_get_margin_mode`, `_get_account_type_label`, `_classify_health`, `print_snapshot`, `resolve_symbol`, `market_status` (+1)

**Functions (2):**

- `_spread_limit()`
- `_test_mode()`

### `data_validator.py`

**Classes (1):**

- `DataValidator`
  - Candle list validate করে, gap detect/fill করে, duplicate বাদ দেয়,
  - Methods: `__init__`, `validate_and_fill`, `_filter_invalid`, `_is_invalid`, `_dedupe`, `_detect_and_fill_gaps`, `_attempt_recover`, `_flat_fill` (+2)

### `economic_calendar.py`

**Classes (1):**

- `EconomicCalendar`
  - Usage:
  - Methods: `__init__`, `add_event`, `clear_past_events`, `get_today_events`, `get_upcoming_events`, `check_news_window`, `_currencies_for`, `_load` (+1)

### `health_monitor.py`

**Classes (1):**

- `HealthMonitor`
  - MT5 connection-এর health track করে এবং disconnect হলে
  - Methods: `__init__`, `check_once`, `run_loop`, `_is_connection_ok`, `_attempt_reconnect`, `get_status`, `print_status`

### `journal_bridge.py`

**Classes (1):**

- `JournalBridge`
  - MT5 demo trade-কে paper trade-এর মতো same `trades` table-এ লেখে,
  - Methods: `__init__`, `log_mt5_open`, `log_mt5_close`, `get_combined_stats`, `get_stats_by_source`, `export_history_xml`, `export_history_html`, `export_history_csv`

### `market_data_manager.py`

**Classes (1):**

- `MarketDataManager`
  - সব downstream agent (rule engine, AIAnalyst, RiskEngine) এই class
  - Methods: `__init__`, `get_clean_bundle`, `scan_market`, `print_status_report`

### `mt5_connection.py`

**Classes (1):**

- `MT5Connection`
  - Methods: `__init__`, `connect`, `_try_connect`, `disconnect`, `is_alive`, `get_account_info`, `_require_connected`, `get_tick` (+2)

### `mt5_data.py`

**Classes (1):**

- `MT5DataFeed`
  - MT5 থেকে tick + multi-timeframe candle data নেয়।
  - Methods: `get_tick`, `get_tick_stream`, `print_tick_stream`, `get_candles`, `get_multi_timeframe`, `print_multi_timeframe_status`, `save_live_csv`

**Functions (2):**

- `_mt5_timeframe()`
  - Label string ('M15', '15m', 'm15') কে actual mt5.TIMEFRAME_*
- `normalize_timeframe()`
  - যেকোনো timeframe string-কে MT5 standard uppercase-এ convert করে।

### `order_manager.py`

**Classes (1):**

- `OrderManager`
  - MT5-এ actual order পাঠায়, modify করে, close করে।
  - Methods: `__init__`, `place_market_order`, `place_limit_order`, `place_stop_order`, `place_stop_limit_order`, `modify_order`, `close_order`, `close_all_orders` (+2)

**Functions (1):**

- `_resolve_filling_mode()`
  - Pick the most permissive filling mode the broker supports.

### `position_manager.py`

**Classes (1):**

- `PositionManager`
  - Usage:
  - Methods: `__init__`, `register_open`, `poll_once`, `run_loop`, `_apply_management_rules`, `_check_breakeven`, `_check_trailing`, `_check_partial_close` (+2)

**Functions (2):**

- `_pip()`
- `_pips()`

### `safety_guard.py`

**Classes (1):**

- `SafetyGuard`
  - Unified pre-trade safety check combining:
  - Methods: `__init__`, `check`, `get_status`

### `spread_monitor.py`

**Classes (1):**

- `SpreadMonitor`
  - Usage:
  - Methods: `check`

### `symbol_manager.py`

**Classes (1):**

- `SymbolManager`
  - Multi-pair scanner — broker symbol resolve করা AccountManager-এর
  - Methods: `__init__`, `resolve_all`, `scan`, `_classify`, `print_scan`, `get_symbol_specification`, `validate_order_against_spec`, `check_volume_limit` (+2)

</details>

## data/

**Purpose:** Data layer — fetcher, indicators, live feed, data orchestrator

**Files:** 8

| File | Lines | Description | Key Classes | Key Functions |
|------|-------|-------------|-------------|---------------|
| `automated_updater.py` | 374 | Automated Forex Data Update System | ForexDataUpdater | — |
| `data_orchestrator.py` | 455 | data/data_orchestrator.py — Day 93 Unified Data Orchestrator | DataOrchestrator | _get_mt5_credentials, get_data_orchestrator |
| `fetcher.py` | 921 | — | DataFetcher | _build_timeframe_map |
| `indicators.py` | 111 | — | Indicators | — |
| `indicators_ext.py` | 699 | data/indicators_ext.py — Day 93 Extended Indicator Library (pandas-ta) | ExtendedIndicators | _safe_set |
| `live_feed.py` | 322 | data/live_feed.py — MT5 Live Tick Intelligence (Day 81+) | TickSnapshot, LiveFeed | _get_spread_limit, get_live_feed |
| `validator.py` | 126 | — | DataValidator | — |
| `verify_data_coverage.py` | 488 | Data Coverage Verification Script | — | verify_data_coverage, check_historical_data, check_data_freshness (+5) |

<details>
<summary>Detailed file descriptions (8 files)</summary>

### `automated_updater.py`

**Description:** Automated Forex Data Update System

**Classes (1):**

- `ForexDataUpdater`
  - Automated daily data update system with validation and error handling
  - Methods: `__init__`, `update_all_pairs`, `update_single_pair`, `fetch_forex_data`, `_fetch_from_oanda`, `_fetch_from_yahoo`, `_generate_synthetic_data`, `validate_and_clean_data` (+2)

### `data_orchestrator.py`

**Description:** data/data_orchestrator.py — Day 93 Unified Data Orchestrator

**Classes (1):**

- `DataOrchestrator`
  - Unified data access layer — MT5 first, API fallback.
  - Methods: `__init__`, `_get_fetcher`, `_get_mt5`, `get_candles`, `_normalize_mt5_candles`, `get_account_info`, `get_open_positions`, `get_pending_orders` (+2)

**Functions (2):**

- `_get_mt5_credentials()`
  - Read MT5 credentials from environment.
- `get_data_orchestrator()`

### `fetcher.py`

**Classes (1):**

- `DataFetcher`
  - MT5-first data fetcher.
  - Methods: `__init__`, `_detect_source`, `fetch_ohlcv`, `_fetch_mt5`, `_fetch_tvdatafeed`, `_fetch_yfinance`, `_to_yahoo_symbol`, `_tf_to_yfinance_interval` (+2)

**Functions (1):**

- `_build_timeframe_map()`
  - Populate TIMEFRAME_MAP from live mt5 constants (called once, lazily).

### `indicators.py`

**Classes (1):**

- `Indicators`
  - Methods: `add_all`, `add_moving_averages`, `add_rsi`, `add_macd`, `add_bollinger_bands`, `add_atr`, `add_trend_signals`, `get_summary` (+2)

### `indicators_ext.py`

**Description:** data/indicators_ext.py — Day 93 Extended Indicator Library (pandas-ta)

**Classes (1):**

- `ExtendedIndicators`
  - Comprehensive indicator layer built on pandas-ta.
  - Methods: `add_all`, `add_moving_averages`, `add_momentum`, `_rsi_zone`, `add_volatility`, `add_volume_indicators`, `add_trend_strength`, `add_volume_rsi` (+2)

**Functions (1):**

- `_safe_set()`
  - Safely assign a pandas-ta result to a single column.

### `live_feed.py`

**Description:** data/live_feed.py — MT5 Live Tick Intelligence (Day 81+)

**Classes (2):**

- `TickSnapshot`
  - One moment-in-time view of a symbol's live market state.
  - Methods: `mid`, `is_tradeable`, `to_dict`
- `LiveFeed`
  - Real-time MT5 tick intelligence layer.
  - Methods: `__init__`, `get_snapshot`, `get_multi_snapshot`, `_compute_velocity`, `_compute_pressure`, `_compute_spread_median`, `_classify_liquidity`, `is_safe_to_trade`

**Functions (2):**

- `_get_spread_limit()`
- `get_live_feed()`

### `validator.py`

**Classes (1):**

- `DataValidator`
  - OHLCV data fetch করার পরে এই class দিয়ে validate করো।
  - Methods: `validate`, `_check_empty`, `_check_columns`, `_check_missing_values`, `_check_duplicates`, `_check_price_sanity`, `_check_ohlc_logic`, `_check_gaps`

### `verify_data_coverage.py`

**Description:** Data Coverage Verification Script

**Functions (8):**

- `verify_data_coverage()`
  - Verify data coverage meets requirements
- `check_historical_data()`
  - Check historical data availability
- `check_data_freshness()`
  - Check data freshness
- `check_automation_capability()`
  - Check automated update capability
- `check_data_validation()`
  - Check data validation implementation
- `check_missing_data_handling()`
  - Check missing data handling
- `check_june_2026_data()`
  - Check if data is available through June 21, 2026
- `generate_report()`
  - Generate comprehensive verification report

</details>

## ml/

**Purpose:** Machine learning — feature engineering, model training, ensemble, RL agent, walk-forward

**Files:** 28

| File | Lines | Description | Key Classes | Key Functions |
|------|-------|-------------|-------------|---------------|
| `__init__.py` | 1 | — | — | — |
| `confidence_fusion.py` | 216 | ml/confidence_fusion.py — Confidence fusion engine (Day 70) | FusionResult, ConfidenceFusion | get_confidence_fusion |
| `data_preprocessor.py` | 193 | ml/data_preprocessor.py — Data preprocessor (Day 68) | ProcessedDataset, DataPreprocessor | get_preprocessor |
| `dataset_builder.py` | 164 | ml/dataset_builder.py — Training dataset assembler (Day 69) | Dataset, DatasetBuilder | get_dataset_builder |
| `diagnostic.py` | 189 | ml/diagnostic.py — ML Diagnostic Engine (Day 74) | DiagnosticReport, MLDiagnostic | get_ml_diagnostic |
| `ensemble.py` | 394 | ml/ensemble.py — Ensemble Engine: AI Brain Fusion Layer (Day 70) | EnsembleDecision, EnsembleEngine | get_ensemble_engine |
| `ensemble_store.py` | 200 | ml/ensemble_store.py — Ensemble decision persistence (Day 70) | EnsembleStore | get_ensemble_store |
| `feature_engineer.py` | 604 | ml/feature_engineer.py — Feature Engineering Layer (Day 68) | FeatureEngineer | _safe_float, _pips, get_feature_engineer |
| `feature_selector.py` | 266 | ml/feature_selector.py — Feature selection + drift detection (Day 68) | ImportanceResult, DriftResult, FeatureSelector | get_feature_selector |
| `feature_store.py` | 224 | ml/feature_store.py — Persistent feature store (Day 68) | FeatureStore | get_feature_store |
| `forecast_engine.py` | 261 | ml/forecast_engine.py — Day 97 Conservative Time-Series Forecast | ForecastEngine | get_forecast_engine |
| `label_generator.py` | 212 | ml/label_generator.py — Target variable generator (Day 68) | LabelResult, LabelGenerator | get_label_generator |
| `model_evaluator.py` | 294 | ml/model_evaluator.py — Model evaluation (Day 69) | ModelMetrics, ModelEvaluator, WalkForwardValidator | get_evaluator, get_walk_forward_validator |
| `model_predictor.py` | 321 | ml/model_predictor.py — Live ensemble prediction (Day 69) | ModelPredictor | get_model_predictor |
| `model_store.py` | 243 | ml/model_store.py — Model version control + persistence (Day 69) | ModelStore | get_model_store |
| `model_trainer.py` | 365 | ml/model_trainer.py — Multi-model training pipeline (Day 69) | TrainingResult, ModelTrainer | get_model_trainer |
| `monte_carlo.py` | 193 | ml/monte_carlo.py — Monte Carlo Trade Simulation (Day 72) | MonteCarloResult, MonteCarloSimulator | get_monte_carlo_simulator |
| `regime_test.py` | 204 | ml/regime_test.py — Market Regime Robustness Test (Day 72) | RegimeResult, RegimeTestResult, RegimeTester | get_regime_tester |
| `reward_engine.py` | 157 | ml/reward_engine.py — RL Reward System (Day 71) | RewardBreakdown, RewardEngine | get_reward_engine |
| `rl_agent.py` | 272 | ml/rl_agent.py — PPO RL Agent (Day 71) | RLAction, RLAgent, TrainingCallback | get_rl_agent |
| `rl_environment.py` | 414 | ml/rl_environment.py — Forex Trading RL Environment (Day 71) | Position, ForexTradingEnv | — |
| `rl_policy_store.py` | 149 | ml/rl_policy_store.py — RL policy versioning (Day 71) | RLPolicyStore | get_rl_policy_store |
| `sensitivity_test.py` | 174 | ml/sensitivity_test.py — Parameter Sensitivity + Leakage Detection (Da | SensitivityResult, LeakageResult, SensitivityTester | get_sensitivity_tester |
| `train_rl.py` | 266 | ml/train_rl.py — RL Training Script with Curriculum Learning (Day 71) | TrainingStage | load_historical_data, build_features_df, split_by_regime (+1) |
| `validation.py` | 437 | ml/validation.py — Validation Engine: The Quant Gatekeeper (Day 72) | ValidationReport, ValidationEngine | get_validation_engine |
| `validation_report.py` | 125 | ml/validation_report.py — Validation Report Generator (Day 72) | — | generate_text_report, validate_all_models, get_validation_status |
| `voting_engine.py` | 186 | ml/voting_engine.py — Voting & agreement system (Day 70) | ModelVote, VoteResult, VotingEngine | get_voting_engine |
| `walk_forward.py` | 214 | ml/walk_forward.py — Walk-Forward Validation (Day 72) | WalkForwardFold, WalkForwardResult, WalkForwardValidator | get_walk_forward_validator |

<details>
<summary>Detailed file descriptions (27 files)</summary>

### `confidence_fusion.py`

**Description:** ml/confidence_fusion.py — Confidence fusion engine (Day 70)

**Classes (2):**

- `FusionResult`
  - Output of the confidence fusion process.
  - Methods: `to_dict`
- `ConfidenceFusion`
  - Fuses multi-model confidences into one calibrated ensemble confidence.
  - Methods: `__init__`, `_load_weights`, `_get_regime_adjustments`, `update_performance`, `_performance_adjustment`, `fuse`

**Functions (1):**

- `get_confidence_fusion()`

### `data_preprocessor.py`

**Description:** ml/data_preprocessor.py — Data preprocessor (Day 68)

**Classes (2):**

- `ProcessedDataset`
  - Result of preprocessing a feature matrix.
  - Methods: `summary`
- `DataPreprocessor`
  - Prepares data for ML training with strict leakage prevention.
  - Methods: `__init__`, `clean_features`, `chronological_split`, `fit_scaler`, `transform`, `save_scaler`, `load_scaler`, `process`

**Functions (1):**

- `get_preprocessor()`

### `dataset_builder.py`

**Description:** ml/dataset_builder.py — Training dataset assembler (Day 69)

**Classes (2):**

- `Dataset`
  - A chronologically-split ML dataset.
  - Methods: `summary`
- `DatasetBuilder`
  - Builds chronologically-split training datasets.
  - Methods: `__init__`, `build_from_store`, `build_from_dataframe`

**Functions (1):**

- `get_dataset_builder()`

### `diagnostic.py`

**Description:** ml/diagnostic.py — ML Diagnostic Engine (Day 74)

**Classes (2):**

- `DiagnosticReport`
  - ML diagnostic report.
  - Methods: `to_dict`
- `MLDiagnostic`
  - Diagnoses ML model and data quality issues.
  - Methods: `diagnose`, `_get_feature_importance`, `_rank_features`, `_find_bad_features`, `_suggest_hyperparameters`, `_general_recommendations`

**Functions (1):**

- `get_ml_diagnostic()`

### `ensemble.py`

**Description:** ml/ensemble.py — Ensemble Engine: AI Brain Fusion Layer (Day 70)

**Classes (2):**

- `EnsembleDecision`
  - The final output of the EnsembleEngine — the single trade decision.
  - Methods: `to_dict`, `to_telegram_alert`
- `EnsembleEngine`
  - The AI Brain Fusion Layer — combines all intelligence into one decision.
  - Methods: `__init__`, `decide`, `record_outcome`, `update_model_weights_from_performance`, `stats`

**Functions (1):**

- `get_ensemble_engine()`

### `ensemble_store.py`

**Description:** ml/ensemble_store.py — Ensemble decision persistence (Day 70)

**Classes (1):**

- `EnsembleStore`
  - Persists ensemble decisions + tracks model performance.
  - Methods: `__init__`, `_conn`, `_init_db`, `save_decision`, `update_outcome`, `update_model_performance`, `get_model_performance`, `stats`

**Functions (1):**

- `get_ensemble_store()`

### `feature_engineer.py`

**Description:** ml/feature_engineer.py — Feature Engineering Layer (Day 68)

**Classes (1):**

- `FeatureEngineer`
  - Generates a flat ~110-feature dict from market data + analysis contexts.
  - Methods: `__init__`, `build_feature_vector`, `_price_features`, `_indicator_features`, `_pattern_features`, `_context_features`, `_mtf_features`, `_smc_liquidity_features` (+1)

**Functions (3):**

- `_safe_float()`
- `_pips()`
  - Convert a price difference to pips (handles JPY pairs).
- `get_feature_engineer()`

### `feature_selector.py`

**Description:** ml/feature_selector.py — Feature selection + drift detection (Day 68)

**Classes (3):**

- `ImportanceResult`
  - Feature importance ranking.
  - Methods: `to_dict`
- `DriftResult`
  - Per-feature drift report.
  - Methods: `to_dict`
- `FeatureSelector`
  - Feature importance + drift detection.
  - Methods: `compute_importance`, `_compute_with_method`, `_lightgbm_importance`, `_forest_importance`, `_variance_importance`, `detect_drift`, `_psi`, `aggregate_multi_timeframe`

**Functions (1):**

- `get_feature_selector()`

### `feature_store.py`

**Description:** ml/feature_store.py — Persistent feature store (Day 68)

**Classes (1):**

- `FeatureStore`
  - SQLite-backed persistent feature store.
  - Methods: `__init__`, `_conn`, `_init_db`, `save_features`, `update_outcome`, `load_training_data`, `stats`, `save_importance`

**Functions (1):**

- `get_feature_store()`

### `forecast_engine.py`

**Description:** ml/forecast_engine.py — Day 97 Conservative Time-Series Forecast

**Classes (1):**

- `ForecastEngine`
  - Conservative short-term price forecast (extra vote only).
  - Methods: `forecast`, `_get_or_compute`, `_compute_ma`, `_compute_ema`, `_compute_rsi`, `_compute_atr`, `_fallback`, `get_ai_context` (+1)

**Functions (1):**

- `get_forecast_engine()`

### `label_generator.py`

**Description:** ml/label_generator.py — Target variable generator (Day 68)

**Classes (2):**

- `LabelResult`
  - Labels for a single row.
  - Methods: `to_dict`
- `LabelGenerator`
  - Generates forward-looking labels for ML training.
  - Methods: `__init__`, `label_for_row`, `label_dataframe`, `label_summary`

**Functions (1):**

- `get_label_generator()`

### `model_evaluator.py`

**Description:** ml/model_evaluator.py — Model evaluation (Day 69)

**Classes (3):**

- `ModelMetrics`
  - Comprehensive evaluation metrics for one model.
  - Methods: `to_dict`, `summary_line`
- `ModelEvaluator`
  - Evaluates classification models with ML + trading metrics.
  - Methods: `evaluate`, `compare_models`
- `WalkForwardValidator`
  - Rolling window validation for time-series models.
  - Methods: `__init__`, `run`

**Functions (2):**

- `get_evaluator()`
- `get_walk_forward_validator()`

### `model_predictor.py`

**Description:** ml/model_predictor.py — Live ensemble prediction (Day 69)

**Classes (1):**

- `ModelPredictor`
  - Live ensemble predictor combining XGBoost + RF + LSTM.
  - Methods: `__init__`, `_init_predictions_db`, `_load_models`, `_load_scaler`, `predict`, `_record_prediction`, `_record_ensemble`, `update_actual_result` (+1)

**Functions (1):**

- `get_model_predictor()`

### `model_store.py`

**Description:** ml/model_store.py — Model version control + persistence (Day 69)

**Classes (1):**

- `ModelStore`
  - Versioned model persistence with rollback support.
  - Methods: `__init__`, `_load_registry`, `_save_registry`, `_pair_dir`, `save_model`, `load_model`, `rollback`, `list_models` (+1)

**Functions (1):**

- `get_model_store()`

### `model_trainer.py`

**Description:** ml/model_trainer.py — Multi-model training pipeline (Day 69)

**Classes (2):**

- `TrainingResult`
  - Result of training all models for one pair.
  - Methods: `to_dict`
- `ModelTrainer`
  - Trains XGBoost, RandomForest, and LSTM models.
  - Methods: `__init__`, `train_all`, `_train_xgboost`, `_train_random_forest`, `_train_lstm`, `walk_forward_validate`

**Functions (1):**

- `get_model_trainer()`

### `monte_carlo.py`

**Description:** ml/monte_carlo.py — Monte Carlo Trade Simulation (Day 72)

**Classes (2):**

- `MonteCarloResult`
  - Monte Carlo simulation results.
  - Methods: `to_dict`
- `MonteCarloSimulator`
  - Monte Carlo trade-sequence simulation.
  - Methods: `__init__`, `simulate`, `_profit_factor`, `_max_drawdown_pct`

**Functions (1):**

- `get_monte_carlo_simulator()`

### `regime_test.py`

**Description:** ml/regime_test.py — Market Regime Robustness Test (Day 72)

**Classes (3):**

- `RegimeResult`
  - One regime's performance.
  - Methods: `to_dict`
- `RegimeTestResult`
  - Aggregated regime robustness results.
  - Methods: `to_dict`
- `RegimeTester`
  - Tests model robustness across market regimes.
  - Methods: `test`, `_classify_regimes`

**Functions (1):**

- `get_regime_tester()`

### `reward_engine.py`

**Description:** ml/reward_engine.py — RL Reward System (Day 71)

**Classes (2):**

- `RewardBreakdown`
  - Detailed reward breakdown for one step.
  - Methods: `to_dict`
- `RewardEngine`
  - Calculates RL rewards with anti-hacking protections.
  - Methods: `__init__`, `calculate`

**Functions (1):**

- `get_reward_engine()`

### `rl_agent.py`

**Description:** ml/rl_agent.py — PPO RL Agent (Day 71)

**Classes (3):**

- `RLAction`
  - RL agent's action recommendation.
  - Methods: `to_dict`
- `RLAgent`
  - PPO-based RL agent with heuristic fallback.
  - Methods: `__init__`, `_check_sb3`, `load_model`, `predict`, `_heuristic_predict`, `train`, `status`
- `TrainingCallback`
  - Methods: `__init__`, `_on_step`

**Functions (1):**

- `get_rl_agent()`

### `rl_environment.py`

**Description:** ml/rl_environment.py — Forex Trading RL Environment (Day 71)

**Classes (2):**

- `Position`
  - Open position state.
- `ForexTradingEnv`
  - Gym-compatible forex trading environment.
  - Methods: `__init__`, `reset`, `step`, `_open_position`, `_close_position`, `_check_sl_tp`, `_close_at_price`, `_get_state` (+2)

### `rl_policy_store.py`

**Description:** ml/rl_policy_store.py — RL policy versioning (Day 71)

**Classes (1):**

- `RLPolicyStore`
  - Versioned RL policy persistence.
  - Methods: `__init__`, `_load_registry`, `_save_registry`, `save_policy`, `load_policy`, `rollback`, `list_versions`, `stats`

**Functions (1):**

- `get_rl_policy_store()`

### `sensitivity_test.py`

**Description:** ml/sensitivity_test.py — Parameter Sensitivity + Leakage Detection (Day 72)

**Classes (3):**

- `SensitivityResult`
  - Parameter sensitivity test result.
  - Methods: `to_dict`
- `LeakageResult`
  - Data leakage detection result.
  - Methods: `to_dict`
- `SensitivityTester`
  - Tests model robustness to parameter perturbation + detects data leakage.
  - Methods: `test_sensitivity`, `detect_leakage`

**Functions (1):**

- `get_sensitivity_tester()`

### `train_rl.py`

**Description:** ml/train_rl.py — RL Training Script with Curriculum Learning (Day 71)

**Classes (1):**

- `TrainingStage`
  - One stage of curriculum learning.

**Functions (4):**

- `load_historical_data()`
  - Load historical OHLCV data for training.
- `build_features_df()`
  - Build feature vectors for each row using FeatureEngineer.
- `split_by_regime()`
  - Split dataframe by market regime for curriculum learning.
- `train_rl_agent()`
  - Train the RL agent with optional curriculum learning.

### `validation.py`

**Description:** ml/validation.py — Validation Engine: The Quant Gatekeeper (Day 72)

**Classes (2):**

- `ValidationReport`
  - Complete validation report for one model.
  - Methods: `to_dict`, `to_telegram_alert`
- `ValidationEngine`
  - The Quant Gatekeeper — validates models before live deployment.
  - Methods: `__init__`, `_init_db`, `validate`, `_benchmark_comparison`, `_save_report`, `_update_champion`, `get_champion`, `stats`

**Functions (1):**

- `get_validation_engine()`

### `validation_report.py`

**Description:** ml/validation_report.py — Validation Report Generator (Day 72)

**Functions (3):**

- `generate_text_report()`
  - Generate a human-readable text validation report.
- `validate_all_models()`
  - Validate all models in the ModelStore for a given pair.
- `get_validation_status()`
  - Get overall validation status for dashboard.

### `voting_engine.py`

**Description:** ml/voting_engine.py — Voting & agreement system (Day 70)

**Classes (3):**

- `ModelVote`
  - One model's vote.
  - Methods: `to_dict`
- `VoteResult`
  - Result of the voting process.
  - Methods: `to_dict`
- `VotingEngine`
  - Collects model votes and produces a decision with position sizing.
  - Methods: `vote`

**Functions (1):**

- `get_voting_engine()`

### `walk_forward.py`

**Description:** ml/walk_forward.py — Walk-Forward Validation (Day 72)

**Classes (3):**

- `WalkForwardFold`
  - One fold of walk-forward validation.
  - Methods: `to_dict`
- `WalkForwardResult`
  - Aggregated walk-forward results.
  - Methods: `to_dict`
- `WalkForwardValidator`
  - Rolling-window validation for time-series models.
  - Methods: `__init__`, `validate`

**Functions (1):**

- `get_walk_forward_validator()`

</details>

## ai/

**Files:** 4

| File | Lines | Description | Key Classes | Key Functions |
|------|-------|-------------|-------------|---------------|
| `__init__.py` | 1 | — | — | — |
| `ai_analyst.py` | 513 | — | AIAnalyst | — |
| `automated_retraining.py` | 378 | Automated Model Retraining System | AutomatedRetrainingSystem | — |
| `model_versioning.py` | 231 | Model Versioning System | ModelVersionManager | — |

<details>
<summary>Detailed file descriptions (3 files)</summary>

### `ai_analyst.py`

**Classes (1):**

- `AIAnalyst`
  - LLM-powered market analyst।
  - Methods: `__init__`, `groq_client`, `gemini_client`, `groq_model`, `gemini_model`, `active_provider`, `_init_clients`, `analyze` (+2)

### `automated_retraining.py`

**Description:** Automated Model Retraining System

**Classes (1):**

- `AutomatedRetrainingSystem`
  - Automated model retraining with performance monitoring
  - Methods: `__init__`, `start_scheduled_retraining`, `_run_scheduler`, `_retrain_models`, `_load_all_forex_data`, `_train_model_for_pair`, `_create_features`, `_calculate_rsi` (+2)

### `model_versioning.py`

**Description:** Model Versioning System

**Classes (1):**

- `ModelVersionManager`
  - Manages model versions, storage, and deployment
  - Methods: `__init__`, `save_model_version`, `load_model_version`, `list_model_versions`, `get_latest_model_version`, `compare_model_versions`, `delete_model_version`

</details>

## intelligence/

**Purpose:** Intelligence layer — news AI, sentiment, confluence, event classification

**Files:** 10

| File | Lines | Description | Key Classes | Key Functions |
|------|-------|-------------|-------------|---------------|
| `__init__.py` | 1 | — | — | — |
| `confidence_calibrator.py` | 184 | intelligence/confidence_calibrator.py — Confidence calibration | ConfidenceCalibrator | _bucket_for, _empty_state, get_calibrator |
| `confluence_engine.py` | 486 | intelligence/confluence_engine.py — Multi-Factor Confluence Engine | ConfluenceDecision, ConfluenceEngine, ConfluenceScore (+3) | get_confluence_engine |
| `currency_impact.py` | 177 | intelligence/currency_impact.py — Currency impact mapping engine | CurrencyImpact, CurrencyImpactEngine | get_currency_impact_engine |
| `decision_score.py` | 193 | intelligence/decision_score.py — Weighted factor scoring system | FactorScore, ConfluenceScore, DecisionScorer | get_scorer |
| `event_classifier.py` | 282 | intelligence/event_classifier.py — Special event detection & risk rule | EventClassification, EventClassifier | get_event_classifier |
| `news_ai.py` | 438 | intelligence/news_ai.py — NewsIntelligence main orchestrator | NewsBiasReport, NewsIntelligence | get_news_intelligence |
| `news_sources.py` | 357 | intelligence/news_sources.py — Multi-source news aggregator | NewsItem, NewsSources | — |
| `sentiment_model.py` | 318 | intelligence/sentiment_model.py — Financial news sentiment analyzer | SentimentResult, SentimentModel | get_sentiment_model |
| `signal_validator.py` | 247 | intelligence/signal_validator.py — Signal validation gates | ValidationResult, SignalValidator | get_signal_validator |

<details>
<summary>Detailed file descriptions (9 files)</summary>

### `confidence_calibrator.py`

**Description:** intelligence/confidence_calibrator.py — Confidence calibration

**Classes (1):**

- `ConfidenceCalibrator`
  - Tracks confidence-vs-actual-win-rate and adjusts future predictions.
  - Methods: `__init__`, `_load`, `_save`, `record_outcome`, `calibrate`, `status`

**Functions (3):**

- `_bucket_for()`
- `_empty_state()`
- `get_calibrator()`

### `confluence_engine.py`

**Description:** intelligence/confluence_engine.py — Multi-Factor Confluence Engine

**Classes (6):**

- `ConfluenceDecision`
  - The final output of the ConfluenceEngine.
  - Methods: `to_dict`, `to_telegram_alert`
- `ConfluenceEngine`
  - Collects all analyses → computes confluence → validates → decides.
  - Methods: `__init__`, `evaluate`, `_collect_factors`, `_smc_factor`, `_liquidity_factor`, `_session_factor`, `_currency_strength_factor`, `_intermarket_factor` (+2)
- `ConfluenceScore`
- `FactorScore`
- `DecisionScorer`
- `SignalValidator`

**Functions (1):**

- `get_confluence_engine()`

### `currency_impact.py`

**Description:** intelligence/currency_impact.py — Currency impact mapping engine

**Classes (2):**

- `CurrencyImpact`
  - Currency-level bias from a news event.
  - Methods: `to_dict`
- `CurrencyImpactEngine`
  - Maps sentiment results to per-pair directional biases.
  - Methods: `__init__`, `calculate`, `merge_impacts`

**Functions (1):**

- `get_currency_impact_engine()`

### `decision_score.py`

**Description:** intelligence/decision_score.py — Weighted factor scoring system

**Classes (3):**

- `FactorScore`
  - One analysis factor's contribution to the confluence score.
  - Methods: `aligned_direction`, `is_meaningful`, `to_dict`
- `ConfluenceScore`
  - The final aggregated confluence score.
  - Methods: `to_dict`
- `DecisionScorer`
  - Computes weighted confluence scores from individual factor inputs.
  - Methods: `__init__`, `score`

**Functions (1):**

- `get_scorer()`

### `event_classifier.py`

**Description:** intelligence/event_classifier.py — Special event detection & risk rules

**Classes (2):**

- `EventClassification`
  - Classification result for a single news event.
  - Methods: `to_dict`
- `EventClassifier`
  - Classifies news events and returns trading rules per category.
  - Methods: `classify`, `is_in_block_window`

**Functions (1):**

- `get_event_classifier()`

### `news_ai.py`

**Description:** intelligence/news_ai.py — NewsIntelligence main orchestrator

**Classes (2):**

- `NewsBiasReport`
  - Top-level output of NewsIntelligence.analyze().
  - Methods: `to_dict`
- `NewsIntelligence`
  - Main orchestrator — wires all 4 sub-modules together.
  - Methods: `__init__`, `set_pairs`, `analyze`, `_pairs_affected_by_currency`, `should_block_trade`, `adjust_confidence`, `format_telegram_alert`, `record_prediction` (+2)

**Functions (1):**

- `get_news_intelligence()`

### `news_sources.py`

**Description:** intelligence/news_sources.py — Multi-source news aggregator

**Classes (2):**

- `NewsItem`
  - Unified news/event item.
  - Methods: `to_dict`
- `NewsSources`
  - Multi-source news aggregator with TTL caching.
  - Methods: `__init__`, `fetch_economic_calendar`, `_fetch_local_calendar`, `fetch_central_bank_events`, `fetch_rss_feeds`, `fetch_all`, `fetch_all_flat`

### `sentiment_model.py`

**Description:** intelligence/sentiment_model.py — Financial news sentiment analyzer

**Classes (2):**

- `SentimentResult`
  - Structured sentiment analysis output.
  - Methods: `to_dict`
- `SentimentModel`
  - Financial news sentiment analyzer (LLM-powered with rule-based fallback).
  - Methods: `__init__`, `analyze`, `_analyze_with_llm`, `_analyze_with_rules`

**Functions (1):**

- `get_sentiment_model()`

### `signal_validator.py`

**Description:** intelligence/signal_validator.py — Signal validation gates

**Classes (2):**

- `ValidationResult`
  - Result of one validation gate.
  - Methods: `to_dict`
- `SignalValidator`
  - Runs all pre-trade validation gates on a ConfluenceScore.
  - Methods: `validate_all`, `_gate_confluence_quality`, `_gate_factor_count`, `_gate_contradiction`, `_gate_risk`, `_gate_news`, `_gate_correlation`

**Functions (1):**

- `get_signal_validator()`

</details>

## learning/

**Purpose:** Learning system — auto optimizer, mistake analyzer, lesson memory, performance feedback

**Files:** 12

| File | Lines | Description | Key Classes | Key Functions |
|------|-------|-------------|-------------|---------------|
| `__init__.py` | 1 | — | — | — |
| `auto_optimizer.py` | 599 | — | AutoOptimizer | — |
| `confidence_engine.py` | 857 | — | ConfidenceEngine | _test_mode |
| `deep_analyzer.py` | 889 | — | DeepMistakeAnalyzer | — |
| `lesson_memory.py` | 319 | — | LessonMemory | — |
| `memory_integration.py` | 227 | — | MemoryIntegration | — |
| `mistake_analyzer.py` | 207 | — | AdvancedMistakeAnalyzer | — |
| `optimizer_rules.py` | 153 | — | ValidationResult | is_statistically_significant, validate_change, volatility_to_risk (+1) |
| `performance_feedback.py` | 314 | — | PerformanceFeedback | — |
| `rule_updater.py` | 208 | — | RuleUpdater | — |
| `strategy_config.py` | 266 | — | StrategyConfig | — |
| `weekly_review.py` | 202 | — | WeeklyReportGenerator | is_review_day, run_weekly_review, _save_report (+1) |

<details>
<summary>Detailed file descriptions (11 files)</summary>

### `auto_optimizer.py`

**Classes (1):**

- `AutoOptimizer`
  - Day 55 Main Class — AI-এর Self-Improvement Controller।
  - Methods: `__init__`, `weekly_optimizer`, `_optimize_pairs`, `_optimize_patterns`, `_optimize_sessions`, `_optimize_risk`, `_dispatch_suggestions`, `_apply_suggestion` (+2)

### `confidence_engine.py`

**Classes (1):**

- `ConfidenceEngine`
  - Day 53 — Pattern + Context ভিত্তিক Dynamic Confidence Scoring।
  - Methods: `__init__`, `calculate`, `record_outcome`, `_get_historical_score`, `_get_recent_score`, `_get_regime_score`, `_bayesian_penalty`, `_check_skip` (+2)

**Functions (1):**

- `_test_mode()`
  - Lazy check for TEST_MODE flag. Returns False if config isn't

### `deep_analyzer.py`

**Classes (1):**

- `DeepMistakeAnalyzer`
  - Day 52 Main Class — AI-এর Self-Learning Intelligence Layer।
  - Methods: `__init__`, `analyze_loss`, `_collect_loss_context`, `_run_llm_analysis`, `_heuristic_analysis`, `_build_counterfactual`, `_validate_statistically`, `_calibrate_confidence` (+2)

### `lesson_memory.py`

**Classes (1):**

- `LessonMemory`
  - AI-এর experience store।
  - Methods: `__init__`, `add_lesson`, `recall`, `update_success_rate`, `get_pattern_stats`, `get_regime_stats`, `print_summary`, `_load` (+1)

### `memory_integration.py`

**Classes (1):**

- `MemoryIntegration`
  - Trading pipeline-এর সাথে learning system-এর bridge।
  - Methods: `__init__`, `get_pre_trade_context`, `record_trade_outcome`, `get_memory_for_master_analyst`

### `mistake_analyzer.py`

**Classes (1):**

- `AdvancedMistakeAnalyzer`
  - LLM এবং ভেক্টর মেমোরির সমন্বয়ে গঠিত ক্লোজড ট্রেড অ্যানালাইসিস লুপ।
  - Methods: `__init__`, `_has_vector_memory`, `_vector_search`, `_vector_add_lesson`, `analyze_closed_trade`, `_process_loss_trade`, `_process_win_trade`

### `optimizer_rules.py`

**Classes (1):**

- `ValidationResult`

**Functions (4):**

- `is_statistically_significant()`
  - One-sided z-test: observed win rate বেসলাইন (default 50%) থেকে
- `validate_change()`
  - Safety Layer-এর কেন্দ্রীয় gate। Day 55 spec-এর তিনটা শর্ত মিলিয়ে
- `volatility_to_risk()`
  - Day 55 formula:  risk = base_risk / volatility_factor
- `clamp_risk_step()`
  - একবারে risk যেন খুব বেশি লাফ না দেয় — gradual change।

### `performance_feedback.py`

**Classes (1):**

- `PerformanceFeedback`
  - Strategy-level performance analytics।
  - Methods: `__init__`, `record_trade`, `get_pattern_performance`, `get_regime_performance`, `get_timeframe_performance`, `get_master_context`, `print_full_report`, `_load` (+1)

### `rule_updater.py`

**Classes (1):**

- `RuleUpdater`
  - Pattern confidence rules manage করে।
  - Methods: `__init__`, `get_confidence`, `get_confidence_adjustment`, `get_all_rules`, `apply_rule`, `reset_rule`, `print_all_rules`, `_key` (+2)

### `strategy_config.py`

**Classes (1):**

- `StrategyConfig`
  - Live trading configuration manager + version control।
  - Methods: `__init__`, `get_active_pairs`, `remove_pair`, `add_pair`, `get_disabled_pairs`, `set_session_preference`, `get_session_preference`, `get_risk` (+2)

### `weekly_review.py`

**Classes (1):**

- `WeeklyReportGenerator`
  - AutoOptimizer-এর run output থেকে spec-format-এ একটা readable report বানায়।
  - Methods: `__init__`, `build`, `_best_strategy`, `_disabled_summary`, `_risk_change_summary`, `_explain_all_actions`

**Functions (4):**

- `is_review_day()`
  - আজ কি saptahik review চালানোর দিন (রবিবার)?
- `run_weekly_review()`
  - Day 55 main entry point।
- `_save_report()`
- `_print_report()`

</details>

## memory/

**Purpose:** Memory system — trade memory, pattern memory, knowledge store, history

**Files:** 10

| File | Lines | Description | Key Classes | Key Functions |
|------|-------|-------------|-------------|---------------|
| `__init__.py` | 1 | — | — | — |
| `confidence_calibrator.py` | 258 | — | ConfidenceCalibrator | — |
| `database.py` | 466 | — | Database | — |
| `history.py` | 124 | — | AnalysisHistory | — |
| `knowledge_store.py` | 354 | AI-এর Long-Term Memory। | KnowledgeStore | _make_embedding_function |
| `learning.py` | 246 | AI তার নিজের trade history দেখে শিখবে। | LearningEngine | — |
| `pattern_memory.py` | 171 | Vector search similarity খুঁজবে, | PatternMemory | — |
| `sentence_model_cache.py` | 69 | memory/sentence_model_cache.py — Shared SentenceTransformer Cache (Day | — | get_sentence_model, reset_cache |
| `trade_context.py` | 1 | — | — | — |
| `trade_memory.py` | 396 | — | TradeMemory | — |

<details>
<summary>Detailed file descriptions (8 files)</summary>

### `confidence_calibrator.py`

**Classes (1):**

- `ConfidenceCalibrator`
  - AI-এর confidence score কতটা accurate সেটা track করে।
  - Methods: `__init__`, `record_prediction`, `record_outcome`, `get_calibration_report`, `get_adjustment_factor`, `get_ai_context`, `print_report`, `_get_bucket` (+2)

### `database.py`

**Classes (1):**

- `Database`
  - AI Trader-এর central memory system।
  - Methods: `__init__`, `create_tables`, `save_trade`, `update_trade_result`, `get_recent_trades`, `get_trade_by_id`, `save_analysis`, `get_similar_setups` (+2)

### `history.py`

**Classes (1):**

- `AnalysisHistory`
  - প্রতিটা analysis run save করো।
  - Methods: `save`, `update_result`, `get_recent`, `get_stats`, `print_recent`, `_load`, `_save`

### `knowledge_store.py`

**Description:** AI-এর Long-Term Memory।

**Classes (1):**

- `KnowledgeStore`
  - AI Trader-এর vector memory।
  - Methods: `__init__`, `_is_ready`, `add_memory`, `add_trade_memory`, `add_rule`, `add_lesson`, `search_memory`, `search_similar_trades` (+2)

**Functions (1):**

- `_make_embedding_function()`
  - Embedding function তৈরি করো।

### `learning.py`

**Description:** AI তার নিজের trade history দেখে শিখবে।

**Classes (1):**

- `LearningEngine`
  - AI-এর self-improvement engine।
  - Methods: `__init__`, `pattern_win_rate`, `regime_win_rate`, `confidence_vs_result`, `common_mistakes`, `get_improvement_plan`, `print_report`, `close`

### `pattern_memory.py`

**Description:** Vector search similarity খুঁজবে,

**Classes (1):**

- `PatternMemory`
  - Structured pattern memory।
  - Methods: `__init__`, `_load`, `_save`, `add_winning_pattern`, `add_losing_pattern`, `add_lesson`, `find_similar_winning`, `find_similar_losing` (+2)

### `sentence_model_cache.py`

**Description:** memory/sentence_model_cache.py — Shared SentenceTransformer Cache (Day 81+)

**Functions (2):**

- `get_sentence_model()`
  - Return a shared SentenceTransformer instance, or None if unavailable.
- `reset_cache()`
  - Force-reload the model on next call. Used in tests only.

### `trade_memory.py`

**Classes (1):**

- `TradeMemory`
  - SQL Database + Closed trade lessons-এর local vector memory (Day 16 & Day 33 Combined).
  - Methods: `__init__`, `_has_model`, `on_signal_generated`, `on_trade_closed`, `_generate_lesson`, `add_vector_lesson`, `find_similar`, `get_memory_context_text` (+2)

</details>

## orchestrator/

**Purpose:** Orchestration — trading orchestrator, scheduler, safety controller, self-healing

**Files:** 12

| File | Lines | Description | Key Classes | Key Functions |
|------|-------|-------------|-------------|---------------|
| `__init__.py` | 1 | — | — | — |
| `audit_trail.py` | 156 | — | AuditTrail | — |
| `communication_bus.py` | 259 | — | AgentMessage, AgentMessageBus | — |
| `daily_routine.py` | 348 | — | DailyRoutineManager | — |
| `decision_journal.py` | 75 | orchestrator/decision_journal.py — Minimal stub (Day 60 placeholder) | DecisionJournal | — |
| `human_override.py` | 247 | — | HumanOverrideSystem | — |
| `mode_manager.py` | 65 | orchestrator/mode_manager.py — Minimal stub (Day 60 placeholder) | ModeManager | — |
| `safety_controller.py` | 68 | orchestrator/safety_controller.py — Minimal stub (Day 60 placeholder) | SafetyController | — |
| `scheduler.py` | 189 | — | ScheduledTask, TaskScheduler | — |
| `self_healing.py` | 60 | orchestrator/self_healing.py — Minimal stub (Day 60 placeholder) | SelfHealingSystem | — |
| `system_state.py` | 264 | — | SystemState, SystemStateManager | — |
| `trading_orchestrator.py` | 855 | — | TradingOrchestrator | — |

<details>
<summary>Detailed file descriptions (11 files)</summary>

### `audit_trail.py`

**Classes (1):**

- `AuditTrail`
  - Complete audit trail for all system events.
  - Methods: `__init__`, `log_event`, `log_message`, `get_events`, `get_trade_history`, `get_rejection_history`, `get_safety_history`, `get_stats` (+2)

### `communication_bus.py`

**Classes (2):**

- `AgentMessage`
  - Single message on the bus.
  - Methods: `__init__`, `to_dict`, `from_dict`, `__repr__`
- `AgentMessageBus`
  - Central communication bus for all agent interactions.
  - Methods: `__init__`, `publish`, `subscribe`, `subscribe_all`, `unsubscribe`, `get_history`, `get_last_message`, `get_cycle_messages` (+2)

### `daily_routine.py`

**Classes (1):**

- `DailyRoutineManager`
  - Manages autonomous daily, weekly, and periodic routines.
  - Methods: `__init__`, `setup`, `execute_scheduled_tasks`, `_morning_routine`, `_evening_routine`, `_sunday_routine`, `_position_monitor`, `_state_backup` (+2)

### `decision_journal.py`

**Description:** orchestrator/decision_journal.py — Minimal stub (Day 60 placeholder)

**Classes (1):**

- `DecisionJournal`
  - Append-only JSON journal of every decision the orchestrator makes.
  - Methods: `__init__`, `_load`, `record`, `recent`, `status`

### `human_override.py`

**Classes (1):**

- `HumanOverrideSystem`
  - Emergency human control system for the AI Trader.
  - Methods: `__init__`, `start`, `check_command_file`, `stop_all`, `close_all`, `pause`, `resume`, `get_status` (+2)

### `mode_manager.py`

**Description:** orchestrator/mode_manager.py — Minimal stub (Day 60 placeholder)

**Classes (1):**

- `ModeManager`
  - Thin shim around `core.approval_mode.ApprovalMode`.
  - Methods: `__init__`, `mode_name`, `set_analysis_only`, `set_autonomous`, `is_trading_allowed`, `status`

### `safety_controller.py`

**Description:** orchestrator/safety_controller.py — Minimal stub (Day 60 placeholder)

**Classes (1):**

- `SafetyController`
  - Centralized safety gate. Currently a thin wrapper around the existing
  - Methods: `__init__`, `check_pre_trade`, `trigger_emergency_stop`, `clear_emergency_stop`, `status`

### `scheduler.py`

**Classes (2):**

- `ScheduledTask`
  - A task scheduled to run at specific intervals.
  - Methods: `__init__`, `should_run`, `execute`
- `TaskScheduler`
  - Central task scheduler for autonomous routines.
  - Methods: `__init__`, `schedule`, `tick`, `start`, `stop_all`, `get_tasks`, `get_stats`

### `self_healing.py`

**Description:** orchestrator/self_healing.py — Minimal stub (Day 60 placeholder)

**Classes (1):**

- `SelfHealingSystem`
  - Detects recurring runtime errors and applies automatic remediation.
  - Methods: `__init__`, `record_issue`, `_try_remediate`, `get_recent_issues`, `status`

### `system_state.py`

**Classes (2):**

- `SystemState`
  - Immutable snapshot of the entire system state.
  - Methods: `__init__`, `to_dict`, `from_dict`, `is_trading_allowed`, `is_analysis_allowed`, `get_summary_line`
- `SystemStateManager`
  - Manages the global system state. Thread-safe state transitions.
  - Methods: `__init__`, `state`, `get_state`, `update`, `update_market_status`, `on_state_change`, `_notify_listeners`, `_load_state` (+2)

### `trading_orchestrator.py`

**Classes (1):**

- `TradingOrchestrator`
  - Central Nervous System of the AI Trading Operating System.
  - Methods: `__init__`, `start_system`, `run_cycle`, `shutdown`, `run`, `_init_mode`, `_init_communication_bus`, `_init_agents` (+2)

</details>

## scanner/

**Purpose:** Market scanner — multi-pair scanning, correlation filter, opportunity ranking

**Files:** 6

| File | Lines | Description | Key Classes | Key Functions |
|------|-------|-------------|-------------|---------------|
| `__init__.py` | 1 | — | — | — |
| `config.py` | 84 | — | — | — |
| `correlation_filter.py` | 85 | — | CorrelationFilter | — |
| `market_scanner.py` | 379 | — | MarketScanner | — |
| `opportunity_ranker.py` | 140 | — | OpportunityRanker | — |
| `scanner.py` | 57 | — | — | — |

<details>
<summary>Detailed file descriptions (3 files)</summary>

### `correlation_filter.py`

**Classes (1):**

- `CorrelationFilter`
  - Usage:
  - Methods: `__init__`, `sync_open`, `register_open`, `register_close`, `allow`, `_find_group`, `print_status`

### `market_scanner.py`

**Classes (1):**

- `MarketScanner`
  - Usage:
  - Methods: `__init__`, `scan`, `get_top_opportunities`, `_scan_pair`, `_rule_signal`, `_mtf_alignment`, `_session_pairs`, `_current_session` (+2)

### `opportunity_ranker.py`

**Classes (1):**

- `OpportunityRanker`
  - Usage:
  - Methods: `rank`, `top_n`, `_compute_score`, `_technical_strength`, `_mtf_alignment`, `_rr_score`, `_news_safety`, `_liquidity_score` (+2)

</details>

## strategies/

**Purpose:** Strategy implementations — breakout, EMA+RSI, reversal, scalping, trend following

**Files:** 6

| File | Lines | Description | Key Classes | Key Functions |
|------|-------|-------------|-------------|---------------|
| `__init__.py` | 6 | — | — | — |
| `breakout.py` | 68 | — | BreakoutStrategy | — |
| `ema_rsi_combo.py` | 197 | strategies/ema_rsi_combo.py — EMA-200 + RSI-50 Combo Strategy (Day 81+ | EmaRsiComboStrategy | — |
| `reversal.py` | 78 | — | ReversalStrategy | — |
| `scalping_strategy.py` | 368 | strategies/scalping_strategy.py — Scalping Intelligence Engine (Day 82 | ScalpingStrategy, _Indicators | — |
| `trend_follow.py` | 76 | — | TrendFollowStrategy | — |

<details>
<summary>Detailed file descriptions (5 files)</summary>

### `breakout.py`

**Classes (1):**

- `BreakoutStrategy`
  - Methods: `__init__`, `generate`, `_signal`, `_stop_pips`, `_pattern`, `_hold`

### `ema_rsi_combo.py`

**Description:** strategies/ema_rsi_combo.py — EMA-200 + RSI-50 Combo Strategy (Day 81+)

**Classes (1):**

- `EmaRsiComboStrategy`
  - EMA-200 trend filter + RSI-50 momentum trigger.
  - Methods: `analyze`

### `reversal.py`

**Classes (1):**

- `ReversalStrategy`
  - Methods: `__init__`, `generate`, `_signal`, `_stop_pips`, `_pattern`, `_hold`

### `scalping_strategy.py`

**Description:** strategies/scalping_strategy.py — Scalping Intelligence Engine (Day 82+)

**Classes (2):**

- `ScalpingStrategy`
  - M1/M5/M15 scalping brain. Returns a UnifiedSignal so it integrates
  - Methods: `analyze`, `_compute_indicators`, `_detect_liquidity_sweep`
- `_Indicators`

### `trend_follow.py`

**Classes (1):**

- `TrendFollowStrategy`
  - Methods: `__init__`, `generate`, `_signal`, `_stop_pips`, `_pattern`, `_hold`

</details>

## strategy/

**Purpose:** Strategy infrastructure — selector, signal engine

**Files:** 3

| File | Lines | Description | Key Classes | Key Functions |
|------|-------|-------------|-------------|---------------|
| `__init__.py` | 1 | — | — | — |
| `selector.py` | 521 | — | StrategySelector | — |
| `signal_engine.py` | 372 | — | SignalEngine | _apply_fib_scoring |

<details>
<summary>Detailed file descriptions (2 files)</summary>

### `selector.py`

**Classes (1):**

- `StrategySelector`
  - Market Regime result থেকে ঠিক কোন strategy activate হবে সেই router।
  - Methods: `__init__`, `select`, `_pick_strategy`, `_detect_conflict`, `_position_multiplier`, `_confidence_level`, `_build_reason`, `_wait` (+1)

### `signal_engine.py`

**Classes (1):**

- `SignalEngine`
  - Mixin — existing SignalEngine-এ যোগ করো।
  - Methods: `_apply_fib_scoring`, `generate`, `_signal_recommendation`, `get_ai_context`, `print_summary`

**Functions (1):**

- `_apply_fib_scoring()`
  - Module-level backward-compat wrapper — delegates to the static method

</details>

## execution/

**Purpose:** Execution layer — execution router, paper trader, simulated executor

**Files:** 3

| File | Lines | Description | Key Classes | Key Functions |
|------|-------|-------------|-------------|---------------|
| `execution_router.py` | 563 | — | ExecutionRouter | _log_event, _check_absolute_safety |
| `paper_trader.py` | 410 | — | PaperTrader | — |
| `simulated_executor.py` | 121 | execution/simulated_executor.py — Dry-run order executor. | SimulatedExecutor | — |

<details>
<summary>Detailed file descriptions (3 files)</summary>

### `execution_router.py`

**Classes (1):**

- `ExecutionRouter`
  - Methods: `__init__`, `_ensure_mt5_connected`, `execute`, `_execute_mt5_demo`, `shutdown`

**Functions (2):**

- `_log_event()`
- `_check_absolute_safety()`

### `paper_trader.py`

**Classes (1):**

- `PaperTrader`
  - Virtual trading account. AITrader.get_signal()-এর output (a `result` dict)
  - Methods: `__init__`, `open_trade_from_signal`, `_build_trade_record`, `update_price`, `close_trade`, `_calculate_pnl`, `get_dashboard`, `print_dashboard` (+2)

### `simulated_executor.py`

**Description:** execution/simulated_executor.py — Dry-run order executor.

**Classes (1):**

- `SimulatedExecutor`
  - Drop-in replacement for OrderManager.place_market_order().
  - Methods: `__init__`, `place_market_order`, `get_open_positions`, `close_order`

</details>

## fundamental/

**Purpose:** Fundamental analysis — economic calendar, news filter, FRED data

**Files:** 8

| File | Lines | Description | Key Classes | Key Functions |
|------|-------|-------------|-------------|---------------|
| `__init__.py` | 1 | — | — | — |
| `economic_calendar_api.py` | 441 | fundamental/economic_calendar_api.py — Day 94 Institutional Economic C | EconomicCalendarAPI | — |
| `economic_surprise.py` | 209 | fundamental/economic_surprise.py — Day 96 Economic Surprise Index | EconomicSurpriseEngine | — |
| `faireconomy_cache.py` | 200 | — | — | fetch_faireconomy |
| `fred_data.py` | 275 | fundamental/fred_data.py — Day 94 FRED API (Federal Reserve Economic D | FREDApi | get_fred_api |
| `fundamental_sentiment.py` | 207 | — | FundamentalSentimentScore | — |
| `news_filter.py` | 472 | — | NewsFilter | — |
| `trading_economics_calendar.py` | 367 | fundamental/trading_economics_calendar.py — Day 95 Economic Calendar ( | TradingEconomicsCalendar | — |

<details>
<summary>Detailed file descriptions (6 files)</summary>

### `economic_calendar_api.py`

**Description:** fundamental/economic_calendar_api.py — Day 94 Institutional Economic Calendar

**Classes (1):**

- `EconomicCalendarAPI`
  - Multi-source economic calendar with automatic fallback.
  - Methods: `__init__`, `get_calendar`, `_fetch_faireconomy`, `_fetch_tradermade`, `_fetch_finnhub`, `_normalize_ff_events`, `_check_block`, `_format_event` (+2)

### `economic_surprise.py`

**Description:** fundamental/economic_surprise.py — Day 96 Economic Surprise Index

**Classes (1):**

- `EconomicSurpriseEngine`
  - Economic surprise index — actual vs forecast comparison.
  - Methods: `__init__`, `analyze`, `_get_events_with_actuals`, `_parse_numeric`, `_fallback_result`, `get_ai_context`, `print_summary`

### `fred_data.py`

**Description:** fundamental/fred_data.py — Day 94 FRED API (Federal Reserve Economic Data)

**Classes (1):**

- `FREDApi`
  - FRED API client for macro-economic data.
  - Methods: `__init__`, `available`, `get_macro_snapshot`, `get_series`, `_analyze_yield_curve`, `_analyze_inflation`, `_analyze_rate_environment`, `_empty_result` (+2)

**Functions (1):**

- `get_fred_api()`

### `fundamental_sentiment.py`

**Classes (1):**

- `FundamentalSentimentScore`
  - Usage:
  - Methods: `__init__`, `score_currency`, `_upcoming_risk_for_currency`, `score_pair`, `get_ai_context`, `print_summary`

### `news_filter.py`

**Classes (1):**

- `NewsFilter`
  - Methods: `__init__`, `check`, `estimate_volatility`, `affected_pairs`, `post_news_status`, `_event_time_from_label`, `_max_risk_level`, `get_weekly_calendar` (+2)

### `trading_economics_calendar.py`

**Description:** fundamental/trading_economics_calendar.py — Day 95 Economic Calendar (Tradermade alternative)

**Classes (1):**

- `TradingEconomicsCalendar`
  - Multi-source economic calendar — Trading Economics + RSS fallbacks.
  - Methods: `__init__`, `get_calendar`, `_fetch_trading_economics`, `_fetch_investing_rss`, `_fetch_dailyfx_rss`, `_check_block`, `_format_event`, `_empty_result` (+2)

</details>

## alerts/

**Purpose:** Alerts — Telegram bot, MT5 alert engine

**Files:** 4

| File | Lines | Description | Key Classes | Key Functions |
|------|-------|-------------|-------------|---------------|
| `__init__.py` | 1 | — | — | — |
| `mt5_alert_engine.py` | 415 | — | AlertField, AlertOperator, AlertAction (+3) | create_price_alert |
| `telegram_bot.py` | 731 | — | _RateLimiter, TelegramNotifier | _get_rate_limiter, register_pause_callback, _set_trading_paused (+12) |
| `telegram_ext.py` | 352 | alerts/telegram_ext.py — Day 93 Telegram command extensions | — | _escape_md, _fmt_pnl, _fmt_position (+9) |

<details>
<summary>Detailed file descriptions (3 files)</summary>

### `mt5_alert_engine.py`

**Classes (6):**

- `AlertField`
- `AlertOperator`
- `AlertAction`
- `AlertCondition`
  - Single MT5-style alert condition.
  - Methods: `to_dict`
- `AlertResult`
  - Result of an alert check.
  - Methods: `to_dict`
- `MT5AlertEngine`
  - MT5-style alert condition engine.
  - Methods: `__init__`, `add_alert`, `remove_alert`, `list_alerts`, `set_action_handler`, `check_all`, `_check_condition`, `_execute_action` (+2)

**Functions (1):**

- `create_price_alert()`
  - Quick helper to create a price-based alert.

### `telegram_bot.py`

**Classes (2):**

- `_RateLimiter`
  - Sliding-window per-channel rate limiter.
  - Methods: `__init__`, `allow`
- `TelegramNotifier`
  - Handles every outgoing notification for the trading bot.
  - Methods: `__init__`, `send_message`, `notify_trade_open`, `notify_trade_close`, `notify_daily_loss_limit`, `notify_drawdown_alert`, `notify_daily_report`, `notify_news_warning` (+2)

**Functions (15):**

- `_get_rate_limiter()`
- `register_pause_callback()`
  - Register a callback that fires the moment IS_TRADING_PAUSED changes.
- `_set_trading_paused()`
  - Internal async helper — updates flag AND invokes callback.
- `_escape_markdown()`
  - Strip characters that break Telegram's legacy Markdown (V1) entity
- `_chunk_message()`
  - Split a long message into chunks of at most `limit` characters,
- `get_notifier()`
  - Return (or lazily create) the shared TelegramNotifier instance.
- `_reply()`
  - Reply to a Telegram update with Markdown, falling back to plain
- `cmd_start()`
  - Welcome message with available commands.
- ... and 7 more

### `telegram_ext.py`

**Description:** alerts/telegram_ext.py — Day 93 Telegram command extensions

**Functions (12):**

- `_escape_md()`
  - Escape Markdown special chars for Telegram.
- `_fmt_pnl()`
  - Format PnL with color icon.
- `_fmt_position()`
  - Format one position as a Telegram line.
- `cmd_positions()`
  - /positions — List all open MT5 positions.
- `cmd_close()`
  - /close <ticket> — Close an open position by ticket.
- `cmd_symbols()`
  - /symbols — List configured trading pairs + spread.
- `cmd_indicators()`
  - /indicators [symbol] — Show latest indicator snapshot.
- `cmd_source()`
  - /source — Show which data sources are active.
- ... and 4 more

</details>

## analytics/

**Purpose:** Analytics — performance report, ranking engine, strategy tracker

**Files:** 5

| File | Lines | Description | Key Classes | Key Functions |
|------|-------|-------------|-------------|---------------|
| `__init__.py` | 17 | — | — | — |
| `analytics.py` | 171 | — | PerformanceAnalyzer | — |
| `performance_report.py` | 537 | — | StrategyVersionControl, MonteCarloSimulator, OptimizationSuggester (+1) | — |
| `ranking_engine.py` | 407 | — | SetupScore, RankingEngine | — |
| `strategy_tracker.py` | 717 | — | StrategyTracker | detect_session |

<details>
<summary>Detailed file descriptions (4 files)</summary>

### `analytics.py`

**Classes (1):**

- `PerformanceAnalyzer`
  - Methods: `summarize`, `rank_strategies`, `monte_carlo`, `parameter_grid`, `_sharpe`, `_max_drawdown`, `_win_rate_by`

### `performance_report.py`

**Classes (4):**

- `StrategyVersionControl`
  - প্রতিটি strategy version-এর performance আলাদাভাবে track করো।
  - Methods: `__init__`, `compare_versions`, `print_version_comparison`
- `MonteCarloSimulator`
  - পরপর N loss হলে account survive করবে?
  - Methods: `simulate`
- `OptimizationSuggester`
  - AI নিজে নিজে rule পরিবর্তন সাজেস্ট করে।
  - Methods: `generate`
- `PerformanceReport`
  - Day 54 — সব analytics একত্র করে weekly report তৈরি করে।
  - Methods: `__init__`, `generate`, `save`, `to_json`, `_print`

### `ranking_engine.py`

**Classes (2):**

- `SetupScore`
  - একটা setup-এর সম্পূর্ণ score breakdown।
  - Methods: `__init__`, `_calculate`, `_recommend`, `to_dict`
- `RankingEngine`
  - Day 54 — সব setup কে score দিয়ে rank করে।
  - Methods: `__init__`, `score_setup`, `_fetch_from_tracker`, `rank_all_setups`, `get_confidence_adjustment`, `print_rankings`

### `strategy_tracker.py`

**Classes (1):**

- `StrategyTracker`
  - Day 54 Performance Intelligence Layer।
  - Methods: `__init__`, `_init_db`, `_conn`, `record_trade`, `update_outcome`, `_get_trade`, `pair_performance`, `session_performance` (+2)

**Functions (1):**

- `detect_session()`
  - UTC hour থেকে trading session বের করো।

</details>

## automation/

**Purpose:** Automation — daily review, error handler, system health

**Files:** 5

| File | Lines | Description | Key Classes | Key Functions |
|------|-------|-------------|-------------|---------------|
| `__init__.py` | 1 | — | — | — |
| `daily_review.py` | 298 | — | DailyReview | — |
| `error_handler.py` | 177 | — | ErrorHandler | — |
| `runtime_metrics.py` | 210 | — | RuntimeMetrics | — |
| `system_health.py` | 176 | — | SystemHealth | — |

<details>
<summary>Detailed file descriptions (4 files)</summary>

### `daily_review.py`

**Classes (1):**

- `DailyReview`
  - Usage:
  - Methods: `run`, `_build_context`, `_call_llm`, `_parse_response`, `_fallback_review`, `save`, `print_summary`

### `error_handler.py`

**Classes (1):**

- `ErrorHandler`
  - Usage:
  - Methods: `__init__`, `log_error`, `log_crash`, `with_retry`, `get_error_summary`, `get_recent_errors`, `print_summary`

### `runtime_metrics.py`

**Classes (1):**

- `RuntimeMetrics`
  - Usage:
  - Methods: `__init__`, `start_session`, `session_duration_sec`, `timer`, `get_stage_stats`, `get_all_stage_stats`, `average_decision_time_sec`, `record_cycle` (+2)

### `system_health.py`

**Classes (1):**

- `SystemHealth`
  - Usage:
  - Methods: `__init__`, `check_all`, `_check_broker`, `_check_database`, `_check_vision`, `_check_internet`, `_check_memory`, `print_status` (+1)

</details>

## hybrid/

**Purpose:** Hybrid system — confidence calibrator, decision validator, flow controller

**Files:** 5

| File | Lines | Description | Key Classes | Key Functions |
|------|-------|-------------|-------------|---------------|
| `__init__.py` | 1 | — | — | — |
| `confidence_calibrator.py` | 223 | — | ConfidenceCalibrator | — |
| `decision_validator.py` | 194 | — | DecisionValidator | — |
| `execution_router.py` | 466 | — | ApprovalMode, EmergencyStopError, ExecutionRouter | — |
| `flow_controller.py` | 379 | — | FlowController | — |

<details>
<summary>Detailed file descriptions (4 files)</summary>

### `confidence_calibrator.py`

**Classes (1):**

- `ConfidenceCalibrator`
  - Usage:
  - Methods: `__init__`, `build_calibration_report`, `calibrate`, `_find_bucket`, `get_calibration_health`, `_load_closed_trades`, `print_report`

### `decision_validator.py`

**Classes (1):**

- `DecisionValidator`
  - Usage:
  - Methods: `validate`, `_is_hard_conflict`, `_normalize`, `print_summary`

### `execution_router.py`

**Classes (3):**

- `ApprovalMode`
  - তোমার .env-এর APPROVAL_MODE-এর সাথে হুবহু সামঞ্জস্যপূর্ণ।
- `EmergencyStopError`
  - Emergency stop ট্রিগার হলে raise হয় — caller (FlowController) এটা catch করে
- `ExecutionRouter`
  - Usage:
  - Methods: `__init__`, `route`, `_approval_gate`, `get_pending_approvals`, `clear_pending_approval`, `_record_shadow_trade`, `resolve_shadow_trade`, `get_shadow_trades` (+2)

### `flow_controller.py`

**Classes (1):**

- `FlowController`
  - Usage:
  - Methods: `__init__`, `run_cycle`, `run_loop`, `_build_health_ctx`, `_log_to_learning`, `_finish`, `_save_cycle_log`, `print_cycle_summary`

</details>

## dashboard/

**Purpose:** Web dashboard — Streamlit app, charts, metrics, live room

**Files:** 13

| File | Lines | Description | Key Classes | Key Functions |
|------|-------|-------------|-------------|---------------|
| `__init__.py` | 1 | — | — | — |
| `app.py` | 542 | — | — | — |
| `components/__init__.py` | 1 | — | — | — |
| `components/alerts.py` | 150 | — | — | alert_banner, system_health_panel, emergency_control_panel (+2) |
| `components/charts.py` | 115 | — | — | equity_curve_chart, drawdown_chart, pattern_performance_chart (+5) |
| `components/data_loader.py` | 389 | — | — | load_json, save_json, get_system_control (+12) |
| `components/metrics.py` | 91 | — | — | kpi_row, status_badge, risk_meter (+4) |
| `pages/__init__.py` | 1 | — | — | — |
| `pages/ai_brain.py` | 73 | — | — | — |
| `pages/learning_center.py` | 67 | — | — | — |
| `pages/live_room.py` | 64 | — | — | — |
| `pages/risk_monitor.py` | 52 | — | — | — |
| `pages/strategy_lab.py` | 110 | — | — | — |

## monitoring/

**Purpose:** Monitoring — execution quality, signal debugger, pipeline tracer

**Files:** 4

| File | Lines | Description | Key Classes | Key Functions |
|------|-------|-------------|-------------|---------------|
| `__init__.py` | 1 | — | — | — |
| `execution_quality.py` | 239 | monitoring/execution_quality.py — Day 96 Execution Quality Monitor | ExecutionQualityMonitor | get_execution_quality_monitor |
| `signal_debugger.py` | 382 | monitoring/signal_debugger.py — Signal Pipeline Debugger (Day 81+) | LayerVerdict, CycleDebug, SignalDebugger | get_signal_debugger |
| `trade_pipeline_tracer.py` | 350 | monitoring/trade_pipeline_tracer.py — Detailed Trade Condition Tracer  | — | _c, _stage_icon, trace_symbol (+2) |

<details>
<summary>Detailed file descriptions (3 files)</summary>

### `execution_quality.py`

**Description:** monitoring/execution_quality.py — Day 96 Execution Quality Monitor

**Classes (1):**

- `ExecutionQualityMonitor`
  - Tracks execution quality metrics + recommends adjustments.
  - Methods: `__init__`, `record_trade`, `get_quality_report`, `_save_history`, `_load_history`, `get_ai_context`, `print_summary`

**Functions (1):**

- `get_execution_quality_monitor()`

### `signal_debugger.py`

**Description:** monitoring/signal_debugger.py — Signal Pipeline Debugger (Day 81+)

**Classes (3):**

- `LayerVerdict`
  - One pipeline layer's verdict for one cycle.
  - Methods: `icon`, `display_detail`, `to_dict`
- `CycleDebug`
  - All layer verdicts + final outcome for one cycle.
  - Methods: `record`, `record_final`, `summary_block`, `to_dict`
- `SignalDebugger`
  - Accumulates layer verdicts across the current cycle and across
  - Methods: `__init__`, `start_cycle`, `record`, `record_final`, `log_cycle_summary`, `save_to_file`, `_ensure_backfilled`, `block_stats` (+2)

**Functions (1):**

- `get_signal_debugger()`

### `trade_pipeline_tracer.py`

**Description:** monitoring/trade_pipeline_tracer.py — Detailed Trade Condition Tracer (Day 81+)

**Functions (5):**

- `_c()`
- `_stage_icon()`
- `trace_symbol()`
  - Run one real cycle and return a detailed stage-by-stage trace.
- `print_trace()`
  - Print a human-readable trace.
- `main()`

</details>

## server/

**Purpose:** Server — webhook server, signal pipeline

**Files:** 3

| File | Lines | Description | Key Classes | Key Functions |
|------|-------|-------------|-------------|---------------|
| `__init__.py` | 1 | — | — | — |
| `signal_pipeline.py` | 223 | — | SignalPipeline | — |
| `webhook_server.py` | 89 | — | — | tradingview_webhook, health_check |

<details>
<summary>Detailed file descriptions (1 files)</summary>

### `signal_pipeline.py`

**Classes (1):**

- `SignalPipeline`
  - Singleton — Flask-এর প্রতি request-এ নতুন AIAnalyst/DecisionAgent
  - Methods: `get_instance`, `__init__`, `_get_risk_engine`, `process`, `on_trade_closed`, `_build_indicator_context`, `_placeholder_rule_signal`

</details>

## system/

**Purpose:** System — network monitor, watchdog

**Files:** 3

| File | Lines | Description | Key Classes | Key Functions |
|------|-------|-------------|-------------|---------------|
| `__init__.py` | 1 | — | — | — |
| `network_monitor.py` | 313 | system/network_monitor.py — Day 97 Network & Latency Monitor | NetworkMonitor | get_network_monitor |
| `watchdog.py` | 261 | system/watchdog.py — Day 96 System Watchdog & Heartbeat | SystemWatchdog | get_watchdog |

<details>
<summary>Detailed file descriptions (2 files)</summary>

### `network_monitor.py`

**Description:** system/network_monitor.py — Day 97 Network & Latency Monitor

**Classes (1):**

- `NetworkMonitor`
  - Background network latency tracker.
  - Methods: `__init__`, `start`, `stop`, `_run_loop`, `check_now`, `_ping_host`, `_ping_mt5`, `_get_execution_latency` (+2)

**Functions (1):**

- `get_network_monitor()`

### `watchdog.py`

**Description:** system/watchdog.py — Day 96 System Watchdog & Heartbeat

**Classes (1):**

- `SystemWatchdog`
  - Background health monitor + auto-recovery.
  - Methods: `__init__`, `start`, `stop`, `_run_loop`, `check_now`, `_check_mt5`, `_check_database`, `_check_signal_freshness` (+2)

**Functions (1):**

- `get_watchdog()`

</details>

## database/

**Purpose:** Database — SQLite trade storage

**Files:** 2

| File | Lines | Description | Key Classes | Key Functions |
|------|-------|-------------|-------------|---------------|
| `__init__.py` | 1 | — | — | — |
| `db.py` | 508 | — | _NumpySafeEncoder, TraderDB | _safe_json_dumps, _safe |

<details>
<summary>Detailed file descriptions (1 files)</summary>

### `db.py`

**Classes (2):**

- `_NumpySafeEncoder`
  - Methods: `default`
- `TraderDB`
  - AI Trader-এর central database।
  - Methods: `__init__`, `_connect`, `_init_tables`, `save_candles`, `save_indicators`, `save_patterns`, `save_analysis`, `save_trade_open` (+2)

**Functions (2):**

- `_safe_json_dumps()`
  - json.dumps that never crashes — converts numpy types + falls back to str.
- `_safe()`
  - NaN → None (SQLite-এর জন্য)

</details>

## utils/

**Purpose:** Utilities — logger, safe pickle, session

**Files:** 4

| File | Lines | Description | Key Classes | Key Functions |
|------|-------|-------------|-------------|---------------|
| `__init__.py` | 1 | — | — | — |
| `logger.py` | 94 | — | — | _utf8_console_stream, get_logger |
| `safe_pickle.py` | 205 | utils/safe_pickle.py — Safe pickle loading with integrity verification | RestrictedUnpickler | _compute_file_hash, safe_pickle_dump, safe_pickle_load |
| `session.py` | 94 | — | SessionAnalyzer | — |

<details>
<summary>Detailed file descriptions (2 files)</summary>

### `safe_pickle.py`

**Description:** utils/safe_pickle.py — Safe pickle loading with integrity verification

**Classes (1):**

- `RestrictedUnpickler`
  - Unpickler that only allows whitelisted classes.
  - Methods: `find_class`

**Functions (3):**

- `_compute_file_hash()`
  - Compute SHA-256 hash of a file.
- `safe_pickle_dump()`
  - Save object to pickle file with integrity hash.
- `safe_pickle_load()`
  - Load object from pickle file with integrity verification.

### `session.py`

**Classes (1):**

- `SessionAnalyzer`
  - Methods: `get_current_session`, `_get_overlap`, `_volatility`, `_trade_quality`, `print_session_info`

</details>

## visualization/

**Purpose:** Visualization — chart rendering

**Files:** 2

| File | Lines | Description | Key Classes | Key Functions |
|------|-------|-------------|-------------|---------------|
| `__init__.py` | 1 | — | — | — |
| `chart.py` | 310 | — | ChartEngine | — |

<details>
<summary>Detailed file descriptions (1 files)</summary>

### `chart.py`

**Classes (1):**

- `ChartEngine`
  - AI Trader-এর visualization engine।
  - Methods: `__init__`, `create_full_chart`, `_add_candlesticks`, `_add_moving_averages`, `_add_sr_levels`, `_add_pattern_annotations`, `_add_rsi`, `_add_macd` (+2)

</details>

## research/

**Purpose:** Research — experiment runner, hypothesis engine, strategy generator

**Files:** 6

| File | Lines | Description | Key Classes | Key Functions |
|------|-------|-------------|-------------|---------------|
| `__init__.py` | 1 | — | — | — |
| `experiment_runner.py` | 614 | — | Experiment, ExperimentRunner, ResearchStrategyAdapter | — |
| `hypothesis_engine.py` | 470 | — | Hypothesis, HypothesisEngine | — |
| `research_agent.py` | 669 | — | ResearchAgent | — |
| `research_report.py` | 386 | — | ResearchReportGenerator | — |
| `strategy_generator.py` | 202 | research/strategy_generator.py — Minimal stub (Day 57 placeholder) | GeneratedStrategy, StrategyGenerator | — |

<details>
<summary>Detailed file descriptions (5 files)</summary>

### `experiment_runner.py`

**Classes (3):**

- `Experiment`
  - Represents a single experiment run.
  - Methods: `__init__`, `to_dict`, `_summarize_backtest`
- `ExperimentRunner`
  - Automated Experiment Execution Engine.
  - Methods: `__init__`, `create_experiment`, `create_auto_experiment`, `run_experiment`, `run_batch`, `_run_backtest`, `_create_strategy_adapter`, `_simulate_backtest` (+2)
- `ResearchStrategyAdapter`
  - Methods: `__init__`, `generate`

### `hypothesis_engine.py`

**Classes (2):**

- `Hypothesis`
  - Represents a single testable hypothesis.
  - Methods: `__init__`, `to_dict`, `set_result`, `__repr__`
- `HypothesisEngine`
  - AI Hypothesis Generation Engine.
  - Methods: `__init__`, `generate`, `generate_from_market_observation`, `generate_batch`, `evaluate_hypothesis`, `analyze_market_behavior`, `get_history`, `get_stats` (+1)

### `research_agent.py`

**Classes (1):**

- `ResearchAgent`
  - Autonomous Research Agent — AI Trader-এর Research Department.
  - Methods: `__init__`, `run_research_cycle`, `analyze_market`, `generate_hypothesis_from_market`, `test_strategy`, `evaluate_result`, `generate_weekly_report`, `mutate_best_strategy` (+2)

### `research_report.py`

**Classes (1):**

- `ResearchReportGenerator`
  - Generates research reports for the AI Trading System.
  - Methods: `generate_weekly`, `generate_single_experiment_report`, `save_report`, `print_report`, `_find_best_discovery`, `_find_best_hypothesis`, `_summarize_findings`, `_generate_recommendations` (+2)

### `strategy_generator.py`

**Description:** research/strategy_generator.py — Minimal stub (Day 57 placeholder)

**Classes (2):**

- `GeneratedStrategy`
  - A strategy produced by StrategyGenerator.
  - Methods: `to_dict`
- `StrategyGenerator`
  - Generates trading strategies by combining filter and entry components.
  - Methods: `__init__`, `random_strategy`, `mutate`, `combine`, `list_components`

</details>

## scripts/

**Purpose:** Scripts — backtest runners, validation, book5 knowledge base

**Files:** 12

| File | Lines | Description | Key Classes | Key Functions |
|------|-------|-------------|-------------|---------------|
| `build_book5_knowledge_base.py` | 422 | scripts/build_book5_knowledge_base.py | — | rule_to_dict, build_knowledge_base, main |
| `build_book5_markdown_reference.py` | 560 | scripts/build_book5_markdown_reference.py | — | build_markdown, main |
| `fix_bare_excepts.py` | 52 | scripts/fix_bare_excepts.py | — | — |
| `run_comprehensive_backtest.py` | 500 | scripts/run_comprehensive_backtest.py | — | parse_args, main, _generate_recommendations (+1) |
| `run_honest_validation.py` | 557 | scripts/run_honest_validation.py | — | make_sr_bounce_strategy, make_sr_resistance_only_strategy, make_donchian_breakout_strategy (+4) |
| `test_day92_apis.py` | 221 | test_day92_apis.py — Day 92 API integration tests | — | banner, main |
| `test_day93_integration.py` | 213 | test_day93_integration.py — Day 93 integration tests | — | banner, main |
| `test_day94_institutional_apis.py` | 238 | test_day94_institutional_apis.py — Day 94 institutional-grade API test | — | banner, main |
| `test_day95_alternatives.py` | 264 | test_day95_alternatives.py — Day 95 alternative API tests | — | banner, main |
| `test_integration_day90.py` | 165 | test_integration_day90.py — End-to-end integration test for Day 90 wir | — | make_synthetic_market_output, main |
| `test_sync_open_positions_fix.py` | 155 | test_sync_open_positions_fix.py — Day 90 bugfix verification | BadIterable | section, main |
| `verify_new_modules.py` | 118 | verify_new_modules.py — Verify all newly added modules load and work | — | — |

<details>
<summary>Detailed file descriptions (12 files)</summary>

### `build_book5_knowledge_base.py`

**Description:** scripts/build_book5_knowledge_base.py

**Functions (3):**

- `rule_to_dict()`
- `build_knowledge_base()`
  - Assemble the complete knowledge base from rule registry + chapter metadata.
- `main()`

### `build_book5_markdown_reference.py`

**Description:** scripts/build_book5_markdown_reference.py

**Functions (2):**

- `build_markdown()`
- `main()`

### `fix_bare_excepts.py`

**Description:** scripts/fix_bare_excepts.py

### `run_comprehensive_backtest.py`

**Description:** scripts/run_comprehensive_backtest.py

**Functions (4):**

- `parse_args()`
- `main()`
- `_generate_recommendations()`
  - Generate actionable recommendations for the live trading system.
- `_write_markdown_report()`
  - Write a human-readable Markdown report.

### `run_honest_validation.py`

**Description:** scripts/run_honest_validation.py

**Functions (7):**

- `make_sr_bounce_strategy()`
  - S/R bounce strategy using INCREMENTAL zone detection (no look-ahead).
- `make_sr_resistance_only_strategy()`
  - S/R resistance-only bounce (your 'best' setup — let's test it honestly).
- `make_donchian_breakout_strategy()`
  - Donchian channel breakout — simple, hard to overfit.
- `make_random_strategy()`
  - Random strategy — BASELINE. If your strategies can't beat this, they're noise.
- `parse_args()`
- `main()`
- `_write_markdown_report()`
  - Write the honest validation Markdown report.

### `test_day92_apis.py`

**Description:** test_day92_apis.py — Day 92 API integration tests

**Functions (2):**

- `banner()`
- `main()`

### `test_day93_integration.py`

**Description:** test_day93_integration.py — Day 93 integration tests

**Functions (2):**

- `banner()`
- `main()`

### `test_day94_institutional_apis.py`

**Description:** test_day94_institutional_apis.py — Day 94 institutional-grade API tests

**Functions (2):**

- `banner()`
- `main()`

### `test_day95_alternatives.py`

**Description:** test_day95_alternatives.py — Day 95 alternative API tests

**Functions (2):**

- `banner()`
- `main()`

### `test_integration_day90.py`

**Description:** test_integration_day90.py — End-to-end integration test for Day 90 wiring.

**Functions (2):**

- `make_synthetic_market_output()`
  - Build a market_output dict like MarketAgent.run() would produce.
- `main()`

### `test_sync_open_positions_fix.py`

**Description:** test_sync_open_positions_fix.py — Day 90 bugfix verification

**Classes (1):**

- `BadIterable`
  - Methods: `__iter__`

**Functions (2):**

- `section()`
- `main()`

### `verify_new_modules.py`

**Description:** verify_new_modules.py — Verify all newly added modules load and work

</details>

## tests/

**Purpose:** Test suites — unit tests, integration tests, step-by-step pipeline tests

**Files:** 32

| File | Lines | Description | Key Classes | Key Functions |
|------|-------|-------------|-------------|---------------|
| `__init__.py` | 1 | — | — | — |
| `steps/run_all_steps.py` | 179 | tests/steps/run_all_steps.py | — | _c, run_step, main |
| `steps/step_01_mt5_connection.py` | 134 | tests/steps/step_01_mt5_connection.py | — | _pass, _fail, _info (+2) |
| `steps/step_02_market_data.py` | 132 | tests/steps/step_02_market_data.py | — | _pass, _fail, _info (+3) |
| `steps/step_03_indicators.py` | 170 | tests/steps/step_03_indicators.py | — | _pass, _fail, _info (+2) |
| `steps/step_04_smc_engine.py` | 150 | tests/steps/step_04_smc_engine.py | — | _pass, _fail, _info (+3) |
| `steps/step_05_session.py` | 149 | tests/steps/step_05_session.py | — | _pass, _fail, _info (+2) |
| `steps/step_06_signal_engine.py` | 126 | tests/steps/step_06_signal_engine.py | — | _pass, _fail, _info (+2) |
| `steps/step_07_llm_analyst.py` | 171 | tests/steps/step_07_llm_analyst.py | — | _pass, _fail, _info (+2) |
| `steps/step_08_decision_agent.py` | 139 | tests/steps/step_08_decision_agent.py | — | _pass, _fail, _info (+2) |
| `steps/step_09_risk_engine.py` | 149 | tests/steps/step_09_risk_engine.py | — | _pass, _fail, _info (+2) |
| `steps/step_10_trade_permission.py` | 155 | tests/steps/step_10_trade_permission.py | — | _pass, _fail, _info (+2) |
| `steps/step_11_execution.py` | 185 | tests/steps/step_11_execution.py | — | _pass, _fail, _info (+2) |
| `test_adaptive_backtest_system.py` | 475 | tests/test_adaptive_backtest_system.py — Test the new adaptive backtes | — | make_synthetic_df, test_mt5_fetcher_pair_discovery, test_mt5_fetcher_synthetic_data (+12) |
| `test_book_pages_106_120.py` | 286 | Tests for new chart patterns from Book pages 106-120. | — | make_df, make_ohlc, test_rectangle_no_trade_state (+6) |
| `test_book_pages_136_151.py` | 242 | Tests for Book Pages 136-151 — Final Risk Management Guardrails. | — | test_correlation_no_open_positions, test_correlation_same_direction_violation, test_correlation_opposite_direction_partial_hedge (+11) |
| `test_cci_state_machine.py` | 305 | tests/test_cci_state_machine.py — Book 5 Chapter 11 CCI State Machine | — | test_long_entry_at_demand_zone, test_short_entry_at_supply_zone, test_no_entry_without_zone (+9) |
| `test_core.py` | 191 | — | TestDataValidator, TestIndicators, TestPatterns (+2) | make_df |
| `test_curve_mtf.py` | 361 | tests/test_curve_mtf.py — Book 5 Chapter 12 Curve MTF Methodology | — | make_book_p131_curve, test_book_p131_worked_example, test_all_5_positions (+12) |
| `test_entry_quality_guardrails.py` | 796 | Tests for Entry Quality Guardrails — 6 red flags from GBPUSD M5 trade  | — | make_ohlc, test_chasing_filter_blocks_extended_move, test_chasing_filter_allows_with_pullback (+12) |
| `test_flip_zones.py` | 406 | tests/test_flip_zones.py — Book 5 Chapter 8 Flip Zone Detector | — | make_df, test_demand_zone_flips_to_supply, test_supply_zone_flips_to_demand (+8) |
| `test_high_reliability_patterns.py` | 371 | Smoke test for High-Reliability Pattern Detector — spec compliance. | — | make_df, test_hammer_detection, test_shooting_star_detection (+12) |
| `test_ict_amd_signal_engine.py` | 417 | Smoke test for ICT/AMD Signal Engine — spec compliance. | — | make_intraday_ohlc, test_schema_conformance, test_insufficient_data (+9) |
| `test_multi_strategy_pa_engine.py` | 479 | Smoke test for Multi-Strategy PA Signal Engine — spec compliance. | — | make_4h_ohlc, make_h2_ohlc, test_schema_conformance (+12) |
| `test_odd_enhancers.py` | 829 | tests/test_odd_enhancers.py — Book 5 Chapter 6 Odd Enhancers Scoring S | — | make_erc_candle, make_indecision_candle, make_df (+12) |
| `test_pipeline.py` | 717 | — | TestResult, PipelineTestRunner | main |
| `test_production_integration.py` | 375 | tests/test_production_integration.py — Full Integration Test | — | make_realistic_df, test_production_system_initializes, test_data_quality_blocks_bad_bars (+11) |
| `test_risk_management.py` | 373 | tests/test_risk_management.py — Book 5 Chapter 14 Risk Management | — | test_risk_per_trade_tiers, test_beginner_graduation, test_position_sizing_stock (+12) |
| `test_sr_zones.py` | 150 | Smoke test for upgraded S/R Zone detection module. | — | make_synthetic_ohlc, test_basic, test_json_output (+6) |
| `test_stop_hunt_signal_engine.py` | 401 | Smoke test for Stop Hunt Signal Engine — spec compliance. | — | make_ohlc, make_ohlc_with_resistance_stop_hunt, make_ohlc_real_breakout_no_stop_hunt (+9) |
| `test_triple_top_bottom.py` | 226 | Tests for Triple Top/Bottom patterns (Book Spreads 6-7). | — | make_ohlc, test_triple_top_confirmed, test_triple_top_forming_no_trade (+4) |
| `test_unified_signal_engine.py` | 342 | Integration tests for Unified Signal Engine — verifies all 5 engines w | — | make_4h_ohlc, make_h2_ohlc, test_unified_schema (+9) |

<details>
<summary>Detailed file descriptions (31 files)</summary>

### `steps/run_all_steps.py`

**Description:** tests/steps/run_all_steps.py

**Functions (3):**

- `_c()`
- `run_step()`
  - একটা step চালায় এবং (success, duration) return করে।
- `main()`

### `steps/step_01_mt5_connection.py`

**Description:** tests/steps/step_01_mt5_connection.py

**Functions (5):**

- `_pass()`
- `_fail()`
- `_info()`
- `_warn()`
- `main()`

### `steps/step_02_market_data.py`

**Description:** tests/steps/step_02_market_data.py

**Functions (6):**

- `_pass()`
- `_fail()`
- `_info()`
- `_warn()`
- `test_symbol()`
  - একটা symbol এর জন্য data fetch test.
- `main()`

### `steps/step_03_indicators.py`

**Description:** tests/steps/step_03_indicators.py

**Functions (5):**

- `_pass()`
- `_fail()`
- `_info()`
- `_warn()`
- `main()`

### `steps/step_04_smc_engine.py`

**Description:** tests/steps/step_04_smc_engine.py

**Functions (6):**

- `_pass()`
- `_fail()`
- `_info()`
- `_warn()`
- `test_symbol()`
- `main()`

### `steps/step_05_session.py`

**Description:** tests/steps/step_05_session.py

**Functions (5):**

- `_pass()`
- `_fail()`
- `_info()`
- `_warn()`
- `main()`

### `steps/step_06_signal_engine.py`

**Description:** tests/steps/step_06_signal_engine.py

**Functions (5):**

- `_pass()`
- `_fail()`
- `_info()`
- `_warn()`
- `main()`

### `steps/step_07_llm_analyst.py`

**Description:** tests/steps/step_07_llm_analyst.py

**Functions (5):**

- `_pass()`
- `_fail()`
- `_info()`
- `_warn()`
- `main()`

### `steps/step_08_decision_agent.py`

**Description:** tests/steps/step_08_decision_agent.py

**Functions (5):**

- `_pass()`
- `_fail()`
- `_info()`
- `_warn()`
- `main()`

### `steps/step_09_risk_engine.py`

**Description:** tests/steps/step_09_risk_engine.py

**Functions (5):**

- `_pass()`
- `_fail()`
- `_info()`
- `_warn()`
- `main()`

### `steps/step_10_trade_permission.py`

**Description:** tests/steps/step_10_trade_permission.py

**Functions (5):**

- `_pass()`
- `_fail()`
- `_info()`
- `_warn()`
- `main()`

### `steps/step_11_execution.py`

**Description:** tests/steps/step_11_execution.py

**Functions (5):**

- `_pass()`
- `_fail()`
- `_info()`
- `_warn()`
- `main()`

### `test_adaptive_backtest_system.py`

**Description:** tests/test_adaptive_backtest_system.py — Test the new adaptive backtest system

**Functions (16):**

- `make_synthetic_df()`
  - Generate synthetic OHLCV for testing.
- `test_mt5_fetcher_pair_discovery()`
  - Fetcher discovers pairs (or falls back to defaults if MT5 unavailable).
- `test_mt5_fetcher_synthetic_data()`
  - Fetcher generates synthetic data when MT5 unavailable.
- `test_trade_simulator_long_tp()`
  - Long trade hits TP when price rises.
- `test_trade_simulator_short_sl()`
  - Short trade hits SL when price rises.
- `test_per_strategy_tester_runs()`
  - PerStrategyTester runs all strategies on synthetic data.
- `test_adaptive_single_mode_allows_solo()`
  - Single mode: one strategy with good WR can trade alone.
- `test_adaptive_strict_mode_blocks_solo()`
  - Strict mode: single strategy cannot trade alone (legacy behavior).
- ... and 8 more

### `test_book_pages_106_120.py`

**Description:** Tests for new chart patterns from Book pages 106-120.

**Functions (9):**

- `make_df()`
- `make_ohlc()`
- `test_rectangle_no_trade_state()`
  - Rectangle forming (no breakout) → NO_TRADE state.
- `test_rectangle_breakout_up()`
  - Rectangle breakout up → LONG signal.
- `test_momentum_screener_near_high()`
  - Price within 10% of high → MOMENTUM_CANDIDATE.
- `test_momentum_screener_far_from_high()`
  - Price far from high (>10% below) → no momentum candidate.
- `test_rectangle_breakout_down()`
  - Rectangle breakout down → SHORT signal.
- `test_new_patterns_in_detect_all()`
  - Verify detect_all includes the new patterns.
- ... and 1 more

### `test_book_pages_136_151.py`

**Description:** Tests for Book Pages 136-151 — Final Risk Management Guardrails.

**Functions (14):**

- `test_correlation_no_open_positions()`
  - No open positions → pass.
- `test_correlation_same_direction_violation()`
  - EURUSD long + GBPUSD long → block (correlation 0.85).
- `test_correlation_opposite_direction_partial_hedge()`
  - EURUSD long + GBPUSD short → partial hedge (less risky, still flagged but allowed).
- `test_correlation_uncorrelated_pairs_pass()`
  - EURUSD long + USDJPY short → low correlation, passes.
- `test_anti_revenge_no_loss_streak()`
  - No loss streak → pass even if oversized.
- `test_anti_revenge_loss_streak_normal_size()`
  - Loss streak + normal size → pass (disciplined).
- `test_anti_revenge_loss_streak_oversized()`
  - Loss streak + oversized → BLOCK.
- `test_cost_aware_ev_positive()`
  - High expected PnL → pass after costs.
- ... and 6 more

### `test_cci_state_machine.py`

**Description:** tests/test_cci_state_machine.py — Book 5 Chapter 11 CCI State Machine

**Functions (12):**

- `test_long_entry_at_demand_zone()`
  - Book P120: CCI < -100 at demand zone → ENTER long.
- `test_short_entry_at_supply_zone()`
  - Book P120/P123: CCI > +100 (e.g. +200) at supply zone → ENTER short.
- `test_no_entry_without_zone()`
  - Book P125: CCI is NOT a standalone signal — needs zone context.
- `test_exit_long_on_cci_retrace()`
  - Book P125: exit long when CCI drops back below +100.
- `test_exit_short_on_cci_retrace()`
  - Book P125: exit short when CCI rises back above -100.
- `test_add_to_long_requires_positive_cci()`
  - Book P125: add to long only if CCI > 0.
- `test_add_to_short_requires_negative_cci()`
  - Book P125: add to short only if CCI < 0.
- `test_ambiguous_zone_hold()`
  - Book P125: |CCI| < 20 = uncertain → HOLD.
- ... and 4 more

### `test_core.py`

**Classes (5):**

- `TestDataValidator`
  - Methods: `test_valid_data_passes`, `test_empty_df_fails`, `test_missing_column_fails`, `test_ohlc_logic_check`
- `TestIndicators`
  - Methods: `test_all_indicators_added`, `test_rsi_range`, `test_ai_context_keys`, `test_trend_values`
- `TestPatterns`
  - Methods: `test_pattern_column_exists`, `test_engulfing_column_exists`, `test_known_hammer`, `test_known_shooting_star`
- `TestSupportResistance`
  - Methods: `test_swing_highs_found`, `test_swing_lows_found`, `test_ai_context_keys`
- `TestSession`
  - Methods: `test_london_session`, `test_new_york_session`, `test_overlap_detected`

**Functions (1):**

- `make_df()`
  - Synthetic OHLCV DataFrame

### `test_curve_mtf.py`

**Description:** tests/test_curve_mtf.py — Book 5 Chapter 12 Curve MTF Methodology

**Functions (16):**

- `make_book_p131_curve()`
  - Book P131 worked example: proximal lines at $10 (demand) and $13 (supply).
- `test_book_p131_worked_example()`
  - Book P131: proximal lines at $10 and $13 → sub-zone width = $1, boundaries at $11 and $12.
- `test_all_5_positions()`
  - Book P131: 5 zone areas — Very Low / Low / Equilibrium / High / Very High.
- `test_directional_bias_mapping()`
  - Book P133: High/Very High=SELL_ONLY; Low/Very Low=BUY_ONLY; Equilibrium=TREND_FOLLOW.
- `test_timeframe_triplet_lookup()`
  - Book P129: each style maps to a specific (long, medium, short) triplet.
- `test_htf_override_conflict_waits()`
  - Book P135: 'longer frame always wins' — conflicting LTF signal → WAIT.
- `test_htf_ltf_aligned_trades()`
  - When LTF signals agree with HTF bias → TRADE.
- `test_equilibrium_mixed_signals_waits()`
  - Book P133: Equilibrium = TREND_FOLLOW_OR_NO_TRADE. Mixed LTF → WAIT.
- ... and 8 more

### `test_entry_quality_guardrails.py`

**Description:** Tests for Entry Quality Guardrails — 6 red flags from GBPUSD M5 trade post-mortem.

**Functions (29):**

- `make_ohlc()`
- `test_chasing_filter_blocks_extended_move()`
  - Flag 1: Extended move without pullback → BLOCK.
- `test_chasing_filter_allows_with_pullback()`
  - Flag 1: Extended move WITH pullback → PASS.
- `test_sl_swing_anchor_detected()`
  - Flag 2: SL near swing low → PASS.
- `test_sl_swing_anchor_fixed_pip_warning()`
  - Flag 2: SL far from any swing → WARNING.
- `test_tp_unconfirmed_territory()`
  - Flag 3: TP beyond any prior swing high → WARNING.
- `test_tp_validated_by_swing()`
  - Flag 3: TP near prior swing high → PASS.
- `test_indecision_candles_block()`
  - Flag 4: 2+ indecision candles → BLOCK.
- ... and 21 more

### `test_flip_zones.py`

**Description:** tests/test_flip_zones.py — Book 5 Chapter 8 Flip Zone Detector

**Functions (11):**

- `make_df()`
  - Build OHLCV DataFrame from lists of closes (others optional).
- `test_demand_zone_flips_to_supply()`
  - Book P89: demand zone broken down (close < distal) → reclassified as supply.
- `test_supply_zone_flips_to_demand()`
  - Book P89: supply zone broken up (close > distal) → reclassified as demand.
- `test_wick_pierce_does_not_flip()`
  - Book P89: only a candle CLOSE beyond distal counts — wick pierce does NOT flip.
- `test_no_break_no_flip()`
  - If price never closes beyond distal, no flip occurs.
- `test_multiple_flips_tracked()`
  - A zone that flips back and forth should be tracked across multiple flips.
- `test_zone_type_inference()`
  - Verify _infer_zone_type correctly identifies supply vs demand.
- `test_max_flips_safety_cap()`
  - A zone that flips too many times should be invalidated.
- ... and 3 more

### `test_high_reliability_patterns.py`

**Description:** Smoke test for High-Reliability Pattern Detector — spec compliance.

**Functions (17):**

- `make_df()`
  - Build DataFrame from list of (open, high, low, close) tuples.
- `test_hammer_detection()`
- `test_shooting_star_detection()`
- `test_doji_detection()`
- `test_bullish_marubozu()`
- `test_bullish_engulfing()`
- `test_bearish_engulfing()`
- `test_tweezer_top()`
- ... and 9 more

### `test_ict_amd_signal_engine.py`

**Description:** Smoke test for ICT/AMD Signal Engine — spec compliance.

**Functions (12):**

- `make_intraday_ohlc()`
  - Random-walk OHLC with hourly timestamps spanning multiple days.
- `test_schema_conformance()`
- `test_insufficient_data()`
- `test_weak_zone_sweep_not_counted_as_manipulation()`
  - Spec: Weak zone sweep → manipulation_detected = false.
- `test_strict_rr_filter()`
  - R:R < 1:6 must always be NO_TRADE even if other steps pass.
- `test_no_trade_when_no_strong_zone_for_tp()`
  - If no Strong zone exists for TP, must be NO_TRADE.
- `test_oneshot_helper()`
- `test_prompt_text()`
- ... and 4 more

### `test_multi_strategy_pa_engine.py`

**Description:** Smoke test for Multi-Strategy PA Signal Engine — spec compliance.

**Functions (15):**

- `make_4h_ohlc()`
  - 4H OHLC spanning multiple days.
- `make_h2_ohlc()`
  - H2 (2-hour) OHLC.
- `test_schema_conformance()`
- `test_unsupported_pair()`
- `test_unsupported_timeframe()`
- `test_insufficient_data()`
- `test_session_filter()`
  - Outside 12:30-14:30 BD Time → NO_TRADE.
- `test_sideways_trend_wait()`
  - Sideways trend → WAIT.
- ... and 7 more

### `test_odd_enhancers.py`

**Description:** tests/test_odd_enhancers.py — Book 5 Chapter 6 Odd Enhancers Scoring System

**Functions (25):**

- `make_erc_candle()`
  - Build a candle with a large body (ERC).
- `make_indecision_candle()`
  - Build a doji-like candle (small body, large wicks).
- `make_df()`
  - Convert list of candle dicts to a DataFrame.
- `make_perfect_demand_zone()`
  - A perfect Tier-A demand zone (Drop-Base-Rally, fresh, 2-candle base).
- `make_perfect_supply_zone()`
  - A perfect Tier-A supply zone (Rally-Base-Drop, fresh, 2-candle base).
- `test_tier_a_demand_zone_perfect()`
  - All 4 compulsory enhancers at max → Tier A, limit entry.
- `test_stale_zone_skipped()`
  - Book P68: ≥2 retests → score 0 → SKIP (hard gate).
- `test_base_too_long_skipped()`
  - Book P66: ≥6 candles in base → SKIP.
- ... and 17 more

### `test_pipeline.py`

**Classes (2):**

- `TestResult`
  - Methods: `__init__`, `__repr__`
- `PipelineTestRunner`
  - Runs all system tests and produces a structured report.
  - Methods: `__init__`, `test`, `print_report`

**Functions (1):**

- `main()`

### `test_production_integration.py`

**Description:** tests/test_production_integration.py — Full Integration Test

**Functions (14):**

- `make_realistic_df()`
  - Generate realistic OHLCV data (positive prices, proper OHLC relationships).
- `test_production_system_initializes()`
  - Test 1: Production system initializes all 6 defense layers.
- `test_data_quality_blocks_bad_bars()`
  - Test 2: Data quality validator blocks bars with negative prices.
- `test_news_blackout_blocks_trades()`
  - Test 3: News blackout blocks trades during high-impact news.
- `test_broker_health_blocks_after_rejections()`
  - Test 4: Broker execution guard blocks after too many rejections.
- `test_risk_manager_blocks_correlated_trades()`
  - Test 5: Risk manager blocks trades in same currency cluster.
- `test_adaptive_decision_blocks_low_wr_strategy()`
  - Test 6: Adaptive decision blocks strategies with WR below threshold.
- `test_volatility_sizer_reduces_risk()`
  - Test 7: Volatility-scaled sizer reduces risk when ATR is high.
- ... and 6 more

### `test_risk_management.py`

**Description:** tests/test_risk_management.py — Book 5 Chapter 14 Risk Management

**Functions (15):**

- `test_risk_per_trade_tiers()`
  - Book P154: experienced=2%, beginner=1%.
- `test_beginner_graduation()`
  - Book P154: beginner graduates to experienced when account triples.
- `test_position_sizing_stock()`
  - Book P155: $1,000 × 2% = $20 risk; entry $10, stop $8 → 10 shares.
- `test_forex_sizing_case_a()`
  - Book P155-156 AUD/USD: quote (USD) = account (USD) → Case A.
- `test_forex_sizing_case_b()`
  - Book P156 USD/JPY: quote (JPY) ≠ account (USD) → Case B.
- `test_margin_call_detection()`
  - Book P154: ×10+10%, ×5+20%, ×100+1% all → margin call.
- `test_max_loss_before_margin_call()`
  - Book P154: max loss % = 100% / leverage.
- `test_drawdown_circuit_breaker()`
  - Book P157: ≥20% drawdown → halt trading for month.
- ... and 7 more

### `test_sr_zones.py`

**Description:** Smoke test for upgraded S/R Zone detection module.

**Functions (9):**

- `make_synthetic_ohlc()`
  - Create realistic OHLC with clear S/R clusters.
- `test_basic()`
- `test_json_output()`
- `test_max_zones_filter()`
- `test_tf_adaptive_window()`
- `test_backward_compat()`
- `test_xau_pip_value()`
- `test_insufficient_candles()`
- ... and 1 more

### `test_stop_hunt_signal_engine.py`

**Description:** Smoke test for Stop Hunt Signal Engine — spec compliance.

**Functions (12):**

- `make_ohlc()`
  - Plain random-walk OHLC.
- `make_ohlc_with_resistance_stop_hunt()`
  - Build OHLC where:
- `make_ohlc_real_breakout_no_stop_hunt()`
  - Real breakout: price breaks ABOVE resistance zone with strong bullish
- `test_schema_conformance()`
- `test_insufficient_data()`
- `test_stop_hunt_confirmed_sell_signal()`
- `test_real_breakout_no_trade()`
- `test_no_zones_no_trade()`
- ... and 4 more

### `test_triple_top_bottom.py`

**Description:** Tests for Triple Top/Bottom patterns (Book Spreads 6-7).

**Functions (7):**

- `make_ohlc()`
- `test_triple_top_confirmed()`
  - Triple Top with neckline break → BEARISH/SHORT.
- `test_triple_top_forming_no_trade()`
  - Triple Top without neckline break → NEUTRAL/NO_TRADE.
- `test_triple_bottom_confirmed()`
  - Triple Bottom with neckline break → BULLISH/LONG.
- `test_triple_bottom_forming_no_trade()`
  - Triple Bottom without neckline break → NEUTRAL/NO_TRADE.
- `test_triple_patterns_in_detect_all()`
  - Verify detect_all includes Triple Top/Bottom.
- `test_no_triple_pattern_in_trending_market()`
  - In a clear trend (no 3 equal peaks) → no Triple pattern.

### `test_unified_signal_engine.py`

**Description:** Integration tests for Unified Signal Engine — verifies all 5 engines work together.

**Functions (12):**

- `make_4h_ohlc()`
- `make_h2_ohlc()`
- `test_unified_schema()`
- `test_insufficient_data()`
- `test_all_engines_run()`
- `test_consensus_voting()`
- `test_engine_disabling()`
- `test_oneshot_helper()`
- ... and 4 more

</details>

---

## Dependency Map

### Core Dependencies (who imports whom)

```
main.py
  ├── core/runtime.py
  ├── core/trading_engine.py
  ├── agents/analysis_agent.py
  │     ├── analysis/unified_signal_engine.py
  │     │     ├── analysis/support_resistance.py
  │     │     ├── analysis/supply_demand_zones.py
  │     │     ├── analysis/stop_hunt_signal_engine.py
  │     │     ├── analysis/ict_amd_signal_engine.py
  │     │     ├── analysis/multi_strategy_pa_engine.py
  │     │     └── analysis/high_reliability_patterns.py
  │     ├── analysis/decision_bridge.py
  │     │     └── analysis/adaptive_decision_engine.py
  │     ├── data/fetcher.py
  │     ├── data/indicators.py
  │     └── analysis/odd_enhancers.py
  ├── agents/decision_agent.py
  ├── agents/risk_agent.py
  │     └── risk/risk_engine.py
  ├── agents/learning_agent.py
  └── broker/mt5_connection.py

core/production_trading_system.py
  ├── core/graceful_shutdown.py
  ├── risk/strict_risk_manager.py
  ├── risk/adversarial_defenses.py
  ├── risk/cognitive_bias_defenses.py
  ├── analysis/decision_bridge.py
  ├── backtest/mt5_bulk_fetcher.py
  └── backtest/honest_backtest_engine.py

backtest/per_strategy_tester.py
  ├── backtest/mt5_bulk_fetcher.py
  ├── analysis/pin_bar_strategy.py
  ├── analysis/high_reliability_patterns.py
  ├── analysis/supply_demand_zones.py
  ├── analysis/support_resistance.py
  ├── analysis/stop_hunt_signal_engine.py
  ├── analysis/ict_amd_signal_engine.py
  ├── analysis/multi_strategy_pa_engine.py
  └── analysis/cci_state_machine.py
```

### Analysis Module Dependencies

```
analysis/odd_enhancers.py
  └── analysis/supply_demand_zones.py (uses zone dicts)

analysis/flip_zones.py
  └── (standalone, uses zone dicts from supply_demand_zones)

analysis/cci_state_machine.py
  └── (standalone, CCI indicator from data/indicators_ext.py)

analysis/curve_mtf.py
  └── (standalone, uses zone dicts)

analysis/risk_management.py
  └── (standalone, Book 5 Chapter 14 rules)

analysis/decision_bridge.py
  └── analysis/adaptive_decision_engine.py

analysis/unified_signal_engine.py
  ├── analysis/support_resistance.py
  ├── analysis/supply_demand_zones.py
  ├── analysis/stop_hunt_signal_engine.py
  ├── analysis/ict_amd_signal_engine.py
  ├── analysis/multi_strategy_pa_engine.py
  └── analysis/high_reliability_patterns.py

analysis/book_rules_index.py
  └── (registry of 119 rules, references all analysis modules)
```

### Risk Module Dependencies

```
risk/strict_risk_manager.py
  └── (standalone, thread-safe)

risk/adversarial_defenses.py
  └── utils/logger.py

risk/cognitive_bias_defenses.py
  └── utils/logger.py

risk/risk_engine.py
  ├── risk/position_sizer.py
  ├── risk/correlation_manager.py
  └── risk/book_guardrails.py

risk/entry_quality_guardrails.py
  ├── analysis/support_resistance.py
  ├── analysis/supply_demand_zones.py
  └── analysis/market_regime.py
```

### Backtest Module Dependencies

```
backtest/honest_backtest_engine.py
  ├── backtest/mt5_bulk_fetcher.py (for data)
  └── utils/logger.py

backtest/per_strategy_tester.py
  ├── backtest/mt5_bulk_fetcher.py
  ├── analysis/*.py (all strategy modules)
  └── analysis/odd_enhancers.py

backtest/mt5_bulk_fetcher.py
  ├── data/fetcher.py (MT5 integration)
  └── utils/logger.py

scripts/run_honest_validation.py
  ├── backtest/honest_backtest_engine.py
  ├── backtest/mt5_bulk_fetcher.py
  └── risk/adversarial_defenses.py

scripts/run_comprehensive_backtest.py
  ├── backtest/mt5_bulk_fetcher.py
  ├── backtest/per_strategy_tester.py
  └── analysis/adaptive_decision_engine.py
```

---

## Test Suite Summary

| Test File | Tests | Covers |
|-----------|-------|--------|
| `test_odd_enhancers.py` | 20 | Scoring tiers A/B/SKIP, PA confluence, Book P78 example |
| `test_flip_zones.py` | 10 | Demand↔Supply flips, wick vs close, multi-flip |
| `test_cci_state_machine.py` | 12 | Entry/exit/add, ambiguous zone, confluence |
| `test_curve_mtf.py` | 15 | Book P131 worked example, HTF override, bias |
| `test_risk_management.py` | 15 | Position sizing, margin call, drawdown circuit breaker |
| `test_adaptive_backtest_system.py` | 15 | MT5 fetcher, per-strategy tester, decision engine |
| `test_production_integration.py` | 13 | All defense layers working together |
| `test_sr_zones.py` | — | S/R zone detection |
| `test_high_reliability_patterns.py` | — | 20 candlestick patterns |
| `test_stop_hunt_signal_engine.py` | — | Stop hunt detection |
| `test_ict_amd_signal_engine.py` | — | ICT/AMD signal engine |
| `test_multi_strategy_pa_engine.py` | — | Multi-strategy PA engine |
| `test_unified_signal_engine.py` | — | Unified signal consensus |
| `test_entry_quality_guardrails.py` | — | Entry quality checks |
| `test_triple_top_bottom.py` | — | Triple top/bottom pattern |
| `test_book_pages_106_120.py` | — | Book rules pages 106-120 |
| `test_book_pages_136_151.py` | — | Book rules pages 136-151 |
| `test_core.py` | — | Core module tests |
| `test_pipeline.py` | — | Full pipeline test |
| `tests/steps/step_01` through `step_11` | 11 | Step-by-step pipeline verification |

**Total test suites:** 19+  
**Total tests:** 100+ (all passing)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    PRODUCTION TRADING SYSTEM                 │
│              core/production_trading_system.py               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │  Layer 1    │  │   Layer 2    │  │    Layer 3      │   │
│  │ Data Quality│  │Cognitive Bias│  │   Adversarial   │   │
│  │             │  │  Defenses    │  │    Defenses     │   │
│  │ • Validator  │  │ • PreReg     │  │ • ExecGuard     │   │
│  │ • Spike filt │  │ • Graveyard  │  │ • NewsBlackout  │   │
│  │ • Gap detect │  │ • Calibratn  │  │ • CrashRecovery │   │
│  │             │  │ • Selection  │  │ • DegradeMonitor│   │
│  │             │  │   Audit      │  │ • VolScaledSizer│   │
│  │             │  │             │  │ • OrderReconcilr│   │
│  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘   │
│         │                │                   │            │
│  ┌──────┴──────────────────┴───────────────────┴──────┐    │
│  │              Layer 4: Risk Management              │    │
│  │         risk/strict_risk_manager.py                │    │
│  │  • 0.5% risk per trade • Correlation control       │    │
│  │  • Daily/weekly limits   • Drawdown circuit breaker│    │
│  └──────────────────────┬────────────────────────────┘    │
│                         │                                  │
│  ┌──────────────────────┴────────────────────────────┐    │
│  │           Layer 5: Decision Engine                │    │
│  │     analysis/adaptive_decision_engine.py          │    │
│  │     analysis/decision_bridge.py                   │    │
│  │  • Calibrated weights • Confluence scoring        │    │
│  │  • Single/confluence/strict modes                 │    │
│  └──────────────────────┬────────────────────────────┘    │
│                         │                                  │
│  ┌──────────────────────┴────────────────────────────┐    │
│  │           Layer 6: Infrastructure                 │    │
│  │     core/graceful_shutdown.py                     │    │
│  │     utils/safe_pickle.py                          │    │
│  │  • Signal handlers   • State persistence          │    │
│  │  • Cleanup callbacks • Tamper detection           │    │
│  └───────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Book 5 (Frank Miller S&D) Implementation Summary

| Chapter | Pages | Module | Rules |
|---------|-------|--------|-------|
| Ch 1: S/R Basics | 1-20 | `analysis/support_resistance.py` | 5 |
| Ch 2: S/D Economics | 21-30 | `analysis/supply_demand_zones.py` | — |
| Ch 3: Zone ID (ERC) | 31-45 | `analysis/supply_demand_zones.py` | — |
| Ch 4: Zone Drawing | 46-55 | `analysis/supply_demand_zones.py` | — |
| Ch 5: Fresh/Original | 56-60 | `analysis/odd_enhancers.py` | — |
| Ch 6: Odd Enhancers | 61-78 | `analysis/odd_enhancers.py` | 15 |
| Ch 7: PA Confluence | 79-88 | `analysis/odd_enhancers.py` (PA layer) | 11 |
| Ch 8: Flip Zones | 89-95 | `analysis/flip_zones.py` | 4 |
| Ch 9: Reversal/Continuation | 96-105 | `analysis/advanced_patterns.py` | — |
| Ch 10: Gap Trading | 106-115 | `analysis/supply_demand_zones.py` | — |
| Ch 11: CCI Confluence | 116-125 | `analysis/cci_state_machine.py` | 8 |
| Ch 12: MTF Curve | 126-135 | `analysis/curve_mtf.py` | 8 |
| Ch 13: Trade Walkthrough | 136-152 | `analysis/curve_mtf.py` | 2 |
| Ch 14: Risk Management | 153-157 | `analysis/risk_management.py` + `risk/strict_risk_manager.py` | 10 |
| **Total** | **1-163** | **8 modules** | **119 rules** |

---

## Quick Start

### 1. Install Dependencies
```bash
pip install MetaTrader5 pandas numpy matplotlib ta scipy
```

### 2. Run Honest Validation (Windows with MT5)
```powershell
cd D:\Projects\forex_ai
python scripts\run_honest_validation.py --pairs EURUSD,GBPUSD,XAUUSD --timeframes M15,H1,H4 --max-candles 5000
```

### 3. Run Comprehensive Backtest
```powershell
python scripts\run_comprehensive_backtest.py --pairs EURUSD,GBPUSD --timeframes H1,H4 --max-candles 3000
```

### 4. Run All Tests
```bash
bash scripts/run_all_tests.sh
```

### 5. Start Production Trading System
```python
from core.production_trading_system import ProductionTradingSystem

system = ProductionTradingSystem(
    account_equity=10_000,
    is_beginner=True,
    mode="confluence",
)

system.startup()
while system.is_running():
    for pair in system.get_pairs():
        df = system.fetch_data(pair, 'H1')
        signals = your_strategy_engine.analyze(df)
        decision = system.evaluate(pair, df, signals)
        if decision:
            system.execute_trade(pair, decision)
system.graceful_shutdown()
```

---

## File Statistics

**Total Python files:** 383  
**Total lines of code:** 125,921  
**Total classes:** 466  
**Total functions:** 755  
**Test suites:** 19+  
**Total tests:** 100+ (all passing)
