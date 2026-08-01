# `analysis/` — Full Module Documentation (Deep/Code-Level)

Ei document ta `analysis/` folder-er **101-ta Python module** (+ `database/` subpackage) ke সরাসরি source code পড়ে (docstring, class, method, constant সব দেখে — শুধু filename দেখে guess kore na) generate kora hoyeche। প্রতিটা module-er জন্য: এটা কী করে, কীভাবে করে (algorithm/logic), input/output shape, এবং important design notes/bug-fix history (যেগুলো code comment-e explicitly লেখা আছে) দেওয়া হয়েছে।

---

## 📁 Folder Structure

```
analysis/
├── __init__.py                  (empty package init)
├── database/
│   └── __init__.py               (empty package init)
├── ... 99 more .py modules
└── support_resistance.py.bak2    (older backup — see notes)
```

---

## 🧠 1. Core Signal & Decision Engines

### `unified_signal_engine.py` — `UnifiedSignalEngine`
Pura codebase-er **master orchestrator**। এটা 5-6 ta আলাদা strategy engine-ke (stop-hunt, ICT/AMD, multi-strategy price-action, candlestick patterns, S/R zones) ekসাথে চালিয়ে ekটা single unified JSON output-e merge kore। Core flow (`analyze(df, symbol, lower_tf_df)`):
1. S/R zones detect kore (support/resistance),
2. আলাদা আলাদা zone source (S/R, supply/demand, trendline) থেকে `_zones_to_unified()` diye ekTa consistent schema-te merge kore (`{type, zone_top, zone_bottom, touches, strength, source}`),
3. প্রতিটা sub-engine call kore (falls back gracefully if any fails via `_fallback_stop_hunt/_fallback_ict/_fallback_pa`),
4. `_compute_consensus()` — voting-based consensus: koyta engine BUY/SELL bole shetar upor final decision nirbhor kore,
5. `_build_unified_result()` diye final dict banay, jeta `to_prompt_text()` diye LLM prompt-er jonno plain text-e o convert kora jay।
Minimum candle requirement enforce kore (`MIN_CANDLES_REQUIRED`); insufficient data hole `_insufficient_data_result()` return kore. One-shot helper: `detect_unified_signal()`.

### `adaptive_decision_engine.py` — `AdaptiveDecisionEngine`
Purano "sob strategy-ke agree korte hobe" rigid consensus gate-er shomoshsha shomadhan kore ("multiple mandatory strategies = no trades" problem)। Egta **soft confluence scoring system** replace kore ja backtest result theke shekhe। 3-ta mode ache:
- **`single`** — ekTa strategy-i jodi tar historical win rate `MIN_WIN_RATE_SOLO`-r beshi hoy tahole alone trade nite pare (kono confluence lagbe na)।
- **`confluence`** (default) — soft scoring; multiple strategy agree korle bonus pay kintu mandatory na।
- **`strict`** — legacy mode, 2+ strategy agree korte hobe (comparison-er jonno rakha)।
`load_backtest_results()`/`load_from_file()` diye backtest JSON theke প্রতিটা strategy-r weight calibrate kore (`_get_weight` = base × confidence_mult × win_rate_mult)। `decide(signals, current_price)` final `Decision` object dey (action, confidence tier High/Medium/Low, score, extracted entry/SL/TP via `_extract_levels`)। `export_weights()` diye calibrated weight JSON file-e save kora jay live trading system-er jonno.

### `decision_bridge.py` — `UnifiedToAdaptiveBridge` + `make_adaptive_decision()`
`UnifiedSignalEngine`-er (rigid voting) output ke `AdaptiveDecisionEngine`-er (calibrated-weight learning) input-e convert kore, jate purono engine implementation reuse hoy কিন্তু নতুন adaptive learning-o pawa jay। `extract_signals()` unified result theke প্রতিটা sub-engine-er signal ke `StrategySignal` object-e map kore। `make_adaptive_decision()` — process-wide, per-(mode, weights_path) **thread-safe singleton bridge** cache kore (double-checked locking diye race condition avoid kora hoyeche)।

### `multi_strategy_pa_engine.py` — `MultiStrategyPAEngine`
Ekta **spec-compliant, 8-step gated** price-action signal engine, session-restricted (`SESSION_START_UTC`–`SESSION_END_UTC`, BD time 11:00–22:00)। Pipeline:
1. S/R zone + touch-based bias,
2. Trend structure + BOS/CHOCH detection,
3. 2-candle shooting-star reversal check,
5. Lower-timeframe (MTF) confirmation,
6. Supply/demand zone (3 consecutive same-direction momentum candle diye base candle mark),
7. Confluence scoring (highest-confluence zone khoja),
8. **6-factor checklist** — kompokkhe 3 pass korte hobe।
Momentum candle (body ≥70% range) o "baby candle" (weak momentum) classify kore, engulfing/rejection-wick detect kore. Final `_generate_signal()`-e 8-gate check pass hole signal dey, na hole `_no_trade_signal`/`_wait_signal`।

### `dat_framework.py` — `DATFramework` (Direction–Area–Trigger)
Masterclass concept: trade newar age 3-ta jinish confirm korte hoy sequentially — **Direction** (uptrend/downtrend, EMA+MTF bias theke), **Area** (S/R / Order Block / FVG / Fibonacci zone-e ache ki na), **Trigger** (oi zone-e strong candlestick confirmation ache ki na)। Tinta-i align korle tobei signal, nahole NO TRADE। Institutional-review note-e bola ache: age Area stage-e sudhu SupportResistance wire kora chilo, ekhon FVG/Fibonacci/OrderBlock — sob module-i properly consult kore `_evaluate_area()`. `DATResult.to_dict()` diye output normalize hoy.

### `trend_level_signal.py` — `TrendLevelSignalFramework`
Boi-er "Trend, Level, Signal" (Page 79-80) unified framework। 3-ta question sequentially answer kore: (1) Market ki korche — regime (trending/ranging/choppy), (2) current price-er kache ki key level ache, (3) shei level-e pattern signal ache ki na — tarpor `_decide_action()` diye final BUY/SELL/WAIT/NO_TRADE decide kore।

### `book_rules_index.py` — `BookRule` registry
Ekta trading-book (chapter/page reference soho) theke extract kora rules-er **searchable index**। Functions: `get_all_rules()`, `get_rules_by_chapter()`, `get_rules_by_category()` (pattern/indicator/risk/trend/strategy/fundamental), `get_no_trade_conditions()`, `get_deterministic_rules()` vs `get_rules_needing_confirmation()`, `get_design_principles()`, `find_rule(id)`, এবং `get_implementation_map()` — kon rule kon file/function-e implement kora ache tar mapping dey।

---

## 🕯️ 2. Price Action & Candlestick Patterns

### `candlestick_patterns_ml.py` — `CandleStickPatterns`
Simplest layer: **8-ta boolean pattern detector** — Inverted Hammer, Hammer, Dragonfly Doji, Bullish Engulfing, Bullish Harami, Piercing Pattern, Morning Star, Morning Doji Star। Sob static-method style, raw OHLC value niye True/False dey — kono trend/volume filter ছাড়া।

### `candlestick_patterns_br.py` — Brazilian-book 11-pattern scanner (functions, no class)
Portuguese-named function set: `marobozu`, `doji`, `spinning_top`, `estrela` (Star), `martelo` (Hammer), `martelo_invertido` (Inverted Hammer), `engolfo_de_alta` (Bullish Engulfing), `harami_fundo` (Bullish Harami), `linha_perfuracao` (Piercing Line), `pinca_fundo` (Tweezer Bottom), `chute_alta` (Bullish Kicking)। Important engineering fixes documented in-code:
- **FX-unsafe absolute thresholds fixed** — `doji()`/`spinning_top()` age stock-scaled absolute price threshold babohar korto ja FX 5-decimal quote-e meaningless chilo (sob candle Doji dhora porto)। Ekhon `_resolve_absolute_threshold()` symbol-based pip-scaling → ATR-based scaling → stock-default fallback — ei 3-tier resolution babohar kore।
- **Performance fix**: trend (9-EMA/50-SMA) age প্রতি pattern function-e আলাদা kore recompute hoto (8x); ekhon `df.attrs` cache babohar kore O(1) per DataFrame kora hoyeche।
- `require_confirmation` (Hammer) o `require_gap` (Piercing Line) — configurable flags, jate FX/crypto (24h market, gap thake na) o equity (gap-dependent pattern) duitai support kore।
- `detect_all(df, **kwargs)` — 11-ta pattern ek shathe run kore; age ekta bug chilo jekhane kwargs accept korto kintu forward korto na — eta fix kora hoyeche।

### `candlestick_patterns_mw.py` — MotiveWave-style **33-pattern scanner** (function-based, no class)
Sobcheye comprehensive candlestick module — 1-bar (Doji variants, Hammer, Shooting Star, Marubozu variants), 2-bar (Engulfing, Harami, Piercing/Dark Cloud, Tweezer, Kicker, Counterattack, Separating Lines), 3-bar (Morning/Evening Star + Doji variant, Three Soldiers/Crows, Abandoned Baby, Three Inside/Outside, Upside Gap Two Crows) — total ~33 pattern। `compute(df, ...)` full scan chalay, প্রতি pattern-ke `csp_pattern/csp_category/csp_bars/csp_signal/csp_trend` column-e output kore। Nijer 50/200-SMA trend classifier ache, kintu `precomputed_trend` parameter diye baire theke shared trend inject kora jay (jate `candlestick_patterns_br.py`-r shathe consistent trend definition babohar hoy)। `add_confirmation()` — 1-bar reversal pattern-er jonno "next bar confirmation" logic add kore, **no-repainting guarantee soho** (last row-e always `csp_confirmation_pending=True` — repaint prevent korar jonno)।

### `patterns.py` — `PatternDetector`
TA-Lib **chhara** (external dependency ছাড়া) full candlestick detector class। `detect_all()`, `is_doji/is_hammer/is_shooting_star/is_pin_bar` row-level check, `detect_engulfing`, `detect_morning_evening_star`, `detect_three_bar_continuation/reversal`, `detect_breakout_candle`। Erpore "Book Page" reference soho aro onek method আছে: `detect_piercing_line`, `detect_harami`, `detect_three_soldiers_crows`, `detect_context_patterns`, `detect_dark_cloud_cover`, `detect_doji_variants`, `detect_three_methods` (Rising/Falling Three Methods), `detect_tweezers`, `detect_inside_bar_false_breakout` (stop-loss hunting pattern), `classify_engulfing_context/doji_context/harami_context`। Shesh-e `get_ai_context()` diye AI Brain-er jonno compressed pattern-context dict dey।

### `high_reliability_patterns.py` — `HighReliabilityPatternDetector`
Strict, **spec-compliant 20-pattern library**, প্রতিটা pattern-er nijer threshold constant ache (`HAMMER_WICK_BODY_RATIO`, `MARUBOZU_BODY_RATIO`, ইত্যাদি)। Sob 20-ta pattern (Hammer, Shooting Star, Inverted Hammer, Hanging Man, Doji, Bullish/Bearish Marubozu, Bullish/Bearish Engulfing, Tweezer Top/Bottom, Piercing Line, Dark Cloud Cover, Harami, Morning/Evening Star, Three White Soldiers/Black Crows, Three Inside Up/Down) — private `_detect_*` method-e implement kora, comment-e number-tagged (1-20)। **Zone confluence validation** — `_check_zone_confluence()` diye check kore pattern-ta kono S/R/Supply/Demand/Trendline zone-er kache ache ki na, na thakle `reliability="Low"` mark hoy। `analyze_repetition()` — multi-bar repetition analysis (spec rule 3)। One-shot helper: `detect_high_reliability_patterns()` — exact spec JSON schema dey।

### `advanced_patterns.py` — `AdvancedPatternDetector`
**Chart pattern** (candlestick na, larger structural pattern) detection engine: Head & Shoulders (+ Inverse), Double Top/Bottom, Triple Top/Bottom, Triangle (Ascending/Descending/Symmetric), Flag, Wedge (Rising/Falling), Cup & Handle, Rectangle, Momentum Screener (52-week high proximity)। `_find_swings()` diye swing high/low khoje, তারপর pattern-shape match kore। `boost_confidence()` — indicator + regime context diye confidence adjust kore; `filter_false_patterns()` — low-quality/regime-mismatched pattern discard kore। `get_ai_context()` full pipeline run kore final AI-ready context dey।

### `long_term_patterns.py` — 8 function-based long-term topping/bottoming detectors
`_swing_points()` diye local swing high/low khoje, tarpor: `detect_three_mountain_top` (triple top), `detect_three_buddha_top` (H&S), `detect_three_river_bottom` (inverse H&S), `detect_inverted_three_buddha` (triple bottom), `detect_dumpling_top` (rounding top/Nison-style), `detect_frypan_bottom` (rounding bottom), `detect_tower_top`/`detect_tower_bottom` (sharp V-reversal)। `detect_all(df)` shob detector run kore unfiltered candidate list dey — confluence engine-e gate kora expected।

### `pin_bar_strategy.py` — `PinBarStrategy`
Boi Page 81-95 — Pin Bar Trading Strategy, full setup-detection class। `detect()` last candle-e pin bar detect kore ebong **3-ta filter criteria** evaluate kore: (1) timeframe check — sudhu 4H/Daily/W1 pin bar seriously nite hobe (Page 83), (2) trend alignment (Page 84) — pin direction prevailing trend-er sathe match korte hobe, (3) level confluence (Page 85) — key S/R ba 21-MA-r kache hote hobe। `_calculate_entries()` diye aggressive+conservative entry/SL/TP calculate kore (Page 92-95), `_score_quality()` diye A/B/C/D grade dey।

### `engulfing_bar_strategy.py` — `EngulfingBarStrategy`
Similar structure `pin_bar_strategy.py`-r moto but engulfing candle-er jonno। `_detect_engulfing()`, tarpor confluence check tin dic theke — `_check_ma_confluence`, `_check_fib_confluence` (golden zone `FIB_GOLDEN_LOW`–`FIB_GOLDEN_HIGH`), `_check_sr_confluence`। `_calculate_entries()` + `_score_quality()` diye A-D grade full setup dict (`EngulfingSetup`) return kore।

### `megaphone_pennant.py` — `classify()` function
Pure classifier — **directional vote dey na**, nijer docstring-e explicitly bola ache। Sudhu shesh 2-ta confirmed swing high + 2-ta confirmed swing low niye classify kore:
- **MEGAPHONE** (expanding range, no trade possible): higher highs AND lower lows
- **PENNANT** (contracting range, breakout imminent, dui side bracket): lower highs AND higher lows
- Onno shob case → **UNKNOWN** (etake bug na — ei module-er scope-er bairer jinish, ordinary trend onno module handle korbe)
Contract note: swing input MUST already confirmed hote hobe, nahole repaint hote pare (`strategies/breakout.py`-r shathe same look-ahead-bias contract share kore)।

---

## 💰 3. Smart Money Concepts (SMC) / ICT

### `smart_money.py` — `SmartMoneyEngine`
Duita usage mode: **single-timeframe fast** (`analyze_single()`) o **full multi-timeframe top-down** (`analyze()`: D1 bias → H4 structure → H1 OB → M15 entry)। `_current_kill_zone()` diye current UTC time kon ICT kill zone (`KILL_ZONES`)-e pore seta detect kore, `_score_confluence()` diye structure + liquidity + nearest OB + nearest FVG + HTF bias mile confluence score dey, tarpor `_build_explanation()` human-readable reasoning dey।

### `smc_engine.py` — `SMCEngine`
`smart_money.py`-r moto kintu independent implementation — nijer `SCORE_WEIGHTS`/`MIN_TRADE_SCORE` constant, `_fetch_with_atr()` diye ATR-soho data pull kore, `_score_confluence()` H4 sweep/BOS/CHOCH + nearest OB/FVG + M15 sweep/BOS/pattern mile confluence score dey, `_rank_zone()` grade dey।

### `smc_advanced.py` — `SMCAdvancedEngine`
`order_block.py`/`breaker_block.py`-r **complementary** — Mitigation Block + Inducement detector। `_find_order_blocks()` — simplified internal OB finder (full logic `order_block.py`-e), `_find_mitigation_blocks()` — kon OB "mitigated" (partially/fully retested) hoyeche khoje, `_find_inducements()` — chhoto swing high/low khoje jegulo liquidity-grab hisebe kaj kore। `_bias_and_signal()` active signal theke final bias/signal ber kore।

### `order_block.py` — `OrderBlockDetector`
Order block detection-er **canonical implementation**। `mode` parameter — `'consecutive'` (default, video-6-backed) ba `'single'` (Day-44 legacy, শুধু A/B backtest-er জন্য)। `detect(df, closed_bars_only, max_results)` — **contract**: `df`-e sudhu CLOSED bar thakte hobe (repaint prevent korte)। `_find_ob_run()` — impulse-er ager candle range ber kore, `_find_confluent_fvg()` — OB-er sathe FVG overlap ache ki na check kore (confluence boost), `_score()` — ATR-ratio + structure-break + FVG-confluence + sweep-conditioning niye score dey, `_dedupe_keep_best()` duplicate zone remove kore, `nearest_active()` current price-er sobcheye kache active OB ber kore।

### `breaker_block.py` — `BreakerBlockDetector`
ICT concept: **failed Order Block** ja support↔resistance flip kore। Bullish breaker: downtrend-e ekta bullish OB break hoy, price abar retest kore, oi broken level ekhon support hisebe act kore। `detect(df, order_blocks)` — existing order_block list nive, `_check_breaker()` diye protita OB break+retest hoyeche ki na check kore। Powerful reversal signal।

### `fvg_detector.py` — `FVGDetector`
**Fair Value Gap (FVG)** detector — 3-candle gap pattern jekhane middle candle-er range surrounding 2-ta candle-r sathe overlap kore na। `detect(df)` — df-e already `atr` column thakte hobe। `nearest_active()` current price-er sobcheye kache unfilled FVG ber kore।

### `liquidity.py` — `LiquidityPoolAnalyzer`
Equal-highs/equal-lows base kore **liquidity pool** khoje। `_find_swings()` → `_find_equal_levels()` (ATR-tolerance-er modhye kache-kachi 2+ swing point ekসাথে cluster kore) → `_build_pools()` (protita pool-er জন্য sweep hoyeche ki na check kore, merge kore)। `_check_sweep()` — pool level toiri howar pore kono candle shei level swept korche ki na। `_premium_discount_zone()` — recent range-er midpoint diye price premium/discount zone-e ache ki na bole।

### `liquidity_engine.py` — `LiquidityEngine`
**Unified liquidity module** — sob liquidity sub-module (equal highs/lows, PDH/PDL, Asian range) ekসাথে চালিয়ে ekটা unified `liquidity_bias` + confidence score dey। `_classify_sweep_pattern()` — sweep pattern classify kore, `_fvg_confluence()` — stop-hunt direction-er sathe matching fresh FVG ache ki na check kore, `_build_liquidity_levels()` sob source ekটা common schema-te anay, `_score_liquidity_bias()` — **recalibrated from backtest evidence** (score breakdown comment-e detailed explain kora ache)।

### `liquidity_structure.py` — `LiquidityStructureAnalyzer`
`classify_internal_external()` — protita level-ke EXTERNAL/INTERNAL scope tag kore। `classify_resistance()` — last 3 confirmed swing point dekhe classify kore। `detect_trendlines()` — recent swing low (rising support)/high (falling resistance)-er upor line fit kore (`_fit_line`, min R² threshold soho)। `check_trendline_sweep()` — trendline-er current-bar value-r opore wick-through-and-close-back check kore। `detect_inducement()` — recent stop-hunt event ekta INTERNAL level swept korle inducement flag kore।

### `liquidity_zones.py` — `LiquidityZoneMapper`
`find_equal_highs()`/`find_equal_lows()` — cluster-based equal level detection। `calculate_previous_levels()` — **PDH/PDL/PWH/PWL** (Previous Day/Week High/Low) — always last FULLY COMPLETED period babohar kore (`unique_days/weeks[-2]`), never in-progress। `asian_session_range()` — Asian session (default 00:00–08:00 UTC) high/low range। **Important no-lookahead contract explicitly documented**: expanding-window (`df.iloc[:i+1]`) e call korte hobe backtest-e, purata dataset-e ekbar call kore treat kora jabe na jeno seta "known at earlier bar" chilo।

### `stop_hunt_detector.py` — `StopHuntDetector` ⚠️ **DEPRECATED**
Level-list driven (Day 62) stop-hunt engine। Code-e explicit **deprecation notice** ache — codebase-e duita independent stop-hunt engine ache (ei ta + newer `stop_hunt_signal_engine.py`) jara same event-e disagree korte pare kono arbitration rule chhara। `detect()` liquidity levels list niye sweep+rejection check kore, `_find_rejection()` — break-er por REJECTION_LOOKBACK candle-er modhye strong reversal khoje। Instantiate korle `DeprecationWarning` dey — new integration-e `stop_hunt_signal_engine.py` babohar korte bola hoyeche।

### `stop_hunt_signal_engine.py` — `StopHuntSignalEngine`
Newer, **spec-compliant, zone-driven** stop-hunt + trade-signal engine, ATR-adaptive rejection scoring soho। Pipeline: S/R zones detect kore → protita zone-e stop-hunt signature khoje (`_check_zone_for_stop_hunt`) → `_find_reversal_confirmation()` → `_check_equal_highs_lows()` (round-number + equal-level confluence) → `_generate_signal()` — full R:R sanity check soho signal dey, `_compute_tp()` opposite-side zone theke TP ber kore, explicit `RR_REJECT_FLOOR` diye bad R:R signal reject kore। No-trade case-e structured `_no_trade_result()`।

### `ict_amd_signal_engine.py` — `ICTAMDSignalEngine`
Full **6-step ICT/SMC AMD + FVG + MSS** signal pipeline, session-boundary constant soho (Asian/London/NY): (1) all S/R zones + strongest/weakest, (2) Asian accumulation range validate, (3) London manipulation (stop-hunt) detect, (4) manipulation move-er modhye FVG khoje, (5) MSS (Market Structure Shift) confirm kore, (6) final signal — **R:R ≥ 1:6 filter** soho। `_strength_to_confidence()` — confluence factor niye Low/Medium/High confidence map kore।

### `amd_strategy.py` — `AMDStrategy`
Simplified/standalone **Accumulation-Manipulation-Distribution** (3-phase session cycle) strategy। `_detect_amd()` — tight range (ATR < 0.7× median) 3+ ghonta thakle Accumulation, range break+reverse hole Manipulation, tarpor strong opposite move-e Distribution। `analyze()` `UnifiedSignal` return kore। *(Note: `extended_modules_adapter.py`-r comment onujayi ei module ekhon dead code — `ict_amd_signal_engine.py` diye superseded, stricter spec soho।)*

### `auction_market_theory.py` — `analyze_auction_market()` function
**Auction Market Theory (AMT)**: Initial Balance (IB, first-hour range), Value Area (VA, 70% trading occurred), Value Area Rotation, Opening Auction, Acceptance vs Rejection, Single Prints (low-volume price levels), Excess High/Low (tails beyond VA)। Output: `price_location` (ABOVE_VA/INSIDE_VA/BELOW_VA/IB_HIGH/IB_LOW), `acceptance` bool, `single_prints` list, `signal`/`score`/`reason`।

### `flip_zones.py` — `FlipZoneDetector`
Boi 5 (Frank Miller S&D) Chapter 8 — **flip zone / role-reversal** state machine। Demand zone (support) confirmed-close diye break hole → SUPPLY zone-e reclassify hoy (future retest = sell opportunity), ulta-o true। "Confirmed break" mane wick na, **candle CLOSE** distal line periye jete hobe। `register_zone()`/`update()` diye zone lifecycle track kore (`ZoneState`: active/flipped/invalidated), `update(df)` protita new candle scan kore, flip hole `FlipZoneEvent` emit kore। `get_active_zones()`/`get_flipped_zones()`/`get_events()` query interface।

### `odd_enhancers.py` — `OddEnhancerScorer` + `TierBEntryStateMachine`
Boi 5 Chapter 6 — sobcheye **quantitative-heavy scoring system**। 4-ta compulsory enhancer (each 0-3 normalized, actual max varies: Strength 0-2, Time-at-Zone 0-2, Freshness 0-3, R:R 0-3 — total max **10**, book-er worked-example onujayi resolved) + 2-ta optional (Original Zone, Overlapping Zones — multi-TF confluence)। Decision tier: **10 = full conviction** (limit order), **7-9 (no zero enhancer) = conditional** (2 entry tactics), **<7 or any enhancer=0 = SKIP**। `score_zone()` full scoring pipeline, `_score_strength_of_move/_score_time_at_zone/_score_freshness/_score_risk_reward` protita enhancer scoring, `_check_original_zone/_check_overlapping/_check_pa_confluence` optional booster check। `TierBEntryStateMachine` — Tier-B (7-9 score) er 2-ta entry method implement kore: **Market Order** (zone-e close hole enter) o **Confirmation Order** (reversal momentum candle wait kore, lower-risk)।

---

## 🏗️ 4. Market Structure & Regime

### `structure.py` — `MarketStructureEngine`
Full market structure pipeline: `_find_swing_points()` (fractal-style, duipashe window), `_label_swings()` (HH/HL/LH/LL labeling), `_determine_structure()` (overall bias), `_detect_bos()` (Break of Structure — bullish: close > last confirmed swing high), `_bos_confidence()` (ATR-normalized break decisiveness), `_detect_choch()` (Change of Character — trend reversal signal), `_detect_displacement()` (choto candle-er pore ekta boro impulsive candle), `_detect_trend_phase()` (Impulsive vs Retracement phase, Candlestick Bible P54-55)।

### `structure_mtf.py` — `MTFStructureEngine`
**Internal vs External structure** analyzer — duita alada timeframe (e.g. H4 external, M15 internal) niye `analyze(df_external, df_internal)` chalay। `_detect_conflict()` — external o internal bias biporit hole conflict, `_alignment()` (ALIGNED/CONFLICT), `_combined_bias()` — external dominate kore kintu conflict thakle NEUTRAL, `_trading_permission()` — TRADE_ALLOWED (aligned) ba caution/block।

### `market_structure.py` — `MarketStructure`
Lean, **causal-only** structure engine (`analyze(df, strength)` — closed bars only)। `_detect_fractals()` diye swing high/low, output-e `events` list (BOS/CHOCH with broke_index/broke_price)। `structure_break_between()` — kono range-e most recent BOS/CHoCH khoje। `liquidity_sweep_before()` — ekta index-er age kono level wick diye sweep hoyeche ki na check kore।

### `market_regime.py` — `MarketRegimeDetector`
**4-dimension regime classifier**: (1) REGIME (Trending/Ranging/Breakout, ADX+structure diye), (2) DIRECTION (Bullish/Bearish/Neutral, EMA+MA alignment), (3) STRENGTH (Strong/Moderate/Weak, ADX value), (4) VOLATILITY (High/Normal/Low, ATR vs historical avg)। `_add_adx()` — nijer ADX calculation kore। `_suggest_strategy()` — 4-ta dimension mile "ekhon kon strategy apply kora uchit" bole দেয়।

### `market_bias.py` — `MarketBiasEngine`
সব input (indicator context, pattern context, S/R context, MTF bias, Fib context) ekসাথে niye final **bias + confidence(0-100%) + conflict warnings + recommendation** dey (`analyze()`)। Ekটা synthesis/aggregation layer, নিজে kono raw calculation kore na।

### `market_state_memory.py` — `MarketStateMemory`
**Rolling memory of market behavior** — normal trade-journal theke different, ei ta CURRENT market character-er upor influence kore, not historical performance in general। `record_trade(strategy, result, reason)` diye last 20 trade track kore (`MAX_RECENT_TRADES`), `get_market_character()` — market-er "character" bole (e.g. `FAKE_BREAKOUT_HEAVY`), `get_strategy_recommendation()` — recent behavior onujayi best strategy suggest kore, `should_skip_trading()` — recent performance kharap hole trading bondho korte bole। JSON file-e persist/load hoy (`MEMORY_PATH`, atomic write)।

### `trendline_engine.py` — `TrendlineEngine`
Book "The Only Technical Analysis Book You Will Ever Need" (Brian Hale) Page 63-66 reference। `_find_swings()` → `_fit_trendline()` (least-squares linear regression swing point-e fit kore) → `_detect_channel()` (parallel up+down trendline diye channel zone khoje) → `_generate_signals()` (pullback-entry-at-trendline signal)। Touchpoint count = trendline strength indicator।

### `window_module.py` — Nison-style **Window (gap)** detector, function-based
`detect_windows(df)` — SHADOW overlap (wick, real-body na) diye rising/falling window detect kore, size irrelevant। Protita window bar-by-bar forward check hoy void condition-er jonno (close diye boundary periye gele void hoy)। `active_window_bias()` — most recent not-yet-voided window-er bias return kore (author-er explicit decision: purono "3 windows = exhaustion" heuristic implement kora hoy nai)। `window_overrides_candle_signal()` — meta priority rule: active window support/resistance always ekta opposing single-candle signal-ke override kore (e.g. bullish hammer ekta open falling window-er bhitore thakle seta ekhono bearish context)।

### `microstructure.py` — `MicrostructureEngine`
Day 97 — **tick-level** market microstructure analyzer (MT5-native, live tick data lagbe)। 4-ta analysis: (1) `_analyze_tick_speed` — tick arrival speed classify (sudden burst = institutional activity), (2) `_analyze_spread` — spread expansion detect (widening = low liquidity/news), (3) `_analyze_volume` — tick volume burst detect, (4) `_analyze_acceleration` — price acceleration (pips/sec) diye displacement measure। `_fallback_result()` — MT5 unavailable hole neutral result dey (trade block kore na)।

---

## 📊 5. Support/Resistance & Zones

### `support_resistance.py` — `SupportResistance` (v2, Zone-Based)
Sobcheye heavily-audited module — protita method-e explicit **evidence-based fix comment** ache। `find_swing_highs/find_swing_lows()` → `cluster_into_zones()` (nearby swing prices-ke box/range-e cluster kore) → `_build_zone()` (protita zone-er `zone_top/zone_bottom/touches/strength/role/last_touch_time/distance_pips` output banay)। `_classify_strength()`: 2 touch = Weak, 3 = Medium, 4+ = Strong। `_is_valid_rejection()`/`_count_valid_rejections()` — rejection candle validate kore। `_raw_swing_levels()` — TRENDING regime-e fallback liquidity source (recent swing-ke thin zone banay)। `_detect_role_reversal()` — Page 25 rule: purono resistance ekhon support (ba ulta) kina check kore। **Evidence-based fixes documented in code**: walk-forward backtest-e EQH/EQL-sourced zone 71.2% bounce rate dekhiyeche (vs 56.1% standard cluster, 53.5% raw swing — statistically significant, p<0.01) → EQH/EQL zone-er strength upgrade kora hoyeche; zone width-o strongest predictor pawa geche (thin zone 61.8% breakout vs wide zone 23.6%, p<1e-77) → thin-zone downgrade logic add kora hoyeche। `to_json()`/`to_prompt_text()` — LLM-ready output। One-shot helper: `detect_zones_for_llm()`।

### `support_resistance.py.bak2` — Backup file
`support_resistance.py`-r **older version**, active codebase-er part na। Diff kore dekha geche: notun version-e logger fix (duplicate `getLogger` call remove kora), distance-normalization bug fix (`dist_to_nearest` calculation-e cluster_min/cluster_max-er against thik vabe normalize kora), ebong upore-bola **evidence-based EQH/EQL strength upgrade + thin-zone downgrade** logic — ei duita major statistical-fix ei bak2 version-e nei। Reference/rollback purpose-e rakha ache।

### `supply_demand_zones.py` — `SupplyDemandZones`
Candlestick Bible-based — S/R theke stronger karon institutional order flow reflect kore। 3-ta quality criterion: (1) move-away speed/strength, (2) favorable R:R, (3) higher-timeframe zone most significant। `detect()` full pipeline। `is_erc()` — ERC (Extended-Range Candlestick, Page 35) check kore, `count_ercs()`/`has_valid_impulse()` — kompokkhe 2-ta ERC thakle genuine imbalance। 3-ta zone-drawing risk-level: `draw_zone_medium_risk`/`draw_zone_high_risk`/`draw_zone_low_risk` (Page 45-48)। `calculate_entry_stop_tp()` — Page 50 order-placement formula। `check_zone_freshness()`/`check_zone_originality()` (Page 57-60), `score_zone_quality()` — comprehensive Ch.5 scoring। `is_staircase_pattern`/`is_doji_only_base`/`has_long_tailed_candles` — zone disqualification check (Page 42-43), `is_zone_tradable()` — Ch.4 overall tradability gate। `_detect_balance_imbalance()` — Ch.2 balance/imbalance cycle। Thread-safe singleton: `get_supply_demand_zones()`।

### `price_target_module.py` — Function-based price-target calculator
`box_breakout_target()` — box height projection breakout direction-e। `swing_target()` — leg A→B impulse-er height leg C (pullback low)-theke apply kore। `flag_pennant_target()` — flagpole height flag-er opposite boundary theke project kore (conservative)। `triangle_target()` — triangle-er widest point theke horizontal line birat measured move। `spring_upthrust_target()` — Wyckoff spring/upthrust-er জন্য trading-range height reuse kore।

---

## 📈 6. Technical Indicators

### `fibonacci.py` — `FibonacciEngine`
**Sobcheye complex indicator module** — full AI-powered Fibonacci analysis pipeline। `find_swing_points()` auto swing detect, `calculate_retracement()`/`calculate_extension()`/`calculate_expansion()` (ABC 3-point) — standard Fib level set (`FIB_RETRACEMENT_LEVELS`, `FIB_EXTENSION_LEVELS`, `LEVEL_SET_PRESETS`)। `detect_flip_zones()` — Fib support/resistance polarity flip। `find_confluence()` — Fib level + S/R + onno indicator level-er kache-kachi thakle confluence zone banay। `find_multi_swings()`/`detect_clusters()` — protita valid swing theke retracement/extension grid overlay kore cluster khoje। `_generate_signal()` — position + confluence + indicator + HTF + flip-zone + MA + ATR + cluster + swing-age mile signal dey, onek configurable filter ache constructor-e (`require_trigger_candle`, `min_rr`, `require_htf_alignment`, `min_confluence_strength`, `require_volume_confirmation`, `require_liquidity_alignment` ইত্যাদি)। `_detect_trigger_candle_pattern()` — confirmation-candle check, `_detect_liquidity_sweep()` — internal liquidity sweep pattern, `_detect_failure_risk()` — level fail howar risk detect kore। `get_memory_record()` — outcome tracking-er jonno database record banay।

### `backtest_fibonacci.py` — `FibonacciBacktester` + `BacktestConfig`
`fibonacci.py`-strategy-r **MT5-integrated backtester**। `connect_mt5()`/`fetch_closed_bars()` — MT5 theke data pull kore, **shudhu closed bar** (currently-forming bar always drop kora hoy — look-ahead bug prevent kora)। `BacktestConfig` — spread_pips, slippage_pips, min_confidence, max_holding_bars (force-close), risk_per_trade_pct externalized (magic number nei)। `run()` — full simulate loop, `_check_exit()` SL/TP touch check kore, `_finalize_pnl()` — realistic cost (spread+slippage) soho P&L calculate kore। `build_report()` diye summary stats।

### `ichimoku.py` — `IchimokuEngine`
Standard 9/26/52/26 parameter Ichimoku Cloud engine। `_donchian_mid()` — core calc (highest_high+lowest_low)/2। `_cloud_position()`, `_tk_cross()` (Tenkan/Kijun cross), `_chikou_clear()` (Chikou Span — current close, `displacement` bar back-e plot kora, clear ache ki na)। `_assess_trend()` sob signal combine kore 0-100 strength dey, `_signal()` — Strong BUY = bullish trend + cloud above + chikou clear।

### `supermao_ichimoku.py` — function `compute(df)`
MQL4 EA theke ported **alternate** Ichimoku variant, Donchian midpoint babohar kore। MQL4 default parameter (9/26/52/26, TP/SL 5%) preserve kora hoyeche। Distinct implementation from `ichimoku.py`।

### `supertrend.py` — `compute(df)`
Classic **Supertrend** indicator, MQL5 default (ATR period 10, multiplier 3.0)। `supertrend`, `st_trend`, `st_color` column output kore।

### `chandelier_exit.py` — `compute(df)`
**Chandelier Exit** — Heikin-Ashi smoothing (`_heikin_ashi`) + Wilder's RMA (`_atr_rma`) ATR-based। MQL5 default (ATR period 1, multiplier 0.75)। `ce_long_stop`/`ce_short_stop`/`ce_dir`/`ce_buy_signal`/`ce_sell_signal` output kore — trailing-exit tool।

### `andean_oscillator.py` — `compute(df)`
**Andean Oscillator** — EMA-based (length 50, signal 9, MQL5 default)। `ao_bull`/`ao_bear`/`ao_signal`/`ao_phase` output।

### `nadaraya_watson_envelope.py` — `compute(df)`
**Gaussian kernel regression envelope** (`_gauss()` helper)। MQL5 default: bandwidth 8.0, multiplier 3.0, window 500। `nwe_mid`/`nwe_upper`/`nwe_lower`/`nwe_pos` output — window-size shesh bar-guloter jonno cap kora hoy jate NaN na hoy।

### `utbot_alerts.py` — `compute(df)`
**UT Bot Alerts** — ATR-based trailing stop signal, MQL5 default (atr_coef 2.0, atr_len 1)। `ut_trail`/`ut_bull_arrow`/`ut_bear_arrow`/`ut_signal` output।

### `vw_macd.py` — Volume-Weighted MACD
`_vwma()` — Volume-Weighted MA helper। `compute()` — standard 12/26/9 MACD kintu VWMA babohar kore। Volume column auto-detect kore (`real_volume` → `tick_volume` → `volume`, priority order-e)। `vwmacd`/`vwmacd_signal`/`vwmacd_hist`/`vwmacd_cross` output।

### `supermao_bands.py` — `compute(df)`
SuperMao.mq4 EA theke ported — multi-band Bollinger + MACD combined signal (default: AvgPeriod 50, 3-ta BBand, MACD 24/52/18)। TP/SL column-o output kore kintu ei ta full trade plan, sizing risk-engine-er kaj।

### `crossover_signals.py` — function-based crossover detection
`cross_above()`/`cross_below()` — 2-bar crossover detect kore, optional `confirmation` (current bar-o cross-side confirm korte hobe, fake crossover filter korte)। `golden_cross()`/`death_cross()` — MA crossover wrapper। `Cruzamentos` class — Portuguese backward-compat alias (`cruzar_acima`/`cruzar_abaixo`)।

### `divergence.py` — `DivergenceEngine`
**Price vs RSI/MACD divergence** detector। `_find_pivots()` — fractal pivot khoje। `_detect_divergence_pair()` — duita same-kind pivot compare kore Regular/Hidden divergence detect kore। `_score_divergence()` — 0-100 score dey। `_reversal_risk()` — recent Regular divergence high score hole HIGH risk। `_trend_continuation()` — Hidden divergence thakle trend-continuation signal।

### `cci_state_machine.py` — `CCIStateMachine`
Boi 5 Ch.11 (Page 120-125) — CCI (Commodity Channel Index) **entry/add/exit state machine**, explicitly **confluence layer** (standalone signal na, book warns against it)। Entry: demand zone-e CCI<-100 (long), supply zone-e CCI>+100 (short)। Exit: CCI retrace back through ±100। Add-to-position: CCI direction trend-confirm korle। Ambiguous zone (CCI near zero) — action recommend kore na। `evaluate()` — full state-machine check, `diagnose_zone_failure()` — retrospective failure-analysis।

### `adx_filters.py` — `adx_rising()` + `bishop_exit()` (function-based)
`adx_rising()` — ADX strictly barche ki na check kore (flat = not rising)। `bishop_exit()` — **"The Bishop"** exit-only signal: ADX prior bar-e `arm_level` (default 40)-er upore chilo, current bar-e down-tick — ekhon exit koro। Pure/stateless function, kono hidden state nei — caller-ke nijer bookkeeping korte hoy।

### `adx_trend_filter.py` — `compute()`, `should_trade()`, `get_trend_context()`
Standard **ADX-based trend filter** (period 14, min_adx 20 threshold)। `adx`, `adx_pos`, `adx_neg`, `adx_trend_strength`, `adx_direction`, `adx_filter_pass` output kore। `should_trade()` quick boolean check।

### `oscillator_regime_gate.py` — `OscillatorRegimeGate`
Book Page 33-34 — oscillator (RSI/Stochastic) sudhu range-bound market-e reliable, trending market-e false signal dey। `adjust_signal()` — regime (ADX) onujayi oscillator signal-er weight adjust/suppress kore (e.g. RSI oversold ekta strong DOWNTREND-e = false buy signal → suppressed)। `get_rsi_signal()` — regime-adjusted RSI signal।

### `atr_sl_finder.py` — `compute(df)`
**ATR Stop-Loss Finder**। `causal` parameter — default **True** (age False chilo, ekta breaking-change fix — original MQL5-e SMA window forward-looking chilo, repaint hoto — live trading-e eta hazard)। `atr_sl_upper`/`atr_sl_lower`/`atr_sl_ma` output।

### `daily_high_low.py` — `compute(df)`
Intraday DataFrame-e **daily high/low level** compute kore। `previous=True` (default) — PREVIOUS din-er level babohar kore (repaint-free); `price_mode`: lowhigh/openclose/closeclose। `dhl_high`/`dhl_low` output।

### `volume_confirmation.py` — `VolumeConfirmation`
Book Page 6/9 — breakout **volume diye confirm** korte hobe, na hole fake breakout hote pare। `check_breakout()` — breakout level + volume ratio check kore (confirmed/adjustment score)। `check_trend_confirmation()` — Page 27 Volume/Price Trend Confirmation rule।

### `volume_profile.py` — `VolumeProfileEngine`
Price-binned **volume distribution** analyzer। `_build_profile()` — price range-ke `num_bins`-e ভাগ kore protita bin-e koto candle porche seta count kore, POC (Point of Control) ber kore। `_value_area()` — POC theke shuru kore upor-nichey extend kore 70% volume cover kora range ber kore (Value Area High/Low)। `_find_zones()` — HVN (High Volume Node, top-25% percentile) o LVN (Low Volume Node) khoje। `_bias()`/`_signal()` — price VA-r upore/niche/modhye onujayi bullish/bearish bias।

---

## ⏱️ 7. Multi-Timeframe Analysis

### `mtf_analyzer.py` — `MTFAnalyzer`
Professional top-down: **H4 trend → H1 zone → M15 setup → M5 entry** (`MTF_CHAIN`, `TF_WEIGHTS`)। `_fetch_all_timeframes()` — চারটা TF fetch kore, `_build_tf_contexts()` protita TF-er indicator context banay। `_detect_bos/_detect_choch/_detect_liquidity_sweep()` — protita TF-e market structure event khoje। `_detect_conflicts()` — TF-der modhye direction conflict khoje। `_check_h4_override()` — rule: H4 strongly bullish/bearish hole (`H4_OVERRIDE_THRESHOLD`) lower TF-er signal override kore, WAIT force kore jodi conflict thake। `_calculate_confidence()` — aligned TF-er weight sum kore confidence dey। `_apply_regime_gate()` — `market_regime.py`-r verdict-ke hard gate hisebe enforce kore।

### `timeframe.py` — `MultiTimeframeAnalyzer`
Simpler version — **Daily trend → 4H confirmation → 1H structure → 15M entry** chain। `analyze()` protita TF-e indicator calculate kore, `get_bias()` sob TF-er trend dekhe overall bias bole।

### `curve_mtf.py` — `CurveMTF` / `Curve` / `TradingStyle`
Boi 5 Ch.12 (Page 126-135) — sobcheye **quantitative MTF filter**। "**Curve**" = nearest demand zone (proximal line) o nearest supply zone (proximal line)-er modhyer price range, jeta 3 equal sub-zone-e bhag kora hoy: Low/Equilibrium/High (plus Very Low/Very High = zone-er bhitorei)। Bias rule (Page 133): Very Low/Low → **BUY_ONLY**, Very High/High → **SELL_ONLY**, Equilibrium → TREND_FOLLOW_OR_NO_TRADE। **HTF override** (Page 135): "longer frame always wins" — lower-TF signal shudhu tokhoni actionable jokhon HTF bias-er sathe agree kore। `TradingStyle` enum-e 4-ta style (Scalper/Day/Swing/Position) ache প্রতিটার নিজস্ব timeframe triplet soho (Page 127-129)। `get_timeframe_triplet()`, `Curve.position_of()`/`bias_for()`, `CurveMTF.check_alignment()`/`resolve_conflict()`, `fib_levels_for_curve()` (alternative 33%/66% split method, Page 132)।

---

## 📰 8. Sentiment & Fundamental Data

### `sentiment.py` — `SentimentEngine`
Core market-psychology engine। `retail_positioning()` — retail long/short % diye contrarian signal, `fear_greed()` — Fear & Greed Index diye market emotion, `currency_strength()` — currency strength diye pair bias, `dxy_analysis()` — DXY trend diye USD-linked pair impact। `final_sentiment_score()` — sob source combine kore final score, `detect_conflict()` — technical signal-er sathe sentiment conflict ache ki na check kore।

### `sentiment_data.py` — `SentimentDataProvider`
`sentiment.py`-r জন্য **central data feeder** — `get_all(pair)` ek call-e retail positioning, Fear&Greed, currency strengths, DXY data sob dey। `_compute_fx_native_fg()` — retail positioning theke 0-100 FX-native Fear&Greed Index banay (crypto index-er upor nirbhor kore na)। yfinance babohar kore DXY o currency strength fetch kore, fallback logic soho।

### `retail_sentiment.py` — `RetailSentimentAPI`
**Multi-source fallback chain**: (1) OANDA v20 (API key thakle, most accurate, order book soho), (2) Myfxbook Community Outlook (no key), (3) price-based (RSI) synthetic proxy (last resort — explicitly warned eta REAL retail sentiment na)। `_fetch_position_book()`/`_fetch_order_book()` — OANDA-specific। `_compute_confidence()` — retail যত one-sided, contrarian confidence tত beshi।

### `myfxbook_sentiment.py` — `MyfxbookSentiment`
**Free, no-API-key** retail sentiment scraper (Myfxbook Community Outlook page, BeautifulSoup diye HTML parse)। Contrarian indicator: 80%+ retail long hole smart money usually short, price reverse hoy। **Circuit breaker** implement kora ache — 5 consecutive failure-er por 30-min cooldown-e source disable hoye jay (log spam prevent kore), `cloudscraper` fallback jodi plain `requests` 403 pay (Cloudflare WAF bypass)। `compute_synthetic_sentiment()` — price-action theke synthetic sentiment।

### `risk_sentiment.py` — `RiskSentimentEngine`
**Risk-on/risk-off** macro sentiment engine। `RISK_ON_ASSETS`/`RISK_OFF_ASSETS` niye `_classify_environment()` — S&P/VIX/DXY trend diye risk environment classify kore, `_classify_fear()` VIX value diye fear level, `_preferred_assets()` — environment onujayi konsob asset favor kora uchit bole।

### `news_api_provider.py` — `NewsAPIProvider`
NewsAPI.org integration — **breaking news** (scheduled economic calendar na, real-time headline) pull kore। Free tier: 100 req/day, 1-day delay। `fetch_headlines_for_pair()` — pair-related headline fetch+score kore, `_score_headline()` — keyword-based bullish/bearish classification (simple, LLM-based na, quick-and-dirty but effective risk-window flag korar jonno)। `_extract_currencies()` — pair theke currency code ber kore (EURUSD→[EUR,USD])। `_check_quota()` — daily request limit track kore। `_rss_fallback()` — API unavailable hole free RSS-based fallback।

### `macro_data.py` — `MacroDataProvider`
Global macro asset (indices, commodities, bonds — `GLOBAL_SYMBOLS`) fetch kore (5-min cache TTL soho)। `_fetch_single_symbol()` — protita symbol individually fetch kore (fail-safe, ekta fail korle onnogula thik thake), `_classify_trend()` — % change diye trend classify kore।

### `institutional_flow.py` — `InstitutionalFlowEngine`
**COT (Commitment of Traders)** data theke institutional positioning track kore (CFTC weekly report)। `_fetch_cot_data()`/`_fetch_cot_from_cftc()` — CFTC text report fetch+parse kore। Unavailable hole `_build_synthetic_result()` — large-candle (displacement) analysis diye institutional flow estimate kore। Output-e `retail_vs_inst` field — retail-institution divergence flag kore।

### `currency_strength.py` — `CurrencyStrengthEngine`
Day 64 — **global currency intelligence**। `calculate_strength()` — sob `CROSS_PAIRS` fetch kore protita currency-r raw contribution ber kore। `calculate_momentum()` — history-r sathe compare kore "koto strong hocche" na, "strong howar gati barche na komche" seta measure kore। `rank_currencies()`/`find_best_pairs()` — strength-difference diye best trading opportunity khoje। `multi_timeframe_strength()` — ekই currency multiple TF-e strong/weak kina check kore। `record_trade_outcome()` — currency-strength-based trade result track kore future calibration-er jonno। Process-wide singleton (`get_currency_strength_engine()`) — first caller-er config lock hoye jay।

### `currency_ranker.py` — `CurrencyRanker`
`rank()` — strength dict theke ranking। `find_best_pairs()` — protita currency-pair combination-er strength-difference calculate kore grade dey। `detect_correlation_risk()` — multiple opportunity ekই currency-r upor base korle flag kore। `build_heatmap()` — pairwise strength-difference matrix। `detect_cycle()` — history theke currency-strength cycle detect kore।

### `strength_calculator.py` — `StrengthCalculator`
`currency_strength.py`-r **computation backend**। `compute_pair_score()` — `_price_change_score` + `_trend_score` + `_momentum_score` + `_volatility_adjustment` combine kore protita pair-er base-currency contribution ber kore (quote currency-r jonno caller negate kore)। `normalize_scores()` — min-max normalize kore 0-100 scale-e anay।

### `correlation_engine.py` — `CorrelationEngine`
Day 96 — **correlation + volatility risk** engine, position sizing/hidden-risk detection-er jonno। `analyze()` — pair + open_pairs niye correlation risk + volatility check kore। `build_matrix()` — multiple pair-er correlation matrix banay। `_compute_live_correlation()` — live Pearson correlation try kore, fail korle `_get_static_correlation()` (pre-defined `STATIC_CORRELATIONS` table) fallback।

### `intermarket.py` — `IntermarketEngine`
Cross-asset macro analysis। `fetch_global_data()`/`calculate_correlations()`/`detect_market_regime()` → `generate_macro_bias()` — protita currency-r BUY/SELL bias ber kore (USD, DXY trend, preferred/avoid asset list diye), `_resolve_pair_bias()` base+quote currency bias combine kore। `_calculate_macro_score()` — unified 0-100 Macro Score। `_cross_asset_confirmation()` — USD bias nite hole confluence lagbe (doc-onujayi)। `_event_risk_integration()` — FOMC/high-impact news-er shomoy confidence komay। `fuse_with_smc()` — Macro + SMC + Session context fuse kore final decision। History persist/load kore (`MACRO_MEMORY_PATH`)।

---

## 🕒 9. Session & Timing

### `session_analysis.py` — `LondonManipulationDetector` (renamed from `SessionAnalyzer`)
**Institutional-review renamed class** — age `SessionAnalyzer` name-e chilo jeta `session_analyzer.py`-r ekই-name class-er sathe collision korto (silent shadowing hazard — jekono ekta import last-e run hole otai win kore, kono warning chhara)। `detect_london_manipulation()` — Asian range + London session candle dekhe fake-breakout→reversal detect kore। **DST-aware fix**: London open window age hardcoded 07:00-10:00 UTC chilo (summer/BST-e thik, winter/GMT-e wrong — 08:00 UTC howa uchit chilo, false-positive alert dito winter-e)। Ekhon `_is_eu_dst_for_timestamp()` diye protita candle-r jonno correct window compute hoy। Backward-compat: deprecated `SessionAnalyzer` alias rakha ache (DeprecationWarning soho)।

### `session_analyzer.py` — `SessionAnalyzer` (Day 63 — canonical session engine)
**Comprehensive session intelligence**। `get_current_session()` — current GMT time diye active session(s) detect kore। `analyze_session_behavior()` — session-er volatility/behavior/characteristics। `get_strategy_mode()` — session onujayi strategy auto-select। `get_pair_preference()` — pair+session combination priority। `detect_session_transition()` — session transition point kache ki na। `calculate_session_confidence()` — session+pair+SMC+signal mile final confidence। `session_smc_fusion()` — SMC score-er sathe fuse। **DST-aware**: `_is_us_dst()`/`_is_eu_dst()` — US (2nd Sunday March→1st Sunday Nov) o EU (last Sunday March→last Sunday Oct) DST approximate kore। `_is_dead_zone()` — low-liquidity dead hour detect। `record_trade_outcome()`/`get_session_performance()` — pair+session combination-er historical win-rate track kore।

### `session_rules.py` — Pure constant/config module
`SESSION_WINDOWS`, `DEAD_ZONES_ENABLED`, `DEAD_ZONES`, `SESSION_CHARACTERISTICS`, `BASE_MIN_CONFIDENCE`, `SESSION_STRATEGIES`, `LONDON_OPEN_WINDOW`, `BASE_MIN_SMC_SCORE`, `SMC_REQUIREMENTS` — kono class/function nei, ekটা **static rule-definition dictionary set** onno session-related module-guloke driving kore।

### `pair_session_map.py` — Function-based pair-session mapper
`PAIR_SESSION_MAP`, `ALWAYS_MONITOR`, `PAIR_PRIORITY` constant। `get_pair_priority()` — session-specific 0-100 priority score। `get_preferred_pairs()` — ekta session-er jonno preferred pair list। `get_pair_session_recommendation()` — single pair+session-er recommendation label।

### `optimal_trading_time.py` — `find_optimal_trading_time()`
**Cost-recovery based** optimal-hour finder — bid/ask spread vs price-change compare kore koto % bar-e cost "recover" hoy (`coverage_pct`) seta protita UTC hour-er jonno calculate kore। `best_trading_hours()` — top-N highest-coverage hour return kore। CLI entry-point (`_cli()`)-o ache parquet file input-er jonno।

---

## ⚠️ 10. Risk Management

### `risk_management.py` — `RiskManager`, `PositionSizer`, `MarginCallDetector`, `DrawdownSimulator`
Boi 5 Ch.14 (Page 153-157) — full risk system। **Risk per trade**: experienced 2%, beginner 1% (account 3x na hওয়া porjonto)। **Margin call threshold**: `account_loss_pct × leverage ≥ 100%` formula (`MarginCallDetector.is_margin_call()`)। **Position sizing**: `position_size = risk_amount / |entry-stop|` (`PositionSizer.size_for_stock/size_for_forex` — quote-currency ≠ account-currency case-o handle kore)। **Drawdown circuit breaker**: ≥20% drawdown hole trading STOP (mash porjonto), any drawdown-e risk 25% reduce hoy, notun equity high hole restore hoy (`RiskManager.current_risk_pct/update/can_trade`)। `DrawdownSimulator.simulate_losing_streak()` — compounding loss math (`(1-risk%)^n_losses`) — book-er number-er sathe minor discrepancy flag kora ache (rounding methodology difference, code-e exact formula babohar kora hoyeche)।

---

## 🔬 11. Quant / Research Domains

### `quantitative_factors.py`
Function-based: `hurst_exponent()` (R/S method — H<0.5 mean-reverting, H>0.5 trending), `kalman_filter()` (1D adaptive price smoothing), `hidden_markov_regime()` (simplified HMM — Bull/Bear/Volatile state, no external lib), `bayesian_win_probability()` (prior + evidence diye win-prob update), `rolling_zscore()`। `compute_quant_factors()` — sob combine kore signal/score dey।

### `research_domains.py`
3-ta "missing research domain"-er **framework/interface module** — actual external data feed na thakle gracefully "data unavailable" e degrade kore। `get_options_intelligence()` (put/call ratio, gamma exposure, max pain — forex options data free-e available na, tai eta framework-only)। `get_futures_data()`/`detect_cme_gap()` (COT + CME gap)। `compute_correlation_graph()` — asset correlation network graph (nodes/edges/clusters/leading_indicators)। `meta_learning_strategy_update()` — kon strategy kon regime-e valo kaj kore seta shekhe (**min_trades_per_regime raised from 3 to 20** — noise/overfitting prevent korte, 2/3 win looking like strong signal kintu actually coin-flip-range-er modhye)।

### `phd_frontier.py`
7-ta **PhD-level niche domain** class hisebe: `InformationTheory` (Shannon entropy, KL divergence, transfer entropy, mutual information), `ChaosDynamics` (Lyapunov exponent, fractal dimension, chaos-based regime), `AnomalyDetector` (Z-score/IQR outlier, simplified Isolation Forest, volume anomaly), `GameTheory` (zero-sum payoff, Nash equilibrium mixed strategy, adversarial market-maker analysis), `DecisionIntelligence` (expected utility, information-value-of-waiting), `KnowledgeGraph` (market entity BFS relationship graph, e.g. Fed→USD→DXY→Gold), `FederatedLearning` (framework **stub only** — multi-broker gradient sharing, full implementation lagbe)।

### `final_frontier.py`
13-ta **institutional-roadmap domain**, sob standalone utility class — `MarketEcology` (participant-type classification: retail/bank/HFT/MM), `StrategyDecayTracker` (win-rate/Sharpe decline detect), `AlphaAttribution` (P&L source breakdown), `TransactionCostAnalyzer` (commission+spread+slippage+impact full TCA), `LatencyAnalyzer` (signal/order/fill latency), `DataProvenance` (source quality tagging), `MarketCalendar` (holiday/expiry/rebalancing), `RegimeProbability` (soft distribution: {"trending":0.65,...}), `CausalInference` (Granger causality), `DigitalTwin` (virtual account trade simulation), `FailureAnalyzer` (loss-cause classification), `EdgePreservation` (alpha decay + strategy crowding detect)। **Explicit wiring-status note in code**: as of the review, NO other module imports this file — sob reference/utility library, live decision pipeline-e ekhono connected na।

---

## 🧰 12. Filters & Utilities

### `advanced_filters.py`
4-ta function: `detect_elliott_wave()`, `analyze_ema_ribbon()`, `detect_fake_breakout()` (volume threshold soho), `detect_wyckoff_pattern()`।

### `global_filters.py`
`doji_density()` — trailing window-e koto doji/small-body candle ache (rolling count, choppiness indicator)। `doji_weight()` — density barle doji-r significance komiye dey (choppy market-e ekta doji-r importance kom, clean trend-e beshi)। `gap_required_for_star_patterns()` — instrument-type onujayi Morning/Evening Star pattern-e strict gap lagbe (daily equity) na relaxed (forex/index/intraday) seta decide kore।

### `_engine_utils.py`
Shared helper library — `atr_series()`/`atr_value()`, `pip_value(symbol, price)` — instrument-aware pip size resolver (**important bug fix documented**: age `(symbol)` 1-arg signature chilo kintu `supply_demand_zones.py` `(symbol, price)` 2-arg-e call korto, try/except silently swallow kore full Supply/Demand feature disable kore rekhechilo — ekhon optional `price` arg add kore fix kora hoyeche), `is_round_number()` — psychological round-number level check (FX 50/100 pip, JPY, XAUUSD $5/$10, indices), `no_trade_signal()`/`wait_signal()` — standard NO_TRADE/WAIT dict builder, multiple schema soho (default/pa/ict)।

### `follow_through_engine.py` — `FollowThroughEngine`
BOS (Break of Structure)-er **por price genuinely follow-through korche ki na** track kore (fakeout na)। `evaluate_from_bos()`/`evaluate()` — breakout-er por N bar dekhe strong-body-ratio, ATR-ratio, max-pullback-ratio, volume niye score dey, session-weighted (`DEFAULT_SESSION_WEIGHTS`)। `_infer_session()` — self-contained UTC session classifier (orchestrator theke import kore na, standalone rakhar jonno)। Standalone singleton: `get_follow_through_engine()`।

### `shadow_follow_through_logger.py` — `ShadowFollowThroughLogger`
`follow_through_engine.py`-r prediction **shadow-mode-e log+resolve** kore (SQLite DB, `DB_PATH`)। `log_prediction()` — de-duplicate kore (symbol, timeframe, breakout_index) diye — same BOS event multiple cycle-e ekbari log hoy। `resolve_pending_outcomes()` — pending prediction-er actual outcome check kore (`RESOLUTION_HORIZON_BARS`, `MIN_MOVE_FRACTION`)। `summary()` — rollout-readiness count (≥500 threshold reference)।

### `extended_modules_adapter.py`
**Adapter layer** — audit-e pawa 17-ta "imported-only-but-never-called" module ke live signal-fusion pipeline-e wire kore (bullish/bearish/weight/reason vote tuple hisebe, jeta `strategy/signal_engine.py`-r existing scoring model-e directly plug hoy)। 17+5 = 22-ta module wired: `andean_oscillator`, `supertrend`, `utbot_alerts`, `nadaraya_watson_envelope`, `daily_high_low`, `auction_market_theory`, `candlestick_patterns_ml`, `breaker_block`, `flip_zones`, `curve_mtf`, `candlestick_patterns_br`, `candlestick_patterns_mw`, `supermao_ichimoku`, `vw_macd`, `supermao_bands`, `crossover_signals` (golden/death cross), `window_module`, `long_term_patterns`, `cci_state_machine` (confluence-only vote)। **12-ta module deliberately NOT wired** — protita-r karon explicit documented: `atr_sl_finder`/`chandelier_exit` (SL sizing tool, directional signal na), `book_rules_index` (static KB), `research_domains`/`quantitative_factors`/`phd_frontier` (confidence-layer, direction na), `risk_management` (position sizing, double-count risk), `adx_trend_filter`/`adx_filters` (gate, vote na), `engulfing_bar_strategy`/`pin_bar_strategy`/`trend_level_signal` (double-counting risk — already-wired module same pattern detect kore), `megaphone_pennant` (design onujayi directional na)। `get_extended_votes()`/`get_zone_dependent_votes()`/`merge_zone_votes_into_signal()`/`apply_extended_votes()` — main entry point functions, `analysis_agent.py`/`strategy/signal_engine.py` diye call hoy। Protita wrapper defensively try/except-e wrapped, ekta module fail korle pipeline crash kore na।

### `volatility.py` — `VolatilityEngine`
**Bollinger Squeeze + ATR regime + breakout release** detector। `_add_bollinger()`/`_add_atr()` — existing column reuse kore jodi thake (`indicators.add_all()` age run kora thakle)। `_percentile()` — value-r historical distribution-e percentile ber kore। `_squeeze_strength()` — lower percentile = stronger squeeze। `_detect_release()` — squeeze breakout kore BB band periye gele release detect kore। `_expansion_probability()` — squeeze theke expansion আসার probability (0-100)।

---

## 🗄️ 13. Package Files

### `__init__.py`
`analysis/` package-er init file — **empty** (kono re-export/config nei, plain namespace marker)।

### `database/__init__.py`
`analysis/database/` subpackage-er init file — **empty**।

---

## 📌 Notes / General Observations (code পড়ে dekha gechey)

- **Code quality/audit culture**: onek module-e explicit "institutional review", "audit fix", "Round-N fix", "Phase-N fix" comment ache — mane ei codebase-e regular internal code-review process cholche, ebong protita fix-er karon (bug, evidence, backward-compatibility concern) detailed likha thake।
- **Evidence-based tuning**: `support_resistance.py`, `liquidity_engine.py`-r moto module-e actual walk-forward backtest statistics (sample size, p-value soho) comment-e cite kora hoyeche design decision justify korte — eta shudhu heuristic na, data-driven।
- **No-lookahead / no-repaint discipline**: bohu module-e explicit contract documented — `order_block.py` ("closed bars only"), `liquidity_zones.py` (expanding-window backtest contract), `atr_sl_finder.py` (`causal=True` default), `candlestick_patterns_mw.py` (`csp_confirmed` last-row-e always pending) — repainting/look-ahead bias systematically avoid korar culture ache।
- **Duplicate/naming-collision fixes**: `session_analysis.py`-r `SessionAnalyzer` class `session_analyzer.py`-r shathe collide korchilo — rename kore fix kora hoyeche, deprecated alias soho।
- **Dead-code wiring**: `extended_modules_adapter.py` ekta specific audit-e "implemented kintu kono-i call hoy na" emon module-guloke systematically live pipeline-e wire kore, defensive try/except soho — kono module-i pipeline crash korte pare na।
- Onek module-e Bengali/mixed-language (Bangla+English) comment/docstring ache — original author-er coding style।
- Kichu module "Day XX" ba "Book 5 (Frank Miller S&D) Chapter XX" / "The Only Technical Analysis Book You Will Ever Need (Brian Hale)" reference kore — ei module-gulo ekta structured trading-course/book-er direct code implementation।
- `support_resistance.py.bak2` — purono backup, ekhono keno rakha ache seta explain kora holo upore (missing evidence-based fixes)।
- Ei README ta shob 101-ta `.py` file **directly Python AST parse kore** (module docstring + class + method + docstring + constant level porjonto) ebong prottekta file-er full context read kore generate kora hoyeche — ekTao module sudhu filename dekhe guess kora hoy nai।