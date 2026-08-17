# Deliverable 1 — System Understanding Report

**Project:** Forex AI Autonomous Trading System (forex-agent)
**Repo:** `/home/z/my-project/download/forex-agent` (~1.6 GB, 2,006 files, 703 Python files)
**Audit date:** 2026-08-17
**Auditor:** Multi-agent forensic audit (4 parallel Explore sub-agents + main agent)
**Forensic source:** `/home/z/my-project/upload/Forex_AI_Forensic_README.md` (used as cross-reference only — code verified independently)

---

## 1. Executive Summary

The system is a **production-style multi-agent autonomous forex trading system**. It boots a service graph of ~80 modules through a strict phased lifecycle, then runs a continuous per-symbol trading loop where every cycle: fetches MT5 market data, runs 17 analysis engines, fuses 4 decision layers (Rule + ML + RL + LLM) into a master signal, gates through 21 sequential risk/permission checks, and finally submits an order to MT5 via `broker/order_manager.py:430` (`mt5.order_send`).

**Backtest parity is structurally guaranteed** by sharing the same `AITrader.evaluate_decision_core()` between live and backtest modes — only the data source (`LiveMT5Provider` vs `HistoricalCSVProvider`) and the fill mechanism (`OrderManager` vs `BrokerSimulator`) differ.

However, **5 critical parity gaps** in the data layer mean that a backtest cannot exactly reproduce live trading without CSV enrichment (see Deliverable 3).

---

## 2. Pipeline (Actual, Verified from Code)

The pipeline below is the **verified runtime call chain** (not the README's nominal flow). File:line markers are from the actual source.

```
main.py main()  →  ForexAISystem.start_trading()                              [main.py:343]
  ↓
AutonomousTraderSystem.run()                                                  [core/trader.py:4120]
  for symbol in active_symbols:
    ↓
    AITrader.run_cycle()                                                       [core/trader.py:1722]
      Stage 0:   HumanOverride check (PAUSED/STOPPED → bail)                   [:1771]
      Stage 0.5: Equity-stop (mt5_demo only)                                  [:1822]
      Stage 0.7: TradeFrequencyController.can_trade_now()                     [:1884]
      Stage 1:   LiveMT5Provider.get_market_out(symbol, tf)                    [:1913]
                    ↓
                    MarketAgent.run()                                          [agents/market_agent.py:156]
                      ├ MultiTimeframeAnalyzer.analyze(["1d","4h","1h","15m"]) [:171]
                      │   → fetch_ohlcv per TF (limit=300) → MT5.copy_rates_from_pos
                      ├ DataValidator().validate(df)                          [:203]
                      ├ add_canonical_indicators(df)                          [data/indicator_registry.py:103]
                      │   → ExtendedIndicators (pandas-ta, 139 cols)
                      │   → legacy Indicators (ta-lib wrapper)
                      └ MarketRegimeDetector.detect(df)                       [:301]
      Stage 1b:  PaperTrader.update_price() + PositionManager.poll_once()      [:1940]
      Stage 2:   CircuitBreaker.allow_trade() → hard block if tripped          [:2025]
      Stage 2b:  production_hardening.check_data_staleness() + is_candle_closed()
      Stage 3:   evaluate_decision_core()                                      [core/trader.py:895]
                    ↓
                    [3/9] AnalysisAgent.run(market_out)                        [agents/analysis_agent.py:192]
                              17 analyzers in order:
                              - SessionAnalyzer (Step 0)
                              - PatternDetector (Step 1)
                              - SupportResistance (Step 2)
                              - LiquidityEngine (Step 2b)
                              - AdvancedPatternDetector (Step 3)
                              - IntermarketEngine (Step 4)
                              - CurrencyStrengthEngine (Step 6.5, DISABLED)
                              - SentimentEngine (Step 7)
                              - SMCEngine (Step 8) — self-fetches H4 + M15
                              - MarketStructureEngine (Step 8.1)
                              - DivergenceEngine (Step 8.2)
                              - VolatilityEngine (Step 8.4)
                              - MTFStructureEngine (Step 8.95) — caller fetches H4
                              - CorrelationEngine (Step 8.95) — fetches 28 cross pairs
                              - InstitutionalFlowEngine (Step 8.96) — fetches COT
                              - MicrostructureEngine (Step 8.975) — fetches MT5 ticks
                              - VolumeProfileEngine (DISABLED 2026-07-30)
                    ↓
                    [4/9] DecisionAgent.decide(market_out, analysis, placeholder)
                              ├ SignalEngine.generate (rule layer, 30% weight)
                              ├ ML ensemble (30% weight)  — DISABLED in live code
                              ├ RLAgent (20% weight)     — see §6
                              ├ LLM Analyst (20% weight) — Groq → Gemini → OpenRouter
                              └ MasterDecisionEngine.decide()  →  SignalFusion.fuse()
                                                              → 4-layer majority vote
                                                              →  opposition penalty
                                                              →  ConfidenceManager.adjust_weights
                    ↓
                    [5/9] RiskEngine.evaluate(signal, entry, atr, regime, correlation)
                              ├ atr × vol_mult × instrument_mult → SL distance
                              ├ balance × MAX_RISK_PC / (sl_pips × pip_value) → lot
                              ├ correlation_adjustment (0.25–1.0)
                              └ PositionSizer override (kelly × vol × conf × corr × dd × loss-streak)
                    ↓
                    [6/9] TradePermission.check(dec_out, risk_out, news_ctx, session_ctx, ...)
                              10 sequential gates:
                              1. Valid signal (BUY/SELL)
                              2. S/R zone alignment
                              3. Trend alignment (regime)
                              4. MTF trend alignment (H4/H1/M15)
                              5. Zone cooldown
                              6. Risk approved
                              7. News safe (fail-safe default DENY)
                              8. Cost-aware EV gate
                              9. Min confidence (adaptive loss-streak bump)
                              10. Session quality / Confluence quality / Min R:R / SMC+Session / LLM availability
                    ↓
                    Post-permission gates:
                    P1. signal_persistence.is_stable()
                    P2. regime_suppression.should_suppress()
                    P3. PaperTrader.has_open_position() (duplicate)
                    P4. CorrelationFilter.allow()
                    P5. final_decision_gate() (watchdog + network + ML drift)
      Stage 4:   LearningAgent.save_decision()
      Stage 5:   _build_result()
      Stage 6:   [8/9] DevilsAdvocateGate.review() — HARD VETO (LLM counter-thesis)
      Stage 7:   ApprovalMode.process() — mode 2 pends, mode 1/3 default deny
      Stage 8:   [9/9] MT5ExecutionAdapter.open_trade()
                    ↓
                    ExecutionRouter.execute()
                      ├ HARD GATE 1: decision ∈ {BUY,SELL}
                      ├ HARD GATE 2: trade_allowed=True
                      ├ HARD GATE 3: lot ≤ MAX_LOT
                      └ HARD GATE 4: lot > 0
                    ↓
                    _execute_mt5_demo()
                      ├ _ensure_mt5_connected()
                      ├ _check_absolute_safety(symbol)
                      ├ AccountManager.trading_permission(symbol)
                      ├ OrderManager._pre_trade_validate(symbol, dir, lot, sl, tp)
                      │   ├ mt5.terminal_info() — Algo Trading ON?
                      │   ├ mt5.symbol_info(broker_symbol)
                      │   └ mt5.symbol_select(broker_symbol, True)
                      ├ mt5.symbol_info_tick(broker_symbol)  — spread check + fill price
                      ├ mt5.symbol_info(broker_symbol)       — pip size
                      ├ mt5.account_info()                   — free margin check
                      ├ mt5.positions_get(symbol=...)       — pre-trade snapshot
                      ├ _resolve_filling_mode()              — IOC/FOK/RETURN
                      └ ★ mt5.order_send(request) ★         [broker/order_manager.py:430]
                            request = {action: TRADE_ACTION_DEAL, symbol, volume, type,
                                       price: round(tick.ask/bid, digits), sl, tp,
                                       deviation: 10, magic: 424242, comment,
                                       type_time: ORDER_TIME_GTC, type_filling}
                            result.retcode == TRADE_RETCODE_DONE (10009) → success
                            _confirm_position_appeared(ticket)  → polls positions_get for 2s
```

**Total sequential gates between "analysis says BUY" and "broker receives order_send": 21** (10 in `TradePermission.check`, 5 post-permission, 6 in `ExecutionRouter` / `OrderManager`).

---

## 3. Execution Mode Switching

`execution/execution_router.py:_init_mode` (line 163) selects backend based on `EXECUTION_MODE` env / config:

| Mode | Condition | Path |
|------|-----------|------|
| `simulation` | `SIMULATION_MODE=true` | `SimulatedExecutor` (no MT5) |
| `backtest` | `mode=backtest` | Router inert; `backtest.unified_engine` drives fills via `BrokerSimulator` + `HistoricalExecutionAdapter` |
| `mt5_demo` (default) | MT5 reachable | `MT5Connection` + `AccountManager` + `OrderManager` + `JournalBridge` |
| `mt5_demo` (fallback) | MT5 unreachable + `MT5_FALLBACK_TO_SIMULATION=true` | Falls back to `SimulatedExecutor` silently |
| `mt5_live` | `ALLOW_REAL_MONEY_TRADING=true` + real creds set | Same as demo but real credentials, **NO silent fallback** |
| `mt5_live` (missing opt-in) | `ALLOW_REAL_MONEY_TRADING=false` | `raise RuntimeError` — refuses to start |

---

## 4. Backtest Mode — Parity Contract

`backtest/unified_engine.py:run_unified_backtest` (line 293) is the historical entry point.

```
python main.py --mode backtest
  → _run_backtest(args)                                         [main.py:687]
      for symbol in pairs:
        csv_path = data/{symbol}_{timeframe}.csv
        df = HistoricalDataLoader().load_csv(file_path=csv_path, pair=symbol, timeframe=timeframe)
        result = run_unified_backtest(symbol, df, timeframe, ...)

  run_unified_backtest():
    trader   = _make_backtest_trader(symbol, timeframe, ...)    # AITrader(execution_mode="backtest")
    broker   = BrokerSimulator(...)                              # Simulated fills
    adapter  = HistoricalExecutionAdapter(broker)
    provider = make_backtest_provider(symbol, timeframe, df, prefer="auto")
                  → HistoricalCSVProvider if data/{SYMBOL}_{TF}.csv exists
                  → HistoricalMT5Provider(df, symbol, tf) as fallback
    set_backtest_mode(True)
    reset_backtest_memory()

    for i in range(warmup_bars, total_bars):
      # Exit check on open trades (uses bar high/low sweep — BrokerSimulator)
      for trade in open_trades: broker.check_exit(trade, df.iloc[i]["high"], df.iloc[i]["low"], df.iloc[i]["close"], current_time)

      provider.advance_to(i)                                    # Move replay cursor
      market_out = provider.get_market_out(symbol, timeframe)   # SAME shape as LiveMT5Provider
      session_ctx = {"current_session": "BACKTEST", "gmt_time": str(current_time), "session_strategy": "n/a"}

      core = trader.evaluate_decision_core(market_out, session_ctx, bypass_checks=bypass_checks)
        # ★ SAME evaluate_decision_core() as live ★
        # Runs AnalysisAgent → DecisionAgent → RiskEngine → PositionSizer → TradePermission
        # → signal_persistence → regime_suppression → duplicate → correlation → final_decision_gate

      # DevilsAdvocateGate + ApprovalMode + MT5ExecutionAdapter are SKIPPED in backtest
      # (broker fills via adapter instead of MT5)

      if dec_out["decision"] in ("BUY","SELL") and risk_out["approved"] and perm_out["allowed"]:
        # Look-ahead fix: fill at NEXT bar's open, not signal bar's close
        entry = df.iloc[i+1]["open"] if i+1 < len(df) else df.iloc[i]["close"]
        trade = adapter.open_trade(symbol, direction, entry_price=entry, sl, tp, lot, bar_time)
        open_trades.append(trade)
```

**Key parity guarantees:**
- Same `AITrader.evaluate_decision_core()` runs in both live and backtest.
- Same provider output shape `{df, ind_ctx, regime, regime_ctx, mtf_bias, symbol, timeframe, data_source}`.
- Same indicator chain: `add_canonical_indicators → ExtendedIndicators → Indicators`.
- Same `MarketRegimeDetector`, same `AnalysisAgent`, same `DecisionAgent`, same `RiskEngine`, same `TradePermission`.

**Differences (intentional):**
- Live fetches via `MT5.copy_rates_from_pos`; backtest replays from CSV.
- Live fills via `OrderManager.place_market_order` → `mt5.order_send`; backtest fills via `BrokerSimulator` (next-bar open, bar high/low sweep for SL/TP).
- Live's `DevilsAdvocateGate` and `ApprovalMode` are bypassed in backtest (LLM veto not replayed).
- News/sentiment context: live fetches news from Forex Factory + RSS + NewsAPI; backtest uses `data/economic_calendar.json` snapshot (live-only).

---

## 5. MT5 Broker Call Catalog (Production)

`mt5.order_send` is called from **8 distinct sites** in production code:
- `broker/order_manager.py`: market (`:430`), limit (`:591`), cancel (`:621`), stop (`:720`), stop-limit (`:778`), SLTP modify (`:821`), close (`:873`), alt-execute (`:1422`)
- `broker/position_manager.py`: partial close (`:469`), Friday close (`:874`), timeout close (`:914`)

Market data (`copy_rates_from_pos` / `copy_rates_range`) is called from:
- `broker/mt5_connection.py:633` (shared locked wrapper — used by `data/fetcher.py`)
- `broker/mt5_data.py:216` (legacy/secondary)
- `broker/mt5_historical_fetcher.py:102` (bulk download)
- `analysis/microstructure.py:156` (`copy_ticks_range` — bypasses shared lock!)

`mt5.symbol_info_tick` (live bid/ask/spread) is called from 11 sites across `broker/mt5_data.py`, `data/live_feed.py`, `data/data_orchestrator.py`, `broker/order_manager.py`, `broker/position_manager.py`, `broker/account_manager.py`.

`mt5.market_book_add` (Level-2 depth) is **NOT used anywhere**.

---

## 6. ML / RL / LLM Status

| Layer | Status | Production weight | Live data path |
|-------|--------|-------------------|----------------|
| **Rule Engine** (`strategy/signal_engine.py`) | ACTIVE | 30% | Reads `ind_ctx` from MarketAgent (OHLC + 139 indicator columns) |
| **ML Ensemble** (`ml/model_predictor.py`) | **DISABLED in live code** (`if False:` at `analysis_agent.py:2001`) | 30% (when re-enabled) | 110-feature dict from `FeatureEngineer.build_feature_vector()` |
| **RL Agent** (`rl/rl_agent.py`) | Constructed but minor | 20% | 29-dim state vector (Box(29)) — 23 market features + 6 account state |
| **LLM Analyst** (`ai/ai_analyst.py` + `agents/master_analyst.py`) | ACTIVE | 20% | Groq → Gemini → OpenRouter cascade; 22 context blocks fed in prompt |
| **MasterDecisionEngine** (`core/master_decision.py`) | ACTIVE | Fuses the 4 layers | `SignalFusion.fuse()` — majority vote + opposition penalty |

**ML is currently commented out** because `memory/ml_models/_registry.json` is empty (no trained models deployed). The training pipeline (`ml/train_historical.py`, `ml/data_bootstrap.py`) exists and works, but no model has been promoted to production. When re-enabled, ML inference needs the **exact same** 110-feature dict that training saw — so historical CSVs MUST reproduce all upstream contexts (session, intermarket, news, SMC, confluence).

**LLM** uses a 3-provider cascade (Groq → Gemini → OpenRouter) with 5-minute in-memory cache (`core/llm_cache.py`, max 200 entries). Ollama was REMOVED from the cascade and is now an opt-in veto gate (`core/ollama_validator.py`).

---

## 7. Configuration Files

| File | Purpose |
|------|---------|
| `config.py` (repo root) | Main config — `EXECUTION_MODE`, `DEFAULT_TIMEFRAME`, `MAX_LOT`, `MAX_RISK_PC`, `DAILY_LOSS_LIMIT_PCT`, `MAX_OPEN_TRADES`, `BYPASS_NEWS_GATE`, etc. |
| `ai/forex/config.py` | Sub-config for the AI/forex module |
| `.env` (gitignored) | Secrets: `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`, `GROQ_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, `NEWSAPI_KEY`, `OANDA_API_KEY`, `MT5_BROKER_TZ_OFFSET_HOURS`, `ALLOW_REAL_MONEY_TRADING`, `SIMULATION_MODE`, `MT5_FALLBACK_TO_SIMULATION`, `DEVILS_ADVOCATE_FAIL_OPEN`, `BYPASS_FUSION_GATE` |
| `data/config.py` | Data layer config — historical dir paths, default symbols/TFs |
| `data/historical/manifest.json` | Downloaded-data manifest — generated by `scripts/download_historical_data.py` |
| `core/constants.py` | Static constants including `DEFAULT_SPREAD_PIPS` table |
| `risk/rr_policy.py` | Minimum R:R policy (default 2.0, capped at [0.5, 5.0]) |
| `backtest/symbol_specs.py` | Per-symbol pip sizes, contract sizes, digits |

---

## 8. Cache / DB Layer

| Layer | Path | Purpose |
|-------|------|---------|
| **SQLite ML feature store** | `memory/ml_features.db` | Persisted training features + labels (`ml/feature_store.py`) |
| **ML model registry** | `memory/ml_models/{PAIR}_{TF}/{xgboost\|random_forest\|lstm}_v*.pkl` + `_registry.json` | Trained models (currently empty) |
| **Scaler** | `memory/ml_processed/scaler.pkl` | StandardScaler fitted on training set |
| **Indicator cache** | `core/indicator_cache.py` | In-process LRU cache for indicator computation |
| **LLM cache** | `core/llm_cache.py` | 5-min TTL in-memory `OrderedDict`, max 200 entries |
| **News cache** | `intelligence/news_ai.py` | 5-min in-memory + persisted to `memory/news_analysis_memory.jsonl` |
| **Trade memory** | `memory/trader.db` | Live trade journal DB (SQLite via `TraderDB`) |
| **Backtest memory** | `memory/_backtest/` | Wiped on each `run_unified_backtest()` call |
| **CircuitBreaker state** | `memory/circuit_breaker_state.json` | Global kill-switch state |
| **Pending approvals** | `memory/pending_approvals.json` | ApprovalMode MODE_SUPERVISED queue |
| **Daily risk** | `memory/daily_risk.json` | Daily loss tracking |
| **Recent entries** | `risk/trade_permission.py:_recent_entries` | In-process duplicate-entry cache |
| **Live tick buffer** | `data/live_feed.py:LiveFeed` | Per-symbol rolling buffer (120 ticks, ~2 min) |
| **MT5 OHLCV cache** | `data/backtest_ohlcv_cache.py` | Backtest-mode point-in-time HTF cache (shared with `SMCEngine` / `MTFAnalyzer`) |

---

## 9. Key Findings

1. **Backtest parity is structurally sound** — both modes run the same `evaluate_decision_core()` with the same provider output shape. No "backtest-only" decision code path exists.

2. **21 sequential gates** between analysis-BUY and `mt5.order_send` — far more defensive than typical trading systems.

3. **ML is currently disabled** in live code (`if False:` block), so the 4-layer fusion effectively becomes 3-layer (Rule + RL + LLM). Historical backtests reproduce this configuration as long as the CSV provider feeds the same upstream contexts that training would need.

4. **5 critical data-layer parity gaps** (see Deliverable 3) must be closed for backtest to truly match live:
   - Bid/Ask not in CSV (historical fills at `close`, not at ask/bid)
   - `real_volume` silently dropped
   - `LiveMT5Provider.current_time()` returns naive UTC (parity violation)
   - `analysis/microstructure.py` bypasses shared MT5 lock
   - `data/automated_updater.py` (orphan) writes to a disconnected path

5. **No global `MAX_REQUIRED_LOOKBACK` constant** — the effective floor is 500 bars primary TF + 200 bars per HTF + 1 trading week for PDH/PWL/Asian range. This is implicitly enforced by `MarketAgent.run(limit=300)` + `HistoricalCSVProvider(lookback_bars=300)`, but the implicit floor should be made explicit.

6. **No future leakage** was found in the production decision path. Label generation in ML training uses `shift(-N)` (correct — labels peek forward, features don't). The NadarayaWatson envelope module self-documents as REPAINTING (forward-looking centered window); it must be consumed with the `nwe_stable=False` flag for the most recent 500 bars.

7. **2 minor leakage risks** to address (not in production-critical paths):
   - `ml/feature_engineer.py:400` uses `pd.Timestamp.now(tz="UTC")` for `hour_utc` / `day_of_week` features — breaks historical replay (must use `df.index[-1]`).
   - `analysis/smart_money.py:390` uses `datetime.now(timezone.utc)` for kill-zone detection without bar-timestamp fallback.

8. **External data dependencies** (not in CSV, fetched live):
   - News: Forex Factory JSON + 4 RSS feeds + NewsAPI.org
   - Macro: DXY, Gold, Oil, US10Y, S&P500, VIX (via `MacroDataProvider`)
   - Sentiment: Myfxbook retail positioning
   - COT: CFTC weekly Commitments of Traders report (live HTTP fetch)
   - 28-pair correlation matrix (fetched live; `CurrencyStrengthEngine` is DISABLED)

---

## 10. References (Evidence)

- `docs/audit/evidence/P1-A-data-provider-audit.md` — 1,374 lines, data provider layer
- `docs/audit/evidence/P1-B-decision-execution-audit.md` — 910 lines, decision + execution chain
- `docs/audit/evidence/P1-C-analysis-indicators-audit.md` — 310 lines, 79 indicators cataloged
- `docs/audit/evidence/P1-D-ml-rl-llm-audit.md` — 655 lines, ML/RL/LLM data flow
- `docs/audit/evidence/P6-csv-audit.md` — 21 existing CSV files audited
- `docs/audit/evidence/P6-csv-audit.json` — machine-readable audit data
