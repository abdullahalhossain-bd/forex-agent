# P1-B — Decision & Execution Audit (Forex-Agent)

**Task ID:** P1-B
**Auditor:** Explore sub-agent
**Scope:** Trace the live trading decision-to-execution path; verify the
README claim that the chain is
`core/trader.py: AITrader.evaluate_decision_core()` →
`broker/order_manager.py: MetaTrader5.order_send(request)`.
**Repo root:** `/home/z/my-project/download/forex-agent`
**Date:** 2026-08-17

---

## 0. README claim — VERIFIED (with nuance)

The forensic README's claim is **structurally correct** but **under-specifies
the real chain**. The actual live path is:

```
main.py main() -> ForexAISystem.start_trading()
  -> trading_engine.run()               (AutonomousTraderSystem.run — core/trader.py:4120)
     for symbol in active_symbols:
        -> trader.run_cycle()           (AITrader.run_cycle — core/trader.py:1722)
           -> [1/9] LiveMT5Provider.get_market_out()
                  -> MarketAgent.run()  (agents/market_agent.py:156)
                     -> DataOrchestrator.get_candles()  (data/data_orchestrator.py:191)
                        -> DataFetcher.fetch_ohlcv()    (data/fetcher.py:506)
                           -> _fetch_mt5()              (data/fetcher.py:727)
                              -> MT5Connection.copy_rates_from_pos()
                                 -> mt5.copy_rates_from_pos()   # broker/mt5_connection.py:633
           -> [2/9] CircuitBreaker.allow_trade()       (risk/circuit_breaker.py:142)
           -> [3-6/9] AITrader.evaluate_decision_core() (core/trader.py:895)
                  -> AnalysisAgent.run(market_out)     (agents/analysis_agent.py:192)
                  -> DecisionAgent.decide(...)          (agents/decision_agent.py:387)
                  -> RiskEngine.evaluate(...)           (risk/risk_engine.py:70)
                  -> PositionSizer (Day-76)            (risk/position_sizer.py)
                  -> advanced_risk_gates (orphan)     (core/_orphan_integration.py)
                  -> TradePermission.check(...)        (risk/trade_permission.py:228)
                  -> signal_persistence + regime_suppression + duplicate + correlation
           -> [7/9] LearningAgent.save_decision(...)   (agents/learning_agent.py)
           -> [8/9] DevilsAdvocateGate.review(...)     (core/devils_advocate.py:154)
                  -> ApprovalMode.process(...)         (core/approval_mode.py:89)
           -> [9/9] MT5ExecutionAdapter.open_trade(...) (core/execution_adapter.py:58)
                  -> ExecutionRouter.execute(...)      (execution/execution_router.py:505)
                     -> _execute_mt5_demo(...)         (execution/execution_router.py:552)
                        -> OrderManager.place_market_order(...)  (broker/order_manager.py:230)
                           -> mt5.order_send(request)          # broker/order_manager.py:430
```

**Nuances the README omits:**

1. `evaluate_decision_core()` does NOT call `mt5.order_send` itself — it stops
   at `perm_out`. The MT5 call happens **two layers later**, through
   `MT5ExecutionAdapter` → `ExecutionRouter` → `OrderManager`.
2. Between `evaluate_decision_core()` and the MT5 call sit two more veto
   layers: **Devil's Advocate** (`core/devils_advocate.py:154`) and
   **ApprovalMode** (`core/approval_mode.py:89`). Either can hard-block the
   trade before it reaches `OrderManager`.
3. `OrderManager.place_market_order()` is not the only `mt5.order_send` call
   site — `mt5.order_send` is also called from `broker/position_manager.py`
   (trailing/breakeven/partial/Friday-close) and from limit-order placement
   (`place_limit_order`, `place_stop_order`, `place_stop_limit_order`,
   `cancel_order`, `modify_sltp`, `close_order`).
4. The **SafetyGuard** wrapper (`broker/safety_guard.py`) is constructed by
   `core/runtime.py:boot_safety` (Phase 13) but **never invoked** in the live
   `run_cycle` — `AITrader` calls `TradePermission.check()` and
   `CorrelationFilter.allow()` directly inline. `SafetyGuard` is effectively
   dead code in the live path (registered in `ServiceRegistry` but no caller).

---

## 1. Full call-chain diagram (actual code path)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ main.py:597  main()                                                         │
│  ├ args.mode == "start"                                                      │
│  └ ForexAISystem(args).initialize()  →  ForexAISystem.start_trading()       │
└──────────────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ main.py:343  ForexAISystem.start_trading()                                  │
│  loop:  report = self.trading_engine.run()    # auto-restart on crash       │
└──────────────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ core/trading_engine.py:40  TradingEngine.run()   (subclass)                │
│  └ super().run()  →  AutonomousTraderSystem.run()                           │
└──────────────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ core/trader.py:4120  AutonomousTraderSystem.run()                           │
│  while not _stop_requested:                                                 │
│    ├ _detect_mt5_position_closes()                                          │
│    ├ should_close_for_weekend()           → close_all_orders() if needed    │
│    └ for symbol in _select_cycle_symbols():                                 │
│         trader = self.traders[symbol]  # _build_trader → AITrader(...)       │
│         result = trader.run_cycle(auto_paper_trade=True)                     │
└──────────────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ core/trader.py:1722  AITrader.run_cycle()                                   │
│  Stage 0: HumanOverride check (PAUSED/STOPPED → bail)                       │
│  Stage 0.5: equity-stop (mt5_demo only) via MT5Connection.account_info()    │
│  Stage 0.7: TradeFrequencyController.can_trade_now()                         │
│  Stage 1: [1/9] market_out = LiveMT5Provider.get_market_out(sym, tf)         │
│  Stage 1b: PaperTrader.update_price() + PositionManager.poll_once()          │
│  Stage 2: [2/9] CircuitBreaker.allow_trade() → hard block if tripped         │
│  Stage 2b: production_hardening.check_data_staleness() + is_candle_closed()  │
│  Stage 3: evaluate_decision_core(market_out, session_ctx, debugger)          │
│           ├── enrich_market_context()  (orphan integration)                   │
│           ├── AnalysisAgent.run(market_out)                                  │
│           │     └ 17 analyzers (see §10)                                     │
│           ├── DecisionAgent.decide(market_out, analysis_out, placeholder)   │
│           ├── apply_signal_scoring()  (signal_scorer)                       │
│           ├── Stop Hunt Direct Lane (override branch)                       │
│           ├── RiskEngine.sync_open_positions(_live_open_pairs)              │
│           ├── RiskEngine.evaluate(signal, entry, atr, regime, correlation_ctx)│
│           ├── MT5 sync fail-closed checks (live only)                        │
│           ├── _apply_advanced_sizing()  → PositionSizer                      │
│           ├── apply_advanced_risk_gates()  (orphan)                         │
│           ├── TradePermission.check(dec_out, risk_out, news, session, ...)   │
│           ├── signal_persistence.is_stable()                                │
│           ├── regime_suppression.should_suppress()                           │
│           ├── PaperTrader.has_open_position()  (duplicate)                  │
│           ├── CorrelationFilter.allow()                                      │
│           └── final_decision_gate()  (orphan)                                │
│  Stage 4: LearningAgent.save_decision()                                      │
│  Stage 5: _build_result(...)                                                 │
│  Stage 6: [8/9] DevilsAdvocateGate.review()  → hard veto                     │
│  Stage 7: ApprovalMode.process()  → MODE_SUPERVISED pends, MODE_AUTONOMOUS OK│
│  Stage 8: [9/9] MT5ExecutionAdapter.open_trade(...)                          │
│           └ ExecutionRouter.execute(decision_result)                       │
│              ├ HARD GATE 1: decision ∈ {BUY,SELL}                            │
│              ├ HARD GATE 2: trade_allowed=True                               │
│              ├ HARD GATE 3: lot ≤ MAX_LOT                                    │
│              ├ HARD GATE 4: lot > 0                                          │
│              └ _execute_mt5_demo(...)                                       │
│                  ├ _ensure_mt5_connected()                                   │
│                  ├ _check_absolute_safety(symbol)                            │
│                  ├ AccountManager.trading_permission(symbol)                 │
│                  └ pullback? OrderManager.place_limit_order(...)            │
│                     : OrderManager.place_market_order(...)                  │
│                        ├ _pre_trade_validate(symbol, dir, lot, sl, tp)       │
│                        │   ├ mt5.terminal_info()  (Algo Trading ON?)         │
│                        │   ├ AccountManager.trading_permission()             │
│                        │   ├ mt5.symbol_info(broker_symbol)  (Market Watch) │
│                        │   └ mt5.symbol_select(broker_symbol, True)         │
│                        ├ mt5.symbol_info_tick(broker_symbol)  (spread chk)  │
│                        ├ mt5.symbol_info(broker_symbol)  (pip_size)          │
│                        ├ mt5.account_info()  (free margin chk)               │
│                        ├ _mt5_positions_get(symbol=...)  (pre-trade snapshot)│
│                        ├ mt5.symbol_info_tick(broker_symbol)  (price/tick)   │
│                        ├ _resolve_filling_mode()  (mt5.symbol_info)         │
│                        ├ request = {action:TRADE_ACTION_DEAL, ...}           │
│                        └ ★ mt5.order_send(request) ★   # line 430             │
│                              ├ result.retcode == TRADE_RETCODE_DONE          │
│                              └ _confirm_position_appeared(ticket)          │
│                                 └ _mt5_positions_get(symbol=...)             │
└──────────────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ ExecutionRouter fills `trade` dict → returns to MT5ExecutionAdapter          │
│ AITrader.run_cycle line 2680-2746:                                          │
│   ├ result["ticket"] = trade.get("ticket")                                  │
│   ├ RiskEngine.record_trade_open(self.symbol)                               │
│   ├ PositionManager.register_open(ticket, db_trade_id)                      │
│   ├ event_bus.publish("trade.execution", ...)                               │
│   └ TelegramNotifier.send_message(...)  (if confluence trade)                │
└──────────────────────────────────────────────────────────────────────────────┘
```

The **single** call that places real broker orders is
`broker/order_manager.py:430` — `result = mt5.order_send(request)`.

---

## 2. Per-symbol cycle order

For each symbol in `self.traders` (built once at boot by `_build_trader`,
`core/trader.py:4105`), one full `run_cycle` is invoked inside
`AutonomousTraderSystem.run` (`core/trader.py:4120`):

```
For each active symbol:
  ┌─ PRE-EXECUTION GATES (in order) ───────────────────────────┐
  │ 1. HumanOverride state check          (core/trader.py:1771)│
  │ 2. Equity stop / balance sync         (core/trader.py:1822)│
  │ 3. TradeFrequencyController.can_trade_now()  (:1884)      │
  │ 4. Market data fetch                  (core/trader.py:1913)│
  │     LiveMT5Provider.get_market_out →  MarketAgent.run     │
  │     → DataOrchestrator.get_candles → DataFetcher._fetch_mt5│
  │       → MT5Connection.copy_rates_from_pos                  │
  │ 5. PaperTrader.update_price() + close detection           │
  │ 6. PositionManager.poll_once()  (mt5_demo/live only)      │
  │ 7. CircuitBreaker.allow_trade()       (core/trader.py:2025)│
  │ 8. Data staleness + candle-close check (production_hardening)│
  └─────────────────────────────────────────────────────────────┘
  ┌─ DECISION CORE (evaluate_decision_core) ────────────────────┐
  │ 3/9  AnalysisAgent.run(market_out)                          │
  │ 4/9  DecisionAgent.decide(market_out, analysis, placeholder)│
  │      + SignalScorer (orphan) + Stop-Hunt Direct Lane        │
  │ 5/9  RiskEngine.evaluate(signal, entry, atr, regime, corr)  │
  │      + PositionSizer (Day-76) + apply_advanced_risk_gates() │
  │ 6/9  TradePermission.check(...)                             │
  │      + signal_persistence + regime_suppression             │
  │      + duplicate + correlation + final_decision_gate       │
  └─────────────────────────────────────────────────────────────┘
  ┌─ POST-DECISION (run_cycle continued) ───────────────────────┐
  │ 7/9  LearningAgent.save_decision(...)                      │
  │ 8/9  DevilsAdvocateGate.review(...)  (HARD VETO)           │
  │      ApprovalMode.process(...)  (MODE_SUPERVISED pends)     │
  │ 9/9  MT5ExecutionAdapter.open_trade(...)                    │
  │      → ExecutionRouter.execute(...)                         │
  │      → OrderManager.place_market_order(...)                 │
  │      → ★ mt5.order_send(request) ★                          │
  │                                                            │
  │ Post-fill: register_open(ticket, db_id), publish events,   │
  │            notify Telegram, log trade_decision_log          │
  └─────────────────────────────────────────────────────────────┘
```

The 9-stage numbering comes from the `log.info("[k/9] ...")` markers
embedded in `run_cycle` and `evaluate_decision_core`.

---

## 3. Market data fetch — where, when, what shape

| Attribute          | Value |
|--------------------|-------|
| **Stage**          | `[1/9] Market Agent` — first MT5-touching step of every cycle (`core/trader.py:1911`) |
| **Module**         | `LiveMT5Provider` (`core/data_provider.py:60`) wraps `agents/market_agent.py:MarketAgent` |
| **Underlying call**| `DataOrchestrator.get_candles(symbol, timeframe, limit=300)` → `DataFetcher.fetch_ohlcv` → `DataFetcher._fetch_mt5` (`data/fetcher.py:727`) → `MT5Connection.copy_rates_from_pos` (`broker/mt5_connection.py:627`) → `mt5.copy_rates_from_pos(symbol, mt5_tf, 0, 300)` |
| **Timeframes**     | Primary TF: whatever `config.DEFAULT_TIMEFRAME` says (default `"15m"` → `mt5.TIMEFRAME_M15`). MTF bias ladder always `["1d","4h","1h","15m"]` (`agents/market_agent.py:171`) fetched via `MultiTimeframeAnalyzer.analyze()`. H4 is fetched separately inside `AnalysisAgent.run()` via `self._h4_fetcher.fetch_ohlcv(symbol, "H4", 300)`. |
| **Bars**           | `limit=300` (matches backtest warmup for parity). |
| **Data shape returned** | `MarketAgentResult` TypedDict (`agents/market_agent.py:49-60`): `{df: DataFrame, ind_ctx: dict, regime: dict, regime_ctx: dict, mtf_bias: dict, symbol: str, timeframe: str, data_source: str}`. |
| **Indicator chain** | `data/indicator_registry.add_canonical_indicators()` → `ExtendedIndicators.add_all` (pandas-ta) → legacy `Indicators.add_all` (three-tier fallback, `agents/market_agent.py:252-295`). |
| **Regime detection** | `analysis/market_regime.MarketRegimeDetector.detect(df)` → stored as `regime` + `regime_ctx` in market_out (`agents/market_agent.py:301-309`). |
| **MTF bias shape** | `{"bias": "BULLISH"\|"BEARISH"\|"NEUTRAL", "confidence": "HIGH"\|"MEDIUM"\|"LOW", "trends": {"4h":..., "1h":..., "15m":...}}` — full per-TF breakdown lived only in `MultiTimeframeAnalyzer`; trader.py reconstructs per-TF trends from `_mtf_bias_full.get("trends")` (`core/trader.py:1322-1354`). |
| **Fallback**       | If MT5 not installed (Linux/Mac dev) `DataOrchestrator._get_mt5_conn()` returns None → `DataFetcher.fetch_ohlcv` falls through to API sources (TwelveData, yfinance). `last_source` flag tells consumers which path fed the data. |

---

## 4. Signal fusion — RuleEngine + ML + RL + LLM

The actual 4-layer fusion lives in `core/signal_fusion.py:SignalFusion.fuse()`
called from `core/master_decision.py:MasterDecisionEngine.decide()`, which is
itself invoked from `agents/analysis_agent.py:run()` at step "Day 73 Master
Decision Engine" (`agents/analysis_agent.py:2259`).

### Layer weights (`core/signal_fusion.py:8-13`)
```
1. Rule Engine (Day 67 Confluence)  — weight 30%
2. ML Ensemble (Day 69-70)          — weight 30%
3. RL Agent (Day 71)                — weight 20%
4. LLM Analyst (MasterAnalyst)      — weight 20%
```

### Fusion algorithm (`core/signal_fusion.py:87-251`)
```python
# 1. Mark non-participating layers (WAIT/HOLD/NOT_READY) → weight=0
working = self._normalize_weights(signals, result)   # zero abstainers
                                                  # then redistribute to 1.0

# 2. Split directional vs abstaining
directional = [s for s in working if s.signal in ("BUY","SELL")]
buy_votes  = [s for s in directional if s.signal=="BUY"]
sell_votes = [s for s in directional if s.signal=="SELL"]

# 3. Majority among directional votes only (WAIT no longer vetoes)
if len(buy_votes) > len(sell_votes):  majority = "BUY";  agreeing = buy_votes
elif len(sell_votes) > len(buy_votes): majority = "SELL"; agreeing = sell_votes
else:                                  majority = "WAIT"

# 4. Weighted confidence from agreeing directional layers
weighted_conf = sum(s.confidence * s.weight for s in agreeing) / sum(s.weight)

# 5. Opposition penalty (only real BUY vs SELL conflict)
if opposing and agreeing:
    if avg_opp > 80:  weighted_conf *= 0.70   # strong opposition
    else:             weighted_conf *= 0.85   # weak opposition

# 6. Final signal from directional agreement
if len(agreeing) >= 3:                              result.final_signal = majority
elif len(agreeing) == 2 and conf >= REDUCED_THRESHOLD (40):  majority
elif len(agreeing) == 1 and conf >= FULL_THRESHOLD (70):    majority  # single-layer safety net
else:                                                 WAIT

# 7. Position size (FULL/HALF/REDUCED/WAIT)
# Confidence ceiling: EntrySafetyFilters.calibrate_confidence (max 99%)
```

### Layer sources (filled in `AnalysisAgent.run`):

| Layer        | Source module                              | Built in analysis_agent.py at |
|--------------|--------------------------------------------|-------------------------------|
| Rule Engine  | `strategy.signal_engine.SignalEngine` + `intelligence.confluence_engine.ConfluenceDecision` | step 6 (`:547`), Day-67 (`:1822`) |
| ML Ensemble  | `ml.ensemble_engine.EnsembleEngine` + `ml.predictor.MLPredictor` (Day-69/70) | step 10 (`:1249`) |
| RL Agent     | `rl.rl_agent.RLAgent` (Day-71)             | step 11 (`:2138`) |
| LLM Analyst  | `ai.ai_analyst.AIAnalyst` (classic LLM) + `agents.master_analyst.MasterAnalyst` (vision) | steps 10 (`:1249`) and 12 (`:1296`) |
| Master Decision | `core/master_decision.MasterDecisionEngine.decide()` (Day-73) — fuses the four | step "Day 73" (`:2259`) |

`MasterDecisionEngine` is the actual consumer of `SignalFusion.fuse()` — it
collects the four `LayerSignal` objects, calls `fusion.fuse(signals)`, then
runs `DecisionValidator.validate()` and `ConfidenceManager.adjust_weights()`.

---

## 5. TradePermission layer — gate catalog and order

Source: `risk/trade_permission.py:TradePermission.check()` (line 228).

**Gates execute in this exact order** (each appends to `checks[]` and
increments `passed`/`total`; final `allowed = passed == total`):

| # | Gate name (string in checks[]) | Bypass key | Source line |
|---|--------------------------------|------------|-------------|
| 0 | Execution filters (from AnalysisAgent) | per-gate | `:290-407` |
| 1 | Valid signal (`BUY`/`SELL`) | `"Valid signal"` | `:409-432` |
| 1b | S/R zone alignment (SELL near support / BUY near resistance) | `"S/R zone alignment"` | `:460-519` |
| 1c | Trend alignment (regime) | `"Trend alignment (regime)"` | `:597` |
| 1d | MTF trend alignment (H4/H1/M15) | `"MTF trend alignment (H4/H1/M15)"` | `:694` |
| 1e | Zone cooldown (duplicate entry within X hrs/pips) | `"Zone cooldown (duplicate entry)"` | `:867` |
| 2 | Risk approved (post-sizer, post-RAG) | `"Risk approved"` | `:911-939` |
| 3 | News safe (fail-safe default DENY if news_ctx empty) | `"News safe"` or `BYPASS_NEWS_GATE=true` env | `:941-974` |
| 3b | Entry quality guardrails (confidence penalty, hard-block only on extreme) | per-rule | `:987-1138` |
| 3c | Confirmation bias defense | `"Confirmation bias defense"` | `:1146` |
| 3d | Revenge trading detector | `"Revenge trading detector"` | `:1248` |
| 3e | Cost-aware EV gate (book_guardrails) | `"Cost-aware EV gate (book_guardrails)"` | `:1362` |
| 4 | Min confidence (with adaptive loss-streak bump, win-rate scaling) | `"Min confidence"` | `:1450-1518` |
| 5 | Session quality (low-quality sessions require ≥60% conf) | `"Session quality"` | `:1580-1634` |
| 6 | Confluence quality (aligned_factors ≥ threshold) | `"Confluence quality"` | `:1682-1713` |
| 7 | Min R:R (≥ MIN_RR from rr_policy) | `"Min R:R"` | `:1714-1794` |
| 8 | SMC + Session fusion gate | `"SMC+Session fusion"` or `BYPASS_FUSION_GATE=true` | `:1794` |
| 9 | LLM availability (MasterAnalyst / Devil's Advocate available) | `LLM_UNAVAILABLE_FAIL_OPEN=true` | `:1830-1866` |
| 10 | `allowed = passed == total` | — | `:1868` |

**Post-permission gates** (run by `AITrader.evaluate_decision_core`, AFTER
TradePermission.check returns):

| # | Gate | Source |
|---|------|--------|
| P1 | `signal_persistence.is_stable()` — flip-flop filter | `core/trader.py:1522` |
| P2 | `regime_suppression.should_suppress()` | `core/trader.py:1546` |
| P3 | Duplicate trade (`PaperTrader.has_open_position`) | `core/trader.py:1575` |
| P4 | `CorrelationFilter.allow()` | `core/trader.py:1598-1602` |
| P5 | `final_decision_gate()` — system watchdog + network monitor + ML drift | `core/trader.py:1626` |

**After approval (veto layers):**

| # | Veto | Source |
|---|------|--------|
| V1 | DevilsAdvocateGate.review() — HARD VETO, fail-closed on exception | `core/trader.py:2489` |
| V2 | ApprovalMode.process() — mode 2 pends, mode 1/3 default deny | `core/trader.py:2597` |

So in total the trade must pass through **~21 sequential gates** between
"analysis says BUY" and "broker receives order_send".

---

## 6. RiskEngine — what it computes

Source: `risk/risk_engine.py:RiskEngine.evaluate()` (line 70).

### Inputs
| Param | From | Notes |
|-------|------|-------|
| `signal` | `dec_out["decision"]` | BUY/SELL (WAIT/NO_TRADE/HOLD/"" → instant reject) |
| `entry` | `signal_data["entry"] or ind["close"] or ind["price"] or latest_price` (`core/trader.py:1015`) | 0/None → reject |
| `atr` | `ind["atr"]` from MarketAgent's `ind_ctx` | NaN/0 → defaults to 0.0010 |
| `regime` | `market_out["regime"]` from MarketRegimeDetector | `{"volatility": "LOW"\|"NORMAL"\|"HIGH_VOLATILITY", ...}` |
| `correlation_ctx` | `analysis_out["correlation_ctx"]` (from CorrelationEngine, Day-96) | `{corr_risk, risk_adjustment, corr_pairs}` |

### Outputs (set on `risk_out` dict)
| Field | How computed | Source |
|-------|--------------|--------|
| `approved` | True unless: WAIT signal, entry 0/None, daily-loss≥limit, open_trades≥max, correlation≥0.90 | `:75-128` |
| `sl_price` / `tp_price` | `entry ± sl_distance` where `sl_distance = atr × vol_mult × instrument_mult`; floor of 10 pips | `:154-189` |
| `sl_pips` / `tp_pips` | `sl_distance / pip_size` ; `tp = sl_pips × MIN_RR` | `:170-192` |
| `rr_ratio` | `tp_pips / sl_pips` | `:192` |
| `vol_mult` | LOW=1.0, NORMAL=1.5 (ATR_SL_MULT), HIGH=2.2 ; JPY pairs ×1.2, XAUUSD/XAGUSD ×1.5, US30/NAS100 ×1.3 | `:132-152` |
| `MIN_RR` | `risk.rr_policy.get_min_rr()` (default 2.0, capped at [0.5, MAX_RR=5.0]) | `:172-182` |
| `risk_usd` | `balance × MAX_RISK_PC / 100` (1.0% default) | `:194` |
| `lot` (raw) | `risk_usd / (sl_pips × pip_value)` ; ×0.5 leverage mult when MAX_LOT>1 ; ×`correlation_adjustment` (0.25-1.0) ; rounded to [0.01, MAX_LOT=0.20] | `:196-228` |
| `risk_pc` | recomputed from final (post-cap) lot — actual exposure, not intended | `:248-261` |
| `risk_usd_max_by_lot`, `risk_pc_max_by_lot` | what risk would be at MAX_LOT | `:245-246` |
| `correlation_risk_score`, `correlation_adjustment` | stored on self for audit | `:116-123` |
| `reject_reason` | when `approved=False` | `_reject()` helper |

### Constants (defaults from `risk/risk_engine.py:20-40`)
```
MAX_RISK_PC       = 1.0       (% of balance per trade)
MIN_RR            = 2.0       (aligned with rr_policy.get_min_rr)
MAX_RR            = 5.0
DAILY_LOSS_LIMIT  = config.DAILY_LOSS_LIMIT_PCT (default 3.0)
MAX_OPEN_TRADES   = config.MAX_OPEN_TRADES      (default 10)
ATR_SL_MULT       = 1.5
MAX_LOT           = config.MAX_LOT              (default 0.20)
```

### Inputs needed from market data
`atr`, `regime.volatility`, `ind_ctx.close`/`price`, `ind_ctx.spread_pips`
(for correlation adjustment), `correlation_ctx.corr_risk` (from analysis_out).

### Day-76 PositionSizer override (`_apply_advanced_sizing`)
Runs AFTER RiskEngine and may further shrink/block the lot via
`kelly × volatility × confidence × correlation × drawdown × loss-streak`
multipliers. See `core/trader.py:459-548`.

---

## 7. Execution mode switching

### Modes (`execution/execution_router.py:93-145`)
```
self.mode = (mode or "mt5_demo").lower()    # honoring config.EXECUTION_MODE
self._simulation_mode = bool(SIMULATION_MODE)  # config.SIMULATION_MODE
self._mt5_fallback_to_sim = bool(MT5_FALLBACK_TO_SIMULATION)  # default True
```

### Switch logic (`_init_mode`, `:163-348`)
| Condition | Resulting path |
|-----------|----------------|
| `SIMULATION_MODE=true` | `_init_simulation_mode()` → `SimulatedExecutor` (no MT5) (`:339-349`) |
| `self.mode == "backtest"` | `_init_simulation_mode()` (router inert; `backtest.unified_engine` drives fills) (`:186-190`) |
| `self.mode == "mt5_demo"` + MT5 unreachable + `MT5_FALLBACK_TO_SIMULATION=true` | `_init_simulation_mode()` (`:240-243`, `:267-270`, `:283-286`) |
| `self.mode == "mt5_demo"` + MT5 reachable | Shared `MT5Connection` + `AccountManager` + `OrderManager` + `JournalBridge` (`:290-334`) |
| `self.mode == "mt5_live"` + `ALLOW_REAL_MONEY_TRADING=true` + `MT5_REAL_LOGIN/PASSWORD/SERVER` all set | Same as demo, real credentials, `_mt5_fallback_to_sim = False` (NO silent fallback) (`:192-253`) |
| `self.mode == "mt5_live"` + missing real-money opt-in | `raise RuntimeError("...ALLOW_REAL_MONEY_TRADING is not set...")` (`:212-227`) |

### The final MT5 call — request shape (`broker/order_manager.py:414-430`)
```python
request = {
    "action":       mt5.TRADE_ACTION_DEAL,
    "symbol":       broker_symbol,
    "volume":       lot,
    "type":         mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL,
    "price":        price,             # round(tick.ask/bid, digits)
    "sl":           sl or 0.0,
    "tp":           tp or 0.0,
    "deviation":    10,                # max slippage (points)
    "magic":        424242,
    "comment":      comment,           # "ai_trader_demo" or "ai_trader_sim"
    "type_time":    mt5.ORDER_TIME_GTC,
    "type_filling": filling_mode,      # _resolve_filling_mode(): IOC/FOK/RETURN
}
result = mt5.order_send(request)     # ★ broker/order_manager.py:430 ★
```

### Confirmation (`broker/order_manager.py:437-485`)
- `result.retcode == mt5.TRADE_RETCODE_DONE (10009)` → success
- `_check_confirmation(result, attempt, requested_volume, symbol)` — handles
  partial fills, retries, duplicate-order detection (pre/post-position poll)
- `_confirm_position_appeared(broker_symbol, ticket)` — polls
  `mt5.positions_get()` for up to 2s to ensure the position is queryable
  before declaring success.

### `copy_rates` vs `order_send`
- **Market data path**: `MT5Connection.copy_rates_from_pos()` →
  `mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, 300)` —
  `broker/mt5_connection.py:633`
- **Order placement path**: `mt5.order_send(request)` —
  `broker/order_manager.py:430` (market), `:591` (limit), `:621` (cancel),
  `:720` (stop), `:778` (stop-limit), `:821` (SLTP modify), `:873` (close)

---

## 8. MT5 broker calls catalog (file:line)

Only production (non-scripts/diagnostics) call sites. Lines are 1-indexed
from the actual source.

### `broker/order_manager.py` (the live trading path)
| Line | Call |
|------|------|
| 142  | `mt5.symbol_info(broker_symbol)` (precision/digits lookup, `_pre_trade_validate`) |
| 173  | `mt5.symbol_info(broker_symbol).filling_mode` (filling-mode probe) |
| 175  | `mt5.ORDER_FILLING_IOC` (constant) |
| 178  | `mt5.ORDER_FILLING_IOC` |
| 180  | `mt5.ORDER_FILLING_FOK` |
| 182  | `mt5.ORDER_FILLING_RETURN` |
| 183  | `mt5.ORDER_FILLING_IOC` (default fallback) |
| 201  | `mt5.ORDER_FILLING_IOC` (try-fail fallback) |
| 266  | `mt5.symbol_info_tick(broker_symbol)` (spread check, pre-trade) |
| 269  | `mt5.symbol_info(broker_symbol)` (pip size) |
| 300  | `mt5.account_info()` (free-margin check) |
| 360  | `mt5.symbol_info_tick(broker_symbol)` (price/tick, per retry attempt) |
| 383  | `mt5.symbol_info(broker_symbol)` (pip size, per retry attempt) |
| 407  | `mt5.ORDER_TYPE_BUY` / `mt5.ORDER_TYPE_SELL` (constant) |
| 415  | `mt5.TRADE_ACTION_DEAL` (constant) |
| 425  | `mt5.ORDER_TIME_GTC` (constant) |
| 430  | ★ `mt5.order_send(request)` — market order placement |
| 560  | `mt5.ORDER_TYPE_BUY_LIMIT` / `mt5.ORDER_TYPE_SELL_LIMIT` |
| 566  | `mt5.TRADE_ACTION_PENDING` |
| 580  | `mt5.ORDER_TIME_SPECIFIED` |
| 588  | `mt5.ORDER_TIME_GTC` |
| 591  | `mt5.order_send(request)` — limit order placement |
| 620  | `mt5.TRADE_ACTION_REMOVE` (cancel pending order) |
| 621  | `mt5.order_send(request)` — cancel order |
| 624  | `mt5.TRADE_RETCODE_DONE` (constant) |
| 642  | `mt5.orders_get()` (pending orders list) |
| 689  | `mt5.ORDER_TYPE_BUY_STOP` |
| 691  | `mt5.symbol_info_tick(broker_symbol)` (stop price) |
| 696  | `mt5.ORDER_TYPE_SELL_STOP` |
| 698  | `mt5.symbol_info_tick(broker_symbol)` |
| 706  | `mt5.TRADE_ACTION_PENDING` |
| 715  | `mt5.ORDER_TIME_GTC` |
| 720  | `mt5.order_send(request)` — stop order |
| 756  | `mt5.ORDER_TYPE_BUY_STOP_LIMIT` |
| 758  | `mt5.ORDER_TYPE_SELL_STOP_LIMIT` |
| 763  | `mt5.TRADE_ACTION_PENDING` |
| 773  | `mt5.ORDER_TIME_GTC` |
| 778  | `mt5.order_send(request)` — stop-limit order |
| 814  | `mt5.TRADE_ACTION_SLTP` (modify SL/TP) |
| 821  | `mt5.order_send(request)` — SLTP modify |
| 848  | `mt5.symbol_info_tick(position.symbol)` (close price) |
| 852  | `mt5.ORDER_TYPE_BUY` |
| 853  | `mt5.ORDER_TYPE_SELL` |
| 860  | `mt5.TRADE_ACTION_DEAL` |
| 869  | `mt5.ORDER_TIME_GTC` |
| 873  | `mt5.order_send(request)` — close position |
| 923  | `mt5.ORDER_TYPE_BUY` (read position type) |
| 959  | `mt5.history_deals_get(start, end)` |
| 967  | `mt5.DEAL_TYPE_BUY` |
| 1000 | `mt5.terminal_info()` (Algo-Trading ON check) |
| 1030 | `mt5.symbol_info(broker_symbol)` (Market Watch check) |
| 1046 | `mt5.symbol_select(broker_symbol, True)` |
| 1057 | `mt5.symbol_info(broker_symbol)` (re-fetch after select) |
| 1089 | `mt5.symbol_info_tick(broker_symbol)` |
| 1130 | `mt5.symbol_info(broker_symbol)` |
| 1214 | `mt5.last_error()` (failure logging) |
| 1402 | `mt5.symbol_info_tick(broker_symbol)` (alt path) |
| 1406 | `mt5.ORDER_TYPE_BUY` / `mt5.ORDER_TYPE_SELL` |
| 1409 | `mt5.TRADE_ACTION_DEAL` |
| 1419 | `mt5.ORDER_TIME_GTC` |
| 1422 | `mt5.order_send(request)` (alt execute path) |

### `broker/position_manager.py` (active trade management)
| Line | Call |
|------|------|
| 36   | `mt5.positions_get(**kwargs)` (retry-wrapped, `_safe_positions_get`) |
| 449  | `mt5.symbol_info_tick(symbol)` (partial close) |
| 453  | `mt5.ORDER_TYPE_SELL` / `mt5.ORDER_TYPE_BUY` |
| 457  | `mt5.TRADE_ACTION_DEAL` |
| 466  | `mt5.ORDER_TIME_GTC` |
| 467  | `mt5.ORDER_FILLING_FOK` |
| 469  | `mt5.order_send(request)` — partial close |
| 757  | `mt5.symbol_info_tick(symbol)` (trailing stop) |
| 853  | `mt5.ORDER_TYPE_SELL` / `mt5.ORDER_TYPE_BUY` (Friday close) |
| 855  | `mt5.symbol_info_tick(pos.symbol)` |
| 862  | `mt5.TRADE_ACTION_DEAL` |
| 870  | `mt5.ORDER_TIME_GTC` |
| 871  | `mt5.ORDER_FILLING_IOC` |
| 874  | `mt5.order_send(request)` — Friday close |
| 875  | `mt5.TRADE_RETCODE_DONE` |
| 894  | `mt5.ORDER_TYPE_SELL` / `mt5.ORDER_TYPE_BUY` (timeout close) |
| 896  | `mt5.symbol_info_tick(symbol)` |
| 903  | `mt5.TRADE_ACTION_DEAL` |
| 910  | `mt5.ORDER_TIME_GTC` |
| 911  | `mt5.ORDER_FILLING_IOC` |
| 914  | `mt5.order_send(request)` — timeout close |

### `broker/mt5_connection.py` (shared wrapper)
| Line | Call |
|------|------|
| 196  | `mt5.shutdown()` (reconnect sequence start) |
| 204  | `mt5.initialize(**init_kwargs)` (with MT5_LOCK) |
| 205  | `mt5.last_error()` |
| 211  | `mt5.login(login, password, server)` |
| 218  | `mt5.last_error()` |
| 222  | `mt5.shutdown()` (login failed) |
| 247  | `mt5.shutdown()` (disconnect) |
| 311  | `mt5.terminal_info()` (health check, cached 5s) |
| 343  | `mt5.account_info()` (health probe, deep check) |
| 430  | `mt5.account_info()` (get_account_info) |
| 434  | `mt5.last_error()` |
| 520  | `mt5.symbol_info_tick(symbol)` (get_tick) |
| 561  | `mt5.symbol_info(symbol)` (get_symbol_info) |
| 563  | `mt5.symbol_select(symbol, True)` |
| 564  | `mt5.symbol_info(symbol)` (re-fetch after select) |
| 596  | `mt5.positions_get(**kwargs)` (locked wrapper) |
| 622  | `mt5.symbol_select(symbol, enable)` |
| 633  | `mt5.copy_rates_from_pos(symbol, timeframe, start_pos, count)` |
| 664  | `mt5.account_info()` (banner) |

### `broker/account_manager.py`
| Line | Call |
|------|------|
| 95   | `mt5.account_info()` (primary) |
| 97   | `mt5.last_error()` |
| 198  | `mt5.symbols_get(f"*{requested}*")` (broker symbol resolution) |
| 204  | `mt5.symbols_get()` (full symbol list) |
| 238  | `mt5.symbol_info(broker_symbol)` |
| 243  | `mt5.symbol_select(broker_symbol, True)` |
| 244  | `mt5.symbol_info(broker_symbol)` (re-fetch) |
| 246  | `mt5.symbol_info_tick(broker_symbol)` |

### `broker/symbol_manager.py`
| Line | Call |
|------|------|
| 206  | `mt5.symbol_info(symbol)` |
| 208  | `mt5.last_error()` |
| 415  | `mt5.symbol_info(symbol)` |

### `broker/health_monitor.py`
| Line | Call |
|------|------|
| 142  | `mt5.terminal_info()` |
| 151  | `mt5.account_info()` |

### `broker/mt5_data.py` (legacy/secondary — DataOrchestrator now uses fetcher.py)
| Line | Call |
|------|------|
| 157  | `mt5.symbol_info_tick(broker_symbol)` |
| 162  | `mt5.symbol_info(broker_symbol)` |
| 216  | `mt5.copy_rates_from_pos(broker_symbol, tf_const, 0, count)` |

### `broker/mt5_historical_fetcher.py` (backtest bulk-download)
| Lines | Calls |
|-------|-------|
| 48-56 | `mt5.TIMEFRAME_M1` ... `mt5.TIMEFRAME_MN1` (constants) |
| 89    | `mt5.TIMEFRAME_M1` (default) |
| 102   | `mt5.copy_rates_range(ticker, timeframe, start_loop, end_loop)` |
| 125   | `mt5.TIMEFRAME_M1` |
| 178   | `mt5.TIMEFRAME_H1` |

### `core/trader.py` (direct MT5 calls — only as fallbacks/edge cases)
| Line | Call |
|------|------|
| 1835 | `mt5.account_info()` (equity stop fallback when no shared conn) |
| 3190 | `mt5.symbol_info(p.symbol)` (weekend guard close_all fallback) |
| 4626 | `mt5.history_deals_get(utc_from, utc_to)` (MT5 close detection) |

### `data/data_orchestrator.py`
| Line | Call |
|------|------|
| 282  | `mt5_lib.orders_get()` (MetaTrader5 imported as `mt5_lib`) |
| 310  | `mt5_lib.symbol_info(symbol)` |

### `data/fetcher.py` (data fetching — primary MT5 candle path)
| Line | Call |
|------|------|
| 776  | `mt5.last_error()` (after `symbol_select` failure) |
| 852  | (via `self._mt5_conn.copy_rates_from_pos`) — actually delegated |

### `core/orphan_cleanup.py`
| Line | Call |
|------|------|
| 25   | `mt5.positions_get()` (retry-wrapped) |
| 66   | `mt5.history_deals_get(start, end)` |

### `core/production_hardening.py`
| Line | Call |
|------|------|
| 36   | `mt5.positions_get()` (retry-wrapped) |
| 59   | `mt5.positions_get()` (retry attempt) |

### Other production call sites (non-`broker/`)
| File:line | Call |
|-----------|-----|
| `analysis/microstructure.py:151` | `mt5.initialize()` (Day-97 tick analysis — its own session!) |
| `analysis/microstructure.py:156` | `mt5.copy_ticks_range(symbol, utc_from, utc_to, mt5.COPY_TICKS_ALL)` |
| `system/watchdog.py:155` | `mt5.initialize()` (its own session) |
| `system/watchdog.py:157` | `mt5.account_info()` |
| `system/watchdog.py:158` | `mt5.shutdown()` |
| `system/network_monitor.py:223-228, 244-246` | `mt5.initialize/symbols_total/shutdown` (latency probe) |

**Full list of all `mt5.*` calls** is also retrievable via:
```bash
rg -n 'mt5\.(initialize|shutdown|login|copy_rates_from_pos|copy_rates_from|copy_rates_range|copy_ticks_from|copy_ticks_range|symbol_info|symbol_info_tick|symbol_select|symbols_total|symbols_get|account_info|positions_get|orders_get|order_send|order_calc_margin|order_calc_profit|history_orders_get|history_deals_get|terminal_info|last_error|version|market_book_add|market_book_get)' \
   --type py --glob '!scripts/**' --glob '!tests/**' --glob '!backtest/**' --glob '!_backtest_validation/**' .
```

---

## 9. Backtest / historical mode entry point

### Entry point
`main.py:687  _run_backtest(args)`  → `python main.py --mode backtest`.

### Path
```
main.py:_run_backtest(args)
  ├ for symbol in pairs:
  │    csv_path = PROJECT_ROOT/"data"/f"{symbol}_{timeframe}.csv"
  │    df = HistoricalDataLoader().load_csv(file_path=csv_path, pair=symbol, timeframe=timeframe)
  │    if --days:  df = df[df.index >= df.index.max() - pd.Timedelta(days=args.days)]
  │    if --bars:  df = df.tail(args.bars)
  │    result = run_unified_backtest(symbol=symbol, df=df, timeframe=timeframe,
  │                                   starting_balance=balance, warmup_bars=50,
  │                                   max_open_trades=3, max_hold_bars=100,
  │                                   db_path=f"backtest/backtest_run_{symbol}_{timeframe}.db",
  │                                   bypass_checks=bypass_checks)
  └ save summary to backtest/results/unified_backtest_summary.json
```

### The historical loop (`backtest/unified_engine.py:run_unified_backtest`)
```python
# backtest/unified_engine.py:293
trader = _make_backtest_trader(symbol, timeframe, starting_balance, db_path)
              # constructs AITrader(execution_mode="backtest",
              #                       paper_trader=PaperTrader(starting_balance, db=TraderDB(db_path)),
              #                       db=TraderDB(db_path))
              # — ISOLATED backtest DB, never touches live trader.db

broker   = BrokerSimulator(starting_balance=starting_balance,
                           commission_per_lot=_commission,
                           slippage_pips=_slippage)
adapter  = HistoricalExecutionAdapter(broker)         # core/execution_adapter.py:85

# CSV provider support — uses HistoricalCSVProvider when CSVs exist
provider = make_backtest_provider(symbol=symbol, timeframe=timeframe,
                                   df=df, prefer="auto")
# Falls back to HistoricalMT5Provider(df, symbol, timeframe) when no CSVs

# Backtest mode flag (so external APIs skip live calls):
set_backtest_mode(True)
reset_backtest_memory()                  # wipes memory/_backtest/ for clean slate

for i in range(warmup_bars, total_bars):
    # Exit check: bar high/low sweep against open trades
    for trade in open_trades:
        result = broker.check_exit(trade, df.iloc[i]["high"], df.iloc[i]["low"],
                                    df.iloc[i]["close"], current_time)
        if result:  closed_trades.append(result)
        elif trade.hold_bars > max_hold_bars:
            closed = broker.close_trade(trade, df.iloc[i]["close"], current_time, "timeout")
            closed_trades.append(closed)
        else: still_open.append(trade)
    open_trades = still_open

    if len(open_trades) >= max_open_trades: continue

    provider.advance_to(i)
    market_out = provider.get_market_out(symbol, timeframe)  # ← SAME shape as LiveMT5Provider
    session_ctx = {"current_session": "BACKTEST", "gmt_time": str(current_time),
                    "session_strategy": "n/a"}

    core = trader.evaluate_decision_core(market_out, session_ctx, bypass_checks=bypass_checks)
                   # ★ SAME evaluate_decision_core() as live — runs AnalysisAgent,
                   #   DecisionAgent, RiskEngine, PositionSizer, TradePermission,
                   #   signal_persistence, regime_suppression, duplicate, correlation
                   #   final_decision_gate — every gate live runs.

    analysis_out, dec_out, risk_out, perm_out = (core["analysis_out"], core["dec_out"],
                                                  core["risk_out"], core["perm_out"])
    if "error" in analysis_out:               continue  # rejection_stats["NO_TRADE_ANALYSIS"]
    if dec_out["decision"] not in ("BUY","SELL"): continue  # rejection_stats["WAIT"]
    if not risk_out.get("approved"):            continue  # rejection_stats["risk_rejected"]
    if not perm_out.get("allowed"):             continue  # rejection_stats["permission_blocked"]

    # Look-ahead fix: fill at NEXT bar's open, not signal bar's close
    entry = dec_out.get("entry") or (df.iloc[i+1]["open"] if i+1 < len(df) else df.iloc[i]["close"])
    trade = adapter.open_trade(symbol=symbol, direction=action, entry_price=entry,
                               sl=risk_out["sl_price"], tp=risk_out["tp_price"],
                               lot=risk_out["lot"], bar_time=current_time, ...)
    if trade is None:
        rejection_stats["permission_blocked"] += 1   # spread-limit rejection
        continue
    open_trades.append(trade)
    if save_forensics:
        trade_forensics[trade.trade_id] = {full decision-core snapshot}
```

### CSV provider (`core/csv_data_provider.py:HistoricalCSVProvider`)
- Loads CSV from `data/historical/{SYMBOL}/{TF}.csv` (nested, preferred) or
  `data/{SYMBOL}_{TF}.csv` (flat, legacy)
- Causal slicing: primary TF slice = rows where `datetime <= cursor`;
  higher TFs (H1, H4, D1) = rows whose OPEN time ≤ `cursor - tf_interval`
- Reuses the same indicator chain (`add_canonical_indicators → ExtendedIndicators → Indicators`)
- Computes real MTF bias via causal H1→4H/1D resampling (no look-ahead)
- Exposes CSV-derived `spread_pips` via `market_out["ind_ctx"]["spread_pips"]`

### Data flow shape (parity contract)
```
Live:     LiveMT5Provider     → MarketAgent.run → MT5 ticks (real-time)
Backtest: HistoricalCSVProvider → CSV bars (replay)
                ↓
  DataProvider.get_market_out(symbol, tf) → market_out dict
                ↓
  AITrader.evaluate_decision_core → SAME analysis/risk/permission gates
                ↓
Live:     MT5ExecutionAdapter  → ExecutionRouter → OrderManager → mt5.order_send
Backtest: HistoricalExecutionAdapter → BrokerSimulator.open_trade (bar high/low touch)
```

---

## 10. Regime / S-R / liquidity / volatility modules in cycle

Called from `AnalysisAgent.run(market_out)` (`agents/analysis_agent.py:192`):

| Module | Class | Stage in analysis_agent.py | Purpose |
|--------|-------|----------------------------|---------|
| `analysis/market_regime.py` | `MarketRegimeDetector` | Pre-step (in MarketAgent.run, `:301-309`) | LOW/NORMAL/HIGH_VOLATILITY + direction + market_state |
| `analysis/support_resistance.py` | `SupportResistance` | Step 2 (`:410`) | Support/resistance zones, `dist_to_support_pips` / `dist_to_resistance_pips` for S/R gate |
| `analysis/liquidity.py` | `LiquidityEngine` | Step 2b (`:416`) | Equal highs/lows + sweep detection |
| `analysis/liquidity_zones.py` | (registered via orphan) | enrichment | Liquidity zone detection |
| `analysis/liquidity_structure.py` | (registered via orphan) | enrichment | Structural liquidity (order blocks + liquidity pools) |
| `analysis/volatility.py` | `VolatilityEngine` | Step 8.4 (`:854`) | Volatility regime + Bollinger squeeze |
| `analysis/structure.py` | `MarketStructureEngine` | Step 8.1 (`:765`) | BOS / CHoCH |
| `analysis/structure_mtf.py` | `MTFStructureEngine` | Step 8.95 (`:943`) | H4/H1 internal vs external bias |
| `analysis/smc_engine.py` | `SMCEngine` | Step 8 (`:729`) | Smart Money Concepts (order blocks, FVG, BOS) |
| `analysis/smc_advanced.py` | `SMCAdvancedEngine` | DISABLED 2026-07-30 (`:871`) | (constructed but not run) |
| `analysis/ichimoku.py` | `IchimokuEngine` | DISABLED 2026-07-30 (`:815`) | (constructed but not run) |
| `analysis/volume_profile.py` | `VolumeProfileEngine` | DISABLED 2026-07-30 (`:865`) | (constructed but not run) |
| `analysis/divergence.py` | `DivergenceEngine` | Step 8.2 (`:804`) | RSI/MACD divergence |
| `analysis/intermarket.py` | `IntermarketEngine` | Step 8.5 (`:925`) | DXY, Gold, Oil, US10Y, S&P500, VIX, risk-on/off |
| `analysis/currency_strength.py` | `CurrencyStrengthEngine` | Step 6.5 (`:669`) | 28-pair MT5 matrix + ranking |
| `analysis/correlation_engine.py` | `CorrelationEngine` | Step 8.95 (`:1094`) | Correlation_risk, risk_adjustment (consumed by RiskEngine) |
| `analysis/institutional_flow.py` | `InstitutionalFlowEngine` | Step 8.96 (`:1137`) | COT + displacement |
| `analysis/microstructure.py` | `MicrostructureEngine` | Step 8.975 (`:1168`) | MT5 tick analysis (own session!) |
| `analysis/stop_hunt_direct_lane.py` | (called from trader.py:1088) | Post-DecisionAgent | Validated standalone signal — overrides blend when blend is WAIT |
| `analysis/follow_through_engine.py` | `FollowThroughEngine` | Step 8.15 (`:777`) SHADOW ONLY | Logs but does not affect signal |
| `analysis/adaptive_decision_engine.py` | (called from trader.py `_orphan_consumers`) | Post-AnalysisAgent | Backtest-calibrated scoring |
| `analysis/market_dna_service.py` | (called from MarketAgent:314) | enrichment | Unsupervised cluster context |
| `analysis/session_analyzer.py` | `SessionAnalyzer` | Step 0 (`:243`) | Asian/London/NY session + dead-zone + pair priority |
| `analysis/fibonacci.py` | `FibonacciEngine` | DISABLED 2026-07-31 (`:510`) | (constructed but not run) |
| `analysis/market_bias.py` | `MarketBiasEngine` | DISABLED 2026-07-31 (`:517`) | (constructed but not run) |
| `analysis/patterns.py` | `PatternDetector` | Step 1 (`:398`) | Candlestick patterns |
| `analysis/advanced_patterns.py` | `AdvancedPatternDetector` | Step 3 (`:474`) | Harmonic / advanced PA |
| `analysis/sentiment.py` | `SentimentEngine` | Step 7 (`:684`) | Retail + news sentiment |
| `analysis/sentiment_data.py` | `SentimentDataProvider` | Step 7 (provider) | yfinance/COT cached data |
| `analysis/retail_sentiment.py` | (registered) | Step 8.92 (`:1061`) | OANDA → Myfxbook → synthetic |
| `analysis/news_api_provider.py` | (registered) | Step 8.65 (`:879`) | NewsAPI.org sentiment |
| `analysis/stop_hunt_signal_engine.py` | (in stop_hunt_direct_lane) | Post-blend | Stop-hunt detection |
| `analysis/supertrend.py` | (orphan) | enrichment | Supertrend indicator |
| `analysis/order_block.py` | (orphan) | enrichment | Order block detection |
| `analysis/fvg_detector.py` | (orphan) | enrichment | Fair Value Gap detection |
| `analysis/breaker_block.py` | (orphan) | enrichment | Breaker block detection |
| `analysis/flip_zones.py` | (orphan) | enrichment | Flip zone detection |
| `analysis/supply_demand_zones.py` | (orphan) | enrichment | Supply/demand zones |
| `analysis/quantitative_factors.py` | (orphan) | enrichment | Quant factor model |

---

## 11. Key findings

### 11.1 Confirmed
1. **README chain claim VERIFIED** with the nuance that two more veto layers
   (Devil's Advocate + ApprovalMode) sit between `evaluate_decision_core()`
   and `mt5.order_send`. The terminal MT5 call is at
   `broker/order_manager.py:430`.
2. **21 sequential gates** stand between "analysis says BUY" and "broker
   receives order_send" (see §5 + post-permission + veto layers).
3. **`SafetyGuard` (`broker/safety_guard.py`) is dead code in the live path** —
   constructed by `core/runtime.py:boot_safety` Phase 13 but never invoked.
   `AITrader` calls `TradePermission.check()` + `CorrelationFilter.allow()`
   inline. `SafetyGuard` is a wrapper that re-implements the same logic.
4. **`paper` execution mode has been removed** — config default is now
   `mt5_demo`; `SIMULATION_MODE=true` activates `SimulatedExecutor` instead.
5. **Backtest shares `evaluate_decision_core()` with live** — no separate
   decision code; only the data source (CSV vs MT5) and fill mechanism
   (BrokerSimulator vs OrderManager) differ. This is the execution-parity
   guarantee.
6. **MT5 calls are heavily consolidated through `MT5Connection`** (locked
   wrappers in `broker/mt5_connection.py`) for thread safety — except
   `analysis/microstructure.py`, `system/watchdog.py`, and
   `system/network_monitor.py`, which build their own throwaway MT5 sessions
   (a known thread-safety hazard noted in `data/data_orchestrator.py:91-117`).

### 11.2 Risks / next actions
1. **Dead `SafetyGuard`** — either delete it or wire `AITrader` to use it
   instead of inlining TradePermission + CorrelationFilter. Currently a
   maintenance trap (operators may believe the gate is active).
2. **Duplicate MT5 sessions** in `analysis/microstructure.py:151`,
   `system/watchdog.py:155`, `system/network_monitor.py:223` — each calls
   `mt5.initialize()` independently, racing the shared `MT5Connection.MT5_LOCK`.
   These should be routed through the shared connection (same pattern fixed
   in `data/data_orchestrator.py:88-185`).
3. **Stop Hunt Direct Lane bypasses the blend** — when blend produces WAIT,
   this lane fires on its own signal with its own entry/SL/TP
   (`core/trader.py:1086-1122`). TradePermission then skips blend-only
   gates for it. Verified safe by `backtest/per_strategy_tester.py` but
   carries risk if the validator's filters drift.
4. **`DEVILS_ADVOCATE_FAIL_OPEN` env var** defaults to `false` (fail-closed)
   — correct for production. Operators must not enable this in live.
5. **`BYPASS_NEWS_GATE` and `BYPASS_FUSION_GATE` env vars** — both default
   to `false`; only operators set them. `trade_permission.py` honors them
   with logged warnings.
6. **`MT5_STRUCTURE_SOFTEN` env var** — defaults to `false` (correctly
   hard-blocks per the B4c fix); `true` softens the MTF structure gate.

---

## 12. Per-stage input/output contract table

| Stage | Module.method | Input shape | Output shape | Side effects |
|-------|---------------|-------------|--------------|--------------|
| Market fetch | `LiveMT5Provider.get_market_out` | `(symbol: str, timeframe: str)` | `{df, ind_ctx, regime, regime_ctx, mtf_bias, symbol, timeframe, data_source}` | Reads from MT5 |
| PaperTrader update | `PaperTrader.update_price` | `(pair, price, high, low)` | `list[closed_trade_dicts]` | Mutates open_positions, balance |
| CircuitBreaker | `CircuitBreaker.allow_trade` | `()` | `{allowed, mode, reason, stats}` | Reads state file |
| Analysis | `AnalysisAgent.run` | `(market_output, memory_ctx=None)` | `{df, ind_ctx, regime, mtf_bias, signal, signal_ctx, llm, master_ctx, master_decision, confluence, sr_result, sr_ctx, smc_ctx, structure_ctx, mtf_structure_ctx, liquidity_ctx, news, session, news_ctx, session_ctx, intermarket_ctx, correlation_ctx, volatility_ctx, divergence_ctx, ichimoku_ctx (None), retail_sentiment_ctx, fred_ctx, econ_calendar_ctx, institutional_flow_ctx, economic_surprise_ctx, microstructure_ctx, forecast_ctx, currency_strength_ctx, final_signal, llm_signal, mtf_bias, volatility, ...}` (see `analysis_agent.py:2668+`) | Calls ~17 analyzers |
| Decision | `DecisionAgent.decide` | `(market_out, analysis_out, risk_out={placeholder})` | `{decision: BUY/SELL/WAIT, confidence: float, entry, sl, tp, lot, rr, reasons: [], strategy, raw_signal, execution_action, ind_ctx, mtf_trends, mtf_stale_tfs, sr_ctx, structure_ctx, smc_ctx, liquidity_ctx, regime, fast_path, _db, _df, _symbol, market_bias, ...}` | Calls SignalFusion + MasterDecision |
| Risk | `RiskEngine.evaluate` | `(signal, entry, atr, regime, correlation_ctx)` | `{approved, lot, sl_pips, tp_pips, sl_price, tp_price, rr_ratio, risk_usd, risk_pc, reject_reason, signal, entry, atr, volatility_mult, instrument_mult, risk_usd_intended, risk_pc_intended, risk_usd_max_by_lot, risk_pc_max_by_lot, correlation_risk_score, correlation_adjustment, is_placeholder}` | Reads `memory/daily_risk.json` |
| Permission | `TradePermission.check` | `(decision_out, risk_out, news_ctx, session_ctx, execution_filters, bypass_checks, symbol)` | `{allowed, execution_allowed, passed, total, checks: [], failed_checks: [], blocked_reason, execution_action, final_action, entry, sl, tp, lot, rr, confidence_pre_penalty, confidence_post_penalty, min_confidence_diagnostic, entry_quality_detail, risk_requested_pc, risk_requested_usd, risk_max_by_lot_pc, risk_max_by_lot_usd}` | Reads `_recent_entries` cache |
| DA review | `DevilsAdvocateGate.review` | `(trade_context, signal, risk_out, decision_out)` | `{decision: TAKE/REJECT, confidence, reasons_for_concern, risk_summary, evidence, raw_decision, thesis_quality, counter_evidence_strength, expected_edge, risk_level, data_quality, critical_failure}` | LLM call |
| Approval | `ApprovalMode.process` | `({symbol, final_action, confidence, entry, sl, tp, lot, rr, llm_analysis})` | `{proceed, mode, action, message, pending_id?}` | Persists pending approvals to DB |
| Execute | `MT5ExecutionAdapter.open_trade` | `(symbol, direction, entry_price, sl, tp, lot, confidence, **kwargs)` | `{id, status, broker_symbol, ticket, entry, sl, tp, lot, type, pair, pending?}` | Calls mt5.order_send |
| Final MT5 call | `OrderManager.place_market_order` | `(symbol, direction, lot, sl, tp, comment)` | `{success, ticket, retcode, price, volume, reason?, partial_fill?, duplicate_prevented?}` | ★ `mt5.order_send(request)` ★ |

---

**End of P1-B audit.**
