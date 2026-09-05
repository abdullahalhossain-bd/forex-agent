# FOREX-AGENT — P0 Live Call-Chain Reconstruction

Branch: `research/live-mirroring-replay`

This document records the call chain from executable code, not from architecture comments.

## 1. Entry point

`main.py::_run_backtest(args)` → `backtest.unified_engine.run_unified_backtest(...)`.

The unified backtest constructs a real `core.trader.AITrader`, rather than a separate strategy object. The historical loop advances the provider and invokes `trader.evaluate_decision_core(...)`. This is the correct parity boundary for Analysis → Decision → Risk → Permission.

## 2. Live runtime object graph

`main.py` → `core.trading_engine.TradingEngine` → `core.trader.AutonomousTraderSystem` → per-symbol `AITrader` → `AITrader.run_cycle()` → `AITrader.evaluate_decision_core()` → analysis / decision / risk / permission / DA → execution router → broker adapter → MT5 or paper backend → position monitoring / close detection → database + logs.

`TradingEngine` subclasses `AutonomousTraderSystem`; it is not a separate wrapper engine.

## 3. Decision-critical calls observed in source

| Stage | Caller | Callee / method | Primary input | Output / mutation | External dependency |
|---|---|---|---|---|---|
| Market data | `AITrader` cycle | `self._data_provider.get_market_out(...)` / live `MarketAgent.run()` | symbol, timeframe | `market_out` | MT5 in live; historical provider in replay |
| Analysis | `evaluate_decision_core()` | `self._analysis.run(market_out)` | market context | `analysis_out` | multiple analyzers; some external sources |
| Memory | `evaluate_decision_core()` | `self._memory.get_context_for_ai(...)` | symbol / prior trades | `memory_ctx`, `pat_ctx`, `vec_ctx` | isolated DB/memory required in replay |
| Decision | `evaluate_decision_core()` | `self._decision.decide(market_out, analysis_out, placeholder_risk)` | market + analysis | `dec_out` | LLM/master-analysis may be downstream of analysis |
| Risk | `evaluate_decision_core()` | `self._risk.evaluate(...)` | decision, price, SL/TP context | `risk_out` | balance/state |
| Position sizing | risk/sizing stage | live position-sizer path inside `AITrader` | `risk_out` + market state | mutates `risk_out[lot]`, risk fields | account/balance state |
| Permission | `evaluate_decision_core()` | `self._perm.check(...)` | decision + risk | `perm_out` | session/spread/news/duplicate/etc. |
| Safety | `AITrader` / safety guard | `TradePermission` + correlation / circuit breaker / approval gates | final candidate | allow/block mutations | global state in live |
| Devil's Advocate | trade decision path | `self._devils_advocate.review(...)` | trade context, final action, risk | review / veto decision | LLM/config dependent |
| Execution | `AITrader` | `self._router` / execution adapter | approved decision | execution result | MT5 or simulated backend |
| Position monitoring | `AITrader` | position manager / paper update / MT5 close detection | open positions + current market | close/P&L state | MT5 live; historical OHLC replay required |
| Persistence | execution / close paths | `TraderDB` + memory/logging | trade/event records | DB rows/files/logs | must be isolated in replay |

## 4. Critical parity finding

The source confirms that `AITrader.evaluate_decision_core()` is the shared decision implementation. `backtest/unified_engine.py` explicitly calls that method. Therefore the backtest is structurally on the live decision core, but this alone does **not** prove complete live equivalence.

Known parity gaps that remain P0/P1 work:

1. External analysis providers must be proven to return no current/future data in replay.
2. Every decision-critical wall-clock read must resolve through `ReplayClock`.
3. Historical spread/bid/ask and broker symbol specifications must be timestamped or explicitly marked `ASSUMED`.
4. Live position state and portfolio constraints must be represented in the replay state, not merely inferred from a separate simulator.
5. LLM and Devil's Advocate requests require deterministic cache keys and replay metadata.
6. Every gate must emit an auditable PASS/FAIL/NOT_REACHED result.

## 5. Verdict vocabulary

- **PROVEN** — source + tests demonstrate equivalence.
- **PARTIALLY PROVEN** — shared implementation exists but one or more dependencies differ.
- **UNKNOWN** — not enough evidence.
- **FAILED** — a parity rule is demonstrably violated.

Current overall verdict: **PARTIALLY PROVEN**. The shared decision core is proven structurally; full live-trading mirroring is not yet proven.
