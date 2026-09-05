# Live-Trading-Mirror Backtest / Historical Replay — Parity Contract

This document is the source-level audit checklist. A result is **PARITY-PROVEN** only when the corresponding live behavior and historical behavior are exercised by the same code path or by an explicitly documented execution-only substitute.

## P0 — correctness / no-false-edge

| Area | Requirement | Status |
|---|---|---|
| Historical input | UTC-aware timestamps | FIXED in `backtest.live_mirror` |
| Historical input | Strictly increasing timestamps | FIXED |
| Historical input | No duplicate timestamps | FIXED |
| Historical input | Valid OHLC geometry | FIXED |
| Decision timing | Decision only sees information available at signal bar | EXISTING + VERIFY |
| Higher TF | Only closed higher-TF bars available | EXISTING in CSV provider + VERIFY |
| Wall clock | Historical decision cannot use machine `datetime.now()` | PARTIAL — audit remaining modules |
| News | No current/future news injected into old bars | FIXED to explicit neutral when historical data unavailable |
| Sentiment | No today's retail/F&G/DXY data in historical replay | FIXED to explicit neutral when historical data unavailable |
| FRED/macro | No current macro values attached to historical bars | BACKTEST MODE GATE EXISTS; VERIFY ALL CALLERS |
| Session | Session calculation uses replay timestamp | FIXED / existing `dt` path |
| TEST_MODE | Cannot force-approve rejected sizing | FIXED in strict facade |
| Memory | Backtest state isolated from live memory | FIXED / isolated memory reset |
| RNG | Slippage/partial-fill replay deterministic | FIXED with seed in unified runner |
| Risk | RiskEngine remains authoritative for SL/TP/lot | FIXED: no ATR fallback in strict runner |
| Fill timing | Market signal on bar i cannot fill on bar i close | FIXED: next-bar open boundary |
| Last bar | No synthetic last-bar market fill | FIXED: no-next-bar rejection |

## P0 — risk parity

- Same `RISK_PER_TRADE` source as live.
- Same risk USD calculation.
- Same pip size / pip value / contract size.
- Same symbol digits and price normalization.
- Same minimum/maximum SL distance.
- Same RR policy.
- Same lot normalization.
- Same `MAX_LOT` cap.
- Same daily-loss gate.
- Same max-open-trades gate.
- Same correlation context and correlation gate.
- Same TradePermission result.
- Same rejection reason classification.
- No backtest-only risk fallback.

Current status: **core risk path is shared; multi-position exposure synchronization and time-based trade-frequency parity remain P0 work.**

## P0 — execution parity

Live:

`TradeDecision → ExecutionRouter.execute() → broker OrderManager → MT5 fill`

Historical target:

`TradeDecision → HistoricalExecutionRouter → BrokerSimulator → OHLC-aware fill`

Requirements:

- Shared decision validation.
- Shared BUY/SELL hard gate.
- Shared explicit permission gate.
- Shared lot-cap validation.
- Shared positive-lot validation.
- Shared entry-price validation.
- Same pullback/limit routing policy.
- Historical market order: next-bar open, never signal-bar close.
- Historical pending limit: remain pending until touched or expired.
- Limit-order expiry measured by replay time, not machine time.
- Historical fill must preserve requested SL/TP levels from RiskEngine.
- Broker symbol normalization must match live mapping.
- Spread-limit behavior must match live.
- Price digits must match live.
- Commission must be applied to filled lot.
- Slippage model must be deterministic for reproducibility.
- BUY/SELL bid/ask semantics must be explicit.
- Same-bar SL+TP ambiguity must have an explicit deterministic policy.
- Partial fills must update actual filled lot, commission and P/L.
- Rejected execution must not become a trade.

Current status: **BrokerSimulator correctness fixes exist, but the backtest still does not traverse the real ExecutionRouter. Historical execution-router work is the remaining P0 boundary.**

## P0 — position lifecycle

- One canonical simulated position state.
- Open positions visible to duplicate-position checks.
- Open positions visible to correlation checks.
- Entry recorded at actual simulated fill time.
- SL/TP monitored after fill, not before.
- Pending orders are distinct from open positions.
- Close detection is deterministic.
- Timeout close uses replay timestamp.
- End-of-data close is explicit and excluded from normal signal statistics where appropriate.
- Realized P/L updates account equity exactly once.
- Commission/spread/slippage included exactly once.
- Closed trade is removed from exposure state exactly once.
- Multi-symbol replay shares one portfolio state when testing portfolio limits.

Current status: **single-symbol broker state works; `_paper` and BrokerSimulator are not yet one canonical exposure state.**

## P0 — time / replay clock

All time-sensitive behavior must consume `ReplayClock` in historical mode:

- session selection
- daily loss/day boundary
- trade frequency
- pending-order expiry
- cooldowns
- circuit breakers
- stale-order cancellation
- news windows
- economic-calendar windows
- rolling event windows
- learning/memory timestamps when they affect decisions

Known blocker: `risk.trade_frequency` directly uses `datetime.now(timezone.utc)` and a global singleton. It needs replay-clock injection rather than being silently skipped. See `backtest/replay_clock_adapter.py` for the deterministic state-machine foundation.

## P1 — analysis stack

Audit every AnalysisAgent module for:

1. wall-clock reads
2. live network/API reads
3. current-date/current-session defaults
4. unbounded caches
5. future-row access
6. full-dataframe statistics computed before the replay cursor
7. resampling that includes an unclosed higher-timeframe bar
8. external sentiment/news/macro values without historical timestamps
9. model features trained/fit using future rows
10. persistent memory from previous live/backtest runs

No module should be allowed to silently substitute today's value for an unavailable historical value. It must either consume timestamped historical data or return an explicit unavailable/neutral result.

## P1 — decision / LLM parity

- Same DecisionAgent.
- Same MasterAnalyst.
- Same Devil's Advocate logic.
- Same prompts/templates where applicable.
- Same model/provider selection policy, unless provider is unavailable offline.
- Same confidence calibration.
- Same confidence caps/penalties.
- No historical prompt may contain future market/news data.
- LLM outputs must be captured in forensic logs.
- Replay must record model/provider/version for reproducibility.
- If a live external LLM is unavailable, the run must fail or use an explicitly configured deterministic substitute; it must not silently invent a different strategy.

## P1 — data-provider parity

- Same canonical indicator chain.
- Same lookback length.
- Same column normalization.
- Same symbol specifications.
- Same spread source when historical spread exists.
- Same volume semantics.
- Same timeframe labels.
- Same higher-TF close alignment.
- Same missing-data policy.
- No future candle in `market_out`.

`core/csv_data_provider.py` already defines a strict anti-look-ahead contract for primary and higher timeframes.

## P1 — accounting / metrics

- Starting balance isolated.
- Balance/equity distinction explicit.
- Realized vs unrealized P/L explicit.
- Commission exactly once.
- Spread cost exactly once.
- Slippage exactly once.
- Partial-fill lot reflected in P/L.
- Same currency conversion rules as live.
- Profit factor based on realized closed trades.
- Win rate based on closed trades.
- Drawdown derived from the canonical equity curve.
- End-of-backtest liquidation labeled separately.
- Rejected signals never counted as trades.

## P1 — forensic evidence

Every executed trade should retain:

- signal timestamp
- fill timestamp
- signal bar index
- fill bar index
- decision
- confidence
- requested entry
- actual fill
- SL
- TP
- requested lot
- filled lot
- risk USD
- analysis output
- decision output
- risk output
- permission output
- execution route
- spread
- slippage
- commission
- exit timestamp
- exit reason
- exit price
- realized P/L
- OHLC path from fill to exit

The strict facade already captures most of this; risk/execution route fields should be extended as the execution router is unified.

## P1 — reproducibility

A replay is reproducible when identical:

- code commit
- config snapshot
- input data hash
- model versions
- random seed
- execution assumptions
- symbol specifications

produce identical decision/trade/metric output.

## P2 — operational quality

- Resume after Ctrl+C without corrupting state.
- Atomic checkpointing.
- Per-bar progress logging.
- Per-stage rejection counters.
- Final parity report.
- Data-quality report.
- Future-leak detector.
- Wall-clock dependency detector.
- Live-vs-replay decision diff tool.
- Trade-by-trade execution diff tool.

## Explicit non-goals

- No threshold optimization.
- No strategy redesign.
- No parameter tuning to improve PF/WR.
- No removing losing trades.
- No changing SL/TP policy to improve historical results.
- No look-ahead data.
- No current news/sentiment substituted as historical truth.

## Acceptance gate

The Live-Trading-Mirror Backtest is **NOT production-validation-grade** until all P0 items above are closed and a deterministic parity test proves that a captured live decision can be replayed through the same risk/permission/execution policy with only the allowed historical fill/data-source differences.
