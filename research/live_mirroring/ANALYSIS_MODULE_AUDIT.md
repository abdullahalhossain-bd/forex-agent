# P0-2 AnalysisAgent Source-Level Audit

Branch: `research/live-mirroring-replay`

## Scope

`agents/analysis_agent.py` is the live source of truth. This audit records whether each analysis dependency is safe for historical replay. A module is not considered replay-safe merely because it returns an empty/default value in backtest mode; the source dependency and timestamp semantics must be explicit.

## Classification

- **A — PURE**: deterministic from supplied historical state; no wall clock/external I/O.
- **B — HISTORICAL**: external-ish dependency, but can be safely supplied from timestamped replay data.
- **C — LIVE EXTERNAL**: current news/calendar/sentiment/macro/network/broker dependency; disabled unless timestamped historical source is injected.
- **D — WALL CLOCK**: decision can depend on real current time/date/session.
- **E — UNKNOWN/UNSAFE**: source semantics are not proven safe; disable until audited.

## Source inventory from `agents/analysis_agent.py`

| Dependency | Class | Replay policy | Reason |
|---|---|---|---|
| PatternDetector | A | ENABLE | Historical OHLC-derived pattern analysis |
| SupportResistance | A/B | ENABLE* | Safe only when all levels derive from data <= T |
| MarketBiasEngine | A/B | ENABLE* | Must consume replay-bounded frame |
| AdvancedPatternDetector | A/B | ENABLE* | Must consume replay-bounded frame |
| FibonacciEngine | A/B | ENABLE* | Must consume replay-bounded frame |
| SentimentEngine | C/E | DISABLE | Sentiment source must not silently use current/live data |
| SMCEngine | A/B | ENABLE* | Must consume replay-bounded frame/context |
| SentimentDataProvider | C/E | DISABLE | External sentiment dependency |
| SessionAnalyzer | D | ENABLE via ReplayClock | Session must be derived from replay timestamp, not wall clock |
| IntermarketEngine | B/C | DISABLE unless historical | DXY/gold/oil/US10Y/SPX/VIX must be timestamp-aligned historical data |
| CurrencyStrengthEngine | B/C | DISABLE unless historical | Must not fetch current strength during replay |
| DivergenceEngine | A/B | ENABLE* | Requires replay-bounded OHLC/indicators |
| IchimokuEngine | A/B | ENABLE* | Requires replay-bounded OHLC |
| VolatilityEngine | A/B | ENABLE* | Requires replay-bounded OHLC |
| VolumeProfileEngine | A/B | ENABLE* | Requires replay-bounded volume/tick-volume history |
| SMCAdvancedEngine | A/B | ENABLE* | Requires replay-bounded structure |
| MTFStructureEngine | B | ENABLE* | Higher TF data only through candles closed by T |
| MarketStructureEngine | A/B | ENABLE* | No future bars/centered windows |
| FollowThroughEngine | A/B | ENABLE* | Historical state only; no future outcome labels at decision T |
| ShadowFollowThroughLogger | E | DISABLE for decisions | Observability may not feed decisions or future labels into replay |
| NewsAPI provider | C | DISABLE | Current news cannot represent historical information |
| EconomicCalendarAPI | C | DISABLE | Current calendar explicitly excluded by parity contract |
| FRED API | C/B | DISABLE unless historical snapshot | Latest-value API is not historical replay |
| RetailSentiment API | C | DISABLE | Current retail sentiment is not timestamped historical input |
| CorrelationEngine | B | ENABLE* | Must use historical aligned prices; no future window |
| InstitutionalFlowEngine | C/B | DISABLE unless historical | Current institutional-flow feeds are not valid replay inputs |
| EconomicSurpriseEngine | B/C | DISABLE unless historical | Requires timestamped releases known at T, with publication-time semantics |
| MicrostructureEngine | B/C | DISABLE unless historical ticks | Current broker/tick state is not valid historical context |
| NetworkMonitor | C/D | DISABLE | Network latency/connectivity is live infrastructure state |
| ForecastEngine | B/E | ENABLE only if frozen model + historical inputs | Model output must be reproducible and use only data <= T |
| StrategySelector | A/B | ENABLE* | Strategy selection must be based on replay-bounded context only |
| NewsFilter | C | DISABLE | News/calendar gating is excluded from historical replay |
| AIAnalyst | E | ENABLE only through deterministic replay adapter | LLM/model payload must be timestamped, cached and reproducible |
| MasterAnalyst | A/B/E | ENABLE* | All upstream context must already satisfy replay policy |
| SignalEngine | A/B | ENABLE* | Must receive only replay-bounded analysis/context |

`*` means source-level dependency still requires the generic look-ahead/data-boundary audit before being declared PROVEN.

## Critical source findings

1. `agents/analysis_agent.py` imports multiple live providers directly. Importing a provider is not itself a violation, but every call must be explicitly backtest-gated or supplied a historical adapter.
2. `news_api_provider`, `EconomicCalendarAPI`, retail sentiment, FRED/latest macro, network monitor and live microstructure are not valid historical inputs by default.
3. `SessionAnalyzer` is decision-relevant and therefore must receive replay time. A real `datetime.now()` anywhere in the session path invalidates parity.
4. MTF analysis requires a strict closed-candle rule. At replay timestamp T, an H1/H4/D1 candle whose close is after T cannot contribute to the decision.
5. Follow-through modules must never use the eventual outcome of the current signal to influence the signal itself. Future labels may exist only in post-trade/autopsy analysis.
6. LLM/AI output is not automatically historical-safe. The exact payload, prompt version and model response must be cached and hashable.

## Required enforcement

The replay adapter must expose an explicit context with:

- `replay_timestamp`
- `primary_tf_cutoff`
- `closed_htf_cutoffs`
- `external_context_policy`
- `source_timestamp` per external field
- `assumption_status` (`OBSERVED`, `ASSUMED`, `DISABLED`)

No module may silently fall back to current/live values.

## Verdict

**P0-2 current status: PARTIALLY PROVEN.** The module inventory and obvious live-only dependencies are identified, but every `*` module still requires source-level boundary tests before the whole AnalysisAgent can be marked PROVEN.
