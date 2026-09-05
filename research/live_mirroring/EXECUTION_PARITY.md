# Execution Parity Contract

## Source of truth
Live order construction, risk, permission, and position-management logic remain authoritative. Historical replay may replace only the MT5 execution backend.

## Canonical order lifecycle
1. Decision timestamp `T` is frozen.
2. RiskEngine produces the live risk result.
3. PositionSizer produces the live final lot.
4. Permission produces PASS/FAIL and reason.
5. ExecutionRouter receives the unchanged approved order payload.
6. Historical adapter selects a fill timestamp `>= T`.
7. Fill uses historical bid/ask when available; otherwise the missing-side assumption is explicitly recorded.
8. Filled lot is recorded separately from requested lot.
9. Position remains OPEN until a historical exit event occurs.
10. SL/TP touch is resolved according to an explicit intrabar policy.
11. Exit price, costs and P/L are recorded.
12. Close state is isolated from live DB/memory/learning/circuit-breaker state.

## Forbidden
- `datetime.now()`, `time.time()`, `today()` for trading decisions or fills.
- Random slippage or random partial fills in canonical replay.
- Future-bar prices for a signal-time decision.
- Current spread/news/sentiment/macro data.
- Silent fallback from missing historical bid/ask to live/current values.
- Changing risk, lot, SL, TP, permission or strategy thresholds.

## Intrabar
If OHLC data shows both SL and TP touched and no ordering source exists, the result is `AMBIGUOUS_INTRABAR`; the engine must not silently choose a profitable outcome. `WORST_CASE` and `BEST_CASE` are explicit research scenarios only.

## Evidence status
This contract defines the required backend boundary. Integration into the existing ExecutionRouter and live position-monitoring call chain remains **PARTIALLY PROVEN** until source-level parity tests exercise the real router payload end-to-end.
