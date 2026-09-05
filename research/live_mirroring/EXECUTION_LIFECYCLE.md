# Canonical Execution Lifecycle

## Research replay path

```text
AITrader.evaluate_decision_core()
  -> analysis_out
  -> dec_out
  -> risk_out
  -> perm_out
  -> CanonicalHistoricalExecutionAdapter
  -> PositionLifecycle
  -> HistoricalPositionMonitor
  -> close detection
  -> deterministic P/L
```

The live path remains:

```text
AITrader -> ExecutionRouter -> MT5 -> broker/server position lifecycle
```

The replay difference is strictly the execution backend. The complete live
`decision_out` plus risk-derived `sl_price`, `tp_price`, and `lot` are passed
to the canonical adapter. The adapter does not resize, reinterpret, or invent
an entry/SL/TP decision.

## Entry rules

- `entry_time >= signal_time` is mandatory.
- Entry timestamps must be timezone-aware.
- Historical BID is supplied by the replay clock/data provider.
- BUY pays the historical spread to ASK plus explicit adverse slippage.
- SELL executes at historical BID minus explicit adverse slippage.
- No random fills or partial-fill probabilities are used.
- No next-bar-open fallback is used by the strict runner.

## Position monitoring

`HistoricalPositionMonitor` consumes one historical bar at a time.

- BUY: low touches SL / high touches TP.
- SELL: high touches SL / low touches TP.
- If both are touched and no ordering source exists, the result is
  `AMBIGUOUS_INTRABAR` rather than silently choosing a winner.
- `WORST_CASE` / `CONSERVATIVE` chooses SL; `BEST_CASE` chooses TP only when
  explicitly requested by the research configuration.
- Exit time must be >= entry time.
- Market-close exits use the supplied historical BID and explicit spread /
  slippage assumptions.

## P/L

P/L is calculated from actual replay fill price to actual historical exit
price using an explicitly supplied `pnl_multiplier` (for example the broker's
contract-size/value conversion). Commission is separately recorded and
subtracted from realized P/L. No implicit default contract value is used.

## Isolation

The bridge owns only replay positions. It does not write live MT5 positions,
live DB state, live memory, circuit-breaker state, or live learning state.

## Current evidence

**PARTIALLY PROVEN** at this stage.

Proven by source structure:
- canonical adapter exists at the execution boundary;
- complete live decision/risk/permission payload can be passed unchanged;
- deterministic fill calculation exists;
- timestamp ordering is enforced;
- historical close detection and explicit intrabar ambiguity exist;
- deterministic P/L and commission accounting exist;
- focused unit tests cover these invariants.

Still required before a PROVEN verdict:
- wire this bridge into the production replay runner's actual per-bar call
  site (rather than maintaining the legacy `BrokerSimulator` path);
- run the focused tests and a real short historical replay in the target
  environment;
- verify live-vs-replay payload parity against the same timestamp;
- add MAE/MFE and portfolio-level lifecycle accounting.
