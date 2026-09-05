# Execution Integration Status

## Canonical path

```text
AITrader.evaluate_decision_core()
  -> decision_out
  -> risk_out
  -> permission_out
  -> Execution boundary
  -> CanonicalHistoricalExecutionAdapter
  -> PositionLifecycle
  -> HistoricalPositionMonitor
  -> SL/TP or market close
  -> deterministic P/L
```

## Source-of-truth rule

`execution/execution_router.py` remains the live order-construction and broker-execution source of truth. Historical replay is permitted to replace only the broker/MT5 side. The replay adapter must receive the same decision/risk/permission payload rather than rebuilding strategy logic.

## Critical parity guarantees

- `decision`, `entry`, `sl`, `tp`, and `lot` are preserved from the live decision/risk output.
- `historical_bid` is the only market observation used to construct the replay fill.
- BUY fill = historical BID + historical/declared spread + adverse slippage.
- SELL fill = historical BID - adverse slippage.
- No random fill, partial-fill probability, or wall-clock fill.
- `entry_time >= signal_time` is enforced.
- Exit time cannot precede entry time.
- SL/TP detection consumes one historical bar at a time.
- If both SL and TP are touched without ordering information, the result is `AMBIGUOUS_INTRABAR` unless an explicit BEST_CASE/WORST_CASE policy is selected.
- Market close uses supplied historical BID and declared spread/slippage only.
- Commission is explicit and deducted from realized P/L.
- Replay positions live only in the isolated adapter state; they are never inserted into live MT5 state.

## Known integration boundary

The existing `backtest/unified_engine.py` still contains its legacy `BrokerSimulator` execution loop. The strict research runner is `backtest/live_mirroring_runner.py`, which uses the canonical lifecycle. Until the legacy runner is removed or made a thin delegating wrapper, the repository verdict for full execution integration remains **PARTIALLY PROVEN**.

No strategy thresholds, indicators, risk parameters, SL/TP formulas, confidence gates, or filters were changed as part of this execution integration.
