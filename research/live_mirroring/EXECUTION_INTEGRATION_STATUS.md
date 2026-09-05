# Execution Integration Status

## Canonical path

```text
AITrader.evaluate_decision_core()
  -> decision_out
  -> risk_out
  -> permission_out
  -> canonical replay execution boundary
  -> CanonicalHistoricalExecutionAdapter
  -> PositionLifecycle
  -> HistoricalPositionMonitor
  -> SL/TP or market close
  -> deterministic P/L
```

## Source-of-truth rule

`execution/execution_router.py` remains the live order-construction and broker-execution source of truth. Historical replay replaces only the broker-facing execution backend. The replay adapter receives the same decision/risk/permission payload rather than rebuilding strategy logic.

## Critical parity guarantees

- `decision`, `entry`, `sl`, `tp`, and `lot` are preserved from live decision/risk output.
- Historical BID is the authoritative sell-side execution observation; explicit historical ASK is supported for BUY fills.
- When ASK is unavailable, BUY ASK is reconstructed from historical BID + declared full spread.
- SELL fill = historical BID - adverse slippage.
- No random fill or probabilistic partial fill in the canonical adapter.
- `entry_time >= signal_time` is enforced.
- Exit time cannot precede entry time.
- SL/TP detection consumes one historical bar at a time.
- If both SL and TP are touched without ordering information, the result is `AMBIGUOUS_INTRABAR` unless an explicit BEST_CASE/WORST_CASE policy is selected.
- Market close uses supplied historical BID and declared spread/slippage only.
- Commission is explicit and deducted from realized P/L.
- Replay positions live only in isolated adapter state; they are never inserted into live MT5 state.
- The compatibility `backtest/unified_engine.py` entry point now delegates to `backtest/live_mirroring_runner.py`; it no longer contains the legacy `BrokerSimulator` lifecycle.

## Remaining research boundaries

The canonical runner still needs end-to-end execution in the user's trading environment before a PROVEN verdict can be issued. In particular, the repository must demonstrate that the real live runtime's execution boundary can be replaced/injected with the canonical historical backend without changing upstream decision/risk/permission outputs. That is an integration/runtime proof, not a strategy change.

No strategy thresholds, indicators, risk parameters, SL/TP formulas, confidence gates, or filters were changed as part of this execution integration.
