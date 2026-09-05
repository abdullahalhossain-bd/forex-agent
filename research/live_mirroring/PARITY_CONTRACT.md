# Live-Trading-Mirroring Replay — Parity Contract

## Source of truth
The live trading implementation is authoritative. Historical replay must reuse live decision-critical objects and may not introduce an independent strategy.

## Allowed differences
Only these boundaries may differ from live:
- market data source (historical replay instead of live MT5 feed)
- execution backend (historical simulator instead of broker execution)
- clock (deterministic replay clock instead of wall clock)
- external historical context source (only timestamped historical data; otherwise explicitly disabled)

## Forbidden during replay
No future candle, indicator, MTF candle, news/sentiment result, model output, trade outcome, balance/P&L, calibration statistic, or other information unavailable at replay timestamp T may influence a decision at T.

## Higher-timeframe rule
A higher-timeframe candle is usable only after its close timestamp. For an M15 decision at 10:45, an H1 candle opened at 10:00 is unavailable; an H1 candle opened at 09:00 is available after 10:00.

## External data policy
Current/live external data must never silently enter historical decisions. Economic calendar, current news, current sentiment, current macro APIs, and similar sources must be explicitly marked DISABLED_BACKTEST unless a timestamped historical dataset is supplied.

## Execution policy
Historical execution must evaluate OHLC intrabar touches. If both SL and TP are touched without ordering information, the result is AMBIGUOUS_INTRABAR and must not silently select the profitable outcome.

## Integrity
No threshold, indicator, SL/TP, confidence, filter, session, pair-profile, or strategy optimization is part of this project. Baseline behavior is measured as-is.

## Verdict vocabulary
Every parity audit uses only: PROVEN, PARTIALLY PROVEN, UNKNOWN, FAILED. Parity is never claimed without evidence.
