# P0 External Context Policy

Historical replay must never silently substitute today's external context for historical context.

## Required replay statuses

| Source | Backtest policy | Required evidence |
|---|---|---|
| Economic calendar | `DISABLED_BACKTEST` unless timestamped historical archive exists | provider returns explicit disabled/available status |
| Live news API | `DISABLED_BACKTEST` unless timestamped historical archive exists | no network fallback to current headlines |
| Current sentiment | `DISABLED_BACKTEST` unless timestamped historical archive exists | source timestamp <= replay timestamp |
| External macro/FRED | `DISABLED_BACKTEST` unless timestamped historical observations exist | observation timestamp <= replay timestamp |
| Intermarket live feeds | `DISABLED_BACKTEST` unless historical series exists | timestamped historical series |
| MT5 current tick/account state | `DISABLED_BACKTEST` | historical replay state only |

## Current repository evidence

`backtest/unified_engine.py` calls `set_backtest_mode(True)` before constructing the replay trader. The repository also contains `is_backtest_mode()` and multiple external modules already branch on it, including the economic-calendar path and microstructure path.

However, a global backtest flag is not sufficient evidence by itself. Every AnalysisAgent dependency must be classified and tested so that a module cannot silently return current data after an exception, cache miss, or fallback.

## Rule for module authors

A replay-safe module must return one of:

- `HISTORICAL` — source data has an explicit timestamp <= replay clock.
- `DERIVED_HISTORICAL` — deterministic calculation from historical inputs.
- `DISABLED_BACKTEST` — live-only source intentionally not used.
- `ASSUMED` — explicitly configured assumption, never disguised as observed data.
- `UNSAFE` — source cannot establish temporal validity; trade decision must not consume it.

Never return an unlabelled live/current value from a replay path.
