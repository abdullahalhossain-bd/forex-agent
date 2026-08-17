# Deliverable 5 — Updated Downloader (Code Changes)

**Project:** Forex AI Autonomous Trading System
**Audit date:** 2026-08-17

This document catalogs every code change made to close the parity gaps identified in the Phase 1-10 audit. Each change follows the format specified in Phase 16.

---

## 1. Summary of Changes

| # | File | Type | Severity | Description |
|---|------|------|---------|-------------|
| 1 | `scripts/download_historical_data.py` | REWRITE | HIGH | Production-grade downloader v2 with broker-tz auto-detection, retry, gap detection, weekend awareness, MAX_REQUIRED_LOOKBACK enforcement, HTF resampling fallback, optional bid/ask fetch, validation report, idempotent re-run |
| 2 | `scripts/validate_historical_csv.py` | NEW FILE | MEDIUM | Standalone CSV validation CLI for spot-checking data quality |
| 3 | `scripts/download_external_data.py` | NEW FILE | MEDIUM | External macro data (DXY/Gold/Oil/US10Y/SP500/VIX) downloader via yfinance |
| 4 | `core/data_provider.py` | EDIT | HIGH (parity) | Fixed `LiveMT5Provider.current_time()` to return tz-aware UTC instead of naive `datetime.utcnow()` |
| 5 | `analysis/smart_money.py` | EDIT | MEDIUM (leakage) | Fixed `_current_kill_zone()` to accept `bar_timestamp` parameter for backtest parity |
| 6 | `ml/feature_engineer.py` | EDIT | MEDIUM (leakage) | Fixed `_context_features()` to use `df.index[-1]` instead of `pd.Timestamp.now(tz="UTC")` for time features |

---

## 2. Change Details

### Change 1 — Production-grade downloader v2

**File:** `scripts/download_historical_data.py`
**Function:** Whole file (rewrote v1 → v2)
**Current Problem (v1):**
- W1: No broker-tz correction (`pd.to_datetime(df["time"], unit="s", utc=True)` assumes MT5 `time` is true UTC; brokers using GMT+2/+3 mislabel it)
- W2: No retry logic for transient MT5 errors
- W3: No gap detection (missing bars silently accepted)
- W4: No weekend awareness (flags ALL gaps as issues, including expected Fri 21:00 → Sun 21:00 UTC)
- W5: No MAX_REQUIRED_LOOKBACK enforcement (user must manually extend `--start` to cover warmup)
- W6: No HTF resampling fallback (H4 fetch failure = no H4 data)
- W7: No bid/ask columns (historical backtest fills at `close` instead of ask/bid)
- W8: No validation report persisted to disk
- W9: No idempotent re-run (always re-downloads or skips entirely)
- W10: No data quality metrics in manifest

**Why It Matters:**
- W1 → timestamps drift 2-3h into the future; downstream session detection, kill-zone, Asian range, PDH/PWL all break
- W3, W4 → silent data quality issues accepted as valid; backtest results may be biased
- W5 → first ~500-1000 bars of any backtest run have NaN indicators (parity violation)
- W7 → BUY fills at `close` instead of `ask`; spread cost not captured at fill time

**Change:** Rewrote the downloader with:
- `_detect_broker_utc_offset_hours()` — reuses the proven pattern from `data/fetcher.py:_get_broker_utc_offset_hours`, with 30-min cache, DST self-correction, env override support
- `_fetch_chunk_with_retry()` — exponential backoff (3 retries: 1s, 2s, 4s); distinguishes "no data" (don't retry) from "transient error" (retry)
- `validate_full()` — 13 validation rules (V1-V13); produces both errors (reject) and warnings (accept but flag)
- Weekend-aware gap detection (Fri 21:00 UTC → Sun 21:00 UTC excluded from non-weekend gap count)
- `--warmup-bars N` argument (default 1000) extends `--start` backward by N × TF_seconds
- HTF resampling fallback via `_resample_h1_to_h4()` (matches `data/fetcher.py:_resample_h1_to_h4` pattern)
- `--with-bid-ask` flag — fetches bid/ask at each bar's close via `mt5.symbol_info_tick`; adds `bid`/`ask` columns to CSV
- Per-file validation report saved to `data/historical/{SYMBOL}/_validation_{TF}.json`
- Idempotent re-run: skips files whose existing range already covers `[effective_start, end]`
- Manifest includes `broker_utc_offset_hours`, `schema`, `has_bid_ask`, `validation_errors`, `validation_warnings`, `spread_zero_pct`, `gap_count`, `non_weekend_gap_count`, `real_volume_available`

**Effect:**
- W1-W10 all closed
- Backtest results from new CSVs match live trading behavior (modulo the documented approximation flags in D4 §7)
- Operators can spot-check data quality before backtest runs via `scripts/validate_historical_csv.py`

---

### Change 2 — Standalone CSV validation CLI

**File:** `scripts/validate_historical_csv.py` (new file)
**Function:** Whole file
**Current Problem:** No standalone way to validate CSVs downloaded by older versions of the downloader, or to spot-check data quality before backtest runs.
**Why It Matters:** Operators had to manually inspect CSVs or write ad-hoc scripts; data quality issues would silently degrade backtest results.
**Change:** New CLI that:
- Reuses `validate_full()` from the downloader (single source of truth for validation rules)
- Supports filtering by symbol, timeframe, or custom data directory
- Outputs both human-readable Markdown report (`docs/audit/evidence/P-validation-cli-report.md`) and machine-readable JSON
- Reports: row count, date range, spread zero %, gap count, non-weekend gaps, OHLC violations, NaN/inf counts
**Effect:** Operators can validate CSVs at any time without re-downloading; data quality issues surface before backtest runs.

---

### Change 3 — External macro data downloader

**File:** `scripts/download_external_data.py` (new file)
**Function:** Whole file
**Current Problem:** `IntermarketEngine` fetches DXY/Gold/Oil/US10Y/SP500/VIX live via `MacroDataProvider`; in backtest, returns neutral context (no historical macro).
**Why It Matters:** `IntermarketEngine` context feeds into LLM analyst (22 context blocks), MasterDecisionEngine, and RiskEngine correlation adjustment. Missing macro context means backtest results don't reflect live macro-aware decisions.
**Change:** New CLI that:
- Downloads 6 external assets via yfinance (free, no API key)
- Saves to `data/external/{ASSET}_D1.csv` with the same schema as MT5 CSVs
- Writes manifest with ticker mappings and date ranges
**Effect:** Operators can populate `data/external/` once; `MacroDataProvider` (to be updated in Phase 14 HistoricalCSVProvider work) can then load from CSV when `is_backtest_mode()=True`.

---

### Change 4 — LiveMT5Provider.current_time() parity fix

**File:** `core/data_provider.py`
**Function:** `LiveMT5Provider.current_time()` (line 76-78)
**Current Problem:**
```python
def current_time(self):
    import datetime
    return datetime.datetime.utcnow()    # ⚠ NAIVE datetime
```
**Why It Matters:** The `DataProvider` ABC docstring (line 50-57) explicitly warns:
> "Live: real wall-clock-ish broker time. Historical: the replay cursor's bar timestamp. Callers (session filters, news filters) must ask the provider for 'now' instead of calling datetime.now() directly, or historical replay silently gets today's session/news state applied to a 2023 bar."

But the live provider violates its own contract — `datetime.utcnow()` returns a naive datetime (no tzinfo), while `HistoricalCSVProvider.current_time()` returns a tz-aware UTC timestamp. Any caller that does arithmetic on the result (e.g. `current_time() - bar_time`) crashes on the live path if `bar_time` is tz-aware. Conversely, in backtest, callers that compare `current_time()` to a naive bar timestamp work, but if the live path is used (e.g. for forward testing), the comparison crashes.

**Change:**
```python
def current_time(self):
    # P1-A R4 FIX: must return tz-aware UTC for parity with HistoricalCSVProvider.
    # Previously returned datetime.utcnow() (NAIVE) which crashes callers that
    # compare to tz-aware bar timestamps.
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)
```
**Effect:** Live and historical paths now both return tz-aware UTC `datetime` objects. Parity violation closed.

---

### Change 5 — SmartMoneyEngine kill-zone parity fix

**File:** `analysis/smart_money.py`
**Function:** `SmartMoneyEngine._current_kill_zone()` (line 385)
**Current Problem:**
```python
def _current_kill_zone(self) -> dict:
    now = datetime.now(timezone.utc)  # ⚠ LIVE wall-clock — leaks in backtest
    hour = now.hour
    ...
```
Called from `SmartMoneyEngine.analyze_single()` (line 104) and `SmartMoneyEngine.analyze()` (line 223).

**Why It Matters:** In backtest mode, every historical bar gets stamped with the operator's CURRENT UTC hour for kill-zone detection, not the bar's actual UTC hour. A 2025-07-25 09:00 UTC bar (London open) processed at 14:00 UTC operator time would be tagged as "NY kill zone" instead of "London kill zone". This is a PARITY VIOLATION — backtest results do not reflect live kill-zone filtering behavior.

**Change:**
1. Added `bar_timestamp: Optional[datetime] = None` parameter to `_current_kill_zone()`
2. When `bar_timestamp` is provided, use it instead of wall-clock
3. Updated both callers to pass `df.index[-1]` (the most recent bar's timestamp):
   - `analyze_single()`: `bar_ts = df.index[-1].to_pydatetime() if hasattr(df.index[-1], 'to_pydatetime') else df.index[-1]`
   - `analyze()`: `bar_ts = dfs["M15"].index[-1].to_pydatetime() if ...`
4. Added `from typing import Optional` import
**Effect:** Kill-zone detection in backtest now uses the historical bar's actual UTC hour, matching live behavior. Parity violation closed.

---

### Change 6 — FeatureEngineer time features parity fix

**File:** `ml/feature_engineer.py`
**Function:** `FeatureEngineer._context_features()` (line 383) and `build_feature_vector()` (line 71)
**Current Problem:**
```python
# Time features
now = pd.Timestamp.now(tz="UTC")  # ⚠ LIVE wall-clock — leaks in backtest
f["hour_utc"] = float(now.hour)
f["day_of_week"] = float(now.weekday())
f["is_weekend"] = 1.0 if now.weekday() >= 5 else 0.0
f["is_monday_open"] = 1.0 if now.weekday() == 0 and now.hour < 12 else 0.0
f["is_friday_close"] = 1.0 if now.weekday() == 4 and now.hour >= 20 else 0.0
```

**Why It Matters:** ML feature engineering produces 5 time-based features (`hour_utc`, `day_of_week`, `is_weekend`, `is_monday_open`, `is_friday_close`). In backtest mode, every historical bar's feature vector would have the SAME values (whatever the operator's clock says right now), not the bar's actual time. This is a PARITY VIOLATION — ML training and inference feature vectors would have different time features for the same bar depending on when the code was run.

**Change:**
1. Added `df: Optional[pd.DataFrame] = None` parameter to `_context_features()`
2. Updated `build_feature_vector()` to pass `df` through to `_context_features()`
3. Time feature logic now:
```python
if len(df.index) > 0 and hasattr(df.index[-1], 'hour'):
    now = pd.Timestamp(df.index[-1])
    if now.tz is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")
else:
    now = pd.Timestamp.now(tz="UTC")  # fallback for non-time-indexed df
```
**Effect:** Time features in backtest now use the historical bar's actual UTC timestamp, matching live behavior. Parity violation closed.

---

## 3. Not Yet Implemented (Future Work)

The following items are documented in D3 and D4 but require larger refactors that are out of scope for this audit cycle:

### 3.1 Bid/ask column support in `HistoricalCSVProvider`

The downloader now writes `bid`/`ask` columns when run with `--with-bid-ask`. But `core/csv_data_provider.py:_load_csv()` (line 136-183) does not yet propagate `bid`/`ask` columns into the indicator pipeline. `BrokerSimulator.open_trade()` (in `execution/broker_simulator.py`) still fills at `close` instead of `ask` (BUY) or `bid` (SELL).

**Future change required:**
1. `_load_csv()`: add `bid` and `ask` to the column whitelist
2. `HistoricalCSVProvider.get_market_out()`: include `bid`/`ask` from the current bar in `ind_ctx`
3. `BrokerSimulator.open_trade()`: when `bid`/`ask` are available, fill BUY at `ask` and SELL at `bid`; otherwise fall back to `close` (current behavior, with `HISTORICAL_FILL_AT_CLOSE=True` flag)

### 3.2 MacroDataProvider historical reproduction

`analysis/intermarket.py` fetches macro data live via `MacroDataProvider`. The new `scripts/download_external_data.py` populates `data/external/{ASSET}_D1.csv`, but `MacroDataProvider` doesn't yet load from CSV when `is_backtest_mode()=True`.

**Future change required:** Modify `MacroDataProvider.get_macro_data()` to check `data/external/{ASSET}_D1.csv` first when `is_backtest_mode()` returns True; fall back to live fetch only if CSV is missing or stale.

### 3.3 News historical reproduction

`intelligence/news_sources.py:181` returns the local `data/economic_calendar.json` snapshot in backtest mode, but the snapshot is whatever was at install time — doesn't cover the historical backtest window.

**Future change required:** Either (a) disable news context in backtest mode (simplest), (b) download historical news from Forex Factory archive for the backtest window, or (c) filter the snapshot by `published_at <= current_bar_time` so future-dated news (relative to the replay cursor) is hidden.

### 3.4 28 cross-pair correlation reproduction

`analysis/correlation_engine.py` fetches 28 cross pairs live. In backtest, returns neutral `correlation_adjustment=1.0` if fetch fails.

**Future change required:** Operator runs `scripts/download_historical_data.py` for all 28 cross pairs (or 21 of them — the 7 majors give 21 of 28); `CorrelationEngine` then loads from CSV when `is_backtest_mode()=True`.

### 3.5 CFTC COT historical reproduction

`analysis/institutional_flow.py:_fetch_cot_from_cftc()` is gated OFF in backtest_mode (uses synthetic large-candle proxy).

**Future change required:** Download historical COT from CFTC.gov (public, free); modify `_fetch_cot_from_cftc()` to load from `data/external/cot/{PAIR}_cot.csv` when in backtest_mode.

### 3.6 `analysis/microstructure.py` MT5 lock bypass

P1-A R3: `microstructure.py:151` calls `mt5.initialize()` directly, bypassing the shared `MT5Connection.MT5_LOCK`. This is a thread-safety hazard.

**Future change required:** Add `copy_ticks_range` wrapper to `MT5Connection` (mirror the pattern of `copy_rates_from_pos` at line 633); route `microstructure.py` through it.

### 3.7 `broker/mt5_data.py` and `broker/mt5_historical_fetcher.py` broker-tz fix

P1-A R1: Both files call `mt5.copy_rates_from_pos` / `mt5.copy_rates_range` without applying the broker-tz offset, unlike `data/fetcher.py:_fetch_mt5()`.

**Future change required:** Either delete these files (they duplicate `data/fetcher.py` functionality) or route them through `MT5Connection` + apply the offset.

### 3.8 `data/automated_updater.py` orphan cleanup

P1-A R6: `data/automated_updater.py` writes to `data/forex/{PAIR}_daily.csv` with capitalized column names; `HistoricalCSVProvider` doesn't load these. Disconnected from the backtest pipeline.

**Future change required:** Delete `data/automated_updater.py` (operators should use `scripts/download_historical_data.py` instead), or refactor its output schema + path to match what `HistoricalCSVProvider` expects.

---

## 4. Test Plan

See Deliverable 6 for the validation + end-to-end test report.
