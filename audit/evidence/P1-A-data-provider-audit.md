# P1-A — Data Provider Layer Audit (Evidence Report)

**Task ID:** P1-A
**Scope:** Live MT5 + Historical CSV data ingestion / providing
**Repo:** `/home/z/my-project/download/forex-agent`
**Auditor:** Explore sub-agent (parallel stage)
**Date:** 2026-08-08

---

## 0. Executive Summary

The data provider layer has been refactored toward an execution-parity
contract (`DataProvider` ABC in `core/data_provider.py`). Three concrete
providers exist:

| Provider                  | File                              | Use-case                                |
|---------------------------|-----------------------------------|-----------------------------------------|
| `LiveMT5Provider`         | `core/data_provider.py:60`        | Live trading (wraps `MarketAgent.run()`)|
| `HistoricalMT5Provider`   | `core/data_provider.py:81`        | Backtest from in-memory pre-fetched df  |
| `HistoricalCSVProvider`   | `core/csv_data_provider.py:188`    | Backtest from local CSVs (preferred)    |

A factory (`core/provider_factory.py`) selects between them. The shared
output dict shape is `{df, ind_ctx, regime, regime_ctx, mtf_bias,
symbol, timeframe, data_source}`.

**Headline risks identified (full list in §8):**

- **R1 — Broker-timezone bug surface area:** Two distinct MT5 fetch
  paths exist. `data/fetcher.py` self-detects the broker UTC offset
  (GMT+2/+3) and corrects timestamps. `broker/mt5_data.py` and
  `broker/mt5_historical_fetcher.py` do **not** — they assume
  `time` is true UTC. Same MT5 terminal, two different time-zone
  behaviors.

- **R2 — `real_volume` silently dropped:** Live MT5 returns both
  `tick_volume` and `real_volume`; the historical CSV schema preserves
  `real_volume` (always 0 in samples). But `data/fetcher.py:966` drops
  `real_volume` from the live df, and `data/backtest_ohlcv_cache.py:98`
  drops it again from the registered cache. So even if a future CSV had
  non-zero real_volume, the indicator pipeline would not see it.

- **R3 — Live MT5 tick path bypasses shared lock:** `analysis/microstructure.py:151`
  calls `mt5.initialize()` directly, then `mt5.copy_ticks_range()`
  (line 156) without going through `broker.mt5_connection.MT5Connection`.
  This is exactly the race class the "Day 90+ hotfix" was meant to
  eliminate across the rest of the codebase.

- **R4 — `LiveMT5Provider.current_time()` returns a naive UTC datetime**
  (line 78: `datetime.datetime.utcnow()`), while `HistoricalCSVProvider`
  returns a tz-aware UTC timestamp (line 346). Downstream callers that
  compare these to `datetime.now(timezone.utc)` will misbehave on the
  live path. Parity violation flagged in the ABC docstring but not
  enforced.

- **R5 — Bid/Ask in CSV is missing entirely:** Live path has tick.bid /
  tick.ask for execution; historical path only has `close`. Backtest
  fills at `close` via `BrokerSimulator` — not at ask (for buys) or bid
  (for sells). This is execution-side parity, not data-side, but it
  means **spread is the only path through which the historical bid/ask
  reality touches the backtest** — and spread comes from a CSV column
  that is zero on many bars.

---

## 1. Live MT5 Data Path

### 1.1 Exact MT5 API calls used

| API Call                                | File:Line                              | Purpose                                              |
|-----------------------------------------|----------------------------------------|------------------------------------------------------|
| `mt5.symbol_info_tick(symbol)`          | `broker/mt5_data.py:157`               | Live bid/ask/spread                                  |
|                                         | `data/live_feed.py:218`                | Tick snapshot                                        |
|                                         | `data/data_orchestrator.py:359`        | Orchestrator tick passthrough                        |
|                                         | `broker/mt5_connection.py:520`         | Thread-safe wrapper                                  |
| `mt5.symbol_info(symbol)`               | `broker/mt5_data.py:162`               | digits for spread conversion                         |
|                                         | `data/live_feed.py:219`                | Same                                                 |
|                                         | `data/data_orchestrator.py:310`        | Symbol metadata (digits, spread, point, contract)   |
|                                         | `broker/mt5_connection.py:561`         | Thread-safe wrapper                                  |
| `mt5.copy_rates_from_pos(symbol,tf,0,count)` | `broker/mt5_data.py:216`          | Recent candles (count-back)                         |
|                                         | `data/fetcher.py:852`                  | Same (via shared connection)                         |
|                                         | `broker/mt5_connection.py:633`         | Thread-safe wrapper                                  |
| `mt5.copy_rates_range(ticker,tf,start,end)` | `broker/mt5_historical_fetcher.py:102` | Bulk historical fetch (monthly chunks)         |
| `mt5.copy_ticks_range(sym,from,to,COPY_TICKS_ALL)` | `analysis/microstructure.py:156` | Last 60s of ticks                      |
| `mt5.symbol_select(symbol, True)`       | `data/fetcher.py:775`                  | Activate in Market Watch                             |
|                                         | `broker/mt5_connection.py:622`         | Same                                                 |
| `mt5.last_error()`                      | `data/fetcher.py:776, 855`             | Error classification                                 |
| `mt5.initialize()`                      | `analysis/microstructure.py:151`      | ⚠ Direct init bypassing shared connection            |
| `mt5.order_send(request)`               | `data/data_orchestrator.py:452`        | Order placement (execution path, not data)           |

`mt5.market_book_add()` (Level-2 depth) is **NOT used anywhere** in the
codebase (verified via grep).

### 1.2 Exact fields read

**From `copy_rates_from_pos` / `copy_rates_range` (structured array
rows):**

| Field         | Read by                                              | Notes                                                  |
|---------------|------------------------------------------------------|--------------------------------------------------------|
| `time`        | All paths                                            | Unix epoch seconds (broker-tz caveat, see §10)         |
| `open`        | All paths                                            | Float                                                  |
| `high`        | All paths                                            | Float                                                  |
| `low`         | All paths                                            | Float                                                  |
| `close`       | All paths                                            | Float                                                  |
| `tick_volume` | `mt5_data.py:228`, `fetcher.py:966`                  | Number of price ticks in bar (NOT consolidated volume) |
| `spread`      | `mt5_data.py:229`                                    | Integer points                                         |
| `real_volume` | **NEVER READ** by any consumer                      | Field exists in MT5 array but dropped everywhere       |

**From `symbol_info_tick`:**

| Field    | Read by                                                     |
|----------|-------------------------------------------------------------|
| `bid`    | `mt5_data.py:169`, `live_feed.py:233`, `data_orchestrator.py:363` |
| `ask`    | `mt5_data.py:170`, `live_feed.py:234`, `data_orchestrator.py:364` |
| `last`   | `live_feed.py:235`, `data_orchestrator.py:365`              |
| `time`   | `mt5_data.py:172`, `live_feed.py:255`, `fetcher.py:702, 709` |
| `time_msc`| **NEVER READ** — millisecond precision lost                  |

**From `symbol_info`:**

| Field                    | Read by                                |
|--------------------------|----------------------------------------|
| `digits`                 | `mt5_data.py:163`, `live_feed.py:225`, `data_orchestrator.py:314` |
| `spread`                 | `data_orchestrator.py:315`             |
| `point`                  | `data_orchestrator.py:316`             |
| `trade_contract_size`    | `data_orchestrator.py:317`             |
| `trade_tick_value`       | `data_orchestrator.py:318`             |
| `trade_tick_size`        | `data_orchestrator.py:319`             |
| `volume_min` / `volume_max` / `volume_step` | `data_orchestrator.py:320-322` |
| `trade_mode`             | `live_feed.py:366`                     |

**From `copy_ticks_range` (tick rows in `microstructure.py`):**

| Field          | Read by                              |
|----------------|--------------------------------------|
| `time`         | `microstructure.py:168`              |
| `bid`          | `microstructure.py:169`              |
| `ask`          | `microstructure.py:170`              |
| `last`         | `microstructure.py:171`              |
| `volume_real`  | `microstructure.py:172` (fallback to `volume`) |
| `volume`       | `microstructure.py:173` (fallback)   |
| `flags`        | `microstructure.py:174`              |

### 1.3 Timezone handling (live)

`data/fetcher.py` lines 895-953 implement broker-tz auto-detection:

```python
# Audit-P1 fix: MT5's `time` field is documented as Unix-epoch seconds (UTC).
# HOWEVER, in practice many brokers configure the MT5 server to return bar
# OPEN time in BROKER SERVER TIME (commonly GMT+2 winter / GMT+3 summer).
broker_offset_hours = self._get_broker_utc_offset_hours(symbol)
df['time'] = pd.to_datetime(df['time'], unit='s', utc=False)
if broker_offset_hours != 0:
    df['time'] = df['time'] - pd.Timedelta(hours=broker_offset_hours)
df['time'] = df['time'].dt.tz_localize('UTC')
```

Offset is detected by comparing `tick.time` to `datetime.now(timezone.utc)`
(cached 30 min, self-corrects for DST flips). Manual override via
`MT5_BROKER_TZ_OFFSET_HOURS` env var.

⚠ `broker/mt5_data.py:223` does NOT do this — it just calls
`datetime.fromtimestamp(int(r["time"]), tz=timezone.utc)`. If the broker
returns server time mislabeled as epoch, this path produces timestamps
2-3h in the future. Same for `broker/mt5_historical_fetcher.py:107`
which uses `pd.to_datetime(chunk["time"], unit="s")` — naive, no
tzinfo, no offset correction.

---

## 2. Historical CSV Data Path

### 2.1 CSV Loader

Single loader: `core/csv_data_provider.py::_load_csv()` lines 136-183.

```python
def _load_csv(filepath: Path, symbol: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, encoding="utf-8-sig")
    ts_col = None
    for candidate in ("datetime_utc", "datetime", "time", "timestamp", "date"):
        if candidate in df.columns:
            ts_col = candidate
            break
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col])
    df = df.rename(columns={ts_col: "time"})
    df = df.set_index("time").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    # Rename tick_volume → volume (if needed)
    if "tick_volume" in df.columns and "volume" not in df.columns:
        df = df.rename(columns={"tick_volume": "volume"})
    if "volume" not in df.columns:
        if "tickvol" in df.columns:
            df = df.rename(columns={"tickvol": "volume"})
        else:
            df["volume"] = 0.0
    # Coerce OHLC to numeric
    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    if "spread" in df.columns:
        df["spread"] = pd.to_numeric(df["spread"], errors="coerce").fillna(0)
    return df
```

Timestamp parsing: **explicit UTC** via `utc=True`. Naive timestamps are
treated as UTC (no broker offset correction needed because historical
CSVs are pre-converted to UTC at download time — see manifest.json
field `"timezone": "UTC"`).

### 2.2 File layout resolution

`_find_csv()` (line 79) tries two layouts:

1. **Nested (preferred):** `data/historical/{SYMBOL}/{TF}.csv`
2. **Flat (legacy):** `data/{SYMBOL}_{TF}.csv`

Current state: only flat layout exists in the repo (`data/*.csv` — 21
files). `data/historical/` contains only `manifest.json`.

### 2.3 Pipeline downstream of CSV load

Historical CSV feeds into the **exact same** indicator chain as live:

```
HistoricalCSVProvider.get_market_out()
  ├─ df_slice = primary_df.iloc[cursor-300 : cursor+1].copy()  (causal)
  ├─ add_canonical_indicators(df_slice, include_patterns=True)
  │    └─ data.indicator_registry → ExtendedIndicators → legacy Indicators
  ├─ MarketRegimeDetector.detect(df_slice) + get_ai_context()
  ├─ _get_spread_pips() → ind_ctx["spread_pips"]
  └─ _compute_mtf_bias_from_csvs()  (causal — only closed higher-TF bars)
```

Same chain as `LiveMT5Provider` (which wraps `MarketAgent.run()`):
```
MarketAgent.run()
  ├─ MultiTimeframeAnalyzer.analyze(["1d","4h","1h","15m"])
  ├─ orchestrator.get_candles(symbol, tf, limit=300)
  │    └─ DataFetcher.fetch_ohlcv → mt5.copy_rates_from_pos
  ├─ DataValidator().validate(df, symbol, timeframe)
  ├─ add_canonical_indicators → ExtendedIndicators → Indicators
  └─ MarketRegimeDetector.detect(df)
```

**Pipeline convergence:** YES — both paths feed into the SAME indicator
registry, SAME regime detector, SAME `MarketAgentResult`-shaped dict.

### 2.4 Anti-look-ahead contract

Documented in `core/csv_data_provider.py:28-35`:

> For every decision timestamp `T`:
> - Primary TF slice: only rows with `datetime <= T`
> - Higher TF (H1, H4, D1) slices: only rows whose bar OPENED at or before
>   `T - tf_interval` (i.e. the bar has CLOSED by T)
> - mtf_bias: computed from the closed higher-TF bars only

Implemented in `_compute_mtf_bias_from_csvs()` (line 458-523):

```python
cutoff = current_time - pd.Timedelta(minutes=tf_minutes)
causal = df[df.index <= cutoff]
```

Equivalent in `data/backtest_ohlcv_cache.py:172-175`:
```python
closed_mask = (df.index + delta) <= asof
visible = df.loc[closed_mask]
visible = visible.loc[visible.index <= asof]
```

Tested: `tests/parity/test_csv_provider_lookahead.py` (referenced in
docstring).

---

## 3. Provider Interface

### 3.1 The `DataProvider` ABC

`core/data_provider.py:38-57`:

```python
class DataProvider(ABC):
    """Contract every provider must satisfy. Return shape must match
    agents/market_agent.py's MarketAgentResult dict exactly:
    {df, ind_ctx, regime, regime_ctx, mtf_bias, symbol, timeframe,
     data_source}.
    """

    @abstractmethod
    def get_market_out(self, symbol: str, timeframe: str) -> dict:
        ...

    @abstractmethod
    def current_time(self):
        """Broker-time timestamp of the last bar this provider has seen.
        Live: real wall-clock-ish broker time. Historical: the replay
        cursor's bar timestamp. Callers (session filters, news filters)
        must ask the provider for "now" instead of calling datetime.now()
        directly, or historical replay silently gets today's session/news
        state applied to a 2023 bar."""
        ...
```

Only **two** methods. No `get_rates`, `get_candles`, `get_tick` on the
ABC — those live on the lower-level `MT5DataFeed` / `DataFetcher` /
`DataOrchestrator` (which are not part of the DataProvider contract).

### 3.2 Concrete providers

| Class                       | Module                              | Constructor Args                                          |
|-----------------------------|-------------------------------------|-----------------------------------------------------------|
| `LiveMT5Provider`           | `core/data_provider.py:60`          | `(market_agent)`                                          |
| `HistoricalMT5Provider`     | `core/data_provider.py:81`          | `(df, symbol, timeframe)`                                 |
| `HistoricalCSVDataProvider` | `core/csv_data_provider.py:188`     | `(symbol, primary_timeframe, data_dir, mtf_timeframes, lookback_bars)` |

### 3.3 Factory

`core/provider_factory.py:38` `make_backtest_provider(symbol, timeframe, df=None, prefer="auto", data_dir=None, **kwargs)`.

Selection logic:
- `prefer="csv"` → `HistoricalCSVDataProvider` (raises `FileNotFoundError` if no CSV)
- `prefer="mt5"` → `HistoricalMT5Provider` (requires `df` arg)
- `prefer="auto"` (default) → CSV if available, else df-based MT5, else raise

### 3.4 Execution-side adapter (sibling, not data)

`core/execution_adapter.py` defines `ExecutionAdapter` ABC with
`open_trade(...)` and `get_balance()` methods; `MT5ExecutionAdapter`
wraps `ExecutionRouter`, `HistoricalExecutionAdapter` wraps
`BrokerSimulator`. Mentioned here for completeness; not part of the
data-provider layer per se.

---

## 4. Multi-Timeframe Handling

### 4.1 Live path

`MarketAgent.run()` (line 156) builds MTF via `MultiTimeframeAnalyzer`:

```python
mtf = MultiTimeframeAnalyzer(self.symbol)
mtf_data = mtf.analyze(["1d", "4h", "1h", "15m"])
```

Each TF triggers a separate `orchestrator.get_candles(symbol, tf, limit=300)`
call → `DataFetcher.fetch_ohlcv` → `mt5.copy_rates_from_pos` per TF.

If MT5 unavailable and external source lacks H4 (e.g. yfinance/Alpha
Vantage free tiers), `DataFetcher._resample_h1_to_h4(df_h1, limit)`
(line 1310) synthesizes H4 from H1 using
`df_h1.resample("4h", origin="epoch")` so bars align to MT5's H4
grid (00-04, 04-08, ..., 20-24 UTC).

### 4.2 Forming-candle guard (live)

`data/fetcher.py:972-1004` explicitly drops the last (still-forming)
bar before returning:

```python
_tf_seconds = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800,
               "H1": 3600, "H4": 14400, "D1": 86400}.get(timeframe.upper())
if _tf_seconds and len(df) > 1:
    _last_open = df.index[-1]
    _implied_close = _last_open + pd.Timedelta(seconds=_tf_seconds)
    _now_utc = pd.Timestamp.now(tz='UTC')
    if _implied_close > _now_utc:
        log.debug(f"[MT5] {symbol} {timeframe}: dropping still-forming "
                  f"last bar (open={_last_open.isoformat()}, implied "
                  f"close={_implied_close.isoformat()} is still in the "
                  f"future) — structural detectors (BOS/CHoCH) require "
                  f"closed candles only.")
        df = df.iloc[:-1].copy()
```

### 4.3 Historical path

`HistoricalCSVProvider._compute_mtf_bias_from_csvs()` (line 458):
**only closed higher-TF bars** are used:

```python
tf_minutes = _tf_to_minutes(tf)
cutoff = current_time - pd.Timedelta(minutes=tf_minutes)
causal = df[df.index <= cutoff]   # bar open_time <= T - tf_interval
                                        # → bar closed by T
```

`data/backtest_ohlcv_cache.py::get_ohlcv()` (line 148): same logic for
the point-in-time HTF cache (used by `SMCEngine` / `MultiTimeframeAnalyzer`
in backtest mode):

```python
closed_mask = (df.index + delta) <= asof
visible = df.loc[closed_mask]
visible = visible.loc[visible.index <= asof]
```

### 4.4 HTF incomplete bars allowed?

**No, on either path.** Live drops the forming bar at fetch time
(`fetcher.py:1004`); historical filters to closed bars only at
cursor time. This matches the `HistoricalMT5Provider._compute_mtf_bias`
docstring (line 226): "live evaluates mid-bar" — but the actual fetcher
code drops mid-bar candles before they reach the analyzer, so the live
pipeline effectively operates on closed bars too. Parity is preserved.

---

## 5. Lookback Hints

No global `MAX_LOOKBACK` constant exists. Each provider has its own:

| Source                              | Default Lookback (bars) | File:Line                  |
|-------------------------------------|--------------------------|----------------------------|
| `MarketAgent.run()` (live)          | 300                      | `agents/market_agent.py:186` (`limit=300`) |
| `DataFetcher.fetch_ohlcv()`         | 300                      | `data/fetcher.py:506` (`limit=300`) |
| `MT5DataFeed.DEFAULT_CANDLE_COUNT`  | 500                      | `broker/mt5_data.py:142`   |
| `HistoricalMT5Provider.LOOKBACK_BARS` | 300                    | `core/data_provider.py:153` |
| `HistoricalCSVProvider.lookback_bars` | 300                    | `core/csv_data_provider.py:219` |
| `DataOrchestrator.get_multi_timeframe()` | 100                  | `data/data_orchestrator.py:380` |
| `failure_cascade_detector.DEFAULT_LOOKBACK_BARS` | 60        | `analysis/failure_cascade_detector.py:62` |

Live `MarketAgent` uses 300, explicitly chosen for parity with
HistoricalCSVProvider's default (see comment in
`core/data_provider.py:150-152`):

> Window size = 300 matches live trading's limit=300 in
> agents/market_agent.py for execution parity.

⚠ **Mismatch:** `DataOrchestrator.get_multi_timeframe()` defaults to
`limit=100` while `MarketAgent.run()` calls `get_candles(... limit=300)`
directly. So the orchestrator's MTF helper would return only 100 bars
if anything called it — but `MarketAgent` does NOT use that helper, it
loops through `MultiTimeframeAnalyzer` which calls `fetch_ohlcv` per TF
with `limit=300`. The 100-bar default in the orchestrator is dead code
for the agent path.

---

## 6. Existing CSV Schema

### 6.1 All 21 CSVs in `data/` share the same schema

Verified via `head -1 data/*.csv`:

```
datetime_utc,open,high,low,close,tick_volume,spread,real_volume
```

Eight columns. Confirmed uniform across every symbol/timeframe in the
repo. (See Appendix A for per-file first 3 lines.)

### 6.2 Field semantics (from sample inspection)

| Column         | Type    | Sample (EURUSD_H1.csv) | Units           |
|----------------|---------|------------------------|------------------|
| `datetime_utc` | ISO 8601| `2025-07-25 07:00:00+00:00` | UTC bar open time |
| `open`         | float   | `1.17473`              | Price            |
| `high`         | float   | `1.1758899999999999`   | Price            |
| `low`          | float   | `1.17473`              | Price            |
| `close`        | float   | `1.17523`              | Price            |
| `tick_volume`  | int     | `3210`                 | Tick count       |
| `spread`       | int     | `8`                    | POINTS (not pips)|
| `real_volume`  | int     | `0` (always)           | Lots (always 0)  |

### 6.3 Coverage (from `data/historical/manifest.json`)

- **Total files:** 145 (across all symbols × 3 timeframes + REPORT)
- **Total rows:** 1,526,342
- **Symbols:** 48 (majors, minors, metals, exotics)
- **Timeframes:** H1, H4, M15 (+ REPORT)
- **Date range:** 2025-07-25 → 2026-07-24 (1 year)
- **Timezone:** UTC (manifest field `"timezone": "UTC"`)
- **`tick_volume_available`:** true (all files)
- **`real_volume_available`:** false (all files)
- **`spread_available`:** true
- **`spread_nonzero_pct`:** varies (77.44% for AUDCAD_H1 — some files have many zero-spread bars; EURUSD_H4 first bar shows spread=0)

### 6.4 Flat vs nested layout

Only the **flat** layout (`data/{SYMBOL}_{TF}.csv`) is currently
populated. The **nested** layout (`data/historical/{SYMBOL}/{TF}.csv`)
is preferred by `_find_csv()` but is empty — `data/historical/` contains
only `manifest.json`. The loader checks nested first, then falls back
to flat.

---

## 7. Bid/Ask/Spread + Volume Handling

### 7.1 Bid/Ask/Spread — Live MT5

Three live entry points:

1. **`broker/mt5_data.py:148` `get_tick()`:**
   ```python
   tick = mt5.symbol_info_tick(broker_symbol)
   info = mt5.symbol_info(broker_symbol)
   digits = info.digits if info else 5
   spread_points = (tick.ask - tick.bid)
   spread_pips = round(spread_points * (10 ** (digits - 1)), 1) if digits else 0
   return {"symbol": broker_symbol, "bid": tick.bid, "ask": tick.ask,
           "spread_pips": spread_pips, "time": ...}
   ```
   Formula: `spread_pips = (ask - bid) × 10^(digits-1)`. For 5-digit
   FX (digits=5): `× 10^4` = `× 10000` — but `(ask-bid)` is in price
   units, so `0.00012 × 10000 = 1.2` pips. ✓ Correct.

2. **`data/live_feed.py:206` `get_snapshot()`:**
   Same formula (`spread_pips = spread_points * (10 ** (digits - 1))`).
   Adds rolling buffer for spread median, velocity, direction pressure.

3. **`data/data_orchestrator.py:346` `get_tick()`:**
   Returns raw `bid, ask, last, time` — **does NOT compute spread_pips**.
   Caller has to do it. Inconsistent with the other two paths.

### 7.2 Bid/Ask/Spread — Historical CSV

- **Bid/Ask:** NOT in CSV at all. CSV has only `open, high, low, close`.
  Live execution uses `tick.ask` (for buys) and `tick.bid` (for sells);
  historical execution (`BrokerSimulator`) fills at `close`. So the
  bid/ask spread cost is captured in backtest **only** through the
  `spread` column → `ind_ctx["spread_pips"]` → downstream cost-aware EV
  gates.

- **Spread:** `HistoricalCSVProvider._get_spread_pips()` (line 421-456)
  converts CSV `spread` (in points) to pips:
  ```python
  spread_points = float(self._primary_df.iloc[self._cursor].get("spread", 0))
  if spread_points == 0:
      # Fallback 1: mean of last 50 non-zero bars
      recent = self._primary_df["spread"].iloc[max(0, c-50):c+1]
      recent_nonzero = recent[recent > 0]
      if len(recent_nonzero) > 0:
          spread_points = float(recent_nonzero.mean())
      else:
          return None  # Fallback 2: caller uses DEFAULT_SPREAD_PIPS table
  pip = get_pip_size(symbol)  # from backtest.symbol_specs
  if pip in (0.0001, 0.01):    # 5-digit FX or 3-digit JPY
      return spread_points / 10.0
  return spread_points         # XAU/indices where pip == point
  ```
  So spread is **never** approximated when the CSV has a non-zero value;
  approximation only kicks in when the CSV bar has `spread=0` (then
  recent-mean fallback), or the column is missing entirely (then
  static DEFAULT_SPREAD_PIPS table).

### 7.3 Volume — Live MT5

`data/fetcher.py:966-967` (live df after `copy_rates_from_pos`):
```python
df = df[['open', 'high', 'low', 'close', 'tick_volume']].copy()
df.rename(columns={'tick_volume': 'volume'}, inplace=True)
```

`real_volume` is **explicitly dropped**. Inline comment (line 959-965):
> NOTE: MT5's 'tick_volume' is the number of price ticks in the bar,
> not consolidated traded volume — forex is decentralized and there is
> no true consolidated volume figure. We keep the column named 'volume'
> for downstream compatibility, but this is documented here and in
> _fetch_yfinance/others so anyone weighting signal confidence by
> 'volume' knows it's a tick activity proxy, not real traded volume.

`broker/mt5_data.py:228` (live candles dict):
```python
"volume": float(r["tick_volume"]),
```
Same — only `tick_volume` is kept.

### 7.4 Volume — Historical CSV

`_load_csv()` (line 166-172):
```python
if "tick_volume" in df.columns and "volume" not in df.columns:
    df = df.rename(columns={"tick_volume": "volume"})
```

So CSV's `tick_volume` becomes the `volume` column. `real_volume` is
preserved in the DataFrame (the loader doesn't drop it), but:

`data/backtest_ohlcv_cache.py:96-105` — when registering the series
into the shared MTF cache, the column whitelist is:
```python
for c in out.columns:
    cl = str(c).lower()
    if cl in ("open", "high", "low", "close", "volume", "tick_volume", "spread"):
        rename[c] = "volume" if cl == "tick_volume" else cl
# ...
keep = [c for c in ("open", "high", "low", "close", "volume", "spread") if c in out.columns]
out = out[keep].astype(float, errors="ignore")
```

`real_volume` is **dropped silently** here. Even if a future CSV had
non-zero `real_volume`, the backtest OHLCV cache would not propagate it
to consumers.

### 7.5 Volume parity check

| Source                | tick_volume | real_volume | volume column name        |
|-----------------------|-------------|-------------|---------------------------|
| Live MT5 (fetcher)   | ✅ kept     | ❌ dropped  | `volume` (renamed from `tick_volume`) |
| Live MT5 (mt5_data)  | ✅ kept     | ❌ ignored | `volume`                  |
| Historical CSV       | ✅ kept     | ✅ kept (in df, dropped in cache) | `volume` (renamed from `tick_volume`) |
| Backtest OHLCV cache | ✅ kept     | ❌ dropped  | `volume`                  |
| Indicator pipeline   | ✅ reads `volume` | n/a   | `volume`                  |

The `volume` column is always tick_volume. `real_volume` is read by
**nothing** in the pipeline. This is documented (forex has no real
consolidated volume), so it is intentional — but it means the system
cannot differentiate "0 real volume" from "no real volume field".

---

## 8. Red Flags / Leakage Risks

### R1 — Broker-timezone bug: inconsistent across MT5 fetch paths

**Severity:** High (data correctness)

`data/fetcher.py:_fetch_mt5()` (lines 895-953) implements broker-tz
auto-detection and correction. `broker/mt5_data.py:get_candles()` (line
223) does NOT — it just calls
`datetime.fromtimestamp(int(r["time"]), tz=timezone.utc)`.

`broker/mt5_historical_fetcher.py:107` also does NOT — uses
`pd.to_datetime(chunk["time"], unit="s")` producing naive timestamps.

If a broker returns server time mislabeled as epoch (GMT+2/+3 common
for FX brokers), `mt5_data.py` and `mt5_historical_fetcher.py` produce
timestamps 2-3h in the future. `fetcher.py` (the path actually used by
`MarketAgent`) is safe — but the other two paths are lurking
foot-guns if any consumer wires them up.

**Recommendation:** route all MT5 calls through `MT5Connection` +
apply broker-tz correction in one place. Or delete `mt5_data.py` and
`mt5_historical_fetcher.py` since they duplicate `fetcher.py`.

### R2 — `real_volume` silently dropped

**Severity:** Medium (information loss)

Live MT5 returns `real_volume` in the structured array; the historical
CSV schema preserves it (`real_volume` column, always 0 in samples).
But `data/fetcher.py:966` explicitly drops it from the live df, and
`data/backtest_ohlcv_cache.py:96-105` drops it from the cache. So even
if a future CSV download populated real_volume, the indicator pipeline
would not see it.

For FX this is benign (real_volume is always 0). For metals (XAUUSD)
and indices (US30, NAS100) where some brokers DO populate real_volume,
this is an information loss.

**Recommendation:** keep `real_volume` in the column whitelist, even
if the indicator pipeline doesn't use it yet.

### R3 — Live tick fetch in `microstructure.py` bypasses shared MT5 lock

**Severity:** High (race condition)

`analysis/microstructure.py:130-186`:

```python
@staticmethod
def _fetch_ticks(symbol: str) -> Optional[list]:
    # ...
    if not mt5.initialize():   # ⚠ DIRECT mt5.initialize()
        return None
    utc_to = datetime.now(timezone.utc)
    utc_from = utc_to - timedelta(seconds=60)
    ticks = mt5.copy_ticks_range(symbol, utc_from, utc_to, mt5.COPY_TICKS_ALL)
    # DO NOT CALL mt5.shutdown() — leave connection alive for main loop
```

The comment says "Do NOT call mt5.shutdown() here" but the
`mt5.initialize()` call is **already** a violation — it bypasses the
shared `MT5Connection.MT5_LOCK` that every other MT5 caller
(`fetcher.py`, `live_feed.py`, `data_orchestrator.py`) uses. A tick
fetch here can race an in-flight, lock-protected order elsewhere in the
process.

This is the **exact same class of bug** the "Day 90+ hotfix" was
designed to eliminate across the rest of the codebase — see the
docstring in `data/data_orchestrator.py:88-117`:

> previously `_get_mt5()`, `get_account_info()`, and
> `place_market_order()` each built and `.connect()`-ed a BRAND
> NEW `MT5Connection(login=..., password=..., server=...)` instance on
> every call — never reusing one another, and never reusing the shared,
> locked connection that fetcher.py ... depend on.

`microstructure.py` was missed.

**Recommendation:** route through `MT5Connection` (add a
`copy_ticks_range` wrapper, mirror the pattern of
`copy_rates_from_pos`).

### R4 — `LiveMT5Provider.current_time()` returns naive datetime

**Severity:** Medium (parity violation)

`core/data_provider.py:76-78`:

```python
def current_time(self):
    import datetime
    return datetime.datetime.utcnow()    # ⚠ NAIVE
```

vs `HistoricalCSVProvider.current_time()` (csv_data_provider.py:344-346):

```python
def current_time(self):
    return self._primary_df.index[self._cursor]   # tz-aware UTC
```

vs `HistoricalMT5Provider.current_time()` (data_provider.py:127-128):

```python
def current_time(self):
    return self._df.index[self._cursor]   # tz-aware UTC (if df was loaded as UTC)
```

The ABC docstring explicitly warns about this
(`core/data_provider.py:50-57`):

> Live: real wall-clock-ish broker time. Historical: the replay
> cursor's bar timestamp. Callers (session filters, news filters) must
> ask the provider for "now" instead of calling datetime.now() directly,
> or historical replay silently gets today's session/news state
> applied to a 2023 bar.

But the live provider violates its own contract — `utcnow()` returns
naive, while the historical provider returns tz-aware. Any caller that
does arithmetic on the result (e.g. `current_time() - bar_time`) will
crash on the live path if `bar_time` is tz-aware.

**Recommendation:** change to `datetime.datetime.now(timezone.utc)`.

### R5 — Bid/Ask in CSV missing → backtest fills at `close`, not at ask/bid

**Severity:** Medium (execution parity, not data parity per se)

Live execution: `MT5ExecutionAdapter.open_trade()` →
`ExecutionRouter.execute()` uses `tick.ask` (BUY) or `tick.bid` (SELL)
as the fill price.

Historical execution: `HistoricalExecutionAdapter.open_trade()` →
`BrokerSimulator.open_trade()` uses `entry_price` (which the calling
strategy passes as the bar's `close`). The bid/ask spread cost is
captured ONLY through `ind_ctx["spread_pips"]` → downstream
cost-aware EV gates.

If `spread == 0` in the CSV bar (and recent-mean fallback also yields
0), the backtest effectively trades at zero spread — the historical
equivalent of "no slippage, no spread" — while live pays the actual
bid/ask. The CSV's spread column is the **only** path through which
historical bid/ask reality touches the backtest, and many EURUSD_H4
bars show `spread=0`.

**Recommendation:** when CSV spread==0 and recent-mean also==0, the
provider currently returns `None` and the caller falls back to
DEFAULT_SPREAD_PIPS — verify this fallback is realistic (the static
table in `core/constants.py` should match typical retail spreads).

### R6 — `data/automated_updater.py` uses yfinance / OANDA, not MT5

**Severity:** Low (orphan module)

`data/automated_updater.py` is the "daily data update" tool. It
fetches from OANDA (if `OANDA_API_KEY` set) → yfinance → "synthetic data
refusal" (recently fixed; was generating synthetic data). It uses
`Open/High/Low/Close/Volume` capitalized column names — totally
different schema from the live MT5 path's lowercase `open/high/low/close/volume`.

It saves daily CSVs to `data/forex/{PAIR}_daily.csv` (note: `forex/`
subdir, not the flat `data/{SYMBOL}_{TF}.csv` layout the backtest CSV
provider expects).

The backtest CSV provider (`HistoricalCSVProvider`) does NOT load these
files — it only looks in `data/historical/{SYMBOL}/{TF}.csv` and
`data/{SYMBOL}_{TF}.csv`.

So `automated_updater.py` is **disconnected from the actual backtest
pipeline**. It looks like a Day 7-era module that was never wired into
the parity refactor.

**Recommendation:** either delete `automated_updater.py` (use
`scripts/download_historical_data.py` instead) or update its output
schema + path to match what `HistoricalCSVProvider` expects.

### R7 — `data/verify_data_coverage.py` hardcodes 2026-06-21 target date

**Severity:** Low (test rot)

`data/verify_data_coverage.py:346` `target_date = datetime(2026, 6, 21)`.
This was a one-off check; running it today (2026-08-08) the target is
in the past, so the test trivially passes. Stale verification.

### R8 — `LiveMT5Provider.get_market_out()` mutates the market_agent state

**Severity:** Low (architecture smell)

`core/data_provider.py:69-74`:

```python
def get_market_out(self, symbol: str, timeframe: str) -> dict:
    self._market_agent.symbol = symbol         # ⚠ MUTATES shared agent
    self._market_agent.timeframe = timeframe
    return self._market_agent.run()
```

If the `MarketAgent` is shared between multiple `LiveMT5Provider`
instances (e.g. one per pair), they race to set `symbol`/`timeframe`.
Live trading typically uses one MarketAgent per pair, so this works in
practice — but the abstraction is leaky. A future caller that builds a
single provider and calls `get_market_out()` for multiple symbols in
the same loop would get stale state.

### R9 — `mt5_data.py` live CSV save uses `time` field, not `time_msc`

**Severity:** Informational

`broker/mt5_data.py:283` `save_live_csv()` writes the candles list to
`data/live/{SYMBOL}_{TF}_live.csv`. The candle dict has `time` (ISO
8601 UTC) but no `time_msc`. Sub-millisecond resolution is lost.

### R10 — `_get_broker_utc_offset_hours` only updates on tick availability

**Severity:** Low (cold-start edge case)

`data/fetcher.py:_get_broker_utc_offset_hours()` requires
`self._mt5_conn.get_tick(symbol)` to return a non-None tick to detect
the offset. On weekends or for thin symbols, the tick may be cached
zero-time → returns the previous cached offset (default 0.0). If the
broker actually uses GMT+3, the cold-start fetch on a Sunday evening
would mislabel bar timestamps until Monday open.

---

## 9. Tick Data / Market Depth

### 9.1 Tick fetch — only one consumer

`analysis/microstructure.py:130-186` is the **only** module that calls
`mt5.copy_ticks_range()`. It fetches the last 60 seconds of ticks and
classifies tick speed (DEAD/SLOW/NORMAL/FAST/HYPER) and direction.

Fields read (line 168-174): `time`, `bid`, `ask`, `last`, `volume_real`
(with `volume` fallback), `flags`.

### 9.2 Market depth (Level 2) — not used

`mt5.market_book_add()` is **not called anywhere** in the codebase
(grep verified). No order-book / Level-2 data is consumed.

### 9.3 Tick caching

`data/live_feed.py:LiveFeed` maintains a per-symbol rolling buffer
(`BUFFER_SIZE = 120` ticks, ~2 minutes at 1 tick/sec) of recent ticks
for velocity / direction pressure / spread median computation. This is
the only in-memory tick cache. No persistent tick storage other than
the live CSV snapshot written by `mt5_data.py:save_live_csv()`.

### 9.4 Tick-to-bar aggregation

There is **no** tick-to-bar aggregation in the live path. Bars are
fetched directly via `copy_rates_from_pos` (broker-side aggregation).
Ticks are only used for real-time intelligence (spread, velocity) —
not for building OHLC bars.

---

## 10. Time / Timezone Handling

### 10.1 MT5 server time

Typical FX broker convention: GMT+2 in winter (US/EU DST off), GMT+3 in
summer (US/EU DST on). IC Markets, Pepperstone, Exness, FXTM all use
this. The actual offset for the connected broker is detected at runtime
by `DataFetcher._get_broker_utc_offset_hours()` (line 662-725):

```python
tick = self._mt5_conn.get_tick(symbol)
broker_now = datetime.fromtimestamp(tick.time, tz=timezone.utc)
utc_now = datetime.now(timezone.utc)
offset_hours = round((broker_now - utc_now).total_seconds() / 3600)
```

Cached for 30 min (`_BROKER_OFFSET_CACHE_TTL_SEC = 1800`). Self-corrects
for DST flips.

Manual override via `MT5_BROKER_TZ_OFFSET_HOURS` env var (line 686-688).
If set, auto-detection is skipped.

### 10.2 Timezone tagging in the live df

`data/fetcher.py:938`: `df['time'] = df['time'].dt.tz_localize('UTC')` —
explicit UTC tzinfo attached after broker-tz correction.

### 10.3 Timezone tagging in historical CSV

`core/csv_data_provider.py:157`: `pd.to_datetime(ts_col, utc=True,
errors="coerce")` — explicit UTC parsing. Both nested and flat CSV
layouts use the `datetime_utc` column name (confirmed across all 21
files).

### 10.4 Sessions

`analysis/session_analyzer.py:get_current_session()` computes London /
NY / Tokyo / Sydney windows from GMT hour using
`datetime.now(timezone.utc)`:

| Session | Winter Hours (UTC) | Summer Hours (UTC) |
|---------|--------------------|--------------------|
| Tokyo   | 00:00 - 09:00      | 00:00 - 09:00      |
| London | 08:00 - 17:00       | 07:00 - 16:00      |
| New York | 13:00 - 22:00     | 12:00 - 21:00      |
| Sydney | 22:00 - 07:00      | 22:00 - 07:00      |

DST detection: `zoneinfo` (Python 3.9+ stdlib) using
`America/New_York` and `Europe/London` IANA zones — 100% accurate per
OS tz database.

### 10.5 Weekend close detection

`broker/data_validator.py:180-181`:
```python
WEEKEND_CLOSE_UTC = (4, 21, 0)   # (weekday Mon=0..Sun=6, hour, minute) — Fri 21:00
WEEKEND_OPEN_UTC  = (6, 21, 0)   # Sun 21:00
```

`data/fetcher.py:_is_forex_market_expected_open()` (line 272-304):
- Saturday → closed
- Sunday before 21:00 UTC → closed
- Friday after 21:00 UTC → closed
- All other times → open

Both checks use UTC. This means the gap-fill logic in
`broker/data_validator.py:_is_market_closed_gap()` (line 187-253)
correctly skips Friday-close→Sunday-open gaps instead of synthesizing
~191 fake flat-fill candles (per the inline comment at line 142-152).

### 10.6 `is_candle_closed()` parity

`core/production_hardening.py:218-279` handles two cases explicitly:

1. `last_bar_time` carries tzinfo → use as-is
2. `last_bar_time` is naive → attach UTC label, emit WARNING, also emit
   CRITICAL if the resulting close time ends up in the future
   ("FUTURE_BAR" sentinel)

This is the consumer-side defense against the broker-tz bug. It catches
the symptom (future-dated bar) but doesn't fix the cause (broker
returning server time mislabeled as UTC) — that fix lives in
`DataFetcher._fetch_mt5()`.

### 10.7 Timezone summary

| Layer                       | Timezone Tag               | Broker Offset Correction         |
|-----------------------------|----------------------------|----------------------------------|
| `DataFetcher._fetch_mt5()` | tz-aware UTC              | ✅ Auto-detected (cached 30 min) |
| `broker/mt5_data.py`       | tz-aware UTC (`fromtimestamp(..., tz=utc)`) | ❌ None |
| `broker/mt5_historical_fetcher.py` | NAIVE (`pd.to_datetime(unit="s")`) | ❌ None |
| `data/live_feed.py`        | tz-aware UTC (ISO 8601)   | ❌ None (uses tick.time directly) |
| `data/data_orchestrator.py:get_tick()` | Raw epoch int (`tick.time`) | ❌ None |
| `HistoricalCSVProvider`    | tz-aware UTC              | N/A (CSVs pre-converted)         |
| `HistoricalMT5Provider`    | Inherits from df          | N/A                              |
| `LiveMT5Provider.current_time()` | NAIVE (`datetime.utcnow()`) | ❌ None |
| Sessions (`session_analyzer.py`) | tz-aware UTC           | N/A                              |
| Weekend gap-fill (`data_validator.py`) | tz-aware UTC    | N/A                              |

The mixed naive/tz-aware situation is the single biggest source of
parity risk in the data layer.

---

## 11. Live Data Path Diagram

```
                    ┌──────────────────────────────────────┐
                    │   MT5 Terminal (Windows + broker)    │
                    └──────────────────┬───────────────────┘
                                       │  (IPC, shared session)
                    ┌──────────────────▼───────────────────┐
                    │  broker/mt5_connection.MT5Connection  │
                    │  (singleton per login+server, locked) │
                    └─┬─────────────┬──────────┬────────────┘
                      │             │          │
              symbol_info_tick   symbol_info   copy_rates_from_pos
                      │             │          │
        ┌─────────────▼──┐   ┌──────▼─────┐  ┌─▼────────────────────┐
        │ broker/mt5_data │   │ data/      │  │ data/fetcher.py       │
        │ .py (get_tick,  │   │ live_feed  │  │ _fetch_mt5 (lines     │
        │  get_candles)   │   │ .py        │  │  727-1158)            │
        │ ⚠ no broker-tz  │   │ (snapshot) │  │ ✅ broker-tz auto-detect│
        │   correction   │   │            │  │ ✅ tz_localize('UTC') │
        └────────┬───────┘   └──────┬─────┘  │ ✅ drops forming bar  │
                 │                  │        └───────────┬───────────┘
                 │                  │                    │
                 ▼                  ▼                    ▼
        ┌────────────────────────────────────────────────────┐
        │  data/data_orchestrator.DataOrchestrator            │
        │  (singleton, shares MT5Connection with fetcher)    │
        │  get_candles → fetcher.fetch_ohlcv                  │
        │  get_tick    → mt5.symbol_info_tick (direct)         │
        │  get_symbol_info → mt5.symbol_info (direct)         │
        └──────────────────────┬─────────────────────────────┘
                               │
                ┌──────────────▼────────────────────────┐
                │  agents/market_agent.MarketAgent.run()│
                │  ├─ MultiTimeframeAnalyzer.analyze(   │
                │  │     ["1d","4h","1h","15m"])         │
                │  │   → fetch_ohlcv per TF (limit=300)  │
                │  ├─ DataValidator().validate(df)       │
                │  ├─ add_canonical_indicators(df)       │
                │  │   → indicator_registry → ext → legacy│
                │  ├─ MarketRegimeDetector.detect(df)    │
                │  └─ returns MarketAgentResult dict     │
                └──────────────┬────────────────────────┘
                               │
                ┌──────────────▼────────────────────────┐
                │  core/data_provider.LiveMT5Provider    │
                │  .get_market_out(symbol, tf)           │
                │  → MarketAgent.run() (thin wrapper)   │
                │  .current_time() → datetime.utcnow()   │
                │                   ⚠ NAIVE              │
                └───────────────────────────────────────┘
```

Side-path (NOT through MT5Connection lock — R3):
```
analysis/microstructure.py:_fetch_ticks()
  ├─ mt5.initialize()         ⚠ DIRECT, bypasses lock
  └─ mt5.copy_ticks_range(symbol, utc_from, utc_to, COPY_TICKS_ALL)
      → reads fields: time, bid, ask, last, volume_real/volume, flags
```

---

## 12. Historical Data Path Diagram

```
            ┌──────────────────────────────────────────┐
            │  data/{SYMBOL}_{TF}.csv (flat layout)    │
            │  Schema: datetime_utc,open,high,low,    │
            │          close,tick_volume,spread,       │
            │          real_volume                      │
            │  (UTC timestamps, pre-converted)         │
            └──────────────────┬───────────────────────┘
                               │
                ┌──────────────▼────────────────────────┐
                │  core/csv_data_provider._load_csv()   │
                │  ├─ pd.read_csv(encoding="utf-8-sig")│
                │  ├─ ts_col = "datetime_utc" (preferred)│
                │  │     fallback: datetime/time/timestamp/date│
                │  ├─ pd.to_datetime(ts_col, utc=True) │ ✅ tz-aware UTC
                │  ├─ dropna, sort_index, dedupe        │
                │  ├─ rename tick_volume → volume      │
                │  └─ coerce OHLC/spread to numeric    │
                └──────────────┬────────────────────────┘
                               │
                ┌──────────────▼────────────────────────┐
                │  HistoricalCSVProvider.__init__       │
                │  ├─ load primary TF CSV               │
                │  ├─ load higher TF CSVs (H1, H4, D1)  │
                │  ├─ resample M15→H1/H4 if CSV missing │
                │  ├─ register_series into              │
                │  │   data.backtest_ohlcv_cache         │
                │  └─ _resolve_indicator_mode() (once)   │
                └──────────────┬────────────────────────┘
                               │
                               │  advance_to(bar_index)
                               │  set_asof(cursor_time)  ←─┐
                               │                            │ shared with
                ┌──────────────▼────────────────────────┐    │ SMCEngine /
                │  HistoricalCSVProvider               │    │ MTFAnalyzer
                │  .get_market_out(symbol, tf)         │    │ (backtest mode)
                │  ├─ df_slice = primary_df.iloc[       │    │
                │  │     cursor-300 : cursor+1].copy()  │    │
                │  ├─ add_canonical_indicators(df_slice)│    │
                │  │   → registry → ext → legacy        │    │
                │  ├─ MarketRegimeDetector.detect       │    │
                │  ├─ _get_spread_pips()                │    │
                │  │   → ind_ctx["spread_pips"]         │    │
                │  └─ _compute_mtf_bias_from_csvs()      │    │
                │      (causal: only closed higher-TF)  │    │
                │  .current_time() → primary_df.index[   │    │
                │                    cursor]  ✅ tz-aware│    │
                └───────────────────────────────────────┘
```

---

## 13. Field-Level Comparison Table — Live MT5 vs Historical CSV

| Field               | Live MT5                                                  | Historical CSV                                              | Parity Status |
|---------------------|-----------------------------------------------------------|-------------------------------------------------------------|---------------|
| **bar open time**   | `time` field (epoch sec, broker-tz corrected → tz-aware UTC) | `datetime_utc` column (ISO 8601 UTC, parsed with `utc=True`) | ✅ Both tz-aware UTC (after fetcher fix) |
| **bar open time precision** | 1 second (epoch)                                    | 1 second (CSV ISO 8601)                                     | ✅ Match (sub-second `time_msc` discarded on live) |
| **open**            | `r["open"]` (float)                                       | `open` column (float)                                       | ✅ Match |
| **high**            | `r["high"]`                                               | `high`                                                      | ✅ Match |
| **low**             | `r["low"]`                                                | `low`                                                       | ✅ Match |
| **close**           | `r["close"]`                                              | `close`                                                     | ✅ Match |
| **tick_volume**     | `r["tick_volume"]` (renamed to `volume`)                  | `tick_volume` column (renamed to `volume`)                  | ✅ Match (both = tick activity proxy) |
| **real_volume**     | `r["real_volume"]` (NEVER READ — dropped at fetcher.py:966) | `real_volume` column (always 0 in samples, preserved in df but dropped at backtest_ohlcv_cache.py:96-105) | ⚠ Both effectively absent; field exists but unused |
| **spread**          | `r["spread"]` (int points)                                | `spread` column (int points)                                | ✅ Match (units: points) |
| **spread_pips (derived)** | `(ask-bid) × 10^(digits-1)` (live_feed.py, mt5_data.py) | `spread_points / 10` for FX, `/1` for XAU/indices (csv_data_provider.py:448-456) | ✅ Match (both convert points→pips using digits) |
| **bid**             | `tick.bid` (live, real-time)                              | ❌ NOT IN CSV                                                | ⚠ Historical has no bid; backtest uses `close` |
| **ask**             | `tick.ask` (live, real-time)                              | ❌ NOT IN CSV                                                | ⚠ Historical has no ask; backtest uses `close` |
| **last**            | `tick.last` (tick struct)                                 | ❌ NOT IN CSV                                                | ⚠ N/A for bar data |
| **time_msc**        | `tick.time_msc` (NEVER READ)                              | ❌ NOT IN CSV                                                | ⚠ Sub-ms precision lost on live |
| **digits**          | `info.digits` (live symbol_info)                          | N/A (looked up from `backtest.symbol_specs.get_pip_size`)   | ⚠ Different sources — live from broker, historical from static table |
| **point**           | `info.point` (live symbol_info)                           | N/A (derived from pip_size)                                  | ⚠ Same as above |
| **trade_contract_size** | `info.trade_contract_size`                           | N/A                                                          | ⚠ Same |
| **forming bar**     | Dropped at fetch time (fetcher.py:972-1004)                | Filtered out by causal slice (cursor < close_time)          | ✅ Both exclude forming bars |
| **MT5 timeframe constant** | `mt5.TIMEFRAME_M15` etc.                            | `"M15"` string → `_normalize_timeframe()` → canonical       | ✅ Match |
| **Default lookback**| `limit=300` (MarketAgent, DataFetcher)                   | `lookback_bars=300` (CSVProvider)                           | ✅ Match (intentional parity) |
| **Volume semantics** | Tick count (decentralized FX, no real volume)            | Tick count (same)                                            | ✅ Match — both labeled `volume` = tick_volume |
| **OHLC ordering**   | Sorted ascending + deduped (fetcher.py:942-943)           | Sorted ascending + deduped (csv_data_provider.py:160-163)  | ✅ Match |
| **NA handling**      | dropna on `time`; coerce numeric with errors="coerce"    | dropna on ts_col; to_numeric(errors="coerce").fillna(0)    | ✅ Match |
| **Indicator chain** | `add_canonical_indicators` → ExtendedIndicators → Indicators | Same (csv_data_provider.py:368-381)                       | ✅ Match (single chain, single source of truth) |
| **Regime detection** | `MarketRegimeDetector.detect(df)`                        | Same (csv_data_provider.py:392-395)                          | ✅ Match |
| **MTF bias source** | Live `MultiTimeframeAnalyzer.analyze()` over fetched HTF bars | `_compute_mtf_bias_from_csvs()` over loaded HTF CSVs     | ⚠ Same EMA logic, different data source (live vs CSV); both causal |
| **MTF bias output shape** | `{"bias": ..., "confidence": ...}`                   | `{"bias": ..., "confidence": ...}`                          | ✅ Match |

---

## 14. Appendix A — CSV Column Lists Per File

### 14.1 All `data/*.csv` files (verified uniform)

Every file in `/home/z/my-project/download/forex-agent/data/*.csv`
(21 files: 7 symbols × 3 timeframes) shares the **identical** schema:

```
datetime_utc,open,high,low,close,tick_volume,spread,real_volume
```

### 14.2 Per-file first 3 lines

**AUDUSD_M15.csv**
```
datetime_utc,open,high,low,close,tick_volume,spread,real_volume
2025-07-25 06:30:00+00:00,0.65771,0.65818,0.65771,0.65798,664,18,0
2025-07-25 06:45:00+00:00,0.65798,0.65801,0.6576,0.65789,571,18,0
```

**EURUSD_H1.csv**
```
datetime_utc,open,high,low,close,tick_volume,spread,real_volume
2025-07-25 07:00:00+00:00,1.17473,1.1758899999999999,1.17473,1.17523,3210,8,0
2025-07-25 08:00:00+00:00,1.17522,1.17575,1.1741,1.17427,3232,8,0
```
(Last bar: `2026-07-24 20:00:00+00:00,1.13701,1.13735,1.13669,1.13705,4425,0,0`)

**USDJPY_H4.csv**
```
datetime_utc,open,high,low,close,tick_volume,spread,real_volume
2025-07-25 09:00:00+00:00,147.747,147.931,147.597,147.847,11659,4,0
2025-07-25 13:00:00+00:00,147.846,147.912,147.521,147.75,18355,4,0
```

**NZDUSD_M15.csv**
```
datetime_utc,open,high,low,close,tick_volume,spread,real_volume
2025-07-25 06:30:00+00:00,0.60225,0.60272,0.60223,0.60253,584,20,0
2025-07-25 06:45:00+00:00,0.60253,0.60257,0.60223,0.60234,522,20,0
```

**USDCAD_H4.csv**
```
datetime_utc,open,high,low,close,tick_volume,spread,real_volume
2025-07-25 09:00:00+00:00,1.3675,1.36876,1.36663,1.36842,5959,4,0
2025-07-25 13:00:00+00:00,1.36841,1.37247,1.3676300000000001,1.37153,12608,4,0
```

**GBPUSD_H1.csv**
```
datetime_utc,open,high,low,close,tick_volume,spread,real_volume
2025-07-25 07:00:00+00:00,1.3489200000000001,1.34914,1.34659,1.3467500000000001,3390,18,0
2025-07-25 08:00:00+00:00,1.34676,1.34756,1.34585,1.3462100000000001,3085,18,0
```

**EURUSD_M15.csv**
```
datetime_utc,open,high,low,close,tick_volume,spread,real_volume
2025-07-25 06:30:00+00:00,1.1749100000000001,1.17603,1.17485,1.17578,777,8,0
2025-07-25 06:45:00+00:00,1.17578,1.17583,1.17463,1.17473,786,8,0
```

**AUDUSD_H1.csv**
```
datetime_utc,open,high,low,close,tick_volume,spread,real_volume
2025-07-25 07:00:00+00:00,0.65789,0.65804,0.65693,0.65713,2531,18,0
2025-07-25 08:00:00+00:00,0.65714,0.65734,0.65634,0.65675,2398,18,0
```

**USDJPY_H1.csv**
```
datetime_utc,open,high,low,close,tick_volume,spread,real_volume
2025-07-25 07:00:00+00:00,147.046,147.407,147.033,147.375,4086,13,0
2025-07-25 08:00:00+00:00,147.375,147.823,147.352,147.746,4548,13,0
```

**USDCHF_H4.csv**
```
datetime_utc,open,high,low,close,tick_volume,spread,real_volume
2025-07-25 09:00:00+00:00,0.79583,0.79752,0.79569,0.79712,8862,11,0
2025-07-25 13:00:00+00:00,0.7971,0.79783,0.79534,0.79536,12309,11,0
```

**EURUSD_H4.csv** (note: spread=0 on early bars)
```
datetime_utc,open,high,low,close,tick_volume,spread,real_volume
2025-07-25 09:00:00+00:00,1.17426,1.17431,1.17112,1.17177,9802,0,0
2025-07-25 13:00:00+00:00,1.1718,1.17381,1.17027,1.1728399999999999,14693,0,0
```

**NZDUSD_H4.csv**
```
datetime_utc,open,high,low,close,tick_volume,spread,real_volume
2025-07-25 09:00:00+00:00,0.60091,0.60152,0.60018,0.60034,6375,10,0
2025-07-25 13:00:00+00:00,0.60035,0.60141,0.59968,0.60091,11446,10,0
```

### 14.3 Schema observations

1. **Column order is identical** across all 21 files: `datetime_utc` is
   always first; `real_volume` is always last.
2. **Timestamp format**: ISO 8601 with explicit `+00:00` UTC offset —
   unambiguous. No naive timestamps in any file.
3. **OHLC precision**: floats with up to 16 significant digits (Python
   `repr` of the float; e.g. `1.1758899999999999` is the inexact
   representation of `1.17589`). Round-trips to the same value.
4. **`tick_volume`**: integer (e.g. `3210`, `11659`).
5. **`spread`**: integer in POINTS (e.g. `8`, `18`, `4`). Conversion to
   pips: divide by 10 for 5-digit FX (digits=5, point=0.00001, pip=0.0001)
   or 3-digit JPY (digits=3, point=0.001, pip=0.01); divide by 1 for
   XAU/indices (pip == point).
6. **`real_volume`**: always `0` in every sampled row across every file
   — matches the manifest's `"real_volume_available": false` field.
7. **`spread=0` bars**: observed on EURUSD_H4 first two bars (and per
   manifest, `spread_nonzero_pct` varies — some files have many zero
   bars). The CSV provider handles this via recent-mean fallback.

### 14.4 Other CSV files (not in the backtest pipeline)

| File                                            | Schema                                  | Used by                          |
|-------------------------------------------------|-----------------------------------------|----------------------------------|
| `data/metrics/*.csv` (22 files)                 | Various metric/trade journal schemas    | Reports only — not data provider |
| `data/forex/{PAIR}_daily.csv` (if exists)       | `Open,High,Low,Close,Volume` (capital) | `data/automated_updater.py` only — DISCONNECTED from `HistoricalCSVProvider` |
| `backtest/final_validated_trades.csv`, etc.     | Trade journals                          | Analytics only                   |
| `_backtest_validation/csv/*.csv`                | Various report CSVs                     | Validation reports only          |

---

## 15. Appendix B — Method/Class Reference (Key Line Numbers)

| Symbol                                  | File:Line                            |
|-----------------------------------------|--------------------------------------|
| `DataProvider` (ABC)                    | `core/data_provider.py:38`           |
| `LiveMT5Provider`                       | `core/data_provider.py:60`           |
| `LiveMT5Provider.get_market_out()`      | `core/data_provider.py:69`           |
| `LiveMT5Provider.current_time()` ⚠       | `core/data_provider.py:76` (naive)   |
| `HistoricalMT5Provider`                 | `core/data_provider.py:81`           |
| `HistoricalMT5Provider.get_market_out()`| `core/data_provider.py:130`          |
| `HistoricalMT5Provider._compute_mtf_bias()` | `core/data_provider.py:213`      |
| `HistoricalCSVDataProvider`             | `core/csv_data_provider.py:188`       |
| `_load_csv()`                           | `core/csv_data_provider.py:136`       |
| `_find_csv()`                           | `core/csv_data_provider.py:79`        |
| `_normalize_timeframe()`                | `core/csv_data_provider.py:109`      |
| `_tf_to_minutes()`                      | `core/csv_data_provider.py:125`      |
| `_get_spread_pips()`                    | `core/csv_data_provider.py:421`      |
| `_compute_mtf_bias_from_csvs()`         | `core/csv_data_provider.py:458`      |
| `make_backtest_provider()`              | `core/provider_factory.py:38`       |
| `ExecutionAdapter` (ABC)                | `core/execution_adapter.py:31`      |
| `MT5ExecutionAdapter`                   | `core/execution_adapter.py:46`       |
| `HistoricalExecutionAdapter`            | `core/execution_adapter.py:85`      |
| `MT5DataFeed`                           | `broker/mt5_data.py:127`            |
| `MT5DataFeed.get_tick()`                | `broker/mt5_data.py:148`            |
| `MT5DataFeed.get_candles()`             | `broker/mt5_data.py:198`            |
| `MT5DataFeed.get_multi_timeframe()`     | `broker/mt5_data.py:235`            |
| `MT5DataFeed.save_live_csv()`           | `broker/mt5_data.py:270`            |
| `fetch_historical_data()`              | `broker/mt5_historical_fetcher.py:60` |
| `fetch_and_cache()`                     | `broker/mt5_historical_fetcher.py:131` |
| `MarketDataManager`                     | `broker/market_data_manager.py:19`  |
| `MarketDataManager.get_clean_bundle()`  | `broker/market_data_manager.py:46`  |
| `DataValidator` (broker)                | `broker/data_validator.py:25`       |
| `DataValidator.validate_and_fill()`     | `broker/data_validator.py:42`       |
| `DataValidator._is_market_closed_gap()` | `broker/data_validator.py:187`      |
| `WEEKEND_CLOSE_UTC` / `WEEKEND_OPEN_UTC`| `broker/data_validator.py:180-181`  |
| `MT5Connection.get_tick()`              | `broker/mt5_connection.py:514`      |
| `MT5Connection.symbol_info()`           | `broker/mt5_connection.py:543`      |
| `MT5Connection.symbol_select()`        | `broker/mt5_connection.py:616`      |
| `MT5Connection.copy_rates_from_pos()`  | `broker/mt5_connection.py:627`      |
| `DataFetcher.__init__()`                | `data/fetcher.py:315`               |
| `DataFetcher.fetch_ohlcv()`             | `data/fetcher.py:506`               |
| `DataFetcher._fetch_mt5()`              | `data/fetcher.py:727`               |
| `DataFetcher._get_broker_utc_offset_hours()` | `data/fetcher.py:662`          |
| `DataFetcher.detect_broker_tz_offset()` | `data/fetcher.py:1167`              |
| `DataFetcher._resample_h1_to_h4()`      | `data/fetcher.py:1310`              |
| `DataFetcher._fetch_tvdatafeed()`        | `data/fetcher.py:1255`              |
| `DataFetcher._fetch_yfinance()`         | `data/fetcher.py:1338`              |
| `DataFetcher._fetch_alpha_vantage()`     | `data/fetcher.py:1524`              |
| `DataFetcher._fetch_polygon()`           | `data/fetcher.py:1665`              |
| `DataFetcher._fetch_finnhub()`           | `data/fetcher.py:1765`              |
| `DataFetcher._fetch_twelve_data()`       | `data/fetcher.py:1859`              |
| `DataOrchestrator.get_candles()`         | `data/data_orchestrator.py:191`     |
| `DataOrchestrator.get_tick()`           | `data/data_orchestrator.py:346`     |
| `DataOrchestrator.get_symbol_info()`    | `data/data_orchestrator.py:304`     |
| `DataOrchestrator.get_multi_timeframe()`| `data/data_orchestrator.py:376`     |
| `LiveFeed`                              | `data/live_feed.py:149`             |
| `LiveFeed.get_snapshot()`               | `data/live_feed.py:206`             |
| `LiveFeed.is_safe_to_trade()`           | `data/live_feed.py:337`             |
| `SPREAD_LIMITS_PIPS` table              | `data/live_feed.py:65-91`           |
| `DataValidator` (data)                  | `data/validator.py:16`              |
| `DataValidator.validate()`              | `data/validator.py:22`              |
| `add_canonical_indicators()`            | `data/indicator_registry.py:103`    |
| `CANONICAL_SOURCES`                     | `data/indicator_registry.py:66`     |
| `ExtendedIndicators`                    | `data/indicators_ext.py`            |
| `Indicators`                            | `data/indicators.py:8`              |
| `set_asof()` / `register_series()` / `get_ohlcv()` / `resample_ohlcv()` / `register_from_m15()` | `data/backtest_ohlcv_cache.py:70/79/148/110/136` |
| `lookahead_self_check()`                | `data/backtest_ohlcv_cache.py:184`   |
| `ForexDataUpdater` (orphan)             | `data/automated_updater.py:24`       |
| `verify_data_coverage()`                | `data/verify_data_coverage.py:19`   |
| `CompressedQuoteStorage`                | `data/compressed_storage.py:99`     |
| `MultiSymbolStorage`                    | `data/compressed_storage.py:354`    |
| `check_data_staleness()`                | `core/production_hardening.py:167`  |
| `is_candle_closed()`                    | `core/production_hardening.py:218`  |
| `Microstructure._fetch_ticks()` ⚠       | `analysis/microstructure.py:130`    |
| `SessionAnalyzer.get_current_session()`| `analysis/session_analyzer.py:63`   |

---

## 16. Summary — Recommended Next Actions

1. **(R1, R3)** Route `analysis/microstructure.py` and `broker/mt5_data.py`
   through `MT5Connection` (add `copy_ticks_range` wrapper; delete
   `mt5_data.py` direct `mt5.symbol_info_tick` call). Eliminates the
   last two unlocked-MT5-call paths.
2. **(R4)** Fix `LiveMT5Provider.current_time()` to return
   `datetime.datetime.now(timezone.utc)` (one-line fix, eliminates a
   tz-mismatch crash).
3. **(R2)** Decide policy on `real_volume`: either keep it in the
   column whitelist (so future CSVs with non-zero real_volume are
   propagated) or document explicitly that real_volume is permanently
   unused and remove the column from the CSV schema for clarity.
4. **(R6)** Delete `data/automated_updater.py` and
   `data/verify_data_coverage.py` (orphan modules from pre-refactor
   era; not wired into the `HistoricalCSVProvider` pipeline). Use
   `scripts/download_historical_data.py` for CSV refresh going forward.
5. **(R5)** Verify the DEFAULT_SPREAD_PIPS fallback table (in
   `core/constants.py` / `broker/spread_monitor.py`) matches typical
   retail broker spreads — this is the only safety net for CSV bars
   with `spread=0`. Spot-check 1-2 symbols against the live MT5
   `info.spread` field.
6. **Document** the broker-tz auto-detection contract: any new MT5
   caller MUST go through `DataFetcher._fetch_mt5()` (which applies
   the offset) or `MT5Connection.copy_rates_from_pos()` PLUS the
   offset correction. Adding a one-line check in CI that no `mt5.copy_*`
   call exists outside `data/fetcher.py`, `broker/mt5_connection.py`,
   and `broker/mt5_historical_fetcher.py` would prevent regression.

---

**End of P1-A audit.**
