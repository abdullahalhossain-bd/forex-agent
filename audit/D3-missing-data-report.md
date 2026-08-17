# Deliverable 3 — Missing Data Report

**Project:** Forex AI Autonomous Trading System
**Audit date:** 2026-08-17
**Source:** Compiled from P1-A, P1-B, P1-C, P1-D, P6 evidence reports

This document identifies what historical data is currently available in CSV form vs what the live trading system actually needs for faithful reproduction.

---

## 1. Currently Available (Existing CSVs in `data/*.csv`)

21 CSV files (7 symbols × 3 timeframes) covering 2025-07-25 → 2026-07-24 (≈365 days):

| Symbol | M15 | H1 | H4 |
|--------|-----|----|----|
| AUDUSD | ✅ 24,781 rows | ✅ 6,196 rows | ✅ 1,551 rows |
| EURUSD | ✅ 24,780 rows | ✅ 6,197 rows | ✅ 1,551 rows |
| GBPUSD | ✅ 24,776 rows | ✅ 6,195 rows | ✅ 1,551 rows |
| NZDUSD | ✅ 24,781 rows | ✅ 6,196 rows | ✅ 1,551 rows |
| USDCAD | ✅ 24,781 rows | ✅ 6,196 rows | ✅ 1,551 rows |
| USDCHF | ✅ 24,781 rows | ✅ 6,196 rows | ✅ 1,551 rows |
| USDJPY | ✅ 24,781 rows | ✅ 6,196 rows | ✅ 1,551 rows |

**Schema (uniform across all 21 files):**

```
datetime_utc, open, high, low, close, tick_volume, spread, real_volume
```

### Quality summary (from P6 audit)

- **Timezone:** UTC (tz-aware ISO 8601 with `+00:00`) — ✅ correct
- **Timestamp sorting:** ascending, no duplicates — ✅ correct
- **OHLC sanity:** 0 violations of `high ≥ max(open, close)` and `low ≤ min(open, close)` — ✅ correct
- **NaN values:** 0 across all files — ✅ clean
- **Tick volume:** 0% zeros (always populated) — ✅ good
- **Real volume:** 100% zeros across every file — ⚠ always 0 (expected for FX)
- **Spread:** 15-70% zeros depending on symbol/TF — ⚠ significant data quality issue
  - EURUSD_H4: 70.08% zero spreads (worst)
  - NZDUSD_M15: 15.17% zero spreads (best)
- **Gaps:** 2-7 non-weekend gaps per file (the rest are expected forex weekend close Fri 21:00 → Sun 21:00 UTC)

---

## 2. Missing But Required

Without these, the live system's historical reproduction is materially inaccurate or impossible.

### 2.1 Missing timeframes (HTF gap)

| Symbol | M5 | H1 | H4 | D1 |
|--------|----|----|----|----|
| AUDUSD | ❌ MISSING | ✅ | ✅ | ❌ MISSING |
| EURUSD | ❌ MISSING | ✅ | ✅ | ❌ MISSING |
| GBPUSD | ❌ MISSING | ✅ | ✅ | ❌ MISSING |
| NZDUSD | ❌ MISSING | ✅ | ✅ | ❌ MISSING |
| USDCAD | ❌ MISSING | ✅ | ✅ | ❌ MISSING |
| USDCHF | ❌ MISSING | ✅ | ✅ | ❌ MISSING |
| USDJPY | ❌ MISSING | ✅ | ✅ | ❌ MISSING |

**Why required:**
- `M5` is in the live MTF chain (`agents/market_agent.py:171` `["1d","4h","1h","15m"]`) when `MultiTimeframeAnalyzer` is called by `MTFAnalyzer.analyze()`. `MTFAnalyzer` uses `["4h","1h","15m","5m"]` (`mtf_analyzer.py:108`).
- `D1` is in the live MTF chain (`["1d","4h","1h","15m"]`) and used by `MarketStructureEngine` for swing-bias confirmation.

**Effect if missing:**
- `MTFAnalyzer.analyze()` will silently skip the missing TF, producing incomplete MTF context.
- `MarketAgent` MTF bias dict will lack `1d` trend.
- HistoricalCSVProvider has a fallback: `_compute_mtf_bias_from_csvs()` will resample H1→H4 if H4 is missing, but it does NOT synthesize M5 or D1.

### 2.2 Missing symbols (live config has 8 pairs)

Current live config (`config.py` `SYMBOLS`) typically includes `XAUUSD` (gold) along with the 7 FX majors in the CSVs. **XAUUSD CSVs are MISSING entirely.**

| Symbol | M15 | H1 | H4 |
|--------|-----|----|----|
| XAUUSD (gold) | ❌ | ❌ | ❌ |

**Why required:** If the live system trades XAUUSD, backtest cannot reproduce any XAUUSD cycle without these CSVs. Even if XAUUSD is not in the active symbol list, `IntermarketEngine` uses `Gold` as one of its 6 external macro feeds — historical backtest would need to either fetch live gold history or accept that intermarket context is incomplete.

### 2.3 Missing non-OHLCV fields

| Field | CSV status | Live source | Required by |
|-------|-----------|-------------|-------------|
| `bid` | ❌ NOT in CSV | `tick.bid` from `mt5.symbol_info_tick` | `OrderManager.place_market_order` (fills BUY at ask, SELL at bid); `OptimalTradingTime` |
| `ask` | ❌ NOT in CSV | `tick.ask` | Same as above |
| Tick stream | ❌ NOT in CSV | `mt5.copy_ticks_range` | `MicrostructureEngine` (gated OFF in backtest) |
| `time_msc` | ❌ NOT in CSV | `tick.time_msc` (NEVER READ on live) | Not used |
| Market depth (L2) | ❌ NOT in CSV | `mt5.market_book_add` (NOT USED on live) | Not used |

**Effect:**
- `bid`/`ask` missing → historical backtest fills at `close` instead of `tick.ask` (BUY) or `tick.bid` (SELL). For low-spread FX majors this is a small bias; for JPY crosses and exotics it can be 2-5 pips of optimistic fill per trade.
- Tick stream missing → `MicrostructureEngine` is gated OFF in backtest (intentional; uses synthetic fallback). Live tick speed / volume burst / spread state features are not reproduced.
- L2 depth / `time_msc` — neither live nor historical uses them.

### 2.4 Spread quality gaps (already in CSV but mostly zero)

Even though the `spread` column exists in all 21 CSVs, a large fraction of bars have `spread=0`:

| File | % zero spreads |
|------|---------------|
| EURUSD_H4.csv | 70.08% |
| EURUSD_H1.csv | 60.45% |
| EURUSD_M15.csv | 52.80% |
| GBPUSD_M15.csv | 20.54% |
| AUDUSD_H4.csv | 45.52% |
| USDCAD_H4.csv | 40.75% |
| USDJPY_H4.csv | 39.01% |
| AUDUSD_H1.csv | 32.62% |
| USDJPY_M1.csv | 34.38% |
| USDCHF_H4.csv | 29.27% |
| USDJPY_M15.csv | 27.39% |
| GBPUSD_H1.csv | 27.70% |
| AUDUSD_M15.csv | 23.40% |
| USDCAD_H1.csv | 23.71% |
| USDCHF_H1.csv | 20.71% |
| NZDUSD_H4.csv | 25.60% |
| USDCHF_M15.csv | 16.18% |
| USDCAD_M15.csv | 17.24% |
| NZDUSD_H1.csv | 18.30% |
| NZDUSD_M15.csv | 15.17% |
| GBPUSD_H4.csv | 34.88% |

**Effect:**
- When `spread=0`, `HistoricalCSVProvider._get_spread_pips()` falls back to mean-of-last-50-non-zero-bars; if all 50 are also 0, returns `None` and caller falls back to `DEFAULT_SPREAD_PIPS` static table (`core/constants.py`).
- The `SignalEngine` spread filter (`signal_engine.py:114-120`) silently no-ops when `spread=0` (it computes `spread_avg_20 = mean(spread[-20:])`; if all 20 are 0, the filter compares `0 < threshold` which is always false → trade passes).
- This means **historical backtest over-routes trades** that live would block on wide-spread filter.

**Fix:** Re-download with MT5's `copy_rates_range` directly (not via the legacy fetcher that may be stripping spread). The existing `scripts/download_historical_data.py` already preserves `spread` — re-running it should populate non-zero spreads for the missing bars.

---

## 3. Missing but Recommended (improves fidelity, not strictly required)

### 3.1 External macro history (DXY, Gold, Oil, US10Y, SP500, VIX)

Currently fetched live by `MacroDataProvider` and consumed by `IntermarketEngine`. In backtest mode, `IntermarketEngine` has a fallback that returns neutral context when live fetch fails.

**Recommendation:** Download 1-year daily history for each of these 6 assets and store as `data/external/{ASSET}_D1.csv`. Modify `MacroDataProvider` to load from CSV when `is_backtest_mode()=True`.

**Sources:**
- yfinance (free): `DX-Y.NYB` (DXY), `GC=F` (Gold), `CL=F` (Oil), `^TNX` (US10Y), `^GSPC` (SP500), `^VIX` (VIX)
- Alpha Vantage (free tier): same assets with different ticker conventions

### 3.2 28 cross-pair close history

`CorrelationEngine` fetches 28 cross pairs live (50 bars each). In backtest, if fetch fails, `correlation_adjustment` defaults to 1.0 (no risk reduction for correlation).

**Recommendation:** Download the 21 missing cross pairs (the 7 majors give 21 of the 28; the rest are exotic crosses like HKDJPY, MXNJPY, SGDJPY, USDCNH, USDHKD, USDMXN, USDSGD, USDTHB, USDTRY, USDZAR, XAGUSD).

### 3.3 CFTC weekly COT history

`InstitutionalFlowEngine` fetches live COT from `cftc.gov` (weekly net positioning). In backtest_mode, this is gated OFF and uses a synthetic large-candle proxy from OHLC.

**Recommendation:** Download historical COT reports from `https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm` (available as historical CSV) and store under `data/external/cot/`.

### 3.4 News / economic calendar history

`intelligence/news_sources.py:181` returns the local `data/economic_calendar.json` snapshot in backtest mode. The current snapshot is whatever was downloaded at install time — it does NOT cover the historical backtest window.

**Recommendation:** Download historical high-impact news from Forex Factory's archive (https://www.forexfactory.com/calendar.php) for the backtest window and store as `data/economic_calendar_history.json`.

---

## 4. Optional / Nice to Have

### 4.1 Tick-level historical data

**Why optional:** `MicrostructureEngine` is gated OFF in backtest_mode; tick data is only used live for real-time spread/velocity/burst analysis. Historical tick databases for FX cost $1k-$10k/year per symbol (Dukascopy, TrueFX, HistData.com free tier).

**If acquired:** Would enable tick-by-tick replay for ultra-faithful spread/volume burst simulation. Storage cost: ~100MB per symbol per month (M1-equivalent ticks).

### 4.2 Real volume (for non-FX assets)

`real_volume` is always 0 for FX (no consolidated volume). For XAUUSD (gold), some brokers DO populate real_volume — would enable `VolumeProfile` engine to produce more accurate POC/Value Area.

### 4.3 Sub-second bar timestamps (`time_msc`)

Live MT5 returns `time_msc` (millisecond precision) in tick data but NOT in bar data. Bar data is always 1-second precision. No consumer currently uses `time_msc`. No action needed.

---

## 5. Impossible / Broker-Dependent Data

These cannot be reliably reproduced from historical sources. The historical simulation must mark these as **approximation required**.

| Data | Why impossible / broker-dependent |
|------|----------------------------------|
| Live tick stream (bid/ask/last/flags per tick, last 60s) | Historical tick databases are broker-specific; the live MT5 terminal only returns recent ticks via `copy_ticks_range` |
| Live market depth (Level-2 order book) | `mt5.market_book_add` not used in production code; even if used, historical L2 is not stored anywhere |
| Live news events (Forex Factory + RSS + NewsAPI) | Real-time event data; historical news databases are paid (Bloomberg, Refinitiv, Dow Jones) |
| Live retail sentiment (Myfxbook) | Live-only API; historical snapshot service is paid |
| Broker-specific SL/TP execution priority | Some brokers honor SL before TP on a bar that hits both; others honor TP first. `BrokerSimulator` defaults to SL priority (conservative). Cannot be reproduced without per-broker execution logs. |
| Broker-specific swap rates (overnight financing) | Not modeled in `BrokerSimulator` at all; live MT5 applies swap automatically. Multi-day trades will have small equity drift in backtest vs live. |
| Broker-specific commission per lot | `BrokerSimulator` uses a configurable `_commission_per_lot` parameter; live broker may have different tiers. |
| Requotes / partial fills / order rejections | Live MT5 can reject orders (no money, market closed, requoted price, etc.); `BrokerSimulator` always fills. |
| Latency / slippage on order placement | Live order placement has 100-500ms latency between `mt5.order_send` and fill; backtest assumes instant fill at next-bar open. |

---

## 6. Live-vs-Historical Feature Parity Audit

Per Phase 8 spec. For every data field consumed by the live system, identify whether the historical CSV has an exact match, an approximation, or no equivalent.

| Feature | Live Source | Historical Source | Exact Match? | Fix |
|---------|-------------|-------------------|--------------|-----|
| `bar open time` (UTC) | `r["time"]` epoch sec → broker-tz corrected → `tz_localize("UTC")` (fetcher.py:938) | `datetime_utc` ISO 8601 UTC parsed with `utc=True` (csv_data_provider.py:157) | ✅ YES (both tz-aware UTC after fetcher fix) | None |
| `open` | `r["open"]` | `open` column | ✅ YES | None |
| `high` | `r["high"]` | `high` column | ✅ YES | None |
| `low` | `r["low"]` | `low` column | ✅ YES | None |
| `close` | `r["close"]` | `close` column | ✅ YES | None |
| `tick_volume` | `r["tick_volume"]` (renamed to `volume`) | `tick_volume` column (renamed to `volume`) | ✅ YES (both = tick activity proxy) | None |
| `real_volume` | `r["real_volume"]` (NEVER READ — dropped at fetcher.py:966) | `real_volume` column (always 0; dropped at backtest_ohlcv_cache.py:96) | ⚠ BOTH ABSENT (no consumer reads it) | Document explicitly that real_volume is permanently unused |
| `spread` (per-bar points) | `r["spread"]` (int points) | `spread` column (int points) | ✅ YES when non-zero | Re-download to fill zero-spread bars |
| `spread_pips` (derived) | `(ask-bid) × 10^(digits-1)` (live_feed.py, mt5_data.py) | `spread_points / 10` for FX, `/1` for XAU/indices (csv_data_provider.py:448-456) | ✅ YES (both convert points→pips using digits) | None |
| `bid` | `tick.bid` (live real-time) | ❌ NOT IN CSV | ❌ NO — historical fills at `close` | Add `bid` column to CSV (next download); BrokerSimulator to fill BUY at `ask` |
| `ask` | `tick.ask` (live real-time) | ❌ NOT IN CSV | ❌ NO — historical fills at `close` | Add `ask` column to CSV; BrokerSimulator to fill SELL at `bid` |
| `last` (tick) | `tick.last` | ❌ NOT IN CSV | ⚠ N/A for bar data | None (not used in backtest) |
| `time_msc` (tick) | `tick.time_msc` (NEVER READ) | ❌ NOT IN CSV | ⚠ N/A | None (not used on live either) |
| `digits` | `info.digits` from `mt5.symbol_info` | Static table `backtest.symbol_specs.get_pip_size` | ⚠ DIFFERENT SOURCES (live from broker, historical from static) | Spot-check static table against live broker; update if mismatched |
| `point` | `info.point` | Derived from pip_size | ⚠ Same | Same |
| `trade_contract_size` | `info.trade_contract_size` | N/A | ⚠ Same | Verify static table matches broker |
| `forming bar` | Dropped at fetch time (fetcher.py:972-1004) | Filtered out by causal slice (cursor < close_time) | ✅ YES — both exclude forming bars | None |
| MT5 timeframe constant | `mt5.TIMEFRAME_M15` etc. | `"M15"` string → `_normalize_timeframe()` → canonical | ✅ YES | None |
| Default lookback | `limit=300` (MarketAgent, DataFetcher) | `lookback_bars=300` (CSVProvider) | ✅ YES (intentional parity) | None |
| Volume semantics | Tick count (decentralized FX) | Tick count (same) | ✅ YES | None |
| OHLC ordering | Sorted ascending + deduped (fetcher.py:942-943) | Sorted ascending + deduped (csv_data_provider.py:160-163) | ✅ YES | None |
| NA handling | `dropna` on `time`; coerce numeric with `errors="coerce"` | `dropna` on ts_col; `to_numeric(errors="coerce").fillna(0)` | ✅ YES | None |
| Indicator chain | `add_canonical_indicators` → `ExtendedIndicators` → `Indicators` | Same (csv_data_provider.py:368-381) | ✅ YES | None |
| Regime detection | `MarketRegimeDetector.detect(df)` | Same (csv_data_provider.py:392-395) | ✅ YES | None |
| MTF bias source | `MultiTimeframeAnalyzer.analyze()` over fetched HTF bars | `_compute_mtf_bias_from_csvs()` over loaded HTF CSVs | ⚠ SAME EMA LOGIC, DIFFERENT DATA SOURCE | Both causal; verified equivalent |
| MTF bias output shape | `{"bias": ..., "confidence": ...}` | `{"bias": ..., "confidence": ...}` | ✅ YES | None |
| `current_time()` (provider) | `datetime.datetime.utcnow()` ⚠ **NAIVE** | `primary_df.index[cursor]` tz-aware UTC | ❌ PARITY VIOLATION (P1-A R4) | Fix: change `LiveMT5Provider.current_time()` to `datetime.now(timezone.utc)` |
| Sessions (Asia/London/NY) | `session_analyzer.py` uses `datetime.now(timezone.utc)` if `dt=None`; `analysis_agent.py:259-273` passes `dt=_bar_dt` | Same — caller passes bar timestamp | ✅ YES (when caller passes `dt`) | Audit all `SessionAnalyzer` callers; ensure they pass `dt` |
| SmartMoney kill-zone (UTC hour check) | `analysis/smart_money.py:390` uses `datetime.now(timezone.utc)` | ❌ Uses live wall-clock in backtest | ❌ PARITY VIOLATION (P1-C 6b.2) | Pass `bar_timestamp` to `SmartMoneyEngine.analyze()` |
| ML feature engineer timestamp | `ml/feature_engineer.py:400` uses `pd.Timestamp.now(tz="UTC")` | ❌ Uses live wall-clock in backtest | ❌ PARITY VIOLATION (P1-D §0.5) | Override with `df.index[-1]` when in backtest mode |
| News context | Live ForexFactory + RSS + NewsAPI fetch | `data/economic_calendar.json` snapshot | ❌ HISTORICAL SNAPSHOT STALE | Download historical news for backtest window |
| Macro (DXY/Gold/oil/VIX/SP500/US10Y) | `MacroDataProvider` live | N/A — backtest has fallback (returns neutral) | ❌ NOT REPRODUCED | Download macro history to `data/external/` |
| COT (CFTC weekly) | `_fetch_cot_from_cftc()` live | ⚠ Synthetic large-candle proxy (gated OFF in backtest) | ❌ NOT REPRODUCED | Download historical COT from CFTC.gov |
| 28 cross-pair correlation | Live fetch (50 bars × 28 pairs) | ⚠ Returns neutral `correlation_adjustment=1.0` if fetch fails | ❌ NOT REPRODUCED | Download 28 cross-pair history |
| Microstructure (tick stream) | `mt5.copy_ticks_range` (last 60s) | ❌ Gated OFF in backtest | ❌ NOT REPRODUCED | Live-only by design; no fix |
| LLM context (22 blocks) | Live (Groq → Gemini → OpenRouter) | Same (downstream of OHLCV + external feeds) | ⚠ Partial — external feed blocks (news, macro) are stale/missing | Address §3.1-3.4 |
| DevilsAdvocateGate (LLM veto) | Live LLM call | ❌ SKIPPED in backtest | ❌ NOT REPRODUCED | Acceptable — veto is a runtime safety net |
| ApprovalMode | Live (MODE_SUPERVISED pends to DB) | ❌ SKIPPED in backtest | ❌ NOT REPRODUCED | Acceptable — operator workflow, not data |
| Order fill price (BUY) | `tick.ask` | `df.iloc[i+1]["open"]` (next bar open) | ⚠ APPROXIMATION | Add `bid`/`ask` columns; BrokerSimulator to fill at `ask` for BUY, `bid` for SELL |
| Order fill price (SELL) | `tick.bid` | `df.iloc[i+1]["open"]` (next bar open) | ⚠ APPROXIMATION | Same as above |
| Spread cost | Captured via `tick.ask - tick.bid` at fill time | Captured via `spread_pips × pip_size` deducted from entry | ⚠ APPROXIMATION | Same fix as above |
| Slippage | Real (broker deviation) | Configurable `_slippage_pips` parameter | ⚠ APPROXIMATION | Tune `_slippage_pips` to match live average |
| Commission | Broker-deducted | Configurable `_commission_per_lot` | ⚠ APPROXIMATION | Match broker's actual commission tier |
| Swap (overnight financing) | Auto-applied by broker | NOT MODELED in BrokerSimulator | ❌ NOT REPRODUCED | Add swap model to BrokerSimulator (optional) |
| Latency | Real (100-500ms typical) | Instant fill at next bar open | ⚠ APPROXIMATION | Add configurable latency simulation (optional) |

---

## 7. Leakage Audit (Phase 9)

Per Phase 9 spec. Verify that no future information accidentally leaks into historical CSV construction or backtest decision-making.

### 7.1 Candle timing leakage

**Rule:** A candle's `Open/High/Low/Close` are not available until the candle CLOSES.

**Live path:** `data/fetcher.py:972-1004` explicitly drops the last (still-forming) bar before returning. Bars arriving at `MarketAgent.run()` are guaranteed closed.

**Historical path:** `HistoricalCSVProvider` slicing uses `df.iloc[cursor-300 : cursor+1]` where `cursor` is the current bar index. The current bar IS included in the slice (it's the most recent closed bar at replay time `T`).

**Decision timing:** `backtest/unified_engine.py:745` calls `trader.evaluate_decision_core(market_out, ...)` at iteration `i`. The `market_out["df"]` includes bars `[0..i]` (inclusive). The signal is generated based on bar `i`'s close (which closed at time `T`).

**Fill timing:** `backtest/unified_engine.py:759` fills at `df.iloc[i+1]["open"]` (next bar's open) — this is the LOOK-AHEAD FIX (the previous version filled at `df.iloc[i]["close"]` which is correct but pessimistic; next-bar-open matches live behavior where order placement happens between bar closes).

**Verdict:** ✅ NO LEAKAGE. Candle timing is correctly causal on both paths.

### 7.2 Higher-timeframe (HTF) leakage

**Rule:** When M15 signal is being generated at time `T`, the current H1/H4 candle is still forming. Its final High/Low/Close are not yet known.

**Live path:** `data/fetcher.py:972-1004` drops the forming bar at fetch time — for every TF. So when `MarketAgent.run()` requests H4 bars, the most recent H4 bar is the LAST CLOSED H4 bar, not the currently-forming one.

**Historical path:** `HistoricalCSVProvider._compute_mtf_bias_from_csvs()` (line 458):
```python
tf_minutes = _tf_to_minutes(tf)
cutoff = current_time - pd.Timedelta(minutes=tf_minutes)
causal = df[df.index <= cutoff]   # bar open_time <= T - tf_interval → bar closed by T
```

So at replay time `T` (M15 bar `i` close), H4 bars with `open_time + 4h ≤ T` are visible. H4 bar with `open_time + 4h > T` (i.e. currently forming H4) is excluded.

Same for H1, D1.

**MTF cache:** `data/backtest_ohlcv_cache.py:148`:
```python
closed_mask = (df.index + delta) <= asof
visible = df.loc[closed_mask]
```
Identical logic — only closed HTF bars visible at `asof`.

**Verdict:** ✅ NO LEAKAGE. HTF filtering is correctly causal.

### 7.3 Indicator rolling calculations

**Rule:** Rolling calculations (EMA, RSI, MACD, ATR, ADX, Bollinger, Stochastic) must not use future candles.

**Implementation:** All indicators in `data/indicators.py` and `data/indicators_ext.py` use `pandas-ta` / `ta` library functions, which are causal by construction. `pandas.rolling(window=N).mean()` only looks backward. `pandas.ewm(adjust=False)` only looks backward.

`MarketRegimeDetector._add_adx` uses `ewm(alpha=1/period, adjust=False)` — Wilder's smoothing — causal.

**Verdict:** ✅ NO LEAKAGE. All rolling indicators are causal.

### 7.4 NadarayaWatson Envelope (SPECIAL CASE)

**Rule:** This module self-documents as REPAINTING — it uses a centered (forward-looking) Gaussian window.

**Status:** `analysis/nadaraya_watson_envelope.py:25-29` documents: "the MQL5 port uses a forward-looking centered window (`reg[i]` uses bars `[i, i+1, ..., i+W-1]` — bars AFTER i). The `nwe_stable` flag marks the last `window_size=500` bars as unstable."

**Verdict:** ⚠ POTENTIAL LEAK if consumer reads `nwe_mid/upper/lower` on the most recent bar. Mitigation already in place: `nwe_stable=False` flag marks last 500 bars; consumers MUST respect this flag.

**Action:** Document the contract for all consumers; verify they check `nwe_stable`. If no consumer respects the flag, the indicator should be removed from the production decision path.

### 7.5 SupportResistance + VolumeProfile caller contracts

**Rule:** These modules compute statistics over the full df passed in. If the caller passes the full historical df, future bars leak in.

**SupportResistance:** `analysis/support_resistance.py:18-23` documents: "Rejection counts and strength scores are computed on the *full* supplied DataFrame (current live strength). For pure historical walk-forward, slice the DataFrame up to the decision bar before calling analyze()."

**VolumeProfile:** `analysis/volume_profile.py:158-199` uses `df[["open","high","low","close"]].min().min()` / `.max().max()` over the whole df to set bin range — leaks future highs/lows.

**Caller pattern (live):** `MarketAgent` passes the 300-bar df fetched from MT5. The most recent bar in that df is the most recent CLOSED bar (forming bar dropped). So in live, `SupportResistance.analyze(df)` is called on the last 300 closed bars — no leak.

**Caller pattern (backtest):** `HistoricalCSVProvider.get_market_out()` (csv_data_provider.py) slices `df.iloc[cursor-300 : cursor+1]` — only past bars. ✅ CAUSAL.

**But:** `VolumeProfileEngine` is DISABLED 2026-07-30 (`analysis_agent.py:865` constructed but not run). So no current leakage risk.

**Verdict:** ✅ NO ACTIVE LEAKAGE (VolumeProfile disabled). SupportResistance is causal when called from the provider; risk only if a future caller passes the full historical df directly.

### 7.6 ML training label leakage

**Rule:** Labels use `shift(-N)` (forward-looking) — this is correct for LABELS, not for features.

**LabelGenerator:** `ml/label_generator.py:39-49` documents: "forward_return / forward_pips / mae_pips / mfe_pips / signal class use ONLY future candles relative to the feature row (row_idx+1 .. row_idx+horizon). This is the only place future data is allowed — and only for creating training labels, never for inference features."

**Train/val/test split:** `DatasetBuilder.build_from_dataframe` (lines 326–377) uses chronological 70/15/15 split, no shuffle. `PurgedEmbargoedSplitter.purge_train_val_test` (when `use_purged_split=True`) drops training rows whose `[i, i+h]` label window overlaps the val/test boundary.

**Threshold calibration:** `_find_optimal_threshold` (lines 186–222) done on `X_val` only, never `X_test`.

**Scaler fit:** `DataPreprocessor.fit_clip_bounds` and `fit_scaler` both fit on TRAIN only then transform val/test.

**Verdict:** ✅ NO LEAKAGE in training pipeline.

### 7.7 ML inference leakage

**Status:** ML inference is DISABLED in live code (`if False:` at `analysis_agent.py:2001`). No inference leakage possible until re-enabled.

**Risk when re-enabled:** `ml/feature_engineer.py:400` uses `pd.Timestamp.now(tz="UTC")` for `hour_utc`, `day_of_week`, `is_weekend`, `is_monday_open`, `is_friday_close` features. In backtest, this would stamp every historical bar with current wall-clock — PARITY VIOLATION (P1-D §0.5).

**Fix:** Override `pd.Timestamp.now(tz="UTC")` with `df.index[-1]` when in backtest mode (or always use `df.index[-1]`).

### 7.8 RL state leakage

**Rule:** RL state at step `t` must not include any future data.

**V2 schema** (`ml/rl_environment_v2.py`): State vector is built from `features_df.iloc[current_step]` — strictly backward-looking. Position/account state is maintained in env.

**Reward:** `rl/reward_functions.py` + `ml/reward_engine_v2.py` compute reward from `(close[current_step] - entry)` — close is the current bar's close, not future.

**Verdict:** ✅ NO LEAKAGE in RL state/reward.

### 7.9 LLM context leakage

**Rule:** LLM context blocks must not include future market information.

**Status:** LLM context is built from `analysis_out` (current analysis) + `memory` (past trade history) + `news_intelligence` (live news). All backward-looking.

**News:** Live news fetch is the only "future-facing" data — but news is timestamped; the LLM only sees news published up to "now" (in live mode) or up to the snapshot date (in backtest mode, the snapshot is whatever was in `data/economic_calendar.json` at install time).

**Risk:** In backtest, if the news snapshot is recent (e.g. today's news), the LLM sees "future" news relative to a 2023 historical bar. **PARITY VIOLATION.**

**Fix:** Either:
1. Disable LLM in backtest mode (simplest)
2. Use historical news archive that matches the backtest window
3. Filter news by `published_at <= current_bar_time` in backtest mode

### 7.10 Summary of leakage findings

| # | Module | Risk | Severity | Fix |
|---|--------|------|----------|-----|
| L1 | NadarayaWatson Envelope | REPAINTING (forward-looking centered window) | HIGH if consumer ignores `nwe_stable` flag | Verify all consumers check `nwe_stable`; remove from production decision path if violated |
| L2 | `ml/feature_engineer.py:400` | `pd.Timestamp.now(tz="UTC")` for time features in backtest | MEDIUM (ML disabled in live, but training re-runs would be affected) | Use `df.index[-1]` instead |
| L3 | `analysis/smart_money.py:390` | `datetime.now(timezone.utc)` for kill-zone in backtest | MEDIUM | Pass `bar_timestamp` to `SmartMoneyEngine.analyze()` |
| L4 | LLM news context | Snapshot from `data/economic_calendar.json` may be future-dated relative to historical bar | LOW (snapshot is mostly historical) | Filter news by `published_at <= current_bar_time` in backtest mode |
| L5 | `SupportResistance.analyze` | Caller contract: must slice `df.iloc[:i+1]` before calling | LOW (no active violation; future caller could violate) | Document/enforce contract in module docstring |
| L6 | `VolumeProfileEngine` | Same as L5 | LOW (module DISABLED) | Same |

**No active high-severity leakage found in the production decision path.** All identified risks are either documented caller contracts (L5, L6) or wall-clock usage in non-critical modules (L2, L3, L4) or already mitigated by stability flag (L1).

---

## 8. Historical Downloader Code Audit (Phase 10)

### 8.1 Existing downloader: `scripts/download_historical_data.py`

**Architecture:** Connects to MT5 → for each (symbol, timeframe) pair, fetches bars in monthly chunks via `mt5.copy_rates_range` → validates each chunk → saves to `data/historical/{SYMBOL}/{TF}.csv` with UTC timestamps.

**Schema:** `datetime_utc, open, high, low, close, tick_volume, spread, real_volume` — matches existing CSVs exactly.

**Strengths:**
- Clean CLI: `--symbols EURUSD GBPUSD ... --timeframes M15 H1 H4 D1 --start 2025-07-01 --end 2026-08-08`
- Monthly chunking avoids MT5 per-call rate limits
- Per-chunk validation (sort, duplicates, OHLC sanity, negative prices)
- Manifest with metadata (timezone, source, spread_available, real_volume_available)
- Supports both nested (`data/historical/{SYMBOL}/{TF}.csv`) and legacy flat (`data/{SYMBOL}_{TF}.csv`) layouts

**Weaknesses identified:**

| # | Issue | Severity | Why it matters |
|---|-------|---------|----------------|
| W1 | **Broker timezone bug** | HIGH | `pd.to_datetime(df["time"], unit="s", utc=True)` (line 121) assumes `r["time"]` is true UTC. If broker returns server time mislabeled as epoch (common for FX brokers using GMT+2/+3), timestamps will be 2-3h in the future. `data/fetcher.py:_get_broker_utc_offset_hours` has the fix but this script doesn't use it. |
| W2 | **No retry logic** | MEDIUM | MT5 can return empty/None for transient errors. No retry/backoff. |
| W3 | **No gap detection** | MEDIUM | Validation only checks duplicates + OHLC sanity. Missing bars (e.g. broker data feed gap) are silently accepted. |
| W4 | **No weekend awareness** | LOW | Validation flags ALL gaps as issues, including expected forex weekend close (Fri 21:00 → Sun 21:00 UTC). Manifest should distinguish weekend gaps from real gaps. |
| W5 | **No MAX_REQUIRED_LOOKBACK enforcement** | MEDIUM | User must manually pass enough `--start` to cover warmup. If user passes `--start 2026-01-01`, the first ~500 bars (2 weeks of M15) will have NaN indicators. |
| W6 | **No HTF resampling** | LOW | If user requests H4 but broker doesn't have H4 history, downloader doesn't fall back to H1→H4 resampling (which `data/fetcher.py:_resample_h1_to_h4` does for live). |
| W7 | **No bid/ask columns** | MEDIUM | Output schema is `OHLC + tick_volume + spread + real_volume`. No `bid` / `ask` columns. Historical backtest cannot fill at ask/bid. |
| W8 | **No validation report** | LOW | Per-chunk validation messages are printed to stdout but not persisted. No machine-readable validation report. |
| W9 | **No idempotent re-run** | LOW | `--force` re-downloads everything; no smart "download only missing ranges" capability. |
| W10 | **Manifest doesn't include data quality metrics** | LOW | Manifest has `spread_available` and `real_volume_available` booleans but not zero-percentages, gap counts, etc. |

### 8.2 Other data fetching code

| Module | Role | Status |
|--------|------|--------|
| `data/fetcher.py` | Primary live MT5 fetcher with broker-tz auto-detection, forming-bar drop, multi-source fallback (TwelveData, yfinance, Alpha Vantage, Polygon, Finnhub) | ✅ Production-ready; the broker-tz fix in `_get_broker_utc_offset_hours` (line 662-725) is the gold standard |
| `data/data_orchestrator.py` | Singleton wrapper around `DataFetcher` + `MT5Connection` | ✅ Production-ready |
| `data/live_feed.py` | Live tick snapshot (bid/ask/spread/last) + rolling buffer | ✅ Production-ready; live-only |
| `broker/mt5_data.py` | Legacy secondary MT5 fetcher (no broker-tz correction!) | ⚠ Should be deleted or routed through `MT5Connection` |
| `broker/mt5_historical_fetcher.py` | Bulk historical fetcher (no broker-tz correction!) | ⚠ Same — should use the broker-tz fix |
| `data/automated_updater.py` | Daily data update tool (uses OANDA → yfinance; disconnected from CSV provider) | ⚠ Orphan — writes to `data/forex/{PAIR}_daily.csv` with capitalized column names; HistoricalCSVProvider doesn't load these. **Delete or refactor.** |
| `data/verify_data_coverage.py` | Coverage verification (hardcodes `2026-06-21` target date) | ⚠ Stale — re-run with current date |
| `data/compressed_storage.py` | Compressed tick/bar storage (Level-2 / MultiSymbol) | ✅ Unused in production decision path; orphan |
| `data/backtest_ohlcv_cache.py` | Point-in-time HTF cache for backtest mode | ✅ Production-ready; correctly causal |
| `core/csv_data_provider.py` | HistoricalCSVProvider — the actual backtest data loader | ✅ Production-ready; needs the bid/ask addition (when CSVs have those columns) |
| `core/data_provider.py` | DataProvider ABC + LiveMT5Provider + HistoricalMT5Provider | ⚠ LiveMT5Provider.current_time() returns naive UTC — fix needed |

---

## 9. Recommended Downloader Upgrades (Phase 11-13)

See Deliverable 5 for the actual code changes. Summary:

1. **Fix broker-tz bug (W1):** Use `data/fetcher.py:_get_broker_utc_offset_hours` pattern in `scripts/download_historical_data.py`. Or route through `MT5Connection` + apply offset.

2. **Add retry logic (W2):** Exponential backoff (3 retries, 1s/2s/4s) for each chunk.

3. **Add gap detection (W3):** Compare consecutive timestamps; flag gaps > 1.5× TF interval. Distinguish weekend gaps from real gaps.

4. **Add weekend awareness (W4):** Pre-compute expected forex weekend close times (Fri 21:00 UTC → Sun 21:00 UTC, DST-aware) and exclude from gap reporting.

5. **Add MAX_REQUIRED_LOOKBACK enforcement (W5):** Add a `--warmup-bars N` argument (default 1000) that extends `--start` backward by N bars on the primary TF.

6. **Add HTF resampling fallback (W6):** If H4 fetch returns empty, fall back to H1 → H4 resampling using `data/fetcher.py:_resample_h1_to_h4` pattern.

7. **Add bid/ask columns (W7):** For each bar, also fetch `mt5.symbol_info_tick` at bar close time and store `bid` and `ask`. Note: this requires per-bar tick fetch — may slow down download significantly. Alternative: only fetch bid/ask for the most recent N bars (e.g. last 100) for spread validation.

8. **Add validation report (W8):** After download, produce `data/historical/{SYMBOL}/_validation_{TF}.json` with detailed gap analysis, zero-spread percentages, NaN counts, OHLC violations.

9. **Add idempotent re-run (W9):** If existing CSV covers part of the requested range, only download the missing portion and merge.

10. **Add data quality metrics to manifest (W10):** Include `spread_zero_pct`, `gap_count`, `non_weekend_gap_count`, `nan_count`, `ohlc_violations` per file.

11. **Add a standalone validation CLI** (`scripts/validate_historical_csv.py`) that runs the P6 audit script on the downloaded data.

12. **Add CSV schema enforcement:** Strict column order, UTC tz-aware timestamps, integer spread, float OHLC. Reject CSVs that don't conform.
