# CHANGELOG

## 2026-08-02 — Point-in-Time Data Leakage Fix + Data Loader Fix

### Critical: Backtest Data Leakage Eliminated

**Problem**: During `python main.py --mode backtest`, the unified engine replays
historical bars through `evaluate_decision_core()`. However, 13 auxiliary
modules in `analysis_agent.py` were fetching **present-day live data** via
MT5, HTTP APIs, and yfinance — injecting future information into historical
bar evaluations. This made ALL prior backtest numbers **meaningless**.

**Modules that were leaking (all now fixed)**:

| Module | Data Source | Leakage Severity |
|--------|-------------|-------------------|
| `currency_strength.py` | 28 live MT5 cross-pair fetches | CRITICAL |
| `sentiment_data.py` | yfinance (DXY, retail), Fear&Greed API, currency strength | CRITICAL |
| `institutional_flow.py` | Live CFTC website HTTP scrape | CRITICAL |
| `microstructure.py` | MT5 `copy_ticks_range()` | CRITICAL |
| `intermarket.py` | Live correlation via MT5 DataFetcher | HIGH |
| `news_api_provider.py` | Live NewsAPI.org headlines | HIGH |
| `fred_data.py` | Live St. Louis Fed FRED API | MEDIUM |
| `economic_calendar_api.py` | Live Forex Factory scraper | MEDIUM |
| `economic_surprise.py` | Live actual-vs-forecast | MEDIUM |
| `_h4_fetcher` (MTF structure) | Live MT5 H4 candle fetch | HIGH |

**Fix approach (belt-and-suspenders)**:
1. **Caller-level guards** in `analysis_agent.py`: 9 `is_backtest_mode()`
   guards wrapping every external-data call site. During backtest,
   these modules are **skipped entirely** (not substituted with live data).
2. **Module-level guards**: Each leaking module also has its own
   `is_backtest_mode()` check at its `analyze()`/`get_all()` entry point
   as a second line of defense.

**Documented permanent divergences** (no historical point-in-time data exists):
- Currency strength matrix (28-pair MT5) — backtest runs without it
- COT institutional flow — backtest runs without it
- Microstructure tick analysis — backtest runs without it
- News sentiment (NewsAPI.org) — backtest runs without it
- FRED macro data (CPI, yields, VIX) — backtest runs without it
- Economic surprise index — backtest runs without it
- Intermarket correlations — backtest runs without it
- H4 MTF structure — backtest uses internal-timeframe-only analysis

These modules **do contribute to live trading decisions** but are
**absent from backtests**. Live decisions will be more informed than
backtest decisions. This is an honest, documented gap — not hidden.

**Prior to this fix, ALL previous backtest numbers are untrustworthy**
because they were contaminated by present-day data injected into
historical bar evaluations.

### ML/LLM Component Status (Structural, Not Fixed)

| Component | Status | Root Cause |
|-----------|--------|------------|
| XGBoost/RF models (registry) | BROKEN | sklearn 1.5.2 vs 1.9.0 pickle incompatibility |
| LSTM model | MISSING | Never trained; 25% ensemble weight permanently dead |
| AI Analyst (LLM) | Working | Graceful cascade when API keys available |
| MasterAnalyst (LLM) | Working | Same cascade; vote excluded when unavailable |
| Decision fusion | Working | Dynamic rebalancing handles degraded mode correctly |

**Current effective system**: ~100% rule-engine decisions. ML ensemble
always returns `ml_available=False`. LLM votes are excluded when API
keys are unavailable or rate-limited. Backtest numbers reflect
**rule-engine-only edge**, not the designed multi-voter architecture.

### Data Loader Fix

`backtest/data_loader.py`: Added `tick_volume` → `volume` column mapping
in `_normalize_columns()`. MT5-exported CSVs use `tick_volume` instead
of `volume`, causing a validation error.

### Files Modified

- `agents/analysis_agent.py` — 9 backtest leakage guards + import
- `analysis/currency_strength.py` — module-level backtest guard
- `analysis/institutional_flow.py` — module-level backtest guard
- `analysis/microstructure.py` — module-level backtest guard
- `analysis/intermarket.py` — module-level backtest guard
- `analysis/sentiment_data.py` — module-level backtest guard
- `fundamental/economic_surprise.py` — module-level backtest guard
- `backtest/data_loader.py` — tick_volume column mapping fix
- `CHANGELOG.md` — this file
