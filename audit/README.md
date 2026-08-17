# Forex-Agent Audit — Master Index

**Project:** Forex AI Autonomous Trading System
**Audit date:** 2026-08-17
**Auditor:** Multi-agent forensic audit (4 parallel Explore sub-agents + main agent)

This is the master index for the 17-phase audit + implementation cycle.

---

## 1. Final Deliverables (Required by User)

| # | Deliverable | File | Status |
|---|------------|------|--------|
| 1 | System Understanding Report | [D1-system-understanding.md](D1-system-understanding.md) | ✅ Complete |
| 2 | Data Dependency Map | [D2-data-dependency-map.md](D2-data-dependency-map.md) | ✅ Complete |
| 3 | Missing Data Report | [D3-missing-data-report.md](D3-missing-data-report.md) | ✅ Complete |
| 4 | Historical Data Specification | [D4-historical-data-specification.md](D4-historical-data-specification.md) | ✅ Complete |
| 5 | Updated Downloader | [D5-updated-downloader.md](D5-updated-downloader.md) | ✅ Complete |
| 6 | Validation + End-to-End Test Report | [D6-validation-test-report.md](D6-validation-test-report.md) | ✅ Complete |

## 2. Final Data Requirements Table

See [D4-historical-data-specification.md §5](D4-historical-data-specification.md#5-final-data-requirements-table) for the full FINAL DATA REQUIREMENTS TABLE.

Summary:

| Data | Required? | Status | Action |
|------|-----------|--------|--------|
| `datetime_utc` (tz-aware UTC) | ✅ REQUIRED | ✅ Present | None |
| `open`, `high`, `low`, `close` | ✅ REQUIRED | ✅ Present | None |
| `tick_volume` | ✅ REQUIRED | ✅ Present | None |
| `spread` (per-bar points) | ✅ REQUIRED | ⚠ 15-70% zeros | Re-download with v2 downloader |
| `bid`, `ask` | ⚠ RECOMMENDED | ❌ Missing | Add via `--with-bid-ask` flag |
| `real_volume` | ❌ OPTIONAL | ⚠ Always 0 | Document (FX has no real volume) |
| M5, D1 timeframes | ⚠ RECOMMENDED | ❌ Missing | Download via `--timeframes M5 D1` |
| XAUUSD CSVs | ⚠ RECOMMENDED (if traded) | ❌ Missing | Download if traded |
| External macro (DXY/Gold/Oil/US10Y/SP500/VIX) | ⚠ RECOMMENDED | ❌ Missing | `python scripts/download_external_data.py` |
| 21 cross pairs | ⚠ RECOMMENDED | ❌ Missing | Download if CorrelationEngine used in backtest |
| CFTC COT history | ❌ OPTIONAL | ❌ Missing | Synthetic proxy used in backtest |
| News history | ⚠ RECOMMENDED | ❌ Stale snapshot | Download Forex Factory archive |
| Live tick stream | ❌ OPTIONAL | ❌ Live-only | MicrostructureEngine gated OFF in backtest |
| Market depth (L2) | ❌ OPTIONAL | ❌ Live-only | Not used in pipeline |

---

## 3. Evidence Reports (Phase 1 Audit)

| Phase | File | Lines | Scope |
|-------|------|------:|-------|
| P1-A | [evidence/P1-A-data-provider-audit.md](evidence/P1-A-data-provider-audit.md) | 1,374 | Live MT5 + Historical CSV data ingestion |
| P1-B | [evidence/P1-B-decision-execution-audit.md](evidence/P1-B-decision-execution-audit.md) | 910 | Decision core + execution + risk chain |
| P1-C | [evidence/P1-C-analysis-indicators-audit.md](evidence/P1-C-analysis-indicators-audit.md) | 310 | 79 indicators cataloged with lookbacks |
| P1-D | [evidence/P1-D-ml-rl-llm-audit.md](evidence/P1-D-ml-rl-llm-audit.md) | 655 | ML/RL/LLM data dependencies |
| P6 | [evidence/P6-csv-audit.md](evidence/P6-csv-audit.md) | ~700 | 21 existing CSVs audited |
| P6 (JSON) | [evidence/P6-csv-audit.json](evidence/P6-csv-audit.json) | - | Machine-readable audit |
| Validation | [evidence/P-validation-cli-report.md](evidence/P-validation-cli-report.md) | - | CLI validation report |

---

## 4. Phase Coverage Matrix

| Phase | Description | Status | Deliverable |
|-------|-------------|--------|-------------|
| 1 | Project audit (4 parallel agents) | ✅ | P1-A, P1-B, P1-C, P1-D evidence reports |
| 2 | Runtime data dependency map | ✅ | D2 §1 (Master Data Dependency Table) |
| 3 | Feature source trace | ✅ | D2 §5 (17 indicator traces) |
| 4 | Multi-timeframe dependency audit | ✅ | D2 §3 (TF × Module matrix) |
| 5 | MAX_REQUIRED_LOOKBACK derivation | ✅ | D2 §4 |
| 6 | Existing CSV audit | ✅ | P6-csv-audit.md |
| 7 | Missing data identify | ✅ | D3 §2 |
| 8 | Live-vs-Historical parity audit | ✅ | D3 §6 |
| 9 | Leakage audit | ✅ | D3 §7 |
| 10 | Existing downloader code audit | ✅ | D3 §8 |
| 11-13 | Design + implement production-grade downloader + validation | ✅ | D5 (Changes 1-3) |
| 14 | Live-Compatible Historical Dataset (HistoricalCSVProvider) | ✅ | D5 (Changes 4-6) + D6 (Tests 6-8) |
| 15 | Exact Missing Data Report | ✅ | D3 §1-5 + D4 §5 (FINAL DATA REQUIREMENTS TABLE) |
| 16 | Code Changes with explanations | ✅ | D5 §2 (per-change details) |
| 17 | Test updated downloader | ✅ | D6 (8 tests, 5 passed, 3 skipped - require MT5) |

---

## 5. Code Changes Summary

| # | File | Type | Severity | Lines changed |
|---|------|------|---------|--------------:|
| 1 | `scripts/download_historical_data.py` | REWRITE | HIGH | ~440 (full file) |
| 2 | `scripts/validate_historical_csv.py` | NEW | MEDIUM | ~100 |
| 3 | `scripts/download_external_data.py` | NEW | MEDIUM | ~110 |
| 4 | `core/data_provider.py` | EDIT | HIGH (parity) | 5 lines (line 76-81) |
| 5 | `analysis/smart_money.py` | EDIT | MEDIUM (leakage) | 18 lines (line 22, 104-106, 225-227, 385-401) |
| 6 | `ml/feature_engineer.py` | EDIT | MEDIUM (leakage) | 22 lines (line 110, 383, 396-416) |

See [D5-updated-downloader.md](D5-updated-downloader.md) for per-change details (File / Function / Current Problem / Why It Matters / Change / Effect).

---

## 6. Key Findings (TL;DR)

1. **Backtest parity is structurally sound** — both live and backtest run the same `AITrader.evaluate_decision_core()` with the same provider output shape. No "backtest-only" decision code path exists.

2. **21 sequential gates** between "analysis says BUY" and `mt5.order_send` — far more defensive than typical trading systems.

3. **ML is currently disabled** in live code (`if False:` at `analysis_agent.py:2001`). Historical backtests reproduce this configuration as long as the CSV provider feeds the same upstream contexts.

4. **6 parity gaps closed** via this audit:
   - W1-W10 downloader weaknesses (full v2 rewrite)
   - P1-A R4: `LiveMT5Provider.current_time()` naive → tz-aware UTC
   - P1-C §6b.2: `SmartMoneyEngine._current_kill_zone` wall-clock → bar timestamp
   - P1-D §0.5: `FeatureEngineer._context_features` wall-clock → `df.index[-1]`

5. **No future leakage** found in the production decision path. All rolling indicators are causal; HTF bars are correctly filtered to closed-only at replay time; ML label generation uses `shift(-N)` (correct — labels peek forward, features don't).

6. **NadarayaWatson envelope** self-documents as REPAINTING — consumers must respect `nwe_stable=False` flag for the most recent 500 bars.

7. **Existing CSVs have data quality issues**:
   - 15-70% of bars have `spread=0` (depending on symbol/TF)
   - 2-7 non-weekend gaps per file
   - `real_volume` always 0 (expected for FX)
   - Missing M5 and D1 timeframes
   - Missing XAUUSD

8. **Operators should re-download** with v2 downloader to fix spread gaps and apply broker-tz correction.

---

## 7. References

- Forensic README (uploaded by user): `/home/z/my-project/upload/Forex_AI_Forensic_README.md`
- Audit scripts: `/home/z/my-project/scripts/audit/`
- Worklog: `/home/z/my-project/worklog.md`
- Updated downloader code: `/home/z/my-project/download/forex-agent/scripts/download_historical_data.py`
- Validation CLI: `/home/z/my-project/download/forex-agent/scripts/validate_historical_csv.py`
- External data downloader: `/home/z/my-project/download/forex-agent/scripts/download_external_data.py`
- E2E test script: `/home/z/my-project/scripts/audit/p17_e2e_provider_test.py`
