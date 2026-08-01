# Confidence → Win-Rate Lookup Report

**Generated:** 2026-07-25  
**Source backtest:** 2026-07-04 14:32:46  
**Pairs tested:** 3 · **Timeframes:** M15, H1, H4 · **Combinations:** 9 · **Successful runs:** 9  
**Suggested live decision mode:** `confluence`

> **Purpose of this file:**  
> When the live decision system tags a signal with a confidence level (`High` / `Medium` / `Low`), this report tells you what win rate that confidence level has historically delivered. Use it as a *risk gate*: if the historical win rate for a (strategy, confidence) pair is too low or the sample is too small, skip or downsize the trade — even if the live signal looks good.

---

## 1. Executive Summary

| Tier | Count | Action |
|------|-------|--------|
| ✅ Use (WR ≥ 50%, n ≥ 10) | 1 | `ict_amd` |
| ⚠️ Use with caution (40% ≤ WR < 50%) | 0 | — |
| ❌ Disable or fix (WR < 40% or n = 0) | 7 | `pin_bar`, `candlestick_patterns`, `sd_zones_scored`, `sr_zones`, `stop_hunt`, `multi_pa`, `cci_state` |

**Headline numbers**
- Only **1 out of 5 strategies** with trades is safe to use: `ict_amd` (100% WR, n=11).
- **4 out of 5 strategies** with trades have WR < 40% — they are bleeding money.
- **3 strategies** generated zero trades — they need parameter tuning before live use.
- `stop_hunt` has an interesting split: **High-confidence = 52.3% WR (use), Medium/Low = 10%/0% (skip)**. This is the single most actionable insight — confidence level *should* gate stop_hunt entries.

---

## 2. Global Thresholds (used by the lookup module)

| Threshold | Value | Meaning |
|-----------|-------|---------|
| `min_trades_for_reliability` | 10 | Below this → "low_n", even good WR is treated as caution |
| `min_trades_for_high_reliability` | 30 | Above this → "high_n", WR is statistically meaningful |
| `winrate_use_threshold` | 50% | WR ≥ this AND n ≥ 10 → "trust", full or near-full size |
| `winrate_caution_threshold` | 40% | 40% ≤ WR < 50% → "caution", reduced size (40%) |
| — | < 40% | → "skip", position_scale = 0.0 |

---

## 3. Strategy Performance Summary

| Strategy | Trades | Win Rate | Avg R | Best Confidence | Best Tactic | Tier |
|----------|-------:|---------:|------:|-----------------|-------------|------|
| `ict_amd` | 11 | **100.0%** | +49.85 | Medium | `ict_amd_default` | ✅ Use |
| `candlestick_patterns` | 3,301 | 37.8% | +0.09 | Low | `Three Black Crows` | ❌ Disable |
| `sr_zones` | 16,598 | 36.8% | +0.06 | Medium | `support` | ❌ Disable |
| `cci_state` | 2,395 | 35.4% | +0.04 | High | `cci_227` | ❌ Disable |
| `stop_hunt` | 71 | 35.2% | +0.52 | High | `stop_hunt_default` | ❌ Disable (but see confidence split) |

---

## 4. Per-Confidence-Level Win Rates — *the heart of this report*

This is the table the live decision system queries. **Position sizing should be scaled by `position_scale`**.

| Strategy | Confidence | Trades | Win Rate | Sample Reliability | Action | Position Scale | Notes |
|----------|-----------|-------:|---------:|---------------------|--------|---------------:|-------|
| `ict_amd` | High | 3 | 100.0% | low_n | **trust** | 1.0 | Promising but tiny sample — cap risk |
| `ict_amd` | Medium | 8 | 100.0% | low_n | **trust** | 1.0 | Promising but tiny sample — cap risk |
| `ict_amd` | Low | 0 | — | no_data | **skip** | 0.0 | No historical data |
| `candlestick_patterns` | any | 3,301 | 37.8% | high_n | **skip** | 0.0 | Below 40% threshold |
| `sr_zones` | any | 16,598 | 36.8% | high_n | **skip** | 0.0 | Below 40% threshold |
| `cci_state` | any | 2,395 | 35.4% | high_n | **skip** | 0.0 | Below 40% threshold |
| `stop_hunt` | High | 44 | 52.3% | high_n | **trust** | 0.7 | Decent WR + n; reduce size for volatility |
| `stop_hunt` | Medium | 20 | 10.0% | med_n | **skip** | 0.0 | 10% WR — losing money |
| `stop_hunt` | Low | 7 | 0.0% | low_n | **skip** | 0.0 | 0% WR |

### Key insight — confidence *does* predict outcomes for `stop_hunt`

`stop_hunt` overall looks bad (35.2% WR), but the confidence split tells a different story:

- **High-confidence stop_hunts**: 52.3% WR (n=44) → safe to trade at 70% size.
- **Medium-confidence stop_hunts**: 10.0% WR (n=20) → **hard skip**.
- **Low-confidence stop_hunts**: 0.0% WR (n=7) → **hard skip**.

If the live decision system tags a stop_hunt signal with `High` confidence, trade it. If it tags it `Medium` or `Low`, the historical data says **do not enter**. Without this confidence-level lookup, you would have disabled stop_hunt entirely and missed a 52.3%-WR edge.

---

## 5. Per-Tactic Win Rates (filtered to actionable rows)

### 5.1 `ict_amd` — only one tactic, 100% WR

| Tactic | Trades | Win Rate |
|--------|-------:|---------:|
| `ict_amd_default` | 11 | 100.0% |

### 5.2 `candlestick_patterns` — sorted by WR (top 10 only)

| Tactic | Trades | Win Rate | Action |
|--------|-------:|---------:|--------|
| `Three Black Crows` | 5 | 60.0% | ⚠️ n < 10, can't trust |
| `Dark Cloud Cover` | 119 | 48.7% | ⚠️ borderline |
| `Three Inside Down` | 63 | 47.6% | ⚠️ borderline |
| `Doji` | 290 | 45.9% | ⚠️ borderline |
| `Shooting Star` | 192 | 45.8% | ⚠️ borderline |
| `Hanging Man` | 124 | 45.2% | ⚠️ borderline |
| `Evening Star` | 140 | 43.6% | ⚠️ borderline |
| `Bearish Harami` | 197 | 43.1% | ⚠️ borderline |
| `Tweezer Top` | 367 | 40.6% | ⚠️ borderline |
| `Bearish Engulfing` | 285 | 40.0% | ⚠️ borderline |

> **Insight:** Most candlestick patterns hover at 40–49% WR. None reach 50% with n ≥ 10. As a category, candlestick_patterns should remain disabled. `Three Black Crows` looks promising (60%) but n=5 is too small to trust.

### 5.3 `sr_zones` — both tactics below 40%

| Tactic | Trades | Win Rate |
|--------|-------:|---------:|
| `support` | 6,728 | 37.8% |
| `resistance` | 9,870 | 36.0% |

### 5.4 `stop_hunt` — single tactic, see confidence split above

| Tactic | Trades | Win Rate |
|--------|-------:|---------:|
| `stop_hunt_default` | 71 | 35.2% |

### 5.5 `cci_state` — top 10 tactics by WR (filtered to n ≥ 10)

| Tactic | Trades | Win Rate | Action |
|--------|-------:|---------:|--------|
| `cci_227` | 6 | 83.3% | ⚠️ n < 10, can't trust |
| `cci_111` | 20 | 70.0% | ✅ promising |
| `cci_157` | 10 | 60.0% | ✅ promising |
| `cci_-107` | 15 | 60.0% | ✅ promising |
| `cci_-148` | 10 | 60.0% | ✅ promising |
| `cci_164` | 13 | 53.8% | ✅ promising |
| `cci_170` | 11 | 54.5% | ✅ promising |
| `cci_-127` | 11 | 54.5% | ✅ promising |
| `cci_-131` | 7 | 57.1% | ⚠️ n < 10 |
| `cci_-153` | 7 | 57.1% | ⚠️ n < 10 |

> **Insight:** Although the overall `cci_state` WR is 35.4%, **a handful of specific CCI levels are above 50% with n ≥ 10**. If your live system can route to only those levels (e.g., `cci_111`, `cci_157`, `cci_-107`, `cci_-148`, `cci_164`, `cci_170`, `cci_-127`), you would flip cci_state from "disable" to "use". The lookup module exposes this via `lw.tactic_winrate("cci_state", "cci_111")`.

---

## 6. Strategies With Zero Trades (parameter tuning needed)

| Strategy | Reason |
|----------|--------|
| `pin_bar` | No trades generated — check params |
| `sd_zones_scored` | No trades generated — check params |
| `multi_pa` | No trades generated — check params |

> These strategies registered but never fired. Likely causes: thresholds too strict, timeframes mismatched, or filters too aggressive. Re-tune parameters and re-run the backtest.

---

## 7. How to use this in the live decision system

### 7.1 Python import

```python
from intelligence.confidence_winrate_lookup import get_lookup

lw = get_lookup()  # singleton — loads JSON once

# When decision system produces a signal:
rec = lw.recommend(strategy="stop_hunt", confidence="High", tactic="stop_hunt_default")

if rec.action == "skip":
    return None              # do NOT enter this trade

# Scale position size by historical confidence
position_size *= rec.position_scale    # 0.0–1.0

# Optional: log the reasoning
log.info(f"Trade approved: {rec.strategy} [{rec.confidence}] "
         f"expected WR={rec.expected_winrate:.1%} (n={rec.n_trades}) "
         f"action={rec.action} scale={rec.position_scale} "
         f"reliability={rec.sample_reliability}")
```

### 7.2 CLI (quick check before entering a trade manually)

```bash
python -m intelligence.confidence_winrate_lookup --strategy ict_amd --confidence High
python -m intelligence.confidence_winrate_lookup --strategy stop_hunt --confidence Medium
python -m intelligence.confidence_winrate_lookup --tactic-lookup cci_state cci_111
python -m intelligence.confidence_winrate_lookup --all
python -m intelligence.confidence_winrate_lookup --top-tactics cci_state
```

### 7.3 Wire into `core/orphan_consumers.py` (recommended integration point)

In `apply_signal_scoring()`, after the signal scorer has computed a confidence level, call the lookup and:

1. **If `action == "skip"`** → veto the signal (or demote confidence to None).
2. **If `action == "caution"`** → reduce position_size by `position_scale`.
3. **If `action == "trust"`** → keep signal; optionally still apply `position_scale` (e.g., 0.7 for stop_hunt High to reduce volatility exposure).

This way the historical win-rate gate becomes part of the decision pipeline, not just a report.

---

## 8. Decision Matrix — quick reference

| Confidence tag from live system | `ict_amd` | `stop_hunt` | `candlestick_patterns` | `sr_zones` | `cci_state` |
|---------------------------------|-----------|-------------|------------------------|------------|-------------|
| **High** | ✅ trust (1.0×) | ✅ trust (0.7×) | ❌ skip | ❌ skip | ❌ skip |
| **Medium** | ✅ trust (1.0×) | ❌ skip (10% WR) | ❌ skip | ❌ skip | ❌ skip |
| **Low** | ❌ skip (no data) | ❌ skip (0% WR) | ❌ skip | ❌ skip | ❌ skip |
| **any** (fallback) | — | — | ❌ skip (37.8%) | ❌ skip (36.8%) | ❌ skip (35.4%) |

> The "fallback" row applies when the live system tags a confidence the lookup has no specific rule for — it falls back to the strategy's overall WR. This means candlestick_patterns / sr_zones / cci_state are **always skipped** regardless of confidence tag, because their best-case WR is below 40%.

---

## 9. Files Generated

| File | Purpose |
|------|---------|
| `intelligence/confidence_winrate_data.json` | Structured data — the source of truth |
| `intelligence/confidence_winrate_lookup.py` | Python module + CLI for live system queries |
| `intelligence/confidence_winrate_report.md` | This document |

---

## 10. Next Steps

1. **Wire the lookup into `core/orphan_consumers.py`** — specifically into `apply_signal_scoring()`, so every signal is gated by historical win rate before reaching the risk engine.
2. **Re-run the backtest monthly** with fresh data, regenerate `confidence_winrate_data.json`, and call `reload_lookup()` to refresh the singleton.
3. **Tune `pin_bar`, `sd_zones_scored`, `multi_pa`** parameters so they actually generate trades in the next backtest.
4. **Investigate the cci_state anomalies** — the 7 specific CCI levels with WR ≥ 50% (cci_111, cci_157, cci_-107, cci_-148, cci_164, cci_170, cci_-127) deserve a closer look. If they're stable across timeframes, consider promoting them to a "filtered cci_state" strategy.
5. **Re-evaluate `ict_amd` with more data** — 11 trades is too small to be confident in 100% WR. Run a longer backtest (more pairs, longer window) to see if the edge holds.
