% Trade Permission System — Evidence-to-Fix Audit Report
% Repository: forex-agent (abdullahalhossain-bd)
% 2026-08-11

# 1. Evidence Reviewed

| File | Rows/Content |
|---|---|
| `blocked_trade_analysis.csv` | 1,022 blocked decision cycles, EURUSD, 2026-08-10 09:55–13:31 UTC |
| `filter_summary.csv` | Per-filter blocked counts (11 filters) |
| `filter_combo_summary.csv` | Filter co-occurrence / overlap counts |
| `penalty_attribution.csv` | Entry-quality rule co-occurrence + average penalty |
| `confidence_counterfactual_summary.json` | Min-confidence failure classification (218 total) |
| `scenario_comparison.csv` | 8 executed backtest runs, 19 trades total, WR 10.5%, PF 0.18 |
| `forensic_analysis_report.md` | Prior forensic pass (Bengali/English), documents that no OHLC price series was available for hypothetical win/loss reconstruction, so most outcome columns are `OUTCOME_NOT_AVAILABLE` |

All original files are preserved unmodified in `audit_fixes/before/`.

**Important scope caveat carried over from the prior report:** without historical OHLC data for every blocked-trade timestamp, no hypothetical WIN/LOSS outcome could be reconstructed for blocked trades. That limitation still applies here — none of the findings below claim to know whether a *specific* blocked trade would have won or lost. Findings are instead based on (a) direct code inspection of the modules that produced the blocks, and (b) internal consistency of the audit data itself.

---

# 2. Evidence → Module → Fix Mapping

| # | Finding | Affected module | Exact code path | Evidence | Severity | Fix |
|---|---|---|---|---|---|---|
| 1 | A structurally NEUTRAL factor's weight was still added to the confidence-normalization denominator, mechanically deflating confidence/aligned% on every single decision | `intelligence/decision_score.py` → `DecisionScorer.score()` | `decision_score.py:118-145` (pre-fix) | `confidence_counterfactual_summary.json`: 218 Min-confidence failures, 158 (72.5%) "naturally low" pre-penalty. Code: `intelligence/confluence_engine.py:336` hardcodes `direction="NEUTRAL"` for the session factor ("Session is direction-neutral — it boosts the OTHER factors' weight") yet the old normalization added its weight to the denominator regardless, contradicting its own documented intent. Same effect hits `intermarket`/`news` whenever their upstream data is missing. | **CONFIRMED BUG** | Only add a factor's weight to `total_weight_used` when `f.direction in ("BUY","SELL")`. **Applied.** |
| 2 | "Confluence quality" gate never fails independently of "Session quality" (100% subset: 136/136) | `risk/trade_permission.py` (gate) + `intelligence/confluence_engine.py` (`_session_factor`) | `trade_permission.py:1161-1284`, `confluence_engine.py:323-360` | Verified directly against `blocked_trade_analysis.csv`: `Confluence-fail ⊆ Session-fail` = 136/136 = 100%. Root cause: the session grade feeds the confluence score (as a permanently-NEUTRAL factor, see #1) **and** is checked again as a separate hard gate — the same signal penalizes the same decision twice. | **STRONG EVIDENCE** (root cause confirmed; whether to remove the *second* gate needs outcome data) | Root numerical cause fixed by #1. The dual-gating itself is treated as a **deliberate defense-in-depth design** (not clearly a bug) — **not removed**, per "do not remove risk controls" and lack of outcome (WR/PF) data to justify removal. Flagged for review with real OHLC data. |
| 3 | `sl_swing_anchor` + `tp_structure_validation` compound penalty (+10) contributes to 60/218 (27.5%) "penalty-created" Min-confidence blocks | `risk/entry_quality_guardrails.py` | `entry_quality_guardrails.py:1867-1892` | `penalty_attribution.csv`: `sl_swing_anchor` co-occurs in 488 cycles (avg total penalty 20.4), `tp_structure_validation` in 516 (avg 19.9); `confidence_counterfactual_summary.json`: `penalty_created_block: 60`. Code comment shows this exact compound rule was added deliberately after a **real-money loss postmortem** (Day 137, GBPCAD 2026-07-20). | **STRONG EVIDENCE, BY DESIGN** — not a bug | **Not changed.** Instructions explicitly forbid removing risk controls or tuning to this dataset; this rule has documented loss-prevention justification and there's no outcome data here to show it's net-harmful. Recommend keeping under observation with outcome-tagged data (see §12). |
| 4 | "S/R zone alignment" is the single largest blocker (397/1,022 = 38.8%) — prior report speculated some of this might be a missing-data artifact | `risk/trade_permission.py` | `trade_permission.py:276-350` | Code inspection: `sr_ok` defaults to `True` ("no S/R zone data — not evaluated") and is **only** set to `False` inside the branch that requires **both** `dist_sup` and `dist_res` to be present (`trade_permission.py:303`). The fail-open path never appends a failure. | **INSUFFICIENT EVIDENCE for a bug** (prior report's suspicion refuted) | No change. All 397 blocks are backed by real computed misalignment data, not missing-data artifacts. Threshold (3.0 pips / 1.15 ratio margin) was already re-tuned in a prior 2026-08-02 audit per the in-code comment. |
| 5 | "Risk approved" (329) and "Valid signal" (291) blocks almost perfectly co-occur (291/291) | `risk/trade_permission.py`, `risk/risk_engine.py` | n/a | `filter_combo_summary.csv`: `Risk approved + Valid signal = 291` | **EXPECTED, NOT A BUG** | No change — a signal that never validated has nothing for the risk engine to approve; this is architecturally necessary sequencing, not redundant filtering. |
| 6 | Session/Confluence/Min-confidence blocks only occur in the 09:00–11:00 UTC window of the sampled log, then disappear entirely | Upstream signal generation (not a specific gate) | n/a | `blocked_trade_analysis.csv` hour-by-hour breakdown (independently reproduced) | **POSSIBLE ISSUE, upstream** | No code change — this is a signal-generation/regime question, not a filter-tuning question; flagged for further investigation with a longer log window. |
| 7 | Zone cooldown / Signal persistence / execution filters (news, confluence_avoid) — low block counts (34, 34, 8, 8) | Various | n/a | `filter_summary.csv` | **INSUFFICIENT EVIDENCE** (too low volume to assess) | No change. |

---

# 3. Confirmed Bug — Full Detail

## 3.1 Root cause

`intelligence/decision_score.py`'s `DecisionScorer.score()` computes a 0–100 confidence score by summing each factor's weighted contribution and dividing by the total weight available (`max_possible`). The bug: **every** factor's weight was added to that denominator, even factors whose `direction` is `"NEUTRAL"` and can therefore *never* contribute to the numerator (`buy_weighted`/`sell_weighted`).

Two distinct sources of NEUTRAL factors exist in `intelligence/confluence_engine.py`:

- **By design, always NEUTRAL:** the `session` factor (`_session_factor`, line 336) is hardcoded `direction = "NEUTRAL"` — its own docstring says *"Session is direction-neutral — it boosts the OTHER factors' weight"*, but the surrounding scoring code never implemented that boosting; it only added dead weight to the denominator.
- **Conditionally NEUTRAL from missing data:** `intermarket` (15% weight) and `news` (10% weight) fall back to `NEUTRAL`/0 strength whenever their upstream context is unavailable or blocked (`confluence_engine.py:404`, `436`) — the identical dilution effect, but this time it's a **missing-data penalty**, not a deliberate design choice.

Net effect: confidence and the aligned-factor percentage were mechanically capped below their true value on every decision that included any NEUTRAL factor — which, because `session` is *always* present and *always* NEUTRAL, was every decision in the audited window.

## 3.2 Fix applied

**File:** `intelligence/decision_score.py`
**Change:** in the `score()` loop, only add `f.weight` to `total_weight_used` when `f.direction in ("BUY", "SELL")`. NEUTRAL factors still appear in `result.factors` / `total_factors` for transparency and downstream contradiction logic — they are only excluded from the confidence-normalization denominator, since they cannot mathematically earn a share of it.

```python
if f.direction in ("BUY", "SELL"):
    total_weight_used += f.weight
```

This is the smallest change that corrects the normalization math. No thresholds, risk controls, or gate logic were touched.

## 3.3 Regression test

`tests/unit/test_decision_score_neutral_weight.py` (3 tests, all passing):

- A "perfect" bullish setup with all directional factors maxed and a NEUTRAL session factor now reaches ~100% buy_score instead of being capped at 95%.
- A setup with `intermarket`/`news` missing (NEUTRAL/0) produces the *same* buy_score as the identical setup with those factors omitted entirely — presence of missing-data factors no longer changes the outcome.
- An all-NEUTRAL factor list still safely resolves to `NEUTRAL` direction / 0 scores (no divide-by-zero).

```
tests/unit/test_decision_score_neutral_weight.py::test_neutral_session_factor_does_not_deflate_a_perfect_setup PASSED
tests/unit/test_decision_score_neutral_weight.py::test_missing_data_neutral_factor_does_not_deflate_confidence PASSED
tests/unit/test_decision_score_neutral_weight.py::test_all_neutral_factors_do_not_divide_by_zero PASSED
```

Full `tests/unit` suite was also re-run: 8 pre-existing failures remain, all verified **unrelated** to this change (missing `typed_config` dependency module in this sandbox, and one news-bypass test whose fixture hardcodes `confidence=80` directly rather than deriving it from `DecisionScorer`).

## 3.4 Before vs after (quantitative)

A realistic borderline BUY setup — decent SMC/liquidity/currency-strength/technical factors, `intermarket` and `news` data unavailable, `session` present as it always is — was scored with both the pre-fix and post-fix normalization logic (`audit_fixes/after/unit_level_before_after.json`):

| | buy_score | confidence | vs. 55% Min-confidence gate |
|---|---|---|---|
| **BEFORE fix** | 29.47 | 45.36% | **BLOCKED** |
| **AFTER fix** | 42.11 | 60.53% | **PASSED** |

Same inputs, same market data, same risk controls — the only difference is that dead weight from data-unavailable/structurally-neutral factors no longer counts against the denominator. This demonstrates the bug was capable of turning a legitimately-passing setup into a false "Min confidence" block, consistent with the 72.5%-naturally-low-confidence pattern in the audit evidence.

## 3.5 End-to-end backtest validation — environment limitation

Per the validation instructions, I attempted:
```
py -3.13 -m backtest.persistent_runner --symbols EURUSD --timeframe H1 --workers 1 --no-llm
```
(adapted to `python3` in this Linux sandbox; ran against `data/EURUSD_H1.csv`, 6,197 bars.)

The run completed end-to-end for both the pre-fix and post-fix code (checkpoints in `audit_fixes/before/backtest_before_metrics.json`), but in **this sandboxed environment** it produced **zero decision cycles of any kind** (0 trades, 0 WAIT, 0 permission_blocked, 0 risk_rejected) for both versions — a verbose/debug run confirmed no `[TradePermission]`/`MinConfidenceDiagnostic` log lines were emitted at all. This indicates the offline harness here isn't exercising the same analysis → decision → permission pipeline that produced the uploaded live-log evidence (most likely due to missing broker/API configuration not present in this sandbox, e.g. MT5/News/Intermarket data sources). Because the *before* and *after* runs behaved identically (both silent), this is an environment gap rather than something introduced by the fix.

**Recommendation:** re-run the exact validation command in your full environment (with MT5/data feeds configured) for both the pre-fix and post-fix code, and compare `blocked_trade_analysis`-style output the same way this audit did. The isolated unit-level before/after in §3.4 is real, reproducible, and directly targets the confirmed bug, but it is not a substitute for a live trade-level A/B run.

---

# 4. Modules — Disposition

## 4.1 KEEP (no evidence supports changing)
- **S/R zone alignment gate** (`trade_permission.py`) — fail-open design confirmed correct in code; 397 blocks are all real computed misalignments, already re-tuned in a prior 2026-08-02 audit.
- **Risk approved / Valid signal sequencing** — co-occurrence is architecturally expected, not redundant filtering.
- **`sl_swing_anchor` + `tp_structure_validation` compound penalty** — deliberate, postmortem-driven risk control (Day 137 GBPCAD loss). Not touched.
- **Zone cooldown, Signal persistence, execution filters (news_intelligence, confluence_avoid)** — low volume (8–34 blocks each), no evidence either way.

## 4.2 EDIT (fixed in this pass)
- **`intelligence/decision_score.py` → `DecisionScorer.score()`** — neutral-factor weight-dilution bug. Fixed, tested.

## 4.3 WEIGHT changes
- None applied. `FACTOR_WEIGHTS` (session 5%, intermarket 15%, news 10%, etc.) were **not** retuned — the fix corrects how existing weights are *normalized*, not what the weights themselves are. Changing the weight values would be optimization against this specific dataset, which the brief explicitly rules out.

## 4.4 REMOVE
- None. No module was removed. The "Confluence quality" gate's redundancy with "Session quality" (finding #2) is architecturally interesting but was **not removed** — that would be removing a risk control without outcome data to justify it.

## 4.5 Requiring more evidence (outcome/OHLC data needed)
- Whether "Confluence quality" as a *second*, independent gate on top of "Session quality" is net-beneficial or purely redundant (needs hypothetical WR/PF on the blocked-trade set).
- Whether the `sl_swing_anchor`/`tp_structure_validation` compound penalty is net-positive vs. simply reducing trade frequency (needs outcome-tagged data).
- Whether the post-11:00-UTC disappearance of Session/Confluence/Min-confidence blocks reflects a real regime shift or an upstream signal-generation issue (needs a longer log window across multiple sessions/days).

## 4.6 Remaining risks / issues
- This sandbox could not reproduce the live decision pipeline for a true before/after trade-level backtest (§3.5) — the fix's real-world trade-frequency and win-rate impact should be validated in the full environment before being treated as final.
- The neutral-factor-weight fix will, by construction, **raise** confidence values across many historical decisions (as shown in §3.4). This is a correctness fix, not a tuning knob, but it does mean the Min-confidence gate will now block fewer trades than before purely due to more accurate math — this should be monitored post-deployment, not assumed to be an unambiguous improvement, since higher confidence numbers alone don't guarantee better trade outcomes.
- `filter_summary.csv`/`filter_combo_summary.csv`/`blocked_trade_analysis.csv` still cannot answer the "which filter blocks the most *good* trades" question — that requires historical OHLC data per the original forensic report's own conclusion, which still holds.

---

# 5. Priority List

**P0 — Fix immediately**
- (Done) Neutral-factor weight-dilution bug in `intelligence/decision_score.py` — confirmed, fixed, tested.

**P1 — Fix next**
- Re-run the full live-environment backtest (`persistent_runner`, with MT5/data feeds configured) before **and** after this fix, over a multi-day window, and compare `blocked_trade_analysis`-style output to confirm the real-world effect on block rate and (once OHLC-based outcome data is available) win rate/expectancy.

**P2 — Investigate**
- Whether "Confluence quality" should remain a fully independent hard gate given its 100% overlap with "Session quality" (needs hypothetical-outcome data, not currently available).
- Whether the `sl_swing_anchor + tp_structure_validation` compound penalty (and the broader entry-quality penalty map in general) is net-positive for expectancy, once outcome data exists.
- The post-11:00-UTC disappearance of several filters' block activity — likely an upstream signal-generation/regime question, not a filter-tuning one.

**P3 — Leave unchanged**
- S/R zone alignment gate (fail-open behavior confirmed correct).
- Risk approved / Valid signal sequencing.
- Zone cooldown, Signal persistence, and low-volume execution filters (news_intelligence, confluence_avoid).
- `FACTOR_WEIGHTS` values (session 5%, smc 25%, etc.) — not retuned.

---

# 6. Files in this delivery

```
audit_fixes/
  before/
    blocked_trade_analysis.csv              (original, unmodified)
    scenario_comparison.csv                 (original, unmodified)
    filter_combo_summary.csv                (original, unmodified)
    filter_summary.csv                      (original, unmodified)
    penalty_attribution.csv                 (original, unmodified)
    forensic_analysis_report.md             (original, unmodified)
    confidence_counterfactual_summary.json  (original, unmodified)
    backtest_before_metrics.json            (pre-fix full backtest run, this sandbox)
  after/
    decision_score_FIXED.py                 (fixed module, for diff reference)
    test_decision_score_neutral_weight.py   (regression tests)
    unit_level_before_after.json            (quantitative before/after evidence)
  reports/
    audit_fix_report.md / .pdf              (this report)
```
