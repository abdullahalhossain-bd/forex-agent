# Devil's Advocate + DA Safety Net — Pre-Implementation Audit
Repo: forex-agent · Files inspected: core/devils_advocate.py, core/da_safety_net.py,
risk/live_risk_manager.py, risk/structure_stop.py, risk/entry_quality_guardrails.py,
risk/trade_permission.py, core/constants.py, core/spread_policy.py, broker/spread_monitor.py

No code has been changed yet. This is Phase 1–20 per your spec.

---

## PHASE 1 — DATA LINEAGE TABLE

| FIELD | CURRENT SOURCE(S) | OTHER SOURCES | AUTHORITATIVE SOURCE | DUPLICATE CALC? | RISK |
|---|---|---|---|---|---|
| symbol (clean) | `DASafetyNet._clean_symbol()` (regex/len heuristic) | `core.constants.strip_mt5_suffix/clean_symbol`, `core.spread_policy.clean_symbol` | `core.constants.clean_symbol` | **YES** — 3rd independent implementation | MEDIUM — edge cases (exotics, indices) can diverge across the 3 |
| pip_size | `core.constants.get_pip_size` (used in `_check_structure_sl` fallback), `core.spread_policy.pip_size_for_symbol`, `entry_quality_guardrails._pip_value` | `_check_spread` also re-derives pip_size via `get_pip_size` | `core.constants.PIP_SIZE` table / `core.spread_policy.pip_size_for_symbol` (MT5-aware superset) | **YES — 3 independent tables** | **HIGH** — see finding F1 below (XAUUSD mismatch) |
| pip_value (USD/lot) | `core.constants.PIP_VALUE_USD` (static), `get_live_pip_value_per_lot` (live MT5) | none else | `get_live_pip_value_per_lot` when MT5 available, else `PIP_VALUE_USD` | NO (proper fallback chain) | LOW, documented |
| spread_pips | `ind_ctx["spread_pips"]` (decision pipeline) | `trade_context` via LiveRiskManager caller | pipeline-calculated (upstream of DA) | NO for DA itself | — |
| max_spread (limit) | `core.spread_policy.get_max_spread_pips` (DA primary), `broker.spread_monitor.MAX_SPREAD_PIPS` (DA fallback), **`risk/live_risk_manager.py:598` hardcoded `5.0`** | — | `core.spread_policy` (per-symbol + asset-class table, most complete) | **YES — 3 independently conflicting tables** | **HIGH** — see finding F2 |
| ATR (price) | `ind_ctx["atr"]` | — | analysis/decision pipeline | NO | — |
| ATR (pips) | `ind_ctx["atr_pips"]` if present, else DA derives via `get_pip_size` + `DA_MIN_ATR_PIPS` floor | — | should be pipeline-calculated; DA currently re-derives as fallback | PARTIAL | MEDIUM — floor value `DA_MIN_ATR_PIPS=4.0` is a flat cross-symbol floor (Phase 8 violation) |
| SL price / TP price | `risk_out["sl_price"/"tp_price"]` | `decision_out["sl"/"tp"]` (fallback in trade_permission.py) | risk pipeline (`risk_engine.py` → `trade_permission.py`) | NO for DA | — |
| SL pips | `risk_out["sl_pips"]`, DA derives pip_size from `sl_distance / sl_pips_hint` when present | `structure_stop.compute_structure_stop_pips` (own floor, see F3) | risk pipeline | NO for DA, but risk pipeline itself has duplicate floors (F3) | MEDIUM |
| support/resistance / swing levels | `sr_ctx["nearest_support"/"nearest_resistance"]` (decision_out) | `mc["nearest_support"/"nearest_resistance"]`, `entry_quality_guardrails._find_swing_lows/_highs` (independent swing finder), `structure_stop.find_fractal_swing_low/high` (a 3rd independent swing finder) | ambiguous — **three separate swing-detection implementations exist** (`entry_quality_guardrails.py`, `structure_stop.py`, and whatever feeds `sr_ctx`) | **YES** | MEDIUM-HIGH — DA's `_check_structure_sl` picks a 4th path (level from `sr_ctx`) instead of consuming the already-computed `entry_quality_detail` swing-anchor result that ran moments earlier in the same pipeline |
| H4 trend | `decision_out["mtf_trends"]["4h"/"H4"]`, `mc["h4_trend"]` | — | decision_agent analysis output | NO | — |
| session | `mc["sessions_active"]` (list) / `mc["session"]` (string, sometimes a quality label not a session name) | `utils.session.SessionAnalyzer` (wall-clock, live-only) | pipeline-supplied session list; wall-clock only for live via explicit opt-in flag | NO (correctly guarded already) | LOW — already well-designed |
| news state | `analysis_out["news_ctx"]["risk_level"]` | — | analysis pipeline | NO | — |
| lot size / balance / risk_pc | `risk_out["lot"/"lot_size"]`, `risk_out["balance"]`, `risk_out["risk_pc"/"risk_percent"]` | `LiveRiskManager.position_sizer` (actual sizing engine) | `position_sizer` output flowing into `risk_out` | NO for DA (consumer only) | — |
| tick_value / tick_size / contract_size | **not consumed by DA safety net at all** — `_check_lot_sizing` uses `core.constants.PIP_VALUE_USD` static table, not broker MT5 metadata | `core.constants.get_live_pip_value_per_lot` (exists, correct, but DA doesn't call it) | `get_live_pip_value_per_lot` | N/A | **HIGH** — see finding F4 |

---

## KEY FINDINGS (dangerous, in priority order)

**F1 — Three independent, disagreeing pip-size tables.**
- `core/constants.py: PIP_SIZE["XAUUSD"] = 0.01`
- `core/spread_policy.py` fallback: XAU → `0.01` (agrees)
- `risk/entry_quality_guardrails.py: _pip_value()` → **XAUUSD = 0.1** (10× off from the other two)

`_pip_value()` feeds `check_sl_swing_anchor` and `check_tp_structure_validation`, i.e. it decides whether a gold SL/TP is "structurally anchored." A 10× pip-size error there silently changes what counts as "near a swing" for every XAUUSD trade. This is exactly the class of bug Phase 3 asks to hunt for, and it's real, not hypothetical — I found it by reading the code, not by testing.

**F2 — Three independently conflicting spread-limit tables, one of them a flat non-symbol-aware default.**
- `core/spread_policy.py` (richest, per-symbol + asset-class, e.g. EURUSD=3.0, XAUUSD=400.0)
- `broker/spread_monitor.py: MAX_SPREAD_PIPS` (EURUSD=2.0, XAUUSD=5.0 — both disagree with spread_policy)
- `risk/live_risk_manager.py:598` — **`max_spread = 5.0` flat, for every symbol including XAUUSD**, ungated by asset class

`DASafetyNet._check_spread` already does the right thing (prefers `core.spread_policy`, falls back to `broker.spread_monitor`, then a `25.0` last resort) — DA itself is fine. But `LiveRiskManager.check_trade_permission()` (a **different, earlier gate in the same pipeline**) uses its own hardcoded `5.0` for every symbol, which would falsely reject virtually every normal XAUUSD spread rejection-check... no — it would falsely **reject majors whose real limit is tighter than 5 pips is not the danger; the danger is XAUUSD, where 5 pips is far too tight for a genuine ~260-pip live gold spread**, meaning `LiveRiskManager` could be vetoing legitimate gold trades on ordinary spread while DA's own (correct) check would have passed them — a silent, upstream, wrong gate that never reaches DA at all. This needs a decision from you: should `LiveRiskManager` be changed to call `core.spread_policy.get_max_spread_pips()` too? (Out of DA's scope strictly, but it's the same authoritative-source violation Phase 2 forbids, sitting one hop upstream of DA.)

**F3 — Two disagreeing SL-noise floors.**
- `core/da_safety_net.py: DA_MIN_SL_PIPS = 6.0` (env-configurable)
- `risk/structure_stop.py: compute_structure_stop_pips()` → `max(round(sl_pips,1), 5.0)` (hardcoded, not env-configurable)

Both are "E-type" (Phase 4) arbitrary developer floors, not broker/spec-derived. Per your Phase 4 instructions these should not simply be deleted, but they disagree with each other (5.0 vs 6.0) and neither is symbol/timeframe-aware (Phase 18: XAUUSD "5 pips" is noise-floor nonsense at $0.01/pip vs EURUSD "5 pips" at 1e-4).

**F4 — `_check_lot_sizing` ignores the authoritative live pip-value source that already exists in this codebase.**
`core/constants.get_live_pip_value_per_lot()` reads real MT5 `tick_value`/`tick_size` and is explicitly documented as required for cent accounts / JPY / exotics (its own docstring warns that the static table is "WRONG by ~100x" on cent accounts). `DASafetyNet._check_lot_sizing` calls `get_pip_value_usd()` (the static table) only — it never attempts the live path. This is a direct Phase 2 violation: an authoritative live source exists and DA silently uses the approximation instead.

**F5 — `_check_structure_sl` invents its own protective-level lookup instead of consuming the already-computed, richer swing-anchor result.**
`entry_quality_guardrails.check_sl_swing_anchor()` runs earlier in the same pipeline (visible on `trade_context["perm_out"]["entry_quality_detail"]`, which DA already reads elsewhere for `evidence["entry_quality"]`) and computes: nearest swing, distance in both pips and ATR-multiples, and an anchored/not-anchored verdict — using fractal-swing detection. `DASafetyNet._check_structure_sl` instead re-derives protective level from `sr_ctx["nearest_support"/"nearest_resistance"]` (a different, coarser signal) and applies its own separate `DA_STRUCT_SL_BUFFER_PIPS`/`DA_STRUCT_PROXIMITY_MULT` thresholds — a second, parallel structure-SL judgment that can disagree with the first one already sitting in evidence. This is the clearest Phase 6 finding: "Do not recreate swing detection inside DA... if the risk manager already calculated a valid structure-aware SL, DA should validate it rather than invent another SL."

**F6 — `DA_MIN_ATR_PIPS = 4.0` is a flat cross-symbol/cross-timeframe floor**, applied identically to EURUSD and XAUUSD alike (Phase 8 explicitly forbids this: "Do not compare XAUUSD ATR to EURUSD ATR using the same absolute pip threshold"). No ATR-percentile or symbol+timeframe historical distribution is consulted anywhere in this module or its neighbors (I did not find one in `analysis/` under a quick pass — flagging as **UNKNOWN, needs confirmation** whether one exists elsewhere before Phase 8 can be implemented properly).

---

## PHASE 4 — THRESHOLD CLASSIFICATION

| Threshold | Current default | Class | Recommendation |
|---|---|---|---|
| `DA_MIN_SL_PIPS` | 6.0 | E (arbitrary, conflicts with F3's 5.0) | Reconcile with `structure_stop.py`'s floor or make both read one shared constant; keep as hard VETO only for genuinely sub-spread SLs (spread-relative, not flat) |
| `DA_STRUCT_SL_BUFFER_PIPS` | 2.0 | D (empirical, untested) | Downgrade influence — feed into WARN scoring, not auto-VETO, per Phase 6 |
| `DA_STRUCT_PROXIMITY_MULT` | 2.0 | D | Keep as a relevance filter (already correctly SKIPs rather than VETOs when level is far) — no change needed |
| `DA_MIN_ATR_PIPS` | 4.0 | E (flat, symbol-blind) | Replace with symbol/timeframe-relative regime check (Phase 8) once ATR-percentile source is confirmed to exist or built |
| `DA_NEWS_SPIKE_ATR_MULT` | 2.5 | C (market-structure rule, plausible) | Keep as VETO candidate (news spike into a blowout is a legitimate hard-safety case per your Phase 5 examples) but log OOS validation status |
| `DA_MAX_SPREAD_RATIO` / `_VETO` | 0.35 / 0.50 | C/D | Already correctly two-tier (WARN then VETO) — good example of the target design; keep |
| `DA_LOT_MISMATCH_TOL` | 0.25 | B (risk-management requirement) | Legitimate hard-VETO candidate — keep, but fix F4 first (wrong pip value source makes the ratio itself wrong) |
| `structure_stop.py` floor `5.0` | 5.0 (hardcoded) | E | Should not exist as a second, disagreeing copy of `DA_MIN_SL_PIPS` |
| `entry_quality_guardrails.DEFAULT_SL_PROXIMITY_ATR` | 1.5×ATR | D | Independent of DA's own buffer — another place doing a similar judgment with a different method; not itself dangerous but part of the "3 swing-detectors" duplication (F5) |

---

## WHICH CHECKS STAY HARD VETO (as currently designed, largely correct)
- Structural SL on wrong side of entry (already in `entry_quality_guardrails`, `BLOCK` severity) ✅ keep hard.
- `lot_sizing_check` risk-overflow / mismatch — legitimate B-type, keep hard **once F4 is fixed**.
- `spread_check` absolute-pip breach of the authoritative `core.spread_policy` max — keep hard.
- `atr_regime_check` news-spike-candle (candle > 2.5×ATR) — plausible hard case, keep pending your OOS confirmation.
- Session `market_closed` (no sessions at all) — keep hard (this is a factual/broker-hours condition, not a heuristic).

## WHICH CHECKS SHOULD MOVE TO WARN (currently over-eager VETOes)
- `session_filter`'s `asian_dead_chop` VETO for non-JPY/AUD/NZD pairs — this is a C-type "historical WR killer" empirical rule, not a hard broker/risk fact. Per your Phase 5/9 guidance this belongs as WARN unless you've validated it OOS as robust policy.
- `h4_trend_filter` VETO — currently vetoes on ANY H4 opposition. Per Phase 10, this should only be a hard VETO if counter-trend is *explicitly* prohibited by strategy policy; otherwise WARN/confidence-penalty. Currently there's no such policy flag being checked.
- `structure_sl_check`'s "SL sits inside structure noise" VETO — should likely become WARN once it's reconciled with `entry_quality_guardrails`'s already-more-rigorous swing-anchor verdict (F5), to avoid a double-jeopardy VETO from two versions of the same judgment.

## WHICH CHECKS SHOULD SKIP MORE OFTEN (currently OK, listing for completeness)
- All checks already SKIP cleanly on missing data — this module is in fact one of the better-designed parts of the codebase for exactly this ("never fabricate a PASS/VETO"). No changes needed here except where noted above.

---

## OVERFITTING / VALIDATION STATUS (Phase 18) — UNKNOWN, needs your input
I found no dataset/date-range/OOS-validation record attached to any of the six checks' thresholds in this codebase (no comments citing a specific backtest run for `DA_MIN_ATR_PIPS`, `DA_STRUCT_SL_BUFFER_PIPS`, etc., the way the spread-ratio two-tier fix (2026-09-01 comment) documents its own origin). Before I harden any of these into permanent WARN/VETO policy, I need to know: do you have `blocked_trade_outcome_audit.py` / `scripts/backtest_da_layers.py` output I should read to classify each threshold as ROBUST vs OVERFIT-RISK? Both scripts exist in the repo but I haven't run them yet.

---

## FILES THAT WOULD NEED CHANGES (Phase 20-A)
1. `core/da_safety_net.py` — `_check_structure_sl` (F5), `_check_lot_sizing` (F4), `_check_atr_regime` (F6), session/H4 VETO→WARN reclassification.
2. `risk/entry_quality_guardrails.py` — `_pip_value()` XAUUSD bug (F1); consider exposing its swing-anchor result in a shape `da_safety_net.py` can consume directly instead of re-deriving.
3. `risk/structure_stop.py` — reconcile `5.0`-pip floor with `DA_MIN_SL_PIPS` (F3), or remove the local one in favor of a single shared constant.
4. `risk/live_risk_manager.py` — flat `max_spread = 5.0` (F2) — **flagging, not touching**, since it's upstream of DA and outside "Devil's Advocate / DA Safety Net" scope you defined; tell me if you want it included.
5. `core/devils_advocate.py` — prompt already implements most of Phase 12/13 well (FACT/UNKNOWN discipline, pillar grouping, WAIT-like conservative resolution via UNCERTAIN). Minor: no changes anticipated here beyond ensuring `_check_structure_sl`'s revised verdicts still surface correctly into `evidence["safety_net"]`.

## BEFORE I TOUCH ANY CODE, I need three decisions from you:

1. **F2 (LiveRiskManager's flat 5.0 spread limit)** — in scope for this task, or leave it (it's upstream of DA, not inside DA/DA-Safety-Net proper)?
2. **F6 (ATR regime, symbol/timeframe-relative)** — do you already have an ATR-percentile/regime data source somewhere I haven't found (e.g. in `analysis/` or `ml/`), or should I build a minimal symbol+timeframe historical-percentile lookup from scratch as part of this task?
3. **Phase 18 OOS validation** — should I run `scripts/backtest_da_layers.py` / `blocked_trade_outcome_audit.py` against existing data first to classify thresholds as ROBUST vs OVERFIT-RISK before I decide VETO-vs-WARN placement, or do you want me to proceed with the conservative default (anything not a broker/risk hard-fact becomes WARN) without waiting for that backtest?

I'll proceed straight into Phase 5–17 implementation as soon as you answer these three — happy to default to the conservative choices (exclude F2, build a minimal ATR-percentile helper, default everything non-authoritative to WARN) if you just say "go."
