# Priority #1 — Leakage Audit & Purged/Embargoed Cross-Validation

Status: **Implemented, default-off, backward compatible.** No live behavior
changes until a caller explicitly opts in.

## What was wrong

1. **`ml/triple_barrier_labels.py`** — a complete, correct path-dependent
   labeling implementation (`triple_barrier_labels()`, `meta_labels()`,
   `compute_label_uniqueness()`) existed with **zero importers** anywhere
   else in the codebase. Dead code.
2. **`ml/label_generator.py`** (the live labeler, via `core/runtime.py`) —
   labels a sample profitable based on price at exactly `t + horizon`,
   regardless of the path taken. A sample can be labeled a winning BUY even
   if price first crashed through what would have been the actual stop-loss
   and only recovered later. The model learns to endorse trades that would
   have been stopped out live.
3. **`ml/dataset_builder.py`** — a single chronological `iloc[:train_end]`
   split with **zero gap** between train/val/test. Since labels look
   `horizon` bars into the future, the last `horizon` rows of `X_train`
   have labels computed from data that falls inside `X_val`'s range — a
   direct, measurable leak, repeated at every fold boundary in
   `ml/walk_forward.py`.

Zero occurrences of "purge" or "embargo" existed anywhere in `ml/`, `risk/`,
`core/` before this change.

## What changed (all additive, all default-off)

| File | Change | Default behavior |
|---|---|---|
| `ml/cv_splitter.py` | **NEW** — `PurgedEmbargoedSplitter` | N/A (new file) |
| `ml/triple_barrier_labels.py` | **+** `TripleBarrierLabeler` class wrapper | Not called unless `labeling_method="triple_barrier"` |
| `ml/dataset_builder.py` | **+** `labeling_method`, `use_purged_split`, `label_horizon` params; `Dataset.sample_weight/labeling_method/cv_method/purge_stats` fields | Unchanged (`fixed_horizon`, unpurged) |
| `ml/walk_forward.py` | **+** `purge_window`, `embargo_pct` params to `validate()`; `WalkForwardFold.rows_purged/embargo_rows`, `WalkForwardResult.cv_method` | Unchanged (0, 0.0) |
| `ml/model_trainer.py` | `sample_weight` forwarded to XGBoost/RandomForest `.fit()`; `labeling_method`/`use_purged_split`/`label_horizon` forwarded through `train_all()`; ModelStore metadata tagged with `labeling_method`/`cv_method` | `sample_weight=None` ≡ omitting the kwarg |
| `ml/feature_store.py` | **+** `labels.labeling_method`, `labels.sample_weight` columns (additive, nullable, defaulted) | Every pre-migration row reads as `fixed_horizon` / `1.0` |
| `ml/validation.py` | **+** `model_validation.cv_method` column; `purge_window`/`embargo_pct` forwarded to `WalkForwardValidator.validate()`; `ValidationReport.cv_method` | Unchanged (`naive_walk_forward`) |
| `core/runtime.py` | **+** `cv_splitter` registered in the service registry | Registered as a no-op singleton |

Nothing is deleted. `ml/label_generator.py` is untouched — its MAE/MFE
computation is genuinely useful independent of this fix, and it stays live
as the default labeler.

## How to run the new (purged / triple-barrier) pipeline

```python
from ml.model_trainer import get_model_trainer

trainer = get_model_trainer()

# Shadow run — does NOT touch the champion model, ModelStore versions it
# separately and tags it labeling_method="triple_barrier", cv_method="purged_embargoed"
result = trainer.train_all(
    pair="EURUSD", timeframe="15m",
    labeling_method="triple_barrier",
    use_purged_split=True,
    label_horizon=48,   # match your real forward-looking window
)
```

Or via the module-level batch wrapper:

```python
from ml.model_trainer import train_all
train_all(labeling_method="triple_barrier", use_purged_split=True, label_horizon=48)
```

## Migration plan (shadow model, never auto-promoted)

1. Ship with default parameters reproducing current behavior exactly —
   zero live impact (done, see tests below).
2. Train a **shadow** model with the new pipeline; `ModelStore` versions
   it alongside the current champion without promoting it.
3. Compare shadow vs. champion via `ValidationEngine.validate(...,
   purge_window=48, embargo_pct=0.01)` on the same untouched final period.
4. Only promote if the shadow beats the champion on that untouched period —
   not on its own internal CV score, since the two pipelines score on
   different folds.
5. Keep the old code paths intact (parameter-gated) for at least one full
   retraining cycle after cutover, for fast revert.

**Sanity check**: a purged/embargoed run's walk-forward score should be
equal-or-lower than the naive run's on the same model — never higher. A
higher score after purging is itself evidence of a bug in the purge logic,
not an improvement (you can't fix a leak and see performance go up).

## Known related issue found during this audit (not fixed — out of scope)

`ml/model_evaluator.py` contains a **second**, duplicate `WalkForwardValidator`
class with a `.run()` method. It's reachable only through
`ModelTrainer.walk_forward_validate()`, which itself has zero callers
anywhere in the codebase — fully dead code, separate from the leakage fix.
Flagging for a future cleanup pass; not touched here to keep this change
scoped to Priority #1.

## Tests

- `tests/test_cv_splitter.py` — 9 tests: backward compat, purge correctness,
  loud-failure-on-empty-train, embargo, k-fold, input validation.
- `tests/test_triple_barrier_labeler.py` — 4 tests: label uniqueness
  (isolated + fully-overlapping samples), SL-before-recovery correctness,
  `TripleBarrierLabeler` class ≡ underlying functions.
- `tests/test_dataset_builder_purged.py` — 5 tests: old path unchanged,
  purged split removes boundary rows, triple-barrier end-to-end, missing-
  OHLC-columns guard, empty-training-set guard.

All 18 new tests pass. Full existing suite (`pytest tests/`) run alongside
them: 37 passed, 3 pre-existing failures in `test_decision_pipeline.py` /
`test_whole_decision_system.py` — unrelated to this change (no `ml.*`
import anywhere in that call chain; confirmed by grep before and after).

Also verified end-to-end against a real (temp) `FeatureStore` + real
XGBoost/RandomForest through `ModelTrainer.train_all()`, and confirmed the
`ml_features.db` / `model_validation.db` schema migrations are idempotent
against a simulated pre-migration database.
