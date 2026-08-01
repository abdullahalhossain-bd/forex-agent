# Market DNA — Integration Guide

Context layer, not a decision engine. It never outputs BUY/SELL — only
`KNOWN`/`UNKNOWN`, a tier, and a position-size multiplier.

## Files

| File | Role |
|---|---|
| `analysis/market_dna.py` | `MarketDNADetector` — frozen-model fit + `approximate_predict` live inference |
| `analysis/dna_journal.py` | Per-cluster trade stats (Wilson CI, evidence tiers), rebuilt from scratch on every refit |
| `analysis/dna_drift.py` | PSI drift monitor with 4-band graduated thresholds |
| `analysis/dna_walkforward.py` | Three-way fold split (clusterer / meta-calibration / evaluation) to prevent nested leakage |
| `database/market_dna_schema.py` | SQLite tables: `market_dna_models`, `market_dna_cluster_stats`, `market_dna_drift_log`, `market_dna_trade_assignments` |

## One-time setup

```bash
pip install hdbscan --break-system-packages   # or: pip install -r requirements.txt
python -m database.market_dna_schema           # creates the 4 tables, idempotent
```

## Do NOT skip: initial training data requirement

Per the sequencing decision from the architecture review, **do not wire this
into live trading before the Day-80 base system has its own walk-forward
validation history** — that trade journal is what `dna_journal.py` needs to
produce non-trivial cluster stats. Training the clusterer itself (fold A)
needs candle data, not trades, so `MarketDNADetector.fit()` can technically
run earlier — but a cluster with zero trade history is `NO_STATISTICAL_EDGE`
by construction, so there's nothing to gate on until the journal fills up.

## Offline: fit + validate (run this before anything touches live code)

```python
from analysis.dna_walkforward import run_full_validation
from analysis.market_dna import DNAConfig

result = run_full_validation(
    historical_df,                  # OHLCV + indicators_v5 features, 2-10yr, one symbol/timeframe
    calibrate_fn=my_calibration_fn, # your Platt/isotonic/logistic-stacking function
    metric_fn=my_metric_fn,         # returns dict of Sharpe/PF/drawdown on fold C only
    config=DNAConfig(min_cluster_size=30, model_freeze_days=90),
)

detector = result["detector"]
detector.save()                      # -> models/market_dna/<model_id>.joblib (+ .json metadata)
print(result["fold_c_metrics"])      # the ONLY numbers allowed to be called "expected live performance"
```

Register the model row and (re)build its journal:

```python
import sqlite3, json
from datetime import datetime, timezone
from database.db import DB_PATH
from analysis.dna_journal import build_cluster_journal

meta = detector.metadata()
with sqlite3.connect(DB_PATH) as conn:
    conn.execute(
        """INSERT INTO market_dna_models
           (model_id, trained_at, train_window_start, train_window_end,
            n_train_rows, n_clusters, min_cluster_size, pca_components,
            feature_cols_json, model_path, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE')""",
        (meta["model_id"], meta["trained_at"], meta["train_window_start"],
         meta["train_window_end"], meta["n_train_rows"], meta["n_clusters"],
         meta["min_cluster_size"], meta["pca_components"],
         json.dumps(meta["feature_cols"]), str(detector.save())),
    )

# Label every historical trade's entry-bar features with THIS model,
# then aggregate -> market_dna_cluster_stats. Never reuse a previous
# model's journal rows for a new model_id.
stats = build_cluster_journal(trades_with_clusters_df, model_id=meta["model_id"])
now = datetime.now(timezone.utc).isoformat()
with sqlite3.connect(DB_PATH) as conn:
    for s in stats:
        conn.execute(
            """INSERT OR REPLACE INTO market_dna_cluster_stats
               (model_id, cluster_id, trades, wins, win_rate, ci_low, ci_high,
                profit_factor, expectancy_r, tier, position_multiplier, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (s.model_id, s.cluster_id, s.trades, s.wins, s.win_rate,
             s.ci_low, s.ci_high, s.profit_factor, s.expectancy_r,
             s.tier, s.position_multiplier, now),
        )
```

## Live: wiring into `agents/market_agent.py`

Add as a step immediately after the existing rule-based
`MarketRegimeDetector.detect(df)` call (does not replace it — see
architecture note in `market_dna.py` header). Sketch:

```python
from analysis.market_dna import MarketDNADetector
from analysis.dna_journal import lookup, decision_context

# Loaded once at process start (not per-tick) — frozen model, cheap to hold in memory.
_dna_detector = MarketDNADetector.load(active_model_path_from_db())
_dna_journal = load_cluster_stats_for(_dna_detector.model_id)   # SELECT * FROM market_dna_cluster_stats WHERE model_id = ?

# ... inside the per-candle flow, AFTER regime_result = regime_detector.detect(df):
dna_result = _dna_detector.predict_live(df.iloc[[-1]])   # last closed bar only
if dna_result["state"] == "KNOWN":
    dna_ctx = decision_context(lookup(_dna_journal, dna_result["cluster_id"]))
else:
    dna_ctx = decision_context(None)   # UNKNOWN path

regime_ctx["market_dna"] = dna_ctx   # merges into the same dict the signal engine already reads
```

Downstream in the signal/risk pipeline, read `regime_ctx["market_dna"]["recommendation"]`
(`APPROVE` / `REJECT` / `REDUCE_SIZE`) and `["position_multiplier"]` — multiply into
existing position sizing, do not let it independently open/close trades.

## Scheduled jobs

1. **Drift check** (e.g. hourly): pull recent live `dna_cluster_id` assignments,
   compare against `detector`'s training label distribution via `dna_drift.py`,
   write to `market_dna_drift_log`. On `MANDATORY_REFIT`, trigger step 2 early.
2. **Refit** (every `model_freeze_days`, or on mandatory-refit trigger): rerun
   the "Offline: fit + validate" section above on the latest window, `RETIRE`
   the old `market_dna_models` row (`status='RETIRED'`), insert the new one as
   `ACTIVE`. The old model's journal rows are left as historical record —
   never deleted, never remapped onto the new model_id.

## What is explicitly OUT of scope for this drop

- **Strategy switching per cluster** (trend-following vs mean-reversion vs
  scalping selection) — flagged as a large, separate overfitting surface in
  review. Not implemented here. Do not add it without its own dedicated
  walk-forward validation cycle per cluster-strategy pairing.
- **Confidence blending / meta-model formula itself** — `dna_walkforward.py`
  enforces *which data* a calibrator is allowed to see, but the calibration
  method (Platt/isotonic/logistic stacking) is intentionally left as a
  caller-supplied function so it isn't bolted on without deliberate design.
