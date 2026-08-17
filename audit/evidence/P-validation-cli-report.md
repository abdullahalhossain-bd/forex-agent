# CSV Validation CLI Report

**Generated:** 2026-08-17T14:33:16.649516+00:00
**Files validated:** 1

## Summary Table

| File | Symbol | TF | Rows | Days | Spread 0% | Gaps | Non-WKND Gaps | Errors | Warnings |
|------|--------|----|-----|------|-----------|------|---------------|--------|----------|
| data/EURUSD_M15.csv | EURUSD | M15 | 24780 | 364.59 | 52.8% | 59 | 59 | 0 | 2 |

## Per-File Detail

### data/EURUSD_M15.csv

**Warnings:**
- 59 non-weekend gaps (total gaps: 59)
- 52.8% of bars have spread=0 (re-download recommended)

**Stats:** {
  "rows": 24780,
  "start_utc": "2025-07-25 06:30:00+00:00",
  "end_utc": "2026-07-24 20:45:00+00:00",
  "date_range_days": 364.59,
  "duplicate_timestamps": 0,
  "gap_count": 59,
  "non_weekend_gap_count": 59,
  "spread_zero_pct": 52.8,
  "tick_volume_zero_pct": 0.0,
  "real_volume_zero_pct": 100.0,
  "ohlc_violations": {
    "high_lt_max_oc": 0,
    "low_gt_min_oc": 0,
    "high_lt_low": 0
  },
  "gaps_sample": [
    {
      "start": "2025-07-25 20:45:00+00:00",
      "end": "2025-07-27 21:00:00+00:00",
      "missing_bars": 192,
      "is_weekend": false
    },
    {
      "start": "2025-08-01 20:45:00+00:00",
      "end": "2025-08-03 21:00:00+00:00",
      "missing_bars": 192,
      "is_weekend": false
    },
    {
      "start": "2025-08-08 20:45:00+00:00",
      "end": "2025-08-10 21:00:00+00:00",
      "missing_bars": 192,
      "is_weekend": false
    },
    {
      "start": "2025-08-15 20:45:00+00:00",
      "end": "2025-08-17 21:00:00+00:00",
      "missing_bars": 192,
      "is_weekend": false
    },
    {
      "start": "2025-08-22 20:45:00+00:00",
      "end": "2025-08-24 21:00:00+00:00",
      "missing_bars": 192,
      "is_weekend": false
    }
  ]
}

