#!/usr/bin/env python3
"""
Bootstrap Sample Generator — Day53 Confidence Bayesian Penalty Fix.

PROBLEM:
  When you start the system fresh, `memory/pattern_stats.json` doesn't
  exist (or is empty). ConfidenceEngine then sees sample_size=0 for
  every pattern and applies a Bayesian penalty of -8 to -37 (depending
  on raw_score). This pushes confidence below the 60% MIN_CONFIDENCE
  gate → NO TRADE → no outcomes recorded → penalty persists forever
  (chicken-and-egg loop).

  The error message you saw:
    "Day53 Confidence: ⚠️ Small sample (0/3) — Bayesian penalty -11"
  means: pattern has 0 trades out of required MIN_SAMPLE_SIZE=3,
  applying a -11pp penalty to confidence.

SOLUTION (3 options in this toolkit):

  1. generate_synthetic_samples.py
     Generate a known-good pattern_stats.json with seed samples based
     on a documented baseline WR (e.g. 50% for unknown patterns).
     This is the HONEST approach: tells the engine "I have no real
     data, assume neutral 50% WR with low confidence".
     → Lifts penalty from -11 to 0 in one shot, but engine will
       still apply small-sample adjustments until 3+ real trades
       confirm the pattern.

  2. generate_paper_trade_samples.py
     Run the system in PAPER mode for N days and let it naturally
     accumulate real outcomes. Slowest but most legitimate.
     → Operator runs main.py --paper for 1-2 weeks.

  3. import_backtest_samples.py
     Run a backtest, then extract the trade outcomes from the
     backtest DB and import them into pattern_stats.json as if
     they were live trades. This seeds the confidence engine
     with REAL (backtest-derived) win/loss data.
     → Fast and legitimate IF the backtest is honest.

USAGE:
  python scripts/bootstrap_tools/generate_synthetic_samples.py --wr 50
  python scripts/bootstrap_tools/import_backtest_samples.py --db backtest/backtest_run_EURUSD_H1.db
  python scripts/bootstrap_tools/check_penalty_status.py

After running any of these, the "Small sample (0/3)" warning should
disappear or downgrade to "(1/3)" then "(2/3)" then "(3/3)" → no penalty.
"""
print(__doc__)
