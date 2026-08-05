# Bootstrap Tools — Day53 Confidence Penalty Fix (Complete Toolkit)

## সমস্যা

তুমি যে এরর দেখছো:
```
🎯 Day53 Confidence: ⚠️ Small sample (0/3) — Bayesian penalty -11
```

**Root cause:** `memory/pattern_stats.json` ফাইলটা ছিল না (fresh system)। `ConfidenceEngine._bayesian_penalty()` দেখে `sample_size=0` → -11pp penalty → confidence 60% এর নিচে → NO TRADE → কোনো outcome record হয় না → **chicken-and-egg loop**।

## সমাধান — ৬টা scripts

| # | Script | Data source | Speed | Legitimacy |
|---|--------|-------------|-------|------------|
| 1 | `generate_synthetic_samples.py` | None (empty entries) | ১ সেকেন্ড | Lowest (placeholder) |
| 2 | `replay_mt5_signals.py` (--use-csv) | `data/*.csv` | ১ মিনিট | Medium (real OHLC, simplified signals) |
| 3 | `replay_mt5_signals.py` (--no flag) | MT5 historical | ৫ মিনিট | High (real broker OHLC) |
| 4 | `import_backtest_samples.py` | Backtest DB | ৫ মিনিট | High (real backtest outcomes) |
| 5 | `import_mt5_history.py` | MT5 trade history | ১ মিনিট | **Highest** (your actual trades) |
| 6 | `run_paper_bootstrap.py` | Live market (paper) | ঘণ্টা | Highest (real-time) |
| ★ | `run_full_bootstrap.py` | Orchestration | — | Runs multiple above |
| ✓ | `check_penalty_status.py` | Verification | ১ সেকেন্ড | — |

## দ্রুত শুরু (Quick Start)

### Linux/CI (এই environment, কোনো MT5 নেই)

```bash
cd /home/z/my-project/download/forex-agent

# সব data/*.csv থেকে pattern stats তৈরি করো (১ মিনিট)
python scripts/bootstrap_tools/run_full_bootstrap.py --source csv --quick
```

### Windows host with MT5 terminal

```bash
cd C:\Projects\forex-agent

# MT5 historical data থেকে (৫ মিনিট)
python scripts/bootstrap_tools/run_full_bootstrap.py --source mt5 --months 12

# অথবা তোমার নিজের MT5 trade history থেকে (১ মিনিট, best)
python scripts/bootstrap_tools/run_full_bootstrap.py --source mt5-history --days 90
```

## বর্তমান অবস্থা (Already done in this session)

আমরা ইতিমধ্যে CSV fallback দিয়ে bootstrap চালিয়েছি:

```
✓ Total entries: 49
✓ Total trades recorded: 625
✓ Overall WR: 36.0%
✓ Mature patterns (3+ trades, NO penalty): 33
✓ Patterns with 1-2 trades (small penalty): 16
✓ Patterns with 0 trades (max penalty): 0
```

**Mature patterns এর উদাহরণ:**
- `Breakout|EURUSD|M15|RANGING` — 36 trades, 69.4% WR
- `Trend_Continuation|EURUSD|M15|RANGING` — 20 trades, 70.0% WR
- `Breakout|AUDUSD|H1|RANGING` — 50 trades, 40.0% WR
- `Breakout|USDJPY|H4|TRENDING` — 36 trades, 30.6% WR

## Penalty Math (reference)

```
sample_size=0:  penalty = -(8 + 0.6×(raw-50)) × 1.0 × (0.5 if bootstrap)
sample_size=1:  penalty = -(8 + 0.6×(raw-50)) × 0.42 × (0.5 if bootstrap)
sample_size=2:  penalty = -(8 + 0.6×(raw-50)) × 0.18 × (0.5 if bootstrap)
sample_size=3+: penalty = 0  ← target
```

raw_score=70, bootstrap=True (default fresh system):
- sample 0: penalty = **-10.0pp**
- sample 1: penalty = **-4.2pp**
- sample 2: penalty = **-1.8pp**
- sample 3: penalty = **0pp**

## প্রতিটা Script এর বিস্তারিত

### 1. `generate_synthetic_samples.py` — Empty entries
সব common pattern×pair×timeframe×regime combo-র জন্য empty entry তৈরি করে। Fake data দেয় না।

```bash
python scripts/bootstrap_tools/generate_synthetic_samples.py
python scripts/bootstrap_tools/generate_synthetic_samples.py --dry-run
python scripts/bootstrap_tools/generate_synthetic_samples.py --pairs EURUSD --timeframes H1
```

### 2. `replay_mt5_signals.py` — Historical OHLC replay
Strategy signals চালায় historical bars-এ, SL/TP হিট হলে outcome record করে।

```bash
# CSV fallback (Linux)
python scripts/bootstrap_tools/replay_mt5_signals.py \
    --pairs EURUSD,GBPUSD --timeframe H1 --use-csv --months 12

# MT5 historical (Windows)
python scripts/bootstrap_tools/replay_mt5_signals.py \
    --pairs EURUSD --timeframe H1 --months 12

# Dry-run
python scripts/bootstrap_tools/replay_mt5_signals.py \
    --pairs EURUSD --timeframe H1 --use-csv --months 6 --dry-run
```

### 3. `import_backtest_samples.py` — Backtest DB → pattern_stats
পুরানো ব্যাকটেস্ট-এর trade outcomes কে import করে।

```bash
python main.py --mode backtest --pairs EURUSD --timeframe 1h --bars 500
python scripts/bootstrap_tools/import_backtest_samples.py \
    --db backtest/backtest_run_EURUSD_H1.db
```

### 4. `import_mt5_history.py` — MT5 trade history (BEST)
তোমার নিজের MT5 account-এর closed trades থেকে outcomes import করে।

```bash
python scripts/bootstrap_tools/import_mt5_history.py --days 90
python scripts/bootstrap_tools/import_mt5_history.py --start 2026-01-01 --end 2026-06-30
python scripts/bootstrap_tools/import_mt5_history.py --days 30 --dry-run
```

### 5. `run_paper_bootstrap.py` — Live paper trading
Paper mode-এ N ঘণ্টা চালায়, real trades নেয়, real outcomes record করে।

```bash
python scripts/bootstrap_tools/run_paper_bootstrap.py --hours 4
python scripts/bootstrap_tools/run_paper_bootstrap.py --minutes 30
python scripts/bootstrap_tools/run_paper_bootstrap.py --status-only
```

### 6. `run_full_bootstrap.py` — Master runner (★ recommended)
সব সঠিক script গুলো একসাথে চালায়।

```bash
python scripts/bootstrap_tools/run_full_bootstrap.py --source csv --quick
python scripts/bootstrap_tools/run_full_bootstrap.py --source mt5 --months 12
python scripts/bootstrap_tools/run_full_bootstrap.py --source mt5-history --days 90
```

### ✓ `check_penalty_status.py` — Verify
বর্তমান penalty status দেখায়।

```bash
python scripts/bootstrap_tools/check_penalty_status.py
python scripts/bootstrap_tools/check_penalty_status.py --pair EURUSD
python scripts/bootstrap_tools/check_penalty_status.py --pattern Breakout
python scripts/bootstrap_tools/check_penalty_status.py --show-empty
```

## সুপারিশকৃত workflow

### সবচেয়ে দ্রুত (Linux/CI):
```bash
python scripts/bootstrap_tools/run_full_bootstrap.py --source csv --quick
```

### সবচেয়ে legitimate (Windows host):
```bash
# Step 1: নিজের MT5 trade history (১ মিনিট)
python scripts/bootstrap_tools/import_mt5_history.py --days 365

# Step 2: বাকি pairs-এর জন্য MT5 historical replay (৫ মিনিট)
python scripts/bootstrap_tools/replay_mt5_signals.py \
    --pairs EURUSD,GBPUSD,USDJPY --timeframe H1 --months 12

# Step 3: Verify
python scripts/bootstrap_tools/check_penalty_status.py
```

### Slowest কিন্তু most legitimate:
```bash
# Paper mode-এ রাতে চালাও
python scripts/bootstrap_tools/run_paper_bootstrap.py --hours 8
```

## সাবধানতা

1. **Fake data দেইনি** — Script 1 total_trades=0 দেয়, scripts 2-6 real outcomes দেয়।
2. **Threshold tune করিনি** — `MIN_SAMPLE_SIZE=3` কমাইনি (overfitting হতো)।
3. **Re-run safe না** — `import_mt5_history.py` একই date range দুইবার চালালে double-count হবে।
4. **Live data overrides** — 10 live trade জমলে সেটা bootstrap data-কে override করবে। এটাই স্বাভাবিক।
5. **Simplified signals** — `replay_mt5_signals.py` simplified pattern detector ব্যবহার করে। Full fidelity-র জন্য `main.py --mode backtest` চালাও।
