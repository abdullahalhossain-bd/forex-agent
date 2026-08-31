# forex-agent — v3.21 Live Trading Update (SZ)

## বাংলা সারসংক্ষেপ

এই zip-এ v3.18→v3.21 ব্যাকটেস্ট সেশনে খুঁজে পাওয়া **সব বাগ-ফিক্স live trading path-এ** বসানো হয়েছে।
ব্যাকটেস্টে যেসব bypass/bug লস করিয়েছিল, সেগুলো লাইভেও একইভাবে ক্ষতি করত — সেগুলোই এখানে ঠিক করা হয়েছে।
`data/`, `.git/`, ব্যাকটেস্ট DB/cache বাদ; শুধু সোর্স + কনফিগ।

**গুরুত্বপূর্ণ:** live-এর জন্য বাধ্যতামূলক — MasterAnalyst-এর জন্য আসল LLM API key (Groq/Gemini) দিতে হবে।
ব্যাকটেস্টের deterministic mock লাইভে কাজ করে না। প্রথমে অবশ্যই MT5 **demo** অ্যাকাউন্টে টেস্ট করুন।

---

## 1. Changes applied to the live path

### 1.1 `risk/risk_engine.py` — risk & SL/TP fixes
| Change | Detail |
|---|---|
| ATR SL multiplier `1.5 → 2.0` | v3.18: wider structural SL, fewer noise stop-outs |
| SL floor `10 → 15 pips` | v3.18: 4 losses were 10-pip noise stop-outs |
| **Structure-SL / RR-floor-TP desync fix (v3.21SZ)** | When the structure SL replaces the ATR SL and is *wider*, the RR-floor TP used to keep the OLD pre-structure distance → `rr_ratio < execution minimum` → mass rejection (every XAUUSD trade rejected: "R:R 1.08 below execution minimum 2.00"). `floor_tp` is now recomputed from the updated SL distance. |

### 1.2 `core/trader.py` — execution-layer fixes
| Change | Detail |
|---|---|
| **Micro-SL snapping fix (v3.19)** | `auto_correct_sl_tp` could snap SL to a swing 0.2–8 pips from entry, silently bypassing the 15-pip floor (13/13 such trades lost). The RiskEngine floored SL is now kept whenever the correction would tighten below 15 pips. |
| **Same-direction loss cooldown (v3.21, NEW in live path)** | After a trade closes at a loss in direction X, new X-direction entries are blocked for `SZ_LOSS_COOLDOWN_MIN` minutes (default **300** = 20 M15 bars). Mirrors the v3.21 backtest guard that stopped 15 consecutive knife-catch re-entries during the Aug-2024 crash window. Recording happens in `_process_closed_trades`, blocking in `evaluate_decision_core`. |

### 1.3 `agents/analysis_agent.py` — authority fixes (no more unauthorized trades)
| Change | Detail |
|---|---|
| `_bt_aggressive = False` | Backtest no longer uses the aggressive rule-signal bypass → backtest = live parity. (Live was already effectively off; the flag can no longer diverge.) |
| Adaptive-fill "resurrection" layer disabled | Was re-entering a trade from `adaptive_decision` even when the MasterAnalyst said WAIT. |
| Unified-consensus "fallback" resurrection disabled | Was filling a lone-engine signal as a trade when the master said WAIT. |
| Override gate (see 1.4) | Single-layer rule override of a master WAIT no longer possible. |

### 1.4 `core/entry_safety_filters.py` — override gate kill-switch (v3.21)
`evaluate_override_gate` now **rejects by default** ("master WAIT stays WAIT").
The v3.18–v3.21 sessions proved this gate lets the rule engine flip a master
WAIT into a trade on its own. Env `SZ_DISABLE_OVERRIDE_GATE=0` restores the
legacy behavior.

### 1.5 `analysis/stop_hunt_direct_lane.py` — direct lane kill-switch (v3.21)
`get_stop_hunt_direct_signal` returns `None` by default — the lane generated
SELLs straight from `StopHuntSignalEngine` even when the DecisionAgent said
WAIT. Env `SZ_DISABLE_STOP_HUNT_LANE=0` restores legacy behavior.

### 1.6 `core/constants.py` — gate defaults restored
| Gate | Old default | New default |
|---|---|---|
| `MIN_CONFIDENCE_PROD` | 55 (drifted) | **80** (matches v3.18–v3.21 backtest gate) |
| `MIN_RR_PROD` | 2.0 | 2.0 (unchanged; env-overridable) |
| `MAX_OPEN_TRADES` | 10 | 10 (unchanged) |
| `DAILY_LOSS_LIMIT_PCT` | 5.0 | 5.0 (unchanged) |

---

## 2. Environment switches (all optional)

| Env var | Default | Meaning |
|---|---|---|
| `SZ_LOSS_COOLDOWN_MIN` | `300` | Minutes to block same-direction re-entry after a loss. `0` = off. |
| `SZ_DISABLE_OVERRIDE_GATE` | `1` | `1` = single-layer override of master WAIT always rejected. `0` = legacy gate. |
| `SZ_DISABLE_STOP_HUNT_LANE` | `1` | `1` = stop-hunt direct lane off. `0` = legacy lane. |
| `MIN_RR_PROD` | `2.0` | Minimum reward:risk for TP floor. |
| `MIN_CONFIDENCE_PROD` | `80` | Master verdict confidence floor. |
| `MAX_OPEN_TRADES` | `10` | Concurrent position cap. |
| `DAILY_LOSS_LIMIT_PCT` | `5.0` | Daily loss circuit breaker. |

---

## 3. Required for live (checklist)

1. **LLM keys** — MasterAnalyst needs a real LLM (Groq / Gemini / Cerebras /
   SambaNova / OpenRouter). Without a key it falls back to the rule-engine
   signal (degraded mode). The deterministic mock used in backtests is NOT
   shipped — it is a test-only harness.
2. **MT5 connection** — set mode to `mt5_demo` first. Live pip-value lookup
   needs a working MT5 connection; on a **Cent account** the static fallback
   is wrong by ~100x (the code logs a loud warning — do not ignore it).
3. **One symbol per AITrader instance** — loss cooldown state is per-instance
   (per symbol), matching the backtest harness semantics.
4. **Paper/demo soak test** — run `paper` mode for at least a week before
   real money. The v3.21 backtest results were produced with a deterministic
   signal harness; live LLM verdicts will differ.
5. **Honest expectation** — validated windows on real historical CSVs
   (GBPUSD M15, 1300 test bars): ~44–54% WR, PF 1.4–2.35, max DD ~2.9%.
   There is no 100%-win-rate setting in live; anyone promising one is selling
   a backtest artifact.

## 4. Not included in this zip

`data/` (1.2 GB historical CSVs), `.git/`, backtest DBs/caches/results,
`memory/` runtime state, `logs/`, `__pycache__`, `*.v318bak` backups.
Clone the GitHub repo for the original data, or re-download MT5 history.

## 5. File manifest of the update (quick diff guide)

```
core/constants.py              MIN_CONFIDENCE_PROD default 55 → 80
core/entry_safety_filters.py   override gate: env-gated reject-by-default
core/trader.py                 v3.19 micro-SL fix; v3.21 loss cooldown
                               (record + gate + helper); ATR SL mult/floor
                               already applied earlier in risk layer
agents/analysis_agent.py       _bt_aggressive=False; 2 resurrection layers off
risk/risk_engine.py            ATR_SL_MULT 2.0; SL floor 15p; structure-SL /
                               RR-floor-TP desync fix (v3.21SZ)
analysis/stop_hunt_direct_lane.py  env-gated disable (default off)
```
