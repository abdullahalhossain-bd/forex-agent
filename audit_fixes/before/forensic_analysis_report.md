# Trade Permission System — Forensic Analysis Report

## ⚠️ সবচেয়ে গুরুত্বপূর্ণ সীমাবদ্ধতা — আগেই বলে নেওয়া দরকার

**Hypothetical WIN/LOSS outcome reconstruction সম্ভব হয়নি।** এই তিনটি zip-এ কোনো independent historical OHLC price series ছিল না:
- প্রতিটি backtest run-এর `trader_EURUSD.db` SQLite file-এ `candles`, `indicators`, `analysis`, `trades` — সব table **খালি (0 rows)**।
- `memory/` বা `logs/` কোথাও bar-by-bar future price data নেই যেটা দিয়ে reliably SL/TP hit reconstruct করা যায়।

তাই spec-এর নির্দেশ অনুযায়ী (§11 "no guessing"), item 4, 5, 6, 8-এর hypothetical outcome / blocked-trade WR / potential wins lost — সবকিছুতে **`OUTCOME_NOT_AVAILABLE`** বসানো হয়েছে। যা কিছু data থেকে সরাসরি derive করা যায় (blocked counts, penalty attribution, confidence classification, overlap counts) সেগুলো পুরোপুরি বের করা হয়েছে।

## ⚠️ দ্বিতীয় গুরুত্বপূর্ণ finding — দুটো ভিন্ন session/config মিশে আছে

Uploaded data-তে আসলে **দুইটা ভিন্ন সময়ের, ভিন্ন config-এর run** আছে:

| Session | Source | Time window (2026-08-10, UTC) | Config |
|---|---|---|---|
| **A — Active filters** | `logs/execution.log` (`permission.checked` events, JSON) | 09:55 – 13:31 | Session quality / Confluence quality / Min confidence **সক্রিয়ভাবে block করছে** |
| **B — Bypass mode** | `logs/trader.log` + rotated `.1`–`.5` (text logs) | 15:53 – 19:31 | `permission_bypass` **flag ON** — Session quality 89.3%, Confluence quality 82.2%, Min confidence 98.0% evaluations **BYPASSED** |

আমার পুরো quantitative analysis (CSV গুলো) **Session A (execution.log)**-এর উপর ভিত্তি করে, কারণ এটাই একমাত্র structured/machine-readable source যেখানে filter গুলো actual কাজ করছিল। Session B আলাদাভাবে নিচে qualitatively রিপোর্ট করা হলো, কারণ সেখানে filter blocking prácticamente disabled ছিল বলে ওই window-এর block data দিয়ে "filter কতটা restrictive" মাপা যাবে না — উল্টো সেটা দেখায় filter গুলো কতটা bypass হচ্ছিল লাইভে।

---

## 1–3. Structured blocked-trade rows, Min-confidence counterfactual, Penalty attribution

**Source:** `logs/execution.log`, 1050 total decision cycles (EURUSD, single symbol — এই log window-এ multi-symbol ছিল না), **1022 blocked, 28 allowed** (allowed-গুলোর মধ্যেও প্রায় সবই ছিল WAIT/NO TRADE — কোনো raw BUY/SELL signal ছিল না বলে trade হয়নি)।

→ `blocked_trade_analysis.csv` (1022 rows, প্রতি row = এক blocked decision cycle)

**Fields যা log-এ সরাসরি ছিল:** timestamp, symbol, raw_signal (BUY/SELL/NO TRADE), decision, allowed, confidence_pre_penalty, confidence_post_penalty, effective_min_confidence (constant 55%, পুরো dataset-এ কোনো dynamic threshold observed হয়নি), entry_quality_penalty, failed_checks, entry_quality_failed_checks, sl, tp, lot, risk_approved.
**যেগুলো log-এ ছিল না (তাই UNKNOWN রাখা হয়েছে):** timeframe (execution.log field দেয়নি — trader.log অনুযায়ী likely H1), entry_price, spread, session/regime label, signal source module votes।

### Min confidence counterfactual (item 2)

Total "Min confidence" failures (যেখানে এই check block-এর কারণ হিসেবে failed_checks-এ ছিল): **218**

| Class | Count | % |
|---|---|---|
| A. naturally_low_confidence (pre < 55%) | 158 | 72.5% |
| B. penalty_created_block (pre ≥ 55%, post < 55%) | **60** | **27.5%** |
| C. threshold_only/dynamic_threshold | 0 | 0.0% — dataset-এ effective threshold সবসময় constant 55%, কোনো dynamic threshold behavior পাওয়া যায়নি |
| D. other_upstream_block (Min confidence + অন্তত আরেকটা check একসাথে fail) | 197 | 90.4% |
| Borderline (post-penalty confidence থ্রেশহোল্ডের ±5 পয়েন্টের মধ্যে) | 8 | 3.7% |

→ `confidence_counterfactual_summary.json`, `confidence_analysis.csv` (confidence bucket-wise breakdown)

### Penalty attribution (item 3)

execution.log-এ per-rule numeric penalty split নেই — শুধু total `entry_quality_penalty` + কোন কোন rule fail করেছে তার list। তাই co-occurrence ভিত্তিতে attribution করা হয়েছে (→ `penalty_attribution.csv`):

| Entry-quality rule | কতবার blocked cycle-এ present | ঐ cycle-গুলোতে avg TOTAL penalty |
|---|---|---|
| rejection_psychology | 665 | 17.8 |
| tp_structure_validation | 516 | 19.9 |
| sl_swing_anchor | 488 | 20.4 |
| indecision_candles | 156 | 21.0 |
| tp_above_unconfirmed_spike | 125 | 19.2 |
| round_number_tp | 76 | 19.4 |
| exhaustion_filter | 55 | 25.8 |
| rejection_wick_at_entry | 41 | 25.0 |
| indicator_confluence | 31 | 15.3 |
| fresh_high_rejection | 16 | 28.4 |

মোট penalty summed across all blocked cycles: **12,194 points**। এর মধ্যে ঠিক **60 টা** cycle-এ penalty একাই confidence-কে threshold-এর নিচে নামিয়ে block তৈরি করেছে (Class B, উপরে)।

**সতর্কতা:** এটা exact per-rule attribution না — কারণ execution.log প্রতিটি rule-এর individual point-value দেয় না, শুধু aggregate penalty + কোন rule-গুলো fail করেছে সেটা দেয়। `trader.log`-এর Session B-তে (আলাদা সময়ের sample) `[MinConfidenceDiagnostic]` লাইনে exact per-rule points পাওয়া গেছে (উদাহরণ: sl_swing_anchor=-3, tp_structure_validation=-3, rejection_psychology=-5, sl_tp_structure_compound=-10) — কিন্তু এটা ভিন্ন session-এর sample বলে Session A-এর প্রতিটি row-এ সরাসরি বসানো হয়নি, শুধু reference হিসেবে দেওয়া হলো।

---

## 4–6, 8. Filter-specific blocked-trade outcomes ও counterfactual dataset

**সব hypothetical outcome column = `OUTCOME_NOT_AVAILABLE`** (কারণ উপরে বর্ণিত)। যা পাওয়া গেছে:

→ `filter_summary.csv` — প্রতিটি filter কত trade block করেছে:

| Filter | Blocked count | % of all blocked cycles |
|---|---|---|
| S/R zone alignment | 397 | 38.8% |
| Risk approved | 329 | 32.2% |
| Valid signal | 291 | 28.5% |
| Min confidence | 218 | 21.3% |
| Session quality | 168 | 16.4% |
| Confluence quality | 136 | 13.3% |
| Trend alignment (regime) | 47 | 4.6% |
| Zone cooldown (duplicate entry) | 34 | 3.3% |
| Signal persistence | 34 | 3.3% |
| Execution filter: news_intelligence | 8 | 0.8% |
| Execution filter: confluence_avoid | 8 | 0.8% |

→ `filter_combo_summary.csv` — item 6-এর requested combo-গুলো (item 6):

| Combo | Count |
|---|---|
| Session quality + Confluence quality | **136** |
| Session quality + Min confidence | 119 |
| Confluence quality + Min confidence | 87 |
| Session quality + Confluence quality + Min confidence | 87 |
| Min confidence + S/R zone alignment | 84 |
| Confluence + Min confidence + Risk + Session + Valid signal (সব ৫টা একসাথে) | 37 |

Top general pairwise overlaps (primary 6 filter): Risk approved+Valid signal (291, প্রায় সবসময় একসাথে — expected, কারণ risk reject হলে upstream-এ valid signal-ও থাকে না), তারপর Confluence+Session (136), Min confidence+Session (119)।

---

## 7. Baseline performance (backtest_runs, ৮টা run মিলিয়ে)

`backtest_runs/*/trades/EURUSD.jsonl` থেকে সব executed trade একত্র করে (→ `scenario_comparison.csv`, per-run + combined):

| Metric | Value |
|---|---|
| Total executed trades | 19 |
| Wins | 2 |
| Losses | 17 |
| Win rate | 10.53% |
| Net P&L | −$306.64 |
| Gross profit | $68.12 |
| Gross loss | −$374.76 |
| Profit factor | 0.182 |
| Expectancy/trade | −$16.14 |
| Avg R | −0.734 R |
| Max drawdown | $306.64 (on $10,000 starting balance) |

**Total signals / trade frequency:** শুধু ২টা run-এ (`20260809_114044`, `20260810_111823`) checkpoint.json/rejection_stats সংরক্ষিত ছিল (বাকি ৬টা run-এ ওই summary মিসিং)।
- `20260809_114044`: 400 bars processed → 125 WAIT, 19 risk_rejected, **254 permission_blocked**, শুধু 2 trades executed.
- `20260810_111823`: 700 bars processed → 213 WAIT, 25 risk_rejected, **462 permission_blocked**, **0 trades executed**।

অর্থাৎ trade frequency বার প্রতি প্রায় 0.3–0.5%। System প্রায় সবসময় trade নিচ্ছে না।

---

## 9. Time / direction breakdown (Session A)

Direction (raw_signal) of blocked cycles: BUY 470, SELL 261, NO TRADE 290, WAIT 1 — অর্থাৎ প্রায় দুই-তৃতীয়াংশ block হয়েছে এমন একটা signal-এর উপর যেটা আসলে BUY বা SELL ছিল (731টা), মানে filter-গুলো বাস্তব directional signal-কেই বেশি block করছে, শুধু noise-কে না।

Hour-by-hour (UTC) pattern লক্ষণীয়:
- **Min confidence / Session quality / Confluence quality** — শুধু 09:00–11:00 window-এ block করেছে (এর পরে দেখা যায়নি)।
- **S/R zone alignment / Risk approved / Valid signal** — পুরো 09:00–13:00 জুড়ে সক্রিয় ছিল।

এটা suggest করে যে ~11:00 UTC-এর পরে upstream signal-scorer/decision-agent স্তরে raw confidence বা signal quality-ই কমে গিয়েছিল (তাই confidence-based filter-গুলো আর trigger হওয়ার মতো candidate পায়নি), filter নিজে relax হয়নি। এটা filter tuning-এর প্রশ্ন না, upstream signal generation-এর প্রশ্ন — এই distinction গুরুত্বপূর্ণ যাতে ভুল layer-এ fix না করা হয়।

Symbol: পুরো dataset EURUSD-only (উভয় session-এ), তাই symbol-wise breakdown প্রযোজ্য না।

---

## 10. Machine-readable outputs (delivered)

| File | Rows | Content |
|---|---|---|
| `blocked_trade_analysis.csv` | 1022 | এক row = এক blocked decision cycle |
| `filter_summary.csv` | 11 | filter-wise blocked count (outcome columns = OUTCOME_NOT_AVAILABLE) |
| `filter_combo_summary.csv` | 22 | item 6-এর সব requested combo + pairwise overlap |
| `confidence_analysis.csv` | 10 | confidence bucket-wise pass/penalty-block/naturally-low breakdown |
| `scenario_comparison.csv` | 9 | per-run + combined baseline performance |
| `penalty_attribution.csv` | 10 | entry-quality rule → co-occurrence count + avg total penalty |
| `confidence_counterfactual_summary.json` | — | item 2-এর aggregate numbers |

---

## 12. Final conclusions

1. **কোন filter সবচেয়ে বেশি trades block করছে?** — S/R zone alignment (397 cycles, 38.8%), তারপর Risk approved (329) ও Valid signal (291)। কিন্তু S/R zone alignment-এর detail log দেখায় বেশিরভাগ ক্ষেত্রেই আসলে "no S/R zone data — not evaluated" (missing data-এর কারণে fail, real filtering না) — এটা raw CSV-তে verify করা দরকার আরও গভীরভাবে data quality issue হিসেবে।
2. **কোন filter-এর blocked trades hypothetical WR সবচেয়ে বেশি?** — `OUTCOME_NOT_AVAILABLE`। এই প্রশ্নের উত্তর দিতে হলে historical OHLC price series লাগবে যেটা এই upload-এ নেই।
3. **কোন filter সবচেয়ে বেশি bad trades বাদ দিচ্ছে?** — একই কারণে `OUTCOME_NOT_AVAILABLE`।
4. **কোন filter সবচেয়ে বেশি good trades block করছে?** — একই কারণে `OUTCOME_NOT_AVAILABLE`।
5. **Min confidence-এর কত % failure আসলে penalty-created?** — **27.5%** (60/218)। বাকি 72.5% naturally low confidence ছিল (penalty ছাড়াই threshold-এর নিচে)। এই 27.5%-ই সবচেয়ে actionable target যদি penalty rule-গুলো relax করার কথা ভাবা হয় — বাকিটা penalty relax করে সমাধান হবে না, upstream confidence generation-এর সমস্যা।
6. **Session + Confluence কি redundant/overlapping?** — **হ্যাঁ, প্রায় সম্পূর্ণ overlapping।** যতবার Confluence quality fail হয়েছে (136), ঠিক ততবারই Session quality-ও fail হয়েছে (136/136 = 100%)। উল্টোদিকে Session quality fail-এর 81% (136/168) ক্ষেত্রে Confluence quality-ও fail করেছে। অর্থাৎ Confluence quality filter-টা independently প্রায় কিছুই আলাদাভাবে block করছে না যা Session quality already block করছে না — দুটো filter একসাথে চলা মানে একটাকে সরালেও block-count খুব একটা কমবে না, কিন্তু কোনটা সরানো "সঠিক" সেটা নির্ধারণ করতে hypothetical WR লাগবে (available না)।
7. **কোন filter relax করার strongest evidence আছে?** — বর্তমান data দিয়ে outcome-ভিত্তিক evidence দেওয়া সম্ভব না। তবে *structural* evidence অনুযায়ী: (a) Min confidence penalty-rule-গুলো (rejection_psychology, tp_structure_validation, sl_swing_anchor) — এগুলো একত্রে 60টা trade block করেছে শুধু penalty দিয়ে, যেটা directly tunable; (b) Confluence quality — Session quality-র সাথে 100% redundant হওয়ায় এটা duplicate filtering করছে, তাই এটাই প্রথম candidate consolidation/removal-এর জন্য (outcome data দিয়ে confirm করা উচিত)।
8. **কোন filter untouched রাখা উচিত?** — Data থেকে নির্ধারণ করা যায় না বাস্তব outcome ছাড়া। Risk approved ও Valid signal-কে untouched রাখাই যুক্তিসঙ্গত মনে হচ্ছে কারণ এগুলো বেশিরভাগ ক্ষেত্রে upstream-এ signal না থাকারই ফল (291 case-এ Risk+Valid signal একসাথে fail — এটা filter অতিরিক্ত restrictive হওয়ার লক্ষণ না, বরং signal generation-এ trade candidate-ই কম থাকার লক্ষণ)।
9. **কোন filter remove করলে frequency বাড়বে কিন্তু WR/PF কমতে পারে?** — নির্ধারণ করা যাচ্ছে না (`OUTCOME_NOT_AVAILABLE`)।
10. **কোন filter remove/relax করলে WR ও expectancy দুটোই improve হওয়ার evidence আছে?** — নির্ধারণ করা যাচ্ছে না বর্তমান data দিয়ে।

### সবচেয়ে গুরুত্বপূর্ণ next step

এই পুরো counterfactual/opportunity-cost analysis (item 4–6, 8, এবং conclusion 2–4, 9–10) সম্পূর্ণ করতে **historical OHLC price data** (যে symbol/timeframe-এ backtest হয়েছে — EURUSD H1, বা execution.log-এর window অনুযায়ী EURUSD সম্ভবত 15m/H1) লাগবে, প্রতিটি blocked trade-এর timestamp-এর পরবর্তী কমপক্ষে `max_hold_bars` (config অনুযায়ী 100 bars) পর্যন্ত। এটা upload করলে `blocked_trade_analysis.csv`-এর প্রতিটি row-এ SL/TP simulate করে real hypothetical WIN/LOSS বসানো সম্ভব, এবং তখনই conclusion 2, 3, 4, 7, 9, 10-এর সঠিক data-driven উত্তর পাওয়া যাবে।
