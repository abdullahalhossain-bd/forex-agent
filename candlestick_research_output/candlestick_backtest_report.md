# Candlestick Pattern Research Backtest Report

> **⚠ THIS RUN USED SYNTHETIC (randomly generated) DATA.** No real OHLC dataset was found in the uploaded project (see Part 14 notes in the accompanying summary). Every number below is a demonstration of the engine's correctness and speed, NOT a real trading result. Re-run against real EURUSD/GBPUSD/USDJPY/XAUUSD... history to get real answers.

- Rows processed: **155,000**
- Pattern occurrences (all patterns, all directions): **136,484**
- Trades simulated (all RR ratios, both entry models): **49,172**
- Runtime: **16.71s** (9,276 rows/sec)


WARNING — MULTIPLE TESTING / DATA-MINING BIAS: this report evaluates many pattern x pair x timeframe x session x month x regime combinations. With enough combinations, some will show an apparently strong edge by chance alone even if no real edge exists. The highest observed win rate or expectancy in any single slice is NOT automatically the best strategy — prefer combinations with N>=100 (ideally >=300), positive out-of-sample expectancy, and stability across the train/test split over combinations that merely look best in one slice of the data.


## 1-2. Best patterns overall (baseline entry, RR 1:1, ranked by expectancy x log(N))

| pattern                  |   n |   win_rate |   avg_r |   expectancy_r |   profit_factor |
|:-------------------------|----:|-----------:|--------:|---------------:|----------------:|
| Bearish Harami           |  34 |     0.6176 |  0.2353 |         0.2353 |          1.6154 |
| Three Outside Up         |  95 |     0.5789 |  0.1579 |         0.1579 |          1.375  |
| Three Inside Down        |  42 |     0.5714 |  0.1429 |         0.1429 |          1.3333 |
| Bullish Separating Lines | 139 |     0.5396 |  0.0791 |         0.0791 |          1.1719 |
| Three Inside Up          |  53 |     0.5472 |  0.0943 |         0.0943 |          1.2083 |
| Bearish Separating Lines | 114 |     0.5351 |  0.0702 |         0.0702 |          1.1509 |
| Dark Cloud Cover         | 296 |     0.527  |  0.0541 |         0.0541 |          1.1143 |
| Tweezer Top              | 523 |     0.5201 |  0.0402 |         0.0402 |          1.0837 |
| Three Outside Down       |  75 |     0.52   |  0.04   |         0.04   |          1.0833 |
| Morning Doji Star        | 429 |     0.5058 |  0.0117 |         0.0117 |          1.0236 |


## 3. Best pattern x pair combinations

| pair   | pattern           |   n |   win_rate |   avg_r |   expectancy_r |   profit_factor |
|:-------|:------------------|----:|-----------:|--------:|---------------:|----------------:|
| XAUUSD | Inverted Hammer   |  48 |     0.6458 |  0.2917 |         0.2917 |          1.8235 |
| GBPUSD | Hammer            |  49 |     0.6327 |  0.2653 |         0.2653 |          1.7222 |
| XAUUSD | Dark Cloud Cover  |  68 |     0.6176 |  0.2353 |         0.2353 |          1.6154 |
| AUDUSD | Hanging Man       |  50 |     0.62   |  0.24   |         0.24   |          1.6316 |
| XAUUSD | Shooting Star     |  36 |     0.6111 |  0.2222 |         0.2222 |          1.5714 |
| XAUUSD | Hanging Man       |  61 |     0.5902 |  0.1803 |         0.1803 |          1.44   |
| USDJPY | Piercing Line     |  55 |     0.5818 |  0.1636 |         0.1636 |          1.3913 |
| GBPUSD | Morning Doji Star |  75 |     0.5733 |  0.1467 |         0.1467 |          1.3438 |
| EURUSD | Evening Doji Star |  93 |     0.5699 |  0.1398 |         0.1398 |          1.325  |
| EURUSD | Bearish Kicker    | 318 |     0.5472 |  0.0943 |         0.0943 |          1.2083 |


## 4. Best pattern x timeframe combinations

| timeframe   | pattern                  |   n |   win_rate |   avg_r |   expectancy_r |   profit_factor |
|:------------|:-------------------------|----:|-----------:|--------:|---------------:|----------------:|
| M15         | Bearish Separating Lines |  72 |     0.625  |  0.25   |         0.25   |          1.6667 |
| H1          | Shooting Star            |  67 |     0.6119 |  0.2239 |         0.2239 |          1.5769 |
| H4          | Tweezer Bottom           |  57 |     0.5965 |  0.193  |         0.193  |          1.4783 |
| H1          | Tweezer Top              | 157 |     0.5732 |  0.1465 |         0.1465 |          1.3433 |
| H1          | Morning Doji Star        | 115 |     0.5652 |  0.1304 |         0.1304 |          1.3    |
| H1          | Hanging Man              |  76 |     0.5658 |  0.1316 |         0.1316 |          1.303  |
| M15         | Three Inside Up          |  37 |     0.5676 |  0.1351 |         0.1351 |          1.3125 |
| H1          | Inverted Hammer          |  70 |     0.5571 |  0.1143 |         0.1143 |          1.2581 |
| H1          | Piercing Line            |  93 |     0.5484 |  0.0968 |         0.0968 |          1.2143 |
| M15         | Three Outside Up         |  60 |     0.55   |  0.1    |         0.1    |          1.2222 |


## 5. Confirmed vs. unconfirmed (Part 6H)

| pattern         | entry_model   |   n |   win_rate |   expectancy_r |   profit_factor |
|:----------------|:--------------|----:|-----------:|---------------:|----------------:|
| Inverted Hammer | confirmed     |  45 |     0.5556 |         0.1111 |          1.25   |
| Shooting Star   | baseline      | 266 |     0.5038 |         0.0075 |          1.0152 |
| Hanging Man     | baseline      | 296 |     0.5034 |         0.0068 |          1.0136 |
| Inverted Hammer | baseline      | 291 |     0.4914 |        -0.0172 |          0.9662 |
| Hammer          | confirmed     | 121 |     0.438  |        -0.1119 |          0.7979 |
| Shooting Star   | confirmed     | 116 |     0.4052 |        -0.1897 |          0.6812 |
| Hammer          | baseline      | 277 |     0.4946 |       nan      |          0.9964 |
| Hanging Man     | confirmed     |  27 |     0.2593 |        -0.4815 |          0.35   |

## 6. Worst patterns (lowest expectancy, N>=30)

| pattern              |   n |   win_rate |   avg_r |   expectancy_r |   profit_factor |
|:---------------------|----:|-----------:|--------:|---------------:|----------------:|
| Hanging Man          | 296 |     0.5034 |  0.0068 |         0.0068 |          1.0136 |
| Piercing Line        | 316 |     0.4968 | -0.0063 |        -0.0063 |          0.9874 |
| Tweezer Bottom       | 566 |     0.4965 | -0.0071 |        -0.0071 |          0.986  |
| Evening Doji Star    | 465 |     0.4946 | -0.0108 |        -0.0108 |          0.9787 |
| Inverted Hammer      | 291 |     0.4914 | -0.0172 |        -0.0172 |          0.9662 |
| Morning Star         | 250 |     0.488  | -0.024  |        -0.024  |          0.9531 |
| Bearish Engulfing    | 746 |     0.4799 | -0.0402 |        -0.0402 |          0.9227 |
| Bullish Engulfing    | 715 |     0.4769 | -0.0462 |        -0.0462 |          0.9118 |
| Bullish Harami       |  33 |     0.4545 | -0.0909 |        -0.0909 |          0.8333 |
| Upside Gap Two Crows |  42 |     0.4524 | -0.0952 |        -0.0952 |          0.8261 |


## 9. In-sample vs out-of-sample (chronological 70/30 split, Part 10)

| pattern                  |   n_in_sample |   win_rate_in_sample |   expectancy_r_in_sample |   n_out_of_sample |   win_rate_out_of_sample |   expectancy_r_out_of_sample | sign_flip   |
|:-------------------------|--------------:|---------------------:|-------------------------:|------------------:|-------------------------:|-----------------------------:|:------------|
| Three Outside Up         |            60 |               0.6667 |                   0.3333 |                35 |                   0.4286 |                      -0.1429 | True        |
| Three Inside Up          |            37 |               0.6216 |                   0.2432 |                16 |                   0.375  |                      -0.25   | True        |
| Bullish Separating Lines |           102 |               0.5686 |                   0.1373 |                37 |                   0.4595 |                      -0.0811 | True        |
| Upside Gap Two Crows     |            33 |               0.5455 |                   0.0909 |                 9 |                   0.1111 |                      -0.7778 | True        |
| Three Outside Down       |            50 |               0.54   |                   0.08   |                25 |                   0.48   |                      -0.04   | True        |
| Tweezer Top              |           376 |               0.516  |                   0.0319 |               147 |                   0.5238 |                     nan      | True        |
| Dark Cloud Cover         |           209 |               0.5167 |                   0.0335 |                87 |                   0.5287 |                     nan      | True        |
| Hanging Man              |           212 |               0.5094 |                   0.0242 |                84 |                   0.4762 |                     nan      | True        |
| Bearish Separating Lines |            79 |               0.5063 |                   0.0127 |                35 |                   0.6    |                       0.2    | False       |
| Tweezer Bottom           |           397 |               0.5013 |                   0.0025 |               169 |                   0.4852 |                      -0.0296 | True        |
| Shooting Star            |           182 |               0.5    |                   0      |                84 |                   0.5119 |                       0.0238 | True        |
| Inverted Hammer          |           201 |               0.4975 |                  -0.005  |                90 |                   0.4778 |                      -0.0444 | False       |
| Evening Doji Star        |           310 |               0.4968 |                  -0.0065 |               155 |                   0.4839 |                     nan      | True        |
| Morning Doji Star        |           298 |               0.4899 |                  -0.0201 |               131 |                   0.542  |                       0.084  | True        |
| Bullish Engulfing        |           491 |               0.4868 |                  -0.0265 |               224 |                   0.4554 |                      -0.0893 | False       |
| Bearish Engulfing        |           534 |               0.4813 |                  -0.0375 |               212 |                   0.4717 |                     nan      | True        |
| Piercing Line            |           215 |               0.4744 |                  -0.0512 |               101 |                   0.5347 |                     nan      | True        |
| Morning Star             |           167 |               0.4671 |                  -0.0659 |                83 |                   0.5301 |                       0.0602 | True        |
| Three White Soldiers     |           625 |               0.5072 |                 nan      |               299 |                   0.5318 |                       0.0635 | True        |
| Bearish Kicker           |          1063 |               0.4976 |                 nan      |               458 |                   0.5022 |                     nan      | True        |
| Three Black Crows        |           677 |               0.4771 |                 nan      |               253 |                   0.4348 |                     nan      | True        |
| Bearish Counterattack    |           250 |               0.496  |                 nan      |               102 |                   0.4608 |                     nan      | True        |
| Bullish Kicker           |          1054 |               0.5019 |                 nan      |               474 |                   0.5042 |                     nan      | True        |
| Three Inside Down        |            29 |               0.5172 |                   0.0345 |                13 |                   0.6923 |                       0.3846 | False       |
| Evening Star             |           177 |               0.4689 |                 nan      |                87 |                   0.4713 |                     nan      | True        |
| Hammer                   |           207 |               0.5121 |                 nan      |                70 |                   0.4429 |                      -0.0928 | True        |
| Bullish Counterattack    |           283 |               0.4523 |                 nan      |               118 |                   0.5    |                     nan      | True        |
| Bearish Harami           |            25 |               0.64   |                   0.28   |                 9 |                   0.5556 |                       0.1111 | False       |
| Bullish Harami           |            18 |               0.4444 |                  -0.1111 |                15 |                   0.4667 |                      -0.0667 | False       |

_A `sign_flip=True` row means the pattern's edge did NOT survive out-of-sample — treat with suspicion regardless of how good the in-sample number looked._


## 10. Sample-size warnings

No patterns fell below the N>=30 threshold in the overall table.
