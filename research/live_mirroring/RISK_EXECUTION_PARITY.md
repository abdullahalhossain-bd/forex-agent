# LIVE RISK → EXECUTION PARITY CONTRACT

Status: **P0 implementation contract / source audit**  
Branch: `research/live-mirroring-replay`

## 1. Source of truth

Historical replay MUST call the live risk/execution path. It must not re-implement the strategy's risk math in a second backtest-only function.

Verified live components:

```text
DecisionAgent
   ↓
RiskEngine.evaluate()
   ↓
Live PositionSizer.calculate()      # additional live sizing layer
   ↓
post-sizing live mutations (if any)
   ↓
TradePermission.check()
   ↓
Devil's Advocate / safety gates
   ↓
ExecutionRouter.execute()
   ↓
execution backend
   ↓
position lifecycle / close detection
```

`backtest/research_engine.py` currently describes an "exact same decision kernel", but its own execution loop still contains backtest-specific behavior (for example next-bar-open fallback and a separate `BrokerSimulator`). Therefore it is **not by itself proof of live execution parity**.

## 2. RiskEngine — exact live inputs and gates

`risk/risk_engine.py::RiskEngine.evaluate()` is authoritative for structural risk geometry and initial risk decision.

Inputs:

- `signal`
- `entry`
- `atr`
- `regime`
- `correlation_ctx`
- `df`
- `strategy`
- `stop_loss`
- `take_profit`

Verified gates before sizing:

1. `NO TRADE`, `WAIT`, `HOLD`, empty signal → reject.
2. Invalid/zero entry → reject; no fabricated entry fallback.
3. Daily-loss limit → reject.
4. Maximum-open-trades → reject.
5. Static correlation group check → reject when disallowed.
6. Live correlation context → reject at `corr_risk >= 0.90`; otherwise retain a correlation adjustment multiplier.
7. Missing/invalid ATR → reject.
8. Missing/short OHLC history (`<20`) → reject.
9. R:R policy must be available; failure is fail-closed.
10. Structural SL geometry is authoritative; ATR is a distance validator/buffer, not a replacement for structure.

## 3. Exact SL construction

RiskEngine uses the following source order:

### A. Signal-provided SL

If a valid `stop_loss` is supplied:

- BUY requires `SL < entry`.
- SELL requires `SL > entry`.
- Minimum distance is `max(15 pips, 0.5 × ATR)`.
- Too-tight SL is stretched to the minimum distance.
- Maximum distance is `2.5 × ATR`.
- Too-wide SL is clamped to the maximum distance.
- Final price is rounded to 5 decimals in this path.

### B. Structural SL when signal SL is absent

`risk.structure_stop.compute_structure_stop()` is called with:

- `method="swing_atr"`
- `lookback=50`
- `atr_buffer_mult=0.20`
- current ATR

Then the same side/distance validation and min/max clamping are applied.

**Replay rule:** the exact same RiskEngine call must be used with a dataframe cut off at the replay timestamp. Do not compute SL in the replay engine independently.

## 4. Exact TP construction

RiskEngine resolves the minimum R:R through `risk.rr_policy.get_min_rr()` and the execution minimum through `get_execution_min_rr()`.

The replay must persist, for every approved signal:

- `raw_tp`
- `rr_policy_min`
- `execution_min_rr`
- `final_tp`
- `tp_source`
- `sl_distance`
- `tp_distance`
- `final_rr`

Any mutation after RiskEngine must be captured rather than silently replaced.

## 5. Risk USD and lot sizing — important finding

The live chain does **not** end at `RiskEngine.evaluate()`.

`core/trader.py` invokes the live `PositionSizer.calculate()` after RiskEngine and stores its complete result under `risk_out["position_sizing"]`.

Verified base formula in `risk/position_sizer.py`:

```text
base_risk_usd = balance × risk_pct
base_lot      = base_risk_usd / (sl_pips × pip_value_per_lot)
base_lot      = clamp(base_lot, MIN_LOT, MAX_LOT)
```

The final live lot is then:

```text
final_lot = base_lot
            × kelly_mult
            × volatility_factor
            × confidence_factor
            × correlation_factor
            × drawdown_mult
            × streak_mult
            × profit_protection_mult
            × tier_mult
            × compounding_mult
```

with the live implementation applying rounding/caps and then recomputing actual risk:

```text
final_lot = max(MIN_LOT, min(round(final_lot, 2), MAX_LOT))
actual_risk_usd = final_lot × sl_pips × pip_value_per_lot
actual_risk_pct = actual_risk_usd / balance
```

A hard maximum-risk cap is subsequently enforced, followed by the tiny-balance minimum-lot safety gate.

### Replay invariant

The replay MUST record all of these separately:

- `balance_before`
- `risk_pct_requested`
- `risk_usd_requested`
- `sl_pips`
- `pip_value_per_lot`
- `base_lot_raw`
- `base_lot_normalized`
- every live multiplier and its source
- `final_lot_raw`
- `final_lot_normalized`
- `actual_risk_usd`
- `actual_risk_pct`
- `max_lot`
- `min_lot`
- rejection/cap reason, if any

**Do not replace this with a simple `risk_usd / SL` calculation in the backtest.** That would lose live PositionSizer behavior.

## 6. Permission parity

`risk/trade_permission.py::TradePermission.check()` is the final permission gate after risk/sizing. It contains gates including signal validity, risk approval, confidence, aligned factors, session quality, confluence, minimum R:R, SMC/session fusion, persistence, regime suppression, duplicate trade, correlation, and execution filters.

The replay must emit one trace record per gate:

```text
stage = TRADE_PERMISSION
name = <gate>
status = PASS | FAIL
reason = <exact live reason>
input_snapshot = <decision-time values only>
```

No replay-only gate may be inserted into this chain.

## 7. Execution Router parity

Live execution boundary: `execution/execution_router.py::ExecutionRouter`.

The replay MUST call the same router contract but inject only the allowed execution backend difference:

```text
LIVE:      ExecutionRouter → MT5
REPLAY:    ExecutionRouter-equivalent boundary → deterministic simulator
```

The simulator must never create or contact an MT5 session.

`execution/simulated_executor.py` is **not sufficient as the research simulator** because it currently fabricates an entry from SL/TP when no tick exists and uses wall-clock/random slippage. Those behaviors are unsuitable for a strict historical replay.

## 8. Historical fill contract

At decision timestamp `T`:

- entry must use only the historical quote available at/after the live execution boundary, according to the declared execution timing contract;
- no future candle may be consulted to manufacture an entry;
- bid/ask must be respected when available;
- spread must come from historical data when available;
- slippage must be an explicit replay assumption or a deterministic historical execution model;
- commission, contract size, pip value, digits, min/max lot and lot step must be explicit;
- if any required broker field is unavailable, mark it `ASSUMED` rather than silently substituting current/live data.

## 9. Position lifecycle

Every accepted execution must create a lifecycle record containing:

- `trade_id`
- symbol/direction
- signal timestamp
- requested entry
- filled entry
- requested/filled lot
- SL/TP at execution
- spread/slippage/commission/swap
- open timestamp
- close timestamp
- close price
- exit reason
- P/L USD
- P/L pips
- P/L R
- hold bars/time
- MAE/MFE
- time-to-MAE/MFE

Exit processing must happen in timestamp order and must use the declared intrabar policy. If OHLC cannot establish SL-vs-TP ordering, mark `AMBIGUOUS_INTRABAR`; do not invent ordering.

## 10. Critical current gap

The existing `backtest/broker_sim.py` is useful infrastructure, but it is **not yet strict live-execution parity** because it currently:

- samples random slippage;
- has probabilistic partial fills;
- has default/assumed spreads;
- resolves simultaneous SL/TP touches using a heuristic;
- can therefore produce a different execution path than a deterministic research replay.

These behaviors may remain available for separate broker-realism experiments, but they must not be silently used as the canonical live-mirroring replay.

## 11. Verdict

**P0 risk/execution parity: PARTIALLY PROVEN.**

PROVEN:

- live RiskEngine is identified;
- live PositionSizer is identified as an additional sizing layer;
- live TradePermission is identified;
- live ExecutionRouter is identified;
- structural SL and position-sizing formulas are source-verified;
- a broker simulator boundary exists.

NOT YET PROVEN:

- every post-RiskEngine lot mutation is captured in replay;
- every permission gate emits a canonical trace;
- the canonical replay execution backend is deterministic and historical-only;
- complete position lifecycle is wired to the same live execution call chain;
- live-vs-replay parity fixtures pass for risk, permission, execution and lifecycle.
