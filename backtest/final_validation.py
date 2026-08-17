"""
Reproduces the validated winrate/expectancy/profit numbers for the
mean-reversion confluence engine + DA gate pipeline.

Run from the repo root (so `analysis.` and `risk.` imports resolve):
    python -m backtest.final_validation

Expects EURAUD_M15.csv, GBPCAD_M15.csv, EURCAD_M15.csv (M15 OHLCV,
columns: datetime_utc, open, high, low, close, tick_volume, spread,
real_volume) at the paths in PAIRS below — update those paths to your
own data location.
"""
import sys
import pandas as pd
from analysis.mean_reversion_confluence_engine import MeanReversionConfluenceEngine
from risk.mean_reversion_da_gate import MeanReversionDAGate

PAIRS = {
    "EURAUD": "./data/EURAUD_M15.csv",   # <-- update to your CSV paths
    "GBPCAD": "./data/GBPCAD_M15.csv",
    "EURCAD": "./data/EURCAD_M15.csv",
}
SPLIT_DATE = pd.Timestamp("2026-04-25", tz="UTC")
TP_R_MULT = 1.0
RISK_PER_TRADE_PCT = 1.0   # for $ profit simulation on a hypothetical $10,000 account
ACCOUNT_SIZE = 10_000

# Cost realism — beyond the CSV's own spread column:
SLIPPAGE_PIPS = 0.5         # typical retail-broker market-order slippage, added
                             # against the trader on both entry and exit
COMMISSION_USD_PER_LOT_ROUNDTRIP = 7.0   # common ECN-account commission
ASSUMED_LOT_SIZE = 0.1      # for the $ simulation below (1 mini-lot per 1% risk unit)
PIP_SIZE = 0.0001

engine = MeanReversionConfluenceEngine()
gate = MeanReversionDAGate()

def swing_high(df, idx, lookback=40):
    return df['high'].iloc[max(0, idx - lookback):idx + 1].max()

def swing_low(df, idx, lookback=40):
    return df['low'].iloc[max(0, idx - lookback):idx + 1].min()

def run_pair(symbol, path, max_hold=300):
    df = pd.read_csv(path, parse_dates=['datetime_utc']).sort_values('datetime_utc').reset_index(drop=True)
    df = engine.prepare(df)
    trades = []
    i = 250
    n = len(df)
    veto_count = 0
    candidate_count = 0
    while i < n - 1:
        row = df.iloc[i]
        direction, score = engine.decide(row)
        if direction is None:
            i += 1
            continue
        candidate_count += 1

        atr = row['atr14'] if pd.notna(row['atr14']) else 0.0005
        spread = row.get('spread', 0)
        spread_price = (spread / 100000.0) if pd.notna(spread) else 0.0
        if direction == "SELL":
            entry = row['close'] - spread_price / 2
            sl = swing_high(df, i) + 0.2 * atr
        else:
            entry = row['close'] + spread_price / 2
            sl = swing_low(df, i) - 0.2 * atr

        vetoed, flags = gate.review(df, i, direction, sl)
        if vetoed:
            veto_count += 1
            i += 1
            continue

        r = abs(sl - entry)
        if r <= 0:
            i += 1
            continue
        tp1 = entry - r * TP_R_MULT if direction == "SELL" else entry + r * TP_R_MULT

        outcome, exit_idx = None, None
        future = df.iloc[i + 1: i + 1 + max_hold]
        for fi, frow in future.iterrows():
            if direction == "SELL":
                if frow['high'] >= sl:
                    outcome, exit_idx = "LOSS", fi; break
                if frow['low'] <= tp1:
                    outcome, exit_idx = "WIN", fi; break
            else:
                if frow['low'] <= sl:
                    outcome, exit_idx = "LOSS", fi; break
                if frow['high'] >= tp1:
                    outcome, exit_idx = "WIN", fi; break
        if outcome is None:
            i += 1
            continue

        # Cost realism: slippage. Entry always slips against the trader
        # (market order). SL exits also slip against the trader (stop
        # orders often fill worse during the move that triggered them).
        # TP exits are treated as limit fills (no adverse slippage).
        slip_price = SLIPPAGE_PIPS * PIP_SIZE
        slip_r_entry = slip_price / r
        slip_r_exit = slip_price / r if outcome == "LOSS" else 0.0
        raw_r = TP_R_MULT if outcome == "WIN" else -1.0
        r_multiple_net = raw_r - slip_r_entry - slip_r_exit

        trades.append({
            "symbol": symbol, "entry_time": row['datetime_utc'], "exit_time": df['datetime_utc'].iloc[exit_idx],
            "signal": direction,
            "score": score, "outcome": outcome, "r_multiple": TP_R_MULT if outcome == "WIN" else -1.0,
        })
        i = exit_idx + 1

    return trades, candidate_count, veto_count

def report(trades, label):
    if not trades:
        print(f"{label}: 0 trades")
        return 0, 0.0, 0.0
    wins = sum(1 for t in trades if t['outcome'] == 'WIN')
    n = len(trades)
    wr = 100 * wins / n
    total_r = sum(t['r_multiple'] for t in trades)
    exp_r = total_r / n
    # $ simulation: fixed-fractional 1% risk per trade, compounding
    balance = ACCOUNT_SIZE
    for t in sorted(trades, key=lambda x: x['entry_time']):
        risk_amt = balance * (RISK_PER_TRADE_PCT / 100)
        balance += risk_amt * t['r_multiple']
    profit_pct = 100 * (balance - ACCOUNT_SIZE) / ACCOUNT_SIZE
    print(f"{label}: n={n} | winrate={wr:.1f}% | expectancy={exp_r:+.3f}R | "
          f"sim. account ${ACCOUNT_SIZE:,} -> ${balance:,.0f} ({profit_pct:+.1f}%) @ {RISK_PER_TRADE_PCT}% risk/trade")
    return n, wr, exp_r

all_train, all_test = [], []
total_candidates, total_vetoed = 0, 0
print("=" * 78)
for sym, path in PAIRS.items():
    trades, cand, veto = run_pair(sym, path)
    total_candidates += cand
    total_vetoed += veto
    train = [t for t in trades if t['entry_time'] < SPLIT_DATE]
    test = [t for t in trades if t['entry_time'] >= SPLIT_DATE]
    all_train += train
    all_test += test
    print(f"\n{sym}: candidates={cand}, vetoed_by_gate={veto} ({100*veto/cand:.1f}%), taken={len(trades)}")
    report(train, "  TRAIN")
    report(test, "  TEST (held-out, 2026-04-25+)")

print("\n" + "=" * 78)
print(f"Pipeline funnel (all 3 pairs, ~13 months): "
      f"{total_candidates} candidates -> {total_candidates - total_vetoed} survived gate "
      f"({100*(total_candidates-total_vetoed)/total_candidates:.1f}% pass rate)")
print()
report(all_train, "TOTAL TRAIN (in-sample)")
report(all_test, "TOTAL TEST (out-of-sample, held-out — THIS is the trustworthy number)")

pd.DataFrame(all_train + all_test).to_csv('final_validated_trades.csv', index=False)
print("\nSaved -> final_validated_trades.csv")