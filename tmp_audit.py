import json
from pathlib import Path
import pandas as pd
from scripts.trading_engine_audit import analyze_strategy_signals

p = Path('data/EURUSD_H1.csv')
if not p.exists():
    print('CSV not found:', p)
else:
    df = pd.read_csv(p, parse_dates=['datetime_utc'], index_col='datetime_utc')
    res = analyze_strategy_signals(df, 'EURUSD')
    print(json.dumps(res, indent=2))
