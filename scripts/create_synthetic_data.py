import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone

OUT = Path(__file__).resolve().parent.parent / 'data' / 'history' / 'EURUSD'
OUT.mkdir(parents=True, exist_ok=True)

n = 2000
end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
start = end - timedelta(minutes=15*(n-1))

timestamps = [start + timedelta(minutes=15*i) for i in range(n)]
price = 1.1000
prices = []
for _ in range(n):
    price += np.random.normal(0, 0.0005)
    prices.append(price)

open_p = prices
close_p = [p + np.random.normal(0, 0.0002) for p in prices]
high_p = [max(o,c) + abs(np.random.normal(0, 0.0003)) for o,c in zip(open_p, close_p)]
low_p = [min(o,c) - abs(np.random.normal(0, 0.0003)) for o,c in zip(open_p, close_p)]
vol = np.random.randint(100, 1000, size=n)

df = pd.DataFrame({
    'timestamp': pd.to_datetime(timestamps),
    'open': open_p,
    'high': high_p,
    'low': low_p,
    'close': close_p,
    'volume': vol,
})

path = OUT / 'EURUSD_M15.parquet'
df.to_parquet(path, index=False)
print('Wrote', path, 'rows=', len(df))
