"""CSV spread remediation policy.

Investigation findings:
- EURUSD H1 data (2025-07-25 to 2026-07-24):
  - 3746/6197 rows (60.4%) have spread=0
  - Timeline: 1.7% zero-spread in Jul 2025 → 100% zero-spread in Jul 2026
  - Transition began Dec 2025 (~60% zero-spread)
  - By Jan 2026, 95% zero-spread
  - Non-zero spreads range 1-38 pips (reasonable for EURUSD)

Likely root cause:
- MT5 `symbol_info_tick().spread` returns 0 when:
  1. Tick is stale (market closed, weekend, holiday)
  2. Data collection script polled at moment when spread was not yet populated
  3. Symbol info cache not refreshed
- This is a DATA COLLECTION BUG, not a broker behavior (live brokers
  always have non-zero spread on tradable symbols during market hours)

Remediation policy:
- DO NOT overwrite raw spread column
- Add new columns: raw_spread, spread_valid, spread_source, effective_spread
- For zero-spread rows, substitute synthetic spread based on:
  - Same symbol's historical non-zero median spread (per session)
  - Falls back to symbol-specific default (e.g. EURUSD=1.5 pips)
- effective_spread = raw_spread if raw_spread > 0 else synthetic_spread
- Track percentage of synthetic spread usage in backtest report
"""
import pandas as pd
import numpy as np
from pathlib import Path


# Symbol-specific default spreads (in pips) — from healthy historical period
SYMBOL_DEFAULT_SPREADS = {
    "EURUSD": 1.5,
    "GBPUSD": 2.0,
    "USDJPY": 1.5,
    "AUDUSD": 1.8,
    "USDCAD": 2.0,
    "USDCHF": 2.0,
    "NZDUSD": 2.0,
    "XAUUSD": 25.0,
}


def derive_session(timestamp_utc):
    """Derive trading session from UTC timestamp.
    Sessions (UTC, approx):
    - Asia: 00:00-07:00 (Tokyo 00-09, Sydney 22-07)
    - London: 07:00-16:00
    - New York: 12:00-21:00
    - Overlap London/NY: 12:00-16:00
    - Off-hours: 21:00-00:00
    """
    h = timestamp_utc.hour
    if 0 <= h < 7:
        return "asia"
    elif 7 <= h < 12:
        return "london"
    elif 12 <= h < 16:
        return "london_ny_overlap"
    elif 16 <= h < 21:
        return "new_york"
    else:
        return "off_hours"


def remediate_spread(df, symbol="EURUSD"):
    """Add raw_spread, spread_valid, spread_source, effective_spread columns.
    Does NOT modify the original 'spread' column.

    Args:
        df: DataFrame with 'spread' column (in points/pips, MT5 format)
        symbol: symbol name for default spread lookup

    Returns:
        df with new columns added
    """
    df = df.copy()
    if 'spread' not in df.columns:
        return df

    # Preserve raw spread
    df['raw_spread'] = df['spread']

    # Determine validity
    df['spread_valid'] = df['spread'] > 0

    # Compute session-based median spread from VALID rows only
    df['session'] = df.index.map(derive_session)
    valid = df[df['spread_valid']]
    if len(valid) > 0:
        session_medians = valid.groupby('session')['spread'].median().to_dict()
    else:
        session_medians = {}

    # Default spread for this symbol
    default_spread = SYMBOL_DEFAULT_SPREADS.get(symbol, 2.0)

    # Build effective_spread column
    effective = []
    sources = []
    for idx, row in df.iterrows():
        if row['spread_valid']:
            effective.append(row['spread'])
            sources.append('raw')
        else:
            # Try session-based median first
            session = row['session']
            session_median = session_medians.get(session)
            if session_median and session_median > 0:
                effective.append(session_median)
                sources.append(f'synthetic_session_median({session})')
            else:
                # Fall back to symbol default
                effective.append(default_spread)
                sources.append(f'synthetic_default({symbol}={default_spread})')

    df['effective_spread'] = effective
    df['spread_source'] = sources

    return df


def audit_spread_remediation(df, symbol="EURUSD"):
    """Return stats on spread remediation."""
    if 'spread_source' not in df.columns:
        return None
    total = len(df)
    raw_count = (df['spread_source'] == 'raw').sum()
    synthetic_count = total - raw_count
    return {
        'symbol': symbol,
        'total_rows': total,
        'raw_spread_rows': int(raw_count),
        'synthetic_spread_rows': int(synthetic_count),
        'synthetic_pct': synthetic_count / total * 100 if total else 0,
        'session_breakdown': {f'{k[0]}|{k[1]}': int(v) for k, v in df.groupby(['session', 'spread_source']).size().items()},
    }


if __name__ == '__main__':
    import sys, os, json
    sys.path.insert(0, '/home/z/my-project/forex-agent')
    os.chdir('/home/z/my-project/forex-agent')

    pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF']
    results = {}
    for pair in pairs:
        path = f'data/{pair}_H1.csv'
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, parse_dates=['datetime_utc']).set_index('datetime_utc')
        df = remediate_spread(df, symbol=pair)
        stats = audit_spread_remediation(df, symbol=pair)
        results[pair] = stats
        print(f'{pair}: {stats["synthetic_pct"]:.1f}% synthetic spread '
              f'({stats["synthetic_spread_rows"]}/{stats["total_rows"]} rows)')

    # Save
    os.makedirs('/home/z/my-project/download', exist_ok=True)
    with open('/home/z/my-project/download/spread_remediation_audit.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\nSaved: /home/z/my-project/download/spread_remediation_audit.json')
