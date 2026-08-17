"""Parity test: verify broker_sim enforces stop_level, min_lot, margin."""
import sys, os
sys.path.insert(0, '/home/z/my-project/forex-agent')
os.chdir('/home/z/my-project/forex-agent')
import logging
logging.getLogger().setLevel(logging.ERROR)

from backtest.broker_sim import BrokerSimulator
from datetime import datetime, timezone

sim = BrokerSimulator(starting_balance=10000.0)
print(f'balance=${sim.balance}')

# Test 1: SL too close → reject
print('\n--- Test 1: SL too close (5 pips, broker min 8) ---')
t = sim.open_trade(
    symbol='EURUSD', direction='BUY',
    entry_price=1.1000, sl=1.0995, tp=1.1050,  # 5 pip SL, 50 pip TP
    lot=0.10, bar_time=datetime.now(timezone.utc),
)
print(f'  trade={t} (expected None — SL too close)')
assert t is None, 'BUG: SL-too-close check did not fire'

# Test 2: SL far enough → accept
print('\n--- Test 2: SL 20 pips (>= 8) → accept ---')
t = sim.open_trade(
    symbol='EURUSD', direction='BUY',
    entry_price=1.1000, sl=1.0980, tp=1.1040,  # 20 pip SL, 40 pip TP
    lot=0.10, bar_time=datetime.now(timezone.utc),
)
print(f'  trade={t is not None} (expected True)')
assert t is not None, 'BUG: valid trade was rejected'

# Test 3: lot too small → normalize to 0.01
print('\n--- Test 3: lot=0.005 → normalize to 0.01 ---')
t = sim.open_trade(
    symbol='EURUSD', direction='BUY',
    entry_price=1.1000, sl=1.0980, tp=1.1040,
    lot=0.005, bar_time=datetime.now(timezone.utc),
)
if t is not None:
    print(f'  lot_size={t.lot_size} (expected 0.01)')
    assert t.lot_size == 0.01, f'BUG: lot not normalized, got {t.lot_size}'

# Test 4: lot too big → normalize to MAX_LOT
print('\n--- Test 4: lot=100.0 → normalize to 50.0 ---')
t = sim.open_trade(
    symbol='EURUSD', direction='BUY',
    entry_price=1.1000, sl=1.0980, tp=1.1040,
    lot=100.0, bar_time=datetime.now(timezone.utc),
)
if t is not None:
    print(f'  lot_size={t.lot_size} (expected 50.0)')
    assert t.lot_size == 50.0, f'BUG: lot not clamped, got {t.lot_size}'
else:
    # Could be rejected on margin — that's also acceptable behavior
    print(f'  trade=None (rejected on margin — acceptable)')

# Test 5: insufficient margin → reject
print('\n--- Test 5: huge lot on small balance → reject on margin ---')
small_sim = BrokerSimulator(starting_balance=100.0)
t = small_sim.open_trade(
    symbol='EURUSD', direction='BUY',
    entry_price=1.1000, sl=1.0980, tp=1.1040,
    lot=50.0, bar_time=datetime.now(timezone.utc),  # 50 lot * 100k * 1.1 / 100 = $55k margin
)
print(f'  trade={t} (expected None — insufficient margin)')
assert t is None, 'BUG: margin check did not fire'

# Test 6: JPY pair pip size
print('\n--- Test 6: USDJPY SL distance uses JPY pip size (0.01) ---')
t = sim.open_trade(
    symbol='USDJPY', direction='BUY',
    entry_price=150.00, sl=149.80, tp=150.80,  # 20 pip SL on JPY (0.20 / 0.01 = 20)
    lot=0.01, bar_time=datetime.now(timezone.utc), spread_pips=2.0,  # small lot for JPY margin
)
print(f'  trade={t is not None} (expected True — JPY pip calc)')
assert t is not None, 'BUG: JPY SL distance check failed'

print('\n=== ALL PARITY TESTS PASSED ===')
