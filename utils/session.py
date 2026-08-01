# utils/session.py
# ============================================================
# Forex Market Session Awareness
# AI Trader কোন session-এ trade করছে সেটা জানবে
# ============================================================

from datetime import datetime, timezone, timedelta
from utils.logger import get_logger

log = get_logger(__name__)

# Session hours (UTC)
SESSIONS = {
    'sydney':   {'open': 21, 'close': 6,  'color': '🔵'},
    'tokyo':    {'open': 0,  'close': 9,  'color': '🟣'},
    'london':   {'open': 7,  'close': 16, 'color': '🟡'},
    'new_york': {'open': 12, 'close': 21, 'color': '🟠'},
}

# Highest volatility — session overlap
OVERLAPS = [
    {'name': 'Tokyo/London',    'start': 7,  'end': 9},
    {'name': 'London/New York', 'start': 12, 'end': 16},   # ← সবচেয়ে volatile
]


class SessionAnalyzer:

    def get_current_session(self, dt: datetime = None) -> dict:
        """
        এখন কোন forex session চলছে।
        Multiple sessions একসাথে active থাকতে পারে।
        """
        if dt is None:
            dt = datetime.now(timezone.utc)

        hour = dt.hour
        active = []

        for name, info in SESSIONS.items():
            o, c = info['open'], info['close']
            # Overnight session (e.g. sydney: 21–6)
            if o > c:
                is_open = hour >= o or hour < c
            else:
                is_open = o <= hour < c
            if is_open:
                active.append(name)

        overlap = self._get_overlap(hour)

        result = {
            'utc_time':       dt.strftime('%H:%M UTC'),
            'active_sessions': active,
            'overlap':        overlap,
            'volatility':     self._volatility(active, overlap),
            'trade_quality':  self._trade_quality(active, overlap),
        }
        return result

    def _get_overlap(self, hour: int) -> str | None:
        for ov in OVERLAPS:
            if ov['start'] <= hour < ov['end']:
                return ov['name']
        return None

    def _volatility(self, active: list, overlap: str | None) -> str:
        if overlap == 'London/New York':  return 'VERY HIGH 🔥'
        if overlap == 'Tokyo/London':     return 'HIGH ⚡'
        if 'london' in active:            return 'HIGH ⚡'
        if 'new_york' in active:          return 'MEDIUM-HIGH'
        if 'tokyo' in active:             return 'MEDIUM'
        return 'LOW 😴'

    def _trade_quality(self, active: list, overlap: str | None) -> str:
        if overlap == 'London/New York':  return '🟢 BEST — highest liquidity'
        if overlap == 'Tokyo/London':     return '🟢 GOOD — active market'
        if 'london' in active:            return '🟡 GOOD — major session'
        if 'new_york' in active:          return '🟡 GOOD — major session'
        if not active:                    return '🔴 AVOID — market closed'
        return '🟠 CAUTION — low liquidity'

    def print_session_info(self):
        info = self.get_current_session()
        print("\n" + "═" * 46)
        print("  🕐  MARKET SESSION INFO")
        print("═" * 46)
        print(f"  Time (UTC)   :  {info['utc_time']}")
        print(f"  Active       :  {', '.join(info['active_sessions']) or 'None (closed)'}")
        print(f"  Overlap      :  {info['overlap'] or '—'}")
        print(f"  Volatility   :  {info['volatility']}")
        print(f"  Trade Quality:  {info['trade_quality']}")
        print("═" * 46 + "\n")
        return info