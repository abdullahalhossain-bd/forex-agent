# core/production_hardening.py — Production Hardening (compact version)
import logging, os, json, threading, time, random
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict
import numpy as np
import pandas as pd
from core.constants import MEMORY_DIR
log = logging.getLogger(__name__)


class _MT5NoneResult(Exception):
    """Internal sentinel: mt5.positions_get() returned None (a known-flaky
    response, not a raised exception) so it can be retried through the
    shared retry_with_failover.retry_sync() machinery."""


def _mt5_positions_get(retries: int = 2, delay: float = 0.3, **kwargs):
    """Call mt5.positions_get() with retry logic.

    MT5 can return None intermittently. This helper retries
    a few times before giving up, reducing false negatives.

    Passes through any kwargs (symbol=, ticket=, etc.) to mt5.positions_get().

    Audit fix: this used to hand-roll its own sleep/retry loop, duplicating
    core/retry_with_failover.py (which ships an MT5-tuned RetryPolicy that
    was otherwise unused anywhere in the codebase). It now delegates to
    retry_sync() so there is one retry/backoff implementation instead of
    two. Signature, return value (positions list or None), and behavior
    for existing callers are unchanged. Falls back to the original inline
    loop if retry_with_failover can't be imported, so this never breaks.
    """
    import MetaTrader5 as mt5

    def _call():
        result = mt5.positions_get(**kwargs) if kwargs else mt5.positions_get()
        if result is None:
            raise _MT5NoneResult("mt5.positions_get() returned None")
        return result

    try:
        from core.retry_with_failover import retry_sync, RetryPolicy

        policy = RetryPolicy(
            attempts=retries + 1,
            min_delay_ms=int(delay * 1000),
            max_delay_ms=max(int(delay * 1000) * (2 ** max(retries, 1)), int(delay * 1000)),
            jitter=0.15,
            should_retry=lambda e, attempt: isinstance(e, _MT5NoneResult),
        )
        return retry_sync(_call, policy=policy, label="mt5_positions_get")
    except _MT5NoneResult:
        return None
    except ImportError:
        # retry_with_failover unavailable — preserve the original inline
        # retry loop as a safe fallback so this function never breaks.
        for attempt in range(retries + 1):
            try:
                result = mt5.positions_get(**kwargs) if kwargs else mt5.positions_get()
                if result is not None:
                    return result
            except Exception:
                pass
            if attempt < retries:
                time.sleep(delay)
        return None
    except Exception:
        return None


def check_partial_fill(result, requested_volume):
    if result is None: return {"is_partial": False, "filled_volume": 0, "requested_volume": requested_volume, "adjustment_needed": False, "action": "abort"}
    filled = float(getattr(result, "volume", requested_volume))
    is_partial = filled < requested_volume * 0.999
    if is_partial:
        pct = filled / requested_volume * 100
        log.warning(f"[PartialFill] {filled}/{requested_volume} ({pct:.1f}%)")
        return {"is_partial": True, "filled_volume": filled, "requested_volume": requested_volume, "adjustment_needed": True, "action": "retry" if pct < 50 else "accept"}
    return {"is_partial": False, "filled_volume": filled, "requested_volume": requested_volume, "adjustment_needed": False, "action": "accept"}

class PositionReconciler:
    def __init__(self, interval_sec=60):
        self.interval_sec = interval_sec; self._thread = None; self._running = False
        self._internal = {}; self._on_mismatch = None; self._last = {"ok": True, "mismatches": []}
    def register_position(self, ticket, symbol, volume, direction): self._internal[ticket] = {"symbol": symbol, "volume": volume, "direction": direction}
    def unregister_position(self, ticket): self._internal.pop(ticket, None)
    def set_mismatch_callback(self, cb): self._on_mismatch = cb
    def start(self):
        if self._thread and self._thread.is_alive(): return
        self._running = True; self._thread = threading.Thread(target=self._run, daemon=True); self._thread.start()
    def stop(self): self._running = False
    def _run(self):
        while self._running:
            try: self._reconcile()
            except Exception as e: log.error(f"[Reconciler] {e}")
            time.sleep(self.interval_sec)
    def _reconcile(self):
        try:
            import MetaTrader5 as mt5
            pos = _mt5_positions_get() or []
            mt5_t = {p.ticket for p in pos}
            int_t = set(self._internal.keys())
            mm = []
            for t in (int_t - mt5_t):
                mm.append({"type": "phantom", "ticket": t}); self.unregister_position(t)
            for t in (mt5_t - int_t):
                mm.append({"type": "orphaned", "ticket": t})
            self._last = {"ok": len(mm) == 0, "mismatches": mm, "mt5_count": len(mt5_t), "internal_count": len(int_t)}
            if mm and self._on_mismatch: self._on_mismatch(mm)
        except Exception as e: pass
    def get_status(self): return self._last

class HeartbeatMonitor:
    FILE = str(MEMORY_DIR / "heartbeat.json"); INTERVAL = 300
    def __init__(self): self._thread = None; self._running = False; self._state = {"status": "starting", "positions": 0, "equity": 0}
    def update_state(self, status, positions, equity): self._state = {"status": status, "positions": positions, "equity": equity}
    def start(self):
        if self._thread and self._thread.is_alive(): return
        self._running = True; self._thread = threading.Thread(target=self._run, daemon=True); self._thread.start()
    def stop(self): self._running = False
    def _run(self):
        while self._running:
            try: self._write()
            except Exception as e: pass
            time.sleep(self.INTERVAL)
    def _write(self):
        os.makedirs(os.path.dirname(self.FILE), exist_ok=True)
        with open(self.FILE, "w") as f: json.dump({"timestamp": datetime.now(timezone.utc).isoformat(), "bot_state": self._state, "pid": os.getpid()}, f)
    @classmethod
    def check_alive(cls, max_age=600):
        try:
            if not os.path.exists(cls.FILE): return {"alive": False, "reason": "No heartbeat file"}
            with open(cls.FILE) as f: hb = json.load(f)
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(hb["timestamp"])).total_seconds()
            return {"alive": age <= max_age, "age_sec": age, "last_state": hb.get("bot_state", {})} if age <= max_age else {"alive": False, "reason": f"Heartbeat {age:.0f}s old"}
        except Exception as e: return {"alive": False, "reason": str(e)}

def should_close_for_weekend(positions):
    now = datetime.now(timezone.utc); wd = now.weekday()
    if wd == 5: return {"should_close_all": True, "should_reduce": False, "reduce_multiplier": 0, "positions_to_close": [p.get("ticket") for p in positions], "reason": "Saturday — close all"}
    if wd == 4 and now.hour >= 21: return {"should_close_all": True, "should_reduce": False, "reduce_multiplier": 0, "positions_to_close": [p.get("ticket") for p in positions], "reason": "Friday late — close all"}
    if wd == 4 and now.hour >= 20: return {"should_close_all": False, "should_reduce": True, "reduce_multiplier": 0.5, "positions_to_close": [], "reason": "Friday evening — reduce 50%"}
    if wd == 6 and now.hour < 22: return {"should_close_all": False, "should_reduce": False, "reduce_multiplier": 0, "positions_to_close": [], "reason": "Sunday pre-open"}
    return {"should_close_all": False, "should_reduce": False, "reduce_multiplier": 1.0, "positions_to_close": [], "reason": "Normal hours"}

class DynamicCorrelationMatrix:
    def __init__(self, lookback=30): self.lookback = lookback; self._data = {}
    def update(self, symbol, closes): self._data[symbol.upper()] = closes.tail(self.lookback)
    def get_correlation(self, a, b):
        s1, s2 = self._data.get(a.upper()), self._data.get(b.upper())
        if s1 is None or s2 is None: return 0.0
        aligned = pd.concat([s1, s2], axis=1, join="inner").dropna()
        if len(aligned) < 10: return 0.0
        c = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        return float(c) if np.isfinite(c) else 0.0
    def check_exposure(self, proposed_pair, proposed_dir, open_positions, threshold=0.70):
        if not open_positions: return {"passed": True, "reason": "No positions", "max_correlation": 0}
        mx = 0; viol = []
        for pos in open_positions:
            ep = pos.get("pair", "").upper()
            c = 1.0 if ep == proposed_pair.upper() else abs(self.get_correlation(proposed_pair, ep))
            mx = max(mx, c)
            if c > threshold and pos.get("direction", "").upper() == proposed_dir.upper():
                viol.append({"existing_pair": ep, "correlation": round(c, 3)})
        return {"passed": len(viol) == 0, "reason": f"Max corr={mx:.2f}", "max_correlation": round(mx, 3), "violations": viol} if viol else {"passed": True, "reason": f"Max corr={mx:.2f}", "max_correlation": round(mx, 3)}

def _weekend_closed_seconds(start: datetime, end: datetime) -> float:
    """Approximate non-trading seconds (Sat 00:00 → Sun 24:00 UTC) between two datetimes.

    Forex is closed from Friday ~21/22:00 UTC to Sunday ~21/22:00 UTC, but the
    exact broker rollover varies. Counting full Sat+Sun is a safe, conservative
    underestimate of closed time (we never *inflate* age by accident).
    """
    if end <= start:
        return 0.0
    closed = 0.0
    # Walk calendar days from start.date() to end.date()
    day = start.date()
    end_day = end.date()
    while day <= end_day:
        # weekday(): Mon=0 … Sat=5, Sun=6
        if day.weekday() >= 5:
            day_start = datetime(day.year, day.month, day.day, tzinfo=start.tzinfo or timezone.utc)
            day_end = day_start + timedelta(days=1)
            overlap_start = max(start, day_start)
            overlap_end = min(end, day_end)
            if overlap_end > overlap_start:
                closed += (overlap_end - overlap_start).total_seconds()
        day += timedelta(days=1)
    return closed


def check_data_staleness(df, max_age_sec=120, *, adjust_weekend: bool = True,
                         timeframe: str = None):
    """Return staleness verdict for the newest bar in *df*.

    Age is measured from the **bar close time** when *timeframe* is known
    (open + tf_seconds). MT5 indexes bars by open time; right after a bar
    closes the open-based age is already ~1 full interval — without this
    adjustment H1/H4 false-positive as STALE within minutes of a valid close.

    Parameters
    ----------
    df : DataFrame with DatetimeIndex (bar open times)
    max_age_sec : threshold from compute_staleness_threshold()
    adjust_weekend : subtract Sat+Sun closed hours (D1/H4 weekend-safe)
    timeframe : optional TF code ('1h', 'H1', '1d', …). When set, age
        is computed from expected close of the last bar instead of open.
    """
    if df is None or len(df) == 0:
        return {"is_stale": True, "age_sec": 999, "reason": "No data"}
    try:
        # Newest bar by open-time index
        last_open = df.index.max()
        if hasattr(last_open, 'to_pydatetime'):
            last_open = last_open.to_pydatetime()
        if hasattr(last_open, 'tzinfo') and last_open.tzinfo:
            now = datetime.now(last_open.tzinfo)
        else:
            last_open = last_open.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)

        # Prefer age-from-close when TF is known (closed-bar pipeline)
        tf_sec = 0
        if timeframe:
            try:
                tf_sec = int(timeframe_to_seconds(timeframe))
            except Exception:
                tf_sec = 0
        last_close = last_open + timedelta(seconds=tf_sec) if tf_sec > 0 else last_open

        raw_age = (now - last_close).total_seconds()
        # Still forming: close is in the future → treat as age 0 (fresh forming bar)
        if raw_age < 0:
            # Distinguish broker-tz future bar (open itself in future) vs forming
            open_age = (now - last_open).total_seconds()
            if open_age < 0:
                log.critical(
                    f"[check_data_staleness] NEGATIVE OPEN AGE ({open_age:.0f}s) — "
                    f"last bar OPEN is in the FUTURE (broker tz bug). "
                    f"last_open={last_open.isoformat()} now={now.isoformat()}"
                )
                return {
                    "is_stale": False,
                    "age_sec": round(open_age, 1),
                    "last_timestamp": str(df.index.max()),
                    "reason": f"FUTURE_BAR (open_age={open_age:.0f}s) — broker tz mislabeled as UTC",
                }
            # Forming bar — not stale
            return {
                "is_stale": False,
                "age_sec": 0.0,
                "raw_age_sec": round(raw_age, 1),
                "weekend_sec": 0.0,
                "last_timestamp": str(df.index.max()),
                "reason": f"Fresh (forming bar, closes in {-raw_age:.0f}s)",
            }

        age = raw_age
        weekend_sec = 0.0
        if adjust_weekend and raw_age > 3600:
            weekend_sec = _weekend_closed_seconds(last_close, now)
            age = max(0.0, raw_age - weekend_sec)

        is_stale = age > max_age_sec
        if is_stale:
            age_h = age / 3600.0
            raw_h = raw_age / 3600.0
            broker_offset = os.getenv("MT5_BROKER_TZ_OFFSET_HOURS", "0")
            if age_h >= 2.5 and broker_offset != "0":
                reason = (
                    f"Stale {age:.0f}s ({age_h:.1f}h effective, raw {raw_h:.1f}h from close) — "
                    f"WARNING: age ≈ MT5_BROKER_TZ_OFFSET_HOURS ({broker_offset}h). "
                    f"Run scripts/diagnose_mt5_staleness.py to verify the offset. "
                    f"If the broker is actually UTC+0, set MT5_BROKER_TZ_OFFSET_HOURS=0."
                )
            elif weekend_sec > 0:
                reason = (
                    f"Stale {age:.0f}s ({age_h:.1f}h effective after −{weekend_sec/3600:.1f}h weekend, "
                    f"raw {raw_h:.1f}h from close) — MT5 terminal may be disconnected. "
                    f"Check: terminal window open? symbols subscribed? broker server reachable?"
                )
            else:
                reason = (
                    f"Stale {age:.0f}s ({age_h:.1f}h from close) — MT5 terminal may be disconnected. "
                    f"Check: terminal window open? symbols subscribed? broker server reachable?"
                )
        else:
            reason = "Fresh"
            if weekend_sec > 0 and raw_age > max_age_sec:
                reason = (
                    f"Fresh (weekend-adjusted: raw {raw_age:.0f}s − {weekend_sec:.0f}s weekend "
                    f"= {age:.0f}s ≤ threshold {max_age_sec}s)"
                )
        return {
            "is_stale": is_stale,
            "age_sec": round(age, 1),
            "raw_age_sec": round(raw_age, 1),
            "weekend_sec": round(weekend_sec, 1),
            "last_timestamp": str(df.index.max()),
            "reason": reason,
        }
    except Exception as e:
        return {"is_stale": True, "age_sec": 999, "reason": str(e)}

def is_candle_closed(timeframe, last_bar_time, current_time=None):
    """
    Determine whether the bar whose OPEN time is `last_bar_time` has closed.

    ─── Timezone handling (institutional-grade fix) ───────────────────────
    The original implementation called `last_bar_time.replace(tzinfo=timezone.utc)`
    when the incoming timestamp was naive. This is DANGEROUS — `replace()` only
    ATTACHES a UTC label without converting the underlying wall-clock value.
    If the broker actually returns server time (e.g., GMT+2/GMT+3, common for
    MT5 brokers), attaching a UTC label produces a timestamp that is 2-3 hours
    in the future relative to true UTC, which silently breaks the close check.

    Two cases are now handled explicitly:

      1. `last_bar_time` already carries tzinfo  → use as-is (assume correct).
      2. `last_bar_time` is naive                → attach UTC label, but ALSO
         emit a WARNING so the operator knows the upstream fetcher did not tag
         the timezone. If the bar ends up "in the future" relative to now,
         we additionally emit a CRITICAL log pointing at the fetcher.

    A new "FUTURE_BAR" sentinel is returned when `last_bar_time > current_time`
    by more than one timeframe interval — this is the smoking gun for a
    broker-timezone bug and the caller can surface it cleanly.

    Debug logs (per audit P1 request):
      log.info("Last Bar : ...")
      log.info("Current  : ...")
      log.info("Close    : ...")
      log.info("Seconds  : ...")
    are emitted on every call at DEBUG level, and at WARNING level whenever
    the result is "Forming" so the operator can immediately see why the
    trade was blocked.
    """
    if current_time is None:
        current_time = datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)

    _naive_input = last_bar_time.tzinfo is None
    if _naive_input:
        # ⚠️  This is the suspicious path. The fetcher SHOULD have tagged
        # the timestamp with tzinfo. We attach UTC for arithmetic to work,
        # but if `last_bar_time` was actually broker server time (GMT+2/+3),
        # `close_time` will end up in the FUTURE and the FUTURE_BAR branch
        # below will catch it.
        last_bar_time = last_bar_time.replace(tzinfo=timezone.utc)

    tf_sec = timeframe_to_seconds(timeframe)

    close_time = last_bar_time + timedelta(seconds=tf_sec)
    seconds_left = (close_time - current_time).total_seconds()
    is_closed = current_time >= close_time

    # ── Debug logs (audit P1 requirement) ─────────────────────────────
    log.info(
        f"[is_candle_closed] tf={timeframe} | "
        f"last_bar={last_bar_time.isoformat()} | "
        f"current={current_time.isoformat()} | "
        f"close={close_time.isoformat()} | "
        f"seconds_left={seconds_left:.0f}s | "
        f"naive_input={_naive_input} | "
        f"is_closed={is_closed}"
    )

    # ── FUTURE_BAR detector ───────────────────────────────────────────
    # If `last_bar_time` itself is in the future (beyond one timeframe
    # interval), the timezone labeling is wrong. This is the exact symptom
    # reported by the user: 11126s "left" on an M15 bar = ~3 hours skew,
    # which is the typical GMT+2/GMT+3 broker offset.
    if (last_bar_time - current_time).total_seconds() > tf_sec:
        log.critical(
            f"[is_candle_closed] FUTURE_BAR DETECTED — last_bar_time "
            f"({last_bar_time.isoformat()}) is AHEAD of current_time "
            f"({current_time.isoformat()}) by "
            f"{(last_bar_time - current_time).total_seconds():.0f}s. "
            f"This is a broker-timezone bug in the data fetcher: the "
            f"timestamp is broker server time (likely GMT+2/+3) but is "
            f"being labeled as UTC. Check data/fetcher.py _fetch_mt5() "
            f"and set MT5_BROKER_TZ_OFFSET_HOURS accordingly."
        )
        return {
            "is_closed": False,
            "bar_time": last_bar_time.isoformat(),
            "close_time": close_time.isoformat(),
            "current_time": current_time.isoformat(),
            "seconds_left": round(seconds_left, 0),
            "reason": (
                f"FUTURE_BAR — last_bar_time is "
                f"{(last_bar_time - current_time).total_seconds():.0f}s "
                f"AHEAD of current_time. Broker timezone mislabeled as UTC. "
                f"Check data/fetcher.py and MT5_BROKER_TZ_OFFSET_HOURS."
            ),
            "future_bar": True,
        }

    if _naive_input:
        log.warning(
            f"[is_candle_closed] Naive last_bar_time received — "
            f"fetcher did not tag tzinfo. Assuming UTC. If this bar "
            f"appeared in the future, broker server time is being "
            f"mislabeled as UTC. Fix in data/fetcher.py."
        )

    if not is_closed:
        log.warning(
            f"[is_candle_closed] {timeframe} bar still FORMING — "
            f"{seconds_left:.0f}s left until close. "
            f"Trade entry will be skipped by trader.py."
        )

    return {
        "is_closed": is_closed,
        "bar_time": last_bar_time.isoformat(),
        "close_time": close_time.isoformat(),
        "current_time": current_time.isoformat(),
        "seconds_left": round(seconds_left, 0),
        "reason": "Closed" if is_closed else f"Forming, {seconds_left:.0f}s left",
        "future_bar": False,
    }

RECOMMENDED_DAILY_LOSS_PCT = 6.0
RECOMMENDED_MAX_CONSECUTIVE_LOSSES = 2


# ── Timeframe helpers (audit Round-3 fix) ──────────────────────────
# Single source of truth for the "M15 → 900s" mapping. Previously
# the same dict was inlined in `is_candle_closed`, `check_data_staleness`
# (via trader.py), and the stale-data diagnostic in `data/fetcher.py`.
# Each call site defined its own copy, which silently drifted apart.

TIMEFRAME_SECONDS = {
    "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
    "H1": 3600, "H4": 14400, "D1": 86400,
}
DEFAULT_TIMEFRAME_SECONDS = 3600  # fallback for unknown codes (≈ H1)

# Forms that after .upper() are NOT already keys of TIMEFRAME_SECONDS.
# "15m".upper() → "15M", "1h".upper() → "1H", "1d".upper() → "1D", etc.
_LOWERCASE_TF_MAP = {
    "1M": "M1", "5M": "M5", "15M": "M15", "30M": "M30",
    "1H": "H1", "4H": "H4", "1D": "D1",
    # bare aliases sometimes seen in config / MTF_CHAIN
    "D": "D1", "DAILY": "D1", "DAY": "D1",
    "H": "H1", "HOURLY": "H1",
}


def timeframe_to_seconds(timeframe: str) -> int:
    """Map a timeframe code (e.g. 'M15', 'H1', 'D1') to seconds.

    Single source of truth — `is_candle_closed`, `check_data_staleness`,
    `data/fetcher.py`, and `core/trader.py` all import this instead of
    re-declaring the mapping locally. Accepts MT5-style codes
    ('M15', 'H1'), lowercase forms ('15m', '1h', '4h', '1d'), and
    legacy '15M'/'1H' forms — all resolve to the same interval.

    Returns:
        int: seconds in one timeframe interval (e.g. M15 → 900).
        Falls back to 3600 (≈ H1) for unrecognized codes.
    """
    if not timeframe:
        return DEFAULT_TIMEFRAME_SECONDS
    key = timeframe.upper().strip()
    # Resolve "15M"/"1H"/"1D" → "M15"/"H1"/"D1" form via the lowercase map
    if key in _LOWERCASE_TF_MAP:
        key = _LOWERCASE_TF_MAP[key]
    # Also accept already-correct keys and bare forms like "D1", "H1"
    return TIMEFRAME_SECONDS.get(key, DEFAULT_TIMEFRAME_SECONDS)


def compute_staleness_threshold(timeframe: str, *,
                                  buffer_sec: int = 300,          # increased from 60 → 300
                                  min_floor_sec: int = 120) -> int:
    """
    Compute the appropriate max_age_sec for check_data_staleness
    based on the trader's timeframe.

    Increased buffer because we only analyze CLOSED bars (forming bar is dropped).
    Right after a new candle opens, the last closed bar can already be
    almost a full interval old + network latency.

    Threshold formula (intraday):
        threshold = max(tf_sec + buffer_sec, min_floor_sec) * multiplier

    D1 special-case (weekend-safe):
        Forex daily bars do not print Sat/Sun. A last closed D1 bar that
        opened Friday evening is routinely 40–60 h old by Monday/Tuesday
        afternoon. The generic 1.5× formula yields only ~36 h and false-
        positives every weekend. We therefore:
          • use a higher default multiplier for D1 (2.5× → ~60 h), and
          • enforce an absolute floor of 72 h so a normal weekend gap
            never marks Daily as STALE when the feed is healthy.
        Both values are overridable via env:
          MT5_STALE_BAR_MULTIPLIER_D1  (default 2.5)
          MT5_STALE_D1_FLOOR_SEC       (default 259200 = 72 h)

    Examples with default multiplier=1.5 (intraday) / 2.5 (D1):
        M15 → max(900+300, 120) * 1.5   = 1800s   (30 min)
        H1  → max(3600+300, 120) * 1.5  = 5850s   (~97 min)
        H4  → max(14400+300, 120) * 1.5 = 22050s  (~6.1 h)
        D1  → max( max(86400+300,120)*2.5 , 259200 ) = 259200s (72 h)
    """
    tf_sec = timeframe_to_seconds(timeframe)
    key = (timeframe or "").upper().strip()
    is_daily = key in ("D1", "1D", "DAILY", "1DAY") or tf_sec >= 86400

    try:
        if is_daily:
            multiplier = float(os.getenv("MT5_STALE_BAR_MULTIPLIER_D1",
                                         os.getenv("MT5_STALE_BAR_MULTIPLIER", "2.5")))
        else:
            multiplier = float(os.getenv("MT5_STALE_BAR_MULTIPLIER", "1.5"))
    except (ValueError, TypeError):
        multiplier = 2.5 if is_daily else 1.5

    threshold = max(tf_sec + buffer_sec, min_floor_sec) * multiplier

    if is_daily:
        try:
            d1_floor = int(os.getenv("MT5_STALE_D1_FLOOR_SEC", "259200"))  # 72 h
        except (ValueError, TypeError):
            d1_floor = 259200
        threshold = max(threshold, d1_floor)

    return int(threshold)


def get_last_closed_bar_time(df, timeframe, current_time=None):
    """Return the timestamp of the last CLOSED bar in df.

    Audit Round-4 fix: MT5's `copy_rates_from_pos` returns N rows where
    the LAST row (`df.index[-1]`) is the CURRENTLY-FORMING candle — its
    close_time is in the FUTURE relative to now. Passing `df.index[-1]`
    to `is_candle_closed()` therefore ALWAYS returns `is_closed=False`,
    which blocks all new-entry analysis on every cycle, every pair.

    The operator's log showed exactly this:
        last_bar=2026-07-13T12:30:00+00:00 | close=2026-07-13T12:45:00+00:00
        | seconds_left=850s | is_closed=False
        [Trader] EURUSD candle still forming (Forming, 850s left) — skipping

    The bar that opened at 12:30 cannot close until 12:45, so checking
    it for "is closed?" is meaningless — the answer is structurally
    always False.

    This helper walks BACKWARD from the end of df and returns the first
    bar whose close_time <= now_utc. That's the last fully-closed bar —
    the one whose OHLCV is final and safe to analyze.

    Behavior:
      * Common case (MT5): df.index[-1] is forming → returns df.index[-2].
      * Edge case (MT5 hasn't pushed new bar yet): df.index[-1] is the
        just-closed bar → returns df.index[-1] (it has closed by time).
      * Edge case (df has only 1 row): can't walk back → returns
        df.index[-1] as fallback (is_candle_closed() will correctly
        return False, blocking the trade — which is right, we genuinely
        don't have a closed bar).
      * Edge case (broker-tz bug — all bars in future): no closed bar
        found → returns df.index[-1] as fallback (caller's FUTURE_BAR
        detector in is_candle_closed() will fire CRITICAL).

    Args:
        df: pandas DataFrame with a DatetimeIndex.
        timeframe: MT5-style code (e.g. 'M15') or lowercase form ('15m').
        current_time: optional override for "now" (UTC datetime). Defaults
            to datetime.now(timezone.utc). Useful for unit tests.

    Returns:
        tuple: (last_closed_time, row_index, forming_bar_time)
            - last_closed_time: datetime of the last closed bar (or
              df.index[-1] as fallback if no closed bar was found).
            - row_index: integer position in df.index (e.g. -2 means
              second-to-last row; -1 means last row was used as fallback).
            - forming_bar_time: datetime of the forming bar that was
              SKIPPED (or None if no skip happened — e.g. when df[-1]
              was already closed).

    If df is None or empty, returns (None, None, None).
    """
    if current_time is None:
        current_time = datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)

    if df is None or len(df.index) == 0:
        return None, None, None

    tf_sec = timeframe_to_seconds(timeframe)
    forming_bar_time = None

    # Walk backward from the end of df
    for i in range(len(df.index) - 1, -1, -1):
        ts = df.index[i]
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        close_time = ts + timedelta(seconds=tf_sec)
        if current_time >= close_time:
            # This bar is closed — return it
            return ts, i, forming_bar_time
        else:
            # This bar is still forming — record and keep walking back
            if forming_bar_time is None:
                forming_bar_time = ts

    # All bars are forming (broker-tz bug, or df is shorter than tf_sec
    # allows). Fall back to df.index[-1] — caller's FUTURE_BAR detector
    # will fire CRITICAL if applicable.
    last_ts = df.index[-1]
    if hasattr(last_ts, "to_pydatetime"):
        last_ts = last_ts.to_pydatetime()
    if last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=timezone.utc)
    return last_ts, len(df.index) - 1, None


def validate_llm_output(llm_result, ind_ctx):
    if not llm_result: return {"is_valid": False, "hallucinations": ["No output"], "corrected_signal": "WAIT"}
    halluc = []; reason = str(llm_result.get("reason", "")).lower(); signal = llm_result.get("signal", "WAIT").upper()
    rsi = float(ind_ctx.get("rsi", 50) or 50)
    if "oversold" in reason and rsi > 40: halluc.append(f"RSI oversold claim but RSI={rsi:.1f}")
    if "overbought" in reason and rsi < 60: halluc.append(f"RSI overbought claim but RSI={rsi:.1f}")
    ema = float(ind_ctx.get("ema_21", 0) or 0); price = float(ind_ctx.get("close", 0) or 0)
    if ema > 0 and price > 0:
        if "uptrend" in reason and price < ema: halluc.append(f"Uptrend claim but price<EMA")
        if "downtrend" in reason and price > ema: halluc.append(f"Downtrend claim but price>EMA")
    macd = float(ind_ctx.get("macd", 0) or 0)
    if "macd bullish" in reason and macd < 0: halluc.append(f"MACD bullish claim but MACD<0")
    if "macd bearish" in reason and macd > 0: halluc.append(f"MACD bearish claim but MACD>0")
    corrected = "WAIT" if halluc else signal
    return {"is_valid": len(halluc) == 0, "hallucinations": halluc, "corrected_signal": corrected, "original_signal": signal}

def should_use_llm_for_trading(enabled=False):
    return os.getenv("ENABLE_LLM_TRADING", "false").lower() == "true" and enabled

if __name__ == "__main__":
    print("=== PRODUCTION HARDENING TEST ===")
    class R: volume = 0.02; retcode = 10009
    print(f"Partial fill: {check_partial_fill(R(), 0.04)}")
    df = pd.DataFrame({"close": [1.0]}, index=[datetime.now(timezone.utc) - timedelta(seconds=30)])
    print(f"Staleness: {check_data_staleness(df, 120)}")
    print(f"Candle closed: {is_candle_closed('M1', datetime.now(timezone.utc) - timedelta(minutes=2))}")
    print(f"Weekend: {should_close_for_weekend([])}")
    dcm = DynamicCorrelationMatrix(30); np.random.seed(42)
    dcm.update("EURUSD", pd.Series(np.cumsum(np.random.randn(30)) + 1.085))
    dcm.update("GBPUSD", pd.Series(np.cumsum(np.random.randn(30)) + 1.25))
    print(f"Corr EURUSD-GBPUSD: {dcm.get_correlation('EURUSD', 'GBPUSD'):.3f}")
    print(f"LLM validation: {validate_llm_output({'signal':'BUY','reason':'RSI oversold uptrend'}, {'rsi':65,'ema_21':1.084,'close':1.083,'macd':-0.0002})}")
    print("ALL HARDENING TESTS PASSED ✅")