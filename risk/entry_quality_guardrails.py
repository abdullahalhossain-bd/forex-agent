# risk/entry_quality_guardrails.py
# ============================================================
# Entry Quality Guardrails — Anti-Chasing & Structure Validation
# ============================================================
# Built from a real-trade post-mortem on GBPUSD M5 (2026-07-02).
# The trade had the right macro direction but wrong tactical
# execution — entered late into an extended move, into former
# resistance, during stalling candles. These 6 filters catch
# that exact failure mode.
#
# Core lesson encoded: direction-right ≠ timing-right.
#
# The 6 Red Flags (hard-coded):
#   1. Chasing filter — block entries after X-pip move in Y min
#      without a pullback. 80 pips in 60 min on M5 GBPUSD is
#      3-4× ATR — entering into that is chasing, full stop.
#   2. SL must be swing-anchored — a stop chosen without
#      reference to the last N swing lows is a guess, not a stop.
#   3. TP must have prior price action test — if there's no
#      historical S/R between current price and TP, the bot is
#      trading a fantasy target.
#   4. Indecision candle filter — small bodies + long wicks on
#      both sides = market stalling. Skip if last 3 candles have
#      body < 30% of range.
#   5. Indicator confluence required — no price-action-only
#      entries. Need RSI / MA / volume confirmation.
#   6. Round number awareness — 1.3400, 1.3500 etc. are real
#      psychological barriers. Treat as soft resistance in TP.
# ============================================================

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ─── Constants ────────────────────────────────────────────────
# Flag 1 — Chasing filter
DEFAULT_CHASE_PIP_THRESHOLD = 50      # 50+ pips in window = "extended"
DEFAULT_CHASE_WINDOW_BARS   = 12      # 12 bars (1h on M5)
DEFAULT_PULLBACK_MIN_PIPS   = 10      # need ≥10-pip pullback within window
DEFAULT_PULLBACK_MIN_PCT    = 0.10    # OR ≥10% retracement of the move

# Flag 2 — SL swing anchor
DEFAULT_SL_SWING_LOOKBACK   = 20      # check last 20 bars for swing low
DEFAULT_SL_PROXIMITY_ATR    = 0.5     # SL within 0.5×ATR of a swing = "anchored"

# Flag 3 — TP structure validation
DEFAULT_TP_STRUCTURE_LOOKBACK = 100   # check last 100 bars for prior tests
DEFAULT_TP_PROXIMITY_PIPS     = 5     # S/R within 5 pips of TP = "validated"

# Flag 4 — Indecision candles
DEFAULT_INDECISION_BODY_PCT  = 0.30   # body < 30% of range = indecision
DEFAULT_INDECISION_LOOKBACK   = 3      # check last 3 candles
DEFAULT_INDECISION_MIN_COUNT  = 2      # 2+ indecision candles → skip

# Flag 5 — Indicator confluence
DEFAULT_MIN_CONFLUENCE_COUNT = 1      # need ≥1 of: RSI, MA, volume alignment

# Flag 6 — Round numbers
DEFAULT_ROUND_NUMBER_STEPS = {
    "EURUSD": [0.0050, 0.0100],   # 50-pip and 100-pip levels
    "GBPUSD": [0.0050, 0.0100],
    "USDJPY": [0.50, 1.00],
    "XAUUSD": [5.0, 10.0],
}
DEFAULT_ROUND_NUMBER_PROXIMITY_PIPS = 3   # TP within 3 pips of round number = "blocked"


# ─── Dataclass ────────────────────────────────────────────────

@dataclass
class EntryQualityResult:
    """Result of a single entry-quality check."""
    flag_name: str
    passed: bool
    reason: str
    severity: str = "WARNING"   # "WARNING" | "BLOCK"
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "flag_name": self.flag_name,
            "passed":    bool(self.passed),
            "reason":    self.reason,
            "severity":  self.severity,
            "details":   self.details,
        }


# ─── Helpers ──────────────────────────────────────────────────

def _pip_value(symbol: str) -> float:
    sym = (symbol or "").upper()
    if sym.endswith("JPY"):
        return 0.01
    if sym == "XAUUSD":
        return 0.1
    return 0.0001


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    try:
        h, l, c = df["high"], df["low"], df["close"]
        tr = pd.concat(
            [(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
        ).max(axis=1)
        atr = tr.rolling(period, min_periods=1).mean().iloc[-1]
        return float(atr) if np.isfinite(atr) and atr > 0 else 0.001
    except Exception as e:
        return 0.001


def _find_swing_lows(df: pd.DataFrame, lookback: int = 20, window: int = 3) -> List[float]:
    """Find swing lows in the last `lookback` bars (window=3 = 3 bars each side)."""
    try:
        lows = df["low"].values[-lookback:]
        swings = []
        for i in range(window, len(lows) - window):
            if lows[i] == min(lows[i-window:i+window+1]):
                swings.append(float(lows[i]))
        return swings
    except Exception as e:
        import logging; logging.warning(f"Suppressed at risk/entry_quality_guardrails.py:127: {e}")
        return []


def _find_swing_highs(df: pd.DataFrame, lookback: int = 100, window: int = 3) -> List[float]:
    """Find swing highs in the last `lookback` bars."""
    try:
        highs = df["high"].values[-lookback:]
        swings = []
        for i in range(window, len(highs) - window):
            if highs[i] == max(highs[i-window:i+window+1]):
                swings.append(float(highs[i]))
        return swings
    except Exception as e:
        import logging; logging.warning(f"Suppressed at risk/entry_quality_guardrails.py:141: {e}")
        return []


def _is_round_number(price: float, symbol: str) -> Tuple[bool, float]:
    """Check if price is near a round number. Returns (is_round, nearest_round)."""
    sym = (symbol or "").upper()
    steps = DEFAULT_ROUND_NUMBER_STEPS.get(sym, [0.0050, 0.0100])
    for step in steps:
        nearest = round(price / step) * step
        if abs(price - nearest) < step * 0.05:
            return True, nearest
    return False, price


# ═════════════════════════════════════════════════════════════
# FLAG 1: Chasing Filter — block entries after extended moves
# ═════════════════════════════════════════════════════════════

def check_chasing_filter(
    df: pd.DataFrame,
    symbol: str,
    direction: str,
    pip_threshold: float = DEFAULT_CHASE_PIP_THRESHOLD,
    window_bars: int = DEFAULT_CHASE_WINDOW_BARS,
    pullback_min_pips: float = DEFAULT_PULLBACK_MIN_PIPS,
    pullback_min_pct: float = DEFAULT_PULLBACK_MIN_PCT,
) -> EntryQualityResult:
    """
    Red Flag 1: "Never enter after X-pip move in Y minutes without a pullback filter."

    Detects if price has made an extended move (≥pip_threshold in last window_bars)
    AND there's been no meaningful pullback. Entering into such a move = chasing.

    Args:
        df: OHLC DataFrame
        symbol: e.g., "GBPUSD"
        direction: "BUY" or "SELL"
        pip_threshold: pips of move to qualify as "extended" (default 50)
        window_bars: bars to look back (default 12 = 1h on M5)
        pullback_min_pips: minimum pullback required (default 10 pips)
        pullback_min_pct: OR minimum retracement % of the move (default 10%)

    Returns:
        EntryQualityResult with passed=False if chasing detected (BLOCK severity).
    """
    if df is None or len(df) < window_bars + 2:
        return EntryQualityResult(
            flag_name="chasing_filter",
            passed=True,
            reason="Insufficient data — skipping chase check",
            details={"bars_available": len(df) if df is not None else 0},
        )

    pip = _pip_value(symbol)
    window = df.tail(window_bars + 1)

    move_start = float(window["close"].iloc[0])
    move_end   = float(window["close"].iloc[-1])
    move_pips  = abs(move_end - move_start) / pip

    # Direction of the move
    move_direction = "BUY" if move_end > move_start else "SELL"

    # Check if move is in the same direction as the proposed trade
    same_direction = (move_direction == direction.upper())

    # Find max adverse excursion (pullback) within the window
    if direction.upper() == "BUY":
        # For a buy entry, pullback = drop from a peak
        running_max = window["high"].cummax()
        pullback = (running_max - window["low"]).max()
    else:
        # For a sell entry, pullback = rise from a trough
        running_min = window["low"].cummin()
        pullback = (window["high"] - running_min).max()
    pullback_pips = float(pullback) / pip

    # Pullback as % of the move
    pullback_pct = (pullback_pips / move_pips) if move_pips > 0 else 0

    is_extended = move_pips >= pip_threshold
    has_pullback = (pullback_pips >= pullback_min_pips) or (pullback_pct >= pullback_min_pct)

    details = {
        "symbol":           symbol,
        "direction":        direction,
        "window_bars":      window_bars,
        "move_pips":        round(move_pips, 1),
        "move_direction":   move_direction,
        "same_direction":   same_direction,
        "pullback_pips":    round(pullback_pips, 1),
        "pullback_pct":     round(pullback_pct * 100, 1),
        "pip_threshold":    pip_threshold,
        "is_extended":      is_extended,
        "has_pullback":     has_pullback,
    }

    # Block only if: extended move + same direction + no pullback
    if is_extended and same_direction and not has_pullback:
        return EntryQualityResult(
            flag_name="chasing_filter",
            passed=False,
            reason=(
                f"CHASING DETECTED — {move_pips:.0f}-pip {move_direction} move in "
                f"last {window_bars} bars with only {pullback_pips:.1f}-pip pullback "
                f"({pullback_pct*100:.1f}%). Need ≥{pullback_min_pips} pips or "
                f"≥{pullback_min_pct*100}% retracement. BLOCKED."
            ),
            severity="BLOCK",
            details=details,
        )

    if is_extended and same_direction and has_pullback:
        return EntryQualityResult(
            flag_name="chasing_filter",
            passed=True,
            reason=(
                f"Extended move ({move_pips:.0f} pips) BUT pullback present "
                f"({pullback_pips:.1f} pips / {pullback_pct*100:.1f}%). Entry acceptable."
            ),
            details=details,
        )

    return EntryQualityResult(
        flag_name="chasing_filter",
        passed=True,
        reason=(
            f"Move {move_pips:.0f} pips {'in trade direction' if same_direction else 'against trade'}. "
            f"No chasing risk."
        ),
        details=details,
    )


# ═════════════════════════════════════════════════════════════
# FLAG 2: SL Swing Anchor — stop must reference structure
# ═════════════════════════════════════════════════════════════

def check_sl_swing_anchor(
    df: pd.DataFrame,
    symbol: str,
    direction: str,
    stop_loss: float,
    entry_price: float,
    lookback: int = DEFAULT_SL_SWING_LOOKBACK,
    proximity_atr: float = DEFAULT_SL_PROXIMITY_ATR,
) -> EntryQualityResult:
    """
    Red Flag 2: "A stop-loss chosen without reference to the last N swing lows is a guess."

    Verifies that the SL is anchored to a recent swing low (for BUY) or swing high (for SELL).
    If SL is just a fixed-pip distance with no structural reference → flag as WARNING.

    Args:
        df: OHLC DataFrame
        symbol: e.g., "GBPUSD"
        direction: "BUY" or "SELL"
        stop_loss: proposed SL price
        entry_price: entry price
        lookback: bars to search for swings (default 20)
        proximity_atr: SL within this ×ATR of a swing = "anchored" (default 0.5)

    Returns:
        EntryQualityResult with passed=False if SL not anchored (WARNING severity).
    """
    if df is None or len(df) < lookback:
        return EntryQualityResult(
            flag_name="sl_swing_anchor",
            passed=True,
            reason="Insufficient data — skipping SL anchor check",
        )

    atr = _atr(df)
    pip = _pip_value(symbol)

    if direction.upper() == "BUY":
        swings = _find_swing_lows(df, lookback=lookback)
        # SL should be below entry AND near a swing low
        nearest_swing = max(swings) if swings else None
        sl_correct_side = stop_loss < entry_price
    else:
        swings = _find_swing_highs(df, lookback=lookback)
        nearest_swing = min(swings) if swings else None
        sl_correct_side = stop_loss > entry_price

    if not swings or nearest_swing is None:
        return EntryQualityResult(
            flag_name="sl_swing_anchor",
            passed=False,
            severity="WARNING",
            reason=(
                f"No swing {'lows' if direction.upper()=='BUY' else 'highs'} found in last "
                f"{lookback} bars — SL cannot be structurally validated."
            ),
            details={"swings_found": 0, "lookback": lookback},
        )

    # Distance from SL to nearest swing
    sl_to_swing_pips = abs(stop_loss - nearest_swing) / pip
    sl_to_swing_atr  = abs(stop_loss - nearest_swing) / atr if atr > 0 else 999

    is_anchored = sl_to_swing_atr <= proximity_atr

    details = {
        "symbol":          symbol,
        "direction":       direction,
        "entry_price":     round(entry_price, 5),
        "stop_loss":       round(stop_loss, 5),
        "nearest_swing":   round(nearest_swing, 5),
        "swings_found":    len(swings),
        "sl_to_swing_pips": round(sl_to_swing_pips, 1),
        "sl_to_swing_atr":  round(sl_to_swing_atr, 2),
        "proximity_atr":   proximity_atr,
        "sl_correct_side": sl_correct_side,
        "is_anchored":     is_anchored,
    }

    if not sl_correct_side:
        return EntryQualityResult(
            flag_name="sl_swing_anchor",
            passed=False,
            severity="BLOCK",
            reason=(
                f"SL {stop_loss:.5f} is on WRONG SIDE of entry {entry_price:.5f} "
                f"for {direction} trade. BLOCKED."
            ),
            details=details,
        )

    if not is_anchored:
        return EntryQualityResult(
            flag_name="sl_swing_anchor",
            passed=False,
            severity="WARNING",
            reason=(
                f"SL {stop_loss:.5f} is {sl_to_swing_pips:.1f} pips ({sl_to_swing_atr:.2f}×ATR) "
                f"from nearest swing {nearest_swing:.5f} — appears to be a fixed-pip stop, "
                f"not structure-anchored. Risk of stop-run wick."
            ),
            details=details,
        )

    return EntryQualityResult(
        flag_name="sl_swing_anchor",
        passed=True,
        reason=(
            f"SL {stop_loss:.5f} anchored to swing {nearest_swing:.5f} "
            f"({sl_to_swing_pips:.1f} pips / {sl_to_swing_atr:.2f}×ATR)."
        ),
        details=details,
    )


# ═════════════════════════════════════════════════════════════
# FLAG 3: TP Structure Validation — target must have prior test
# ═════════════════════════════════════════════════════════════

def check_tp_structure_validation(
    df: pd.DataFrame,
    symbol: str,
    direction: str,
    take_profit: float,
    entry_price: float,
    lookback: int = DEFAULT_TP_STRUCTURE_LOOKBACK,
    proximity_pips: float = DEFAULT_TP_PROXIMITY_PIPS,
) -> EntryQualityResult:
    """
    Red Flag 3: "TP should never be placed beyond the last visible price action."

    Verifies that TP is at or near a prior swing high (for BUY) or swing low (for SELL)
    in the lookback window. If TP is in "unconfirmed territory" → flag as WARNING.

    Args:
        df: OHLC DataFrame
        symbol: e.g., "GBPUSD"
        direction: "BUY" or "SELL"
        take_profit: proposed TP price
        entry_price: entry price
        lookback: bars to search for prior tests (default 100)
        proximity_pips: TP within this many pips of a swing = "validated" (default 5)

    Returns:
        EntryQualityResult with passed=False if TP not validated (WARNING severity).
    """
    if df is None or len(df) < lookback:
        return EntryQualityResult(
            flag_name="tp_structure_validation",
            passed=True,
            reason="Insufficient data — skipping TP validation",
        )

    pip = _pip_value(symbol)

    if direction.upper() == "BUY":
        # TP should be above entry, near a prior swing high
        swings = _find_swing_highs(df, lookback=lookback)
        tp_correct_side = take_profit > entry_price
        # Filter swings that are above entry (potential TP targets)
        target_swings = [s for s in swings if s > entry_price]
    else:
        swings = _find_swing_lows(df, lookback=lookback)
        tp_correct_side = take_profit < entry_price
        target_swings = [s for s in swings if s < entry_price]

    if not tp_correct_side:
        return EntryQualityResult(
            flag_name="tp_structure_validation",
            passed=False,
            severity="BLOCK",
            reason=(
                f"TP {take_profit:.5f} is on WRONG SIDE of entry {entry_price:.5f} "
                f"for {direction} trade. BLOCKED."
            ),
            details={"tp": take_profit, "entry": entry_price, "direction": direction},
        )

    if not target_swings:
        return EntryQualityResult(
            flag_name="tp_structure_validation",
            passed=False,
            severity="WARNING",
            reason=(
                f"TP {take_profit:.5f} is in UNCONFIRMED TERRITORY — no prior "
                f"{'swing highs' if direction.upper()=='BUY' else 'swing lows'} above entry "
                f"in last {lookback} bars. Trading a fantasy target."
            ),
            details={
                "tp": take_profit,
                "entry": entry_price,
                "swings_above_entry": 0,
                "lookback": lookback,
            },
        )

    # Find nearest swing to TP
    nearest_swing = min(target_swings, key=lambda s: abs(s - take_profit))
    tp_to_swing_pips = abs(take_profit - nearest_swing) / pip
    is_validated = tp_to_swing_pips <= proximity_pips

    details = {
        "symbol":          symbol,
        "direction":       direction,
        "entry_price":     round(entry_price, 5),
        "take_profit":     round(take_profit, 5),
        "nearest_swing":   round(nearest_swing, 5),
        "tp_to_swing_pips": round(tp_to_swing_pips, 1),
        "proximity_pips":  proximity_pips,
        "swings_above_entry": len(target_swings),
        "is_validated":    is_validated,
    }

    if not is_validated:
        return EntryQualityResult(
            flag_name="tp_structure_validation",
            passed=False,
            severity="WARNING",
            reason=(
                f"TP {take_profit:.5f} is {tp_to_swing_pips:.1f} pips from nearest "
                f"prior swing {nearest_swing:.5f} — no confirmed S/R at target. "
                f"Consider moving TP to {nearest_swing:.5f}."
            ),
            details=details,
        )

    return EntryQualityResult(
        flag_name="tp_structure_validation",
        passed=True,
        reason=(
            f"TP {take_profit:.5f} validated by prior swing {nearest_swing:.5f} "
            f"({tp_to_swing_pips:.1f} pips apart)."
        ),
        details=details,
    )


# ═════════════════════════════════════════════════════════════
# FLAG 4: Indecision Candle Filter — skip stalling entries
# ═════════════════════════════════════════════════════════════

def check_indecision_candles(
    df: pd.DataFrame,
    body_pct_threshold: float = DEFAULT_INDECISION_BODY_PCT,
    lookback: int = DEFAULT_INDECISION_LOOKBACK,
    min_indecision_count: int = DEFAULT_INDECISION_MIN_COUNT,
) -> EntryQualityResult:
    """
    Red Flag 4: "Skip entry if last 3 candles have body < 30% of range."

    Detects stalling/indecision candles (small body, long wicks both sides).
    Entering during indecision = entering when market is telling you to wait.

    Args:
        df: OHLC DataFrame
        body_pct_threshold: body/range below this = indecision (default 0.30)
        lookback: candles to check (default 3)
        min_indecision_count: number of indecision candles to trigger skip (default 2)

    Returns:
        EntryQualityResult with passed=False if indecision detected (BLOCK severity).
    """
    if df is None or len(df) < lookback:
        return EntryQualityResult(
            flag_name="indecision_candles",
            passed=True,
            reason="Insufficient data — skipping indecision check",
        )

    recent = df.tail(lookback)
    indecision_count = 0
    candle_details = []

    for idx, row in recent.iterrows():
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        body = abs(c - o)
        total_range = h - l
        if total_range <= 0:
            continue
        body_pct = body / total_range
        is_indecision = body_pct < body_pct_threshold
        if is_indecision:
            indecision_count += 1
        candle_details.append({
            "time": str(idx),
            "body_pct": round(body_pct, 3),
            "is_indecision": is_indecision,
        })

    is_stalling = indecision_count >= min_indecision_count

    details = {
        "lookback": lookback,
        "indecision_count": indecision_count,
        "min_indecision_count": min_indecision_count,
        "body_pct_threshold": body_pct_threshold,
        "candles": candle_details,
        "is_stalling": is_stalling,
    }

    if is_stalling:
        return EntryQualityResult(
            flag_name="indecision_candles",
            passed=False,
            severity="BLOCK",
            reason=(
                f"INDECISION DETECTED — {indecision_count}/{lookback} recent candles have "
                f"body < {body_pct_threshold*100:.0f}% of range. Market is stalling. "
                f"Wait for confirmation candle. BLOCKED."
            ),
            details=details,
        )

    return EntryQualityResult(
        flag_name="indecision_candles",
        passed=True,
        reason=(
            f"Only {indecision_count}/{lookback} indecision candles — market has direction."
        ),
        details=details,
    )


# ═════════════════════════════════════════════════════════════
# FLAG 5: Indicator Confluence Required
# ═════════════════════════════════════════════════════════════

def check_indicator_confluence(
    df: pd.DataFrame,
    direction: str,
    ind_ctx: Optional[dict] = None,
    min_confluence: int = DEFAULT_MIN_CONFLUENCE_COUNT,
) -> EntryQualityResult:
    """
    Red Flag 5: "No indicator confluence was used — no RSI, MA, volume logic."

    Verifies that at least one indicator aligns with the proposed trade direction.
    A bot running on price action alone will systematically chase tops.

    Checks (any one suffices for default min_confluence=1):
      - RSI alignment: BUY needs RSI > 50 (or oversold reversal), SELL needs RSI < 50
      - MA alignment: BUY needs price above MA, SELL needs price below MA
      - Volume alignment: BUY needs rising volume on up-candles, SELL on down-candles

    Args:
        df: OHLC DataFrame
        direction: "BUY" or "SELL"
        ind_ctx: indicator context dict (optional — will compute basic if None)
        min_confluence: minimum number of aligning indicators (default 1)

    Returns:
        EntryQualityResult with passed=False if no confluence (WARNING severity).
    """
    if df is None or len(df) < 20:
        return EntryQualityResult(
            flag_name="indicator_confluence",
            passed=True,
            reason="Insufficient data — skipping confluence check",
        )

    direction = direction.upper()
    confluences = []

    # ── RSI check ──
    rsi_value = None
    if ind_ctx and "rsi" in ind_ctx:
        rsi_value = ind_ctx.get("rsi")
    elif "rsi" in df.columns:
        rsi_value = float(df["rsi"].iloc[-1])

    if rsi_value is not None and not np.isnan(rsi_value):
        if direction == "BUY" and rsi_value > 50:
            confluences.append({"indicator": "RSI", "value": round(rsi_value, 1), "alignment": "BUY"})
        elif direction == "SELL" and rsi_value < 50:
            confluences.append({"indicator": "RSI", "value": round(rsi_value, 1), "alignment": "SELL"})

    # ── MA check ──
    ma_value = None
    if ind_ctx and "ema_20" in ind_ctx:
        ma_value = ind_ctx.get("ema_20")
    elif "ema_20" in df.columns:
        ma_value = float(df["ema_20"].iloc[-1])

    current_price = float(df["close"].iloc[-1])
    if ma_value is not None and not np.isnan(ma_value) and ma_value > 0:
        if direction == "BUY" and current_price > ma_value:
            confluences.append({"indicator": "EMA20", "value": round(ma_value, 5), "alignment": "BUY"})
        elif direction == "SELL" and current_price < ma_value:
            confluences.append({"indicator": "EMA20", "value": round(ma_value, 5), "alignment": "SELL"})

    # ── Volume check ──
    if "volume" in df.columns:
        recent_vol = df["volume"].tail(10).values
        avg_vol = float(np.mean(recent_vol))
        current_vol = float(recent_vol[-1])
        # Check if recent candle in trade direction had above-average volume
        last_candle = df.iloc[-1]
        if direction == "BUY" and last_candle["close"] > last_candle["open"] and current_vol > avg_vol:
            confluences.append({"indicator": "Volume", "value": current_vol, "alignment": "BUY"})
        elif direction == "SELL" and last_candle["close"] < last_candle["open"] and current_vol > avg_vol:
            confluences.append({"indicator": "Volume", "value": current_vol, "alignment": "SELL"})

    confluence_count = len(confluences)
    is_aligned = confluence_count >= min_confluence

    details = {
        "direction": direction,
        "confluences": confluences,
        "confluence_count": confluence_count,
        "min_confluence": min_confluence,
        "rsi_available": rsi_value is not None,
        "ma_available": ma_value is not None,
        "volume_available": "volume" in df.columns,
    }

    if not is_aligned:
        return EntryQualityResult(
            flag_name="indicator_confluence",
            passed=False,
            severity="WARNING",
            reason=(
                f"NO INDICATOR CONFLUENCE — 0 of RSI/MA/Volume align with {direction}. "
                f"Price-action-only entries systematically chase tops. "
                f"Need ≥{min_confluence} indicator alignment."
            ),
            details=details,
        )

    return EntryQualityResult(
        flag_name="indicator_confluence",
        passed=True,
        reason=(
            f"{confluence_count} indicator(s) align with {direction}: "
            f"{[c['indicator'] for c in confluences]}."
        ),
        details=details,
    )


# ═════════════════════════════════════════════════════════════
# FLAG 6: Round Number Awareness in TP
# ═════════════════════════════════════════════════════════════

def check_round_number_tp(
    symbol: str,
    direction: str,
    take_profit: float,
    entry_price: float,
    proximity_pips: float = DEFAULT_ROUND_NUMBER_PROXIMITY_PIPS,
) -> EntryQualityResult:
    """
    Red Flag 6: "Round numbers (1.3400, 1.3500) are real psychological barriers."

    Checks if TP is placed just below/above a round number — these act as soft
    resistance/support and often cause TP to miss by a few pips.

    For BUY: TP just below a round number (e.g., 1.3398) → flag (price may reverse at 1.3400)
    For SELL: TP just above a round number (e.g., 1.3402) → flag

    Args:
        symbol: e.g., "GBPUSD"
        direction: "BUY" or "SELL"
        take_profit: proposed TP
        entry_price: entry price
        proximity_pips: TP within this many pips of round number = "blocked" (default 3)

    Returns:
        EntryQualityResult with passed=False if TP too close to round number (WARNING).
    """
    pip = _pip_value(symbol)
    is_round, nearest_round = _is_round_number(take_profit, symbol)

    # Find the nearest round number (even if TP isn't exactly on one)
    steps = DEFAULT_ROUND_NUMBER_STEPS.get(symbol.upper(), [0.0050, 0.0100])
    nearest = None
    min_dist = float("inf")
    for step in steps:
        candidate = round(take_profit / step) * step
        dist = abs(take_profit - candidate)
        if dist < min_dist:
            min_dist = dist
            nearest = candidate

    dist_pips = min_dist / pip if pip > 0 else 999
    is_too_close = dist_pips <= proximity_pips

    # Check if round number is between entry and TP (acts as barrier)
    if direction.upper() == "BUY":
        round_between = entry_price < nearest < take_profit
        # If TP is just below a round number, it might not reach
        tp_just_below = take_profit < nearest and dist_pips <= proximity_pips
    else:
        round_between = entry_price > nearest > take_profit
        tp_just_below = take_profit > nearest and dist_pips <= proximity_pips

    details = {
        "symbol": symbol,
        "direction": direction,
        "take_profit": round(take_profit, 5),
        "entry_price": round(entry_price, 5),
        "nearest_round_number": round(nearest, 5),
        "distance_pips": round(dist_pips, 1),
        "proximity_pips": proximity_pips,
        "round_between_entry_tp": round_between,
        "is_too_close": is_too_close,
        "tp_just_below": tp_just_below,
    }

    if round_between and is_too_close:
        return EntryQualityResult(
            flag_name="round_number_tp",
            passed=False,
            severity="WARNING",
            reason=(
                f"TP {take_profit:.5f} is {dist_pips:.1f} pips from round number "
                f"{nearest:.5f} (between entry and TP). Round numbers act as soft "
                f"resistance — consider moving TP to {nearest:.5f} or beyond."
            ),
            details=details,
        )

    # FIX (bug 21): tp_just_below was computed but never consulted here — this
    # branch used to fire on raw proximity (is_too_close) alone, which also
    # WARNs when TP already sits safely *past* the round number (no barrier
    # left ahead of it). Gate on tp_just_below so it only fires when the round
    # number is still an un-crossed barrier in front of TP.
    if tp_just_below:
        return EntryQualityResult(
            flag_name="round_number_tp",
            passed=False,
            severity="WARNING",
            reason=(
                f"TP {take_profit:.5f} is only {dist_pips:.1f} pips short of round number "
                f"{nearest:.5f}. Price may reverse at the round number before reaching TP. "
                f"Consider adjusting TP."
            ),
            details=details,
        )

    return EntryQualityResult(
        flag_name="round_number_tp",
        passed=True,
        reason=(
            f"TP {take_profit:.5f} is {dist_pips:.1f} pips from nearest round "
            f"{nearest:.5f} — safe distance."
        ),
        details=details,
    )


# ═════════════════════════════════════════════════════════════
# FLAG 7: Rejection Wick at Entry (shooting-star when buying)
# ═════════════════════════════════════════════════════════════

def check_rejection_wick_at_entry(
    df: pd.DataFrame,
    direction: str,
    wick_body_ratio: float = 2.0,
    lookback: int = 1,
) -> EntryQualityResult:
    """
    Red Flag 7 (XAUUSD post-mortem): "Entered right on a rejection candle —
    shooting-star wick at the high — rather than waiting for rejection to resolve."

    Detects a bearish rejection wick (long upper wick) when buying,
    or a bullish rejection wick (long lower wick) when selling.
    Entering right as a rejection wick prints = entering against immediate pressure.

    Args:
        df: OHLC DataFrame
        direction: "BUY" or "SELL"
        wick_body_ratio: rejection wick must be ≥ this × body (default 2.0)
        lookback: candles to check at the end (default 1 = just last candle)

    Returns:
        EntryQualityResult with passed=False if rejection wick detected (BLOCK).
    """
    if df is None or len(df) < lookback:
        return EntryQualityResult(
            flag_name="rejection_wick_at_entry",
            passed=True,
            reason="Insufficient data — skipping rejection wick check",
        )

    recent = df.tail(lookback)
    rejection_detected = False
    candle_details = []

    for idx, row in recent.iterrows():
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        body = abs(c - o)
        if body < 1e-9:
            continue
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        # For BUY: bearish rejection = long upper wick (shooting star)
        if direction.upper() == "BUY":
            is_rejection = upper_wick >= body * wick_body_ratio and upper_wick > lower_wick * 1.5
            rejection_type = "bearish_upper_wick"
        # For SELL: bullish rejection = long lower wick (hammer)
        else:
            is_rejection = lower_wick >= body * wick_body_ratio and lower_wick > upper_wick * 1.5
            rejection_type = "bullish_lower_wick"

        if is_rejection:
            rejection_detected = True
        candle_details.append({
            "time": str(idx),
            "upper_wick": round(upper_wick, 5),
            "lower_wick": round(lower_wick, 5),
            "body": round(body, 5),
            "wick_body_ratio": round(max(upper_wick, lower_wick) / body, 2),
            "is_rejection": is_rejection,
            "rejection_type": rejection_type if is_rejection else None,
        })

    details = {
        "direction": direction,
        "lookback": lookback,
        "wick_body_ratio_threshold": wick_body_ratio,
        "rejection_detected": rejection_detected,
        "candles": candle_details,
    }

    if rejection_detected:
        wick_type = "upper (shooting-star)" if direction.upper() == "BUY" else "lower (hammer)"
        return EntryQualityResult(
            flag_name="rejection_wick_at_entry",
            passed=False,
            severity="BLOCK",
            reason=(
                f"REJECTION WICK at entry — {wick_type} wick ≥ {wick_body_ratio}× body detected. "
                f"Entering right as rejection prints = entering against immediate pressure. "
                f"Wait for rejection to resolve. BLOCKED."
            ),
            details=details,
        )

    return EntryQualityResult(
        flag_name="rejection_wick_at_entry",
        passed=True,
        reason=f"No rejection wick at entry candle(s).",
        details=details,
    )


# ═════════════════════════════════════════════════════════════
# FLAG 8: Averaging Into Losers (un-planned same-direction adds)
# ═════════════════════════════════════════════════════════════

def check_averaging_into_losers(
    symbol: str,
    direction: str,
    proposed_entry: float,
    open_positions: List[dict],
    max_adds_underwater: int = 1,
    underwater_threshold_pips: float = 5.0,
    time_window_minutes: int = 60,
) -> EntryQualityResult:
    """
    Red Flag 8 (XAUUSD post-mortem): "Three XAUUSD buys within ~6 points as price
    fell — un-planned averaging, not a pre-defined scaling strategy."

    Detects when the bot is adding to a losing position in the same direction
    multiple times within a short window. This is averaging into a drawdown,
    NOT disciplined pyramiding on confirmation.

    Args:
        symbol: e.g., "XAUUSD"
        direction: "BUY" or "SELL"
        proposed_entry: entry price of the proposed new trade
        open_positions: list of dicts with keys:
            {"pair": str, "direction": str, "entry": float,
             "current_price": float, "open_time": datetime/ISO string}
        max_adds_underwater: max allowed same-direction adds while underwater (default 1)
        underwater_threshold_pips: position is "underwater" if down by this many pips (default 5)
        time_window_minutes: only count adds within this recent window (default 60)

    Returns:
        EntryQualityResult with passed=False if averaging detected (BLOCK).
    """
    pip = _pip_value(symbol)
    sym = symbol.upper()
    dir_upper = direction.upper()

    # Filter open positions: same symbol + same direction + within time window
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(minutes=time_window_minutes)

    same_direction_underwater = []
    for pos in open_positions:
        if pos.get("pair", "").upper() != sym:
            continue
        if pos.get("direction", "").upper() != dir_upper:
            continue

        entry = float(pos.get("entry", 0))
        current = float(pos.get("current_price", entry))

        # Check if underwater
        if dir_upper == "BUY":
            pnl_pips = (current - entry) / pip
        else:
            pnl_pips = (entry - current) / pip

        is_underwater = pnl_pips < -underwater_threshold_pips

        # Check time window
        open_time = pos.get("open_time")
        in_window = True
        if open_time:
            try:
                if isinstance(open_time, str):
                    pos_time = datetime.fromisoformat(open_time.replace("Z", "+00:00"))
                else:
                    pos_time = open_time
                if pos_time.tzinfo is None:
                    pos_time = pos_time.replace(tzinfo=timezone.utc)
                in_window = pos_time >= window_start
            except Exception as e:
                in_window = True  # if can't parse, assume in window

        if is_underwater and in_window:
            same_direction_underwater.append({
                "entry": entry,
                "current": current,
                "pnl_pips": round(pnl_pips, 1),
                "open_time": str(open_time) if open_time else None,
            })

    add_count = len(same_direction_underwater)
    is_averaging = add_count > max_adds_underwater

    details = {
        "symbol": symbol,
        "direction": direction,
        "proposed_entry": proposed_entry,
        "existing_underwater_positions": same_direction_underwater,
        "underwater_count": add_count,
        "max_adds_allowed": max_adds_underwater,
        "time_window_minutes": time_window_minutes,
        "is_averaging": is_averaging,
    }

    if is_averaging:
        return EntryQualityResult(
            flag_name="averaging_into_losers",
            passed=False,
            severity="BLOCK",
            reason=(
                f"AVERAGING INTO LOSERS — {add_count} existing {direction} {symbol} positions "
                f"underwater within {time_window_minutes}min. Adding more = un-planned averaging, "
                f"not disciplined pyramiding. Max allowed: {max_adds_underwater}. BLOCKED."
            ),
            details=details,
        )

    if add_count > 0:
        return EntryQualityResult(
            flag_name="averaging_into_losers",
            passed=True,
            reason=(
                f"{add_count} underwater {direction} {symbol} position(s) — within "
                f"max {max_adds_underwater} allowed. Add acceptable."
            ),
            details=details,
        )

    return EntryQualityResult(
        flag_name="averaging_into_losers",
        passed=True,
        reason=f"No existing underwater {direction} {symbol} positions — no averaging risk.",
        details=details,
    )


# ═════════════════════════════════════════════════════════════
# FLAG 9: Fresh High + Rejection (distance from swing high)
# ═════════════════════════════════════════════════════════════

def check_fresh_high_rejection(
    df: pd.DataFrame,
    symbol: str,
    direction: str,
    entry_price: float,
    swing_lookback: int = 50,
    fresh_high_proximity_pips: float = 10.0,
) -> EntryQualityResult:
    """
    Red Flag 9 (XAUUSD post-mortem): "Add a 'distance from recent swing high/impulse'
    filter that blocks new longs when price is making a fresh high with a rejection wick."

    Detects when:
      - For BUY: price is at/near a fresh swing high (within proximity_pips)
        AND the last candle has a rejection wick (long upper wick)
      - For SELL: price is at/near a fresh swing low AND last candle has
        long lower wick

    This is the highest-leverage fix per the trade analysis.

    Args:
        df: OHLC DataFrame
        symbol: e.g., "XAUUSD"
        direction: "BUY" or "SELL"
        entry_price: proposed entry
        swing_lookback: bars to find swing high/low (default 50)
        fresh_high_proximity_pips: within this many pips of swing = "at high" (default 10)

    Returns:
        EntryQualityResult with passed=False if fresh high + rejection (BLOCK).
    """
    if df is None or len(df) < swing_lookback:
        return EntryQualityResult(
            flag_name="fresh_high_rejection",
            passed=True,
            reason="Insufficient data — skipping fresh high check",
        )

    pip = _pip_value(symbol)

    if direction.upper() == "BUY":
        swing_extreme = float(df["high"].values[-swing_lookback:].max())
        dist_pips = (entry_price - swing_extreme) / pip  # negative if below high
        at_extreme = abs(dist_pips) <= fresh_high_proximity_pips
        extreme_name = "swing high"
    else:
        swing_extreme = float(df["low"].values[-swing_lookback:].min())
        dist_pips = (swing_extreme - entry_price) / pip
        at_extreme = abs(dist_pips) <= fresh_high_proximity_pips
        extreme_name = "swing low"

    # Check for rejection wick on last candle
    last = df.iloc[-1]
    o, h, l, c = float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"])
    body = abs(c - o)
    has_rejection = False
    if body > 0:
        if direction.upper() == "BUY":
            upper_wick = h - max(o, c)
            has_rejection = upper_wick >= body * 2.0
        else:
            lower_wick = min(o, c) - l
            has_rejection = lower_wick >= body * 2.0

    details = {
        "symbol": symbol,
        "direction": direction,
        "entry_price": round(entry_price, 5),
        "swing_extreme": round(swing_extreme, 5),
        "extreme_type": extreme_name,
        "distance_pips": round(dist_pips, 1),
        "at_extreme": at_extreme,
        "has_rejection_wick": has_rejection,
        "fresh_high_proximity_pips": fresh_high_proximity_pips,
    }

    if at_extreme and has_rejection:
        return EntryQualityResult(
            flag_name="fresh_high_rejection",
            passed=False,
            severity="BLOCK",
            reason=(
                f"FRESH {extreme_name.upper()} + REJECTION — entry {entry_price:.5f} is "
                f"{abs(dist_pips):.1f} pips from {swing_extreme:.5f} AND last candle has "
                f"rejection wick. Wait for pullback-and-confirm. BLOCKED. "
                f"(Highest-leverage fix per trade analysis.)"
            ),
            details=details,
        )

    if at_extreme:
        return EntryQualityResult(
            flag_name="fresh_high_rejection",
            passed=True,
            reason=(
                f"At {extreme_name} ({abs(dist_pips):.1f} pips from {swing_extreme:.5f}) "
                f"but NO rejection wick — entry acceptable."
            ),
            details=details,
        )

    return EntryQualityResult(
        flag_name="fresh_high_rejection",
        passed=True,
        reason=(
            f"Not at {extreme_name} ({abs(dist_pips):.1f} pips away from {swing_extreme:.5f})."
        ),
        details=details,
    )


# ═════════════════════════════════════════════════════════════
# FLAG 10: TP Above Unconfirmed Spike (liquidity sweep)
# ═════════════════════════════════════════════════════════════

def check_tp_above_unconfirmed_spike(
    df: pd.DataFrame,
    symbol: str,
    direction: str,
    take_profit: float,
    spike_lookback: int = 50,
    spike_wick_body_ratio: float = 2.5,
    retest_lookback: int = 10,
) -> EntryQualityResult:
    """
    Red Flag 10 (XAUUSD post-mortem): "TP sits above the 4146 spike high with
    no evidence that level will be reclaimed — aspirational, not derived from
    a measured move or clear next resistance."

    Detects when TP is placed above (for BUY) or below (for SELL) an unconfirmed
    spike high/low — a wick extreme that looks like a liquidity sweep, not an
    accepted price level. TP beyond such a spike = trading a fantasy target.

    A spike is "unconfirmed" if:
      - The wick is ≥ spike_wick_body_ratio × body (long wick = sweep)
      - Price has NOT closed back above/below the spike extreme in the last
        retest_lookback bars (no retest = not accepted)

    Args:
        df: OHLC DataFrame
        symbol: e.g., "XAUUSD"
        direction: "BUY" or "SELL"
        take_profit: proposed TP
        spike_lookback: bars to search for spike (default 50)
        spike_wick_body_ratio: wick/body ratio to qualify as spike (default 2.5)
        retest_lookback: bars to check for retest/acceptance (default 10)

    Returns:
        EntryQualityResult with passed=False if TP above unconfirmed spike (WARNING).
    """
    if df is None or len(df) < spike_lookback:
        return EntryQualityResult(
            flag_name="tp_above_unconfirmed_spike",
            passed=True,
            reason="Insufficient data — skipping spike check",
        )

    pip = _pip_value(symbol)
    recent = df.tail(spike_lookback)

    if direction.upper() == "BUY":
        # Find the highest high
        spike_idx = recent["high"].idxmax()
        spike_high = float(recent.loc[spike_idx, "high"])
        spike_o = float(recent.loc[spike_idx, "open"])
        spike_c = float(recent.loc[spike_idx, "close"])
        spike_body = abs(spike_c - spike_o)
        spike_wick = spike_high - max(spike_o, spike_c)

        # TP above spike?
        tp_above_spike = take_profit > spike_high

        # Is spike a wick spike?
        is_wick_spike = (spike_body > 0 and spike_wick >= spike_body * spike_wick_body_ratio)

        # Has price closed back above spike high in last retest_lookback bars?
        post_spike = recent.loc[spike_idx:].tail(retest_lookback)
        retest_confirmed = any(float(c) > spike_high for c in post_spike["close"].values)

        extreme_name = "spike high"
        tp_beyond = tp_above_spike

    else:  # SELL
        spike_idx = recent["low"].idxmin()
        spike_low = float(recent.loc[spike_idx, "low"])
        spike_o = float(recent.loc[spike_idx, "open"])
        spike_c = float(recent.loc[spike_idx, "close"])
        spike_body = abs(spike_c - spike_o)
        spike_wick = min(spike_o, spike_c) - spike_low

        tp_above_spike = take_profit < spike_low  # TP below spike low for SELL
        is_wick_spike = (spike_body > 0 and spike_wick >= spike_body * spike_wick_body_ratio)

        post_spike = recent.loc[spike_idx:].tail(retest_lookback)
        retest_confirmed = any(float(c) < spike_low for c in post_spike["close"].values)

        extreme_name = "spike low"
        tp_beyond = tp_above_spike

    is_unconfirmed = is_wick_spike and not retest_confirmed
    tp_beyond_unconfirmed = tp_beyond and is_unconfirmed

    details = {
        "symbol": symbol,
        "direction": direction,
        "take_profit": round(take_profit, 5),
        "spike_extreme": round(spike_high if direction.upper() == "BUY" else spike_low, 5),
        "spike_type": extreme_name,
        "tp_beyond_spike": tp_beyond,
        "is_wick_spike": is_wick_spike,
        "spike_wick_body_ratio": round(spike_wick / spike_body, 2) if spike_body > 0 else 0,
        "retest_confirmed": retest_confirmed,
        "is_unconfirmed": is_unconfirmed,
        "tp_beyond_unconfirmed": tp_beyond_unconfirmed,
    }

    if tp_beyond_unconfirmed:
        spike_price = spike_high if direction.upper() == "BUY" else spike_low
        return EntryQualityResult(
            flag_name="tp_above_unconfirmed_spike",
            passed=False,
            severity="WARNING",
            reason=(
                f"TP {take_profit:.5f} is beyond UNCONFIRMED {extreme_name} {spike_price:.5f} "
                f"(wick/body={spike_wick/spike_body:.1f}×, no retest in last {retest_lookback} bars). "
                f"Likely a liquidity sweep — TP is aspirational, not validated. "
                f"Consider moving TP to {spike_price:.5f} or wait for retest."
            ),
            details=details,
        )

    return EntryQualityResult(
        flag_name="tp_above_unconfirmed_spike",
        passed=True,
        reason=(
            f"TP {take_profit:.5f} "
            f"{'beyond' if tp_beyond else 'not beyond'} {extreme_name} "
            f"({spike_high if direction.upper()=='BUY' else spike_low:.5f}), "
            f"spike {'unconfirmed' if is_unconfirmed else 'confirmed'}."
        ),
        details=details,
    )


# ═════════════════════════════════════════════════════════════
# FLAG 11: Opposite-Direction Stacking Guard (signal whipsaw)
# ═════════════════════════════════════════════════════════════

def check_opposite_direction_stacking(
    symbol: str,
    direction: str,
    open_positions: List[dict],
) -> EntryQualityResult:
    """
    Red Flag 11 (USDJPY post-mortem): "Never open an opposite-direction
    position on the same symbol while an existing position is open —
    force a close-then-reverse logic, never a stack."

    Detects when the bot is about to open a BUY while SELL(s) are open
    on the same symbol (or vice versa). This is signal whipsaw, NOT a
    deliberate hedge. If bias flipped, the bot should CLOSE the existing
    position first, not stack an opposing one.

    Args:
        symbol: e.g., "USDJPY"
        direction: "BUY" or "SELL" (proposed new trade)
        open_positions: list of dicts with keys {"pair": str, "direction": str}

    Returns:
        EntryQualityResult with passed=False if opposite-direction position
        exists on same symbol (BLOCK).
    """
    sym = symbol.upper()
    dir_upper = direction.upper()
    opposite_dir = "SELL" if dir_upper == "BUY" else "BUY"

    # Find existing opposite-direction positions on same symbol
    opposite_positions = []
    for pos in open_positions:
        if pos.get("pair", "").upper() != sym:
            continue
        if pos.get("direction", "").upper() == opposite_dir:
            opposite_positions.append({
                "pair": pos.get("pair"),
                "direction": pos.get("direction"),
                "entry": pos.get("entry"),
            })

    has_opposite = len(opposite_positions) > 0

    details = {
        "symbol": symbol,
        "proposed_direction": direction,
        "opposite_direction": opposite_dir,
        "opposite_positions_found": opposite_positions,
        "opposite_count": len(opposite_positions),
        "has_opposite_open": has_opposite,
    }

    if has_opposite:
        return EntryQualityResult(
            flag_name="opposite_direction_stacking",
            passed=False,
            severity="BLOCK",
            reason=(
                f"OPPOSITE-DIRECTION STACKING — {len(opposite_positions)} existing "
                f"{opposite_dir} {symbol} position(s) open. Proposed {direction} = "
                f"signal whipsaw, NOT hedge. Close existing position first, then reverse. "
                f"Never stack opposing trades. BLOCKED."
            ),
            details=details,
        )

    return EntryQualityResult(
        flag_name="opposite_direction_stacking",
        passed=True,
        reason=f"No existing {opposite_dir} {symbol} positions — no stacking risk.",
        details=details,
    )


# ═════════════════════════════════════════════════════════════
# FLAG 12: Exhaustion Filter (shrinking bodies after momentum)
# ═════════════════════════════════════════════════════════════

def check_exhaustion_filter(
    df: pd.DataFrame,
    direction: str,
    momentum_lookback: int = 5,
    exhaustion_lookback: int = 3,
    momentum_body_pct: float = 0.60,
    exhaustion_body_pct: float = 0.35,
) -> EntryQualityResult:
    """
    Red Flag 12 (USDJPY post-mortem): "Block new entries in the direction
    of a move if the last 2-3 candles show shrinking bodies/consolidation
    at a support/resistance extreme — that's the market telling you the
    move may be done, not a re-entry cue."

    Detects exhaustion pattern:
      1. Recent momentum candles (body ≥ 60% of range) in trade direction
      2. Followed by small-bodied consolidation (body < 35% of range)
    Entering in the direction of the exhausted move = entering at the worst time.

    Args:
        df: OHLC DataFrame
        direction: "BUY" or "SELL"
        momentum_lookback: bars to check for momentum (default 5)
        exhaustion_lookback: bars to check for exhaustion at end (default 3)
        momentum_body_pct: body/range to qualify as momentum candle (default 0.60)
        exhaustion_body_pct: body/range below this = exhaustion (default 0.35)

    Returns:
        EntryQualityResult with passed=False if exhaustion detected (BLOCK).
    """
    total_needed = momentum_lookback + exhaustion_lookback
    if df is None or len(df) < total_needed:
        return EntryQualityResult(
            flag_name="exhaustion_filter",
            passed=True,
            reason="Insufficient data — skipping exhaustion check",
        )

    # Split into momentum window and exhaustion window
    momentum_window = df.iloc[-(total_needed):-exhaustion_lookback]
    exhaustion_window = df.iloc[-exhaustion_lookback:]

    # Check momentum candles: were they in the trade direction AND large-bodied?
    momentum_count = 0
    for idx, row in momentum_window.iterrows():
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        body = abs(c - o)
        total_range = h - l
        if total_range <= 0:
            continue
        body_pct = body / total_range

        # Direction check: bullish candle (c>o) for BUY, bearish (c<o) for SELL
        is_directional = (c > o) if direction.upper() == "BUY" else (c < o)
        is_momentum = body_pct >= momentum_body_pct

        if is_directional and is_momentum:
            momentum_count += 1

    # Check exhaustion candles: small-bodied consolidation at the end
    exhaustion_count = 0
    candle_details = []
    for idx, row in exhaustion_window.iterrows():
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        body = abs(c - o)
        total_range = h - l
        if total_range <= 0:
            continue
        body_pct = body / total_range
        is_exhaustion = body_pct < exhaustion_body_pct
        if is_exhaustion:
            exhaustion_count += 1
        candle_details.append({
            "time": str(idx),
            "body_pct": round(body_pct, 3),
            "is_exhaustion": is_exhaustion,
        })

    # Exhaustion pattern: ≥2 momentum candles + ≥2 exhaustion candles
    has_momentum = momentum_count >= 2
    has_exhaustion = exhaustion_count >= 2
    is_exhausted = has_momentum and has_exhaustion

    details = {
        "direction": direction,
        "momentum_lookback": momentum_lookback,
        "exhaustion_lookback": exhaustion_lookback,
        "momentum_candles_found": momentum_count,
        "exhaustion_candles_found": exhaustion_count,
        "momentum_body_pct_threshold": momentum_body_pct,
        "exhaustion_body_pct_threshold": exhaustion_body_pct,
        "has_momentum": has_momentum,
        "has_exhaustion": has_exhaustion,
        "is_exhausted": is_exhausted,
        "exhaustion_candles": candle_details,
    }

    if is_exhausted:
        return EntryQualityResult(
            flag_name="exhaustion_filter",
            passed=False,
            severity="BLOCK",
            reason=(
                f"EXHAUSTION DETECTED — {momentum_count} momentum candles followed by "
                f"{exhaustion_count}/{exhaustion_lookback} small-bodied consolidation candles. "
                f"Move in {direction} direction is exhausted. Market is telling you to WAIT, "
                f"not re-enter. BLOCKED."
            ),
            details=details,
        )

    return EntryQualityResult(
        flag_name="exhaustion_filter",
        passed=True,
        reason=(
            f"No exhaustion pattern — {momentum_count} momentum + "
            f"{exhaustion_count} consolidation candles."
        ),
        details=details,
    )


# ═════════════════════════════════════════════════════════════
# AGGREGATE: Run All 12 Entry Quality Checks
# ═════════════════════════════════════════════════════════════

def run_all_entry_quality_checks(
    df: pd.DataFrame,
    symbol: str,
    direction: str,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    ind_ctx: Optional[dict] = None,
    open_positions: Optional[List[dict]] = None,
) -> dict:
    """
    Run all 10 entry-quality guardrails + return aggregate result.

    BLOCK severity = hard block (trade must NOT execute)
    WARNING severity = soft warning (trade can proceed but quality is questionable)

    Returns:
        {
            "all_passed": bool,
            "passed_count": int,
            "block_count": int,
            "warning_count": int,
            "should_execute": bool,
            "results": [result.to_dict(), ...],
            "block_reason": str | None,
            "warnings": [str, ...],
            "quality_score": int,
        }
    """
    results = []

    # 1. Chasing filter
    results.append(check_chasing_filter(df, symbol, direction))

    # 2. SL swing anchor
    results.append(check_sl_swing_anchor(df, symbol, direction, stop_loss, entry_price))

    # 3. TP structure validation
    results.append(check_tp_structure_validation(df, symbol, direction, take_profit, entry_price))

    # 4. Indecision candles
    results.append(check_indecision_candles(df))

    # 5. Indicator confluence
    results.append(check_indicator_confluence(df, direction, ind_ctx))

    # 6. Round number TP
    results.append(check_round_number_tp(symbol, direction, take_profit, entry_price))

    # 7. Rejection wick at entry (NEW — XAUUSD post-mortem)
    results.append(check_rejection_wick_at_entry(df, direction))

    # 8. Averaging into losers (NEW — XAUUSD post-mortem)
    results.append(check_averaging_into_losers(
        symbol, direction, entry_price, open_positions or []
    ))

    # 9. Fresh high + rejection (NEW — XAUUSD post-mortem, highest-leverage fix)
    results.append(check_fresh_high_rejection(df, symbol, direction, entry_price))

    # 10. TP above unconfirmed spike (NEW — XAUUSD post-mortem)
    results.append(check_tp_above_unconfirmed_spike(df, symbol, direction, take_profit))

    # 11. Opposite-direction stacking guard (NEW — USDJPY post-mortem)
    results.append(check_opposite_direction_stacking(
        symbol, direction, open_positions or []
    ))

    # 12. Exhaustion filter (NEW — USDJPY post-mortem)
    results.append(check_exhaustion_filter(df, direction))

    passed_count = sum(1 for r in results if r.passed)
    block_count = sum(1 for r in results if not r.passed and r.severity == "BLOCK")
    warning_count = sum(1 for r in results if not r.passed and r.severity == "WARNING")
    all_passed = passed_count == len(results)
    should_execute = block_count == 0

    block_reason = next(
        (r.reason for r in results if not r.passed and r.severity == "BLOCK"), None
    )
    warnings = [r.reason for r in results if not r.passed and r.severity == "WARNING"]

    # Quality score: 100 base - (blocks × 25) - (warnings × 10), clamped [0, 100]
    quality_score = max(0, min(100, 100 - (block_count * 25) - (warning_count * 10)))

    return {
        "all_passed":      bool(all_passed),
        "passed_count":    int(passed_count),
        "total_count":     len(results),
        "block_count":     int(block_count),
        "warning_count":   int(warning_count),
        "should_execute":  bool(should_execute),
        "results":         [r.to_dict() for r in results],
        "block_reason":    block_reason,
        "warnings":        warnings,
        "quality_score":   int(quality_score),
    }


# ═════════════════════════════════════════════════════════════
# CLI entry — reproduce the XAUUSD H1 trade analysis
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Reproduce the trade from the XAUUSD post-mortem:
    # XAUUSD H1, BUY 0.01 @ 4130.70, SL 4098.58, TP 4194.27
    # After ~130-point impulsive rally, at fresh high 4146.26 with rejection wick,
    # then averaged into 2 more buys at 4126.08 and 4124.21 as price fell.

    np.random.seed(42)
    n = 100
    dates = pd.date_range("2026-06-30 12:00", periods=n, freq="1h")

    # Build the session:
    # Phase 1: low 3984 → consolidation 3984-4020
    # Phase 2: breakout leg to 4065
    # Phase 3: shallow pullback → impulsive leg to spike high 4146.26
    # Phase 4: pullback to current 4124.72
    close = np.empty(n)
    for i in range(n):
        if i < 20:  # consolidation 3984-4020
            close[i] = 3984 + (i % 4) * 9
        elif i < 40:  # breakout to 4065
            close[i] = 4020 + (i - 20) * 2.25
        elif i < 50:  # shallow pullback
            close[i] = 4065 - (i - 40) * 0.5
        elif i < 75:  # impulsive leg to 4146
            close[i] = 4060 + (i - 50) * 3.4
        else:  # pullback to 4124
            close[i] = 4146 - (i - 75) * 0.88

    close += np.random.randn(n) * 0.5  # small noise

    # Build candles — the spike high candle (i=74) has a long upper wick (rejection)
    opens = close - np.random.randn(n) * 0.3
    highs = np.maximum(close, opens) + abs(np.random.randn(n)) * 0.5
    lows = np.minimum(close, opens) - abs(np.random.randn(n)) * 0.5
    # Force spike high at i=74 with long upper wick (rejection)
    highs[74] = 4146.26  # spike high
    # Last candle (entry time) — small body, slight upper wick (indecision)
    opens[-1] = 4124.5
    close[-1] = 4124.72
    highs[-1] = 4126.0
    lows[-1] = 4123.5

    df = pd.DataFrame({
        "open":   opens,
        "high":   highs,
        "low":    lows,
        "close":  close,
        "volume": np.random.randint(100, 1000, n),
        "rsi":    np.random.uniform(50, 70, n),
        "ema_20": close - 2.0,  # below price (bullish)
    }, index=dates)

    # Existing underwater positions (the 2 earlier buys)
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)
    open_positions = [
        {"pair": "XAUUSD", "direction": "BUY", "entry": 4130.70,
         "current_price": 4124.72, "open_time": (now - timedelta(minutes=30)).isoformat()},
        {"pair": "XAUUSD", "direction": "BUY", "entry": 4126.08,
         "current_price": 4124.72, "open_time": (now - timedelta(minutes=15)).isoformat()},
    ]

    print("=" * 70)
    print("  XAUUSD H1 TRADE POST-MORTEM — Entry Quality Guardrails (10 flags)")
    print("=" * 70)
    print(f"  Trade: BUY 0.01 @ 4130.70 | SL 4098.58 | TP 4194.27")
    print(f"  After 130-pt rally, at fresh high 4146.26 with rejection wick")
    print(f"  + 2 existing underwater BUY positions (averaging into losers)")
    print("=" * 70)
    print()

    result = run_all_entry_quality_checks(
        df=df,
        symbol="XAUUSD",
        direction="BUY",
        entry_price=4130.70,
        stop_loss=4098.58,
        take_profit=4194.27,
        ind_ctx={"rsi": 60, "ema_20": 4122.0},
        open_positions=open_positions,
    )

    print(f"Quality Score: {result['quality_score']}/100")
    print(f"Should Execute: {result['should_execute']}")
    print(f"Passed: {result['passed_count']}/{result['total_count']}")
    print(f"Blocks: {result['block_count']} | Warnings: {result['warning_count']}")
    print()

    if result["block_reason"]:
        print(f"🚫 BLOCK REASON: {result['block_reason']}")
        print()

    if result["warnings"]:
        print("⚠️  WARNINGS:")
        for w in result["warnings"]:
            print(f"  • {w}")
        print()

    print("─" * 70)
    print("DETAILED RESULTS:")
    print("─" * 70)
    for r in result["results"]:
        icon = "✅" if r["passed"] else ("🚫" if r["severity"] == "BLOCK" else "⚠️")
        print(f"\n{icon} {r['flag_name']} ({r['severity']})")
        print(f"   {r['reason']}")
    print()