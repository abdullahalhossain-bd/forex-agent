# analysis/structure.py  —  Day 61 | Market Structure Engine
# ============================================================
# Institutional price-action foundation।
#
# এই module দেখে:
#   1. Swing High / Swing Low (HH, HL, LH, LL)
#   2. Overall structure (BULLISH / BEARISH / RANGING)
#   3. Break of Structure (BOS)
#   4. Change of Character (CHoCH)
#   5. Displacement (strong institutional move)
#
# এটা Day 44-এর mtf_analyzer._detect_bos/_detect_choch থেকে আলাদা —
# এখানে swing label (HH/HL/LH/LL) সহ একটা সম্পূর্ণ independent engine,
# যেটা smart_money.py SMC pipeline-এর foundation হিসেবে ব্যবহার হবে।
# ============================================================

import numpy as np
import pandas as pd
from utils.logger import get_logger

log = get_logger("structure_engine")


class MarketStructureEngine:
    """
    Usage:
        engine  = MarketStructureEngine(swing_window=5)
        result  = engine.analyze(df)        # df: OHLC(+atr) DataFrame
        ctx     = engine.get_ai_context(result)
    """

    def __init__(self, swing_window: int = 5):
        """
        swing_window : কতটা candle দুই পাশে দেখবে swing high/low ধরতে।
                       ছোট timeframe (M5/M15) → ছোট window (3-5)
                       বড় timeframe (H1/H4)   → বড় window (5-10)
        """
        self.swing_window = swing_window

    # ═══════════════════════════════════════════════════════
    # MAIN METHOD
    # ═══════════════════════════════════════════════════════

    def analyze(self, df: pd.DataFrame) -> dict:
        """
        Full market structure pipeline:
          swing points -> label (HH/HL/LH/LL) -> structure bias
          -> BOS -> CHoCH -> displacement
        """
        w = self.swing_window
        if len(df) < w * 4 + 10:
            return self._empty_result("Insufficient data for structure analysis")

        swing_points = self._find_swing_points(df)
        if len(swing_points) < 3:
            return self._empty_result("Not enough swing points detected")

        labeled = self._label_swings(swing_points)
        structure_bias = self._determine_structure(labeled)

        bos   = self._detect_bos(df, labeled)
        choch = self._detect_choch(df, labeled, structure_bias)
        displacement = self._detect_displacement(df)

        result = {
            "valid":          True,
            "structure":      structure_bias,
            "swing_points":   labeled,
            "bos":            bos,
            "choch":          choch,
            "displacement":   displacement,
            "trend_phase":    self._detect_trend_phase(df, labeled, structure_bias),
            "last_price":     round(float(df["close"].iloc[-1]), 5),
        }

        log.info(
            f"[Structure] Bias={structure_bias} | BOS={bos['event']} | "
            f"CHoCH={choch['event']} | Displacement={displacement['detected']}"
        )
        return result

    # ═══════════════════════════════════════════════════════
    # STEP 1: SWING POINT DETECTION
    # ═══════════════════════════════════════════════════════

    def _find_swing_points(self, df: pd.DataFrame) -> list[dict]:
        """
        Local swing high/low খুঁজো (fractal-style: দুই পাশের window-এর
        চেয়ে বেশি/কম)। Output chronological order-এ — type এখনো label
        করা হয়নি (raw high/low), পরের step-এ HH/HL/LH/LL label হবে।
        """
        highs = df["high"].values
        lows  = df["low"].values
        n     = len(df)
        w     = self.swing_window

        raw_points = []
        for i in range(w, n - w):
            window_high = highs[i - w: i + w + 1]
            window_low  = lows[i - w: i + w + 1]

            if highs[i] >= window_high.max() - 1e-12:
                raw_points.append({"index": i, "price": float(highs[i]), "kind": "high"})
            elif lows[i] <= window_low.min() + 1e-12:
                raw_points.append({"index": i, "price": float(lows[i]), "kind": "low"})

        # Consecutive same-kind points -> keep most extreme one only
        cleaned = []
        for p in raw_points:
            if cleaned and cleaned[-1]["kind"] == p["kind"]:
                if p["kind"] == "high" and p["price"] > cleaned[-1]["price"]:
                    cleaned[-1] = p
                elif p["kind"] == "low" and p["price"] < cleaned[-1]["price"]:
                    cleaned[-1] = p
            else:
                cleaned.append(p)

        return cleaned

    # ═══════════════════════════════════════════════════════
    # STEP 2: LABEL SWINGS (HH / HL / LH / LL)
    # ═══════════════════════════════════════════════════════

    def _label_swings(self, points: list[dict]) -> list[dict]:
        """
        প্রতিটা high-কে আগের high-এর সাথে, প্রতিটা low-কে আগের low-এর
        সাথে compare করে HH/LH (highs) এবং HL/LL (lows) label করো।
        """
        labeled = []
        last_high = None
        last_low  = None

        for p in points:
            if p["kind"] == "high":
                if last_high is None:
                    label = "H"   # প্রথম high, এখনো compare করার কিছু নেই
                elif p["price"] > last_high:
                    label = "HH"
                else:
                    label = "LH"
                last_high = p["price"]
            else:
                if last_low is None:
                    label = "L"
                elif p["price"] > last_low:
                    label = "HL"
                else:
                    label = "LL"
                last_low = p["price"]

            labeled.append({
                "index": p["index"],
                "price": round(p["price"], 5),
                "kind":  p["kind"],
                "type":  label,
            })

        return labeled

    # ═══════════════════════════════════════════════════════
    # STEP 3: OVERALL STRUCTURE BIAS
    # ═══════════════════════════════════════════════════════

    def _determine_structure(self, labeled: list[dict]) -> str:
        """
        সাম্প্রতিক swing labels দেখে overall bias বলো।

        BULLISH : HH + HL pattern dominant
        BEARISH : LH + LL pattern dominant
        RANGING : mixed / no clear sequence
        """
        recent = labeled[-6:] if len(labeled) >= 6 else labeled
        bullish_votes = sum(1 for p in recent if p["type"] in ("HH", "HL"))
        bearish_votes = sum(1 for p in recent if p["type"] in ("LH", "LL"))

        if bullish_votes > bearish_votes and bullish_votes >= 2:
            return "BULLISH"
        if bearish_votes > bullish_votes and bearish_votes >= 2:
            return "BEARISH"
        return "RANGING"

    # ═══════════════════════════════════════════════════════
    # STEP 4: BREAK OF STRUCTURE (BOS)
    # ═══════════════════════════════════════════════════════

    def _detect_bos(
        self,
        df: pd.DataFrame,
        labeled: list[dict],
        max_age_bars: int | None = None,
    ) -> dict:
        """
        Bullish BOS : close (primary) or wick (secondary) breaks above the
                      most recent CONFIRMED swing high.
        Bearish BOS : mirror on swing lows.

        Two fixes applied here:

        1. PERSISTENCE — previously this only ever inspected the LAST bar,
           so a genuine break followed by a normal pullback candle reported
           NONE the very next bar (identical bug to the one already fixed
           in mtf_analyzer._detect_bos). Now the whole df is walked
           bar-by-bar and the *latest* BOS event is returned, so structure
           persists through a pullback instead of vanishing.

        2. NEAR-BOS FALSE POSITIVE REMOVED — the old "tertiary" branch
           returned a fake BULLISH_BOS/BEARISH_BOS (confidence 35) just
           because price was within 0.2% of a swing level. Proximity to a
           level is not a break — it's never emitted as `event` anymore.
           It's now only ever a non-overriding `near_bos` diagnostic on
           the current bar, and only shown when there's no real BOS
           already in force (so it can never mask genuine structure).

        LOOK-AHEAD SAFETY: a swing pivot from `labeled` is only registered
        into the walk once its confirmation window has actually closed
        (`pivot_index + swing_window`), matching the same rule
        `_find_swing_points` uses to decide a pivot exists at all.

        Args:
            max_age_bars: optional — if set, a BOS older than this many
                bars is reported as NONE (stale). None = no cutoff
                (structure stays valid until invalidated, which matches
                real market-structure semantics).
        """
        highs = [p for p in labeled if p["kind"] == "high"]
        lows  = [p for p in labeled if p["kind"] == "low"]

        if not highs or not lows:
            return {"event": "NONE", "level": None, "confidence": 0, "near_bos": False}

        w = self.swing_window
        n = len(df)
        closes    = df["close"].values
        bar_highs = df["high"].values
        bar_lows  = df["low"].values

        # A pivot becomes "known" only once its right-side confirmation
        # window has closed — register it then, not at its own index.
        hi_events = sorted((p["index"] + w, p) for p in highs)
        lo_events = sorted((p["index"] + w, p) for p in lows)

        last_high = last_low = None
        latest = {
            "event": "NONE", "level": None, "confidence": 0,
            "note": "No structural break detected", "near_bos": False,
        }
        hi_ptr = lo_ptr = 0

        for i in range(n):
            while hi_ptr < len(hi_events) and hi_events[hi_ptr][0] <= i:
                last_high = hi_events[hi_ptr][1]
                hi_ptr += 1
            while lo_ptr < len(lo_events) and lo_events[lo_ptr][0] <= i:
                last_low = lo_events[lo_ptr][1]
                lo_ptr += 1

            if last_high is None or last_low is None:
                continue

            c, h, l = closes[i], bar_highs[i], bar_lows[i]

            if c > last_high["price"]:
                conf = self._bos_confidence(df, i, last_high["price"], "bullish")
                latest = {
                    "event": "BULLISH_BOS", "level": last_high["price"], "confidence": conf,
                    "broke_at": i, "bars_ago": n - 1 - i,
                    "note": f"Close broke above swing high {last_high['price']:.5f}",
                    "near_bos": False,
                }
            elif c < last_low["price"]:
                conf = self._bos_confidence(df, i, last_low["price"], "bearish")
                latest = {
                    "event": "BEARISH_BOS", "level": last_low["price"], "confidence": conf,
                    "broke_at": i, "bars_ago": n - 1 - i,
                    "note": f"Close broke below swing low {last_low['price']:.5f}",
                    "near_bos": False,
                }
            elif h > last_high["price"] * 1.001:  # 0.1% wick-through
                conf = max(self._bos_confidence(df, i, last_high["price"], "bullish"), 55)
                latest = {
                    "event": "BULLISH_BOS", "level": last_high["price"], "confidence": conf,
                    "broke_at": i, "bars_ago": n - 1 - i,
                    "note": f"High wick broke above swing high {last_high['price']:.5f} (wicking BOS)",
                    "near_bos": False,
                }
            elif l < last_low["price"] * 0.999:
                conf = max(self._bos_confidence(df, i, last_low["price"], "bearish"), 55)
                latest = {
                    "event": "BEARISH_BOS", "level": last_low["price"], "confidence": conf,
                    "broke_at": i, "bars_ago": n - 1 - i,
                    "note": f"Low wick broke below swing low {last_low['price']:.5f} (wicking BOS)",
                    "near_bos": False,
                }
            # (Old tertiary "near level → fake BOS" branch removed —
            # proximity is handled separately by `_near_bos_flag` below,
            # and only ever as a diagnostic, never as `event`.)

        if max_age_bars is not None and latest["event"] != "NONE" and latest["bars_ago"] > max_age_bars:
            latest = {
                "event": "NONE", "level": None, "confidence": 0,
                "note": f"Last BOS was {latest['bars_ago']} bars ago (> max_age_bars={max_age_bars}) — treated as stale",
                "near_bos": False,
            }

        if latest["event"] == "NONE":
            latest["near_bos"] = self._near_bos_flag(df, last_high, last_low)

        return latest

    def _near_bos_flag(self, df: pd.DataFrame, last_high: dict | None, last_low: dict | None,
                        threshold: float = 0.002):
        """
        Proximity heads-up ONLY. "Price is close to a swing level" is not
        a Break of Structure — this must never be surfaced as `event`
        (that was the bug). Returns False, or a dict describing which
        side price is approaching, purely as extra context for a caller
        that wants "signal building" awareness.
        """
        curr_close = float(df["close"].iloc[-1])
        if last_high is not None:
            gap = (last_high["price"] - curr_close) / last_high["price"]
            if 0 < gap < threshold:
                return {"direction": "bullish", "level": last_high["price"], "distance_pct": round(gap * 100, 3)}
        if last_low is not None:
            gap = (curr_close - last_low["price"]) / last_low["price"]
            if 0 < gap < threshold:
                return {"direction": "bearish", "level": last_low["price"], "distance_pct": round(gap * 100, 3)}
        return False

    def _bos_confidence(self, df: pd.DataFrame, idx: int, level: float, direction: str) -> int:
        """
        Break কতটা decisive — close, level থেকে কতদূর সরে গেছে (ATR-normalized),
        evaluated AT BAR `idx` (not necessarily the last row) — needed so a
        persisted, older BOS still gets a correctly-dated confidence score
        instead of being scored against today's close.
        """
        atr = self._atr_value(df.iloc[:idx + 1])
        close_at = float(df["close"].iloc[idx])
        dist = abs(close_at - level)
        ratio = dist / atr if atr else 0
        confidence = int(min(95, 50 + ratio * 25))
        return confidence

    # ═══════════════════════════════════════════════════════
    # STEP 5: CHANGE OF CHARACTER (CHoCH)
    # ═══════════════════════════════════════════════════════

    def _detect_choch(self, df: pd.DataFrame, labeled: list[dict], structure_bias: str) -> dict:
        """
        CHoCH = trend reversal signal।

        Bullish structure-এ থাকার সময় একটা HL break হয়ে নতুন LL
        তৈরি হলে -> BEARISH_CHOCH (Bullish -> Bearish reversal শুরু)

        Bearish structure-এ থাকার সময় একটা LH break হয়ে নতুন HH
        তৈরি হলে -> BULLISH_CHOCH (Bearish -> Bullish reversal শুরু)
        """
        if len(labeled) < 4:
            return {"event": "NONE", "confidence": 0, "note": "Insufficient swings"}

        # Check last 8 swings for better pattern detection (not just last 4)
        lookback_swings = min(8, len(labeled))
        recent = labeled[-lookback_swings:]
        types = [p["type"] for p in recent]

        # Bullish -> Bearish: ...HH, HL ... then LL appears breaking prior HL
        if structure_bias in ("BULLISH", "RANGING"):
            # Look for LL in recent swings
            if "LL" in types:
                ll_idx = len(types) - 1 - types[::-1].index("LL")  # Find last LL
                # Find most recent HL before this LL
                prior_hl = [p for p in recent[:ll_idx] if p["type"] == "HL"]
                if prior_hl:
                    broken_level = prior_hl[-1]["price"]
                    curr_close = float(df["close"].iloc[-1])
                    # Check if we're actually below the HL level
                    if curr_close < broken_level:
                        confidence = 75 if curr_close < broken_level * 0.99 else 60
                        return {
                            "event": "BEARISH_CHOCH",
                            "confidence": confidence,
                            "broken_level": broken_level,
                            "note": (
                                f"Bullish HL at {broken_level:.5f} broken — "
                                f"character shifting to bearish"
                            ),
                        }
            
            # Secondary detection: check for sustained bearish pattern (LH, LL sequence)
            if len(types) >= 3:
                if types[-2] == "LH" and types[-1] == "LL":
                    broken_level = recent[-3]["price"] if len(recent) >= 3 and recent[-3]["type"] == "HL" else None
                    if broken_level:
                        confidence = 65
                        return {
                            "event": "BEARISH_CHOCH",
                            "confidence": confidence,
                            "broken_level": broken_level,
                            "note": f"Bearish pattern LH→LL confirmed at {broken_level:.5f}",
                        }

        # Bearish -> Bullish: ...LH, LL ... then HH appears breaking prior LH
        if structure_bias in ("BEARISH", "RANGING"):
            # Look for HH in recent swings
            if "HH" in types:
                hh_idx = len(types) - 1 - types[::-1].index("HH")  # Find last HH
                # Find most recent LH before this HH
                prior_lh = [p for p in recent[:hh_idx] if p["type"] == "LH"]
                if prior_lh:
                    broken_level = prior_lh[-1]["price"]
                    curr_close = float(df["close"].iloc[-1])
                    # Check if we're actually above the LH level
                    if curr_close > broken_level:
                        confidence = 75 if curr_close > broken_level * 1.01 else 60
                        return {
                            "event": "BULLISH_CHOCH",
                            "confidence": confidence,
                            "broken_level": broken_level,
                            "note": (
                                f"Bearish LH at {broken_level:.5f} broken — "
                                f"character shifting to bullish"
                            ),
                        }
            
            # Secondary detection: check for sustained bullish pattern (HH, HL sequence)
            if len(types) >= 3:
                if types[-2] == "HH" and types[-1] == "HL":
                    broken_level = recent[-3]["price"] if len(recent) >= 3 and recent[-3]["type"] == "LH" else None
                    if broken_level:
                        confidence = 65
                        return {
                            "event": "BULLISH_CHOCH",
                            "confidence": confidence,
                            "broken_level": broken_level,
                            "note": f"Bullish pattern HH→HL confirmed at {broken_level:.5f}",
                        }

        return {"event": "NONE", "confidence": 0, "note": "No character change detected"}

    # ═══════════════════════════════════════════════════════
    # STEP 6: DISPLACEMENT DETECTION
    # ═══════════════════════════════════════════════════════

    def _detect_displacement(self, df: pd.DataFrame, lookback: int = 10) -> dict:
        """
        Displacement = ছোট ছোট candle-এর পরে একটা বড় impulsive candle,
        যেটা institutional ("real money") entry-র signature ধরা হয়।

        Rule: candle body, পূর্ববর্তী N candle-এর average body-র
        নির্দিষ্ট গুণের বেশি হলে displacement।
        """
        if len(df) < lookback + 1:
            return {"detected": False, "direction": "NONE", "note": "Insufficient data"}

        opens  = df["open"].values
        closes = df["close"].values

        recent_bodies = np.abs(closes[-(lookback + 1):-1] - opens[-(lookback + 1):-1])
        avg_body = float(np.mean(recent_bodies)) if len(recent_bodies) else 0.0

        last_body = float(closes[-1] - opens[-1])

        if avg_body == 0:
            return {"detected": False, "direction": "NONE", "note": "Flat market"}

        ratio = abs(last_body) / avg_body

        if ratio >= 2.5:
            direction = "BULLISH" if last_body > 0 else "BEARISH"
            return {
                "detected": True,
                "direction": direction,
                "ratio": round(ratio, 2),
                "note": (
                    f"{direction} displacement candle — body {ratio:.1f}x "
                    f"average. Real money likely entered."
                ),
            }

        return {"detected": False, "direction": "NONE", "ratio": round(ratio, 2), "note": "No displacement"}

    # ═══════════════════════════════════════════════════════
    # Day 97+ Candlestick Bible Pages 54-55: Trend Phase Detection
    # ═══════════════════════════════════════════════════════

    def _detect_trend_phase(self, df: pd.DataFrame, labeled: list, structure_bias: str) -> dict:
        """Candlestick Bible Pages 54-55: Impulsive vs Retracement phase.

        Book: "Professional traders enter at the START of an impulsive move,
        not during a retracement."

        This method classifies the current price position:
          - 'impulsive' — strong trend-direction move in progress (entry late/risky)
          - 'retracement' — counter-trend pullback (potential entry zone)
          - 'retracement_ending' — pullback stalling near S/R (entry window OPEN)
          - 'unknown' — no clear phase

        Book Page 55 failure mode: "entering at the beginning of a retracement
        (mistaking it for impulsive) gets you trapped."
        """
        result = {
            "phase": "unknown",
            "entry_window": "closed",
            "note": "No clear trend phase detected",
        }

        if structure_bias not in ("BULLISH", "BEARISH") or len(labeled) < 3:
            return result

        closes = df["close"].values
        n = len(df)

        # Get recent swing points
        recent_swings = labeled[-4:]
        prev_swing = recent_swings[-2] if len(recent_swings) >= 2 else None

        # Round-21 audit fix follow-up: the Round-21 cleanup removed this
        # computation as "unused," but it IS used below (elif recent_change
        # < 0 / > 0) to tell a pullback apart from trend continuation. That
        # earlier removal caused a live NameError every time displacement
        # was NOT detected (i.e. most cycles), silently degrading MTF
        # structure analysis for the rest of that cycle (caught by an
        # outer try/except in analysis_agent, but mtf_structure_ctx came
        # back empty). Restored: recent_change = short-lookback price
        # delta, used only to classify pullback (opposite sign to trend)
        # vs continuation (same sign).
        recent_lookback = min(3, n - 1)
        recent_change = (
            float(closes[-1] - closes[-1 - recent_lookback])
            if recent_lookback > 0 else 0.0
        )

        # Check if displacement (impulsive) is happening now
        displacement = self._detect_displacement(df)
        is_impulsive_now = displacement.get("detected", False)

        if structure_bias == "BULLISH":
            if is_impulsive_now and displacement["direction"] == "BULLISH":
                result.update({
                    "phase": "impulsive",
                    "entry_window": "closed",
                    "note": "Bullish impulsive move in progress — entering late is risky",
                })
            elif recent_change < 0:
                # Price pulling back (counter-trend) → retracement
                # Check if near a swing low (potential support)
                near_support = False
                if prev_swing and prev_swing["kind"] == "low":
                    near_support = abs(float(closes[-1]) - prev_swing["price"]) < (prev_swing["price"] * 0.002)

                if near_support:
                    result.update({
                        "phase": "retracement_ending",
                        "entry_window": "open",
                        "note": "Bullish retracement ending near swing-low support — entry window OPEN",
                    })
                else:
                    result.update({
                        "phase": "retracement",
                        "entry_window": "wait",
                        "note": "Bullish retracement in progress — wait for it to end at support",
                    })
            else:
                result.update({
                    "phase": "impulsive",
                    "entry_window": "closed",
                    "note": "Bullish trend continuing — wait for next retracement",
                })

        elif structure_bias == "BEARISH":
            if is_impulsive_now and displacement["direction"] == "BEARISH":
                result.update({
                    "phase": "impulsive",
                    "entry_window": "closed",
                    "note": "Bearish impulsive move in progress — entering late is risky",
                })
            elif recent_change > 0:
                # Price bouncing up (counter-trend) → retracement
                near_resistance = False
                if prev_swing and prev_swing["kind"] == "high":
                    near_resistance = abs(float(closes[-1]) - prev_swing["price"]) < (prev_swing["price"] * 0.002)

                if near_resistance:
                    result.update({
                        "phase": "retracement_ending",
                        "entry_window": "open",
                        "note": "Bearish retracement ending near swing-high resistance — entry window OPEN",
                    })
                else:
                    result.update({
                        "phase": "retracement",
                        "entry_window": "wait",
                        "note": "Bearish retracement in progress — wait for it to end at resistance",
                    })
            else:
                result.update({
                    "phase": "impulsive",
                    "entry_window": "closed",
                    "note": "Bearish trend continuing — wait for next retracement",
                })

        return result

    # ═══════════════════════════════════════════════════════
    # AI CONTEXT
    # ═══════════════════════════════════════════════════════

    def get_ai_context(self, result: dict) -> dict:
        if not result.get("valid"):
            return {
                "structure_valid":   False,
                "structure_bias":    "NEUTRAL",
                "structure_bos":     "NONE",
                "structure_choch":   "NONE",
                "displacement":      False,
                "displacement_dir":  "NONE",
                "swing_points":      [],
            }

        bos   = result.get("bos", {})
        choch = result.get("choch", {})
        disp  = result.get("displacement", {})

        return {
            "structure_valid":   True,
            "structure_bias":    result.get("structure"),
            "structure_bos":     bos.get("event", "NONE"),
            "structure_bos_level": bos.get("level"),
            "structure_bos_confidence": bos.get("confidence", 0),
            "structure_choch":   choch.get("event", "NONE"),
            "structure_choch_confidence": choch.get("confidence", 0),
            "displacement":      disp.get("detected", False),
            "displacement_dir":  disp.get("direction", "NONE"),
            "swing_points":      result.get("swing_points", [])[-6:],
        }

    # ═══════════════════════════════════════════════════════
    # UTILITIES
    # ═══════════════════════════════════════════════════════

    def _atr_value(self, df: pd.DataFrame, period: int = 14) -> float:
        if "atr" in df.columns:
            val = df["atr"].iloc[-1]
            if not np.isnan(val):
                return float(val)
        highs  = df["high"].values[-period:]
        lows   = df["low"].values[-period:]
        closes = df["close"].values[-period:]
        trs = [
            max(h - l, abs(h - c), abs(l - c))
            for h, l, c in zip(highs[1:], lows[1:], closes[:-1])
        ]
        return float(np.mean(trs)) if trs else 0.0001

    def _empty_result(self, reason: str) -> dict:
        return {
            "valid": False, "reason": reason,
            "structure": "NEUTRAL", "swing_points": [],
            "bos": {"event": "NONE", "level": None, "confidence": 0},
            "choch": {"event": "NONE", "confidence": 0},
            "displacement": {"detected": False, "direction": "NONE"},
        }

    # ═══════════════════════════════════════════════════════
    # PRINT SUMMARY
    # ═══════════════════════════════════════════════════════

    def print_summary(self, result: dict) -> None:
        bar = "═" * 56
        log.info(bar)
        log.info("  🏛️  MARKET STRUCTURE ENGINE  (Day 61)")
        log.info(bar)

        if not result.get("valid"):
            log.info(f"  ⚠️  {result.get('reason', 'No structure detected')}")
            log.info(bar)
            return

        icon = {"BULLISH": "🟢", "BEARISH": "🔴", "RANGING": "🟡"}.get(result["structure"], "⚪")
        log.info(f"  Structure    : {icon} {result['structure']}")

        bos = result["bos"]
        log.info(f"  BOS          : {bos['event']}" + (
            f"  @ {bos['level']}  (conf {bos['confidence']}%)" if bos["event"] != "NONE" else ""
        ))

        choch = result["choch"]
        log.info(f"  CHoCH        : {choch['event']}" + (
            f"  (conf {choch['confidence']}%)" if choch["event"] != "NONE" else ""
        ))

        disp = result["displacement"]
        if disp.get("detected"):
            log.info(f"  Displacement : ✅ {disp['direction']}  ({disp.get('ratio')}x avg body)")
        else:
            log.info("  Displacement : ❌ None")

        log.info("")
        log.info("  ── Recent Swing Points ──")
        for p in result["swing_points"][-6:]:
            arrow = "▲" if p["kind"] == "high" else "▼"
            log.info(f"  {arrow} {p['type']:<3}  @ {p['price']}")

        log.info(bar)


# ═══════════════════════════════════════════════════════════
# QUICK RUN — Direct test
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    from data.fetcher import DataFetcher
    from data.indicators import Indicators

    fetcher = DataFetcher()
    ind     = Indicators()

    df = fetcher.fetch_ohlcv("EURUSD", "1h", limit=200)
    if df is not None:
        df = ind.add_all(df)

        engine = MarketStructureEngine(swing_window=5)
        result = engine.analyze(df)
        engine.print_summary(result)

        ctx = engine.get_ai_context(result)
        print("\nAI Context:")
        for k, v in ctx.items():
            print(f"  {k:<28}: {v}")