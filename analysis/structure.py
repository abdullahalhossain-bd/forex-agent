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

    def _detect_bos(self, df: pd.DataFrame, labeled: list[dict]) -> dict:
        """
        Bullish BOS : close > সর্বশেষ confirmed swing high
        Bearish BOS : close < সর্বশেষ confirmed swing low

        Trend continuation signal — গঠন একই দিকে আরও এগোচ্ছে।
        """
        highs = [p for p in labeled if p["kind"] == "high"]
        lows  = [p for p in labeled if p["kind"] == "low"]

        if not highs or not lows:
            return {"event": "NONE", "level": None, "confidence": 0}

        last_high = highs[-1]
        last_low  = lows[-1]
        curr_close = float(df["close"].iloc[-1])

        if curr_close > last_high["price"]:
            confidence = self._bos_confidence(df, last_high["price"], "bullish")
            return {
                "event":      "BULLISH_BOS",
                "level":      last_high["price"],
                "confidence": confidence,
                "note": f"Price broke above swing high {last_high['price']:.5f}",
            }

        if curr_close < last_low["price"]:
            confidence = self._bos_confidence(df, last_low["price"], "bearish")
            return {
                "event":      "BEARISH_BOS",
                "level":      last_low["price"],
                "confidence": confidence,
                "note": f"Price broke below swing low {last_low['price']:.5f}",
            }

        return {"event": "NONE", "level": None, "confidence": 0}

    def _bos_confidence(self, df: pd.DataFrame, level: float, direction: str) -> int:
        """
        Break কতটা decisive — close, level থেকে কতদূর সরে গেছে (ATR-normalized)।
        """
        atr = self._atr_value(df)
        curr_close = float(df["close"].iloc[-1])
        dist = abs(curr_close - level)
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

        last4 = labeled[-4:]
        types = [p["type"] for p in last4]

        # Bullish -> Bearish: ...HH, HL ... then LL appears breaking prior HL
        if structure_bias in ("BULLISH", "RANGING") and "LL" in types:
            ll_idx = types.index("LL")
            prior_hl = [p for p in last4[:ll_idx] if p["type"] == "HL"]
            if prior_hl:
                broken_level = prior_hl[-1]["price"]
                return {
                    "event": "BEARISH_CHOCH",
                    "confidence": 70,
                    "broken_level": broken_level,
                    "note": (
                        f"Bullish HL at {broken_level:.5f} broken — "
                        f"character shifting to bearish"
                    ),
                }

        # Bearish -> Bullish: ...LH, LL ... then HH appears breaking prior LH
        if structure_bias in ("BEARISH", "RANGING") and "HH" in types:
            hh_idx = types.index("HH")
            prior_lh = [p for p in last4[:hh_idx] if p["type"] == "LH"]
            if prior_lh:
                broken_level = prior_lh[-1]["price"]
                return {
                    "event": "BULLISH_CHOCH",
                    "confidence": 70,
                    "broken_level": broken_level,
                    "note": (
                        f"Bearish LH at {broken_level:.5f} broken — "
                        f"character shifting to bullish"
                    ),
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