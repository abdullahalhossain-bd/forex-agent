# analysis/smc_advanced.py  —  Day 87 | Advanced SMC Concepts
# ============================================================
# এই module আপনার বর্তমান SMC suite-এ যে ২টা professional-level
# concept missing ছিল সেগুলো যোগ করে:
#
#   1. Mitigation Block
#   2. Inducement (Buy-Side / Sell-Side Liquidity)
#
# ── ১. Mitigation Block ──
# যখন একটা Order Block (OB) ভাঙা যায় (price তাকে violate করে),
# কিন্তু তারপর price reverse করে ও সেই ভাঙা zone-কে retest করে —
# সেই zone তখন Mitigation Block।
# এটা এমন একটা level যেখানে institutional order গুলো "mitigate" হয়েছে
# (অর্থাৎ partial close হয়েছে), এবং পরবর্তী retest-এ strong reversal দেয়।
#
# Detection:
#   - Find a broken OB (price closed beyond it then reversed)
#   - Mark that zone as Mitigation Block
#   - When price returns to retest it → reversal signal
#
# ── ২. Inducement ──
# Inducement হলো "trap" level — ছোট swing high/low যেটা retail
# trader-দের breakout ট্রেড নিতে প্রলুব্ধ করে, তারপর price reverse
# করে এবং সত্যিকারের liquidity sweep করে।
#
# Two types:
#   - Buy-Side Inducement (BSI): ছোট swing high যেটা break হলে
#     buy-side liquidity grab করে, তারপর bearish reversal।
#   - Sell-Side Inducement (SSI): ছোট swing low যেটা break হলে
#     sell-side liquidity grab করে, তারপর bullish reversal।
#
# Output:
#   {
#     "mitigation_blocks": [ {type, zone_top, zone_bottom, status, note}, ... ],
#     "inducements":       [ {type, level, swept, note}, ... ],
#     "active_signals":    [ "MITIGATION_BULLISH", "INDUCEMENT_BEARISH", ... ],
#     "bias":              "BULLISH" | "BEARISH" | "NEUTRAL",
#     "signal":            "BUY" | "SELL" | "WAIT",
#     "note":              str
#   }
# ============================================================

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("smc_advanced_engine")


class SMCAdvancedEngine:
    """
    Mitigation Block + Inducement detector।
    OrderBlockDetector ও BreakerBlockDetector এর complementary।
    """

    def __init__(
        self,
        impulse_atr_mult:    float = 1.5,
        ob_lookback:         int   = 3,
        max_zones:           int   = 10,
        mitigation_proximity: float = 0.3,   # ATR fraction for "retest"
        inducement_window:   int   = 5,
    ):
        self.impulse_atr_mult     = impulse_atr_mult
        self.ob_lookback          = ob_lookback
        self.max_zones            = max_zones
        self.mitigation_proximity = mitigation_proximity
        self.inducement_window    = inducement_window

    # ═══════════════════════════════════════════════════════
    # MAIN ENTRY
    # ═══════════════════════════════════════════════════════

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        df-এ high/low/close এবং atr লাগবে (indicators.add_all করা থাকলে কাজ করবে)।
        """
        if df is None or len(df) < 30:
            return self._empty_result("Insufficient data")

        if "atr" not in df.columns:
            df = df.copy()
            df["atr"] = self._compute_atr(df)

        # Step 1: Find all candidate Order Blocks
        obs = self._find_order_blocks(df)

        # Step 2: Find Mitigation Blocks (broken OBs that got retested)
        mitigation_blocks = self._find_mitigation_blocks(df, obs)

        # Step 3: Find Inducement levels (small swing points that get swept)
        inducements = self._find_inducements(df)

        # Step 4: Build active signals
        active_signals = []
        for mb in mitigation_blocks:
            if mb["status"] == "ACTIVE_RETEST":
                if mb["type"] == "BULLISH_MITIGATION":
                    active_signals.append("MITIGATION_BULLISH")
                else:
                    active_signals.append("MITIGATION_BEARISH")

        for ind in inducements:
            if ind["swept"]:
                if ind["type"] == "SELL_SIDE_INDUCEMENT":
                    active_signals.append("INDUCEMENT_BULLISH")  # SSL sweep = bullish reversal
                else:
                    active_signals.append("INDUCEMENT_BEARISH")

        # Step 5: Bias + signal
        bias, signal, note = self._bias_and_signal(
            mitigation_blocks, inducements, active_signals
        )

        result = {
            "valid":            True,
            "mitigation_blocks": mitigation_blocks,
            "inducements":      inducements,
            "active_signals":   active_signals,
            "bias":             bias,
            "signal":           signal,
            "note":             note,
        }

        log.info(
            f"[SMCAdvanced] MBs={len(mitigation_blocks)} | "
            f"Inducements={len(inducements)} | "
            f"active={len(active_signals)} | bias={bias} | signal={signal}"
        )
        return result

    # ═══════════════════════════════════════════════════════
    # ORDER BLOCK DETECTION (simplified, for internal use)
    # ═══════════════════════════════════════════════════════

    def _find_order_blocks(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        এটা analysis/order_block.py এর simplified version। শুধু internal
        use করার জন্য — mitigation detection এর জন্য zone info লাগে।
        """
        opens  = df["open"].values
        closes = df["close"].values
        highs  = df["high"].values
        lows   = df["low"].values
        atrs   = df["atr"].values
        n      = len(df)

        obs: List[Dict[str, Any]] = []

        for i in range(5, n - 2):
            atr = atrs[i]
            if np.isnan(atr) or atr == 0:
                continue

            body = closes[i] - opens[i]
            if abs(body) < atr * self.impulse_atr_mult:
                continue

            is_bullish_impulse = body > 0
            ob_idx = None
            for j in range(i - 1, max(i - 1 - self.ob_lookback, -1), -1):
                c_body = closes[j] - opens[j]
                if is_bullish_impulse and c_body < 0:
                    ob_idx = j
                    break
                if not is_bullish_impulse and c_body > 0:
                    ob_idx = j
                    break

            if ob_idx is None:
                continue

            zone_top    = float(highs[ob_idx])
            zone_bottom = float(lows[ob_idx])
            ob_type     = "BULLISH_OB" if is_bullish_impulse else "BEARISH_OB"

            obs.append({
                "ob_idx":      ob_idx,
                "impulse_idx": i,
                "type":        ob_type,
                "zone_top":    zone_top,
                "zone_bottom": zone_bottom,
                "zone_mid":    round((zone_top + zone_bottom) / 2, 5),
                "broken":      False,
                "mitigated":   False,
            })

            if len(obs) >= self.max_zones * 3:
                break

        return obs

    # ═══════════════════════════════════════════════════════
    # MITIGATION BLOCK DETECTION
    # ═══════════════════════════════════════════════════════

    def _find_mitigation_blocks(
        self, df: pd.DataFrame, obs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        একটা OB-কে খুঁজে বের করো যেটা:
          1. Price তাকে break করেছে (closed beyond)
          2. তারপর reverse করেছে
          3. এখন zone-এর কাছে এসে পৌঁছেছে → active retest
        """
        if not obs:
            return []

        closes = df["close"].values
        highs  = df["high"].values
        lows   = df["low"].values
        atrs   = df["atr"].values
        last_close = float(closes[-1])
        last_atr   = float(atrs[-1]) if not np.isnan(atrs[-1]) else 1e-5

        mitigation_blocks: List[Dict[str, Any]] = []

        for ob in obs:
            ob_top    = ob["zone_top"]
            ob_bot    = ob["zone_bottom"]
            ob_type   = ob["type"]
            imp_idx   = ob["impulse_idx"]

            # Check if OB was broken after impulse
            broken_idx = None
            broken_dir = None
            for k in range(imp_idx + 1, len(df)):
                # Bullish OB broken if price closes BELOW zone_bottom
                if ob_type == "BULLISH_OB" and closes[k] < ob_bot:
                    broken_idx = k
                    broken_dir = "DOWN"
                    break
                # Bearish OB broken if price closes ABOVE zone_top
                if ob_type == "BEARISH_OB" and closes[k] > ob_top:
                    broken_idx = k
                    broken_dir = "UP"
                    break

            if broken_idx is None:
                continue   # OB still intact, not a mitigation block

            # Check if price has reversed since the break and now retesting
            post_break_closes = closes[broken_idx + 1:]
            if len(post_break_closes) < 2:
                continue

            # Did price reverse?
            if broken_dir == "DOWN":
                # Need price to recover back toward zone
                reversed = float(post_break_closes[-1]) > float(post_break_closes[0])
                # Currently retesting zone?
                near_zone = abs(last_close - ob_top) <= last_atr * 2 or \
                            abs(last_close - ob_bot) <= last_atr * 2
                retest_active = last_close >= ob_bot - last_atr * 0.5 and last_close <= ob_top + last_atr * 0.5

                mb_type = "BULLISH_MITIGATION"
            else:
                reversed = float(post_break_closes[-1]) < float(post_break_closes[0])
                near_zone = abs(last_close - ob_top) <= last_atr * 2 or \
                            abs(last_close - ob_bot) <= last_atr * 2
                retest_active = last_close <= ob_top + last_atr * 0.5 and last_close >= ob_bot - last_atr * 0.5

                mb_type = "BEARISH_MITIGATION"

            # Status
            if retest_active and reversed:
                status = "ACTIVE_RETEST"
            elif reversed:
                status = "REVERSED_NO_RETEST"
            else:
                status = "BROKEN_NOT_REVERSED"

            mitigation_blocks.append({
                "type":          mb_type,
                "zone_top":      round(ob_top, 5),
                "zone_bottom":   round(ob_bot, 5),
                "zone_mid":      round((ob_top + ob_bot) / 2, 5),
                "broken_at":     broken_idx,
                "broken_dir":    broken_dir,
                "status":        status,
                "note":          self._mb_note(mb_type, status),
            })

            if len(mitigation_blocks) >= self.max_zones:
                break

        return mitigation_blocks

    def _mb_note(self, mb_type: str, status: str) -> str:
        if status == "ACTIVE_RETEST":
            return f"{mb_type} retest active — reversal likely"
        if status == "REVERSED_NO_RETEST":
            return f"{mb_type} reversed but not yet retested"
        return f"{mb_type} broken, no reversal yet"

    # ═══════════════════════════════════════════════════════
    # INDUCEMENT DETECTION
    # ═══════════════════════════════════════════════════════

    def _find_inducements(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        ছোট swing high/low (inducement) খুঁজে বের করো যেগুলো
        পরবর্তী candle-এ break + reverse হয়েছে (swept)।

        BSI = Buy-Side Inducement (small swing high, swept → bearish)
        SSI = Sell-Side Inducement (small swing low, swept → bullish)

        Swing detection window: শুধু পূর্ববর্তী w candle + পরের 1 candle
        (পরের 1 candle হলো confirmation candle, sweep-টা তার পরে ঘটে)।
        এটা নিশ্চিত করে যে sweep candle-টা নিজে swing min হিসেবে ধরা হয় না।
        """
        w = self.inducement_window
        highs  = df["high"].values
        lows   = df["low"].values
        closes = df["close"].values
        n = len(df)

        inducements: List[Dict[str, Any]] = []

        # Look for small swing highs in the last 50 candles
        lookback = min(n - 1, 50)

        for i in range(n - lookback, n - 3):
            if i < w or i + 3 >= n:
                continue

            # Swing low: check that lows[i] is the minimum of
            # [i-w, i+w] window (NOT including future sweep candles).
            # Confirmation candle at i+1 must be higher.
            past_window_low  = lows[i - w:i + 1]    # i-w to i (inclusive)
            next_window_low  = lows[i + 1:i + 2]    # just the confirmation candle

            if lows[i] == past_window_low.min() and len(next_window_low) > 0 and next_window_low[0] > lows[i]:
                # Now check sweep: any of next 1-3 candles breaks below lows[i]
                # AND the candle after that closes back above
                swept = False
                for k in range(i + 1, min(i + 4, n - 1)):
                    if closes[k] < lows[i] and closes[k + 1] > lows[i]:
                        swept = True
                        break
                if swept:
                    inducements.append({
                        "type":      "SELL_SIDE_INDUCEMENT",
                        "level":     round(float(lows[i]), 5),
                        "idx":       i,
                        "swept":     True,
                        "sweep_dir": "DOWN_THEN_REVERSE",
                        "note":      f"SSI at {lows[i]:.5f} swept — bullish reversal likely",
                    })

            # Swing high: similar logic mirrored
            past_window_high = highs[i - w:i + 1]
            next_window_high = highs[i + 1:i + 2]

            if highs[i] == past_window_high.max() and len(next_window_high) > 0 and next_window_high[0] < highs[i]:
                swept = False
                for k in range(i + 1, min(i + 4, n - 1)):
                    if closes[k] > highs[i] and closes[k + 1] < highs[i]:
                        swept = True
                        break
                if swept:
                    inducements.append({
                        "type":      "BUY_SIDE_INDUCEMENT",
                        "level":     round(float(highs[i]), 5),
                        "idx":       i,
                        "swept":     True,
                        "sweep_dir": "UP_THEN_REVERSE",
                        "note":      f"BSI at {highs[i]:.5f} swept — bearish reversal likely",
                    })

        return inducements[-self.max_zones:]

    # ═══════════════════════════════════════════════════════
    # BIAS + SIGNAL
    # ═══════════════════════════════════════════════════════

    def _bias_and_signal(
        self,
        mitigation_blocks: List[Dict[str, Any]],
        inducements:       List[Dict[str, Any]],
        active_signals:    List[str],
    ) -> tuple[str, str, str]:
        """
        Active signals থেকে bias + final signal বের করো।
        """
        bull = sum(1 for s in active_signals if "BULLISH" in s)
        bear = sum(1 for s in active_signals if "BEARISH" in s)

        if bull > bear:
            return "BULLISH", "BUY", f"{bull} bullish SMC signals active (MB+Inducement)"
        if bear > bull:
            return "BEARISH", "SELL", f"{bear} bearish SMC signals active (MB+Inducement)"
        if bull > 0 and bear > 0:
            return "NEUTRAL", "WAIT", "Conflicting SMC signals — wait for clarity"
        return "NEUTRAL", "WAIT", "No active SMC advanced signals"

    # ═══════════════════════════════════════════════════════
    # AI CONTEXT
    # ═══════════════════════════════════════════════════════

    def get_ai_context(self, result: Dict[str, Any]) -> Dict[str, Any]:
        if not result.get("valid"):
            return {
                "smc_adv_valid":         False,
                "smc_adv_bias":          "NEUTRAL",
                "smc_adv_signal":        "WAIT",
                "smc_adv_active_count":  0,
            }

        return {
            "smc_adv_valid":              True,
            "smc_adv_bias":               result.get("bias"),
            "smc_adv_signal":             result.get("signal"),
            "smc_adv_active_count":       len(result.get("active_signals", [])),
            "smc_adv_active_signals":     result.get("active_signals", []),
            "smc_adv_mitigation_count":   len(result.get("mitigation_blocks", [])),
            "smc_adv_inducement_count":   len(result.get("inducements", [])),
            "smc_adv_has_active_retest":  any(
                mb.get("status") == "ACTIVE_RETEST"
                for mb in result.get("mitigation_blocks", [])
            ),
        }

    # ═══════════════════════════════════════════════════════
    # UTILITIES
    # ═══════════════════════════════════════════════════════

    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high  = df["high"]
        low   = df["low"]
        close = df["close"]
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low  - close.shift(1)).abs()
        tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()

    def _empty_result(self, reason: str) -> Dict[str, Any]:
        return {
            "valid":             False,
            "reason":            reason,
            "mitigation_blocks": [],
            "inducements":       [],
            "active_signals":    [],
            "bias":              "NEUTRAL",
            "signal":            "WAIT",
            "note":              reason,
        }

    # ═══════════════════════════════════════════════════════
    # PRINT SUMMARY
    # ═══════════════════════════════════════════════════════

    def print_summary(self, result: Dict[str, Any]) -> None:
        bar = "═" * 56
        log.info(bar)
        log.info("  🎯  SMC ADVANCED  (Day 87)")
        log.info(bar)

        if not result.get("valid"):
            log.info(f"  ⚠️  {result.get('reason', 'No analysis')}")
            log.info(bar)
            return

        mbs = result.get("mitigation_blocks", [])
        if mbs:
            log.info(f"  Mitigation Blocks ({len(mbs)}):")
            for mb in mbs[:3]:
                icon = "🟢" if mb["type"] == "BULLISH_MITIGATION" else "🔴"
                log.info(f"    {icon} {mb['type']}  zone={mb['zone_bottom']}-{mb['zone_top']}  status={mb['status']}")
        else:
            log.info("  Mitigation Blocks: none")

        inds = result.get("inducements", [])
        if inds:
            log.info(f"  Inducements ({len(inds)}):")
            for ind in inds[:3]:
                icon = "🟢" if ind["type"] == "SELL_SIDE_INDUCEMENT" else "🔴"
                log.info(f"    {icon} {ind['type']}  level={ind['level']}  swept={ind['swept']}")
        else:
            log.info("  Inducements: none")

        active = result.get("active_signals", [])
        log.info(f"  Active Signals: {', '.join(active) if active else 'none'}")
        log.info(f"  Bias   : {result['bias']}")
        log.info(f"  Signal : {result['signal']}")
        log.info(f"  Note   : {result['note']}")
        log.info(bar)


# ═══════════════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)
    n = 200
    # Create data with a clear swing low at index 150 (SSI)
    prices = 1.1000 + np.cumsum(np.random.randn(n) * 0.0005)
    # Force a small swing low at 150 then sweep at 155
    prices[150] = prices[149] - 0.002
    prices[151] = prices[150] + 0.0005
    prices[155] = prices[150] - 0.0003   # sweep below
    prices[156] = prices[150] + 0.001    # reversal

    df = pd.DataFrame({
        "open":  prices,
        "high":  prices + 0.0005,
        "low":   prices - 0.0005,
        "close": prices,
        "atr":   pd.Series(np.full(n, 0.001)),
    })

    engine = SMCAdvancedEngine()
    result = engine.analyze(df)
    engine.print_summary(result)

    ctx = engine.get_ai_context(result)
    print("\nAI Context:")
    for k, v in ctx.items():
        print(f"  {k:<28}: {v}")
