# analysis/patterns.py
# ============================================================
# Day 4 — Candlestick Pattern Detection Engine
# TA-Lib ছাড়া — Pure Python Logic
# AI Trader-এর Price Action Brain
# ============================================================

import pandas as pd


class PatternDetector:
    """
    Candlestick pattern detector — TA-Lib ছাড়া।

    প্রতিটা pattern-এর নিজস্ব mathematical rule আছে।
    AI এই patterns দেখে price behavior বুঝবে।
    """

    # ─────────────────────────────────────────────
    # MAIN METHOD — সব pattern একসাথে
    # ─────────────────────────────────────────────

    def detect_all(self, df):
        """
        DataFrame-এ প্রতিটা candle-এর জন্য pattern detect করো।
        নতুন column যোগ হবে: 'pattern'
        """
        df = df.copy()
        df['pattern'] = df.apply(self._detect_row, axis=1)
        detected = df[df['pattern'] != 'none']['pattern'].value_counts()
        print(f"✅ Pattern detection done | Unique patterns: {len(detected)}")
        return df

    def _detect_row(self, row):
        """একটা candle দেখে pattern বলো"""
        checks = [
            self.is_doji(row),
            self.is_hammer(row),
            self.is_shooting_star(row),
            self.is_bullish_engulfing_row(row),
            self.is_bearish_engulfing_row(row),
            self.is_pin_bar(row),
        ]
        # প্রথম যেটা match করে সেটাই return
        for result in checks:
            if result and result != 'none':
                return result
        return 'none'

    # ─────────────────────────────────────────────
    # INDIVIDUAL PATTERNS
    # ─────────────────────────────────────────────

    def is_doji(self, row):
        """
        Doji — open ও close প্রায় সমান।
        অনিশ্চয়তা দেখায় — reversal সম্ভব।
        """
        body       = abs(row['close'] - row['open'])
        full_range = row['high'] - row['low']

        if full_range == 0:
            return 'none'

        # Body, full candle range-এর ১০% এর কম হলে Doji
        if body / full_range < 0.1:
            return 'doji'
        return 'none'

    def is_hammer(self, row):
        """
        Hammer — ছোট body, লম্বা lower wick।
        Downtrend-এ দেখা দিলে bullish reversal signal।

        Rule:
          lower_wick > body * 2
          upper_wick < body * 0.5
        """
        body        = abs(row['close'] - row['open'])
        upper_wick  = row['high'] - max(row['open'], row['close'])
        lower_wick  = min(row['open'], row['close']) - row['low']

        if body == 0:
            return 'none'

        if lower_wick > body * 2 and upper_wick <= body * 0.5:
            return 'hammer'
        return 'none'

    def is_shooting_star(self, row):
        """
        Shooting Star — ছোট body, লম্বা upper wick।
        Uptrend-এ দেখা দিলে bearish reversal signal।

        Rule:
          upper_wick > body * 2
          lower_wick < body * 0.5
        """
        body        = abs(row['close'] - row['open'])
        upper_wick  = row['high'] - max(row['open'], row['close'])
        lower_wick  = min(row['open'], row['close']) - row['low']

        if body == 0:
            return 'none'

        if upper_wick > body * 2 and lower_wick < body * 0.5:
            return 'shooting_star'
        return 'none'

    def is_pin_bar(self, row):
        """
        Pin Bar — strong rejection candle।
        Hammer/Shooting Star-এর চেয়ে আরো extreme।

        Rule:
          dominant wick, body-এর ৩ গুণের বেশি
          AND opposite wick ছোট (dominant wick-এর 1.5x এর কম) — নাহলে
          এটা rejection candle না, বরং indecision/doji-type candle।

        Note: this mirrors the asymmetry check in
        analysis/pin_bar_strategy.py (the book-validated implementation)
        so the two pin-bar detectors in this codebase can't disagree on
        the same candle — previously this function only checked the
        dominant wick and ignored the opposite one.
        """
        body        = abs(row['close'] - row['open'])
        upper_wick  = row['high'] - max(row['open'], row['close'])
        lower_wick  = min(row['open'], row['close']) - row['low']

        if body == 0:
            return 'none'

        if lower_wick > body * 3 and lower_wick > upper_wick * 1.5:
            return 'bullish_pin_bar'
        if upper_wick > body * 3 and upper_wick > lower_wick * 1.5:
            return 'bearish_pin_bar'
        return 'none'

    def is_bullish_engulfing_row(self, row):
        """Single row দিয়ে detect হয় না — দুটো candle লাগে।
        detect_engulfing() ব্যবহার করো DataFrame-এ।"""
        return 'none'

    def is_bearish_engulfing_row(self, row):
        return 'none'

    # ─────────────────────────────────────────────
    # MULTI-CANDLE PATTERNS (DataFrame দরকার)
    # ─────────────────────────────────────────────

    def detect_engulfing(self, df):
        """
        Bullish/Bearish Engulfing — দুটো consecutive candle দেখে।

        Bullish Engulfing:
          আগের candle bearish (red)
          পরের candle bullish (green) এবং আগেরটাকে পুরো ঢেকে দেয়

        Bearish Engulfing:
          আগের candle bullish (green)
          পরের candle bearish (red) এবং আগেরটাকে পুরো ঢেকে দেয়
        """
        df = df.copy()
        df['engulfing'] = 'none'

        for i in range(1, len(df)):
            prev = df.iloc[i - 1]
            curr = df.iloc[i]

            prev_bearish = prev['close'] < prev['open']
            curr_bullish = curr['close'] > curr['open']
            bullish_engulf = (
                prev_bearish and curr_bullish
                and curr['open']  < prev['close']
                and curr['close'] > prev['open']
            )

            prev_bullish = prev['close'] > prev['open']
            curr_bearish = curr['close'] < curr['open']
            bearish_engulf = (
                prev_bullish and curr_bearish
                and curr['open']  > prev['close']
                and curr['close'] < prev['open']
            )

            if bullish_engulf:
                df.iloc[i, df.columns.get_loc('engulfing')] = 'bullish_engulfing'
            elif bearish_engulf:
                df.iloc[i, df.columns.get_loc('engulfing')] = 'bearish_engulfing'

        return df

    def detect_morning_evening_star(self, df):
        """
        Morning Star (bullish reversal) — ৩টা candle:
          1. বড় bearish candle
          2. ছোট body candle (star)
          3. বড় bullish candle

        Evening Star (bearish reversal) — ৩টা candle:
          1. বড় bullish candle
          2. ছোট body candle (star)
          3. বড় bearish candle
        """
        df = df.copy()
        df['star_pattern'] = 'none'

        for i in range(2, len(df)):
            c1 = df.iloc[i - 2]
            c2 = df.iloc[i - 1]
            c3 = df.iloc[i]

            c2_body = abs(c2['close'] - c2['open'])
            c1_body = abs(c1['close'] - c1['open'])
            c3_body = abs(c3['close'] - c3['open'])

            # Morning Star
            if (
                c1['close'] < c1['open']            # c1 bearish
                and c2_body < c1_body * 0.3          # c2 small body
                and c3['close'] > c3['open']          # c3 bullish
                and c3_body > c1_body * 0.5           # c3 significant
            ):
                df.iloc[i, df.columns.get_loc('star_pattern')] = 'morning_star'

            # Evening Star
            elif (
                c1['close'] > c1['open']             # c1 bullish
                and c2_body < c1_body * 0.3           # c2 small body
                and c3['close'] < c3['open']           # c3 bearish
                and c3_body > c1_body * 0.5            # c3 significant
            ):
                df.iloc[i, df.columns.get_loc('star_pattern')] = 'evening_star'

        return df

    # ─────────────────────────────────────────────
    # Day 81+ — 3 NEW MASTERCLASS PATTERNS
    # ─────────────────────────────────────────────

    def detect_three_bar_continuation(self, df):
        """
        Three Bar Continuation — trend continuation signal.

        Bullish (uptrend continues):
          1. Bullish candle (green)
          2. Small pullback candle (red, small body)
          3. Bullish candle that closes above candle 1's high

        Bearish (downtrend continues):
          1. Bearish candle (red)
          2. Small pullback candle (green, small body)
          3. Bearish candle that closes below candle 1's low
        """
        df = df.copy()
        df['three_bar_cont'] = 'none'

        for i in range(2, len(df)):
            c1 = df.iloc[i - 2]
            c2 = df.iloc[i - 1]
            c3 = df.iloc[i]

            c1_body = abs(c1['close'] - c1['open'])
            c2_body = abs(c2['close'] - c2['open'])
            c3_body = abs(c3['close'] - c3['open'])

            # Bullish continuation
            if (
                c1['close'] > c1['open']              # c1 bullish
                and c2['close'] < c2['open']           # c2 bearish pullback
                and c2_body < c1_body * 0.5            # c2 small body
                and c3['close'] > c3['open']           # c3 bullish
                and c3['close'] > c1['high']           # c3 closes above c1 high
            ):
                df.iloc[i, df.columns.get_loc('three_bar_cont')] = 'three_bar_continuation_bullish'

            # Bearish continuation
            elif (
                c1['close'] < c1['open']              # c1 bearish
                and c2['close'] > c2['open']           # c2 bullish pullback
                and c2_body < c1_body * 0.5            # c2 small body
                and c3['close'] < c3['open']           # c3 bearish
                and c3['close'] < c1['low']            # c3 closes below c1 low
            ):
                df.iloc[i, df.columns.get_loc('three_bar_cont')] = 'three_bar_continuation_bearish'

        return df

    def detect_three_bar_reversal(self, df):
        """
        Three Bar Reversal — trend reversal signal.

        Bullish reversal (downtrend → uptrend):
          1. Bearish candle (red, big body)
          2. Lower low but small body (indecision)
          3. Bullish candle that closes above candle 2's high

        Bearish reversal (uptrend → downtrend):
          1. Bullish candle (green, big body)
          2. Higher high but small body (indecision)
          3. Bearish candle that closes below candle 2's low
        """
        df = df.copy()
        df['three_bar_rev'] = 'none'

        for i in range(2, len(df)):
            c1 = df.iloc[i - 2]
            c2 = df.iloc[i - 1]
            c3 = df.iloc[i]

            c1_body = abs(c1['close'] - c1['open'])
            c2_body = abs(c2['close'] - c2['open'])
            c3_body = abs(c3['close'] - c3['open'])

            # Bullish reversal
            if (
                c1['close'] < c1['open']              # c1 bearish (big red)
                and c1_body > c2_body                  # c1 bigger than c2
                and c2['low'] < c1['low']              # c2 makes lower low
                and c2_body < c1_body * 0.5            # c2 small body (indecision)
                and c3['close'] > c3['open']           # c3 bullish
                and c3['close'] > c2['high']           # c3 closes above c2 high
            ):
                df.iloc[i, df.columns.get_loc('three_bar_rev')] = 'three_bar_reversal_bullish'

            # Bearish reversal
            elif (
                c1['close'] > c1['open']              # c1 bullish (big green)
                and c1_body > c2_body                  # c1 bigger than c2
                and c2['high'] > c1['high']            # c2 makes higher high
                and c2_body < c1_body * 0.5            # c2 small body (indecision)
                and c3['close'] < c3['open']           # c3 bearish
                and c3['close'] < c2['low']            # c3 closes below c2 low
            ):
                df.iloc[i, df.columns.get_loc('three_bar_rev')] = 'three_bar_reversal_bearish'

        return df

    def detect_breakout_candle(self, df, lookback=20):
        """
        Breakout Candle — candle that breaks a recent high/low with strong momentum.

        Bullish breakout:
          - Current candle closes above the highest high of last `lookback` candles
          - Body is at least 1.5x average body size (strong momentum)

        Bearish breakout:
          - Current candle closes below the lowest low of last `lookback` candles
          - Body is at least 1.5x average body size (strong momentum)
        """
        df = df.copy()
        df['breakout_candle'] = 'none'

        if len(df) < lookback + 1:
            return df

        for i in range(lookback, len(df)):
            window = df.iloc[i - lookback:i]
            current = df.iloc[i]

            recent_high = window['high'].max()
            recent_low = window['low'].min()
            avg_body = (window['close'] - window['open']).abs().mean()

            body = abs(current['close'] - current['open'])

            # Bullish breakout
            if (
                current['close'] > recent_high
                and current['close'] > current['open']
                and body > avg_body * 1.5
            ):
                df.iloc[i, df.columns.get_loc('breakout_candle')] = 'breakout_bullish'

            # Bearish breakout
            elif (
                current['close'] < recent_low
                and current['close'] < current['open']
                and body > avg_body * 1.5
            ):
                df.iloc[i, df.columns.get_loc('breakout_candle')] = 'breakout_bearish'

        return df

    # ─────────────────────────────────────────────
    # FULL PIPELINE — সব একসাথে
    # ─────────────────────────────────────────────

    def run_full_detection(self, df):
        """সব pattern একসাথে detect করো এবং final df return করো"""
        df = self.detect_all(df)
        df = self.detect_engulfing(df)
        df = self.detect_morning_evening_star(df)
        # Day 81+ — masterclass patterns
        df = self.detect_three_bar_continuation(df)
        df = self.detect_three_bar_reversal(df)
        df = self.detect_breakout_candle(df)
        # Day 97+ Book Rules (Pages 76-90): additional patterns
        df = self.detect_piercing_line(df)
        df = self.detect_harami(df)
        df = self.detect_three_soldiers_crows(df)
        df = self.detect_context_patterns(df)  # Hammer vs Hanging Man (trend-aware)
        # Day 97+ Book Pages 91-98: more patterns
        df = self.detect_dark_cloud_cover(df)
        df = self.detect_doji_variants(df)
        df = self.detect_three_methods(df)
        # Day 97+ Candlestick Bible: Tweezers + Inside Bar False Breakout
        df = self.detect_tweezers(df)
        df = self.detect_inside_bar_false_breakout(df)
        # Day 97+ Candlestick Bible context classifiers
        df = self.classify_engulfing_context(df)
        df = self.classify_doji_context(df)
        # Day 97+ Candlestick Bible Page 41: Harami context
        df = self.classify_harami_context(df)
        return df

    # ═══════════════════════════════════════════════════════
    # Day 97+ Book Rules (Pages 76-90): Advanced Patterns
    # ═══════════════════════════════════════════════════════

    def detect_piercing_line(self, df):
        """Day 97+ Book Page 82: Piercing Line pattern.

        Two-candle pattern in downtrend:
          - Candle 1: bearish (large body)
          - Candle 2: opens below candle 1's close, closes ABOVE 50% of candle 1's body
        Strength = how far candle 2 closes into candle 1's body.
        """
        df = df.copy()
        df['piercing_line'] = 'none'
        for i in range(1, len(df)):
            c1_open = df.iloc[i-1]['open']
            c1_close = df.iloc[i-1]['close']
            c2_open = df.iloc[i]['open']
            c2_close = df.iloc[i]['close']

            if c1_close >= c1_open:  # candle 1 must be bearish
                continue
            midpoint = (c1_open + c1_close) / 2
            if c2_open < c1_close and c2_close > midpoint and c2_close < c1_open:
                strength = (c2_close - midpoint) / (c1_open - midpoint) if (c1_open - midpoint) > 0 else 0
                df.iloc[i, df.columns.get_loc('piercing_line')] = f'piercing_line_{strength:.2f}'
        return df

    def detect_harami(self, df):
        """Day 97+ Book Pages 85-86: Bullish/Bearish Harami + Harami Cross.

        Two-candle: candle 2's body fully contained within candle 1's body.
        Bullish Harami: candle 1 bearish, candle 2 small bullish.
        Bearish Harami: candle 1 bullish, candle 2 small bearish.
        Harami Cross: candle 2 is a doji (open ≈ close).
        """
        df = df.copy()
        df['harami'] = 'none'
        for i in range(1, len(df)):
            c1_open = df.iloc[i-1]['open']
            c1_close = df.iloc[i-1]['close']
            c2_open = df.iloc[i]['open']
            c2_close = df.iloc[i]['close']

            c1_body = abs(c1_close - c1_open)
            c2_body = abs(c2_close - c2_open)

            if c1_body == 0:
                continue

            # Candle 2 body inside candle 1 body
            inside = c2_open >= min(c1_open, c1_close) and c2_close <= max(c1_open, c1_close)
            if not inside or c2_body >= c1_body:
                continue

            # Doji check (Harami Cross)
            is_doji = c2_body <= c1_body * 0.05

            if c1_close < c1_open:  # candle 1 bearish
                if is_doji:
                    df.iloc[i, df.columns.get_loc('harami')] = 'bullish_harami_cross'
                elif c2_close > c2_open:
                    df.iloc[i, df.columns.get_loc('harami')] = 'bullish_harami'
            elif c1_close > c1_open:  # candle 1 bullish
                if is_doji:
                    df.iloc[i, df.columns.get_loc('harami')] = 'bearish_harami_cross'
                elif c2_close < c2_open:
                    df.iloc[i, df.columns.get_loc('harami')] = 'bearish_harami'
        return df

    def detect_three_soldiers_crows(self, df):
        """Day 97+ Book Pages 84, 90: Three White Soldiers / Three Black Crows.

        Three White Soldiers: 3 consecutive bullish candles, each closing higher.
        Three Black Crows: 3 consecutive bearish candles, each closing lower.
        """
        df = df.copy()
        df['three_soldiers_crows'] = 'none'
        for i in range(2, len(df)):
            c0 = df.iloc[i-2]
            c1 = df.iloc[i-1]
            c2 = df.iloc[i]

            # Three White Soldiers
            if (c0['close'] > c0['open'] and c1['close'] > c1['open'] and c2['close'] > c2['open']
                and c1['close'] > c0['close'] and c2['close'] > c1['close']):
                df.iloc[i, df.columns.get_loc('three_soldiers_crows')] = 'three_white_soldiers'

            # Three Black Crows
            if (c0['close'] < c0['open'] and c1['close'] < c1['open'] and c2['close'] < c2['open']
                and c1['close'] < c0['close'] and c2['close'] < c1['close']):
                df.iloc[i, df.columns.get_loc('three_soldiers_crows')] = 'three_black_crows'
        return df

    def detect_context_patterns(self, df, lookback=5):
        """Day 97+ Book Page 87-88: Context-aware pattern detection.

        Hammer and Hanging Man are geometrically IDENTICAL.
        The ONLY difference is the preceding trend:
          - After downtrend → Hammer (bullish reversal)
          - After uptrend → Hanging Man (bearish reversal)

        Same for Inverted Hammer (downtrend) vs Shooting Star (uptrend).

        This method adds a 'context_pattern' column that correctly
        identifies the pattern based on trend context.
        """
        df = df.copy()
        df['context_pattern'] = 'none'

        for i in range(lookback, len(df)):
            row = df.iloc[i]
            body = abs(row['close'] - row['open'])
            if body == 0:
                continue
            upper_wick = row['high'] - max(row['open'], row['close'])
            lower_wick = min(row['open'], row['close']) - row['low']

            # Determine preceding trend from last N candles
            recent = df.iloc[i-lookback:i]
            price_change = recent['close'].iloc[-1] - recent['close'].iloc[0]
            trend_before = "up" if price_change > 0 else "down" if price_change < 0 else "flat"

            # Hammer / Hanging Man shape: long lower wick, small body
            if lower_wick > body * 2 and upper_wick <= body * 0.5:
                if trend_before == "down":
                    df.iloc[i, df.columns.get_loc('context_pattern')] = 'hammer'
                elif trend_before == "up":
                    df.iloc[i, df.columns.get_loc('context_pattern')] = 'hanging_man'

            # Inverted Hammer / Shooting Star shape: long upper wick
            if upper_wick > body * 2 and lower_wick <= body * 0.5:
                if trend_before == "down":
                    df.iloc[i, df.columns.get_loc('context_pattern')] = 'inverted_hammer'
                elif trend_before == "up":
                    df.iloc[i, df.columns.get_loc('context_pattern')] = 'shooting_star'

        return df

    # ═══════════════════════════════════════════════════════
    # Day 97+ Book Rules (Pages 91-98): More Patterns
    # ═══════════════════════════════════════════════════════

    def detect_dark_cloud_cover(self, df):
        """Day 97+ Book Page 92: Dark Cloud Cover (bearish Piercing Line mirror).

        Two-candle pattern at top of uptrend:
          - Candle 1: bullish (large body)
          - Candle 2: opens ABOVE candle 1's high (new high), closes BELOW 50% of body
        Strength = how far candle 2 closes down into candle 1's body.
        """
        df = df.copy()
        df['dark_cloud_cover'] = 'none'
        for i in range(1, len(df)):
            c1_open = df.iloc[i-1]['open']
            c1_close = df.iloc[i-1]['close']
            c1_high = df.iloc[i-1]['high']
            c2_open = df.iloc[i]['open']
            c2_close = df.iloc[i]['close']

            if c1_close <= c1_open:  # candle 1 must be bullish
                continue
            midpoint = (c1_open + c1_close) / 2
            if c2_open > c1_high and c2_close < midpoint and c2_close > c1_open:
                strength = (midpoint - c2_close) / (midpoint - c1_open) if (midpoint - c1_open) > 0 else 0
                df.iloc[i, df.columns.get_loc('dark_cloud_cover')] = f'dark_cloud_{strength:.2f}'
        return df

    def detect_doji_variants(self, df, doji_threshold=0.001):
        """Day 97+ Book Pages 95-96: Doji sub-types.

        - Four-Price Doji: open≈close≈high≈low (extreme indecision)
        - Long-Legged Doji: open≈close, both wicks large (high volatility)
        - Dragonfly Doji: open≈close≈high, long lower wick (bullish at support)
        - Gravestone Doji: open≈close≈low, long upper wick (bearish at resistance)
        - Spinning Top: small body, both wicks present (weak indecision)
        """
        df = df.copy()
        df['doji_variant'] = 'none'

        # Calculate average body for relative threshold
        bodies = (df['close'] - df['open']).abs()
        avg_body = bodies.rolling(20, min_periods=5).mean().iloc[-1] if len(bodies) >= 5 else 0.0010
        threshold = max(doji_threshold, avg_body * 0.1)

        for i in range(len(df)):
            row = df.iloc[i]
            body = abs(row['close'] - row['open'])
            high = row['high']
            low = row['low']
            upper_wick = high - max(row['open'], row['close'])
            lower_wick = min(row['open'], row['close']) - low
            full_range = high - low

            if full_range == 0:
                continue

            # Doji condition
            if body <= threshold:
                # Four-Price Doji
                if full_range <= threshold * 2:
                    df.iloc[i, df.columns.get_loc('doji_variant')] = 'four_price_doji'
                # Dragonfly (lower wick dominant, no upper wick)
                elif lower_wick > full_range * 0.7 and upper_wick <= body * 0.5:
                    df.iloc[i, df.columns.get_loc('doji_variant')] = 'dragonfly_doji'
                # Gravestone (upper wick dominant, no lower wick)
                elif upper_wick > full_range * 0.7 and lower_wick <= body * 0.5:
                    df.iloc[i, df.columns.get_loc('doji_variant')] = 'gravestone_doji'
                # Long-Legged (both wicks large)
                elif upper_wick > full_range * 0.3 and lower_wick > full_range * 0.3:
                    df.iloc[i, df.columns.get_loc('doji_variant')] = 'long_legged_doji'
            # Spinning Top (small body but not doji-small, both wicks)
            elif body <= avg_body * 0.5 and upper_wick > body * 0.5 and lower_wick > body * 0.5:
                df.iloc[i, df.columns.get_loc('doji_variant')] = 'spinning_top'

        return df

    def detect_three_methods(self, df):
        """Day 97+ Book Pages 97-98: Rising/Falling Three Methods (continuation).

        Falling Three Methods (bearish continuation):
          - Candle 1: long bearish
          - Candles 2-4: small bullish, contained within candle 1's range
          - Candle 5: long bearish, closes below candle 1's close

        Rising Three Methods (bullish continuation, mirror):
          - Candle 1: long bullish
          - Candles 2-4: small bearish, contained within candle 1's range
          - Candle 5: long bullish, closes above candle 1's close
        """
        df = df.copy()
        df['three_methods'] = 'none'

        if len(df) < 6:
            return df

        bodies = (df['close'] - df['open']).abs()
        avg_body = bodies.rolling(20, min_periods=5).mean().iloc[-1] if len(bodies) >= 5 else 0.0010

        for i in range(4, len(df)):
            c1 = df.iloc[i-4]
            c2 = df.iloc[i-3]
            c3 = df.iloc[i-2]
            c4 = df.iloc[i-1]
            c5 = df.iloc[i]

            c1_body = abs(c1['close'] - c1['open'])
            if c1_body < avg_body * 0.8:  # candle 1 must be "long"
                continue

            # Falling Three Methods (bearish continuation)
            if c1['close'] < c1['open']:  # candle 1 bearish
                small_bullish = all(
                    c['close'] > c['open'] and abs(c['close'] - c['open']) < c1_body * 0.5
                    for c in [c2, c3, c4]
                )
                contained = all(
                    c['high'] <= c1['high'] and c['low'] >= c1['low']
                    for c in [c2, c3, c4]
                )
                if small_bullish and contained and c5['close'] < c5['open'] and c5['close'] < c1['close']:
                    df.iloc[i, df.columns.get_loc('three_methods')] = 'falling_three_methods'

            # Rising Three Methods (bullish continuation)
            if c1['close'] > c1['open']:  # candle 1 bullish
                small_bearish = all(
                    c['close'] < c['open'] and abs(c['close'] - c['open']) < c1_body * 0.5
                    for c in [c2, c3, c4]
                )
                contained = all(
                    c['high'] <= c1['high'] and c['low'] >= c1['low']
                    for c in [c2, c3, c4]
                )
                if small_bearish and contained and c5['close'] > c5['open'] and c5['close'] > c1['close']:
                    df.iloc[i, df.columns.get_loc('three_methods')] = 'rising_three_methods'

        return df

    # ═══════════════════════════════════════════════════════
    # Day 97+ Candlestick Bible: Tweezers + Inside Bar False Breakout
    # ═══════════════════════════════════════════════════════

    def detect_tweezers(self, df, tolerance=0.0003):
        """Candlestick Bible: Tweezers Top/Bottom.

        Tweezers Top: bullish candle followed by bearish candle that closes
          back down near/at the first candle's open — bearish reversal at uptrend tops.
        Tweezers Bottom: bearish candle followed by bullish candle closing
          back up near/at the first candle's open — bullish reversal at downtrend bottoms.

        More reliable when occurring at S/R levels (book states this).
        """
        df = df.copy()
        df['tweezers'] = 'none'
        for i in range(1, len(df)):
            c1 = df.iloc[i-1]
            c2 = df.iloc[i]

            # Tweezers Top: c1 bullish, c2 bearish, c2 close ≈ c1 open
            if c1['close'] > c1['open'] and c2['close'] < c2['open']:
                if abs(c2['close'] - c1['open']) <= tolerance:
                    df.iloc[i, df.columns.get_loc('tweezers')] = 'tweezers_top'

            # Tweezers Bottom: c1 bearish, c2 bullish, c2 close ≈ c1 open
            if c1['close'] < c1['open'] and c2['close'] > c2['open']:
                if abs(c2['close'] - c1['open']) <= tolerance:
                    df.iloc[i, df.columns.get_loc('tweezers')] = 'tweezers_bottom'
        return df

    def detect_inside_bar_false_breakout(self, df, lookback=2):
        """Candlestick Bible: Inside Bar False Breakout (stop-loss hunting pattern).

        Structure:
          1. Mother candle (large body)
          2. Inside bar (small body within mother's range)
          3. Price breaks out of inside bar's range
          4. Price reverses back inside mother candle's range

        Bullish false breakout: forms in downtrend → reversal/bullish signal
        Bearish false breakout: forms in uptrend → reversal/bearish signal

        Book frames this as institutional stop-loss hunting.
        """
        df = df.copy()
        df['ib_false_breakout'] = 'none'

        if len(df) < 4:
            return df

        for i in range(3, len(df)):
            mother = df.iloc[i-3]    # mother candle
            inside = df.iloc[i-2]    # inside bar
            breakout = df.iloc[i-1]  # breakout candle
            current = df.iloc[i]     # reversal candle

            # Validate inside bar: inside bar's range within mother's range
            if not (inside['high'] <= mother['high'] and inside['low'] >= mother['low']):
                continue
            # Inside bar must be smaller than mother
            if abs(inside['close'] - inside['open']) >= abs(mother['close'] - mother['open']):
                continue

            # Bullish false breakout: price broke BELOW inside bar low, then reversed back up
            if (breakout['close'] < inside['low']  # broke below
                and current['close'] > inside['low']  # reversed back above
                and current['close'] > current['open']  # bullish reversal candle
                and current['close'] >= mother['low']):  # back inside mother range
                df.iloc[i, df.columns.get_loc('ib_false_breakout')] = 'bullish_false_breakout'

            # Bearish false breakout: price broke ABOVE inside bar high, then reversed back down
            if (breakout['close'] > inside['high']  # broke above
                and current['close'] < inside['high']  # reversed back below
                and current['close'] < current['open']  # bearish reversal candle
                and current['close'] <= mother['high']):  # back inside mother range
                df.iloc[i, df.columns.get_loc('ib_false_breakout')] = 'bearish_false_breakout'

        return df

    # ═══════════════════════════════════════════════════════
    # Day 97+ Candlestick Bible (Pages 15-30): Context Classifiers
    # ═══════════════════════════════════════════════════════

    def classify_engulfing_context(self, df, lookback=5):
        """Candlestick Bible Pages 17-19: Engulfing context classifier.

        Same engulfing shape means different things:
          - At downtrend bottom → 'reversal' (high weight, "capitulation bottom")
          - Mid-uptrend → 'continuation' (lower weight, buyers reasserting)
          - At uptrend top → 'reversal' (bearish, sellers absorb buyers)
          - Mid-downtrend → 'continuation' (lower weight, sellers reasserting)

        Book: "candle color is NOT diagnostic — only open/close positioning matters"
        """
        df = df.copy()
        df['engulfing_context'] = 'none'

        for i in range(1, len(df)):
            if df.iloc[i].get('engulfing', 'none') == 'none':
                continue

            # Determine preceding trend
            if i < lookback:
                continue
            recent = df.iloc[i-lookback:i]
            price_change = recent['close'].iloc[-1] - recent['close'].iloc[0]
            trend_before = "up" if price_change > 0 else "down" if price_change < 0 else "flat"

            engulfing_type = df.iloc[i]['engulfing']

            if engulfing_type == 'bullish_engulfing':
                if trend_before == "down":
                    df.iloc[i, df.columns.get_loc('engulfing_context')] = 'reversal_high_weight'
                elif trend_before == "up":
                    df.iloc[i, df.columns.get_loc('engulfing_context')] = 'continuation'
                else:
                    df.iloc[i, df.columns.get_loc('engulfing_context')] = 'neutral'

            elif engulfing_type == 'bearish_engulfing':
                if trend_before == "up":
                    df.iloc[i, df.columns.get_loc('engulfing_context')] = 'reversal_high_weight'
                elif trend_before == "down":
                    df.iloc[i, df.columns.get_loc('engulfing_context')] = 'continuation'
                else:
                    df.iloc[i, df.columns.get_loc('engulfing_context')] = 'neutral'

        return df

    def classify_doji_context(self, df, lookback=10, extreme_window=3):
        """Candlestick Bible Pages 20-22: Doji context classifier.

        Two distinct meanings:
          - 'exhaustion' (at trend extreme, swing high/low) → reversal signal
          - 'pause' (mid-trend, not at extreme) → consolidation, NOT reversal

        Book: "not every Doji marks a reversal; some just mark a rest/pause"
        Also: Doji can be used as profit-taking signal if in a trade against it.
        """
        df = df.copy()
        df['doji_context'] = 'none'

        for i in range(lookback, len(df)):
            row = df.iloc[i]
            body = abs(row['close'] - row['open'])
            full_range = row['high'] - row['low']
            if full_range == 0:
                continue

            # Doji condition (body ≤ 10% of range)
            is_doji = body <= full_range * 0.10
            if not is_doji:
                continue

            # Determine preceding trend
            recent = df.iloc[i-lookback:i]
            price_change = recent['close'].iloc[-1] - recent['close'].iloc[0]
            trend_before = "up" if price_change > 0 else "down" if price_change < 0 else "flat"

            # Check if at local extreme (N-bar high or low)
            window_start = max(0, i - extreme_window)
            window_end = min(len(df), i + extreme_window + 1)
            local_high = df.iloc[window_start:window_end]['high'].max()
            local_low = df.iloc[window_start:window_end]['low'].min()

            is_at_high = row['high'] >= local_high * 0.999
            is_at_low = row['low'] <= local_low * 1.001

            if trend_before == "up" and is_at_high:
                df.iloc[i, df.columns.get_loc('doji_context')] = 'exhaustion_bearish'
            elif trend_before == "down" and is_at_low:
                df.iloc[i, df.columns.get_loc('doji_context')] = 'exhaustion_bullish'
            elif trend_before in ("up", "down"):
                df.iloc[i, df.columns.get_loc('doji_context')] = 'pause'
            else:
                df.iloc[i, df.columns.get_loc('doji_context')] = 'neutral'

        return df

    def classify_harami_context(self, df, lookback=5, extreme_window=3):
        """Candlestick Bible Page 41: Harami context classifier.

        Same Harami shape means different things:
          - At trend extreme (top/bottom) → 'reversal' (market entering indecision)
          - Mid-trend → 'continuation' (consolidation before trend resumes)

        Book: "color of the two candles is unimportant — only containment matters"
        """
        df = df.copy()
        df['harami_context'] = 'none'

        for i in range(lookback, len(df)):
            if df.iloc[i].get('harami', 'none') == 'none':
                continue

            # Determine preceding trend
            recent = df.iloc[i-lookback:i]
            price_change = recent['close'].iloc[-1] - recent['close'].iloc[0]
            trend_before = "up" if price_change > 0 else "down" if price_change < 0 else "flat"

            # Check if at local extreme
            window_start = max(0, i - extreme_window)
            window_end = min(len(df), i + extreme_window + 1)
            local_high = df.iloc[window_start:window_end]['high'].max()
            local_low = df.iloc[window_start:window_end]['low'].min()

            is_at_high = df.iloc[i]['high'] >= local_high * 0.999
            is_at_low = df.iloc[i]['low'] <= local_low * 1.001

            harami_type = df.iloc[i]['harami']

            if harami_type in ('bullish_harami', 'bullish_harami_cross'):
                if trend_before == "down" and is_at_low:
                    df.iloc[i, df.columns.get_loc('harami_context')] = 'reversal_bullish'
                elif trend_before == "down":
                    df.iloc[i, df.columns.get_loc('harami_context')] = 'continuation_bullish'
                else:
                    df.iloc[i, df.columns.get_loc('harami_context')] = 'neutral'

            elif harami_type in ('bearish_harami', 'bearish_harami_cross'):
                if trend_before == "up" and is_at_high:
                    df.iloc[i, df.columns.get_loc('harami_context')] = 'reversal_bearish'
                elif trend_before == "up":
                    df.iloc[i, df.columns.get_loc('harami_context')] = 'continuation_bearish'
                else:
                    df.iloc[i, df.columns.get_loc('harami_context')] = 'neutral'

        return df

    def get_latest_patterns(self, df, lookback=5):
        """
        সর্বশেষ N candle-এ কী কী pattern ছিল।
        AI Brain এই list দেখে context বুঝবে।
        """
        recent = df.tail(lookback)
        found  = []

        # Day 81+ — include new pattern columns
        pattern_cols = ['pattern', 'engulfing', 'star_pattern',
                       'three_bar_cont', 'three_bar_rev', 'breakout_candle']
        for _, row in recent.iterrows():
            for col in pattern_cols:
                if col in row and row[col] != 'none':
                    found.append({
                        'time':    str(row.name),
                        'pattern': row[col],
                    })

        print("\n" + "═" * 46)
        print(f"  🕯️  CANDLESTICK PATTERNS  (Last {lookback} candles)")
        print("═" * 46)
        if found:
            for item in found:
                signal = self._pattern_signal(item['pattern'])
                print(f"  {item['time'][-8:]}  |  {item['pattern']:<22}  {signal}")
        else:
            print("  No significant patterns in recent candles.")
        print("═" * 46 + "\n")

        return found

    def get_ai_pattern_context(self, df, lookback=5):
        """Day 4 → Day 5 handoff: AI Brain-এর জন্য pattern context dict"""
        patterns = self.get_latest_patterns(df, lookback)
        latest   = df.iloc[-1]

        # সবচেয়ে recent pattern টা নাও
        last_pattern = 'none'
        if patterns:
            last_pattern = patterns[-1]['pattern']

        return {
            "latest_pattern":  last_pattern,
            "pattern_signal":  self._pattern_signal(last_pattern),
            "recent_patterns": [p['pattern'] for p in patterns],
            "candle_type":     'bullish' if latest['close'] > latest['open'] else 'bearish',
            "body_size":       round(abs(latest['close'] - latest['open']), 5),
        }

    @staticmethod
    def _pattern_signal(pattern):
        """Pattern দেখে signal বলো"""
        bullish = [
            'hammer', 'bullish_engulfing', 'morning_star', 'bullish_pin_bar',
            # Day 81+ masterclass patterns
            'three_bar_continuation_bullish', 'three_bar_reversal_bullish',
            'breakout_bullish',
        ]
        bearish = [
            'shooting_star', 'bearish_engulfing', 'evening_star', 'bearish_pin_bar',
            # Day 81+ masterclass patterns
            'three_bar_continuation_bearish', 'three_bar_reversal_bearish',
            'breakout_bearish',
        ]
        neutral = ['doji']

        if pattern in bullish: return '🟢 Bullish Signal'
        if pattern in bearish: return '🔴 Bearish Signal'
        if pattern in neutral: return '🟡 Neutral / Wait'
        return '⬜ No Signal'