# data/indicators.py
import pandas as pd
import ta

class Indicators:

    def add_all(self, df):
        df = self.add_moving_averages(df)
        df = self.add_rsi(df)
        df = self.add_macd(df)
        df = self.add_bollinger_bands(df)
        df = self.add_atr(df)
        df = self.add_trend_signals(df)
        print(f"✅ All indicators added | Total columns: {len(df.columns)}")
        return df

    def add_moving_averages(self, df):
        df['sma_20']  = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50']  = ta.trend.sma_indicator(df['close'], window=50)
        df['sma_200'] = ta.trend.sma_indicator(df['close'], window=200)
        df['ema_9']   = ta.trend.ema_indicator(df['close'], window=9)
        df['ema_21']  = ta.trend.ema_indicator(df['close'], window=21)
        return df

    def add_rsi(self, df):
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['rsi_signal'] = df['rsi'].apply(self._rsi_zone)
        return df

    def add_macd(self, df):
        macd = ta.trend.MACD(df['close'])
        df['macd']        = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist']   = macd.macd_diff()
        df['macd_cross']  = df.apply(
            lambda r: 'bullish_cross' if r['macd'] > r['macd_signal']
                      else 'bearish_cross', axis=1
        )
        return df

    def add_bollinger_bands(self, df):
        bb = ta.volatility.BollingerBands(df['close'], window=20)
        df['bb_upper']  = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower']  = bb.bollinger_lband()
        df['bb_width']  = bb.bollinger_wband()
        df['bb_pct']    = bb.bollinger_pband()
        return df

    def add_atr(self, df):
        df['atr'] = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close'], window=14
        ).average_true_range()
        return df

    def add_trend_signals(self, df):
        def trend_direction(row):
            try:
                p, s20, s50, s200 = row['close'], row['sma_20'], row['sma_50'], row['sma_200']
                if p > s20 > s50 > s200:   return 'strong_bullish'
                elif p > s20 and s20 > s50: return 'bullish'
                elif p < s20 < s50 < s200:  return 'strong_bearish'
                elif p < s20 and s20 < s50: return 'bearish'
                else:                       return 'sideways'
            except Exception:
                return 'unknown'
        df['trend'] = df.apply(trend_direction, axis=1)
        return df

    def get_summary(self, df):
        last = df.iloc[-1]
        print("\n" + "═" * 46)
        print("  📊  MARKET SNAPSHOT  (Latest Candle)")
        print("═" * 46)
        print(f"  Close       :  {last['close']:.5f}")
        print(f"  Trend       :  {last['trend'].upper()}")
        print(f"  RSI (14)    :  {last['rsi']:.2f}  →  {last['rsi_signal'].upper()}")
        print(f"  MACD        :  {last['macd']:.5f}  ({last['macd_cross']})")
        print(f"  ATR         :  {last['atr']:.5f}")
        print(f"  BB Upper    :  {last['bb_upper']:.5f}")
        print(f"  BB Lower    :  {last['bb_lower']:.5f}")
        print(f"  SMA 20/50/200: {last['sma_20']:.5f} / {last['sma_50']:.5f} / {last['sma_200']:.5f}")
        print("═" * 46 + "\n")
        return last

    def get_ai_context(self, df):
        last = df.iloc[-1]
        return {
            "price":      round(float(last['close']), 5),
            "trend":      last['trend'],
            "rsi":        round(float(last['rsi']), 2),
            "rsi_signal": last['rsi_signal'],
            "macd":       round(float(last['macd']), 5),
            "macd_cross": last['macd_cross'],
            "atr":        round(float(last['atr']), 5),
            "bb_upper":   round(float(last['bb_upper']), 5),
            "bb_lower":   round(float(last['bb_lower']), 5),
            "bb_pct":     round(float(last['bb_pct']), 2),
            "sma_20":     round(float(last['sma_20']), 5),
            "sma_50":     round(float(last['sma_50']), 5),
            "sma_200":    round(float(last['sma_200']), 5),
        }

    @staticmethod
    def _rsi_zone(x):
        if pd.isna(x):    return 'unknown'
        if x >= 70:       return 'overbought'
        if x <= 30:       return 'oversold'
        if x >= 55:       return 'bullish_zone'
        if x <= 45:       return 'bearish_zone'
        return 'neutral'