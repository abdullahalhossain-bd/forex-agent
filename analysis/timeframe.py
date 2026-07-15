# analysis/timeframe.py
# ============================================================
# Multi-Timeframe Analysis (MTF)
# Daily → 4H → 1H → 15M
# AI Trader top-down analysis করবে
# ============================================================

from data.fetcher import get_data_fetcher
from data.indicators import Indicators
from utils.logger import get_logger

log = get_logger(__name__)

# Top-down timeframe hierarchy
MTF_CHAIN = ['1d', '4h', '1h', '15m']


class MultiTimeframeAnalyzer:
    """
    Professional trader-এর মতো top-down analysis।

    Daily trend → 4H confirmation → 1H structure → 15M entry

    AI এটা দেখে বলবে:
        "Daily bullish, 4H pullback, 15M entry opportunity"
    """

    def __init__(self, symbol: str = "EUR/USDT"):
        self.symbol  = symbol
        self.fetcher = get_data_fetcher()
        self.ind     = Indicators()

    def analyze(self, timeframes: list = None) -> dict:
        """
        Multiple timeframe-এ indicator calculate করো।
        Return: dict { '1d': context, '4h': context, ... }
        """
        timeframes = timeframes or MTF_CHAIN
        results    = {}

        for tf in timeframes:
            log.info(f"MTF: Fetching {self.symbol} {tf}")
            df = self.fetcher.fetch_ohlcv(
                symbol    = self.symbol,
                timeframe = tf,
                limit     = 200,
            )
            if df is None:
                log.warning(f"MTF: Could not fetch {tf}")
                continue

            df  = self.ind.add_all(df)
            ctx = self.ind.get_ai_context(df)
            ctx['timeframe'] = tf
            results[tf] = ctx
            log.info(f"MTF: {tf} → trend={ctx['trend']} rsi={ctx['rsi']}")

        return results

    def get_bias(self, mtf_results: dict) -> dict:
        """
        সব timeframe-এর trend দেখে overall bias বলো।

        Rule:
          Daily + 4H bullish  → Look for BUY on 15M
          Daily + 4H bearish  → Look for SELL on 15M
          Mixed               → Wait for alignment
        """
        trends = {
            tf: ctx.get('trend', 'unknown')
            for tf, ctx in mtf_results.items()
        }

        bullish_count = sum(1 for t in trends.values() if 'bullish' in t)
        bearish_count = sum(1 for t in trends.values() if 'bearish' in t)
        total         = len(trends)

        if bullish_count >= total * 0.75:
            bias, conf = 'BULLISH', 'HIGH'
        elif bearish_count >= total * 0.75:
            bias, conf = 'BEARISH', 'HIGH'
        elif bullish_count > bearish_count:
            bias, conf = 'BULLISH', 'MEDIUM'
        elif bearish_count > bullish_count:
            bias, conf = 'BEARISH', 'MEDIUM'
        else:
            bias, conf = 'NEUTRAL', 'LOW'

        return {
            'bias':       bias,
            'confidence': conf,
            'trends':     trends,
            'bullish_tf': bullish_count,
            'bearish_tf': bearish_count,
        }

    def print_summary(self, mtf_results: dict):
        bias = self.get_bias(mtf_results)

        print("\n" + "═" * 46)
        print("  📊  MULTI-TIMEFRAME ANALYSIS")
        print("═" * 46)
        for tf in MTF_CHAIN:
            if tf not in mtf_results:
                print(f"  {tf:<6}  :  ⚠️  Not available")
                continue
            ctx = mtf_results[tf]
            arrow = '▲' if 'bullish' in ctx['trend'] else ('▼' if 'bearish' in ctx['trend'] else '→')
            print(f"  {tf:<6}  :  {arrow} {ctx['trend']:<18}  RSI {ctx['rsi']:.1f}")
        print()
        print(f"  Overall Bias :  {bias['bias']}  (confidence: {bias['confidence']})")
        if bias['bias'] == 'BULLISH':
            print(f"  Suggestion   :  🟢 Look for BUY setups on lower TF")
        elif bias['bias'] == 'BEARISH':
            print(f"  Suggestion   :  🔴 Look for SELL setups on lower TF")
        else:
            print(f"  Suggestion   :  🟡 Wait for timeframe alignment")
        print("═" * 46 + "\n")
        return bias