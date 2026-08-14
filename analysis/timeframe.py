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

        P4 fix: returns an additional `_stale_tfs` key listing any TFs
        that were excluded due to staleness, so downstream consumers
        (TradePermission) can HARD BLOCK instead of silently trading
        without confirmation.
        """
        timeframes = timeframes or MTF_CHAIN
        results    = {}
        stale_tfs  = []   # P4: track excluded TFs for downstream hard-block
        # `stale_tfs` will contain structured records when a TF is excluded
        # due to staleness: {'tf': '1d', 'age_sec': 12345.6, 'last_timestamp': '...', 'reason': '...'}

        # Stale-data fix: this fetch path is independent of the primary
        # pipeline's df (core/trader.py), which already gates on
        # check_data_staleness()/compute_staleness_threshold(). This one
        # never did — each MTF leg (1d/4h/1h/15m) was used for bias
        # calculation no matter how old its last candle was. In practice
        # a stale H1 fetch (e.g. ~6583s / ~1h50m old — nearly 2 closed H1
        # bars behind) would silently feed a wrong trend into get_bias(),
        # which MarketBiasEngine/SignalEngine then treat as current.
        try:
            from core.production_hardening import (
                check_data_staleness,
                compute_staleness_threshold,
            )
            _staleness_available = True
        except Exception as e:
            log.debug(f"MTF: staleness check unavailable ({e}) — skipping freshness gate")
            _staleness_available = False

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

            if _staleness_available:
                try:
                    max_age = compute_staleness_threshold(tf)
                    staleness = check_data_staleness(df, max_age_sec=max_age)
                    if staleness.get("is_stale"):
                        reason = staleness.get('reason') or ''
                        age = staleness.get('age_sec') if staleness.get('age_sec') is not None else None
                        last_ts = staleness.get('last_timestamp')
                        log.warning(
                            f"MTF: {tf} data is STALE — {reason} "
                            f"(threshold={max_age}s, age={age}s) — excluding {tf} from bias "
                            f"calculation instead of trading on a stale candle"
                        )
                        stale_tfs.append({'tf': tf, 'age_sec': age, 'last_timestamp': last_ts, 'reason': reason})
                        continue
                except Exception as e:
                    log.debug(f"MTF: staleness check failed for {tf} (non-fatal): {e}")

            df  = self.ind.add_all(df)
            ctx = self.ind.get_ai_context(df)
            ctx['timeframe'] = tf
            results[tf] = ctx
            log.info(f"MTF: {tf} → trend={ctx['trend']} rsi={ctx['rsi']}")

        # P4: surface the stale-TF list so TradePermission can hard-block.
        # Empty list = all TFs fresh; non-empty = at least one TF excluded.
        if stale_tfs:
            names = [s['tf'] if isinstance(s, dict) else s for s in stale_tfs]
            log.warning(
                f"[MTF] Stale TFs detected: {names} — downstream TradePermission "
                f"will HARD BLOCK unless MTF_STALE_FAIL_OPEN=true"
            )
        # Export structured stale records for richer downstream handling.
        results["_stale_tfs"] = stale_tfs
        return results

    def get_bias(self, mtf_results: dict) -> dict:
        """
        সব timeframe-এর trend দেখে overall bias বলো।

        Rule:
          Daily + 4H bullish  → Look for BUY on 15M
          Daily + 4H bearish  → Look for SELL on 15M
          Mixed               → Wait for alignment

        B4a fix: skip non-dict entries (e.g. the `_stale_tfs` list added
        by `analyze()` in P4). Previously the dict comprehension iterated
        ALL items including `_stale_tfs`, and `[].get('trend')` raised
        AttributeError → caught upstream → `mtf_bias` defaulted to no
        `trends` key → `mtf_trends={}` → P4 hard-blocked EVERY live trade.
        """
        trends = {
            tf: ctx.get('trend', 'unknown')
            for tf, ctx in mtf_results.items()
            if isinstance(ctx, dict) and hasattr(ctx, 'get')
        }

        # B4a fix: surface stale TFs so downstream can see WHY data is missing
        # Support both legacy list-of-strings and new structured records
        raw_stale = mtf_results.get('_stale_tfs', []) if isinstance(mtf_results, dict) else []
        if raw_stale and isinstance(raw_stale[0], dict):
            stale_names = [r.get('tf') for r in raw_stale]
        else:
            stale_names = list(raw_stale)
        stale_tfs = stale_names

        bullish_count = sum(1 for t in trends.values() if 'bullish' in t)
        bearish_count = sum(1 for t in trends.values() if 'bearish' in t)
        total         = len(trends)

        if total == 0:
            # All TFs stale or missing — explicitly NEUTRAL/LOW
            bias, conf = 'NEUTRAL', 'LOW'
        elif bullish_count >= total * 0.75:
            bias, conf = 'BULLISH', 'HIGH'
        elif bearish_count >= total * 0.75:
            bias, conf = 'BEARISH', 'HIGH'
        elif bullish_count > bearish_count:
            bias, conf = 'BULLISH', 'MEDIUM'
        elif bearish_count > bullish_count:
            bias, conf = 'BEARISH', 'MEDIUM'
        else:
            bias, conf = 'NEUTRAL', 'LOW'

        # B4a fix: downgrade confidence if some TFs were stale
        if stale_tfs:
            # Stale TFs present → cap confidence at LOW regardless of alignment
            conf = 'LOW'
            log.warning(
                f"[MTF] get_bias: {len(stale_tfs)} TF(s) stale ({stale_tfs}) — "
                f"confidence downgraded to LOW; bias={bias} based on {total} fresh TF(s)"
            )

        return {
            'bias':       bias,
            'confidence': conf,
            'trends':     trends,
            'bullish_tf': bullish_count,
            'bearish_tf': bearish_count,
            'stale_tfs':  stale_tfs,   # B4a: propagate names for downstream hard-block logic
            'stale_details': raw_stale, # structured records when available
        }

    def print_summary(self, mtf_results: dict):
        bias = self.get_bias(mtf_results)

        print("\n" + "═" * 46)
        print("  📊  MULTI-TIMEFRAME ANALYSIS")
        print("═" * 46)
        # Prepare stale names for friendly checks (support legacy and structured)
        raw_stale = mtf_results.get('_stale_tfs', []) if isinstance(mtf_results, dict) else []
        if raw_stale and isinstance(raw_stale[0], dict):
            stale_names = [r.get('tf') for r in raw_stale]
            stale_map = {r.get('tf'): r for r in raw_stale}
        else:
            stale_names = list(raw_stale)
            stale_map = {s: None for s in stale_names}

        for tf in MTF_CHAIN:
            if tf not in mtf_results or not isinstance(mtf_results.get(tf), dict):
                if tf in stale_names:
                    rec = stale_map.get(tf) or {}
                    age = rec.get('age_sec')
                    last_ts = rec.get('last_timestamp')
                    reason = rec.get('reason') or ''
                    age_str = f" age={age}s" if age is not None else ""
                    ts_str = f" last={last_ts}" if last_ts else ""
                    print(f"  {tf:<6}  :  ⚠️  Not available (STALE){age_str}{ts_str} {reason}")
                else:
                    print(f"  {tf:<6}  :  ⚠️  Not available")
                continue
            ctx = mtf_results[tf]
            arrow = '▲' if 'bullish' in ctx.get('trend','') else ('▼' if 'bearish' in ctx.get('trend','') else '→')
            print(f"  {tf:<6}  :  {arrow} {ctx.get('trend',''):<18}  RSI {ctx.get('rsi',0):.1f}")
        print()
        print(f"  Overall Bias :  {bias['bias']}  (confidence: {bias['confidence']})")
        if bias.get('stale_tfs'):
            print(f"  ⚠️  Stale TFs :  {bias['stale_tfs']}  (confidence downgraded to LOW)")
        if bias['bias'] == 'BULLISH':
            print(f"  Suggestion   :  🟢 Look for BUY setups on lower TF")
        elif bias['bias'] == 'BEARISH':
            print(f"  Suggestion   :  🔴 Look for SELL setups on lower TF")
        else:
            print(f"  Suggestion   :  🟡 Wait for timeframe alignment")
        print("═" * 46 + "\n")
        return bias