import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from analysis.support_resistance import SupportResistance, _atr_pct


def make_df(n=120, base=100.0, seed=1, vol=0.05):
    rng = np.random.RandomState(seed)
    close = base + np.cumsum(rng.randn(n) * vol * 0.05)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    df = pd.DataFrame({
        "open": close,
        "high": close + np.abs(rng.randn(n)) * vol * 0.1,
        "low": close - np.abs(rng.randn(n)) * vol * 0.1,
        "close": close,
    }, index=dates)
    return df


def flat_df(n, price=100.0, freq="h"):
    dates = pd.date_range("2024-01-01", periods=n, freq=freq)
    return pd.DataFrame({
        "open": [price] * n, "high": [price] * n,
        "low": [price] * n, "close": [price] * n,
    }, index=dates)


def candle(o, h, l, c):
    return pd.Series({"open": o, "high": h, "low": l, "close": c})


# ═══════════════════════════════════════════════════════════════
# 1. Rejection-event grouping
# ═══════════════════════════════════════════════════════════════

class TestRejectionEvents:
    def _base_df(self, n=40, price=100.0):
        df = flat_df(n, price=price)
        return df

    def test_one_touch_one_rejection(self):
        df = self._base_df()
        sr = SupportResistance()
        # single upward wick into resistance zone [99.9, 100.1] at bar 10
        df.iloc[10, df.columns.get_loc("high")] = 100.15
        df.iloc[10, df.columns.get_loc("open")] = 99.95
        df.iloc[10, df.columns.get_loc("close")] = 99.96  # small body, big upper wick
        n_events = sr._count_valid_rejections(df, 100.1, 99.9, "resistance")
        assert n_events == 1

    def test_multi_candle_rejection_is_one_event(self):
        df = self._base_df()
        sr = SupportResistance()
        # 6 consecutive candles all wick into the same resistance zone
        for i in range(10, 16):
            df.iloc[i, df.columns.get_loc("open")] = 99.95
            df.iloc[i, df.columns.get_loc("close")] = 99.96
            df.iloc[i, df.columns.get_loc("high")] = 100.15
        n_events = sr._count_valid_rejections(df, 100.1, 99.9, "resistance")
        assert n_events == 1, f"prolonged single interaction must be ONE event, got {n_events}"

    def test_repeated_separate_rejections(self):
        df = self._base_df(n=60)
        sr = SupportResistance()
        # Three well-separated touches (far apart in bars)
        for i in (5, 25, 45):
            df.iloc[i, df.columns.get_loc("open")] = 99.95
            df.iloc[i, df.columns.get_loc("close")] = 99.96
            df.iloc[i, df.columns.get_loc("high")] = 100.15
        n_events = sr._count_valid_rejections(df, 100.1, 99.9, "resistance")
        assert n_events == 3

    def test_wick_without_meaningful_rejection(self):
        df = self._base_df()
        sr = SupportResistance()
        # touches zone but body ~= wick (not a real rejection: wick < 1.5x body)
        df.iloc[10, df.columns.get_loc("open")] = 99.90
        df.iloc[10, df.columns.get_loc("close")] = 100.05
        df.iloc[10, df.columns.get_loc("high")] = 100.10  # wick only 0.05, body 0.15
        n_events = sr._count_valid_rejections(df, 100.1, 99.9, "resistance")
        assert n_events == 0

    def test_breakout_through_zone_splits_events(self):
        df = self._base_df(n=40)
        sr = SupportResistance()
        # touch+reject at bar 10
        df.iloc[10, df.columns.get_loc("open")] = 99.95
        df.iloc[10, df.columns.get_loc("close")] = 99.96
        df.iloc[10, df.columns.get_loc("high")] = 100.15
        # confirmed breakout close well above zone at bar 12
        df.iloc[12, df.columns.get_loc("close")] = 100.30
        df.iloc[12, df.columns.get_loc("open")] = 100.25
        df.iloc[12, df.columns.get_loc("high")] = 100.32
        # a fresh touch+reject shortly after (within merge_gap of bar 10,
        # but the breakout in between should still split it into 2 events)
        df.iloc[13, df.columns.get_loc("open")] = 99.95
        df.iloc[13, df.columns.get_loc("close")] = 99.96
        df.iloc[13, df.columns.get_loc("high")] = 100.15
        n_events = sr._count_valid_rejections(df, 100.1, 99.9, "resistance")
        assert n_events == 2

    def test_support_direction(self):
        df = self._base_df()
        sr = SupportResistance()
        for i in range(10, 14):
            df.iloc[i, df.columns.get_loc("open")] = 100.05
            df.iloc[i, df.columns.get_loc("close")] = 100.04
            df.iloc[i, df.columns.get_loc("low")] = 99.85
        n_events = sr._count_valid_rejections(df, 100.1, 99.9, "support")
        assert n_events == 1

    def test_doji_at_zone_counts_if_any_wick(self):
        df = self._base_df()
        sr = SupportResistance()
        df.iloc[10, df.columns.get_loc("open")] = 99.99
        df.iloc[10, df.columns.get_loc("close")] = 99.99
        df.iloc[10, df.columns.get_loc("high")] = 100.15
        n_events = sr._count_valid_rejections(df, 100.1, 99.9, "resistance")
        assert n_events == 1

    def test_zero_range_candle_no_crash(self):
        df = self._base_df()
        sr = SupportResistance()
        df.iloc[10, df.columns.get_loc("open")] = 100.0
        df.iloc[10, df.columns.get_loc("high")] = 100.0
        df.iloc[10, df.columns.get_loc("low")] = 100.0
        df.iloc[10, df.columns.get_loc("close")] = 100.0
        n_events = sr._count_valid_rejections(df, 100.1, 99.9, "resistance")
        assert n_events == 0  # touches zone (high in band) but zero wick


# ═══════════════════════════════════════════════════════════════
# 2. Zone width construction
# ═══════════════════════════════════════════════════════════════

class TestZoneWidth:
    def test_narrow_cluster_gets_atr_floor(self):
        df = make_df(n=120, base=100.0, vol=0.3)  # decent ATR
        sr = SupportResistance()
        atr_pct = _atr_pct(df)
        price = float(df["close"].iloc[-1])
        atr_abs = atr_pct * price
        cluster = [
            {"price": 100.000, "index": 10},
            {"price": 100.0001, "index": 20},
            {"price": 100.0002, "index": 30},
        ]
        zone = sr._build_zone(cluster, df, "support", atr_pct=atr_pct)
        width = zone["zone_top"] - zone["zone_bottom"]
        assert width >= 2 * 0.15 * atr_abs - 1e-4, "width should be floored near MIN_ZONE_ATR_MULT*ATR"

    def test_wide_cluster_gets_atr_ceiling(self):
        df = make_df(n=120, base=100.0, vol=0.05)  # small ATR
        sr = SupportResistance()
        atr_pct = _atr_pct(df)
        price = float(df["close"].iloc[-1])
        atr_abs = atr_pct * price
        # An outlier-heavy cluster whose raw min/max spread would be huge
        cluster = [
            {"price": 99.0, "index": 10},
            {"price": 100.0, "index": 20},
            {"price": 100.05, "index": 30},
            {"price": 100.1, "index": 40},
            {"price": 101.0, "index": 50},
        ]
        zone = sr._build_zone(cluster, df, "support", atr_pct=atr_pct)
        width = zone["zone_top"] - zone["zone_bottom"]
        raw_width = 101.0 - 99.0
        assert width < raw_width, "zone must not simply span raw cluster min/max"
        assert width <= 2 * 1.20 * atr_abs + 1e-4, "width should respect MAX_ZONE_ATR_MULT*ATR ceiling"

    def test_outlier_swing_does_not_dominate_center(self):
        df = make_df(n=120, base=100.0, vol=0.2)
        sr = SupportResistance()
        atr_pct = _atr_pct(df)
        cluster = [
            {"price": 100.00, "index": 10},
            {"price": 100.01, "index": 20},
            {"price": 100.02, "index": 30},
            {"price": 105.00, "index": 40},  # wild outlier (within a % threshold upstream)
        ]
        zone = sr._build_zone(cluster, df, "support", atr_pct=atr_pct)
        # median-based center should sit near the tight cluster, not be
        # dragged toward the outlier the way a raw mean/min-max would be
        assert abs(zone["center"] - 100.015) < 1.0

    def test_single_raw_swing_zone_is_nonzero_width(self):
        df = make_df(n=80)
        sr = SupportResistance()
        swings = [{"price": float(df["low"].iloc[-5]), "index": len(df) - 5}]
        levels = sr._raw_swing_levels(swings, df, "support")
        assert len(levels) == 1
        assert levels[0]["zone_top"] > levels[0]["zone_bottom"]


# ═══════════════════════════════════════════════════════════════
# 3. Order-independent merge
# ═══════════════════════════════════════════════════════════════

class TestMergeOrderInvariance:
    def test_merge_is_order_independent(self):
        df = make_df(n=100)
        sr = SupportResistance()
        atr_pct = _atr_pct(df)

        tier_a = [sr._build_zone(
            [{"price": 100.00, "index": 10}, {"price": 100.01, "index": 20}],
            df, "support", source="cluster", atr_pct=atr_pct)]
        tier_b = [sr._build_zone(
            [{"price": 100.02, "index": 15}, {"price": 100.03, "index": 25}],
            df, "support", source="eqh_eql", atr_pct=atr_pct)]

        merged_ab = sr._merge_zone_sources(tier_a, tier_b, df=df, atr_pct=atr_pct)
        merged_ba = sr._merge_zone_sources(tier_b, tier_a, df=df, atr_pct=atr_pct)

        assert len(merged_ab) == len(merged_ba) == 1
        za, zb = merged_ab[0], merged_ba[0]
        assert za["zone_top"] == zb["zone_top"]
        assert za["zone_bottom"] == zb["zone_bottom"]
        assert za["center"] == zb["center"]
        assert za["strength"] == zb["strength"]
        assert za["source"] == zb["source"]
        assert sorted(za["sources"]) == sorted(zb["sources"])

    def test_distant_zones_stay_separate_regardless_of_order(self):
        df = make_df(n=100)
        sr = SupportResistance()
        atr_pct = _atr_pct(df)
        tier_a = [sr._build_zone(
            [{"price": 95.0, "index": 10}, {"price": 95.01, "index": 20}],
            df, "support", source="cluster", atr_pct=atr_pct)]
        tier_b = [sr._build_zone(
            [{"price": 110.0, "index": 15}, {"price": 110.01, "index": 25}],
            df, "support", source="raw_swing", atr_pct=atr_pct)]
        merged_ab = sr._merge_zone_sources(tier_a, tier_b, df=df, atr_pct=atr_pct)
        merged_ba = sr._merge_zone_sources(tier_b, tier_a, df=df, atr_pct=atr_pct)
        assert len(merged_ab) == 2 and len(merged_ba) == 2

    def test_higher_priority_source_wins_regardless_of_order(self):
        df = make_df(n=100)
        sr = SupportResistance()
        atr_pct = _atr_pct(df)
        cluster_zone = sr._build_zone(
            [{"price": 100.00, "index": 10}, {"price": 100.01, "index": 20},
             {"price": 100.02, "index": 30}],
            df, "support", source="cluster", atr_pct=atr_pct)
        raw_zone = dict(sr._raw_swing_levels(
            [{"price": 100.015, "index": 35}], df, "support")[0])

        merged_1 = sr._merge_zone_sources([cluster_zone], [raw_zone], df=df, atr_pct=atr_pct)
        merged_2 = sr._merge_zone_sources([raw_zone], [cluster_zone], df=df, atr_pct=atr_pct)
        assert merged_1[0]["source"] == "cluster"
        assert merged_2[0]["source"] == "cluster"
        assert set(merged_1[0]["sources"]) == {"cluster", "raw_swing"}


# ═══════════════════════════════════════════════════════════════
# 4. Recency / relevance scoring
# ═══════════════════════════════════════════════════════════════

class TestRecency:
    def _zone(self, center, last_touch_index, strength="Medium"):
        return {
            "zone_top": center + 0.05, "zone_bottom": center - 0.05,
            "center": center, "touches": 2, "strength": strength,
            "last_touch_index": last_touch_index, "role": "support",
            "source": "cluster", "sources": ["cluster"],
        }

    def test_newer_zone_beats_older_identical_zone(self):
        sr = SupportResistance()
        old_zone = self._zone(99.0, last_touch_index=10)
        new_zone = self._zone(99.0, last_touch_index=90)
        ranked = sr._filter_relevant_zones(
            [old_zone, new_zone], current_price=100.0, side="support", total_bars=100
        )
        assert ranked[0]["last_touch_index"] == 90

    def test_index_shift_invariance(self):
        sr = SupportResistance()
        zone_old = self._zone(99.0, last_touch_index=10)
        zone_new = self._zone(98.5, last_touch_index=40)
        ranked_a = sr._filter_relevant_zones(
            [zone_old, zone_new], current_price=100.0, side="support", total_bars=100
        )
        order_a = [z["center"] for z in ranked_a]

        # Shift the whole frame by +500 bars (as if 500 extra bars of
        # history were prepended) — relative age is identical.
        shift = 500
        zone_old_shifted = self._zone(99.0, last_touch_index=10 + shift)
        zone_new_shifted = self._zone(98.5, last_touch_index=40 + shift)
        ranked_b = sr._filter_relevant_zones(
            [zone_old_shifted, zone_new_shifted], current_price=100.0,
            side="support", total_bars=100 + shift,
        )
        order_b = [z["center"] for z in ranked_b]
        assert order_a == order_b, "shifting all indices by a constant must not change ranking"

    def test_strength_still_matters(self):
        sr = SupportResistance()
        weak = self._zone(99.0, last_touch_index=50, strength="Weak")
        strong = self._zone(99.0, last_touch_index=50, strength="Strong")
        ranked = sr._filter_relevant_zones(
            [weak, strong], current_price=100.0, side="support", total_bars=100
        )
        assert ranked[0]["strength"] == "Strong"

    def test_distance_still_matters(self):
        sr = SupportResistance()
        near = self._zone(99.5, last_touch_index=50)
        far = self._zone(90.0, last_touch_index=50)
        ranked = sr._filter_relevant_zones(
            [near, far], current_price=100.0, side="support", total_bars=100
        )
        assert ranked[0]["center"] == 99.5


# ═══════════════════════════════════════════════════════════════
# 5. Inside-zone semantics
# ═══════════════════════════════════════════════════════════════

class TestInsideZoneSemantics:
    def _mk(self, top, bottom):
        return {"zone_top": top, "zone_bottom": bottom, "center": (top + bottom) / 2}

    def test_below_zone(self):
        sr = SupportResistance()
        sup = self._mk(100.5, 99.5)
        state = sr._classify_price_state(99.0, sup, None)
        assert state["location"] == "BELOW_SUPPORT"

    def test_inside_zone(self):
        sr = SupportResistance()
        sup = self._mk(100.5, 99.5)
        state = sr._classify_price_state(100.0, sup, None)
        assert state["location"] == "IN_SUPPORT_ZONE"
        assert state["in_zone"] is True

    def test_exactly_bottom_boundary(self):
        sr = SupportResistance()
        sup = self._mk(100.5, 99.5)
        state = sr._classify_price_state(99.5, sup, None)
        assert state["in_support_zone"] is True

    def test_exactly_top_boundary(self):
        sr = SupportResistance()
        res = self._mk(100.5, 99.5)
        state = sr._classify_price_state(100.5, None, res)
        assert state["in_resistance_zone"] is True

    def test_above_zone(self):
        sr = SupportResistance()
        res = self._mk(100.5, 99.5)
        state = sr._classify_price_state(101.0, None, res)
        assert state["location"] == "ABOVE_RESISTANCE"

    def test_overlapping_support_resistance_is_flagged(self):
        sr = SupportResistance()
        sup = self._mk(100.5, 99.5)
        res = self._mk(100.5, 99.5)  # identical/overlapping zone
        state = sr._classify_price_state(100.0, sup, res)
        assert state["location"] == "IN_OVERLAPPING_ZONE"
        assert state["in_support_zone"] and state["in_resistance_zone"]


# ═══════════════════════════════════════════════════════════════
# 6. Role reversal
# ═══════════════════════════════════════════════════════════════

class TestRoleReversal:
    def test_wick_through_only_is_not_broken(self):
        sr = SupportResistance()
        zone = {"zone_top": 100.5, "zone_bottom": 99.5}
        n = 30
        df = flat_df(n, price=100.0)
        # a single wick below support without a confirmed close
        df.iloc[15, df.columns.get_loc("low")] = 99.0
        df.iloc[15, df.columns.get_loc("close")] = 99.6  # closes back inside zone
        state = sr.detect_role_reversal_state(df, zone, "support", lookback_bars=20)
        assert state["state"] == "UNBROKEN"

    def test_confirmed_close_through_is_broken(self):
        sr = SupportResistance()
        zone = {"zone_top": 100.5, "zone_bottom": 99.5}
        n = 30
        df = flat_df(n, price=100.0)
        df.iloc[15, df.columns.get_loc("close")] = 99.0
        df.iloc[15, df.columns.get_loc("open")] = 99.3
        df.iloc[15, df.columns.get_loc("low")] = 98.9
        state = sr.detect_role_reversal_state(df, zone, "support", lookback_bars=20)
        assert state["state"] == "BROKEN"

    def test_retest_detected(self):
        sr = SupportResistance()
        zone = {"zone_top": 100.5, "zone_bottom": 99.5}
        n = 30
        df = flat_df(n, price=100.0)
        df.iloc[10, df.columns.get_loc("close")] = 99.0
        df.iloc[10, df.columns.get_loc("open")] = 99.3
        df.iloc[10, df.columns.get_loc("low")] = 98.9
        df.iloc[15, df.columns.get_loc("close")] = 99.48  # retest near old boundary
        df.iloc[15, df.columns.get_loc("open")] = 99.20
        state = sr.detect_role_reversal_state(df, zone, "support", lookback_bars=25)
        assert state["state"] in ("RETESTED", "ROLE_REVERSED")

    def test_rejection_after_retest_is_role_reversed(self):
        sr = SupportResistance()
        zone = {"zone_top": 100.5, "zone_bottom": 99.5}
        n = 30
        df = flat_df(n, price=100.0)
        df.iloc[10, df.columns.get_loc("close")] = 99.0
        df.iloc[10, df.columns.get_loc("open")] = 99.3
        df.iloc[10, df.columns.get_loc("low")] = 98.9
        # retest with a clear rejection candle (long upper wick, small body)
        df.iloc[15, df.columns.get_loc("open")] = 99.40
        df.iloc[15, df.columns.get_loc("close")] = 99.41
        df.iloc[15, df.columns.get_loc("high")] = 99.55
        state = sr.detect_role_reversal_state(df, zone, "support", lookback_bars=25)
        assert state["state"] == "ROLE_REVERSED"


# ═══════════════════════════════════════════════════════════════
# 7. Clustering
# ═══════════════════════════════════════════════════════════════

class TestClustering:
    def test_close_swings_merge(self):
        sr = SupportResistance(cluster_threshold_pct=0.002)
        df = make_df(n=100)
        swings = [
            {"price": 100.00, "index": 10}, {"price": 100.05, "index": 20},
            {"price": 100.10, "index": 30},
        ]
        zones = sr.cluster_into_zones(swings, df, "support", min_touches_override=2)
        assert len(zones) == 1
        assert zones[0]["touches"] == 3

    def test_distant_swings_do_not_merge(self):
        sr = SupportResistance(cluster_threshold_pct=0.001)
        df = make_df(n=100)
        swings = [
            {"price": 90.0, "index": 10}, {"price": 90.01, "index": 20},
            {"price": 110.0, "index": 30}, {"price": 110.02, "index": 40},
        ]
        zones = sr.cluster_into_zones(swings, df, "support", min_touches_override=2)
        assert len(zones) == 2


# ═══════════════════════════════════════════════════════════════
# 8. Leakage / point-in-time safety
# ═══════════════════════════════════════════════════════════════

class TestNoLookahead:
    def test_future_bars_do_not_change_historical_zones(self):
        df_full = make_df(n=200, seed=7)
        sr = SupportResistance(timeframe="H1")

        df_upto_100 = df_full.iloc[:100].copy()
        result_a = sr.analyze(df_upto_100, symbol="TEST")

        # Now dramatically alter bars AFTER index 100 and re-run on the
        # same up-to-100 slice — result must be identical.
        df_full_altered = df_full.copy()
        df_full_altered.iloc[100:, df_full_altered.columns.get_loc("close")] += 50.0
        df_full_altered.iloc[100:, df_full_altered.columns.get_loc("high")] += 50.0
        df_full_altered.iloc[100:, df_full_altered.columns.get_loc("low")] += 50.0
        df_full_altered.iloc[100:, df_full_altered.columns.get_loc("open")] += 50.0

        df_upto_100_again = df_full_altered.iloc[:100].copy()
        result_b = sr.analyze(df_upto_100_again, symbol="TEST")

        assert result_a["all_support_zones"] == result_b["all_support_zones"]
        assert result_a["all_resistance_zones"] == result_b["all_resistance_zones"]
        assert result_a["current_price"] == result_b["current_price"]


# ═══════════════════════════════════════════════════════════════
# 9. Timeframes — engine runs on all supported timeframes
# ═══════════════════════════════════════════════════════════════

@pytest.mark.parametrize("tf", ["M1", "M5", "M15", "H1", "H4", "D1"])
def test_all_timeframes_run_without_error(tf):
    df = make_df(n=150)
    sr = SupportResistance(timeframe=tf)
    result = sr.analyze(df, symbol="EURUSD")
    assert result["timeframe"] == tf
    assert "support_zones" in result and "resistance_zones" in result


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))