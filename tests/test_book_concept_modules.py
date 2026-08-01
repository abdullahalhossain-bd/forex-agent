"""
tests/test_book_concept_modules.py — Unit tests for the new book-concept
modules: analysis/adx_filters.py, analysis/megaphone_pennant.py,
risk/channel_breakout_stops.py.

Run: python -m pytest tests/test_book_concept_modules.py -v
  or: python tests/test_book_concept_modules.py
"""

from __future__ import annotations

import math
import os
import sys
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from analysis.adx_filters import adx_rising, bishop_exit
from analysis.megaphone_pennant import classify, Formation
from risk.channel_breakout_stops import (
    five_day_condition,
    evaluate_rejection,
    last_bar_stop,
    select_active_stop,
)


class TestAdxRising(unittest.TestCase):
    def test_strictly_increasing_is_rising(self):
        self.assertTrue(adx_rising(25.0, 22.0))

    def test_flat_is_not_rising(self):
        self.assertFalse(adx_rising(25.0, 25.0))

    def test_declining_is_not_rising(self):
        self.assertFalse(adx_rising(20.0, 25.0))

    def test_nan_is_not_rising(self):
        self.assertFalse(adx_rising(float("nan"), 20.0))
        self.assertFalse(adx_rising(20.0, float("nan")))

    def test_missing_is_not_rising(self):
        self.assertFalse(adx_rising(None, 20.0))
        self.assertFalse(adx_rising(20.0, None))


class TestBishopExit(unittest.TestCase):
    def test_fires_on_downtick_after_above_40(self):
        self.assertTrue(bishop_exit(39.0, 41.0))

    def test_fires_on_small_downtick(self):
        self.assertTrue(bishop_exit(40.99, 41.0))

    def test_does_not_fire_below_arm_level(self):
        self.assertFalse(bishop_exit(36.0, 38.0))  # never crossed 40

    def test_does_not_fire_at_exactly_40(self):
        # prior bar must be STRICTLY above 40 to arm the rule
        self.assertFalse(bishop_exit(39.5, 40.0))

    def test_does_not_fire_on_uptick(self):
        self.assertFalse(bishop_exit(43.0, 41.0))

    def test_does_not_fire_on_flat(self):
        self.assertFalse(bishop_exit(45.0, 45.0))

    def test_fires_again_on_second_downtick_after_a_bounce(self):
        # 45 -> 43 (fires) -> 44 (no) -> 41 (fires again)
        self.assertTrue(bishop_exit(43.0, 45.0))
        self.assertFalse(bishop_exit(44.0, 43.0))
        self.assertTrue(bishop_exit(41.0, 44.0))

    def test_nan_safe(self):
        self.assertFalse(bishop_exit(float("nan"), 45.0))
        self.assertFalse(bishop_exit(39.0, float("nan")))

    def test_custom_arm_level(self):
        self.assertTrue(bishop_exit(29.0, 31.0, arm_level=30.0))
        self.assertFalse(bishop_exit(29.0, 31.0, arm_level=40.0))


class TestMegaphonePennant(unittest.TestCase):
    def test_megaphone(self):
        result = classify(swing_highs=[100.0, 105.0], swing_lows=[90.0, 85.0])
        self.assertEqual(result.formation, Formation.MEGAPHONE)

    def test_pennant(self):
        result = classify(swing_highs=[105.0, 102.0], swing_lows=[90.0, 92.0])
        self.assertEqual(result.formation, Formation.PENNANT)
        self.assertEqual(result.buy_stop, 102.0)
        self.assertEqual(result.sell_stop, 92.0)

    def test_ordinary_uptrend_is_unknown(self):
        # higher high AND higher low = trending, not a megaphone/pennant
        result = classify(swing_highs=[100.0, 105.0], swing_lows=[90.0, 95.0])
        self.assertEqual(result.formation, Formation.UNKNOWN)

    def test_insufficient_swings(self):
        result = classify(swing_highs=[100.0], swing_lows=[90.0, 85.0])
        self.assertEqual(result.formation, Formation.UNKNOWN)
        self.assertIn("insufficient", result.reason)

    def test_exact_tie_is_unknown(self):
        result = classify(swing_highs=[100.0, 100.0], swing_lows=[90.0, 85.0])
        self.assertEqual(result.formation, Formation.UNKNOWN)

    def test_nan_safe(self):
        result = classify(swing_highs=[100.0, float("nan")], swing_lows=[90.0, 85.0])
        self.assertEqual(result.formation, Formation.UNKNOWN)


class TestFiveDayCondition(unittest.TestCase):
    def test_flat_declining_level_arms_for_buy(self):
        # non-increasing sequence
        self.assertTrue(five_day_condition([110, 109, 109, 108, 107], "BUY"))

    def test_rising_level_does_not_arm_for_buy(self):
        self.assertFalse(five_day_condition([100, 101, 102, 103, 104], "BUY"))

    def test_flat_rising_level_arms_for_sell(self):
        self.assertTrue(five_day_condition([90, 91, 91, 92, 93], "SELL"))

    def test_insufficient_history(self):
        self.assertFalse(five_day_condition([100, 99], "BUY", lookback=5))

    def test_nan_in_window(self):
        self.assertFalse(five_day_condition([110, 109, float("nan"), 108, 107], "BUY"))

    def test_invalid_direction_raises(self):
        with self.assertRaises(ValueError):
            five_day_condition([1, 2, 3, 4, 5], "SIDEWAYS")


class TestRejectionRule(unittest.TestCase):
    def test_triggers_on_bar1_close_back_through_for_buy(self):
        check = evaluate_rejection("BUY", breakout_level=1.5000, bar_close=1.4990,
                                    bars_in_trade=1, five_day_condition_met=True)
        self.assertTrue(check.triggered)

    def test_triggers_on_bar2(self):
        check = evaluate_rejection("BUY", breakout_level=1.5000, bar_close=1.4995,
                                    bars_in_trade=2, five_day_condition_met=True)
        self.assertTrue(check.triggered)

    def test_does_not_trigger_on_bar3(self):
        check = evaluate_rejection("BUY", breakout_level=1.5000, bar_close=1.4990,
                                    bars_in_trade=3, five_day_condition_met=True)
        self.assertFalse(check.triggered)

    def test_inactive_when_five_day_condition_not_met(self):
        check = evaluate_rejection("BUY", breakout_level=1.5000, bar_close=1.4990,
                                    bars_in_trade=1, five_day_condition_met=False)
        self.assertFalse(check.triggered)

    def test_no_trigger_when_close_holds_above_level_buy(self):
        check = evaluate_rejection("BUY", breakout_level=1.5000, bar_close=1.5010,
                                    bars_in_trade=1, five_day_condition_met=True)
        self.assertFalse(check.triggered)

    def test_sell_direction_mirrors_buy(self):
        check = evaluate_rejection("SELL", breakout_level=1.5000, bar_close=1.5010,
                                    bars_in_trade=1, five_day_condition_met=True)
        self.assertTrue(check.triggered)


class TestLastBarStop(unittest.TestCase):
    def test_uses_breakout_bar_low_normally(self):
        stop = last_bar_stop(
            "BUY", breakout_bar_high=1.5050, breakout_bar_low=1.4980,
            breakout_bar_open=1.4960, breakout_level=1.5000,
            prior_bar_high=1.4990, prior_bar_low=1.4950,
            barely_clear_atr_fraction=0.05, atr=0.0020,
        )
        self.assertEqual(stop, 1.4980)

    def test_falls_back_to_prior_bar_on_gap_through(self):
        stop = last_bar_stop(
            "BUY", breakout_bar_high=1.5100, breakout_bar_low=1.5020,
            breakout_bar_open=1.5010,  # opened ABOVE the level -> gap-through
            breakout_level=1.5000,
            prior_bar_high=1.4990, prior_bar_low=1.4950,
            barely_clear_atr_fraction=0.05, atr=0.0020,
        )
        self.assertEqual(stop, 1.4950)

    def test_falls_back_to_prior_bar_when_barely_cleared(self):
        # threshold = 0.05 * 0.0020 = 0.0001; breakout high clears the level
        # by only 0.00005, well inside that threshold -> "barely cleared"
        stop = last_bar_stop(
            "BUY", breakout_bar_high=1.50005, breakout_bar_low=1.4980,
            breakout_bar_open=1.4960, breakout_level=1.5000,
            prior_bar_high=1.4990, prior_bar_low=1.4950,
            barely_clear_atr_fraction=0.05, atr=0.0020,
        )
        self.assertEqual(stop, 1.4950)

    def test_raises_on_non_positive_atr(self):
        with self.assertRaises(ValueError):
            last_bar_stop("BUY", 1.51, 1.49, 1.495, 1.50, 1.49, 1.48, 0.05, atr=0.0)


class TestSelectActiveStop(unittest.TestCase):
    def test_rejection_takes_priority(self):
        result = select_active_stop(
            "BUY", bars_in_trade=1, five_day_condition_met=True,
            breakout_level=1.5000, current_close=1.4990,
            last_bar_stop_price=1.4950, channel_stop_price=1.4900,
        )
        self.assertEqual(result["action"], "EXIT")
        self.assertEqual(result["stage"], "rejection")

    def test_last_bar_stage_when_tighter(self):
        result = select_active_stop(
            "BUY", bars_in_trade=3, five_day_condition_met=True,
            breakout_level=1.5000, current_close=1.5100,
            last_bar_stop_price=1.4950, channel_stop_price=1.4900,
        )
        self.assertEqual(result["action"], "HOLD")
        self.assertEqual(result["stage"], "last_bar")
        self.assertEqual(result["stop_price"], 1.4950)

    def test_channel_stage_once_tighter(self):
        result = select_active_stop(
            "BUY", bars_in_trade=10, five_day_condition_met=True,
            breakout_level=1.5000, current_close=1.5300,
            last_bar_stop_price=1.4950, channel_stop_price=1.5100,  # now tighter
        )
        self.assertEqual(result["action"], "HOLD")
        self.assertEqual(result["stage"], "channel")
        self.assertEqual(result["stop_price"], 1.5100)


if __name__ == "__main__":
    unittest.main()
