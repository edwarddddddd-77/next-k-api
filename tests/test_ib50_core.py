"""IB50 核心信号测试。"""

from __future__ import annotations

import unittest

from quant.ib50.core import (
    IntrabarOhlc,
    bar_exit_reason,
    build_ib50_setup,
    compute_midpoint,
    continuation_side,
    finalize_initial_balance,
    first_extreme_on_bar,
    parse_weekday_filter,
    replay_ib_from_bars,
    trade_side,
    update_ib_range,
    weekday_allowed,
)


class TestIb50Core(unittest.TestCase):
    def test_compute_midpoint(self):
        self.assertAlmostEqual(compute_midpoint(110.0, 100.0), 105.0)

    def test_first_extreme_on_bar_bearish(self):
        self.assertEqual(first_extreme_on_bar(open_=105.0, high=106.0, low=100.0), "low")

    def test_first_extreme_on_bar_bullish(self):
        self.assertEqual(first_extreme_on_bar(open_=101.0, high=110.0, low=100.0), "high")

    def test_update_ib_range_tracks_first_breakout(self):
        hi, lo, ext = update_ib_range(
            ib_high=105.0,
            ib_low=100.0,
            first_extreme=None,
            open_=104.0,
            high=104.5,
            low=98.0,
        )
        self.assertAlmostEqual(hi, 105.0)
        self.assertAlmostEqual(lo, 98.0)
        self.assertEqual(ext, "low")

    def test_continuation_low_first_is_long(self):
        ib = finalize_initial_balance(ib_high=110.0, ib_low=100.0, first_extreme="low")
        assert ib is not None
        self.assertEqual(continuation_side("low"), 1)
        setup = build_ib50_setup(ib, 105.0, direction_mode="continuation")
        self.assertEqual(setup.side, 1)
        self.assertAlmostEqual(setup.stop, 100.0)
        self.assertAlmostEqual(setup.target, 110.0)

    def test_inverse_flips_direction(self):
        ib = finalize_initial_balance(ib_high=110.0, ib_low=100.0, first_extreme="low")
        assert ib is not None
        self.assertEqual(trade_side("low", direction_mode="inverse"), -1)
        setup = build_ib50_setup(ib, 105.0, direction_mode="inverse")
        self.assertEqual(setup.side, -1)
        self.assertAlmostEqual(setup.stop, 110.0)
        self.assertAlmostEqual(setup.target, 100.0)

    def test_bar_exit_reason_stop_before_target(self):
        reason = bar_exit_reason(
            side=1,
            high=108.0,
            low=99.0,
            stop=100.0,
            target=110.0,
            prev_high=107.0,
            prev_low=101.0,
        )
        self.assertEqual(reason, "stop_loss")

    def test_bar_exit_reason_target_hit(self):
        reason = bar_exit_reason(
            side=1,
            high=111.0,
            low=104.0,
            stop=100.0,
            target=110.0,
            prev_high=108.0,
            prev_low=104.0,
        )
        self.assertEqual(reason, "target_hit")

    def test_weekday_filter_mon_tue_thu(self):
        allowed = parse_weekday_filter("mon,tue,thu")
        assert allowed is not None
        self.assertTrue(weekday_allowed(0, allowed))
        self.assertTrue(weekday_allowed(1, allowed))
        self.assertFalse(weekday_allowed(2, allowed))
        self.assertTrue(weekday_allowed(3, allowed))

    def test_replay_ib_from_bars(self):
        bars = [
            IntrabarOhlc(0, 105.0, 106.0, 100.0, 101.0),
            IntrabarOhlc(60_000, 101.0, 108.0, 101.0, 107.0),
            IntrabarOhlc(120_000, 107.0, 110.0, 106.0, 109.0),
        ]
        ib = replay_ib_from_bars(bars, anchor_ms=0, ib_minutes=60)
        assert ib is not None
        self.assertAlmostEqual(ib.high, 110.0)
        self.assertAlmostEqual(ib.low, 100.0)
        self.assertEqual(ib.first_extreme, "low")


if __name__ == "__main__":
    unittest.main()
