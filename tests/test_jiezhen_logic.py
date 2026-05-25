"""接针距离与触价逻辑单元测试。"""

from __future__ import annotations

import unittest

from jiezhen_logic import (
    build_spike_plan,
    ema_last,
    klines_chronological,
    selected_distance_pct,
)


def _kl(close: float, *, high=None, low=None) -> list:
    h = high if high is not None else close * 1.002
    lo = low if low is not None else close * 0.998
    return [0, close, h, lo, close, 0]


class TestJiezhenLogic(unittest.TestCase):
    def test_binance_klines_order_preserved(self):
        rows = [_kl(float(i)) for i in range(5)]
        chrono = klines_chronological(rows)
        self.assertEqual(float(chrono[-1][4]), 4.0)

    def test_ema_last(self):
        closes = [1.0, 2.0, 3.0, 4.0, 5.0]
        v = ema_last(closes, 3)
        self.assertGreater(v, 0)

    def test_selected_min_mode_floor(self):
        sel, _ = selected_distance_pct(
            mark=100.0,
            atr=1.0,
            average_amplitude_pct=0.2,
            value_multiplier=2.0,
            min_distance_pct=0.8,
            distance_mode="min",
        )
        self.assertGreaterEqual(sel, 0.8)

    def test_long_fill_when_low_touches_target(self):
        mark = 100.0
        rows = [_kl(90.0 + i * 0.08) for i in range(119)]
        # 最新一根：low 刺到 ~1% 下方接针价
        rows.append(_kl(100.0, high=101.0, low=98.5))
        plan = build_spike_plan(
            mark=mark,
            klines_raw=rows,
            ema_period=60,
            atr_period=60,
            amplitude_period=60,
            value_multiplier=2.0,
            min_distance_pct=0.8,
            distance_mode="min",
            touch_lookback_bars=1,
        )
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertLess(plan.target_long, mark)
        if plan.is_bullish:
            self.assertTrue(plan.long_fill)

    def test_long_fill_false_without_touch(self):
        mark = 100.0
        rows = [_kl(100.0, high=100.5, low=99.9) for _ in range(120)]
        plan = build_spike_plan(
            mark=mark,
            klines_raw=rows,
            ema_period=60,
            atr_period=60,
            amplitude_period=60,
            value_multiplier=2.0,
            min_distance_pct=0.8,
            distance_mode="min",
            touch_lookback_bars=1,
        )
        assert plan is not None
        if plan.is_bullish:
            self.assertFalse(plan.long_fill)

    def test_ema_zero_allows_both_sides(self):
        rows = [_kl(50.0) for _ in range(80)]
        plan = build_spike_plan(
            mark=50.0,
            klines_raw=rows,
            ema_period=0,
            atr_period=30,
            amplitude_period=30,
            value_multiplier=1.0,
            min_distance_pct=0.5,
            distance_mode="min",
        )
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertTrue(plan.is_bullish)
        self.assertTrue(plan.is_bearish)


if __name__ == "__main__":
    unittest.main()
