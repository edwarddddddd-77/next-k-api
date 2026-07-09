"""Aberration 核心逻辑测试。"""

from __future__ import annotations

import unittest

from orb.aberration.core import aberration_action, aberration_bands


class TestAberrationCore(unittest.TestCase):
    def test_bands_requires_enough_closes(self):
        self.assertIsNone(aberration_bands([1.0] * 10, n_period=35))

    def test_bands_symmetric_k(self):
        closes = [float(100 + i) for i in range(40)]
        upper, mid, lower = aberration_bands(closes, n_period=35, k_up=2.0, k_down=2.0)
        self.assertGreater(upper, mid)
        self.assertLess(lower, mid)

    def test_action_breakout_long(self):
        act = aberration_action(0, 110.0, upper=105.0, middle=100.0, lower=95.0)
        self.assertEqual(act, "long")

    def test_action_breakout_short(self):
        act = aberration_action(0, 90.0, upper=105.0, middle=100.0, lower=95.0)
        self.assertEqual(act, "short")

    def test_action_close_long_at_mid(self):
        act = aberration_action(1, 99.0, upper=105.0, middle=100.0, lower=95.0)
        self.assertEqual(act, "close_long")

    def test_action_close_short_at_mid(self):
        act = aberration_action(-1, 101.0, upper=105.0, middle=100.0, lower=95.0)
        self.assertEqual(act, "close_short")


if __name__ == "__main__":
    unittest.main()
