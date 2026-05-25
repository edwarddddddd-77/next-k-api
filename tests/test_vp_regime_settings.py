"""vp_regime 周期默认。"""

from __future__ import annotations

import os
import unittest

from vp_regime_scanner import _interval_defaults, load_vp_settings


class TestVpRegimeSettings(unittest.TestCase):
    def test_interval_defaults_5m(self):
        ma, lim = _interval_defaults("5m")
        self.assertEqual(ma, 12)
        self.assertEqual(lim, 150)

    def test_interval_defaults_1m(self):
        ma, lim = _interval_defaults("1m")
        self.assertEqual(ma, 60)
        self.assertEqual(lim, 300)

    def test_load_5m_explicit(self):
        s = load_vp_settings(interval="5m")
        self.assertEqual(s.interval, "5m")
        self.assertEqual(s.vol_ma_period, 12)
        self.assertGreaterEqual(s.kline_limit, 42)


if __name__ == "__main__":
    unittest.main()
