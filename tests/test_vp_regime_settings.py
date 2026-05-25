"""vp_regime 周期默认。"""

from __future__ import annotations

import os
import unittest

import numpy as np
import pandas as pd

from vp_regime_scanner import (
    VPRegimeResult,
    VPSettings,
    _classify_on_closed,
    _interval_defaults,
    load_vp_settings,
)


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

    def test_classify_flat_branch_does_not_shadow_settings(self):
        """回归：flat 段 std 不得覆盖 VPSettings 变量 s（曾导致 flat_cv_max AttributeError）。"""
        n = 30
        vol = np.full(n, 200_000.0)
        df = pd.DataFrame(
            {
                "open_time": np.arange(n, dtype=np.int64) * 300_000,
                "open": np.full(n, 100.0),
                "high": np.full(n, 101.0),
                "low": np.full(n, 99.0),
                "close": np.full(n, 100.0),
                "volume": vol / 100.0,
                "vol_usd": vol,
                "range_pct": np.full(n, 0.001),
            }
        )
        settings = VPSettings(
            interval="5m",
            kline_limit=150,
            vol_ma_period=12,
            min_vol_usd_ma=0.0,
            spike_vol_mult=2.5,
            spike_range_min_pct=0.004,
            spike_burst_mult=1.8,
            increase_lookback=7,
            increase_min_up=4,
            flat_lookback=12,
            flat_cv_max=0.22,
        )
        res = _classify_on_closed(df, settings)
        self.assertIsInstance(res, VPRegimeResult)
        self.assertIn(res.scheme, ("MOMENTUM", "MEAN_REVERT", "REVERSAL_WATCH", "WATCH", "NO_TRADE"))


if __name__ == "__main__":
    unittest.main()
