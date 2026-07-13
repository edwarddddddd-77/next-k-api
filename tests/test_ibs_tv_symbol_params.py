"""IBS TV 按标的 AlgoTradeKit 参数对齐测试。"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from quant.ibs_tv.config import IbsTvConfig
from quant.ibs_tv.symbol_params import resolve_tv_symbol_params


class TestIbsTvSymbolParams(unittest.TestCase):
    def test_spy_recommended(self):
        sp = resolve_tv_symbol_params("SPYUSDT")
        self.assertAlmostEqual(sp.entry_threshold, 0.11)
        self.assertAlmostEqual(sp.exit_threshold, 0.995)
        self.assertEqual(sp.trend_ma_period, 200)
        self.assertEqual(sp.max_trade_duration_days, 12)

    def test_qqq_recommended(self):
        sp = resolve_tv_symbol_params("QQQ")
        self.assertAlmostEqual(sp.entry_threshold, 0.09)
        self.assertAlmostEqual(sp.exit_threshold, 0.985)
        self.assertEqual(sp.trend_ma_period, 220)
        self.assertEqual(sp.max_trade_duration_days, 14)

    def test_lane_config_for_symbol_spy(self):
        cfg = IbsTvConfig(
            lane="ibs_tv",
            profile="tv",
            entry_threshold=0.09,
            exit_threshold=0.985,
            trend_ma_period=220,
            max_trade_duration_days=14,
        )
        spy = cfg.lane_config_for_symbol("SPYUSDT")
        self.assertAlmostEqual(spy.entry_threshold, 0.11)
        self.assertAlmostEqual(spy.exit_threshold, 0.995)
        self.assertEqual(spy.trend_ma_period, 200)
        self.assertEqual(spy.max_trade_duration_days, 12)

    def test_env_override_beats_symbol_default(self):
        env = {"IBS_TV_VNPY_ENTRY_THRESHOLD": "0.20"}
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = IbsTvConfig(
                lane="ibs_tv",
                profile="tv",
                entry_threshold=0.20,
            )
            spy = cfg.lane_config_for_symbol("SPYUSDT")
        self.assertAlmostEqual(spy.entry_threshold, 0.20)


if __name__ == "__main__":
    unittest.main()
