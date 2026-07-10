"""MtfMomo 配置与注册测试。"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from quant.engine.registry import get_enabled_vnpy_lanes, vnpy_lane_plugins
from quant.mtfmomo.config import MtfMomoConfig


class TestMtfMomoConfig(unittest.TestCase):
    def test_default_disabled(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("STRATEGY_MTFMOMO_ENABLED", None)
            os.environ.pop("MTFMOMO_VNPY_ENABLED", None)
            cfg = MtfMomoConfig.from_env()
        self.assertFalse(cfg.enabled)

    def test_enabled_via_switch(self):
        env = {"STRATEGY_MTFMOMO_ENABLED": "1"}
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = MtfMomoConfig.from_env()
        self.assertTrue(cfg.enabled)
        self.assertAlmostEqual(cfg.risk_pct, 0.02)

    def test_default_symbols_pool(self):
        cfg = MtfMomoConfig.from_env()
        pool = {s.replace("USDT", "") for s in cfg.symbol_list()}
        self.assertIn("SOL", pool)
        self.assertIn("ETH", pool)

    def test_plugin_registered(self):
        names = [p.name for p in vnpy_lane_plugins()]
        self.assertIn("mtfmomo", names)

    def test_lane_enabled_with_env(self):
        env = {
            "STRATEGY_MTFMOMO_ENABLED": "1",
            "STRATEGY_TRADING_ORB_ENABLED": "0",
            "ORB_VNPY_ENABLED": "0",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            lanes = get_enabled_vnpy_lanes()
        self.assertEqual([n for n, _ in lanes], ["mtfmomo"])


if __name__ == "__main__":
    unittest.main()
