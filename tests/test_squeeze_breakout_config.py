"""Smart Breakout 配置与注册测试。"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from quant.engine.registry import get_enabled_vnpy_lanes, vnpy_lane_plugins
from quant.squeeze_breakout.config import SqueezeBreakoutConfig


class TestSqueezeBreakoutConfig(unittest.TestCase):
    def test_default_disabled(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("STRATEGY_SQUEEZE_BREAKOUT_ENABLED", None)
            cfg = SqueezeBreakoutConfig.from_env()
        self.assertFalse(cfg.enabled)

    def test_enabled_pool(self):
        with mock.patch.dict(os.environ, {"STRATEGY_SQUEEZE_BREAKOUT_ENABLED": "1"}, clear=False):
            cfg = SqueezeBreakoutConfig.from_env()
        self.assertTrue(cfg.enabled)
        self.assertIn("AVAXUSDT", cfg.symbol_list())
        self.assertIn("LINKUSDT", cfg.symbol_list())

    def test_plugin_registered(self):
        self.assertIn("squeeze_breakout", [p.name for p in vnpy_lane_plugins()])

    def test_lane_enabled(self):
        env = {
            "STRATEGY_SQUEEZE_BREAKOUT_ENABLED": "1",
            "STRATEGY_TRADING_ORB_ENABLED": "0",
            "STRATEGY_MTFMOMO_ENABLED": "0",
            "STRATEGY_KAMA_TREND_ENABLED": "0",
            "ORB_VNPY_ENABLED": "0",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            lanes = get_enabled_vnpy_lanes()
        self.assertEqual([n for n, _ in lanes], ["squeeze_breakout"])


if __name__ == "__main__":
    unittest.main()
