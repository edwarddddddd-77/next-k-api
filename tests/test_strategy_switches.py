"""策略开关解析与注册表。"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from quant.common.strategy_switch import StrategySwitchSpec, resolve_switch, vnpy_master_enabled
from quant.engine.registry import get_enabled_vnpy_lanes, list_strategy_switch_status


class TestStrategySwitch(unittest.TestCase):
    def test_unified_enabled_key(self):
        spec = StrategySwitchSpec(
            lane="demo",
            title="Demo",
            enabled_keys=("STRATEGY_DEMO_ENABLED", "DEMO_ENABLED"),
        )
        with mock.patch.dict(os.environ, {"STRATEGY_DEMO_ENABLED": "1"}, clear=False):
            self.assertTrue(spec.enabled())

    def test_legacy_enabled_fallback(self):
        spec = StrategySwitchSpec(
            lane="demo",
            title="Demo",
            enabled_keys=("STRATEGY_DEMO_ENABLED", "DEMO_ENABLED"),
        )
        with mock.patch.dict(os.environ, {"DEMO_ENABLED": "1"}, clear=False):
            os.environ.pop("STRATEGY_DEMO_ENABLED", None)
            self.assertTrue(spec.enabled())

    def test_vnpy_master_switch_off(self):
        with mock.patch.dict(os.environ, {"VNPY_ENABLED": "0", "STRATEGY_TRADING_ORB_ENABLED": "1"}):
            self.assertFalse(vnpy_master_enabled())
            self.assertEqual(get_enabled_vnpy_lanes(), [])

    def test_trading_orb_enabled_via_unified_key(self):
        env = {
            "STRATEGY_TRADING_ORB_ENABLED": "1",
            "STRATEGY_TRADING_ORB_LIVE": "0",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            os.environ.pop("ORB_VNPY_ENABLED", None)
            lanes = get_enabled_vnpy_lanes()
        self.assertEqual([name for name, _ in lanes], ["trading_orb"])

    def test_list_strategy_switch_status_shape(self):
        with mock.patch.dict(os.environ, {"STRATEGY_TRADING_ORB_ENABLED": "0"}, clear=False):
            out = list_strategy_switch_status()
        self.assertTrue(out["ok"])
        self.assertIn("strategies", out)
        self.assertEqual(out["strategies"][0]["id"], "trading_orb")
        self.assertEqual(out["strategies"][0]["env"]["enabled"], "STRATEGY_TRADING_ORB_ENABLED")


if __name__ == "__main__":
    unittest.main()
