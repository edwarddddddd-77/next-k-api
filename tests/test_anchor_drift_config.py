"""Anchor Drift 配置、注册与标的池测试。"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from quant.anchor_drift.config import AnchorDriftConfig
from quant.engine.lane import find_symbol_pool_overlaps
from quant.engine.registry import get_enabled_vnpy_lanes, vnpy_lane_plugins
from quant.trading_orb.config import OrbVnpyConfig


class TestAnchorDriftConfig(unittest.TestCase):
    def test_default_disabled(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("STRATEGY_ANCHOR_DRIFT_ENABLED", None)
            cfg = AnchorDriftConfig.from_env()
        self.assertFalse(cfg.enabled)

    def test_enabled_and_pool(self):
        with mock.patch.dict(os.environ, {"STRATEGY_ANCHOR_DRIFT_ENABLED": "1"}, clear=False):
            cfg = AnchorDriftConfig.from_env()
        self.assertTrue(cfg.enabled)
        self.assertTrue(cfg.non_rth_only)
        self.assertIn("MSTRUSDT", cfg.symbol_list())

    def test_plugin_registered(self):
        self.assertIn("anchor_drift", [p.name for p in vnpy_lane_plugins()])

    def test_lane_enabled(self):
        env = {
            "STRATEGY_ANCHOR_DRIFT_ENABLED": "1",
            "STRATEGY_TRADING_ORB_ENABLED": "0",
            "STRATEGY_MTFMOMO_ENABLED": "0",
            "ORB_VNPY_ENABLED": "0",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            lanes = get_enabled_vnpy_lanes()
        self.assertEqual([n for n, _ in lanes], ["anchor_drift"])

    def test_time_complementary_overlap_allowed(self):
        orb = OrbVnpyConfig(
            enabled=True,
            symbols=["MSTRUSDT"],
            rth_only=True,
            eod_flat=True,
        )
        drift = AnchorDriftConfig(
            enabled=True,
            symbols=["MSTRUSDT"],
            non_rth_only=True,
        )
        overlaps = find_symbol_pool_overlaps([("trading_orb", orb), ("anchor_drift", drift)])
        self.assertEqual(overlaps, [])


if __name__ == "__main__":
    unittest.main()
