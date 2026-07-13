"""IBS lane 配置、注册与标的池测试。"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from quant.engine.lane import find_symbol_pool_overlaps
from quant.engine.registry import get_enabled_vnpy_lanes, vnpy_lane_plugins
from quant.ibs.profile import PROFILE_AGGRESSIVE, PROFILE_CONSERVATIVE, PROFILE_TV
from quant.ibs_aggressive.config import IbsAggressiveConfig
from quant.ibs_conservative.config import IbsConservativeConfig
from quant.ibs_tv.config import IbsTvConfig


class TestIbsConfig(unittest.TestCase):
    def test_conservative_default_disabled(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("STRATEGY_IBS_CONSERVATIVE_ENABLED", None)
            cfg = IbsConservativeConfig.from_env()
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.profile, PROFILE_CONSERVATIVE)
        self.assertAlmostEqual(cfg.entry_threshold, 0.20)
        self.assertAlmostEqual(cfg.exit_threshold, 0.50)
        self.assertEqual(cfg.sma_period, 200)
        self.assertEqual(cfg.trend_price_mode, "prev_close")
        self.assertEqual(cfg.daily_bar_source, "session_5m")
        self.assertTrue(cfg.execute_at_next_open)
        self.assertAlmostEqual(cfg.stop_loss_pct, 0.0)
        self.assertEqual(cfg.trade_type, "long_only")

    def test_aggressive_default_disabled(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("STRATEGY_IBS_AGGRESSIVE_ENABLED", None)
            cfg = IbsAggressiveConfig.from_env()
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.profile, PROFILE_AGGRESSIVE)
        self.assertAlmostEqual(cfg.entry_threshold, 0.19)
        self.assertAlmostEqual(cfg.exit_threshold, 0.95)
        self.assertEqual(cfg.sma_period, 0)

    def test_conservative_pool(self):
        with mock.patch.dict(os.environ, {"STRATEGY_IBS_CONSERVATIVE_ENABLED": "1"}, clear=False):
            cfg = IbsConservativeConfig.from_env()
        self.assertTrue(cfg.enabled)
        self.assertIn("SPYONUSDT", cfg.symbol_list())
        self.assertEqual(cfg.product_type, "spot")
        self.assertEqual(cfg.live_exchange, "bitget_spot")

    def test_aggressive_pool(self):
        with mock.patch.dict(os.environ, {"STRATEGY_IBS_AGGRESSIVE_ENABLED": "1"}, clear=False):
            cfg = IbsAggressiveConfig.from_env()
        self.assertTrue(cfg.enabled)
        self.assertIn("RTQQQUSDT", cfg.symbol_list())

    def test_tv_default_disabled(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("STRATEGY_IBS_TV_ENABLED", None)
            cfg = IbsTvConfig.from_env()
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.profile, PROFILE_TV)
        self.assertAlmostEqual(cfg.entry_threshold, 0.09)
        self.assertAlmostEqual(cfg.exit_threshold, 0.985)
        self.assertEqual(cfg.trend_ma_type, "ema")
        self.assertEqual(cfg.trend_ma_period, 220)
        self.assertEqual(cfg.trend_price_mode, "current")
        self.assertAlmostEqual(cfg.stop_loss_pct, 0.0)
        self.assertEqual(cfg.max_trade_duration_days, 14)
        self.assertEqual(cfg.trade_type, "long_short")

    def test_tv_pool(self):
        with mock.patch.dict(os.environ, {"STRATEGY_IBS_TV_ENABLED": "1"}, clear=False):
            cfg = IbsTvConfig.from_env()
        self.assertTrue(cfg.enabled)
        self.assertIn("SPYUSDT", cfg.symbol_list())
        self.assertIn("QQQUSDT", cfg.symbol_list())
        self.assertEqual(cfg.product_type, "perp")
        self.assertEqual(cfg.trade_type, "long_short")
        self.assertEqual(cfg.live_exchange, "bitget")

    def test_plugins_registered(self):
        names = [p.name for p in vnpy_lane_plugins()]
        self.assertIn("ibs_conservative", names)
        self.assertIn("ibs_aggressive", names)
        self.assertIn("ibs_tv", names)

    def test_both_lanes_enabled(self):
        env = {
            "STRATEGY_IBS_CONSERVATIVE_ENABLED": "1",
            "STRATEGY_IBS_AGGRESSIVE_ENABLED": "1",
            "STRATEGY_TRADING_ORB_ENABLED": "0",
            "STRATEGY_MTFMOMO_ENABLED": "0",
            "STRATEGY_KAMA_TREND_ENABLED": "0",
            "STRATEGY_ANCHOR_DRIFT_ENABLED": "0",
            "STRATEGY_SQUEEZE_BREAKOUT_ENABLED": "0",
            "ORB_VNPY_ENABLED": "0",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            lanes = get_enabled_vnpy_lanes()
        self.assertEqual(sorted(n for n, _ in lanes), ["ibs_aggressive", "ibs_conservative"])

    def test_all_three_ibs_lanes_enabled(self):
        env = {
            "STRATEGY_IBS_CONSERVATIVE_ENABLED": "1",
            "STRATEGY_IBS_AGGRESSIVE_ENABLED": "1",
            "STRATEGY_IBS_TV_ENABLED": "1",
            "STRATEGY_TRADING_ORB_ENABLED": "0",
            "STRATEGY_MTFMOMO_ENABLED": "0",
            "STRATEGY_KAMA_TREND_ENABLED": "0",
            "STRATEGY_ANCHOR_DRIFT_ENABLED": "0",
            "STRATEGY_SQUEEZE_BREAKOUT_ENABLED": "0",
            "ORB_VNPY_ENABLED": "0",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            lanes = get_enabled_vnpy_lanes()
        self.assertEqual(
            sorted(n for n, _ in lanes),
            ["ibs_aggressive", "ibs_conservative", "ibs_tv"],
        )

    def test_no_overlap_between_ibs_lanes(self):
        conservative = IbsConservativeConfig(
            lane="ibs_conservative",
            profile=PROFILE_CONSERVATIVE,
            enabled=True,
            symbols=["SPYONUSDT"],
        )
        aggressive = IbsAggressiveConfig(
            lane="ibs_aggressive",
            profile=PROFILE_AGGRESSIVE,
            enabled=True,
            symbols=["RTQQQUSDT"],
        )
        tv = IbsTvConfig(
            lane="ibs_tv",
            profile=PROFILE_TV,
            enabled=True,
            symbols=["QQQUSDT"],
        )
        overlaps = find_symbol_pool_overlaps(
            [
                ("ibs_conservative", conservative),
                ("ibs_aggressive", aggressive),
                ("ibs_tv", tv),
            ]
        )
        self.assertEqual(overlaps, [])

    def test_spy_overlap_between_conservative_and_tv_perp(self):
        conservative = IbsConservativeConfig(
            lane="ibs_conservative",
            profile=PROFILE_CONSERVATIVE,
            enabled=True,
            product_type="perp",
            symbols=["SPYUSDT"],
        )
        tv = IbsTvConfig(
            lane="ibs_tv",
            profile=PROFILE_TV,
            enabled=True,
            product_type="perp",
            symbols=["SPYUSDT", "QQQUSDT"],
        )
        overlaps = find_symbol_pool_overlaps([("ibs_conservative", conservative), ("ibs_tv", tv)])
        self.assertEqual(overlaps, ["SPYUSDT"])

    def test_spot_conservative_and_perp_tv_no_symbol_overlap(self):
        conservative = IbsConservativeConfig(
            lane="ibs_conservative",
            profile=PROFILE_CONSERVATIVE,
            enabled=True,
            symbols=["SPY"],
        )
        tv = IbsTvConfig(
            lane="ibs_tv",
            profile=PROFILE_TV,
            enabled=True,
            product_type="perp",
            symbols=["SPY", "QQQ"],
        )
        self.assertEqual(conservative.symbol_list(), ["SPYONUSDT"])
        self.assertEqual(tv.symbol_list(), ["SPYUSDT", "QQQUSDT"])
        overlaps = find_symbol_pool_overlaps([("ibs_conservative", conservative), ("ibs_tv", tv)])
        self.assertEqual(overlaps, [])

    def test_tv_trade_type_long_short_env(self):
        env = {
            "STRATEGY_IBS_TV_ENABLED": "1",
            "IBS_TV_VNPY_TRADE_TYPE": "long_short",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = IbsTvConfig.from_env()
        self.assertEqual(cfg.trade_type, "long_short")

    def test_conservative_trade_type_forced_long_only(self):
        env = {
            "STRATEGY_IBS_CONSERVATIVE_ENABLED": "1",
            "IBS_CONSERVATIVE_VNPY_TRADE_TYPE": "long_short",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = IbsConservativeConfig.from_env()
        self.assertEqual(cfg.trade_type, "long_only")

    def test_overlap_detected_between_ibs_lanes(self):
        conservative = IbsConservativeConfig(
            lane="ibs_conservative",
            profile=PROFILE_CONSERVATIVE,
            enabled=True,
            symbols=["MSTRUSDT"],
        )
        aggressive = IbsAggressiveConfig(
            lane="ibs_aggressive",
            profile=PROFILE_AGGRESSIVE,
            enabled=True,
            symbols=["MSTRUSDT"],
        )
        overlaps = find_symbol_pool_overlaps(
            [("ibs_conservative", conservative), ("ibs_aggressive", aggressive)]
        )
        self.assertEqual(overlaps, ["MSTRUSDT"])


if __name__ == "__main__":
    unittest.main()
