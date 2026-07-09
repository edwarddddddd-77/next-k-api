"""交易所 / 行情环境变量解析测试。"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from quant.common.exchange_env import (
    resolve_live_exchange_id,
    resolve_lanes_live_exchange,
    resolve_market_data_exchange_id,
)
from quant.trading_orb.config import OrbVnpyConfig


class TestExchangeEnv(unittest.TestCase):
    def test_live_lane_env_beats_global(self):
        with mock.patch.dict(
            os.environ,
            {"ORB_VNPY_LIVE_EXCHANGE": "bybit", "VNPY_LIVE_EXCHANGE": "binance"},
            clear=True,
        ):
            self.assertEqual(resolve_live_exchange_id(), "bybit")

    def test_market_lane_env_beats_global(self):
        with mock.patch.dict(
            os.environ,
            {"ORB_MARKET_DATA_EXCHANGE": "bybit", "MARKET_DATA_EXCHANGE": "binance"},
            clear=True,
        ):
            self.assertEqual(resolve_market_data_exchange_id(), "bybit")

    def test_config_matches_registry_resolution(self):
        with mock.patch.dict(
            os.environ,
            {"ORB_VNPY_LIVE_EXCHANGE": "bybit", "VNPY_LIVE_EXCHANGE": "binance"},
            clear=True,
        ):
            cfg = OrbVnpyConfig.from_env()
            self.assertEqual(cfg.live_exchange, resolve_live_exchange_id())
            self.assertEqual(cfg.live_exchange, "bybit")

    def test_lanes_live_exchange_rejects_mismatch(self):
        cfg_a = OrbVnpyConfig(live_exchange="binance")
        cfg_b = OrbVnpyConfig(live_exchange="bybit")
        with self.assertRaises(ValueError):
            resolve_lanes_live_exchange([("a", cfg_a), ("b", cfg_b)])


if __name__ == "__main__":
    unittest.main()
