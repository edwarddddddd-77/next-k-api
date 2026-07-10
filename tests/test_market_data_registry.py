"""行情源 registry 测试。"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from quant.common.exchange_env import resolve_market_data_exchange_id
from quant.market.registry import (
    PROVIDER_BINANCE,
    PROVIDER_BITGET,
    get_market_adapter,
)


class TestMarketDataRegistry(unittest.TestCase):
    def test_resolve_defaults_binance(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_market_data_exchange_id(), PROVIDER_BINANCE)

    def test_resolve_bitget_env(self):
        with mock.patch.dict(os.environ, {"MARKET_DATA_EXCHANGE": "bitget"}, clear=True):
            self.assertEqual(resolve_market_data_exchange_id(), PROVIDER_BITGET)

    def test_bybit_alias_maps_to_bitget(self):
        with mock.patch.dict(os.environ, {"MARKET_DATA_EXCHANGE": "bybit"}, clear=True):
            self.assertEqual(resolve_market_data_exchange_id(), PROVIDER_BITGET)

    def test_resolve_falls_back_to_live_exchange(self):
        with mock.patch.dict(os.environ, {"VNPY_LIVE_EXCHANGE": "bitget"}, clear=True):
            self.assertEqual(resolve_market_data_exchange_id(), PROVIDER_BITGET)

    def test_market_data_overrides_live_exchange(self):
        with mock.patch.dict(
            os.environ,
            {"VNPY_LIVE_EXCHANGE": "bitget", "MARKET_DATA_EXCHANGE": "binance"},
            clear=True,
        ):
            self.assertEqual(resolve_market_data_exchange_id(), PROVIDER_BINANCE)

    def test_bitget_adapter_has_fetchers(self):
        with mock.patch.dict(os.environ, {"MARKET_DATA_EXCHANGE": "bitget"}, clear=True):
            adapter = get_market_adapter()
        self.assertEqual(adapter.id, PROVIDER_BITGET)
        self.assertTrue(callable(adapter.fetch_mark_price))
        self.assertTrue(callable(adapter.fetch_klines_forward))


if __name__ == "__main__":
    unittest.main()
