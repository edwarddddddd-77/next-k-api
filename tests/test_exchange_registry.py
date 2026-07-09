"""实盘交易所 registry 测试。"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from quant.common.exchange_env import resolve_live_exchange_id
from quant.engine.exchanges.registry import (
    EXCHANGE_BINANCE,
    EXCHANGE_BYBIT,
    get_adapter,
    symbol_from_vt,
    vnpy_vt_symbol,
)


class TestExchangeRegistry(unittest.TestCase):
    def test_resolve_defaults_binance(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_live_exchange_id(), EXCHANGE_BINANCE)

    def test_resolve_lane_env_beats_global(self):
        with mock.patch.dict(
            os.environ,
            {"ORB_VNPY_LIVE_EXCHANGE": "bybit", "VNPY_LIVE_EXCHANGE": "binance"},
            clear=True,
        ):
            self.assertEqual(resolve_live_exchange_id(), EXCHANGE_BYBIT)

    def test_resolve_unknown_falls_back(self):
        with mock.patch.dict(os.environ, {"VNPY_LIVE_EXCHANGE": "okx"}, clear=True):
            self.assertEqual(resolve_live_exchange_id(), EXCHANGE_BINANCE)

    def test_binance_vt_symbol_roundtrip(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            vt = vnpy_vt_symbol("intc")
        self.assertEqual(vt, "INTCUSDT_SWAP_BINANCE.GLOBAL")
        self.assertEqual(symbol_from_vt(vt), "INTCUSDT")

    def test_bybit_vt_symbol_format(self):
        with mock.patch.dict(os.environ, {"VNPY_LIVE_EXCHANGE": "bybit"}, clear=True):
            adapter = get_adapter()
        self.assertEqual(adapter.id, EXCHANGE_BYBIT)
        self.assertEqual(adapter.vnpy_vt_symbol("ETH"), "ETHUSDT_SWAP_BYBIT.GLOBAL")
        self.assertEqual(adapter.symbol_from_vt("ETHUSDT_SWAP_BYBIT.GLOBAL"), "ETHUSDT")


if __name__ == "__main__":
    unittest.main()
