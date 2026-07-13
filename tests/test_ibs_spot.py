"""IBS 现货标的与 Bitget spot 适配测试。"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from quant.engine.exchanges.registry import EXCHANGE_BITGET_SPOT, get_adapter, symbol_from_vt, vnpy_vt_symbol
from quant.ibs.symbols import resolve_ibs_trading_symbol
from quant.ibs_conservative.config import IbsConservativeConfig
from quant.market.registry import get_market_adapter


class TestIbsSpotSymbols(unittest.TestCase):
    def test_resolve_spy_to_ondo_spot(self):
        self.assertEqual(resolve_ibs_trading_symbol("SPY", "spot"), "SPYONUSDT")
        self.assertEqual(resolve_ibs_trading_symbol("QQQ", "spot"), "QQQONUSDT")
        self.assertEqual(resolve_ibs_trading_symbol("TQQQ", "spot"), "RTQQQUSDT")

    def test_resolve_perp_unchanged(self):
        self.assertEqual(resolve_ibs_trading_symbol("SPY", "perp"), "SPYUSDT")

    def test_bitget_spot_vt_symbol_roundtrip(self):
        with mock.patch.dict(os.environ, {"VNPY_LIVE_EXCHANGE": "bitget_spot"}, clear=True):
            adapter = get_adapter()
        self.assertEqual(adapter.id, EXCHANGE_BITGET_SPOT)
        vt = adapter.vnpy_vt_symbol("SPYONUSDT")
        self.assertEqual(vt, "SPYONUSDT_SPOT_BITGET.GLOBAL")
        self.assertEqual(adapter.symbol_from_vt(vt), "SPYONUSDT")

    def test_ibs_perp_override(self):
        env = {
            "STRATEGY_IBS_CONSERVATIVE_ENABLED": "1",
            "IBS_CONSERVATIVE_VNPY_PRODUCT_TYPE": "perp",
            "VNPY_LIVE_EXCHANGE": "bitget",
            "MARKET_DATA_EXCHANGE": "bitget",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = IbsConservativeConfig.from_env()
        self.assertEqual(cfg.product_type, "perp")
        self.assertEqual(cfg.live_exchange, "bitget")
        self.assertIn("SPYUSDT", cfg.symbol_list())

    def test_bitget_spot_market_adapter_registered(self):
        adapter = get_market_adapter("bitget_spot")
        self.assertEqual(adapter.id, "bitget_spot")


if __name__ == "__main__":
    unittest.main()
