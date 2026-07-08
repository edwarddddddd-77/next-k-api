"""官方 BinanceLinearGateway 辅助函数与守卫测试。"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from orb.trading_orb.config import OrbVnpyConfig
from orb.vnpy.bootstrap import ensure_vnpy_path

ensure_vnpy_path()

from vnpy.event import EventEngine  # noqa: E402
from vnpy.trader.constant import Direction, Exchange, Offset, OrderType  # noqa: E402
from vnpy.trader.object import OrderRequest, TradeData  # noqa: E402

from orb.vnpy.binance_gateway import (  # noqa: E402
    VnpyBinanceLinearGateway,
    binance_connect_setting,
    binance_credentials_configured,
    symbol_from_vt,
    vnpy_vt_symbol,
)


class TestBinanceGatewayHelpers(unittest.TestCase):
    def test_vnpy_vt_symbol(self):
        self.assertEqual(vnpy_vt_symbol("intc"), "INTCUSDT_SWAP_BINANCE.GLOBAL")
        self.assertEqual(vnpy_vt_symbol("INTC"), "INTCUSDT_SWAP_BINANCE.GLOBAL")

    def test_symbol_from_vt(self):
        self.assertEqual(symbol_from_vt("INTCUSDT_SWAP_BINANCE.GLOBAL"), "INTCUSDT")
        self.assertEqual(symbol_from_vt("INTCUSDT_SWAP_BINANCE"), "INTCUSDT")

    def test_binance_credentials_configured(self):
        with mock.patch.dict(
            os.environ,
            {"BINANCE_API_KEY": "k", "BINANCE_API_SECRET": "s"},
            clear=False,
        ):
            self.assertTrue(binance_credentials_configured())
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(binance_credentials_configured())

    def test_binance_connect_setting_orb_defaults_kline_stream(self):
        orb = OrbVnpyConfig(enabled=True, engine="vnpy")
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch(
                "orb.vnpy.binance_gateway.get_enabled_vnpy_lanes",
                return_value=[("trading_orb", orb)],
            ):
                s = binance_connect_setting()
        self.assertEqual(s["Kline Stream"], "True")

    def test_binance_connect_setting(self):
        with mock.patch(
            "orb.vnpy.binance_gateway.get_enabled_vnpy_lanes",
            return_value=[],
        ):
            with mock.patch.dict(
                os.environ,
                {
                    "BINANCE_API_KEY": "key1",
                    "BINANCE_API_SECRET": "sec1",
                    "BINANCE_SERVER": "TESTNET",
                    "BINANCE_KLINE_STREAM": "true",
                    "BINANCE_PROXY_HOST": "127.0.0.1",
                    "BINANCE_PROXY_PORT": "7890",
                },
                clear=True,
            ):
                s = binance_connect_setting()
        self.assertEqual(s["API Key"], "key1")
        self.assertEqual(s["API Secret"], "sec1")
        self.assertEqual(s["Server"], "TESTNET")
        self.assertEqual(s["Kline Stream"], "True")
        self.assertEqual(s["Proxy Host"], "127.0.0.1")
        self.assertEqual(s["Proxy Port"], 7890)


class TestVnpyBinanceGatewayGuards(unittest.TestCase):
    def setUp(self) -> None:
        self._lane_patcher = mock.patch(
            "orb.vnpy.binance_gateway.cfg_for_symbol",
            return_value=OrbVnpyConfig.from_env(),
        )
        self._lane_patcher.start()

    def tearDown(self) -> None:
        self._lane_patcher.stop()

    def _gateway(self) -> VnpyBinanceLinearGateway:
        return VnpyBinanceLinearGateway(EventEngine())

    def _open_req(self, symbol: str = "INTCUSDT_SWAP_BINANCE") -> OrderRequest:
        return OrderRequest(
            symbol=symbol,
            exchange=Exchange.GLOBAL,
            direction=Direction.LONG,
            type=OrderType.LIMIT,
            volume=1.0,
            price=100.0,
            offset=Offset.OPEN,
        )

    @mock.patch("orb.vnpy.binance_gateway.BinanceLinearGateway.send_order")
    def test_shadow_rejects_order(self, mock_super):
        with mock.patch.dict(os.environ, {"ORB_VNPY_SHADOW": "1"}, clear=False):
            gw = self._gateway()
            out = gw.send_order(self._open_req())
        self.assertEqual(out, "")
        mock_super.assert_not_called()

    @mock.patch("orb.vnpy.binance_gateway.lane_live_enabled", return_value=False)
    @mock.patch("orb.vnpy.binance_gateway.BinanceLinearGateway.send_order")
    def test_live_disabled_rejects_order(self, mock_super, _live):
        gw = self._gateway()
        out = gw.send_order(self._open_req())
        self.assertEqual(out, "")
        mock_super.assert_not_called()

    @mock.patch("orb.vnpy.binance_gateway.lane_live_enabled", return_value=True)
    @mock.patch("orb.vnpy.binance_gateway.BinanceLinearGateway.send_order", return_value="BINANCE_LINEAR.1")
    def test_live_enabled_forwards_order(self, mock_super, _live):
        gw = self._gateway()
        out = gw.send_order(self._open_req())
        self.assertEqual(out, "BINANCE_LINEAR.1")
        mock_super.assert_called_once()

    @mock.patch("orb.vnpy.binance_gateway.lane_live_enabled", return_value=True)
    @mock.patch("orb.vnpy.binance_gateway.BinanceLinearGateway.send_order")
    def test_zero_volume_rejects_order(self, mock_super, _live):
        gw = self._gateway()
        req = self._open_req()
        req.volume = 0.0
        out = gw.send_order(req)
        self.assertEqual(out, "")
        mock_super.assert_not_called()

    @mock.patch("orb.vnpy.binance_gateway.lane_live_enabled", return_value=True)
    @mock.patch("orb.vnpy.binance_gateway.cfg_for_symbol")
    @mock.patch("orb.vnpy.binance_gateway.BinanceLinearGateway.send_order")
    def test_max_open_positions_rejects(self, mock_super, mock_cfg, _live):
        mock_cfg.return_value = OrbVnpyConfig(
            live_enabled=True,
            max_open_positions=2,
            symbols=["INTCUSDT", "SOXLUSDT", "HOODUSDT"],
        )
        gw = self._gateway()
        gw._active_symbols = {"SOXLUSDT", "HOODUSDT"}
        out = gw.send_order(self._open_req("INTCUSDT_SWAP_BINANCE"))
        self.assertEqual(out, "")
        mock_super.assert_not_called()

    @mock.patch("orb.vnpy.binance_gateway.orb_record_vnpy_fill")
    @mock.patch("orb.vnpy.binance_gateway.lane_live_enabled", return_value=True)
    @mock.patch("orb.vnpy.binance_gateway.BinanceLinearGateway.on_trade")
    def test_on_trade_persists_open(self, mock_super, _live, mock_record):
        gw = self._gateway()
        trade = TradeData(
            symbol="INTCUSDT_SWAP_BINANCE",
            exchange=Exchange.GLOBAL,
            orderid="1",
            tradeid="t1",
            direction=Direction.LONG,
            offset=Offset.OPEN,
            price=50.0,
            volume=2.0,
            gateway_name=gw.gateway_name,
        )
        gw.on_trade(trade)
        mock_super.assert_called_once()
        mock_record.assert_called_once()
        self.assertEqual(mock_record.call_args.kwargs["event"], "open")
        self.assertEqual(mock_record.call_args.kwargs["symbol"], "INTCUSDT")


if __name__ == "__main__":
    unittest.main()
