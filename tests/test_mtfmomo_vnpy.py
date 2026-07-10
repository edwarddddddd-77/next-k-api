"""MtfMomo vnpy 策略测试。"""

from __future__ import annotations

import unittest
from unittest import mock

from quant.mtfmomo.mtfmomo_vnpy import MtfMomoVnpyStrategy


class TestMtfMomoVnpyStrategy(unittest.TestCase):
    def _strategy(self) -> MtfMomoVnpyStrategy:
        strat = MtfMomoVnpyStrategy.__new__(MtfMomoVnpyStrategy)
        strat.entry_lb = 26
        strat.ema_exit = 35
        strat.ema_4h = 21
        strat.ema_1d = 16
        strat.stop_atr = 3.3
        strat.tp_atr = 8.7
        strat.pos = 0
        strat.vt_symbol = "SOLUSDT_SWAP_BINANCE.GLOBAL"
        strat._hour_bars = [
            (i * 3_600_000, 100.0, 101.0, 99.0, 100.0 + i * 0.1)
            for i in range(80)
        ]
        strat._anchor_4h_closes = []
        strat._anchor_1d_closes = []
        strat.entry_price = 0.0
        strat.stop_price = 0.0
        strat.target_price = 0.0
        strat.write_log = mock.MagicMock()
        return strat

    def test_restore_synced_position_sets_levels(self):
        strat = self._strategy()
        strat.restore_synced_position(entry_px=180.0, pos=1.0)
        self.assertGreater(strat.stop_price, 0.0)
        self.assertGreater(strat.target_price, strat.entry_price)
        self.assertLess(strat.stop_price, strat.entry_price)


if __name__ == "__main__":
    unittest.main()
