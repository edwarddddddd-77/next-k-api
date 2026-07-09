"""Aberration vnpy lane 测试。"""

from __future__ import annotations

import os
import unittest
from datetime import datetime, timezone
from unittest import mock

from orb.aberration.config import AberrationVnpyConfig
from orb.aberration.vnpy.strategies.aberration_vnpy import AberrationVnpyStrategy
from orb.vnpy.lane import find_symbol_pool_overlaps, get_enabled_vnpy_lanes


class TestAberrationVnpy(unittest.TestCase):
    def _strategy(self) -> AberrationVnpyStrategy:
        strat = AberrationVnpyStrategy.__new__(AberrationVnpyStrategy)
        strat.n_period = 5
        strat.k_up = 2.0
        strat.k_down = 2.0
        strat.bar_hours = 1
        strat.position_pct = 1.0
        strat.leverage = 2.0
        strat.fixed_size = 0.01
        strat.up_track = 0.0
        strat.mid_track = 0.0
        strat.down_track = 0.0
        strat.last_signal = ""
        strat._closes = [100.0, 101.0, 102.0, 103.0, 104.0]
        strat._shadow_pos = 0.0
        strat._last_bar = None
        strat.pos = 0
        strat.trading = False
        strat.cta_engine = None
        strat.vt_symbol = "BTCUSDT_SWAP_BINANCE.GLOBAL"
        strat.strategy_name = "test"
        strat.write_log = lambda _m: None
        strat.put_event = lambda: None
        return strat

    def test_from_aberration_config(self):
        cfg = AberrationVnpyConfig(n_period=35, k_up=2.0, k_down=2.0, bar_hours=1)
        d = AberrationVnpyStrategy.from_aberration_config(cfg)
        self.assertEqual(d["n_period"], 35)
        self.assertEqual(d["bar_hours"], 1)

    def test_warmup_does_not_trade(self):
        from vnpy.trader.constant import Exchange, Interval
        from vnpy.trader.object import BarData

        s = self._strategy()
        s.trading = False
        bar = BarData(
            symbol="BTCUSDT",
            exchange=Exchange.GLOBAL,
            datetime=datetime(2026, 1, 1, tzinfo=timezone.utc),
            interval=Interval.HOUR,
            volume=1.0,
            turnover=0.0,
            open_interest=0.0,
            open_price=200.0,
            high_price=200.0,
            low_price=200.0,
            close_price=200.0,
            gateway_name="BINANCE",
        )
        s.on_signal_bar(bar)
        self.assertEqual(s._shadow_pos, 0.0)
        self.assertEqual(s.last_signal, "")

    def test_lane_registration_with_env(self):
        env = {
            "ICT_VNPY_ENABLED": "0",
            "ORB_VNPY_ENABLED": "0",
            "ABERRATION_VNPY_ENABLED": "1",
            "ABERRATION_VNPY_SYMBOLS": "BTCUSDT,SOLUSDT",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            lanes = get_enabled_vnpy_lanes()
        names = [n for n, _ in lanes]
        self.assertIn("aberration", names)
        self.assertEqual(len(lanes), 1)

    def test_symbol_overlap_detected(self):
        from orb.ict.config import IctVnpyConfig
        from orb.trading_orb.config import OrbVnpyConfig

        lanes = [
            ("ict_2022", IctVnpyConfig(enabled=True, symbols=["ETHUSDT"])),
            ("aberration", AberrationVnpyConfig(enabled=True, symbols=["ETHUSDT", "BTCUSDT"])),
        ]
        overlaps = find_symbol_pool_overlaps(lanes)
        self.assertIn("ETHUSDT", overlaps)


if __name__ == "__main__":
    unittest.main()
