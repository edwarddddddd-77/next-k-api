"""Trading ORB vnpy 策略测试。"""

from __future__ import annotations

import os
import unittest
from datetime import datetime, timezone
from unittest import mock

import pandas as pd

from orb.trading_orb.config import OrbVnpyConfig
from orb.trading_orb.vnpy.strategies.trading_orb_vnpy import TradingOrbVnpyStrategy


class _Bar:
    def __init__(self, dt: datetime, *, close: float = 100.0, high: float = 101.0, low: float = 99.0, vol: float = 1000.0):
        self.datetime = dt
        self.close_price = close
        self.high_price = high
        self.low_price = low
        self.volume = vol
        self.open_price = close


class TestTradingOrbVnpyStrategy(unittest.TestCase):
    def _strategy(self) -> TradingOrbVnpyStrategy:
        strat = TradingOrbVnpyStrategy.__new__(TradingOrbVnpyStrategy)
        strat.orb_rth_only = True
        strat.orb_eod_flat = True
        strat.orb_exit_hour = 15
        strat.orb_exit_minute = 50
        strat.entry_start_hour = 10
        strat.entry_start_minute = 0
        strat.entry_end_hour = 11
        strat.entry_end_minute = 30
        strat.or_minutes = 20
        strat.vol_thresh = 1.2
        strat.pos = 0
        strat.vt_symbol = "INTCUSDT_SWAP_BINANCE.GLOBAL"
        strat.session_date = ""
        strat.or_high = 0.0
        strat.or_low = 0.0
        strat.or_range = 0.0
        strat.traded_today = False
        strat._entry_pending = False
        strat._exit_pending = False
        strat._restore_entry_px = 0.0
        strat._vol_baselines = {"10:00": 500.0}
        strat.or_range_at_entry = 0.0
        strat.stop_price = 0.0
        strat.target_price = 0.0
        strat.entry_price = 0.0
        strat.stop_or_mult = 0.5
        strat.target_or_mult = 0.75
        strat.write_log = mock.MagicMock()
        return strat

    def test_from_orb_config_maps_trading_orb_params(self):
        with mock.patch.dict(
            os.environ,
            {
                "ORB_VNPY_ENABLED": "1",
                "ORB_VNPY_OR_MINUTES": "20",
                "ORB_VNPY_VOL_THRESH": "1.2",
                "ORB_VNPY_STOP_OR_MULT": "0.5",
                "ORB_VNPY_TARGET_OR_MULT": "0.75",
                "ORB_VNPY_ENTRY_START": "10:00",
                "ORB_VNPY_ENTRY_END": "11:30",
                "ORB_VNPY_EXIT_MINUTE": "50",
            },
            clear=False,
        ):
            cfg = OrbVnpyConfig.from_env()
            d = TradingOrbVnpyStrategy.from_orb_config(cfg)
        self.assertEqual(d["or_minutes"], 20)
        self.assertAlmostEqual(d["vol_thresh"], 1.2)
        self.assertEqual(d["entry_start_hour"], 10)
        self.assertEqual(d["entry_end_minute"], 30)
        self.assertEqual(d["orb_exit_minute"], 50)

    def test_entry_window_1030_et(self):
        strat = self._strategy()
        ts = strat._bar_session_ts(_Bar(datetime(2026, 6, 2, 14, 30, tzinfo=timezone.utc)))
        with mock.patch.object(strat, "_session_cfg") as cfg_mock:
            cfg_mock.return_value.session_tz = "America/New_York"
            self.assertTrue(strat._in_entry_window(ts))

    def test_entry_window_rejects_0950_et(self):
        strat = self._strategy()
        ts = strat._bar_session_ts(_Bar(datetime(2026, 6, 2, 13, 50, tzinfo=timezone.utc)))
        with mock.patch.object(strat, "_session_cfg") as cfg_mock:
            cfg_mock.return_value.session_tz = "America/New_York"
            self.assertFalse(strat._in_entry_window(ts))

    def test_eod_bar_1550_et(self):
        strat = self._strategy()
        strat.pos = 1
        bar = _Bar(datetime(2026, 6, 2, 19, 50, tzinfo=timezone.utc))
        with mock.patch.object(strat, "_session_cfg") as cfg_mock:
            cfg_mock.return_value.session_tz = "America/New_York"
            cfg_mock.return_value.session_open_time = "09:30"
            cfg_mock.return_value.session_close_time = "16:00"
            cfg_mock.return_value.market = "us_equity"
            self.assertTrue(strat._is_eod_bar(bar))

    def test_vnpy_or_minutes_isolated_from_legacy_orb_env(self):
        with mock.patch.dict(
            os.environ,
            {
                "ORB_VNPY_ENABLED": "1",
                "ORB_VNPY_OR_MINUTES": "20",
                "ORB_OR_MINUTES": "15",
            },
            clear=False,
        ):
            cfg = OrbVnpyConfig.from_env()
        self.assertEqual(cfg.or_minutes, 20)

    def test_opening_range_excludes_bar_at_or_end(self):
        strat = self._strategy()
        strat.or_minutes = 20
        # 9:50 ET = 13:50 UTC — 20m OR 结束，不应纳入区间
        bar = _Bar(datetime(2026, 6, 2, 13, 50, tzinfo=timezone.utc), high=105.0, low=95.0)
        with mock.patch.object(strat, "_session_cfg") as cfg_mock:
            cfg_mock.return_value.session_tz = "America/New_York"
            cfg_mock.return_value.session_open_time = "09:30"
            strat.or_high = 102.0
            strat.or_low = 98.0
            strat._update_opening_range(bar)
        self.assertEqual(strat.or_high, 102.0)
        self.assertEqual(strat.or_low, 98.0)
        self.assertAlmostEqual(strat.or_range, 4.0)

    def test_try_entry_does_not_mark_traded_before_fill(self):
        strat = self._strategy()
        strat.or_high = 100.0
        strat.or_low = 99.0
        strat.or_range = 1.0
        strat.session_date = "2026-06-02"
        strat._vol_baselines = {"10:30": 500.0}
        bar = _Bar(datetime(2026, 6, 2, 14, 30, tzinfo=timezone.utc), close=101.0, vol=1000.0)
        strat._send_market = mock.MagicMock(return_value=["oid1"])
        strat.write_log = mock.MagicMock()
        with mock.patch.object(strat, "_orb_cfg") as cfg_mock:
            cfg_mock.return_value.one_trade_per_session = True
            cfg_mock.return_value.macro_filter = False
            cfg_mock.return_value.compound = False
            cfg_mock.return_value.equity_usdt = 100.0
            cfg_mock.return_value.shadow = False
            cfg_mock.return_value.live_enabled = True
            with mock.patch.object(strat, "_bar_session_ts") as ts_mock:
                ts_mock.return_value = pd.Timestamp("2026-06-02 10:30", tz="America/New_York")
                with mock.patch.object(strat, "_in_entry_window", return_value=True):
                    with mock.patch(
                        "orb.trading_orb.vnpy.strategies.trading_orb_vnpy.fixed_size_for_orb",
                        return_value=1.0,
                    ):
                        strat._try_entry(bar)
        self.assertFalse(strat.traded_today)
        self.assertTrue(strat._entry_pending)
        strat._send_market.assert_called_once()

    def test_entry_pending_blocks_second_signal(self):
        strat = self._strategy()
        strat.or_high = 100.0
        strat.or_low = 99.0
        strat.or_range = 1.0
        strat.session_date = "2026-06-02"
        strat._entry_pending = True
        bar = _Bar(datetime(2026, 6, 2, 14, 35, tzinfo=timezone.utc), close=101.0, vol=1000.0)
        strat._open_market = mock.MagicMock()
        with mock.patch.object(strat, "_orb_cfg") as cfg_mock:
            cfg_mock.return_value.macro_filter = False
            with mock.patch.object(strat, "_in_entry_window", return_value=True):
                with mock.patch.object(strat, "_bar_session_ts") as ts_mock:
                    ts_mock.return_value = pd.Timestamp("2026-06-02 10:35", tz="America/New_York")
                    strat._try_entry(bar)
        strat._open_market.assert_not_called()

    def test_shadow_signal_marks_traded_without_order(self):
        strat = self._strategy()
        strat.trading = True
        with mock.patch.object(strat, "_send_market", return_value=[]):
            with mock.patch.object(strat, "_orb_cfg") as cfg_mock:
                cfg_mock.return_value.shadow = True
                cfg_mock.return_value.live_enabled = False
                cfg_mock.return_value.one_trade_per_session = True
                strat.write_log = mock.MagicMock()
                strat._open_market(1, 1.0)
        self.assertTrue(strat.traded_today)
        self.assertFalse(strat._entry_pending)

    def test_opening_range_uses_1min_bars(self):
        strat = self._strategy()
        strat.trading = True
        strat.or_minutes = 20
        strat.session_date = "2026-06-02"
        bar = _Bar(datetime(2026, 6, 2, 13, 35, tzinfo=timezone.utc), high=101.0, low=99.5)
        with mock.patch.object(strat, "_session_cfg") as cfg_mock:
            cfg_mock.return_value.session_tz = "America/New_York"
            cfg_mock.return_value.session_open_time = "09:30"
            strat.bg = mock.MagicMock()
            strat._on_1min_bar(bar)
        self.assertEqual(strat.or_high, 101.0)
        self.assertEqual(strat.or_low, 99.5)
        strat.bg.update_bar.assert_called_once()

    def test_exit_pending_blocks_duplicate_close(self):
        strat = self._strategy()
        strat.pos = 1
        strat.stop_price = 99.0
        strat.target_price = 102.0
        strat._exit_pending = True
        bar = _Bar(datetime(2026, 6, 2, 14, 35, tzinfo=timezone.utc), low=98.5)
        strat._send_market = mock.MagicMock()
        self.assertFalse(strat._check_exit_on_bar(bar))
        strat._send_market.assert_not_called()

    def test_on_trade_open_recomputes_levels_from_fill(self):
        from vnpy.trader.constant import Offset

        strat = self._strategy()
        strat.pos = 1
        strat.or_range = 2.0
        strat.or_range_at_entry = 2.0
        strat.inited = False
        strat.put_event = mock.MagicMock()
        strat.cancel_all = mock.MagicMock()
        trade = mock.MagicMock()
        trade.offset = Offset.OPEN
        trade.price = 102.0
        strat.on_trade(trade)
        self.assertTrue(strat.traded_today)
        self.assertFalse(strat._entry_pending)
        self.assertAlmostEqual(strat.entry_price, 102.0)
        self.assertAlmostEqual(strat.stop_price, 101.0)
        self.assertAlmostEqual(strat.target_price, 103.5)

    def test_restore_synced_position_after_or_ready(self):
        strat = self._strategy()
        strat.pos = 1
        strat.or_range = 2.0
        with mock.patch.object(strat, "_orb_cfg") as cfg_mock:
            cfg_mock.return_value.one_trade_per_session = True
            strat.restore_synced_position(entry_px=100.0, pos=1.0)
        self.assertTrue(strat.traded_today)
        self.assertAlmostEqual(strat.stop_price, 99.0)
        self.assertAlmostEqual(strat.target_price, 101.5)

    def test_deferred_restore_waits_for_or_range(self):
        strat = self._strategy()
        strat.pos = 1
        strat._restore_entry_px = 100.0
        strat.or_range = 0.0
        bar = _Bar(datetime(2026, 6, 2, 13, 55, tzinfo=timezone.utc), high=101.0, low=99.0)
        with mock.patch.object(strat, "_session_cfg") as cfg_mock:
            cfg_mock.return_value.session_tz = "America/New_York"
            cfg_mock.return_value.session_open_time = "09:30"
            strat.session_date = "2026-06-02"
            strat.or_high = 100.0
            strat.or_low = 98.0
            strat._update_opening_range(bar)
        strat.or_range = strat.or_high - strat.or_low
        strat._try_deferred_restore()
        self.assertAlmostEqual(strat.entry_price, 100.0)
        self.assertGreater(strat.stop_price, 0)

    def test_on_trade_open_sets_traded_today(self):
        from vnpy.trader.constant import Offset

        strat = self._strategy()
        strat.pos = 1
        strat.or_range = 1.0
        strat.or_range_at_entry = 1.0
        strat.inited = False
        strat.put_event = mock.MagicMock()
        strat.cancel_all = mock.MagicMock()
        trade = mock.MagicMock()
        trade.offset = Offset.OPEN
        trade.price = 100.5
        strat.on_trade(trade)
        self.assertTrue(strat.traded_today)
        strat.cancel_all.assert_not_called()

    def test_exit_stop_triggers_market_close(self):
        strat = self._strategy()
        strat.pos = 1
        strat.stop_price = 99.0
        strat.target_price = 102.0
        strat.cancel_all = mock.MagicMock()
        strat.write_log = mock.MagicMock()
        strat._send_market = mock.MagicMock(return_value=["oid1"])
        bar = _Bar(datetime(2026, 6, 2, 14, 35, tzinfo=timezone.utc), low=98.5, high=100.0)
        self.assertTrue(strat._check_exit_on_bar(bar))
        strat._send_market.assert_called_once()

    def test_eod_flatten_uses_market(self):
        from vnpy.trader.constant import Direction, Offset

        strat = self._strategy()
        strat.pos = 2
        strat.cancel_all = mock.MagicMock()
        strat.write_log = mock.MagicMock()
        strat._send_market = mock.MagicMock(return_value=["oid1"])
        bar = _Bar(datetime(2026, 6, 2, 19, 50, tzinfo=timezone.utc))
        strat._flatten_at_bar(bar)
        strat._send_market.assert_called_once_with(Direction.SHORT, Offset.CLOSE, 2)


if __name__ == "__main__":
    unittest.main()
