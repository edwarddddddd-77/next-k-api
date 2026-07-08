"""ICT 2022 vnpy strategy smoke tests."""

from __future__ import annotations

import unittest

from orb.ict.config import IctVnpyConfig
from orb.ict.vnpy.strategies.trading_ict_2022_vnpy import TradingIct2022VnpyStrategy


class TestTradingIct2022VnpyStrategy(unittest.TestCase):
    def _strategy(self) -> TradingIct2022VnpyStrategy:
        strat = TradingIct2022VnpyStrategy.__new__(TradingIct2022VnpyStrategy)
        strat.position_pct = 1.0
        strat.leverage = 2.0
        strat.rr = 1.5
        strat.hmm_filter = True
        strat.hmm_stick = 0.97
        strat.hmm_confirm = 3
        strat.fixed_size = 0.01
        strat.hmm_regime = -1
        strat.active_setups = 0
        strat.entry_price = 0.0
        strat.stop_price = 0.0
        strat.target_price = 0.0
        strat._bars = []
        strat._setups = []
        strat._all_fvgs = []
        strat._hmm_df = None
        strat._pending_limit_px = 0.0
        strat._pending_sl = 0.0
        strat._pending_tp = 0.0
        strat._pending_side = ""
        strat._pending_born_ms = 0
        strat._entry_pending = False
        strat._exit_pending = False
        strat._last_bar = None
        strat._shadow_pos = 0.0
        strat._used_fvg_keys = set()
        strat.pos = 0
        strat.trading = False
        strat.cta_engine = None
        strat.vt_symbol = "ETHUSDT.BINANCE"
        strat.strategy_name = "test"
        strat.write_log = lambda _m: None
        return strat

    def test_tp_from_rr(self):
        s = self._strategy()
        self.assertGreater(s._tp_from_rr(100.0, 95.0, "long"), 100.0)
        self.assertLess(s._tp_from_rr(100.0, 105.0, "short"), 100.0)

    def test_hmm_allows_range_only(self):
        s = self._strategy()
        s.hmm_filter = True
        s.hmm_regime = 1
        self.assertTrue(s._hmm_allows_entry())
        s.hmm_regime = 0
        self.assertFalse(s._hmm_allows_entry())
        s.hmm_filter = False
        s.hmm_regime = 0
        self.assertTrue(s._hmm_allows_entry())

    def test_from_ict_config(self):
        cfg = IctVnpyConfig(leverage=2.0, hmm_filter=True)
        d = TradingIct2022VnpyStrategy.from_ict_config(cfg)
        self.assertEqual(d["leverage"], 2.0)
        self.assertTrue(d["hmm_filter"])

    def test_on_5min_bar_skips_indicators_during_warmup(self):
        from datetime import datetime, timezone
        from unittest import mock

        from vnpy.trader.constant import Exchange, Interval
        from vnpy.trader.object import BarData

        s = self._strategy()
        s.trading = False
        bar = BarData(
            symbol="ETHUSDT",
            exchange=Exchange.GLOBAL,
            datetime=datetime(2026, 7, 8, 12, 0, tzinfo=timezone.utc),
            interval=Interval.MINUTE,
            volume=1.0,
            open_price=100.0,
            high_price=101.0,
            low_price=99.0,
            close_price=100.5,
            gateway_name="BINANCE_LINEAR",
        )
        with mock.patch.object(s, "_refresh_indicators") as refresh_mock:
            s.on_5min_bar(bar)
        refresh_mock.assert_not_called()
        self.assertEqual(len(s._bars), 1)

    def test_on_5min_bar_refreshes_when_trading(self):
        from datetime import datetime, timezone
        from unittest import mock

        from vnpy.trader.constant import Exchange, Interval
        from vnpy.trader.object import BarData

        s = self._strategy()
        s.trading = True
        bar = BarData(
            symbol="ETHUSDT",
            exchange=Exchange.GLOBAL,
            datetime=datetime(2026, 7, 8, 12, 0, tzinfo=timezone.utc),
            interval=Interval.MINUTE,
            volume=1.0,
            open_price=100.0,
            high_price=101.0,
            low_price=99.0,
            close_price=100.5,
            gateway_name="BINANCE_LINEAR",
        )
        with mock.patch.object(s, "_refresh_indicators") as refresh_mock:
            with mock.patch.object(s, "_check_pending_limit", return_value=False):
                with mock.patch.object(s, "_effective_pos", return_value=0):
                    with mock.patch.object(s, "_process_setups"):
                        with mock.patch.object(s, "put_event"):
                            s.on_5min_bar(bar)
        refresh_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
