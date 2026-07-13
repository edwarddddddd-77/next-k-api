"""IBS 日线拉取与合并测试。"""

from __future__ import annotations

import unittest
from unittest import mock

import pandas as pd

from quant.common.config import OrbConfig
from quant.ibs.core import SessionDailyBar
from quant.ibs.session_daily import _merge_session_today, fetch_daily_bars


class TestIbsSessionDaily(unittest.TestCase):
    def test_merge_session_today_replaces_last(self):
        history = [
            SessionDailyBar("2026-01-01", 0, 100.0, 110.0, 99.0, 105.0),
            SessionDailyBar("2026-01-02", 0, 105.0, 108.0, 104.0, 106.0),
        ]
        today = [SessionDailyBar("2026-01-02", 0, 105.0, 112.0, 103.0, 111.0)]
        merged = _merge_session_today(history, today)
        self.assertEqual(len(merged), 2)
        self.assertAlmostEqual(merged[-1].close, 111.0)
        self.assertAlmostEqual(merged[-1].high, 112.0)

    def test_fetch_daily_bars_session_5m_merges_history(self):
        history = [
            SessionDailyBar("2026-01-01", 0, 100.0, 110.0, 99.0, 105.0),
        ]
        intraday = pd.DataFrame(
            [
                {
                    "open_time": 1_768_000_000_000,
                    "open": 105.0,
                    "high": 112.0,
                    "low": 103.0,
                    "close": 111.0,
                    "volume": 1.0,
                }
            ]
        )
        sess = OrbConfig.from_env()
        with mock.patch(
            "quant.ibs.session_daily.fetch_session_daily_bars",
            return_value=history,
        ), mock.patch(
            "quant.ibs.session_daily.aggregate_session_daily",
            return_value=[SessionDailyBar("2026-01-02", 0, 105.0, 112.0, 103.0, 111.0)],
        ):
            bars = fetch_daily_bars(
                "SPYUSDT",
                days=30,
                exchange_id="bitget",
                sess=sess,
                source="session_5m",
                intraday_df=intraday,
            )
        self.assertEqual(len(bars), 2)
        self.assertAlmostEqual(bars[-1].close, 111.0)

    def test_fetch_daily_bars_bitget_1d(self):
        sess = OrbConfig.from_env()
        bars = fetch_daily_bars(
            "SPYUSDT",
            days=60,
            exchange_id="bitget",
            sess=sess,
            source="exchange_1d",
        )
        self.assertGreaterEqual(len(bars), 20)
        days = [b.session_day for b in bars]
        self.assertEqual(days, sorted(days))


if __name__ == "__main__":
    unittest.main()
