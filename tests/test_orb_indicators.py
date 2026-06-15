"""ORB 指标单元测试。"""

from __future__ import annotations

import unittest

import pandas as pd

from orb.core.indicators import compute_atr_series, daily_atr_asof


class TestOrbIndicators(unittest.TestCase):
    def test_daily_atr_uses_prior_day(self):
        tz = "America/New_York"
        rows = []
        for i in range(20):
            day = pd.Timestamp("2024-03-01", tz=tz) + pd.Timedelta(days=i)
            rows.append(
                {
                    "open_time": int(day.value // 1_000_000),
                    "open": 100.0 + i * 0.1,
                    "high": 101.0 + i * 0.1,
                    "low": 99.0 + i * 0.1,
                    "close": 100.5 + i * 0.1,
                    "volume": 1e6,
                }
            )
        df = pd.DataFrame(rows)
        asof = int(pd.Timestamp("2024-03-16 10:00", tz=tz).value // 1_000_000)
        atr = daily_atr_asof(df, asof, period=14, tz=tz)
        self.assertIsNotNone(atr)
        assert atr is not None
        self.assertGreater(atr, 0)
        series = compute_atr_series(df, period=14)
        asof_day = pd.Timestamp(asof, unit="ms", tz=tz).normalize()
        day_ts = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(tz).dt.normalize()
        completed = df[day_ts < asof_day]
        expected = float(compute_atr_series(completed, period=14).iloc[-1])
        self.assertAlmostEqual(atr, expected)


if __name__ == "__main__":
    unittest.main()
