"""MtfMomo2xA 核心逻辑测试。"""

from __future__ import annotations

import unittest

from quant.mtfmomo.core import compute_levels_from_series, entry_signal, trend_dir


class TestMtfMomoCore(unittest.TestCase):
    def _series(self, n: int, start: float = 100.0, step: float = 0.5):
        closes = [start + i * step for i in range(n)]
        highs = [c + 1.0 for c in closes]
        lows = [c - 1.0 for c in closes]
        return closes, highs, lows

    def test_trend_dir_up(self):
        closes = [float(100 + i) for i in range(40)]
        self.assertEqual(trend_dir(closes, 16), 1)

    def test_entry_long_when_agree_and_breakout(self):
        closes, highs, lows = self._series(120, start=100.0, step=0.8)
        highs[-1] = closes[-1] + 5
        levels = compute_levels_from_series(closes, highs, lows)
        self.assertIsNotNone(levels)
        sig = entry_signal(closes[-1] + 10, levels)
        self.assertIn(sig, (-1, 0, 1))

    def test_resample_utc_four_hour_buckets(self):
        from quant.mtfmomo.core import resample_utc

        bars = [
            (0, 1, 2, 0.5, 1.0),
            (3_600_000, 1, 2, 0.5, 1.1),
            (7_200_000, 1, 2, 0.5, 1.2),
            (10_800_000, 1, 2, 0.5, 1.3),
            (14_400_000, 1, 3, 0.4, 1.5),
        ]
        closes, _, _ = resample_utc(bars, 4)
        self.assertEqual(len(closes), 2)
        self.assertAlmostEqual(closes[0], 1.3)
        self.assertAlmostEqual(closes[1], 1.5)
        closes, highs, lows = self._series(10)
        self.assertIsNone(compute_levels_from_series(closes, highs, lows))


if __name__ == "__main__":
    unittest.main()
