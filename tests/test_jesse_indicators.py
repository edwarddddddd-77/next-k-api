"""Jesse 指标对齐测试。"""

from __future__ import annotations

import unittest

from quant.common.jesse_indicators import (
    atr_last,
    bollinger_bands_width_pct,
    kama_last,
)


class TestJesseIndicators(unittest.TestCase):
    def _series(self, n: int, start: float = 100.0, step: float = 0.3):
        closes = [start + i * step for i in range(n)]
        highs = [c + 0.5 for c in closes]
        lows = [c - 0.5 for c in closes]
        return closes, highs, lows

    def test_kama_last(self):
        closes = [float(100 + i) for i in range(50)]
        self.assertIsNotNone(kama_last(closes, period=14))

    def test_atr_wilder(self):
        closes, highs, lows = self._series(40)
        self.assertIsNotNone(atr_last(highs, lows, closes, 14))

    def test_bb_width_pct(self):
        closes, _, _ = self._series(30)
        bbw = bollinger_bands_width_pct(closes, 20)
        self.assertIsNotNone(bbw)
        self.assertGreater(bbw, 0.0)


if __name__ == "__main__":
    unittest.main()
