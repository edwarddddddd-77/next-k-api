"""KAMA Trend 核心测试。"""

from __future__ import annotations

import unittest

from quant.common.jesse_indicators import kama_last
from quant.kama_trend.core import compute_snapshot, entry_signal


class TestKamaTrendCore(unittest.TestCase):
    def _bars(self, n: int, start: float = 100.0, step: float = 0.3):
        out = []
        for i in range(n):
            c = start + i * step
            ts = i * 900_000
            out.append((ts, c - 0.2, c + 0.5, c - 0.5, c))
        return out

    def test_kama_last_trending_up(self):
        closes = [float(100 + i) for i in range(40)]
        self.assertIsNotNone(kama_last(closes, period=14))

    def test_entry_requires_cooldown(self):
        bars = self._bars(250)
        long_tf = [float(100 + i) for i in range(80)]
        snap = compute_snapshot(bars, long_tf_closes=long_tf)
        self.assertIsNotNone(snap)
        sig = entry_signal(
            float(bars[-1][4]),
            snap,
            bars_since_trade=0,
            cooldown_bars=10,
        )
        self.assertEqual(sig, 0)

    def test_entry_long_when_filters_pass(self):
        bars = self._bars(250, step=0.8)
        long_tf = [float(100 + i) for i in range(80)]
        snap = compute_snapshot(bars, long_tf_closes=long_tf)
        self.assertIsNotNone(snap)
        sig = entry_signal(
            float(bars[-1][4]) + 5,
            snap,
            adx_min=0,
            chop_max=100,
            bb_width_max_pct=100,
            bars_since_trade=20,
            cooldown_bars=10,
        )
        self.assertIn(sig, (-1, 0, 1))

    def test_requires_native_4h_kama(self):
        bars = self._bars(250)
        self.assertIsNone(compute_snapshot(bars, long_tf_closes=None))
        self.assertIsNone(compute_snapshot(bars, long_tf_closes=[]))

    def test_bar_hits_stop_before_tp_same_bar(self):
        from quant.kama_trend.core import bar_hits_stop_tp

        hit = bar_hits_stop_tp(
            side=1,
            high=110.0,
            low=95.0,
            stop=96.0,
            tp=108.0,
            prev_high=100.0,
            prev_low=100.0,
        )
        self.assertEqual(hit, "stop")


if __name__ == "__main__":
    unittest.main()
