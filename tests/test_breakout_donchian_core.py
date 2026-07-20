"""Breakout Donchian core/resonance tests."""

from __future__ import annotations

import unittest

from quant.breakout_donchian.config import BreakoutDonchianConfig
from quant.breakout_donchian.core import bar_exit_reason, resolve_exit_target_price
from quant.breakout_donchian.resonance import evaluate_resonance, weekly_trend_ok


class TestBreakoutDonchianOptimized(unittest.TestCase):
    def test_exit_target_tp1(self):
        px, tag = resolve_exit_target_price(tp1=102.0, tp2=105.0, tp3=110.0, exit_target="tp1")
        self.assertEqual(tag, "tp1")
        self.assertEqual(px, 102.0)

    def test_bar_exit_hits_tp1(self):
        reason = bar_exit_reason(
            side=1,
            high=103.0,
            low=100.0,
            stop=95.0,
            tp1=102.0,
            tp2=105.0,
            tp3=110.0,
            prev_high=101.0,
            prev_low=100.0,
            exit_target="tp1",
        )
        self.assertEqual(reason, "tp1")

    def test_weekly_trend_ok(self):
        bars = []
        for i in range(12):
            c = 100.0 + i
            bars.append((i * 604_800_000, c, c + 1, c - 1, c, 1000.0))
        cfg = BreakoutDonchianConfig(weekly_trend_ma_period=10)
        self.assertTrue(weekly_trend_ok(bars, cfg))

    def test_evaluate_resonance_trend_mode(self):
        bars = []
        for i in range(12):
            c = 100.0 + i
            bars.append((i * 604_800_000, c, c + 1, c - 1, c, 1000.0))
        cfg = BreakoutDonchianConfig(weekly_confirm_mode="trend", check_1h_bonus=False)
        res = evaluate_resonance(cfg, weekly_bars=bars, hourly_bars=None)
        self.assertTrue(res.weekly_ok)
        self.assertEqual(res.tier, "dual")


if __name__ == "__main__":
    unittest.main()
