"""分档移动止盈逻辑单元测试。"""

from __future__ import annotations

import unittest

from momentum_trail import (
    TIER_LOW,
    TIER_TIER1,
    TIER_TIER2,
    TrailConfig,
    evaluate_trail,
)


def _cfg(**kw) -> TrailConfig:
    base = dict(
        enabled=True,
        stop_loss_pct=2.0,
        low_trail_stop_loss_pct=0.3,
        trail_stop_loss_pct=0.2,
        higher_trail_stop_loss_pct=0.25,
        low_trail_profit_threshold=0.4,
        first_trail_profit_threshold=1.0,
        second_trail_profit_threshold=3.0,
    )
    base.update(kw)
    return TrailConfig(**base)


class TestMomentumTrail(unittest.TestCase):
    def test_tier1_trail_exit(self):
        ev = evaluate_trail(
            side="LONG",
            entry=100.0,
            mark=101.5,
            peak_profit_pct=2.0,
            cfg=_cfg(),
        )
        self.assertEqual(ev.trail_tier, TIER_TIER1)
        self.assertEqual(ev.exit_rule, "trail_tier1")

    def test_tier2_floor(self):
        ev = evaluate_trail(
            side="SHORT",
            entry=100.0,
            mark=97.0,
            peak_profit_pct=4.0,
            cfg=_cfg(),
        )
        self.assertEqual(ev.trail_tier, TIER_TIER2)
        self.assertAlmostEqual(ev.profit_pct, 3.0)
        self.assertEqual(ev.exit_rule, "trail_tier2")

    def test_low_tier_lock(self):
        ev = evaluate_trail(
            side="LONG",
            entry=100.0,
            mark=100.29,
            peak_profit_pct=0.5,
            cfg=_cfg(),
        )
        self.assertEqual(ev.trail_tier, TIER_LOW)
        self.assertEqual(ev.exit_rule, "trail_low")

    def test_hard_stop(self):
        ev = evaluate_trail(
            side="LONG",
            entry=100.0,
            mark=97.5,
            peak_profit_pct=0.0,
            cfg=_cfg(),
        )
        self.assertEqual(ev.exit_rule, "trail_stop")

    def test_hold_below_thresholds(self):
        ev = evaluate_trail(
            side="LONG",
            entry=100.0,
            mark=100.2,
            peak_profit_pct=0.1,
            cfg=_cfg(),
        )
        self.assertIsNone(ev.exit_rule)


if __name__ == "__main__":
    unittest.main()
