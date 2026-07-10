"""CtaEngine RTH tick 补丁测试。"""

from __future__ import annotations

import unittest
from unittest import mock

from quant.engine import lane


class _FakeStrategy:
    def __init__(self, pos: int = 0, *, orb_rth_only: bool = False):
        self.pos = pos
        self.orb_rth_only = orb_rth_only


class _FakeEngine:
    def __init__(self, strategies: dict):
        self.strategies = strategies


class TestCtaRthPatch(unittest.TestCase):
    def test_crypto_strategy_accepts_tick_outside_rth(self):
        from quant.engine.cta_rth_patch import _strategy_accepts_tick

        strat = _FakeStrategy(orb_rth_only=False)
        tick = mock.Mock()
        with mock.patch("quant.engine.cta_rth_patch.lane_rth_only", return_value=True):
            with mock.patch("quant.engine.cta_rth_patch.tick_in_lane_rth", return_value=False):
                with mock.patch("quant.engine.cta_rth_patch.lane_vnpy_idle_outside_rth", return_value=True):
                    self.assertTrue(_strategy_accepts_tick(strat, tick, eod_flat_active=False))

    def test_orb_strategy_blocks_tick_outside_rth_when_idle(self):
        from quant.engine.cta_rth_patch import _strategy_accepts_tick

        strat = _FakeStrategy(orb_rth_only=True)
        tick = mock.Mock()
        with mock.patch("quant.engine.cta_rth_patch.lane_rth_only", return_value=True):
            with mock.patch("quant.engine.cta_rth_patch.tick_in_lane_rth", return_value=False):
                with mock.patch("quant.engine.cta_rth_patch.lane_vnpy_idle_outside_rth", return_value=True):
                    self.assertFalse(_strategy_accepts_tick(strat, tick, eod_flat_active=False))

    def test_orb_strategy_allows_tick_with_open_position_and_eod_flat(self):
        from quant.engine.cta_rth_patch import _strategy_accepts_tick

        strat = _FakeStrategy(pos=1, orb_rth_only=True)
        tick = mock.Mock()
        with mock.patch("quant.engine.cta_rth_patch.lane_rth_only", return_value=True):
            with mock.patch("quant.engine.cta_rth_patch.tick_in_lane_rth", return_value=False):
                with mock.patch("quant.engine.cta_rth_patch.lane_vnpy_idle_outside_rth", return_value=True):
                    self.assertTrue(_strategy_accepts_tick(strat, tick, eod_flat_active=True))

    def test_eod_flat_allows_tick_when_open_position(self):
        cfg = mock.Mock()
        cfg.eod_flat = True
        cfg.enabled = True
        engine = _FakeEngine({"s1": _FakeStrategy(pos=1)})
        with mock.patch("quant.engine.lane.get_enabled_vnpy_lanes", return_value=[("trading_orb", cfg)]):
            self.assertTrue(lane.lane_eod_flat_and_enabled(engine))

    def test_eod_flat_blocks_tick_when_flat(self):
        cfg = mock.Mock()
        cfg.eod_flat = True
        cfg.enabled = True
        engine = _FakeEngine({"s1": _FakeStrategy(pos=0)})
        with mock.patch("quant.engine.lane.get_enabled_vnpy_lanes", return_value=[("trading_orb", cfg)]):
            self.assertFalse(lane.lane_eod_flat_and_enabled(engine))


if __name__ == "__main__":
    unittest.main()
