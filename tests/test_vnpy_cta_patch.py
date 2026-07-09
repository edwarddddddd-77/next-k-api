"""CtaEngine RTH tick 补丁测试。"""

from __future__ import annotations

import unittest
from unittest import mock

from quant.engine import lane


class _FakeStrategy:
    def __init__(self, pos: int = 0):
        self.pos = pos


class _FakeEngine:
    def __init__(self, strategies: dict):
        self.strategies = strategies


class TestCtaRthPatch(unittest.TestCase):
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
