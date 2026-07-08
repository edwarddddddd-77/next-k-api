"""CtaEngine RTH tick 补丁测试。"""

from __future__ import annotations

import unittest
from unittest import mock

from orb.vnpy import cta_rth_patch as patch


class _FakeStrategy:
    def __init__(self, pos: int = 0):
        self.pos = pos


class _FakeEngine:
    def __init__(self, strategies: dict):
        self.strategies = strategies


class TestCtaRthPatch(unittest.TestCase):
    def test_allow_tick_outside_rth_when_eod_flat_and_open_position(self):
        cfg = mock.Mock()
        cfg.eod_flat = True
        cfg.enabled = True
        engine = _FakeEngine({"s1": _FakeStrategy(pos=1)})
        self.assertTrue(patch._allow_tick_outside_rth(engine, cfg))

    def test_block_tick_outside_rth_when_flat(self):
        cfg = mock.Mock()
        cfg.eod_flat = True
        cfg.enabled = True
        engine = _FakeEngine({"s1": _FakeStrategy(pos=0)})
        self.assertFalse(patch._allow_tick_outside_rth(engine, cfg))


if __name__ == "__main__":
    unittest.main()
