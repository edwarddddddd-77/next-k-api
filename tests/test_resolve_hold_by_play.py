#!/usr/bin/env python3
"""resolve_max_hold_ms / resolve_max_bars 按 Play 族分流。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_API_ROOT = Path(__file__).resolve().parent.parent
if str(_API_ROOT) not in sys.path:
    sys.path.insert(0, str(_API_ROOT))

import zct_vwap_signal_scanner as z


class TestResolveHoldByPlay(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = (
            z.RESOLVE_MAX_HOLD_MS,
            z.RESOLVE_MAX_BARS,
            z.RESOLVE_MAX_HOLD_MS_PLAY01,
            z.RESOLVE_MAX_BARS_PLAY01,
            z.RESOLVE_MAX_HOLD_MS_PLAY02,
            z.RESOLVE_MAX_BARS_PLAY02,
            z.RESOLVE_MAX_HOLD_MS_PLAY03,
            z.RESOLVE_MAX_BARS_PLAY03,
        )

    def tearDown(self) -> None:
        (
            z.RESOLVE_MAX_HOLD_MS,
            z.RESOLVE_MAX_BARS,
            z.RESOLVE_MAX_HOLD_MS_PLAY01,
            z.RESOLVE_MAX_BARS_PLAY01,
            z.RESOLVE_MAX_HOLD_MS_PLAY02,
            z.RESOLVE_MAX_BARS_PLAY02,
            z.RESOLVE_MAX_HOLD_MS_PLAY03,
            z.RESOLVE_MAX_BARS_PLAY03,
        ) = self._saved

    def _set_defaults(self) -> None:
        z.RESOLVE_MAX_HOLD_MS = 4 * 3_600_000
        z.RESOLVE_MAX_BARS = 240
        z.RESOLVE_MAX_HOLD_MS_PLAY01 = 5 * 3_600_000
        z.RESOLVE_MAX_BARS_PLAY01 = 300
        z.RESOLVE_MAX_HOLD_MS_PLAY02 = 4 * 3_600_000
        z.RESOLVE_MAX_BARS_PLAY02 = 240
        z.RESOLVE_MAX_HOLD_MS_PLAY03 = 3 * 3_600_000
        z.RESOLVE_MAX_BARS_PLAY03 = 180

    def test_per_play_defaults(self) -> None:
        self._set_defaults()
        self.assertEqual(z.resolve_max_hold_ms("PLAY01_BREAKOUT_LONG"), 5 * 3_600_000)
        self.assertEqual(z.resolve_max_bars("PLAY01_BIAS_LONG"), 300)
        self.assertEqual(z.resolve_max_hold_ms("PLAY02_BREAKDOWN_SHORT"), 4 * 3_600_000)
        self.assertEqual(z.resolve_max_bars("PLAY02_BIAS_SHORT"), 240)
        self.assertEqual(z.resolve_max_hold_ms("PLAY03_REV_LONG"), 3 * 3_600_000)
        self.assertEqual(z.resolve_max_bars("PLAY03_REV_SHORT"), 180)

    def test_fallback_global_when_play_override_zero(self) -> None:
        self._set_defaults()
        z.RESOLVE_MAX_HOLD_MS_PLAY03 = 0
        z.RESOLVE_MAX_BARS_PLAY03 = 0
        self.assertEqual(z.resolve_max_hold_ms("PLAY03_REV_LONG"), 4 * 3_600_000)
        self.assertEqual(z.resolve_max_bars("PLAY03_REV_LONG"), 240)

    def test_transition_uses_global(self) -> None:
        self._set_defaults()
        self.assertEqual(z.resolve_max_hold_ms("TRANSITION_BIAS_LONG"), 4 * 3_600_000)


if __name__ == "__main__":
    unittest.main()
