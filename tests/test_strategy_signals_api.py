"""F-MR strategy signals API tests."""

from __future__ import annotations

import unittest
from unittest import mock

from quant.engine import strategy_signals as ss


class TestStrategySignals(unittest.TestCase):
    def test_valid_lanes_only_fmr(self):
        self.assertEqual(ss.VALID_LANES, {ss.LANE_AVAX_F_MR})

    def test_invalid_lane(self):
        out = ss.list_strategy_signals(lane="trading_orb", limit=10)
        self.assertFalse(out["ok"])
        self.assertEqual(out.get("error"), "invalid_lane")

    def test_record_open_is_noop(self):
        self.assertIsNone(
            ss.record_strategy_open_signal(
                lane=ss.LANE_AVAX_F_MR,
                symbol="AVAXUSDT",
                side="LONG",
                entry_price=6.5,
            )
        )

    def test_list_fmr_feed(self):
        fake = {
            "ok": True,
            "lane": "avax_f_mr",
            "count": 1,
            "signals": [
                {
                    "symbol": "AVAXUSDT",
                    "side": "SHORT",
                    "action": "open",
                    "status": "shadow",
                }
            ],
        }
        with mock.patch(
            "utils.avax_f_mr_indicator.strategy_signal_feed",
            return_value=fake,
        ):
            out = ss.list_strategy_signals(lane=ss.LANE_AVAX_F_MR, limit=10)
        self.assertTrue(out["ok"])
        self.assertEqual(out["count"], 1)
        self.assertEqual(out["signals"][0]["symbol"], "AVAXUSDT")


if __name__ == "__main__":
    unittest.main()
