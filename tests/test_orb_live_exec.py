"""ORB live_exec 载荷构建测试。"""

from __future__ import annotations

import unittest
from dataclasses import replace

from orb.core.config import OrbConfig
from orb.core.live_exec import build_close_payload, build_open_payload
from orb.core.signals import OrbSignal


class TestOrbLiveExec(unittest.TestCase):
    def test_open_payload_uses_default_leverage(self):
        cfg = replace(OrbConfig(), live_enabled=True)
        sig = OrbSignal(
            symbol="COINUSDT",
            price=160.0,
            side="LONG",
            play="ORB_BREAKOUT_LONG",
            confidence="high",
            reasons=[],
            sl_price=159.0,
            tp_price=None,
            session_date="2026-06-09",
            entry_bar_open_ms=1_700_000_000_000,
            paper_notional_usdt=2500.0,
        )
        p = build_open_payload(sig, cfg)
        self.assertEqual(p["leverage"], 10.0)
        self.assertAlmostEqual(p["margin_usdt"], 250.0)
        self.assertAlmostEqual(p["margin_usdt"] * p["leverage"], 2500.0)

    def test_open_payload_margin_from_notional(self):
        cfg = replace(OrbConfig(), live_enabled=True, leverage=5.0, margin_usdt=100.0)
        sig = OrbSignal(
            symbol="QQQUSDT",
            price=400.0,
            side="LONG",
            play="ORB_BREAKOUT_LONG",
            confidence="high",
            reasons=[],
            sl_price=395.0,
            tp_price=None,
            session_date="2026-06-09",
            entry_bar_open_ms=1_700_000_000_000,
            paper_notional_usdt=500.0,
        )
        p = build_open_payload(sig, cfg)
        self.assertEqual(p["source"], "orb")
        self.assertEqual(p["symbol"], "QQQUSDT")
        self.assertEqual(p["action"], "open")
        self.assertAlmostEqual(p["margin_usdt"], 100.0)
        self.assertEqual(p["leverage"], 5.0)
        self.assertIsNone(p["tp_price"])

    def test_close_payload_session_close_uses_market(self):
        p = build_close_payload("COINUSDT", "SHORT", close_price=155.0, tag="session_close")
        self.assertEqual(p["action"], "close")
        self.assertEqual(p["side"], "SHORT")
        self.assertNotIn("close_price", p)
        self.assertIn(":session_close:", str(p["api_signal_id"]))

    def test_close_payload_loss_keeps_limit_price(self):
        p = build_close_payload("COINUSDT", "SHORT", close_price=155.0, tag="loss")
        self.assertAlmostEqual(p["close_price"], 155.0)


if __name__ == "__main__":
    unittest.main()
