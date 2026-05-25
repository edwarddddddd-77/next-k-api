"""动量开仓过滤单元测试。"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import momentum_config as cfg
from momentum_filters import check_open_allowed


class TestMomentumFilters(unittest.TestCase):
    NOW_MS = 1_779_671_304_452

    def setUp(self):
        self._enabled = cfg.MOM_FILTER_ENABLED
        cfg.MOM_FILTER_ENABLED = True

    def tearDown(self):
        cfg.MOM_FILTER_ENABLED = self._enabled

    def _fresh_event(self, price_change: str) -> dict:
        return {
            "createTimestamp": self.NOW_MS - 10 * 60 * 1000,
            "priceChange": price_change,
        }

    def test_same_symbol_both_legs(self):
        ok, reason = check_open_allowed(
            side="LONG",
            symbol="BTCUSDT",
            event_raw={"createTimestamp": 1_700_000_000_000, "priceChange": "0.05"},
            peer_symbol="BTCUSDT",
        )
        self.assertFalse(ok)
        self.assertIn("same_symbol", reason)

    def test_stale_event(self):
        ok, reason = check_open_allowed(
            side="LONG",
            symbol="ETHUSDT",
            event_raw={
                "createTimestamp": self.NOW_MS - 120 * 60 * 1000,
                "priceChange": "0.05",
            },
            now_ms=self.NOW_MS,
        )
        self.assertFalse(ok)
        self.assertIn("stale", reason)

    def test_pullback_too_small(self):
        ok, reason = check_open_allowed(
            side="LONG",
            symbol="ETHUSDT",
            event_raw=self._fresh_event("0.01"),
            now_ms=self.NOW_MS,
        )
        self.assertFalse(ok)
        self.assertIn("pullback_small", reason)

    @patch("vp_regime_scanner.analyze_symbol_vp")
    def test_vp_scheme_rejects_long(self, mock_vp):
        mock_vp.return_value = MagicMock(scheme="REVERSAL_WATCH")
        ok, reason = check_open_allowed(
            side="LONG",
            symbol="PLAYUSDT",
            event_raw=self._fresh_event("0.05"),
            now_ms=self.NOW_MS,
        )
        self.assertFalse(ok)
        self.assertIn("filter:vp:scheme", reason)

    @patch("vp_regime_scanner.analyze_symbol_vp")
    def test_pass_long(self, mock_vp):
        mock_vp.return_value = MagicMock(scheme="MOMENTUM")
        ok, reason = check_open_allowed(
            side="LONG",
            symbol="PLAYUSDT",
            event_raw=self._fresh_event("0.05"),
            now_ms=self.NOW_MS,
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "")


if __name__ == "__main__":
    unittest.main()
