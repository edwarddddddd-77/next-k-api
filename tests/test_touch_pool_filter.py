#!/usr/bin/env python3
"""触轨池主筛条件单元测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_API_ROOT = Path(__file__).resolve().parent.parent
if str(_API_ROOT) not in sys.path:
    sys.path.insert(0, str(_API_ROOT))

from zct_touch_pool_metrics import (
    net_pnl_after_friction,
    round_trip_friction_usdt,
    taker_bps_per_side,
    slippage_bps_per_side,
    trailing_consecutive_losses_at_end,
)
from zct_vwap_asset_pool import _filter_pool


class TouchPoolFilterTests(unittest.TestCase):
    def test_trailing_consecutive_losses(self) -> None:
        rows = [
            {"outcome": "win"},
            {"outcome": "loss"},
            {"outcome": "loss"},
            {"outcome": "loss"},
        ]
        self.assertEqual(trailing_consecutive_losses_at_end(rows), 3)

    def test_filter_requires_profit_factor(self) -> None:
        summary = {
            "per_symbol": {
                "BTCUSDT": {
                    "win": 8,
                    "loss": 2,
                    "n_trades": 25,
                    "expired": 0,
                    "unresolved": 0,
                    "win_rate_touch_sl_tp": 0.8,
                    "profit_factor_net": 1.1,
                    "consecutive_losses_at_end": 0,
                },
                "ETHUSDT": {
                    "win": 8,
                    "loss": 2,
                    "n_trades": 25,
                    "expired": 0,
                    "unresolved": 0,
                    "win_rate_touch_sl_tp": 0.8,
                    "profit_factor_net": 1.5,
                    "consecutive_losses_at_end": 1,
                },
            }
        }
        filt = _filter_pool(
            summary,
            min_touch_trades=1,
            strict_greater_touch=False,
            min_touch_win_rate=0.72,
            strict_greater_rate=False,
            min_total_trades=20,
            max_expired_ratio=1.0,
            min_win_loss_abs=0,
            min_touch_share=0.0,
            min_profit_factor=1.25,
            max_consecutive_losses_at_end=2,
        )
        syms = {m["symbol"] for m in filt["matched"]}
        self.assertNotIn("BTCUSDT", syms)
        self.assertIn("ETHUSDT", syms)

    def test_net_pnl_after_friction_reduces_win(self) -> None:
        gross = net_pnl_after_friction("LONG", 100.0, 101.0, 1000.0)
        self.assertLess(gross, 10.0)

    def test_round_trip_friction_bps(self) -> None:
        n = 10_000.0
        expected = n * 2.0 * (taker_bps_per_side() + slippage_bps_per_side()) / 10_000.0
        self.assertAlmostEqual(round_trip_friction_usdt(n), expected, places=4)


if __name__ == "__main__":
    unittest.main()
