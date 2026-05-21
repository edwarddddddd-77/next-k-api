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
    t4_bucket_touch_metrics,
    taker_bps_per_side,
    slippage_bps_per_side,
    trailing_consecutive_losses_at_end,
)
from touch_pool_config import touch_pool_4h_filter_params
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

    def test_touch_pool_4h_default_params(self) -> None:
        p = touch_pool_4h_filter_params()
        self.assertAlmostEqual(p["days"], 6.0 / 24.0)
        self.assertEqual(p["min_total_trades"], 5)
        self.assertEqual(p["min_win_loss_abs"], 3)
        self.assertEqual(p["min_touch_trades"], 3)
        self.assertAlmostEqual(p["min_touch_win_rate"], 0.70)
        self.assertAlmostEqual(p["min_profit_factor"], 1.30)
        self.assertEqual(p["max_consecutive_losses_at_end"], 1)
        self.assertEqual(p["min_t4_touch_win_rate"], 0.0)

    def test_criteria_doc_matches_walk_days(self) -> None:
        from zct_vwap_asset_pool import touch_pool_4h_criteria_doc

        doc = touch_pool_4h_criteria_doc(
            walk_days=0.5,
            min_total_trades=5,
            min_win_loss_abs=3,
            min_touch_win_rate=0.70,
            min_profit_factor=1.30,
            max_consecutive_losses_at_end=1,
            min_t4_touch_win_rate=0.0,
        )
        self.assertAlmostEqual(doc["walk_hours"], 12.0)
        self.assertIn("12h walk", doc["entry_rule"])

    def test_filter_requires_profit_factor(self) -> None:
        summary = {
            "per_symbol": {
                "BTCUSDT": {
                    "win": 3,
                    "loss": 1,
                    "n_trades": 8,
                    "expired": 0,
                    "unresolved": 0,
                    "win_rate_touch_sl_tp": 0.75,
                    "profit_factor_net": 1.1,
                    "consecutive_losses_at_end": 0,
                },
                "ETHUSDT": {
                    "win": 3,
                    "loss": 1,
                    "n_trades": 8,
                    "expired": 0,
                    "unresolved": 0,
                    "win_rate_touch_sl_tp": 0.75,
                    "profit_factor_net": 1.5,
                    "consecutive_losses_at_end": 1,
                },
            }
        }
        filt = _filter_pool(
            summary,
            min_touch_trades=3,
            strict_greater_touch=False,
            min_touch_win_rate=0.70,
            strict_greater_rate=False,
            min_total_trades=5,
            max_expired_ratio=1.0,
            min_win_loss_abs=3,
            min_touch_share=0.0,
            min_profit_factor=1.30,
            max_consecutive_losses_at_end=1,
            min_t4_touch_win_rate=0.0,
        )
        syms = {m["symbol"] for m in filt["matched"]}
        self.assertNotIn("BTCUSDT", syms)
        self.assertIn("ETHUSDT", syms)

    def test_filter_rejects_win_loss_below_3(self) -> None:
        summary = {
            "per_symbol": {
                "XUSDT": {
                    "win": 1,
                    "loss": 1,
                    "n_trades": 8,
                    "expired": 0,
                    "win_rate_touch_sl_tp": 1.0,
                    "profit_factor_net": 2.0,
                    "consecutive_losses_at_end": 0,
                },
            }
        }
        filt = _filter_pool(
            summary,
            min_touch_trades=3,
            strict_greater_touch=False,
            min_touch_win_rate=0.70,
            strict_greater_rate=False,
            min_total_trades=5,
            max_expired_ratio=1.0,
            min_win_loss_abs=3,
            min_touch_share=0.0,
            min_profit_factor=1.30,
            max_consecutive_losses_at_end=1,
            min_t4_touch_win_rate=0.0,
        )
        self.assertEqual(len(filt["matched"]), 0)

    def test_filter_rejects_pf_at_threshold_exclusive(self) -> None:
        summary = {
            "per_symbol": {
                "XUSDT": {
                    "win": 3,
                    "loss": 1,
                    "n_trades": 8,
                    "expired": 0,
                    "win_rate_touch_sl_tp": 0.75,
                    "profit_factor_net": 1.30,
                    "consecutive_losses_at_end": 0,
                },
            }
        }
        filt = _filter_pool(
            summary,
            min_touch_trades=3,
            strict_greater_touch=False,
            min_touch_win_rate=0.70,
            strict_greater_rate=False,
            min_total_trades=5,
            max_expired_ratio=1.0,
            min_win_loss_abs=3,
            min_touch_share=0.0,
            min_profit_factor=1.30,
            max_consecutive_losses_at_end=1,
            min_t4_touch_win_rate=0.0,
        )
        self.assertEqual(len(filt["matched"]), 0)

    def test_filter_rejects_end_streak_over_1(self) -> None:
        summary = {
            "per_symbol": {
                "XUSDT": {
                    "win": 3,
                    "loss": 1,
                    "n_trades": 8,
                    "expired": 0,
                    "win_rate_touch_sl_tp": 0.75,
                    "profit_factor_net": 1.5,
                    "consecutive_losses_at_end": 2,
                },
            }
        }
        filt = _filter_pool(
            summary,
            min_touch_trades=3,
            strict_greater_touch=False,
            min_touch_win_rate=0.70,
            strict_greater_rate=False,
            min_total_trades=5,
            max_expired_ratio=1.0,
            min_win_loss_abs=3,
            min_touch_share=0.0,
            min_profit_factor=1.30,
            max_consecutive_losses_at_end=1,
            min_t4_touch_win_rate=0.0,
        )
        self.assertEqual(len(filt["matched"]), 0)

    def test_filter_requires_t4_win_rate(self) -> None:
        summary = {
            "per_symbol": {
                "OKUSDT": {
                    "win": 20,
                    "loss": 5,
                    "n_trades": 30,
                    "expired": 0,
                    "win_rate_touch_sl_tp": 0.8,
                    "profit_factor_net": 1.5,
                    "consecutive_losses_at_end": 0,
                    "t4_win_rate_touch_sl_tp": 0.55,
                    "t4_win_plus_loss": 8,
                },
                "BADUSDT": {
                    "win": 20,
                    "loss": 5,
                    "n_trades": 30,
                    "expired": 0,
                    "win_rate_touch_sl_tp": 0.8,
                    "profit_factor_net": 1.5,
                    "consecutive_losses_at_end": 0,
                    "t4_win_rate_touch_sl_tp": 0.45,
                    "t4_win_plus_loss": 8,
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
            min_profit_factor=1.25,
            max_consecutive_losses_at_end=2,
            min_t4_touch_win_rate=0.50,
        )
        syms = {m["symbol"] for m in filt["matched"]}
        self.assertIn("OKUSDT", syms)
        self.assertNotIn("BADUSDT", syms)

    def test_filter_t4_unavailable_reason(self) -> None:
        summary = {
            "per_symbol": {
                "XUSDT": {
                    "win": 20,
                    "loss": 5,
                    "n_trades": 30,
                    "expired": 0,
                    "win_rate_touch_sl_tp": 0.8,
                    "profit_factor_net": 1.5,
                    "consecutive_losses_at_end": 0,
                    "t4_win_rate_touch_sl_tp": None,
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
            min_profit_factor=1.25,
            max_consecutive_losses_at_end=2,
            min_t4_touch_win_rate=0.50,
        )
        self.assertEqual(len(filt["matched"]), 0)
        self.assertIn(
            "t4_touch_win_rate_unavailable",
            filt["rejected"][0]["reject_reason"],
        )

    def test_t4_bucket_metrics(self) -> None:
        end = 100_000_000
        six_h = 6 * 3_600_000
        trades = [
            {
                "symbol": "XUSDT",
                "signal_open_ms": end - 3_600_000,
                "outcome": "win",
            },
            {
                "symbol": "XUSDT",
                "signal_open_ms": end - 2_000_000,
                "outcome": "loss",
            },
            {
                "symbol": "XUSDT",
                "signal_open_ms": end - six_h - 60_000,
                "outcome": "win",
            },
        ]
        m = t4_bucket_touch_metrics(trades, "XUSDT", window_end_ms=end, bucket_hours=6)
        self.assertEqual(m["t4_win"], 1)
        self.assertEqual(m["t4_loss"], 1)
        self.assertAlmostEqual(m["t4_win_rate_touch_sl_tp"], 0.5)

    def test_net_pnl_after_friction_reduces_win(self) -> None:
        gross = net_pnl_after_friction("LONG", 100.0, 101.0, 1000.0)
        self.assertLess(gross, 10.0)

    def test_round_trip_friction_bps(self) -> None:
        n = 10_000.0
        expected = n * 2.0 * (taker_bps_per_side() + slippage_bps_per_side()) / 10_000.0
        self.assertAlmostEqual(round_trip_friction_usdt(n), expected, places=4)


if __name__ == "__main__":
    unittest.main()
