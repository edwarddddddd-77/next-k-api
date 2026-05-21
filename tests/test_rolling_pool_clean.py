#!/usr/bin/env python3
"""触轨池已废弃的 Phase 2 滚动淘汰规则单元测试（逻辑仍保留供回归）。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_API_ROOT = Path(__file__).resolve().parent.parent
if str(_API_ROOT) not in sys.path:
    sys.path.insert(0, str(_API_ROOT))

from zct_vwap_asset_pool import rolling_evict_reason


def _row(**kwargs):
    base = {
        "win": 10,
        "loss": 5,
        "n_trades": 20,
        "win_rate_touch_sl_tp": 0.75,
        "profit_factor_net": 1.3,
        "consecutive_losses_at_end": 0,
        "t4_win_rate_touch_sl_tp": 0.75,
    }
    base.update(kwargs)
    return base


class RollingPoolCleanTests(unittest.TestCase):
    def test_keep_when_all_pass(self) -> None:
        self.assertIsNone(rolling_evict_reason(_row()))

    def test_evict_low_win_rate(self) -> None:
        r = rolling_evict_reason(_row(win_rate_touch_sl_tp=0.69))
        self.assertEqual(r, "touch_win_rate_below_rolling_min")

    def test_keep_at_exactly_70_pct(self) -> None:
        self.assertIsNone(rolling_evict_reason(_row(win_rate_touch_sl_tp=0.70)))

    def test_evict_low_pf(self) -> None:
        r = rolling_evict_reason(_row(profit_factor_net=1.10))
        self.assertEqual(r, "profit_factor_below_rolling_min")

    def test_keep_pf_at_115(self) -> None:
        self.assertIsNone(rolling_evict_reason(_row(profit_factor_net=1.15)))

    def test_evict_consecutive_losses_veto(self) -> None:
        r = rolling_evict_reason(_row(consecutive_losses_at_end=3))
        self.assertEqual(r, "consecutive_losses_at_end_veto")

    def test_keep_two_consecutive_losses(self) -> None:
        self.assertIsNone(rolling_evict_reason(_row(consecutive_losses_at_end=2)))

    def test_evict_when_win_rate_missing(self) -> None:
        r = rolling_evict_reason(_row(win_rate_touch_sl_tp=None))
        self.assertEqual(r, "touch_win_rate_unavailable")

    def test_skip_wr_evict_when_sample_too_small(self) -> None:
        r = rolling_evict_reason(
            _row(win=1, loss=1, win_rate_touch_sl_tp=0.5, profit_factor_net=0.5),
            min_win_loss_abs=5,
        )
        self.assertIsNone(r)

    def test_consecutive_veto_ignores_min_sample(self) -> None:
        r = rolling_evict_reason(
            _row(win=1, loss=2, win_rate_touch_sl_tp=0.33, consecutive_losses_at_end=3),
            min_win_loss_abs=10,
        )
        self.assertEqual(r, "consecutive_losses_at_end_veto")

    def test_evict_low_t4_win_rate(self) -> None:
        r = rolling_evict_reason(
            _row(t4_win_rate_touch_sl_tp=0.35),
            min_t4_touch_win_rate_evict=0.40,
        )
        self.assertEqual(r, "t4_touch_win_rate_below_rolling_min")

    def test_keep_t4_at_40_pct(self) -> None:
        self.assertIsNone(
            rolling_evict_reason(
                _row(t4_win_rate_touch_sl_tp=0.40),
                min_t4_touch_win_rate_evict=0.40,
            )
        )

    def test_t4_evict_even_when_24h_sample_below_floor(self) -> None:
        """小样本豁免 24h 胜率/PF，但不豁免 T4。"""
        r = rolling_evict_reason(
            _row(
                win=2,
                loss=2,
                win_rate_touch_sl_tp=0.80,
                profit_factor_net=2.0,
                t4_win_rate_touch_sl_tp=0.30,
            ),
            min_win_loss_abs=10,
            min_t4_touch_win_rate_evict=0.40,
        )
        self.assertEqual(r, "t4_touch_win_rate_below_rolling_min")


if __name__ == "__main__":
    unittest.main()
