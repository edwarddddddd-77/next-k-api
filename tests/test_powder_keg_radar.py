#!/usr/bin/env python3
"""火药桶雷达筛选与打分单元测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_API_ROOT = Path(__file__).resolve().parent.parent
if str(_API_ROOT) not in sys.path:
    sys.path.insert(0, str(_API_ROOT))

from powder_keg_config import powder_keg_params
from powder_keg_radar import (
    _dedupe_items_for_insert,
    _funding_side_metadata,
    _passes_hard_filters,
    _retention_cutoff_ms,
    _score_row,
)


class PowderKegRadarTests(unittest.TestCase):
    def setUp(self) -> None:
        self.p = powder_keg_params()

    def test_passes_all_filters(self) -> None:
        row = {
            "vol": 10_000_000,
            "px_chg": 2.0,
            "range_6h_pct": 3.0,
            "fr_pct": -0.08,
            "oi_usd": 5_000_000,
            "oi_delta_1h_pct": 4.0,
            "oi_delta_6h_pct": 8.0,
        }
        self.assertTrue(_passes_hard_filters(row, self.p))

    def test_passes_practical_sideways_bounds(self) -> None:
        """实战级：24h≤10%、6h振幅≤8.5% 可通过。"""
        row = {
            "vol": 10_000_000,
            "px_chg": 9.0,
            "range_6h_pct": 8.0,
            "fr_pct": -0.03,
            "oi_usd": 5_000_000,
            "oi_delta_1h_pct": 3.0,
            "oi_delta_6h_pct": 6.0,
        }
        self.assertTrue(_passes_hard_filters(row, self.p))

    def test_rejects_high_px_chg(self) -> None:
        row = {
            "vol": 10_000_000,
            "px_chg": 11.0,
            "range_6h_pct": 3.0,
            "fr_pct": -0.08,
            "oi_usd": 5_000_000,
            "oi_delta_1h_pct": 4.0,
            "oi_delta_6h_pct": 8.0,
        }
        self.assertFalse(_passes_hard_filters(row, self.p))

    def test_rejects_weak_oi(self) -> None:
        row = {
            "vol": 10_000_000,
            "px_chg": 2.0,
            "range_6h_pct": 3.0,
            "fr_pct": -0.08,
            "oi_usd": 5_000_000,
            "oi_delta_1h_pct": 1.0,
            "oi_delta_6h_pct": 1.5,
        }
        self.assertFalse(_passes_hard_filters(row, self.p))

    def test_rejects_oi_decrease_even_if_large_abs(self) -> None:
        """OI 减仓（负变化）不计入激增，即使 |Δ| 很大。"""
        row = {
            "vol": 10_000_000,
            "px_chg": 2.0,
            "range_6h_pct": 3.0,
            "fr_pct": -0.08,
            "oi_usd": 5_000_000,
            "oi_delta_1h_pct": -4.0,
            "oi_delta_6h_pct": -8.0,
        }
        self.assertFalse(_passes_hard_filters(row, self.p))

    def test_passes_oi_if_only_6h_positive(self) -> None:
        row = {
            "vol": 10_000_000,
            "px_chg": 2.0,
            "range_6h_pct": 3.0,
            "fr_pct": -0.08,
            "oi_usd": 5_000_000,
            "oi_delta_1h_pct": 0.5,
            "oi_delta_6h_pct": 6.0,
        }
        self.assertTrue(_passes_hard_filters(row, self.p))

    def test_score_orders_stronger_oi(self) -> None:
        base = {
            "vol": 10_000_000,
            "px_chg": 1.0,
            "range_6h_pct": 2.0,
            "fr_pct": -0.06,
            "oi_delta_1h_pct": 3.0,
            "oi_delta_6h_pct": 6.0,
        }
        weak = {**base, "oi_delta_6h_pct": 5.5}
        strong = {**base, "oi_delta_6h_pct": 15.0}
        self.assertGreater(_score_row(strong, self.p), _score_row(weak, self.p))


    def test_funding_side_negative_long_positive_short(self) -> None:
        neg = _funding_side_metadata(-0.08)
        self.assertEqual(neg["funding_sign"], "negative")
        self.assertEqual(neg["allowed_side"], "LONG")
        self.assertAlmostEqual(neg["funding_rate_abs_pct"], 0.08)

        pos = _funding_side_metadata(0.06)
        self.assertEqual(pos["funding_sign"], "positive")
        self.assertEqual(pos["allowed_side"], "SHORT")

    def test_dedupe_items_keeps_higher_score(self) -> None:
        items = [
            {"symbol": "BTCUSDT", "score": 10.0},
            {"symbol": "btcusdt", "score": 99.0},
            {"symbol": "ETHUSDT", "score": 5.0},
        ]
        out = _dedupe_items_for_insert(items)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["symbol"], "BTCUSDT")
        self.assertEqual(out[0]["score"], 99.0)

    def test_pre_sort_pool_score_desc(self) -> None:
        rows = [
            {"pool_score": 40.0, "fr_pct": -0.05, "vol": 1e6},
            {"pool_score": 90.0, "fr_pct": -0.03, "vol": 5e5},
        ]
        rows.sort(
            key=lambda r: (float(r["pool_score"]), abs(r["fr_pct"]), r["vol"]),
            reverse=True,
        )
        self.assertEqual(rows[0]["pool_score"], 90.0)

    def test_retention_cutoff_24h(self) -> None:
        from datetime import datetime, timedelta, timezone

        cst = timezone(timedelta(hours=8))
        now = datetime(2025, 5, 22, 12, 0, tzinfo=cst)
        cutoff = _retention_cutoff_ms(retention_hours=24, now=now)
        expect = int((now - timedelta(hours=24)).timestamp() * 1000)
        self.assertEqual(cutoff, expect)


if __name__ == "__main__":
    unittest.main()
