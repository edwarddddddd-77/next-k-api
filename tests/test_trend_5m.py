#!/usr/bin/env python3
"""5m 趋势判定单元测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_API_ROOT = Path(__file__).resolve().parent.parent
if str(_API_ROOT) not in sys.path:
    sys.path.insert(0, str(_API_ROOT))

from trend_5m import assess_trend_5m, partition_trend_items


class ApplyTrendPayloadTests(unittest.TestCase):
    def test_apply_trend_refreshes_note_from_period(self) -> None:
        import top_trader_radar as mod

        out = mod.apply_trend_5m_to_payload(
            {
                "ok": True,
                "period": "15m",
                "items": [],
                "note": "5m 趋势：旧说明",
            }
        )
        self.assertEqual(
            out["note"],
            "15m 趋势：PosLSR+Taker+OI/价；无 Smart Money 盈利/均价",
        )


def _snap(pos_lsr: float, taker: float, tags=None):
    return {
        "symbol": "TESTUSDT",
        "top_position_lsr": pos_lsr,
        "taker_bsr": taker,
        "top_account_long_pct": 50,
        "top_position_long_pct": 50,
        "signal_tags": tags or [],
    }


class Trend5mTests(unittest.TestCase):
    def test_long_trend(self) -> None:
        out = assess_trend_5m(
            _snap(1.25, 1.12, ["大户+Taker同向多"]),
            {"d6h": 5.0, "px_chg": 2.0},
            pool_sources=["hot_oi", "focus"],
        )
        self.assertEqual(out["trend_verdict"], "long")
        self.assertGreaterEqual(out["trend_score"], 65)

    def test_short_trend(self) -> None:
        out = assess_trend_5m(
            _snap(0.75, 0.88, ["大户+Taker同向空"]),
            {"d6h": -4.0, "px_chg": -3.0},
            pool_sources=["focus"],
        )
        self.assertEqual(out["trend_verdict"], "short")
        self.assertGreater(out["trend_score"], 0)

    def test_avoid_price_up_oi_drain(self) -> None:
        out = assess_trend_5m(
            _snap(1.3, 1.1, ["大户+Taker同向多"]),
            {"d6h": -8.0, "px_chg": 3.0},
        )
        self.assertEqual(out["trend_verdict"], "avoid")
        self.assertIn("价涨OI流出", out["trend_reasons"])

    def test_partition(self) -> None:
        items = [
            {"symbol": "A", "trend_verdict": "long", "trend_score": 80},
            {"symbol": "B", "trend_verdict": "long", "trend_score": 60},
            {"symbol": "C", "trend_verdict": "short", "trend_score": 70},
        ]
        parts = partition_trend_items(items)
        self.assertEqual(len(parts["trend_long"]), 2)
        self.assertEqual(parts["trend_long"][0]["symbol"], "A")


class RehydratePoolSourcesTests(unittest.TestCase):
    def test_rehydrate_from_db_tables(self) -> None:
        import sqlite3
        import top_trader_radar as mod

        conn = sqlite3.connect(":memory:")
        conn.execute(
            """CREATE TABLE worth_watch_hot_oi (
            symbol TEXT PRIMARY KEY, coin TEXT, generated_date TEXT,
            last_seen_cst TEXT, rank_in_category INTEGER,
            summary_line TEXT, detail_json TEXT, bpc_json TEXT, bpc_updated_cst TEXT)"""
        )
        conn.execute(
            "INSERT INTO worth_watch_hot_oi (symbol, coin, generated_date, last_seen_cst) "
            "VALUES ('BTCUSDT', 'BTC', '2026-01-01', 'x')"
        )
        conn.commit()

        orig_init = mod.init_db
        orig_params = mod.top_trader_params
        try:
            mod.init_db = lambda: conn
            mod.top_trader_params = lambda: orig_params().__class__(
                universe="trend_5m",
                period=orig_params().period,
                pool_max=0,
                explicit_symbols=(),
                spacing_sec=1.0,
                jitter_sec=0.2,
                retention_days=7,
                min_interval_sec=0.12,
            )
            items = [{"symbol": "BTCUSDT"}]
            mod.rehydrate_pool_sources(items, universe="trend_5m")
            self.assertEqual(items[0]["pool_sources"], ["hot_oi"])
        finally:
            mod.init_db = orig_init
            mod.top_trader_params = orig_params
            conn.close()


if __name__ == "__main__":
    unittest.main()
