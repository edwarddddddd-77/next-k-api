#!/usr/bin/env python3
"""大户多空 + Taker 信号推导单元测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_API_ROOT = Path(__file__).resolve().parent.parent
if str(_API_ROOT) not in sys.path:
    sys.path.insert(0, str(_API_ROOT))

from top_trader_radar import clear_top_trader_data, derive_signal_tags, load_top_trader_snapshot_auto


class TopTraderRadarTests(unittest.TestCase):
    def test_long_bias_and_taker_buy(self) -> None:
        snap = {
            "top_position_lsr": 1.35,
            "top_account_long_pct": 55,
            "top_position_long_pct": 58,
            "taker_bsr": 1.12,
        }
        tags, summary = derive_signal_tags(snap)
        self.assertIn("大户持仓偏多", tags)
        self.assertIn("主动买盘", tags)
        self.assertIn("PosLSR", summary)

    def test_account_position_divergence(self) -> None:
        snap = {
            "top_position_lsr": 0.9,
            "top_account_long_pct": 62,
            "top_position_long_pct": 45,
            "taker_bsr": 1.0,
        }
        tags, _ = derive_signal_tags(snap)
        self.assertIn("账户偏多/持仓偏空", tags)

    def test_aligned_short(self) -> None:
        snap = {
            "top_position_lsr": 0.75,
            "top_account_long_pct": 40,
            "top_position_long_pct": 38,
            "taker_bsr": 0.88,
        }
        tags, _ = derive_signal_tags(snap)
        self.assertIn("大户持仓偏空", tags)
        self.assertIn("主动卖盘", tags)
        self.assertIn("大户+Taker同向空", tags)


class TopTraderSnapshotAutoTests(unittest.TestCase):
    def test_auto_prefers_disk_ok(self) -> None:
        import top_trader_radar as mod

        orig_disk = mod.load_top_trader_snapshot_from_disk
        orig_db = mod.load_latest_top_trader_from_db
        try:
            mod.load_top_trader_snapshot_from_disk = lambda: {"ok": True, "items": [{"symbol": "BTCUSDT"}]}
            mod.load_latest_top_trader_from_db = lambda **_: {"ok": True, "items": [{"symbol": "ETHUSDT"}]}
            out = load_top_trader_snapshot_auto()
            self.assertTrue(out.get("ok"))
            self.assertEqual(out["items"][0]["symbol"], "BTCUSDT")
        finally:
            mod.load_top_trader_snapshot_from_disk = orig_disk
            mod.load_latest_top_trader_from_db = orig_db

    def test_auto_falls_back_to_db_when_disk_missing(self) -> None:
        import top_trader_radar as mod

        orig_disk = mod.load_top_trader_snapshot_from_disk
        orig_db = mod.load_latest_top_trader_from_db
        try:
            mod.load_top_trader_snapshot_from_disk = lambda: {"ok": False, "error": "no_snapshot"}
            mod.load_latest_top_trader_from_db = lambda **_: {"ok": True, "items": [{"symbol": "ETHUSDT"}]}
            out = load_top_trader_snapshot_auto()
            self.assertEqual(out["items"][0]["symbol"], "ETHUSDT")
        finally:
            mod.load_top_trader_snapshot_from_disk = orig_disk
            mod.load_latest_top_trader_from_db = orig_db


class TopTraderClearTests(unittest.TestCase):
    def test_clear_top_trader_data(self) -> None:
        import sqlite3
        import tempfile

        import top_trader_radar as mod

        conn = sqlite3.connect(":memory:")
        mod.ensure_top_trader_schema(conn)
        conn.execute(
            """INSERT INTO top_trader_snapshots
               (run_id, run_at_ms, generated_date, symbol, period, ts)
               VALUES ('r1', 1, '2026-01-01', 'BTCUSDT', '15m', 1)"""
        )
        conn.commit()

        with tempfile.TemporaryDirectory() as tmp:
            orig = mod._data_dir
            try:
                mod._data_dir = lambda: Path(tmp)
                snap = mod.snapshot_path()
                snap.write_text('{"ok": true}', encoding="utf-8")
                out = clear_top_trader_data(conn)
                self.assertEqual(out["deleted_top_trader_rows"], 1)
                self.assertTrue(out["disk_snapshot_removed"])
                self.assertFalse(snap.is_file())
                self.assertEqual(
                    conn.execute("SELECT COUNT(*) FROM top_trader_snapshots").fetchone()[0],
                    0,
                )
            finally:
                mod._data_dir = orig
        conn.close()


if __name__ == "__main__":
    unittest.main()
