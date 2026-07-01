"""KK dashboard API 测试。"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orb.kk.config import KKConfig
from orb.kk.dashboard import build_kk_summary, clear_kk_db, fetch_kk_trades
from orb.kk.db import migrate_kk_tables, save_state_json, save_wallet


class TestKKDashboard(unittest.TestCase):
    def _with_db(self):
        tmp = tempfile.TemporaryDirectory()
        db_path = Path(tmp.name) / "kk_test.db"
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        migrate_kk_tables(cur)
        conn.commit()

        import accumulation_radar

        orig = accumulation_radar.init_db

        def _init():
            return sqlite3.connect(str(db_path))

        accumulation_radar.init_db = _init
        return tmp, conn, orig

    def test_summary_and_trades_empty(self):
        kk = KKConfig(
            symbols=["INTCUSDT", "COINUSDT"],
            equity_usdt=14.0,
            compound=True,
            engine="paper",
        )
        tmp, conn, orig = self._with_db()
        try:
            summary = build_kk_summary(kk=kk)
            self.assertTrue(summary.get("ok"))
            self.assertEqual(summary.get("robot_count"), 2)
            self.assertEqual(len(summary.get("robots") or []), 2)
            self.assertTrue(summary.get("robot_bound"))
            trades = fetch_kk_trades(limit=10, kk=kk)
            self.assertTrue(trades.get("ok"))
            self.assertEqual(trades.get("trades"), [])
        finally:
            import accumulation_radar

            accumulation_radar.init_db = orig
            conn.close()
            tmp.cleanup()

    def test_open_position_in_trades(self):
        kk = KKConfig(symbols=["INTCUSDT"], equity_usdt=14.0, engine="paper")
        tmp, conn, orig = self._with_db()
        cur = conn.cursor()
        save_wallet(cur, "INTCUSDT", 14.0, now_utc="2026-06-01T00:00:00Z")
        save_state_json(
            cur,
            "INTCUSDT",
            "2026-06-01",
            state={"ctx": {"pos": {"side": 1, "entry": 100.0, "notional": 50.0}}},
            last_bar_ms=1,
        )
        conn.commit()
        try:
            with mock.patch("orb.kk.dashboard._session_date_now", return_value="2026-06-01"):
                summary = build_kk_summary(kk=kk)
                self.assertEqual(summary.get("open_positions"), 1)
                trades = fetch_kk_trades(limit=10, kk=kk)
                self.assertEqual(len(trades.get("trades") or []), 1)
                self.assertEqual(trades["trades"][0].get("side"), "LONG")
        finally:
            import accumulation_radar

            accumulation_radar.init_db = orig
            conn.close()
            tmp.cleanup()

    def test_clear_db(self):
        tmp, conn, orig = self._with_db()
        cur = conn.cursor()
        save_wallet(cur, "INTCUSDT", 14.0, now_utc="2026-06-01T00:00:00Z")
        conn.commit()
        try:
            out = clear_kk_db()
            self.assertTrue(out.get("ok"))
            self.assertEqual(out["deleted"].get("kk_symbol_bots"), 1)
        finally:
            import accumulation_radar

            accumulation_radar.init_db = orig
            conn.close()
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
