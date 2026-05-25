"""动量纸面扫描 — 换仓与 mark 逻辑。"""

from __future__ import annotations

import sqlite3
import unittest
from unittest.mock import patch

from momentum_db import fetch_open_by_side, migrate_mom_tables
from momentum_scanner import _mark_open_row, run_scan_conn


class TestMomentumScanner(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        migrate_mom_tables(self.conn.cursor())
        self.conn.commit()

    def tearDown(self):
        self.conn.close()

    @patch("momentum_config.mom_filter_enabled", return_value=False)
    @patch("momentum_scanner.fetch_momentum_targets")
    @patch("momentum_scanner.fetch_mark_price")
    def test_open_long_and_short(self, mock_px, mock_targets, _mock_filter):
        mock_targets.return_value = (
            "BTCUSDT",
            "ETHUSDT",
            {"movers_total": 2, "long_event_raw": {}, "short_event_raw": {}},
        )
        mock_px.side_effect = lambda s: {"BTCUSDT": 50000.0, "ETHUSDT": 3000.0}[s]

        stats = run_scan_conn(self.conn, notify=False)
        self.assertEqual(stats["opens"], 2)
        cur = self.conn.cursor()
        long_row = fetch_open_by_side(cur, "LONG")
        short_row = fetch_open_by_side(cur, "SHORT")
        self.assertIsNotNone(long_row)
        self.assertIsNotNone(short_row)
        self.assertEqual(long_row["symbol"], "BTCUSDT")
        self.assertEqual(short_row["symbol"], "ETHUSDT")

    @patch("momentum_config.mom_filter_enabled", return_value=False)
    @patch("momentum_scanner.fetch_momentum_targets")
    @patch("momentum_scanner.fetch_mark_price")
    def test_rotate_long_closes_and_reopens(self, mock_px, mock_targets, _mock_filter):
        mock_px.side_effect = lambda s: {"BTCUSDT": 50000.0, "ETHUSDT": 3000.0, "SOLUSDT": 100.0}[s]

        mock_targets.return_value = (
            "BTCUSDT",
            None,
            {"movers_total": 1, "long_event_raw": {}, "short_event_raw": {}},
        )
        run_scan_conn(self.conn, notify=False)

        mock_targets.return_value = (
            "SOLUSDT",
            None,
            {"movers_total": 1, "long_event_raw": {}, "short_event_raw": {}},
        )
        stats = run_scan_conn(self.conn, notify=False)
        self.assertEqual(stats["closes"], 1)
        self.assertEqual(stats["opens"], 1)
        cur = self.conn.cursor()
        row = fetch_open_by_side(cur, "LONG")
        self.assertEqual(row["symbol"], "SOLUSDT")
        cur.execute("SELECT COUNT(*) FROM mom_settlements")
        self.assertEqual(int(cur.fetchone()[0]), 1)

    @patch("momentum_config.mom_filter_enabled", return_value=False)
    @patch("momentum_scanner.fetch_momentum_targets")
    @patch("momentum_scanner.fetch_mark_price")
    def test_mark_preserves_recorded_at(self, mock_px, mock_targets, _mock_filter):
        mock_targets.return_value = (
            "BTCUSDT",
            None,
            {"movers_total": 1, "long_event_raw": {}, "short_event_raw": {}},
        )
        mock_px.side_effect = [50000.0, 51000.0]

        run_scan_conn(self.conn, notify=False)
        cur = self.conn.cursor()
        row = fetch_open_by_side(cur, "LONG")
        opened_at = row["recorded_at_utc"]

        _mark_open_row(cur, row, mark=51000.0, now_utc="2099-01-01T00:00:00Z")
        self.conn.commit()

        cur.execute("SELECT recorded_at_utc, updated_at_utc FROM mom_signals WHERE id=?", (row["id"],))
        r = cur.fetchone()
        self.assertEqual(r["recorded_at_utc"], opened_at)
        self.assertEqual(r["updated_at_utc"], "2099-01-01T00:00:00Z")

    @patch("momentum_config.mom_filter_enabled", return_value=False)
    @patch("momentum_scanner.cfg.MOM_COOLDOWN_SEC", 3600)
    @patch("momentum_scanner.fetch_momentum_targets")
    @patch("momentum_scanner.fetch_mark_price")
    def test_cooldown_blocks_open_not_close(self, mock_px, mock_targets, _mock_filter):
        mock_px.side_effect = lambda s: {"BTCUSDT": 100.0, "ETHUSDT": 200.0}[s]
        mock_targets.return_value = (
            "BTCUSDT",
            None,
            {"long_event_raw": {}, "short_event_raw": {}},
        )
        run_scan_conn(self.conn, notify=False)

        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO mom_settlements (
                settled_at_utc, signal_id, symbol, side, outcome,
                entry_price, exit_price, pnl_usdt, virtual_notional_usdt, exit_rule
            ) VALUES (?, 1, 'ETHUSDT', 'LONG', 'flat', 100, 100, 0, 100, 'rotate')
            """,
            ("2099-01-01T00:00:00Z",),
        )
        self.conn.commit()

        mock_targets.return_value = (
            "ETHUSDT",
            None,
            {"long_event_raw": {}, "short_event_raw": {}},
        )
        stats = run_scan_conn(self.conn, notify=False)
        self.assertEqual(stats["closes"], 1)
        self.assertEqual(stats["opens"], 0)
        self.assertTrue(
            any("closed_without_reopen:cooldown:ETHUSDT" in s for s in stats["skipped"])
        )
        self.assertIsNone(fetch_open_by_side(cur, "LONG"))

    @patch("momentum_config.mom_filter_enabled", return_value=False)
    @patch("momentum_scanner.cfg.MOM_COOLDOWN_SEC", 0)
    @patch("momentum_scanner.cfg.MOM_TRAIL_REOPEN_COOLDOWN_SEC", 3600)
    @patch("momentum_scanner.fetch_momentum_targets")
    @patch("momentum_scanner.fetch_mark_price")
    def test_trail_reopen_cooldown_blocks_same_symbol(
        self, mock_px, mock_targets, _mock_filter
    ):
        mock_px.side_effect = lambda s: {"BTCUSDT": 100.0}[s]
        mock_targets.return_value = (
            "BTCUSDT",
            None,
            {"long_event_raw": {}, "short_event_raw": {}},
        )
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO mom_settlements (
                settled_at_utc, signal_id, symbol, side, outcome,
                entry_price, exit_price, pnl_usdt, virtual_notional_usdt, exit_rule
            ) VALUES (?, 1, 'BTCUSDT', 'LONG', 'win', 100, 105, 50, 1000, 'trail_tier1')
            """,
            ("2099-01-01T00:00:00Z",),
        )
        self.conn.commit()
        stats = run_scan_conn(self.conn, notify=False)
        self.assertEqual(stats["opens"], 0)
        self.assertTrue(
            any("trail_reopen_cooldown:BTCUSDT" in s for s in stats["skipped"])
        )

    @patch("momentum_scanner.fetch_momentum_targets")
    def test_top_movers_error_writes_run(self, mock_targets):
        mock_targets.return_value = (None, None, {"error": "empty_top_movers"})
        stats = run_scan_conn(self.conn, notify=False)
        self.assertFalse(stats["ok"])
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM mom_runs")
        self.assertEqual(int(cur.fetchone()[0]), 1)
        cur.execute("SELECT detail_json FROM mom_runs ORDER BY id DESC LIMIT 1")
        detail = cur.fetchone()[0]
        self.assertIn("empty_top_movers", detail)

    @patch("momentum_scanner.cfg.MOM_TRAIL_ENABLED", True)
    @patch("momentum_config.mom_filter_enabled", return_value=False)
    @patch("momentum_scanner.fetch_momentum_targets")
    @patch("momentum_scanner.fetch_mark_price")
    def test_trail_stop_before_rotate(self, mock_px, mock_targets, _mock_filter):
        mock_px.side_effect = lambda s: {"BTCUSDT": 50000.0, "SOLUSDT": 100.0}[s]
        mock_targets.return_value = (
            "BTCUSDT",
            None,
            {"movers_total": 1, "long_event_raw": {}, "short_event_raw": {}},
        )
        run_scan_conn(self.conn, notify=False)

        mock_px.side_effect = lambda s: {"BTCUSDT": 48200.0, "SOLUSDT": 100.0}[s]
        mock_targets.return_value = (
            "SOLUSDT",
            None,
            {"movers_total": 1, "long_event_raw": {}, "short_event_raw": {}},
        )
        stats = run_scan_conn(self.conn, notify=False)
        self.assertEqual(stats["closes"], 1)
        self.assertEqual(stats["opens"], 1)
        cur = self.conn.cursor()
        cur.execute(
            "SELECT exit_rule FROM mom_settlements ORDER BY id DESC LIMIT 1"
        )
        self.assertEqual(cur.fetchone()[0], "trail_stop")
        row = fetch_open_by_side(cur, "LONG")
        self.assertEqual(row["symbol"], "SOLUSDT")

    @patch("momentum_scanner.cfg.MOM_TRAIL_ENABLED", True)
    @patch("momentum_config.mom_filter_enabled", return_value=False)
    @patch("momentum_scanner.fetch_momentum_targets")
    @patch("momentum_scanner.fetch_mark_price")
    def test_trail_stop_on_mark_same_scan(self, mock_px, mock_targets, _mock_filter):
        """同轮：开头 trail 未触发，持币 mark 更新时触发硬止损。"""
        mock_px.side_effect = lambda s: 50000.0 if s == "BTCUSDT" else 100.0
        mock_targets.return_value = (
            "BTCUSDT",
            None,
            {"movers_total": 1, "long_event_raw": {}, "short_event_raw": {}},
        )
        run_scan_conn(self.conn, notify=False)

        prices = iter([49900.0, 48200.0])

        def _px(sym: str) -> float:
            if sym == "BTCUSDT":
                return next(prices)
            return 100.0

        mock_px.side_effect = _px
        mock_targets.return_value = (
            "BTCUSDT",
            None,
            {"movers_total": 1, "long_event_raw": {}, "short_event_raw": {}},
        )
        stats = run_scan_conn(self.conn, notify=False)
        self.assertEqual(stats["closes"], 1)
        cur = self.conn.cursor()
        cur.execute(
            "SELECT exit_rule FROM mom_settlements ORDER BY id DESC LIMIT 1"
        )
        self.assertEqual(cur.fetchone()[0], "trail_stop")
        self.assertIsNone(fetch_open_by_side(cur, "LONG"))

    @patch("momentum_scanner.cfg.MOM_TRAIL_ENABLED", True)
    @patch("momentum_config.mom_filter_enabled", return_value=False)
    @patch("momentum_scanner.fetch_momentum_targets")
    @patch("momentum_scanner.fetch_mark_price")
    def test_trail_on_top_movers_error(self, mock_px, mock_targets, _mock_filter):
        mock_px.side_effect = lambda s: 50000.0 if s == "BTCUSDT" else 100.0
        mock_targets.return_value = (
            "BTCUSDT",
            None,
            {"movers_total": 1, "long_event_raw": {}, "short_event_raw": {}},
        )
        run_scan_conn(self.conn, notify=False)

        mock_px.side_effect = lambda s: 48200.0 if s == "BTCUSDT" else 100.0
        mock_targets.return_value = (None, None, {"error": "empty_top_movers"})
        stats = run_scan_conn(self.conn, notify=False)
        self.assertFalse(stats["ok"])
        self.assertEqual(stats["closes"], 1)
        cur = self.conn.cursor()
        cur.execute(
            "SELECT exit_rule FROM mom_settlements ORDER BY id DESC LIMIT 1"
        )
        self.assertEqual(cur.fetchone()[0], "trail_stop")
        self.assertIsNone(fetch_open_by_side(cur, "LONG"))

    @patch("momentum_scanner.cfg.MOM_NOTIONAL_USDT", 0.0)
    @patch("momentum_scanner.fetch_momentum_targets")
    def test_zero_notional_skips(self, mock_targets):
        mock_targets.return_value = ("BTCUSDT", "ETHUSDT", {})
        stats = run_scan_conn(self.conn, notify=False)
        self.assertFalse(stats["ok"])
        self.assertEqual(stats["error"], "zero_notional")
        mock_targets.assert_not_called()
        cur = self.conn.cursor()
        self.assertIsNone(fetch_open_by_side(cur, "LONG"))


if __name__ == "__main__":
    unittest.main()
