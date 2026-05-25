"""动量移动止盈独立定时任务。"""

from __future__ import annotations

import sqlite3
import unittest
from unittest.mock import patch

from momentum_db import fetch_open_by_side, migrate_mom_tables
from momentum_scanner import run_trail_checks_conn


class TestMomentumTrailTask(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        migrate_mom_tables(self.conn.cursor())
        self.conn.commit()

    def tearDown(self):
        self.conn.close()

    @patch("momentum_scanner.cfg.MOM_TRAIL_ENABLED", False)
    def test_trail_disabled_skips(self):
        stats = run_trail_checks_conn(self.conn, notify=False)
        self.assertIn("trail_disabled", stats["skipped"])

    @patch("momentum_scanner.cfg.MOM_TRAIL_ENABLED", True)
    @patch("momentum_config.mom_filter_enabled", return_value=False)
    @patch("momentum_scanner.fetch_momentum_targets")
    @patch("momentum_scanner.fetch_mark_price", return_value=50000.0)
    def test_trail_only_closes_without_targets(
        self, _mock_px, mock_targets, _mock_filter
    ):
        from momentum_scanner import run_scan_conn

        mock_targets.return_value = (
            "BTCUSDT",
            None,
            {"movers_total": 1, "long_event_raw": {}, "short_event_raw": {}},
        )
        run_scan_conn(self.conn, notify=False)

        mock_targets.reset_mock()
        with patch("momentum_scanner.fetch_mark_price", return_value=48200.0):
            stats = run_trail_checks_conn(self.conn, notify=False)
        mock_targets.assert_not_called()
        self.assertEqual(stats["closes"], 1)
        self.assertEqual(stats["task"], "trail")
        cur = self.conn.cursor()
        self.assertIsNone(fetch_open_by_side(cur, "LONG"))


if __name__ == "__main__":
    unittest.main()
