# -*- coding: utf-8 -*-
"""纸面自动开平单元测试。"""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest import mock

import alpha_paper
import xarb_paper


class XarbPaperAutoTests(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.path = Path(self._td.name) / "xarb_paper_trades.json"
        self._path_patch = mock.patch.object(xarb_paper, "_path", return_value=self.path)
        self._path_patch.start()
        os.environ["XARB_PAPER_AUTO"] = "1"
        os.environ["XARB_PAPER_SIZE_USD"] = "1000"
        os.environ["XARB_PAPER_MAX_OPEN"] = "2"
        os.environ["XARB_PAPER_MAX_HOLD_HOURS"] = "24"
        os.environ["XARB_PAPER_CLOSE_FR_RATIO"] = "0.4"

    def tearDown(self) -> None:
        self._path_patch.stop()
        self._td.cleanup()
        for k in (
            "XARB_PAPER_AUTO",
            "XARB_PAPER_SIZE_USD",
            "XARB_PAPER_MAX_OPEN",
            "XARB_PAPER_MAX_HOLD_HOURS",
            "XARB_PAPER_CLOSE_FR_RATIO",
            "XARB_PAPER_AUTO_SECONDARY",
        ):
            os.environ.pop(k, None)

    def _alert_row(self, *, base="BTC", fr_diff=0.001, primary=True):
        # HL funding higher → short HL, long BP
        return {
            "base": base,
            "pair": "hyperliquid/backpack",
            "ex_a": "hyperliquid",
            "ex_b": "backpack",
            "primary": primary,
            "mark_a": 100.0,
            "mark_b": 101.0,
            "funding_8h_a": 0.001,
            "funding_8h_b": 0.001 - fr_diff,
            "funding_diff_8h": fr_diff,
            "funding_diff_8h_pct": fr_diff * 100,
            "funding_alert": True,
            "advice": "short HL / long BP",
        }

    def test_auto_open_primary_funding_alert(self):
        row = self._alert_row()
        board = {
            "rows": [row],
            "funding_alerts": [row],
            "thresholds": {"funding_alert_8h": 0.00025},
        }
        out = xarb_paper.auto_manage_from_board(board)
        self.assertTrue(out["enabled"])
        self.assertEqual(len(out["opened"]), 1)
        book = xarb_paper.list_paper()
        self.assertEqual(book["open_count"], 1)
        t = book["trades"][0]
        self.assertEqual(t["ex_short"], "hyperliquid")
        self.assertEqual(t["ex_long"], "backpack")
        self.assertTrue(t["auto"])

    def test_auto_skip_duplicate_open(self):
        row = self._alert_row()
        board = {
            "rows": [row],
            "funding_alerts": [row],
            "thresholds": {"funding_alert_8h": 0.00025},
        }
        xarb_paper.auto_manage_from_board(board)
        out2 = xarb_paper.auto_manage_from_board(board)
        self.assertEqual(out2["opened"], [])
        self.assertEqual(xarb_paper.list_paper()["open_count"], 1)

    def test_auto_close_on_funding_converge(self):
        row = self._alert_row(fr_diff=0.001)
        board = {
            "rows": [row],
            "funding_alerts": [row],
            "thresholds": {"funding_alert_8h": 0.00025},
        }
        xarb_paper.auto_manage_from_board(board)
        # converge below 0.00025 * 0.4 = 0.0001
        quiet = dict(row)
        quiet["funding_diff_8h"] = 0.00005
        quiet["funding_8h_a"] = 0.0001
        quiet["funding_8h_b"] = 0.00005
        quiet["funding_alert"] = False
        quiet["mark_a"] = 99.0
        quiet["mark_b"] = 100.5
        out = xarb_paper.auto_manage_from_board(
            {
                "rows": [quiet],
                "funding_alerts": [],
                "thresholds": {"funding_alert_8h": 0.00025},
            }
        )
        self.assertEqual(len(out["closed"]), 1)
        book = xarb_paper.list_paper()
        self.assertEqual(book["open_count"], 0)
        self.assertEqual(book["closed_count"], 1)
        self.assertIsNotNone(book["trades"][0]["net_pnl_usd"])

    def test_find_row_case_insensitive(self):
        row = self._alert_row(base="btc")
        found = xarb_paper._find_row(
            [row],
            base="BTC",
            ex_long="backpack",
            ex_short="hyperliquid",
        )
        self.assertIsNotNone(found)

    def test_hours_held_naive_timestamp(self):
        naive = (xarb_paper._now_cst() - timedelta(hours=3)).replace(tzinfo=None).isoformat()
        h = xarb_paper._hours_held(naive)
        self.assertGreaterEqual(h, 2.9)
        self.assertLessEqual(h, 3.1)


class AlphaPaperAutoTests(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.path = Path(self._td.name) / "alpha_paper_trades.json"
        self._path_patch = mock.patch.object(alpha_paper, "_path", return_value=self.path)
        self._path_patch.start()
        os.environ["ALPHA_PAPER_AUTO"] = "1"
        os.environ["ALPHA_PAPER_SIZE_USD"] = "500"
        os.environ["ALPHA_PAPER_MAX_OPEN"] = "2"
        os.environ["ALPHA_PAPER_MAX_HOLD_HOURS"] = "12"

    def tearDown(self) -> None:
        self._path_patch.stop()
        self._td.cleanup()
        for k in (
            "ALPHA_PAPER_AUTO",
            "ALPHA_PAPER_SIZE_USD",
            "ALPHA_PAPER_MAX_OPEN",
            "ALPHA_PAPER_MAX_HOLD_HOURS",
        ):
            os.environ.pop(k, None)

    def test_auto_open_and_close_on_window(self):
        with mock.patch.object(
            alpha_paper,
            "fetch_symbol_prices",
            return_value=(1.2, 1.21, "binance_spot+fut"),
        ):
            out = alpha_paper.auto_manage_from_watches(
                [
                    {
                        "symbol": "ERA",
                        "coingecko_id": "caldera",
                        "aggregate": {"signal": "airdrop_dump"},
                    }
                ]
            )
            self.assertEqual(len(out["opened"]), 1)
            book = alpha_paper.list_paper()
            self.assertEqual(book["open_count"], 1)

            out2 = alpha_paper.auto_manage_from_watches(
                [
                    {
                        "symbol": "ERA",
                        "coingecko_id": "caldera",
                        "aggregate": {"signal": "quiet"},
                    }
                ]
            )
            self.assertEqual(len(out2["closed"]), 1)
            book2 = alpha_paper.list_paper()
            self.assertEqual(book2["open_count"], 0)
            self.assertEqual(book2["closed_count"], 1)

    def test_auto_disabled(self):
        os.environ["ALPHA_PAPER_AUTO"] = "0"
        out = alpha_paper.auto_manage_from_watches(
            [{"symbol": "ERA", "aggregate": {"signal": "airdrop_dump"}}]
        )
        self.assertFalse(out["enabled"])
        self.assertEqual(out["opened"], [])


if __name__ == "__main__":
    unittest.main()
