# -*- coding: utf-8 -*-
"""Alpha 纸面自动开平单元测试。"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import alpha_paper


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
