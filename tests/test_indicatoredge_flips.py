# -*- coding: utf-8 -*-
"""IndicatorEdge Just flipped 解析测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from utils import indicatoredge as ie


SAMPLE = """
Live · updated 2026-07-16
<span>Just flipped</span><h2>Latest signal changes</h2></div>
<div class="flip-row">
<a class="flip-chip to-flat" href="/assets/bch">Bitcoin Cash (BCH) → FLAT</a>
<a class="flip-chip to-long" href="/assets/xrp">XRP (XRP) → LONG</a>
<a class="flip-chip to-long" href="/assets/crv">Curve → LONG</a>
<a class="flip-chip to-short" href="/assets/foo">Foo (FOO) → SHORT</a>
</div>
"""


class ScreenerFlipsTests(unittest.TestCase):
    def test_parse_screener_flips(self):
        out = ie.parse_screener_flips(SAMPLE)
        self.assertEqual(out["source_updated"], "2026-07-16")
        self.assertEqual(out["count"], 4)
        self.assertEqual(out["counts"]["LONG"], 2)
        self.assertEqual(out["counts"]["FLAT"], 1)
        self.assertEqual(out["counts"]["SHORT"], 1)
        by_slug = {x["slug"]: x for x in out["flips"]}
        self.assertEqual(by_slug["xrp"]["signal"], "LONG")
        self.assertEqual(by_slug["xrp"]["ticker"], "XRP")
        self.assertEqual(by_slug["crv"]["signal"], "LONG")
        self.assertEqual(by_slug["bch"]["signal"], "FLAT")

    def test_refresh_detects_new(self):
        td = tempfile.TemporaryDirectory()
        path = Path(td.name) / "indicatoredge_flips_snapshot.json"
        with mock.patch.object(ie, "_flips_path", return_value=path):
            with mock.patch.object(
                ie,
                "fetch_screener_flips",
                return_value={
                    **ie.parse_screener_flips(SAMPLE),
                    "fetched_at_cst": "2026-07-17T10:00:00+08:00",
                },
            ):
                first = ie.refresh_screener_flips(force=True)
                self.assertEqual(first["new_count"], 0)
                self.assertEqual(first["count"], 4)

            sample2 = SAMPLE.replace("XRP (XRP) → LONG", "Solana (SOL) → LONG").replace(
                'href="/assets/xrp"', 'href="/assets/sol"'
            )
            with mock.patch.object(
                ie,
                "fetch_screener_flips",
                return_value={
                    **ie.parse_screener_flips(sample2),
                    "fetched_at_cst": "2026-07-17T11:00:00+08:00",
                },
            ):
                second = ie.refresh_screener_flips(force=True)
                keys = {x["key"] for x in second["new_flips"]}
                self.assertIn("sol:LONG", keys)
                self.assertTrue(second["new_count"] >= 1)
        td.cleanup()


if __name__ == "__main__":
    unittest.main()
