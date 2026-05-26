"""接针严选 jz_universe v2 单元测试。"""

from __future__ import annotations

import json
import sqlite3
import unittest
from unittest.mock import patch

from jiezhen_universe import migrate_jz_universe_table, refresh_jiezhen_universe


class TestJiezhenUniverseV2(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        c = self.conn.cursor()
        migrate_jz_universe_table(c)
        c.execute(
            """
            CREATE TABLE worth_watch_patrick_core (
                symbol TEXT PRIMARY KEY,
                coin TEXT,
                generated_date TEXT,
                last_seen_cst TEXT,
                rank_in_category INTEGER,
                summary_line TEXT,
                detail_json TEXT
            )
            """
        )
        c.execute(
            """
            CREATE TABLE worth_watch_heat_accum (
                symbol TEXT PRIMARY KEY,
                coin TEXT,
                generated_date TEXT,
                last_seen_cst TEXT,
                rank_in_category INTEGER,
                summary_line TEXT,
                detail_json TEXT
            )
            """
        )
        c.execute(
            """
            CREATE TABLE worth_watch_hot_oi (
                symbol TEXT PRIMARY KEY,
                coin TEXT,
                generated_date TEXT,
                last_seen_cst TEXT,
                rank_in_category INTEGER,
                summary_line TEXT,
                detail_json TEXT
            )
            """
        )
        c.execute(
            """
            CREATE TABLE patrick_core_watch (
                symbol TEXT PRIMARY KEY,
                coin TEXT,
                generated_date TEXT,
                last_seen_cst TEXT,
                d6h REAL,
                px_chg REAL,
                est_mcap REAL,
                sideways_days INTEGER,
                summary_line TEXT
            )
            """
        )
        patrick_ok = json.dumps({"d6h": 10.0, "px_chg": 2.0, "sw_days": 5})
        patrick_bad = json.dumps({"d6h": 30.0, "px_chg": 2.0})
        heat_ok = json.dumps({"heat": 8, "d6h": 8.0, "px_chg": 1.0, "sw_days": 4})
        hot_only = json.dumps({"heat": 20, "d6h": 15.0, "px_chg": 1.0})
        stale_worth_only = json.dumps({"d6h": 11.0, "px_chg": 1.0})

        c.execute(
            """
            INSERT INTO worth_watch_patrick_core VALUES
            ('AAAUSDT', 'AAA', 'd', 't', 1, 'patrick', ?),
            ('BBBUSDT', 'BBB', 'd', 't', 1, 'bad', ?),
            ('STALEUSDT', 'ST', 'd', 't', 1, 'stale worth only', ?)
            """,
            (patrick_ok, patrick_bad, stale_worth_only),
        )
        c.execute(
            """
            INSERT INTO worth_watch_heat_accum VALUES
            ('CCCUSDT', 'CCC', 'd', 't', 1, 'heat', ?)
            """,
            (heat_ok,),
        )
        c.execute(
            """
            INSERT INTO worth_watch_hot_oi VALUES
            ('HOTONLYUSDT', 'H', 'd', 't', 1, 'hot only', ?)
            """,
            (hot_only,),
        )
        c.execute(
            """
            INSERT INTO patrick_core_watch VALUES
            ('AAAUSDT', 'AAA', 'd', 't', 10, 2, 1e8, 5, 'pat'),
            ('BBBUSDT', 'BBB', 'd', 't', 30, 2, 1e8, 0, 'bad core')
            """
        )
        self.conn.commit()

    def tearDown(self) -> None:
        self.conn.close()

    @patch(
        "jiezhen_universe.filter_symbols_to_binance_usdt_perps",
        side_effect=lambda xs: list(xs),
    )
    def test_v2_primary_sources_and_core_row_gate(self, _mock_perp: object) -> None:
        import jiezhen_config as cfg

        orig_max = cfg.JIEZHEN_UNIVERSE_MAX
        try:
            cfg.JIEZHEN_UNIVERSE_MAX = 10
            meta = refresh_jiezhen_universe(self.conn)
        finally:
            cfg.JIEZHEN_UNIVERSE_MAX = orig_max

        syms = meta.get("symbols") or []
        self.assertEqual(meta.get("scheme"), "curated_v2")
        self.assertIn("AAAUSDT", syms)
        self.assertIn("CCCUSDT", syms)
        self.assertNotIn("BBBUSDT", syms)
        self.assertNotIn("HOTONLYUSDT", syms)
        self.assertNotIn("STALEUSDT", syms)


if __name__ == "__main__":
    unittest.main()
