"""ORB V2 robot pool tests."""

from __future__ import annotations

import sqlite3
import unittest

from orb.core.db import migrate_orb_tables
from orb.ml.gate import LiveGateDayState, rollback_open_decision
from orb.v2.robots import (
    init_robot_wallets,
    next_free_robot_id,
    next_robot_index,
    robot_equity_for_signals,
)
from orb.core.config import OrbConfig


class TestOrbV2Robots(unittest.TestCase):
    def test_init_robot_wallets(self):
        ws = init_robot_wallets(count=8, equity_usdt=10_000.0)
        self.assertEqual(len(ws), 8)
        self.assertEqual(ws[0], 10_000.0)

    def test_next_robot_index_skips_used_and_depleted(self):
        wallets = [10_000.0, 0.0, 10_000.0]
        self.assertEqual(next_robot_index(set(), wallets), 0)
        self.assertEqual(next_robot_index({0}, wallets), 2)
        self.assertIsNone(next_robot_index({0, 2}, [0.0, 0.0, 0.0]))

    def test_robot_equity_for_signals(self):
        cfg = OrbConfig.from_env()
        self.assertEqual(robot_equity_for_signals([1000, 2000, 0], cfg), 2000)

    def test_next_free_robot_id_from_db(self):
        conn = sqlite3.connect(":memory:")
        cur = conn.cursor()
        migrate_orb_tables(cur)
        from orb.v2.robots import ensure_orb_robots

        ensure_orb_robots(cur, count=2, initial_equity_usdt=10_000)
        cur.execute(
            """
            INSERT INTO orb_signals (
                recorded_at_utc, updated_at_utc, symbol, play, side, confidence,
                sl_price, robot_id, outcome
            ) VALUES ('t','t','AAAUSDT','p','LONG','h',1.0,1,NULL)
            """
        )
        self.assertEqual(next_free_robot_id(cur, count=2), 2)
        self.assertIsNone(next_free_robot_id(cur, count=1))

    def test_rollback_open_decision(self):
        st = LiveGateDayState(opens=1, opened=[{"symbol": "COINUSDT"}])
        rollback_open_decision(st, symbol="COINUSDT")
        self.assertEqual(st.opens, 0)
        self.assertEqual(st.opened, [])


if __name__ == "__main__":
    unittest.main()
