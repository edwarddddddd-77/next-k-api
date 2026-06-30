"""ORB V2 robot pool tests."""

from __future__ import annotations

import os
import sqlite3
import unittest
from unittest.mock import patch

from orb.core.config import OrbConfig
from orb.core.db import migrate_orb_tables
from orb.ml.gate import LiveGateDayState, rollback_open_decision
from orb.ml.gate import LiveGateConfig
from orb.v2.robots import (
    apply_robot_wallet_after_pnl,
    bound_robot_index_available,
    ensure_orb_robots,
    init_robot_wallets,
    next_free_robot_id,
    next_robot_index,
    resolve_robot_pool_size,
    robot_bound_mode,
    robot_equity_for_signals,
    robot_symbol_bindings,
    symbol_to_robot_id,
)


class TestOrbV2Robots(unittest.TestCase):
    @patch.dict(os.environ, {"ORB_V2_ROBOT_BOUND": ""}, clear=False)
    def test_robot_bound_mode_default(self):
        self.assertTrue(robot_bound_mode(symbol_count=8, robot_count=8))
        self.assertFalse(robot_bound_mode(symbol_count=8, robot_count=4))
        self.assertFalse(robot_bound_mode(symbol_count=25, robot_count=8))

    @patch.dict(os.environ, {"ORB_V2_ROBOT_BOUND": "1"}, clear=False)
    def test_robot_bound_explicit_requires_matching_counts(self):
        self.assertTrue(robot_bound_mode(symbol_count=8, robot_count=8))
        self.assertFalse(robot_bound_mode(symbol_count=25, robot_count=8))

    def test_robot_symbol_bindings(self):
        syms = ["COINUSDT", "HOODUSDT", "PLTRUSDT"]
        self.assertEqual(robot_symbol_bindings(syms), {1: "COINUSDT", 2: "HOODUSDT", 3: "PLTRUSDT"})
        self.assertEqual(symbol_to_robot_id("HOOD", syms), 2)
        self.assertIsNone(symbol_to_robot_id("TSLAUSDT", syms))

    def test_bound_robot_index_available(self):
        wallets = [1000.0, 1000.0]
        syms = ["COINUSDT", "HOODUSDT"]
        busy = {0: {"symbol": "COINUSDT", "exit_ms": 1, "pnl_usdt": 0}}
        self.assertIsNone(bound_robot_index_available(busy, wallets, "COINUSDT", syms))
        self.assertEqual(bound_robot_index_available(busy, wallets, "HOODUSDT", syms), 1)

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

    def test_apply_robot_wallet_after_pnl_compounds(self):
        self.assertAlmostEqual(apply_robot_wallet_after_pnl(2000.0, 100.0), 2100.0)
        self.assertAlmostEqual(apply_robot_wallet_after_pnl(2400.0, 200.0), 2600.0)

    def test_next_free_robot_id_from_db(self):
        conn = sqlite3.connect(":memory:")
        cur = conn.cursor()
        migrate_orb_tables(cur)
        ensure_orb_robots(cur, count=2, initial_equity_usdt=10_000)
        cur.execute(
            """
            INSERT INTO orb_robots(robot_id, initial_equity_usdt, enabled, created_at_utc, updated_at_utc)
            VALUES (9, 10_000, 1, 't', 't')
            """
        )
        ensure_orb_robots(cur, count=2, initial_equity_usdt=10_000)
        cur.execute("SELECT enabled FROM orb_robots WHERE robot_id=9")
        self.assertEqual(int(cur.fetchone()[0]), 0)
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

    @patch.dict(os.environ, {"ORB_V2_ROBOT_COUNT": "3", "ORB_V2_ROBOT_BOUND": "0"}, clear=False)
    def test_resolve_robot_pool_size_uses_gate_over_env(self):
        gate = LiveGateConfig(max_opens_per_day=8, robot_pool_size=8)
        self.assertEqual(resolve_robot_pool_size(gate=gate, symbol_count=25), 8)

    def test_ensure_orb_robots_reenables_after_pool_grows(self):
        conn = sqlite3.connect(":memory:")
        cur = conn.cursor()
        migrate_orb_tables(cur)
        ensure_orb_robots(cur, count=3, initial_equity_usdt=14)
        ensure_orb_robots(cur, count=8, initial_equity_usdt=14)
        cur.execute("SELECT COUNT(*) FROM orb_robots WHERE enabled=1")
        self.assertEqual(int(cur.fetchone()[0]), 8)
        cur.execute("SELECT enabled FROM orb_robots WHERE robot_id=8")
        self.assertEqual(int(cur.fetchone()[0]), 1)


if __name__ == "__main__":
    unittest.main()
