"""ORB V2 robot pool tests."""

from __future__ import annotations

import sqlite3
import unittest

from orb.core.db import migrate_orb_tables
from orb.ml.gate import LiveGateDayState, rollback_open_decision
from orb.v2.robots import (
    apply_robot_wallet_after_pnl,
    init_robot_wallets,
    maybe_reset_robot_wallet_after_settle,
    next_free_robot_id,
    next_robot_index,
    robot_equity_for_signals,
    robot_reset_policy,
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

    def test_robot_reset_policy_defaults(self):
        policy = robot_reset_policy()
        self.assertEqual(policy["cap_usdt"], 2500.0)
        self.assertEqual(policy["floor_usdt"], 1500.0)
        self.assertTrue(policy["enabled"])

    def test_apply_robot_wallet_after_pnl_no_reset(self):
        balance, evt = apply_robot_wallet_after_pnl(2000.0, 100.0)
        self.assertAlmostEqual(balance, 2100.0)
        self.assertIsNone(evt)

    def test_apply_robot_wallet_after_pnl_triggers_reset(self):
        balance, evt = apply_robot_wallet_after_pnl(2400.0, 200.0)
        self.assertAlmostEqual(balance, 1500.0)
        self.assertIsNotNone(evt)
        self.assertAlmostEqual(evt["balance_before"], 2600.0)
        self.assertAlmostEqual(evt["withdrawn_usdt"], 1100.0)
        self.assertAlmostEqual(evt["balance_after"], 1500.0)

    def test_apply_robot_wallet_after_pnl_at_cap(self):
        balance, evt = apply_robot_wallet_after_pnl(2500.0, 0.0)
        self.assertAlmostEqual(balance, 1500.0)
        self.assertIsNotNone(evt)
        self.assertAlmostEqual(evt["withdrawn_usdt"], 1000.0)

    def test_maybe_reset_robot_wallet_after_settle_db(self):
        conn = sqlite3.connect(":memory:")
        cur = conn.cursor()
        migrate_orb_tables(cur)
        from orb.v2.robots import ensure_orb_robots

        ensure_orb_robots(cur, count=1, initial_equity_usdt=1000.0)
        cur.execute(
            """
            INSERT INTO orb_settlements (
                settled_at_utc, signal_id, symbol, side, play, outcome,
                entry_price, exit_price, pnl_r, pnl_usdt, virtual_notional_usdt,
                exit_rule, session_date, robot_id
            ) VALUES ('t', 1, 'AAAUSDT', 'LONG', 'p', 'win', 1, 2, 1, 1600, 1000, 'tp', '2026-06-01', 1)
            """
        )
        conn.commit()
        evt = maybe_reset_robot_wallet_after_settle(conn, 1, trigger_signal_id=1, session_date="2026-06-01")
        self.assertIsNotNone(evt)
        self.assertAlmostEqual(evt["balance_before"], 2600.0)
        self.assertAlmostEqual(evt["withdrawn_usdt"], 1100.0)
        cur.execute("SELECT COALESCE(SUM(pnl_usdt), 0) FROM orb_settlements WHERE robot_id = 1")
        self.assertAlmostEqual(float(cur.fetchone()[0]), 500.0)
        from orb.v2.robots import robot_wallet_balance

        self.assertAlmostEqual(robot_wallet_balance(conn, 1, initial_equity_usdt=1000.0, sync=False), 1500.0)


if __name__ == "__main__":
    unittest.main()
