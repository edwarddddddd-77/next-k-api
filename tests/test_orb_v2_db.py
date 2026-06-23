"""ORB V2 breakout_seen / session lock tests."""

from __future__ import annotations

import sqlite3
import unittest

from orb.v2.db import (
    breakout_opened_today,
    breakout_seen_today,
    mark_breakout_seen,
    migrate_orb_v2_tables,
    rollback_breakout_opened,
)
from orb.v2.gate_state import v2_session_traded
from orb.v2.config import OrbV2Config


class TestOrbV2BreakoutSeen(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        cur = self.conn.cursor()
        migrate_orb_v2_tables(cur)
        self.conn.commit()
        self.cur = self.conn.cursor()
        self.day = "2026-06-15"
        self.cfg = OrbV2Config(base=__import__("orb.core.config", fromlist=["OrbConfig"]).OrbConfig.from_env())

    def test_gate_reject_does_not_lock_session(self) -> None:
        mark_breakout_seen(
            self.cur,
            session_date=self.day,
            symbol="TSLAUSDT",
            now_utc="t",
            scan_open_ms=1,
            p_true=0.4,
            opened=False,
            reason="early_sync_trap",
        )
        self.conn.commit()
        self.assertTrue(breakout_seen_today(self.cur, "TSLAUSDT", self.day))
        self.assertFalse(breakout_opened_today(self.cur, "TSLAUSDT", self.day))
        self.assertFalse(v2_session_traded(self.cur, "TSLAUSDT", self.day, self.cfg))

    def test_opened_locks_session(self) -> None:
        mark_breakout_seen(
            self.cur,
            session_date=self.day,
            symbol="COINUSDT",
            now_utc="t",
            scan_open_ms=2,
            p_true=0.6,
            opened=True,
            reason="open_ok",
        )
        self.conn.commit()
        self.assertTrue(breakout_opened_today(self.cur, "COINUSDT", self.day))
        self.assertTrue(v2_session_traded(self.cur, "COINUSDT", self.day, self.cfg))

    def test_gate_reject_then_open_upgrades_opened_flag(self) -> None:
        mark_breakout_seen(
            self.cur,
            session_date=self.day,
            symbol="USARUSDT",
            now_utc="t1",
            scan_open_ms=1,
            p_true=0.4,
            opened=False,
            reason="early_sync_trap",
        )
        mark_breakout_seen(
            self.cur,
            session_date=self.day,
            symbol="USARUSDT",
            now_utc="t2",
            scan_open_ms=2,
            p_true=0.55,
            opened=True,
            reason="open_ok",
        )
        self.conn.commit()
        self.assertTrue(breakout_opened_today(self.cur, "USARUSDT", self.day))
        self.assertTrue(v2_session_traded(self.cur, "USARUSDT", self.day, self.cfg))

    def test_opened_not_downgraded_by_later_gate_reject(self) -> None:
        mark_breakout_seen(
            self.cur,
            session_date=self.day,
            symbol="NVDAUSDT",
            now_utc="t1",
            scan_open_ms=1,
            p_true=0.6,
            opened=True,
            reason="open_ok",
        )
        mark_breakout_seen(
            self.cur,
            session_date=self.day,
            symbol="NVDAUSDT",
            now_utc="t2",
            scan_open_ms=2,
            p_true=0.4,
            opened=False,
            reason="early_sync_trap",
        )
        self.conn.commit()
        self.assertTrue(breakout_opened_today(self.cur, "NVDAUSDT", self.day))
        self.assertTrue(v2_session_traded(self.cur, "NVDAUSDT", self.day, self.cfg))

    def test_rollback_breakout_opened_unlocks_session(self) -> None:
        mark_breakout_seen(
            self.cur,
            session_date=self.day,
            symbol="USARUSDT",
            now_utc="t1",
            scan_open_ms=1,
            p_true=0.55,
            opened=True,
            reason="open_ok",
        )
        self.assertTrue(breakout_opened_today(self.cur, "USARUSDT", self.day))
        self.assertTrue(
            rollback_breakout_opened(self.cur, self.day, "USARUSDT", reason="live_open_failed")
        )
        self.conn.commit()
        self.assertFalse(breakout_opened_today(self.cur, "USARUSDT", self.day))
        self.assertFalse(v2_session_traded(self.cur, "USARUSDT", self.day, self.cfg))


if __name__ == "__main__":
    unittest.main()
