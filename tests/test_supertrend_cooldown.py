"""Supertrend 冷却：同向挡、反向放行。"""

from __future__ import annotations

import sqlite3
import unittest

from supertrend_db import (
    cooldown_blocks_entry,
    migrate_st_tables,
    upsert_symbol_cooldown,
)


class TestStCooldown(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        cur = self.conn.cursor()
        migrate_st_tables(cur)
        self.conn.commit()
        self.cur = self.conn.cursor()

    def tearDown(self) -> None:
        self.conn.close()

    def test_loss_cooldown_blocks_same_side_only(self) -> None:
        upsert_symbol_cooldown(
            self.cur,
            symbol="BTCUSDT",
            until_bar_open_ms=0,
            until_utc_ms=9_999_999_999_999,
            reason="loss_cooldown",
            updated_at_utc="2026-01-01T00:00:00Z",
            blocked_side="LONG",
        )
        self.assertIsNotNone(
            cooldown_blocks_entry(
                self.cur,
                "BTCUSDT",
                bar_open_ms=1_000_000,
                now_utc_ms=1_000,
                entry_side="LONG",
            )
        )
        self.assertIsNone(
            cooldown_blocks_entry(
                self.cur,
                "BTCUSDT",
                bar_open_ms=1_000_000,
                now_utc_ms=1_000,
                entry_side="SHORT",
            )
        )


if __name__ == "__main__":
    unittest.main()
