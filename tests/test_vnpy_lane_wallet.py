"""vnpy lane 钱包持久化测试。"""

from __future__ import annotations

import sqlite3
import unittest

from quant.common.vnpy_wallet import (
    lane_equity_usdt,
    load_lane_wallet,
    migrate_vnpy_lane_tables,
    save_lane_wallet,
)
from quant.kama_trend.config import KamaTrendConfig


class TestVnpyLaneWallet(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.cur = self.conn.cursor()
        migrate_vnpy_lane_tables(self.cur)

    def tearDown(self) -> None:
        self.conn.close()

    def test_compound_wallet_load_save(self):
        save_lane_wallet(self.cur, lane="kama_trend", symbol="BTCUSDT", wallet=108.0, now_utc="t")
        self.conn.commit()
        wallet = load_lane_wallet(self.cur, lane="kama_trend", symbol="BTCUSDT", default=100.0)
        self.assertAlmostEqual(wallet, 108.0)

    def test_lane_equity_uses_wallet_when_compound(self):
        cfg = KamaTrendConfig(lane="kama_trend", equity_usdt=100.0, compound=True)
        save_lane_wallet(self.cur, lane="kama_trend", symbol="BTCUSDT", wallet=125.5, now_utc="t")
        self.conn.commit()
        self.assertAlmostEqual(lane_equity_usdt(cfg, "BTCUSDT", cur=self.cur), 125.5)

    def test_non_compound_keeps_base_equity(self):
        cfg = KamaTrendConfig(lane="kama_trend", equity_usdt=50.0, compound=False)
        save_lane_wallet(self.cur, lane="kama_trend", symbol="ETHUSDT", wallet=999.0, now_utc="t")
        self.conn.commit()
        self.assertEqual(lane_equity_usdt(cfg, "ETHUSDT", cur=self.cur), 50.0)


if __name__ == "__main__":
    unittest.main()
