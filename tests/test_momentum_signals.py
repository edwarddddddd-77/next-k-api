"""动量信号解析单元测试。"""

from __future__ import annotations

import unittest

from momentum_signals import filter_symbol, pick_momentum_targets, to_binance_symbol


class TestMomentumSignals(unittest.TestCase):
    def test_to_binance_symbol(self):
        self.assertEqual(to_binance_symbol("LUMIA"), "LUMIAUSDT")
        self.assertEqual(to_binance_symbol("LUMIAUSDT"), "LUMIAUSDT")
        self.assertEqual(to_binance_symbol("btc/usdt:usdt"), "BTCUSDT")

    def test_filter_blacklist(self):
        self.assertFalse(filter_symbol("XNYUSDT", blacklist=("XNY",), allow_usdc=False))
        self.assertFalse(filter_symbol("ESPORTSUSDT", blacklist=("ESPORTS",), allow_usdc=False))
        self.assertTrue(filter_symbol("BTCUSDT", blacklist=("XNY", "ESPORTS"), allow_usdc=False))

    def test_filter_usdc(self):
        self.assertFalse(filter_symbol("BTCUSDC", allow_usdc=False))
        self.assertTrue(filter_symbol("BTCUSDC", allow_usdc=True))

    def test_pick_targets_latest_event(self):
        movers = [
            {
                "symbol": "AAA",
                "eventType": "PULLBACK",
                "createTimestamp": 100,
            },
            {
                "symbol": "BBB",
                "eventType": "PULLBACK",
                "createTimestamp": 200,
            },
            {
                "symbol": "CCC",
                "eventType": "RALLY",
                "createTimestamp": 150,
            },
            {
                "symbol": "XNY",
                "eventType": "PULLBACK",
                "createTimestamp": 999,
            },
        ]
        long_sym, short_sym, meta = pick_momentum_targets(
            movers, blacklist=("XNY",), allow_usdc=False
        )
        self.assertEqual(long_sym, "BBBUSDT")
        self.assertEqual(short_sym, "CCCUSDT")
        self.assertEqual(meta["movers_valid"], 3)


if __name__ == "__main__":
    unittest.main()
