"""maker/taker 分拆手续费。"""

from __future__ import annotations

import unittest

from quant.common.fees import entry_fee_bps, trade_fee_usdt


class TestOrbFees(unittest.TestCase):
    def test_fvg_prox_maker_open_taker_close(self):
        self.assertEqual(entry_fee_bps("fvg_prox", maker_bps=2.0, taker_bps=4.0), 2.0)
        # 500 × (2+4)/10000 = 0.3
        self.assertAlmostEqual(
            trade_fee_usdt(500.0, entry_mode="fvg_prox", maker_bps=2.0, taker_bps=4.0),
            0.3,
        )

    def test_signal_both_taker(self):
        self.assertEqual(entry_fee_bps("signal", maker_bps=2.0, taker_bps=4.0), 4.0)
        self.assertAlmostEqual(
            trade_fee_usdt(500.0, entry_mode="signal", maker_bps=2.0, taker_bps=4.0),
            0.4,
        )

    def test_legacy_uniform_override(self):
        self.assertEqual(trade_fee_usdt(500.0, fee_bps_per_side=4.0), 0.4)

    def test_scales_with_notional(self):
        small = trade_fee_usdt(100.0, entry_mode="fvg_prox", maker_bps=2.0, taker_bps=4.0)
        large = trade_fee_usdt(700.0, entry_mode="fvg_prox", maker_bps=2.0, taker_bps=4.0)
        self.assertAlmostEqual(large / small, 7.0, places=6)


if __name__ == "__main__":
    unittest.main()
