import unittest

from quant.common.fees import trade_fee_usdt


class TestSimLiveFee(unittest.TestCase):
    def test_trade_fee_round_trip_uniform_legacy(self):
        self.assertEqual(trade_fee_usdt(500.0, fee_bps_per_side=4.0), 0.4)
        self.assertEqual(trade_fee_usdt(0.0, fee_bps_per_side=4.0), 0.0)

    def test_scales_with_notional(self):
        small = trade_fee_usdt(100.0, fee_bps_per_side=4.0)
        large = trade_fee_usdt(700.0, fee_bps_per_side=4.0)
        self.assertAlmostEqual(large / small, 7.0, places=6)


if __name__ == "__main__":
    unittest.main()
