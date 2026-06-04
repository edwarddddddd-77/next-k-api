"""MOSS_ACTIVE_LANE 互斥。"""

from __future__ import annotations

import unittest


class TestMossLane(unittest.TestCase):
    def test_default_lane_moss2(self):
        import moss_lane as ml

        self.assertEqual(ml.active_moss_lane(), "moss2")
        self.assertTrue(ml.lane_allows_moss2())
        self.assertFalse(ml.lane_allows_moss_quant())

    def test_moss_quant_real_off_when_lane_moss2(self):
        import moss_quant.config as mq

        self.assertFalse(mq.MOSS_QUANT_REAL_MODE)

    def test_moss2_real_on_when_lane_moss2(self):
        import moss2.config as m2

        self.assertTrue(m2.MOSS2_REAL_MODE)


if __name__ == "__main__":
    unittest.main()
