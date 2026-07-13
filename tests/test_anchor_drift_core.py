"""Anchor Drift 信号逻辑测试。"""

from __future__ import annotations

import unittest

from quant.anchor_drift.core import (
    adverse_drift_stop,
    calculate_drift,
    generate_signal,
)


class TestAnchorDriftCore(unittest.TestCase):
    def test_calculate_drift(self):
        self.assertAlmostEqual(calculate_drift(anchor_price=100.0, current_price=102.0), 0.02)

    def test_signal_short_on_positive_drift(self):
        sig = generate_signal(0.02, signal_threshold=0.015, converge_threshold=0.003)
        self.assertEqual(sig.signal, "SHORT")

    def test_signal_long_on_negative_drift(self):
        sig = generate_signal(-0.02, signal_threshold=0.015, converge_threshold=0.003)
        self.assertEqual(sig.signal, "LONG")

    def test_converged(self):
        sig = generate_signal(0.002, signal_threshold=0.015, converge_threshold=0.003)
        self.assertEqual(sig.signal, "CONVERGED")

    def test_flat_between_thresholds(self):
        sig = generate_signal(0.01, signal_threshold=0.015, converge_threshold=0.003)
        self.assertEqual(sig.signal, "FLAT")

    def test_adverse_drift_stop_short(self):
        self.assertTrue(
            adverse_drift_stop(
                0.05,
                side=-1,
                signal_threshold=0.015,
                max_adverse_extension=0.025,
            )
        )

    def test_adverse_drift_stop_long(self):
        self.assertTrue(
            adverse_drift_stop(
                -0.05,
                side=1,
                signal_threshold=0.015,
                max_adverse_extension=0.025,
            )
        )


if __name__ == "__main__":
    unittest.main()
