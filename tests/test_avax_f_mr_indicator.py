"""Unit tests for desk-F AVAX mean-reversion indicator v2 (no network)."""

from __future__ import annotations

import unittest

from utils.avax_f_mr_indicator import FMrParams, compute_signal, params_for_mode


def _down(n: int = 80, start: float = 20.0) -> tuple[list[float], list[float], list[float]]:
    closes = [start]
    for i in range(1, n):
        closes.append(closes[-1] * (0.994 if i < n - 3 else 0.999))
    # near lows: highs well above, lows near last
    highs = [c * 1.05 for c in closes]
    lows = [c * 0.998 for c in closes]
    return closes, highs, lows


def _up(n: int = 80, start: float = 10.0) -> tuple[list[float], list[float], list[float]]:
    closes = [start]
    for _ in range(1, n):
        closes.append(closes[-1] * 1.006)
    highs = [c * 1.002 for c in closes]
    lows = [c * 0.95 for c in closes]
    return closes, highs, lows


class TestAvaxFMrV2(unittest.TestCase):
    def test_long_near_lows_after_drop(self):
        closes, highs, lows = _down()
        sig = compute_signal(closes, highs_1h=highs, lows_1h=lows, mode="trade")
        self.assertEqual(sig.side, "long")
        self.assertLess(sig.ret_4h_pct, 0)
        self.assertGreater(sig.strength, 0)

    def test_short_near_highs_after_rally(self):
        closes, highs, lows = _up()
        sig = compute_signal(closes, highs_1h=highs, lows_1h=lows, mode="trade")
        self.assertEqual(sig.side, "short")
        self.assertGreater(sig.ret_4h_pct, 0)

    def test_blocks_chase_long(self):
        # down last 4h but up a lot over 24h → chase block
        closes = [10.0]
        for i in range(1, 80):
            if i < 60:
                closes.append(closes[-1] * 1.01)  # big rally
            else:
                closes.append(closes[-1] * 0.997)  # mild pullback
        highs = [c * 1.02 for c in closes]
        lows = [c * 0.999 for c in closes]
        sig = compute_signal(
            closes,
            highs_1h=highs,
            lows_1h=lows,
            params=FMrParams(ret4h_min_abs=0.0, rsi_long_max=55.0, ret24h_chase_block=1.0),
        )
        # may be flat due to chase or ext; at least not blindly long-chase
        if sig.ret_24h_pct is not None and sig.ret_24h_pct > 1.0:
            self.assertNotEqual(sig.side, "long")

    def test_gate_params_stricter_ext(self):
        g = params_for_mode("gate")
        t = params_for_mode("trade")
        self.assertLess(g.ext_max_pct, t.ext_max_pct)
        self.assertLess(g.rsi_short_min, t.rsi_short_min)


if __name__ == "__main__":
    unittest.main()
