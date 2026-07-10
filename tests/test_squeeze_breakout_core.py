"""Smart Breakout 核心逻辑测试。"""

from __future__ import annotations

import unittest

from quant.squeeze_breakout.core import (
    BreakoutEngineState,
    RangeBox,
    bar_exit_reason,
    breakout_levels,
    process_signal_bar,
    scan_range_breakouts,
)


class TestSqueezeBreakoutCore(unittest.TestCase):
    def _bars(self, n: int, start: float = 100.0, step: float = 0.05, vol: float = 1000.0):
        out = []
        for i in range(n):
            c = start + i * step
            ts = i * 900_000
            out.append((ts, c - 0.1, c + 0.2, c - 0.2, c, vol))
        return out

    def test_breakout_levels_long(self):
        stop, tp1, tp2, tp3, risk = breakout_levels(
            105.0,
            1,
            range_top=104.0,
            range_bottom=100.0,
            atr=1.0,
            sl_atr_buffer=0.5,
            tp1_rr=1.0,
            tp2_rr=2.0,
            tp3_rr=3.0,
        )
        self.assertAlmostEqual(stop, 99.5)
        self.assertAlmostEqual(risk, 5.5)
        self.assertAlmostEqual(tp3, 105.0 + 5.5 * 3)

    def test_stale_breakout_consumes_range(self):
        rng = RangeBox(top=101.0, bottom=99.0, squeeze_bars=8, start_ts=0, end_ts=0)
        kept, sig = scan_range_breakouts(
            close=102.0,
            open_=101.5,
            ranges=[rng],
            atr=1.0,
            impulse_mult=0.8,
            squeeze_bars_for_strength=5,
            min_squeeze_bars=5,
            sl_atr_buffer=0.5,
            tp1_rr=1.0,
            tp2_rr=2.0,
            tp3_rr=3.0,
        )
        self.assertIsNone(sig)
        self.assertEqual(kept, [])

    def test_qualified_breakout_long(self):
        rng = RangeBox(top=100.5, bottom=99.0, squeeze_bars=10, start_ts=0, end_ts=0)
        kept, sig = scan_range_breakouts(
            close=101.5,
            open_=100.0,
            ranges=[rng],
            atr=1.0,
            impulse_mult=0.8,
            squeeze_bars_for_strength=5,
            min_squeeze_bars=5,
            sl_atr_buffer=0.5,
            tp1_rr=1.0,
            tp2_rr=2.0,
            tp3_rr=3.0,
        )
        self.assertIsNotNone(sig)
        self.assertEqual(sig.side, 1)
        self.assertEqual(kept, [])

    def test_bar_exit_same_bar_ambiguity_is_sl(self):
        reason = bar_exit_reason(
            side=1,
            high=110.0,
            low=95.0,
            stop=96.0,
            tp1=102.0,
            tp2=105.0,
            tp3=108.0,
            prev_high=100.0,
            prev_low=100.0,
        )
        self.assertEqual(reason, "sl")

    def test_bar_tp_milestones_first_touch(self):
        from quant.squeeze_breakout.core import bar_tp_milestones

        hits = bar_tp_milestones(
            side=1,
            high=103.0,
            low=100.0,
            tp1=102.0,
            tp2=105.0,
            prev_high=101.0,
            prev_low=100.0,
            tp1_hit=False,
            tp2_hit=False,
        )
        self.assertEqual(hits, ["tp1"])

    def test_process_signal_bar_runs(self):
        bars = self._bars(120)
        state = BreakoutEngineState()
        for i in range(60, len(bars)):
            state, sig = process_signal_bar(bars[: i + 1], state)
        self.assertIsInstance(state, BreakoutEngineState)


    def test_volume_filter_blocks_breakout(self):
        rng = RangeBox(top=100.5, bottom=99.0, squeeze_bars=10, start_ts=0, end_ts=0)
        kept, sig = scan_range_breakouts(
            close=101.5,
            open_=100.0,
            ranges=[rng],
            atr=1.0,
            impulse_mult=0.8,
            squeeze_bars_for_strength=5,
            min_squeeze_bars=5,
            sl_atr_buffer=0.5,
            tp1_rr=1.0,
            tp2_rr=2.0,
            tp3_rr=3.0,
            volume=100.0,
            vol_sma=200.0,
            volume_filter=True,
            volume_mult=1.5,
        )
        self.assertIsNone(sig)
        self.assertEqual(kept, [])

    def test_strength_counts_volume_without_filter_toggle(self):
        from quant.squeeze_breakout.core import _signal_strength

        score = _signal_strength(
            body=2.0,
            atr=1.0,
            impulse_mult=0.8,
            squeeze_bars=12,
            min_squeeze_bars=5,
            volume=500.0,
            vol_sma=100.0,
            volume_filter=False,
            volume_mult=1.5,
        )
        self.assertGreaterEqual(score, 3)

    def test_replay_engine_state(self):
        from quant.squeeze_breakout.core import replay_engine_state

        bars = self._bars(120)
        state = replay_engine_state(bars)
        self.assertIsInstance(state, BreakoutEngineState)


if __name__ == "__main__":
    unittest.main()
