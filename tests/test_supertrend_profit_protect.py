"""Supertrend 利润保护单元测试。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import supertrend_config as cfg
from supertrend_profit_protect import (
    ProtectState,
    evaluate_profit_exit,
    profit_exit_fill_price,
    update_protect_state,
)


class TestProfitProtect(unittest.TestCase):
    def test_trail_long_triggered_by_low_not_close(self):
        with patch.multiple(
            cfg,
            ST_EXIT_MODE="trail_atr",
            ST_TRAIL_ATR_MULT=2.0,
            ST_TRAIL_ARM_ATR=0.5,
        ):
            entry, atr = 100.0, 2.0
            st = update_protect_state(
                ProtectState(entry, 0.0, False),
                side="LONG",
                entry=entry,
                high=105.0,
                low=99.0,
                close=104.0,
                atr=atr,
            )
            self.assertEqual(st.trail_stop, 101.0)
            rule = evaluate_profit_exit(
                st,
                side="LONG",
                entry=entry,
                high=104.0,
                low=100.5,
                close=103.0,
                atr=atr,
            )
            self.assertEqual(rule, "trail_atr")
            fill = profit_exit_fill_price(
                "trail_atr", side="LONG", close=103.0, high=104.0, low=100.5, state=st
            )
            self.assertEqual(fill, 101.0)

    def test_giveback_uses_close_peak_only_by_default(self):
        with patch.multiple(
            cfg,
            ST_EXIT_MODE="giveback",
            ST_GIVEBACK_PCT=0.85,
            ST_GIVEBACK_MIN_PEAK_PCT=0.03,
            ST_GIVEBACK_PEAK_USE_CLOSE=True,
            ST_GIVEBACK_REQUIRE_POSITIVE_PCT=True,
        ):
            st = ProtectState(mfe_price=105.0, peak_pnl_pct=0.04, trail_armed=False)
            rule = evaluate_profit_exit(
                st,
                side="LONG",
                entry=100.0,
                high=105.0,
                low=99.0,
                close=100.5,
                atr=2.0,
            )
            self.assertEqual(rule, "giveback")

    def test_giveback_skips_when_not_positive(self):
        with patch.multiple(
            cfg,
            ST_EXIT_MODE="giveback",
            ST_GIVEBACK_PCT=0.85,
            ST_GIVEBACK_MIN_PEAK_PCT=0.03,
            ST_GIVEBACK_REQUIRE_POSITIVE_PCT=True,
        ):
            st = ProtectState(mfe_price=102.0, peak_pnl_pct=0.04, trail_armed=False)
            rule = evaluate_profit_exit(
                st,
                side="LONG",
                entry=100.0,
                high=101.0,
                low=98.0,
                close=99.0,
                atr=2.0,
            )
            self.assertIsNone(rule)

    def test_trail_arm_requires_close_when_flag_on(self):
        with patch.multiple(
            cfg,
            ST_EXIT_MODE="trail_atr",
            ST_TRAIL_ATR_MULT=2.0,
            ST_TRAIL_ARM_ATR=1.0,
            ST_TRAIL_ARM_USE_CLOSE=True,
        ):
            entry, atr = 100.0, 2.0
            st = update_protect_state(
                ProtectState(entry, 0.0, False),
                side="LONG",
                entry=entry,
                high=102.5,
                low=99.0,
                close=100.5,
                atr=atr,
            )
            self.assertFalse(st.trail_armed)

    def test_update_peak_when_atr_zero(self):
        with patch.multiple(cfg, ST_GIVEBACK_PEAK_USE_CLOSE=True):
            st = update_protect_state(
                ProtectState(100.0, 0.0, False),
                side="LONG",
                entry=100.0,
                high=101.0,
                low=99.5,
                close=100.8,
                atr=0.0,
            )
            self.assertGreater(st.peak_pnl_pct, 0)
            self.assertFalse(st.trail_armed)

    def test_short_trail_high_trigger(self):
        with patch.multiple(
            cfg,
            ST_EXIT_MODE="trail_atr",
            ST_TRAIL_ATR_MULT=1.5,
            ST_TRAIL_ARM_ATR=0.5,
        ):
            entry, atr = 50.0, 1.0
            st = update_protect_state(
                ProtectState(entry, 0.0, False),
                side="SHORT",
                entry=entry,
                high=51.0,
                low=46.0,
                close=47.0,
                atr=atr,
            )
            rule = evaluate_profit_exit(
                st,
                side="SHORT",
                entry=entry,
                high=48.0,
                low=47.5,
                close=47.8,
                atr=atr,
            )
            self.assertEqual(rule, "trail_atr")


if __name__ == "__main__":
    unittest.main()
