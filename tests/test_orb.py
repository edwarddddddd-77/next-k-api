"""ORB 策略单元测试（无网络）。"""

from __future__ import annotations

import sqlite3
import unittest

import pandas as pd

from orb.config import OrbConfig
from orb.db import (
    archive_settlement,
    ensure_symbol_bots,
    migrate_orb_tables,
    symbol_bot_wallet_balance,
)
from orb.tz import normalize_session_tz, session_utc_offset_hours
from orb.us_equity_calendar import (
    is_us_equity_early_close_day,
    is_us_equity_trading_day,
    us_equity_session_close_time,
)
from orb.session import (
    compute_opening_range,
    is_regular_session,
    is_trading_session,
    session_anchor_ms,
    session_day_floor_ms,
    trading_session_block_reason,
)
from orb.signals import classify_signal, compute_sl_tp, compute_position_notional


def _utc_day0(date_str: str = "2024-03-15") -> int:
    return int(pd.Timestamp(date_str, tz="UTC").value // 1_000_000)


def _make_df(n: int, *, step_ms: int = 300_000, start_ms: int | None = None) -> pd.DataFrame:
    start_ms = _utc_day0() if start_ms is None else start_ms
    rows = []
    for i in range(n):
        o = 100.0
        rows.append(
            {
                "open_time": start_ms + i * step_ms,
                "open": o,
                "high": 100.2,
                "low": 99.8,
                "close": o,
                "volume": 1000.0 + i * 10,
            }
        )
    return pd.DataFrame(rows)


class TestOrb(unittest.TestCase):
    def test_opening_range(self):
        day0 = _utc_day0()
        df = _make_df(8, step_ms=300_000, start_ms=day0)
        pack = compute_opening_range(
            df, or_minutes=15, bar_step_ms=300_000, asof_open_ms=day0 + 7 * 300_000
        )
        self.assertIsNotNone(pack)
        assert pack is not None
        self.assertAlmostEqual(pack["or_high"], 100.2)

    def test_breakout_long_signal(self):
        day0 = _utc_day0()
        step = 300_000
        df = _make_df(10, step_ms=step, start_ms=day0)
        df.at[4, "close"] = 100.0
        df.at[4, "high"] = 100.15
        df.at[4, "low"] = 99.9
        df.at[4, "volume"] = 1000.0
        for i in range(5, 10):
            px = 100.5 + (i - 5) * 0.1
            df.at[i, "close"] = px
            df.at[i, "high"] = px + 0.05
            df.at[i, "low"] = px - 0.05
            df.at[i, "volume"] = 5000.0
        cfg = OrbConfig(
            or_minutes=15,
            session_tz="UTC",
            entry_mode="breakout",
            sl_mode="or_range",
            exit_mode="fixed_r",
            tp_r_multiple=2.0,
            session_open_time="",
            session_close_time="",
            regular_session_only=False,
            vwap_filter=False,
            confirm_bars=1,
            vol_mult=0.0,
            min_or_width_pct=0.01,
            max_or_width_pct=10.0,
        )
        asof = int(df.iloc[5]["open_time"])
        sig = classify_signal("BTCUSDT", df, asof_open_ms=asof, cfg=cfg)
        self.assertEqual(sig.side, "LONG", sig.reasons)

    def test_sl_tp_short(self):
        cfg = OrbConfig(sl_mode="or_range", exit_mode="fixed_r", tp_r_multiple=2.0, tick_size=0.01)
        sl, tp, r = compute_sl_tp(side="SHORT", entry=100.22, or_high=100.2, or_low=99.8, cfg=cfg)
        self.assertIsNotNone(sl)
        assert sl is not None and tp is not None
        self.assertGreater(sl, 100.2)
        self.assertLess(tp, 100.22)

    def test_atr_sl_eod_no_tp(self):
        cfg = OrbConfig(sl_mode="atr_pct", exit_mode="eod", atr_sl_fraction=0.05, min_sl_pct=0.0)
        sl, tp, r = compute_sl_tp(
            side="LONG", entry=100.0, or_high=100.2, or_low=99.8, cfg=cfg, daily_atr=1.6
        )
        self.assertAlmostEqual(sl, 100.0 - 1.6 * 0.05)
        self.assertIsNone(tp)
        self.assertAlmostEqual(r, 1.6 * 0.05)

    def test_risk_position_notional(self):
        cfg = OrbConfig(
            risk_pct=0.01,
            symbol_bot_equity_usdt=25_000.0,
            account_equity_usdt=25_000.0,
            position_safety_pct=0.15,
            fixed_notional_usdt=0.0,
        )
        notion = compute_position_notional(entry=100.0, sl=99.92, cfg=cfg)
        budget = 25_000 * 0.01 * 0.85
        expected = budget / 0.0008
        self.assertAlmostEqual(notion, expected, places=0)

    def test_fixed_notional_overrides_risk(self):
        cfg = OrbConfig(
            risk_pct=0.01,
            symbol_bot_equity_usdt=10_000.0,
            account_equity_usdt=10_000.0,
            position_safety_pct=0.15,
            fixed_notional_usdt=10_000.0,
        )
        self.assertFalse(cfg.uses_risk_sizing())
        notion = compute_position_notional(entry=100.0, sl=99.92, cfg=cfg)
        self.assertAlmostEqual(notion, 10_000.0)

    def test_risk_sizing_uses_symbol_bot_equity(self):
        cfg = OrbConfig(
            risk_pct=0.01,
            symbol_bot_equity_usdt=10_000.0,
            position_safety_pct=0.15,
            fixed_notional_usdt=0.0,
        )
        notion = compute_position_notional(entry=100.0, sl=99.92, cfg=cfg)
        budget = 10_000 * 0.01 * 0.85
        expected = budget / 0.0008
        self.assertAlmostEqual(notion, expected, places=0)

    def test_default_paper_notional_us_equity(self):
        cfg = OrbConfig.from_env()
        self.assertEqual(cfg.default_paper_notional(), cfg.virtual_notional_usdt)

    def test_symbol_bot_wallet_after_settlement(self):
        conn = sqlite3.connect(":memory:")
        migrate_orb_tables(conn.cursor())
        cur = conn.cursor()
        ensure_symbol_bots(cur, ["QQQUSDT"], initial_equity_usdt=10_000.0)
        archive_settlement(
            cur,
            signal_id=1,
            symbol="QQQUSDT",
            side="LONG",
            play="ORB_BREAKOUT_LONG",
            outcome="win",
            entry_price=100.0,
            exit_price=101.0,
            pnl_r=1.0,
            pnl_usdt=50.0,
            notional=10_000.0,
            exit_rule="tp",
            settled_at_utc="2024-03-15T16:00:00Z",
            session_date="2024-03-15",
        )
        conn.commit()
        bal = symbol_bot_wallet_balance(conn, "QQQUSDT", initial_equity_usdt=10_000.0, sync=True)
        self.assertAlmostEqual(bal, 10_050.0)
        conn.commit()
        cur.execute("SELECT virtual_equity_usdt FROM orb_symbol_bots WHERE symbol='QQQUSDT'")
        self.assertAlmostEqual(float(cur.fetchone()[0]), 10_050.0)

    def test_us_equity_session_anchor(self):
        tz = "America/New_York"
        asof = int(pd.Timestamp("2024-03-15 10:05", tz=tz).value // 1_000_000)
        anchor = session_anchor_ms(asof, tz=tz, session_open_time="09:30")
        expected = int(pd.Timestamp("2024-03-15 09:30", tz=tz).value // 1_000_000)
        self.assertEqual(anchor, expected)

    def test_us_equity_or_window(self):
        tz = "America/New_York"
        step = 300_000
        anchor = int(pd.Timestamp("2024-03-15 09:30", tz=tz).value // 1_000_000)
        df = _make_df(6, step_ms=step, start_ms=anchor)
        asof = anchor + 5 * step
        pack = compute_opening_range(
            df,
            or_minutes=15,
            bar_step_ms=step,
            asof_open_ms=asof,
            tz=tz,
            session_open_time="09:30",
        )
        self.assertIsNotNone(pack)
        assert pack is not None
        self.assertEqual(pack["session_date"], "2024-03-15")
        self.assertAlmostEqual(pack["or_high"], 100.2)

    def test_us_equity_before_open_flat(self):
        tz = "America/New_York"
        step = 300_000
        anchor = int(pd.Timestamp("2024-03-15 09:30", tz=tz).value // 1_000_000)
        df = _make_df(4, step_ms=step, start_ms=anchor - 2 * step)
        asof = int(pd.Timestamp("2024-03-15 09:10", tz=tz).value // 1_000_000)
        cfg = OrbConfig(
            market="us_equity",
            session_tz=tz,
            session_open_time="09:30",
            session_close_time="16:00",
            regular_session_only=True,
            confirm_bars=1,
            vol_mult=0.0,
            min_or_width_pct=0.01,
        )
        sig = classify_signal("MSTRUSDT", df, asof_open_ms=asof, cfg=cfg)
        self.assertEqual(sig.side, "FLAT")
        self.assertIn("session_not_open", sig.reasons)

    def test_us_equity_overnight_blocked(self):
        tz = "America/New_York"
        asof = int(pd.Timestamp("2024-03-15 20:00", tz=tz).value // 1_000_000)
        self.assertFalse(
            is_trading_session(
                asof,
                tz=tz,
                session_open_time="09:30",
                session_close_time="16:00",
                market="us_equity",
            )
        )
        self.assertEqual(
            trading_session_block_reason(
                asof,
                tz=tz,
                session_open_time="09:30",
                session_close_time="16:00",
                market="us_equity",
            ),
            "outside_regular_session",
        )

    def test_us_equity_weekend_blocked(self):
        tz = "America/New_York"
        asof = int(pd.Timestamp("2024-03-16 10:00", tz=tz).value // 1_000_000)
        self.assertFalse(
            is_trading_session(
                asof,
                tz=tz,
                session_open_time="09:30",
                session_close_time="16:00",
                market="us_equity",
            )
        )
        cfg = OrbConfig(
            market="us_equity",
            session_tz=tz,
            session_open_time="09:30",
            session_close_time="16:00",
            regular_session_only=True,
        )
        sig = classify_signal("QQQUSDT", _make_df(3), asof_open_ms=asof, cfg=cfg)
        self.assertEqual(sig.side, "FLAT")
        self.assertIn("weekend", sig.reasons)

    def test_us_equity_exchange_holiday_blocked(self):
        tz = "America/New_York"
        asof = int(pd.Timestamp("2024-07-04 10:00", tz=tz).value // 1_000_000)
        self.assertEqual(
            trading_session_block_reason(
                asof,
                tz=tz,
                session_open_time="09:30",
                session_close_time="16:00",
                market="us_equity",
            ),
            "exchange_holiday",
        )

    def test_us_equity_early_close_after_1300_blocked(self):
        tz = "America/New_York"
        asof = int(pd.Timestamp("2024-07-03 14:00", tz=tz).value // 1_000_000)
        self.assertFalse(
            is_trading_session(
                asof,
                tz=tz,
                session_open_time="09:30",
                session_close_time="16:00",
                market="us_equity",
            )
        )
        self.assertTrue(
            is_trading_session(
                int(pd.Timestamp("2024-07-03 12:00", tz=tz).value // 1_000_000),
                tz=tz,
                session_open_time="09:30",
                session_close_time="16:00",
                market="us_equity",
            )
        )

    def test_us_equity_2026_july3_early_close(self):
        self.assertTrue(is_us_equity_trading_day("2026-07-03"))
        self.assertTrue(is_us_equity_early_close_day("2026-07-03"))
        self.assertEqual(us_equity_session_close_time("2026-07-03"), "13:00")

    def test_macro_skip_2026_dates(self):
        import os

        from orb.macro_calendar import clear_macro_calendar_cache, is_macro_skip_day

        os.environ["ORB_MACRO_CALENDAR_FETCH"] = "0"
        clear_macro_calendar_cache()
        self.assertTrue(is_macro_skip_day("2026-05-12"))  # CPI
        self.assertTrue(is_macro_skip_day("2026-03-18"))  # FOMC
        self.assertFalse(is_macro_skip_day("2026-05-13"))

    def test_dst_winter_vs_summer_open_utc(self):
        """墙钟 09:30 不变；UTC 在 EST=-5h / EDT=-4h 间切换。"""
        tz = "America/New_York"
        winter_open = int(pd.Timestamp("2025-01-08 09:30", tz=tz).value // 1_000_000)
        summer_open = int(pd.Timestamp("2025-07-08 09:30", tz=tz).value // 1_000_000)
        self.assertEqual(
            str(pd.Timestamp(winter_open, unit="ms", tz="UTC")),
            "2025-01-08 14:30:00+00:00",
        )
        self.assertEqual(
            str(pd.Timestamp(summer_open, unit="ms", tz="UTC")),
            "2025-07-08 13:30:00+00:00",
        )
        self.assertEqual(
            session_anchor_ms(winter_open + 3_600_000, tz=tz, session_open_time="09:30"),
            winter_open,
        )
        self.assertEqual(
            session_anchor_ms(summer_open + 3_600_000, tz=tz, session_open_time="09:30"),
            summer_open,
        )
        self.assertEqual(session_utc_offset_hours(winter_open, tz), -5.0)
        self.assertEqual(session_utc_offset_hours(summer_open, tz), -4.0)

    def test_est_alias_maps_to_dst_aware_tz(self):
        self.assertEqual(normalize_session_tz("EST", market="us_equity"), "America/New_York")
        self.assertEqual(normalize_session_tz("EDT", market="us_equity"), "America/New_York")
        self.assertEqual(normalize_session_tz("US/Eastern", market="us_equity"), "America/New_York")

    def test_image_retest_long_with_vwap(self):
        tz = "America/New_York"
        step = 300_000
        anchor = int(pd.Timestamp("2024-03-15 09:30", tz=tz).value // 1_000_000)
        df = _make_df(8, step_ms=step, start_ms=anchor)
        for i in range(3):
            df.at[i, "high"] = 100.2
            df.at[i, "low"] = 99.8
            df.at[i, "close"] = 100.0
            df.at[i, "volume"] = 1000.0
        df.at[3, "close"] = 100.5
        df.at[3, "high"] = 100.6
        df.at[3, "low"] = 100.0
        df.at[3, "volume"] = 5000.0
        df.at[4, "close"] = 100.3
        df.at[4, "high"] = 100.4
        df.at[4, "low"] = 100.15
        df.at[4, "volume"] = 2000.0
        df.at[5, "close"] = 100.35
        df.at[5, "high"] = 100.45
        df.at[5, "low"] = 100.18
        df.at[5, "volume"] = 1800.0
        cfg = OrbConfig(
            session_tz=tz,
            session_open_time="09:30",
            session_close_time="16:00",
            regular_session_only=False,
            entry_mode="retest",
            sl_mode="or_range",
            exit_mode="fixed_r",
            tp_r_multiple=2.0,
            vol_mult=1.5,
            vol_ma_period=3,
            vwap_filter=True,
            min_or_width_pct=0.0,
            max_or_width_pct=0.0,
            trade_window_minutes=90,
        )
        asof = int(df.iloc[5]["open_time"])
        sig = classify_signal("NVDAUSDT", df, asof_open_ms=asof, cfg=cfg)
        self.assertEqual(sig.side, "LONG", sig.reasons)

    def test_min_sl_pct_widens_stop(self):
        cfg = OrbConfig(sl_mode="or_range", exit_mode="eod", min_sl_pct=0.01, tick_size=0.01)
        sl, tp, r = compute_sl_tp(
            side="LONG", entry=100.0, or_high=100.05, or_low=99.99, cfg=cfg
        )
        self.assertIsNotNone(sl)
        assert sl is not None and r is not None
        self.assertAlmostEqual(r, 1.0)
        self.assertAlmostEqual(sl, 99.0)

    def test_paper_defaults_from_env(self):
        import os

        old = os.environ.pop("ORB_ENTRY_MODE", None)
        try:
            cfg = OrbConfig.from_env()
            self.assertEqual(cfg.entry_mode, "breakout")
            self.assertFalse(cfg.vwap_filter)
            self.assertEqual(cfg.vol_mult, 0.0)
            self.assertEqual(cfg.sl_mode, "atr_pct")
            self.assertEqual(cfg.exit_mode, "eod")
            self.assertEqual(cfg.atr_sl_fraction, 0.05)
            self.assertEqual(cfg.risk_pct, 0.01)
            self.assertEqual(cfg.symbol_bot_equity_usdt, 10_000.0)
            self.assertEqual(cfg.account_equity_usdt, 10_000.0)
            self.assertEqual(cfg.fixed_notional_usdt, 0.0)
            self.assertTrue(cfg.uses_risk_sizing())
            self.assertEqual(cfg.min_or_width_pct, 0.0)
            self.assertEqual(cfg.entry_tick_offset, 2)
            self.assertEqual(cfg.early_exit_minutes, 0)
            self.assertEqual(cfg.max_open_positions, 7)
            self.assertTrue(cfg.macro_filter)
            self.assertEqual(cfg.signal_interval, "5m")
            self.assertEqual(
                cfg.symbol_list(),
                [
                    "COINUSDT",
                    "INTCUSDT",
                    "PAYPUSDT",
                    "GOOGLUSDT",
                    "PLTRUSDT",
                    "EWYUSDT",
                    "QQQUSDT",
                ],
            )
            from orb.config import default_scan_interval_minutes, scan_interval_minutes_for_signal

            self.assertEqual(scan_interval_minutes_for_signal(cfg.signal_interval), 5)
            self.assertEqual(default_scan_interval_minutes(), 5)
        finally:
            if old is not None:
                os.environ["ORB_ENTRY_MODE"] = old


if __name__ == "__main__":
    unittest.main()
