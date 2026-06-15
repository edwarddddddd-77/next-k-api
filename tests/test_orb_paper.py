"""ORB 纸面入库测试。"""

from __future__ import annotations

import json
import sqlite3
import unittest

import pandas as pd

from orb.core.config import OrbConfig
from orb.core.db import count_open_positions, fetch_open_for_resolve, migrate_orb_tables, symbol_session_traded
from orb.core.paper import (
    _drop_forming_bar,
    _idle_scan_skip_reason,
    in_regular_session,
    _upsert_signal,
    resolve_open_positions,
)
from orb.core.resolve import resolve_forward
from orb.core.signals import OrbSignal


class TestOrbPaper(unittest.TestCase):
    def test_upsert_open(self):
        conn = sqlite3.connect(":memory:")
        migrate_orb_tables(conn.cursor())
        conn.commit()
        cur = conn.cursor()
        sig = OrbSignal(
            symbol="ETHUSDT",
            price=3500.0,
            side="LONG",
            play="ORB_BREAKOUT_LONG",
            confidence="high",
            reasons=["test"],
            or_high=3490.0,
            or_low=3480.0,
            session_date="2024-03-15",
            entry_bar_open_ms=1_710_000_000_000,
            sl_price=3475.0,
            tp_price=3525.0,
            r_unit=25.0,
            paper_notional_usdt=1000.0,
            volume=1200.0,
            vol_ma=800.0,
        )
        _upsert_signal(cur, ts="2024-03-15T08:00:00Z", sig=sig, scan_params={"strategy": "orb"}, cfg=OrbConfig())
        conn.commit()
        self.assertEqual(count_open_positions(cur), 1)

    def test_flat_upsert_preserves_settled(self):
        conn = sqlite3.connect(":memory:")
        migrate_orb_tables(conn.cursor())
        conn.commit()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO orb_signals (
                recorded_at_utc, symbol, play, side, confidence,
                entry_price, sl_price, tp_price, session_date, outcome, pnl_usdt
            ) VALUES ('t','BTCUSDT','ORB_BREAKOUT_LONG','LONG','high',100,99,102,'2024-03-15','win',50)
            """
        )
        flat = OrbSignal("BTCUSDT", 0.0, "FLAT", "ORB_NO_TRADE", "low", ["no_breakout"])
        _upsert_signal(cur, ts="2024-03-16T08:00:00Z", sig=flat, scan_params={}, cfg=OrbConfig())
        conn.commit()
        cur.execute("SELECT outcome, pnl_usdt FROM orb_signals WHERE symbol='BTCUSDT'")
        row = cur.fetchone()
        self.assertEqual(row[0], "win")
        self.assertEqual(row[1], 50)

    def test_session_traded_from_settlements(self):
        conn = sqlite3.connect(":memory:")
        migrate_orb_tables(conn.cursor())
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO orb_settlements (
                settled_at_utc, signal_id, symbol, side, outcome,
                entry_price, exit_price, pnl_r, pnl_usdt, virtual_notional_usdt, session_date
            ) VALUES ('t',1,'ETHUSDT','LONG','win',1,2,1,10,1000,'2024-03-15')
            """
        )
        self.assertTrue(symbol_session_traded(cur, "ETHUSDT", "2024-03-15"))

    def test_resolve_starts_after_signal_bar(self):
        import pandas as pd

        cfg = OrbConfig(signal_interval="5m", early_exit_minutes=0)
        step = cfg.bar_step_ms()
        entry_bo = 1_710_000_000_000
        rows = []
        for i in range(8):
            o = entry_bo + i * 60_000
            rows.append({"open_time": o, "open": 100.0, "high": 100.8, "low": 99.5, "close": 100.2, "volume": 1.0})
        df = pd.DataFrame(rows)
        out, _, note, _, _ = resolve_forward(
            df,
            entry=100.0,
            entry_bar_open_ms=entry_bo,
            side="LONG",
            sl=99.0,
            tp=103.0,
            hist_end_ms=entry_bo + 5 * 60_000,
            bar_step_ms=step,
            cfg=cfg,
        )
        self.assertIsNone(out, note)

    def test_resolve_eod_no_tp(self):
        import pandas as pd

        tz = "America/New_York"
        anchor = int(pd.Timestamp("2024-03-15 09:30", tz=tz).value // 1_000_000)
        rows = []
        for i in range(390):
            bo = anchor + (i + 3) * 60_000
            rows.append(
                {
                    "open_time": bo,
                    "open": 100.0,
                    "high": 100.4,
                    "low": 99.8,
                    "close": 100.1,
                    "volume": 1.0,
                }
            )
        df = pd.DataFrame(rows)
        cfg = OrbConfig(
            session_tz=tz,
            session_open_time="09:30",
            session_close_time="16:00",
            resolve_at_session_close=True,
            exit_mode="eod",
            signal_interval="5m",
            early_exit_minutes=0,
        )
        out, px, note, _, _ = resolve_forward(
            df,
            entry=100.0,
            entry_bar_open_ms=anchor,
            side="LONG",
            sl=99.0,
            tp=None,
            hist_end_ms=int(df["open_time"].iloc[-1]),
            bar_step_ms=cfg.bar_step_ms(),
            cfg=cfg,
        )
        self.assertEqual(out, "session_close")
        self.assertAlmostEqual(px, 100.1)

    def test_resolve_session_close(self):
        import pandas as pd

        tz = "America/New_York"
        anchor = int(pd.Timestamp("2024-03-15 09:30", tz=tz).value // 1_000_000)
        rows = []
        for i in range(390):
            bo = anchor + (i + 3) * 60_000
            rows.append(
                {
                    "open_time": bo,
                    "open": 100.0,
                    "high": 100.4,
                    "low": 99.8,
                    "close": 100.1,
                    "volume": 1.0,
                }
            )
        df = pd.DataFrame(rows)
        cfg = OrbConfig(
            session_tz=tz,
            session_open_time="09:30",
            session_close_time="16:00",
            resolve_at_session_close=True,
            resolve_max_hold_ms=0,
            signal_interval="5m",
            early_exit_minutes=0,
        )
        out, px, note, _, _ = resolve_forward(
            df,
            entry=100.0,
            entry_bar_open_ms=anchor,
            side="LONG",
            sl=99.0,
            tp=103.0,
            hist_end_ms=int(df["open_time"].iloc[-1]),
            bar_step_ms=cfg.bar_step_ms(),
            cfg=cfg,
        )
        self.assertEqual(out, "session_close")
        self.assertEqual(note, "resolved:session_close")
        self.assertAlmostEqual(px, 100.1)

    def test_fetch_open_for_resolve_eod_null_tp(self):
        conn = sqlite3.connect(":memory:")
        migrate_orb_tables(conn.cursor())
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO orb_signals (
                recorded_at_utc, symbol, play, side, confidence,
                entry_price, entry_bar_open_ms, sl_price, tp_price, session_date
            ) VALUES ('t','QQQUSDT','ORB_BREAKOUT_LONG','LONG','high',100,1_710_000_000_000,99,NULL,'2024-03-15')
            """
        )
        conn.commit()
        rows = fetch_open_for_resolve(cur, default_notional=1000.0)
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0][6])

    def test_drop_forming_bar(self):
        cfg = OrbConfig(signal_interval="5m")
        step = cfg.bar_step_ms()
        base = 1_710_000_000_000
        df = pd.DataFrame(
            [
                {"open_time": base, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
                {"open_time": base + step, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
            ]
        )
        trimmed = _drop_forming_bar(df, cfg, now_ms=base + step + 1)
        self.assertEqual(len(trimmed), 1)
        kept = _drop_forming_bar(df, cfg, now_ms=base + step + step)
        self.assertEqual(len(kept), 2)

    def test_resolve_eod_open_position_pipeline(self):
        import orb.core.paper as paper_mod

        conn = sqlite3.connect(":memory:")
        migrate_orb_tables(conn.cursor())
        cur = conn.cursor()
        tz = "America/New_York"
        anchor = int(pd.Timestamp("2024-03-15 09:30", tz=tz).value // 1_000_000)
        cur.execute(
            """
            INSERT INTO orb_signals (
                recorded_at_utc, symbol, play, side, confidence,
                entry_price, entry_bar_open_ms, sl_price, tp_price, session_date
            ) VALUES ('t','QQQUSDT','ORB_BREAKOUT_LONG','LONG','high',100,?,99,NULL,'2024-03-15')
            """,
            (anchor,),
        )
        conn.commit()
        rows = []
        for i in range(390):
            bo = anchor + (i + 3) * 60_000
            rows.append(
                {
                    "open_time": bo,
                    "open": 100.0,
                    "high": 100.4,
                    "low": 99.8,
                    "close": 100.1,
                    "volume": 1.0,
                }
            )
        df = pd.DataFrame(rows)
        cfg = OrbConfig(
            market="us_equity",
            session_tz=tz,
            session_open_time="09:30",
            session_close_time="16:00",
            resolve_at_session_close=True,
            exit_mode="eod",
            signal_interval="5m",
        )
        old_fetch = paper_mod.fetch_klines_forward
        old_df = paper_mod.klines_to_df
        try:
            paper_mod.fetch_klines_forward = lambda *a, **k: rows
            paper_mod.klines_to_df = lambda r: df.copy()
            stats = resolve_open_positions(conn, cfg=cfg)
        finally:
            paper_mod.fetch_klines_forward = old_fetch
            paper_mod.klines_to_df = old_df
        self.assertEqual(stats["resolved"], 1)
        cur.execute("SELECT outcome FROM orb_signals WHERE symbol='QQQUSDT'")
        self.assertEqual(cur.fetchone()[0], "session_close")


class TestOrbIdleSkip(unittest.TestCase):
    def setUp(self):
        self.cfg = OrbConfig(
            market="us_equity",
            regular_session_only=True,
            session_tz="America/New_York",
            session_open_time="09:30",
            session_close_time="16:00",
        )
        self.conn = sqlite3.connect(":memory:")
        migrate_orb_tables(self.conn.cursor())
        self.conn.commit()
        self.cur = self.conn.cursor()
        self.pre_open_ms = int(pd.Timestamp("2024-03-15 08:00", tz="America/New_York").value // 1_000_000)
        self.rth_ms = int(pd.Timestamp("2024-03-15 10:00", tz="America/New_York").value // 1_000_000)
        self.after_close_ms = int(pd.Timestamp("2024-03-15 17:00", tz="America/New_York").value // 1_000_000)
        self.weekend_ms = int(pd.Timestamp("2024-03-16 10:00", tz="America/New_York").value // 1_000_000)

    def test_idle_skip_pre_market_no_positions(self):
        self.assertEqual(
            _idle_scan_skip_reason(self.cfg, self.cur, now_ms=self.pre_open_ms),
            "outside_regular_session_no_open_positions",
        )

    def test_idle_skip_after_close_no_positions(self):
        self.assertEqual(
            _idle_scan_skip_reason(self.cfg, self.cur, now_ms=self.after_close_ms),
            "outside_regular_session_no_open_positions",
        )

    def test_no_idle_skip_when_open_position(self):
        self.cur.execute(
            """
            INSERT INTO orb_signals (
                recorded_at_utc, symbol, play, side, confidence,
                entry_price, sl_price, session_date, outcome
            ) VALUES ('t','QQQUSDT','ORB_BREAKOUT_LONG','LONG','high',100,99,'2024-03-14',NULL)
            """
        )
        self.conn.commit()
        self.assertIsNone(_idle_scan_skip_reason(self.cfg, self.cur, now_ms=self.pre_open_ms))

    def test_idle_skip_weekend_no_positions(self):
        self.assertEqual(
            _idle_scan_skip_reason(self.cfg, self.cur, now_ms=self.weekend_ms),
            "outside_regular_session_no_open_positions",
        )

    def test_no_idle_skip_during_rth(self):
        self.assertFalse(in_regular_session(self.cfg, now_ms=self.pre_open_ms))
        self.assertTrue(in_regular_session(self.cfg, now_ms=self.rth_ms))
        self.assertIsNone(_idle_scan_skip_reason(self.cfg, self.cur, now_ms=self.rth_ms))


if __name__ == "__main__":
    unittest.main()
