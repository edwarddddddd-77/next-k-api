"""ORB 回测结算边界测试。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from orb.backtest import _LOAD_1M_CHUNK_MS, _iter_scan_ms, _load_range
from orb.config import OrbConfig
from orb.resolve import resolve_forward


def _1m_bars(day: str, start_hm: str, end_hm: str, *, close: float = 100.1) -> pd.DataFrame:
    tz = "America/New_York"
    t0 = pd.Timestamp(f"{day} {start_hm}", tz=tz)
    t1 = pd.Timestamp(f"{day} {end_hm}", tz=tz)
    rows = []
    t = t0
    while t <= t1:
        rows.append(
            {
                "open_time": int(t.value // 1_000_000),
                "open": close,
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close,
                "volume": 1.0,
            }
        )
        t += pd.Timedelta(minutes=1)
    return pd.DataFrame(rows)


class TestOrbBacktestResolve(unittest.TestCase):
    def test_eod_needs_1m_hist_end_not_5m_cutoff(self):
        """5m 最后一根 15:55 时，应用 1m 数据到 15:59 才能触发 16:00 收盘平仓。"""
        tz = "America/New_York"
        entry_bo = int(pd.Timestamp("2024-06-03 10:00", tz=tz).value // 1_000_000)
        hist_end_5m = int(pd.Timestamp("2024-06-03 15:55", tz=tz).value // 1_000_000)
        df_1m = _1m_bars("2024-06-03", "10:01", "15:59", close=101.0)
        cfg = OrbConfig(
            session_tz=tz,
            session_open_time="09:30",
            session_close_time="16:00",
            resolve_at_session_close=True,
            exit_mode="eod",
            early_exit_minutes=0,
            signal_interval="5m",
        )
        out_short, _, note_short, _, _ = resolve_forward(
            df_1m,
            entry=100.0,
            entry_bar_open_ms=entry_bo,
            side="LONG",
            sl=99.0,
            tp=None,
            hist_end_ms=hist_end_5m,
            bar_step_ms=cfg.bar_step_ms(),
            cfg=cfg,
        )
        out_long, px, note, _, _ = resolve_forward(
            df_1m,
            entry=100.0,
            entry_bar_open_ms=entry_bo,
            side="LONG",
            sl=99.0,
            tp=None,
            hist_end_ms=int(df_1m["open_time"].iloc[-1]),
            bar_step_ms=cfg.bar_step_ms(),
            cfg=cfg,
        )
        self.assertIsNone(out_short)
        self.assertEqual(out_long, "session_close")
        self.assertEqual(note, "resolved:session_close")
        self.assertAlmostEqual(px, 101.0)

    def test_incremental_resolve_hits_sl_before_eod(self):
        """逐 scan 结算：1m 触 SL 后不应等到 EoD 才平仓。"""
        tz = "America/New_York"
        entry_bo = int(pd.Timestamp("2026-06-09 10:05", tz=tz).value // 1_000_000)
        sl = 112.4469
        rows = []
        t = pd.Timestamp("2026-06-09 10:06", tz=tz)
        while t <= pd.Timestamp("2026-06-09 16:00", tz=tz):
            h, low = 112.20, 112.10
            if t == pd.Timestamp("2026-06-09 11:30", tz=tz):
                h = 112.46
            rows.append(
                {
                    "open_time": int(t.value // 1_000_000),
                    "open": 112.04,
                    "high": h,
                    "low": low,
                    "close": 112.15,
                    "volume": 1.0,
                }
            )
            t += pd.Timedelta(minutes=1)
        df_1m = pd.DataFrame(rows)
        cfg = OrbConfig(
            session_tz=tz,
            session_open_time="09:30",
            session_close_time="16:00",
            resolve_at_session_close=True,
            exit_mode="eod",
            signal_interval="5m",
        )
        scan_before = int(pd.Timestamp("2026-06-09 11:29:05", tz=tz).value // 1_000_000)
        scan_after = int(pd.Timestamp("2026-06-09 11:35:05", tz=tz).value // 1_000_000)
        out_pre, _, _, _, _ = resolve_forward(
            df_1m,
            entry=112.04,
            entry_bar_open_ms=entry_bo,
            side="SHORT",
            sl=sl,
            tp=None,
            hist_end_ms=scan_before,
            bar_step_ms=cfg.bar_step_ms(),
            cfg=cfg,
        )
        out_post, px, note, _, _ = resolve_forward(
            df_1m,
            entry=112.04,
            entry_bar_open_ms=entry_bo,
            side="SHORT",
            sl=sl,
            tp=None,
            hist_end_ms=scan_after,
            bar_step_ms=cfg.bar_step_ms(),
            cfg=cfg,
        )
        self.assertIsNone(out_pre)
        self.assertEqual(out_post, "loss")
        self.assertAlmostEqual(px, sl)
        self.assertIn("resolved", note)

    def test_aligned_sl_avoids_wick_stop_eod_wins(self):
        """统一 ATR 窗口后 SL 略宽，6/9 类毛刺（112.46）不触发止损，EoD 止盈。"""
        tz = "America/New_York"
        entry_bo = int(pd.Timestamp("2026-06-09 09:45", tz=tz).value // 1_000_000)
        eod_ms = int(pd.Timestamp("2026-06-09 16:00", tz=tz).value // 1_000_000)
        rows = []
        t = pd.Timestamp("2026-06-09 09:50", tz=tz)
        while t <= pd.Timestamp("2026-06-09 16:00", tz=tz):
            h, low, c = 112.20, 112.10, 112.15
            if t == pd.Timestamp("2026-06-09 10:05", tz=tz):
                h = 112.46
            if t >= pd.Timestamp("2026-06-09 15:55", tz=tz):
                c = 108.01
            rows.append(
                {
                    "open_time": int(t.value // 1_000_000),
                    "open": 112.04,
                    "high": h,
                    "low": low,
                    "close": c,
                    "volume": 1.0,
                }
            )
            t += pd.Timedelta(minutes=1)
        df_1m = pd.DataFrame(rows)
        cfg = OrbConfig(
            session_tz=tz,
            session_open_time="09:30",
            session_close_time="16:00",
            resolve_at_session_close=True,
            exit_mode="eod",
            signal_interval="5m",
        )
        aligned_sl = 112.4788
        old_live_sl = 112.4469
        out_aligned, px_a, note_a, _, _ = resolve_forward(
            df_1m,
            entry=112.04,
            entry_bar_open_ms=entry_bo,
            side="SHORT",
            sl=aligned_sl,
            tp=None,
            hist_end_ms=eod_ms,
            bar_step_ms=cfg.bar_step_ms(),
            cfg=cfg,
        )
        out_old, px_o, _, _, _ = resolve_forward(
            df_1m,
            entry=112.04,
            entry_bar_open_ms=entry_bo,
            side="SHORT",
            sl=old_live_sl,
            tp=None,
            hist_end_ms=eod_ms,
            bar_step_ms=cfg.bar_step_ms(),
            cfg=cfg,
        )
        self.assertEqual(out_old, "loss")
        self.assertAlmostEqual(px_o, old_live_sl)
        self.assertEqual(out_aligned, "session_close")
        self.assertAlmostEqual(px_a, 108.01)
        self.assertEqual(note_a, "resolved:session_close")

    def test_scan_times_align_utc_cron(self):
        bar = 300_000
        delay = 5_000
        t0 = (1_740_000_000_000 // bar) * bar
        scans = _iter_scan_ms(t0, t0 + bar * 3 + delay, bar_step_ms=bar)
        self.assertEqual(len(scans), 3)
        self.assertEqual(scans[0], t0 + bar + delay)
        for s in scans:
            self.assertEqual(s % bar, delay)

    def test_load_range_1m_splits_into_chunks(self):
        calls: list[tuple[int, int]] = []

        def _fake_fetch(_sym: str, _iv: str, start: int, end: int) -> list:
            calls.append((start, end))
            return []

        span = _LOAD_1M_CHUNK_MS * 3 + 1
        with patch("orb.backtest.fetch_klines_forward", side_effect=_fake_fetch):
            _load_range("INTCUSDT", "1m", 1_000_000, 1_000_000 + span)
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0][0], 1_000_000)
        self.assertEqual(calls[-1][1], 1_000_000 + span)


if __name__ == "__main__":
    unittest.main()
