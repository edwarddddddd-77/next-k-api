"""突破 bar 质量分单元测试。"""

from __future__ import annotations

import unittest
from unittest import mock

import pandas as pd

from orb.core.breakout_score import (
    breakout_score_for_signal,
    df5_for_breakout_score,
    passes_breakout_score,
    score_breakout_bar,
)


def _bar(open_time: int, o: float, h: float, l: float, c: float, vol: float = 1000.0) -> dict:
    return {"open_time": open_time, "open": o, "high": h, "low": l, "close": c, "volume": vol}


class TestBreakoutScore(unittest.TestCase):
    def test_strong_breakout_scores_high(self):
        sc = score_breakout_bar(
            side="LONG",
            level=100.0,
            open_=100.2,
            high=101.5,
            low=100.1,
            close=101.2,
            atr=1.0,
            vol_comp=0.8,
        )
        self.assertGreater(sc, 60.0)

    def test_weak_breakout_scores_low(self):
        sc = score_breakout_bar(
            side="LONG",
            level=100.0,
            open_=100.1,
            high=100.8,
            low=99.9,
            close=100.05,
            atr=1.0,
            vol_comp=0.3,
        )
        self.assertLess(sc, 45.0)

    def test_breakout_score_for_signal(self):
        from orb.core.config import OrbConfig
        from orb.core.signals import OrbSignal

        cfg = OrbConfig.from_env()
        sig = OrbSignal(
            "TESTUSDT",
            100.5,
            "LONG",
            "ORB_BREAKOUT_LONG",
            "high",
            or_high=100.0,
            or_low=98.0,
            entry_bar_open_ms=900_000,
        )
        df = pd.DataFrame(
            [{"open_time": 900_000, "open": 99.7, "high": 100.8, "low": 99.5, "close": 100.5, "volume": 1500.0}]
        )
        sc = breakout_score_for_signal(sig, df, cfg, now_ms=900_000 + cfg.bar_step_ms())
        self.assertGreater(sc, 0.0)

    def test_df5_for_breakout_score_live_fallback_when_cache_empty(self):
        from orb.core.config import OrbConfig
        from orb.core.signals import OrbSignal

        cfg = OrbConfig.from_env()
        sig = OrbSignal(
            "TESTUSDT",
            100.5,
            "LONG",
            "ORB_BREAKOUT_LONG",
            "high",
            or_high=100.0,
            or_low=98.0,
            entry_bar_open_ms=900_000,
        )
        now_ms = 900_000 + cfg.bar_step_ms()
        live_df = pd.DataFrame(
            [{"open_time": 900_000, "open": 99.7, "high": 100.8, "low": 99.5, "close": 100.5, "volume": 1500.0}]
        )
        cache: dict = {}
        with mock.patch("orb.core.kline_cache.load_klines", return_value=pd.DataFrame()):
            with mock.patch("orb.core.paper._load_signal_df", return_value=live_df) as live_mock:
                out = df5_for_breakout_score(
                    "TESTUSDT",
                    sig,
                    cfg,
                    session_day="2026-06-22",
                    now_ms=now_ms,
                    df5_cache=cache,
                )
        live_mock.assert_called_once()
        self.assertFalse(out.empty)
        self.assertIs(cache.get("TESTUSDT"), out)

    def test_df5_for_breakout_score_uses_scan_cache_without_live_or_disk(self):
        from orb.core.config import OrbConfig
        from orb.core.signals import OrbSignal

        cfg = OrbConfig.from_env()
        now_ms = 900_000 + cfg.bar_step_ms()
        sig = OrbSignal(
            "TESTUSDT",
            100.5,
            "LONG",
            "ORB_BREAKOUT_LONG",
            "high",
            or_high=100.0,
            or_low=98.0,
            entry_bar_open_ms=900_000,
        )
        scan_df = pd.DataFrame(
            [{"open_time": 900_000, "open": 99.7, "high": 100.8, "low": 99.5, "close": 100.5, "volume": 1500.0}]
        )
        cache = {"TESTUSDT": scan_df}
        with mock.patch("orb.core.kline_cache.load_klines") as disk_mock:
            with mock.patch("orb.core.paper._load_signal_df") as live_mock:
                out = df5_for_breakout_score(
                    "TESTUSDT",
                    sig,
                    cfg,
                    session_day="2026-06-22",
                    now_ms=now_ms,
                    df5_cache=cache,
                )
        disk_mock.assert_not_called()
        live_mock.assert_not_called()
        self.assertIs(out, scan_df)

    def test_df5_for_breakout_score_refreshes_stale_scan_cache(self):
        from orb.core.config import OrbConfig
        from orb.core.signals import OrbSignal

        cfg = OrbConfig.from_env()
        now_ms = 1_000_000 + cfg.bar_step_ms()
        sig = OrbSignal(
            "TESTUSDT",
            100.5,
            "LONG",
            "ORB_BREAKOUT_LONG",
            "high",
            or_high=100.0,
            or_low=98.0,
            entry_bar_open_ms=1_000_000,
        )
        stale_df = pd.DataFrame(
            [{"open_time": 900_000, "open": 99.7, "high": 100.8, "low": 99.5, "close": 100.5, "volume": 1500.0}]
        )
        fresh_df = pd.DataFrame(
            [{"open_time": 1_000_000, "open": 100.1, "high": 101.0, "low": 100.0, "close": 100.8, "volume": 1800.0}]
        )
        cache = {"TESTUSDT": stale_df}
        with mock.patch("orb.core.kline_cache.load_klines", return_value=pd.DataFrame()):
            with mock.patch("orb.core.paper._load_signal_df", return_value=fresh_df) as live_mock:
                out = df5_for_breakout_score(
                    "TESTUSDT",
                    sig,
                    cfg,
                    session_day="2026-06-22",
                    now_ms=now_ms,
                    df5_cache=cache,
                )
        live_mock.assert_called_once()
        self.assertIs(out, fresh_df)
        self.assertIs(cache["TESTUSDT"], fresh_df)

    def test_df5_for_breakout_score_uses_cache_when_entry_present(self):
        from orb.core.config import OrbConfig
        from orb.core.signals import OrbSignal

        cfg = OrbConfig.from_env()
        now_ms = 900_000 + cfg.bar_step_ms()
        sig = OrbSignal(
            "TESTUSDT",
            100.5,
            "LONG",
            "ORB_BREAKOUT_LONG",
            "high",
            or_high=100.0,
            or_low=98.0,
            entry_bar_open_ms=900_000,
        )
        cached_df = pd.DataFrame(
            [{"open_time": 900_000, "open": 99.7, "high": 100.8, "low": 99.5, "close": 100.5, "volume": 1500.0}]
        )
        cache = {"TESTUSDT": cached_df}
        with mock.patch("orb.core.kline_cache.load_klines") as disk_mock:
            with mock.patch("orb.core.paper._load_signal_df") as live_mock:
                out = df5_for_breakout_score(
                    "TESTUSDT",
                    sig,
                    cfg,
                    session_day="2026-06-22",
                    now_ms=now_ms,
                    df5_cache=cache,
                )
        disk_mock.assert_not_called()
        live_mock.assert_not_called()
        self.assertIs(out, cached_df)

    def test_passes_breakout_score(self):
        ok, _ = passes_breakout_score(55.0, min_score=50.0)
        self.assertTrue(ok)
        ok2, reason = passes_breakout_score(42.0, min_score=50.0)
        self.assertFalse(ok2)
        self.assertEqual(reason, "breakout_score<50")
        ok3, reason3 = passes_breakout_score(None, min_score=50.0)
        self.assertFalse(ok3)
        self.assertEqual(reason3, "breakout_score_missing")

    def test_gate_should_open_breakout_filter(self):
        from orb.ml.gate import LiveGateConfig, LiveGateDayState, should_open

        gate = LiveGateConfig(min_p_true=0.35, min_breakout_score=45.0, max_opens_per_day=8)
        state = LiveGateDayState()
        feat = {"minutes_after_or": 30.0}
        ok, reason = should_open(
            p_true=0.5,
            symbol="TSLAUSDT",
            feat=feat,
            sync=0,
            state=state,
            gate=gate,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "breakout_score_missing")
        ok2, reason2 = should_open(
            p_true=0.5,
            symbol="TSLAUSDT",
            feat=feat,
            sync=0,
            state=state,
            gate=gate,
            breakout_score=42.0,
        )
        self.assertFalse(ok2)
        self.assertEqual(reason2, "breakout_score<45")
        ok3, reason3 = should_open(
            p_true=0.5,
            symbol="TSLAUSDT",
            feat=feat,
            sync=0,
            state=state,
            gate=gate,
            breakout_score=55.0,
        )
        self.assertTrue(ok3)
        self.assertEqual(reason3, "open_ok")

    def test_gate_should_open_skips_breakout_when_disabled(self):
        from orb.ml.gate import LiveGateConfig, LiveGateDayState, should_open

        gate = LiveGateConfig(min_p_true=0.35, min_breakout_score=0.0, max_opens_per_day=8)
        state = LiveGateDayState()
        feat = {"minutes_after_or": 30.0}
        ok, reason = should_open(
            p_true=0.5,
            symbol="TSLAUSDT",
            feat=feat,
            sync=0,
            state=state,
            gate=gate,
            breakout_score=None,
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "open_ok")


    def test_gate_replay_respects_breakout_score(self):
        from orb.ml.gate import LiveGateConfig
        from orb.ml.gate_replay import replay_day

        gate = LiveGateConfig(min_p_true=0.35, min_breakout_score=45.0, max_opens_per_day=8)
        timeline = [
            {
                "p_true": 0.5,
                "sync_same_side": 0,
                "minutes_after_or": 30.0,
                "breakout_score": 40.0,
                "pnl_usdt": 100.0,
                "true_breakout": True,
            },
            {
                "p_true": 0.5,
                "sync_same_side": 0,
                "minutes_after_or": 30.0,
                "breakout_score": 50.0,
                "pnl_usdt": 80.0,
                "true_breakout": True,
            },
        ]
        out = replay_day(timeline, min_p=0.35, max_opens=8, gate=gate)
        self.assertEqual(out["opens"], 1)
        self.assertAlmostEqual(out["pnl"], 80.0)


if __name__ == "__main__":
    unittest.main()
