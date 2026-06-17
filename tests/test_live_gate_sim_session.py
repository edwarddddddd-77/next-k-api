"""live_gate_sim 与 paper.py 一致的 session 锁 / robot 释放规则。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from orb.core.backtest import _SimOpen
from orb.core.config import OrbConfig
from orb.ml.gate import LiveGateConfig
from orb.ml.live_gate_sim import _first_resolve_scan_ms, simulate_live_gate_day
from orb.ml.ranker import BreakoutRanker


class _LowRanker(BreakoutRanker):
    def predict_true(self, feat, *, symbol: str = "") -> float:
        return 0.2

    def predict_fake(self, feat, *, symbol: str = "") -> float:
        return 0.8


class TestLiveGateSimSessionLock(unittest.TestCase):
    def test_gate_reject_does_not_block_later_scan(self) -> None:
        cfg = OrbConfig.from_env()
        gate = LiveGateConfig(min_p_true=0.35, max_opens_per_day=8, robot_reuse_after_exit=True)
        ranker = _LowRanker()
        calls = {"n": 0}

        def fake_analyze(sym, *, cfg, now_ms, session_traded=False, daily_df=None, bot_equity_usdt=None, df5=None):
            calls["n"] += 1
            from orb.core.signals import OrbSignal

            return OrbSignal(
                symbol=str(sym),
                price=100.0,
                side="LONG",
                play="ORB_BREAKOUT_LONG",
                confidence="high",
                reasons=["test"],
                sl_price=95.0,
                entry_bar_open_ms=int(now_ms) - cfg.bar_step_ms(),
            )

        dummy_df = pd.DataFrame({"open_time": [0], "close": [100.0]})

        def fake_load_klines(sym, interval, start_ms=0, end_ms=0):
            return dummy_df

        with patch("orb.ml.live_gate_sim.analyze_at_ms", side_effect=fake_analyze):
            with patch("orb.ml.live_gate_sim.is_actionable", return_value=True):
                with patch("orb.ml.live_gate_sim.load_klines", side_effect=fake_load_klines):
                    with patch("orb.ml.live_gate_sim._resolve_trade_row", return_value=None):
                        day = simulate_live_gate_day(
                            "2026-06-15",
                            ["AAAUSDT"],
                            cfg,
                            ranker,
                            gate,
                            robot_wallets=[1000.0] * 8,
                        )

        self.assertGreaterEqual(calls["n"], 2, "gate 拒绝后后续 scan 仍应分析该标的")
        self.assertEqual(day.get("opens"), 0)
        self.assertGreater(len(day.get("skipped_sample") or []), 0)

    def test_first_resolve_scan_after_open_scan(self) -> None:
        cfg = OrbConfig.from_env()
        entry_bo = 1_700_000_000_000
        open_scan = entry_bo + cfg.bar_step_ms() * 3
        scans = [open_scan + cfg.bar_step_ms() * i for i in range(1, 6)]
        df1 = pd.DataFrame(
            {
                "open_time": [entry_bo + cfg.bar_step_ms(), open_scan + cfg.bar_step_ms()],
                "high": [100.0, 100.0],
                "low": [94.0, 94.0],
                "close": [99.0, 95.0],
            }
        )
        pos = _SimOpen(
            symbol="AAAUSDT",
            side="LONG",
            play="ORB",
            entry=100.0,
            sl=95.0,
            tp=None,
            entry_bar_open_ms=entry_bo,
            notional=1000.0,
            session_date="2026-06-15",
            scan_open_ms=open_scan,
        )
        release = _first_resolve_scan_ms(pos, df1, scans, open_scan_ms=open_scan, cfg=cfg)
        self.assertIsNotNone(release)
        self.assertGreater(int(release), int(open_scan))
        self.assertIn(int(release), scans)


if __name__ == "__main__":
    unittest.main()
