"""Anchor Drift 回测引擎测试（合成 K 线，无网络）。"""

from __future__ import annotations

import unittest

import pandas as pd

from quant.anchor_drift.backtest import BacktestParams, simulate_bars
from quant.anchor_drift.config import AnchorDriftConfig


def _et_ms(y, m, d, hh, mm) -> int:
    ts = pd.Timestamp(year=y, month=m, day=d, hour=hh, minute=mm, tz="America/New_York")
    return int(ts.value // 1_000_000)


class TestAnchorDriftBacktest(unittest.TestCase):
    def _cfg(self) -> AnchorDriftConfig:
        return AnchorDriftConfig(
            enabled=True,
            signal_threshold=0.015,
            converge_threshold=0.003,
            max_adverse_extension=0.025,
            preopen_flat_minutes=5,
            tick_interval_sec=1.0,
            equity_usdt=100.0,
            risk_pct=0.01,
        )

    def test_weekend_mean_reversion_round_trip(self):
        """周五 16:00 anchor=100；周六 drift +2% 做空；周日回落收敛平仓。"""
        bars = [
            (_et_ms(2026, 3, 6, 15, 59), 100.0),
            (_et_ms(2026, 3, 7, 12, 0), 102.0),
            (_et_ms(2026, 3, 7, 18, 0), 100.1),
        ]
        df = pd.DataFrame(bars, columns=["open_time", "close"])
        df["open"] = df["close"]
        df["high"] = df["close"]
        df["low"] = df["close"]
        df["volume"] = 1.0

        params = BacktestParams(
            symbols=["MSTRUSDT"],
            equity_usdt=100.0,
            compound=False,
            slippage_bps=0.0,
            taker_fee_bps=0.0,
            cfg=self._cfg(),
        )
        res = simulate_bars(df, symbol="MSTRUSDT", params=params)
        self.assertGreaterEqual(len(res.trades), 1)
        t = res.trades[0]
        self.assertEqual(t.side, "SHORT")
        self.assertEqual(t.period, "weekend")
        self.assertGreater(t.pnl_gross_usdt, 0.0)
        self.assertIn(t.exit_reason, ("converged", "preopen_flat"))

    def test_preopen_flat_closes_before_open(self):
        """周一 9:27 仍在 preopen flat 窗口，应强平。"""
        bars = [
            (_et_ms(2026, 3, 6, 15, 59), 100.0),
            (_et_ms(2026, 3, 7, 10, 0), 103.0),
            (_et_ms(2026, 3, 9, 9, 27), 101.5),
        ]
        df = pd.DataFrame(bars, columns=["open_time", "close"])
        df["open"] = df["close"]
        df["high"] = df["close"]
        df["low"] = df["close"]
        df["volume"] = 1.0

        params = BacktestParams(
            symbols=["MSTRUSDT"],
            equity_usdt=100.0,
            compound=False,
            slippage_bps=0.0,
            taker_fee_bps=0.0,
            cfg=self._cfg(),
        )
        res = simulate_bars(df, symbol="MSTRUSDT", params=params)
        self.assertEqual(len(res.trades), 1)
        self.assertEqual(res.trades[0].exit_reason, "preopen_flat")


if __name__ == "__main__":
    unittest.main()
