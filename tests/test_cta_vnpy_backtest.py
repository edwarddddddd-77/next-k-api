"""vnpy 官方 CTA 回测 smoke tests。"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

import pandas as pd

from orb.core.config import OrbConfig
from orb.cta.vnpy.backtest import (
    CtaVnpyBacktestConfig,
    bar_symbol_from_vt,
    klines_df_to_bars,
    run_vnpy_cta_backtest,
    session_bounds_for_date,
)
from orb.cta.vnpy.registry import list_vnpy_strategies
from orb.vnpy.binance_gateway import vnpy_vt_symbol

ensure = __import__("orb.vnpy.bootstrap", fromlist=["ensure_vnpy_path"]).ensure_vnpy_path
ensure()

from vnpy.trader.constant import Exchange, Interval  # noqa: E402
from vnpy.trader.object import BarData  # noqa: E402


def _synthetic_bars(n: int = 500, *, base: float = 100.0) -> list[BarData]:
    sym = bar_symbol_from_vt(vnpy_vt_symbol("COINUSDT"))
    bars: list[BarData] = []
    t0 = datetime(2026, 2, 3, 14, 30, tzinfo=timezone.utc)
    px = base
    for i in range(n):
        dt = t0 + pd.Timedelta(minutes=i)
        px *= 1.0 + (0.001 if i % 17 else -0.0008)
        bars.append(
            BarData(
                symbol=sym,
                exchange=Exchange.GLOBAL,
                datetime=dt,
                interval=Interval.MINUTE,
                open_price=px * 0.999,
                high_price=px * 1.002,
                low_price=px * 0.998,
                close_price=px,
                volume=10.0,
                gateway_name="BACKTESTING",
            )
        )
    return bars


class TestCtaVnpyBacktest(unittest.TestCase):
    def test_list_strategies(self) -> None:
        keys = list_vnpy_strategies()
        self.assertIn("double_ma", keys)
        self.assertIn("king_keltner", keys)

    def test_run_double_ma_smoke(self) -> None:
        cfg = OrbConfig.from_env()
        bars = _synthetic_bars(600)
        start, end, _, _, _ = session_bounds_for_date("2026-02-03", cfg)
        bt = CtaVnpyBacktestConfig(equity_usdt=1000.0, compound=True)
        out = run_vnpy_cta_backtest(
            "COINUSDT",
            bars,
            strategy_key="double_ma",
            bt_cfg=bt,
            start=start,
            end=end,
            price=100.0,
            quiet=True,
            replay_start=start,
            replay_end=end,
            orb_cfg=cfg,
        )
        self.assertIsNone(out.get("error"))
        self.assertIn("end_wallet", out)
        self.assertIsInstance(out.get("trades"), list)

    def test_klines_to_bars(self) -> None:
        df = pd.DataFrame(
            {
                "open_time": [1700000000000],
                "open": [1.0],
                "high": [2.0],
                "low": [0.5],
                "close": [1.5],
                "volume": [1.0],
            }
        )
        bars = klines_df_to_bars(df, "COINUSDT", vt_symbol=vnpy_vt_symbol("COINUSDT"))
        self.assertEqual(len(bars), 1)
        self.assertEqual(bars[0].symbol, bar_symbol_from_vt(vnpy_vt_symbol("COINUSDT")))


if __name__ == "__main__":
    unittest.main()
