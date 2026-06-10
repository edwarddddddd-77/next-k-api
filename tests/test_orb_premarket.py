"""盘前 ORB 过滤器单元测试（无网络）。"""

from __future__ import annotations

import unittest

import pandas as pd

from orb.config import OrbConfig
from orb.premarket import (
    apply_premarket_filter,
    classify_premarket_regime,
    premarket_anchor_ms,
    premarket_slice,
)
from orb.signals import classify_signal


def _ms(day: str, hm: str, tz: str = "America/New_York") -> int:
    return int(pd.Timestamp(f"{day} {hm}", tz=tz).value // 1_000_000)


def _bar(open_ms: int, o: float, h: float, l: float, c: float, vol: float) -> dict:
    return {
        "open_time": open_ms,
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": vol,
    }


class TestPremarket(unittest.TestCase):
    def test_premarket_slice_excludes_rth(self):
        tz = "America/New_York"
        day = "2024-03-15"
        pm0 = _ms(day, "04:00", tz)
        rth = _ms(day, "09:30", tz)
        rows = [
            _bar(pm0, 100, 101, 99.5, 100.5, 1000),
            _bar(pm0 + 300_000, 100.5, 102, 100, 101.5, 2000),
            _bar(rth, 101.5, 103, 101, 102.5, 5000),
        ]
        df = pd.DataFrame(rows)
        pm = premarket_slice(
            df, rth + 600_000, tz=tz, session_open_time="09:30", premarket_open_time="04:00"
        )
        self.assertEqual(len(pm), 2)
        self.assertAlmostEqual(float(pm["high"].max()), 102.0)

    def test_gap_and_go_regime(self):
        from orb.premarket import PremarketStats

        s = PremarketStats(
            pm_bars=10,
            pm_volume=50_000,
            pm_rvol=3.5,
            gap_pct=2.0,
            rth_open=102.0,
            pm_vwap=101.0,
            pm_late_below_vwap=False,
        )
        cfg = OrbConfig(
            premarket_filter=True,
            premarket_mode="gap_go_fade",
            premarket_min_gap_pct=0.5,
            premarket_rvol_min=3.0,
            premarket_min_volume=10_000,
        )
        self.assertEqual(classify_premarket_regime(s, cfg), "gap_and_go")

    def test_gap_and_fade_regime(self):
        from orb.premarket import PremarketStats

        s = PremarketStats(
            pm_bars=8,
            pm_volume=500,
            pm_rvol=0.5,
            gap_pct=1.5,
            rth_open=101.0,
            pm_vwap=101.5,
            pm_late_below_vwap=True,
        )
        cfg = OrbConfig(
            premarket_filter=True,
            premarket_mode="gap_go_fade",
            premarket_min_gap_pct=0.5,
            premarket_rvol_min=3.0,
            premarket_min_volume=10_000,
        )
        self.assertEqual(classify_premarket_regime(s, cfg), "gap_and_fade")

    def test_block_long_below_pmh(self):
        from orb.premarket import PremarketStats

        stats = PremarketStats(pm_bars=5, pm_high=105.0, pm_vwap=103.0, pm_volume=20_000)
        cfg = OrbConfig(
            premarket_filter=True,
            premarket_require_pmh_long=True,
            premarket_require_vwap_long=True,
        )
        ok, reason = apply_premarket_filter(
            "LONG",
            stats,
            cfg=cfg,
            entry_px=104.0,
            session_high=104.5,
            session_low=103.0,
            or_high=104.0,
            or_low=103.0,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "below_pmh")

    def test_allow_long_above_pmh_and_vwap(self):
        from orb.premarket import PremarketStats

        stats = PremarketStats(pm_bars=5, pm_high=105.0, pm_vwap=103.0, pm_volume=20_000, pm_rvol=2.0)
        cfg = OrbConfig(
            premarket_filter=True,
            premarket_require_pmh_long=True,
            premarket_require_vwap_long=True,
            premarket_rvol_min=0.0,
        )
        ok, reason = apply_premarket_filter(
            "LONG",
            stats,
            cfg=cfg,
            entry_px=105.5,
            session_high=105.5,
            session_low=103.0,
            or_high=104.0,
            or_low=103.0,
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_classify_signal_premarket_rejects_long(self):
        tz = "America/New_York"
        day = "2024-03-15"
        step = 300_000
        pm0 = premarket_anchor_ms(
            _ms(day, "10:00", tz), tz=tz, session_open_time="09:30", premarket_open_time="04:00"
        )
        rth = _ms(day, "09:30", tz)
        rows = []
        for i in range(4):
            rows.append(_bar(pm0 + i * step, 100, 102.0, 99.8, 101.0, 5000))
        or_bars = [
            (101.0, 101.5, 100.8, 101.2),
            (101.2, 101.6, 101.0, 101.4),
            (101.4, 101.7, 101.2, 101.5),
        ]
        for i, (o, h, l, c) in enumerate(or_bars):
            rows.append(_bar(rth + i * step, o, h, l, c, 5000))
        # 突破 OR 高点 101.7，但 session high 仍低于 PMH 102
        rows.append(_bar(rth + 3 * step, 101.5, 101.85, 101.4, 101.75, 8000))
        df = pd.DataFrame(rows)
        asof = int(df.iloc[-1]["open_time"])
        cfg = OrbConfig(
            market="us_equity",
            session_tz=tz,
            session_open_time="09:30",
            session_close_time="16:00",
            regular_session_only=False,
            or_minutes=15,
            entry_mode="breakout",
            sl_mode="or_range",
            exit_mode="eod",
            vol_mult=0.0,
            min_or_width_pct=0.0,
            macro_filter=False,
            premarket_filter=True,
            premarket_source="binance",
            premarket_require_pmh_long=True,
            premarket_require_vwap_long=False,
            premarket_rvol_min=0.0,
        )
        sig = classify_signal("QQQUSDT", df, asof_open_ms=asof, cfg=cfg, full_df=df)
        self.assertEqual(sig.side, "FLAT")
        self.assertIn("below_pmh", sig.reasons)

    def test_pmh_buffer_allows_near_miss(self):
        from orb.premarket import PremarketStats

        stats = PremarketStats(pm_bars=5, pm_high=100.0, pm_volume=20_000)
        cfg = OrbConfig(
            premarket_filter=True,
            premarket_require_pmh_long=True,
            premarket_pmh_buffer_bps=10.0,
            premarket_rvol_min=0.0,
        )
        ok, _ = apply_premarket_filter(
            "LONG",
            stats,
            cfg=cfg,
            entry_px=100.0,
            session_high=99.95,
            session_low=99.0,
            or_high=99.5,
            or_low=99.0,
        )
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
