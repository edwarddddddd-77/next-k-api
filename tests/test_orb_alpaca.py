"""Alpaca 盘前 provider 单元测试（无网络）。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from orb.config import OrbConfig
from orb.premarket import compute_premarket_stats, extended_fetch_anchor_ms, uses_alpaca_premarket
from orb.providers.alpaca import _bar_rows_to_df, binance_to_alpaca_symbol


class TestAlpacaProvider(unittest.TestCase):
    def test_symbol_map(self):
        self.assertEqual(binance_to_alpaca_symbol("QQQUSDT"), "QQQ")
        self.assertEqual(binance_to_alpaca_symbol("PLTRUSDT"), "PLTR")
        self.assertEqual(binance_to_alpaca_symbol("GOOGLUSDT"), "GOOGL")

    def test_bar_rows_to_df(self):
        rows = [
            {
                "t": "2024-03-15T08:00:00Z",
                "o": 100.0,
                "h": 101.0,
                "l": 99.5,
                "c": 100.5,
                "v": 12000,
            }
        ]
        df = _bar_rows_to_df(rows)
        self.assertEqual(len(df), 1)
        self.assertAlmostEqual(float(df["close"].iloc[0]), 100.5)
        self.assertAlmostEqual(float(df["volume"].iloc[0]), 12000.0)

    def test_extended_fetch_rth_when_alpaca(self):
        tz = "America/New_York"
        asof = int(pd.Timestamp("2024-03-15 10:00", tz=tz).value // 1_000_000)
        cfg = OrbConfig(
            market="us_equity",
            session_tz=tz,
            session_open_time="09:30",
            premarket_filter=True,
            premarket_source="alpaca",
        )
        self.assertTrue(uses_alpaca_premarket(cfg))
        anchor = extended_fetch_anchor_ms(asof, cfg)
        expected = int(pd.Timestamp("2024-03-15 09:30", tz=tz).value // 1_000_000)
        self.assertEqual(anchor, expected)

    @patch("orb.providers.alpaca.alpaca_configured", return_value=True)
    @patch("orb.providers.alpaca.fetch_stock_bars")
    @patch("orb.providers.alpaca.fetch_daily_bars")
    def test_compute_stats_from_alpaca_bars(self, mock_daily, mock_bars, _mock_cfg):
        tz = "America/New_York"
        day = "2024-03-15"
        pm0 = int(pd.Timestamp(f"{day} 04:00", tz=tz).value // 1_000_000)
        rth = int(pd.Timestamp(f"{day} 09:30", tz=tz).value // 1_000_000)
        step = 300_000
        pm_rows = [
            {
                "t": pd.Timestamp(pm0 + i * step, unit="ms", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
                "o": 100 + i * 0.1,
                "h": 102.0,
                "l": 99.8,
                "c": 101 + i * 0.1,
                "v": 5000 + i * 100,
            }
            for i in range(4)
        ]
        alpaca_df = _bar_rows_to_df(pm_rows)
        alpaca_df = pd.concat(
            [
                alpaca_df,
                _bar_rows_to_df(
                    [
                        {
                            "t": pd.Timestamp(rth, unit="ms", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "o": 101.5,
                            "h": 102.0,
                            "l": 101.2,
                            "c": 101.8,
                            "v": 8000,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        daily_df = _bar_rows_to_df(
            [
                {
                    "t": "2024-03-14T04:00:00Z",
                    "o": 98,
                    "h": 99,
                    "l": 97,
                    "c": 100.0,
                    "v": 1e6,
                }
            ]
        )
        mock_bars.return_value = alpaca_df
        mock_daily.return_value = daily_df

        from orb.providers.alpaca import load_premarket_history

        cfg = OrbConfig(
            market="us_equity",
            session_tz=tz,
            session_open_time="09:30",
            premarket_filter=True,
            premarket_source="alpaca",
            premarket_rvol_min=0,
        )
        bars, daily = load_premarket_history("QQQUSDT", cfg, asof_ms=rth + step)
        stats = compute_premarket_stats(
            pd.DataFrame(),
            rth + step,
            cfg=cfg,
            session_df=alpaca_df[alpaca_df["open_time"] >= rth],
            pm_history_df=bars,
            pm_daily_df=daily,
        )
        self.assertGreater(stats.pm_bars, 0)
        self.assertAlmostEqual(stats.pm_high, 102.0)
        self.assertGreater(stats.gap_pct, 0)

    def test_live_day_cache_avoids_refetch(self):
        from orb.providers import alpaca as ap

        ap.clear_alpaca_live_cache()
        calls = {"n": 0}

        def fake_bars(*_a, **_k):
            calls["n"] += 1
            return _bar_rows_to_df(
                [
                    {
                        "t": "2024-03-15T08:00:00Z",
                        "o": 100.0,
                        "h": 101.0,
                        "l": 99.5,
                        "c": 100.5,
                        "v": 5000,
                    }
                ]
            )

        def fake_daily(*_a, **_k):
            calls["n"] += 1
            return _bar_rows_to_df(
                [
                    {
                        "t": "2024-03-14T04:00:00Z",
                        "o": 98,
                        "h": 99,
                        "l": 97,
                        "c": 100.0,
                        "v": 1e6,
                    }
                ]
            )

        cfg = OrbConfig(
            market="us_equity",
            session_tz="America/New_York",
            session_open_time="09:30",
            premarket_filter=True,
            premarket_source="alpaca",
        )
        tz = cfg.session_tz
        asof1 = int(pd.Timestamp("2024-03-15 10:00", tz=tz).value // 1_000_000)
        asof2 = int(pd.Timestamp("2024-03-15 10:05", tz=tz).value // 1_000_000)

        with patch.object(ap, "alpaca_configured", return_value=True):
            with patch.object(ap, "fetch_stock_bars", side_effect=fake_bars):
                with patch.object(ap, "fetch_daily_bars", side_effect=fake_daily):
                    ap.load_premarket_history("QQQUSDT", cfg, asof_ms=asof1)
                    ap.load_premarket_history("QQQUSDT", cfg, asof_ms=asof2)
        self.assertEqual(calls["n"], 2)
        ap.clear_alpaca_live_cache()

    def test_sip_forbidden_falls_back_to_iex(self):
        from orb.providers import alpaca as ap

        def fake_once(sym, *, start_ms, end_ms, interval="5m", feed="iex", adjustment="raw"):
            if feed == "sip":
                raise RuntimeError("Alpaca data feed 'sip' forbidden")
            return _bar_rows_to_df(
                [
                    {
                        "t": "2024-03-15T08:00:00Z",
                        "o": 100.0,
                        "h": 101.0,
                        "l": 99.5,
                        "c": 100.5,
                        "v": 100,
                    }
                ]
            )

        with patch.object(ap, "_fetch_stock_bars_once", side_effect=fake_once):
            df = ap.fetch_stock_bars("QQQ", start_ms=0, end_ms=86_400_000, feed="sip")
        self.assertEqual(len(df), 1)


if __name__ == "__main__":
    unittest.main()
