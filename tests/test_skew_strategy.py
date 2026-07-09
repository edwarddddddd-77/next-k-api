"""Skew 中性策略单元测试（纯逻辑，不调用币安 API）。"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import skew_strategy as sk
from orb.vnpy import strategy_signals as ss


class TestSkewZScores(unittest.TestCase):
    def test_z_scores_centered(self):
        zs = sk._z_scores([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(sum(zs), 0.0, places=5)

    def test_single_value_zero(self):
        self.assertEqual(sk._z_scores([42.0]), [0.0])


class TestSkewClassify(unittest.TestCase):
    def test_squeeze_long(self):
        t, side, score = sk._classify_signal(fr_z=-1.5, oi_z=1.0, px_chg=5.0)
        self.assertEqual(t, "squeeze_long")
        self.assertEqual(side, "LONG")
        self.assertGreater(score, 0)

    def test_cover_trap(self):
        t, side, _ = sk._classify_signal(fr_z=0.0, oi_z=-1.5, px_chg=8.0)
        self.assertEqual(t, "cover_trap")
        self.assertEqual(side, "SKIP")

    def test_weak_short(self):
        t, side, _ = sk._classify_signal(fr_z=1.2, oi_z=-1.0, px_chg=-2.0)
        self.assertEqual(t, "weak_short")
        self.assertEqual(side, "SHORT")


class TestSkewRegime(unittest.TestCase):
    def test_defensive_on_btc_crash(self):
        regime = sk._btc_regime({"px_chg": -10.0, "fr_z": 0.0, "oi_z": 0.0})
        self.assertEqual(regime["regime"], "defensive")
        self.assertTrue(regime["halt_new_pairs"])

    def test_defensive_at_minus_5pct_default(self):
        regime = sk._btc_regime({"px_chg": -5.5, "fr_z": 0.0, "oi_z": 0.0})
        self.assertEqual(regime["regime"], "defensive")

    def test_normal(self):
        regime = sk._btc_regime({"px_chg": 1.0, "fr_z": 0.0, "oi_z": 0.0})
        self.assertEqual(regime["regime"], "normal")
        self.assertFalse(regime["halt_new_pairs"])


class TestSkewPairPick(unittest.TestCase):
    def test_picks_long_and_short(self):
        rows = [
            {
                "symbol": "AAAUSDT",
                "coin": "AAA",
                "signal_type": "squeeze_long",
                "side": "LONG",
                "skew_score": 3.0,
            },
            {
                "symbol": "BBBUSDT",
                "coin": "BBB",
                "signal_type": "weak_short",
                "side": "SHORT",
                "skew_score": 2.5,
            },
        ]
        pair = sk.pick_pair_trade(
            rows,
            {"regime": "normal", "halt_new_pairs": False},
            sk.SkewConfig(min_skew_gap=0.0),
        )
        self.assertIsNotNone(pair)
        assert pair is not None
        self.assertEqual(pair["long_symbol"], "AAAUSDT")
        self.assertEqual(pair["short_symbol"], "BBBUSDT")

    def test_halt_in_defensive(self):
        rows = [
            {"symbol": "AAAUSDT", "signal_type": "squeeze_long", "side": "LONG", "skew_score": 3.0},
            {"symbol": "BBBUSDT", "signal_type": "weak_short", "side": "SHORT", "skew_score": 2.5},
        ]
        pair = sk.pick_pair_trade(
            rows,
            {"regime": "defensive", "halt_new_pairs": True},
            sk.SkewConfig(min_skew_gap=0.0),
        )
        self.assertIsNone(pair)


class TestSkewSignalsLane(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "test.db"

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_record_skew_neutral_lane(self):
        def _init_db():
            conn = sqlite3.connect(str(self.db_path))
            return conn

        with mock.patch("accumulation_radar.init_db", side_effect=_init_db):
            ss.record_strategy_signal(
                lane=ss.LANE_SKEW_NEUTRAL,
                symbol="ethusdt",
                side="LONG",
                status="emitted",
                detail={"pair_id": "abc", "leg": "long"},
            )
            out = ss.list_strategy_signals(lane=ss.LANE_SKEW_NEUTRAL, limit=5)
        self.assertTrue(out["ok"])
        self.assertEqual(out["count"], 1)
        self.assertEqual(out["signals"][0]["lane"], "skew_neutral")


class TestSkewTables(unittest.TestCase):
    def test_ensure_tables(self):
        conn = sqlite3.connect(":memory:")
        sk.ensure_skew_tables(conn)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        self.assertIn("skew_signals", tables)
        self.assertIn("skew_pairs", tables)
        conn.close()


if __name__ == "__main__":
    unittest.main()
