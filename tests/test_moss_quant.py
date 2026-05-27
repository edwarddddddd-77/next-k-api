"""Moss 量化基础单测。"""

import unittest

import pandas as pd

from moss_quant.params import build_initial_params, lock_personality
from moss_quant.core.decision import (
    DecisionParams,
    compute_last_composite,
    compute_signals,
)
from moss_quant.core.regime import classify_regime
from moss_quant.universe import (
    active_symbols_taken,
    base_to_binance_symbol,
    list_universe,
)


class TestMossQuant(unittest.TestCase):
    def test_momentum_template_weights_sum(self):
        p = build_initial_params(template="momentum")
        s = (
            p["trend_weight"]
            + p["momentum_weight"]
            + p["mean_revert_weight"]
            + p["volume_weight"]
            + p["volatility_weight"]
        )
        self.assertAlmostEqual(s, 1.0, places=5)
        self.assertGreater(p["momentum_weight"], p["mean_revert_weight"])

    def test_lock_personality(self):
        initial = build_initial_params(template="balanced")
        tweaked = dict(initial)
        tweaked["momentum_weight"] = 0.99
        tweaked["entry_threshold"] = 0.5
        locked = lock_personality(tweaked, initial)
        self.assertEqual(locked["momentum_weight"], initial["momentum_weight"])
        self.assertEqual(locked["entry_threshold"], 0.5)

    def test_compute_signals_on_synthetic(self):
        n = 120
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
                "open": [100 + i * 0.1 for i in range(n)],
                "high": [101 + i * 0.1 for i in range(n)],
                "low": [99 + i * 0.1 for i in range(n)],
                "close": [100 + i * 0.1 for i in range(n)],
                "volume": [1000.0] * n,
            }
        )
        params = DecisionParams.from_dict(build_initial_params(template="momentum"))
        regime = classify_regime(df, version="v1")
        sig = compute_signals(df, params, regime)
        self.assertEqual(len(sig), n)
        self.assertTrue(sig.isin([-1, 0, 1]).all())

    def test_universe_has_btc(self):
        syms = {u["symbol"] for u in list_universe()}
        self.assertIn(base_to_binance_symbol("BTC"), syms)

    def test_compute_last_composite_bounded(self):
        n = 120
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
                "open": [100.0] * n,
                "high": [101.0] * n,
                "low": [99.0] * n,
                "close": [100.0] * n,
                "volume": [1000.0] * n,
            }
        )
        params = DecisionParams.from_dict(build_initial_params(template="balanced"))
        regime = classify_regime(df, version="v1")
        c = compute_last_composite(df, params, regime)
        self.assertGreaterEqual(c, -2.0)
        self.assertLessEqual(c, 2.0)

    def test_active_symbols_taken_from_universe_module(self):
        import sqlite3

        conn = sqlite3.connect(":memory:")
        conn.execute(
            """CREATE TABLE moss_profiles (
                id INTEGER PRIMARY KEY, symbol TEXT, enabled INTEGER)"""
        )
        conn.execute(
            "INSERT INTO moss_profiles(symbol, enabled) VALUES ('BTCUSDT', 1)"
        )
        conn.execute(
            "INSERT INTO moss_profiles(symbol, enabled) VALUES ('ETHUSDT', 0)"
        )
        taken = active_symbols_taken(conn)
        self.assertEqual(taken, {"BTCUSDT"})
        conn.close()

    def test_extract_tactical_params_locks_personality(self):
        from moss_quant.params import build_initial_params, extract_tactical_params

        initial = build_initial_params(template="momentum")
        final = dict(initial)
        final["entry_threshold"] = 0.36
        final["momentum_weight"] = 0.99
        tactical = extract_tactical_params(final, initial)
        self.assertEqual(tactical["entry_threshold"], 0.36)
        self.assertNotIn("momentum_weight", tactical)

    def test_reflect_extract_json_array(self):
        from moss_quant.reflect import _extract_json_array

        raw = '说明\n```json\n[{"round": 1, "params": {"entry_threshold": 0.35}}]\n```'
        arr = _extract_json_array(raw)
        self.assertEqual(len(arr), 1)
        self.assertEqual(arr[0]["round"], 1)

    def test_format_scan_detail_message_wait(self):
        from moss_quant.paper_scanner import format_scan_detail_message

        label = "p1:BTCUSDT:balanced"
        msg = format_scan_detail_message(
            label,
            {
                "action": "wait",
                "composite": 0.12,
                "entry_threshold": 0.4,
                "regime": "SIDEWAYS",
                "reason": "below_threshold",
            },
        )
        self.assertIn("[moss]", msg)
        self.assertIn("WAIT", msg)
        self.assertIn("SIDEWAYS", msg)
        self.assertIn(label, msg)

    def test_optimize_returns_ranking(self):
        from moss_quant.optimize_service import run_strategy_optimize

        out = run_strategy_optimize(
            symbol="BTCUSDT",
            capital=10000,
            refresh_klines=False,
            max_combinations=8,
            entry_thresholds=[0.40, 0.50],
            sl_atr_mults=[2.0],
            tp_rr_ratios=[2.0, 3.0],
            top_n=5,
        )
        self.assertTrue(out.get("ok"))
        self.assertGreaterEqual(out.get("combinations_tested", 0), 1)
        self.assertIn("ranking", out)
        if out.get("combinations_ok", 0) > 0:
            self.assertIsNotNone(out.get("best"))

    def test_delete_profile_blocks_open_position(self):
        import sqlite3

        from moss_quant.db import delete_profile, migrate_moss_tables

        conn = sqlite3.connect(":memory:")
        migrate_moss_tables(conn.cursor())
        now = "2024-01-01T00:00:00Z"
        conn.execute(
            """INSERT INTO moss_profiles(
                   id, name, symbol, template, enabled, initial_params_json,
                   tactical_params_json, created_at_utc, updated_at_utc)
               VALUES (1, 't', 'BTCUSDT', 'balanced', 1, '{}', '{}', ?, ?)""",
            (now, now),
        )
        conn.execute(
            """INSERT INTO moss_signals(profile_id, recorded_at_utc, side, symbol, entry_price)
               VALUES (1, '2024-01-01T00:00:00Z', 'LONG', 'BTCUSDT', 1.0)"""
        )
        with self.assertRaises(ValueError):
            delete_profile(conn, 1)
        conn.close()


if __name__ == "__main__":
    unittest.main()
