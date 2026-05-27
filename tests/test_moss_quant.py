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

    def test_symbol_to_ccxt(self):
        from moss_quant.hyperliquid_klines import symbol_to_ccxt

        self.assertEqual(symbol_to_ccxt("BTCUSDT"), "BTC/USDC:USDC")
        self.assertEqual(symbol_to_ccxt("SOLUSDC"), "SOL/USDC:USDC")

    def test_hyperliquid_factory_csv_if_present(self):
        from moss_quant.hyperliquid_klines import _factory_csv_path, _load_factory_csv

        path = _factory_csv_path("BTCUSDT")
        if not path or not path.is_file():
            self.skipTest("factory BTC CSV not in workspace")
        df = _load_factory_csv("BTCUSDT", 100)
        self.assertGreaterEqual(len(df), 50)

    def test_default_data_source_hyperliquid(self):
        from moss_quant import config as cfg

        self.assertEqual(cfg.MOSS_QUANT_DATA_SOURCE, "hyperliquid")

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

    def test_daily_profile_name(self):
        from moss_quant.db import daily_profile_name

        self.assertEqual(daily_profile_name("btcusdt"), "daily-BTCUSDT")

    def test_parse_daily_optimize_utc(self):
        from moss_quant import config as cfg

        h, m = cfg.parse_daily_optimize_utc()
        self.assertGreaterEqual(h, 0)
        self.assertLessEqual(h, 23)
        self.assertGreaterEqual(m, 0)
        self.assertLessEqual(m, 59)

    def test_sync_daily_profiles_upsert(self):
        import json
        import sqlite3

        from moss_quant.db import (
            DAILY_PROFILE_SOURCE,
            daily_profile_name,
            get_daily_profile_by_symbol,
            migrate_moss_tables,
        )
        from moss_quant.daily_optimize_service import sync_daily_profiles

        conn = sqlite3.connect(":memory:")
        migrate_moss_tables(conn.cursor())
        now = "2024-01-01T00:00:00Z"
        cur = conn.execute(
            """INSERT INTO moss_daily_optimize_batches(
                   ran_at_utc, status, symbols_total, capital, data_source)
               VALUES (?,?,?,?,?)""",
            (now, "completed", 1, 10000.0, "hyperliquid"),
        )
        batch_id = int(cur.lastrowid)
        summary = {"total_return": 0.05, "total_trades": 3}
        conn.execute(
            """INSERT INTO moss_daily_optimize_items(
                   batch_id, symbol, template, tactical_params_json,
                   summary_json, score)
               VALUES (?,?,?,?,?,?)""",
            (
                batch_id,
                "BTCUSDT",
                "momentum",
                json.dumps({"entry_threshold": 0.35}),
                json.dumps(summary),
                0.05,
            ),
        )
        conn.commit()
        out = sync_daily_profiles(conn, batch_id)
        self.assertIn("BTCUSDT", out)
        prof = get_daily_profile_by_symbol(conn, "BTCUSDT")
        self.assertIsNotNone(prof)
        self.assertEqual(prof["name"], daily_profile_name("BTCUSDT"))
        self.assertEqual(prof["profile_source"], DAILY_PROFILE_SOURCE)
        self.assertTrue(prof["enabled"])
        conn.close()

    def test_max_active_profiles_default_23(self):
        from moss_quant import config as cfg

        self.assertEqual(cfg.MOSS_QUANT_MAX_ACTIVE_PROFILES, 23)

    def test_daily_optimize_defaults_on(self):
        from moss_quant import config as cfg

        self.assertTrue(cfg.MOSS_QUANT_DAILY_OPTIMIZE_ENABLED)
        self.assertTrue(cfg.MOSS_QUANT_DAILY_OPTIMIZE_BOOTSTRAP)
        self.assertTrue(cfg.MOSS_QUANT_DAILY_OPTIMIZE_APPLY_PROFILES)
        self.assertTrue(cfg.daily_optimize_scheduler_enabled())

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
