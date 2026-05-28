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
    is_research_symbol_allowed,
    is_symbol_allowed,
    list_universe,
    normalize_usdt_perp_symbol,
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

    def test_enrich_scan_details_with_positions(self):
        from moss_quant.paper_scanner import enrich_scan_details_with_positions

        details = [
            {
                "profile_id": 1,
                "label": "p1:SEIUSDT:trend",
                "action": "hold",
                "side": "LONG",
                "upnl": 0,
            },
        ]
        open_map = {
            1: {
                "entry_price": 0.3125,
                "mark_price": 0.318,
                "upnl": 176.0,
                "side": "LONG",
                "notional": 10000.0,
            }
        }
        out = enrich_scan_details_with_positions(details, open_map)
        self.assertEqual(out[0]["entry_price"], 0.3125)
        self.assertEqual(out[0]["upnl"], 176.0)
        self.assertIn("entry=0.3125", out[0]["message"])

    def test_enrich_wait_becomes_hold_when_position(self):
        from moss_quant.paper_scanner import enrich_scan_details_with_positions

        details = [
            {
                "profile_id": 2,
                "label": "p2:TIAUSDT:balanced",
                "action": "wait",
                "composite": 0.5,
            },
        ]
        open_map = {
            2: {
                "profile_id": 2,
                "side": "SHORT",
                "symbol": "TIAUSDT",
                "entry_price": 5.1,
                "mark_price": 5.0,
                "notional": 10000.0,
                "upnl": 196.08,
            }
        }
        out = enrich_scan_details_with_positions(details, open_map)
        self.assertEqual(out[0]["action"], "hold")
        self.assertEqual(out[0]["upnl"], 196.08)

    def test_fetch_open_positions_includes_profile_id(self):
        from accumulation_radar import init_db
        from moss_quant.db import _utc_now
        from moss_quant.paper_scanner import fetch_open_positions_map

        conn = init_db()
        conn.row_factory = __import__("sqlite3").Row
        now = _utc_now()
        conn.execute(
            """INSERT INTO moss_profiles(
                   name, symbol, template, enabled, profile_source,
                   initial_params_json, tactical_params_json,
                   virtual_equity_usdt, evolution_enabled, created_at_utc, updated_at_utc)
               VALUES ('t','TESTUSDT','balanced',1,'manual','{}','{}',10000,0,?,?)""",
            (now, now),
        )
        pid = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        conn.execute(
            """INSERT INTO moss_signals(
                   profile_id, recorded_at_utc, side, symbol, entry_price,
                   virtual_notional_usdt, mark_price, composite, regime,
                   unrealized_pnl_usdt, updated_at_utc)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (pid, now, "LONG", "TESTUSDT", 1.0, 10000, 1.1, 0.5, "BULL", 1000, now),
        )
        conn.commit()
        try:
            m = fetch_open_positions_map(conn)
            self.assertIn(pid, m)
            self.assertEqual(m[pid]["profile_id"], pid)
        finally:
            conn.execute("DELETE FROM moss_signals WHERE profile_id=?", (pid,))
            conn.execute("DELETE FROM moss_profiles WHERE id=?", (pid,))
            conn.commit()
            conn.close()

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

    def test_normalize_usdt_perp_symbol(self):
        self.assertEqual(normalize_usdt_perp_symbol("xvg"), "XVGUSDT")
        self.assertEqual(normalize_usdt_perp_symbol("XVG/USDT"), "XVGUSDT")

    def test_research_symbol_default_accepts_any_usdt_format(self):
        self.assertTrue(is_research_symbol_allowed("XVGUSDT"))
        self.assertTrue(is_research_symbol_allowed("xvg"))
        self.assertFalse(is_research_symbol_allowed(""))

    def test_research_symbol_strict_uses_binance_perp_filter(self):
        from unittest.mock import patch

        from moss_quant import config as cfg

        with patch.object(cfg, "MOSS_QUANT_RESEARCH_RELAX_SYMBOL_CHECK", False):
            with patch(
                "moss_quant.universe.filter_symbols_to_binance_usdt_perps",
                return_value=["XVGUSDT"],
            ) as m:
                self.assertTrue(is_research_symbol_allowed("xvg"))
                m.assert_called_with(["XVGUSDT"])
            with patch(
                "moss_quant.universe.filter_symbols_to_binance_usdt_perps",
                return_value=[],
            ):
                self.assertFalse(is_research_symbol_allowed("XVGUSDT"))

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

    def test_reconcile_stale_daily_batches(self):
        import sqlite3

        from moss_quant.db import migrate_moss_tables
        from moss_quant.daily_optimize_service import (
            is_daily_batch_running,
            reconcile_stale_daily_batches,
        )

        conn = sqlite3.connect(":memory:")
        migrate_moss_tables(conn.cursor())
        conn.execute(
            """INSERT INTO moss_daily_optimize_batches(
                   ran_at_utc, status, symbols_total, capital, data_source)
               VALUES (?,?,?,?,?)""",
            ("2024-01-01T00:00:00Z", "running", 2, 10000.0, "hyperliquid"),
        )
        conn.commit()
        self.assertTrue(is_daily_batch_running(conn))
        n = reconcile_stale_daily_batches(conn)
        self.assertEqual(n, 1)
        self.assertFalse(is_daily_batch_running(conn))
        row = conn.execute(
            "SELECT status, error FROM moss_daily_optimize_batches WHERE id=1"
        ).fetchone()
        self.assertEqual(row[0], "failed")
        self.assertIn("中断", row[1])
        conn.close()

    def test_annotate_daily_batch_items(self):
        import json
        import sqlite3

        from moss_quant.db import get_profile_by_symbol, migrate_moss_tables
        from moss_quant.daily_optimize_service import annotate_daily_batch_items

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
        summary = {"total_return": 0.05, "total_trades": 3, "win_rate": 0.4}
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
        stats = annotate_daily_batch_items(conn, batch_id)
        self.assertEqual(stats["fail"], 1)
        self.assertIsNone(get_profile_by_symbol(conn, "BTCUSDT"))
        item = conn.execute(
            "SELECT summary_json FROM moss_daily_optimize_items WHERE batch_id=?",
            (batch_id,),
        ).fetchone()
        stored = json.loads(item[0])
        self.assertFalse(stored["auto_enabled"])
        self.assertIn("回合", stored["auto_enable_reason"])
        conn.close()

    def test_evaluate_profile_auto_enable_pass(self):
        from moss_quant.daily_auto_enable import evaluate_profile_auto_enable

        gate = evaluate_profile_auto_enable(
            {
                "total_return": 0.32,
                "total_trades": 20,
                "win_rate": 0.55,
                "max_drawdown": 0.12,
                "blowup_count": 0,
            }
        )
        self.assertTrue(gate["auto_enabled"])
        self.assertEqual(gate["auto_enable_label"], "开")

    def test_evaluate_profile_auto_enable_fail_return(self):
        from moss_quant.daily_auto_enable import evaluate_profile_auto_enable

        gate = evaluate_profile_auto_enable(
            {"total_return": -0.05, "total_trades": 20, "max_drawdown": 0.1}
        )
        self.assertFalse(gate["auto_enabled"])

    def test_max_active_profiles_matches_universe(self):
        from moss_quant import config as cfg
        from moss_quant.universe import moss_catalog_bases

        self.assertEqual(cfg.MOSS_QUANT_MAX_ACTIVE_PROFILES, len(moss_catalog_bases()))

    def test_binance_kline_weight_for_1500(self):
        from binance_fapi import kline_request_weight

        self.assertEqual(kline_request_weight(1500), 10)
        self.assertEqual(kline_request_weight(100), 1)

    def test_universe_includes_new_alts(self):
        from moss_quant.universe import list_universe

        syms = {u["base"] for u in list_universe()}
        for base in ("TON", "PEPE", "ENA", "OP", "SUI"):
            self.assertIn(base, syms, msg=f"missing {base}")

    def test_daily_optimize_defaults_on(self):
        from moss_quant import config as cfg

        self.assertTrue(cfg.MOSS_QUANT_DAILY_OPTIMIZE_ENABLED)
        self.assertFalse(cfg.MOSS_QUANT_DAILY_OPTIMIZE_BOOTSTRAP)
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
