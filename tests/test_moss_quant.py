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

    def test_spot_sizing_defaults_no_leverage(self):
        p = build_initial_params(template="balanced")
        self.assertEqual(p["base_leverage"], 1.0)
        self.assertEqual(p["max_leverage"], 1.0)
        self.assertEqual(p["risk_per_trade"], 1.0)
        self.assertEqual(p["max_position_pct"], 1.0)

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

    def test_sync_enabled_profiles_from_batch(self):
        import json
        import sqlite3

        from moss_quant.db import migrate_moss_tables
        from moss_quant.daily_optimize_service import sync_enabled_profiles_from_batch

        conn = sqlite3.connect(":memory:")
        migrate_moss_tables(conn.cursor())
        now = "2024-01-01T00:00:00Z"
        cur = conn.execute(
            """INSERT INTO moss_daily_optimize_batches(
                   ran_at_utc, status, symbols_total, capital, data_source)
               VALUES (?,?,?,?,?)""",
            (now, "completed", 2, 10000.0, "hyperliquid"),
        )
        batch_id = int(cur.lastrowid)
        conn.execute(
            """INSERT INTO moss_daily_optimize_items(
                   batch_id, symbol, template, tactical_params_json,
                   summary_json, score)
               VALUES (?,?,?,?,?,?)""",
            (
                batch_id,
                "HYPEUSDT",
                "momentum",
                json.dumps({"entry_threshold": 0.48, "sl_atr_mult": 2.5}),
                json.dumps(
                    {
                        "total_return": 0.5,
                        "total_trades": 20,
                        "max_drawdown": -0.08,
                        "blowup_count": 0,
                        "win_rate": 0.55,
                        "validation_passed": True,
                        "validation_reason": "验证通过",
                        "val_return": 0.1,
                    }
                ),
                0.5,
            ),
        )
        conn.execute(
            """INSERT INTO moss_daily_optimize_items(
                   batch_id, symbol, template, tactical_params_json,
                   summary_json, score)
               VALUES (?,?,?,?,?,?)""",
            (
                batch_id,
                "BTCUSDT",
                "momentum",
                json.dumps({"entry_threshold": 0.48, "sl_atr_mult": 2.5}),
                json.dumps(
                    {
                        "total_return": 0.2,
                        "total_trades": 12,
                        "max_drawdown": -0.1,
                        "blowup_count": 0,
                        "win_rate": 0.5,
                        "validation_passed": True,
                        "validation_reason": "验证通过",
                    }
                ),
                0.2,
            ),
        )
        conn.execute(
            """INSERT INTO moss_profiles(
                   name, symbol, template, enabled, profile_source,
                   initial_params_json, tactical_params_json,
                   virtual_equity_usdt, evolution_enabled,
                   created_at_utc, updated_at_utc)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                "hype",
                "HYPEUSDT",
                "balanced",
                1,
                "manual",
                json.dumps({"entry_threshold": 0.4}),
                json.dumps({"entry_threshold": 0.4}),
                10000.0,
                0,
                now,
                now,
            ),
        )
        conn.execute(
            """INSERT INTO moss_profiles(
                   name, symbol, template, enabled, profile_source,
                   initial_params_json, tactical_params_json,
                   virtual_equity_usdt, evolution_enabled,
                   created_at_utc, updated_at_utc)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                "btc",
                "BTCUSDT",
                "momentum",
                1,
                "manual",
                json.dumps({}),
                json.dumps({"entry_threshold": 0.48, "sl_atr_mult": 2.5}),
                10000.0,
                0,
                now,
                now,
            ),
        )
        conn.commit()
        stats = sync_enabled_profiles_from_batch(
            conn, batch_id, trigger_paper_scan=False
        )
        self.assertEqual(stats["updated"], 1)
        self.assertEqual(stats["already_optimal"], 1)
        row = conn.execute(
            "SELECT template, tactical_params_json FROM moss_profiles WHERE symbol='HYPEUSDT'"
        ).fetchone()
        self.assertEqual(row[0], "momentum")
        tact = json.loads(row[1])
        self.assertAlmostEqual(tact["entry_threshold"], 0.48)
        self.assertAlmostEqual(tact["sl_atr_mult"], 2.5)

    def test_sync_profiles_with_open_position_when_disabled(self):
        import json
        import sqlite3

        from moss_quant.db import migrate_moss_tables
        from moss_quant.daily_optimize_service import sync_enabled_profiles_from_batch

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
        conn.execute(
            """INSERT INTO moss_daily_optimize_items(
                   batch_id, symbol, template, tactical_params_json,
                   summary_json, score)
               VALUES (?,?,?,?,?,?)""",
            (
                batch_id,
                "SOLUSDT",
                "trend",
                json.dumps({"entry_threshold": 0.44}),
                json.dumps(
                    {
                        "total_return": 0.2,
                        "total_trades": 10,
                        "max_drawdown": -0.1,
                        "blowup_count": 0,
                        "win_rate": 0.5,
                        "validation_passed": True,
                        "validation_reason": "验证通过",
                    }
                ),
                0.2,
            ),
        )
        cur = conn.execute(
            """INSERT INTO moss_profiles(
                   name, symbol, template, enabled, profile_source,
                   initial_params_json, tactical_params_json,
                   virtual_equity_usdt, evolution_enabled,
                   created_at_utc, updated_at_utc)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                "sol",
                "SOLUSDT",
                "balanced",
                0,
                "manual",
                json.dumps({}),
                json.dumps({"entry_threshold": 0.4}),
                10000.0,
                0,
                now,
                now,
            ),
        )
        pid = int(cur.lastrowid)
        conn.execute(
            """INSERT INTO moss_signals(
                   profile_id, recorded_at_utc, side, symbol, entry_price,
                   virtual_notional_usdt, mark_price, composite, regime,
                   unrealized_pnl_usdt, updated_at_utc)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (pid, now, "LONG", "SOLUSDT", 100.0, 1000.0, 100.0, 0.5, "UP", 0.0, now),
        )
        conn.commit()
        stats = sync_enabled_profiles_from_batch(
            conn, batch_id, trigger_paper_scan=False
        )
        self.assertEqual(stats["updated"], 1)
        self.assertEqual(stats["updated_with_open_position"], 1)
        row = conn.execute(
            "SELECT template, tactical_params_json, enabled FROM moss_profiles WHERE id=?",
            (pid,),
        ).fetchone()
        self.assertEqual(row[0], "trend")
        self.assertAlmostEqual(json.loads(row[1])["entry_threshold"], 0.44)
        self.assertEqual(row[2], 0)

    def test_mcap_candidates_exclude_symbol_added_to_daily_core_db(self):
        import sqlite3
        from unittest.mock import patch

        from moss_quant.binance_mcap_universe import build_mcap_scan_candidates
        from moss_quant.db import add_symbol_to_daily_core, migrate_moss_tables

        conn = sqlite3.connect(":memory:")
        migrate_moss_tables(conn.cursor())
        conn.commit()
        add_symbol_to_daily_core(conn, "RENDERUSDT", note="test")
        mcap = {
            "RENDER": 5e9,
            "BTC": 1e12,
            "ETH": 4e11,
            "SOL": 8e10,
        }
        with patch("accumulation_radar.init_db", return_value=conn):
            candidates = build_mcap_scan_candidates(
                mcap_limit=10, mcap_map=mcap
            )
        bases = {c["base"] for c in candidates}
        syms = {c["symbol"] for c in candidates}
        self.assertNotIn("RENDER", bases)
        self.assertNotIn("RENDERUSDT", syms)

    def test_build_mcap_scan_candidates_excludes_daily_and_stables(self):
        from moss_quant.binance_mcap_universe import build_mcap_scan_candidates
        from moss_quant.universe import list_universe

        daily_bases = {u["base"] for u in list_universe()}
        mcap = {
            "USDC": 9e10,
            "USDT": 8e10,
            "BTC": 1e12,
            "ETH": 4e11,
        }

        candidates = build_mcap_scan_candidates(
            mcap_limit=10, mcap_map=mcap
        )
        bases = {c["base"] for c in candidates}
        self.assertNotIn("USDC", bases)
        self.assertNotIn("USDT", bases)
        for b in daily_bases:
            self.assertNotIn(b, bases)
        syms = {c["symbol"] for c in candidates}
        for u in list_universe():
            self.assertNotIn(u["symbol"], syms)

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

    def test_max_active_profiles_default_five(self):
        from moss_quant import config as cfg

        self.assertEqual(cfg.MOSS_QUANT_MAX_ACTIVE_PROFILES, 5)
        self.assertEqual(cfg.MOSS_QUANT_POOL_MAX_AUTO_ENABLED, 5)
        self.assertTrue(cfg.pool_governance_enabled())

    def test_pool_governance_auto_disable_c_tier(self):
        import json
        import sqlite3

        from moss_quant.db import migrate_moss_tables
        from moss_quant.pool_governance import apply_pool_governance

        conn = sqlite3.connect(":memory:")
        migrate_moss_tables(conn.cursor())
        conn.commit()
        now = "2024-01-01T00:00:00Z"
        cur = conn.execute(
            """INSERT INTO moss_daily_optimize_batches(
                   ran_at_utc, status, symbols_total, capital, data_source)
               VALUES (?,?,?,?,?)""",
            (now, "completed", 1, 10000.0, "hyperliquid"),
        )
        batch_id = int(cur.lastrowid)
        conn.execute(
            """INSERT INTO moss_daily_optimize_items(
                   batch_id, symbol, template, tactical_params_json,
                   summary_json, score)
               VALUES (?,?,?,?,?,?)""",
            (
                batch_id,
                "NEARUSDT",
                "trend",
                json.dumps({"entry_threshold": 0.44}),
                json.dumps(
                    {
                        "total_return": -0.05,
                        "total_trades": 10,
                        "max_drawdown": -0.1,
                        "blowup_count": 0,
                        "win_rate": 0.4,
                    }
                ),
                -999.0,
            ),
        )
        conn.execute(
            """INSERT INTO moss_profiles(
                   name, symbol, template, enabled, profile_source,
                   initial_params_json, tactical_params_json,
                   virtual_equity_usdt, evolution_enabled,
                   created_at_utc, updated_at_utc)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                "near",
                "NEARUSDT",
                "trend",
                1,
                "manual",
                json.dumps({}),
                json.dumps({}),
                10000.0,
                0,
                now,
                now,
            ),
        )
        conn.commit()
        stats = apply_pool_governance(conn, batch_id, trigger_paper_scan=False)
        self.assertEqual(stats["disabled"], 1)
        row = conn.execute(
            "SELECT enabled FROM moss_profiles WHERE symbol='NEARUSDT'"
        ).fetchone()
        self.assertEqual(int(row[0]), 0)

    def test_pool_governance_auto_add_a_pool(self):
        import json
        import sqlite3
        from unittest.mock import patch

        from moss_quant import config as cfg
        from moss_quant.db import migrate_moss_tables
        from moss_quant.pool_governance import apply_pool_governance

        conn = sqlite3.connect(":memory:")
        migrate_moss_tables(conn.cursor())
        conn.commit()
        now = "2024-01-01T00:00:00Z"
        cur = conn.execute(
            """INSERT INTO moss_daily_optimize_batches(
                   ran_at_utc, status, symbols_total, capital, data_source)
               VALUES (?,?,?,?,?)""",
            (now, "completed", 1, 10000.0, "hyperliquid"),
        )
        batch_id = int(cur.lastrowid)
        good_summary = {
            "total_return": 0.2,
            "total_trades": 12,
            "max_drawdown": -0.08,
            "blowup_count": 0,
            "win_rate": 0.55,
            "validation_passed": True,
            "validation_reason": "验证通过",
            "val_return": 0.05,
        }
        conn.execute(
            """INSERT INTO moss_daily_optimize_items(
                   batch_id, symbol, template, tactical_params_json,
                   summary_json, score)
               VALUES (?,?,?,?,?,?)""",
            (
                batch_id,
                "HYPEUSDT",
                "momentum",
                json.dumps({"entry_threshold": 0.48, "sl_atr_mult": 2.5}),
                json.dumps(good_summary),
                0.9,
            ),
        )
        conn.commit()
        with patch.object(cfg, "MOSS_QUANT_POOL_UPGRADE_STREAK", 1):
            stats = apply_pool_governance(conn, batch_id, trigger_paper_scan=False)
        self.assertEqual(stats["added"], 1)
        self.assertEqual(stats["enabled_auto"], 1)
        row = conn.execute(
            "SELECT enabled, profile_source FROM moss_profiles WHERE symbol='HYPEUSDT'"
        ).fetchone()
        self.assertEqual(int(row[0]), 1)
        self.assertEqual(row[1], "governance_auto")

    def test_pool_governance_manual_lock_blocks_auto_enable(self):
        import json
        import sqlite3
        from unittest.mock import patch

        from moss_quant import config as cfg
        from moss_quant.db import migrate_moss_tables
        from moss_quant.pool_governance import apply_pool_governance

        conn = sqlite3.connect(":memory:")
        migrate_moss_tables(conn.cursor())
        conn.commit()
        now = "2024-01-01T00:00:00Z"
        cur = conn.execute(
            """INSERT INTO moss_daily_optimize_batches(
                   ran_at_utc, status, symbols_total, capital, data_source)
               VALUES (?,?,?,?,?)""",
            (now, "completed", 1, 10000.0, "hyperliquid"),
        )
        batch_id = int(cur.lastrowid)
        good_summary = {
            "total_return": 0.2,
            "total_trades": 12,
            "max_drawdown": -0.08,
            "blowup_count": 0,
            "win_rate": 0.55,
            "validation_passed": True,
            "validation_reason": "验证通过",
            "val_return": 0.05,
        }
        conn.execute(
            """INSERT INTO moss_daily_optimize_items(
                   batch_id, symbol, template, tactical_params_json,
                   summary_json, score)
               VALUES (?,?,?,?,?,?)""",
            (
                batch_id,
                "HYPEUSDT",
                "momentum",
                json.dumps({"entry_threshold": 0.48}),
                json.dumps(good_summary),
                0.9,
            ),
        )
        conn.execute(
            """INSERT INTO moss_profiles(
                   name, symbol, template, enabled, profile_source,
                   initial_params_json, tactical_params_json,
                   virtual_equity_usdt, evolution_enabled,
                   governance_manual_lock, created_at_utc, updated_at_utc)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                "hype",
                "HYPEUSDT",
                "momentum",
                0,
                "manual",
                json.dumps({}),
                json.dumps({}),
                10000.0,
                0,
                1,
                now,
                now,
            ),
        )
        conn.commit()
        with patch.object(cfg, "MOSS_QUANT_POOL_UPGRADE_STREAK", 1):
            stats = apply_pool_governance(conn, batch_id, trigger_paper_scan=False)
        self.assertEqual(stats["skipped_manual_lock"], 1)
        self.assertEqual(stats["enabled_auto"], 0)
        row = conn.execute(
            "SELECT enabled FROM moss_profiles WHERE symbol='HYPEUSDT'"
        ).fetchone()
        self.assertEqual(int(row[0]), 0)

    def test_binance_kline_weight_for_1500(self):
        from binance_fapi import kline_request_weight

        self.assertEqual(kline_request_weight(1500), 10)
        self.assertEqual(kline_request_weight(100), 1)

    def test_universe_is_daily_core_25_by_default(self):
        from moss_quant.universe import MOSS_DAILY_CORE_BASES, list_universe

        syms = {u["base"] for u in list_universe()}
        self.assertEqual(len(syms), len(MOSS_DAILY_CORE_BASES))
        self.assertEqual(len(MOSS_DAILY_CORE_BASES), 25)
        for base in ("BTC", "ETH", "HYPE", "ARB", "SUI", "ICP", "TON"):
            self.assertIn(base, syms, msg=f"missing {base}")
        for base in ("PEPE",):
            self.assertNotIn(base, syms, msg=f"extended {base} should be excluded")

    def test_top_qualified_mcap_items_filters_and_limits(self):
        from moss_quant.mcap_scan_service import top_qualified_mcap_items

        items = [
            {
                "symbol": "AAAUSDT",
                "score": 10,
                "summary": {
                    "total_return": 0.2,
                    "total_trades": 10,
                    "max_drawdown": 0.1,
                    "blowup_count": 0,
                    "auto_enabled": True,
                },
            },
            {
                "symbol": "BBBUSDT",
                "score": 20,
                "summary": {
                    "total_return": -0.1,
                    "total_trades": 10,
                    "max_drawdown": 0.1,
                    "blowup_count": 0,
                    "auto_enabled": False,
                },
            },
            {
                "symbol": "CCCUSDT",
                "score": 5,
                "summary": {
                    "total_return": 0.5,
                    "total_trades": 12,
                    "max_drawdown": 0.2,
                    "blowup_count": 0,
                },
            },
        ]
        top = top_qualified_mcap_items(items, 15)
        self.assertEqual(len(top), 2)
        self.assertEqual(top[0]["symbol"], "AAAUSDT")
        self.assertEqual(top[1]["symbol"], "CCCUSDT")

    def test_add_symbol_to_daily_core(self):
        import sqlite3

        from moss_quant.db import (
            add_symbol_to_daily_core,
            list_daily_core_symbols,
            migrate_moss_tables,
        )

        conn = sqlite3.connect(":memory:")
        migrate_moss_tables(conn.cursor())
        conn.commit()
        out = add_symbol_to_daily_core(conn, "RENDERUSDT", note="test")
        self.assertTrue(out["added"])
        rows = list_daily_core_symbols(conn)
        syms = {r["symbol"] for r in rows if int(r.get("enabled") or 0)}
        self.assertIn("RENDERUSDT", syms)
        out2 = add_symbol_to_daily_core(conn, "RENDERUSDT")
        self.assertFalse(out2.get("added"))
        self.assertTrue(out2.get("already_in_daily_core"))

    def test_daily_core_symbol_allowed_for_paper_profile(self):
        import sqlite3

        from moss_quant.db import add_symbol_to_daily_core, migrate_moss_tables

        conn = sqlite3.connect(":memory:")
        migrate_moss_tables(conn.cursor())
        conn.commit()
        self.assertFalse(is_symbol_allowed("ZECUSDT", conn=conn))
        add_symbol_to_daily_core(conn, "ZECUSDT", note="from_mcap_scan")
        self.assertTrue(is_symbol_allowed("ZECUSDT", conn=conn))
        merged = list_universe(conn)
        self.assertIn("ZECUSDT", {u["symbol"] for u in merged})

    def test_daily_core_symbols_table_seeded(self):
        import sqlite3

        from moss_quant.db import list_daily_core_bases, migrate_moss_tables
        from moss_quant.universe import MOSS_DAILY_CORE_BASES

        conn = sqlite3.connect(":memory:")
        migrate_moss_tables(conn.cursor())
        conn.commit()
        bases = list_daily_core_bases(conn)
        self.assertEqual(len(bases), len(MOSS_DAILY_CORE_BASES))
        self.assertEqual(set(bases), set(MOSS_DAILY_CORE_BASES))

    def test_daily_optimize_defaults_on(self):
        from moss_quant import config as cfg

        self.assertTrue(cfg.MOSS_QUANT_DAILY_OPTIMIZE_ENABLED)
        self.assertFalse(cfg.MOSS_QUANT_DAILY_OPTIMIZE_BOOTSTRAP)
        self.assertTrue(cfg.MOSS_QUANT_DAILY_OPTIMIZE_APPLY_PROFILES)
        self.assertTrue(cfg.daily_optimize_scheduler_enabled())

    def test_delete_profile_preserves_settlements_and_wallet_pnl(self):
        import sqlite3

        from moss_quant.db import (
            delete_profile,
            get_moss_wallet,
            migrate_moss_tables,
        )

        conn = sqlite3.connect(":memory:")
        migrate_moss_tables(conn.cursor())
        now = "2024-01-01T00:00:00Z"
        conn.execute(
            """INSERT INTO moss_profiles(
                   id, name, symbol, template, enabled, initial_params_json,
                   tactical_params_json, created_at_utc, updated_at_utc)
               VALUES (1, 'icp', 'ICPUSDT', 'balanced', 1, '{}', '{}', ?, ?)""",
            (now, now),
        )
        conn.execute(
            """INSERT INTO moss_signals(
                   id, profile_id, recorded_at_utc, side, symbol,
                   outcome, outcome_at_utc, pnl_usdt, updated_at_utc)
               VALUES (10, 1, ?, 'LONG', 'ICPUSDT', 'win', ?, 100.5, ?)""",
            (now, now, now),
        )
        conn.execute(
            """INSERT INTO moss_settlements(
                   settled_at_utc, signal_id, profile_id, symbol, side,
                   outcome, pnl_usdt)
               VALUES (?, 10, 1, 'ICPUSDT', 'LONG', 'win', 100.5)""",
            (now,),
        )
        conn.commit()
        deleted = delete_profile(conn, 1)
        conn.commit()
        self.assertIsNotNone(deleted)
        self.assertEqual(deleted["settlements_preserved"], 1)
        self.assertEqual(deleted["signals_preserved"], 1)
        n_set = conn.execute("SELECT COUNT(*) FROM moss_settlements").fetchone()[0]
        n_sig = conn.execute("SELECT COUNT(*) FROM moss_signals").fetchone()[0]
        self.assertEqual(n_set, 1)
        self.assertEqual(n_sig, 1)
        wallet = get_moss_wallet(conn)
        self.assertAlmostEqual(wallet["realized_pnl_usdt"], 100.5, places=2)
        conn.close()

    def test_backfill_settlements_from_closed_signals(self):
        import sqlite3

        from moss_quant.db import (
            backfill_settlements_from_closed_signals,
            get_moss_wallet,
            migrate_moss_tables,
        )

        conn = sqlite3.connect(":memory:")
        migrate_moss_tables(conn.cursor())
        now = "2024-01-01T00:00:00Z"
        conn.execute(
            """INSERT INTO moss_profiles(
                   id, name, symbol, template, enabled, initial_params_json,
                   tactical_params_json, created_at_utc, updated_at_utc)
               VALUES (1, 'ton', 'TONUSDT', 'balanced', 1, '{}', '{}', ?, ?)""",
            (now, now),
        )
        conn.execute(
            """INSERT INTO moss_signals(
                   profile_id, recorded_at_utc, side, symbol,
                   outcome, outcome_at_utc, pnl_usdt, updated_at_utc)
               VALUES (1, ?, 'LONG', 'TONUSDT', 'win', ?, 42.0, ?)""",
            (now, now, now),
        )
        conn.commit()
        out = backfill_settlements_from_closed_signals(conn)
        self.assertEqual(out["inserted"], 1)
        wallet = get_moss_wallet(conn, reconcile=False)
        self.assertAlmostEqual(wallet["realized_pnl_usdt"], 42.0, places=2)
        conn.close()

    def test_backfill_settlements_from_paper_runs(self):
        import json
        import sqlite3

        from moss_quant.db import (
            backfill_settlements_from_paper_runs,
            get_moss_wallet,
            migrate_moss_tables,
        )

        conn = sqlite3.connect(":memory:")
        migrate_moss_tables(conn.cursor())
        now = "2024-01-02T00:00:00Z"
        detail = [
            {
                "profile_id": 99,
                "symbol": "ICPUSDT",
                "action": "close",
                "side": "LONG",
                "pnl": 77.25,
                "rule": "tp",
            }
        ]
        conn.execute(
            """INSERT INTO moss_paper_runs(ran_at_utc, profiles_scanned, opens, closes, detail_json)
               VALUES (?, 1, 0, 1, ?)""",
            (now, json.dumps(detail)),
        )
        conn.commit()
        out = backfill_settlements_from_paper_runs(conn)
        self.assertEqual(out["inserted"], 1)
        self.assertAlmostEqual(out["pnl_usdt"], 77.25, places=2)
        wallet = get_moss_wallet(conn, reconcile=False)
        self.assertAlmostEqual(wallet["realized_pnl_usdt"], 77.25, places=2)
        conn.close()

    def test_paper_sizing_matches_factory_free_margin(self):
        import sqlite3

        from moss_quant.db import migrate_moss_tables, profile_wallet_balance
        from moss_quant.paper_scanner import (
            _notional_for_profile,
            _open_notional_from_free_margin,
        )

        conn = sqlite3.connect(":memory:")
        migrate_moss_tables(conn.cursor())
        now = "2024-01-01T00:00:00Z"
        conn.execute(
            """INSERT INTO moss_profiles(
                   id, name, symbol, template, enabled, initial_params_json,
                   tactical_params_json, virtual_equity_usdt, created_at_utc, updated_at_utc)
               VALUES (1, 't', 'BTCUSDT', 'balanced', 1, '{}', '{}', 10000, ?, ?)""",
            (now, now),
        )
        conn.commit()
        params = {
            "base_leverage": 1,
            "max_leverage": 1,
            "risk_per_trade": 1.0,
            "max_position_pct": 1.0,
        }
        cap = float(__import__("moss_quant.config", fromlist=["MOSS_QUANT_PROFILE_CAPITAL"]).MOSS_QUANT_PROFILE_CAPITAL)
        self.assertAlmostEqual(
            _open_notional_from_free_margin(cap, params), cap, places=2
        )
        self.assertAlmostEqual(
            _notional_for_profile(conn, 1, params, leverage=1), cap, places=2
        )
        conn.execute(
            """INSERT INTO moss_settlements(
                   settled_at_utc, signal_id, profile_id, symbol, side,
                   outcome, pnl_usdt)
               VALUES (?, 1, 1, 'BTCUSDT', 'LONG', 'win', -200)""",
            (now,),
        )
        conn.commit()
        bal = profile_wallet_balance(conn, 1)
        self.assertAlmostEqual(bal, cap - 200.0, places=2)
        self.assertAlmostEqual(
            _notional_for_profile(conn, 1, params, leverage=1), cap - 200.0, places=2
        )
        conn.close()

    def test_wallet_and_profile_capital_split(self):
        import sqlite3

        from moss_quant import config as cfg
        from moss_quant.db import get_moss_wallet, migrate_moss_tables, profile_wallet_balance

        conn = sqlite3.connect(":memory:")
        migrate_moss_tables(conn.cursor())
        conn.commit()
        cap = float(cfg.MOSS_QUANT_PROFILE_CAPITAL)
        wallet = get_moss_wallet(conn, reconcile=False)
        self.assertAlmostEqual(wallet["initial_capital_usdt"], 0.0, places=2)
        self.assertAlmostEqual(wallet["balance_usdt"], 0.0, places=2)
        now = "2024-01-01T00:00:00Z"
        conn.execute(
            """INSERT INTO moss_profiles(
                   id, name, symbol, template, enabled, initial_params_json,
                   tactical_params_json, virtual_equity_usdt, created_at_utc, updated_at_utc)
               VALUES (1, 't', 'BTCUSDT', 'balanced', 1, '{}', '{}', 10000, ?, ?)""",
            (now, now),
        )
        conn.commit()
        migrate_moss_tables(conn.cursor())
        conn.commit()
        wallet = get_moss_wallet(conn, reconcile=False)
        self.assertAlmostEqual(wallet["initial_capital_usdt"], cap, places=2)
        self.assertAlmostEqual(wallet["balance_usdt"], cap, places=2)
        bal = profile_wallet_balance(conn, 1, sync=False)
        self.assertAlmostEqual(bal, cap, places=2)
        conn.close()

    def test_paper_sizing_profiles_independent_not_global_wallet(self):
        import sqlite3

        from moss_quant.db import (
            get_moss_wallet,
            migrate_moss_tables,
            sync_moss_wallet_from_settlements,
        )
        from moss_quant.paper_scanner import _notional_for_profile

        conn = sqlite3.connect(":memory:")
        migrate_moss_tables(conn.cursor())
        now = "2024-01-01T00:00:00Z"
        for pid, sym in ((1, "BTCUSDT"), (2, "ETHUSDT")):
            conn.execute(
                """INSERT INTO moss_profiles(
                       id, name, symbol, template, enabled, initial_params_json,
                       tactical_params_json, virtual_equity_usdt, created_at_utc, updated_at_utc)
                   VALUES (?, ?, ?, 'balanced', 1, '{}', '{}', 10000, ?, ?)""",
                (pid, sym, sym, now, now),
            )
        conn.execute(
            """INSERT INTO moss_settlements(
                   settled_at_utc, signal_id, profile_id, symbol, side,
                   outcome, pnl_usdt)
               VALUES (?, 1, 1, 'BTCUSDT', 'LONG', 'win', 5000)""",
            (now,),
        )
        conn.commit()
        sync_moss_wallet_from_settlements(conn)
        global_bal = get_moss_wallet(conn, reconcile=False)["balance_usdt"]
        cap = float(__import__("moss_quant.config", fromlist=["MOSS_QUANT_PROFILE_CAPITAL"]).MOSS_QUANT_PROFILE_CAPITAL)
        self.assertAlmostEqual(global_bal, 2 * cap + 5000.0, places=2)
        params = {
            "base_leverage": 1,
            "max_leverage": 1,
            "risk_per_trade": 1.0,
            "max_position_pct": 1.0,
        }
        cap = float(__import__("moss_quant.config", fromlist=["MOSS_QUANT_PROFILE_CAPITAL"]).MOSS_QUANT_PROFILE_CAPITAL)
        n1 = _notional_for_profile(conn, 1, params, leverage=1)
        n2 = _notional_for_profile(conn, 2, params, leverage=1)
        self.assertAlmostEqual(n1, cap + 5000.0, places=2)
        self.assertAlmostEqual(n2, cap, places=2)
        conn.close()

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

    def test_optimize_policy_composite_and_pool(self):
        from moss_quant.optimize_policy import (
            can_sync_profile_params,
            classify_pool_tier,
            composite_optimize_score,
            enrich_summary,
            hard_reject_reason,
        )

        good = {
            "total_return": 0.12,
            "sharpe": 1.2,
            "max_drawdown": -0.08,
            "total_trades": 10,
            "blowup_count": 0,
            "validation_passed": True,
            "validation_reason": "验证通过",
        }
        self.assertIsNone(hard_reject_reason(good))
        self.assertGreater(composite_optimize_score(good), 0)
        enriched = enrich_summary(good)
        self.assertEqual(enriched.get("pool_tier"), "A")
        self.assertTrue(can_sync_profile_params(enriched))

        weak_val = {**good, "validation_passed": False, "validation_reason": "验证收益≤0"}
        tier_b = classify_pool_tier(weak_val)
        self.assertEqual(tier_b["pool_tier"], "B")
        self.assertFalse(can_sync_profile_params(enrich_summary(weak_val)))

        bad = {**good, "total_return": -0.01}
        self.assertIsNotNone(hard_reject_reason(bad))
        self.assertEqual(composite_optimize_score(bad), -999.0)

    def test_optimize_train_val_split(self):
        import pandas as pd

        from moss_quant.optimize_policy import split_train_validation_df

        n = 500
        df = pd.DataFrame({"timestamp": range(n), "close": [1.0] * n})
        train, val = split_train_validation_df(df, train_ratio=0.7)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(val), 0)
        self.assertEqual(len(train) + len(val), n)

    def test_optimize_trailing_off_by_default(self):
        from moss_quant.optimize_service import _run_one, _trailing_for_template
        import pandas as pd

        from moss_quant.core.regime import classify_regime

        self.assertFalse(_trailing_for_template("momentum"))
        self.assertFalse(_trailing_for_template("mean_revert"))
        n = 120
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min"),
                "open": [100.0] * n,
                "high": [101.0] * n,
                "low": [99.0] * n,
                "close": [100.0 + (i % 5) * 0.1 for i in range(n)],
                "volume": [1000.0] * n,
            }
        )
        regime = classify_regime(df)
        row = _run_one(
            df,
            regime,
            symbol="BTCUSDT",
            template="momentum",
            tactical={
                "entry_threshold": 0.48,
                "sl_atr_mult": 2.5,
                "tp_rr_ratio": 2.5,
                "exit_threshold": 0.12,
                "regime_sensitivity": 0.55,
            },
            capital=10000.0,
        )
        self.assertFalse(row.get("tactical_params", {}).get("trailing_enabled"))


if __name__ == "__main__":
    unittest.main()
