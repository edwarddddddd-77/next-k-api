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

    def test_default_data_source_binance(self):
        from moss_quant import config as cfg

        self.assertEqual(cfg.MOSS_QUANT_DATA_SOURCE, "binance")

    def test_optimize_returns_ranking(self):
        from unittest.mock import patch

        from moss_quant import config as cfg
        from moss_quant.optimize_service import run_strategy_optimize

        small = dict(
            MOSS_QUANT_OPTIMIZE_MAX_COMBINATIONS=8,
            MOSS_QUANT_OPTIMIZE_ENTRY_THRESHOLDS=(0.40, 0.50),
            MOSS_QUANT_OPTIMIZE_SL_ATR_MULTS=(2.0,),
            MOSS_QUANT_OPTIMIZE_TP_RR_RATIOS=(2.0, 3.0),
        )
        with patch.multiple(cfg, **small):
            out = run_strategy_optimize(
                symbol="BTCUSDT",
                capital=10000,
                refresh_klines=False,
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
                        "total_return": 0.2,
                        "total_trades": 20,
                        "max_drawdown": -0.08,
                        "blowup_count": 0,
                        "win_rate": 0.55,
                        "validation_passed": True,
                        "validation_reason": "验证通过",
                        "val_return": 0.12,
                        "val_sharpe": 1.1,
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
                        "val_return": 0.15,
                        "val_sharpe": 0.9,
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

    def test_pool_governance_auto_disable_paper_loss(self):
        import json
        import sqlite3
        from datetime import datetime, timedelta, timezone

        from moss_quant.db import migrate_moss_tables
        from moss_quant.pool_governance import apply_pool_governance

        conn = sqlite3.connect(":memory:")
        migrate_moss_tables(conn.cursor())
        conn.commit()
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        cur = conn.execute(
            """INSERT INTO moss_daily_optimize_batches(
                   ran_at_utc, status, symbols_total, capital, data_source)
               VALUES (?,?,?,?,?)""",
            (now, "completed", 1, 1000.0, "binance"),
        )
        batch_id = int(cur.lastrowid)
        conn.execute(
            """INSERT INTO moss_daily_optimize_items(
                   batch_id, symbol, template, tactical_params_json,
                   summary_json, score)
               VALUES (?,?,?,?,?,?)""",
            (
                batch_id,
                "BTCUSDT",
                "balanced",
                json.dumps({"entry_threshold": 0.44}),
                json.dumps(
                    {
                        "total_return": 0.1,
                        "total_trades": 10,
                        "max_drawdown": -0.05,
                        "blowup_count": 0,
                        "win_rate": 0.5,
                        "validation_passed": True,
                        "wf_validation_passed": True,
                    }
                ),
                0.5,
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
                "balanced",
                1,
                "manual",
                json.dumps({}),
                json.dumps({}),
                1000.0,
                0,
                now,
                now,
            ),
        )
        pid = int(
            conn.execute("SELECT id FROM moss_profiles WHERE symbol='BTCUSDT'").fetchone()[0]
        )
        loss_time = (datetime.now(timezone.utc) - timedelta(days=1)).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
        conn.execute(
            """INSERT INTO moss_signals(
                   profile_id, recorded_at_utc, side, symbol, entry_price,
                   outcome, outcome_at_utc, pnl_usdt, updated_at_utc)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                pid,
                loss_time,
                "LONG",
                "BTCUSDT",
                100.0,
                "closed",
                loss_time,
                -80.0,
                loss_time,
            ),
        )
        conn.commit()
        stats = apply_pool_governance(conn, batch_id, trigger_paper_scan=False)
        self.assertEqual(stats["disabled"], 1)
        self.assertTrue(
            any(a.get("action") == "auto_disabled_paper_loss" for a in stats["actions"])
        )

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
            "val_return": 0.12,
            "val_sharpe": 1.0,
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

    def test_pool_governance_streak_a_nonsync_freezes_degrade(self):
        import json
        import sqlite3

        from moss_quant.db import migrate_moss_tables, upsert_symbol_pool_streak
        from moss_quant.pool_governance import _next_streaks

        conn = sqlite3.connect(":memory:")
        migrate_moss_tables(conn.cursor())
        conn.commit()
        upsert_symbol_pool_streak(
            conn,
            "SOLUSDT",
            last_pool_tier="B",
            last_batch_id=1,
            degrade_streak=1,
            upgrade_streak=0,
        )
        deg, up = _next_streaks(
            {"degrade_streak": 1, "upgrade_streak": 0},
            upgrade_round=False,
            pool_tier="A",
        )
        self.assertEqual(deg, 1)
        self.assertEqual(up, 0)
        deg2, up2 = _next_streaks(
            {"degrade_streak": 1, "upgrade_streak": 0},
            upgrade_round=False,
            pool_tier="B",
        )
        self.assertEqual(deg2, 2)
        self.assertEqual(up2, 0)

    def test_auto_enable_eligible_requires_sync_top5(self):
        import json
        import sqlite3
        from unittest.mock import patch

        from moss_quant import config as cfg
        from moss_quant.db import migrate_moss_tables, upsert_symbol_pool_streak
        from moss_quant.optimize_policy import enrich_summary
        from moss_quant.pool_governance import (
            _batch_items_by_symbol,
            apply_pool_governance,
            auto_enable_eligible_symbols,
        )

        def _good(sym: str, val_sharpe: float) -> dict:
            return enrich_summary(
                {
                    "total_return": 0.2,
                    "total_trades": 12,
                    "max_drawdown": -0.08,
                    "blowup_count": 0,
                    "win_rate": 0.55,
                    "validation_passed": True,
                    "validation_reason": "验证通过",
                    "val_return": 0.12,
                    "val_sharpe": val_sharpe,
                    "wf_validation_passed": True,
                    "wf_folds": 3,
                    "wf_passed_folds": 2,
                }
            )

        conn = sqlite3.connect(":memory:")
        migrate_moss_tables(conn.cursor())
        conn.commit()
        now = "2024-01-01T00:00:00Z"
        cur = conn.execute(
            """INSERT INTO moss_daily_optimize_batches(
                   ran_at_utc, status, symbols_total, capital, data_source)
               VALUES (?,?,?,?,?)""",
            (now, "completed", 6, 10000.0, "binance"),
        )
        batch_id = int(cur.lastrowid)
        for sym, sh in [
            ("AAAUSDT", 3.0),
            ("BBBUSDT", 2.5),
            ("CCCUSDT", 2.0),
            ("DDDUSDT", 1.5),
            ("EEEUSDT", 1.0),
            ("FFFUSDT", 0.5),
        ]:
            summary = _good(sym, sh)
            conn.execute(
                """INSERT INTO moss_daily_optimize_items(
                       batch_id, symbol, template, tactical_params_json,
                       summary_json, score)
                   VALUES (?,?,?,?,?,?)""",
                (
                    batch_id,
                    sym,
                    "balanced",
                    json.dumps({"entry_threshold": 0.44}),
                    json.dumps(summary, ensure_ascii=False),
                    sh,
                ),
            )
            upsert_symbol_pool_streak(
                conn,
                sym,
                last_pool_tier="A",
                last_batch_id=batch_id,
                degrade_streak=0,
                upgrade_streak=2,
            )
        conn.commit()
        items = _batch_items_by_symbol(conn, batch_id)
        streak_map = {sym: {"upgrade_streak": 2, "degrade_streak": 0} for sym in items}
        with patch.object(cfg, "MOSS_QUANT_POOL_AUTO_ADD_TOP_N", 5):
            with patch.object(cfg, "MOSS_QUANT_POOL_UPGRADE_STREAK", 2):
                eligible = auto_enable_eligible_symbols(items, streak_map)
        self.assertIn("AAAUSDT", eligible)
        self.assertIn("EEEUSDT", eligible)
        self.assertNotIn("FFFUSDT", eligible)

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
            "val_return": 0.12,
            "val_sharpe": 1.0,
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

    def test_fetch_klines_history_paginates(self):
        from unittest.mock import patch

        from binance_fapi import BINANCE_KLINE_MAX_PER_REQUEST, fetch_klines_history

        calls: list = []

        def _fake(sym, interval, limit, end_time_ms=None):
            calls.append((limit, end_time_ms))
            if end_time_ms is None:
                return [[10 + i, 1, 1, 1, 1, 1] for i in range(3)]
            return [[7 + i, 1, 1, 1, 1, 1] for i in range(3)]

        with patch("binance_fapi.fetch_klines", side_effect=_fake):
            with patch("binance_fapi.BINANCE_KLINE_MAX_PER_REQUEST", 3):
                rows = fetch_klines_history("BTCUSDT", "15m", 5)
        self.assertEqual(len(rows), 5)
        self.assertEqual(int(rows[0][0]), 8)
        self.assertEqual(int(rows[-1][0]), 12)
        self.assertGreaterEqual(len(calls), 2)

    def test_research_kline_bar_limits(self):
        from moss_quant import config as cfg
        from moss_quant.kline_cache import kline_bar_limit

        self.assertGreaterEqual(
            kline_bar_limit(research=True), kline_bar_limit(research=False)
        )
        self.assertEqual(cfg.MOSS_QUANT_RESEARCH_KLINE_BARS, 6720)
        self.assertEqual(cfg.MOSS_QUANT_KLINE_LIMIT, 1500)
        self.assertEqual(cfg.MOSS_QUANT_DATA_SOURCE, "binance")
        self.assertEqual(kline_bar_limit(research=False), cfg.MOSS_QUANT_KLINE_LIMIT)

    def test_universe_matches_daily_optimize_catalog(self):
        from moss_quant.universe import (
            MOSS_DAILY_CORE_BASES,
            list_universe,
            moss_daily_optimize_bases,
        )

        catalog = moss_daily_optimize_bases()
        syms = {u["base"] for u in list_universe()}
        self.assertGreaterEqual(len(catalog), len(MOSS_DAILY_CORE_BASES))
        self.assertEqual(len(syms), len(catalog))
        for base in ("BTC", "ETH", "HYPE", "ARB", "SUI", "ICP", "TON", "PEPE", "ZEC"):
            self.assertIn(base, syms, msg=f"missing {base}")

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
        out = add_symbol_to_daily_core(conn, "JUPUSDT", note="test")
        self.assertTrue(out["added"])
        rows = list_daily_core_symbols(conn)
        syms = {r["symbol"] for r in rows if int(r.get("enabled") or 0)}
        self.assertIn("JUPUSDT", syms)
        out2 = add_symbol_to_daily_core(conn, "JUPUSDT")
        self.assertFalse(out2.get("added"))
        self.assertTrue(out2.get("already_in_daily_core"))

    def test_daily_core_symbol_allowed_for_paper_profile(self):
        import sqlite3

        from moss_quant.db import add_symbol_to_daily_core, migrate_moss_tables

        conn = sqlite3.connect(":memory:")
        migrate_moss_tables(conn.cursor())
        conn.commit()
        self.assertTrue(is_symbol_allowed("ZECUSDT", conn=conn))
        merged = list_universe(conn)
        self.assertIn("ZECUSDT", {u["symbol"] for u in merged})

    def test_daily_core_symbols_table_seeded(self):
        import sqlite3

        from moss_quant.db import list_daily_core_bases, migrate_moss_tables
        from moss_quant.universe import moss_daily_optimize_bases

        conn = sqlite3.connect(":memory:")
        migrate_moss_tables(conn.cursor())
        conn.commit()
        expected = moss_daily_optimize_bases()
        bases = list_daily_core_bases(conn)
        self.assertEqual(len(bases), len(expected))
        self.assertEqual(set(bases), set(expected))

    def test_daily_optimize_defaults_on(self):
        from moss_quant import config as cfg

        self.assertTrue(cfg.MOSS_QUANT_DAILY_OPTIMIZE_ENABLED)
        self.assertTrue(cfg.MOSS_QUANT_DAILY_OPTIMIZE_BOOTSTRAP)
        self.assertTrue(cfg.MOSS_QUANT_DAILY_OPTIMIZE_APPLY_PROFILES)
        self.assertTrue(cfg.daily_optimize_scheduler_enabled())
        self.assertTrue(cfg.daily_optimize_bootstrap_enabled())

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

    def test_walk_forward_folds_and_stability(self):
        import pandas as pd

        from moss_quant.optimize_policy import (
            aggregate_walk_forward_validation,
            can_sync_profile_params,
            enrich_summary,
            split_walk_forward_folds,
            stability_adjusted_val_score,
            templates_for_regime,
            train_val_ratio_ok,
        )

        n = 600
        df = pd.DataFrame({"timestamp": range(n), "close": [1.0] * n})
        folds = split_walk_forward_folds(df, n_folds=3)
        self.assertGreaterEqual(len(folds), 2)

        agg = aggregate_walk_forward_validation(
            [
                {"validation_passed": True, "val_sharpe": 1.0, "val_return": 0.1},
                {"validation_passed": True, "val_sharpe": 0.8, "val_return": 0.08},
                {"validation_passed": False, "val_sharpe": -1.0, "val_return": -0.2},
            ]
        )
        self.assertTrue(agg["wf_validation_passed"])
        self.assertEqual(agg["wf_passed_folds"], 2)

        hi = stability_adjusted_val_score(1.0, train_return=0.5, val_return=0.48)
        lo = stability_adjusted_val_score(1.0, train_return=0.5, val_return=0.05)
        self.assertGreater(hi, lo)

        self.assertIsNotNone(train_val_ratio_ok(0.5, 0.01))

        overfit = {
            "total_return": 0.5,
            "val_return": 0.05,
            "sharpe": 1.0,
            "max_drawdown": -0.1,
            "total_trades": 10,
            "blowup_count": 0,
            "validation_passed": True,
            "wf_validation_passed": True,
        }
        enriched_over = enrich_summary(overfit)
        self.assertFalse(enriched_over.get("sync_allowed"))
        self.assertIn("收益比", str(enriched_over.get("sync_block_reason") or ""))

        tpl = templates_for_regime({"regime_note": "sideways_heavy"})
        self.assertIn("mean_revert", tpl)
        self.assertNotIn("trend", tpl)

    def test_validation_context_and_gate_proxy(self):
        import pandas as pd

        from moss_quant import config as cfg
        from moss_quant.gate_proxy import funding_extreme_stats, validation_gate_penalty
        from moss_quant.optimize_service import (
            _build_validation_context,
            _validation_window_bounds,
        )

        n = 200
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1.0,
            }
        )
        train, val = df.iloc[:140].copy(), df.iloc[140:].copy()
        ctx = _build_validation_context(train, val)
        self.assertEqual(len(ctx), min(len(train), cfg.MOSS_QUANT_OPTIMIZE_VAL_WARMUP_BARS) + len(val))
        ws, we = _validation_window_bounds(val, ctx)
        self.assertLessEqual(ws, pd.Timestamp(val["timestamp"].iloc[0]))

        stats = funding_extreme_stats(val.iloc[:10], "BTCUSDT")
        self.assertIn("extreme_ratio", stats)
        pen = validation_gate_penalty({"extreme_ratio": 0.5})
        self.assertGreater(pen, 0.0)

    def test_paper_recent_loss_blocks_sync(self):
        import json
        import sqlite3
        from datetime import datetime, timedelta, timezone

        from moss_quant.db import migrate_moss_tables
        from moss_quant.optimize_policy import enrich_summary, paper_recent_pnl_block_reason

        conn = sqlite3.connect(":memory:")
        migrate_moss_tables(conn.cursor())
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
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
                "trend",
                1,
                "manual",
                "{}",
                "{}",
                1000.0,
                0,
                now,
                now,
            ),
        )
        conn.execute(
            """INSERT INTO moss_signals(
                   profile_id, recorded_at_utc, side, symbol,
                   virtual_notional_usdt, outcome, outcome_at_utc, pnl_usdt, updated_at_utc)
               VALUES (1,?,?,?,?,?,?,?,?)""",
            (now, "LONG", "BTCUSDT", 1000, "win", now, -80.0, now),
        )
        conn.commit()
        reason = paper_recent_pnl_block_reason(conn, 1, profile_capital=1000.0)
        self.assertIsNotNone(reason)
        good = {
            "total_return": 0.2,
            "val_return": 0.12,
            "sharpe": 1.0,
            "max_drawdown": -0.1,
            "total_trades": 10,
            "blowup_count": 0,
            "validation_passed": True,
            "wf_validation_passed": True,
        }
        out = enrich_summary(good, conn=conn, profile_id=1, profile_capital=1000.0)
        self.assertFalse(out.get("sync_allowed"))
        self.assertIn("近", str(out.get("sync_block_reason") or ""))

    def test_regime_align_bear_relaxes_short_threshold(self):
        from moss_quant import config as cfg
        from moss_quant.trade_gates import regime_aligned_threshold_deltas

        with self.subTest("bear_aligned"):
            out = regime_aligned_threshold_deltas(
                0.50,
                train_regime_note="trend_heavy",
                live_regime="BEAR",
                template="balanced",
                allow_relax=True,
            )
            self.assertEqual(out["alignment"], "aligned")
            self.assertLess(out["short_delta"], 0)
            self.assertEqual(out["long_delta"], 0)

        with self.subTest("sideways_mismatch_tightens"):
            out = regime_aligned_threshold_deltas(
                0.50,
                train_regime_note="trend_heavy",
                live_regime="SIDEWAYS",
                template="balanced",
                allow_relax=True,
            )
            self.assertEqual(out["alignment"], "misaligned")
            self.assertGreater(out["long_delta"], 0)
            self.assertGreater(out["short_delta"], 0)

        with self.subTest("paper_loss_blocks_relax"):
            out = regime_aligned_threshold_deltas(
                0.50,
                train_regime_note="trend_heavy",
                live_regime="BEAR",
                allow_relax=False,
            )
            self.assertEqual(out["long_delta"], 0)
            self.assertEqual(out["short_delta"], 0)

        with self.subTest("sideways_heavy_relaxes_momentum_too"):
            out = regime_aligned_threshold_deltas(
                0.50,
                train_regime_note="sideways_heavy",
                live_regime="SIDEWAYS",
                template="momentum",
                allow_relax=True,
            )
            self.assertEqual(out["alignment"], "aligned")
            self.assertLess(out["long_delta"], 0)
            self.assertLess(out["short_delta"], 0)

    def test_entry_snapshot_asymmetric_thresholds(self):
        import numpy as np

        from moss_quant.core.decision import DecisionParams
        from moss_quant.core.regime import classify_regime
        from moss_quant.paper_scanner import entry_snapshot

        n = 120
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
                "open": np.linspace(100, 90, n),
                "high": np.linspace(101, 91, n),
                "low": np.linspace(99, 89, n),
                "close": np.linspace(100, 90, n),
                "volume": np.full(n, 1000.0),
            }
        )
        params = DecisionParams.from_dict({"entry_threshold": 0.50})
        regime_s = classify_regime(df, version="v1")
        ent = entry_snapshot(
            df, params, regime_s, long_threshold=0.50, short_threshold=0.40
        )
        self.assertIn(ent["signal"], (-1, 0, 1))
        self.assertEqual(ent["entry_threshold_long"], 0.50)
        self.assertEqual(ent["entry_threshold_short"], 0.40)

    def test_intraday_threshold_bump_uses_pnl_usdt_column(self):
        import sqlite3
        from unittest.mock import patch

        from moss_quant import config as cfg
        from moss_quant.db import migrate_moss_tables
        from moss_quant.trade_gates import intraday_threshold_bump

        conn = sqlite3.connect(":memory:")
        migrate_moss_tables(conn.cursor())
        conn.commit()
        conn.execute(
            """INSERT INTO moss_profiles(
                   id, name, symbol, template, enabled,
                   initial_params_json, tactical_params_json,
                   virtual_equity_usdt, created_at_utc, updated_at_utc)
               VALUES (1,'t','TONUSDT','balanced',1,'{}','{}',1000,'t','t')"""
        )
        conn.execute(
            """INSERT INTO moss_signals(
                   profile_id, recorded_at_utc, side, symbol, entry_price,
                   outcome, outcome_at_utc, pnl_usdt, updated_at_utc)
               VALUES (1,'2024-01-01T00:00:00Z','LONG','TONUSDT',1.0,
                       'closed','2024-01-01T01:00:00Z',-60.0,'t')"""
        )
        conn.execute(
            """INSERT INTO moss_signals(
                   profile_id, recorded_at_utc, side, symbol, entry_price,
                   unrealized_pnl_usdt, outcome, updated_at_utc)
               VALUES (1,'2024-01-02T00:00:00Z','LONG','TONUSDT',1.0,
                       -10.0,NULL,'t')"""
        )
        conn.commit()
        with patch.object(cfg, "MOSS_QUANT_INTRADAY_ADJUST_ENABLED", True):
            with patch.object(cfg, "MOSS_QUANT_INTRADAY_DRAWDOWN_PCT", 0.05):
                with patch.object(cfg, "MOSS_QUANT_INTRADAY_DRAWDOWN_BUMP", 0.08):
                    bump = intraday_threshold_bump(conn, 1, profile_capital=1000.0)
        self.assertEqual(bump, 0.08)

    def test_oi_spike_flat_detect(self):
        from moss_quant.trade_gates import _oi_spike_flat_price

        self.assertTrue(_oi_spike_flat_price({"d6h": 5.0, "px_chg": 2.0}))
        self.assertFalse(_oi_spike_flat_price({"d6h": 1.0, "px_chg": 2.0}))
        self.assertFalse(_oi_spike_flat_price({"d6h": 5.0, "px_chg": 8.0}))

    def test_portfolio_open_cap(self):
        import sqlite3
        from unittest.mock import patch

        from moss_quant.db import migrate_moss_tables
        from moss_quant.portfolio_risk import check_portfolio_open, list_open_signal_exposure

        conn = sqlite3.connect(":memory:")
        migrate_moss_tables(conn.cursor())
        conn.commit()
        conn.execute(
            """INSERT INTO moss_profiles(id, name, symbol, template, enabled,
               initial_params_json, tactical_params_json, created_at_utc, updated_at_utc)
               VALUES (1,'a','BTCUSDT','trend',1,'{}','{}','t','t')"""
        )
        conn.execute(
            """INSERT INTO moss_signals(
                   profile_id, recorded_at_utc, side, symbol,
                   virtual_notional_usdt, outcome, updated_at_utc)
               VALUES (1,'2024-01-01T00:00:00Z','LONG','BTCUSDT',9000,NULL,'t')"""
        )
        conn.commit()
        self.assertEqual(len(list_open_signal_exposure(conn)), 1)
        with patch("moss_quant.portfolio_risk._wallet_budget_usdt", return_value=10000.0):
            ok, scale, reason = check_portfolio_open(
                conn, symbol="ETHUSDT", side="LONG", proposed_notional=5000
            )
        self.assertFalse(ok)
        self.assertIn("同向", reason)

    def test_optimize_trailing_on_for_trend_templates(self):
        from moss_quant.optimize_service import _run_one, _trailing_for_template
        import pandas as pd

        from moss_quant.core.regime import classify_regime

        self.assertTrue(_trailing_for_template("momentum"))
        self.assertTrue(_trailing_for_template("trend"))
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
        self.assertTrue(row.get("tactical_params", {}).get("trailing_enabled"))

    def test_backtest_diagnosis_neighbors_and_holdout(self):
        from moss_quant.backtest_diagnosis import (
            build_neighbor_candidates,
            compare_holdout_validation,
            suggest_tactical_adjustments,
        )
        from moss_quant.core.engine import Trade

        trades = [
            Trade(
                entry_idx=i,
                entry_price=100.0,
                direction=1,
                margin=100.0,
                leverage=1,
                gross_pnl=-5.0,
                pnl_pct=-0.005,
                exit_reason="flip_close_long",
            )
            for i in range(10)
        ]
        analysis = __import__(
            "moss_quant.backtest_diagnosis", fromlist=["analyze_trades"]
        ).analyze_trades(trades)
        tact = {
            "entry_threshold": 0.44,
            "sl_atr_mult": 2.0,
            "tp_rr_ratio": 2.5,
            "exit_threshold": 0.12,
            "regime_sensitivity": 0.55,
        }
        sug = suggest_tactical_adjustments(
            analysis, tact, template="momentum", regime_note="sideways_heavy"
        )
        neighbors = build_neighbor_candidates(tact, sug.get("tuned_tactical"))
        self.assertGreater(len(neighbors), 1)
        hold = compare_holdout_validation(
            {"total_return": 0.01, "sharpe": 0.4, "max_drawdown": -0.1, "total_trades": 4},
            {"total_return": 0.04, "sharpe": 0.8, "max_drawdown": -0.08, "total_trades": 5},
        )
        self.assertTrue(hold["adopted"])

    def test_post_grid_pipeline_uses_walk_forward(self):
        from moss_quant import config as cfg
        from moss_quant.backtest_diagnosis import run_post_grid_pipeline
        from moss_quant.optimize_service import (
            _build_run_params,
            _validate_candidate_walk_forward,
            split_walk_forward_folds,
        )
        from moss_quant.core.regime import classify_regime
        import pandas as pd

        if not cfg.MOSS_QUANT_OPTIMIZE_LOCAL_REFINE_ENABLED:
            self.skipTest("local refine disabled")
        n = 200
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
                "open": [100.0 + i * 0.03 for i in range(n)],
                "high": [101.0 + i * 0.03 for i in range(n)],
                "low": [99.0 + i * 0.03 for i in range(n)],
                "close": [100.0 + i * 0.03 for i in range(n)],
                "volume": [1000.0] * n,
            }
        )
        cut = int(n * 0.7)
        df_train = df.iloc[:cut].reset_index(drop=True)
        regime_train = classify_regime(df_train)
        tact = {
            "entry_threshold": 0.44,
            "sl_atr_mult": 2.0,
            "tp_rr_ratio": 2.5,
            "exit_threshold": 0.12,
            "regime_sensitivity": 0.55,
        }
        wf_folds = split_walk_forward_folds(df)
        grid_wf = _validate_candidate_walk_forward(
            {"tactical_params": tact, "template": "balanced", "summary": {"total_return": 0.05}},
            wf_folds,
            symbol="BTCUSDT",
            capital=10000.0,
            regime_version="v1",
        )

        def validate_wf_fn(tact, tpl):
            return _validate_candidate_walk_forward(
                {"tactical_params": tact, "template": tpl, "summary": {"total_return": 0.05}},
                wf_folds,
                symbol="BTCUSDT",
                capital=10000.0,
                regime_version="v1",
            )

        pipe = run_post_grid_pipeline(
            df_train=df_train,
            regime_train=regime_train,
            symbol="BTCUSDT",
            template="balanced",
            tactical=tact,
            capital=10000.0,
            regime_note="sideways_heavy",
            build_params_fn=_build_run_params,
            validate_wf_fn=validate_wf_fn,
            grid_wf_validation=grid_wf,
        )
        self.assertEqual(pipe.get("validation_mode"), "walk_forward")
        self.assertIn("final_tactical_params", pipe)
        refine = pipe.get("local_refine") or {}
        self.assertEqual(refine.get("validation_mode"), "walk_forward")

    def test_recent_pick_guards(self):
        from moss_quant.recent_window_pick import _passes_guards

        ok, _ = _passes_guards(
            {
                "total_trades": 10,
                "total_return": 0.05,
                "blowup_count": 0,
                "profit_factor": 1.2,
            },
            {"total_return": 0.01},
        )
        self.assertTrue(ok)
        bad, reason = _passes_guards(
            {"total_trades": 3, "total_return": 0.05, "blowup_count": 0},
            {"total_return": 0.01},
        )
        self.assertFalse(bad)
        self.assertIn("笔数", reason)
        bad_pf, reason_pf = _passes_guards(
            {
                "total_trades": 10,
                "total_return": 0.05,
                "blowup_count": 0,
                "profit_factor": 0.5,
            },
            {"total_return": 0.01},
        )
        self.assertFalse(bad_pf)
        self.assertIn("盈亏比", reason_pf)

    def test_rank_recent_score_prefers_more_trades(self):
        from moss_quant.recent_window_pick import _rank_recent_score

        high_ret_few = _rank_recent_score(
            {"total_return": 0.20, "total_trades": 4, "profit_factor": 1.5}
        )
        mid_ret_many = _rank_recent_score(
            {"total_return": 0.12, "total_trades": 12, "profit_factor": 1.4}
        )
        self.assertGreater(mid_ret_many[0], high_ret_few[0])

    def test_side_attribution_raises_short_threshold(self):
        from moss_quant.trade_gates import side_attribution_threshold_deltas

        d = side_attribution_threshold_deltas(
            {
                "long_count": 8,
                "short_count": 6,
                "long_win_rate": 0.55,
                "short_win_rate": 0.30,
            },
            base_threshold=0.44,
        )
        self.assertGreater(d["short_delta"], 0.0)
        self.assertEqual(d["long_delta"], 0.0)
        self.assertEqual(d["reason"], "short_side_weak")

    def test_env_truthy_explicit_off(self):
        import importlib
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {"MOSS_QUANT_ENABLED": "off"}):
            import moss_quant.config as mq_cfg

            importlib.reload(mq_cfg)
            self.assertFalse(mq_cfg.MOSS_QUANT_ENABLED)

    def test_moss_runtime_switches_default_on(self):
        import importlib
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {}, clear=False):
            for key in (
                "MOSS_QUANT_ENABLED",
                "MOSS_QUANT_PAPER_ENABLED",
                "MOSS_QUANT_SCHEDULER_ENABLED",
                "MOSS_QUANT_DAILY_OPTIMIZE_ENABLED",
                "MOSS_QUANT_DAILY_OPTIMIZE_BOOTSTRAP",
                "MOSS_QUANT_RECENT_PICK_ENABLED",
                "MOSS_QUANT_POOL_GOVERNANCE_ENABLED",
            ):
                os.environ.pop(key, None)
            import moss_quant.config as mq_cfg

            importlib.reload(mq_cfg)
            snap = mq_cfg.moss_runtime_switch_snapshot()
            self.assertTrue(snap["enabled"])
            self.assertTrue(snap["paper_scheduler"])
            self.assertTrue(snap["daily_optimize_bootstrap"])
            self.assertTrue(snap["recent_pick"])
            self.assertTrue(snap["pool_governance"])

    def test_side_stats_from_post_grid_local_refine(self):
        from moss_quant.trade_gates import _side_stats_from_summary

        ss = _side_stats_from_summary(
            {
                "post_grid_pipeline": {
                    "local_refine": {
                        "rounds": [
                            {
                                "train_analysis": {
                                    "side_stats": {
                                        "long_count": 5,
                                        "short_count": 4,
                                        "long_win_rate": 0.6,
                                        "short_win_rate": 0.25,
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        )
        self.assertEqual(ss.get("short_count"), 4)

    def test_sync_allow_when_recent_pick_no_combo(self):
        from moss_quant.optimize_policy import sync_deny_reason

        reason = sync_deny_reason(
            {
                "pool_tier": "A",
                "validation_passed": True,
                "wf_validation_passed": True,
                "train_return": 0.1,
                "val_return": 0.08,
                "total_return": 0.1,
                "total_trades": 20,
                "win_rate": 0.5,
                "max_drawdown": 0.1,
                "blowup_count": 0,
                "auto_enabled": True,
                "recent_pick": {
                    "skipped": False,
                    "bars": 1500,
                    "adopted": False,
                    "reason": "1500窗无满足门槛的组合",
                },
            }
        )
        self.assertIsNone(reason)

    def test_sync_deny_when_recent_tail_below_floor(self):
        from moss_quant.optimize_policy import sync_deny_reason

        reason = sync_deny_reason(
            {
                "pool_tier": "A",
                "validation_passed": True,
                "wf_validation_passed": True,
                "train_return": 0.1,
                "val_return": 0.08,
                "total_return": 0.1,
                "total_trades": 20,
                "win_rate": 0.5,
                "max_drawdown": 0.1,
                "blowup_count": 0,
                "auto_enabled": True,
                "recent_pick": {
                    "skipped": False,
                    "tail_return_pct": -5.0,
                    "adopted": False,
                },
            }
        )
        self.assertIsNotNone(reason)
        self.assertIn("尾段", reason or "")

    def test_recent_pick_skips_non_a_pool(self):
        from moss_quant.recent_window_pick import pick_best_on_recent_window

        out = pick_best_on_recent_window(
            "BTCUSDT",
            l1_summary={"pool_tier": "B", "sync_allowed": True},
        )
        self.assertFalse(out.get("adopted"))
        self.assertTrue(out.get("skipped"))

    def test_grid_combos_prefers_l1_template(self):
        from moss_quant.recent_window_pick import _grid_combos

        combos = _grid_combos({}, prefer_template="momentum")
        self.assertGreater(len(combos), 0)
        self.assertEqual(str(combos[0][0]).lower(), "momentum")
        exits = {float(c[1].get("exit_threshold") or 0) for c in combos}
        self.assertGreater(len(exits), 1)

    def test_apply_recent_pick_preserves_l1_fields(self):
        from moss_quant.recent_window_pick import apply_recent_pick_to_best

        best = {
            "template": "balanced",
            "tactical_params": {"entry_threshold": 0.44},
            "summary": {
                "pool_tier": "B",
                "sync_allowed": False,
                "train_return": 0.1,
                "val_return": 0.08,
                "param_source": "grid",
            },
        }
        out = apply_recent_pick_to_best(best, "BTCUSDT", refresh_klines=False)
        sm = out.get("summary") or {}
        self.assertEqual(sm.get("l1_train_return"), 0.1)
        self.assertEqual(sm.get("l1_val_return"), 0.08)
        self.assertIn("recent_pick", sm)
        self.assertFalse(sm.get("recent_applied"))


    def test_evaluate_entry_signal_margin_and_confirm(self):
        from unittest.mock import patch

        from moss_quant import config as cfg
        from moss_quant.signal_entry import evaluate_entry_signal

        with self.subTest("margin_blocks_weak_long"):
            ev = evaluate_entry_signal(
                [0.44, 0.46],
                long_threshold=0.40,
                short_threshold=0.40,
                entry_margin=0.05,
                confirm_bars=2,
            )
            self.assertEqual(ev["signal"], 0)
            self.assertIn("margin", ev["reason"])
        with self.subTest("strong_long_passes"):
            ev2 = evaluate_entry_signal(
                [0.50, 0.52],
                long_threshold=0.40,
                short_threshold=0.40,
                entry_margin=0.05,
                confirm_bars=2,
            )
            self.assertEqual(ev2["signal"], 1)
        with self.subTest("disabled_quality_legacy"):
            with patch.object(cfg, "MOSS_QUANT_ENTRY_QUALITY_ENABLED", False):
                ev3 = evaluate_entry_signal(
                    [0.42],
                    long_threshold=0.40,
                    short_threshold=0.40,
                )
                self.assertEqual(ev3["signal"], 1)

    def test_validation_reachable_penalty_low_ratio(self):
        from moss_quant.gate_proxy import validation_reachable_penalty

        pen = validation_reachable_penalty(
            {"reachable_ratio": 0.002, "reachable_sub_trades": 0}
        )
        self.assertGreater(pen, 0.0)
        pen_ok = validation_reachable_penalty(
            {"reachable_ratio": 0.05, "reachable_sub_trades": 8, "reachable_sub_pf": 1.2}
        )
        self.assertLess(pen_ok, pen)

    def test_pick_best_validated_sorts_by_reachable_penalty(self):
        from moss_quant.optimize_policy import pick_best_validated

        hi_pen = {
            "score": 10.0,
            "summary": {"total_return": 0.2},
            "validation": {
                "validation_passed": True,
                "val_return": 0.1,
                "val_sharpe": 1.0,
                "gate_penalty": 0.0,
                "reachable_penalty": 0.12,
                "stability_score": 0.5,
            },
        }
        lo_pen = {
            "score": 9.0,
            "summary": {"total_return": 0.18},
            "validation": {
                "validation_passed": True,
                "val_return": 0.09,
                "val_sharpe": 0.9,
                "gate_penalty": 0.0,
                "reachable_penalty": 0.01,
                "stability_score": 0.85,
            },
        }
        best = pick_best_validated([hi_pen, lo_pen])
        self.assertIs(best, lo_pen)


if __name__ == "__main__":
    unittest.main()
