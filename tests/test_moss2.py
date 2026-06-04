"""Moss2 隔离 lane 基础测试。"""

from __future__ import annotations

import sqlite3
import unittest


class TestMoss2(unittest.TestCase):
    def test_factory_roots_exist(self):
        from moss2.config import (
            en_data_cache_dir,
            factory_en_root,
            factory_hl_root,
            hl_data_cache_dir,
            moss2_runtime_snapshot,
        )

        self.assertTrue(factory_hl_root().name.startswith("moss-trade-bot-factory"))
        self.assertTrue("factory" in factory_en_root().name.lower())
        snap = moss2_runtime_snapshot()
        self.assertIn("hl_data_cache", snap)
        self.assertIn("scripts", str(hl_data_cache_dir()))
        str(en_data_cache_dir())  # 不因 Path 拼接报错

    def test_hl_catalog_non_empty_when_data_present(self):
        from moss2.config import hl_data_cache_dir
        from moss2.dataset import list_hl_catalog

        if not hl_data_cache_dir().is_dir():
            self.skipTest("HL data_cache missing")
        cat = list_hl_catalog()
        self.assertGreater(len(cat), 0)

    def test_schema_defaults_cover_all_decision_fields(self):
        from moss2.variants.hl.core.decision import DecisionParams
        from moss2.params import build_initial_params, schema_default_params

        defaults = schema_default_params("hl")
        self.assertEqual(defaults["entry_threshold"], 0.20)
        self.assertEqual(defaults["long_bias"], 0.50)
        self.assertEqual(defaults["fast_ma_period"], 10)
        merged = build_initial_params("balanced", variant="hl")
        valid = {f.name for f in DecisionParams.__dataclass_fields__.values()}
        for name in valid:
            self.assertIn(name, merged, msg=f"missing {name}")

    def test_migrate_and_profile(self):
        from moss2.db import create_profile, get_profile, list_profiles, migrate_moss2_tables
        from moss2.params import build_initial_params, split_profile_params

        conn = sqlite3.connect(":memory:")
        migrate_moss2_tables(conn.cursor())
        conn.commit()
        merged = build_initial_params("balanced", variant="hl")
        initial, tactical = split_profile_params(merged, variant="hl")
        self.assertIn("entry_threshold", tactical)
        self.assertIn("trend_weight", initial)
        pid = create_profile(
            conn,
            name="btc-hl",
            symbol="BTCUSDC",
            variant="hl",
            template="balanced",
            enabled=True,
            initial_params=initial,
            tactical_params=tactical,
            virtual_equity_usdt=10000,
        )
        prof = get_profile(conn, pid)
        self.assertIsNotNone(prof)
        self.assertEqual(prof["variant"], "hl")
        self.assertEqual(prof.get("params_version"), "v1")
        self.assertEqual(prof.get("evolution_status"), "baseline")
        self.assertEqual(len(list_profiles(conn, enabled_only=True)), 1)

    def test_ops_variant_binance_only(self):
        from moss2 import config as c

        self.assertEqual(c.MOSS2_OPS_VARIANT, "en")
        self.assertFalse(c.MOSS2_HL_ENABLED)
        self.assertEqual(c.effective_variant(None), "en")
        self.assertEqual(c.effective_variant("en"), "en")
        with self.assertRaises(ValueError):
            c.effective_variant("hl")

    def test_protocol_ingest_open_traded(self):
        from moss2.protocol_result import (
            protocol_ingest_close_result,
            protocol_ingest_open_result,
        )

        traded = {
            "ok": True,
            "details": [
                {
                    "action": "traded",
                    "position_id": 9,
                    "entry_price": 50123.5,
                    "client_ref": "moss2-abc",
                }
            ],
        }
        open_r = protocol_ingest_open_result(traded)
        self.assertTrue(open_r.ok)
        self.assertEqual(open_r.position_id, 9)
        self.assertEqual(open_r.entry_price, 50123.5)
        self.assertEqual(open_r.client_ref, "moss2-abc")

        skipped = {"ok": True, "details": [{"action": "skipped", "reason": "dup"}]}
        self.assertFalse(protocol_ingest_open_result(skipped).ok)
        self.assertFalse(protocol_ingest_close_result(skipped).ok)

    def test_paper_run_stores_full_stats(self):
        from moss2.db import insert_paper_run, latest_paper_run, migrate_moss2_tables

        conn = sqlite3.connect(":memory:")
        migrate_moss2_tables(conn.cursor())
        conn.commit()
        stats = {
            "profiles_scanned": 2,
            "opens": 1,
            "closes": 0,
            "protocol_opens": 1,
            "protocol_closes": 0,
            "real_mode": False,
            "details": [{"label": "x", "action": "hold"}],
        }
        insert_paper_run(conn, stats)
        run = latest_paper_run(conn)
        self.assertIsNotNone(run)
        self.assertEqual(run["protocol_opens"], 1)
        self.assertEqual(run["real_mode"], False)
        self.assertEqual(run["details"][0]["action"], "hold")

    def test_list_profiles_for_paper_scan_includes_disabled_with_open(self):
        from moss2.db import (
            create_profile,
            list_profiles_for_paper_scan,
            migrate_moss2_tables,
        )
        from moss2.params import build_initial_params, split_profile_params

        conn = sqlite3.connect(":memory:")
        migrate_moss2_tables(conn.cursor())
        conn.commit()
        merged = build_initial_params("balanced", variant="hl")
        initial, tactical = split_profile_params(merged, variant="hl")
        pid = create_profile(
            conn,
            name="eth-hl",
            symbol="ETHUSDC",
            variant="hl",
            template="balanced",
            enabled=False,
            initial_params=initial,
            tactical_params=tactical,
            virtual_equity_usdt=10000,
        )
        conn.execute(
            """INSERT INTO moss2_signals(
                   profile_id, recorded_at_utc, side, symbol,
                   entry_price, virtual_notional_usdt, updated_at_utc)
               VALUES (?,?,?,?,?,?,?)""",
            (pid, "2026-01-01T00:00:00Z", "LONG", "ETHUSDC", 100.0, 10000, "2026-01-01T00:00:00Z"),
        )
        conn.commit()
        scanned = list_profiles_for_paper_scan(conn)
        self.assertEqual(len(scanned), 1)
        self.assertEqual(scanned[0]["id"], pid)

    def test_paper_scan_reads_open_row_as_mapping(self):
        from unittest.mock import patch

        import pandas as pd

        from moss2.db import (
            create_profile,
            migrate_moss2_tables,
        )
        from moss2.paper_scanner import run_paper_scan
        from moss2.params import build_initial_params, split_profile_params

        conn = sqlite3.connect(":memory:")
        migrate_moss2_tables(conn.cursor())
        conn.commit()
        merged = build_initial_params("balanced", variant="en")
        initial, tactical = split_profile_params(merged, variant="en")
        pid = create_profile(
            conn,
            name="btc-en",
            symbol="BTCUSDT",
            variant="en",
            template="balanced",
            enabled=True,
            initial_params=initial,
            tactical_params=tactical,
            virtual_equity_usdt=10000,
        )
        conn.execute(
            """INSERT INTO moss2_signals(
                   profile_id, recorded_at_utc, side, symbol,
                   entry_price, virtual_notional_usdt, meta_json, updated_at_utc)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                pid,
                "2026-01-01T00:00:00Z",
                "LONG",
                "BTCUSDT",
                50000.0,
                10000,
                '{"regime":"TREND"}',
                "2026-01-01T00:00:00Z",
            ),
        )
        conn.commit()
        idx = pd.date_range("2026-01-01", periods=120, freq="15min", tz="UTC")
        df = pd.DataFrame(
            {
                "open": 50000.0,
                "high": 50100.0,
                "low": 49900.0,
                "close": 50050.0,
                "volume": 1.0,
            },
            index=idx,
        )

        with patch("moss2.paper_scanner.load_market_df", return_value=df):
            stats = run_paper_scan(conn)
        self.assertEqual(stats["profiles_scanned"], 1)
        self.assertIn(stats["details"][0]["action"], ("hold", "close"))

    def test_en_backtest_btc_if_csv(self):
        from moss2.backtest_service import run_factory_backtest
        from moss2.dataset import resolve_csv_path
        from moss2.params import build_initial_params

        if not resolve_csv_path("BTCUSDT", "en"):
            self.skipTest("BTC EN CSV missing")
        out = run_factory_backtest(
            symbol="BTCUSDT",
            params=build_initial_params("balanced", variant="en"),
            variant="en",
            capital=10000,
            limit_bars=800,
        )
        self.assertEqual(out["variant"], "en")
        self.assertEqual(out["engine"], "en_cross_margin")
        self.assertIn("summary", out)

    def test_paper_wallet_summary_fields(self):
        from moss2.db import (
            create_profile,
            get_moss2_wallet,
            migrate_moss2_tables,
            summarize_lane,
        )
        from moss2.params import build_initial_params, split_profile_params

        conn = sqlite3.connect(":memory:")
        migrate_moss2_tables(conn.cursor())
        conn.commit()
        merged = build_initial_params("balanced", variant="en")
        initial, tactical = split_profile_params(merged, variant="en")
        create_profile(
            conn,
            name="btc-en",
            symbol="BTCUSDT",
            variant="en",
            template="balanced",
            enabled=True,
            initial_params=initial,
            tactical_params=tactical,
            virtual_equity_usdt=10000,
        )
        wallet = get_moss2_wallet(conn)
        self.assertEqual(wallet["initial_capital_usdt"], 10000.0)
        self.assertEqual(wallet["balance_usdt"], 10000.0)
        summary = summarize_lane(conn)
        self.assertEqual(summary["mode"], "paper")
        self.assertIn("wallet_balance_usdt", summary)
        self.assertIn("equity_usdt", summary)

    def test_protocol_open_failure_still_records_paper(self):
        from unittest.mock import patch

        import pandas as pd

        from moss2.db import create_profile, migrate_moss2_tables
        from moss2.paper_scanner import run_paper_scan
        from moss2.params import build_initial_params, split_profile_params

        conn = sqlite3.connect(":memory:")
        migrate_moss2_tables(conn.cursor())
        conn.commit()
        merged = build_initial_params("balanced", variant="en")
        initial, tactical = split_profile_params(merged, variant="en")
        pid = create_profile(
            conn,
            name="btc-en",
            symbol="BTCUSDT",
            variant="en",
            template="balanced",
            enabled=True,
            initial_params=initial,
            tactical_params=tactical,
            virtual_equity_usdt=10000,
        )
        idx = pd.date_range("2026-01-01", periods=120, freq="15min", tz="UTC")
        df = pd.DataFrame(
            {
                "open": 100.0,
                "high": 200.0,
                "low": 99.0,
                "close": 150.0,
                "volume": 1.0,
            },
            index=idx,
        )
        bad_open = {"ok": True, "details": [{"action": "skipped_invalid_source"}]}

        with patch("moss2.paper_scanner.load_market_df", return_value=df), patch(
            "moss2.paper_scanner.compute_current_signal", return_value=(1, 0.55, "TREND")
        ), patch(
            "moss2.paper_scanner.check_open_gate", return_value=(True, "", {})
        ), patch("moss2.signal_sender.is_real_mode", return_value=True), patch(
            "moss2.signal_sender.send_open", return_value=bad_open
        ):
            stats = run_paper_scan(conn)
        row = conn.execute(
            "SELECT id FROM moss2_signals WHERE profile_id=? AND outcome IS NULL",
            (pid,),
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(stats["opens"], 1)
        self.assertTrue(
            any(d.get("action") == "open" for d in stats.get("details") or [])
        )

    def test_protocol_close_failure_still_settles_paper(self):
        from unittest.mock import patch

        import pandas as pd

        from moss2.db import create_profile, migrate_moss2_tables
        from moss2.paper_scanner import run_paper_scan
        from moss2.params import build_initial_params, split_profile_params

        conn = sqlite3.connect(":memory:")
        migrate_moss2_tables(conn.cursor())
        conn.commit()
        merged = build_initial_params("balanced", variant="en")
        initial, tactical = split_profile_params(merged, variant="en")
        pid = create_profile(
            conn,
            name="btc-en",
            symbol="BTCUSDT",
            variant="en",
            template="balanced",
            enabled=True,
            initial_params=initial,
            tactical_params=tactical,
            virtual_equity_usdt=10000,
        )
        conn.execute(
            """INSERT INTO moss2_signals(
                   profile_id, recorded_at_utc, side, symbol,
                   entry_price, virtual_notional_usdt, meta_json, updated_at_utc)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                pid,
                "2026-01-01T00:00:00Z",
                "LONG",
                "BTCUSDT",
                100.0,
                10000,
                '{"regime":"TREND"}',
                "2026-01-01T00:00:00Z",
            ),
        )
        conn.commit()
        idx = pd.date_range("2026-01-01", periods=120, freq="15min", tz="UTC")
        df = pd.DataFrame(
            {
                "open": 100.0,
                "high": 200.0,
                "low": 99.0,
                "close": 150.0,
                "volume": 1.0,
            },
            index=idx,
        )
        bad_close = {"ok": True, "details": [{"action": "skipped", "reason": "no_position"}]}

        with patch("moss2.paper_scanner.load_market_df", return_value=df), patch(
            "moss2.signal_sender.is_real_mode", return_value=True
        ), patch("moss2.signal_sender.send_close", return_value=bad_close):
            stats = run_paper_scan(conn)
        row = conn.execute(
            "SELECT outcome FROM moss2_signals WHERE profile_id=?", (pid,)
        ).fetchone()
        self.assertIsNotNone(row[0])
        self.assertEqual(stats["closes"], 1)
        self.assertTrue(
            any(d.get("action") == "close" for d in stats.get("details") or [])
        )
        sett = conn.execute(
            "SELECT COUNT(*) FROM moss2_settlements WHERE profile_id=?", (pid,)
        ).fetchone()[0]
        self.assertEqual(int(sett), 1)


    def test_delete_profile_blocks_open_position(self):
        from moss2.db import create_profile, delete_profile, migrate_moss2_tables
        from moss2.params import build_initial_params, split_profile_params

        conn = sqlite3.connect(":memory:")
        migrate_moss2_tables(conn.cursor())
        conn.commit()
        merged = build_initial_params("balanced", variant="en")
        initial, tactical = split_profile_params(merged, variant="en")
        pid = create_profile(
            conn,
            name="btc-en",
            symbol="BTCUSDT",
            variant="en",
            template="balanced",
            enabled=True,
            initial_params=initial,
            tactical_params=tactical,
            virtual_equity_usdt=10000,
        )
        conn.execute(
            """INSERT INTO moss2_signals(
                   profile_id, recorded_at_utc, side, symbol,
                   entry_price, virtual_notional_usdt, updated_at_utc)
               VALUES (?,?,?,?,?,?,?)""",
            (pid, "2026-01-01T00:00:00Z", "LONG", "BTCUSDT", 100.0, 10000, "2026-01-01T00:00:00Z"),
        )
        conn.commit()
        with self.assertRaises(ValueError) as ctx:
            delete_profile(conn, pid)
        self.assertEqual(str(ctx.exception), "profile_has_open_position")


if __name__ == "__main__":
    unittest.main()
