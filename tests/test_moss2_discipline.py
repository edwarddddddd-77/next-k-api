"""Moss2 discipline / gates / versioning."""

from __future__ import annotations

import sqlite3
import unittest


class TestMoss2Discipline(unittest.TestCase):
    def test_trade_stats_ev(self):
        from moss2.discipline.metrics import trade_stats_from_rows

        trades = [{"pnl_pct": 0.02}, {"pnl_pct": -0.01}, {"pnl_pct": 0.03}]
        s = trade_stats_from_rows(trades)
        self.assertEqual(s["trade_count"], 3)
        self.assertGreater(s["ev_per_trade_pct"], 0)

    def test_recent_trades_use_profile_notional(self):
        from moss2.db import create_profile, migrate_moss2_tables
        from moss2.discipline.gates import recent_settled_trades
        from moss2.params import build_initial_params, split_profile_params

        conn = sqlite3.connect(":memory:")
        migrate_moss2_tables(conn.cursor())
        conn.commit()
        merged = build_initial_params("balanced", variant="en")
        initial, tactical = split_profile_params(merged, variant="en")
        pid = create_profile(
            conn,
            name="t",
            symbol="BTCUSDT",
            variant="en",
            template="balanced",
            enabled=True,
            initial_params=initial,
            tactical_params=tactical,
            virtual_equity_usdt=5000,
        )
        conn.execute(
            """INSERT INTO moss2_signals(
                   profile_id, recorded_at_utc, side, symbol,
                   entry_price, virtual_notional_usdt, outcome, pnl_usdt, updated_at_utc)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                pid,
                "2026-01-01T00:00:00Z",
                "LONG",
                "BTCUSDT",
                100.0,
                5000.0,
                "closed",
                50.0,
                "2026-01-02T00:00:00Z",
            ),
        )
        conn.commit()
        rows = recent_settled_trades(conn, pid)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["pnl_pct"], 0.01)

    def test_open_gate_margin(self):
        from unittest.mock import patch

        from moss2 import config as cfg
        from moss2.db import create_profile, migrate_moss2_tables
        from moss2.discipline.gates import check_open_gate
        from moss2.params import build_initial_params, split_profile_params

        conn = sqlite3.connect(":memory:")
        migrate_moss2_tables(conn.cursor())
        conn.commit()
        merged = build_initial_params("balanced", variant="hl")
        initial, tactical = split_profile_params(merged, variant="hl")
        pid = create_profile(
            conn,
            name="t",
            symbol="BTCUSDC",
            variant="hl",
            template="balanced",
            enabled=True,
            initial_params=initial,
            tactical_params=tactical,
            virtual_equity_usdt=10000,
        )
        with patch.object(cfg, "MOSS2_ENTRY_QUALITY_ENABLED", False):
            ok, reason, _ = check_open_gate(
                conn, pid, composite=0.05, entry_threshold=0.2
            )
        self.assertFalse(ok)
        self.assertEqual(reason, "margin_below_threshold")

    def test_entry_quality_requires_confirm(self):
        import pandas as pd

        from moss2.discipline.entry_quality import evaluate_open_signal
        from moss2.params import build_initial_params

        n = 120
        df = pd.DataFrame(
            {
                "open": [100.0 + i * 0.01 for i in range(n)],
                "high": [101.0 + i * 0.01 for i in range(n)],
                "low": [99.0 + i * 0.01 for i in range(n)],
                "close": [100.0 + i * 0.01 for i in range(n)],
                "volume": [1.0] * n,
            }
        )
        params = build_initial_params("balanced", variant="en")
        params["_symbol"] = "BTCUSDT"
        params["entry_threshold"] = 0.40
        out = evaluate_open_signal(
            df, params, "en", entry_threshold=0.40
        )
        self.assertEqual(out["signal"], 0)
        self.assertIn(out["reason"], ("composite_below_threshold", "confirm_bars_insufficient"))

    def test_effective_entry_threshold_floor(self):
        from moss2.discipline.entry_quality import effective_entry_threshold

        self.assertAlmostEqual(effective_entry_threshold(0.26), 0.40)
        self.assertAlmostEqual(effective_entry_threshold(0.44), 0.44)

    def test_params_for_quality_backtest_bumps_threshold(self):
        from moss2.discipline.entry_quality import params_for_quality_backtest

        out = params_for_quality_backtest({"entry_threshold": 0.40})
        self.assertAlmostEqual(out["entry_threshold"], 0.45)

    def test_params_hash_stable(self):
        from moss2.params import build_initial_params, split_profile_params
        from moss2.versioning import params_hash

        merged = build_initial_params("balanced", variant="en")
        initial, tactical = split_profile_params(merged, variant="en")
        prof = {
            "variant": "en",
            "symbol": "BTCUSDT",
            "initial_params_json": __import__("json").dumps(initial),
            "tactical_params_json": __import__("json").dumps(tactical),
        }
        h1 = params_hash(prof)
        h2 = params_hash(prof)
        self.assertEqual(h1, h2)


if __name__ == "__main__":
    unittest.main()
