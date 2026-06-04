"""Moss2 全自动 Profile 运维。"""

from __future__ import annotations

import sqlite3
import unittest
from unittest.mock import patch

from moss2.db import get_profile, list_profiles, migrate_moss2_tables


class TestMoss2AutoProvision(unittest.TestCase):
    def _mem_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        migrate_moss2_tables(conn.cursor())
        conn.commit()
        return conn

    def test_provision_creates_and_enables_on_discipline_pass(self):
        from moss2.auto_provision import provision_symbol

        conn = self._mem_conn()
        fake_suggest = {
            "ok": True,
            "symbol": "BTCUSDT",
            "variant": "en",
            "recommended_template": "balanced",
            "recommended_name": "btc-en-balanced",
            "recommended_enabled": True,
            "reason": "backtest_selection_pass",
        }
        fake_evolve = {
            "ok": True,
            "profile_id": 1,
            "status": "approved",
            "candidate": {
                "summary": {"total_trades": 10, "sharpe": 0.5, "max_drawdown": 0.2},
                "discipline": {"ev": {"ev_per_trade_pct": 0.1}},
            },
        }
        with (
            patch("moss2.auto_provision.suggest_profile", return_value=fake_suggest),
            patch(
                "moss2.auto_provision.run_profile_evolve",
                return_value=fake_evolve,
            ),
            patch("moss2.auto_provision.cfg.MOSS2_AUTO_ENABLE_PROFILES", True),
            patch("moss2.auto_provision.cfg.MOSS2_EVOLVE_AUTO_APPROVE", True),
        ):
            out = provision_symbol(conn, "BTCUSDT")
        self.assertEqual(out["action"], "create")
        self.assertTrue(out.get("auto_enabled") or out.get("enabled"))
        profs = list_profiles(conn, enabled_only=True)
        self.assertEqual(len(profs), 1)
        self.assertEqual(profs[0]["symbol"], "BTCUSDT")

    def test_provision_skips_when_suggest_fails(self):
        from moss2.auto_provision import provision_symbol

        conn = self._mem_conn()
        with patch(
            "moss2.auto_provision.suggest_profile",
            return_value={"ok": False, "reason": "klines_unavailable"},
        ):
            out = provision_symbol(conn, "ETHUSDT")
        self.assertEqual(out["action"], "skip")
        self.assertEqual(len(list_profiles(conn)), 0)

    def test_finalize_skips_full_evolve_when_selection_pass(self):
        from moss2.auto_provision import _finalize_profile
        from moss2.db import create_profile, get_profile, migrate_moss2_tables
        from moss2.params import build_initial_params, split_profile_params

        conn = sqlite3.connect(":memory:")
        migrate_moss2_tables(conn.cursor())
        conn.commit()
        merged = build_initial_params("balanced", variant="en")
        initial, tactical = split_profile_params(merged, variant="en")
        pid = create_profile(
            conn,
            name="x",
            symbol="BTCUSDT",
            variant="en",
            template="balanced",
            enabled=False,
            initial_params=initial,
            tactical_params=tactical,
            virtual_equity_usdt=10_000,
        )
        suggestion = {
            "symbol": "BTCUSDT",
            "reason": "backtest_selection_pass",
            "recommended_template": "balanced",
            "recommended_params": merged,
            "selection_best": {
                "discipline": {"ev": {"ev_per_trade_pct": 0.1}},
                "summary": {
                    "total_trades": 10,
                    "sharpe": 0.5,
                    "max_drawdown": 0.2,
                },
            },
        }
        with patch("moss2.auto_provision.run_profile_evolve") as mock_ev:
            _finalize_profile(conn, pid, suggestion, force_evolve=False)
            mock_ev.assert_not_called()
        prof = get_profile(conn, pid)
        self.assertEqual(prof.get("evolution_status"), "approved")

    def test_run_lane_counts_actions(self):
        from moss2.auto_provision import run_lane_auto_provision

        conn = self._mem_conn()
        with (
            patch("moss2.auto_provision.cfg.MOSS2_SEED_BASES", ("BTC", "ETH")),
            patch(
                "moss2.auto_provision.provision_symbol",
                side_effect=[
                    {"action": "create", "symbol": "BTCUSDT"},
                    {"action": "skip", "symbol": "ETHUSDT"},
                ],
            ),
        ):
            stats = run_lane_auto_provision(conn)
        self.assertTrue(stats["ok"])
        self.assertEqual(stats["created"], 1)
        self.assertEqual(stats["skipped"], 1)


if __name__ == "__main__":
    unittest.main()
