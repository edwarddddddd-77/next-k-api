"""Moss2 淘汰服务。"""

from __future__ import annotations

import sqlite3
import unittest
from unittest.mock import patch

from moss2.db import create_profile, get_profile, migrate_moss2_tables
from moss2.params import build_initial_params, split_profile_params


class TestMoss2Cull(unittest.TestCase):
    def _seed_enabled_profile(self, conn: sqlite3.Connection) -> int:
        merged = build_initial_params("balanced", variant="en")
        initial, tactical = split_profile_params(merged, variant="en")
        return create_profile(
            conn,
            name="btc-en",
            symbol="BTCUSDT",
            variant="en",
            template="balanced",
            enabled=True,
            initial_params=initial,
            tactical_params=tactical,
            virtual_equity_usdt=10_000,
        )

    def test_cull_disables_failing_profile(self):
        from moss2.cull_service import cull_profile

        conn = sqlite3.connect(":memory:")
        migrate_moss2_tables(conn.cursor())
        conn.commit()
        pid = self._seed_enabled_profile(conn)
        with (
            patch(
                "moss2.cull_service._live_fitness",
                return_value=(False, "live_ev_below_floor", {}),
            ),
            patch(
                "moss2.cull_service._backtest_fitness",
                return_value=(True, "rebacktest_ok", {}),
            ),
        ):
            out = cull_profile(conn, pid)
        self.assertEqual(out["action"], "culled")
        prof = get_profile(conn, pid)
        self.assertFalse(prof["enabled"])
        self.assertEqual(prof.get("evolution_status"), "culled")

    def test_recompete_refresh_then_recheck_fails_culls(self):
        from moss2.cull_service import run_lane_cull

        conn = sqlite3.connect(":memory:")
        migrate_moss2_tables(conn.cursor())
        conn.commit()
        pid = self._seed_enabled_profile(conn)
        with (
            patch(
                "moss2.cull_service.recompete_and_refresh",
                return_value={"ok": True, "refreshed": True, "template": "trend"},
            ),
            patch(
                "moss2.cull_service.evaluate_profile",
                return_value={
                    "ok": True,
                    "profile_id": pid,
                    "enabled": True,
                    "keep": False,
                    "live_ok": False,
                    "backtest_ok": True,
                },
            ),
            patch("moss2.cull_service.cull_profile") as mock_cull,
        ):
            mock_cull.return_value = {"action": "culled", "profile_id": pid}
            stats = run_lane_cull(conn)
        self.assertEqual(stats["culled"], 1)


if __name__ == "__main__":
    unittest.main()
