"""Moss2 线上 data_cache 路径与 bootstrap 逻辑。"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestMoss2DataBootstrap(unittest.TestCase):
    def test_en_data_cache_default_without_skills(self):
        from moss2 import config as c

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MOSS2_PREFER_SKILLS_DATA_CACHE", None)
            os.environ.pop("MOSS2_EN_DATA_CACHE", None)
            p = c.en_data_cache_dir()
        self.assertIn("moss2_en_data_cache", str(p).replace("\\", "/"))
        self.assertTrue(p.is_dir())

    def test_base_to_fetch_slash_multiplier(self):
        from moss2.config import base_to_fetch_slash

        self.assertEqual(base_to_fetch_slash("PEPE"), "1000PEPE/USDT")
        self.assertEqual(base_to_fetch_slash("BTC"), "BTC/USDT")

    def test_canonical_csv_path(self):
        from moss2.data_bootstrap import canonical_csv_path

        with tempfile.TemporaryDirectory() as td:
            path = canonical_csv_path(Path(td), "ETH")
            self.assertEqual(
                path.name,
                "binanceusdm_ETH_USDT_USDT_15m_2025-10-06_148d.csv",
            )

    def test_startup_bootstrap_needed(self):
        from moss2.data_bootstrap import (
            bootstrap_marker_path,
            canonical_csv_path,
            startup_bootstrap_needed,
            write_bootstrap_marker,
        )

        with tempfile.TemporaryDirectory() as td:
            cache = Path(td)
            with patch("moss2.data_bootstrap.en_data_cache_dir", return_value=cache):
                need, reason = startup_bootstrap_needed()
                self.assertTrue(need)
                self.assertIn("missing", reason)
                lines = ["timestamp,open,high,low,close,volume\n"]
                lines += [f"2025-10-06 00:{i:02d}:00,1,1,1,1,1\n" for i in range(20)]
                canonical_csv_path(cache, "BTC").write_text("".join(lines), encoding="utf-8")
                need2, _ = startup_bootstrap_needed()
                self.assertTrue(need2)
                write_bootstrap_marker(
                    {
                        "ran_at_utc": "2026-06-02T00:00:00Z",
                        "saved": 0,
                        "skipped": 1,
                        "bases": 1,
                        "ok": True,
                    },
                    cache,
                )
                self.assertTrue(bootstrap_marker_path(cache).is_file())
                with patch("moss2.data_bootstrap.MOSS2_SEED_BASES", ("BTC",)):
                    need3, reason3 = startup_bootstrap_needed()
                self.assertFalse(need3)
                self.assertEqual(reason3, "seed_cache_ready")

    def test_bootstrap_skips_fresh(self):
        from moss2.data_bootstrap import bootstrap_seed_data, canonical_csv_path
        from moss2 import config as c

        with tempfile.TemporaryDirectory() as td:
            cache = Path(td)
            lines = ["timestamp,open,high,low,close,volume\n"]
            lines += [f"2025-10-06 00:{i:02d}:00,1,1,1,1,1\n" for i in range(20)]
            canonical_csv_path(cache, "BTC").write_text("".join(lines), encoding="utf-8")
            with patch("moss2.data_bootstrap.en_data_cache_dir", return_value=cache):
                stats = bootstrap_seed_data(bases=["BTC"], force=False)
            self.assertEqual(stats["skipped"], 1)
            self.assertEqual(stats["saved"], 0)
            self.assertTrue(stats["ok"])


if __name__ == "__main__":
    unittest.main()
