"""ORB 2.0 配置测试。"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from orb.v2.config import OrbV2Config, _load_symbols


from tools.orb.ml.eval_live_gate import init_robot_wallets  # noqa: E402
from orb.v2.robots import next_robot_index as _next_robot_index  # noqa: E402


class TestEightRobots(unittest.TestCase):
    def test_next_robot_index_skips_used_and_depleted(self):
        wallets = [10_000.0, 0.0, 10_000.0]
        self.assertEqual(_next_robot_index(set(), wallets), 0)
        self.assertEqual(_next_robot_index({0}, wallets), 2)
        self.assertIsNone(_next_robot_index({0, 2}, [0.0, 0.0, 0.0]))

    def test_init_robot_wallets(self):
        ws = init_robot_wallets(count=8, equity_usdt=10_000.0)
        self.assertEqual(len(ws), 8)
        self.assertEqual(ws[0], 10_000.0)


class TestOrbV2Config(unittest.TestCase):
    def test_load_symbols_from_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("COIN, PAYP\nMSTR\n")
            path = Path(f.name)
        try:
            syms = _load_symbols(env_override="", symbols_file=path)
            self.assertEqual(syms, ["COINUSDT", "PAYPUSDT", "MSTRUSDT"])
        finally:
            path.unlink(missing_ok=True)

    def test_load_symbols_env_overrides_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("TSLA\n")
            path = Path(f.name)
        try:
            syms = _load_symbols(env_override="COIN", symbols_file=path)
            self.assertEqual(syms, ["COINUSDT"])
        finally:
            path.unlink(missing_ok=True)

    def test_from_env_uses_config_symbols_file(self):
        v2 = OrbV2Config.from_env()
        self.assertTrue(str(v2.symbols_file).replace("\\", "/").endswith("config/orb/v2/symbols.txt"))
        syms = v2.symbol_list()
        self.assertEqual(len(syms), 43)
        self.assertIn("COINUSDT", syms)

    def test_from_env_uses_v2_symbols_file_not_orb_symbols(self):
        saved = {
            k: os.environ.pop(k, None)
            for k in ("ORB_SYMBOLS", "ORB_V2_SYMBOLS", "ORB_V2_SYMBOLS_FILE", "ORB_V2_ENABLED")
        }
        try:
            os.environ["ORB_SYMBOLS"] = "BTCUSDT"
            os.environ.pop("ORB_V2_SYMBOLS", None)
            os.environ.pop("ORB_V2_SYMBOLS_FILE", None)
            v2 = OrbV2Config.from_env()
            syms = v2.symbol_list()
            self.assertGreaterEqual(len(syms), 40)
            self.assertIn("COINUSDT", syms)
            self.assertNotIn("BTCUSDT", syms)
            self.assertNotEqual(syms, v2.base.symbol_list())
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


    def test_session_dates_from_cache_skips_weekends(self):
        import os
        from env_loader import load_env_oi
        from orb.core.config import OrbConfig
        from tools.orb.v2.backtest_symbol import session_dates_from_cache
        import pandas as pd

        load_env_oi()
        saved = os.environ.pop("ORB_SYMBOLS", None)
        try:
            cfg = OrbConfig.from_env()
            cfg.macro_filter = False
            dates = session_dates_from_cache("SPYUSDT", cfg)
            self.assertTrue(dates)
            for d in dates:
                self.assertLess(pd.Timestamp(d).dayofweek, 5, d)
        finally:
            if saved is None:
                os.environ.pop("ORB_SYMBOLS", None)
            else:
                os.environ["ORB_SYMBOLS"] = saved


if __name__ == "__main__":
    unittest.main()
