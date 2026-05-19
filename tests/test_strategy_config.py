#!/usr/bin/env python3
"""StrategyConfig 行为测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_API_ROOT = Path(__file__).resolve().parent.parent
if str(_API_ROOT) not in sys.path:
    sys.path.insert(0, str(_API_ROOT))

from zct_strategy_config import StrategyConfig


class StrategyConfigTests(unittest.TestCase):
    def test_copy_for_scan_resets_btc_macro(self) -> None:
        base = StrategyConfig.from_env()
        base.btc_macro_state["slope_bps"] = 99.0
        base.btc_macro_state["chop"] = "low"
        copy = base.copy_for_scan()
        self.assertIsNot(copy, base)
        self.assertEqual(copy.btc_macro_state["slope_bps"], 0.0)
        self.assertEqual(copy.btc_macro_state["chop"], "high")
        self.assertEqual(base.btc_macro_state["slope_bps"], 99.0)

    def test_for_backtest_disables_btc_macro_by_default(self) -> None:
        bt = StrategyConfig.for_backtest()
        self.assertFalse(bt.btc_macro_filter_enabled)
        self.assertFalse(bt.use_db_cooldown)

if __name__ == "__main__":
    unittest.main()
