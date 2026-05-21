#!/usr/bin/env python3
"""touch_pool_config 默认常量与 env 解析。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_API_ROOT = Path(__file__).resolve().parent.parent
if str(_API_ROOT) not in sys.path:
    sys.path.insert(0, str(_API_ROOT))

from touch_pool_config import (
    TOUCH_POOL_CRON_HOURS,
    TOUCH_POOL_CRON_MINUTE,
    TOUCH_POOL_MIN_PF,
    TOUCH_POOL_WALK_HOURS,
    apply_touch_pool_default_env,
    touch_pool_4h_cron_slots,
    touch_pool_4h_filter_params,
    touch_pool_default_child_env,
)


class TouchPoolConfigTests(unittest.TestCase):
    def test_filter_params_match_constants(self) -> None:
        p = touch_pool_4h_filter_params()
        self.assertAlmostEqual(p["days"], TOUCH_POOL_WALK_HOURS / 24.0)
        self.assertAlmostEqual(p["min_profit_factor"], TOUCH_POOL_MIN_PF)
        self.assertEqual(p["min_win_loss_abs"], 3)
        self.assertEqual(p["min_touch_trades"], 3)

    def test_cron_slots_default(self) -> None:
        slots = touch_pool_4h_cron_slots()
        hours = [h for h, _ in slots]
        self.assertEqual(hours, list(TOUCH_POOL_CRON_HOURS))
        self.assertTrue(all(m == TOUCH_POOL_CRON_MINUTE for _, m in slots))

    def test_apply_default_env_fills_missing_only(self) -> None:
        base = {"ZCT_TOUCH_POOL_WALK_HOURS": "12"}
        out = apply_touch_pool_default_env(base)
        self.assertEqual(out["ZCT_TOUCH_POOL_WALK_HOURS"], "12")
        self.assertEqual(out["ZCT_TOUCH_POOL_MIN_PF"], str(TOUCH_POOL_MIN_PF))

    def test_child_env_has_universe(self) -> None:
        d = touch_pool_default_child_env()
        self.assertEqual(d["ZCT_TOUCH_POOL_UNIVERSE"], "1")


if __name__ == "__main__":
    unittest.main()
